"""Activation recording: one value per (prompt, neuron), with streaming per-set statistics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .prompts import Encoded, encode_chat
from .runtime import forward_logits

REDUCTIONS = ("mean", "max", "last")
FP16_MAX = 65000.0


class Cancelled(Exception):
    pass


class Welford:
    """Streaming mean / variance (Chan et al. batch merge), float32 accumulators."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, np.float32)
        self.m2 = np.zeros(shape, np.float32)
        self.n_pos = np.zeros(shape, np.float32)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, np.float32)
        nb = batch.shape[0]
        if nb == 0:
            return
        mb = batch.mean(0, dtype=np.float32)
        m2b = ((batch - mb) ** 2).sum(0, dtype=np.float32)
        tot = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / tot)
        self.m2 = self.m2 + m2b + delta ** 2 * (self.n * nb / tot)
        self.n_pos += (batch > 0).sum(0)
        self.n = tot

    @property
    def var(self) -> np.ndarray:
        """Sample variance (ddof=1)."""
        return self.m2 / max(self.n - 1, 1)

    @property
    def frac_pos(self) -> np.ndarray:
        return self.n_pos / max(self.n, 1)


def reduce_masked(x: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """[B,T,D] x [B,T] -> [B,D] in float32. Rows with an empty mask come back NaN."""
    x = x.float()
    m = mask.to(x.device)
    cnt = m.sum(1)
    if how == "mean":
        out = (x * m[..., None]).sum(1) / cnt.clamp(min=1)[:, None]
    elif how == "max":
        out = x.masked_fill(~m[..., None], float("-inf")).amax(1)
    elif how == "last":
        idx = (m * torch.arange(m.shape[1], device=x.device)).argmax(1)
        out = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        raise ValueError(how)
    return out.masked_fill((cnt == 0)[:, None], float("nan"))


@dataclass
class SetRecording:
    """Per-prompt collapsed activations for one prompt list."""
    texts: list[str]
    acts: np.ndarray            # float16 [n, L, d_mlp]
    resid: np.ndarray | None    # float32 [n, L, d_model], mean over masked tokens
    valid: np.ndarray           # bool [n]: prompt had at least one masked token
    mean: np.ndarray            # float32 [L, d_mlp]
    var: np.ndarray
    frac_pos: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.valid.sum())

    def valid_acts(self) -> np.ndarray:
        return self.acts[self.valid]


def _progress_tracker(total: int, cb: Callable | None):
    t0 = time.time()

    def tick(done: int):
        if cb:
            el = time.time() - t0
            cb(done, total, el * (total - done) / max(done, 1))

    return tick


def record_set(lab, texts: list[str], system: str | None = None, mask: str = "user_content",
               reduction: str = "mean", keywords=None, responses: list[str] | None = None,
               want_resid: bool = True, batch_size: int | None = None,
               progress: Callable | None = None, cancel: Callable[[], bool] | None = None) -> SetRecording:
    """Run `texts` through the model with read-only hooks on every neuron site."""
    assert reduction in REDUCTIONS
    bs = batch_size or lab.batch_sizes["record"]
    ad, L = lab.adapter, lab.adapter.n_layers
    dm = [ad.d_mlp(i) for i in range(L)]
    assert len(set(dm)) == 1, "uneven d_mlp across layers is not supported by the dense storage"
    D = dm[0]
    n = len(texts)
    acts = np.zeros((n, L, D), np.float16)
    resid = np.zeros((n, L, ad.d_model), np.float32) if want_resid else None
    valid = np.zeros(n, bool)
    wf = Welford((L, D))
    tick = _progress_tracker(n, progress)

    for s in range(0, n, bs):
        if cancel and cancel():
            raise Cancelled()
        e = encode_chat(lab.tokenizer, texts[s:s + bs], system, mask, None if responses is None else responses[s:s + bs],
                        keywords).to(lab.device)
        got_a, got_r, handles = {}, {}, []
        for i in range(L):
            def pre(mod, args, i=i):
                got_a[i] = reduce_masked(args[0], e.token_mask, reduction)

            handles.append(ad.neuron_site(i).register_forward_pre_hook(pre))
            if want_resid:
                def post(mod, args, out, i=i):
                    from .adapters import residual_tensor
                    got_r[i] = reduce_masked(residual_tensor(out), e.token_mask, "mean")

                handles.append(ad.residual_site(i).register_forward_hook(post))
        try:
            forward_logits(lab, e.input_ids, e.attention_mask)
        finally:
            for h in handles:
                h.remove()
        b = e.input_ids.shape[0]
        batch = torch.stack([got_a[i] for i in range(L)], 1).clamp(-FP16_MAX, FP16_MAX)
        batch16 = batch.to(torch.float16).cpu().numpy()
        ok = ~np.isnan(batch16[:, 0, 0])
        acts[s:s + b] = batch16
        valid[s:s + b] = ok
        if want_resid:
            resid[s:s + b] = torch.stack([got_r[i] for i in range(L)], 1).cpu().numpy()
        wf.update(batch16[ok].astype(np.float32))  # stats of the *stored* (rounded) values
        tick(min(s + bs, n))

    return SetRecording(texts=list(texts), acts=acts, resid=resid, valid=valid, mean=wf.mean, var=wf.var,
                        frac_pos=wf.frac_pos,
                        meta=dict(mask=mask, reduction=reduction, system=system, batch_size=bs, n=int(valid.sum())))


# --- calibration: r_L and control means -------------------------------------

@torch.no_grad()
def compute_resid_norms(lab, texts: list[str], system: str | None = None, batch_size: int | None = None) -> np.ndarray:
    """r_L: median L2 norm of the residual after layer L over user-content tokens, first token excluded."""
    from .adapters import residual_tensor

    bs = batch_size or lab.batch_sizes["record"]
    L = lab.adapter.n_layers
    norms: list[list[torch.Tensor]] = [[] for _ in range(L)]
    for s in range(0, len(texts), bs):
        e = encode_chat(lab.tokenizer, texts[s:s + bs], system, "user_content").to(lab.device)
        handles = []
        for i in range(L):
            def post(mod, args, out, i=i):
                h = residual_tensor(out).float()
                norms[i].append(h.norm(dim=-1)[e.token_mask].cpu())

            handles.append(lab.adapter.residual_site(i).register_forward_hook(post))
        try:
            forward_logits(lab, e.input_ids, e.attention_mask)
        finally:
            for h in handles:
                h.remove()
    return np.array([torch.cat(n).median().item() for n in norms], np.float32)


def compute_control_means(lab, texts: list[str], system: str | None = None, batch_size: int | None = None) -> np.ndarray:
    """Per-neuron mean activation on control prompts (mean of per-prompt user-content means)."""
    rec = record_set(lab, texts, system, "user_content", "mean", want_resid=False, batch_size=batch_size)
    return rec.mean


def set_calibration(lab, resid_norm: np.ndarray, control_means: np.ndarray) -> None:
    lab.cache["resid_norm"] = np.asarray(resid_norm, np.float32)
    lab.cache["control_means"] = torch.as_tensor(control_means, dtype=torch.float32, device=lab.device)


@torch.no_grad()
def token_activations(lab, text: str, layer: int, neuron: int, system: str | None = None):
    """Per-token activation of one neuron on one prompt, for the token-highlight view (computed on demand)."""
    e = encode_chat(lab.tokenizer, [text], system, "all_non_special").to(lab.device)
    got = {}
    h = lab.adapter.neuron_site(layer).register_forward_pre_hook(lambda m, a: got.update(x=a[0][0, :, neuron].float().cpu()))
    try:
        forward_logits(lab, e.input_ids, e.attention_mask)
    finally:
        h.remove()
    toks = lab.tokenizer.convert_ids_to_tokens(e.input_ids[0].tolist())
    user = encode_chat(lab.tokenizer, [text], system, "user_content")
    return toks, got["x"].numpy(), user.token_mask[0].numpy()
