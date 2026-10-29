"""Per-neuron scores. Every score is an array of shape [n_layers, d_mlp]."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EPS = 1e-6  # variance floor in d and t (activation units squared)


def group_stats(x: torch.Tensor):
    mean = x.mean(0)
    var = x.var(0, unbiased=True) if x.shape[0] > 1 else torch.zeros_like(mean)
    return mean, var


def cohens_d(mp, vp, mn, vn, eps: float = EPS):
    return (mp - mn) / torch.sqrt((vp + vn) / 2 + eps)


def welch_t(mp, vp, n1, mn, vn, n2, eps: float = EPS):
    return (mp - mn) / torch.sqrt(vp / n1 + vn / n2 + eps)


@torch.no_grad()
def permutation_null(pos: np.ndarray, neg: np.ndarray, n_perm: int = 500, seed: int = 0, q: float = 0.95,
                     device: str | torch.device = "cpu", chunk: int = 25):
    """Shuffle pos/neg labels n_perm times and record the max |d| per layer each time.

    Returns (thresholds [L], null_max [n_perm, L]). All permutations of a chunk are computed with
    one matmul: group sums = membership[P, N] @ X[N, L*d].
    """
    n1, n2 = pos.shape[0], neg.shape[0]
    L, D = pos.shape[1], pos.shape[2]
    X = torch.as_tensor(np.concatenate([pos, neg]), dtype=torch.float32, device=device).reshape(n1 + n2, -1)
    X = X - X.mean(0, keepdim=True)  # d is shift invariant; centring keeps the sum-of-squares stable
    X2 = X * X
    N = n1 + n2
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for s in range(0, n_perm, chunk):
        P = min(chunk, n_perm - s)
        order = torch.rand(P, N, generator=g).argsort(1)
        M = torch.zeros(P, N)
        M.scatter_(1, order[:, :n1], 1.0)
        M = M.to(device)
        S1, Q1 = M @ X, M @ X2
        S2, Q2 = X.sum(0, keepdim=True) - S1, X2.sum(0, keepdim=True) - Q1
        m1, m2 = S1 / n1, S2 / n2
        v1 = ((Q1 - n1 * m1 ** 2) / (n1 - 1)).clamp_min(0)
        v2 = ((Q2 - n2 * m2 ** 2) / (n2 - 1)).clamp_min(0)
        d = cohens_d(m1, v1, m2, v2).abs().reshape(P, L, D)
        out.append(d.amax(-1).cpu())
    null_max = torch.cat(out).numpy()
    return np.quantile(null_max, q, axis=0).astype(np.float32), null_max


@dataclass
class ContrastResult:
    d: np.ndarray
    t: np.ndarray
    diff: np.ndarray
    mean_pos: np.ndarray
    mean_neg: np.ndarray
    threshold: np.ndarray      # [L] family-wise per-layer threshold on |d|
    significant: np.ndarray    # bool [L, D]
    generic: np.ndarray        # bool [L, D]
    null_max: np.ndarray       # [n_perm, L]
    n_pos: int
    n_neg: int
    n_control: int
    seed: int
    n_perm: int

    def arrays(self) -> dict:
        return {k: getattr(self, k) for k in ("d", "t", "diff", "mean_pos", "mean_neg", "threshold", "significant",
                                              "generic", "null_max")}

    def meta(self) -> dict:
        return dict(n_pos=self.n_pos, n_neg=self.n_neg, n_control=self.n_control, seed=self.seed,
                    n_perm=self.n_perm, split="train")

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "ContrastResult":
        return cls(**{k: arrays[k] for k in ("d", "t", "diff", "mean_pos", "mean_neg", "threshold", "significant",
                                              "generic", "null_max")}, **{k: meta[k] for k in (
                       "n_pos", "n_neg", "n_control", "seed", "n_perm")})

    def significant_per_layer(self, hide_generic: bool = True) -> np.ndarray:
        m = self.significant & ~self.generic if hide_generic else self.significant
        return m.sum(1)


def contrast_scores(pos: np.ndarray, neg: np.ndarray, control: np.ndarray | None = None, n_perm: int = 500,
                    seed: int = 0, device: str | torch.device = "cpu") -> ContrastResult:
    """Cohen's d, Welch t, permutation family-wise threshold and generic-neuron flag.

    pos / neg / control: per-prompt activations [n, L, d] from the *training split* only.
    n is the number of prompts, never the number of tokens.
    """
    P = torch.as_tensor(pos, dtype=torch.float32, device=device)
    N = torch.as_tensor(neg, dtype=torch.float32, device=device)
    mp, vp = group_stats(P)
    mn, vn = group_stats(N)
    d = cohens_d(mp, vp, mn, vn)
    t = welch_t(mp, vp, P.shape[0], mn, vn, N.shape[0])
    thr, null_max = permutation_null(pos, neg, n_perm=n_perm, seed=seed, device=device)
    significant = (d.abs().cpu().numpy() > thr[:, None])
    generic = np.zeros_like(significant)
    n_ctl = 0
    if control is not None and len(control):
        C = torch.as_tensor(control, dtype=torch.float32, device=device)
        n_ctl = C.shape[0]
        generic = ((C.mean(0) > mp) | ((C > 0).float().mean(0) > 0.9)).cpu().numpy()
    return ContrastResult(d=d.cpu().numpy(), t=t.cpu().numpy(), diff=(mp - mn).cpu().numpy(),
                          mean_pos=mp.cpu().numpy(), mean_neg=mn.cpu().numpy(), threshold=thr,
                          significant=significant, generic=generic, null_max=null_max, n_pos=P.shape[0],
                          n_neg=N.shape[0], n_control=n_ctl, seed=seed, n_perm=n_perm)


def rank_neurons(score: np.ndarray, k: int, mask: np.ndarray | None = None, sign: str = "abs"):
    """Top-k (layer, neuron, score) by `score`. sign: 'abs' ranks |score|, 'pos' ranks score, 'neg' ranks -score."""
    s = {"abs": np.abs(score), "pos": score, "neg": -score}[sign].astype(np.float64).copy()
    if mask is not None:
        s[~mask] = -np.inf
    flat = np.argsort(-s, axis=None, kind="stable")[:k]
    flat = [f for f in flat if np.isfinite(s.flat[f])]
    L, D = score.shape
    return [(int(f // D), int(f % D), float(score.flat[f])) for f in flat]


def candidate_mask(res: ContrastResult, require_significant: bool = True, hide_generic: bool = True) -> np.ndarray:
    m = np.ones_like(res.d, dtype=bool)
    if require_significant:
        m &= res.significant
    if hide_generic:
        m &= ~res.generic
    return m


# --- attribution (first-order estimate of mean-ablating each neuron) ------------------------------------

from contextlib import contextmanager  # noqa: E402


@contextmanager
def full_precision(lab):
    """Temporarily run the model in float32 (lossless round trip from bf16 / fp16)."""
    orig = next(lab.model.parameters()).dtype
    if orig == torch.float32:
        yield
        return
    lab.model.float()
    try:
        yield
    finally:
        lab.model.to(orig)


def _attribution_pass(lab, probes, abar: torch.Tensor, batch_size: int, exclude_first: bool) -> np.ndarray:
    from .evaluation import probe_sequences
    from .prompts import position_ids

    seqs, starts = probe_sequences(lab, probes)
    n_total, L = len(seqs), lab.adapter.n_layers
    D = lab.adapter.d_mlp(0)
    eff = torch.zeros(L, D, device=lab.device, dtype=torch.float32)
    pad = lab.tokenizer.pad_token_id
    for s in range(0, n_total, batch_size):
        chunk = seqs[s:s + batch_size]
        T = max(len(q) for q in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        tgt = torch.zeros((len(chunk), T), dtype=torch.bool)
        valid = torch.zeros((len(chunk), T), dtype=torch.bool)
        for b, q in enumerate(chunk):
            ids[b, T - len(q):] = torch.tensor(q)
            att[b, T - len(q):] = 1
            tgt[b, T - len(q) + starts[s + b]:] = True
            valid[b, T - len(q) + (1 if exclude_first else 0):] = True
        deltas, acts, handles = {}, {}, []
        for i in range(L):
            def pre(mod, args, i=i):
                x = args[0]
                deltas[i] = torch.zeros_like(x, requires_grad=True)  # d(M)/d(delta) = total d(M)/d(a)
                acts[i] = x.detach()
                return (x + deltas[i],) + tuple(args[1:])

            handles.append(lab.adapter.neuron_site(i).register_forward_pre_hook(pre))
        try:
            with torch.enable_grad():
                logits = lab.model(input_ids=ids.to(lab.device), attention_mask=att.to(lab.device),
                                   position_ids=position_ids(att).to(lab.device), use_cache=False).logits
                lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:].to(lab.device)[..., None])[..., 0]
                M = (lp * tgt[:, 1:].to(lab.device)).sum() / n_total  # mean over probes of log P(target)
                M.backward()
        finally:
            for h in handles:
                h.remove()
        vmask = valid.to(lab.device)[..., None]
        for i in range(L):
            eff[i] += ((abar[i].to(acts[i].dtype) - acts[i]).float() * deltas[i].grad.float() * vmask).sum((0, 1))
        del deltas, acts, logits, lp, M
    return eff.cpu().numpy()


def attribution(lab, probes, abar, batch_size: int = 2, exclude_first: bool = True, precision: str = "float32"):
    """effect_n ~ sum_positions (abar_n - a_n) * dM/da_n   with  M = mean_probes log P(target | prompt).

    A first-order (attribution-patching) estimate of the change in M from mean-ablating neuron n.
    Negative = ablation lowers M. Gradients are taken w.r.t. activations only (parameters are frozen).
    Tries float32 first and falls back to the model's own dtype on OOM. Returns (effects [L, D], precision_used).
    """
    abar = torch.as_tensor(abar, dtype=torch.float32, device=lab.device)
    if precision == "float32":
        try:
            with full_precision(lab):
                return _attribution_pass(lab, probes, abar, batch_size, exclude_first), "float32"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return _attribution_pass(lab, probes, abar, 1, exclude_first), str(lab.dtype).replace("torch.", "")


def attribution_scores(lab, probes_pos, probes_neg, abar, **kw) -> dict:
    """Attribution on the positive and negative probes, and their difference.

    attr_sel = effect_on_M_pos - effect_on_M_neg. More negative = ablation hurts the positive concept more.
    """
    pos, prec = attribution(lab, probes_pos, abar, **kw)
    neg, _ = attribution(lab, probes_neg, abar, **kw)
    return dict(attr_pos=pos, attr_neg=neg, attr_sel=pos - neg, precision=prec)


# --- weight-based scores (no forward pass) -------------------------------------------------------

@torch.no_grad()
def logit_lens(lab, layer: int, neurons: list[int], k: int = 20):
    """Top / bottom-k tokens of each neuron: unembed(final_norm.weight * out_dir_n). Returns ids, scores."""
    W_U = lab.adapter.unembed()
    g = getattr(lab.adapter.final_norm(), "weight", None)
    dirs = lab.out_dirs(layer)[torch.as_tensor(neurons, device=lab.device)]
    if g is not None:
        dirs = dirs * g.float()
    logits = (dirs.to(W_U.dtype) @ W_U.T).float()
    top, bot = logits.topk(k, dim=-1), (-logits).topk(k, dim=-1)
    return dict(top_ids=top.indices.cpu().numpy(), top_vals=top.values.cpu().numpy(),
                bottom_ids=bot.indices.cpu().numpy(), bottom_vals=(-bot.values).cpu().numpy())


def keyword_token_ids(tokenizer, keywords) -> set[int]:
    """Single-token encodings of each keyword (bare, space-prefixed, capitalised)."""
    out = set()
    for kw in keywords:
        for v in {kw, " " + kw, kw.capitalize(), " " + kw.capitalize()}:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out


@torch.no_grad()
def keyword_lens_scores(lab, keywords, layers=None, k: int = 100, chunk: int = 512) -> np.ndarray:
    """[L, D] count of concept keyword tokens among each neuron's top-k logit-lens tokens (0 elsewhere)."""
    ids = torch.tensor(sorted(keyword_token_ids(lab.tokenizer, keywords)), device=lab.device)
    L, D = lab.adapter.n_layers, lab.adapter.d_mlp(0)
    out = np.zeros((L, D), np.float32)
    for layer in (layers if layers is not None else range(L)):
        for s in range(0, D, chunk):
            top = logit_lens(lab, layer, list(range(s, min(s + chunk, D))), k)["top_ids"]
            out[layer, s:s + chunk] = np.isin(top, ids.cpu().numpy()).sum(1)
    return out


# --- combined rank ------------------------------------------------------------------------------

def zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z-score using the masked population only."""
    sel = x[mask].astype(np.float64)
    mu, sd = sel.mean(), sel.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


def combined_score(components: dict[str, np.ndarray], weights: dict[str, float] | None, mask: np.ndarray) -> np.ndarray:
    """Weighted mean of per-component z-scores computed over the candidate set; -inf outside it.

    Default (spec): equal weights on |d| and -attr_selectivity, candidates = significant & non-generic.
    """
    weights = weights or {k: 1.0 for k in components}
    tot = sum(weights.values())
    out = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    for name, w in weights.items():
        out += w * zscore(components[name], mask)
    out /= tot
    out[~mask] = -np.inf
    return out
