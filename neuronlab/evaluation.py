"""Evaluation harness: probe log-probs, KL, perplexity, selectivity, random baselines, generation diffs.

Every comparison (baseline, intervention, each random control) goes through the same functions with
the same batch size and ordering, so numbers are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .interventions import Entry, InterventionSpec, ResidualEdit, intervene
from .prompts import Probe, PromptSet, encode_chat
from .runtime import clean_reply, forward_logits, generate_ids

GROUPS = ("pos", "neg", "ctl")
RATIO_UNSTABLE_BELOW = 0.05  # |delta_neg| below this (nats) and the ratio is flagged as unstable
Z_NOT_DISTINGUISHABLE = 2.0


# --- scoring token sequences ---------------------------------------------------------------

@torch.no_grad()
def sequence_logprobs(lab, seqs: list[list[int]], starts: list[int], batch_size: int | None = None,
                      spec: InterventionSpec | None = None):
    """Sum and count of log P(token) over positions >= start for each sequence (teacher forced).

    `starts[i]` is the first scored token of sequence i; it is also the prompt/generated boundary used
    by the prompt_only / generated_only scopes.
    """
    bs = batch_size or lab.batch_sizes["eval"]
    pad = lab.tokenizer.pad_token_id
    sums, counts = np.zeros(len(seqs)), np.zeros(len(seqs), int)
    for s in range(0, len(seqs), bs):
        chunk = seqs[s:s + bs]
        T = max(len(q) for q in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        boundary = torch.zeros(len(chunk), dtype=torch.long)
        for b, q in enumerate(chunk):
            ids[b, T - len(q):] = torch.tensor(q)
            att[b, T - len(q):] = 1
            boundary[b] = T - len(q) + starts[s + b]
        with intervene(lab, spec, boundary=boundary):
            logits = forward_logits(lab, ids, att, boundary=boundary)
        lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:].to(lab.device)[..., None])[..., 0]
        pos = torch.arange(1, T, device=lab.device)[None, :]
        sel = (pos >= boundary.to(lab.device)[:, None]) & att[:, 1:].bool().to(lab.device)
        sums[s:s + len(chunk)] = (lp * sel).sum(1).cpu().numpy()
        counts[s:s + len(chunk)] = sel.sum(1).cpu().numpy()
    return sums, counts


def probe_sequences(lab, probes: list[Probe]):
    seqs, starts = [], []
    tok = lab.tokenizer
    for p in probes:
        prompt_ids = tok(p.prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(p.prompt + p.target, add_special_tokens=False)["input_ids"]
        k = 0
        while k < min(len(prompt_ids), len(full_ids)) and prompt_ids[k] == full_ids[k]:
            k += 1  # tokenisation can merge across the boundary; score everything after the shared prefix
        seqs.append(full_ids)
        starts.append(max(k, 1))
    return seqs, starts


def probe_logprobs(lab, probes: list[Probe], spec=None, batch_size=None) -> np.ndarray:
    """log P(full target | prompt) per probe (sum over target tokens)."""
    seqs, starts = probe_sequences(lab, probes)
    sums, _ = sequence_logprobs(lab, seqs, starts, batch_size, spec)
    return sums


def perplexity(lab, texts: list[str], spec=None, batch_size=None, max_len: int = 256) -> float:
    seqs = [lab.tokenizer(t, add_special_tokens=False)["input_ids"][:max_len] for t in texts]
    keep = [i for i, q in enumerate(seqs) if len(q) > 1]
    sums, counts = sequence_logprobs(lab, [seqs[i] for i in keep], [1] * len(keep), batch_size, spec)
    return float(np.exp(-sums.sum() / max(counts.sum(), 1)))


@torch.no_grad()
def next_token_logprobs(lab, prompts: list[str], system=None, spec=None, batch_size=None) -> np.ndarray:
    """log-softmax of the next-token distribution at the end of each chat prompt: [n, V] float32."""
    bs = batch_size or lab.batch_sizes["eval"]
    out = []
    for s in range(0, len(prompts), bs):
        e = encode_chat(lab.tokenizer, prompts[s:s + bs], system, "last_token")
        with intervene(lab, spec, boundary=None):
            logits = forward_logits(lab, e.input_ids, e.attention_mask)
        out.append(torch.log_softmax(logits[:, -1].float(), -1).cpu().numpy())
    return np.concatenate(out)


def mean_kl(base_lp: np.ndarray, new_lp: np.ndarray) -> float:
    """Mean over prompts of KL(base || new)."""
    p = np.exp(base_lp)
    return float((p * (base_lp - new_lp)).sum(-1).mean())


# --- evaluation data & result ---------------------------------------------------------------

@dataclass
class EvalData:
    """Everything an evaluation needs, with the baseline computed once and reused."""
    probes: dict                    # group -> list[(split, Probe)]
    kl_prompts: list[str]           # held-out prompts (pos + neg)
    control_text: list[str]
    system: str | None
    batch_size: int
    seed: int = 0
    _base: dict = field(default_factory=dict)

    @classmethod
    def from_promptset(cls, ps: PromptSet, heldout: PromptSet, batch_size: int, probe_splits=("train", "heldout"),
                       seed: int = 0, max_kl_prompts: int = 24) -> "EvalData":
        def tag(train, held):
            out = []
            if "train" in probe_splits:
                out += [("train", p) for p in train]
            if "heldout" in probe_splits:
                out += [("heldout", p) for p in held]
            return out

        probes = {"pos": tag(ps.probes_positive, ps.probes_positive_heldout),
                  "neg": tag(ps.probes_negative, ps.probes_negative_heldout),
                  "ctl": [("heldout", p) for p in ps.probes_control]}
        kl = (heldout.positive + heldout.negative)[:max_kl_prompts]
        return cls(probes, kl, ps.control_text, ps.system_prompt, batch_size, seed)

    def all_probes(self):
        return [(g, sp, p) for g in GROUPS for sp, p in self.probes[g]]

    def baseline(self, lab):
        if not self._base:
            self._base = self._measure(lab, None)
        return self._base

    def _measure(self, lab, spec):
        flat = self.all_probes()
        lp = probe_logprobs(lab, [p for _, _, p in flat], spec, self.batch_size) if flat else np.zeros(0)
        out = {"probe": lp,
               "ppl": perplexity(lab, self.control_text, spec, self.batch_size) if self.control_text else float("nan"),
               "next": next_token_logprobs(lab, self.kl_prompts, self.system, spec, self.batch_size)
               if self.kl_prompts else None}
        return out


@dataclass
class EvalResult:
    delta: dict            # group -> mean change in log P(target)
    base: dict
    after: dict
    kl: float
    ppl_base: float
    ppl_after: float
    per_probe: list[dict]
    n: dict
    meta: dict

    @property
    def ppl_change(self) -> float:
        return self.ppl_after - self.ppl_base

    def selectivity_ratio(self):
        return selectivity_ratio(self.delta["pos"], self.delta["neg"])

    def summary(self) -> dict:
        return {"d_pos": self.delta["pos"], "d_neg": self.delta["neg"], "d_ctl": self.delta["ctl"],
                "kl": self.kl, "ppl_change": self.ppl_change}


def selectivity_ratio(d_pos: float, d_neg: float):
    """Delta_pos / Delta_neg, or None when |Delta_neg| is too small for the ratio to mean anything.

    Both deltas are log-prob changes, so ablating a concept should make d_pos strongly negative.
    A positive ratio means both groups moved the same way; a larger magnitude means more selective
    for the positive concept. The ratio explodes as d_neg -> 0, so it is withheld below 0.05 nats.
    """
    if abs(d_neg) < RATIO_UNSTABLE_BELOW:
        return None
    return d_pos / d_neg


def evaluate(lab, ev: EvalData, spec: InterventionSpec) -> EvalResult:
    base = ev.baseline(lab)
    after = ev._measure(lab, spec)
    flat = ev.all_probes()
    rows, delta, b_mean, a_mean, n = [], {}, {}, {}, {}
    for g in GROUPS:
        idx = [i for i, (gg, _, _) in enumerate(flat) if gg == g]
        n[g] = len(idx)
        if not idx:
            delta[g] = b_mean[g] = a_mean[g] = float("nan")
            continue
        b, a = base["probe"][idx], after["probe"][idx]
        delta[g], b_mean[g], a_mean[g] = float((a - b).mean()), float(b.mean()), float(a.mean())
        for i in idx:
            _, sp, p = flat[i]
            rows.append(dict(group=g, split=sp, prompt=p.prompt, target=p.target, base=float(base["probe"][i]),
                             after=float(after["probe"][i]), delta=float(after["probe"][i] - base["probe"][i])))
    kl = mean_kl(base["next"], after["next"]) if base["next"] is not None else float("nan")
    return EvalResult(delta, b_mean, a_mean, kl, base["ppl"], after["ppl"], rows,
                      n={**n, "kl_prompts": len(ev.kl_prompts), "control_text": len(ev.control_text)},
                      meta=dict(batch_size=ev.batch_size, n_neurons=spec.n_neurons, seed=ev.seed, n_residual_edits=len(spec.residual)))


# --- random baseline --------------------------------------------------------------------------

def random_like(lab, spec: InterventionSpec, seed: int) -> InterventionSpec:
    """Same per-layer counts, mode and positions; neurons drawn uniformly from the non-selected ones."""
    rng = np.random.default_rng(seed)
    entries = []
    for e in spec.entries:
        d = lab.adapter.d_mlp(e.layer)
        taken = {n for x in spec.entries if x.layer == e.layer for n in x.neurons}
        pool = np.setdiff1d(np.arange(d), np.fromiter(taken, int))
        k = min(len(e.neurons), len(pool))
        entries.append(Entry(e.layer, rng.choice(pool, size=k, replace=False).tolist(), e.mode, e.positions))
    residual = []
    for r in spec.residual:  # control for a direction: random unit directions, same layers / strength / positions
        v = rng.normal(size=r.vectors.shape).astype(np.float32)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        residual.append(ResidualEdit(list(r.layers), torch.as_tensor(v), r.kind, list(r.coef), r.positions,
                                     label=f"random {r.label}"))
    return InterventionSpec(entries, residual)


@dataclass
class BaselineComparison:
    real: EvalResult
    random: list[EvalResult]
    stats: dict           # metric -> dict(real, mean, std, z, verdict)
    seeds: list[int]

    def table(self) -> list[dict]:
        return [dict(metric=k, **v) for k, v in self.stats.items()]


def verdict(real: float, mean: float, std: float) -> str:
    if not np.isfinite(std) or std < 1e-9:
        return "random std is ~0: z undefined"
    if abs(real - mean) <= Z_NOT_DISTINGUISHABLE * std:
        return "not distinguishable from random neurons"
    return "differs from random neurons"


def compare_to_random(lab, ev: EvalData, spec: InterventionSpec, n_seeds: int = 5, seed0: int = 0,
                      progress=None) -> BaselineComparison:
    real = evaluate(lab, ev, spec)
    seeds = [seed0 + i for i in range(n_seeds)]
    rand = []
    for i, s in enumerate(seeds):
        rand.append(evaluate(lab, ev, random_like(lab, spec, s)))
        if progress:
            progress(i + 1, n_seeds, 0.0)
    stats = {}
    for key in ("d_pos", "d_neg", "d_ctl", "kl", "ppl_change"):
        r = real.summary()[key]
        vals = np.array([x.summary()[key] for x in rand])
        mean = float(np.nanmean(vals)) if len(vals) else float("nan")
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        z = (r - mean) / std if np.isfinite(std) and std > 1e-9 else None
        stats[key] = dict(real=r, rand_mean=mean, rand_std=std, z=z, n_seeds=len(vals),
                          verdict=verdict(r, mean, std))
    return BaselineComparison(real, rand, stats, seeds)


# --- ablation curve --------------------------------------------------------------------------------

KS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def ablation_curve(lab, ev: EvalData, ranked, mode, positions="all", ks=KS, n_random: int = 5, seed0: int = 0,
                   progress=None) -> dict:
    """Ablate the top-k of `ranked` ((layer, neuron, ...) in rank order) for each k, each with a random baseline."""
    ks = [k for k in ks if k <= len(ranked)]
    out = dict(ks=ks, mode=str(mode), positions=positions, n_random=n_random, seed0=seed0,
               real={m: [] for m in ("d_pos", "d_neg", "d_ctl", "kl", "ppl_change")},
               rand_mean={m: [] for m in ("d_pos", "d_neg", "d_ctl", "kl", "ppl_change")},
               rand_std={m: [] for m in ("d_pos", "d_neg", "d_ctl", "kl", "ppl_change")}, z={m: [] for m in ("d_pos", "d_neg", "d_ctl")})
    for i, k in enumerate(ks):
        spec = InterventionSpec.from_neurons([(l, n) for l, n, *_ in ranked[:k]], mode, positions)
        c = compare_to_random(lab, ev, spec, n_random, seed0)
        for m, st in c.stats.items():
            out["real"][m].append(st["real"]); out["rand_mean"][m].append(st["rand_mean"])
            out["rand_std"][m].append(0.0 if not np.isfinite(st["rand_std"]) else st["rand_std"])
            if m in out["z"]:
                out["z"][m].append(st["z"])
        if progress:
            progress(i + 1, len(ks), 0.0)
    out["n"] = {g: len(ev.probes[g]) for g in GROUPS}
    return out


# --- generation ---------------------------------------------------------------------------------

def repetition_rate(ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (0 when the reply is shorter than n)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def generate_replies(lab, prompts: list[str], spec=None, system=None, max_new_tokens=160, batch_size=None):
    """Greedy replies (text, token ids), batched identically for every spec."""
    bs = batch_size or lab.batch_sizes["generate"]
    out = []
    for s in range(0, len(prompts), bs):
        e = encode_chat(lab.tokenizer, prompts[s:s + bs], system, "last_token")
        with intervene(lab, spec):
            gen = generate_ids(lab, e.input_ids, e.attention_mask, max_new_tokens)
        out += [clean_reply(lab, g) for g in gen.cpu()]
    return out


def generate_replies_rows(lab, prompts: list[str], specs: list, system=None, max_new_tokens=160):
    """One batch where row i is generated under specs[i] (per-row interventions; a spec may be None).

    Rows share one batch, so a strong push on one row can shift numerics of the others slightly.
    """
    e = encode_chat(lab.tokenizer, prompts, system, "last_token")
    with intervene(lab, [s if s is not None else InterventionSpec() for s in specs]):
        gen = generate_ids(lab, e.input_ids, e.attention_mask, max_new_tokens)
    return [clean_reply(lab, g) for g in gen.cpu()]


@torch.no_grad()
def reply_perplexity(lab, prompts: list[str], reply_ids: list[list[int]], system=None, batch_size=None) -> np.ndarray:
    """Per-reply perplexity under the *unsteered* model."""
    seqs, starts, cache = [], [], {}
    for p, r in zip(prompts, reply_ids):
        if p not in cache:  # sweeps score thousands of replies to the same prompt
            e = encode_chat(lab.tokenizer, [p], system, "last_token")
            cache[p] = e.input_ids[0][e.attention_mask[0].bool()].tolist()
        seqs.append(cache[p] + r)
        starts.append(len(cache[p]))
    sums, counts = sequence_logprobs(lab, seqs, starts, batch_size, None)
    return np.where(counts > 0, np.exp(-sums / np.maximum(counts, 1)), np.nan)


def generation_diff(lab, prompts: list[str], spec, system=None, max_new_tokens=160, batch_size=None) -> list[dict]:
    """Baseline vs intervention for the given prompts, in the order given (no cherry-picking)."""
    base = generate_replies(lab, prompts, None, system, max_new_tokens, batch_size)
    new = generate_replies(lab, prompts, spec, system, max_new_tokens, batch_size)
    ppl_b = reply_perplexity(lab, prompts, [b[1] for b in base], system, batch_size)
    ppl_n = reply_perplexity(lab, prompts, [n[1] for n in new], system, batch_size)
    return [dict(prompt=p, baseline=b[0], intervened=n[0], rep_base=repetition_rate(b[1]),
                 rep_new=repetition_rate(n[1]), ppl_base=float(pb), ppl_new=float(pn))
            for p, b, n, pb, pn in zip(prompts, base, new, ppl_b, ppl_n)]
