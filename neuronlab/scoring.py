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
