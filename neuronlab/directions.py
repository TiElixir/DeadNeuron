"""Concept / persona directions in the residual stream: estimation, steering, directional ablation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .interventions import InterventionSpec, ResidualEdit


@dataclass
class Directions:
    raw: np.ndarray    # [L, d_model] mean(positive) - mean(negative) residual, per layer
    unit: np.ndarray   # [L, d_model]
    norm: np.ndarray   # [L]
    n_pos: int
    n_neg: int


def compute_directions(pos_resid: np.ndarray, neg_resid: np.ndarray) -> Directions:
    """Difference of means of per-prompt residual means ([n, L, d_model] each), training split only."""
    raw = pos_resid.mean(0) - neg_resid.mean(0)
    norm = np.linalg.norm(raw, axis=-1)
    return Directions(raw.astype(np.float32), (raw / np.maximum(norm, 1e-12)[:, None]).astype(np.float32),
                      norm.astype(np.float32), len(pos_resid), len(neg_resid))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney U)."""
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = csum - (cnt - 1) / 2.0  # average 1-based rank of each tie group
    ranks = avg_rank[inv]
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def layer_separation(dirs: Directions, pos_resid: np.ndarray, neg_resid: np.ndarray) -> np.ndarray:
    """Held-out AUROC of the projection onto each layer's direction ([n, L, d_model] held-out residual means)."""
    return np.array([auroc(pos_resid[:, l] @ dirs.unit[l], neg_resid[:, l] @ dirs.unit[l])
                     for l in range(dirs.unit.shape[0])])


def best_layer(sep: np.ndarray) -> int:
    """Layer with the highest held-out AUROC (lowest index on ties)."""
    return int(np.argmax(sep))


def steering_spec(lab, dirs: Directions, layers: list[int], alpha: float, positions: str = "all",
                  source_layer: int | None = None) -> InterventionSpec:
    """h_L <- h_L + alpha * r_L * u  at each layer in `layers` (u = direction of `source_layer`, default each own)."""
    r = lab.cache["resid_norm"]
    vecs = np.stack([dirs.unit[source_layer if source_layer is not None else l] for l in layers])
    return InterventionSpec(residual=[ResidualEdit(list(layers), torch.as_tensor(vecs), "add",
                                                  [float(alpha * r[l]) for l in layers], positions,
                                                  label=f"steer alpha={alpha} layers={list(layers)}")])


def direction_ablation_spec(dirs: Directions, layers: list[int], source_layer: int, positions: str = "all") -> InterventionSpec:
    """h <- h - (h.u)u at every layer in `layers`, using the direction estimated at `source_layer`."""
    vecs = np.stack([dirs.unit[source_layer]] * len(layers))
    return InterventionSpec(residual=[ResidualEdit(list(layers), torch.as_tensor(vecs), "project_out", [], positions,
                                                  label=f"project out u[{source_layer}] at {len(layers)} layers")])


def neurons_by_alignment(lab, dirs: Directions, layer: int, k: int = 50, sign: str = "pos"):
    """Top-k (layer, neuron, cosine) over layers <= `layer` by cos(out_dir_n, u_layer)."""
    u = torch.as_tensor(dirs.unit[layer], device=lab.device)
    rows = []
    for l in range(layer + 1):
        od = lab.out_dirs(l)
        cos = (od / od.norm(dim=-1, keepdim=True).clamp_min(1e-8)) @ u
        key = cos if sign == "pos" else (-cos if sign == "neg" else cos.abs())
        v, i = key.topk(min(k, len(cos)))
        rows += [(l, int(n), float(cos[n])) for n in i.cpu()]
    rows.sort(key=lambda r: -(r[2] if sign == "pos" else (-r[2] if sign == "neg" else abs(r[2]))))
    return rows[:k]


def alignment_with(lab, layer: int, direction_unit: np.ndarray) -> np.ndarray:
    """cos(out_dir_n, direction) for every neuron of `layer`."""
    od = lab.out_dirs(layer)
    u = torch.as_tensor(direction_unit, device=lab.device)
    return ((od / od.norm(dim=-1, keepdim=True).clamp_min(1e-8)) @ u).cpu().numpy()
