"""Neuron interventions as forward pre-hooks on the MLP down-projection input.

A hook sees x = act_fn(gate(h)) * up(h), shape [B, T, d_mlp], and returns x * mult + add, where
mult / add are dense [R, d_mlp] tensors (R = 1 for a spec shared by the whole batch, R = B when
every row carries its own spec). All modes are expressed that way:

    zero      mult 0, add 0
    mean      mult 0, add = mean activation on control text
    clamp(v)  mult 0, add v
    scale(f)  mult f, add 0
    push      mult 1, add c = sign * alpha * r_L / ||W_down[:, n]||   (== adding c * W_down[:, n]
              to the residual stream after the MLP; see tests/test_interventions.py)
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

POSITIONS = ("all", "prompt_only", "generated_only")


@dataclass(frozen=True)
class Mode:
    kind: str
    value: float = 0.0  # clamp value, scale factor or push alpha
    sign: int = 1       # push direction

    @staticmethod
    def zero():
        return Mode("zero")

    @staticmethod
    def mean():
        return Mode("mean")

    @staticmethod
    def clamp(value: float):
        return Mode("clamp", float(value))

    @staticmethod
    def scale(factor: float):
        return Mode("scale", float(factor))

    @staticmethod
    def push(alpha: float, sign: int = 1):
        return Mode("push", float(alpha), 1 if sign >= 0 else -1)

    def __str__(self):
        return {"zero": "zero", "mean": "mean", "clamp": f"clamp({self.value})", "scale": f"scale({self.value})",
                "push": f"push(alpha={self.value}, sign={self.sign:+d})"}[self.kind]


@dataclass
class Entry:
    layer: int
    neurons: list[int]
    mode: Mode = field(default_factory=Mode.zero)
    positions: str = "all"

    def __post_init__(self):
        assert self.positions in POSITIONS, self.positions
        self.neurons = [int(n) for n in self.neurons]


@dataclass
class ResidualEdit:
    """Edit of the residual stream after the given layers (steering / directional ablation).

    kind="add":          h <- h + coef[l] * u[l]            (steering; coef = alpha * r_L)
    kind="project_out":  h <- h - (h . u[l]) u[l]           (directional ablation)
    vectors are unit vectors, shape [len(layers), d_model].
    """
    layers: list[int]
    vectors: torch.Tensor
    kind: str = "add"
    coef: list[float] = field(default_factory=list)
    positions: str = "all"
    label: str = ""

    def __post_init__(self):
        assert self.kind in ("add", "project_out") and self.positions in POSITIONS
        self.vectors = torch.as_tensor(self.vectors, dtype=torch.float32)
        if self.kind == "add" and len(self.coef) != len(self.layers):
            raise ValueError("kind='add' needs one coef per layer")


@dataclass
class InterventionSpec:
    entries: list[Entry] = field(default_factory=list)
    residual: list[ResidualEdit] = field(default_factory=list)

    def __bool__(self):
        return bool(self.entries) or bool(self.residual)

    @property
    def n_neurons(self) -> int:
        return sum(len(e.neurons) for e in self.entries)

    def per_layer_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self.entries:
            out[e.layer] = out.get(e.layer, 0) + len(e.neurons)
        return out

    @classmethod
    def from_neurons(cls, neurons: list[tuple[int, int]], mode: Mode, positions: str = "all") -> "InterventionSpec":
        """Build a spec from (layer, neuron) pairs."""
        by_layer: dict[int, list[int]] = {}
        for layer, n in neurons:
            by_layer.setdefault(int(layer), []).append(int(n))
        return cls([Entry(l, ns, mode, positions) for l, ns in sorted(by_layer.items())])

    def neuron_pairs(self) -> list[tuple[int, int]]:
        return [(e.layer, n) for e in self.entries for n in e.neurons]


class InterventionState:
    """Tracks where we are in a (possibly KV-cached) sequence so position scopes work."""

    def __init__(self, boundary=None):
        self.boundary = boundary  # int, or LongTensor [B]: first "generated" index in tensor coordinates
        self.seen: dict[int, int] = {}

    def reset(self, boundary=None):
        self.seen = {}
        if boundary is not None:
            self.boundary = boundary


@dataclass
class _Op:
    mult: torch.Tensor  # [R, d_mlp]
    add: torch.Tensor   # [R, d_mlp]
    scope: str


def _coefficients(lab, entry: Entry):
    """(mult, add) vectors over entry.neurons."""
    n = torch.tensor(entry.neurons, dtype=torch.long, device=lab.device)
    m = entry.mode
    k = len(entry.neurons)
    ones, zeros = torch.ones(k, device=lab.device), torch.zeros(k, device=lab.device)
    if m.kind == "zero":
        return n, zeros, zeros
    if m.kind == "mean":
        cm = lab.cache.get("control_means")
        if cm is None:
            raise RuntimeError("mean mode needs lab.cache['control_means'] (see recording.compute_control_means)")
        return n, zeros, cm[entry.layer].to(lab.device)[n].float()
    if m.kind == "clamp":
        return n, zeros, torch.full((k,), m.value, device=lab.device)
    if m.kind == "scale":
        return n, torch.full((k,), m.value, device=lab.device), zeros
    if m.kind == "push":
        r = lab.cache.get("resid_norm")
        if r is None:
            raise RuntimeError("push mode needs lab.cache['resid_norm'] (see recording.compute_resid_norms)")
        norms = lab.out_dirs(entry.layer)[n].norm(dim=-1).clamp_min(1e-8)
        return n, ones, m.sign * m.value * float(r[entry.layer]) / norms
    raise ValueError(m.kind)


def push_coefficient(lab, layer: int, neuron: int, alpha: float, sign: int) -> float:
    """c in  h <- h + c * W_down[:, n]  (and equivalently a_n <- a_n + c)."""
    norm = float(lab.out_dirs(layer)[neuron].norm())
    return sign * alpha * float(lab.cache["resid_norm"][layer]) / max(norm, 1e-8)


def compile_spec(lab, specs: list[InterventionSpec]) -> dict[int, list[_Op]]:
    """Dense per-layer ops. `specs` has length 1 (broadcast) or B (one per batch row)."""
    R = len(specs)
    ops: dict[tuple[int, str], list[torch.Tensor]] = {}
    for r, spec in enumerate(specs):
        for e in spec.entries:
            key = (e.layer, e.positions)
            if key not in ops:
                d = lab.adapter.d_mlp(e.layer)
                ops[key] = [torch.ones(R, d, device=lab.device), torch.zeros(R, d, device=lab.device)]
            mult, add = ops[key]
            n, em, ea = _coefficients(lab, e)
            # compose with whatever earlier entries did to these neurons: (x*m1+a1)*m2+a2
            add[r, n] = add[r, n] * em + ea
            mult[r, n] = mult[r, n] * em
    out: dict[int, list[_Op]] = {}
    for (layer, scope), (mult, add) in ops.items():
        out.setdefault(layer, []).append(_Op(mult, add, scope))
    return out


def _make_hook(layer: int, ops: list[_Op], state: InterventionState, n_rows: int):
    def hook(module, args):
        x = args[0]
        B, T = x.shape[0], x.shape[1]
        if T > 1:  # a multi-token forward is always the start of a new sequence
            state.seen[layer] = 0
        start = state.seen.get(layer, 0)
        state.seen[layer] = start + T
        for op in ops:
            if op.mult.shape[0] not in (1, B):
                raise ValueError(f"per-row spec has {op.mult.shape[0]} rows but the batch has {B}")
            new = x * op.mult.to(x.dtype)[:, None, :] + op.add.to(x.dtype)[:, None, :]
            if op.scope == "all":
                x = new
                continue
            pos = start + torch.arange(T, device=x.device)
            b = state.boundary
            if b is None:
                gen = torch.zeros(1, T, dtype=torch.bool, device=x.device)
            else:
                b = torch.as_tensor(b, device=x.device).reshape(-1, 1)
                gen = pos[None, :] >= b
            sel = gen if op.scope == "generated_only" else ~gen
            x = torch.where(sel[..., None], new, x)
        return (x,) + tuple(args[1:])

    return hook


def _scope_mask(state: InterventionState, scope: str, start: int, T: int, device):
    """[B or 1, T] bool of positions this scope applies to, or None for 'all'."""
    if scope == "all":
        return None
    pos = start + torch.arange(T, device=device)
    b = state.boundary
    if b is None:
        gen = torch.zeros(1, T, dtype=torch.bool, device=device)
    else:
        gen = pos[None, :] >= torch.as_tensor(b, device=device).reshape(-1, 1)
    return gen if scope == "generated_only" else ~gen


def _make_residual_hook(lab, key, edit_layer_idx: int, edit: ResidualEdit, state: InterventionState):
    from .adapters import replace_residual, residual_tensor

    u = edit.vectors[edit_layer_idx].to(lab.device)
    c = edit.coef[edit_layer_idx] if edit.kind == "add" else 0.0

    def hook(module, args, out):
        h = residual_tensor(out)
        T = h.shape[1]
        if T > 1:
            state.seen[key] = 0
        start = state.seen.get(key, 0)
        state.seen[key] = start + T
        if edit.kind == "add":
            new = h + (c * u).to(h.dtype)
        else:
            uu = u.to(torch.float32)
            new = (h.float() - (h.float() @ uu)[..., None] * uu).to(h.dtype)
        sel = _scope_mask(state, edit.positions, start, T, h.device)
        if sel is not None:
            new = torch.where(sel[..., None], new, h)
        return replace_residual(out, new)

    return hook


@contextmanager
def intervene(lab, spec, boundary=None):
    """`with intervene(lab, spec): ...` Removes every hook on exit, including on exceptions.

    spec: InterventionSpec (applies to every row), a list of specs (one per row), or None/empty.
    boundary: int or LongTensor[B], first index (in padded tensor coordinates) that counts as
    "generated" for the prompt_only / generated_only scopes. lab.generate sets this itself.
    """
    specs = spec if isinstance(spec, (list, tuple)) else [spec if spec is not None else InterventionSpec()]
    state = InterventionState(boundary)
    handles = []
    try:
        for layer, ops in compile_spec(lab, list(specs)).items():
            site = lab.adapter.neuron_site(layer)
            handles.append(site.register_forward_pre_hook(_make_hook(layer, ops, state, len(specs))))
        if any(sp.residual for sp in specs):
            if len(specs) != 1:
                raise NotImplementedError("residual edits are not supported in per-row specs")
            for ei, edit in enumerate(specs[0].residual):
                for li, layer in enumerate(edit.layers):
                    handles.append(lab.adapter.residual_site(layer).register_forward_hook(
                        _make_residual_hook(lab, ("r", ei, layer), li, edit, state)))
        lab.cache.setdefault("_active", []).append(state)
        try:
            yield state
        finally:
            lab.cache["_active"].remove(state)
    finally:
        for h in handles:
            h.remove()
