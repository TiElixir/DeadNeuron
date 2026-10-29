"""The experiment pipeline shared by the CLI, the examples and the UI (so all report the same numbers)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .evaluation import BaselineComparison, EvalData, compare_to_random, generation_diff
from .interventions import InterventionSpec, Mode
from .model_loader import ROOT, Lab
from .prompts import PromptSet
from .recording import (SetRecording, compute_control_means, compute_resid_norms, record_set, set_calibration)
from .scoring import ContrastResult, candidate_mask, contrast_scores, rank_neurons
from .storage import RunStore


@dataclass
class Settings:
    split_seed: int = 0
    n_per_side: int = 60
    mask: str = "user_content"
    reduction: str = "mean"
    n_perm: int = 500
    perm_seed: int = 0
    record_batch: int | None = None
    eval_batch: int | None = None

    def record_key(self, lab: Lab) -> dict:
        d = asdict(self)
        for k in ("n_perm", "perm_seed", "eval_batch"):
            d.pop(k)
        d["record_batch"] = self.record_batch or lab.batch_sizes["record"]
        d["dtype"] = str(lab.dtype)
        d["device"] = lab.device.type
        return d


@dataclass
class Experiment:
    lab: Lab
    preset: PromptSet          # expanded
    train: PromptSet
    heldout: PromptSet
    settings: Settings
    store: RunStore
    rec: dict                  # name -> SetRecording  (pos / neg / ctl, training split)
    contrast: ContrastResult
    timings: dict = field(default_factory=dict)

    def candidates(self, require_significant=True, hide_generic=True) -> np.ndarray:
        return candidate_mask(self.contrast, require_significant, hide_generic)

    def top_neurons(self, k: int, sign: str = "abs", require_significant=True, hide_generic=True):
        return rank_neurons(self.contrast.d, k, self.candidates(require_significant, hide_generic), sign)

    def eval_data(self, probe_splits=("train", "heldout")) -> EvalData:
        return EvalData.from_promptset(self.preset, self.heldout, self.settings.eval_batch or self.lab.batch_sizes["eval"],
                                       probe_splits, seed=self.settings.split_seed)


def _cached_recording(store: RunStore, name: str, timings: dict, fn) -> SetRecording:
    rec = store.load_recording(name)
    if rec is None:
        t0 = time.time()
        rec = fn()
        timings[f"record_{name}_s"] = time.time() - t0
        store.save_recording(name, rec)
    return rec


def prepare(lab: Lab, preset: PromptSet | str | Path, settings: Settings | None = None,
            runs_dir: str | Path | None = None, progress=None, cancel=None) -> Experiment:
    """Record (training split) -> calibrate -> contrast scores. Everything is cached on disk."""
    settings = settings or Settings()
    ps = preset if isinstance(preset, PromptSet) else PromptSet.from_yaml(preset)
    ps = ps.expand(settings.n_per_side, settings.split_seed)
    train, held = ps.split(settings.split_seed)
    store = RunStore(runs_dir or ROOT / "runs", lab.fingerprint, ps.hash(), settings.record_key(lab))
    timings: dict = {}
    sys_p, bs = ps.system_prompt, settings.record_batch

    recs = {}
    for name, texts in (("pos", train.positive), ("neg", train.negative), ("ctl", train.control)):
        recs[name] = _cached_recording(store, name, timings, lambda texts=texts: record_set(
            lab, texts, sys_p, settings.mask, settings.reduction, ps.keywords, None, True, bs, progress, cancel))

    cal = store.load_arrays("calibration")
    if cal is None:
        t0 = time.time()
        resid = compute_resid_norms(lab, train.control, sys_p, bs)
        # mean-ablation value: per-neuron mean on control prompts over user-content tokens
        cmeans = (recs["ctl"].mean if settings.mask == "user_content" and settings.reduction == "mean"
                  else compute_control_means(lab, train.control, sys_p, bs))
        store.save_arrays("calibration", {"resid_norm": resid, "control_means": cmeans}, {"n_control": len(train.control)})
        timings["calibration_s"] = time.time() - t0
    else:
        resid, cmeans = cal[0]["resid_norm"], cal[0]["control_means"]
    set_calibration(lab, resid, cmeans)

    cname = f"contrast_p{settings.n_perm}_s{settings.perm_seed}"
    got = store.load_arrays(cname)
    if got is None:
        t0 = time.time()
        res = contrast_scores(recs["pos"].valid_acts(), recs["neg"].valid_acts(), recs["ctl"].valid_acts(),
                              settings.n_perm, settings.perm_seed, device=lab.device)
        store.save_arrays(cname, res.arrays(), res.meta())
        timings["contrast_s"] = time.time() - t0
    else:
        res = ContrastResult.from_arrays(*got)
    if timings:
        old = store.load_json("timings") or {}
        store.save_json("timings", {**old, **timings})
    return Experiment(lab, ps, train, held, settings, store, recs, res, timings)


def ablation_spec(neurons, mode: Mode, positions: str = "all") -> InterventionSpec:
    return InterventionSpec.from_neurons([(l, n) for l, n, *_ in neurons], mode, positions)


def run_ablation(exp: Experiment, neurons, mode: Mode, positions="all", n_random=5, probe_splits=("train", "heldout"),
                 progress=None) -> BaselineComparison:
    spec = ablation_spec(neurons, mode, positions)
    return compare_to_random(exp.lab, exp.eval_data(probe_splits), spec, n_random, seed0=exp.settings.split_seed,
                             progress=progress)


# --- text reports ---------------------------------------------------------------------------------

def format_top_neurons(exp: Experiment, neurons) -> str:
    r = exp.contrast
    lines = [f"{'rank':>4} {'layer':>5} {'neuron':>6} {'d':>8} {'welch_t':>8} {'mean_pos':>9} {'mean_neg':>9}  flags"]
    for i, (l, n, d) in enumerate(neurons, 1):
        flags = ("sig" if r.significant[l, n] else "") + (" generic" if r.generic[l, n] else "")
        lines.append(f"{i:>4} {l:>5} {n:>6} {d:>+8.2f} {r.t[l, n]:>+8.2f} {r.mean_pos[l, n]:>+9.4f} {r.mean_neg[l, n]:>+9.4f}  {flags}")
    return "\n".join(lines)


def format_comparison(cmp: BaselineComparison, ev_meta: dict) -> str:
    names = {"d_pos": "Δ_pos  (log P target, positive probes)", "d_neg": "Δ_neg  (negative probes)",
             "d_ctl": "Δ_control (general-knowledge probes)", "kl": "KL(base||new) held-out prompts",
             "ppl_change": "Δ control-text perplexity"}
    n = cmp.real.n
    lines = [f"n probes: pos={n['pos']} neg={n['neg']} ctl={n['ctl']}; KL prompts={n['kl_prompts']} (held-out); "
             f"control paragraphs={n['control_text']}; batch_size={cmp.real.meta['batch_size']}; "
             f"random seeds={cmp.seeds}",
             f"{'metric':<42}{'real':>10}{'random mean':>13}{'± std':>9}{'z':>8}  verdict"]
    for k, label in names.items():
        s = cmp.stats[k]
        z = "n/a" if s["z"] is None else f"{s['z']:+.2f}"
        lines.append(f"{label:<42}{s['real']:>+10.4f}{s['rand_mean']:>+13.4f}{s['rand_std']:>9.4f}{z:>8}  {s['verdict']}")
    ratio = cmp.real.selectivity_ratio()
    lines.append("selectivity Δ_pos/Δ_neg: " + (f"{ratio:.2f}" if ratio is not None else
                                                 "withheld (|Δ_neg| < 0.05, ratio unstable)"))
    return "\n".join(lines)


# --- Phase 2: attribution, combined ranking, directions, method comparison -----------------------------

from .directions import Directions, best_layer, compute_directions, layer_separation  # noqa: E402
from .evaluation import ablation_curve  # noqa: E402
from .scoring import attribution_scores, combined_score  # noqa: E402


def ensure_attribution(exp: Experiment, batch_size: int | None = None) -> dict:
    """Attribution on the *training* probes (cached). Uses the control means as the ablation value."""
    got = exp.store.load_arrays("attribution")
    if got is not None:
        return {**got[0], "precision": got[1]["precision"]}
    t0 = time.time()
    abar = exp.lab.cache["control_means"].cpu().numpy()
    res = attribution_scores(exp.lab, exp.preset.probes_positive, exp.preset.probes_negative, abar,
                             batch_size=batch_size or exp.lab.batch_sizes["attribution"])
    exp.store.save_arrays("attribution", {k: v for k, v in res.items() if k != "precision"},
                          {"precision": res["precision"], "n_probes_pos": len(exp.preset.probes_positive),
                           "n_probes_neg": len(exp.preset.probes_negative), "split": "train", "exclude_first": True})
    exp.timings["attribution_s"] = time.time() - t0
    return res


def combined_ranking(exp: Experiment, attr: dict | None, weights: dict | None = None, positive_only: bool = False,
                     require_significant: bool = True, hide_generic: bool = True):
    """(score [L, D], candidate mask). Default: mean of z(|d|) and z(-attr_selectivity) over significant,
    non-generic neurons. Without attribution it falls back to |d| alone. positive_only keeps d > 0."""
    mask = exp.candidates(require_significant, hide_generic)
    if positive_only:
        mask = mask & (exp.contrast.d > 0)
    comps = {"abs_d": np.abs(exp.contrast.d)}
    if attr is not None:
        comps["neg_attr_sel"] = -attr["attr_sel"]
    return combined_score(comps, weights if weights else None, mask), mask


def ranked_list(score: np.ndarray, k: int):
    from .scoring import rank_neurons
    return rank_neurons(score, k, np.isfinite(score), "pos")


def ensure_directions(exp: Experiment):
    """(Directions from training residuals, held-out AUROC per layer, best layer)."""
    lab, st = exp.lab, exp.settings
    held = {}
    for name, texts in (("ho_pos", exp.heldout.positive), ("ho_neg", exp.heldout.negative)):
        held[name] = _cached_recording(exp.store, name, exp.timings, lambda texts=texts: record_set(
            lab, texts, exp.preset.system_prompt, st.mask, st.reduction, exp.preset.keywords, None, True,
            st.record_batch))
    dirs = compute_directions(exp.rec["pos"].resid[exp.rec["pos"].valid], exp.rec["neg"].resid[exp.rec["neg"].valid])
    sep = layer_separation(dirs, held["ho_pos"].resid[held["ho_pos"].valid], held["ho_neg"].resid[held["ho_neg"].valid])
    return dirs, sep, best_layer(sep)


def compare_methods(exp: Experiment, methods: dict, n_random: int = 5, probe_splits=("train", "heldout")) -> list[dict]:
    """Evaluate each named spec with the same EvalData (same batching), each against its random control."""
    ev = exp.eval_data(probe_splits)
    rows = []
    for name, spec in methods.items():
        c = compare_to_random(exp.lab, ev, spec, n_random, exp.settings.split_seed)
        rows.append(dict(method=name, stats=c.stats, n=c.real.n, seeds=c.seeds, ratio=c.real.selectivity_ratio()))
    return rows


def format_method_table(rows: list[dict]) -> str:
    """One table: each cell is  real  (random mean ± std, z).  Same harness for every row."""
    def cell(s):
        z = "n/a" if s["z"] is None else f"{s['z']:+.1f}"
        return f"{s['real']:+.3f} (r {s['rand_mean']:+.3f}±{s['rand_std']:.3f}, z {z})"

    heads = ["method", "Δ_pos", "Δ_neg", "Δ_control", "KL held-out"]
    keys = [None, "d_pos", "d_neg", "d_ctl", "kl"]
    body = [[r["method"]] + [cell(r["stats"][k]) for k in keys[1:]] for r in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(heads)]
    line = lambda cells: "  ".join(c.ljust(w[i]) for i, c in enumerate(cells))
    n = rows[0]["n"]
    out = [f"probes: pos n={n['pos']}, neg n={n['neg']}, control n={n['ctl']}; KL over {n['kl_prompts']} held-out prompts; "
           f"random baseline seeds={rows[0]['seeds']}", line(heads), "  ".join("-" * x for x in w)]
    out += [line(b) for b in body]
    return "\n".join(out)
