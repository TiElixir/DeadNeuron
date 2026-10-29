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
    if not store.has("setup"):
        store.save_json("setup", {"preset": ps.to_dict(), "settings": asdict(settings), "name": ps.name})

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


def compare_methods(exp: Experiment, methods: dict, n_random: int = 5) -> list[dict]:
    """Evaluate each method with the same batching and its own random control.

    methods: name -> (spec, probe_splits). A ranking that used attribution (training probes) must be evaluated
    on ("heldout",) only; rankings that never touched the probes may use ("train", "heldout").
    """
    evs, rows = {}, []
    for name, (spec, splits) in methods.items():
        ev = evs.setdefault(tuple(splits), exp.eval_data(tuple(splits)))
        c = compare_to_random(exp.lab, ev, spec, n_random, exp.settings.split_seed)
        rows.append(dict(method=name, stats=c.stats, n=c.real.n, seeds=c.seeds, ratio=c.real.selectivity_ratio(),
                         splits=list(splits), selected=c.real.summary()))
    return rows


def format_method_table(rows: list[dict]) -> str:
    """One table: each cell is  real  (random mean ± std, z).  n and probe split are printed per row."""
    def cell(s):
        z = "n/a" if s["z"] is None else f"{s['z']:+.1f}"
        return f"{s['real']:+.3f} (r {s['rand_mean']:+.3f}±{s['rand_std']:.3f}, z {z})"

    heads = ["method", "probes (n pos/neg/ctl)", "Δ_pos", "Δ_neg", "Δ_control", "KL held-out"]
    keys = ["d_pos", "d_neg", "d_ctl", "kl"]
    body = [[r["method"], f"{'+'.join(r['splits'])} ({r['n']['pos']}/{r['n']['neg']}/{r['n']['ctl']})"]
            + [cell(r["stats"][k]) for k in keys] for r in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(heads)]
    line = lambda cells: "  ".join(c.ljust(w[i]) for i, c in enumerate(cells))
    out = [f"each cell: real effect (random control mean ± std over seeds {rows[0]['seeds']}, z); "
           f"KL over {rows[0]['n']['kl_prompts']} held-out prompts", line(heads), "  ".join("-" * x for x in w)]
    return "\n".join(out + [line(b) for b in body])


# --- reopening saved runs & cached basket evaluation (used by the UI and the CLI alike) ---------------------

import hashlib  # noqa: E402
import json  # noqa: E402

from .evaluation import generate_replies, generation_diff  # noqa: E402
from .storage import list_runs  # noqa: E402


def describe_runs(root: str | Path) -> list[dict]:
    rows = []
    for d in list_runs(root):
        setup = json.loads((d / "setup.json").read_text()) if (d / "setup.json").exists() else {}
        meta = json.loads((d / "run.json").read_text())
        rows.append(dict(path=str(d), model_fp=d.parents[1].name, name=setup.get("name", "?"),
                         created=meta.get("created", ""), dtype=meta["settings"].get("dtype", "?"),
                         n_per_side=setup.get("settings", {}).get("n_per_side"), setup=setup))
    return rows


def load_experiment(lab: Lab, run_dir: str | Path, progress=None, cancel=None) -> Experiment:
    """Reopen a saved run. Everything comes from disk; nothing is recomputed unless a file is missing."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    setup = json.loads((run_dir / "setup.json").read_text())
    if run_dir.parents[1].name != lab.fingerprint:
        raise ValueError("this run belongs to a different model; load that model first")
    want = meta["settings"]
    if want.get("dtype") != str(lab.dtype) or want.get("device") != lab.device.type:
        raise ValueError(f"run was recorded with dtype={want.get('dtype')} on {want.get('device')}; "
                         f"the loaded model is {lab.dtype} on {lab.device.type}")
    return prepare(lab, PromptSet(**setup["preset"]), Settings(**setup["settings"]), run_dir.parents[2],
                   progress, cancel)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def basket_key(neurons, mode: Mode, positions: str, n_random: int, splits, gen_prompts, curve: bool, max_new: int) -> str:
    payload = json.dumps([sorted(map(tuple, neurons)), str(mode), positions, n_random, list(splits), list(gen_prompts),
                          curve, max_new], default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def evaluate_basket(exp: Experiment, neurons, mode: Mode, positions: str = "all", n_random: int = 5,
                    splits=("train", "heldout"), gen_prompts=(), with_curve: bool = False, curve_ranking=None,
                    max_new_tokens: int = 120, progress=None) -> dict:
    """Full evaluation of a neuron selection (cached on disk by its inputs). JSON-serialisable result."""
    neurons = [(int(l), int(n)) for l, n, *_ in neurons]
    key = basket_key(neurons, mode, positions, n_random, splits, gen_prompts, with_curve, max_new_tokens)
    cached = exp.store.load_json(f"eval_{key}")
    if cached is not None:
        cached["from_cache"] = True
        return cached
    spec = ablation_spec(neurons, mode, positions)
    ev = exp.eval_data(tuple(splits))
    cmp_ = compare_to_random(exp.lab, ev, spec, n_random, exp.settings.split_seed, progress)
    out = dict(key=key, neurons=neurons, mode=str(mode), positions=positions, splits=list(splits), n=cmp_.real.n,
               batch_size=cmp_.real.meta["batch_size"], seeds=cmp_.seeds, stats=cmp_.stats,
               ratio=cmp_.real.selectivity_ratio(), per_probe=cmp_.real.per_probe,
               ppl=dict(base=cmp_.real.ppl_base, after=cmp_.real.ppl_after), kl=cmp_.real.kl,
               n_per_layer=spec.per_layer_counts(), generations=[], curve=None, from_cache=False)
    if gen_prompts:
        out["generations"] = generation_diff(exp.lab, list(gen_prompts), spec, exp.preset.system_prompt, max_new_tokens,
                                             exp.lab.batch_sizes["generate"])
    if with_curve and curve_ranking:
        out["curve"] = ablation_curve(exp.lab, ev, curve_ranking, mode, positions, n_random=n_random,
                                      seed0=exp.settings.split_seed)
    out = _jsonable(out)
    exp.store.save_json(f"eval_{key}", out)
    return out


def first_heldout_prompts(exp: Experiment, n: int) -> list[str]:
    """The first n held-out prompts in a fixed order (positive then negative), never hand-picked."""
    return (exp.heldout.positive + exp.heldout.negative)[:n]


# --- neuron card ---------------------------------------------------------------------------------------------

from .evaluation import generate_replies_rows  # noqa: E402
from .interventions import Entry as _Entry  # noqa: E402
from .recording import token_activations  # noqa: E402
from .scoring import logit_lens  # noqa: E402


def neuron_card(exp: Experiment, layer: int, neuron: int, attr: dict | None = None, dirs: Directions | None = None,
                k: int = 20) -> dict:
    """Everything the neuron card shows, from cached data plus one cheap weight-only computation."""
    r, tok = exp.contrast, exp.lab.tokenizer
    scores = dict(d=float(r.d[layer, neuron]), welch_t=float(r.t[layer, neuron]), diff=float(r.diff[layer, neuron]),
                  mean_pos=float(r.mean_pos[layer, neuron]), mean_neg=float(r.mean_neg[layer, neuron]),
                  significant=bool(r.significant[layer, neuron]), generic=bool(r.generic[layer, neuron]),
                  threshold=float(r.threshold[layer]))
    if attr is not None:
        scores.update(attr_pos=float(attr["attr_pos"][layer, neuron]), attr_neg=float(attr["attr_neg"][layer, neuron]),
                      attr_sel=float(attr["attr_sel"][layer, neuron]))
    lens = logit_lens(exp.lab, layer, [neuron], k)
    dec = lambda ids: [tok.decode([int(i)]) for i in ids]
    acts = {n: exp.rec[n].valid_acts()[:, layer, neuron].astype(np.float32) for n in ("pos", "neg", "ctl")}
    out = dict(scores=scores, acts=acts, top=list(zip(dec(lens["top_ids"][0]), lens["top_vals"][0].tolist())),
               bottom=list(zip(dec(lens["bottom_ids"][0]), lens["bottom_vals"][0].tolist())))
    if dirs is not None:
        from .directions import alignment_with
        out["alignment"] = {int(l): float(alignment_with(exp.lab, layer, dirs.unit[l])[neuron])
                            for l in range(layer, dirs.unit.shape[0])}
    return out


def token_view(exp: Experiment, layer: int, neuron: int, n_per_set: int = 10) -> dict:
    """Per-token activations for the first n training prompts of each set (fixed order), computed on demand."""
    out = {}
    for name, texts in (("positive", exp.train.positive), ("negative", exp.train.negative), ("control", exp.train.control)):
        rows = []
        for t in texts[:n_per_set]:
            toks, acts, counted = token_activations(exp.lab, t, layer, neuron, exp.preset.system_prompt)
            rows.append(dict(text=t, tokens=toks, acts=acts, counted=counted))
        out[name] = rows
    return out


def quick_push_test(exp: Experiment, layer: int, neuron: int, prompts: list[str], alphas=(0.5, 1.0, 1.25),
                    max_new_tokens: int = 100) -> list[dict]:
    """Baseline plus push up / down at each alpha for each prompt, in ONE batch of per-row interventions."""
    conds = [("baseline", None)] + [(f"{'up' if s > 0 else 'down'} alpha={a}", ablation_spec([(layer, neuron)], Mode.push(a, s)))
                                    for a in alphas for s in (+1, -1)]
    rows_p = [p for p in prompts for _ in conds]
    rows_s = [s for _ in prompts for _, s in conds]
    replies = generate_replies_rows(exp.lab, rows_p, rows_s, exp.preset.system_prompt, max_new_tokens)
    out, i = [], 0
    for p in prompts:
        for name, _ in conds:
            out.append(dict(prompt=p, condition=name, reply=replies[i][0]))
            i += 1
    return out
