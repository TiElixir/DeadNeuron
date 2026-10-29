"""Phase 2 acceptance: land-vs-sea end to end.

attribution -> combined rank -> neuron ablation vs directional ablation vs steering in ONE table
-> ablation curve saved as Plotly HTML. All numbers are printed as measured.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronlab.directions import direction_ablation_spec, steering_spec
from neuronlab.evaluation import ablation_curve, probe_logprobs
from neuronlab.interventions import Entry, InterventionSpec, Mode, intervene
from neuronlab.model_loader import ROOT, load_lab
from neuronlab.pipeline import (Settings, ablation_spec, combined_ranking, compare_methods, ensure_attribution,
                                ensure_directions, format_method_table, format_top_neurons, prepare, ranked_list)
from neuronlab.scoring import attribution, full_precision
from neuronlab.viz.curves import ablation_curve_figure

PRESET = sys.argv[1] if len(sys.argv) > 1 else "land_vs_sea_animals"
K = 50
lab = load_lab()
exp = prepare(lab, ROOT / "presets" / f"{PRESET}.yaml", Settings())
r = exp.contrast
print(f"{PRESET}: n_pos={r.n_pos} n_neg={r.n_neg} n_control={r.n_control} (training split); "
      f"{int(r.significant.sum())} significant, {int((r.significant & ~r.generic).sum())} after hiding generic")

t0 = time.time()
attr = ensure_attribution(exp)
print(f"attribution: {time.time() - t0:.1f}s, precision={attr['precision']}, "
      f"{len(exp.preset.probes_positive)}+{len(exp.preset.probes_negative)} training probes (target: < 300 s)")

# --- how good is the first-order estimate? compare with the real effect of mean-ablating single neurons -------------
abar = lab.cache["control_means"].cpu().numpy()
top_abs = np.dstack(np.unravel_index(np.argsort(-np.abs(attr["attr_pos"]), axis=None)[:40], attr["attr_pos"].shape))[0]
est_first_excl = np.array([attr["attr_pos"][l, n] for l, n in top_abs])
with full_precision(lab):
    est_all, _ = attribution(lab, exp.preset.probes_positive, abar, exclude_first=False, precision="native")
    base = probe_logprobs(lab, exp.preset.probes_positive).mean()
    actual = np.array([probe_logprobs(lab, exp.preset.probes_positive,
                                      InterventionSpec([Entry(int(l), [int(n)], Mode.mean())])).mean() - base
                       for l, n in top_abs])
print(f"attribution validation on the 40 neurons with the largest |estimate| (fp32, training positive probes):\n"
      f"  Pearson r(estimate, actual single-neuron mean-ablation) = {np.corrcoef(est_first_excl, actual)[0, 1]:.3f} "
      f"(first token excluded, default) / {np.corrcoef([est_all[l, n] for l, n in top_abs], actual)[0, 1]:.3f} "
      f"(first token included)")

# --- rankings ---------------------------------------------------------------------------------------------------------
comb, _ = combined_ranking(exp, attr)
comb_pos, _ = combined_ranking(exp, attr, positive_only=True)
d_only, _ = combined_ranking(exp, None, positive_only=True)
top_comb, top_comb_pos, top_d = ranked_list(comb, K), ranked_list(comb_pos, K), ranked_list(d_only, K)
print("\nTop 15 by combined rank (z|d| and z(-attr_sel); significant, non-generic):")
print(format_top_neurons(exp, [(l, n, exp.contrast.d[l, n]) for l, n, _ in top_comb[:15]]))
print("   attr_sel for these:", [round(float(attr["attr_sel"][l, n]), 4) for l, n, _ in top_comb[:15]])

# --- directions -------------------------------------------------------------------------------------------------------
dirs, sep, best = ensure_directions(exp)
print(f"\ndirection separation (held-out AUROC per layer): best layer {best} ({sep[best]:.3f}); "
      f"layers: {np.round(sep, 2).tolist()}")
all_layers = list(range(lab.adapter.n_layers))
methods = {
    f"neurons: |d| only, top {K}, d>0, mean-abl": (ablation_spec(top_d, Mode.mean()), ("train", "heldout")),
    f"neurons: combined, top {K}, mean-abl": (ablation_spec(top_comb, Mode.mean()), ("heldout",)),
    f"neurons: combined d>0, top {K}, mean-abl": (ablation_spec(top_comb_pos, Mode.mean()), ("heldout",)),
    f"direction ablation: u[{best}] at all layers": (direction_ablation_spec(dirs, all_layers, best), ("train", "heldout")),
    f"steering: +1.0*r_L*u[{best}] at layer {best}": (steering_spec(lab, dirs, [best], 1.0), ("train", "heldout")),
}
print("\n=== neuron ablation vs directional ablation vs steering (same harness, same batching) ===")
print(format_method_table(compare_methods(exp, methods, n_random=5)))

# --- ablation curve ---------------------------------------------------------------------------------------------------
out = exp.store.dir
for name, ranked, splits in (("combined", top_comb_all := ranked_list(comb, 500), ("heldout",)),
                             ("d_only", ranked_list(d_only, 500), ("train", "heldout"))):
    t0 = time.time()
    curve = ablation_curve(lab, exp.eval_data(splits), ranked, Mode.mean(), n_random=5, seed0=0)
    fig = ablation_curve_figure(curve, "light", f"{PRESET}: mean-ablating the top-k neurons ({name} ranking, {'+'.join(splits)} probes)")
    path = out / f"ablation_curve_{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"\nablation curve ({name}, {time.time() - t0:.0f}s) -> {path}")
    print("  k      :", curve["ks"])
    print("  d_pos  :", np.round(curve["real"]["d_pos"], 3).tolist())
    print("  d_neg  :", np.round(curve["real"]["d_neg"], 3).tolist())
    print("  d_ctl  :", np.round(curve["real"]["d_ctl"], 3).tolist())
    print("  rand d_pos mean:", np.round(curve["rand_mean"]["d_pos"], 3).tolist())

# --- steering strength ladder: every value is reported, none is selected -------------------------------------------
ladder = {f"steering: alpha={a} * r_L * u[{best}] at layer {best}": (steering_spec(lab, dirs, [best], a), ("train", "heldout"))
          for a in (0.05, 0.1, 0.25, 0.5, 1.0)}
print("\n=== steering strength ladder (all alphas shown) ===")
print(format_method_table(compare_methods(exp, ladder, n_random=5)))
