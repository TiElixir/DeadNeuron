"""End-to-end aeroplane-vs-submarine experiment on the real model.

    python examples/quickstart.py            # uses config.yaml / NEURONLAB_MODEL_PATH

Prints the top 20 neurons, mean-ablates the top 50 positive-selective neurons, and reports the effect
next to a 5-seed random-neuron baseline. Whatever the numbers are, they are printed as they are.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronlab.evaluation import generation_diff
from neuronlab.interventions import Mode
from neuronlab.model_loader import ROOT, load_lab
from neuronlab.pipeline import Settings, ablation_spec, format_comparison, format_top_neurons, prepare, run_ablation

t0 = time.time()
lab = load_lab()
print(f"model loaded in {time.time() - t0:.1f}s on {lab.device} ({lab.dtype}); "
      f"{lab.adapter.n_layers} layers x {lab.adapter.d_mlp(0)} neurons = {lab.adapter.n_neurons}")

exp = prepare(lab, ROOT / "presets" / "aeroplane_vs_submarine.yaml", Settings(),
              progress=lambda d, t, eta: print(f"\r  recording {d}/{t}, eta {eta:4.0f}s", end="", flush=True))
print(f"\ntimings (s, only for stages computed this run): { {k: round(v, 1) for k, v in exp.timings.items()} }")

r = exp.contrast
print(f"\ntraining split: n_pos={r.n_pos}, n_neg={r.n_neg}, n_control={r.n_control}; {int(r.significant.sum())} "
      f"neurons pass the per-layer permutation threshold ({r.n_perm} permutations), "
      f"{int((r.significant & ~r.generic).sum())} after hiding generic neurons")
print("significant per layer:", r.significant_per_layer().tolist())

# Positive-selective neurons: higher on aeroplane prompts than on submarine prompts (decided before looking).
top = exp.top_neurons(50, sign="pos")
print("\nTop 20 positive-selective neurons (ranked by Cohen's d):")
print(format_top_neurons(exp, top[:20]))

print(f"\nMean-ablating the top {len(top)} (mode=mean, all positions):")
cmp_ = run_ablation(exp, top, Mode.mean(), "all", n_random=5)
print(format_comparison(cmp_, {}))

held = exp.heldout.positive[:3] + exp.heldout.negative[:3]
print("\nGeneration diff on the first held-out prompts (fixed order, greedy):")
for row in generation_diff(lab, held, ablation_spec(top, Mode.mean()), exp.preset.system_prompt, 80):
    print(f"\n> {row['prompt']}\n  baseline : {row['baseline'][:300]!r}\n  ablated  : {row['intervened'][:300]!r}")
