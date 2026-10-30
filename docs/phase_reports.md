# NeuronLab phase reports

Numbers below come from real runs on Qwen2.5-0.5B-Instruct (bf16) on an RTX 3050 Laptop GPU with 4 GB VRAM
(the brief assumed 12 GB). Nothing was tuned after looking at results.

## Phase 1: core library

**Built:** model_loader (`Lab`), adapters (Qwen2 / Llama / Mistral / Gemma gated MLP, GPT-2, auto-discovery),
prompts (chat formatting, five token masks, seeded 70/30 split, template expansion), recording (Welford, in-hook
reduction), scoring (Cohen's d, Welch t, permutation null, generic flag), interventions (zero / mean / clamp /
scale / push, three position scopes, per-row specs), runtime (padding-invariant forward, greedy generation),
evaluation (probes, KL, perplexity, random baseline, generation diff), storage (atomic, resumable), pipeline,
CLI, quickstart, aeroplane + control presets.

**Tests:** 78 pass / 2 skipped with `NEURONLAB_MODEL_PATH` set (the skips are the GPT-2-only Conv1D test run under
the other two architectures); 74 pass / 6 skipped without the model. Covers every equivalence in 4.5 (empty spec,
zero == subtract a_n * out_dir, push == residual addition, hook cleanup incl. on exceptions), per-row vs row-by-row,
`generated_only` / `prompt_only` / `all` through the KV cache vs an uncached reference, padding invariance,
Welford vs numpy, permutation matmul vs brute-force shuffles, planted-neuron recovery, storage round-trip and resume.

**Real model (aeroplane vs submarine; 42 + 42 training prompts, 28 control; seed 0):**

| stage | time |
|---|---|
| record 112 prompts, all 116,736 neurons | ~2.5 s (target < 2 min: met) |
| permutation null, 500 perms | 0.2 s |
| mean-ablate 50 neurons + 5 random seeds (full eval each) | ~8 s |

3,616 neurons pass the per-layer permutation threshold; 1,734 remain after hiding generic neurons.
Mean-ablating the top 50 positive-selective neurons (n probes: pos 6, neg 6, control 10; 5 random seeds):

| metric | real | random mean ± std | z |
|---|---|---|---|
| Δ_pos | -0.886 | -0.086 ± 0.198 | -4.04 |
| Δ_neg | -0.169 | +0.053 ± 0.052 | -4.23 |
| Δ_control | -0.072 | +0.025 ± 0.056 | -1.72 (not distinguishable) |
| KL (held-out prompts) | 0.035 | 0.024 ± 0.041 | +0.28 (not distinguishable) |

**Reading:** the selected neurons matter more for aeroplane cloze items than random neurons do, but they are
**not selective**: submarine probes are also hurt beyond random. In free generation the ablated model still describes
fighter jets and gliders; the concept is not erased. With 6 probes per group and 5 seeds the z-scores are crude
(std from 5 samples); read them as direction, not proof.

**Known issues / caveats**
- Top neurons by Cohen's d include layer-0 neurons with activations around 0.01-0.08. d is scale-free and says nothing
  about causal size; attribution (Phase 2) is meant to address this.
- The 84 training prompts come from 15 shared frames x 4-6 objects, so they are not independent. The permutation test
  treats them as exchangeable, which overstates significance relative to "aeroplanes vs submarines" in general.
- Per-token activations for the display prompts are recomputed on demand (`recording.token_activations`) instead of
  stored: all-layer per-token storage would be ~200 MB for 30 prompts.
- Deviation: expansion (planned for Phase 2) was pulled into Phase 1, because 5 seeds per side leaves no held-out split.
- Hardware: 4 GB GPU. Phase 1 fits easily; fp32 attribution in Phase 2 may need smaller batches.

## Phase 2: better scoring and analysis

**Built:** attribution (total-derivative first-order estimate via zero-valued `delta` tensors, fp32 with OOM fallback),
logit lens + keyword lens, direction alignment, combined rank (z-score over the candidate set), held-out probes in
every preset, ablation curves (+ Plotly figure, light/dark, validated colours), residual-stream edits (steering,
directional ablation) inside `InterventionSpec` so they go through the same harness and random controls,
`land_vs_sea_animals` and `elephant_vs_other_land_animals` presets. Run: `python examples/phase2_land_vs_sea.py`.

**Tests:** 86 pass / 6 skipped (no model env). New: attribution vs real single-neuron ablation (r > 0.95 on the tiny
model), planted-neuron sign convention, logit lens, combined score, AUROC vs brute force, direction recovery,
steering == manual add, projection leaves zero component, residual edits through `evaluate`, ablation curve + figure.

**Real model, land vs sea (42 + 42 training prompts, 28 control):** 571 neurons significant, 258 after hiding generic.

| check | result |
|---|---|
| attribution time (4+4 probes, fp32) | 0.3 s (target < 5 min: met) |
| attribution vs real single-neuron mean-ablation, top-40 by estimate | Pearson r = 0.954 (first token excluded), 0.958 (included) |
| best direction layer (held-out AUROC) | layer 22, AUROC 0.972 |

Same harness, same batching, each row against its own 5-seed random control (cells: real (random mean ± std, z)):

| method | probes | Δ_pos | Δ_neg | Δ_control |
|---|---|---|---|---|
| neurons: d>0 only, top 50, mean-abl | 6/6/10 | -0.130 (r -0.001±0.053, z -2.4) | **-0.292** (r +0.040±0.081, z -4.1) | -0.006 (not distinguishable) |
| neurons: combined rank, top 50 | 2/2/10 held-out | -0.124 (r -0.034±0.058, z -1.5) | +0.064 (z +0.5) | +0.008 |
| neurons: combined, d>0, top 50 | 2/2/10 held-out | +0.052 (z +1.9) | +0.047 (z +0.1) | +0.017 |
| directional ablation u[22], all layers | 6/6/10 | -0.362 (r -0.289±0.329, z -0.2) | -0.200 (r -0.487±0.704, z +0.4) | -0.264 (z -0.6) |

Steering toward "land" with u[22] at layer 22 (all alphas, none selected): alpha 0.05 gives Δ_pos +0.056 vs random
-0.075 and Δ_neg -0.158 vs -0.011; alpha 0.1 gives +0.076 / -0.322; from alpha 0.25 up it degrades the model
(alpha 1.0: Δ_control -2.6, KL 1.6) and random directions degrade it more.

**Reading**
- Neuron ablation is **not selective** for land animals here. Ablating the top-50 land-leaning neurons hurts sea probes
  more than land probes. One untested explanation: control text is non-animal, so mean-ablation to the control mean
  hurts animal knowledge in general.
- Directional ablation is **not distinguishable from random directions**, because projecting out a random direction at
  all 24 layers is itself very disruptive (random Δ_neg std 0.70).
- Small-alpha steering is the only clearly directional signal, and it rests on 6 probes per side and 5 seeds.
- Ablation curves: the combined ranking has only about 258 candidates, so k = 500 is dropped rather than padded
  with non-significant neurons.

**Known issues**
- The spec's default combined rank (mean of z(|d|) and z(-attr_sel)) lets a very large |d| outvote an attribution of
  the wrong sign: rank 2 has attr_sel = +0.14 (ablation helps the positive probes). Kept as specified; use
  `positive_only` or per-component weights to change it.
- When attribution drives the ranking, evaluation is limited to the 2 held-out probes per side. That is a thin
  basis; the |d|-only row (all 6 probes) is the better-powered comparison.
- Keyword lens only uses keywords that encode to one token; multi-token keywords are ignored.
- The ablation-curve chart was checked structurally (label collisions, palette validator: pass in both modes, aqua
  contrast WARN covered by direct labels). No browser was available to screenshot it.

## Phase 3: UI

**Built:** `app/Home.py` plus seven pages (Model architecture, Experiment setup, Layer overview, Layer drill-down,
Neuron card, Basket and ablation lab, Directions and steering), a persistent sidebar (model path and load status, device
and dtype, active run, GPU memory, basket size), background jobs with progress, ETA and cancel for recording and
evaluation, and `neuronlab/viz` (pure Plotly / HTML builders). Run: `streamlit run app/Home.py`.
Dark theme from `.streamlit/config.toml`, with the minimalist-ui principles applied inside it (warm off-black, hairline
borders, no shadows / gradients / emoji, serif titles, mono for IDs, colour only where it carries meaning).
Chart colours are categorical slots 1 to 3 of the reference palette, run through the validator in both modes.

**No recomputation on reload:** runs reopen from disk via `load_experiment`; basket evaluations are stored by the hash of
their inputs (`eval_<key>.json`) and shown as "Loaded from disk". One model copy (`st.cache_resource`, max 1) and one
experiment in memory.

**Tests:** 106 pass / 2 skipped with the model; AppTest renders all 8 scripts against the real model (entering through
`Home.py` and switching pages), checks the no-run message, the cached-evaluation path and the neuron card / drill-down with
a selected neuron. `tests/test_figures.py` builds every figure in both themes. The server starts headless (health and
index return 200).

**Bugs the tests caught:** `_axes` passed `gridcolor` twice (broke the neuron card and the evaluation chart).

**Not verified (no browser in this environment):** clicking inside a Plotly chart (architecture block, layer bar, grid
cell), `st.data_editor` interaction, the fragment polling a running background job, and how the pages look. Each chart
click has a plain selector next to it (layer / neuron number inputs, table selection), so no function depends on a click.
Treat the first manual click-through as part of the acceptance check.

**Known issues**
- Evaluation cancel is checked between random seeds and curve points, not inside a single forward pass.
- A background job holds the GPU lock, so other pages wait while a recording or evaluation runs.
- Persona mode is not offered in Experiment setup until the persona pages exist (Phase 4).
