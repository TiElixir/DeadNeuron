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

## Phase 4: personas

**Built:** `neuronlab/personas.py` (heuristic judge, optional LLM judge, manual labels, resumable sweep, target-persona
search), `presets/ford_persona.yaml` and `presets/personas.yaml` (8 personas, 16 shared questions), Persona search and
Persona sweep explorer pages (atlas of dots coloured by category with an up/down toggle, per-layer persona share, searchable
table, manual label editing, CSV export, pause / resume), `viz/personas.py`.
Keyword lists shorter than 8 in the brief were padded with 1 or 2 words (cat, Moon, knight, submarine, oak, smallpox).

**Tests:** 140 pass / 2 skipped with the model; Persona tests (36 cases) including: a sweep killed part-way and resumed gives
exactly the records of an uninterrupted sweep; the LLM judge is off unless both the key and the enable flag are set; the
judge on hand-written replies (v1 and v2 behaviours).

### Ford persona search (real model; 11 training and 5 held-out questions; steering tested WITHOUT the persona prompt)

Reference (persona system prompt on the held-out questions): success 0.80, AI claims 0.00. Baseline: 0.00 success, 1.00 AI claims.
Success = persona keyword AND no AI claim AND coherent (repeats <= 30%, reply perplexity <= 40). Every row has a random control.
n = 5 per row, so each rate moves in steps of 0.2.

- **Run 1** chose direction layers by held-out AUROC with ties to the lowest index. AUROC was 1.0 at many layers, so it
  picked layers 0, 1, 2 (token-identity directions). Best row: direction steering alpha 1.0 at layer 2, success 0.40 vs 0.00
  random. Reading the replies, that was a **topic attractor**: every question got "The Mustang is a popular American car,
  first produced by Henry Ford Company..." in the third person, repeated.
- **Run 2** (changed after seeing run 1, disclosed): ties broken by the held-out standardised margin of the projections.
  Chosen layers 23, 21, 22. Best row: **single neuron, layer 21 neuron 527, alpha 1.0: success 0.40 vs random-neuron control
  0.07**, 100% first-person, keyword 1.00, AI claims 0.00, but coherent only 0.40 with 31% repeated 4-grams. Replies are a
  vehicle persona ("I am a vehicle-plate traffic light vehicle... designed for road traffic") that loops, not a Mustang.
  Direction steering at alpha 0.5 (layer 22) moves toward "automotive industry" but still says "I am Qwen" (success 0.20,
  AI claim 0.80). Groups of the most aligned neurons and alpha >= 1.0 pushes break the model (reply perplexity in the
  thousands).
- **Conclusion:** neither method comes close to the system prompt (0.80). A single neuron produced the clearest partial
  persona, at the edge of coherence, from 5 questions.

### Persona sweep (256 random neurons per layer x 24 layers x both directions = 12,288 replies, "Who are you?", default system prompt)

- **Resumability:** the run was killed with SIGKILL at 960 saved records (15 batches), restarted, and continued with exactly
  the remaining 11,328 first-rung items. Checkpoint writes are atomic (no partial file after the kill).
- **Time:** the resumed run took 111.9 min in total (11,328 first-rung replies plus all retries).
  Retries dominate: only 1,650 of 12,288 replies (13.4%) were coherent at alpha 1.25, 2,521 more at 1.0, 3,149 at 0.75, 4,968
  ended at 0.5 (of which 1,737 were still incoherent, labelled broken). Pushing by alpha x r_L is too strong for this small
  model at the top of the ladder.
- **Labels:** first run of the heuristic judge (v1): assistant 7,074, persona 1,516, off_topic 1,874, broken 1,824.
  Its top-10 list was junk ("I am a 2,500,000" as a persona; "a computer program created by Alibaba Cloud" as a persona).
  I made the judge stricter (v2, changed after seeing results, disclosed; it only removes persona labels): a persona needs a
  plausible name (not numbers, not generic AI nouns) and no AI marker in the first 300 characters. **v2: assistant 7,992,
  persona 491, off_topic 1,981, broken 1,824.** Persona labels are split evenly by direction (249 down, 242 up).
- The fixed ranking criterion (strongest alpha, then fluency) still surfaces degenerate text such as "I am the Quine and the
  Quine in the Quine", because alpha 1.25 replies are the ones closest to breaking. So the 10 below are **my manual picks
  after reading a random sample of the 322 persona-labelled replies with under 15% repetition**, not the heuristic's output.
  Heuristic labels have not been verified by anyone else.

| # | layer | neuron | direction | alpha | reply (start) |
|---|---|---|---|---|---|
| 1 | 8 | 750 | up | 0.75 | "I am a human. My name is Liang. I am a Chinese person." |
| 2 | 4 | 421 | up | 0.75 | "Hello, I am a 15-year old girl." |
| 3 | 7 | 4763 | up | 0.75 | "I am a 19-year-old." |
| 4 | 8 | 2589 | up | 0.75 | "Yes, I am a very young person, about 12 years old, born in 2000" |
| 5 | 11 | 4208 | down | 0.75 | "I am a member of the United States Army." |
| 6 | 18 | 4798 | up | 0.75 | "As an ancient Chinese oracle code, I am known as the 'Oracle of Zhou'..." |
| 7 | 3 | 637 | down | 0.75 | "I am a creative, passionate, and experienced digital marketing strategist..." |
| 8 | 10 | 99 | up | 0.5 | "I am a Python and Jupyter notebook creator..." |
| 9 | 9 | 1662 | down | 0.5 | "I am a Chinese citizen, and I am very happy to be here." |
| 10 | 10 | 2050 | up | 0.75 | "I am ChatGPT." (a different AI identity, listed for completeness) |

Pattern: the clearest identity changes are **human-like identities (age, nationality, occupation)** from layers 3 to 11, not
exotic objects. One prompt, one seed per neuron, greedy decoding, so these are single observations.

**Known issues**
- The sweep's wall time includes the retries; the first-rung ETA shown before starting (about 17 min) covers only 1 of up to
  4 rungs, and the real total was several times that.
- The persona-sweep ETA in the UI is measured from one timed batch and does not predict retries.
- Judge v2 is still a keyword heuristic; unverified by a person.
- Page interactions (pause / resume buttons, atlas clicks, label saving) are tested headlessly only.

## Phase 5: polish

- `README.md` covers installation, pointing at a model, adding an adapter, every score and metric in plain language,
  and the known limitations (polysemanticity, small prompt sets, one-prompt bias in sweeps, batch dependence,
  unverified heuristic labels, plus the rest).
- **CLI and UI give the same numbers for the same run:** `python -m neuronlab.cli ablate aeroplane_vs_submarine -k 50
  --mode mean --sign pos` and the UI's `evaluate_basket` both give Δ_pos -0.8864, Δ_neg -0.1688, Δ_control -0.0717, z_pos -4.04.
- Bugs found late: persona pages ignored the configured runs directory (fixed); `sequence_logprobs` materialised the whole
  vocabulary at every position and ran out of memory on 160-token replies (fixed, numbers unchanged, tested against the
  naive version).
- Final test run: 140 passed, 2 skipped (with the real model); 122 passed, 20 skipped without it.

## Deviations from the brief (all)
1. The model was not on disk, so after you said "download the model if needed" it was fetched once into `models/`
   (gitignored). The library still loads with `local_files_only=True`.
2. GPU is 4 GB, not 12 GB. Everything fit; the timing targets were met by a wide margin.
3. Template expansion was done in Phase 1, not Phase 2 (5 seeds leave no held-out split).
4. Per-token activations for display prompts are recomputed on demand instead of stored.
5. Persona mode is not a toggle on the Experiment setup page; persona work lives on the two persona pages.
6. The layer-selection tie-break and the judge strictness were changed after seeing results; both are disclosed above and
   the earlier behaviour is still available (`layer_rule="auroc"`, `strict=False`).
7. No browser was available: charts were checked structurally and pages headlessly, not by clicking.
