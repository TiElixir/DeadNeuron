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
