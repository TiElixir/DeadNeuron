# NeuronLab

A local tool for finding the MLP neurons (and residual-stream directions) that separate a concept from a closely
matched contrast, and for testing whether they matter: ablate, clamp, scale or push them, and compare the effect with
the same number of random neurons. It also steers a model toward a persona.

It runs on one GPU (tested on a 4 GB RTX 3050 laptop GPU with Qwen2.5-0.5B-Instruct in bf16) and, slowly, on CPU.
Everything is local: the model is loaded with `local_files_only=True`, and no code path downloads weights.

The honest-reporting rules are built in: every effect is shown next to a random-neuron baseline, every number states
its sample size and split, and "not distinguishable from random neurons" is printed when an effect is within 2 standard
deviations of the random draws.

## Install

```
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

`pyproject.toml` pins the versions the tests were run against (torch 2.14+, transformers 5.16.1, huggingface-hub below
2.0). Python 3.10 or newer.

## Point it at a model

Put a Hugging Face style model directory on disk (it needs `config.json` and weights), then either edit `model_path` in
`config.yaml` or set `NEURONLAB_MODEL_PATH`. To fetch Qwen2.5-0.5B-Instruct once, outside NeuronLab:

```
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')"
```

Architectures supported out of the box: Qwen2, Llama, Mistral, Gemma (gated MLP) and GPT-2. Anything else falls back to
auto-discovery, which looks for the decoder `ModuleList` and a down projection named `down_proj`, `c_proj`, `fc2`,
`dense_4h_to_h` or `w2`, and otherwise raises an error that prints the module tree.

## Use it

```
streamlit run app/Home.py                      # the UI (dark theme, from .streamlit/config.toml)
python examples/quickstart.py                  # aeroplane vs submarine, end to end
python examples/phase2_land_vs_sea.py          # attribution, directions, ablation curve, comparison table
python examples/phase4_personas.py search      # persona search (default: Ford Mustang)
python examples/phase4_personas.py sweep 256   # persona sweep, resumable: rerun the same command to resume
python -m neuronlab.cli ablate aeroplane_vs_submarine -k 50 --mode mean --sign pos
pytest                                         # NEURONLAB_MODEL_PATH=... also runs the real-model tests
```

The CLI, the examples and the UI call the same library functions (`neuronlab/pipeline.py`), and results are cached on
disk under `runs/<model>/<preset>/<settings>/`. Reopening a run, or repeating an evaluation, reads the cache.

## Add an adapter for a new architecture

An adapter says where a neuron lives. Subclass `DownProjAdapter` and set four paths:

```python
from neuronlab.adapters import register_adapter
from neuronlab.adapters.base import DownProjAdapter

class MyAdapter(DownProjAdapter):
    layers_path = "model.layers"        # the ModuleList of decoder blocks
    mlp_name, attn_name = "mlp", "self_attn"
    down_name = "down_proj"             # the module whose INPUT is the neuron activation vector
    final_norm_path = "model.norm"

    def _down_weight_as_out_dirs(self, w):
        return w.t()                    # nn.Linear weight is (d_model, d_mlp): neuron n is column n
                                        # (a GPT-2 style Conv1D weight is (d_mlp, d_model): return w as is)

register_adapter("my_model_type", MyAdapter)   # key = config.model_type
```

`neuron_out_dirs(i)` must return `[d_mlp, d_model]`, one output direction per row. Check yours with the reconstruction
test in `tests/test_adapters.py`: `neuron_acts @ out_dirs (+ bias) == mlp_output`.

## What the scores and metrics mean

**Neuron activation.** For gated MLPs, `mlp(x) = down_proj(act(gate(x)) * up(x))`. A neuron's activation is one element
of the vector that goes *into* `down_proj`, and its output direction is the matching column of `down_proj.weight`.

**Per-prompt value.** Each prompt is reduced to one number per neuron (mean, max or last over the counted tokens). The
counted tokens are chosen by a mask (default: the user's text only, never the chat template, never the first token,
which is an attention sink). The sample size n is the number of prompts, never tokens.

**Cohen's d.** Difference of the two group means in units of their pooled standard deviation. Large |d| means the neuron
separates the groups consistently. It says nothing about how much the neuron matters for the output, and tiny-activation
neurons can have large d.

**Welch t.** The same contrast with a test statistic that allows unequal variances.

**Permutation threshold and "significant".** The positive/negative labels are shuffled 500 times; each time the largest
|d| in each layer is recorded. The 95th percentile of those maxima is that layer's threshold. A neuron is significant if
its |d| exceeds it, which controls the chance of any false positive in the layer at about 5%.

**Generic neuron.** Also active on the control prompts (control mean above the positive mean, or active on more than 90%
of control prompts). Hidden by default.

**Attribution.** A first-order estimate of what mean-ablating a neuron does to the log-probability of the probe targets:
`sum over positions of (control mean - activation) x gradient`. Negative means ablation lowers it. Attribution
selectivity is the effect on the positive probes minus the effect on the negative probes; more negative means the neuron
helps the positive concept more. It was checked against real single-neuron ablations (Pearson r = 0.95 on the land/sea
run). It uses the training probes, so evaluate on the held-out probes when attribution picked the neurons.

**Combined rank.** Mean of the z-scored |d| and the z-scored negative attribution selectivity over significant,
non-generic neurons. One large component can outvote a small wrong-signed one.

**Logit lens.** The tokens a neuron's output direction promotes or suppresses, through the final norm and unembedding.

**Direction.** Mean residual of the positive prompts minus mean residual of the negative prompts, per layer. Its quality
is the held-out AUROC of the projection (0.5 chance, 1.0 perfect).

**Push.** Adds `± alpha x r_L / ||W_down[:, n]||` to a neuron's activation, where `r_L` is the median residual norm at
layer L. That moves the residual stream by `alpha x r_L` along the neuron's output direction. Equivalent to adding to the
residual stream directly (tested).

**Evaluation numbers.** `Δ_pos`, `Δ_neg`, `Δ_control` are changes in the mean log-probability of the full target
continuation on the positive, negative and general-knowledge probes (0 = no effect, more negative = more damage). The
selectivity ratio `Δ_pos / Δ_neg` is withheld when `|Δ_neg| < 0.05` because it explodes. KL is the mean
KL(baseline || intervened) of the next-token distribution on held-out prompts. Control perplexity is on neutral
paragraphs. The random baseline draws the same number of neurons per layer from the non-selected ones, with the same mode
and positions, for several seeds; the z-score is the real effect against that distribution.

**Persona metrics.** Keyword rate (reply contains a persona keyword), AI-claim rate (reply says it is Qwen or an AI),
coherent rate (repeated 4-grams at most 30% and reply perplexity under the unsteered model at most 40), and success
(keyword and no AI claim and coherent). First-person rate is a diagnostic only.

## Known limitations

- **Polysemanticity.** A neuron that separates two prompt sets usually does many other things. In the land/sea check,
  ablating the top land-leaning neurons hurt sea probes more than land probes.
- **Small prompt sets.** Prompts come from a few frames and subjects, so they are not independent, and the permutation
  test treats them as exchangeable. Each preset has 4 + 2 probes per side, and persona search has 5 held-out questions, so
  rates move in steps of 0.2. Treat z-scores from 5 random seeds as direction, not proof.
- **One-prompt bias in sweeps.** The persona sweep asks only "Who are you?". A neuron that changes identity in other
  contexts but not here is missed, and one prompt cannot show that a neuron is a persona neuron in general.
- **Batch dependence under strong pushes.** bf16 numerics and left padding mean a very strong push can give slightly
  different text when batch composition changes. All comparisons use identical batching, and batch size is recorded.
- **Heuristic labels are not checked by a human.** The persona judge is a transparent keyword and regex rule. The optional
  LLM judge is off unless `NEURONLAB_JUDGE_API_KEY` is set and the box is ticked, and it sends reply text to an external
  API. Manual overrides are saved separately.
- **Teacher-forced probes are not free generation.** Ablations that lower cloze log-probs often leave generated text
  almost unchanged.
- **Attention heads are out of scope.** The adapter has `attn_module` and `n_heads` as hooks for a later head-level view.
- **Keyword lens** only uses keywords that are a single token.
- **No browser tests.** The UI is exercised headlessly with Streamlit's `AppTest`; chart clicks have not been tested in a
  real browser.

See `docs/phase_reports.md` for the numbers behind each phase and every deviation from the brief.
