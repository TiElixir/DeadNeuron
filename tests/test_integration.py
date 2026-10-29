"""End-to-end on the real model. Skipped unless NEURONLAB_MODEL_PATH is set."""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("NEURONLAB_MODEL_PATH"), reason="NEURONLAB_MODEL_PATH not set")

from neuronlab.evaluation import generate_replies  # noqa: E402
from neuronlab.interventions import Mode  # noqa: E402
from neuronlab.model_loader import ROOT, load_lab  # noqa: E402
from neuronlab.pipeline import Settings, prepare, run_ablation  # noqa: E402
from neuronlab.prompts import encode_chat  # noqa: E402
from neuronlab.runtime import forward_logits  # noqa: E402


@pytest.fixture(scope="module")
def real():
    return load_lab()


def test_structure_is_read_from_config(real):
    ad, cfg = real.adapter, real.model.config
    assert ad.n_layers == cfg.num_hidden_layers and ad.d_model == cfg.hidden_size
    assert ad.d_mlp(0) == cfg.intermediate_size and ad.n_neurons == ad.n_layers * cfg.intermediate_size


def test_neuron_site_input_reconstructs_mlp_output(real):
    e = encode_chat(real.tokenizer, ["Explain how an aeroplane stays in the air."], None, "user_content").to(real.device)
    got = {}
    site = real.adapter.neuron_site(5)
    h1 = site.register_forward_pre_hook(lambda m, a: got.update(a=a[0].clone()))
    h2 = real.adapter.mlp_module(5).register_forward_hook(lambda m, a, o: got.update(o=o.clone()))
    forward_logits(real, e.input_ids, e.attention_mask)
    h1.remove(); h2.remove()
    assert torch.equal(site(got["a"]), got["o"])


def test_greedy_generation_ignores_sampling_defaults_of_the_checkpoint(real):
    gc = real.model.generation_config
    assert gc.do_sample and gc.repetition_penalty != 1.0  # the shipped defaults would otherwise sample
    a = generate_replies(real, ["Who are you?"], max_new_tokens=12)
    b = generate_replies(real, ["Who are you?"], max_new_tokens=12)
    assert a == b and a[0][0]


def test_aeroplane_preset_end_to_end_and_cached(real, tmp_path):
    st = Settings(n_per_side=12, n_perm=50, record_batch=4, eval_batch=4)
    exp = prepare(real, ROOT / "presets" / "aeroplane_vs_submarine.yaml", st, tmp_path)
    assert exp.contrast.d.shape == (24, 4864) and exp.contrast.n_pos == 8
    assert exp.timings, "first run computes"
    again = prepare(real, ROOT / "presets" / "aeroplane_vs_submarine.yaml", st, tmp_path)
    assert not again.timings, "second run must not recompute anything"
    assert np.array_equal(again.contrast.d, exp.contrast.d)

    top = exp.top_neurons(10, "pos")
    assert len(top) == 10
    a = run_ablation(exp, top, Mode.mean(), n_random=2)
    b = run_ablation(exp, top, Mode.mean(), n_random=2)
    assert a.stats["d_pos"]["real"] == b.stats["d_pos"]["real"]  # identical batching => identical numbers
    assert all(np.isfinite(v["real"]) for v in a.stats.values())
