import pytest
import torch

from neuronlab.interventions import Entry, InterventionSpec, Mode, intervene, push_coefficient
from neuronlab.prompts import encode_chat
from neuronlab.runtime import forward_logits, generate_ids

PROMPTS = ["Explain how an aeroplane stays in the air.", "Why do submarines dive to great depths?",
           "What is the Pythagorean theorem?"]


def batch(lab, prompts=PROMPTS):
    return encode_chat(lab.tokenizer, prompts, None, "user_content")


def logits(lab, spec=None, prompts=PROMPTS):
    e = batch(lab, prompts)
    with intervene(lab, spec):
        return forward_logits(lab, e.input_ids, e.attention_mask)


def spec_of(layer, neurons, mode, positions="all"):
    return InterventionSpec([Entry(layer, neurons, mode, positions)])


def test_a_empty_spec_is_identical_to_no_hooks(lab):
    e = batch(lab)
    base = forward_logits(lab, e.input_ids, e.attention_mask)
    assert torch.equal(logits(lab, InterventionSpec()), base)
    assert torch.equal(logits(lab, None), base)


def test_b_zero_equals_subtracting_a_n_times_out_dir(lab):
    layer, n = 1, 7
    e = batch(lab)
    ad = lab.adapter
    acts, out_dir = {}, lab.out_dirs(layer)[n]
    hp = ad.neuron_site(layer).register_forward_pre_hook(lambda m, a: acts.update(x=a[0][..., n].clone()))
    hf = ad.mlp_module(layer).register_forward_hook(lambda m, a, o: o - acts["x"][..., None] * out_dir)
    ref = forward_logits(lab, e.input_ids, e.attention_mask)
    hp.remove(); hf.remove()
    got = logits(lab, spec_of(layer, [n], Mode.zero()))
    assert torch.allclose(got, ref, atol=1e-5)
    assert not torch.allclose(got, logits(lab, None), atol=1e-6)  # the intervention does something


def test_c_push_equals_residual_addition(lab):
    layer, n, alpha, sign = 2, 11, 1.0, -1
    c = push_coefficient(lab, layer, n, alpha, sign)
    out_dir = lab.out_dirs(layer)[n]
    # c * W_down[:, n] has norm alpha * r_L, as in the neuronpersonas formula
    assert abs(float((c * out_dir).norm()) - alpha * float(lab.cache["resid_norm"][layer])) < 1e-4
    e = batch(lab)
    h = lab.adapter.mlp_module(layer).register_forward_hook(lambda m, a, o: o + c * out_dir)
    ref = forward_logits(lab, e.input_ids, e.attention_mask)
    h.remove()
    got = logits(lab, spec_of(layer, [n], Mode.push(alpha, sign)))
    assert torch.allclose(got, ref, atol=1e-4)


def test_d_logits_return_to_baseline_after_exit(lab):
    base = logits(lab)
    with intervene(lab, spec_of(0, [1, 2], Mode.clamp(5.0))):
        assert not torch.allclose(logits(lab), base)
    assert torch.equal(logits(lab), base)


def test_hooks_removed_after_exception(lab):
    with pytest.raises(RuntimeError):
        with intervene(lab, spec_of(1, [3], Mode.zero())):
            assert lab.adapter.neuron_site(1)._forward_pre_hooks
            raise RuntimeError("boom")
    for i in range(lab.adapter.n_layers):
        assert not lab.adapter.neuron_site(i)._forward_pre_hooks
    assert not lab.cache.get("_active")


@pytest.mark.parametrize("mode,expect", [
    (Mode.clamp(2.5), lambda x, cm: torch.full_like(x, 2.5)),
    (Mode.scale(3.0), lambda x, cm: x * 3.0),
    (Mode.zero(), lambda x, cm: torch.zeros_like(x)),
    (Mode.mean(), lambda x, cm: cm.expand_as(x)),
])
def test_modes_set_activation_as_specified(lab, mode, expect):
    layer, ns = 1, [4, 9]
    e = batch(lab)
    seen = {}
    spec = spec_of(layer, ns, mode)
    base = {}
    h0 = lab.adapter.neuron_site(layer).register_forward_pre_hook(lambda m, a: base.update(x=a[0][..., ns].clone()))
    forward_logits(lab, e.input_ids, e.attention_mask)
    h0.remove()
    with intervene(lab, spec):
        # registered after the intervention hook, so it sees the *modified* activations
        h = lab.adapter.neuron_site(layer).register_forward_pre_hook(lambda m, a: seen.update(x=a[0][..., ns].clone()))
        forward_logits(lab, e.input_ids, e.attention_mask)
        h.remove()
    cm = lab.cache["control_means"][layer][ns]
    assert torch.allclose(seen["x"], expect(base["x"], cm), atol=1e-5)


def test_per_row_specs_match_row_by_row_runs(lab):
    specs = [spec_of(0, [1], Mode.clamp(4.0)), spec_of(1, [2, 3], Mode.zero()),
             spec_of(2, [5], Mode.push(1.25, +1))]
    e = batch(lab)
    with intervene(lab, specs):
        together = forward_logits(lab, e.input_ids, e.attention_mask)
    for i, s in enumerate(specs):
        e1 = batch(lab, [PROMPTS[i]])
        with intervene(lab, s):
            alone = forward_logits(lab, e1.input_ids, e1.attention_mask)
        T = alone.shape[1]
        assert torch.allclose(together[i, -T:], alone[0], atol=1e-4), f"row {i}"


def test_per_row_wrong_batch_size_raises(lab):
    with pytest.raises(ValueError):
        with intervene(lab, [spec_of(0, [1], Mode.zero())] * 2):
            e = batch(lab)  # 3 rows
            forward_logits(lab, e.input_ids, e.attention_mask)


def test_padding_does_not_change_logits(lab):
    e = batch(lab)
    together = forward_logits(lab, e.input_ids, e.attention_mask)
    for i in range(len(PROMPTS)):
        e1 = batch(lab, [PROMPTS[i]])
        alone = forward_logits(lab, e1.input_ids, e1.attention_mask)
        assert torch.allclose(together[i, -alone.shape[1]:], alone[0], atol=1e-4)


def _manual_greedy(lab, prompt_ids, att, spec, steps):
    """Reference decoding without a KV cache: re-run the whole sequence each step."""
    P = prompt_ids.shape[1]
    ids, mask = prompt_ids.clone(), att.clone()
    for _ in range(steps):
        with intervene(lab, spec, boundary=P):
            lg = forward_logits(lab, ids, mask, boundary=P)
        nxt = lg[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        mask = torch.cat([mask, torch.ones_like(nxt)], 1)
    return ids[:, P:]


@pytest.mark.parametrize("positions", ["all", "generated_only", "prompt_only"])
def test_scopes_with_kv_cache_match_uncached_reference(lab, positions):
    e = batch(lab)
    # strong push on several neurons so the continuation really changes
    spec = InterventionSpec([Entry(l, list(range(0, 24)), Mode.push(1.5, +1), positions) for l in range(3)])
    steps = 8
    with intervene(lab, spec):
        cached = generate_ids(lab, e.input_ids, e.attention_mask, steps, stop_on_repetition=False)
    ref = _manual_greedy(lab, e.input_ids, e.attention_mask, spec, steps)
    assert torch.equal(cached[:, :steps], ref[:, :cached.shape[1]])


def test_scopes_act_on_the_right_positions(lab):
    """Non-vacuous scope check at the logit level (argmax tokens of a random model are too coarse)."""
    e = batch(lab)
    P = e.input_ids.shape[1]
    base = forward_logits(lab, e.input_ids, e.attention_mask)

    def run(pos):
        spec = InterventionSpec([Entry(l, list(range(0, 24)), Mode.push(1.5, +1), pos) for l in range(3)])
        with intervene(lab, spec, boundary=P - 3):  # last 3 positions count as "generated"
            return forward_logits(lab, e.input_ids, e.attention_mask, boundary=P - 3)

    all_, gen, prompt = run("all"), run("generated_only"), run("prompt_only")
    d = lambda a, b, sl: float((a[:, sl] - b[:, sl]).abs().max())
    # prompt positions: generated_only leaves them untouched, the other scopes change them
    assert d(gen, base, slice(0, P - 3)) == 0.0
    assert d(prompt, base, slice(0, P - 3)) > 1e-3 and d(all_, base, slice(0, P - 3)) > 1e-3
    # at the last positions generated_only acts directly, and `all` differs from `prompt_only` there
    assert d(gen, base, slice(P - 3, P)) > 1e-3
    assert d(all_, prompt, slice(P - 3, P)) > 1e-3


def test_mean_mode_requires_calibration(tok, preset):
    from neuronlab.model_loader import build_lab
    from tests.conftest import make_model
    fresh = build_lab(make_model("qwen2", len(tok)), tok, device="cpu")
    with pytest.raises(RuntimeError, match="control_means"):
        with intervene(fresh, spec_of(0, [1], Mode.mean())):
            pass
