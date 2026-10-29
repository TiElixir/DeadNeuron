import numpy as np
import pytest
import torch

from neuronlab.directions import (Directions, auroc, best_layer, compute_directions, direction_ablation_spec,
                                  layer_separation, neurons_by_alignment, steering_spec)
from neuronlab.evaluation import EvalData, ablation_curve, evaluate, random_like
from neuronlab.interventions import Entry, InterventionSpec, Mode, intervene
from neuronlab.prompts import Probe, encode_chat
from neuronlab.runtime import forward_logits
from neuronlab.scoring import (attribution, attribution_scores, combined_score, keyword_lens_scores, logit_lens,
                               rank_neurons, zscore)
from neuronlab.viz.curves import ablation_curve_figure
from tests.conftest import _calibrated, make_model

POS = [Probe("The person who flies an aeroplane is called a", " pilot"), Probe("An aeroplane has two", " wings")]
NEG = [Probe("A submarine looks above the surface using a", " periscope"), Probe("Submarines detect objects using", " sonar")]


def mbar(lab, probes):
    from neuronlab.evaluation import probe_logprobs
    return float(probe_logprobs(lab, probes, batch_size=2).mean())


def test_attribution_matches_actual_single_neuron_ablation(lab):
    """First-order estimate vs the real effect of clamping the neuron to the control mean."""
    abar = lab.cache["control_means"].cpu().numpy()
    est, prec = attribution(lab, POS, abar, batch_size=2, exclude_first=False)
    assert prec == "float32" and est.shape == (3, 64)
    base = mbar(lab, POS)
    rng = np.random.default_rng(0)
    cells = [(int(l), int(n)) for l, n in zip(rng.integers(0, 3, 60), rng.integers(0, 64, 60))]
    cells = sorted(set(cells))
    actual = []
    from neuronlab.evaluation import probe_logprobs
    for l, n in cells:
        spec = InterventionSpec([Entry(l, [n], Mode.clamp(float(abar[l, n])))])
        with intervene(lab, spec):
            pass
        actual.append(float(probe_logprobs(lab, POS, spec, 2).mean()) - base)
    e = np.array([est[l, n] for l, n in cells]); a = np.array(actual)
    assert np.corrcoef(e, a)[0, 1] > 0.95
    big = np.abs(a) > np.abs(a).max() * 0.3
    assert (np.sign(e[big]) == np.sign(a[big])).all()


def test_attribution_sign_convention_on_a_planted_neuron(tok, preset):
    """Neuron whose output direction promotes ' pilot': ablating it must hurt the pilot probe, not the sonar one."""
    model = make_model("qwen2", len(tok))
    lab = _calibrated(model, tok, preset)
    layer, n = 2, 11
    pilot = tok(" pilot", add_special_tokens=False)["input_ids"][-1]
    with torch.no_grad():
        w = lab.adapter.unembed()[pilot]
        model.model.layers[layer].mlp.down_proj.weight[:, n] = 6 * w / w.norm()
    lab.cache.pop(("out_dirs", layer), None)
    abar = np.zeros((3, 64), np.float32)  # ablate to zero
    pos_probe = [Probe("The person who flies an aeroplane is called a", " pilot")]
    with intervene(lab, InterventionSpec([Entry(layer, [n], Mode.clamp(3.0))])):  # make the neuron clearly active
        res = attribution_scores(lab, pos_probe, NEG, abar, batch_size=1, exclude_first=False)
    assert res["attr_pos"][layer, n] < 0, "ablating a neuron that promotes the target must lower log P(target)"
    # Not asserted: attr_neg ~ 0. A strong push in the last layer also rescales the final RMSNorm, which shifts
    # every other token's log-prob a little (a real first-order effect, not an attribution error).
    assert np.unravel_index(res["attr_pos"].argmin(), (3, 64)) == (layer, n)  # most harmful for the pos probe
    assert res["attr_sel"][layer, n] < 0 and res["attr_sel"][layer, n] == res["attr_sel"].min()
    assert np.unravel_index(res["attr_sel"].argmax(), (3, 64)) != (layer, n)
    assert np.unravel_index(res["attr_sel"].argmin(), (3, 64)) == (layer, n)  # most negative = most selective
    # and the real effect has the same sign
    from neuronlab.evaluation import probe_logprobs
    base = probe_logprobs(lab, pos_probe, InterventionSpec([Entry(layer, [n], Mode.clamp(3.0))]))[0]
    ablated = probe_logprobs(lab, pos_probe, InterventionSpec([Entry(layer, [n], Mode.zero())]))[0]
    assert ablated < base


def test_attribution_restores_model_dtype(tok, preset):
    model = make_model("qwen2", len(tok)).to(torch.bfloat16)
    lab = _calibrated(model, tok, preset) if False else None  # calibration in bf16 on cpu is slow-path; check dtype only
    from neuronlab.model_loader import build_lab
    lab = build_lab(model, tok, device="cpu")
    attribution(lab, POS[:1], np.zeros((3, 64), np.float32), batch_size=1)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_logit_lens_top_token_of_a_planted_direction(tok, preset):
    model = make_model("qwen2", len(tok))
    lab = _calibrated(model, tok, preset)
    t = tok("ar", add_special_tokens=False)["input_ids"]
    assert len(t) == 1, "keyword must be a single token in the tiny vocab"
    t = t[0]
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[:, 5] = lab.adapter.unembed()[t]
    lab.cache.pop(("out_dirs", 1), None)
    res = logit_lens(lab, 1, [5, 6], k=5)
    assert res["top_ids"][0, 0] == t and res["top_ids"].shape == (2, 5) and res["bottom_ids"].shape == (2, 5)
    kw = keyword_lens_scores(lab, ["ar"], layers=[1], k=3)
    assert kw[1, 5] == 1 and kw[0].sum() == 0
    # multi-token keywords have no single-token id and are ignored (documented limitation)
    from neuronlab.scoring import keyword_token_ids
    assert keyword_token_ids(tok, ["sonar"]) == set()


def test_combined_score_default_and_mask():
    a = np.array([[1.0, 5.0, 3.0]]); b = np.array([[3.0, 1.0, 2.0]])
    mask = np.array([[True, True, False]])
    s = combined_score({"abs_d": a, "neg_attr_sel": b}, None, mask)
    za, zb = zscore(a, mask), zscore(b, mask)
    assert s[0, 0] == pytest.approx((za[0, 0] + zb[0, 0]) / 2) and np.isinf(s[0, 2]) and s[0, 2] < 0
    top = rank_neurons(s, 3, np.isfinite(s), "pos")
    assert len(top) == 2 and {t[1] for t in top} == {0, 1}
    w = combined_score({"abs_d": a, "neg_attr_sel": b}, {"abs_d": 1.0, "neg_attr_sel": 0.0}, mask)
    assert w[0, 1] > w[0, 0]  # weight only |d|: neuron 1 wins


def test_auroc_values():
    assert auroc(np.array([3., 4.]), np.array([1., 2.])) == 1.0
    assert auroc(np.array([1., 2.]), np.array([3., 4.])) == 0.0
    assert auroc(np.array([1., 1.]), np.array([1., 1.])) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 200), rng.normal(0, 1, 200)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert auroc(a, b) == pytest.approx(brute)


def synth_resid(n=200, L=3, dm=32, shift=2.0, seed=0):
    v = np.random.default_rng(1234).normal(size=dm); v /= np.linalg.norm(v)  # the concept axis is fixed
    rng = np.random.default_rng(seed)                                          # only the noise changes
    pos = rng.normal(size=(n, L, dm)).astype(np.float32); neg = rng.normal(size=(n, L, dm)).astype(np.float32)
    pos[:, 1] += shift * v  # only layer 1 carries the concept
    return pos, neg, v


def test_direction_recovers_planted_axis_and_best_layer():
    pos, neg, v = synth_resid()
    d = compute_directions(pos, neg)
    # expected cos ~ 2 / sqrt(2^2 + 2*32/200) = 0.96 for n=200; 0.9 leaves room for sampling noise
    assert abs(d.unit[1] @ v) > 0.9 and np.allclose(np.linalg.norm(d.unit, axis=-1), 1, atol=1e-5)
    hp, hn, _ = synth_resid(seed=1)
    ph, nh = hp.copy(), hn.copy()
    sep = layer_separation(d, ph, nh)
    assert best_layer(sep) == 1 and sep[1] > 0.85 and abs(sep[0] - 0.5) < 0.15


def test_steering_adds_alpha_r_u_and_directional_ablation_projects_out(lab):
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 32)).astype(np.float32); u /= np.linalg.norm(u, axis=-1, keepdims=True)
    dirs = Directions(u, u, np.ones(3, np.float32), 1, 1)
    e = encode_chat(lab.tokenizer, ["Explain how an aeroplane stays in the air."], None, "user_content")
    got, base = {}, {}
    layer = 1
    h = lab.adapter.residual_site(layer).register_forward_hook(lambda m, a, o: base.update(h=o.clone()))
    forward_logits(lab, e.input_ids, e.attention_mask); h.remove()

    spec = steering_spec(lab, dirs, [layer], alpha=0.5)
    with intervene(lab, spec):
        h = lab.adapter.residual_site(layer).register_forward_hook(lambda m, a, o: got.update(h=o.clone()))
        forward_logits(lab, e.input_ids, e.attention_mask); h.remove()
    expect = base["h"] + 0.5 * float(lab.cache["resid_norm"][layer]) * torch.tensor(u[layer])
    # hooks run in registration order, so our read-hook (registered later) sees the steered output
    assert torch.allclose(got["h"], expect, atol=1e-4)

    spec = direction_ablation_spec(dirs, [layer], source_layer=layer)
    with intervene(lab, spec):
        h = lab.adapter.residual_site(layer).register_forward_hook(lambda m, a, o: got.update(h=o.clone()))
        forward_logits(lab, e.input_ids, e.attention_mask); h.remove()
    assert (got["h"] @ torch.tensor(u[layer])).abs().max() < 1e-4


def test_residual_edits_use_the_same_harness_and_random_control(lab, preset):
    _, ho = preset.split(0)
    ev = EvalData.from_promptset(preset, ho, 4)
    rng = np.random.default_rng(1)
    u = rng.normal(size=(3, 32)).astype(np.float32); u /= np.linalg.norm(u, axis=-1, keepdims=True)
    spec = direction_ablation_spec(Directions(u, u, np.ones(3), 1, 1), [0, 1, 2], source_layer=1)
    r = evaluate(lab, ev, spec)
    assert r.kl > 0 and r.meta["n_residual_edits"] == 1
    rl = random_like(lab, spec, 0)
    assert rl.residual[0].layers == [0, 1, 2] and rl.residual[0].kind == "project_out"
    assert not torch.allclose(rl.residual[0].vectors, spec.residual[0].vectors)
    assert torch.allclose(rl.residual[0].vectors.norm(dim=-1), torch.ones(3), atol=1e-5)
    # scopes: generated_only residual edit leaves prompt-only logits untouched
    e = encode_chat(lab.tokenizer, ["Why is the sky blue?"], None, "user_content")
    base = forward_logits(lab, e.input_ids, e.attention_mask)
    gspec = direction_ablation_spec(Directions(u, u, np.ones(3), 1, 1), [1], 1, "generated_only")
    with intervene(lab, gspec, boundary=e.input_ids.shape[1]):
        out = forward_logits(lab, e.input_ids, e.attention_mask, boundary=e.input_ids.shape[1])
    assert torch.equal(out, base)


def test_neurons_by_alignment_finds_aligned_neuron(lab):
    od = lab.out_dirs(1)[7].cpu().numpy()
    u = np.zeros((3, 32), np.float32); u[2] = od / np.linalg.norm(od)
    top = neurons_by_alignment(lab, Directions(u, u, np.ones(3), 1, 1), layer=2, k=3)
    assert top[0][:2] == (1, 7) and top[0][2] == pytest.approx(1.0, abs=1e-5)
    assert all(l <= 2 for l, _, _ in top)


def test_ablation_curve_and_figure(lab, preset, tmp_path):
    _, ho = preset.split(0)
    ev = EvalData.from_promptset(preset, ho, 4)
    ranked = [(l, n, 1.0) for l in range(3) for n in range(10)]
    curve = ablation_curve(lab, ev, ranked, Mode.zero(), ks=(1, 2, 5, 10, 20, 500), n_random=2)
    assert curve["ks"] == [1, 2, 5, 10, 20]  # k larger than the ranking is dropped, not faked
    assert all(len(curve["real"][m]) == 5 for m in curve["real"]) and curve["n_random"] == 2
    for mode in ("light", "dark"):
        fig = ablation_curve_figure(curve, mode)
        assert len(fig.data) == 9  # per series: band, random mean, selected
    fig.write_html(tmp_path / "curve.html")
    assert (tmp_path / "curve.html").stat().st_size > 10_000
