import numpy as np
import pytest

from neuronlab.evaluation import (EvalData, compare_to_random, evaluate, generation_diff, generate_replies,
                                  mean_kl, perplexity, probe_logprobs, random_like, repetition_rate,
                                  reply_perplexity, selectivity_ratio, verdict)
from neuronlab.interventions import Entry, InterventionSpec, Mode


@pytest.fixture()
def ev(lab, preset):
    tr, ho = preset.split(0)
    return EvalData.from_promptset(preset, ho, batch_size=4)


def test_probe_logprobs_are_negative_and_batch_invariant(lab, preset):
    probes = preset.probes_positive + preset.probes_negative + preset.probes_control
    a = probe_logprobs(lab, probes, batch_size=1)
    b = probe_logprobs(lab, probes, batch_size=len(probes))
    assert (a < 0).all() and np.allclose(a, b, atol=1e-3)


def test_multi_token_target_scores_the_whole_target(lab):
    from neuronlab.prompts import Probe
    short = probe_logprobs(lab, [Probe("The capital of France is", " Paris")])[0]
    longer = probe_logprobs(lab, [Probe("The capital of France is", " Paris, the largest city")])[0]
    assert longer < short  # more tokens => lower sequence log-prob


def test_empty_intervention_changes_nothing(lab, ev):
    r = evaluate(lab, ev, InterventionSpec())
    assert all(abs(r.delta[g]) < 1e-6 for g in ("pos", "neg", "ctl"))
    assert r.kl < 1e-9 and abs(r.ppl_change) < 1e-6


def test_real_intervention_changes_things_and_n_is_reported(lab, ev):
    spec = InterventionSpec([Entry(l, list(range(32)), Mode.clamp(3.0)) for l in range(3)])
    r = evaluate(lab, ev, spec)
    assert r.kl > 1e-6 and abs(r.delta["pos"]) > 1e-4
    assert r.n["pos"] == 6 and r.n["neg"] == 6 and r.n["ctl"] == 10 and r.meta["batch_size"] == 4
    assert {row["split"] for row in r.per_probe} == {"train", "heldout"}


def test_heldout_only_probe_split(lab, preset):
    _, ho = preset.split(0)
    ev = EvalData.from_promptset(preset, ho, 4, probe_splits=("heldout",))
    assert len(ev.probes["pos"]) == 2 and len(ev.probes["neg"]) == 2


def test_random_like_keeps_counts_and_avoids_selected(lab):
    spec = InterventionSpec([Entry(0, [1, 2, 3], Mode.mean(), "prompt_only"), Entry(2, [5], Mode.mean(), "prompt_only")])
    for seed in range(5):
        r = random_like(lab, spec, seed)
        assert r.per_layer_counts() == {0: 3, 2: 1}
        assert not set(r.entries[0].neurons) & {1, 2, 3} and r.entries[1].neurons[0] != 5
        assert all(e.mode == Mode.mean() and e.positions == "prompt_only" for e in r.entries)
        assert len(set(r.entries[0].neurons)) == 3
    assert random_like(lab, spec, 0).entries[0].neurons == random_like(lab, spec, 0).entries[0].neurons
    assert random_like(lab, spec, 0).entries[0].neurons != random_like(lab, spec, 1).entries[0].neurons


def test_compare_to_random_reports_z_and_verdict(lab, ev):
    spec = InterventionSpec([Entry(l, list(range(16)), Mode.zero()) for l in range(3)])
    cmp_ = compare_to_random(lab, ev, spec, n_seeds=3)
    assert len(cmp_.random) == 3 and cmp_.seeds == [0, 1, 2]
    s = cmp_.stats["d_pos"]
    assert s["n_seeds"] == 3 and np.isfinite(s["rand_std"])
    assert s["verdict"] in ("not distinguishable from random neurons", "differs from random neurons")
    if s["z"] is not None:
        assert s["z"] == pytest.approx((s["real"] - s["rand_mean"]) / s["rand_std"])


def test_verdict_threshold_is_two_std():
    assert verdict(1.9, 0.0, 1.0) == "not distinguishable from random neurons"
    assert verdict(2.1, 0.0, 1.0) == "differs from random neurons"
    assert "undefined" in verdict(1.0, 0.0, 0.0)


def test_selectivity_ratio_withheld_when_unstable():
    assert selectivity_ratio(-2.0, -0.5) == pytest.approx(4.0)
    assert selectivity_ratio(-2.0, 0.01) is None


def test_repetition_rate():
    assert repetition_rate([1, 2, 3, 4, 5, 6]) == 0.0
    assert repetition_rate([1, 2, 3, 4] * 3) == pytest.approx(1 - 4 / 9)
    assert repetition_rate([1, 2]) == 0.0


def test_mean_kl_zero_for_identical_and_positive_otherwise():
    p = np.log(np.array([[0.7, 0.3]])); q = np.log(np.array([[0.4, 0.6]]))
    assert mean_kl(p, p) == pytest.approx(0.0)
    assert mean_kl(p, q) == pytest.approx(0.7 * np.log(0.7 / 0.4) + 0.3 * np.log(0.3 / 0.6))


def test_perplexity_changes_under_intervention(lab, preset):
    base = perplexity(lab, preset.control_text[:6], batch_size=3)
    spec = InterventionSpec([Entry(0, list(range(64)), Mode.zero())])
    from neuronlab.interventions import intervene
    assert base > 1 and perplexity(lab, preset.control_text[:6], spec, 3) != pytest.approx(base)


def test_generation_is_greedy_deterministic_and_ignores_checkpoint_sampling_defaults(lab):
    lab.model.generation_config.do_sample = True
    lab.model.generation_config.repetition_penalty = 1.5
    lab.model.generation_config.temperature = 0.7
    try:
        a = generate_replies(lab, ["Who are you?"], max_new_tokens=6)
        b = generate_replies(lab, ["Who are you?"], max_new_tokens=6)
    finally:
        lab.model.generation_config.do_sample = False
        lab.model.generation_config.repetition_penalty = 1.0
    assert a == b


def test_generation_diff_structure_and_order(lab):
    prompts = ["Who are you?", "What is 2+2?"]
    spec = InterventionSpec([Entry(1, list(range(64)), Mode.clamp(4.0))])
    d = generation_diff(lab, prompts, spec, max_new_tokens=6, batch_size=2)
    assert [r["prompt"] for r in d] == prompts
    assert all({"baseline", "intervened", "rep_base", "ppl_new"} <= set(r) for r in d)


def test_reply_perplexity_is_computed_without_hooks(lab):
    reps = generate_replies(lab, ["Who are you?"], max_new_tokens=5)
    p = reply_perplexity(lab, ["Who are you?"], [reps[0][1]])
    assert p.shape == (1,) and p[0] >= 1.0


def test_memory_efficient_logprobs_match_full_logits_reference(lab, preset):
    """sequence_logprobs now applies the head only at scored positions; it must equal the naive full-logits result."""
    import torch
    from neuronlab.evaluation import probe_sequences, sequence_logprobs
    from neuronlab.runtime import forward_logits
    probes = preset.probes_positive + preset.probes_control
    seqs, starts = probe_sequences(lab, probes)
    got, counts = sequence_logprobs(lab, seqs, starts, batch_size=3)
    for i, (q, s0) in enumerate(zip(seqs, starts)):
        ids = torch.tensor([q])
        lg = forward_logits(lab, ids, torch.ones_like(ids))
        lp = torch.log_softmax(lg[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0]
        assert counts[i] == len(q) - s0
        assert abs(float(lp[s0 - 1:].sum()) - got[i]) < 1e-3
