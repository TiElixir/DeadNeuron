import numpy as np
import pytest
import torch

from neuronlab.prompts import encode_chat
from neuronlab.recording import Welford, record_set, reduce_masked, Cancelled
from neuronlab.runtime import forward_logits
from neuronlab.scoring import (candidate_mask, cohens_d, contrast_scores, group_stats, permutation_null,
                               rank_neurons)


def test_welford_matches_numpy_over_uneven_batches():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(97, 5, 11)).astype(np.float32)
    w = Welford((5, 11))
    for s in range(0, 97, 13):
        w.update(x[s:s + 13])
    assert np.allclose(w.mean, x.mean(0), atol=1e-5)
    assert np.allclose(w.var, x.var(0, ddof=1), atol=1e-4)
    assert np.allclose(w.frac_pos, (x > 0).mean(0))
    assert w.n == 97


def test_reduce_masked():
    x = torch.tensor([[[1., 10.], [3., 20.], [5., 30.]]])
    m = torch.tensor([[False, True, True]])
    assert torch.allclose(reduce_masked(x, m, "mean"), torch.tensor([[4., 25.]]))
    assert torch.allclose(reduce_masked(x, m, "max"), torch.tensor([[5., 30.]]))
    assert torch.allclose(reduce_masked(x, m, "last"), torch.tensor([[5., 30.]]))
    assert torch.isnan(reduce_masked(x, torch.zeros(1, 3, dtype=torch.bool), "mean")).all()


def test_record_set_matches_manual_hook_and_excludes_template(lab, preset):
    texts = preset.positive[:3]
    rec = record_set(lab, texts, mask="user_content", reduction="mean", batch_size=2)
    assert rec.acts.shape == (3, 3, 64) and rec.valid.all() and rec.n == 3
    # manual: single prompt, read the site at the masked positions only
    e = encode_chat(lab.tokenizer, [texts[1]], None, "user_content")
    got = {}
    hs = [lab.adapter.neuron_site(i).register_forward_pre_hook(
        lambda m, a, i=i: got.update({i: a[0][0][e.token_mask[0]].mean(0)})) for i in range(3)]
    forward_logits(lab, e.input_ids, e.attention_mask)
    for h in hs:
        h.remove()
    manual = torch.stack([got[i] for i in range(3)]).numpy()
    assert np.allclose(rec.acts[1].astype(np.float32), manual, atol=2e-3)
    # n counts prompts: stats come from 3 prompts regardless of token counts
    assert np.allclose(rec.mean, rec.acts.astype(np.float32).mean(0), atol=1e-4)


def test_record_is_batch_size_invariant(lab, preset):
    a = record_set(lab, preset.positive[:6], batch_size=1, want_resid=False)
    b = record_set(lab, preset.positive[:6], batch_size=6, want_resid=False)
    assert np.allclose(a.acts.astype(np.float32), b.acts.astype(np.float32), atol=2e-3)


def test_record_progress_and_cancel(lab, preset):
    seen = []
    record_set(lab, preset.positive[:4], batch_size=2, progress=lambda d, t, eta: seen.append((d, t)))
    assert seen[-1] == (4, 4)
    with pytest.raises(Cancelled):
        record_set(lab, preset.positive[:4], batch_size=2, cancel=lambda: True)
    assert not lab.cache.get("_active")
    assert all(not lab.adapter.neuron_site(i)._forward_pre_hooks for i in range(3))


def test_concept_tokens_mask_gives_nan_for_prompts_without_keywords(lab):
    rec = record_set(lab, ["The pilot flew.", "Boil an egg."], mask="concept_tokens", keywords=["pilot"],
                     want_resid=False)
    assert rec.valid.tolist() == [True, False] and rec.n == 1


def planted(n_pos=30, n_neg=30, L=3, D=50, seed=0, effect=2.0, where=(1, 17)):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, (n_pos, L, D)).astype(np.float32)
    neg = rng.normal(0, 1, (n_neg, L, D)).astype(np.float32)
    pos[:, where[0], where[1]] += effect
    return pos, neg, where


def test_permutation_test_finds_the_planted_neuron():
    pos, neg, (l, n) = planted()
    ctl = np.random.default_rng(5).normal(0, 1, (20, 3, 50)).astype(np.float32)
    res = contrast_scores(pos, neg, ctl, n_perm=300, seed=1)
    assert res.significant[l, n]
    assert res.significant.sum() == 1  # nothing else clears the family-wise threshold
    top = rank_neurons(res.d, 1, candidate_mask(res), "abs")[0]
    assert top[:2] == (l, n)
    assert res.d[l, n] > 1.0


def test_no_planted_effect_gives_few_false_positives():
    rng = np.random.default_rng(3)
    pos, neg = rng.normal(0, 1, (30, 3, 50)).astype(np.float32), rng.normal(0, 1, (30, 3, 50)).astype(np.float32)
    res = contrast_scores(pos, neg, None, n_perm=300, seed=2)
    # per-layer FWER 5%: expect <= 1 hit in 3 layers with overwhelming probability
    assert res.significant.sum() <= 2


def test_cohens_d_formula():
    pos, neg, _ = planted(effect=1.0)
    P, N = torch.tensor(pos), torch.tensor(neg)
    mp, vp = group_stats(P)
    mn, vn = group_stats(N)
    d = cohens_d(mp, vp, mn, vn).numpy()
    ref = (pos.mean(0) - neg.mean(0)) / np.sqrt((pos.var(0, ddof=1) + neg.var(0, ddof=1)) / 2 + 1e-6)
    assert np.allclose(d, ref, atol=1e-4)


def test_permutation_matmul_matches_explicit_shuffle():
    pos, neg, _ = planted(n_pos=8, n_neg=8, D=10, effect=0.0, where=(1, 3))
    thr, null_max = permutation_null(pos, neg, n_perm=40, seed=7)
    # brute-force the same label draws
    X = np.concatenate([pos, neg]); X = X - X.mean(0)
    g = torch.Generator().manual_seed(7)
    order = torch.rand(25, 16, generator=g).argsort(1)[:, :8].numpy()  # first chunk of 25
    for k in range(3):
        A = np.zeros(16, bool); A[order[k]] = True
        a, b = X[A], X[~A]
        d = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2 + 1e-6)
        assert np.allclose(np.abs(d).max(-1), null_max[k], atol=1e-4)


def test_generic_flag():
    pos, neg, _ = planted(effect=0.0)
    ctl = np.zeros((20, 3, 50), np.float32); ctl[:, 0, 4] = 5.0  # active on 100% of control, above pos mean
    res = contrast_scores(pos, neg, ctl, n_perm=20, seed=0)
    assert res.generic[0, 4]
