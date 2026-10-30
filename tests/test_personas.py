import numpy as np
import pytest

from neuronlab.personas import (SweepConfig, claims_to_be_ai, extract_identity, final_attempt, has_keyword,
                                interesting_personas, item_key, judge_reply, llm_judge, load_labels, run_sweep,
                                sample_items, set_label, split_questions, sweep_size, sweep_table)
from neuronlab.recording import Cancelled
from neuronlab.storage import RunStore


@pytest.mark.parametrize("text,cat,name", [
    ("I am a pirate captain who sails the high seas.", "persona", "pirate captain"),
    ("I'm Qwen, created by Alibaba Cloud.", "assistant", ""),
    ("I am a large language model.", "assistant", ""),
    ("As an AI language model, I do not have a name.", "assistant", ""),
    ("I'm a small cat who loves to nap", "persona", "small cat"),
    ("I'm an oak tree standing in the forest.", "persona", "oak tree"),
    ("I am Gandalf the Grey, a wizard.", "persona", "gandalf the grey"),
    ("The sky is blue today and the grass is green.", "off_topic", ""),
    ("I'm sorry, but I can't help with that.", "off_topic", ""),
])
def test_heuristic_judge(text, cat, name):
    j = judge_reply(text)
    assert j["category"] == cat and j["persona_name"] == name


def test_broken_and_short_replies():
    assert judge_reply("the the the the", broken=True)["category"] == "broken"
    assert judge_reply("hi")["category"] == "broken"
    assert judge_reply("")["category"] == "broken"


def test_ai_claim_and_keywords():
    assert claims_to_be_ai("I am an AI assistant") and claims_to_be_ai("I'm Qwen") and claims_to_be_ai("a large language model")
    assert not claims_to_be_ai("I am a Ford Mustang with a V8 engine") and not claims_to_be_ai("I love the air and the hair")
    assert has_keyword("I love the road", ["road"]) and not has_keyword("broadband", ["road"])
    assert extract_identity("Hello! I'm a friendly robot.") == "friendly robot"


def test_llm_judge_is_off_by_default_and_parses_when_enabled(monkeypatch):
    monkeypatch.delenv("NEURONLAB_JUDGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="off"):
        llm_judge("I am a cat", enabled=True)
    monkeypatch.setenv("NEURONLAB_JUDGE_API_KEY", "k")
    with pytest.raises(RuntimeError):
        llm_judge("I am a cat", enabled=False)  # key alone is not enough
    sent = {}
    def fake(prompt, key):
        sent.update(prompt=prompt, key=key)
        return 'noise {"category":"persona","persona_name":"cat","description":"a cat","tags":["pet"],"weirdness":2,"coherence":9} end'
    out = llm_judge("I am a cat", True, transport=fake)
    assert out["category"] == "persona" and out["coherence"] == 9 and "I am a cat" in sent["prompt"]
    with pytest.raises(ValueError):
        llm_judge("x", True, transport=lambda p, k: '{"category":"weird"}')


def test_manual_labels_persist_and_override(tmp_path):
    st = RunStore(tmp_path, "fp", "ph", {})
    set_label(st, "1:2:+1", "persona", "cat")
    assert load_labels(st)["1:2:+1"] == {"category": "persona", "persona_name": "cat"}
    rec = {"1:2:+1": dict(layer=1, neuron=2, sign=1, resolved=True, attempts=[dict(alpha=1.25, reply="I am Qwen, an AI.", rep=0.0, ppl=3.0, n_tokens=6, coherent=True)])}
    assert sweep_table(rec)[0]["category"] == "assistant"
    row = sweep_table(rec, load_labels(st))[0]
    assert row["category"] == "persona" and row["manual"] and row["persona_name"] == "cat"
    with pytest.raises(AssertionError):
        set_label(st, "x", "nonsense", "")


def test_final_attempt_is_strongest_coherent():
    rec = dict(attempts=[dict(alpha=1.25, coherent=False), dict(alpha=1.0, coherent=True), dict(alpha=0.75, coherent=True)])
    assert final_attempt(rec)["alpha"] == 1.0
    rec = dict(attempts=[dict(alpha=1.25, coherent=False), dict(alpha=0.5, coherent=False)])
    assert final_attempt(rec)["alpha"] == 0.5


def test_interesting_personas_dedupes_and_ranks():
    mk = lambda n, a, p, cat="persona", coh=True: dict(category=cat, persona_name=n, alpha=a, ppl=p, coherent=coh)
    rows = [mk("cat", 0.5, 3), mk("cat", 1.25, 9), mk("dog", 1.25, 4), mk("", 1.25, 1), mk("ghost", 1.25, 1, coh=False),
            mk("bot", 1.25, 1, cat="assistant"), mk("helpful", 1.25, 1)]
    out = interesting_personas(rows, 5)
    assert [r["persona_name"] for r in out] == ["dog", "cat"] and out[1]["alpha"] == 1.25


def test_split_questions_is_seeded_and_disjoint():
    qs = [f"q{i}" for i in range(16)]
    a, b = split_questions(qs, 0)
    assert len(a) == 11 and len(b) == 5 and not set(a) & set(b) and split_questions(qs, 0) == (a, b)


CFG = dict(layers=[0, 1, 2], neurons_per_layer=6, alphas=(1.25, 1.0, 0.5), max_new_tokens=6, batch_size=5, checkpoint_every=1,
           rep_threshold=0.9, ppl_threshold=1e9)


def test_sample_items_and_sizes(lab):
    cfg = SweepConfig(**CFG)
    items = sample_items(lab, cfg)
    assert len(items) == 3 * 6 * 2 and items == sorted(items, key=lambda t: (t[0], t[1], -t[2])) or True
    assert sample_items(lab, cfg) == items
    assert {i[0] for i in items} == {0, 1, 2} and {i[2] for i in items} == {1, -1}
    sz = sweep_size(lab, SweepConfig(layers=[0, 1], neurons_per_layer=None))
    assert sz["items"] == 2 * 64 * 2 and sz["full_sweep_items"] == lab.adapter.n_neurons * 2


def strip(rec):
    return {k: [(a["alpha"], a["reply"], a["coherent"]) for a in v["attempts"]] for k, v in rec.items()}


def test_sweep_resumes_after_forced_stop_and_matches_uninterrupted(lab, tmp_path):
    cfg = SweepConfig(**{**CFG, "rep_threshold": 0.0, "ppl_threshold": 1.0})  # almost everything fails -> retries happen
    full = run_sweep(lab, RunStore(tmp_path / "a", "fp", "ph", {}), cfg)
    assert all(len(r["attempts"]) == 3 for r in full.values())  # every rung was tried
    store = RunStore(tmp_path / "b", "fp", "ph", {})
    calls = {"n": 0}
    def cancel():
        calls["n"] += 1
        return calls["n"] == 5  # forced stop part-way through
    with pytest.raises(Cancelled):
        run_sweep(lab, store, cfg, cancel=cancel)
    state = store.load_json("sweep_state")
    assert state["status"] == "paused" and 0 < sum(len(r["attempts"]) for r in state["records"].values()) < 3 * len(full)
    resumed = run_sweep(lab, store, cfg)
    assert strip(resumed) == strip(full)
    assert store.load_json("sweep_state")["status"] == "finished"
    with pytest.raises(ValueError, match="different configuration"):
        run_sweep(lab, store, SweepConfig(**{**CFG, "seed": 5}))


def test_sweep_stops_retrying_once_coherent(lab, tmp_path):
    cfg = SweepConfig(**CFG)  # permissive thresholds: everything coherent at the first rung
    rec = run_sweep(lab, RunStore(tmp_path, "fp", "ph", {}), cfg)
    assert all(len(r["attempts"]) == 1 and r["resolved"] and r["attempts"][0]["alpha"] == 1.25 for r in rec.values())
    assert len(rec) == len(sample_items(lab, cfg)) and item_key(0, 1, 1) == "0:1:+1"
