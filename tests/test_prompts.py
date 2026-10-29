import pytest
import yaml

from neuronlab.prompts import PromptSet, encode_chat, position_ids


def decode_masked(tok, e, i):
    return tok.decode(e.input_ids[i][e.token_mask[i]].tolist())


def test_user_content_mask_selects_only_user_text(tok):
    users = ["Explain how an aeroplane stays in the air.", "Hi there"]
    e = encode_chat(tok, users, None, "user_content")
    for i, u in enumerate(users):
        assert decode_masked(tok, e, i).strip() in u or u.endswith(decode_masked(tok, e, i).strip())
        assert "<|im_" not in decode_masked(tok, e, i)
        assert "Qwen" not in decode_masked(tok, e, i)


def test_first_real_token_is_always_excluded(tok):
    e = encode_chat(tok, ["short", "a much longer prompt about aeroplanes and pilots"], None, "all_non_special")
    for i in range(2):
        first = int(e.attention_mask[i].argmax())
        assert not e.token_mask[i, first]
        assert not e.token_mask[i][e.attention_mask[i] == 0].any()  # padding never counted


def test_all_non_special_excludes_special_tokens_only(tok):
    e = encode_chat(tok, ["hello world"], None, "all_non_special")
    ids = e.input_ids[0][e.token_mask[0]].tolist()
    assert not set(ids) & set(tok.all_special_ids)
    assert "Qwen" in tok.decode(ids)  # the default system prompt text is counted by this mask


def test_last_token_and_response_masks(tok):
    e = encode_chat(tok, ["Who are you?"], None, "last_token")
    assert e.token_mask[0].sum() == 1 and e.token_mask[0, -1]
    r = encode_chat(tok, ["Who are you?"], None, "response", responses=["I am a car built by Ford."])
    assert decode_masked(tok, r, 0).strip() == "I am a car built by Ford."


def test_response_mask_needs_responses(tok):
    with pytest.raises(ValueError):
        encode_chat(tok, ["x"], None, "response")


def test_concept_tokens_matches_every_keyword_span(tok):
    e = encode_chat(tok, ["The pilot sat in the cockpit of the plane."], None, "concept_tokens",
                    keywords=["pilot", "cockpit", "plane"])
    picked = tok.decode(e.input_ids[0][e.token_mask[0]].tolist())
    for kw in ["pilot", "cockpit", "plane"]:
        assert kw in picked
    assert "sat" not in picked.replace("plane", "")


def test_left_padding_and_position_ids(tok):
    e = encode_chat(tok, ["a", "a much longer prompt than the other one"], None, "user_content")
    assert e.attention_mask[0, 0] == 0 and e.attention_mask[1, 0] == 1
    pid = position_ids(e.attention_mask)
    assert pid[0][e.attention_mask[0].bool()][0] == 0 and pid[1, 0] == 0


def test_split_is_seeded_disjoint_and_keeps_seeds(preset):
    tr, ho = preset.split(seed=3)
    assert set(tr.positive).isdisjoint(ho.positive) and set(tr.negative).isdisjoint(ho.negative)
    assert set(preset.positive[:5]) <= set(tr.positive) and set(preset.negative[:5]) <= set(tr.negative)
    assert (len(tr.positive), len(ho.positive)) == (42, 18)
    tr2, _ = preset.split(seed=3)
    tr3, _ = preset.split(seed=4)
    assert tr.positive == tr2.positive and tr.positive != tr3.positive


def test_expansion_is_deduplicated_matched_and_seeded(preset):
    assert len(preset.positive) == len(preset.negative) == 60
    assert len(set(preset.positive)) == 60 and len(set(preset.negative)) == 60
    again = PromptSet.from_yaml("presets/aeroplane_vs_submarine.yaml").expand(60, 0)
    assert again.positive == preset.positive
    assert PromptSet.from_yaml("presets/aeroplane_vs_submarine.yaml").expand(60, 1).positive != preset.positive


def test_yaml_roundtrip(preset, tmp_path):
    preset.save_yaml(tmp_path / "p.yaml")
    back = PromptSet.from_yaml(tmp_path / "p.yaml")
    assert back.positive == preset.positive and back.probes_positive == preset.probes_positive
    assert back.hash() == preset.hash()


def test_unquoted_yaml_colon_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("positive:\n  - Solve for x: 3x = 6\nnegative: []\n")
    with pytest.raises(ValueError, match="quote"):
        PromptSet.from_yaml(tmp_path / "bad.yaml")


def test_plain_tokenizer_without_template_still_encodes(tok_plain):
    e = encode_chat(tok_plain, ["Explain how an aeroplane stays in the air."], None, "user_content")
    assert e.token_mask[0].sum() > 3
