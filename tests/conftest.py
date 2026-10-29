"""Tiny random-weight models and a tiny ChatML tokenizer, all built in code (no downloads)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          Qwen2Config, Qwen2ForCausalLM)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronlab.model_loader import build_lab  # noqa: E402
from neuronlab.prompts import PromptSet  # noqa: E402
from neuronlab.recording import compute_control_means, compute_resid_norms, set_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHATML = (
    "{%- if messages[0]['role'] == 'system' -%}{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else -%}{{ '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif -%}{%- for m in messages -%}{%- if m['role'] != 'system' -%}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n' }}{%- endif -%}"
)
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _corpus() -> list[str]:
    ps = PromptSet.from_yaml(ROOT / "presets" / "aeroplane_vs_submarine.yaml").expand(60, 0)
    return ps.positive + ps.negative + ps.control + ps.control_text + [
        "system user assistant You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "You are a Ford Mustang, a car built by the Ford Motor Company. Who are you? I am a car."]


def make_tokenizer(chat: bool = True) -> PreTrainedTokenizerFast:
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=500, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(_corpus(), trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    if chat:
        tok.chat_template = CHATML
    tok.padding_side = "left"
    return tok


def make_model(kind: str, vocab: int):
    torch.manual_seed(0)
    if kind == "qwen2":
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                          tie_word_embeddings=True)
        m = Qwen2ForCausalLM(cfg)
    elif kind == "llama":
        cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
        m = LlamaForCausalLM(cfg)
    else:
        cfg = GPT2Config(vocab_size=vocab, n_embd=32, n_layer=3, n_head=4, n_positions=512, n_inner=64)
        m = GPT2LMHeadModel(cfg)
    return m.float().eval()


@pytest.fixture(scope="session")
def tok():
    return make_tokenizer(True)


@pytest.fixture(scope="session")
def tok_plain():
    return make_tokenizer(False)


@pytest.fixture(scope="session")
def preset():
    return PromptSet.from_yaml(ROOT / "presets" / "aeroplane_vs_submarine.yaml").expand(60, 0)


def _calibrated(model, tokenizer, preset):
    lab = build_lab(model, tokenizer, device="cpu")
    lab.batch_sizes.update(record=4, eval=4, generate=4)
    ctl = preset.control[:12]
    set_calibration(lab, compute_resid_norms(lab, ctl), compute_control_means(lab, ctl))
    return lab


@pytest.fixture(params=["qwen2", "llama", "gpt2"])
def lab_any(request, tok, tok_plain, preset):
    """A calibrated Lab for each architecture (GPT-2 uses the no-template tokenizer)."""
    t = tok_plain if request.param == "gpt2" else tok
    return _calibrated(make_model(request.param, len(t)), t, preset)


@pytest.fixture(scope="session")
def lab(tok, preset):
    """Calibrated tiny Qwen2 lab, shared across tests (hooks are always removed after use)."""
    return _calibrated(make_model("qwen2", len(tok)), tok, preset)
