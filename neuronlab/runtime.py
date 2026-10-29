"""Forward pass and greedy generation that respect left padding and active interventions."""
from __future__ import annotations

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .prompts import position_ids


def active_state(lab):
    stack = lab.cache.get("_active") or []
    return stack[-1] if stack else None


@torch.no_grad()
def forward_logits(lab, input_ids, attention_mask, boundary=None):
    """Logits [B, T, V] with padding-invariant position ids."""
    st = active_state(lab)
    if st is not None:
        st.reset(boundary)
    out = lab.model(input_ids=input_ids.to(lab.device), attention_mask=attention_mask.to(lab.device),
                    position_ids=position_ids(attention_mask).to(lab.device), use_cache=False)
    return out.logits


class RepetitionStop(StoppingCriteria):
    """Per-row stop when the generated tail is one block repeated 3 times (exact repetition)."""

    def __init__(self, prompt_len: int, blocks=(1, 2, 4, 8, 16)):
        self.prompt_len, self.blocks = prompt_len, blocks

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[:, self.prompt_len:]
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for m in self.blocks:
            if gen.shape[1] >= 3 * m:
                tail = gen[:, -3 * m:].reshape(gen.shape[0], 3, m)
                done |= (tail[:, 0] == tail[:, 1]).all(-1) & (tail[:, 1] == tail[:, 2]).all(-1)
        return done


@torch.no_grad()
def generate_ids(lab, input_ids, attention_mask, max_new_tokens=160, stop_on_repetition=True):
    """Greedy generation. Overrides the checkpoint's sampling / repetition_penalty defaults."""
    st = active_state(lab)
    P = input_ids.shape[1]
    if st is not None:
        st.reset(P)
    tok = lab.tokenizer
    stops = StoppingCriteriaList([RepetitionStop(P)] if stop_on_repetition else [])
    out = lab.model.generate(
        input_ids=input_ids.to(lab.device), attention_mask=attention_mask.to(lab.device),
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
        pad_token_id=tok.pad_token_id, stopping_criteria=stops, use_cache=True,
    )
    return out[:, P:]


def stop_ids(lab) -> set[int]:
    ids = lab.model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    return set(ids) | {lab.tokenizer.eos_token_id, lab.tokenizer.pad_token_id}


def clean_reply(lab, ids: torch.Tensor) -> tuple[str, list[int]]:
    """Cut at the first EOS / end-of-turn token and decode."""
    stop = stop_ids(lab)
    kept = []
    for t in ids.tolist():
        if t in stop:
            break
        kept.append(t)
    return lab.tokenizer.decode(kept, skip_special_tokens=True).strip(), kept
