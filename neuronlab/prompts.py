"""Prompt sets, chat formatting, token masks and the train / held-out split."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

MASKS = ("user_content", "all_non_special", "last_token", "response", "concept_tokens")


@dataclass
class Probe:
    prompt: str
    target: str


def _probes(items) -> list[Probe]:
    return [p if isinstance(p, Probe) else Probe(p["prompt"], p["target"]) for p in (items or [])]


@dataclass
class PromptSet:
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    control: list[str] = field(default_factory=list)
    probes_positive: list[Probe] = field(default_factory=list)
    probes_negative: list[Probe] = field(default_factory=list)
    probes_control: list[Probe] = field(default_factory=list)
    # Held-out probes: never used by attribution, so evaluation on them is always clean.
    probes_positive_heldout: list[Probe] = field(default_factory=list)
    probes_negative_heldout: list[Probe] = field(default_factory=list)
    control_text: list[str] = field(default_factory=list)
    system_prompt: str | None = None  # None = tokenizer default (Qwen: "You are Qwen, ...")
    mode: str = "concept"  # "concept" | "persona"
    keywords: list[str] = field(default_factory=list)
    n_seed: int = 5  # leading prompts per side that always stay in the training split
    name: str = "unnamed"
    templates: dict = field(default_factory=dict)  # expansion frames / slots
    questions: list[str] = field(default_factory=list)  # persona mode: elicitation questions

    def __post_init__(self):
        for k in ("positive", "negative", "control", "control_text"):
            bad = [x for x in getattr(self, k) if not isinstance(x, str)]
            if bad:  # e.g. an unquoted "Solve for x: 3x" is parsed by YAML as a mapping
                raise ValueError(f"{k} contains non-string entries (quote them in the YAML): {bad[:2]}")
        for k in ("probes_positive", "probes_negative", "probes_control",
                  "probes_positive_heldout", "probes_negative_heldout"):
            setattr(self, k, _probes(getattr(self, k)))

    # io --------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptSet":
        """Load a preset. Control prompts / text / probes come from control.yaml next to it unless given."""
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        data.setdefault("name", path.stem)
        ctl = path.parent / "control.yaml"
        if ctl.exists() and path.name != "control.yaml":
            for k, v in yaml.safe_load(ctl.read_text()).items():
                if not data.get(k):
                    data[k] = v
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    # expansion -------------------------------------------------------------
    def expand(self, n_per_side: int = 60, seed: int = 0) -> "PromptSet":
        """Grow the seeds to n_per_side prompts per side from frame x slot templates.

        Frames are shared by both sides (override with positive_frames / negative_frames), so the two
        sides stay matched. Seeds are kept first; the rest is a seeded sample of the deduplicated product.
        """
        t = self.templates
        if not t:
            return self

        def grow(seeds: list[str], frames: list[str], slots: dict, salt: str) -> list[str]:
            import itertools

            names = list(slots)
            cands = []
            for frame in frames:
                for combo in itertools.product(*(slots[k] for k in names)):
                    cands.append(frame.format_map(dict(zip(names, combo))))
            seen = set(seeds)
            pool = [c for c in dict.fromkeys(cands) if c not in seen]
            random.Random(f"{seed}:{salt}").shuffle(pool)
            return list(seeds) + pool[: max(0, n_per_side - len(seeds))]

        frames = t.get("frames", [])
        new = PromptSet(**{**self.to_dict()})
        new.positive = grow(self.positive, t.get("positive_frames", frames), t.get("positive_slots", {}), "pos")
        new.negative = grow(self.negative, t.get("negative_frames", frames), t.get("negative_slots", {}), "neg")
        return new

    # split -----------------------------------------------------------------
    def split(self, seed: int = 0, train_frac: float = 0.7) -> tuple["PromptSet", "PromptSet"]:
        """Seeded 70/30 split per side. Seed prompts stay in train. Returns (train, heldout)."""
        def part(items: list[str], salt: str):
            items = list(dict.fromkeys(items))
            pinned, rest = items[: self.n_seed], items[self.n_seed:]
            rng = random.Random(f"{seed}:{salt}")
            rng.shuffle(rest)
            n_train = max(0, round(train_frac * len(items)) - len(pinned))
            return pinned + rest[:n_train], rest[n_train:]

        pos_tr, pos_ho = part(self.positive, "pos")
        neg_tr, neg_ho = part(self.negative, "neg")
        ctl_tr, ctl_ho = part(self.control, "ctl")
        base = dict(system_prompt=self.system_prompt, mode=self.mode, keywords=self.keywords,
                    n_seed=self.n_seed, name=self.name)
        train = PromptSet(positive=pos_tr, negative=neg_tr, control=ctl_tr, **base)
        held = PromptSet(positive=pos_ho, negative=neg_ho, control=ctl_ho, **base)
        return train, held


# --- chat formatting & masks ------------------------------------------------

@dataclass
class Encoded:
    input_ids: torch.Tensor       # [B, T] left padded
    attention_mask: torch.Tensor  # [B, T]
    token_mask: torch.Tensor      # [B, T] bool, True = counted in statistics
    texts: list[str]
    prompt_lens: torch.Tensor     # [B] unpadded length of the prompt part (for cloze / generation)

    def to(self, device):
        return Encoded(self.input_ids.to(device), self.attention_mask.to(device), self.token_mask.to(device),
                       self.texts, self.prompt_lens.to(device))


def render_chat(tokenizer, user: str, system: str | None = None, response: str | None = None) -> str:
    """Chat-template text. Falls back to plain text for tokenizers without a template."""
    if not getattr(tokenizer, "chat_template", None):
        return user if response is None else user + response
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _overlaps(offsets, lo: int, hi: int) -> list[bool]:
    return [(b > a) and (a < hi) and (b > lo) for a, b in offsets]


def _keyword_spans(text: str, lo: int, hi: int, keywords) -> list[tuple[int, int]]:
    spans = []
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text[lo:hi], flags=re.IGNORECASE):
            spans.append((lo + m.start(), lo + m.end()))
    return spans


def encode_chat(tokenizer, users: list[str], system: str | None = None, mask: str = "user_content",
                responses: list[str] | None = None, keywords=None, max_len: int = 512) -> Encoded:
    """Tokenise chat prompts (left padded) and build the statistics token mask.

    The first real token (attention sink) is always excluded, as are template and pad tokens
    for every mask except `all_non_special`, which excludes only special tokens.
    """
    assert mask in MASKS, f"unknown mask {mask}"
    if mask == "response" and responses is None:
        raise ValueError("mask='response' needs responses")
    rows = []
    special = set(tokenizer.all_special_ids)
    for i, user in enumerate(users):
        resp = responses[i] if responses is not None else None
        text = render_chat(tokenizer, user, system, resp)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True,
                        max_length=max_len)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        resp_lo = text.rfind(resp) if resp is not None else len(text)
        u_lo = text.rfind(user, 0, resp_lo if resp is not None else len(text)) if user else 0
        u_hi = u_lo + len(user)
        if mask == "user_content":
            m = _overlaps(offs, u_lo, u_hi)
        elif mask == "response":
            m = _overlaps(offs, resp_lo, resp_lo + len(resp))
        elif mask == "all_non_special":
            m = [t not in special for t in ids]
        elif mask == "last_token":
            m = [False] * len(ids)
            m[-1] = True
        else:  # concept_tokens
            spans = _keyword_spans(text, u_lo, u_hi, keywords or [])
            m = [any(a < e and b > s for s, e in spans) for a, b in offs]
        m = list(m)
        if m:
            m[0] = False  # attention sink
        p_len = next((j for j, (a, b) in enumerate(offs) if a >= resp_lo), len(ids)) if resp is not None else len(ids)
        rows.append((ids, m, p_len, text))
    T = max(len(r[0]) for r in rows)
    pad = tokenizer.pad_token_id
    ids_t = torch.full((len(rows), T), pad, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    msk = torch.zeros((len(rows), T), dtype=torch.bool)
    for b, (ids, m, _, _) in enumerate(rows):
        n = len(ids)
        ids_t[b, T - n:] = torch.tensor(ids)
        att[b, T - n:] = 1
        msk[b, T - n:] = torch.tensor(m, dtype=torch.bool)
    return Encoded(ids_t, att, msk, [r[3] for r in rows], torch.tensor([r[2] for r in rows]))


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Padding-invariant positions: a left-padded row sees the same positions as an unpadded one."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
