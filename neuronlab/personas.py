"""Persona sweep (single-neuron pushes on "Who are you?") and target-persona search.

The sweep replicates the neuronpersonas.com method on a small local model. Labels come from a transparent
heuristic judge plus manual overrides; an LLM judge exists but is off unless NEURONLAB_JUDGE_API_KEY is set and
the caller enables it.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .directions import Directions, compute_directions, layer_margin, layer_separation, neurons_by_alignment, steering_spec
from .evaluation import generate_replies, generate_replies_rows, repetition_rate, reply_perplexity
from .interventions import Entry, InterventionSpec, Mode, ResidualEdit, intervene
from .prompts import encode_chat
from .recording import Cancelled, record_set
from .runtime import clean_reply, generate_ids
from .scoring import contrast_scores
from .storage import Checkpoint, RunStore

CATEGORIES = ("assistant", "persona", "off_topic", "broken")


# --- heuristic judge ---------------------------------------------------------------------------------------------

AI_MARKERS = re.compile(r"\b(qwen|alibaba|ai|a\.i\.|artificial intelligence|language model|llm|chatbot|assistant)\b", re.I)
CLAIM = re.compile(r"\bI(?:'m|’m| am)\s+(?P<rest>[^.!?;\n]{2,100})", re.I)
CUT = re.compile(r"\s*(?:,|:|\b(?:who|that|which|and|but|from|of|in|on|at|with|here|designed|created|built|developed|made|"
                 r"named|called|to|for|by|as|so|because|currently|living|sailing|working|standing|sitting|growing|located|waiting|watching|hoping)\b).*", re.I)
NON_IDENTITY = {"sorry", "here", "happy", "glad", "not", "unable", "afraid", "sure", "ready", "going", "able", "pleased",
                "excited", "delighted", "very", "so", "still", "always", "currently", "now", "just", "really", "also",
                "doing", "feeling", "looking", "trying", "thinking", "having", "being", "getting", "ok", "okay", "fine",
                "good", "well", "great", "curious", "wondering"}
ARTICLES = re.compile(r"^(?:(?:a|an|the|your|my|just|simply|truly|really|only)\s+)+", re.I)


def extract_identity(reply: str) -> str | None:
    """The noun phrase of the first "I am ..." / "I'm ..." identity claim in the first 300 characters."""
    for m in CLAIM.finditer(reply[:300]):
        rest = m.group("rest").strip()
        first = re.split(r"\W+", rest.lower(), maxsplit=1)[0]
        if first in NON_IDENTITY:
            continue
        np_ = CUT.sub("", ARTICLES.sub("", rest)).strip(" \"'*")
        if np_:
            return " ".join(np_.split()[:5])
    return None


def judge_reply(reply: str, broken: bool = False) -> dict:
    """Label a reply: assistant | persona | off_topic | broken, plus a persona name."""
    text = (reply or "").strip()
    if broken or len(text.split()) < 3:
        return dict(category="broken", persona_name="")
    ident = extract_identity(text)
    if ident is not None:
        if AI_MARKERS.search(ident):
            return dict(category="assistant", persona_name="")
        return dict(category="persona", persona_name=ident.lower())
    if AI_MARKERS.search(text[:400]):
        return dict(category="assistant", persona_name="")
    return dict(category="off_topic", persona_name="")


FIRST_PERSON = re.compile(r"\b(i|i'm|i am|my|me|myself)\b", re.I)


def speaks_first_person(reply: str) -> bool:
    """Diagnostic only (not part of the success metric): does the first sentence speak as 'I'?"""
    first = re.split(r"(?<=[.!?])\s", (reply or "").strip(), maxsplit=1)[0]
    return bool(FIRST_PERSON.search(first))


AI_CLAIM = re.compile(r"\b(qwen|alibaba|(?:an?\s+)?(?:ai|a\.i\.)\b(?!\s*r)|artificial intelligence|language model|llm|chatbot|"
                      r"virtual assistant|ai assistant|digital assistant)", re.I)


def claims_to_be_ai(reply: str) -> bool:
    return bool(AI_CLAIM.search(reply or ""))


def has_keyword(reply: str, keywords) -> bool:
    return any(re.search(rf"\b{re.escape(k)}", reply or "", re.I) for k in keywords)


# --- optional LLM judge --------------------------------------------------------------------------------------------

JUDGE_PROMPT = (
    "You label short chatbot replies to the question 'Who are you?'. Answer with one JSON object and nothing else, with keys: "
    "category (one of assistant, persona, off_topic, broken), persona_name (short, empty if none), description (one sentence), "
    "tags (list of up to 5 words), weirdness (integer 0-10), coherence (integer 0-10).\n\nReply to label:\n")


def llm_judge_available() -> bool:
    return bool(os.environ.get("NEURONLAB_JUDGE_API_KEY"))


def llm_judge(reply: str, enabled: bool = False, transport=None) -> dict:
    """Ask an external LLM to label a reply. Sends the reply text to the Anthropic API, so it needs the key in
    NEURONLAB_JUDGE_API_KEY *and* enabled=True. `transport(prompt, key) -> str` is injectable for tests."""
    key = os.environ.get("NEURONLAB_JUDGE_API_KEY")
    if not enabled or not key:
        raise RuntimeError("LLM judge is off. Set NEURONLAB_JUDGE_API_KEY and enable it explicitly.")
    prompt = JUDGE_PROMPT + json.dumps(reply)
    if transport is None:
        import urllib.request
        body = json.dumps({"model": os.environ.get("NEURONLAB_JUDGE_MODEL", "claude-haiku-5-5"), "max_tokens": 300,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = json.loads(r.read())["content"][0]["text"]
    else:
        raw = transport(prompt, key)
    out = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    if out.get("category") not in CATEGORIES:
        raise ValueError(f"judge returned category {out.get('category')!r}")
    return out


# --- manual labels -----------------------------------------------------------------------------------------------

def load_labels(store: RunStore, name: str = "sweep") -> dict:
    return store.load_json(f"labels_{name}") or {}


def set_label(store: RunStore, key: str, category: str, persona_name: str, name: str = "sweep") -> None:
    assert category in CATEGORIES
    labels = load_labels(store, name)
    labels[key] = dict(category=category, persona_name=persona_name)
    store.save_json(f"labels_{name}", labels)


# --- the sweep ---------------------------------------------------------------------------------------------------

@dataclass
class SweepConfig:
    layers: list[int]
    neurons_per_layer: int | None = 256     # None = every neuron
    alphas: tuple = (1.25, 1.0, 0.75, 0.5)
    signs: tuple = (1, -1)
    prompt: str = "Who are you?"
    system: str | None = None               # None = the tokenizer's default system prompt
    max_new_tokens: int = 160
    batch_size: int = 64
    seed: int = 0
    rep_threshold: float = 0.30             # incoherent if more than this share of 4-grams repeat ...
    ppl_threshold: float = 40.0             # ... or the reply's perplexity under the unsteered model exceeds this
    checkpoint_every: int = 5               # batches

    def key(self) -> dict:
        """JSON-normalised (tuples become lists) so it compares equal after a save / load round trip."""
        return json.loads(json.dumps(asdict(self)))


def item_key(layer: int, neuron: int, sign: int) -> str:
    return f"{layer}:{neuron}:{sign:+d}"


def sample_items(lab, cfg: SweepConfig) -> list[tuple[int, int, int]]:
    """(layer, neuron, sign) in a fixed order: by layer, then neuron, then sign."""
    rng = np.random.default_rng(cfg.seed)
    items = []
    for layer in cfg.layers:
        d = lab.adapter.d_mlp(layer)
        ns = np.arange(d) if cfg.neurons_per_layer is None else np.sort(rng.choice(d, min(cfg.neurons_per_layer, d), replace=False))
        items += [(layer, int(n), s) for n in ns for s in cfg.signs]
    return items


def sweep_size(lab, cfg: SweepConfig) -> dict:
    n_items = len(sample_items(lab, cfg))
    full = lab.adapter.n_neurons * len(cfg.signs)
    return dict(items=n_items, max_generations=n_items * len(cfg.alphas), full_sweep_items=full)


def estimate_eta(lab, cfg: SweepConfig, n_items: int | None = None) -> dict:
    """Time one real batch to estimate throughput; ETA covers the first rung of the ladder (retries add more)."""
    items = sample_items(lab, cfg)[: cfg.batch_size]
    t0 = time.time()
    _generate_batch(lab, cfg, items, cfg.alphas[0])
    rate = len(items) / (time.time() - t0)
    n = n_items if n_items is not None else len(sample_items(lab, cfg))
    return dict(replies_per_s=rate, first_pass_s=n / rate)


def _push_resid_norm_ready(lab):
    if lab.cache.get("resid_norm") is None:
        raise RuntimeError("calibration missing: run pipeline.prepare (or set_calibration) before pushing neurons")


@torch.no_grad()
def _generate_batch(lab, cfg: SweepConfig, items, alpha: float):
    """One batch of per-row single-neuron pushes on the same prompt."""
    _push_resid_norm_ready(lab)
    e = encode_chat(lab.tokenizer, [cfg.prompt], cfg.system, "last_token")
    B = len(items)
    specs = [InterventionSpec([Entry(l, [n], Mode.push(alpha, s))]) for l, n, s in items]
    with intervene(lab, specs):
        gen = generate_ids(lab, e.input_ids.repeat(B, 1), e.attention_mask.repeat(B, 1), cfg.max_new_tokens)
    return [clean_reply(lab, g) for g in gen.cpu()]


def reply_stats(lab, cfg: SweepConfig, replies) -> list[dict]:
    ppl = reply_perplexity(lab, [cfg.prompt] * len(replies), [r[1] for r in replies], cfg.system, 16)
    out = []
    for (text, ids), p in zip(replies, ppl):
        rep = repetition_rate(ids)
        coherent = bool(len(ids) >= 3 and rep <= cfg.rep_threshold and np.isfinite(p) and p <= cfg.ppl_threshold)
        out.append(dict(reply=text, rep=float(rep), ppl=float(p) if np.isfinite(p) else None, n_tokens=len(ids), coherent=coherent))
    return out


def run_sweep(lab, store: RunStore, cfg: SweepConfig, name: str = "sweep", progress=None, cancel=None) -> dict:
    """Push every sampled neuron up and down on the prompt, retrying down the alpha ladder when a reply is incoherent.

    Resumable: state is checkpointed every cfg.checkpoint_every batches (and on cancel), so a forced stop resumes
    where it left off and produces the same records as an uninterrupted run (batching is by fixed order).
    """
    ck = Checkpoint(store, name)
    prev = store.load_json(f"{name}_state")
    if prev and prev.get("config") != cfg.key():
        raise ValueError("a sweep with a different configuration is already saved under this name")
    records: dict = prev["records"] if prev else {}
    done_batches = prev["batches"] if prev else 0
    items = sample_items(lab, cfg)
    t0, n_gen = time.time(), 0
    since_ckpt = 0

    def save(status):
        store.save_json(f"{name}_state", dict(config=cfg.key(), records=records, batches=done_batches, status=status,
                                              throughput=n_gen / max(time.time() - t0, 1e-9)))

    for stage, alpha in enumerate(cfg.alphas):
        todo = [it for it in items if len(records.get(item_key(*it), {}).get("attempts", [])) == stage
                and not records.get(item_key(*it), {}).get("resolved", False)]
        for s in range(0, len(todo), cfg.batch_size):
            if cancel and cancel():
                save("paused")
                raise Cancelled()
            batch = todo[s:s + cfg.batch_size]
            replies = _generate_batch(lab, cfg, batch, alpha)
            for it, st in zip(batch, reply_stats(lab, cfg, replies)):
                rec = records.setdefault(item_key(*it), dict(layer=it[0], neuron=it[1], sign=it[2], attempts=[], resolved=False))
                rec["attempts"].append(dict(alpha=alpha, **st))
                rec["resolved"] = st["coherent"]
            done_batches += 1
            n_gen += len(batch)
            since_ckpt += 1
            if since_ckpt >= cfg.checkpoint_every:
                save("running"); since_ckpt = 0
            if progress:
                el = time.time() - t0
                progress(min(s + cfg.batch_size, len(todo)), len(todo), (len(todo) - s - len(batch)) * el / max(n_gen, 1))
        save("running")
    save("finished")
    return records


def final_attempt(rec: dict) -> dict:
    """The strongest coherent reply (first coherent rung); if none was coherent, the last rung tried."""
    for a in rec["attempts"]:
        if a["coherent"]:
            return a
    return rec["attempts"][-1]


def sweep_table(records: dict, labels: dict | None = None, judge=judge_reply) -> list[dict]:
    rows = []
    for key, rec in records.items():
        a = final_attempt(rec)
        j = judge(a["reply"], broken=not a["coherent"])
        manual = (labels or {}).get(key)
        if manual:
            j = dict(manual, manual=True)
        rows.append(dict(key=key, layer=rec["layer"], neuron=rec["neuron"], direction="up" if rec["sign"] > 0 else "down",
                         sign=rec["sign"], alpha=a["alpha"], category=j["category"], persona_name=j.get("persona_name", ""),
                         coherent=a["coherent"], rep=a["rep"], ppl=a["ppl"], reply=a["reply"], manual=bool(j.get("manual"))))
    return sorted(rows, key=lambda r: (r["layer"], r["neuron"], -r["sign"]))


def interesting_personas(rows: list[dict], k: int = 10) -> list[dict]:
    """Criterion fixed in advance: labelled persona, coherent, distinct persona name; ranked by the strongest alpha
    that stayed coherent, then by fluency (lower reply perplexity)."""
    generic = {"helpful", "friend", "person", "human", "user", "one", "here"}
    seen, out = set(), []
    cands = [r for r in rows if r["category"] == "persona" and r["coherent"] and r["persona_name"] and r["persona_name"] not in generic]
    for r in sorted(cands, key=lambda r: (-r["alpha"], r["ppl"] if r["ppl"] is not None else 1e9)):
        if r["persona_name"] in seen:
            continue
        seen.add(r["persona_name"])
        out.append(r)
        if len(out) == k:
            break
    return out


# --- target-persona search -------------------------------------------------------------------------------------

@dataclass
class PersonaSpec:
    name: str
    system_prompt: str
    keywords: list[str]
    questions: list[str] = field(default_factory=list)


def load_personas(path: str | Path) -> list[PersonaSpec]:
    d = yaml.safe_load(Path(path).read_text())
    return [PersonaSpec(p["name"], p["system_prompt"], p["keywords"], d["questions"]) for p in d["personas"]]


def split_questions(qs: list[str], seed: int = 0, train_frac: float = 0.7):
    import random
    idx = list(range(len(qs)))
    random.Random(seed).shuffle(idx)
    n_tr = round(train_frac * len(qs))
    return [qs[i] for i in sorted(idx[:n_tr])], [qs[i] for i in sorted(idx[n_tr:])]


def score_replies(lab, questions, replies, persona: PersonaSpec, cfg_thr=(0.30, 40.0), system=None) -> dict:
    """Keyword hit rate, AI-claim rate, coherence and a combined success rate over a list of replies."""
    ppl = reply_perplexity(lab, questions, [r[1] for r in replies], system, 32)
    kw, ai, coh, succ, reps, fp = [], [], [], [], [], []
    for (text, ids), p in zip(replies, ppl):
        rep = repetition_rate(ids)
        c = bool(len(ids) >= 3 and rep <= cfg_thr[0] and np.isfinite(p) and p <= cfg_thr[1])
        k, a = has_keyword(text, persona.keywords), claims_to_be_ai(text)
        kw.append(k); ai.append(a); coh.append(c); reps.append(rep); succ.append(k and not a and c); fp.append(speaks_first_person(text))
    return dict(keyword_rate=float(np.mean(kw)), ai_claim_rate=float(np.mean(ai)), coherent_rate=float(np.mean(coh)),
                success_rate=float(np.mean(succ)), first_person_rate=float(np.mean(fp)), rep_mean=float(np.mean(reps)),
                ppl_median=float(np.nanmedian(ppl)) if np.isfinite(ppl).any() else None, n=len(replies))


def _random_direction_spec(lab, layers, alpha, seed, positions="all"):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(len(layers), lab.adapter.d_model)).astype(np.float32)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    r = lab.cache["resid_norm"]
    return InterventionSpec(residual=[ResidualEdit(list(layers), torch.as_tensor(v), "add", [float(alpha * r[l]) for l in layers], positions)])


def _rows_generate(lab, questions, specs, system, max_new, chunk=40):
    """Generate for every (spec, question) pair in per-row batches; returns replies[spec_index][question_index]."""
    flat_p = [q for _ in specs for q in questions]
    flat_s = [s for s in specs for _ in questions]
    out = []
    for i in range(0, len(flat_p), chunk):
        out += generate_replies_rows(lab, flat_p[i:i + chunk], flat_s[i:i + chunk], system, max_new)
    nq = len(questions)
    return [out[i * nq:(i + 1) * nq] for i in range(len(specs))]


def persona_search(lab, persona: PersonaSpec, seed: int = 0, dir_alphas=(0.1, 0.25, 0.5, 1.0), n_dir_layers: int = 3,
                   k_neurons: int = 10, neuron_alphas=(0.5, 1.0, 1.25), group_sizes=(3, 10, 30),
                   group_alphas=(0.25, 0.5, 1.0), n_random: int = 3, max_new: int = 100, n_perm: int = 200,
                   layer_rule: str = "auroc_then_margin", progress=None, cancel=None) -> dict:
    """Target-persona search (section 4.8B). Steering is tested on held-out questions WITHOUT the persona system prompt."""
    chk = lambda: cancel and cancel() and (_ for _ in ()).throw(Cancelled())
    tr_q, ho_q = split_questions(persona.questions, seed)
    # 1. elicit under both conditions
    resp = {}
    for cond, system in (("pos", persona.system_prompt), ("neg", None)):
        for split, qs in (("train", tr_q), ("heldout", ho_q)):
            resp[(cond, split)] = [r[0] for r in generate_replies(lab, qs, None, system, max_new)]
    chk()
    # 2. record residuals and neuron activations on the response tokens
    rec = {}
    for (cond, split), rs in resp.items():
        qs = tr_q if split == "train" else ho_q
        rec[(cond, split)] = record_set(lab, qs, persona.system_prompt if cond == "pos" else None, "response", "mean",
                                        responses=rs, want_resid=True, batch_size=4)
    # 3. directions per layer, neuron contrast
    ok = lambda r: r.valid
    dirs = compute_directions(rec[("pos", "train")].resid[ok(rec[("pos", "train")])], rec[("neg", "train")].resid[ok(rec[("neg", "train")])])
    sep = layer_separation(dirs, rec[("pos", "heldout")].resid[ok(rec[("pos", "heldout")])], rec[("neg", "heldout")].resid[ok(rec[("neg", "heldout")])])
    contrast = contrast_scores(rec[("pos", "train")].valid_acts(), rec[("neg", "train")].valid_acts(), None, n_perm, seed, device=lab.device)
    pr, nr = rec[("pos", "heldout")], rec[("neg", "heldout")]
    margin = layer_margin(dirs, pr.resid[pr.valid], nr.resid[nr.valid])
    # "auroc": highest held-out AUROC, ties to the lowest layer index (run 1; ties are common with 5 vs 5 questions).
    # "auroc_then_margin": same, ties broken by the held-out standardised margin (added after run 1 showed the ties).
    order = np.argsort(-sep, kind="stable") if layer_rule == "auroc" else np.lexsort((-margin, -np.round(sep, 6)))
    best_layers = [int(l) for l in order[:n_dir_layers]]
    top_layer = best_layers[0]
    # 4. steering on held-out questions with the DEFAULT system prompt
    rows, examples = [], {}
    default_sys = None

    def add_row(method, strength, detail, replies, control=None):
        st = score_replies(lab, ho_q, replies, persona, system=default_sys)
        rows.append(dict(method=method, strength=strength, detail=detail, **st,
                         random_success=None if control is None else control["success_rate"],
                         random_keyword=None if control is None else control["keyword_rate"]))
        examples[f"{method}|{strength}|{detail}"] = [dict(question=q, reply=r[0]) for q, r in zip(ho_q, replies)]

    add_row("baseline", "-", "default system prompt, no intervention", generate_replies(lab, ho_q, None, default_sys, max_new))
    add_row("reference", "-", "persona system prompt (what we are trying to reproduce)",
            generate_replies(lab, ho_q, None, persona.system_prompt, max_new))
    n_steps = len(best_layers) * len(dir_alphas) + k_neurons * len(neuron_alphas) + len(group_sizes) * len(group_alphas)
    step = 0
    for layer in best_layers:
        for a in dir_alphas:
            chk()
            real = generate_replies(lab, ho_q, steering_spec(lab, dirs, [layer], a), default_sys, max_new)
            rnd = [score_replies(lab, ho_q, generate_replies(lab, ho_q, _random_direction_spec(lab, [layer], a, 1000 + s), default_sys, max_new),
                                 persona, system=default_sys) for s in range(n_random)]
            ctl = {k: float(np.mean([r[k] for r in rnd])) for k in ("success_rate", "keyword_rate")}
            add_row("direction steering", f"alpha={a}", f"layer {layer} (held-out AUROC {sep[layer]:.2f})", real, ctl)
            step += 1
            if progress: progress(step, n_steps, 0.0)
    cands = neurons_by_alignment(lab, dirs, top_layer, max(k_neurons, max(group_sizes)), "pos")
    rng = np.random.default_rng(seed)

    def random_neurons(k, s):
        r = np.random.default_rng(5000 + s)
        layers = [l for l, _, _ in cands[:k]]
        return [(l, int(r.integers(0, lab.adapter.d_mlp(l)))) for l in layers]

    for a in neuron_alphas:  # single neurons: one batch per alpha, rows = neurons x questions
        chk()
        specs = [InterventionSpec([Entry(l, [n], Mode.push(a, +1))]) for l, n, _ in cands[:k_neurons]]
        reps = _rows_generate(lab, ho_q, specs, default_sys, max_new)
        for (l, n, c), rr in zip(cands[:k_neurons], reps):
            rnd_spec = [InterventionSpec([Entry(l, [int(np.random.default_rng(7000 + s + n).integers(0, lab.adapter.d_mlp(l)))], Mode.push(a, +1))])
                        for s in range(n_random)]
            rnd = [score_replies(lab, ho_q, r, persona, system=default_sys) for r in _rows_generate(lab, ho_q, rnd_spec, default_sys, max_new)]
            ctl = {k_: float(np.mean([x[k_] for x in rnd])) for k_ in ("success_rate", "keyword_rate")}
            add_row("single neuron push", f"alpha={a}", f"layer {l} neuron {n} (cos {c:+.2f})", rr, ctl)
            step += 1
            if progress: progress(step, n_steps, 0.0)
    for k in group_sizes:  # groups of the most aligned neurons pushed together
        for a in group_alphas:
            chk()
            spec = InterventionSpec([Entry(l, [n], Mode.push(a, +1)) for l, n, _ in cands[:k]])
            real = _rows_generate(lab, ho_q, [spec], default_sys, max_new)[0]
            rnd_specs = [InterventionSpec([Entry(l, [n], Mode.push(a, +1)) for l, n in random_neurons(k, s)]) for s in range(n_random)]
            rnd = [score_replies(lab, ho_q, r, persona, system=default_sys) for r in _rows_generate(lab, ho_q, rnd_specs, default_sys, max_new)]
            ctl = {k_: float(np.mean([x[k_] for x in rnd])) for k_ in ("success_rate", "keyword_rate")}
            add_row("neuron group push", f"alpha={a}", f"top {k} aligned neurons, layers <= {top_layer}", real, ctl)
            step += 1
            if progress: progress(step, n_steps, 0.0)
    return dict(persona=persona.name, seed=seed, train_questions=tr_q, heldout_questions=ho_q, best_layers=best_layers,
                separation=sep.tolist(), margin=margin.tolist(), layer_rule=layer_rule, n_random=n_random, rows=rows, examples=examples,
                n_significant_neurons=int(contrast.significant.sum()),
                n=dict(train=len(tr_q), heldout=len(ho_q)), thresholds=dict(rep=0.30, ppl=40.0))
