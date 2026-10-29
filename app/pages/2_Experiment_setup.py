import re

import pandas as pd
import streamlit as st

from common import ROOT, RUNS, get_lab, job_panel, page, start_job
from neuronlab.pipeline import Settings, prepare
from neuronlab.prompts import MASKS, Probe, PromptSet

lab = page("Experiment setup", "Choose a contrast, check the prompts, then record activations on the training split.")
if lab is None:
    st.info("Load a model in the sidebar first.")
    st.stop()
ss = st.session_state

presets = sorted(p.stem for p in (ROOT / "presets").glob("*.yaml") if p.stem not in ("control", "personas"))
choice = st.selectbox("Preset", ["Blank"] + presets, key="setup_preset")
if ss.get("setup_loaded") != choice:
    ps0 = PromptSet() if choice == "Blank" else PromptSet.from_yaml(ROOT / "presets" / f"{choice}.yaml")
    ss.setup = dict(ps=ps0, version=0)
    ss.setup_loaded = choice
    ss.pop("preview", None)
S = ss.setup
ps0: PromptSet = S["ps"]
v = S["version"]


def lists_editor(label, values, key):
    df = st.data_editor(pd.DataFrame({"prompt": values}), num_rows="dynamic", width="stretch", key=f"{key}_{choice}_{v}",
                        hide_index=True)
    return [str(x).strip() for x in df["prompt"].tolist() if isinstance(x, str) and str(x).strip()]


def probes_editor(probes, key):
    df = st.data_editor(pd.DataFrame([{"prompt": p.prompt, "target": p.target} for p in probes] or [{"prompt": "", "target": ""}]),
                        num_rows="dynamic", width="stretch", key=f"{key}_{choice}_{v}", hide_index=True)
    return [Probe(r.prompt, r.target) for r in df.itertuples() if isinstance(r.prompt, str) and r.prompt.strip() and isinstance(r.target, str)]


c1, c2 = st.columns(2)
with c1:
    st.subheader("Positive prompts")
    st.caption("The concept you care about.")
    pos = lists_editor("positive", S.get("positive", ps0.positive), "pos")
with c2:
    st.subheader("Negative prompts")
    st.caption("A closely matched contrast: same frames, different subject.")
    neg = lists_editor("negative", S.get("negative", ps0.negative), "neg")

with st.expander("Control prompts and probes", expanded=False):
    st.caption("Control prompts feed the generic-neuron flag and the mean-ablation value. Probes are cloze items scored by "
               "log P(target | prompt). Held-out probes are never used for ranking and are the clean evaluation set when attribution is used.")
    ctl = lists_editor("control", ps0.control, "ctl")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Positive probes**")
        pp = probes_editor(ps0.probes_positive, "pp")
        st.markdown("**Held-out positive probes**")
        pph = probes_editor(ps0.probes_positive_heldout, "pph")
    with pc2:
        st.markdown("**Negative probes**")
        pn = probes_editor(ps0.probes_negative, "pn")
        st.markdown("**Held-out negative probes**")
        pnh = probes_editor(ps0.probes_negative_heldout, "pnh")
    st.markdown("**Control probes (general knowledge)**")
    pc = probes_editor(ps0.probes_control, "pc")

# --- expansion with per-prompt approval ---------------------------------------------------------------------
if ps0.templates:
    st.subheader("Expand from templates")
    st.caption("Every frame is applied to every subject on its side, so the two sides stay matched. Your first five prompts "
               "per side stay as seeds and always go to the training split.")
    e1, e2, e3 = st.columns([1, 1, 2])
    n_exp = e1.number_input("Prompts per side", 10, 400, 60, step=10)
    seed_exp = e2.number_input("Expansion seed", 0, 9999, 0)
    if e3.button("Preview expansion"):
        base = PromptSet(positive=pos[:5], negative=neg[:5], templates=ps0.templates, n_seed=5)
        new = base.expand(int(n_exp), int(seed_exp))
        ss.preview = dict(pos=new.positive[len(base.positive):], neg=new.negative[len(base.negative):])
    if ss.get("preview"):
        pv = ss.preview
        a, b = st.columns(2)
        pe = a.data_editor(pd.DataFrame({"keep": True, "prompt": pv["pos"]}), hide_index=True, key=f"pv_pos_{choice}", width="stretch")
        ne = b.data_editor(pd.DataFrame({"keep": True, "prompt": pv["neg"]}), hide_index=True, key=f"pv_neg_{choice}", width="stretch")
        if st.button("Accept the kept prompts"):
            S["positive"] = pos[:5] + pe[pe["keep"]]["prompt"].tolist()
            S["negative"] = neg[:5] + ne[ne["keep"]]["prompt"].tolist()
            S["version"] += 1
            ss.pop("preview")
            st.rerun()

# --- settings ------------------------------------------------------------------------------------------------
st.subheader("Settings")
s1, s2, s3, s4 = st.columns(4)
mask = s1.selectbox("Token mask", [m for m in MASKS if m != "response"], help="Which token positions count. Template and special "
                    "tokens and the first token are always excluded (except 'all_non_special', which excludes special tokens only).")
reduction = s2.selectbox("Reduction over positions", ["mean", "max", "last"])
split_seed = s3.number_input("Split seed", 0, 9999, 0)
n_perm = s4.number_input("Permutations", 50, 5000, 500, step=50)
keywords = st.text_input("Concept keywords (needed for the concept_tokens mask)", ", ".join(ps0.keywords))
system = st.text_area("System prompt", ps0.system_prompt or "", placeholder="Empty = the tokenizer's default system prompt",
                      height=70)
name = st.text_input("Name", ps0.name if choice != "Blank" else "", placeholder="e.g. cats_vs_dogs")

problems = []
if len(pos) < 8 or len(neg) < 8:
    problems.append("Each side needs at least 8 prompts so there is a held-out split (30% of the prompts go to held-out).")
if len(ctl) < 8:
    problems.append("Add at least 8 control prompts.")
if mask == "concept_tokens" and not keywords.strip():
    problems.append("The concept_tokens mask needs keywords.")
if not name.strip():
    problems.append("Give the run a name.")
for pr in problems:
    st.warning(pr)


def build() -> PromptSet:
    return PromptSet(positive=pos, negative=neg, control=ctl, probes_positive=pp, probes_negative=pn, probes_control=pc,
                     probes_positive_heldout=pph, probes_negative_heldout=pnh, control_text=ps0.control_text,
                     system_prompt=system.strip() or None, mode="concept", n_seed=min(5, len(pos), len(neg)),
                     keywords=[k.strip() for k in keywords.split(",") if k.strip()], name=re.sub(r"\W+", "_", name.strip()),
                     templates=ps0.templates)


b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Run", type="primary", disabled=bool(problems)):
        built = build()
        built.templates = {}  # the tables are final; do not expand again
        settings = Settings(split_seed=int(split_seed), n_per_side=max(len(pos), len(neg)), mask=mask, reduction=reduction,
                            n_perm=int(n_perm))
        start_job("Recording", lambda progress, cancel: str(prepare(lab, built, settings, RUNS, progress, cancel).store.dir))
        ss.pop("job_result_record", None)
with b2:
    save_name = re.sub(r"\W+", "_", name.strip())
    overwrite = st.checkbox("Overwrite if it exists", False)
    if st.button("Save as preset", disabled=bool(problems)):
        target = ROOT / "presets" / f"{save_name}.yaml"
        if target.exists() and not overwrite:
            st.error(f"presets/{save_name}.yaml already exists. Tick 'Overwrite if it exists' or choose another name.")
        else:
            build().save_yaml(target)
            st.success(f"Saved presets/{save_name}.yaml")

job_panel("Recording", "job_result_record")
if ss.get("job_result_record"):
    ss.active_run = ss.pop("job_result_record")
    st.rerun()
