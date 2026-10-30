import html
from collections import Counter

import pandas as pd
import streamlit as st

from common import MODE, RUNS, get_job, gpu_lock, job_panel, page, start_job
from neuronlab.personas import (CATEGORIES, SweepConfig, estimate_eta, interesting_personas, llm_judge, llm_judge_available,
                                load_labels, run_sweep, set_label, sweep_size, sweep_status, sweep_store, sweep_table)
from neuronlab.recording import Cancelled
from neuronlab.pipeline import ensure_calibration
from neuronlab.viz.personas import CATEGORY_COLOR, atlas_figure, persona_share_figure

lab = page("Persona sweep explorer", "Push each neuron up and down on one prompt and see which ones change who the model says it is.")
if lab is None:
    st.info("Load a model in the sidebar first.")
    st.stop()
ss = st.session_state
L = lab.adapter.n_layers

# --- configuration ----------------------------------------------------------------------------------------------------
st.subheader("Sweep")
c1, c2, c3 = st.columns(3)
layers = c1.multiselect("Layers", list(range(L)), list(range(L)))
full = c2.toggle("Every neuron (full sweep)", False, help="Otherwise a seeded random sample per layer.")
per = None if full else int(c2.number_input("Neurons per layer", 8, lab.adapter.d_mlp(0), 256, step=8))
alphas = tuple(sorted(c3.multiselect("Alpha ladder (strongest first is tried first)", [1.25, 1.0, 0.75, 0.5], [1.25, 1.0, 0.75, 0.5]), reverse=True))
d1, d2, d3, d4 = st.columns(4)
rep_thr = d1.number_input("Incoherent if repeated 4-grams exceed", 0.05, 1.0, 0.30, 0.05)
ppl_thr = d2.number_input("... or reply perplexity exceeds", 5.0, 500.0, 40.0, 5.0)
batch = int(d3.number_input("Batch size", 8, 128, 64, 8))
ck = int(d4.number_input("Checkpoint every N batches", 1, 50, 5))
if not layers or not alphas:
    st.warning("Choose at least one layer and one alpha.")
    st.stop()
cfg = SweepConfig(layers=layers, neurons_per_layer=per, alphas=alphas, rep_threshold=float(rep_thr), ppl_threshold=float(ppl_thr),
                  batch_size=batch, checkpoint_every=ck)
sz = sweep_size(lab, cfg)
store = sweep_store(lab, cfg, RUNS)
status = sweep_status(store)
st.markdown(f"**{sz['items']:,}** pushes at the first alpha (each neuron up and down), up to **{sz['max_generations']:,}** generations "
            f"counting retries. A full sweep of this model would be **{sz['full_sweep_items']:,}** replies. "
            f"Prompt: \"Who are you?\" with the default system prompt; one prompt only, so a neuron that matters for other "
            f"prompts but not this one is missed.")
if st.button("Measure speed for an ETA"):
    with st.spinner("Timing one batch"), gpu_lock():
        ensure_calibration(lab, RUNS)
        ss.eta = estimate_eta(lab, cfg, sz["items"])
if ss.get("eta"):
    e = ss.eta
    st.write(f"About {e['replies_per_s']:.1f} replies/s, so the first pass takes about {e['first_pass_s'] / 60:.0f} min "
             f"(retries add more). Measured throughput is updated while the sweep runs.")

running = get_job("Sweep") is not None and get_job("Sweep").status == "running"
b1, b2, b3 = st.columns([1, 1, 3])
label = "Resume" if status and status["status"] != "finished" else "Start"
if b1.button(label, type="primary", disabled=running or bool(status and status["status"] == "finished")):
    def job(progress, cancel):
        ensure_calibration(lab, RUNS)
        return run_sweep(lab, store, cfg, "sweep", progress, cancel) and True
    start_job("Sweep", job)
    st.rerun()
if b2.button("Pause", disabled=not running):
    get_job("Sweep").cancel_event.set()
job_panel("Sweep", "sweep_done")
if status:
    b3.markdown(f"<span class='muted'>Saved: {status['records']:,} neurons with a reply, status {status['status']}. "
                f"Pausing keeps a checkpoint; Resume continues from it with identical results.</span>", unsafe_allow_html=True)

state = store.load_json("sweep_state")
if not state or not state["records"]:
    st.info("Nothing to explore yet. Start the sweep above; results appear here as soon as the first checkpoint is saved.")
    st.stop()

labels = load_labels(store)
rows = sweep_table(state["records"], labels)
if state["status"] != "finished":
    st.warning(f"The sweep is {state['status']}: {len(rows):,} neurons so far. Everything below reflects the saved checkpoint.")

# --- atlas ----------------------------------------------------------------------------------------------------------------
st.subheader("Atlas")
cnt = Counter(r["category"] for r in rows)
st.markdown("  ".join(f"<span class='tag' style='background:{CATEGORY_COLOR[MODE][c]}33;color:{CATEGORY_COLOR[MODE][c] if c != 'assistant' else '#9a988f'}'>"
                      f"{c} {cnt.get(c, 0):,}</span>" for c in ("persona", "off_topic", "broken", "assistant")), unsafe_allow_html=True)
direction = st.radio("Direction", ["up", "down"], horizontal=True)
ev = st.plotly_chart(atlas_figure(rows, L, lab.adapter.d_mlp(0), 1 if direction == "up" else -1, MODE), on_select="rerun",
                     selection_mode="points", key="atlas", width="stretch")
pts = ev.selection.points if ev and ev.selection else []
if pts and pts[0].get("customdata"):
    ss.sweep_pick = pts[0]["customdata"][0]
st.plotly_chart(persona_share_figure(rows, L, MODE), width="stretch")

# --- table -------------------------------------------------------------------------------------------------------------------
st.subheader("Replies")
f1, f2, f3, f4 = st.columns(4)
cats = f1.multiselect("Category", list(CATEGORIES), ["persona"])
dirs_ = f2.multiselect("Direction ", ["up", "down"], ["up", "down"])
lo, hi = f3.slider("Layers ", 0, L - 1, (0, L - 1))
q = f4.text_input("Search reply or persona name")
df = pd.DataFrame(rows)
m = df["category"].isin(cats) & df["direction"].isin(dirs_) & df["layer"].between(lo, hi)
if q.strip():
    m &= df["reply"].str.contains(q, case=False, regex=False) | df["persona_name"].str.contains(q, case=False, regex=False)
view = df[m][["layer", "neuron", "direction", "alpha", "category", "persona_name", "coherent", "manual", "reply", "key"]].reset_index(drop=True)
view["reply excerpt"] = view["reply"].str.slice(0, 110)
if view.empty:
    st.info("No replies match these filters. Add categories, widen the layer range or clear the search.")
sel = st.dataframe(view.drop(columns=["reply", "key"]), hide_index=True, on_select="rerun", selection_mode="single-row",
                   width="stretch", height=330)
r = sel.selection.rows if sel and sel.selection else []
if r:
    ss.sweep_pick = view.loc[r[0], "key"]
st.download_button("Export the filtered rows (CSV)", view.drop(columns=["key", "reply excerpt"]).to_csv(index=False),
                   file_name="persona_sweep.csv", mime="text/csv", disabled=view.empty)

# --- one neuron ------------------------------------------------------------------------------------------------------------
pick = ss.get("sweep_pick")
rec = state["records"].get(pick) if pick else None
if rec:
    row = next(x for x in rows if x["key"] == pick)
    st.subheader(f"Layer {rec['layer']}, neuron {rec['neuron']}, pushed {'up' if rec['sign'] > 0 else 'down'}")
    st.markdown(f"<span class='muted'>Label: {row['category']}{' (manual)' if row['manual'] else ''}"
                f"{' · ' + row['persona_name'] if row['persona_name'] else ''} · alpha used {row['alpha']} · repeated 4-grams "
                f"{row['rep']:.0%} · reply perplexity {row['ppl'] if row['ppl'] is None else round(row['ppl'], 1)}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='reply'>{html.escape(row['reply'])}</div>", unsafe_allow_html=True)
    with st.expander("Every alpha tried"):
        for a in rec["attempts"]:
            st.markdown(f"alpha {a['alpha']}: {'coherent' if a['coherent'] else 'incoherent'} · repeated {a['rep']:.0%} · ppl "
                        f"{'n/a' if a['ppl'] is None else round(a['ppl'], 1)}<div class='reply'>{html.escape(a['reply'])}</div>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1, 1, 1])
    new_cat = e1.selectbox("Set label", list(CATEGORIES), CATEGORIES.index(row["category"]))
    new_name = e2.text_input("Persona name", row["persona_name"])
    if e3.button("Save label"):
        set_label(store, pick, new_cat, new_name)
        st.rerun()
    if st.button("Use this neuron in the Neuron card"):
        ss.sel_layer, ss.sel_neuron = rec["layer"], rec["neuron"]
        st.switch_page("pages/5_Neuron_card.py")

# --- interesting ------------------------------------------------------------------------------------------------------------
st.subheader("Most interesting personas")
st.caption("Fixed criterion: labelled persona, coherent, a distinct persona name; ranked by the strongest alpha that stayed coherent, "
           "then by fluency. Labels come from a keyword heuristic and have not been checked by a person.")
for i, x in enumerate(interesting_personas(rows, 10), 1):
    st.markdown(f"**{i}. {x['persona_name']}** · layer {x['layer']}, neuron {x['neuron']}, {x['direction']}, alpha {x['alpha']}"
                f"<div class='reply'>{html.escape(x['reply'][:260])}</div>", unsafe_allow_html=True)

# --- optional LLM judge -------------------------------------------------------------------------------------------------------
with st.expander("Optional: label the filtered rows with an LLM judge"):
    if not llm_judge_available():
        st.write("Off. To enable it, set the environment variable NEURONLAB_JUDGE_API_KEY before starting the app. "
                 "The reply text of every labelled row would then be sent to the Anthropic API.")
    else:
        ok = st.checkbox("I understand that the reply text of the rows below is sent to an external API")
        n_lab = st.number_input("Rows to label (first N of the filtered table)", 1, 200, 20)
        if st.button("Label with the LLM judge", disabled=not ok or view.empty):
            done = 0
            for _, v in view.head(int(n_lab)).iterrows():
                try:
                    j = llm_judge(v["reply"], enabled=True)
                    set_label(store, v["key"], j["category"], j.get("persona_name", ""))
                    done += 1
                except Exception as e:  # noqa: BLE001
                    st.error(f"Stopped after {done} rows: {e}")
                    break
            st.success(f"Labelled {done} rows (saved as manual labels).")
            st.rerun()
