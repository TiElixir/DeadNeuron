import html

import numpy as np
import pandas as pd
import streamlit as st

from common import basket_add, get_directions, gpu_lock, page, require_experiment
from neuronlab.directions import direction_ablation_spec, neurons_by_alignment, steering_spec
from neuronlab.evaluation import generate_replies, repetition_rate

lab = page("Directions and steering", "The residual-stream alternative to single neurons: one direction per layer.")
exp = require_experiment(lab)
ss = st.session_state
dirs, sep, best = get_directions(exp, ss.active_run)
L = len(sep)

st.subheader("Direction per layer")
st.caption("mean(positive) − mean(negative) residual from the training split. AUROC is the held-out separation of the projection onto it.")
st.dataframe(pd.DataFrame({"layer": range(L), "norm": dirs.norm, "held-out AUROC": sep}), hide_index=True, height=260, width="stretch",
             column_config={"norm": st.column_config.NumberColumn(format="%.2f"), "held-out AUROC": st.column_config.ProgressColumn(
                 min_value=0.0, max_value=1.0, format="%.3f")})
st.write(f"Best held-out separation: layer **{best}** (AUROC {sep[best]:.3f}).")

st.subheader("Steer or erase")
c1, c2, c3 = st.columns(3)
method = c1.radio("Method", ["Steer", "Directional ablation"], help="Steer adds alpha × r_L × u. Directional ablation removes the component along u: h − (h·u)u.")
src = int(c2.number_input("Direction from layer", 0, L - 1, best))
if method == "Steer":
    alpha = c3.slider("Alpha (fraction of the typical residual norm)", -1.0, 1.0, 0.1, 0.05,
                      help="Positive moves toward the positive set. In the land-vs-sea check, 0.05 to 0.1 steered and 0.25 or more broke the model.")
    layers = st.multiselect("Apply at layers", list(range(L)), [src])
else:
    layers = st.multiselect("Apply at layers", list(range(L)), list(range(L)))
positions = st.selectbox("Positions", ["all", "prompt_only", "generated_only"])
default_prompt = (exp.heldout.positive or ["Who are you?"])[0]
prompt = st.text_input("Prompt", default_prompt)
max_new = int(st.number_input("Max new tokens", 16, 200, 100))
if st.button("Generate", type="primary", disabled=not layers or not prompt.strip()):
    spec = (steering_spec(lab, dirs, layers, alpha, positions, source_layer=src) if method == "Steer"
            else direction_ablation_spec(dirs, layers, src, positions))
    with st.spinner("Generating"), gpu_lock():
        base = generate_replies(lab, [prompt], None, exp.preset.system_prompt, max_new)[0]
        new = generate_replies(lab, [prompt], spec, exp.preset.system_prompt, max_new)[0]
    x, y = st.columns(2)
    x.markdown(f"<span class='muted'>baseline · repeated 4-grams {repetition_rate(base[1]):.0%}</span><div class='reply'>{html.escape(base[0])}</div>",
               unsafe_allow_html=True)
    y.markdown(f"<span class='muted'>{method.lower()} · repeated 4-grams {repetition_rate(new[1]):.0%}</span><div class='reply'>{html.escape(new[0])}</div>",
               unsafe_allow_html=True)

st.subheader("Neurons aligned with the direction")
st.caption("Neurons in layers up to the chosen one, ranked by the cosine between their output direction and the direction. "
           "These are candidates for single-neuron pushes.")
k = int(st.number_input("How many", 5, 200, 30))
sign = st.selectbox("Sign", ["pos", "neg", "abs"], format_func={"pos": "aligned with the positive set", "neg": "aligned against it", "abs": "either"}.get)
with gpu_lock():
    al = neurons_by_alignment(lab, dirs, src, k, sign)
adf = pd.DataFrame([{"layer": l, "neuron": n, "cosine": c, "d": exp.contrast.d[l, n], "significant": bool(exp.contrast.significant[l, n])}
                    for l, n, c in al])
sel = st.dataframe(adf, hide_index=True, on_select="rerun", selection_mode="multi-row", width="stretch", height=320,
                   column_config={"cosine": st.column_config.NumberColumn(format="%.3f"), "d": st.column_config.NumberColumn(format="%.2f")})
rows = sel.selection.rows if sel and sel.selection else []
if st.button(f"Send {len(rows)} selected to the basket", disabled=not rows):
    basket_add([(int(adf.loc[i, "layer"]), int(adf.loc[i, "neuron"])) for i in rows])
    st.success("Added. Open the Basket and ablation lab to test them.")
