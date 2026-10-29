import streamlit as st

from common import MODE, get_directions, page, require_experiment
from neuronlab.viz.figures import layers_figure, score_histogram, separation_figure

lab = page("Layer overview", "Where in the network do the two prompt sets separate more than shuffled labels would?")
exp = require_experiment(lab)
r = exp.contrast

hide = st.toggle("Hide generic neurons", True, help="Generic = also active on the control prompts (control mean above the "
                                                     "positive mean, or active on more than 90% of control prompts).")
st.caption(f"Training split only: n_pos={r.n_pos}, n_neg={r.n_neg}, n_control={r.n_control} prompts. "
           f"Threshold: 95th percentile of the per-layer maximum |d| over {r.n_perm} label shuffles (seed {r.seed}).")
ev = st.plotly_chart(layers_figure(r, MODE, hide), on_select="rerun", selection_mode="points", key="layers", width="stretch")
pts = ev.selection.points if ev and ev.selection else []
if pts and pts[0].get("x") is not None:
    st.session_state.sel_layer = int(pts[0]["x"])

c1, c2 = st.columns([3, 1])
with c2:
    layer = st.number_input("Layer", 0, r.d.shape[0] - 1, st.session_state.sel_layer, key="ov_layer")
    st.session_state.sel_layer = int(layer)
    n_sig = int(r.significant_per_layer(hide)[int(layer)])
    st.write(f"{n_sig} significant neurons in layer {int(layer)}.")
    if st.button("Open in Layer drill-down", type="primary"):
        st.switch_page("pages/4_Layer_drill-down.py")
with c1:
    st.metric("Significant neurons overall", f"{int(r.significant_per_layer(hide).sum()):,}",
              help="Out of all MLP neurons, after the per-layer permutation threshold.")

st.plotly_chart(score_histogram(r, MODE), width="stretch")

dirs, sep, best = get_directions(exp, st.session_state.active_run)
st.plotly_chart(separation_figure(sep, MODE), width="stretch")
st.caption(f"Direction = mean(positive) minus mean(negative) residual, from the training split; AUROC is measured on "
           f"{len(exp.heldout.positive)} + {len(exp.heldout.negative)} held-out prompts.")
