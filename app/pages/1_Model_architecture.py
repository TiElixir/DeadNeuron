import pandas as pd
import streamlit as st

from common import MODE, page
from neuronlab.viz.figures import architecture_figure

lab = page("Model architecture", "Each row is a decoder block. Select a block to open that layer in the drill-down.")
if lab is None:
    st.info("Load a model in the sidebar first.")
    st.stop()

desc = lab.adapter.describe()
sig = None
if st.session_state.active_run:
    from common import get_experiment
    try:
        exp = get_experiment(lab, st.session_state.active_run)
        sig = exp.contrast.significant_per_layer(True)
        st.caption("MLP boxes are shaded by the number of significant, non-generic neurons in the active run.")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read the active run, showing the plain schematic: {e}")
else:
    st.caption("Pick a run in the sidebar to shade the MLP boxes by significant neurons.")

left, right = st.columns([3, 2])
with left:
    ev = st.plotly_chart(architecture_figure(desc, sig, MODE), on_select="rerun", selection_mode="points",
                         key="arch", width="stretch")
    pts = ev.selection.points if ev and ev.selection else []
    if pts:
        cd = pts[0].get("customdata")
        if cd is not None and 0 <= int(cd[0]) < desc["n_layers"]:
            st.session_state.sel_layer = int(cd[0])
with right:
    st.subheader("Open a layer")
    layer = st.number_input("Layer", 0, desc["n_layers"] - 1, st.session_state.sel_layer, key="arch_layer")
    st.session_state.sel_layer = int(layer)
    L = desc["layers"][int(layer)]
    st.markdown(f"Layer **{int(layer)}**: attention with {L['attn_params']:,} parameters, MLP with {desc['d_mlp'][int(layer)]:,} neurons "
                f"and {L['mlp_params']:,} parameters.")
    if sig is not None:
        st.markdown(f"Significant neurons here: **{int(sig[int(layer)])}**")
    if st.button("Open in Layer drill-down", type="primary"):
        st.switch_page("pages/4_Layer_drill-down.py")

st.subheader("Structure")
summary = {k: v for k, v in desc.items() if k not in ("layers", "d_mlp")}
st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "value"}), width="stretch")
st.dataframe(pd.DataFrame(desc["layers"]), hide_index=True, width="stretch")
