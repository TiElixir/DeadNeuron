import numpy as np
import pandas as pd
import streamlit as st

from common import MODE, get_attribution, gpu_lock, page, require_experiment
from neuronlab.scoring import logit_lens
from neuronlab.viz.figures import layer_grid_figure

lab = page("Layer drill-down", "Every neuron of one layer as a grid. Select a cell, or work from the table.")
exp = require_experiment(lab)
ss = st.session_state
r, L = exp.contrast, exp.contrast.d.shape[0]

layer = int(st.number_input("Layer", 0, L - 1, ss.sel_layer, key="dd_layer"))
ss.sel_layer = layer
attr = get_attribution(exp)

scores = {"Cohen's d": r.d[layer], "Welch t": r.t[layer], "Mean difference (pos - neg)": r.diff[layer],
          "Mean on positive prompts": r.mean_pos[layer], "Mean on negative prompts": r.mean_neg[layer]}
if attr is not None:
    scores["Attribution selectivity (pos - neg)"] = attr["attr_sel"][layer]
c1, c2, c3 = st.columns([2, 1, 1])
score_name = c1.selectbox("Colour by", list(scores))
only_sig = c2.toggle("Significant only", False)
hide_generic = c3.toggle("Hide generic", True)
if attr is None:
    st.caption("Attribution is not computed for this run. It estimates how much mean-ablating each neuron changes the "
               "probe log-probs and takes a few seconds.")
    if st.button("Compute attribution"):
        with st.spinner("Attributing"):
            get_attribution(exp, compute=True)
        st.rerun()

vals = scores[score_name].astype(float).copy()
shown = np.ones_like(vals, bool)
if only_sig:
    shown &= r.significant[layer]
if hide_generic:
    shown &= ~r.generic[layer]
if only_sig or hide_generic:
    vals[~shown] = np.nan

left, right = st.columns([5, 6])
with left:
    in_basket = {n for l, n in ss.basket if l == layer}
    ev = st.plotly_chart(layer_grid_figure(vals, score_name, True, MODE, in_basket), on_select="rerun",
                         selection_mode="points", key=f"grid_{layer}", width="stretch")
    pts = ev.selection.points if ev and ev.selection else []
    for p in pts:
        if p.get("customdata") is not None:
            ss.sel_neuron = int(p["customdata"] if not isinstance(p["customdata"], (list, tuple)) else p["customdata"][0])
            break
    st.caption(f"{r.d.shape[1]:,} neurons as a grid; outlined cells are in the basket. "
               f"{int(r.significant[layer].sum())} significant, {int((r.significant[layer] & ~r.generic[layer]).sum())} after hiding generic.")
    nid = int(st.number_input("Neuron", 0, r.d.shape[1] - 1, ss.sel_neuron, key="dd_neuron"))
    ss.sel_neuron = nid
    b1, b2 = st.columns(2)
    if b1.button("Open neuron card", type="primary"):
        st.switch_page("pages/5_Neuron_card.py")
    if b2.button("Add to basket"):
        from common import basket_add
        basket_add([(layer, nid)])
        st.rerun()

with right:
    df = pd.DataFrame({"neuron": np.arange(r.d.shape[1]), "d": r.d[layer], "welch_t": r.t[layer],
                       "mean_pos": r.mean_pos[layer], "mean_neg": r.mean_neg[layer],
                       "significant": r.significant[layer], "generic": r.generic[layer]})
    if attr is not None:
        df["attr_sel"] = attr["attr_sel"][layer]
    f1, f2, f3 = st.columns(3)
    min_d = f1.number_input("Minimum |d|", 0.0, 10.0, 0.0, 0.1)
    sign = f2.selectbox("Direction", ["either", "positive-leaning (d > 0)", "negative-leaning (d < 0)"])
    top_n = int(f3.number_input("Rows", 10, 1000, 100, 10))
    m = shown & (np.abs(r.d[layer]) >= min_d)
    if sign.startswith("positive"):
        m &= r.d[layer] > 0
    elif sign.startswith("negative"):
        m &= r.d[layer] < 0
    view = df[m].reindex(df[m]["d"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
    if len(view):
        with gpu_lock():
            lens = logit_lens(lab, layer, view["neuron"].tolist(), 5)
        view["promotes (logit lens)"] = [" ".join(lab.tokenizer.decode([int(i)]).strip() or "·" for i in row) for row in lens["top_ids"]]
    st.caption(f"{int(m.sum())} neurons match; showing the {len(view)} with the largest |d|. Logit lens = tokens this neuron's "
               f"output direction promotes most.")
    sel = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="multi-row", width="stretch", height=430,
                       column_config={c: st.column_config.NumberColumn(format="%.3f") for c in view.columns
                                      if view[c].dtype.kind == "f"})
    rows = sel.selection.rows if sel and sel.selection else []
    if st.button(f"Add {len(rows)} selected to basket", disabled=not rows, type="primary"):
        from common import basket_add
        basket_add([(layer, int(view.loc[i, "neuron"])) for i in rows])
        st.rerun()
    if len(view) == 0:
        st.info("No neurons match these filters. Lower the minimum |d|, turn off 'Significant only', or show generic neurons.")
