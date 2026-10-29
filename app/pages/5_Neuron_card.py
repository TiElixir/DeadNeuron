import html

import streamlit as st

from common import MODE, basket_add, get_attribution, get_directions, gpu_lock, page, require_experiment
from neuronlab.pipeline import first_heldout_prompts, neuron_card, quick_push_test, token_view
from neuronlab.viz.figures import strip_figure
from neuronlab.viz.html import token_html

lab = page("Neuron card", "One neuron: what it responds to, what it writes, and what happens when you push it.")
exp = require_experiment(lab)
ss = st.session_state
r = exp.contrast
L, D = r.d.shape

c1, c2, c3 = st.columns([1, 1, 2])
layer = int(c1.number_input("Layer", 0, L - 1, ss.sel_layer, key="nc_layer"))
neuron = int(c2.number_input("Neuron", 0, D - 1, ss.sel_neuron, key="nc_neuron"))
ss.sel_layer, ss.sel_neuron = layer, neuron
with c3:
    st.write("")
    if st.button("Add to basket", type="primary"):
        basket_add([(layer, neuron)])
    if (layer, neuron) in set(map(tuple, ss.basket)):
        st.caption("In the basket.")

attr = get_attribution(exp)
dirs, sep, best = get_directions(exp, ss.active_run)
with gpu_lock():
    card = neuron_card(exp, layer, neuron, attr, dirs)
sc = card["scores"]

st.subheader("Scores")
m = st.columns(4)
m[0].metric("Cohen's d", f"{sc['d']:+.2f}", help="Difference of means in pooled standard deviations (training split).")
m[1].metric("Welch t", f"{sc['welch_t']:+.1f}")
m[2].metric("Mean pos / neg", f"{sc['mean_pos']:+.3f} / {sc['mean_neg']:+.3f}")
if attr is not None:
    m[3].metric("Attribution selectivity", f"{sc['attr_sel']:+.4f}",
                help="Estimated effect of mean-ablating on positive minus negative probes. Negative = hurts the positive concept more.")
flags = []
flags.append(('<span class="tag tag-green">significant</span>' if sc["significant"] else '<span class="tag tag-yellow">not significant</span>'))
if sc["generic"]:
    flags.append('<span class="tag tag-red">generic</span>')
st.markdown(" ".join(flags) + f"<span class='muted'> &nbsp;|d| must exceed {sc['threshold']:.2f} in this layer "
            f"({r.n_perm} shuffles); n = {r.n_pos} positive, {r.n_neg} negative, {r.n_control} control prompts.</span>",
            unsafe_allow_html=True)

a, b = st.columns([3, 2])
with a:
    st.plotly_chart(strip_figure(card["acts"]["pos"], card["acts"]["neg"], card["acts"]["ctl"], MODE), width="stretch")
with b:
    st.subheader("Logit lens")
    st.caption("Tokens whose logit this neuron's output direction raises (top) or lowers (bottom), through the final norm and unembedding.")
    t1, t2 = st.columns(2)
    t1.dataframe([{"token": repr(t), "score": f"{v:.2f}"} for t, v in card["top"]], hide_index=True, height=360)
    t2.dataframe([{"token": repr(t), "score": f"{v:.2f}"} for t, v in card["bottom"]], hide_index=True, height=360)

st.subheader("Alignment with the concept direction")
al = card.get("alignment", {})
if al:
    src = best if best in al else min(al, key=lambda k: abs(k - best))
    st.write(f"Cosine between this neuron's output direction and the direction estimated at layer **{src}** "
             f"(the layer with the best held-out separation): **{al[src]:+.3f}**.")
    with st.expander("All layers at or after this one"):
        st.dataframe([{"direction layer": k, "cosine": round(v, 4)} for k, v in al.items()], hide_index=True)
else:
    st.write("This neuron writes after the last layer that has a direction, so there is no alignment to show.")

st.subheader("Where it fires")
n_tok = st.slider("Prompts per set", 3, 10, 10)
with gpu_lock():
    tv = token_view(exp, layer, neuron, n_tok)
st.caption("Shaded tokens are the ones counted in the statistics (blue = positive activation, orange = negative); "
           "grey tokens are template or first-token positions and are not counted.")
for name, rows in tv.items():
    with st.expander(f"{name} prompts ({len(rows)})", expanded=(name == "positive")):
        for row in rows:
            st.markdown(token_html(row["tokens"], row["acts"], row["counted"], MODE), unsafe_allow_html=True)

st.subheader("Quick push test")
st.caption("Adds ± alpha × r_L / ‖W_down[:, n]‖ to this neuron's activation, so the residual stream moves by alpha × the typical "
           "residual norm along its output direction. All conditions run in one batch (batch composition can shift bf16 "
           "numerics slightly).")
fixed = first_heldout_prompts(exp, 2)
p1 = st.text_input("Prompt 1", "Who are you?")
p2 = st.text_input("Prompt 2", fixed[0] if fixed else "")
p3 = st.text_input("Prompt 3", fixed[-1] if fixed else "")
prompts = [p for p in (p1, p2, p3) if p.strip()]
if st.button("Run push test", disabled=not prompts):
    with st.spinner("Generating"), gpu_lock():
        ss.push = (layer, neuron, quick_push_test(exp, layer, neuron, prompts))
if ss.get("push") and ss.push[:2] == (layer, neuron):
    out = ss.push[2]
    for p in dict.fromkeys(o["prompt"] for o in out):
        st.markdown(f"**{p}**")
        rows = [o for o in out if o["prompt"] == p]
        cols = st.columns(3)
        for i, o in enumerate(rows):
            with cols[i % 3]:
                st.markdown(f"<span class='muted'>{o['condition']}</span><div class='reply'>{html.escape(o['reply'][:600])}</div>",
                            unsafe_allow_html=True)
