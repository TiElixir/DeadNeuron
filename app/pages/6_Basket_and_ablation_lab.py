import json

import pandas as pd
import streamlit as st

from common import MODE, basket_add, get_attribution, job_panel, page, require_experiment, start_job
from neuronlab.evaluation import GROUPS
from neuronlab.interventions import Mode
from neuronlab.pipeline import (basket_key, combined_ranking, evaluate_basket, first_heldout_prompts, ranked_list)
from neuronlab.recording import Cancelled
from neuronlab.viz.curves import ablation_curve_figure
from neuronlab.viz.figures import eval_bars_figure

lab = page("Basket and ablation lab", "Intervene on a set of neurons and compare the effect with the same number of random neurons.")
exp = require_experiment(lab)
ss = st.session_state
r = exp.contrast
attr = get_attribution(exp)

# --- the basket ------------------------------------------------------------------------------------------------
st.subheader("Selected neurons")
basket = list(dict.fromkeys(map(tuple, ss.basket)))
if not basket:
    st.write("The basket is empty. Add neurons from the drill-down or a neuron card, or use the buttons below.")
else:
    df = pd.DataFrame([{"layer": l, "neuron": n, "d": r.d[l, n], **({"attr_sel": attr["attr_sel"][l, n]} if attr else {}),
                        "significant": bool(r.significant[l, n]), "generic": bool(r.generic[l, n])} for l, n in sorted(basket)])
    sel = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="multi-row", width="stretch", height=240)
    rows = sel.selection.rows if sel and sel.selection else []
    cnt = df.groupby("layer").size()
    st.caption("Per layer: " + ", ".join(f"L{l}: {c}" for l, c in cnt.items()))
    b1, b2 = st.columns(2)
    if b1.button(f"Remove {len(rows)} selected", disabled=not rows):
        drop = {(int(df.loc[i, "layer"]), int(df.loc[i, "neuron"])) for i in rows}
        ss.basket = [p for p in basket if tuple(p) not in drop]
        st.rerun()
    if b2.button("Clear basket"):
        ss.basket = []
        st.rerun()

with st.expander("Add neurons in bulk", expanded=not basket):
    a1, a2, a3 = st.columns(3)
    k = int(a1.number_input("Top k", 1, 2000, 50))
    positive_only = a2.toggle("Only d > 0", False, help="Keep neurons that are more active on the positive prompts.")
    rank_by = a3.selectbox("Rank by", ["Combined (|d| and attribution)", "|d| only"] if attr else ["|d| only"])
    use_attr = attr if rank_by.startswith("Combined") else None
    score, _ = combined_ranking(exp, use_attr, positive_only=positive_only)
    ranked = ranked_list(score, 2000)
    if st.button(f"Add top {k}"):
        basket_add([(l, n) for l, n, _ in ranked[:k]])
        st.rerun()
    st.caption(f"{len(ranked)} candidates (significant, non-generic).")
    lo, hi = st.slider("Layer range", 0, r.d.shape[0] - 1, (0, r.d.shape[0] - 1))
    sig_range = [(l, int(n)) for l in range(lo, hi + 1) for n in (r.significant[l] & ~r.generic[l]).nonzero()[0]]
    if st.button(f"Add the {len(sig_range)} significant neurons in layers {lo} to {hi}", disabled=not sig_range):
        basket_add(sig_range)
        st.rerun()

# --- the intervention ---------------------------------------------------------------------------------------------
st.subheader("Intervention")
i1, i2, i3, i4 = st.columns(4)
mode_name = i1.selectbox("Mode", ["mean", "zero", "clamp", "scale", "push"],
                         help="mean: set to the control mean. zero: set to 0. clamp: set to a value. scale: multiply. "
                              "push: add ± alpha × r_L / ‖W_down[:, n]‖ to the activation.")
param = None
if mode_name == "clamp":
    param = i2.number_input("Value", -50.0, 50.0, 0.0)
elif mode_name == "scale":
    param = i2.number_input("Factor", -10.0, 10.0, 0.0)
elif mode_name == "push":
    param = i2.selectbox("Alpha", [0.5, 0.75, 1.0, 1.25], index=2)
    sign = i2.radio("Direction", ["up", "down"], horizontal=True)
mode = {"mean": Mode.mean(), "zero": Mode.zero(), "clamp": lambda: Mode.clamp(param), "scale": lambda: Mode.scale(param),
        "push": lambda: Mode.push(param, 1 if (mode_name == "push" and sign == "up") else -1)}[mode_name]
mode = mode() if callable(mode) else mode
positions = i3.selectbox("Positions", ["all", "prompt_only", "generated_only"])
n_random = int(i4.number_input("Random seeds", 1, 20, 5, help="Each seed draws the same number of neurons per layer from the non-selected ones."))
j1, j2 = st.columns(2)
default_split = 1 if attr is not None else 0
split_choice = j1.radio("Probes used for the evaluation", ["training + held-out", "held-out only"], index=default_split, horizontal=True,
                        help="If attribution helped you choose the neurons, use held-out only: attribution used the training probes.")
splits = ("train", "heldout") if split_choice.startswith("training") else ("heldout",)
n_gen = int(j2.number_input("Generation examples (first held-out prompts, fixed order)", 0, 12, 3))
extra = st.text_input("Extra prompt for the generation diff (optional)", "")
gen_prompts = first_heldout_prompts(exp, n_gen) + ([extra.strip()] if extra.strip() else [])
with_curve = st.checkbox("Also compute the ablation curve (slower)", False,
                         help="Ablates the top-k of the ranking above for k = 1, 2, 5, ... with a random baseline at each k.")
curve_rank = ranked_list(combined_ranking(exp, attr)[0], 2000) if with_curve else None

key = basket_key(basket, mode, positions, n_random, splits, gen_prompts, with_curve, 120)
result = exp.store.load_json(f"eval_{key}") if basket else None
if basket and result is None:
    if st.button("Run evaluation", type="primary"):
        def job(progress, cancel):
            def prog(d, t, eta):
                progress(d, t, eta)
                if cancel():
                    raise Cancelled()
            return evaluate_basket(exp, basket, mode, positions, n_random, splits, gen_prompts, with_curve, curve_rank, 120, prog)
        start_job("Evaluation", job)
        ss.pop("job_result_eval", None)
    job_panel("Evaluation", "job_result_eval")
    if ss.get("job_result_eval") and ss.job_result_eval.get("key") == key:
        result = ss.job_result_eval
elif result is not None:
    st.caption("Loaded from disk: this exact evaluation was computed before.")

if result:
    stt = result["stats"]
    n = result["n"]
    st.subheader("Result")
    z = stt["d_pos"]
    verdict = z["verdict"]
    tag = "tag-yellow" if verdict.startswith("not") else "tag-blue"
    zt = "n/a" if z["z"] is None else f"{z['z']:+.1f}"
    st.markdown(f"<span class='tag {tag}'>{verdict}</span> &nbsp; Δ_pos z-score against random neurons: <b>{zt}</b> "
                f"<span class='muted'>({z['n_seeds']} seeds; a verdict of 'not distinguishable' means the effect is within 2 std of random)</span>",
                unsafe_allow_html=True)
    st.caption(f"n probes: positive {n['pos']}, negative {n['neg']}, control {n['ctl']} ({' + '.join(result['splits'])}); KL over {n['kl_prompts']} held-out "
               f"prompts; {n['control_text']} control paragraphs; batch size {result['batch_size']}; random seeds {result['seeds']}.")
    st.plotly_chart(eval_bars_figure(stt, MODE), width="stretch")
    m = st.columns(4)
    m[0].metric("KL(base || new)", f"{stt['kl']['real']:.4f}", f"random {stt['kl']['rand_mean']:.4f} ± {stt['kl']['rand_std']:.4f}", delta_color="off")
    m[1].metric("Control perplexity", f"{result['ppl']['after']:.2f}", f"{stt['ppl_change']['real']:+.2f} (random {stt['ppl_change']['rand_mean']:+.2f})", delta_color="off")
    ratio = result["ratio"]
    m[2].metric("Δ_pos / Δ_neg", "withheld" if ratio is None else f"{ratio:.2f}",
                help="Both are changes in log-prob; ablating a concept should make Δ_pos strongly negative. A larger ratio means more selective "
                     "for the positive concept. The ratio is unstable (and withheld) when |Δ_neg| < 0.05.")
    m[3].metric("Neurons", len(result["neurons"]))
    st.subheader("Per-probe table")
    st.dataframe(pd.DataFrame(result["per_probe"]), hide_index=True, width="stretch")
    if result.get("curve"):
        st.plotly_chart(ablation_curve_figure(result["curve"], MODE), width="stretch")
    if result.get("generations"):
        st.subheader("Generation diff")
        st.caption("Greedy decoding, same prompts in a fixed order; the first held-out prompts, not hand-picked.")
        import html
        for g in result["generations"]:
            st.markdown(f"**{g['prompt']}**")
            x, y = st.columns(2)
            x.markdown(f"<span class='muted'>baseline · repeated 4-grams {g['rep_base']:.0%} · reply perplexity {g['ppl_base']:.1f}</span>"
                       f"<div class='reply'>{html.escape(g['baseline'])}</div>", unsafe_allow_html=True)
            y.markdown(f"<span class='muted'>intervened · repeated 4-grams {g['rep_new']:.0%} · reply perplexity {g['ppl_new']:.1f}</span>"
                       f"<div class='reply'>{html.escape(g['intervened'])}</div>", unsafe_allow_html=True)
    st.download_button("Export basket and results (JSON)", json.dumps(result, indent=2), file_name=f"neuronlab_eval_{result['key']}.json",
                       mime="application/json")
