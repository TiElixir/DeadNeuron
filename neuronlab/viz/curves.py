"""Plotly figures for ablation curves. Pure functions: no Streamlit, no model access."""
from __future__ import annotations

import plotly.graph_objects as go

from .theme import base_layout, entity_color, hex_to_rgba, theme

LABELS = {"d_pos": ("pos", "Δ positive probes"), "d_neg": ("neg", "Δ negative probes"),
          "d_ctl": ("ctl", "Δ control probes")}


def label_positions(ys: list[float], y_range: float, min_gap_frac: float = 0.07) -> list[float]:
    """Spread end-of-line labels so none overlap (each moved as little as possible, order preserved)."""
    gap = y_range * min_gap_frac
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (sum(ys) - sum(out)) / len(ys)  # re-centre the block on the original mean
    return [o + shift for o in out]


def ablation_curve_figure(curve: dict, mode: str = "light", title: str | None = None) -> go.Figure:
    """Change in log P(target) vs number of ablated neurons, each series against its random-neuron band.

    curve: output of evaluation.ablation_curve. One y axis (nats). Band = random mean +/- 1 std.
    """
    t = theme(mode)
    ks = curve["ks"]
    fig = go.Figure()
    allv = [v for k in LABELS for v in curve["real"][k]] + [a + b for k in LABELS for a, b in zip(curve["rand_mean"][k], curve["rand_std"][k])] \
        + [a - b for k in LABELS for a, b in zip(curve["rand_mean"][k], curve["rand_std"][k])]
    y_range = (max(allv) - min(allv)) or 1.0
    ends = label_positions([curve["real"][k][-1] for k in LABELS], y_range)
    for (key, (ent, label)), y_end in zip(LABELS.items(), ends):
        col = entity_color(ent, mode)
        m, s = curve["rand_mean"][key], curve["rand_std"][key]
        fig.add_trace(go.Scatter(x=ks + ks[::-1], y=[a + b for a, b in zip(m, s)] + [a - b for a, b in zip(m, s)][::-1],
                                 fill="toself", fillcolor=hex_to_rgba(col, 0.16), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, legendgroup=ent))
        fig.add_trace(go.Scatter(x=ks, y=m, mode="lines", line=dict(color=col, width=1.5, dash="dot"),
                                 name=f"{label}, random neurons", legendgroup=ent,
                                 hovertemplate="random mean %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ks, y=curve["real"][key], mode="lines+markers", name=f"{label}, selected neurons",
                                 legendgroup=ent, line=dict(color=col, width=2),
                                 marker=dict(size=8, color=col, line=dict(color=t["surface"], width=2)),
                                 hovertemplate="selected %{y:.3f}<extra></extra>"))
        fig.add_annotation(x=ks[-1], y=y_end, xref="x", yref="y", xanchor="left", showarrow=False,
                           text=label.replace("Δ ", ""), font=dict(color=t["text"], size=12), xshift=10)
    n = curve.get("n", {})
    fig.update_layout(**base_layout(
        mode, title=dict(text=title or "How much does ablating the top-k neurons change log P(target)?",
                         font=dict(color=t["text"], size=16)),
        xaxis=dict(title="neurons ablated (k)", type="log", tickvals=ks, ticktext=[str(k) for k in ks],
                   gridcolor=t["grid"], zeroline=False, linecolor=t["grid"]),
        yaxis=dict(title="change in log P(target), nats  (0 = no effect)", gridcolor=t["grid"],
                   zeroline=True, zerolinecolor=t["text2"], zerolinewidth=1),
        annotations=list(fig.layout.annotations) + [dict(
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False, xanchor="left", font=dict(size=12),
            text=(f"mode {curve.get('mode', '?')}; probes pos={n.get('pos', '?')} neg={n.get('neg', '?')} "
                  f"ctl={n.get('ctl', '?')}; band = random neurons, mean ± 1 std over {curve.get('n_random', '?')} seeds"))]))
    return fig
