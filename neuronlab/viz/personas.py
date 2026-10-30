"""Plotly figures for the persona sweep. Pure functions of the sweep table (list of row dicts)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import base_layout, theme

# Fixed identity per category. Only three hues plus a recessive grey for the default ("assistant"): the first three
# categorical slots validate for scatter (all-pairs) in both modes; a fourth hue would not.
CATEGORY_COLOR = {
    "dark": {"assistant": "#4a4a46", "persona": "#3987e5", "broken": "#d95926", "off_topic": "#199e70"},
    "light": {"assistant": "#c9c6bb", "persona": "#2a78d6", "broken": "#eb6834", "off_topic": "#1baf7a"},
}
ORDER = ("persona", "off_topic", "broken", "assistant")  # draw the rare ones on top of the common one


def atlas_figure(rows: list[dict], n_layers: int, d_mlp: int, sign: int, mode: str = "dark", cols: int = 6) -> go.Figure:
    """One small panel per layer; each sampled neuron is a dot at its position in the layer's neuron grid."""
    t = theme(mode)
    nrows = -(-n_layers // cols)
    width = 64
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[f"layer {l}" for l in range(n_layers)],
                        horizontal_spacing=0.012, vertical_spacing=0.05)
    rows = [r for r in rows if r["sign"] == sign]
    for cat in ORDER:
        for layer in range(n_layers):
            sel = [r for r in rows if r["layer"] == layer and r["category"] == cat]
            if not sel:
                continue
            n = np.array([r["neuron"] for r in sel])
            fig.add_trace(go.Scattergl(
                x=n % width, y=n // width, mode="markers", name=cat, legendgroup=cat, showlegend=(layer == 0 or False),
                marker=dict(size=6, color=CATEGORY_COLOR[mode][cat]),
                customdata=[[r["key"], r["persona_name"], r["alpha"]] for r in sel],
                hovertemplate="layer " + str(layer) + " neuron %{text}<br>" + cat + " %{customdata[1]}<br>alpha %{customdata[2]}<extra></extra>",
                text=[r["neuron"] for r in sel]), row=layer // cols + 1, col=layer % cols + 1)
    fig.update_layout(**base_layout(mode, height=150 * nrows + 90, hovermode="closest", margin=dict(l=20, r=20, t=50, b=60),
                                    legend=dict(orientation="h", y=-0.04, itemsizing="constant", font=dict(color=t["text2"]))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    for a in fig.layout.annotations:
        a.update(font=dict(color=t["text2"], size=11))
    return fig


def persona_share_figure(rows: list[dict], n_layers: int, mode: str = "dark") -> go.Figure:
    t = theme(mode)
    share, counts = [], []
    for l in range(n_layers):
        r = [x for x in rows if x["layer"] == l]
        counts.append(len(r))
        share.append(sum(x["category"] == "persona" for x in r) / len(r) if r else 0.0)
    fig = go.Figure(go.Bar(x=list(range(n_layers)), y=share, marker_color=CATEGORY_COLOR[mode]["persona"], customdata=counts,
                           hovertemplate="layer %{x}: %{y:.1%} of %{customdata} replies labelled persona<extra></extra>"))
    fig.update_layout(**base_layout(mode, height=320, hovermode="x unified", bargap=0.35,
                                    title=dict(text="Share of pushed neurons whose reply is labelled persona", font=dict(color=t["text"], size=15))))
    fig.update_xaxes(title_text="layer", dtick=2, gridcolor=t["grid"])
    fig.update_yaxes(title_text="share", tickformat=".0%", gridcolor=t["grid"], rangemode="tozero")
    return fig
