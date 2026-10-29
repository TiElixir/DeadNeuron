"""Plotly figures for the UI. Pure functions of arrays and dicts: no Streamlit, no model access."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import base_layout, entity_color, hex_to_rgba, theme

# Diverging pair for signed scores: the two series hues meet at a neutral midpoint, never at a hue.
DIVERGING = {"dark": [[0, "#d95926"], [0.5, "#2c2c2a"], [1, "#3987e5"]],
             "light": [[0, "#eb6834"], [0.5, "#ecebe6"], [1, "#2a78d6"]]}
SEQUENTIAL = {"dark": [[0, "#222220"], [1, "#3987e5"]], "light": [[0, "#f3f2ee"], [1, "#2a78d6"]]}
NEUTRAL = {"dark": "#2a2a28", "light": "#e4e2dc"}


def _axes(fig, mode, **kw):
    t = theme(mode)
    base = dict(gridcolor=t["grid"], zeroline=False, linecolor=t["grid"])
    fig.update_xaxes(**{**base, **kw.get("x", {})})
    fig.update_yaxes(**{**base, **kw.get("y", {})})
    return fig


def grid_shape(n: int) -> tuple[int, int]:
    """Rows x cols with rows * cols == n and the aspect closest to square (4864 -> 64 x 76)."""
    best = (1, n)
    for r in range(1, int(math.isqrt(n)) + 1):
        if n % r == 0:
            best = (r, n // r)
    return best


# --- architecture ---------------------------------------------------------------------------------

def architecture_figure(desc: dict, sig_counts: np.ndarray | None = None, mode: str = "dark") -> go.Figure:
    """Vertical schematic: embedding, N blocks (attention | MLP), final norm, unembedding.

    MLP boxes are shaded by number of significant neurons when sig_counts is given. customdata = layer index
    (-1 embedding, n_layers for the head), so a click can be routed to the drill-down.
    """
    t = theme(mode)
    L = desc["n_layers"]
    fig = go.Figure()
    ys = list(range(L))

    def boxes(x, y, text, color, hover, cd, size=26, name=""):
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=text, textfont=dict(size=10, color=t["text"]),
                                 marker=dict(symbol="square", size=size, color=color,
                                             line=dict(color=t["surface"], width=2)),
                                 hovertemplate="%{customdata[1]}<extra></extra>",
                                 customdata=list(zip(cd, hover)), name=name, showlegend=False))

    boxes([0] * L, ys, [str(h) for h in [desc["n_heads"]] * L], [NEUTRAL[mode]] * L,
          [f"layer {l}: attention, {desc['n_heads']} heads, {desc['layers'][l]['attn_params']:,} parameters" for l in ys],
          ys, name="attention")
    if sig_counts is not None:
        mx = max(int(sig_counts.max()), 1)
        mlp_color = [hex_to_rgba("#3987e5" if mode == "dark" else "#2a78d6", 0.15 + 0.85 * c / mx) for c in sig_counts]
        mlp_text = [str(int(c)) for c in sig_counts]
        extra = [f", {int(c)} significant neurons" for c in sig_counts]
    else:
        mlp_color, mlp_text, extra = [NEUTRAL[mode]] * L, [""] * L, [""] * L
    boxes([1] * L, ys, mlp_text, mlp_color,
          [f"layer {l}: MLP, {desc['d_mlp'][l]:,} neurons, {desc['layers'][l]['mlp_params']:,} parameters{extra[l]}"
           for l in ys], ys, name="mlp")
    for x, y, label, cd in ((0.5, -1.4, f"embedding ({desc['vocab_size']:,} x {desc['d_model']})", -1),
                            (0.5, L + 0.4, f"final norm, then unembedding{' (tied)' if desc['tied_embeddings'] else ''}", L)):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[label], textposition="middle center",
                                 marker=dict(symbol="square", size=1, color=t["surface"]), textfont=dict(color=t["text2"], size=12),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(**base_layout(mode, hovermode="closest", margin=dict(l=40, r=40, t=30, b=20), height=max(640, 30 * L + 140)))
    fig.update_xaxes(visible=False, range=[-0.7, 1.7])
    fig.update_yaxes(visible=False, range=[L + 1, -2.2])
    fig.add_annotation(x=0, y=-0.9, text="attention", showarrow=False, font=dict(color=t["text2"], size=11))
    fig.add_annotation(x=1, y=-0.9, text="MLP", showarrow=False, font=dict(color=t["text2"], size=11))
    return fig


# --- layer overview ---------------------------------------------------------------------------------

def layers_figure(res, mode: str = "dark", hide_generic: bool = True) -> go.Figure:
    """Top: significant neurons per layer. Bottom: max and mean |d| with the permutation threshold (same units)."""
    t = theme(mode)
    L = res.d.shape[0]
    x = list(range(L))
    counts = res.significant_per_layer(hide_generic)
    absd = np.abs(res.d)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.45, 0.55],
                        subplot_titles=("Significant neurons per layer" + (" (generic hidden)" if hide_generic else ""),
                                        "|Cohen's d| per layer against the permutation threshold"))
    fig.add_trace(go.Bar(x=x, y=counts, marker_color=entity_color("pos", mode), customdata=x, name="significant",
                         hovertemplate="layer %{x}: %{y} significant<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=absd.max(1), mode="lines+markers", name="max |d|",
                             line=dict(color=entity_color("pos", mode), width=2), marker=dict(size=8)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=absd.mean(1), mode="lines+markers", name="mean |d|",
                             line=dict(color=entity_color("neg", mode), width=2), marker=dict(size=8)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=res.threshold, mode="lines", name=f"threshold (95th pct of max |d|, {res.n_perm} perms)",
                             line=dict(color=t["text2"], width=1.5, dash="dash")), 2, 1)
    fig.update_layout(**base_layout(mode, height=620, hovermode="x unified", bargap=0.35,
                                    legend=dict(orientation="h", y=-0.12, font=dict(color=t["text2"]))))
    fig.update_xaxes(title_text="layer", row=2, col=1, dtick=2)
    fig.update_yaxes(title_text="neurons", row=1, col=1)
    fig.update_yaxes(title_text="|d|", row=2, col=1)
    for a in fig.layout.annotations:
        a.update(font=dict(color=t["text"], size=13), xanchor="left", x=0)
    return _axes(fig, mode)


def score_histogram(res, mode: str = "dark") -> go.Figure:
    """All observed |d| values against the permutation null (max |d| per layer per shuffle)."""
    t = theme(mode)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.abs(res.d).ravel(), nbinsx=80, histnorm="probability", name="observed |d|, all neurons",
                               marker_color=entity_color("pos", mode), opacity=0.9))
    fig.add_trace(go.Histogram(x=res.null_max.ravel(), nbinsx=80, histnorm="probability",
                               name="null: max |d| per layer per shuffle", marker_color=entity_color("neg", mode), opacity=0.8))
    fig.add_vrect(x0=float(res.threshold.min()), x1=float(res.threshold.max()), fillcolor=t["text2"], opacity=0.12,
                  line_width=0, annotation_text="per-layer thresholds", annotation_font=dict(color=t["text2"], size=11))
    fig.update_layout(**base_layout(mode, height=380, barmode="overlay", hovermode="closest",
                                    title=dict(text="Observed scores against the shuffled-label null", font=dict(color=t["text"], size=15)),
                                    legend=dict(orientation="h", y=-0.25)))
    fig.update_xaxes(title_text="|d|")
    fig.update_yaxes(title_text="share of values (log scale)", type="log")
    return _axes(fig, mode)


def separation_figure(sep: np.ndarray, mode: str = "dark") -> go.Figure:
    t = theme(mode)
    x = list(range(len(sep)))
    fig = go.Figure(go.Scatter(x=x, y=sep, mode="lines+markers", line=dict(color=entity_color("pos", mode), width=2),
                               marker=dict(size=8), hovertemplate="layer %{x}: AUROC %{y:.3f}<extra></extra>", name="held-out AUROC"))
    fig.add_hline(y=0.5, line=dict(color=t["text2"], width=1, dash="dash"),
                  annotation_text="chance", annotation_font=dict(color=t["text2"], size=11))
    b = int(np.argmax(sep))
    fig.add_annotation(x=b, y=sep[b], text=f"best: layer {b}", showarrow=True, arrowcolor=t["text2"], ax=0, ay=-30,
                       font=dict(color=t["text"], size=12))
    fig.update_layout(**base_layout(mode, height=340, hovermode="x unified",
                                    title=dict(text="Does the projection onto the direction separate held-out prompts?",
                                               font=dict(color=t["text"], size=15))))
    fig.update_xaxes(title_text="layer", dtick=2)
    fig.update_yaxes(title_text="held-out AUROC", range=[0, 1.02])
    return _axes(fig, mode)


# --- layer drill-down & neuron card -------------------------------------------------------------------

def layer_grid_figure(values: np.ndarray, title: str, signed: bool, mode: str = "dark", marked: set | None = None) -> go.Figure:
    """One layer's neurons as a rows x cols heatmap. An invisible scatter overlay makes every cell hoverable and
    selectable (customdata = neuron id). Basket members get an outline."""
    t = theme(mode)
    n = len(values)
    r, c = grid_shape(n)
    z = np.asarray(values, float).reshape(r, c)
    ids = np.arange(n).reshape(r, c)
    vmax = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(go.Heatmap(z=z, x=list(range(c)), y=list(range(r)), customdata=ids,
                               colorscale=DIVERGING[mode] if signed else SEQUENTIAL[mode],
                               zmin=-vmax if signed else 0, zmax=vmax, xgap=1, ygap=1,
                               hovertemplate="neuron %{customdata}<br>" + title + " %{z:.3f}<extra></extra>",
                               colorbar=dict(thickness=10, tickfont=dict(color=t["text2"]))))
    yy, xx = np.divmod(np.arange(n), c)
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="markers", customdata=np.arange(n), showlegend=False,
                             marker=dict(size=9, opacity=0.0), hoverinfo="skip"))
    if marked:
        mm = np.array(sorted(marked))
        fig.add_trace(go.Scatter(x=mm % c, y=mm // c, mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(symbol="square-open", size=10, color=t["text"], line=dict(width=1.5))))
    fig.update_layout(**base_layout(mode, height=520, hovermode="closest", margin=dict(l=40, r=30, t=30, b=30), dragmode="select"))
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


def strip_figure(pos: np.ndarray, neg: np.ndarray, ctl: np.ndarray, mode: str = "dark", seed: int = 0) -> go.Figure:
    """Per-prompt activation of one neuron (one dot per prompt: n is prompts) for the three prompt sets."""
    t = theme(mode)
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    for i, (name, ent, vals) in enumerate((("positive", "pos", pos), ("negative", "neg", neg), ("control", "ctl", ctl))):
        fig.add_trace(go.Scatter(x=i + rng.uniform(-0.18, 0.18, len(vals)), y=vals, mode="markers", name=f"{name} (n={len(vals)})",
                                 marker=dict(size=8, color=entity_color(ent, mode), line=dict(color=t["surface"], width=1.5)),
                                 hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[i - 0.3, i + 0.3], y=[float(np.mean(vals))] * 2, mode="lines", showlegend=False,
                                 line=dict(color=t["text"], width=2), hoverinfo="skip"))
    fig.update_layout(**base_layout(mode, height=340, hovermode="closest",
                                    title=dict(text="Activation per prompt (line = mean)", font=dict(color=t["text"], size=15))))
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=["positive", "negative", "control"], range=[-0.6, 2.6])
    fig.update_yaxes(title_text="activation (training split)")
    return _axes(fig, mode, x=dict(gridcolor="rgba(0,0,0,0)"))


def eval_bars_figure(stats: dict, mode: str = "dark") -> go.Figure:
    """Selected neurons (bars) vs random neurons (marker +- 1 std) for the three probe groups. One axis, nats."""
    t = theme(mode)
    groups = [("d_pos", "positive probes", "pos"), ("d_neg", "negative probes", "neg"), ("d_ctl", "control probes", "ctl")]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[g[1] for g in groups], y=[stats[g[0]]["real"] for g in groups], name="selected neurons",
                         marker_color=[entity_color(g[2], mode) for g in groups], width=0.28, offset=-0.3,
                         hovertemplate="selected %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=[g[1] for g in groups], y=[stats[g[0]]["rand_mean"] for g in groups], name="random neurons (mean ± 1 std)",
                         marker=dict(color=[hex_to_rgba(entity_color(g[2], mode), 0.35) for g in groups]), width=0.28, offset=0.02,
                         error_y=dict(type="data", array=[stats[g[0]]["rand_std"] for g in groups], color=t["text2"], thickness=1.5),
                         hovertemplate="random mean %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(mode, height=380, barmode="group", hovermode="closest",
                                    title=dict(text="Change in log P(target) after the intervention", font=dict(color=t["text"], size=15))))
    fig.update_yaxes(title_text="Δ log P(target), nats (0 = no effect)", zeroline=True, zerolinecolor=t["text2"])
    return _axes(fig, mode, x=dict(gridcolor="rgba(0,0,0,0)"))
