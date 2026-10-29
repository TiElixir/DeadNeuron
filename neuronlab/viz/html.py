"""HTML snippets (no Streamlit): token highlighting."""
from __future__ import annotations

import html

import numpy as np


def token_html(tokens: list[str], acts: np.ndarray, counted: np.ndarray, mode: str = "dark") -> str:
    """Tokens as spans shaded by activation. Blue = positive, orange = negative (same pair as the heatmap).
    Tokens outside the statistics mask (template, system prompt, first token) are dimmed and not scaled."""
    pos, neg = ("57,135,229", "217,89,38") if mode == "dark" else ("42,120,214", "235,104,52")
    fg, dim = ("#ecebe6", "#8a8880") if mode == "dark" else ("#111111", "#9a988f")
    vmax = float(np.abs(acts[counted]).max()) if counted.any() else float(np.abs(acts).max()) or 1.0
    vmax = vmax or 1.0
    out = ['<div style="line-height:2.1;font-family:ui-monospace,\'SF Mono\',Menlo,monospace;font-size:13px">']
    for tok, a, c in zip(tokens, acts, counted):
        text = html.escape(tok.replace("Ġ", " ").replace("Ċ", "↵")) or "&nbsp;"
        if not c:
            out.append(f'<span style="color:{dim};padding:1px 0" title="{a:+.4f} (not counted)">{text}</span>')
            continue
        alpha = min(abs(a) / vmax, 1.0) * 0.85
        rgb = pos if a >= 0 else neg
        out.append(f'<span style="background:rgba({rgb},{alpha:.2f});color:{fg};border-radius:2px;padding:1px 1px" '
                   f'title="{a:+.4f}">{text}</span>')
    out.append("</div>")
    return "".join(out)
