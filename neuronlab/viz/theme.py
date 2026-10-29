"""Chart colours. Categorical slots 1-3 of the reference palette, validated all-pairs in both modes
(scripts/validate_palette.js): CVD dE >= 9.2 / 9.4, normal-vision dE >= 20.9. Aqua is below 3:1 on the
light surface, so every chart using it carries direct labels."""

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="rgba(11,11,11,0.08)",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="rgba(240,239,236,0.10)",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
# fixed identity: the colour follows the entity, never its rank
ENTITY = {"pos": 0, "neg": 1, "ctl": 2}


def theme(mode: str = "light") -> dict:
    return THEMES[mode]


def entity_color(name: str, mode: str = "light") -> str:
    return THEMES[mode]["series"][ENTITY[name]]


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def base_layout(mode: str = "light", **kw) -> dict:
    t = THEMES[mode]
    layout = dict(paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
                  font=dict(color=t["text2"], size=13, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
                  title=dict(font=dict(color=t["text"], size=16), x=0.0, xanchor="left"),
                  margin=dict(l=60, r=110, t=70, b=55), hovermode="x unified",
                  legend=dict(orientation="h", y=-0.2, font=dict(color=t["text2"])))
    layout.update(kw)
    return layout
