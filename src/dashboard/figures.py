"""Plotly figures for the coverage dashboard.

Color: zones are ordered (DARK < THIN < ADEQUATE), so they're an ordinal
encoding, one blue ramp stepped by lightness, with the most urgent zone the
most prominent against the surface. I didn't use a red/amber/green status set,
because red vs green collapses under deuteranopia and the spec's RdYlGn is a
rainbow ramp. Both ramps below pass the ordinal checks (monotone lightness,
visible steps, light end >= 2:1 on the surface), and each zone also gets its
own marker shape so identity never depends on color alone. Dark mode is its
own set of steps with the anchor flipped: DARK zones are the lightest there,
because that's what stands out on a dark surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

ZONES = ("DARK", "THIN", "ADEQUATE")
NOISE_ID = -1

THEMES: Dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "zone": {"DARK": "#104281", "THIN": "#3987e5", "ADEQUATE": "#86b6ef"},
        "noise": "#c3c2b7",
        # coverage-score view: darker = less covered
        "sequential": [[0.0, "#0d366b"], [0.25, "#1c5cab"], [0.5, "#3987e5"], [0.75, "#86b6ef"], [1.0, "#cde2fb"]],
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "zone": {"DARK": "#b7d3f6", "THIN": "#3987e5", "ADEQUATE": "#184f95"},
        "noise": "#383835",
        "sequential": [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#184f95"]],
    },
}
SYMBOLS = {"DARK": "diamond", "THIN": "triangle-up", "ADEQUATE": "circle"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, theme: str, **kwargs) -> go.Figure:
    t = THEMES[theme]
    settings = dict(
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["secondary"], size=12),
        margin=dict(l=16, r=16, t=16, b=16),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=FONT, color=t["ink"])),
        legend=dict(font=dict(color=t["secondary"]), bgcolor="rgba(0,0,0,0)"),
    )
    settings.update(kwargs)
    fig.update_layout(**settings)
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["grid"], tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["grid"], tickfont=dict(color=t["muted"]))
    return fig


def coverage_map(points: pd.DataFrame, clusters: Sequence[dict], theme: str = "light", color_by: str = "zone",
                 selected: Optional[int] = None) -> go.Figure:
    """points: one row per probe with x, y, coverage_score, query, cluster_id, cluster_name, zone, strategy."""
    t = THEMES[theme]
    fig = go.Figure()
    hover = ("%{customdata[0]}<br><b>%{customdata[1]}</b> · %{customdata[2]}<br>"
             "coverage %{customdata[3]:.2f} · %{customdata[4]}<extra></extra>")
    cols = ["query", "cluster_name", "zone", "coverage_score", "strategy", "cluster_id"]
    ring = dict(width=1.5, color=t["surface"])

    noise = points[points.cluster_id == NOISE_ID]
    if len(noise):
        fig.add_trace(go.Scatter(
            x=noise.x, y=noise.y, mode="markers", name="noise (unclustered)", customdata=noise[cols].values,
            hovertemplate=hover, marker=dict(size=7, color=t["noise"], line=ring),
        ))
    clustered = points[points.cluster_id != NOISE_ID]
    if color_by == "score":
        fig.add_trace(go.Scatter(
            x=clustered.x, y=clustered.y, mode="markers", name="probes", customdata=clustered[cols].values,
            hovertemplate=hover, showlegend=False,
            marker=dict(size=9, color=clustered.coverage_score, colorscale=t["sequential"], cmin=0.1, cmax=0.8,
                        symbol=[SYMBOLS[z] for z in clustered.zone], line=ring,
                        colorbar=dict(title=dict(text="coverage score", font=dict(color=t["secondary"])),
                                      tickfont=dict(color=t["muted"]), outlinewidth=0, thickness=10)),
        ))
    else:
        for zone in ZONES:  # legend order: most urgent first
            part = clustered[clustered.zone == zone]
            if len(part):
                fig.add_trace(go.Scatter(
                    x=part.x, y=part.y, mode="markers", name=zone.lower(), customdata=part[cols].values,
                    hovertemplate=hover,
                    marker=dict(size=9, color=t["zone"][zone], symbol=SYMBOLS[zone], line=ring),
                ))

    # label the dark and thin clusters (spec §9), plus whichever one is selected
    for c in clusters:
        if c["cluster_id"] == NOISE_ID or (c["zone"] == "ADEQUATE" and c["cluster_id"] != selected):
            continue
        # offset with a leader line: UMAP packs a missing topic's probes into a
        # tight clump, and a label sitting on the centroid hides the whole cluster
        fig.add_annotation(
            x=c["centroid_x"], y=c["centroid_y"], text=c["name"], showarrow=True, arrowhead=0, arrowwidth=1,
            arrowcolor=t["muted"], ax=0, ay=-36, standoff=8,
            font=dict(size=11, color=t["ink"]), bgcolor=t["surface"], opacity=0.92,
            borderpad=3, bordercolor=t["axis"] if c["cluster_id"] == selected else t["surface"], borderwidth=1,
        )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, title=None)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, title=None)
    return _layout(fig, theme, legend_orientation="h", legend_y=1.02, legend_x=0, clickmode="event", dragmode="pan",
                   height=560)


def severity_bars(clusters: Sequence[dict], theme: str = "light", top: int = 15) -> go.Figure:
    """Most severe clusters of every tier, so depth gaps in ADEQUATE clusters stay visible."""
    t = THEMES[theme]
    rows = [c for c in clusters if c["cluster_id"] != NOISE_ID][:top][::-1]  # already severity-sorted
    fig = go.Figure()
    for zone in ZONES:
        part = [c for c in rows if c["zone"] == zone]
        fig.add_trace(go.Bar(
            y=[c["name"] for c in part], x=[c["severity"] for c in part], orientation="h", name=zone.lower(),
            marker=dict(color=t["zone"][zone], cornerradius=4, line=dict(width=0)),
            customdata=[[c["cluster_id"], c["query_count"], c["mean_cs"], c["unanswered_share"]] for c in part],
            text=[f"{c['severity']:.2f}" for c in part], textposition="outside",
            textfont=dict(color=t["secondary"], size=11), cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>severity %{x:.2f} · %{customdata[1]} probes<br>"
                          "mean coverage %{customdata[2]:.2f} · %{customdata[3]:.0%} unanswered<extra></extra>",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=[c["name"] for c in rows], showgrid=False,
                     tickfont=dict(color=t["secondary"]))
    fig.update_xaxes(title=dict(text="severity", font=dict(color=t["muted"])), showgrid=True, zeroline=False)
    return _layout(fig, theme, barmode="overlay", bargap=0.35, height=26 * len(rows) + 90,
                   legend_orientation="h", legend_y=1.08, legend_x=0, margin=dict(l=16, r=48, t=24, b=16))


def strategy_breakdown(breakdown: Dict[str, Dict[str, int]], theme: str = "light") -> go.Figure:
    """Share of each strategy's probes that landed in each zone (100% stacked bars)."""
    t = THEMES[theme]
    # horizontal bars draw bottom-up, so sort ascending to put the most-dark strategy on top
    strategies: List[str] = sorted(breakdown, key=lambda s: breakdown[s].get("DARK", 0) / max(1, sum(breakdown[s].values())))
    fig = go.Figure()
    for zone in ZONES:
        shares = [breakdown[s].get(zone, 0) / max(1, sum(breakdown[s].values())) for s in strategies]
        counts = [breakdown[s].get(zone, 0) for s in strategies]
        fig.add_trace(go.Bar(
            y=strategies, x=shares, orientation="h", name=zone.lower(), customdata=counts,
            marker=dict(color=t["zone"][zone], line=dict(width=2, color=t["surface"])),  # 2px surface gap
            hovertemplate="%{y}: %{x:.0%} in " + zone.lower() + " clusters (%{customdata} probes)<extra></extra>",
        ))
    fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=t["secondary"]))
    return _layout(fig, theme, barmode="stack", bargap=0.45, height=44 * len(strategies) + 90,
                   legend_orientation="h", legend_y=1.15, legend_x=0, legend_traceorder="normal")
