"""Coverage dashboard: the map, the severity ranking and the recommendations for one GapReport.

    umbra dashboard --report out/report.json [--audit out/audit.csv] [--port 8050]

Click a point on the map, a bar, or a table row to open that cluster.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html

from src.dashboard.figures import THEMES, coverage_map, severity_bars, strategy_breakdown
from src.reports.schema import GapReport

ASSETS = Path(__file__).parent / "assets"


def load_points(audit_csv: Path, report: GapReport) -> pd.DataFrame:
    by_id = {c.cluster_id: c for c in report.clusters}
    rows = []
    for r in csv.DictReader(open(audit_csv, newline="", encoding="utf-8")):
        if not r["coverage_score"] or r.get("cluster_id") in ("", None) or not r.get("umap_x"):
            continue
        c = by_id.get(int(r["cluster_id"]))
        if c is None:
            continue
        rows.append({
            "x": float(r["umap_x"]), "y": float(r["umap_y"]), "coverage_score": float(r["coverage_score"]),
            "query": r["query"], "cluster_id": c.cluster_id, "cluster_name": c.name,
            "zone": c.zone,
            "strategy": "kb_blind" if r["generation_strategy"] == "ground_truth" else r["generation_strategy"],
        })
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def table_rows(report: GapReport):
    return [{
        "rank": c.severity_rank, "id": c.cluster_id, "name": c.name, "zone": c.zone, "severity": round(c.severity, 2),
        "mean_cs": round(c.mean_cs, 3), "size": c.query_count, "unanswered": round(100 * c.unanswered_share),
        "recs": len(c.recommendations),
    } for c in report.clusters]


def detail_panel(report: GapReport, cluster_id: Optional[int]):
    if cluster_id is None:
        return html.Div([html.H2("Cluster detail"),
                         html.P("Click a point, a bar or a table row to see its questions and recommendations.",
                                className="subtle")])
    c = next(x for x in report.clusters if x.cluster_id == cluster_id)
    recs = [
        html.Div([
            html.Div([html.Span(f"{r.rank}. "),
                      html.A(r.title, href=r.url, target="_blank") if r.url else html.Span(r.title)], className="title"),
            html.Div(r.snippet[:280], className="snippet"),
            html.Div(f"{r.action} Simulated gain {r.expected_gain:+.3f}.", className="muted"),
        ], className="rec")
        for r in c.recommendations[:3]
    ]
    if not recs:
        if c.recommendation_reason is None:
            why = "No recommendations: this cluster isn't a dark or thin zone and isn't in the top severity list."
        elif report.config.get("web_search") == "none":
            why = ("Nothing in the KB answers these questions. Web search was off for this report, so there are "
                   "no external candidates.")
        else:
            why = "No candidate passed the filters: nothing in the KB answers these questions, and no web result "\
                  "raised the simulated score."
        recs = [html.P(why, className="subtle")]
    return html.Div([
        html.H2(c.name),
        html.Div([html.Span(className=f"zone-key {c.zone}"), html.Span(c.zone.lower()),
                  html.Span(f"  ·  severity rank {c.severity_rank}", className="muted")]),
        html.Div([
            html.Div("mean coverage", className="k"), html.Div("severity", className="k"),
            html.Div("probes", className="k"), html.Div("unanswered", className="k"),
            html.Div(f"{c.mean_cs:.3f}"), html.Div(f"{c.severity:.2f}"),
            html.Div(str(c.query_count)), html.Div(_pct(c.unanswered_share)),
        ], className="stats"),
        html.Div("Sample questions", className="k muted"),
        html.Ul([html.Li(q) for q in c.sample_queries[:8]]),
        html.Div("Top recommendations", className="k muted"),
        html.Div(recs),
    ], className="detail")


def create_app(report_path: str | Path, audit_csv: Optional[str | Path] = None) -> Dash:
    report = GapReport.model_validate_json(Path(report_path).read_text())
    audit_csv = Path(audit_csv or report.config.get("audit", ""))
    points = load_points(audit_csv, report)
    clusters = [c.model_dump() for c in report.clusters]
    d = report.estimated_improvement_detail

    app = Dash(__name__, assets_folder=str(ASSETS), title="Umbra coverage report")

    def tile(label, value, hero=False):
        return html.Div([html.Div(label, className="label"), html.Div(value, className="value")],
                        className="tile hero" if hero else "tile")

    app.layout = html.Div(id="root", className="viz-root", children=[
        dcc.Location(id="url"),
        dcc.Store(id="selected", data=None),
        html.Div(className="header", children=[
            html.Div([
                html.H1("Knowledge base coverage"),
                html.Div(f"Report {str(report.report_id)[:8]} · generated {report.generated_at:%Y-%m-%d %H:%M} UTC · "
                         f"zones: dark < {report.thresholds.dark_below:.3f} ≤ thin ≤ "
                         f"{report.thresholds.adequate_above:.3f} < adequate ({report.thresholds.source})",
                         className="subtle"),
            ]),
            html.Div(className="theme-toggle", children=[
                dcc.RadioItems(id="theme", options=[{"label": "Light", "value": "light"},
                                                    {"label": "Dark", "value": "dark"}],
                               value="light", inline=True, inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
            ]),
        ]),
        html.Div(className="tiles", children=[
            tile("Overall coverage", f"{report.overall_coverage_score:.3f}", hero=True),
            tile("Probes", f"{report.probe_count:,}"),
            tile("Clusters", str(report.cluster_count)),
            tile("Dark zones", str(len(report.dark_zones))),
            tile("Thin zones", str(len(report.thin_zones))),
            tile(f"After top {d.recommendations_applied} recs (simulated)", f"{d.overall_after:.3f}"),
        ]),
        html.Div(className="controls", children=[
            html.Span("Color map by"),
            dcc.RadioItems(id="color-by", options=[{"label": "Zone", "value": "zone"},
                                                   {"label": "Coverage score", "value": "score"}],
                           value="zone", inline=True, inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
            html.Span("Strategies"),
            dcc.Checklist(id="strategies", options=sorted(points.strategy.unique()) if len(points) else [],
                          value=sorted(points.strategy.unique()) if len(points) else [], inline=True,
                          inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
        ]),
        html.Div(className="grid", children=[
            html.Div(className="card", children=[
                html.H2("Coverage map"),
                html.Div("Each point is a probe question, placed by meaning (UMAP). Dark and thin zones are labeled.",
                         className="muted"),
                dcc.Graph(id="map", config={"displaylogo": False, "scrollZoom": True}),
            ]),
            html.Div(id="detail", className="card"),
        ]),
        html.Div(className="card section", children=[
            html.H2("Clusters by severity"),
            html.Div("All tiers. The tier says whether a topic exists in the KB; severity also ranks depth gaps "
                     "inside covered topics.", className="muted"),
            dcc.Graph(id="severity", config={"displaylogo": False, "displayModeBar": False}),
        ]),
        html.Div(className="card section", children=[
            html.H2("All clusters"),
            dash_table.DataTable(
                id="table", data=table_rows(report), sort_action="native", page_size=30,
                columns=[
                    {"name": "rank", "id": "rank", "type": "numeric"}, {"name": "cluster", "id": "name"},
                    {"name": "zone", "id": "zone"}, {"name": "severity", "id": "severity", "type": "numeric"},
                    {"name": "mean coverage", "id": "mean_cs", "type": "numeric"},
                    {"name": "probes", "id": "size", "type": "numeric"},
                    {"name": "unanswered %", "id": "unanswered", "type": "numeric"},
                    {"name": "recs", "id": "recs", "type": "numeric"},
                ],
                style_as_list_view=True,
            ),
        ]),
        html.Div(className="card section", children=[
            html.H2("Where each probe strategy's questions landed"),
            dcc.Graph(id="strategy", config={"displaylogo": False, "displayModeBar": False}),
        ]),
    ])

    @app.callback(Output("root", "className"), Input("theme", "value"))
    def _theme(theme):
        return "viz-root theme-dark" if theme == "dark" else "viz-root"

    # ?theme=dark&cluster=4 so coverage_map_uri can link straight to a cluster
    @app.callback(Output("theme", "value"), Input("url", "search"), prevent_initial_call=False)
    def _theme_from_url(search):
        return "dark" if parse_qs((search or "").lstrip("?")).get("theme") == ["dark"] else "light"

    @app.callback(
        Output("selected", "data"),
        Input("url", "search"), Input("map", "clickData"), Input("severity", "clickData"),
        Input("table", "active_cell"), State("table", "derived_viewport_data"),
    )
    def _select(search, map_click, bar_click, cell, viewport):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else "url"
        if trigger == "url":
            wanted = parse_qs((search or "").lstrip("?")).get("cluster")
            ids = {c.cluster_id for c in report.clusters}
            return int(wanted[0]) if wanted and wanted[0].lstrip("-").isdigit() and int(wanted[0]) in ids else None
        if trigger == "map" and map_click:
            return int(map_click["points"][0]["customdata"][5])
        if trigger == "severity" and bar_click:
            return int(bar_click["points"][0]["customdata"][0])
        if trigger == "table" and cell and viewport:
            return int(viewport[cell["row"]]["id"])
        return None

    @app.callback(
        Output("map", "figure"), Output("severity", "figure"), Output("strategy", "figure"), Output("detail", "children"),
        Output("table", "style_header"), Output("table", "style_cell"), Output("table", "style_data_conditional"),
        Input("theme", "value"), Input("color-by", "value"), Input("strategies", "value"), Input("selected", "data"),
    )
    def _render(theme, color_by, strategies, selected):
        t = THEMES[theme]
        shown = points[points.strategy.isin(strategies or [])] if len(points) else points
        style_cell = {"backgroundColor": t["surface"], "color": t["secondary"], "border": "none",
                      "borderBottom": f"1px solid {t['grid']}", "fontFamily": "system-ui, sans-serif",
                      "fontSize": "13px", "padding": "6px 8px", "textAlign": "left"}
        style_header = {**style_cell, "color": t["muted"], "fontWeight": "600", "borderBottom": f"1px solid {t['axis']}"}
        conditional = [{"if": {"filter_query": f"{{id}} = {selected}"}, "color": t["ink"], "fontWeight": "600"}] \
            if selected is not None else []
        return (
            coverage_map(shown, clusters, theme, color_by, selected),
            severity_bars(clusters, theme),
            strategy_breakdown(report.strategy_breakdown, theme),
            detail_panel(report, selected),
            style_header, style_cell, conditional,
        )

    return app


def main(report: str, audit: Optional[str] = None, port: int = 8050, debug: bool = False) -> None:
    create_app(report, audit).run(port=port, debug=debug)
