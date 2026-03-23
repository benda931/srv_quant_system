"""
data_ops/health_panel.py

Dash UI components for the Data Health tab.

Single public function:
    build_health_tab(report: DataHealthReport) -> dbc.Container

Wire into main.py render_tab callback:
    if active_tab == "tab-health":
        return build_health_tab(_health)
"""
from __future__ import annotations

from typing import Any, List, Optional

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, html

from data_ops.status_report import DataHealthReport


# =========================================================================
# Helpers
# =========================================================================

def _badge_color(label: str) -> str:
    return {
        "HEALTHY": "success",
        "FRESH":   "success",
        "DEGRADED": "warning",
        "STALE":   "warning",
        "CRITICAL": "danger",
        "MISSING": "danger",
        "UNKNOWN": "secondary",
    }.get(label.upper(), "secondary")


# =========================================================================
# Panel builders
# =========================================================================

def _score_card(report: DataHealthReport) -> dbc.Card:
    pct   = round(report.health_score * 100, 1)
    color = _badge_color(report.health_label)
    # Format datetime outside the f-string to stay compatible with Python < 3.12
    as_of_str = report.as_of.strftime("%Y-%m-%d %H:%M")

    return dbc.Card(
        dbc.CardBody([
            html.Div("Data Health Score", className="text-muted small mb-1"),
            html.Div(
                [
                    html.Span(f"{pct}%", className="h2 mb-0 me-2"),
                    dbc.Badge(report.health_label, color=color),
                ],
                className="d-flex align-items-baseline",
            ),
            html.Div(f"As of {as_of_str}", className="text-muted small mt-1"),
            (dbc.Badge(
                "DATA DEGRADED — treat analytics with caution",
                color="warning",
                className="mt-1 d-block",
            ) if report.degraded else None),
        ]),
        className=f"border-{color}",
    )


def _freshness_row(report: DataHealthReport) -> dbc.Row:
    fq   = report.freshness
    cols: List[Any] = []

    for name in ("prices", "fundamentals", "weights"):
        art   = fq.artifacts.get(name)
        state = art.state if art else "MISSING"
        color = _badge_color(state)

        if art and art.age_hours is not None:
            age_str = f"{art.age_hours:.1f}h ago"
        else:
            age_str = "—"

        cols.append(dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.Div(name.capitalize(), className="text-muted small"),
                    html.Div(age_str, className="h4 mb-0"),
                    html.Div(state, className=f"text-{color} small mt-1"),
                ]),
                className=f"border-{color}",
            ),
            md=3,
        ))

    # Last price date card
    pd_ = fq.price_detail
    if pd_ and pd_.last_price_date:
        color    = "success" if pd_.price_date_gap_ok else "warning"
        gap_str  = f"{pd_.days_since_last_price}d gap"
        date_str = str(pd_.last_price_date)
        cols.append(dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.Div("Last price date", className="text-muted small"),
                    html.Div(date_str, className="h4 mb-0"),
                    html.Div(gap_str, className=f"text-{color} small mt-1"),
                ]),
                className=f"border-{color}",
            ),
            md=3,
        ))

    return dbc.Row(cols, className="g-2 mt-2")


def _coverage_table(report: DataHealthReport) -> dbc.Card:
    df = report.to_coverage_table()

    if df.empty:
        return dbc.Card(dbc.CardBody([
            html.H6("Coverage"),
            html.Div("No coverage data available.", className="text-muted small"),
        ]))

    tbl = dash_table.DataTable(
        id="health-coverage-table",
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#111",
            "color": "#eaeaea",
            "border": "1px solid #333",
            "fontFamily": "monospace",
            "fontSize": "12px",
            "padding": "5px 8px",
            "textAlign": "center",
        },
        style_header={
            "backgroundColor": "#1f1f1f",
            "fontWeight": "bold",
            "border": "1px solid #444",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{Status} = "DEGRADED"'},
                "backgroundColor": "rgba(255,193,7,.1)",
                "borderLeft": "3px solid #ffc107",
            },
            {
                "if": {"filter_query": '{Fallback} = "YES"'},
                "color": "#ffc107",
            },
            {
                "if": {"filter_query": '{Price} = "MISSING"'},
                "color": "#dc3545",
            },
        ],
    )

    return dbc.Card(dbc.CardBody([
        html.H6("Coverage by ETF", className="mb-1"),
        html.Div(
            "Holdings rows, PE coverage, and ETF-level fallback usage.",
            className="text-muted small mb-2",
        ),
        tbl,
    ]))


def _warnings_panel(report: DataHealthReport) -> dbc.Card:
    if not report.warnings and not report.errors:
        return dbc.Card(dbc.CardBody([
            html.H6("Warnings"),
            html.Div("No warnings — data layer healthy.", className="text-success small"),
        ]))

    items: List[Any] = (
        [dbc.Alert(e, color="danger",  className="py-1 px-2 mb-1 small")
         for e in report.errors]
        + [dbc.Alert(w, color="warning", className="py-1 px-2 mb-1 small")
           for w in report.warnings[:25]]
    )

    if len(report.warnings) > 25:
        items.append(html.Div(
            f"… and {len(report.warnings) - 25} more (see logs).",
            className="text-muted small",
        ))

    n_w = len(report.warnings)
    n_e = len(report.errors)
    return dbc.Card(dbc.CardBody([
        html.H6(f"Warnings ({n_w}) / Errors ({n_e})", className="mb-2"),
        *items,
    ]))


def _validation_panel(report: DataHealthReport) -> dbc.Card:
    if not report.validation.issues:
        return dbc.Card(dbc.CardBody([
            html.H6("Validation"),
            html.Div("All structural checks passed.", className="text-success small"),
        ]))

    severity_color = {"ERROR": "danger", "WARNING": "warning", "INFO": "info"}

    rows = [
        html.Tr([
            html.Td(dbc.Badge(
                issue.severity,
                color=severity_color.get(issue.severity, "secondary"),
            )),
            html.Td(issue.code, style={"fontFamily": "monospace", "fontSize": "12px"}),
            html.Td(issue.message, style={"fontSize": "12px"}),
            html.Td(issue.remediation, style={"fontSize": "11px", "color": "#888"}),
        ])
        for issue in report.validation.issues
    ]

    n_e = report.validation.error_count
    n_w = report.validation.warning_count

    return dbc.Card(dbc.CardBody([
        html.H6(
            f"Validation — {n_e} error(s), {n_w} warning(s)",
            className="mb-2",
        ),
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th(c) for c in ["Level", "Code", "Message", "Remediation"]
                ])),
                html.Tbody(rows),
            ],
            className="table table-dark table-sm table-bordered",
            style={"fontSize": "12px"},
        ),
    ]))


# =========================================================================
# Public entry point
# =========================================================================

def build_health_tab(report: DataHealthReport) -> dbc.Container:
    """
    Build the full Data Health tab layout from a DataHealthReport.

    Wire into main.py render_tab callback:
        if active_tab == "tab-health":
            return build_health_tab(_health)
    """
    n_e = report.validation.error_count
    n_w = report.validation.warning_count

    return dbc.Container(
        fluid=True,
        children=[
            # Score card
            dbc.Row([
                dbc.Col(_score_card(report), md=12),
            ], className="mt-3"),

            # Freshness
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H5("Snapshot Freshness", className="mb-2"),
                        html.Div(
                            "STALE = artifact age exceeds cache_max_age_hours threshold.",
                            className="text-muted small mb-1",
                        ),
                        _freshness_row(report),
                    ])),
                    md=12,
                ),
            ], className="mt-3"),

            # Coverage table
            dbc.Row([
                dbc.Col(_coverage_table(report), md=12),
            ], className="mt-3"),

            # Validation + Warnings side by side
            dbc.Row([
                dbc.Col(_validation_panel(report), md=6),
                dbc.Col(_warnings_panel(report),   md=6),
            ], className="mt-3"),
        ],
    )
