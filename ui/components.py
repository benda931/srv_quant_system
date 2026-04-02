"""
ui/components.py
==================
Shared UI component library for all dashboard tabs.

Eliminates duplication of KPI cards, formatters, charts, and export links
across 15 independent tab builders.

Usage:
    from ui.components import kpi, pct, ff, csv_download_link, section_header
"""
from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def ff(x: Any, fmt: str = "{:.2f}") -> str:
    """Format float safely — returns '—' for NaN/None/errors."""
    try:
        v = float(x)
        return "—" if v != v else fmt.format(v)
    except Exception:
        return "—"


def pct(x: Any, decimals: int = 1) -> str:
    """Format as percentage — e.g., 0.123 → '12.3%'."""
    try:
        v = float(x)
        return "—" if v != v else f"{v * 100:.{decimals}f}%"
    except Exception:
        return "—"


def bps(x: Any) -> str:
    """Format as basis points — e.g., 0.0015 → '15.0bps'."""
    try:
        v = float(x)
        return "—" if v != v else f"{v * 10000:.1f}bps"
    except Exception:
        return "—"


def money(x: Any, prefix: str = "$") -> str:
    """Format as money — e.g., 1234567 → '$1,234,567'."""
    try:
        v = float(x)
        return "—" if v != v else f"{prefix}{v:,.0f}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, color: str = "primary", small: bool = False) -> dbc.Col:
    """
    Standard KPI card with colored top border.

    Usage:
        kpi("Sharpe", "1.52", "success")
        kpi("VaR 95%", "-2.1%", "danger", small=True)
    """
    size_class = "h5" if small else "h4"
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, className="text-muted",
                         style={"fontSize": "11px", "textAlign": "center"}),
                html.Div(value, className=f"{size_class} mb-0 text-center fw-bold"),
            ]),
            className=f"border-{color} text-center h-100",
            style={"borderTop": f"3px solid var(--bs-{color})"},
        ),
    )


def kpi_row(items: List[dict], class_name: str = "g-2 mb-3") -> dbc.Row:
    """
    Build a row of KPI cards from a list of dicts.

    Usage:
        kpi_row([
            {"label": "Sharpe", "value": "1.52", "color": "success"},
            {"label": "VaR", "value": "-2.1%", "color": "danger"},
        ])
    """
    return dbc.Row(
        [kpi(item["label"], item["value"], item.get("color", "primary"),
             item.get("small", False))
         for item in items],
        className=class_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section Headers
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "", icon: str = "") -> html.Div:
    """Section header with optional icon and subtitle."""
    children = []
    if icon:
        children.append(html.Span(f"{icon} ", style={"fontSize": "18px"}))
    children.append(html.Span(title, className="fw-bold"))
    header = html.H6(children, className="mb-1")

    if subtitle:
        return html.Div([header, html.Small(subtitle, className="text-muted")])
    return header


# ─────────────────────────────────────────────────────────────────────────────
# Cards & Containers
# ─────────────────────────────────────────────────────────────────────────────

def info_card(title: str, body, color: str = "secondary",
              border_top: bool = True) -> dbc.Card:
    """Standard info card with header + body."""
    style = {"borderTop": f"3px solid var(--bs-{color})"} if border_top else {}
    return dbc.Card([
        dbc.CardHeader(html.H6(title, className="mb-0 text-center")),
        dbc.CardBody(body if isinstance(body, list) else [body]),
    ], className=f"border-{color} mb-3", style=style)


def chart_card(figure: go.Figure, height: int = 300) -> dbc.Card:
    """Wrap a Plotly figure in a card."""
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=figure, config={"displayModeBar": False})),
        className="border-0 bg-transparent mb-3",
    )


def scrollable_table(table: Any, max_height: int = 350) -> html.Div:
    """Wrap a table in a scrollable container."""
    return html.Div(table, style={"maxHeight": f"{max_height}px", "overflowY": "auto"})


# ─────────────────────────────────────────────────────────────────────────────
# Export / Download
# ─────────────────────────────────────────────────────────────────────────────

def csv_download_link(df: pd.DataFrame, filename: str,
                      label: str = "Export CSV") -> html.A:
    """
    Data-URI based CSV download link (no callback needed).
    Uses UTF-8 BOM for Excel compatibility.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    b64 = base64.b64encode(csv_bytes).decode("ascii")
    return html.A(
        label,
        href=f"data:text/csv;charset=utf-8;base64,{b64}",
        download=filename,
        className="btn btn-sm btn-outline-secondary ms-2",
        style={"fontSize": "0.75rem", "direction": "ltr"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dark_layout(title: str = "", height: int = 300,
                y_title: str = "", y_suffix: str = "",
                **kwargs) -> dict:
    """Standard dark theme layout for Plotly figures."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        height=height,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=kwargs.get("showlegend", True),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13))
    if y_title:
        layout["yaxis"] = dict(title=y_title)
        if y_suffix:
            layout["yaxis"]["ticksuffix"] = y_suffix
    layout.update(kwargs)
    return layout


def color_for_value(value: float, thresholds: tuple = (0, 0.5),
                    colors: tuple = ("danger", "warning", "success")) -> str:
    """Return Bootstrap color based on value thresholds."""
    if value < thresholds[0]:
        return colors[0]
    elif value < thresholds[1]:
        return colors[1]
    return colors[2]


# ─────────────────────────────────────────────────────────────────────────────
# Regime helpers
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "CALM": "#4caf50", "NORMAL": "#2196f3",
    "TENSION": "#ff9800", "CRISIS": "#f44336",
}

REGIME_HEBREW = {
    "CALM": "רגוע", "NORMAL": "רגיל",
    "TENSION": "מתח", "CRISIS": "משבר",
}

REGIME_BADGE_COLORS = {
    "CALM": "success", "NORMAL": "info",
    "TENSION": "warning", "CRISIS": "danger",
}


def regime_badge(regime: str) -> dbc.Badge:
    """Colored badge for a regime label."""
    color = REGIME_BADGE_COLORS.get(regime, "secondary")
    heb = REGIME_HEBREW.get(regime, regime)
    return dbc.Badge(f"{regime} ({heb})", color=color, style={"fontSize": "10px"})
