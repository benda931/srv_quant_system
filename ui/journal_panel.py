"""ui/journal_panel.py — Decision Journal tab for the SRV Quantamental DSS Dash app."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table import DataTable

if TYPE_CHECKING:
    from data_ops.journal import PMJournal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants (mirror main.py)
# ---------------------------------------------------------------------------
RTL_STYLE: Dict[str, str] = {"direction": "rtl", "textAlign": "right"}
CARD_STYLE: Dict[str, str] = {"direction": "rtl", "textAlign": "right", "height": "100%"}

_DARK_TABLE_STYLE: Dict[str, Any] = {
    "backgroundColor": "#111111",
    "color": "#eaeaea",
    "border": "1px solid #333",
}
_DARK_HEADER_STYLE: Dict[str, Any] = {
    "backgroundColor": "#1a1a2e",
    "color": "#eaeaea",
    "fontWeight": "bold",
    "border": "1px solid #333",
    "textAlign": "right",
    "direction": "rtl",
}

_SECTOR_OPTIONS = [
    {"label": "XLC — Communication Services", "value": "XLC"},
    {"label": "XLY — Consumer Discretionary", "value": "XLY"},
    {"label": "XLP — Consumer Staples", "value": "XLP"},
    {"label": "XLE — Energy", "value": "XLE"},
    {"label": "XLF — Financials", "value": "XLF"},
    {"label": "XLV — Health Care", "value": "XLV"},
    {"label": "XLI — Industrials", "value": "XLI"},
    {"label": "XLB — Materials", "value": "XLB"},
    {"label": "XLRE — Real Estate", "value": "XLRE"},
    {"label": "XLK — Technology", "value": "XLK"},
    {"label": "XLU — Utilities", "value": "XLU"},
]

_COL_LABELS = {
    "timestamp": "תאריך",
    "sector": "סקטור",
    "model_direction": "מודל",
    "pm_direction": "החלטה",
    "conviction_score": "שכנוע",
    "regime": "רג'ים",
    "notes": "הערות",
}


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------
def _kpi_card(title: str, value: str, subtitle: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small", style=RTL_STYLE),
                html.Div(value, className="h3 mb-0", style=RTL_STYLE),
                html.Div(subtitle, className="text-muted small mt-1", style=RTL_STYLE),
            ]
        ),
        className=f"border-{color}",
        style=CARD_STYLE,
    )


def _build_kpi_row(journal: "PMJournal") -> dbc.Row:
    """Top KPI row: Total Decisions (30d), Override Rate (%), PM vs Model Accuracy."""
    try:
        stats = journal.get_stats()
        recent = journal.get_recent(n=50)

        # Total decisions last 30 days
        n_total = stats.get("n_decisions", 0)
        if not recent.empty and "timestamp" in recent.columns:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
            ts_col = pd.to_datetime(recent["timestamp"], utc=True, errors="coerce")
            n_30d = int((ts_col >= cutoff).sum())
        else:
            n_30d = 0

        # Override rate
        n_overrides = stats.get("n_overrides", 0)
        override_rate = n_overrides / n_total * 100 if n_total else 0.0

        # PM accuracy
        acc_stats = journal.get_override_accuracy()
        pm_acc = acc_stats.get("pm_accuracy")
        if pm_acc is not None:
            pm_acc_str = f"{pm_acc:.1%}"
        else:
            pm_acc_str = "N/A"

    except Exception:
        logger.exception("Failed to compute journal KPIs")
        n_30d, override_rate, pm_acc_str = 0, 0.0, "N/A"

    return dbc.Row(
        [
            dbc.Col(_kpi_card("סה״כ החלטות (30 יום)", str(n_30d), "תקופה אחרונה", "info"), md=4),
            dbc.Col(_kpi_card("שיעור Override", f"{override_rate:.1f}%", "מכלל ההחלטות", "warning"), md=4),
            dbc.Col(_kpi_card("דיוק PM vs. מודל", pm_acc_str, "על overrides שנפתרו", "success"), md=4),
        ],
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Recent decisions table
# ---------------------------------------------------------------------------
def _build_decisions_table(journal: "PMJournal") -> html.Div:
    """Table of the last 50 PM decisions with Hebrew column headers and row color-coding."""
    try:
        df = journal.get_recent(n=50)
    except Exception:
        logger.exception("Failed to load recent decisions")
        df = pd.DataFrame()

    if df.empty:
        return html.Div("אין החלטות מתועדות", className="text-muted text-center my-4", style=RTL_STYLE)

    display_cols = [c for c in _COL_LABELS if c in df.columns]
    df_display = df[display_cols].copy()

    # Truncate timestamps to date only
    if "timestamp" in df_display.columns:
        df_display["timestamp"] = pd.to_datetime(df_display["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    # Conviction to 1 decimal
    if "conviction_score" in df_display.columns:
        df_display["conviction_score"] = pd.to_numeric(df_display["conviction_score"], errors="coerce").round(1)

    columns = [{"name": _COL_LABELS.get(c, c), "id": c} for c in display_cols]

    # Row colour-coding: OVERRIDE → amber; agreement True → subtle green
    style_data_conditional: List[Dict[str, Any]] = [
        {
            "if": {"filter_query": '{pm_direction} = "OVERRIDE"'},
            "backgroundColor": "#3d2b00",
            "color": "#f0a500",
        },
        {
            "if": {"filter_query": '{pm_direction} = {model_direction}'},
            "backgroundColor": "#0d2b1a",
        },
    ]

    table = DataTable(
        id="journal-decisions-table",
        columns=columns,
        data=df_display.to_dict("records"),
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={**_DARK_TABLE_STYLE, "textAlign": "right", "direction": "rtl", "padding": "6px 10px"},
        style_header=_DARK_HEADER_STYLE,
        style_data_conditional=style_data_conditional,
        sort_action="native",
    )

    return html.Div(
        [
            html.Div("החלטות אחרונות (עד 50)", className="fw-bold mb-2", style=RTL_STYLE),
            table,
        ]
    )


# ---------------------------------------------------------------------------
# Override accuracy bar chart
# ---------------------------------------------------------------------------
def _build_accuracy_chart(journal: "PMJournal") -> dcc.Graph:
    """Sector-level PM vs model accuracy bar chart."""
    try:
        df = journal.get_recent(n=500)
    except Exception:
        logger.exception("Failed to load decisions for accuracy chart")
        df = pd.DataFrame()

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title="דיוק PM vs. מודל לפי סקטור",
        margin=dict(l=20, r=20, t=55, b=40),
        xaxis_title="סקטור",
        yaxis_title="שיעור הסכמה",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
    )

    if df.empty or "sector" not in df.columns or "agreement" not in df.columns:
        return dcc.Graph(figure=fig, id="journal-accuracy-chart")

    agg = (
        df.groupby("sector")["agreement"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    agg = agg.sort_values("accuracy", ascending=False)

    colors = ["#2ecc71" if v >= 0.6 else "#e74c3c" for v in agg["accuracy"]]

    fig.add_trace(
        go.Bar(
            x=agg["sector"],
            y=agg["accuracy"],
            marker_color=colors,
            text=[f"{v:.0%} ({n})" for v, n in zip(agg["accuracy"], agg["n"])],
            textposition="outside",
        )
    )

    return dcc.Graph(figure=fig, id="journal-accuracy-chart")


# ---------------------------------------------------------------------------
# Log-decision form
# ---------------------------------------------------------------------------
def _build_log_form() -> dbc.Card:
    """Form for logging a new PM decision."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("תיעוד החלטה חדשה", className="fw-bold mb-3 h5", style=RTL_STYLE),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("סקטור", style=RTL_STYLE),
                                dcc.Dropdown(
                                    id="journal-sector-dropdown",
                                    options=_SECTOR_OPTIONS,
                                    placeholder="בחר סקטור...",
                                    style={"direction": "rtl", "backgroundColor": "#222", "color": "#eaeaea"},
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("כיוון PM", style=RTL_STYLE),
                                dbc.RadioItems(
                                    id="journal-pm-direction",
                                    options=[
                                        {"label": "LONG", "value": "LONG"},
                                        {"label": "SHORT", "value": "SHORT"},
                                        {"label": "NEUTRAL", "value": "NEUTRAL"},
                                        {"label": "OVERRIDE", "value": "OVERRIDE"},
                                    ],
                                    value="LONG",
                                    inline=True,
                                    style=RTL_STYLE,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("כיוון מודל", style=RTL_STYLE),
                                dbc.RadioItems(
                                    id="journal-model-direction",
                                    options=[
                                        {"label": "LONG", "value": "LONG"},
                                        {"label": "SHORT", "value": "SHORT"},
                                        {"label": "NEUTRAL", "value": "NEUTRAL"},
                                    ],
                                    value="LONG",
                                    inline=True,
                                    style=RTL_STYLE,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("ציון שכנוע (0–100)", style=RTL_STYLE),
                                dbc.Input(
                                    id="journal-conviction-input",
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=0.5,
                                    placeholder="לדוגמה: 72.5",
                                    style={"backgroundColor": "#222", "color": "#eaeaea", "textAlign": "right"},
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("הערות", style=RTL_STYLE),
                                dbc.Textarea(
                                    id="journal-notes-input",
                                    placeholder="הסבר קצר לאות / סיבה לסטייה...",
                                    rows=2,
                                    style={
                                        "backgroundColor": "#222",
                                        "color": "#eaeaea",
                                        "textAlign": "right",
                                        "direction": "rtl",
                                    },
                                ),
                            ],
                            md=8,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "תעד החלטה",
                                id="journal-submit-btn",
                                color="primary",
                                n_clicks=0,
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(id="journal-submit-feedback", style=RTL_STYLE),
                            md=9,
                        ),
                    ]
                ),
            ]
        ),
        style={"backgroundColor": "#1a1a1a", "border": "1px solid #333"},
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_journal_tab(journal: "PMJournal") -> dbc.Tab:
    """Build and return the Decision Journal dbc.Tab component."""
    content = dbc.Container(
        fluid=True,
        children=[
            html.Div("יומן החלטות PM", className="h4 mt-3 mb-3 fw-bold", style=RTL_STYLE),
            _build_kpi_row(journal),
            dbc.Row(
                [
                    dbc.Col(_build_decisions_table(journal), md=8),
                    dbc.Col(_build_accuracy_chart(journal), md=4),
                ],
                className="mb-3",
            ),
            _build_log_form(),
        ],
    )

    return dbc.Tab(
        label="Decision Journal",
        tab_id="tab-journal",
        children=[content],
    )
