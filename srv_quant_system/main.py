"""
main.py

Institutional-grade SRV dashboard built with:
- Dash
- dash_bootstrap_components (CYBORG dark theme)
- Plotly

Required dashboard zones:
1) Top KPI Cards: Market Dispersion, Credit Stress status, Top Long/Short candidates
2) Cross-Sectional Scanner: interactive DataTable + scatter (Z-score vs Hedge Ratio)
3) Sector Tear Sheet: callback on row selection:
   - Residual X-Ray chart with +/-2 std bands
   - Cards for Macro Betas, Fundamentals, Execution directives

The dashboard is designed to "separate everything" visually and also provide
"linking metrics" (rv_alignment, layer scores) clearly.

Run:
  python main.py
Then open the local Dash server URL printed in console.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.dash_table import DataTable

# Ensure local imports work when executed as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.stat_arb import QuantEngine
from config.settings import get_settings
from data.pipeline import DataLakeManager


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "srv_system.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _kpi_card(title: str, value: str, subtitle: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.Div(value, className="h3 mb-0"),
                html.Div(subtitle, className="text-muted small mt-1"),
            ]
        ),
        className=f"border-{color}",
    )


def _format_float(x: Any, fmt: str = "{:.2f}") -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        if not (v == v):  # NaN
            return "—"
        return fmt.format(v)
    except Exception:
        return "—"


def build_app() -> dash.Dash:
    settings = get_settings()
    configure_logging(settings.log_dir)
    logger = logging.getLogger("main")

    # 1) Data pipeline -> Parquet snapshot
    manager = DataLakeManager(settings)
    manager.build_snapshot(force_refresh=False)

    # 2) Analytics engine
    engine = QuantEngine(settings)
    engine.load()
    master_df = engine.calculate_conviction_score()

    # Prepare top candidates
    tradable = master_df[master_df["direction"].isin(["LONG", "SHORT"])].copy()
    top_long = tradable[tradable["direction"] == "LONG"].head(1)
    top_short = tradable[tradable["direction"] == "SHORT"].head(1)

    disp_ratio = master_df["market_dispersion_ratio"].iloc[0] if len(master_df) else None
    disp_z = master_df["market_dispersion_z"].iloc[0] if len(master_df) else None
    credit_stress = bool(master_df["credit_stress"].iloc[0]) if len(master_df) else False
    credit_z = master_df["credit_z"].iloc[0] if len(master_df) else None
    vix_level = master_df["vix_level"].iloc[0] if len(master_df) else None
    vix_pct = master_df["vix_percentile"].iloc[0] if len(master_df) else None

    # DataTable config
    display_cols = [
        "sector_name",
        "sector_ticker",
        "direction",
        "conviction_score",
        "pca_residual_z",
        "hedge_ratio",
        "score_stat",
        "score_macro",
        "score_fund",
        "score_vol",
        "pe_sector_portfolio",
        "rel_pe_vs_spy",
        "neg_or_missing_earnings_weight",
        "beta_tnx_60d",
        "beta_dxy_60d",
        "rv_alignment",
    ]

    table_df = master_df[display_cols].copy()
    table_df["conviction_score"] = table_df["conviction_score"].round(1)
    table_df["pca_residual_z"] = table_df["pca_residual_z"].round(2)
    table_df["hedge_ratio"] = table_df["hedge_ratio"].round(2)
    table_df["pe_sector_portfolio"] = table_df["pe_sector_portfolio"].round(2)
    table_df["rel_pe_vs_spy"] = table_df["rel_pe_vs_spy"].round(2)
    table_df["neg_or_missing_earnings_weight"] = (table_df["neg_or_missing_earnings_weight"] * 100.0).round(1)
    table_df["beta_tnx_60d"] = table_df["beta_tnx_60d"].round(3)
    table_df["beta_dxy_60d"] = table_df["beta_dxy_60d"].round(3)
    table_df["rv_alignment"] = table_df["rv_alignment"].round(3)

    # Scatter: Z vs Hedge Ratio
    scatter_fig = px.scatter(
        master_df,
        x="pca_residual_z",
        y="hedge_ratio",
        color="direction",
        hover_name="sector_name",
        hover_data={
            "sector_ticker": True,
            "conviction_score": True,
            "pe_sector_portfolio": True,
            "rel_pe_vs_spy": True,
            "rv_alignment": True,
        },
        title="Cross-Sectional Scanner: Z-Score vs Hedge Ratio",
    )
    scatter_fig.update_layout(template="plotly_dark", height=420)
    scatter_fig.update_xaxes(zeroline=True)

    # Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "SRV Quantamental System"

    # KPI zone values
    disp_value = f"{_format_float(disp_ratio)}"
    disp_sub = f"Dispersion Z: {_format_float(disp_z)} | VIX: {_format_float(vix_level)} | VIX Pctl: {_format_float(vix_pct, '{:.0%}')}"
    credit_value = "STRESS" if credit_stress else "NORMAL"
    credit_sub = f"HYG-IEF Z: {_format_float(credit_z)} (threshold {settings.credit_stress_z})"
    top_value = "—"
    top_sub = "No actionable signals."
    if not top_long.empty or not top_short.empty:
        long_txt = (
            f"LONG: {top_long['sector_name'].iloc[0]} ({top_long['sector_ticker'].iloc[0]}) "
            f"Score={_format_float(top_long['conviction_score'].iloc[0], '{:.1f}')}"
            if not top_long.empty
            else "LONG: —"
        )
        short_txt = (
            f"SHORT: {top_short['sector_name'].iloc[0]} ({top_short['sector_ticker'].iloc[0]}) "
            f"Score={_format_float(top_short['conviction_score'].iloc[0], '{:.1f}')}"
            if not top_short.empty
            else "SHORT: —"
        )
        top_value = "Top Candidates"
        top_sub = f"{long_txt} | {short_txt}"

    # Layout
    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(_kpi_card("Market Dispersion", disp_value, disp_sub, color="info"), md=4),
                    dbc.Col(_kpi_card("Credit Stress Regime", credit_value, credit_sub, color="danger" if credit_stress else "success"), md=4),
                    dbc.Col(_kpi_card("Top Long / Short", top_value, top_sub, color="primary"), md=4),
                ],
                className="mt-3",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Cross-Sectional Table", className="mb-2"),
                                    html.Div(
                                        "Tip: select a row to populate the tear sheet below.",
                                        className="text-muted small mb-2",
                                    ),
                                    DataTable(
                                        id="signals-table",
                                        columns=[{"name": c, "id": c} for c in table_df.columns],
                                        data=table_df.to_dict("records"),
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        row_selectable="single",
                                        selected_rows=[0] if len(table_df) else [],
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "backgroundColor": "#111111",
                                            "color": "#eaeaea",
                                            "border": "1px solid #333",
                                            "fontFamily": "monospace",
                                            "fontSize": "12px",
                                            "padding": "6px",
                                        },
                                        style_header={
                                            "backgroundColor": "#1f1f1f",
                                            "fontWeight": "bold",
                                            "border": "1px solid #444",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"filter_query": "{direction} = 'LONG'"},
                                                "backgroundColor": "rgba(0, 255, 0, 0.08)",
                                            },
                                            {
                                                "if": {"filter_query": "{direction} = 'SHORT'"},
                                                "backgroundColor": "rgba(255, 0, 0, 0.10)",
                                            },
                                            {
                                                "if": {"filter_query": "{conviction_score} >= 80"},
                                                "borderLeft": "4px solid #00ff99",
                                            },
                                        ],
                                    ),
                                ]
                            ),
                            className="mt-3",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Cross-Sectional Scatter", className="mb-2"),
                                    html.Div(
                                        "Each point is a sector. Use it to spot extreme dislocations (|Z|) and practical hedge ratios.",
                                        className="text-muted small mb-2",
                                    ),
                                    dcc.Graph(id="scatter-z-hr", figure=scatter_fig),
                                ]
                            ),
                            className="mt-3",
                        ),
                        md=5,
                    ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Sector Tear Sheet", className="mb-2"),
                                    html.Div(
                                        "X-Ray view of the OOS PCA residual (mispricing) with ±2σ bands, plus connected macro/fundamental/execution context.",
                                        className="text-muted small mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Card(dbc.CardBody(id="card-macro")), md=4),
                                            dbc.Col(dbc.Card(dbc.CardBody(id="card-fund")), md=4),
                                            dbc.Col(dbc.Card(dbc.CardBody(id="card-exec")), md=4),
                                        ],
                                        className="mb-3",
                                    ),
                                    dcc.Graph(id="residual-xray"),
                                ]
                            ),
                            className="mt-3",
                        ),
                        md=12,
                    ),
                ]
            ),

            # Store full master data (optional; helps callback determinism)
            dcc.Store(id="master-store", data=master_df.to_dict("records")),
        ],
    )

    # Callback: tear sheet updates on row selection
    @app.callback(
        Output("residual-xray", "figure"),
        Output("card-macro", "children"),
        Output("card-fund", "children"),
        Output("card-exec", "children"),
        Input("signals-table", "selected_rows"),
        State("signals-table", "data"),
        prevent_initial_call=False,
    )
    def update_tearsheet(selected_rows: Optional[List[int]], table_data: List[Dict[str, Any]]):
        if not table_data:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark", title="No data")
            return fig, "—", "—", "—"

        idx = 0
        if selected_rows and len(selected_rows) > 0:
            idx = int(selected_rows[0])

        idx = max(0, min(idx, len(table_data) - 1))
        row = table_data[idx]
        sector_ticker = row["sector_ticker"]
        sector_name = row["sector_name"]

        ts = engine.get_sector_tearsheet_series(sector_ticker)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts["residual_level"], name="Residual Level", mode="lines"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts["mean"], name="Rolling Mean", mode="lines"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts["upper_2s"], name="+2σ", mode="lines"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts["lower_2s"], name="-2σ", mode="lines"))

        fig.update_layout(
            template="plotly_dark",
            height=520,
            title=f"{sector_name} ({sector_ticker}) — OOS PCA Residual X-Ray (±2σ)",
            legend=dict(orientation="h"),
            margin=dict(l=30, r=20, t=60, b=30),
        )

        # Cards
        macro_card = html.Div(
            [
                html.Div("Macro Betas (60D)", className="text-muted small"),
                html.Div(f"β(TNX): {row['beta_tnx_60d']}", className="h5 mb-0"),
                html.Div(f"β(DXY): {row['beta_dxy_60d']}", className="h5 mb-0"),
                html.Div(f"β Magnitude: {row['beta_mag']}", className="text-muted small mt-1"),
            ]
        )

        fund_card = html.Div(
            [
                html.Div("Fundamentals (Holdings-Weighted)", className="text-muted small"),
                html.Div(f"P/E (portfolio): {row['pe_sector_portfolio']}", className="h5 mb-0"),
                html.Div(f"Rel P/E vs SPY: {row['rel_pe_vs_spy']}", className="h5 mb-0"),
                html.Div(f"Neg/Missing Earnings Weight: {row['neg_or_missing_earnings_weight']}%", className="text-muted small mt-1"),
                html.Div(f"EPS (weighted median): {row['eps_weighted_median']}", className="text-muted small"),
            ]
        )

        exec_card = html.Div(
            [
                html.Div("Execution Directives", className="text-muted small"),
                html.Div(f"Conviction: {row['conviction_score']}", className="h4 mb-0"),
                html.Div(f"Direction: {row['direction']}", className="h5 mb-0"),
                html.Div(f"Hedge Ratio: {row['hedge_ratio']}", className="h5 mb-0"),
                html.Div(f"Leg 1: {row.get('trade_leg_sector', '—')}", className="text-muted small mt-1"),
                html.Div(f"Leg 2: {row.get('trade_leg_spy', '—')}", className="text-muted small"),
                html.Div(f"RV Alignment: {row.get('rv_alignment', '—')}", className="text-muted small"),
            ]
        )

        return fig, macro_card, fund_card, exec_card

    logger.info("Dashboard ready. Starting server...")
    return app


if __name__ == "__main__":
    app = build_app()
    app.run_server(debug=False, host="0.0.0.0", port=8050)
