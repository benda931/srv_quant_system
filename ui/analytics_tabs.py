"""
ui/analytics_tabs.py
--------------------
Dash tab builders for the heavy-analytics panels:

  - build_stress_tab(stress_results, settings)          → Stress Testing
  - build_risk_tab(risk_report, master_df)              → Portfolio Risk
  - build_backtest_tab(backtest_result | None)          → Walk-Forward Backtest
  - build_corr_vol_tab(corr_vol_analysis)               → Correlation & Vol Pricing
  - build_daily_brief_panel(brief_txt)                  → Daily Brief modal body
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc

# ── Local imports ────────────────────────────────────────────────────────────
try:
    from analytics.stress import StressResult
except ImportError:
    StressResult = Any  # type: ignore

try:
    from analytics.portfolio_risk import RiskReport
except ImportError:
    RiskReport = Any  # type: ignore

try:
    from analytics.backtest import BacktestResult
except ImportError:
    BacktestResult = Any  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ff(x: Any, fmt: str = "{:.2f}") -> str:
    """Format float safely."""
    try:
        v = float(x)
        if v != v:
            return "—"
        return fmt.format(v)
    except Exception:
        return "—"


def _pct(x: Any, decimals: int = 1) -> str:
    try:
        v = float(x)
        if v != v:
            return "—"
        return f"{v * 100:.{decimals}f}%"
    except Exception:
        return "—"


def _kpi(label: str, value: str, color: str = "primary", small: bool = False) -> dbc.Col:
    size_class = "h5" if small else "h4"
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="text-muted", style={"fontSize": "11px", "textAlign": "center"}),
                    html.Div(value, className=f"{size_class} mb-0 text-center fw-bold"),
                ]
            ),
            className=f"border-{color} text-center h-100",
            style={"borderTop": f"3px solid var(--bs-{color})"},
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TESTING TAB
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_HEBREW: Dict[str, str] = {
    "RATES_SHOCK_UP":         "⬆ עליית ריבית חדה (+150bps)",
    "RATES_SHOCK_DOWN":       "⬇ ירידת ריבית (-100bps)",
    "RISK_OFF_ACUTE":         "🔴 Risk-Off חריף (VIX 45)",
    "RISK_OFF_CHRONIC":       "🟠 Risk-Off כרוני (6 חודשים)",
    "STAGFLATION":            "📉 סטגפלציה",
    "TECH_SELLOFF":           "💻 טכנולוגיה -25%",
    "CREDIT_CRISIS":          "💳 משבר אשראי",
    "DOLLAR_SURGE":           "💵 חיזוק דולר (+10%)",
    "EARNINGS_MISS":          "📊 רווחים נמוכים מציפיות",
    "CORRELATION_BREAKDOWN":  "🔗 שבירת מבנה קורלציות",
}


def build_stress_tab(stress_results: Optional[List[Any]], master_df: Optional[pd.DataFrame] = None) -> html.Div:
    """Full stress-testing tab layout."""

    if not stress_results:
        return html.Div(
            dbc.Alert("לא הורצו stress tests — נסה לאחד המודולים.", color="warning"),
            className="mt-3",
        )

    # ── KPI summary row ──────────────────────────────────────────────────────
    worst = stress_results[0]
    best  = stress_results[-1]
    avg_pnl = sum(r.portfolio_pnl_estimate for r in stress_results) / len(stress_results)
    n_negative = sum(1 for r in stress_results if r.portfolio_pnl_estimate < 0)

    worst_color = "danger" if worst.portfolio_pnl_estimate < -0.05 else "warning" if worst.portfolio_pnl_estimate < 0 else "success"
    kpi_row = dbc.Row(
        [
            _kpi("תרחיש הגרוע", f"{_pct(worst.portfolio_pnl_estimate)} ({_SCENARIO_HEBREW.get(worst.scenario_name, worst.scenario_name)[:20]})", worst_color),
            _kpi("תרחיש הטוב",  f"{_pct(best.portfolio_pnl_estimate)} ({_SCENARIO_HEBREW.get(best.scenario_name, best.scenario_name)[:20]})", "success"),
            _kpi("ממוצע P&L",   _pct(avg_pnl), "warning" if avg_pnl < 0 else "primary"),
            _kpi("תרחישים שליליים", f"{n_negative} / {len(stress_results)}", "danger" if n_negative >= 7 else "warning"),
        ],
        className="g-2 mb-3",
    )

    # ── P&L bar chart ────────────────────────────────────────────────────────
    names  = [_SCENARIO_HEBREW.get(r.scenario_name, r.scenario_name) for r in stress_results]
    pnls   = [r.portfolio_pnl_estimate * 100 for r in stress_results]
    colors = ["#dc3545" if p < 0 else "#20c997" for p in pnls]

    bar_fig = go.Figure(
        go.Bar(
            x=names, y=pnls,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pnls],
            textposition="outside",
            hovertemplate="%{x}<br>P&L: %{y:.2f}%<extra></extra>",
        )
    )
    bar_fig.add_hline(y=0, line_color="#555", line_width=1)
    bar_fig.update_layout(
        template="plotly_dark",
        height=340,
        title="השפעת תרחישים על P&L הספר (אחוזים)",
        margin=dict(l=10, r=10, t=50, b=80),
        yaxis_title="P&L (%)",
        showlegend=False,
        xaxis_tickangle=-25,
    )

    # ── Reliability bar chart ────────────────────────────────────────────────
    reliabilities = [r.signal_reliability_score * 100 for r in stress_results]
    rel_fig = go.Figure(
        go.Bar(
            x=names, y=reliabilities,
            marker_color=["#0dcaf0" if r >= 60 else "#ffc107" if r >= 40 else "#dc3545" for r in reliabilities],
            text=[f"{r:.0f}%" for r in reliabilities],
            textposition="outside",
            hovertemplate="%{x}<br>איכות אות: %{y:.0f}%<extra></extra>",
        )
    )
    rel_fig.update_layout(
        template="plotly_dark",
        height=300,
        title="אמינות האות בכל תרחיש (Signal Reliability %)",
        margin=dict(l=10, r=10, t=50, b=80),
        yaxis_title="Reliability (%)",
        yaxis_range=[0, 110],
        showlegend=False,
        xaxis_tickangle=-25,
    )

    # ── Detail table ─────────────────────────────────────────────────────────
    tbl_rows = []
    for r in stress_results:
        pnl_pct = r.portfolio_pnl_estimate * 100
        pnl_color = "#dc3545" if pnl_pct < 0 else "#20c997"
        rel_pct = r.signal_reliability_score * 100
        rel_color = "#0dcaf0" if rel_pct >= 60 else "#ffc107" if rel_pct >= 40 else "#dc3545"
        tbl_rows.append(
            html.Tr(
                [
                    html.Td(_SCENARIO_HEBREW.get(r.scenario_name, r.scenario_name), style={"textAlign": "right", "fontSize": "12px"}),
                    html.Td(
                        html.Strong(f"{pnl_pct:+.2f}%", style={"color": pnl_color}),
                        className="text-center",
                    ),
                    html.Td(
                        html.Span(f"{rel_pct:.0f}%", style={"color": rel_color}),
                        className="text-center",
                    ),
                    html.Td(str(r.worst_sector or "—"), className="text-center", style={"fontSize": "11px"}),
                    html.Td(str(r.best_sector or "—"),  className="text-center", style={"fontSize": "11px"}),
                    html.Td(str(r.regime_label or "—"), className="text-center", style={"fontSize": "11px"}),
                    html.Td(
                        html.Small(str(r.notes or "")[:80], className="text-muted"),
                        style={"textAlign": "right", "fontSize": "10px"},
                    ),
                ],
                style={
                    "backgroundColor": "rgba(220,53,69,0.06)" if pnl_pct < -5 else
                                       "rgba(255,193,7,0.05)" if pnl_pct < 0 else
                                       "rgba(32,201,151,0.05)"
                },
            )
        )

    detail_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("תרחיש",               style={"textAlign": "right"}),
                        html.Th("P&L",                  className="text-center"),
                        html.Th("אמינות אות",           className="text-center"),
                        html.Th("Worst Sector",          className="text-center"),
                        html.Th("Best Sector",           className="text-center"),
                        html.Th("Regime",                className="text-center"),
                        html.Th("הערות",                 style={"textAlign": "right"}),
                    ],
                    style={"backgroundColor": "#1a1a2e", "fontSize": "11px"},
                )
            ),
            html.Tbody(tbl_rows),
        ],
        className="table table-dark table-sm table-hover",
        style={"fontSize": "12px"},
    )

    return html.Div(
        [
            kpi_row,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=bar_fig, config={"displayModeBar": False}))), md=7),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=rel_fig, config={"displayModeBar": False}))), md=5),
                ],
                className="mb-3 g-2",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("פירוט כל התרחישים", style={"textAlign": "right"}),
                        html.Div(detail_table, style={"overflowX": "auto"}),
                    ]
                ),
                className="mb-3",
            ),
        ],
        className="mt-3",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_tab(risk_report: Optional[Any], master_df: Optional[pd.DataFrame] = None) -> html.Div:
    """Full portfolio risk tab layout."""

    if risk_report is None:
        return html.Div(
            dbc.Alert("לא חושב Portfolio Risk — נתונים חסרים.", color="warning"),
            className="mt-3",
        )

    rr = risk_report  # alias for brevity

    # Breach badges
    vol_breach  = getattr(rr, "vol_target_breach",  False)
    wgt_breach  = getattr(rr, "max_weight_breach",  False)

    breach_alerts = []
    if vol_breach:
        breach_alerts.append(
            dbc.Alert("⚠ סטיית תקן הפורטפוליו חורגת מהיעד (12%)", color="danger", className="py-2 mb-2")
        )
    if wgt_breach:
        breach_alerts.append(
            dbc.Alert("⚠ משקל סקטור בודד חורג מהמקסימום המותר (20%)", color="warning", className="py-2 mb-2")
        )

    # ── KPI row ──────────────────────────────────────────────────────────────
    vol_color  = "danger" if vol_breach else "success"
    var_color  = "danger" if getattr(rr, "var_95_1d", 0) < -0.02 else "warning"

    kpi_row = dbc.Row(
        [
            _kpi("תנודתיות שנתית",    _pct(getattr(rr, "portfolio_vol_ann",  None)), vol_color),
            _kpi("VaR 95% (יומי)",    _pct(getattr(rr, "var_95_1d",          None)), var_color),
            _kpi("CVaR 95% (יומי)",   _pct(getattr(rr, "cvar_95_1d",         None)), "danger"),
            _kpi("HHI ריכוזיות",      _ff(getattr(rr, "concentration_hhi",   None), "{:.3f}"), "secondary"),
            _kpi("מקסימום סקטור",     _pct(getattr(rr, "max_sector_weight",  None)), "warning" if wgt_breach else "secondary"),
        ],
        className="g-2 mb-3",
    )

    # ── MCTR bar chart ───────────────────────────────────────────────────────
    mctr: pd.Series = getattr(rr, "mctr_series", pd.Series(dtype=float))
    mctr_fig = go.Figure()
    if not mctr.empty:
        mctr_sorted = mctr.sort_values(ascending=False)
        mctr_colors = ["#dc3545" if v > 0 else "#20c997" for v in mctr_sorted.values]
        mctr_fig.add_trace(
            go.Bar(
                x=mctr_sorted.index.tolist(),
                y=(mctr_sorted * 100).tolist(),
                marker_color=mctr_colors,
                text=[f"{v*100:.2f}%" for v in mctr_sorted.values],
                textposition="outside",
            )
        )
    mctr_fig.update_layout(
        template="plotly_dark",
        height=300,
        title="תרומה שולית לסיכון (MCTR) — כמה כל סקטור תורם לתנודתיות הספר",
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis_title="MCTR (%)",
        showlegend=False,
    )

    # ── Risk budget bar chart ────────────────────────────────────────────────
    rb: pd.Series = getattr(rr, "risk_budget_series", pd.Series(dtype=float))
    rb_fig = go.Figure()
    if not rb.empty:
        rb_sorted = rb.sort_values(ascending=False)
        rb_fig.add_trace(
            go.Bar(
                x=rb_sorted.index.tolist(),
                y=(rb_sorted * 100).tolist(),
                marker_color="#0dcaf0",
                text=[f"{v*100:.1f}%" for v in rb_sorted.values],
                textposition="outside",
            )
        )
    rb_fig.update_layout(
        template="plotly_dark",
        height=300,
        title="תקציב סיכון (Risk Budget) — % מהסיכון הכולל לפי סקטור",
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis_title="% מהסיכון",
        showlegend=False,
    )

    # ── Factor VaR decomposition ─────────────────────────────────────────────
    factor_d: dict = {}
    try:
        from analytics.portfolio_risk import PortfolioRiskEngine
        # factor_var_decomp already run inside full_risk_report; access stored attrs if present
        factor_d = getattr(rr, "_factor_decomp", {})
    except Exception:
        pass

    factor_section: Any = html.Div()
    if factor_d:
        sys_pnl  = factor_d.get("systematic_var_1d", float("nan"))
        idio_pnl = factor_d.get("idiosyncratic_var_1d", float("nan"))
        factor_section = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("פירוק VaR לגורמים", style={"textAlign": "right"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.Div("VaR שיטתי (Factor)", className="text-muted small text-center"),
                                 html.Div(_pct(sys_pnl), className="h5 text-center fw-bold text-warning")],
                            ),
                            dbc.Col(
                                [html.Div("VaR אידיוסינקרטי", className="text-muted small text-center"),
                                 html.Div(_pct(idio_pnl), className="h5 text-center fw-bold text-info")],
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-3",
        )

    return html.Div(
        [
            html.Div(breach_alerts),
            kpi_row,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=mctr_fig, config={"displayModeBar": False}))), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=rb_fig,   config={"displayModeBar": False}))), md=6),
                ],
                className="mb-3 g-2",
            ),
            factor_section,
        ],
        className="mt-3",
    )


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_backtest_tab(backtest_result: Optional[Any]) -> html.Div:
    """Full walk-forward backtest tab."""

    if backtest_result is None:
        return html.Div(
            [
                dbc.Alert(
                    "הבאקטסט טרם הורץ. לחץ על 'הרץ Backtest' (תהליך ארוך — כ-2-3 דקות).",
                    color="secondary",
                    className="mt-3",
                ),
                dbc.Button("הרץ Backtest", id="run-backtest-btn", color="primary", className="mt-2"),
                html.Div(id="backtest-status", className="mt-2"),
                html.Div(id="backtest-output"),
            ],
            className="mt-3",
        )

    br = backtest_result

    # ── KPI row ──────────────────────────────────────────────────────────────
    ic_mean  = getattr(br, "ic_mean",  float("nan"))
    ic_ir    = getattr(br, "ic_ir",    float("nan"))
    hit_rate = getattr(br, "hit_rate", float("nan"))
    sharpe   = getattr(br, "sharpe",   float("nan"))
    max_dd   = getattr(br, "max_drawdown", float("nan"))
    n_walks  = getattr(br, "n_walks",  0)

    ic_color   = "success" if ic_mean > 0.05 else "warning" if ic_mean > 0 else "danger"
    hit_color  = "success" if hit_rate > 0.55 else "warning" if hit_rate > 0.5 else "danger"
    sh_color   = "success" if sharpe > 1.0 else "warning" if sharpe > 0.5 else "danger"

    kpi_row = dbc.Row(
        [
            _kpi("IC ממוצע",           _ff(ic_mean,  "{:.3f}"), ic_color),
            _kpi("IC IR (IC/StdIC)",    _ff(ic_ir,    "{:.2f}"), "primary"),
            _kpi("Hit Rate",            _pct(hit_rate),          hit_color),
            _kpi("Sharpe (אות)",        _ff(sharpe,   "{:.2f}"), sh_color),
            _kpi("Max Drawdown",        _pct(max_dd),            "danger"),
            _kpi("חלונות שנבדקו",      str(n_walks),             "secondary", small=True),
        ],
        className="g-2 mb-3",
    )

    # ── Walk-Forward Equity Curve ─────────────────────────────────────────────
    eq_fig = go.Figure()
    walk_metrics = getattr(br, "walk_metrics", [])
    if walk_metrics:
        try:
            dates_eq = [w.test_start for w in walk_metrics]
            returns_eq = [w.signal_return for w in walk_metrics]
            cum_eq = np.cumprod(1 + np.array(returns_eq, dtype=float))
            eq_fig.add_trace(go.Scatter(
                x=dates_eq, y=cum_eq,
                mode="lines", name="Strategy (signal-weighted)",
                line=dict(color="#0dcaf0", width=2),
                fill="tozeroy", fillcolor="rgba(13,202,240,0.08)",
            ))
            # Drawdown shading
            running_max = np.maximum.accumulate(cum_eq)
            dd_eq = (cum_eq - running_max) / running_max
            eq_fig.add_trace(go.Scatter(
                x=dates_eq, y=dd_eq,
                mode="lines", name="Drawdown",
                line=dict(color="#dc3545", width=1),
                fill="tozeroy", fillcolor="rgba(220,53,69,0.12)",
                yaxis="y2",
            ))
        except Exception:
            pass
    eq_fig.add_hline(y=1.0, line_dash="dot", line_color="#555", opacity=0.5)
    eq_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=340,
        title=dict(text="Walk-Forward Equity Curve — עקומת שווי מצטבר", font=dict(size=13)),
        margin=dict(l=50, r=50, t=45, b=35),
        yaxis=dict(title="Growth of $1", tickformat=".2f"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%",
                    showgrid=False, range=[-0.5, 0]),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
    )

    # ── Standalone Drawdown Chart ─────────────────────────────────────────────
    fig_dd = go.Figure()
    if walk_metrics:
        try:
            returns_dd = [w.signal_return for w in walk_metrics]
            dates_dd = [w.test_start for w in walk_metrics]
            equity_dd = np.cumprod(1 + np.array(returns_dd, dtype=float))
            peak_dd = np.maximum.accumulate(equity_dd)
            drawdown_dd = (equity_dd - peak_dd) / np.where(peak_dd > 0, peak_dd, 1.0) * 100
            fig_dd.add_trace(go.Scatter(
                x=dates_dd, y=drawdown_dd,
                fill="tozeroy", fillcolor="rgba(231,76,60,0.3)",
                line=dict(color="#e74c3c", width=1),
                name="Drawdown %",
            ))
        except Exception:
            pass
    fig_dd.update_layout(
        template="plotly_dark", height=250,
        title=dict(text="Strategy Drawdown — שפל מצטבר", font=dict(size=13)),
        yaxis_title="Drawdown %",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        margin=dict(l=50, r=50, t=45, b=35),
    )

    # ── Regime Performance Heatmap ────────────────────────────────────────────
    regime_hm_section = html.Div()
    regime_bd = getattr(br, "regime_breakdown", None)
    if regime_bd is not None:
        try:
            reg_labels, reg_ic, reg_hit, reg_avg_ret = [], [], [], []
            for reg in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
                rd = getattr(regime_bd, reg.lower(), None) or (regime_bd.get(reg) if isinstance(regime_bd, dict) else None)
                if rd is None:
                    continue
                reg_labels.append(reg)
                _get = lambda obj, k, d=0: getattr(obj, k, obj.get(k, d) if isinstance(obj, dict) else d)
                reg_ic.append(float(_get(rd, "ic_mean", 0)))
                reg_hit.append(float(_get(rd, "hit_rate", 0)))
                reg_avg_ret.append(float(_get(rd, "avg_return", _get(rd, "signal_return", 0))))
            if reg_labels:
                metric_names = ["IC Mean", "Hit Rate", "Avg Return"]
                z_data = [reg_ic, reg_hit, reg_avg_ret]
                fig_regime_hm = go.Figure(go.Heatmap(
                    z=z_data, x=reg_labels, y=metric_names,
                    colorscale="RdYlGn", texttemplate="%{z:.3f}",
                    textfont=dict(size=11),
                    showscale=True,
                ))
                fig_regime_hm.update_layout(
                    template="plotly_dark", height=220,
                    title=dict(text="ביצועים לפי משטר — Regime Heatmap", font=dict(size=13)),
                    paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                    margin=dict(l=80, r=30, t=45, b=30),
                )
                regime_hm_section = dbc.Card(
                    dbc.CardBody(dcc.Graph(figure=fig_regime_hm, config={"displayModeBar": False})),
                    className="border-0 bg-transparent mb-3",
                )
        except Exception:
            pass

    # ── IC time series ───────────────────────────────────────────────────────
    ic_series: pd.Series = getattr(br, "ic_series", pd.Series(dtype=float))
    ic_fig = go.Figure()
    if not ic_series.empty:
        ic_fig.add_trace(
            go.Scatter(
                x=ic_series.index.tolist(),
                y=ic_series.values.tolist(),
                mode="lines",
                name="IC",
                line=dict(color="#0dcaf0", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(13,202,240,0.08)",
            )
        )
        # Rolling mean
        ic_roll = ic_series.rolling(8, min_periods=4).mean()
        ic_fig.add_trace(
            go.Scatter(
                x=ic_roll.index.tolist(),
                y=ic_roll.values.tolist(),
                mode="lines",
                name="IC ממוצע גולל (8 walks)",
                line=dict(color="#ffc107", width=2),
            )
        )
    ic_fig.add_hline(y=0, line_color="#555", line_width=1)
    ic_fig.add_hline(y=0.05, line_dash="dash", line_color="#20c997", annotation_text="IC=0.05 (טוב)")
    ic_fig.update_layout(
        template="plotly_dark",
        height=320,
        title="IC לאורך זמן (Spearman Rank Correlation: אות vs תשואה קדימה)",
        margin=dict(l=10, r=10, t=50, b=30),
    )

    # ── Regime breakdown ─────────────────────────────────────────────────────
    regime_bd = getattr(br, "regime_breakdown", None)
    regime_fig = go.Figure()
    if regime_bd is not None:
        try:
            regimes = list(regime_bd.__dict__.keys()) if hasattr(regime_bd, "__dict__") else []
            labels, ic_vals, hit_vals, sharpe_vals = [], [], [], []
            for reg in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
                rd = getattr(regime_bd, reg.lower(), None) or regime_bd.get(reg) if isinstance(regime_bd, dict) else None
                if rd is None:
                    continue
                labels.append(reg)
                ic_vals.append(getattr(rd, "ic_mean", rd.get("ic_mean", 0) if isinstance(rd, dict) else 0))
                hit_vals.append(getattr(rd, "hit_rate", rd.get("hit_rate", 0) if isinstance(rd, dict) else 0))
                sharpe_vals.append(getattr(rd, "sharpe", rd.get("sharpe", 0) if isinstance(rd, dict) else 0))

            if labels:
                regime_colors = {"CALM": "#20c997", "NORMAL": "#0d6efd", "TENSION": "#ffc107", "CRISIS": "#dc3545"}
                colors = [regime_colors.get(l, "#6c757d") for l in labels]
                regime_fig.add_trace(go.Bar(x=labels, y=ic_vals,    name="IC ממוצע",   marker_color=colors, offsetgroup=0))
                regime_fig.add_trace(go.Bar(x=labels, y=sharpe_vals, name="Sharpe",    marker_color=colors, offsetgroup=1, opacity=0.6))
        except Exception:
            pass

    regime_fig.update_layout(
        template="plotly_dark",
        height=300,
        title="ביצועי האות לפי משטר (Regime Breakdown)",
        margin=dict(l=10, r=10, t=50, b=30),
        barmode="group",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df: pd.DataFrame = getattr(br, "summary_df", pd.DataFrame())
    summary_section: Any = html.Div()
    if not summary_df.empty:
        from dash.dash_table import DataTable
        cols_to_show = [c for c in summary_df.columns if c in [
            "walk_id", "period_start", "period_end", "ic", "hit_rate", "sharpe", "n_signals", "regime"
        ]]
        if cols_to_show:
            summary_section = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("פירוט Walk-Forward", style={"textAlign": "right"}),
                        DataTable(
                            columns=[{"name": c, "id": c} for c in cols_to_show],
                            data=summary_df[cols_to_show].round(3).to_dict("records"),
                            page_size=15,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"backgroundColor": "#111", "color": "#eee", "fontSize": "11px", "padding": "5px", "textAlign": "center"},
                            style_header={"backgroundColor": "#1a1a2e", "fontWeight": "bold", "fontSize": "11px"},
                            style_data_conditional=[
                                {"if": {"filter_query": "{ic} > 0.05"}, "color": "#20c997"},
                                {"if": {"filter_query": "{ic} < 0"},    "color": "#dc3545"},
                            ],
                        ),
                    ]
                ),
                className="mt-3",
            )

    return html.Div(
        [
            kpi_row,
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=eq_fig, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent mb-3"),
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_dd, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent mb-3"),
            regime_hm_section,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=ic_fig,     config={"displayModeBar": False}))), md=7),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=regime_fig, config={"displayModeBar": False}))), md=5),
                ],
                className="mb-3 g-2",
            ),
            summary_section,
        ],
        className="mt-3",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAILY BRIEF PANEL (modal body)
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_brief_panel(brief_txt: str) -> html.Div:
    """Render the daily brief text inside a modal."""
    if not brief_txt:
        return html.Div("לא נמצא brief.", className="text-muted")  # noqa: E501 (preserved for context)
    lines = brief_txt.split("\n")
    rendered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("==="):
            rendered.append(html.H5(stripped.replace("=", "").strip(), className="mt-2 text-warning"))
        elif stripped.startswith("---"):
            rendered.append(html.Hr(style={"borderColor": "#333"}))
        elif stripped.startswith("[") or stripped.startswith("TOP") or stripped.startswith("RISK") or stripped.startswith("STRESS"):
            rendered.append(html.Strong(stripped, style={"display": "block", "marginTop": "6px", "fontSize": "12px"}))
        elif stripped:
            rendered.append(html.Div(stripped, style={"fontSize": "11px", "color": "#ccc", "paddingLeft": "8px"}))
        else:
            rendered.append(html.Br())
    return html.Div(rendered, style={"fontFamily": "monospace", "textAlign": "left", "direction": "ltr"})


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION VOLATILITY TAB
# ─────────────────────────────────────────────────────────────────────────────
def build_corr_vol_tab(analysis: Any) -> html.Div:
    """
    Build the Correlation & Volatility Pricing tab.

    Parameters
    ----------
    analysis : CorrVolAnalysis | None
    """
    import plotly.express as px

    if analysis is None:
        return html.Div("Correlation analysis not available.", className="text-muted p-3")

    def _sf(v, fmt="{:.3f}"):
        try:
            f = float(v)
            return "—" if f != f else fmt.format(f)
        except Exception:
            return "—"

    # ── Signal gauge color ────────────────────────────────────────────────────
    score = analysis.short_vol_score or 0
    gauge_color = (
        "success" if score >= 65
        else "warning" if score >= 35
        else "danger" if score < 20
        else "info"
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi_row = dbc.Row([
        _kpi("Avg Corr (60d)",     _sf(analysis.avg_corr_current),  "primary", small=True),
        _kpi("Avg Corr (252d)",    _sf(analysis.avg_corr_baseline), "secondary", small=True),
        _kpi("Implied Corr",       _sf(analysis.implied_corr),      "info", small=True),
        _kpi("Fair Value Corr",    _sf(analysis.fair_value_corr),   "secondary", small=True),
        _kpi("Corr Risk Premium",  _sf(analysis.corr_risk_premium,  "{:+.3f}"),
             "danger" if (analysis.corr_risk_premium or 0) > 0.02 else "success", small=True),
        _kpi("Dispersion Index",   f"{analysis.dispersion_index:.1f}%" if analysis.dispersion_index == analysis.dispersion_index else "—",
             "info", small=True),
    ], className="g-2 mb-3")

    # ── Short vol signal card ─────────────────────────────────────────────────
    signal_card = dbc.Card([
        dbc.CardHeader(html.H6("🎯 Short Volatility / Dispersion Trade Signal", className="mb-0 text-center")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(f"{score:.0f} / 100", className="display-4 fw-bold text-center",
                             style={"color": f"var(--bs-{gauge_color})"}),
                    html.Div(analysis.short_vol_label, className="h5 text-center mt-1 fw-bold"),
                    dbc.Progress(value=score, color=gauge_color, style={"height": "12px"}, className="mt-2"),
                ], width=4),
                dbc.Col([
                    html.H6("בסיס הsignal:", className="text-muted mb-1"),
                    html.Pre(
                        analysis.short_vol_rationale or "—",
                        style={"fontSize": "11px", "whiteSpace": "pre-wrap", "color": "#ccc",
                               "background": "transparent", "border": "none"},
                    ),
                ], width=8),
            ]),
            html.Hr(style={"borderColor": "#444"}),
            html.Div([
                html.Span("Corr Regime: ", className="text-muted"),
                html.Strong(analysis.corr_regime, className="text-warning me-3"),
                html.Span("Market Mode Strength: ", className="text-muted"),
                html.Strong(f"{analysis.market_mode_strength:.3f}" if analysis.market_mode_strength == analysis.market_mode_strength else "—"),
                html.Span(" | Eigenvalue HHI: ", className="text-muted ms-2"),
                html.Strong(f"{analysis.eigenvalue_concentration:.3f}" if analysis.eigenvalue_concentration == analysis.eigenvalue_concentration else "—"),
            ], style={"fontSize": "12px"}),
        ]),
    ], className=f"border-{gauge_color} mb-3", style={"borderTop": f"3px solid var(--bs-{gauge_color})"})

    # ── Implied corr vs fair value time series ────────────────────────────────
    fig_impl = go.Figure()
    if analysis.implied_corr_ts is not None and not analysis.implied_corr_ts.empty:
        ts = analysis.implied_corr_ts.dropna()
        fv = analysis.fair_value_ts.dropna() if analysis.fair_value_ts is not None else None
        crp = analysis.crp_ts.dropna() if analysis.crp_ts is not None else None

        fig_impl.add_trace(go.Scatter(
            x=ts.index, y=ts.values,
            name="Implied Corr", line=dict(color="#4da6ff", width=1.5),
        ))
        if fv is not None and not fv.empty:
            fig_impl.add_trace(go.Scatter(
                x=fv.index, y=fv.values,
                name="Fair Value (EWMA)", line=dict(color="#ffd700", width=1.5, dash="dash"),
            ))
        if crp is not None and not crp.empty:
            fig_impl.add_trace(go.Bar(
                x=crp.index, y=crp.values,
                name="Corr Risk Premium",
                marker_color=["#ff4444" if v > 0 else "#44aa44" for v in crp.values],
                opacity=0.6, yaxis="y2",
            ))
        fig_impl.update_layout(
            template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            margin=dict(l=40, r=20, t=30, b=30), height=260,
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(title="Correlation", tickformat=".2f"),
            yaxis2=dict(title="CRP", overlaying="y", side="right", tickformat=".3f"),
            title=dict(text="Implied Correlation vs Fair Value + Risk Premium", font=dict(size=12)),
        )
    else:
        fig_impl.add_annotation(text="נתונים לא זמינים", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_impl.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", height=260)

    # ── Avg correlation time series ───────────────────────────────────────────
    fig_avg = go.Figure()
    if analysis.avg_corr_ts is not None and not analysis.avg_corr_ts.empty:
        ts = analysis.avg_corr_ts.dropna()
        fig_avg.add_trace(go.Scatter(
            x=ts.index, y=ts.values,
            name="Avg Corr (rolling)", fill="tozeroy",
            line=dict(color="#4da6ff", width=1.5),
            fillcolor="rgba(77,166,255,0.15)",
        ))
        # Regime threshold lines
        fig_avg.add_hline(y=0.45, line_dash="dot", line_color="#44cc44", annotation_text="CALM threshold")
        fig_avg.add_hline(y=0.60, line_dash="dot", line_color="#ffaa00", annotation_text="TENSION threshold")
        fig_avg.add_hline(y=0.75, line_dash="dot", line_color="#ff4444", annotation_text="CRISIS threshold")
        # Baseline
        if analysis.avg_corr_baseline == analysis.avg_corr_baseline:
            fig_avg.add_hline(y=analysis.avg_corr_baseline, line_dash="dash",
                              line_color="#aaaaaa", annotation_text="252d baseline")
        fig_avg.update_layout(
            template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            margin=dict(l=40, r=20, t=30, b=30), height=220,
            yaxis=dict(title="Avg Pairwise Correlation", tickformat=".2f", range=[0, 1]),
            title=dict(text="Average Cross-Sector Correlation — Rolling History", font=dict(size=12)),
        )
    else:
        fig_avg.add_annotation(text="נתונים לא זמינים", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_avg.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", height=220)

    # ── Correlation heatmaps ──────────────────────────────────────────────────
    def _heatmap(df, title, colorscale, zmid=None):
        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(text="N/A", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", height=280)
            return fig
        z = df.values.round(3).tolist()
        labels = list(df.columns)
        kw = dict(zmid=zmid) if zmid is not None else {}
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=colorscale,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=True,
            **kw,
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            margin=dict(l=5, r=5, t=30, b=5), height=280,
            title=dict(text=title, font=dict(size=11)),
        )
        return fig

    fig_ct   = _heatmap(analysis.corr_current,  "C_t — קורלציה נוכחית (60d)",    "RdYlGn")
    fig_cb   = _heatmap(analysis.corr_baseline,  "C_b — קורלציה Baseline (252d)",  "RdYlGn")
    fig_cdel = _heatmap(analysis.corr_delta,     "ΔC = C_t − C_b  (עיוות)",       "RdBu",   zmid=0)

    # ── Anomalous pairs table ─────────────────────────────────────────────────
    pairs = analysis.anomalous_pairs[:8]
    pairs_rows = [
        html.Tr([
            html.Td(f"{p['sector_a']}–{p['sector_b']}", style={"fontFamily": "monospace", "fontSize": "11px"}),
            html.Td(p["name_a"][:12], style={"fontSize": "10px", "color": "#aaa"}),
            html.Td(p["name_b"][:12], style={"fontSize": "10px", "color": "#aaa"}),
            html.Td(f"{p['corr_current']:.3f}", style={"textAlign": "right"}),
            html.Td(f"{p['corr_baseline']:.3f}", style={"textAlign": "right"}),
            html.Td(
                f"{p['delta']:+.3f}",
                style={"textAlign": "right", "fontWeight": "bold",
                       "color": "#ff6666" if p["delta"] > 0 else "#66ff66"},
            ),
            html.Td(p["direction"], style={"fontSize": "10px", "color": "#aaa"}),
        ]) for p in pairs
    ]
    pairs_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("זוג"),
                html.Th("סקטור A"),
                html.Th("סקטור B"),
                html.Th("Corr נוכחי"),
                html.Th("Corr Baseline"),
                html.Th("Δ"),
                html.Th("כיוון"),
            ])),
            html.Tbody(pairs_rows),
        ],
        bordered=True, dark=True, hover=True, size="sm",
        style={"fontSize": "11px"},
    )

    # ── Market mode loadings bar ──────────────────────────────────────────────
    loadings = analysis.market_mode_loadings or {}
    if loadings:
        sorted_load = sorted(loadings.items(), key=lambda x: x[1], reverse=True)
        tickers_l = [x[0] for x in sorted_load]
        values_l  = [x[1] for x in sorted_load]
        fig_mode = go.Figure(go.Bar(
            x=tickers_l, y=values_l,
            marker_color=["#ff6666" if v < 0 else "#4da6ff" for v in values_l],
            text=[f"{v:.3f}" for v in values_l],
            textposition="outside",
        ))
        fig_mode.update_layout(
            template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            margin=dict(l=20, r=20, t=30, b=20), height=200,
            yaxis=dict(range=[-1, 1], title="PC1 Loading"),
            title=dict(text=f"Market Mode (PC1) Loadings | Strength={analysis.market_mode_strength:.3f}", font=dict(size=11)),
        )
    else:
        fig_mode = go.Figure()
        fig_mode.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", height=200)

    # ── Layout assembly ───────────────────────────────────────────────────────
    return html.Div([
        kpi_row,
        signal_card,

        # Implied corr vs fair value
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_impl, config={"displayModeBar": False})),
                 className="border-0 bg-transparent mb-3"),

        # Avg corr history
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_avg, config={"displayModeBar": False})),
                 className="border-0 bg-transparent mb-3"),

        # Heatmaps row — with rolling window selector
        dbc.Row([
            dbc.Col(
                html.H6("מטריצות קורלציה", className="text-muted mb-0 mt-1"),
                width="auto",
            ),
            dbc.Col(
                dbc.Select(
                    id="corr-window-select",
                    options=[
                        {"label": "30 ימים",  "value": "30"},
                        {"label": "60 ימים",  "value": "60"},
                        {"label": "90 ימים",  "value": "90"},
                        {"label": "120 ימים", "value": "120"},
                    ],
                    value="60",
                    style={"width": "160px", "backgroundColor": "#1a1a2e",
                           "color": "#fff", "border": "1px solid #333",
                           "fontSize": "12px"},
                ),
                width="auto",
            ),
            dbc.Col(
                html.Small(
                    "* חלון גלילה — callback wiring נדרש ב-main.py",
                    className="text-muted",
                    style={"fontSize": "10px", "lineHeight": "2.2"},
                ),
                width="auto",
            ),
        ], align="center", className="mb-2 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="corr-heatmap-current",  figure=fig_ct,   config={"displayModeBar": False}), width=4),
            dbc.Col(dcc.Graph(id="corr-heatmap-baseline", figure=fig_cb,   config={"displayModeBar": False}), width=4),
            dbc.Col(dcc.Graph(id="corr-heatmap-delta",    figure=fig_cdel, config={"displayModeBar": False}), width=4),
        ], className="mb-3"),

        # Market mode + anomalous pairs
        dbc.Row([
            dbc.Col([
                html.H6("Market Mode — PC1 Loadings", className="text-muted mb-2"),
                dcc.Graph(figure=fig_mode, config={"displayModeBar": False}),
            ], width=5),
            dbc.Col([
                html.H6("עיוותי קורלציה — זוגות חריגים", className="text-muted mb-2"),
                pairs_table,
            ], width=7),
        ]),
    ], style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────
# LIVE P&L TRACKER TAB
# ─────────────────────────────────────────────────────────────────────────

def build_pnl_tracker_tab(pnl_result: Any = None) -> html.Div:
    """Build Live P&L Tracker tab content."""
    if pnl_result is None:
        return html.Div(
            dbc.Alert("מעקב P&L לא זמין. הרץ את הדשבורד עם נתונים.", color="info"),
            style={"padding": "20px"},
        )

    import math as _m

    # ── KPI row ──────────────────────────────────────────────────────────
    total_color = "success" if pnl_result.total_pnl > 0 else "danger"
    sharpe_color = "success" if (pnl_result.sharpe > 0.5 if _m.isfinite(pnl_result.sharpe) else False) else "warning"

    kpi_row = dbc.Row([
        _kpi("P&L כולל", _pct(pnl_result.total_pnl), total_color),
        _kpi("Sharpe", _ff(pnl_result.sharpe), sharpe_color),
        _kpi("Max Drawdown", _pct(pnl_result.max_drawdown), "danger"),
        _kpi("Hit Rate", _pct(pnl_result.hit_rate), "primary"),
        _kpi("ימי מסחר", str(pnl_result.n_trading_days), "secondary"),
    ], className="mb-3 g-2")

    # ── Secondary KPIs ───────────────────────────────────────────────────
    calmar_str = _ff(pnl_result.calmar) if _m.isfinite(pnl_result.calmar) else "—"
    kpi_row2 = dbc.Row([
        _kpi("Calmar", calmar_str, "info", small=True),
        _kpi("P&L יומי ממוצע", f"{pnl_result.avg_daily_pnl*10000:.1f}bps", "info", small=True),
        _kpi("תנודתיות P&L", f"{pnl_result.pnl_volatility*10000:.0f}bps", "warning", small=True),
        _kpi("ימי רווח", f"{pnl_result.win_days}", "success", small=True),
        _kpi("ימי הפסד", f"{pnl_result.loss_days}", "danger", small=True),
        _kpi("יום הכי טוב", f"{pnl_result.best_day*10000:.0f}bps", "success", small=True),
        _kpi("יום הכי גרוע", f"{pnl_result.worst_day*10000:.0f}bps", "danger", small=True),
    ], className="mb-3 g-2")

    # ── Cumulative P&L chart ─────────────────────────────────────────────
    fig_cum = go.Figure()
    cum = pnl_result.cumulative_pnl
    if cum is not None and not cum.empty:
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            mode="lines", name="Cumulative P&L",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)",
        ))
    fig_cum.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cum.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        title=dict(text="P&L מצטבר", font=dict(size=13)),
        yaxis=dict(title="P&L (%)", ticksuffix="%"),
        height=320, margin=dict(l=60, r=20, t=40, b=40),
    )

    # ── Drawdown chart ───────────────────────────────────────────────────
    fig_dd = go.Figure()
    dd = pnl_result.drawdown
    if dd is not None and not dd.empty:
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            mode="lines", name="Drawdown",
            line=dict(color="#f44336", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(244, 67, 54, 0.15)",
        ))
    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        title=dict(text="Drawdown", font=dict(size=13)),
        yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
        height=220, margin=dict(l=60, r=20, t=40, b=30),
    )

    # ── Rolling Sharpe chart ─────────────────────────────────────────────
    fig_sharpe = go.Figure()
    rs = pnl_result.rolling_sharpe
    if rs is not None and not rs.empty:
        fig_sharpe.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            mode="lines", name="Rolling Sharpe (63d)",
            line=dict(color="#ffa726", width=1.5),
        ))
    fig_sharpe.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_sharpe.add_hline(y=1, line_dash="dash", line_color="#4caf50", opacity=0.3,
                         annotation_text="Sharpe = 1")
    fig_sharpe.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        title=dict(text="Rolling Sharpe (63 ימים)", font=dict(size=13)),
        yaxis=dict(title="Sharpe Ratio"),
        height=250, margin=dict(l=60, r=20, t=40, b=30),
    )

    # ── Sector contribution stacked area ─────────────────────────────────
    fig_sector = go.Figure()
    sc = pnl_result.sector_contribution
    if sc is not None and not sc.empty:
        cum_contrib = sc.cumsum()
        sector_colors = [
            "#e91e63", "#9c27b0", "#673ab7", "#3f51b5", "#2196f3",
            "#00bcd4", "#009688", "#4caf50", "#8bc34a", "#ffeb3b", "#ff9800",
        ]
        for i, col in enumerate(cum_contrib.columns):
            fig_sector.add_trace(go.Scatter(
                x=cum_contrib.index,
                y=cum_contrib[col].values * 100,
                mode="lines", name=col,
                line=dict(width=1.2, color=sector_colors[i % len(sector_colors)]),
            ))
    fig_sector.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        title=dict(text="תרומת P&L מצטברת לפי סקטור", font=dict(size=13)),
        yaxis=dict(title="P&L (%)", ticksuffix="%"),
        height=350, margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(font=dict(size=9), x=0.01, y=0.99),
    )

    # ── Regime P&L breakdown ─────────────────────────────────────────────
    regime_colors = {"CALM": "#4caf50", "NORMAL": "#2196f3", "TENSION": "#ff9800", "CRISIS": "#f44336"}
    regime_heb = {"CALM": "רגוע", "NORMAL": "רגיל", "TENSION": "מתח", "CRISIS": "משבר"}
    regime_rows = []
    for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
        rp = (pnl_result.regime_pnl or {}).get(regime)
        if rp is None:
            continue
        bg = regime_colors.get(regime, "#333")
        regime_rows.append(html.Tr([
            html.Td(html.Span(
                regime_heb.get(regime, regime),
                style={"backgroundColor": bg, "color": "white",
                       "padding": "2px 8px", "borderRadius": "4px", "fontSize": "12px"},
            )),
            html.Td(f"{rp.n_days}", className="text-center"),
            html.Td(f"{rp.total_pnl * 100:+.2f}%", className="text-center",
                     style={"color": "#4caf50" if rp.total_pnl > 0 else "#f44336"}),
            html.Td(f"{rp.avg_daily_pnl * 10000:+.1f}bps", className="text-center"),
            html.Td(_ff(rp.sharpe), className="text-center"),
            html.Td(f"{rp.hit_rate:.0%}", className="text-center"),
            html.Td(f"{rp.max_drawdown * 100:.2f}%", className="text-center"),
        ]))

    regime_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("רגים"), html.Th("ימים", className="text-center"),
            html.Th("P&L", className="text-center"),
            html.Th("ממוצע יומי", className="text-center"),
            html.Th("Sharpe", className="text-center"),
            html.Th("Hit Rate", className="text-center"),
            html.Th("Max DD", className="text-center"),
        ])),
        html.Tbody(regime_rows),
    ], bordered=True, dark=True, hover=True, size="sm", className="mb-3")

    # ── Sector P&L table ─────────────────────────────────────────────────
    sector_rows = []
    for _, row in (pnl_result.summary_df if pnl_result.summary_df is not None and not pnl_result.summary_df.empty else pd.DataFrame()).iterrows():
        pnl_val = row.get("total_pnl", 0)
        color = "#4caf50" if pnl_val > 0 else "#f44336"
        sector_rows.append(html.Tr([
            html.Td(row.get("sector", ""), style={"fontWeight": "bold"}),
            html.Td(f"{pnl_val:+.2f}%", style={"color": color}, className="text-center"),
            html.Td(f"{row.get('hit_rate', 0):.0f}%", className="text-center"),
            html.Td(_ff(row.get('sharpe')), className="text-center"),
            html.Td(f"{row.get('max_dd', 0):.2f}%", className="text-center"),
            html.Td(f"{row.get('n_days', 0)}", className="text-center"),
        ]))

    sector_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("סקטור"), html.Th("P&L", className="text-center"),
            html.Th("Hit Rate", className="text-center"),
            html.Th("Sharpe", className="text-center"),
            html.Th("Max DD", className="text-center"),
            html.Th("ימים פעילים", className="text-center"),
        ])),
        html.Tbody(sector_rows),
    ], bordered=True, dark=True, hover=True, size="sm")

    # ── Monthly returns heatmap ──────────────────────────────────────────
    fig_monthly = go.Figure()
    mr = pnl_result.monthly_returns
    if mr is not None and not mr.empty:
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        display_cols = [c for c in mr.columns if c != "Total"]
        z = mr[display_cols].values * 100
        x_labels = [month_names.get(c, str(c)) for c in display_cols]

        fig_monthly = go.Figure(data=go.Heatmap(
            z=z,
            x=x_labels,
            y=[str(y) for y in mr.index],
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(z, 2),
            texttemplate="%{text:.1f}",
            hovertemplate="Year: %{y}<br>Month: %{x}<br>P&L: %{z:.2f}%<extra></extra>",
        ))
    fig_monthly.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        title=dict(text="תשואות חודשיות (%)", font=dict(size=13)),
        height=max(200, 40 * len(mr) + 80) if mr is not None and not mr.empty else 200,
        margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )

    # ── Assembly ─────────────────────────────────────────────────────────
    return html.Div([
        html.H5("💰 Live P&L Tracker", className="mb-3"),
        kpi_row,
        kpi_row2,
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_cum, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=8),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_dd, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=4),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_sharpe, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=6),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_monthly, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=6),
        ], className="mb-2"),
        dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_sector, config={"displayModeBar": False}),
        ), className="border-0 bg-transparent mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6("ביצועים לפי רגים", className="text-muted mb-2"),
                regime_table,
            ], width=6),
            dbc.Col([
                html.H6("ביצועים לפי סקטור", className="text-muted mb-2"),
                sector_table,
            ], width=6),
        ]),
    ], style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────
# SIGNAL DECAY TAB
# ─────────────────────────────────────────────────────────────────────────

def build_signal_decay_tab(decay_result: Any = None) -> html.Div:
    """Build Signal Decay Analysis tab content."""
    if decay_result is None:
        return html.Div(
            dbc.Alert("אנליזת Signal Decay לא זמינה. הרץ את הפייפליין עם --backtest.", color="info"),
            style={"padding": "20px"},
        )

    # ── KPI row ──────────────────────────────────────────────────────────
    opt_h = decay_result.optimal_horizon
    opt_label = {1: "1D", 5: "1W", 10: "2W", 21: "1M", 42: "2M", 63: "3M"}.get(opt_h, f"{opt_h}d")
    kpi_row = dbc.Row([
        _kpi("אופק מיטבי", opt_label, "success"),
        _kpi("IC באופק מיטבי", _ff(decay_result.optimal_ic, "{:.4f}"), "primary"),
        _kpi("Turnover שנתי", _ff(decay_result.annualised_turnover, "{:.1f}"), "warning"),
        _kpi("עלות bps/שנה", _ff(decay_result.estimated_cost_bps_pa, "{:.0f}"), "danger"),
    ], className="mb-3 g-2")

    # ── IC Decay Curve ───────────────────────────────────────────────────
    summary = decay_result.summary_df
    fig_decay = go.Figure()
    if summary is not None and not summary.empty:
        fig_decay.add_trace(go.Scatter(
            x=summary["horizon_days"],
            y=summary["ic_mean"],
            mode="lines+markers",
            name="IC Mean",
            line=dict(color="#00d4ff", width=3),
            marker=dict(size=10),
        ))
        if "ic_median" in summary.columns:
            fig_decay.add_trace(go.Scatter(
                x=summary["horizon_days"],
                y=summary["ic_median"],
                mode="lines+markers",
                name="IC Median",
                line=dict(color="#ffa726", width=2, dash="dash"),
                marker=dict(size=7),
            ))
        # Zero reference line
        fig_decay.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig_decay.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text="IC Decay Curve — כוח הסיגנל לאורך זמן", font=dict(size=13)),
        xaxis=dict(
            title="Forward Horizon (ימי מסחר)",
            tickvals=[1, 5, 10, 21, 42, 63],
            ticktext=["1D", "1W", "2W", "1M", "2M", "3M"],
        ),
        yaxis=dict(title="Spearman IC"),
        height=360,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(x=0.7, y=0.95),
    )

    # ── Regime-conditional decay curves ──────────────────────────────────
    fig_regime = go.Figure()
    regime_colors = {"CALM": "#4caf50", "NORMAL": "#2196f3", "TENSION": "#ff9800", "CRISIS": "#f44336"}
    horizons_list = [1, 5, 10, 21, 42, 63]

    if decay_result.regime_decay:
        for regime, rd in decay_result.regime_decay.items():
            ics = [rd.ic_by_horizon.get(h, None) for h in horizons_list]
            valid_h = [h for h, ic in zip(horizons_list, ics) if ic is not None]
            valid_ic = [ic for ic in ics if ic is not None]
            if valid_ic:
                fig_regime.add_trace(go.Scatter(
                    x=valid_h,
                    y=valid_ic,
                    mode="lines+markers",
                    name=f"{regime} (n={rd.n_walks})",
                    line=dict(color=regime_colors.get(regime, "#888"), width=2),
                    marker=dict(size=7),
                ))

    fig_regime.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_regime.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text="IC Decay לפי רגים — התנהגות הסיגנל בתנאי שוק שונים", font=dict(size=13)),
        xaxis=dict(title="Forward Horizon", tickvals=horizons_list,
                   ticktext=["1D", "1W", "2W", "1M", "2M", "3M"]),
        yaxis=dict(title="Spearman IC"),
        height=360,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99),
    )

    # ── Sector × Horizon heatmap ─────────────────────────────────────────
    fig_heatmap = go.Figure()
    hm = decay_result.heatmap_df
    if hm is not None and not hm.empty:
        horizon_cols = [c for c in ["1D", "1W", "2W", "1M", "2M", "3M"] if c in hm.columns]
        z = hm[horizon_cols].values.astype(float) if horizon_cols else np.array([])
        if z.size:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=z,
                x=horizon_cols,
                y=hm["sector"].tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(z, 4),
                texttemplate="%{text:.4f}",
                hovertemplate="Sector: %{y}<br>Horizon: %{x}<br>IC: %{z:.4f}<extra></extra>",
            ))

    fig_heatmap.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text="Signal Decay Heatmap — IC לפי סקטור ואופק", font=dict(size=13)),
        height=400,
        margin=dict(l=80, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )

    # ── Sector optimal holding + turnover table ──────────────────────────
    sector_rows = []
    for sec, sd in (decay_result.sector_decay or {}).items():
        sec_label = sec.replace("sector_", "")
        opt_lbl = {1: "1D", 5: "1W", 10: "2W", 21: "1M", 42: "2M", 63: "3M"}.get(sd.optimal_horizon, f"{sd.optimal_horizon}d")
        sector_rows.append(html.Tr([
            html.Td(sec_label, style={"fontWeight": "bold"}),
            html.Td(opt_lbl, className="text-center"),
            html.Td(f"{sd.half_life_ic:.0f}d" if sd.half_life_ic else "—", className="text-center"),
            html.Td(_ff(sd.avg_turnover, "{:.3f}"), className="text-center"),
            html.Td(_ff(sd.turnover_cost_bps, "{:.0f}"), className="text-center"),
        ]))

    sector_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("סקטור"),
                html.Th("אופק מיטבי", className="text-center"),
                html.Th("IC Half-Life", className="text-center"),
                html.Th("Turnover (ממוצע)", className="text-center"),
                html.Th("עלות bps/שנה", className="text-center"),
            ])),
            html.Tbody(sector_rows),
        ],
        bordered=True, dark=True, hover=True, size="sm",
        className="mb-3",
    )

    # ── Assembly ─────────────────────────────────────────────────────────
    return html.Div([
        html.H5("📉 Signal Decay Analysis", className="mb-3"),
        kpi_row,
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_decay, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=6),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_regime, config={"displayModeBar": False}),
            ), className="border-0 bg-transparent"), width=6),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_heatmap, config={"displayModeBar": False}),
        ), className="border-0 bg-transparent mb-3"),
        html.H6("פרופיל Decay לפי סקטור", className="text-muted mb-2"),
        sector_table,
    ], style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────
# REGIME TIMELINE TAB
# ─────────────────────────────────────────────────────────────────────────

def build_regime_timeline_tab(regime_result: Any = None) -> html.Div:
    """Build Regime Transition Alerts + Timeline tab content."""
    if regime_result is None:
        return html.Div(
            dbc.Alert("אנליזת רגימים לא זמינה.", color="info"),
            style={"padding": "20px"},
        )

    regime_colors_bg = {
        "CALM": "#1b5e20", "NORMAL": "#0d47a1",
        "TENSION": "#e65100", "CRISIS": "#b71c1c",
    }
    regime_heb = {
        "CALM": "שוק רגוע", "NORMAL": "שוק רגיל",
        "TENSION": "מתח בשוק", "CRISIS": "משבר",
    }

    # ── Current regime hero ──────────────────────────────────────────────
    snap = regime_result.current_snapshot
    curr = regime_result.current_regime
    hero_color = regime_colors_bg.get(curr, "#333")

    hero = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H3(regime_heb.get(curr, curr), className="mb-1 text-white"),
                html.Div(f"הסתברות משבר: {snap.crisis_probability:.0%}" if snap else "",
                         className="text-white-50"),
            ], width=4),
            dbc.Col([
                dbc.Row([
                    _kpi("VIX", _ff(snap.vol_score, "{:.0%}") if snap else "—", "light", small=True),
                    _kpi("אשראי", _ff(snap.credit_score, "{:.0%}") if snap else "—", "light", small=True),
                    _kpi("קורלציה", _ff(snap.corr_score, "{:.0%}") if snap else "—", "light", small=True),
                    _kpi("מעבר", _ff(snap.transition_score, "{:.0%}") if snap else "—", "light", small=True),
                ], className="g-2"),
            ], width=8),
        ]),
    ]), style={"backgroundColor": hero_color, "borderRadius": "8px"}, className="mb-3")

    # ── Active alerts ────────────────────────────────────────────────────
    alert_cards = []
    level_colors = {"CRITICAL": "danger", "WARNING": "warning", "INFO": "info"}
    for alert in (regime_result.active_alerts or []):
        color = level_colors.get(alert.get("level", "INFO"), "info")
        alert_cards.append(dbc.Alert([
            html.Strong(f"[{alert.get('level', '')}] "),
            html.Span(alert.get("message", "")),
            html.Small(f"  ({alert.get('date', '')})", className="text-muted ms-2"),
        ], color=color, className="py-2 mb-1"))

    if not alert_cards:
        alert_cards = [dbc.Alert("אין התראות פעילות", color="success", className="py-2")]

    # ── Component scores time series ─────────────────────────────────────
    df = regime_result.snapshot_history
    fig_scores = go.Figure()
    if df is not None and not df.empty:
        for col, name, color in [
            ("vol_score", "VIX Score", "#f44336"),
            ("credit_score", "Credit Score", "#ff9800"),
            ("corr_score", "Correlation Score", "#2196f3"),
            ("transition_score", "Transition Score", "#9c27b0"),
            ("crisis_probability", "Crisis Probability", "#e91e63"),
        ]:
            if col in df.columns:
                fig_scores.add_trace(go.Scatter(
                    x=df["date"], y=df[col],
                    mode="lines", name=name,
                    line=dict(color=color, width=1.5),
                ))

    fig_scores.add_hline(y=0.78, line_dash="dash", line_color="#f44336",
                         annotation_text="סף משבר", opacity=0.5)
    fig_scores.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text="ציוני רכיבים לאורך זמן", font=dict(size=13)),
        yaxis=dict(title="Score (0-1)", range=[-0.05, 1.05]),
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
    )

    # ── Regime timeline bar ──────────────────────────────────────────────
    fig_timeline = go.Figure()
    regime_colors_plot = {"CALM": "#4caf50", "NORMAL": "#2196f3", "TENSION": "#ff9800", "CRISIS": "#f44336"}

    if regime_result.timeline:
        for entry in regime_result.timeline:
            fig_timeline.add_trace(go.Bar(
                x=[(entry.end_date - entry.start_date).days or 1],
                y=["Regime"],
                base=[entry.start_date],
                orientation="h",
                name=entry.regime,
                marker_color=regime_colors_plot.get(entry.regime, "#888"),
                showlegend=False,
                hovertemplate=(
                    f"{regime_heb.get(entry.regime, entry.regime)}<br>"
                    f"מ-{entry.start_date.strftime('%Y-%m-%d')} עד {entry.end_date.strftime('%Y-%m-%d')}<br>"
                    f"משך: {entry.duration_days} ימים<br>"
                    f"VIX ממוצע: {entry.avg_vix:.1f}<extra></extra>"
                ),
            ))

    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text="ציר זמן רגימים", font=dict(size=13)),
        barmode="stack",
        height=120,
        margin=dict(l=60, r=20, t=40, b=20),
        yaxis=dict(visible=False),
        xaxis=dict(type="date"),
    )

    # ── Regime statistics table ──────────────────────────────────────────
    stat_rows = []
    for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
        rs = (regime_result.regime_stats or {}).get(regime, {})
        bg = regime_colors_bg.get(regime, "#333")
        stat_rows.append(html.Tr([
            html.Td(
                html.Span(regime_heb.get(regime, regime),
                          style={"backgroundColor": bg, "color": "white",
                                 "padding": "2px 8px", "borderRadius": "4px", "fontSize": "12px"}),
            ),
            html.Td(f"{rs.get('count', 0)}", className="text-center"),
            html.Td(f"{rs.get('pct_time', 0):.0%}", className="text-center"),
            html.Td(f"{rs.get('avg_duration', 0):.0f}", className="text-center"),
            html.Td(f"{rs.get('max_duration', 0):.0f}", className="text-center"),
        ]))

    stats_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("רגים"), html.Th("מספר תקופות", className="text-center"),
                html.Th("% מהזמן", className="text-center"),
                html.Th("משך ממוצע (ימים)", className="text-center"),
                html.Th("משך מקסימלי", className="text-center"),
            ])),
            html.Tbody(stat_rows),
        ],
        bordered=True, dark=True, hover=True, size="sm", className="mb-3",
    )

    # ── Transition history table ─────────────────────────────────────────
    trans_rows = []
    for t in reversed(regime_result.transitions[-20:]):
        level_badge = {
            "CRITICAL": "danger", "WARNING": "warning", "INFO": "info",
        }.get(t.level.value, "secondary")
        trans_rows.append(html.Tr([
            html.Td(str(t.date.date()) if hasattr(t.date, "date") else str(t.date)),
            html.Td(dbc.Badge(t.level.value, color=level_badge)),
            html.Td(f"{t.from_regime} → {t.to_regime}"),
            html.Td(t.message, style={"fontSize": "11px", "direction": "rtl"}),
        ]))

    if not trans_rows:
        trans_rows = [html.Tr(html.Td("אין מעברי רגים בתקופה", colSpan=4, className="text-muted text-center"))]

    trans_table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("תאריך"), html.Th("רמה"),
                html.Th("מעבר"), html.Th("פירוט"),
            ])),
            html.Tbody(trans_rows),
        ],
        bordered=True, dark=True, hover=True, size="sm",
    )

    # ── Assembly ─────────────────────────────────────────────────────────
    return html.Div([
        html.H5("🔔 מעברי רגים והתראות", className="mb-3"),
        hero,
        html.Div(alert_cards, className="mb-3"),
        dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_timeline, config={"displayModeBar": False}),
        ), className="border-0 bg-transparent mb-3"),
        dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_scores, config={"displayModeBar": False}),
        ), className="border-0 bg-transparent mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6("סטטיסטיקת רגימים", className="text-muted mb-2"),
                stats_table,
            ], width=5),
            dbc.Col([
                html.H6("היסטוריית מעברים", className="text-muted mb-2"),
                trans_table,
            ], width=7),
        ]),
    ], style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────
# DSS — DECISION SUPPORT SYSTEM TAB
# Short Vol / Dispersion Trade DSS
# ─────────────────────────────────────────────────────────────────────────

_SAFETY_BADGE = {
    "SAFE": "success", "CAUTION": "warning", "DANGER": "danger", "KILLED": "danger",
}
_SAFETY_HEB = {
    "SAFE": "בטוח לטריידינג",
    "CAUTION": "זהירות — צמצם גדלים",
    "DANGER": "סכנה — גדלים מינימליים",
    "KILLED": "חסום — אין מסחר",
}
_DIR_BADGE_DSS = {"LONG": "success", "SHORT": "danger", "NEUTRAL": "secondary", "LONG_DISPERSION": "info"}


def build_dss_tab(
    signal_results: Optional[List] = None,
    trade_tickets: Optional[List] = None,
    regime_safety: Any = None,
    corr_snapshot: Any = None,
    monitor_summary: Any = None,
    options_surface: Any = None,
    tail_risk_es: Any = None,
    methodology_ranking: Optional[List[Dict]] = None,
    paper_portfolio: Optional[Dict] = None,
    dispersion_result: Any = None,
) -> html.Div:
    """
    Build the Decision Support System tab:
    - Regime Safety gauge + alerts
    - Signal Stack conviction table (all 4 layers)
    - Trade Book with legs, Greeks, exit conditions
    - Correlation distortion visual
    """

    if signal_results is None and trade_tickets is None:
        return html.Div(
            dbc.Alert(
                "מערכת תומכת החלטה לא זמינה — הרץ את הPipeline כדי לייצר אותות.",
                color="info",
            ),
            style={"padding": "20px"},
        )

    sections = []

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: Regime Safety Banner
    # ══════════════════════════════════════════════════════════════════════
    if regime_safety is not None:
        rs = regime_safety
        badge_color = _SAFETY_BADGE.get(rs.label, "secondary")
        safety_pct = rs.regime_safety_score * 100

        penalty_items = []
        for name, val in [("VIX", rs.vix_penalty), ("Credit", rs.credit_penalty),
                          ("Correlation", rs.corr_penalty), ("Transition", rs.transition_penalty)]:
            bar_color = "success" if val < 0.3 else "warning" if val < 0.6 else "danger"
            penalty_items.append(
                dbc.Col([
                    html.Div(name, className="text-muted", style={"fontSize": "10px", "textAlign": "center"}),
                    dbc.Progress(value=val * 100, color=bar_color, style={"height": "8px"}, className="mt-1"),
                    html.Div(f"{val:.0%}", className="text-center", style={"fontSize": "10px"}),
                ], width=3)
            )

        alert_items = []
        for a in getattr(rs, "alerts", []):
            a_color = "danger" if "HARD KILL" in a else "warning"
            alert_items.append(dbc.Alert(a, color=a_color, className="py-1 px-2 mb-1",
                                          style={"fontSize": "11px"}))

        safety_section = dbc.Card([
            dbc.CardHeader(html.H6("🛡️ Regime Safety — Gate Layer 4", className="mb-0 text-center")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{safety_pct:.0f}%", className="display-4 fw-bold text-center",
                                 style={"color": f"var(--bs-{badge_color})"}),
                        html.Div(_SAFETY_HEB.get(rs.label, rs.label), className="h6 text-center"),
                        dbc.Badge(rs.market_state, color=_SAFETY_BADGE.get(rs.label, "secondary"),
                                  className="d-block mx-auto mt-1",
                                  style={"fontSize": "11px", "width": "fit-content"}),
                        dbc.Progress(value=safety_pct, color=badge_color,
                                     style={"height": "12px"}, className="mt-2"),
                    ], width=3),
                    dbc.Col([
                        html.H6("Penalty Breakdown", className="text-muted mb-2", style={"fontSize": "11px"}),
                        dbc.Row(penalty_items, className="g-2"),
                        html.Div(f"Size Cap: {rs.size_cap:.0%}", className="text-center mt-2 fw-bold",
                                 style={"fontSize": "12px", "color": f"var(--bs-{badge_color})"}),
                    ], width=5),
                    dbc.Col(alert_items or [html.Div("אין התראות", className="text-muted",
                                                      style={"fontSize": "11px"})], width=4),
                ]),
            ]),
        ], className=f"border-{badge_color} mb-3",
           style={"borderTop": f"3px solid var(--bs-{badge_color})"})
        sections.append(safety_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: Signal Stack — Conviction Table
    # ══════════════════════════════════════════════════════════════════════
    if signal_results:
        n_total = len(signal_results)
        n_pass = sum(1 for r in signal_results if r.passes_entry)
        dist_score = signal_results[0].distortion_score if signal_results else 0
        top = signal_results[0] if signal_results else None

        kpi_row = dbc.Row([
            _kpi("Candidates", str(n_total), "primary", small=True),
            _kpi("Passing Entry", str(n_pass), "success" if n_pass > 0 else "secondary", small=True),
            _kpi("Distortion", f"{dist_score:.0%}", "warning" if dist_score > 0.6 else "info", small=True),
            _kpi("Top Conviction",
                 f"{top.conviction_score:.2f}" if top else "—",
                 "success" if top and top.conviction_score > 0.3 else "info", small=True),
        ], className="g-2 mb-3")

        table_rows = []
        for r in signal_results[:15]:
            dir_badge = _DIR_BADGE_DSS.get(r.direction, "secondary")
            entry_icon = "✅" if r.passes_entry else "❌"

            def _mini_bar(val, color):
                return html.Div([
                    dbc.Progress(value=val * 100, color=color,
                                 style={"height": "6px", "width": "60px", "display": "inline-block"}),
                    html.Span(f" {val:.2f}", style={"fontSize": "10px"}),
                ])

            table_rows.append(html.Tr([
                html.Td(r.ticker, style={"fontWeight": "bold", "fontSize": "12px"}),
                html.Td(dbc.Badge(r.direction, color=dir_badge, style={"fontSize": "10px"})),
                html.Td(f"{r.residual_z:+.2f}", style={"fontSize": "11px", "fontFamily": "monospace"}),
                html.Td(_mini_bar(r.distortion_score, "info")),
                html.Td(_mini_bar(r.dislocation_score, "warning")),
                html.Td(_mini_bar(r.mean_reversion_score,
                                  "success" if r.mean_reversion_score > 0.5 else "danger")),
                html.Td(_mini_bar(r.regime_safety_score,
                                  "success" if r.regime_safety_score > 0.5 else "danger")),
                html.Td(html.Span(
                    f"{r.conviction_score:.3f}",
                    style={"fontWeight": "bold", "fontSize": "12px",
                           "color": "var(--bs-success)" if r.conviction_score > 0.3
                           else "var(--bs-warning)" if r.conviction_score > 0.15
                           else "var(--bs-secondary)"})),
                html.Td(entry_icon, style={"textAlign": "center"}),
            ]))

        signal_table = dbc.Card([
            dbc.CardHeader(html.H6("📊 Signal Stack — 4-Layer Conviction Scoring", className="mb-0 text-center")),
            dbc.CardBody([
                kpi_row,
                html.Div(
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Ticker", style={"fontSize": "10px"}),
                            html.Th("Dir", style={"fontSize": "10px"}),
                            html.Th("Z", style={"fontSize": "10px"}),
                            html.Th("L1: Distortion", style={"fontSize": "10px"}),
                            html.Th("L2: Dislocation", style={"fontSize": "10px"}),
                            html.Th("L3: Mean Rev", style={"fontSize": "10px"}),
                            html.Th("L4: Safety", style={"fontSize": "10px"}),
                            html.Th("Conviction", style={"fontSize": "10px"}),
                            html.Th("Entry", style={"fontSize": "10px"}),
                        ])),
                        html.Tbody(table_rows),
                    ], bordered=True, hover=True, responsive=True, size="sm",
                       className="mb-0", style={"fontSize": "11px"}),
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
            ]),
        ], className="border-primary mb-3", style={"borderTop": "3px solid var(--bs-primary)"})
        sections.append(signal_table)

        # ── Signal Z-Score Horizontal Bar Chart ──────────────────────────
        try:
            z_tickers = [r.ticker for r in signal_results[:15]]
            z_vals = [r.residual_z for r in signal_results[:15]]
            z_colors = ['#00bc8c' if z < 0 else '#e74c3c' for z in z_vals]
            z_fig = go.Figure(go.Bar(
                x=z_vals, y=z_tickers,
                orientation='h',
                marker_color=z_colors,
                text=[f"{z:+.2f}" for z in z_vals],
                textposition="outside",
                textfont=dict(size=10),
            ))
            z_fig.add_vline(x=0, line_color="#555", line_width=1)
            z_fig.add_vline(x=2, line_dash="dash", line_color="#ffc107", opacity=0.6,
                            annotation_text="+2σ", annotation_position="top")
            z_fig.add_vline(x=-2, line_dash="dash", line_color="#ffc107", opacity=0.6,
                            annotation_text="-2σ", annotation_position="top")
            z_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                height=max(250, 28 * len(z_tickers) + 80),
                title=dict(text="Sector Z-Scores — PCA Residual", font=dict(size=13)),
                margin=dict(l=70, r=40, t=45, b=30),
                xaxis=dict(title="Z-Score (σ)", zeroline=True),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            z_chart_card = dbc.Card([
                dbc.CardHeader(html.H6("📊 Z-Score Map — סטיית תמחור לפי סקטור",
                                        className="mb-0 text-center")),
                dbc.CardBody(dcc.Graph(figure=z_fig, config={"displayModeBar": False})),
            ], className="border-info mb-3", style={"borderTop": "3px solid var(--bs-info)"})
            sections.append(z_chart_card)
        except Exception:
            pass  # graceful fallback if z-score data is missing

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: Trade Book
    # ══════════════════════════════════════════════════════════════════════
    if trade_tickets:
        active_tickets = [t for t in trade_tickets if t.is_active]
        gross = sum(t.final_weight for t in active_tickets)
        g_spy = sum(t.greeks.delta_spy for t in active_tickets)
        g_vega = sum(t.greeks.vega_net for t in active_tickets)
        g_gamma = sum(t.greeks.gamma_net for t in active_tickets)
        g_rho_corr = sum(t.greeks.rho_corr for t in active_tickets)

        greeks_row = dbc.Row([
            _kpi("Gross", f"{gross:.1%}", "primary", small=True),
            _kpi("Δ SPY", f"{g_spy:+.4f}", "success" if abs(g_spy) < 0.05 else "warning", small=True),
            _kpi("Vega", f"{g_vega:+.4f}", "info", small=True),
            _kpi("Γ synth", f"{g_gamma:.4f}", "secondary", small=True),
            _kpi("ρ corr", f"{g_rho_corr:+.4f}", "warning" if g_rho_corr < -0.1 else "info", small=True),
        ], className="g-2 mb-3")

        trade_cards = []
        for t in active_tickets[:8]:
            dir_color = _DIR_BADGE_DSS.get(t.direction, "secondary")
            type_emoji = {"sector_rv": "📈", "dispersion": "🔀", "rv_spread": "⚖️"}.get(t.trade_type, "📊")

            legs_list = html.Ul([
                html.Li(leg.description, style={"fontSize": "10px", "fontFamily": "monospace"})
                for leg in t.legs
            ], style={"paddingRight": "15px", "marginBottom": "4px"})

            trade_cards.append(dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Span(f"{type_emoji} ", style={"fontSize": "14px"}),
                        html.Strong(t.ticker, style={"fontSize": "13px"}),
                        dbc.Badge(t.direction, color=dir_color, className="ms-2", style={"fontSize": "9px"}),
                        html.Span(f" | conv={t.conviction_score:.2f}", className="text-muted",
                                  style={"fontSize": "10px"}),
                    ], style={"padding": "6px 10px"}),
                    dbc.CardBody([
                        html.Div(f"Weight: {t.final_weight:.2%}",
                                 style={"fontSize": "11px", "fontWeight": "bold"}),
                        html.Div(f"Z: {t.entry_z:+.2f} | HL: {t.half_life_est:.0f}d",
                                 style={"fontSize": "10px"}, className="text-muted"),
                        html.Hr(style={"margin": "4px 0", "borderColor": "#444"}),
                        html.Div("Legs:", style={"fontSize": "10px", "fontWeight": "bold"}),
                        legs_list,
                        html.Div(t.exit_conditions.description,
                                 style={"fontSize": "9px", "color": "#888"}),
                    ], style={"padding": "8px 10px"}),
                ], className=f"border-{dir_color} h-100",
                   style={"borderLeft": f"3px solid var(--bs-{dir_color})"}),
                width=6, className="mb-2",
            ))

        trade_section = dbc.Card([
            dbc.CardHeader(html.H6("📋 Trade Book — Active Positions", className="mb-0 text-center")),
            dbc.CardBody([
                greeks_row,
                dbc.Row(trade_cards),
                html.Div(
                    f"סה\"כ {len(active_tickets)} טריידים פעילים מתוך {len(trade_tickets)} candidates",
                    className="text-muted text-center mt-2", style={"fontSize": "11px"}),
            ]),
        ], className="border-success mb-3", style={"borderTop": "3px solid var(--bs-success)"})
        sections.append(trade_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: Correlation Distortion Summary
    # ══════════════════════════════════════════════════════════════════════
    if corr_snapshot is not None:
        cs = corr_snapshot
        corr_kpis = dbc.Row([
            _kpi("Frob Distortion Z", _ff(cs.frob_distortion_z, "{:+.2f}"),
                 "danger" if abs(getattr(cs, "frob_distortion_z", 0) or 0) > 2 else "info", small=True),
            _kpi("Market Mode Share", _ff(cs.market_mode_share, "{:.1%}"),
                 "warning" if (getattr(cs, "market_mode_share", 0) or 0) > 0.5 else "info", small=True),
            _kpi("Avg Corr Current", _ff(cs.avg_corr_current, "{:.3f}"), "primary", small=True),
            _kpi("CoC Instability Z", _ff(cs.coc_instability_z, "{:+.2f}"),
                 "danger" if abs(getattr(cs, "coc_instability_z", 0) or 0) > 1.5 else "info", small=True),
        ], className="g-2")

        corr_section = dbc.Card([
            dbc.CardHeader(html.H6("🔗 Correlation Structure — Distortion Metrics",
                                    className="mb-0 text-center")),
            dbc.CardBody(corr_kpis),
        ], className="border-info mb-3", style={"borderTop": "3px solid var(--bs-info)"})
        sections.append(corr_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: Trade Monitor — Health & Exit Signals
    # ══════════════════════════════════════════════════════════════════════
    _HEALTH_BADGE = {"HEALTHY": "success", "AGING": "info", "AT_RISK": "warning", "CRITICAL": "danger"}
    _HEALTH_HEB = {"HEALTHY": "בריא", "AGING": "מתבגר", "AT_RISK": "בסיכון", "CRITICAL": "קריטי"}

    if monitor_summary is not None:
        ms = monitor_summary

        health_kpis = dbc.Row([
            _kpi("Trades", str(ms.n_trades), "primary", small=True),
            _kpi("Healthy", str(ms.n_healthy), "success", small=True),
            _kpi("At Risk", str(ms.n_at_risk + ms.n_critical),
                 "danger" if ms.n_critical > 0 else "warning" if ms.n_at_risk > 0 else "success", small=True),
            _kpi("Exit Signals", str(ms.n_exit_signals),
                 "danger" if ms.n_exit_signals > 0 else "success", small=True),
            _kpi("Avg Health", f"{ms.avg_health:.0%}",
                 "success" if ms.avg_health >= 0.6 else "warning" if ms.avg_health >= 0.3 else "danger",
                 small=True),
        ], className="g-2 mb-3")

        # Individual trade health rows
        monitor_rows = []
        for r in ms.trade_reports:
            h_color = _HEALTH_BADGE.get(r.health_label, "secondary")
            primary = r.primary_signal

            monitor_rows.append(html.Tr([
                html.Td(r.ticker, style={"fontWeight": "bold", "fontSize": "12px"}),
                html.Td(dbc.Badge(r.health_label, color=h_color, style={"fontSize": "9px"})),
                html.Td(f"{r.current_z:+.2f}", style={"fontSize": "11px", "fontFamily": "monospace"}),
                html.Td(html.Div([
                    dbc.Progress(value=r.z_compression_pct * 100, color="success",
                                 style={"height": "6px", "width": "50px", "display": "inline-block"}),
                    html.Span(f" {r.z_compression_pct:.0%}", style={"fontSize": "10px"}),
                ])),
                html.Td(html.Div([
                    dbc.Progress(value=r.time_decay_pct * 100,
                                 color="danger" if r.time_decay_pct > 0.8 else "warning" if r.time_decay_pct > 0.5 else "success",
                                 style={"height": "6px", "width": "50px", "display": "inline-block"}),
                    html.Span(f" {r.days_held}d", style={"fontSize": "10px"}),
                ])),
                html.Td(f"{r.health_score:.0%}", style={"fontSize": "11px", "fontWeight": "bold",
                         "color": f"var(--bs-{h_color})"}),
                html.Td(
                    dbc.Badge(r.recommended_action, color="danger" if "CLOSE" in r.recommended_action
                              else "warning" if "REDUCE" in r.recommended_action else "secondary",
                              style={"fontSize": "9px"})
                ),
                html.Td(
                    html.Span(primary.reason[:50] + "..." if primary and len(primary.reason) > 50
                              else primary.reason if primary else "—",
                              style={"fontSize": "9px", "color": "#999"}),
                ),
            ]))

        # Urgent exits highlight
        urgent_alerts = []
        for r in ms.urgent_exits[:3]:
            ps = r.primary_signal
            urgent_alerts.append(dbc.Alert([
                html.Strong(f"⚠️ {r.ticker} — {r.recommended_action}: "),
                html.Span(ps.reason if ps else ""),
                dbc.Badge(ps.urgency if ps else "", color="danger", className="ms-2",
                          style={"fontSize": "9px"}),
            ], color="danger", className="py-1 px-2 mb-1", style={"fontSize": "11px"}))

        monitor_section = dbc.Card([
            dbc.CardHeader(html.H6("🔔 Trade Monitor — Health & Exit Signals", className="mb-0 text-center")),
            dbc.CardBody([
                health_kpis,
                html.Div(urgent_alerts) if urgent_alerts else html.Div(),
                html.Div(
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Ticker", style={"fontSize": "10px"}),
                            html.Th("Health", style={"fontSize": "10px"}),
                            html.Th("Z Now", style={"fontSize": "10px"}),
                            html.Th("Z Compress", style={"fontSize": "10px"}),
                            html.Th("Time", style={"fontSize": "10px"}),
                            html.Th("Score", style={"fontSize": "10px"}),
                            html.Th("Action", style={"fontSize": "10px"}),
                            html.Th("Signal", style={"fontSize": "10px"}),
                        ])),
                        html.Tbody(monitor_rows),
                    ], bordered=True, hover=True, responsive=True, size="sm",
                       className="mb-0", style={"fontSize": "11px"}),
                    style={"maxHeight": "300px", "overflowY": "auto"},
                ),
            ]),
        ], className=f"border-{'danger' if ms.n_exit_signals > 0 else 'success'} mb-3",
           style={"borderTop": f"3px solid var(--bs-{'danger' if ms.n_exit_signals > 0 else 'success'})"})
        sections.append(monitor_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: Options Analytics
    # ══════════════════════════════════════════════════════════════════════
    if options_surface is not None:
        os_ = options_surface
        vix_val = getattr(os_, "vix_current", None)
        term_slope = getattr(os_, "term_slope", None)
        impl_corr = getattr(os_, "implied_corr", None)
        disp_idx = getattr(os_, "dispersion_index", None)
        vrp_idx = getattr(os_, "vrp_index", None)

        slope_label = "Contango" if (term_slope or 0) > 0 else "Backwardation"
        slope_color = "success" if (term_slope or 0) > 0 else "danger"

        opt_kpis = dbc.Row([
            _kpi("VIX", f"{vix_val:.1f}" if vix_val is not None else "—",
                 "danger" if (vix_val or 0) > 25 else "warning" if (vix_val or 0) > 18 else "success", small=True),
            _kpi("Term Slope", f"{term_slope:+.3f} ({slope_label})" if term_slope is not None else "—",
                 slope_color, small=True),
            _kpi("Implied Corr", f"{impl_corr:.3f}" if impl_corr is not None else "—", "info", small=True),
            _kpi("Dispersion Idx", f"{disp_idx:.2f}%" if disp_idx is not None else "—", "primary", small=True),
            _kpi("VRP Index", f"{vrp_idx:+.4f}" if vrp_idx is not None else "—",
                 "success" if (vrp_idx or 0) > 0 else "danger", small=True),
        ], className="g-2 mb-3")

        # Per-sector Greeks table
        sector_greeks = getattr(os_, "sector_greeks", {})
        opt_rows = []
        if sector_greeks:
            sorted_sectors = sorted(sector_greeks.items(),
                                    key=lambda kv: getattr(kv[1], "iv", 0), reverse=True)
            for sec_name, g in sorted_sectors:
                iv = getattr(g, "iv", 0)
                rv_20 = getattr(g, "rv_20d", 0)
                rv_60 = getattr(g, "rv_60d", 0)
                vrp = getattr(g, "vrp", 0)
                theta = getattr(g, "theta", 0)
                iv_rank = getattr(g, "iv_rank_252d", 0)
                rank_pct = min(max(iv_rank * 100, 0), 100)
                rank_color = "danger" if iv_rank > 0.7 else "warning" if iv_rank > 0.4 else "success"
                opt_rows.append(html.Tr([
                    html.Td(sec_name, style={"fontSize": "11px", "fontWeight": "bold"}),
                    html.Td(f"{iv:.1f}%", style={"fontSize": "11px"}),
                    html.Td(f"{rv_20:.1f}%", style={"fontSize": "11px"}),
                    html.Td(f"{vrp:+.1f}%", style={"fontSize": "11px",
                             "color": "var(--bs-success)" if vrp > 0 else "var(--bs-danger)"}),
                    html.Td(f"{theta:.4f}", style={"fontSize": "11px"}),
                    html.Td(
                        dbc.Progress(value=rank_pct, color=rank_color,
                                     style={"height": "10px", "minWidth": "60px"},
                                     className="my-0"),
                        style={"fontSize": "10px", "verticalAlign": "middle"},
                    ),
                ]))

        opt_table = html.Div(
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("סקטור", style={"fontSize": "10px"}),
                    html.Th("IV", style={"fontSize": "10px"}),
                    html.Th("RV 20d", style={"fontSize": "10px"}),
                    html.Th("VRP", style={"fontSize": "10px"}),
                    html.Th("Theta", style={"fontSize": "10px"}),
                    html.Th("IV Rank", style={"fontSize": "10px"}),
                ])),
                html.Tbody(opt_rows),
            ], bordered=True, hover=True, responsive=True, size="sm",
               className="mb-0", style={"fontSize": "11px"}),
            style={"maxHeight": "300px", "overflowY": "auto"},
        ) if opt_rows else html.Div()

        options_section = dbc.Card([
            dbc.CardHeader(html.H6("📈 Options Analytics — IV / Greeks / VRP", className="mb-0 text-center")),
            dbc.CardBody([opt_kpis, opt_table]),
        ], className="border-info mb-3", style={"borderTop": "3px solid var(--bs-info)"})
        sections.append(options_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: Tail Risk
    # ══════════════════════════════════════════════════════════════════════
    if tail_risk_es is not None:
        tr = tail_risk_es
        es_pct = getattr(tr, "es_pct", None)
        var_pct = getattr(tr, "var_pct", None)
        es_var_ratio = getattr(tr, "es_to_var_ratio", None)
        skew = getattr(tr, "skewness", None)
        kurt = getattr(tr, "kurtosis", None)

        ratio_color = "danger" if (es_var_ratio or 0) > 2.0 else "warning" if (es_var_ratio or 0) > 1.5 else "success"

        tail_kpis = dbc.Row([
            _kpi("ES (97.5%)", f"{es_pct:.2%}" if es_pct is not None else "—", "danger", small=True),
            _kpi("VaR", f"{var_pct:.2%}" if var_pct is not None else "—", "warning", small=True),
            _kpi("ES/VaR Ratio", f"{es_var_ratio:.2f}" if es_var_ratio is not None else "—", ratio_color, small=True),
            _kpi("Skewness", f"{skew:.2f}" if skew is not None else "—", "info", small=True),
            _kpi("Excess Kurtosis", f"{kurt:.2f}" if kurt is not None else "—",
                 "danger" if (kurt or 0) > 3 else "info", small=True),
        ], className="g-2 mb-3")

        tail_note = html.Div()
        if (es_var_ratio or 0) > 2.0:
            tail_note = dbc.Alert(
                "⚠️ ES/VaR > 2.0 — זנבות שמנים קיצוניים, סיכון זנב גבוה מאוד",
                color="danger", className="py-1 px-2 mb-1", style={"fontSize": "11px"},
            )

        tail_section = dbc.Card([
            dbc.CardHeader(html.H6("🔻 Tail Risk — Expected Shortfall", className="mb-0 text-center")),
            dbc.CardBody([tail_kpis, tail_note]),
        ], className=f"border-{ratio_color} mb-3",
           style={"borderTop": f"3px solid var(--bs-{ratio_color})"})
        sections.append(tail_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: Methodology Lab Ranking
    # ══════════════════════════════════════════════════════════════════════
    if methodology_ranking:
        best_sharpe = max(m.get("sharpe", -999) for m in methodology_ranking)
        meth_rows = []
        for m in methodology_ranking[:8]:
            is_best = m.get("sharpe", -999) == best_sharpe
            row_style = {"fontSize": "11px", "backgroundColor": "rgba(25,135,84,0.15)"} if is_best else {"fontSize": "11px"}
            sharpe = m.get("sharpe", 0)
            wr = m.get("win_rate", 0)
            pnl = m.get("total_pnl", 0)
            dd = m.get("max_drawdown", 0)
            trades = m.get("total_trades", 0)
            meth_rows.append(html.Tr([
                html.Td([
                    html.Span(m.get("name", "—")),
                    dbc.Badge("BEST", color="success", className="ms-1", style={"fontSize": "8px"}) if is_best else html.Span(),
                ], style=row_style),
                html.Td(f"{sharpe:.2f}", style={**row_style,
                         "color": "var(--bs-success)" if sharpe > 0 else "var(--bs-danger)"}),
                html.Td(f"{wr:.0%}" if isinstance(wr, float) else str(wr), style=row_style),
                html.Td(f"${pnl:,.0f}" if isinstance(pnl, (int, float)) else str(pnl), style={**row_style,
                         "color": "var(--bs-success)" if (pnl or 0) > 0 else "var(--bs-danger)"}),
                html.Td(f"{dd:.1%}" if isinstance(dd, float) else str(dd), style=row_style),
                html.Td(str(trades), style=row_style),
            ]))

        meth_section = dbc.Card([
            dbc.CardHeader(html.H6("🧪 Methodology Lab — דירוג אסטרטגיות", className="mb-0 text-center")),
            dbc.CardBody(
                html.Div(
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("אסטרטגיה", style={"fontSize": "10px"}),
                            html.Th("Sharpe", style={"fontSize": "10px"}),
                            html.Th("Win Rate", style={"fontSize": "10px"}),
                            html.Th("P&L", style={"fontSize": "10px"}),
                            html.Th("Max DD", style={"fontSize": "10px"}),
                            html.Th("Trades", style={"fontSize": "10px"}),
                        ])),
                        html.Tbody(meth_rows),
                    ], bordered=True, hover=True, responsive=True, size="sm",
                       className="mb-0", style={"fontSize": "11px"}),
                    style={"maxHeight": "300px", "overflowY": "auto"},
                ),
            ),
        ], className="border-primary mb-3", style={"borderTop": "3px solid var(--bs-primary)"})
        sections.append(meth_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9: Paper Portfolio Summary
    # ══════════════════════════════════════════════════════════════════════
    if paper_portfolio:
        pp = paper_portfolio
        pp_capital = pp.get("capital", 0)
        pp_pnl = pp.get("total_pnl", 0)
        pp_pnl_pct = pp.get("total_pnl_pct", 0)
        pp_positions = pp.get("positions", [])
        pp_n_trades = pp.get("n_trades", 0)
        pp_wr = pp.get("win_rate", 0)

        pnl_color = "success" if pp_pnl >= 0 else "danger"

        pp_kpis = dbc.Row([
            _kpi("שווי תיק", f"${pp_capital:,.0f}", "primary", small=True),
            _kpi("P&L $", f"${pp_pnl:+,.0f}", pnl_color, small=True),
            _kpi("P&L %", f"{pp_pnl_pct:+.2f}%", pnl_color, small=True),
            _kpi("פוזיציות", str(len(pp_positions)), "info", small=True),
            _kpi("Win Rate", f"{pp_wr:.0%}" if isinstance(pp_wr, float) else str(pp_wr), "success", small=True),
        ], className="g-2 mb-3")

        pp_rows = []
        for pos in pp_positions:
            pos_pnl = pos.get("unrealized_pnl", 0)
            pos_pnl_pct = pos.get("unrealized_pnl_pct", 0)
            dir_badge = "success" if pos.get("direction", "").upper() == "LONG" else "danger"
            pp_rows.append(html.Tr([
                html.Td(pos.get("ticker", "—"), style={"fontSize": "11px", "fontWeight": "bold"}),
                html.Td(
                    dbc.Badge(pos.get("direction", "—"), color=dir_badge, style={"fontSize": "9px"}),
                    style={"fontSize": "11px"},
                ),
                html.Td(f"${pos.get('entry_price', 0):,.2f}", style={"fontSize": "11px"}),
                html.Td(f"${pos.get('current_price', 0):,.2f}", style={"fontSize": "11px"}),
                html.Td(f"${pos_pnl:+,.0f} ({pos_pnl_pct:+.1f}%)", style={"fontSize": "11px",
                         "color": "var(--bs-success)" if pos_pnl >= 0 else "var(--bs-danger)"}),
                html.Td(str(pos.get("days_held", 0)), style={"fontSize": "11px"}),
            ]))

        pp_table = html.Div(
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Ticker", style={"fontSize": "10px"}),
                    html.Th("כיוון", style={"fontSize": "10px"}),
                    html.Th("כניסה", style={"fontSize": "10px"}),
                    html.Th("נוכחי", style={"fontSize": "10px"}),
                    html.Th("P&L", style={"fontSize": "10px"}),
                    html.Th("ימים", style={"fontSize": "10px"}),
                ])),
                html.Tbody(pp_rows),
            ], bordered=True, hover=True, responsive=True, size="sm",
               className="mb-0", style={"fontSize": "11px"}),
            style={"maxHeight": "300px", "overflowY": "auto"},
        ) if pp_rows else html.Div("אין פוזיציות פתוחות", className="text-muted small")

        pp_section = dbc.Card([
            dbc.CardHeader(html.H6("📋 Paper Portfolio — תיק נייר", className="mb-0 text-center")),
            dbc.CardBody([pp_kpis, pp_table]),
        ], className=f"border-{pnl_color} mb-3",
           style={"borderTop": f"3px solid var(--bs-{pnl_color})"})
        sections.append(pp_section)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: Dispersion Backtest Performance
    # ══════════════════════════════════════════════════════════════════════
    if dispersion_result is not None:
        _rtl = {"direction": "rtl", "textAlign": "right"}

        # Build equity curve figure
        _eq = dispersion_result.equity_curve
        if _eq is not None and len(_eq) > 0:
            _eq_fig = go.Figure()
            _eq_fig.add_trace(go.Scatter(
                x=_eq.index, y=_eq.values,
                mode="lines", fill="tozeroy",
                line=dict(color="#00bc8c", width=1.5),
                fillcolor="rgba(0,188,140,0.15)",
                name="Equity",
            ))
            _eq_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                margin=dict(l=40, r=20, t=30, b=30),
                height=220,
                title=dict(text="Equity Curve — Dispersion Strategy", font=dict(size=12)),
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Cumulative P&L", showgrid=True, gridcolor="#333"),
            )
        else:
            _eq_fig = go.Figure()
            _eq_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                height=120,
                annotations=[dict(text="No equity data", showarrow=False,
                                  font=dict(color="#888", size=14), xref="paper", yref="paper", x=0.5, y=0.5)],
            )

        _disp_section = dbc.Card([
            dbc.CardBody([
                html.H6("Dispersion Trade Backtest", className="mb-3", style=_rtl),
                dbc.Row([
                    _kpi("Sharpe (OOS)", f"{dispersion_result.sharpe:.2f}", "success", small=True),
                    _kpi("Win Rate", f"{dispersion_result.win_rate:.0%}", "success", small=True),
                    _kpi("Total P&L", f"{dispersion_result.total_pnl:.1%}",
                         "success" if dispersion_result.total_pnl > 0 else "danger", small=True),
                    _kpi("Max DD", f"{dispersion_result.max_drawdown:.2%}", "danger", small=True),
                ], className="g-2 mb-3"),
                dcc.Graph(figure=_eq_fig, config={"displayModeBar": False}),
                html.Hr(style={"borderColor": "#444"}),
                html.Div("P&L Decomposition", className="text-muted small mb-2 text-center"),
                dbc.Row([
                    _kpi("Vega P&L", f"{dispersion_result.total_vega_pnl:.2%}", "info", small=True),
                    _kpi("Theta P&L", f"{dispersion_result.total_theta_pnl:.2%}", "secondary", small=True),
                    _kpi("Gamma P&L", f"{dispersion_result.total_gamma_pnl:.2%}", "warning", small=True),
                ], className="g-2"),
                html.Div(
                    f"{dispersion_result.total_trades} trades | "
                    f"Avg hold {dispersion_result.avg_holding_days:.0f}d | "
                    f"Calmar {dispersion_result.calmar:.1f}",
                    className="text-muted small mt-2 text-center",
                ),
            ])
        ], className="mb-3", style={"backgroundColor": "#2d2d44"})
        sections.append(_disp_section)

    if not sections:
        return html.Div(dbc.Alert("DSS: אין נתונים זמינים.", color="secondary"),
                         style={"padding": "20px"})

    return html.Div(sections, style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_tab(paper_portfolio: Optional[Dict] = None, prices: Optional[pd.DataFrame] = None) -> html.Div:
    """Full Paper Trading Portfolio tab layout."""
    if not paper_portfolio:
        return html.Div(
            dbc.Alert("אין נתוני Paper Portfolio זמינים. הרץ את ה-Paper Trader כדי לייצר.", color="secondary",
                      style={"textAlign": "right", "direction": "rtl"}),
            style={"padding": "20px"},
        )

    pp = paper_portfolio
    sections: List = []

    # ── 1. KPI Row ──────────────────────────────────────────────────────────
    capital = pp.get("capital", 0)
    cash = pp.get("cash", 0)
    total_pnl = pp.get("total_pnl", 0)
    total_pnl_pct = pp.get("total_pnl_pct", 0)
    max_dd = pp.get("max_drawdown", 0)
    positions = pp.get("positions", [])
    win_rate = pp.get("win_rate", 0)
    total_value = capital + total_pnl

    pnl_color = "success" if total_pnl >= 0 else "danger"

    kpi_row = dbc.Row(
        [
            _kpi("הון התחלתי", f"${capital:,.0f}", "info", small=True),
            _kpi("שווי כולל", f"${total_value:,.0f}", pnl_color, small=True),
            _kpi("P&L $", f"${total_pnl:+,.0f}", pnl_color, small=True),
            _kpi("P&L %", f"{total_pnl_pct * 100:+.2f}%", pnl_color, small=True),
            _kpi("מזומן", f"${cash:,.0f}", "secondary", small=True),
            _kpi("פוזיציות", str(len(positions)), "primary", small=True),
            _kpi("Win Rate", f"{win_rate * 100:.1f}%" if win_rate else "—", "warning", small=True),
            _kpi("Max DD", f"{max_dd * 100:.1f}%" if max_dd else "—", "danger", small=True),
        ],
        className="g-2 mb-3",
    )
    sections.append(kpi_row)

    # ── 2. Equity Curve Chart ───────────────────────────────────────────────
    snapshots = pp.get("daily_snapshots", [])
    if snapshots:
        snap_df = pd.DataFrame(snapshots)
        if "date" in snap_df.columns and "total_value" in snap_df.columns:
            snap_df["date"] = pd.to_datetime(snap_df["date"])
            snap_df = snap_df.sort_values("date")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=snap_df["date"], y=snap_df["total_value"],
                mode="lines", name="Total Value",
                line=dict(color="#0dcaf0", width=2),
                fill="tozeroy", fillcolor="rgba(13,202,240,0.1)",
            ))
            if "pnl_pct" in snap_df.columns:
                eq_fig.add_trace(go.Scatter(
                    x=snap_df["date"],
                    y=snap_df["pnl_pct"].apply(lambda v: capital * (1 + v) if v == v else None),
                    mode="lines", name="Benchmark (start)",
                    line=dict(color="#6c757d", width=1, dash="dot"),
                    visible="legendonly",
                ))
            eq_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                title=dict(text="Equity Curve — עקומת שווי", x=0.5, font=dict(size=14)),
                xaxis_title="תאריך", yaxis_title="שווי ($)",
                height=340, margin=dict(l=50, r=30, t=50, b=40),
                legend=dict(orientation="h", y=-0.15),
            )
            sections.append(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=eq_fig, config={"displayModeBar": False})),
            ], className="mb-3"))

    # ── 3. Open Positions Table ─────────────────────────────────────────────
    if positions:
        pos_header = html.Thead(html.Tr([
            html.Th("Ticker"), html.Th("כיוון"), html.Th("כניסה"),
            html.Th("מחיר כניסה"), html.Th("מחיר נוכחי"),
            html.Th("Notional"), html.Th("P&L $"), html.Th("P&L %"),
            html.Th("ימים"), html.Th("Conviction"),
        ]))
        pos_rows = []
        for p in positions:
            pnl_val = p.get("unrealized_pnl", 0)
            pnl_pct_val = p.get("unrealized_pnl_pct", 0)
            row_color = "#1b4332" if pnl_val >= 0 else "#4a1526"
            pos_rows.append(html.Tr([
                html.Td(p.get("ticker", ""), style={"fontWeight": "bold"}),
                html.Td(p.get("direction", ""), style={"color": "#4caf50" if p.get("direction") == "LONG" else "#ef5350"}),
                html.Td(str(p.get("entry_date", ""))[:10]),
                html.Td(f"${p.get('entry_price', 0):.2f}"),
                html.Td(f"${p.get('current_price', 0):.2f}"),
                html.Td(f"${p.get('notional', 0):,.0f}"),
                html.Td(f"${pnl_val:+,.0f}", style={"color": "#4caf50" if pnl_val >= 0 else "#ef5350", "fontWeight": "bold"}),
                html.Td(f"{pnl_pct_val * 100:+.2f}%", style={"color": "#4caf50" if pnl_pct_val >= 0 else "#ef5350"}),
                html.Td(str(p.get("days_held", 0))),
                html.Td(f"{p.get('conviction', 0):.1f}"),
            ], style={"backgroundColor": row_color}))

        pos_table = dbc.Table(
            [pos_header, html.Tbody(pos_rows)],
            bordered=True, hover=True, size="sm", dark=True,
            style={"fontSize": "12px"},
        )
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("📊 פוזיציות פתוחות — Open Positions", className="mb-0 text-center")),
            dbc.CardBody(pos_table, style={"maxHeight": "350px", "overflowY": "auto"}),
        ], className="border-info mb-3", style={"borderTop": "3px solid var(--bs-info)"}))
    else:
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("📊 פוזיציות פתוחות — Open Positions", className="mb-0 text-center")),
            dbc.CardBody(html.Div("אין פוזיציות פתוחות", className="text-muted text-center")),
        ], className="border-secondary mb-3"))

    # ── 4. Closed Trades Table ──────────────────────────────────────────────
    closed = pp.get("closed_trades", [])
    if closed:
        ct_header = html.Thead(html.Tr([
            html.Th("Ticker"), html.Th("כיוון"), html.Th("כניסה"),
            html.Th("יציאה"), html.Th("מחיר כניסה"), html.Th("מחיר יציאה"),
            html.Th("P&L $"), html.Th("P&L %"), html.Th("ימים"),
            html.Th("סיבת יציאה"),
        ]))
        ct_rows = []
        for t in closed:
            rpnl = t.get("realized_pnl", 0)
            rpct = t.get("realized_pnl_pct", 0)
            row_color = "#1b4332" if rpnl >= 0 else "#4a1526"
            ct_rows.append(html.Tr([
                html.Td(t.get("ticker", ""), style={"fontWeight": "bold"}),
                html.Td(t.get("direction", "")),
                html.Td(str(t.get("entry_date", ""))[:10]),
                html.Td(str(t.get("exit_date", ""))[:10]),
                html.Td(f"${t.get('entry_price', 0):.2f}"),
                html.Td(f"${t.get('exit_price', 0):.2f}"),
                html.Td(f"${rpnl:+,.0f}", style={"color": "#4caf50" if rpnl >= 0 else "#ef5350", "fontWeight": "bold"}),
                html.Td(f"{rpct * 100:+.2f}%", style={"color": "#4caf50" if rpct >= 0 else "#ef5350"}),
                html.Td(str(t.get("holding_days", 0))),
                html.Td(t.get("exit_reason", "—"), style={"fontSize": "11px"}),
            ], style={"backgroundColor": row_color}))

        ct_table = dbc.Table(
            [ct_header, html.Tbody(ct_rows)],
            bordered=True, hover=True, size="sm", dark=True,
            style={"fontSize": "12px"},
        )
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("📜 היסטוריית עסקאות — Closed Trades", className="mb-0 text-center")),
            dbc.CardBody(ct_table, style={"maxHeight": "350px", "overflowY": "auto"}),
        ], className="border-warning mb-3", style={"borderTop": "3px solid var(--bs-warning)"}))

    # ── 5. Exposure Breakdown Pie ───────────────────────────────────────────
    if positions:
        long_notional = sum(abs(p.get("notional", 0)) for p in positions if p.get("direction") == "LONG")
        short_notional = sum(abs(p.get("notional", 0)) for p in positions if p.get("direction") == "SHORT")
        if long_notional + short_notional > 0:
            pie_fig = go.Figure(go.Pie(
                labels=["Long", "Short"],
                values=[long_notional, short_notional],
                marker=dict(colors=["#4caf50", "#ef5350"]),
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent:.1%}<br>$%{value:,.0f}",
                hole=0.45,
            ))
            pie_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                title=dict(text="חשיפה — Long vs Short Exposure", x=0.5, font=dict(size=14)),
                height=300, margin=dict(l=30, r=30, t=50, b=30),
                showlegend=False,
            )
            sections.append(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=pie_fig, config={"displayModeBar": False})),
            ], className="mb-3"))

    return html.Div(sections, style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY LAB TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_methodology_tab(lab_data: Optional[Dict] = None) -> html.Div:
    """Full Methodology Lab tab layout — strategy comparison and analysis."""
    if not lab_data:
        return html.Div(
            dbc.Alert("אין נתוני Methodology Lab זמינים. הרץ methodology_lab כדי לייצר.", color="secondary",
                      style={"textAlign": "right", "direction": "rtl"}),
            style={"padding": "20px"},
        )

    sections: List = []
    strategies = list(lab_data.keys())
    n_strats = len(strategies)

    # Find best strategy by Sharpe
    best_name, best_sharpe, best_wr = "—", -999.0, 0.0
    for name, data in lab_data.items():
        s = data.get("sharpe", -999)
        if s > best_sharpe:
            best_sharpe = s
            best_name = name
        wr = data.get("win_rate", 0)
        if wr > best_wr:
            best_wr = wr

    # ── 1. KPI Row ──────────────────────────────────────────────────────────
    kpi_row = dbc.Row(
        [
            _kpi("# אסטרטגיות", str(n_strats), "info", small=True),
            _kpi("אסטרטגיה מובילה", best_name[:20], "success", small=True),
            _kpi("Best Sharpe", _ff(best_sharpe), "warning", small=True),
            _kpi("Best WR", _pct(best_wr), "primary", small=True),
        ],
        className="g-2 mb-3",
    )
    sections.append(kpi_row)

    # ── 2. Comparison Table ─────────────────────────────────────────────────
    sorted_strats = sorted(lab_data.items(), key=lambda x: x[1].get("sharpe", -999), reverse=True)

    cmp_header = html.Thead(html.Tr([
        html.Th("#"), html.Th("שם אסטרטגיה"), html.Th("Sharpe"),
        html.Th("Win Rate"), html.Th("P&L כולל"), html.Th("# עסקאות"),
        html.Th("Max DD"), html.Th("תיאור"),
    ]))
    cmp_rows = []
    max_sharpe_val = max(d.get("sharpe", 0) for d in lab_data.values()) if lab_data else 1
    max_sharpe_val = max(max_sharpe_val, 0.01)

    for i, (name, data) in enumerate(sorted_strats, 1):
        sharpe_val = data.get("sharpe", 0)
        sharpe_pct = max(0, min(100, (sharpe_val / max_sharpe_val) * 100)) if sharpe_val > 0 else 0
        sharpe_color = "#4caf50" if sharpe_val > 1.0 else "#ff9800" if sharpe_val > 0.5 else "#ef5350"

        sharpe_cell = html.Td([
            html.Div(
                style={
                    "width": f"{sharpe_pct}%", "height": "6px",
                    "backgroundColor": sharpe_color, "borderRadius": "3px",
                    "marginBottom": "2px", "minWidth": "2px",
                },
            ),
            html.Span(_ff(sharpe_val), style={"fontSize": "12px", "fontWeight": "bold", "color": sharpe_color}),
        ])

        cmp_rows.append(html.Tr([
            html.Td(str(i), style={"fontWeight": "bold"}),
            html.Td(name, style={"fontWeight": "bold", "color": "#0dcaf0"}),
            sharpe_cell,
            html.Td(_pct(data.get("win_rate", 0))),
            html.Td(f"${data.get('total_pnl', 0):+,.0f}"),
            html.Td(str(data.get("total_trades", 0))),
            html.Td(_pct(data.get("max_drawdown", 0))),
            html.Td(data.get("description", "—")[:60], style={"fontSize": "11px", "color": "#aaa"}),
        ]))

    cmp_table = dbc.Table(
        [cmp_header, html.Tbody(cmp_rows)],
        bordered=True, hover=True, size="sm", dark=True,
        style={"fontSize": "12px"},
    )
    sections.append(dbc.Card([
        dbc.CardHeader(html.H6("📊 השוואת אסטרטגיות — Strategy Comparison", className="mb-0 text-center")),
        dbc.CardBody(cmp_table, style={"overflowX": "auto"}),
    ], className="border-info mb-3", style={"borderTop": "3px solid var(--bs-info)"}))

    # ── 3. Parameter Details (Accordion) ────────────────────────────────────
    accordion_items = []
    for name, data in sorted_strats:
        params = data.get("params", {})
        exits = data.get("exits", {})
        if params or exits:
            param_rows = [
                html.Tr([html.Td(k, style={"fontWeight": "bold"}), html.Td(str(v))])
                for k, v in params.items()
            ]
            if exits:
                param_rows.append(html.Tr([
                    html.Td("Exit Rules", style={"fontWeight": "bold", "color": "#ff9800"}),
                    html.Td(""),
                ]))
                param_rows.extend([
                    html.Tr([html.Td(f"  {k}"), html.Td(str(v))])
                    for k, v in exits.items()
                ])
            param_table = dbc.Table(
                [html.Tbody(param_rows)],
                bordered=True, size="sm", dark=True,
                style={"fontSize": "11px", "marginBottom": "0"},
            )
            accordion_items.append(dbc.AccordionItem(
                param_table,
                title=f"{name} — Sharpe {_ff(data.get('sharpe', 0))}",
            ))

    if accordion_items:
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("⚙️ פרמטרים — Parameter Details", className="mb-0 text-center")),
            dbc.CardBody(dbc.Accordion(accordion_items, start_collapsed=True)),
        ], className="border-secondary mb-3", style={"borderTop": "3px solid var(--bs-secondary)"}))

    # ── 4. Regime Breakdown ─────────────────────────────────────────────────
    has_regime = any(data.get("regime_stats") for _, data in sorted_strats)
    if has_regime:
        # Collect all regimes
        all_regimes = set()
        for _, data in sorted_strats:
            rs = data.get("regime_stats", {})
            all_regimes.update(rs.keys())
        all_regimes = sorted(all_regimes)

        rg_header = html.Thead(html.Tr(
            [html.Th("אסטרטגיה")] + [html.Th(r) for r in all_regimes]
        ))
        rg_rows = []
        for name, data in sorted_strats:
            rs = data.get("regime_stats", {})
            cells = [html.Td(name, style={"fontWeight": "bold"})]
            for regime in all_regimes:
                val = rs.get(regime, {})
                if isinstance(val, dict):
                    wr = val.get("win_rate", val.get("wr", None))
                else:
                    wr = val
                if wr is not None:
                    wr_f = float(wr)
                    color = "#4caf50" if wr_f > 0.55 else "#ff9800" if wr_f > 0.45 else "#ef5350"
                    cells.append(html.Td(f"{wr_f * 100:.1f}%", style={"color": color, "fontWeight": "bold"}))
                else:
                    cells.append(html.Td("—", style={"color": "#666"}))
            rg_rows.append(html.Tr(cells))

        rg_table = dbc.Table(
            [rg_header, html.Tbody(rg_rows)],
            bordered=True, hover=True, size="sm", dark=True,
            style={"fontSize": "12px"},
        )
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("🔔 ביצועים לפי רגים — Regime Breakdown", className="mb-0 text-center")),
            dbc.CardBody(rg_table, style={"overflowX": "auto"}),
        ], className="border-warning mb-3", style={"borderTop": "3px solid var(--bs-warning)"}))

    # ── 5. Key Findings ─────────────────────────────────────────────────────
    findings = []
    if best_sharpe > 1.5:
        findings.append(f"האסטרטגיה המובילה ({best_name}) מציגה Sharpe גבוה של {best_sharpe:.2f} — ביצועים מצוינים.")
    elif best_sharpe > 0.8:
        findings.append(f"האסטרטגיה המובילה ({best_name}) מציגה Sharpe סביר של {best_sharpe:.2f}.")
    else:
        findings.append(f"שימו לב: Sharpe מקסימלי נמוך ({best_sharpe:.2f}). יש לשקול אופטימיזציה.")

    # Find most consistent
    if n_strats > 1:
        positive_strats = [n for n, d in lab_data.items() if d.get("sharpe", 0) > 0]
        findings.append(f"{len(positive_strats)} מתוך {n_strats} אסטרטגיות הציגו Sharpe חיובי.")

    # Max DD warning
    worst_dd_name, worst_dd = "—", 0
    for name, data in lab_data.items():
        dd = abs(data.get("max_drawdown", 0))
        if dd > worst_dd:
            worst_dd = dd
            worst_dd_name = name
    if worst_dd > 0.15:
        findings.append(f"אזהרה: {worst_dd_name} הציגה Max Drawdown של {worst_dd * 100:.1f}% — בדקו ניהול סיכונים.")

    if findings:
        findings_content = [
            html.Li(f, className="mb-1", style={"fontSize": "13px"})
            for f in findings
        ]
        sections.append(dbc.Card([
            dbc.CardHeader(html.H6("💡 ממצאים עיקריים — Key Findings", className="mb-0 text-center")),
            dbc.CardBody(html.Ul(findings_content, style={"direction": "rtl", "textAlign": "right"})),
        ], className="border-success mb-3", style={"borderTop": "3px solid var(--bs-success)"}))

    return html.Div(sections, style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────────
# ML Insights Tab
# ─────────────────────────────────────────────────────────────────────────────

_RTL = {"direction": "rtl", "textAlign": "right"}


def build_ml_insights_tab(
    feature_importances: Optional[Dict] = None,
    regime_forecast: Optional[Dict] = None,
    ml_signals: Optional[Dict] = None,
    drift_status: Optional[Dict] = None,
) -> html.Div:
    """Build the ML Insights analytics tab.

    Parameters
    ----------
    feature_importances : dict | None
        Mapping of feature name -> importance score.
    regime_forecast : dict | None
        Regime probability dict with keys like CALM, NORMAL, TENSION, CRISIS.
    ml_signals : dict | None
        Per-sector ML conviction scores and aggregate metrics.
    drift_status : dict | None
        Drift detection output: ``is_drifting``, ``current_version``.
    """
    drift_status = drift_status or {}
    ml_signals = ml_signals or {}

    # ── Row 1: KPI cards ──────────────────────────────────────────────────
    accuracy_val = ml_signals.get("accuracy")
    accuracy_str = _pct(accuracy_val) if accuracy_val is not None else "N/A"

    ic_val = ml_signals.get("ic")
    ic_str = _ff(ic_val, "{:.3f}") if ic_val is not None else "N/A"

    is_drifting = drift_status.get("is_drifting", False)
    drift_label = "DRIFT" if is_drifting else "STABLE"
    drift_color = "danger" if is_drifting else "success"

    version_str = str(drift_status.get("current_version", "v1"))

    kpi_row = dbc.Row(
        [
            _kpi("Model Accuracy — דיוק מודל", accuracy_str, color="info"),
            _kpi("IC Score — ניקוד IC", ic_str, color="primary"),
            _kpi("Drift Status — סטטוס סטייה", drift_label, color=drift_color),
            _kpi("Model Version — גרסת מודל", version_str, color="warning"),
        ],
        className="g-3 mb-4",
    )

    # ── Row 2 col-8: Feature importance bar chart ─────────────────────────
    if feature_importances and isinstance(feature_importances, dict) and len(feature_importances) > 0:
        sorted_feats = dict(
            sorted(feature_importances.items(), key=lambda kv: kv[1])
        )
        fi_fig = go.Figure(
            go.Bar(
                x=list(sorted_feats.values()),
                y=list(sorted_feats.keys()),
                orientation="h",
                marker_color="#00bc8c",
            )
        )
        fi_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            title={"text": "Feature Importance — חשיבות פיצ'רים", "x": 0.5},
            xaxis_title="Importance",
            yaxis_title="",
            margin=dict(l=180, r=20, t=50, b=40),
            height=max(350, len(sorted_feats) * 28),
        )
        fi_content = dcc.Graph(figure=fi_fig, config={"displayModeBar": False})
    else:
        fi_content = dbc.Alert(
            "ML models not trained yet — הרצת pipeline נדרשת לחישוב חשיבות פיצ'רים",
            color="info",
            className="mt-3",
            style=_RTL,
        )

    # ── Row 2 col-4: Regime forecast donut ────────────────────────────────
    if regime_forecast and isinstance(regime_forecast, dict):
        probs = regime_forecast.get("probabilities", regime_forecast)
        # Accept either top-level keys or nested 'probabilities'
        regime_labels = []
        regime_values = []
        regime_colors = {
            "CALM": "#00bc8c",
            "NORMAL": "#3498db",
            "TENSION": "#f39c12",
            "CRISIS": "#e74c3c",
        }
        for k in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
            v = probs.get(k, probs.get(k.lower(), 0))
            if v:
                regime_labels.append(k)
                regime_values.append(float(v))

        if regime_labels:
            rf_fig = go.Figure(
                go.Pie(
                    labels=regime_labels,
                    values=regime_values,
                    hole=0.5,
                    marker_colors=[regime_colors.get(l, "#888") for l in regime_labels],
                    textinfo="label+percent",
                )
            )
            rf_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                title={"text": "Regime Forecast — תחזית רגים", "x": 0.5},
                margin=dict(l=20, r=20, t=50, b=20),
                height=350,
                showlegend=False,
            )
            rf_content = dcc.Graph(figure=rf_fig, config={"displayModeBar": False})
        else:
            rf_content = dbc.Alert(
                "Regime forecast data incomplete — אין נתוני הסתברויות",
                color="warning",
                className="mt-3",
                style=_RTL,
            )
    else:
        rf_content = dbc.Alert(
            "Regime forecast not available — תחזית רגים לא זמינה",
            color="info",
            className="mt-3",
            style=_RTL,
        )

    row2 = dbc.Row(
        [
            dbc.Col(fi_content, md=8),
            dbc.Col(rf_content, md=4),
        ],
        className="g-3 mb-4",
    )

    # ── Row 3: Sector ML conviction table ─────────────────────────────────
    sector_scores = ml_signals.get("sector_scores", {})
    if sector_scores and isinstance(sector_scores, dict) and len(sector_scores) > 0:
        table_header = html.Thead(
            html.Tr([
                html.Th("Sector — סקטור", style={"textAlign": "center"}),
                html.Th("ML Score — ציון ML", style={"textAlign": "center"}),
                html.Th("Signal — סיגנל", style={"textAlign": "center"}),
            ]),
            style={"backgroundColor": "#2d2d44", "color": "#ddd"},
        )
        rows = []
        for sector, score in sorted(sector_scores.items(), key=lambda kv: kv[1], reverse=True):
            score_val = float(score)
            if score_val > 0.5:
                badge_color = "success"
                signal_text = "LONG"
            elif score_val < -0.5:
                badge_color = "danger"
                signal_text = "SHORT"
            else:
                badge_color = "secondary"
                signal_text = "NEUTRAL"
            rows.append(html.Tr([
                html.Td(sector, style={"textAlign": "center"}),
                html.Td(_ff(score_val, "{:+.3f}"), style={"textAlign": "center"}),
                html.Td(
                    dbc.Badge(signal_text, color=badge_color, className="px-2"),
                    style={"textAlign": "center"},
                ),
            ]))
        table_body = html.Tbody(rows)
        sector_table = dbc.Table(
            [table_header, table_body],
            bordered=True,
            dark=True,
            hover=True,
            striped=True,
            size="sm",
            className="mb-0",
        )
        table_card = dbc.Card(
            [
                dbc.CardHeader(
                    html.H6("Sector ML Conviction — ציוני ML לפי סקטור", className="mb-0 text-center"),
                ),
                dbc.CardBody(sector_table),
            ],
            className="border-primary",
            style={"borderTop": "3px solid var(--bs-primary)"},
        )
    else:
        table_card = dbc.Alert(
            "Run ML pipeline to see signals — הרצת ML pipeline נדרשת לצפייה בסיגנלים",
            color="info",
            className="mt-3",
            style=_RTL,
        )

    row3 = dbc.Row(
        [dbc.Col(table_card, md=12)],
        className="g-3 mb-4",
    )

    return html.Div(
        [kpi_row, row2, row3],
        style={"padding": "12px", "backgroundColor": "#1a1a2e", "borderRadius": "8px"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATUS TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_status_tab(
    registry_data: Optional[Dict] = None,
    last_reports: Optional[Dict] = None,
) -> html.Div:
    """
    Build the Agent Status monitoring tab.

    Parameters
    ----------
    registry_data : dict | None
        Output of AgentRegistry.all_agents() — keyed by agent name.
    last_reports : dict | None
        Optional dict of {agent_name: last_report_dict} with key metrics
        and GPT conversation snippets.
    """
    _RTL = {"direction": "rtl", "textAlign": "right"}

    if registry_data is None:
        registry_data = {}
    if last_reports is None:
        last_reports = {}

    # ── Status color mapping ─────────────────────────────────────────────
    status_colors = {
        "IDLE": "secondary",
        "RUNNING": "primary",
        "COMPLETED": "success",
        "FAILED": "danger",
        "STALE": "warning",
    }
    status_icons = {
        "IDLE": "⏸",
        "RUNNING": "▶",
        "COMPLETED": "✓",
        "FAILED": "✗",
        "STALE": "⚠",
    }

    # ── Agent definitions (display order) ────────────────────────────────
    agent_defs = [
        ("agent_methodology", "Methodology Agent", "methodology evaluation & benchmarking"),
        ("agent_optimizer", "Optimizer Agent", "parameter & code optimization"),
        ("agent_math", "Math Agent", "mathematical research & proposals"),
    ]

    # ── Build agent cards ────────────────────────────────────────────────
    agent_cards = []
    timeline_items = []

    for agent_key, agent_label, default_role in agent_defs:
        rec = registry_data.get(agent_key, {})
        report = last_reports.get(agent_key, {})

        status = rec.get("status", "IDLE")
        role = rec.get("role", default_role)
        last_hb = rec.get("last_heartbeat", None)
        last_run = rec.get("last_run", None)
        run_count = rec.get("run_count", 0)
        last_error = rec.get("last_error", None)
        color = status_colors.get(status, "secondary")
        icon = status_icons.get(status, "?")

        # Format timestamps
        hb_display = last_hb[:19].replace("T", " ") if last_hb else "Never"
        run_display = last_run[:19].replace("T", " ") if last_run else "Never"

        # Key metrics from last report
        metrics_items = []
        if report.get("metrics"):
            m = report["metrics"]
            for k, v in list(m.items())[:4]:
                metrics_items.append(
                    html.Span(
                        f"{k}: {_ff(v, '{:.4f}')}",
                        className="badge bg-dark me-1",
                    )
                )

        # GPT conversation snippet
        gpt_snippet = ""
        if report.get("gpt_conversation"):
            conv = report["gpt_conversation"]
            if isinstance(conv, list) and conv:
                last_msg = conv[-1]
                content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
                gpt_snippet = content[:150] + "..." if len(content) > 150 else content
            elif isinstance(conv, str):
                gpt_snippet = conv[:150] + "..." if len(conv) > 150 else conv

        # Error display
        error_badge = html.Span()
        if last_error and status == "FAILED":
            error_badge = dbc.Alert(
                f"Error: {last_error[:200]}",
                color="danger",
                className="mt-2 mb-0 py-1 px-2",
                style={"fontSize": "0.8rem"},
            )

        card = dbc.Col(
            dbc.Card([
                dbc.CardHeader([
                    html.Span(f"{icon} ", style={"fontSize": "1.2rem"}),
                    html.Strong(agent_label),
                    dbc.Badge(status, color=color, className="ms-2"),
                ], style={"backgroundColor": "#16213e"}),
                dbc.CardBody([
                    html.Div([
                        html.Small(role, className="text-muted d-block mb-2"),
                        html.Div([
                            html.Span("Last heartbeat: ", className="text-muted"),
                            html.Span(hb_display),
                        ], className="mb-1", style={"fontSize": "0.85rem"}),
                        html.Div([
                            html.Span("Last run: ", className="text-muted"),
                            html.Span(run_display),
                        ], className="mb-1", style={"fontSize": "0.85rem"}),
                        html.Div([
                            html.Span("Total runs: ", className="text-muted"),
                            html.Span(str(run_count)),
                        ], className="mb-2", style={"fontSize": "0.85rem"}),
                        html.Div(metrics_items, className="mb-2") if metrics_items else html.Div(),
                        html.Div([
                            html.Small("GPT: ", className="text-muted"),
                            html.Small(gpt_snippet, className="text-info"),
                        ], style={"fontSize": "0.8rem"}) if gpt_snippet else html.Div(),
                        error_badge,
                    ]),
                ], style={"backgroundColor": "#1a1a2e"}),
            ], className="h-100", style={"border": "1px solid #2a2a4a"}),
            md=4,
            className="mb-3",
        )
        agent_cards.append(card)

        # Timeline entry
        if last_run:
            timeline_items.append({
                "agent": agent_label,
                "time": run_display,
                "status": status,
                "color": color,
            })

    # ── Agent cards row ──────────────────────────────────────────────────
    cards_row = dbc.Row(agent_cards, className="g-3 mb-4")

    # ── Timeline section ─────────────────────────────────────────────────
    timeline_items.sort(key=lambda x: x["time"], reverse=True)
    timeline_rows = []
    for item in timeline_items[:10]:
        timeline_rows.append(
            html.Tr([
                html.Td(item["agent"], style={"fontSize": "0.85rem"}),
                html.Td(item["time"], style={"fontSize": "0.85rem"}),
                html.Td(
                    dbc.Badge(item["status"], color=item["color"]),
                ),
            ])
        )

    timeline_table = dbc.Card([
        dbc.CardHeader(
            html.Strong("Agent Orchestration Timeline"),
            style={"backgroundColor": "#16213e"},
        ),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("Agent"), html.Th("Last Run"), html.Th("Status"),
                ]))] + [html.Tbody(timeline_rows)] if timeline_rows else [
                    html.Div("No agent runs recorded yet.", className="text-muted p-3"),
                ],
                bordered=True,
                dark=True,
                hover=True,
                size="sm",
                className="mb-0",
            ) if timeline_rows else html.Div(
                "No agent runs recorded yet.",
                className="text-muted p-3",
            ),
            style={"backgroundColor": "#1a1a2e", "padding": "0"},
        ),
    ], className="mb-4", style={"border": "1px solid #2a2a4a"})

    # ── Error log section ────────────────────────────────────────────────
    error_items = []
    for agent_key, _, _ in agent_defs:
        rec = registry_data.get(agent_key, {})
        err = rec.get("last_error")
        if err:
            agent_label_short = agent_key.replace("agent_", "").title()
            error_items.append(
                dbc.ListGroupItem(
                    [
                        dbc.Badge(agent_label_short, color="danger", className="me-2"),
                        html.Span(err[:300]),
                    ],
                    style={"backgroundColor": "#1a1a2e", "border": "1px solid #2a2a4a",
                            "fontSize": "0.85rem"},
                )
            )

    error_card = dbc.Card([
        dbc.CardHeader(
            html.Strong("Error Log"),
            style={"backgroundColor": "#16213e"},
        ),
        dbc.CardBody(
            dbc.ListGroup(error_items, flush=True) if error_items else html.Div(
                "No errors — all agents healthy.",
                className="text-success p-2",
            ),
            style={"backgroundColor": "#1a1a2e", "padding": "8px"},
        ),
    ], style={"border": "1px solid #2a2a4a"})

    return html.Div(
        [
            html.H4("Agent Status Monitor", className="text-info mb-3"),
            html.P(
                "Real-time status of the three SRV agents: Methodology, Optimizer, and Math.",
                className="text-muted mb-3",
            ),
            cards_row,
            dbc.Row([
                dbc.Col(timeline_table, md=7),
                dbc.Col(error_card, md=5),
            ], className="g-3"),
        ],
        style={"padding": "12px", "backgroundColor": "#1a1a2e", "borderRadius": "8px"},
    )
