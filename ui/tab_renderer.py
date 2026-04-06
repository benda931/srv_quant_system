"""
ui/tab_renderer.py
====================
Tab rendering logic extracted from main.py's 701-line render_tab callback.

Each tab is rendered by a dedicated function that receives a TabContext
containing all pre-computed analytics results and agent data.

Usage (from main.py):
    from ui.tab_renderer import TabContext, render_tab_content

    tab_ctx = TabContext(...)
    content = render_tab_content(active_tab, tab_ctx)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

log = logging.getLogger("tab_renderer")

# ── Shared styles ────────────────────────────────────────────────────────────
RTL_STYLE = {"direction": "rtl", "textAlign": "right"}

def _loading(children) -> dcc.Loading:
    return dcc.Loading(
        children=[dbc.Container(fluid=True, children=children)],
        type="circle", color="#00bc8c", style={"minHeight": "200px"},
    )

def _tab_header(title: str, subtitle: str = "", style: dict = RTL_STYLE) -> html.Div:
    children = [html.H5(title, className="mt-2", style=style)]
    if subtitle:
        children.append(html.Div(subtitle, className="text-muted small mb-3", style=style))
    return html.Div(children)


@dataclass
class TabContext:
    """
    Bundle of all pre-computed data needed by tab renderers.
    Created once in build_app(), passed to render_tab_content().
    Eliminates 50+ closure variable references.
    """
    # Core
    master_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Any = None
    engine: Any = None

    # DSS / Signals
    signal_results: Optional[List] = None
    trade_tickets: Optional[List] = None
    regime_safety: Any = None
    corr_snapshot: Any = None
    monitor_summary: Any = None
    trade_book_history: Optional[pd.DataFrame] = None
    methodology_ranking: Optional[List[Dict]] = None

    # Options / Tail Risk
    options_surface: Any = None
    tail_risk_es: Any = None

    # Stress / Risk
    stress_results: Optional[List] = None
    mc_stress_result: Any = None
    risk_report: Any = None

    # Correlation
    corr_vol_analysis: Any = None

    # Tracking
    pnl_result: Any = None
    decay_result: Any = None
    regime_result: Any = None
    backtest_result: Any = None

    # Paper
    paper_portfolio: Optional[Dict] = None
    dispersion_result: Any = None

    # ML
    ml_feature_importances: Optional[Dict] = None
    ml_regime_forecast: Any = None
    ml_signals_result: Any = None
    ml_drift_status: Any = None
    ensemble_results: Any = None

    # Agent data
    agent_registry: Optional[Dict] = None
    decay_data: Optional[Dict] = None
    regime_agent_data: Optional[Dict] = None
    risk_agent_data: Optional[Dict] = None
    scout_data: Optional[Dict] = None
    portfolio_alloc: Optional[Dict] = None
    auto_improve_data: Optional[Dict] = None
    optimizer_data: Optional[Dict] = None
    architect_data: Optional[Dict] = None

    # Engine errors (for banners)
    errors: Dict[str, str] = field(default_factory=dict)

    # Callables (thin wrappers from main.py)
    compute_momentum_ranking: Optional[Callable] = None
    load_improvement_log: Optional[Callable] = None
    load_methodology_results: Optional[Callable] = None
    load_json_safe: Optional[Callable] = None
    engine_error_banner: Optional[Callable] = None

    # Brief
    brief_txt: str = ""

    # Health
    data_health: Any = None


def render_tab_content(active_tab: str, ctx: TabContext) -> Any:
    """
    Render the content for a given tab ID.
    Dispatches to per-tab render functions.
    """
    renderer = _TAB_RENDERERS.get(active_tab)
    if renderer:
        return renderer(ctx)
    return None  # Falls through to Overview in main.py


# ─────────────────────────────────────────────────────────────────────────────
# Per-tab renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_dss(ctx: TabContext):
    from ui.analytics_tabs import build_dss_tab
    banner = ctx.engine_error_banner("dss", "Decision Support System") if ctx.engine_error_banner else None
    mom = ctx.compute_momentum_ranking() if ctx.compute_momentum_ranking else None
    return _loading([
        banner or html.Div(),
        _tab_header("🎯 Decision Support System — Short Vol / Dispersion",
                     "מערכת תומכת החלטה: Signal Stack (4 שכבות), Trade Book עם Greeks, "
                     "Regime Safety gate, ותנאי כניסה/יציאה."),
        build_dss_tab(ctx.signal_results, ctx.trade_tickets,
                      ctx.regime_safety, ctx.corr_snapshot,
                      ctx.monitor_summary, ctx.options_surface,
                      ctx.tail_risk_es, ctx.methodology_ranking,
                      ctx.paper_portfolio, ctx.dispersion_result,
                      ctx.trade_book_history, momentum_ranking=mom),
    ])


def _render_scanner(ctx: TabContext):
    from ui.scanner_pro import build_scanner_pro
    return _loading([build_scanner_pro(ctx.master_df)])


def _render_correlation(ctx: TabContext):
    import plotly.graph_objects as go
    import numpy as np

    # Build correlation figures from available data
    master_df = ctx.master_df
    engine = ctx.engine

    # Current correlation heatmap
    corr_fig = go.Figure()
    delta_fig = go.Figure()
    ts_fig = go.Figure()
    contrib_fig = go.Figure()

    try:
        if engine and hasattr(engine, 'prices') and engine.prices is not None:
            sectors = [s for s in ctx.settings.sector_list() if s in engine.prices.columns] if ctx.settings else []
            if sectors and len(engine.prices) > 60:
                log_rets = np.log(engine.prices[sectors] / engine.prices[sectors].shift(1)).dropna()

                # Current correlation matrix
                C = log_rets.tail(60).corr()
                corr_fig = go.Figure(data=go.Heatmap(
                    z=C.values, x=sectors, y=sectors,
                    colorscale="RdYlGn", zmid=0,
                    text=np.round(C.values, 2), texttemplate="%{text:.2f}",
                ))
                corr_fig.update_layout(
                    template="plotly_dark", height=400,
                    title="Sector Correlation Matrix (60d)",
                    margin=dict(l=80, r=20, t=50, b=80),
                )

                # Baseline correlation
                C_base = log_rets.tail(252).corr()
                delta_C = C - C_base
                delta_fig = go.Figure(data=go.Heatmap(
                    z=delta_C.values, x=sectors, y=sectors,
                    colorscale="RdBu_r", zmid=0,
                    text=np.round(delta_C.values, 3), texttemplate="%{text:.3f}",
                ))
                delta_fig.update_layout(
                    template="plotly_dark", height=400,
                    title="Correlation Change (60d vs 252d baseline)",
                    margin=dict(l=80, r=20, t=50, b=80),
                )

                # Rolling average correlation
                n = len(log_rets)
                iu = np.triu_indices(len(sectors), k=1)
                dates = []
                avg_corrs = []
                for t in range(60, n, 5):
                    window = log_rets.iloc[t-60:t]
                    Cw = window.corr().values
                    avg_corrs.append(float(np.mean(Cw[iu])))
                    dates.append(log_rets.index[t])

                ts_fig = go.Figure()
                ts_fig.add_trace(go.Scatter(x=dates, y=avg_corrs, mode="lines",
                    line=dict(color="#00d4ff", width=1.5), name="Avg Correlation"))
                ts_fig.add_hline(y=0.5, line_dash="dash", line_color="#ffc107", opacity=0.5,
                                 annotation_text="Tension threshold")
                ts_fig.update_layout(
                    template="plotly_dark", height=300,
                    title="Rolling Average Pairwise Correlation (60d)",
                    yaxis_title="Avg Correlation",
                    margin=dict(l=60, r=20, t=50, b=30),
                )

                # Sector contribution to distortion
                delta_abs = np.abs(delta_C.values)
                sector_contrib = delta_abs.sum(axis=1) / (delta_abs.sum() + 1e-10)
                contrib_fig = go.Figure(go.Bar(
                    x=sectors, y=sector_contrib * 100,
                    marker_color=["#dc3545" if v > 0.12 else "#ffc107" if v > 0.08 else "#20c997" for v in sector_contrib],
                    text=[f"{v:.1f}%" for v in sector_contrib * 100],
                    textposition="outside",
                ))
                contrib_fig.update_layout(
                    template="plotly_dark", height=300,
                    title="Sector Contribution to Correlation Distortion",
                    yaxis_title="% of total distortion",
                    margin=dict(l=60, r=20, t=50, b=30),
                    showlegend=False,
                )
    except Exception:
        pass

    return _loading([
        _tab_header("Correlation Structure",
                     "Sector pairwise correlation matrix, change from baseline, rolling time series, and distortion contribution"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=corr_fig, config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=delta_fig, config={"displayModeBar": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=ts_fig, config={"displayModeBar": False}), md=7),
            dbc.Col(dcc.Graph(figure=contrib_fig, config={"displayModeBar": False}), md=5),
        ]),
    ])


def _render_stress(ctx: TabContext):
    from ui.analytics_tabs import build_stress_tab
    banner = ctx.engine_error_banner("stress", "Stress Testing") if ctx.engine_error_banner else None
    return _loading([
        banner or html.Div(),
        _tab_header("⚡ Stress Testing — 10 תרחישים מוסדיים",
                     "ניתוח קדימה: השפעת תרחישי קיצון על הספר, P&L מוערך ואמינות האות בכל משטר."),
        build_stress_tab(ctx.stress_results, ctx.master_df, mc_result=ctx.mc_stress_result),
    ])


def _render_risk(ctx: TabContext):
    from ui.analytics_tabs import build_risk_tab
    banner = ctx.engine_error_banner("risk", "Portfolio Risk") if ctx.engine_error_banner else None
    return _loading([
        banner or html.Div(),
        _tab_header("🛡️ Portfolio Risk — סיכון פורטפוליו",
                     "VaR, CVaR, MCTR, Risk Budget, Factor Decomposition"),
        build_risk_tab(ctx.risk_report, ctx.master_df),
    ])


def _render_corrvol(ctx: TabContext):
    from ui.analytics_tabs import build_corr_vol_tab
    return _loading([
        _tab_header("📈 Correlation & Vol Pricing",
                     "Implied correlation, VRP, dispersion index, short vol signal, term structure"),
        build_corr_vol_tab(ctx.corr_vol_analysis),
    ])


def _render_pnl(ctx: TabContext):
    from ui.analytics_tabs import build_pnl_tracker_tab
    return _loading([
        _tab_header("💰 P&L Tracker — מעקב ביצועים",
                     "P&L מצטבר, Sharpe, Drawdown, Factor Attribution, Monthly Returns"),
        build_pnl_tracker_tab(ctx.pnl_result),
    ])


def _render_backtest(ctx: TabContext):
    from ui.analytics_tabs import build_backtest_tab
    return _loading([
        _tab_header("🔬 Walk-Forward Backtest",
                     "OOS validation, IC decay, regime breakdown, equity curves"),
        build_backtest_tab(ctx.backtest_result),
    ])


def _render_decay(ctx: TabContext):
    from ui.analytics_tabs import build_signal_decay_tab
    return _loading([
        _tab_header("📉 Signal Decay — דעיכת אות",
                     "IC decay across horizons, optimal holding period, turnover analysis"),
        build_signal_decay_tab(ctx.decay_result),
    ])


def _render_regime(ctx: TabContext):
    from ui.analytics_tabs import build_regime_timeline_tab
    return _loading([
        _tab_header("🌡️ Regime Timeline",
                     "Historical regime transitions, alert history, crisis probability"),
        build_regime_timeline_tab(ctx.regime_result),
    ])


def _render_health(ctx: TabContext):
    """System health diagnostics — data freshness, quality checks, pipeline status."""
    sections = []
    sections.append(_tab_header("Health Diagnostics",
                                 "Data freshness, quality checks, pipeline status, agent health"))

    try:
        from db.repository import Repository
        from config.settings import get_settings
        repo = Repository(get_settings().db_path)

        # Data freshness
        freshness = repo.data_freshness()
        fresh_color = "success" if freshness.is_fresh else "danger"
        sections.append(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Data Freshness", className="text-muted", style={"fontSize": "11px"}),
                html.Div("FRESH" if freshness.is_fresh else "STALE",
                         className=f"h4 text-{fresh_color} fw-bold text-center"),
            ]), className=f"border-{fresh_color}"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Prices Latest", className="text-muted", style={"fontSize": "11px"}),
                html.Div(str(freshness.prices_latest or "—"), className="h5 text-center"),
            ])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Price Rows", className="text-muted", style={"fontSize": "11px"}),
                html.Div(f"{freshness.prices_rows:,}", className="h5 text-center"),
            ])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Total Runs", className="text-muted", style={"fontSize": "11px"}),
                html.Div(str(freshness.runs_count), className="h5 text-center"),
            ])), md=3),
        ], className="g-2 mb-3"))

        if freshness.warnings:
            for w in freshness.warnings:
                sections.append(dbc.Alert(w, color="warning", className="py-1 mb-1"))

        # Quality checks
        checks = repo.run_data_quality_checks()
        check_rows = []
        for c in checks:
            icon = "✓" if c.status == "PASS" else "⚠" if c.status == "WARN" else "✗"
            color = "success" if c.status == "PASS" else "warning" if c.status == "WARN" else "danger"
            check_rows.append(html.Tr([
                html.Td(icon, className=f"text-{color} text-center"),
                html.Td(c.check_name, style={"fontSize": "12px"}),
                html.Td(c.table_name or "—", style={"fontSize": "11px", "color": "#888"}),
                html.Td(dbc.Badge(c.status, color=color, style={"fontSize": "10px"})),
                html.Td(c.message[:60], style={"fontSize": "11px"}),
            ]))

        if check_rows:
            sections.append(dbc.Card([
                dbc.CardHeader(html.Strong("Data Quality Checks")),
                dbc.CardBody(dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("", style={"width": "30px"}),
                        html.Th("Check"), html.Th("Table"),
                        html.Th("Status"), html.Th("Details"),
                    ])),
                    html.Tbody(check_rows),
                ], bordered=True, dark=True, hover=True, size="sm")),
            ], className="mb-3"))

        # Latest run
        run_summary = repo.run_summary()
        if run_summary:
            sections.append(dbc.Card([
                dbc.CardHeader(html.Strong("Latest Pipeline Run")),
                dbc.CardBody(dbc.Row([
                    dbc.Col([html.Div("Run ID", className="text-muted small"), html.Div(f"#{run_summary.run_id}", className="fw-bold")]),
                    dbc.Col([html.Div("Date", className="text-muted small"), html.Div(run_summary.run_date)]),
                    dbc.Col([html.Div("Regime", className="text-muted small"), html.Div(run_summary.regime)]),
                    dbc.Col([html.Div("Duration", className="text-muted small"), html.Div(f"{run_summary.duration_s:.0f}s")]),
                    dbc.Col([html.Div("Steps", className="text-muted small"), html.Div(f"{run_summary.steps_ok} OK / {run_summary.steps_fail} fail")]),
                    dbc.Col([html.Div("Health", className="text-muted small"), html.Div(run_summary.data_health)]),
                ])),
            ], className="mb-3"))

    except Exception as e:
        sections.append(dbc.Alert(f"Health diagnostics error: {str(e)[:100]}", color="danger"))

    return _loading(sections)


def _render_journal(ctx: TabContext):
    from ui.journal_panel import build_journal_tab
    return _loading([build_journal_tab()])


def _render_portfolio(ctx: TabContext):
    from ui.analytics_tabs import build_portfolio_tab
    return _loading([
        _tab_header("💼 Paper Portfolio — פורטפוליו וירטואלי",
                     "פוזיציות פתוחות, P&L, מעקב ביצועים, signal sources (DSS + momentum)"),
        build_portfolio_tab(ctx.paper_portfolio, prices=ctx.engine.prices if ctx.engine else None,
                            portfolio_alloc=ctx.portfolio_alloc),
    ])


def _render_methodology(ctx: TabContext):
    from ui.analytics_tabs import build_methodology_tab
    _mlab = None
    _alpha = None
    try:
        if ctx.load_json_safe:
            _reports_dir = ctx.settings.project_root / "agents" / "methodology" / "reports"
            _mlab_files = sorted(_reports_dir.glob("*methodology_lab*"), reverse=True)
            if _mlab_files:
                _mlab = ctx.load_json_safe(str(_mlab_files[0]))
            _alpha_files = sorted(_reports_dir.glob("*alpha_research*"), reverse=True)
            if _alpha_files:
                _alpha = ctx.load_json_safe(str(_alpha_files[0]))
    except Exception:
        pass

    _gov_data = None
    try:
        if ctx.load_json_safe:
            _gov_path = ctx.settings.project_root / "agents" / "methodology" / "reports" / "methodology_governance.json"
            _gov_data = ctx.load_json_safe(str(_gov_path))
    except Exception:
        pass

    return _loading([
        _tab_header("🧪 Methodology Lab — מעבדת אסטרטגיות",
                     "השוואת מתודולוגיות, OOS validation, regime fitness, governance"),
        build_methodology_tab(
            lab_data=_mlab,
            alpha_research=_alpha,
            governance_data=_gov_data,
        ),
    ])


def _render_ml(ctx: TabContext):
    from ui.analytics_tabs import build_ml_insights_tab
    return _loading([
        _tab_header("🧠 ML Insights — תובנות מודלים",
                     "Feature importance, regime forecast, drift detection"),
        build_ml_insights_tab(
            feature_importances=ctx.ml_feature_importances,
            regime_forecast=ctx.ml_regime_forecast,
            ml_signals=ctx.ml_signals_result,
            drift_status=ctx.ml_drift_status,
            ensemble_results=ctx.ensemble_results,
            scout_data=ctx.scout_data,
        ),
    ])


def _render_agents(ctx: TabContext):
    from ui.analytics_tabs import build_agent_monitor_tab

    _agent_reg_data = ctx.agent_registry or {}
    _audit_changes = []
    try:
        from agents.shared.agent_registry import get_registry
        _reg = get_registry()
        for _aname in ["methodology", "optimizer", "math", "architect"]:
            if not _reg.get_status(_aname):
                _reg.register(_aname, role=f"SRV {_aname} agent")
    except Exception:
        pass
    try:
        from agents.shared.agent_registry import AgentRegistry
        _agent_reg = AgentRegistry()
        _agent_reg_data = _agent_reg.all_agents() or _agent_reg_data
    except Exception:
        pass
    try:
        from db.audit import AuditTrail
        _audit = AuditTrail()
        _audit_changes = _audit._conn.execute(
            "SELECT * FROM audit.param_changes ORDER BY timestamp DESC LIMIT 20"
        ).fetchdf().to_dict("records")
    except Exception:
        pass

    _imp_log = ctx.load_improvement_log() if ctx.load_improvement_log else None
    _meth_res = ctx.load_methodology_results() if ctx.load_methodology_results else None

    return _loading([
        _tab_header("🤖 Agent Monitor — מעקב סוכנים",
                     "סטטוס סוכנים, היסטוריית שינויים, ביצועי אסטרטגיות"),
        build_agent_monitor_tab(
            registry_data=_agent_reg_data,
            audit_changes=_audit_changes,
            risk_data=ctx.risk_agent_data,
            regime_data=ctx.regime_agent_data,
            decay_data=ctx.decay_data,
            scout_data=ctx.scout_data,
            portfolio_alloc=ctx.portfolio_alloc,
            auto_improve_data=ctx.auto_improve_data,
            optimizer_data=ctx.optimizer_data,
            architect_data=ctx.architect_data,
            project_root=str(ctx.settings.project_root) if ctx.settings else None,
            improvement_log=_imp_log,
            methodology_results=_meth_res,
        ),
    ])


def _render_optimization(ctx: TabContext):
    from ui.analytics_tabs import build_optimization_tab
    _mlab = None
    _optuna = None
    try:
        if ctx.load_json_safe:
            _rd = ctx.settings.project_root / "agents" / "methodology" / "reports"
            _mf = sorted(_rd.glob("*methodology_lab*"), reverse=True)
            if _mf:
                _mlab = ctx.load_json_safe(str(_mf[0]))
            _optuna = ctx.load_json_safe(ctx.settings.project_root / "data" / "optuna_pareto.json")
    except Exception:
        pass
    return _loading([
        _tab_header("🎯 Optimization — אופטימיזציה",
                     "מרחב פרמטרים, השוואת אסטרטגיות, ומצב שיפור אוטומטי."),
        build_optimization_tab(
            optimizer_history=ctx.optimizer_data,
            auto_improve_data=ctx.auto_improve_data,
            methodology_lab=_mlab,
            optuna_pareto=_optuna,
            settings_obj=ctx.settings,
        ),
    ])


def _render_tearsheet(ctx: TabContext):
    from ui.panels import build_tearsheet_panel
    # NOTE: build_tearsheet_explainer lives in main.py — access via ctx if available
    explainer = html.Div()
    if hasattr(ctx, '_tearsheet_explainer') and ctx._tearsheet_explainer:
        explainer = ctx._tearsheet_explainer
    return dbc.Container(fluid=True, children=[explainer, build_tearsheet_panel()])


# ─────────────────────────────────────────────────────────────────────────────
# Tab dispatch table
# ─────────────────────────────────────────────────────────────────────────────
_TAB_RENDERERS = {
    "tab-dss": _render_dss,
    "tab-scanner": _render_scanner,
    "tab-correlation": _render_correlation,
    "tab-stress": _render_stress,
    "tab-risk": _render_risk,
    "tab-corrvol": _render_corrvol,
    "tab-pnl": _render_pnl,
    "tab-backtest": _render_backtest,
    "tab-decay": _render_decay,
    "tab-regime": _render_regime,
    "tab-health": _render_health,
    "tab-journal": _render_journal,
    "tab-portfolio": _render_portfolio,
    "tab-methodology": _render_methodology,
    "tab-ml": _render_ml,
    "tab-agents": _render_agents,
    "tab-optimization": _render_optimization,
    "tab-tearsheet": _render_tearsheet,
}


def _render_overview(ctx: TabContext):
    """Render the Overview tab with KPI snapshot cards."""
    from ui.panels import (
        build_market_narrative, build_regime_hero, build_action_plan,
        build_opportunities_section, build_stat_analysis_panel,
        build_correlation_summary,
    )
    # These are passed via TabContext from main.py — no circular import
    build_health_overview_banner = lambda h: html.Div()
    build_overview_kpi_rows = lambda *a, **k: (html.Div(), html.Div())
    if hasattr(ctx, '_build_health_banner') and ctx._build_health_banner:
        build_health_overview_banner = ctx._build_health_banner
    if hasattr(ctx, '_build_kpi_rows') and ctx._build_kpi_rows:
        build_overview_kpi_rows = ctx._build_kpi_rows

    row0 = ctx.master_df.iloc[0].to_dict() if len(ctx.master_df) else {}
    cards_top, cards_bottom = build_overview_kpi_rows(ctx.master_df, ctx.settings, ctx.data_health)

    # ── Snapshot cards ─────────────────────────────────
    stress_kpis = html.Div()
    if ctx.stress_results:
        worst = ctx.stress_results[0]
        best = ctx.stress_results[-1]
        stress_kpis = dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Div("⚡ Stress Snapshot", className="fw-bold"), width="auto"),
            dbc.Col([
                html.Span("Worst: ", className="text-muted small"),
                html.Span(f"{worst.scenario_name} ({worst.portfolio_pnl_estimate*100:+.1f}%)",
                          className="text-danger small fw-bold me-3"),
                html.Span("Best: ", className="text-muted small"),
                html.Span(f"{best.scenario_name} ({best.portfolio_pnl_estimate*100:+.1f}%)",
                          className="text-success small fw-bold"),
            ]),
        ], align="center")), className="mb-3 border-warning", style={"borderWidth": "1px"})

    risk_kpis = html.Div()
    if ctx.risk_report:
        vol = getattr(ctx.risk_report, "portfolio_vol_ann", 0) or 0
        var95 = getattr(ctx.risk_report, "var_95_1d", 0) or 0
        cvar95 = getattr(ctx.risk_report, "cvar_95_1d", 0) or 0
        risk_kpis = dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Div("🎯 Risk", className="fw-bold"), width="auto"),
            dbc.Col([
                html.Span(f"Vol: {vol*100:.1f}% ", className="small fw-bold me-2"),
                html.Span(f"VaR95: {var95*100:.2f}% ", className="small me-2"),
                html.Span(f"CVaR95: {cvar95*100:.2f}%", className="text-danger small"),
            ]),
        ], align="center")), className="mb-3 border-secondary", style={"borderWidth": "1px"})

    dss_kpis = html.Div()
    if ctx.signal_results and ctx.regime_safety:
        _n_pass = sum(1 for r in ctx.signal_results if r.passes_entry)
        _n_active = sum(1 for t in (ctx.trade_tickets or []) if t.is_active)
        _safety = ctx.regime_safety
        _sc = {"SAFE": "success", "CAUTION": "warning", "DANGER": "danger", "KILLED": "danger"}.get(_safety.label, "secondary")
        dss_kpis = dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Div("🎯 DSS", className="fw-bold"), width="auto"),
            dbc.Col([
                html.Span(f"Safety: {_safety.regime_safety_score*100:.0f}% ", className=f"small fw-bold text-{_sc} me-2"),
                html.Span(f"Trades: {_n_active} active ", className="small me-2"),
                html.Span(f"Signals: {_n_pass} passing", className="small"),
            ]),
        ], align="center")), className=f"mb-3 border-{_sc}", style={"borderWidth": "1px"})

    mc_kpis = html.Div()
    if ctx.mc_stress_result:
        _mc = ctx.mc_stress_result
        mc_kpis = dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Div("🎲 MC Risk (21d)", className="fw-bold"), width="auto"),
            dbc.Col([
                html.Span(f"VaR95: {_mc.var_95*100:.2f}% ", className="small fw-bold text-warning me-2"),
                html.Span(f"CVaR95: {_mc.cvar_95*100:.2f}% ", className="small text-danger me-2"),
                html.Span(f"Skew: {_mc.skewness:+.2f}", className="small"),
            ]),
        ], align="center")), className="mb-3 border-info", style={"borderWidth": "1px"})

    # ── Best Strategy card ──
    strategy_kpis = html.Div()
    try:
        meth_results = ctx.load_methodology_results() if ctx.load_methodology_results else None
        if meth_results:
            ranked = sorted(meth_results.items(),
                           key=lambda x: x[1].get("sharpe", 0) if isinstance(x[1], dict) else 0,
                           reverse=True)
            best_name, best_data = ranked[0] if ranked else ("?", {})
            if isinstance(best_data, dict):
                bs = best_data.get("sharpe", 0)
                bw = best_data.get("win_rate", 0)
                bp = best_data.get("total_pnl", 0)
                n_positive = sum(1 for _, d in ranked if isinstance(d, dict) and d.get("sharpe", 0) > 0)
                strategy_kpis = dbc.Card(dbc.CardBody(dbc.Row([
                    dbc.Col(html.Div("📈 Best Strategy", className="fw-bold"), width="auto"),
                    dbc.Col([
                        html.Span(f"{best_name} ", className="small fw-bold me-2"),
                        html.Span(f"Sharpe: {bs:.2f} ", className=f"small fw-bold text-{'success' if bs > 0.5 else 'warning'} me-2"),
                        html.Span(f"WR: {bw:.0%} ", className="small me-2"),
                        html.Span(f"PnL: {bp:.1%} ", className="small me-2"),
                        html.Span(f"({n_positive} positive / {len(ranked)} total)", className="small text-muted"),
                    ]),
                ], align="center")), className="mb-3 border-success", style={"borderWidth": "1px"})
    except Exception:
        pass

    # ── Momentum card ──
    momentum_kpis = html.Div()
    try:
        mom = ctx.compute_momentum_ranking() if ctx.compute_momentum_ranking else None
        if mom and len(mom) >= 3:
            top3 = [m["ticker"] for m in mom[:3]]
            bot3 = [m["ticker"] for m in mom[-3:]]
            momentum_kpis = dbc.Card(dbc.CardBody(dbc.Row([
                dbc.Col(html.Div("🔄 Momentum Ranking (21d)", className="fw-bold"), width="auto"),
                dbc.Col([
                    html.Span("LONG: ", className="small text-muted"),
                    html.Span(f"{', '.join(top3)} ", className="small fw-bold text-success me-3"),
                    html.Span("SHORT: ", className="small text-muted"),
                    html.Span(f"{', '.join(bot3)} ", className="small fw-bold text-danger me-3"),
                    html.Span(f"Top: {mom[0]['momentum_21d']:+.1%}", className="small text-muted"),
                ]),
            ], align="center")), className="mb-3 border-primary", style={"borderWidth": "1px"})
    except Exception:
        pass

    return dbc.Container(fluid=True, children=[
        build_market_narrative(ctx.master_df),
        build_health_overview_banner(ctx.data_health) if ctx.data_health else html.Div(),
        build_regime_hero(row0),
        cards_top,
        cards_bottom,
        strategy_kpis,
        momentum_kpis,
        dss_kpis,
        mc_kpis,
        stress_kpis,
        risk_kpis,
        build_action_plan(ctx.master_df),
        build_opportunities_section(ctx.master_df),
        build_stat_analysis_panel(ctx.master_df),
        build_correlation_summary(ctx.master_df),
    ])

_TAB_RENDERERS["tab-overview"] = _render_overview
