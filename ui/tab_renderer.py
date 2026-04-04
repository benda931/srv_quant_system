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
    from ui.panels import build_correlation_panel
    return _loading([
        _tab_header("🔗 Correlation Structure — מבנה קורלציות סקטוריאליות",
                     "PCA eigenvector decomposition, rolling regime classification, "
                     "factor loadings, and sectoral distortion scoring"),
        build_correlation_panel(ctx.master_df),
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
    try:
        from ui.panels import build_data_health_tab
        return _loading([build_data_health_tab(ctx.data_health)])
    except ImportError:
        return _loading([html.Div("Health tab unavailable", className="text-muted p-3")])


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
    try:
        from ui.panels import build_tearsheet_panel
        from main import build_tearsheet_explainer
        return dbc.Container(fluid=True, children=[
            build_tearsheet_explainer(), build_tearsheet_panel(),
        ])
    except ImportError:
        from ui.panels import build_tearsheet_panel
        return dbc.Container(fluid=True, children=[build_tearsheet_panel()])


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
    # These are defined in main.py — import from there
    try:
        from main import build_health_overview_banner, build_overview_kpi_rows
    except ImportError:
        build_health_overview_banner = lambda h: html.Div()
        build_overview_kpi_rows = lambda *a, **k: (html.Div(), html.Div())

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

    return dbc.Container(fluid=True, children=[
        build_market_narrative(ctx.master_df),
        build_health_overview_banner(ctx.data_health) if ctx.data_health else html.Div(),
        build_regime_hero(row0),
        cards_top,
        cards_bottom,
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
