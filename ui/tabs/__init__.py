"""
ui/tabs/ — Per-tab UI builders.

Re-exports all build_* functions from analytics_tabs.py for backward compatibility.
New tabs should be created as individual modules in this directory.

Usage:
    from ui.tabs import build_stress_tab, build_dss_tab
"""
# Re-export from analytics_tabs.py (legacy monolith — gradually split)
from ui.analytics_tabs import (
    build_stress_tab,
    build_risk_tab,
    build_backtest_tab,
    build_daily_brief_panel,
    build_corr_vol_tab,
    build_pnl_tracker_tab,
    build_signal_decay_tab,
    build_regime_timeline_tab,
    build_dss_tab,
    build_portfolio_tab,
    build_methodology_tab,
    build_ml_insights_tab,
    build_agent_status_tab,
    build_agent_monitor_tab,
    build_optimization_tab,
)

__all__ = [
    "build_stress_tab",
    "build_risk_tab",
    "build_backtest_tab",
    "build_daily_brief_panel",
    "build_corr_vol_tab",
    "build_pnl_tracker_tab",
    "build_signal_decay_tab",
    "build_regime_timeline_tab",
    "build_dss_tab",
    "build_portfolio_tab",
    "build_methodology_tab",
    "build_ml_insights_tab",
    "build_agent_status_tab",
    "build_agent_monitor_tab",
    "build_optimization_tab",
]
