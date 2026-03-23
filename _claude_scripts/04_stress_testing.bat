@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM: QuantEngine in analytics/stat_arb.py computes sector ETF signals using PCA residuals, macro betas (TNX, DXY, SPY), VIX regime classification, and cross-sectional z-scores. Settings are in config/settings.py with Pydantic validation.

TASK: Create analytics/stress.py - a stress testing and scenario analysis module for the sector book.

PHILOSOPHY: PM needs to understand how current positions/signals behave under shock scenarios before committing capital. This is forward-looking risk assessment, not backtest.

WHAT TO BUILD:

A) SCENARIO DEFINITIONS:
- `ScenarioDefinition` dataclass: name, description, shocks dict
- Pre-built institutional scenarios as class constants:
  1. RATES_SHOCK_UP: TNX +100bps, DXY +3%, SPY -5%, VIX +10
  2. RATES_SHOCK_DOWN: TNX -75bps, DXY -2%, SPY +3%, VIX -5
  3. RISK_OFF_MILD: SPY -8%, VIX +15, credit spreads +50bps
  4. RISK_OFF_SEVERE: SPY -15%, VIX +30, credit spreads +150bps (crisis)
  5. USD_SURGE: DXY +5%, commodities -8%, EM risk-off
  6. STAGFLATION: TNX +80bps, SPY -10%, oil +20%, VIX +20
  7. TECH_SELLOFF: XLK -15%, rotation to value sectors
  8. ENERGY_SPIKE: XLE +20%, consumer sectors -5%, TNX +30bps
  9. CALM_ENVIRONMENT: VIX -8, SPY +3%, correlations compress
  10. CUSTOM: user-defined via dict

B) STRESS ENGINE - StressEngine class:
- `__init__(settings, prices_df, macro_betas_df)`
  - macro_betas_df: sector x factor betas from QuantEngine._compute_macro_betas()

- `apply_scenario(scenario: ScenarioDefinition, current_positions_df: pd.DataFrame) -> ScenarioResult`
  - For each sector in positions:
    - Compute estimated P&L = sum(beta_factor * shock_factor) for all factors
    - Use EWMA vol to compute shock-adjusted z-score range
    - Flag if position would enter stop-loss territory
  - Aggregate book-level: total estimated P&L bps, worst sector, best sector
  - Compute correlation stress: do positions become more correlated under shock?

- `run_all_scenarios(current_positions_df) -> list[ScenarioResult]`
  - Run all pre-built scenarios
  - Return ranked by worst-case book P&L

- `compute_tail_risk(prices_df, percentile=5) -> TailRiskMetrics`
  - Historical simulation: for each rolling 21-day window, compute book return
  - Report VaR(5%), CVaR(5%), max drawdown, worst single-week
  - Parametric VaR using EWMA covariance matrix

- `correlation_stress_test(returns_df, shock_correlation=0.85) -> pd.DataFrame`
  - Re-compute book risk assuming all sector correlations jump to shock_correlation
  - Compare stressed vs normal diversification benefit (ratio of portfolio vol)

C) KEY DATACLASSES:
- ScenarioDefinition: name, description, factor_shocks (dict), created_at
- ScenarioResult: scenario_name, sector_pnl_bps (dict), book_total_bps, worst_sector, best_sector, correlation_delta, flags (list of warnings)
- TailRiskMetrics: var_5pct_bps, cvar_5pct_bps, max_drawdown_bps, worst_21d_bps, parametric_var_bps, computation_date

D) DESIGN:
- Pure analytics - no UI code, no side effects
- All functions accept DataFrames and return dataclasses/DataFrames
- Vectorized operations only
- Handle NaN gracefully
- Logging for all scenario runs
- Full type hints and docstrings

Read analytics/stat_arb.py to understand the exact structure of macro_betas and the factor names (TNX, DXY, SPY) before writing. Create analytics/stress.py now."
