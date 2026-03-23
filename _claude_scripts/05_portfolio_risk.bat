@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM: QuantEngine (analytics/stat_arb.py) generates per-sector signals and synthetic Greeks (delta, vega, beta equivalents). The system targets 12% portfolio vol (from settings). No portfolio-level risk aggregation currently exists.

TASK: Create analytics/portfolio_risk.py - portfolio-level risk engine for the full sector book.

PHILOSOPHY: PM needs a single-view of aggregate book risk - not just per-sector signals but the combined exposure, concentration, and risk budget. Institutional standard: risk decomposed by factor, sector, regime.

WHAT TO BUILD:

A) PORTFOLIO RISK ENGINE - PortfolioRiskEngine class:
- `__init__(settings, prices_df)` - settings from config, prices_df from parquet

- `compute_covariance_matrix(returns_df, method='ewma', halflife=42) -> pd.DataFrame`
  - Methods: 'ewma' (exponential), 'sample' (rolling 252d), 'shrinkage' (Ledoit-Wolf)
  - Ledoit-Wolf via sklearn.covariance.LedoitWolf
  - Return sector x sector annualized covariance matrix

- `compute_portfolio_vol(weights: pd.Series, cov_matrix: pd.DataFrame) -> float`
  - Annualized portfolio volatility given weights (can be long/short)
  - weights: {sector: weight} where weight is signed (-1 to +1)

- `compute_marginal_contribution_to_risk(weights, cov_matrix) -> pd.Series`
  - MCTR per sector: partial derivative of portfolio vol w.r.t. each weight
  - Percent contribution to risk (PCTR) = weight * MCTR / portfolio_vol

- `compute_factor_var(weights, macro_betas_df, factor_cov_matrix) -> FactorVaRResult`
  - Factor-based VaR decomposition
  - Factors: SPY_beta, TNX_beta, DXY_beta
  - Total VaR = systematic VaR + idiosyncratic VaR
  - Report factor contribution to total VaR

- `compute_concentration_metrics(weights) -> ConcentrationMetrics`
  - HHI (Herfindahl-Hirschman Index) of abs weights
  - Effective N (1/HHI)
  - Top-3 concentration (% of gross exposure)
  - Long/short gross and net exposure
  - Sector-level limit breach flags (max_single_name from settings)

- `compute_correlation_contribution(weights, corr_matrix) -> pd.DataFrame`
  - Pairwise correlation contribution to portfolio risk
  - Which sector pairs are the largest risk concentrators?

- `build_risk_report(weights, prices_df, macro_betas_df) -> PortfolioRiskReport`
  - Comprehensive risk snapshot combining all above
  - Compute: portfolio vol, target vol utilization, factor VaR, concentration, regime-adjusted risk

B) RISK BUDGET MODULE:
- `RiskBudget` dataclass: target_vol, current_vol, utilization_pct, headroom_bps, budget_per_sector
- `compute_risk_budget(weights, cov_matrix, target_vol_pct) -> RiskBudget`
  - How much risk budget is used vs available?
  - Sector-level budget allocation based on MCTR

C) KEY DATACLASSES:
- FactorVaRResult: total_var_bps, systematic_var_bps, idio_var_bps, factor_contributions (dict), confidence_level
- ConcentrationMetrics: hhi, effective_n, top3_concentration_pct, gross_exposure, net_exposure, long_exposure, short_exposure, limit_breaches (list)
- PortfolioRiskReport: as_of, portfolio_vol_pct, vol_utilization_pct, factor_var, concentration, mctr (Series), risk_budget, warnings (list)

D) INTEGRATION:
- Must accept settings from config/settings.py (target_vol=12%, max_single_name=20%)
- Read settings.py to understand exact field names before coding
- No UI code - pure analytics
- Vectorized, NaN-safe, typed, documented
- Log warnings when limits are breached

Read config/settings.py and analytics/stat_arb.py first to understand settings fields and macro_betas structure. Create analytics/portfolio_risk.py now."
