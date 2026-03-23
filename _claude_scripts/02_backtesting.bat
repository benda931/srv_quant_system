@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM OVERVIEW:
- analytics/stat_arb.py (~2017 lines): QuantEngine class - computes PCA residuals, z-scores, EWMA vol, regime classification, conviction scores across 11 S&P 500 sector ETFs vs SPY
- analytics/attribution.py: AttributionResult dataclass with 5 scoring layers (SDS, FJS, MSS, STF, MC)
- config/settings.py: Pydantic settings with all analytical parameters
- data/pipeline.py: DataLakeManager fetching from FMP API, storing parquets
- Parquet data: data_lake/parquet/prices.parquet (daily OHLCV + macro + vol), fundamentals.parquet, weights.parquet

TASK: Create analytics/backtest.py - a vectorized signal backtesting and performance attribution module.

STRICT REQUIREMENTS:
1. NO auto-trading, NO execution logic - this is PM decision support only
2. Modular, institutional-grade code - no monolithic functions
3. Must integrate cleanly with existing QuantEngine output (master_df with 70+ columns)
4. Hebrew-friendly: all display strings use English but design allows RTL UI integration

WHAT TO BUILD - BacktestEngine class with these capabilities:

A) SIGNAL PERFORMANCE ANALYSIS:
- `run_signal_backtest(master_df_history: list[pd.DataFrame], holding_period_days: [5,10,21,63]) -> BacktestResult`
  - For each sector, for each historical signal (ENTER/WATCH/REDUCE/AVOID), track forward returns
  - Compute hit rate, avg return, Sharpe, max drawdown per signal type
  - Use log returns from prices.parquet

B) FACTOR ATTRIBUTION:
- `compute_factor_pnl_attribution(returns_df, factor_exposures_df) -> FactorAttribution`
  - Decompose realized PnL into: SDS contribution, FJS contribution, MSS contribution, regime beta
  - Use Brinson-Hood-Beebower style attribution adapted for long/short sector positioning

C) WALK-FORWARD VALIDATION:
- `walk_forward_oos_test(prices_df, n_splits=8, min_train_years=3) -> WalkForwardResult`
  - TimeSeriesSplit-style expanding window
  - Re-run PCA residuals OOS in each fold
  - Track z-score predictive power (IC - Information Coefficient) across folds
  - Report IC mean, IC std, IR (Information Ratio of ICs)

D) REGIME-CONDITIONAL PERFORMANCE:
- `regime_conditional_returns(master_df_history, regime_col='market_state') -> RegimePerformance`
  - Performance of each signal type split by market regime (CALM/NORMAL/TENSION/CRISIS)
  - Shows whether signals are regime-dependent

E) KEY DATACLASSES:
- BacktestResult: dataclass with fields for each signal type's metrics
- FactorAttribution: factor contributions with t-stats
- WalkForwardResult: per-fold IC, overall IR, stability score
- RegimePerformance: per-regime Sharpe, hit-rate, avg-holding-period-return

F) DESIGN PRINCIPLES:
- All functions accept DataFrames, return dataclasses or DataFrames
- Vectorized pandas/numpy operations only - no Python loops on rows
- NaN-safe: handle missing data gracefully with warnings
- Logging via standard library logging module
- Type hints throughout
- Docstrings with parameter descriptions
- No external dependencies beyond: pandas, numpy, scipy, scikit-learn (already in requirements)

Read the existing files first (especially stat_arb.py and attribution.py) to understand the exact column names and data structures before writing. Create analytics/backtest.py now."
