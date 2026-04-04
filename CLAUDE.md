# SRV Quantamental DSS — Architecture Guide

## System Purpose

Institutional-grade Decision Support System for a sector-focused hedge fund.
Two co-equal alpha sources:
1. **Short Vol / Dispersion** — sell implied correlation, profit from sector dispersion
2. **Sector Relative Value** — momentum + mean-reversion on 11 SPDR sector ETFs

## Quick Start

```bash
# Dashboard (primary interface)
python main.py                    # → http://localhost:8050

# Pipeline (daily data refresh + analytics)
python scripts/run_all.py         # Full pipeline (19 steps)
python scripts/run_all.py --force-refresh  # Force FMP re-fetch

# Agents (autonomous operation)
python agents/orchestrator.py     # Daemon mode (24/7)
python agents/orchestrator.py --once  # Single cycle
python agents/orchestrator.py --status  # Show status

# Paper Trading
python analytics/paper_trader.py --update  # Daily update
python analytics/paper_trader.py --status  # Show portfolio

# Daily Report
python scripts/daily_report.py --preview  # Preview report
```

## Architecture

```
main.py (1,652 lines) — Dash dashboard + callbacks
│
├── services/                      — Service layer
│   ├── run_context.py             — Run lineage metadata (RunContext)
│   ├── engine_service.py          — 19-step analytics orchestration (EngineService)
│   ├── data_loader.py             — Centralized JSON/agent I/O (DataLoader)
│   └── alerting.py                — Unified alert routing (Slack/file/bus)
│
├── analytics/                     — 42+ quant modules (~30K LOC)
│   ├── stat_arb.py                — Core QuantEngine (PCA, z-scores, conviction)
│   ├── correlation_structure.py   — DCC-GARCH conditional correlation
│   ├── correlation_engine.py      — CRP, dispersion surface, short-vol signal
│   ├── signal_stack.py            — 4-layer signal scoring
│   ├── signal_regime_safety.py    — Regime safety gating
│   ├── signal_mean_reversion.py   — OU MLE, Hurst, variance ratio
│   ├── trade_structure.py         — Trade construction + Greeks + hedge recalc
│   ├── trade_monitor.py           — Exit management, trailing stops, aging
│   ├── stress.py                  — Deterministic + Monte Carlo stress
│   ├── portfolio_risk.py          — VaR, CVaR, MCTR, risk budgeting
│   ├── risk_decomposition.py      — Euler marginal risk, factor VaR
│   ├── tail_risk.py               — EVT (GPD), Hill estimator, regime tails
│   ├── optimizer.py               — Risk-parity, MV, Black-Litterman
│   ├── leverage_engine.py         — Kelly, GARCH vol, asymmetric DD
│   ├── backtest.py                — Walk-forward backtester
│   ├── methodology_lab.py         — 26 strategies (incl. RelativeMomentum Sharpe 1.52)
│   ├── alpha_research.py          — OOS validation, ensemble, stability
│   ├── pair_scanner.py            — Kalman, Johansen, basket scanner
│   ├── dispersion_backtest.py     — Variance swap simulation + Greeks
│   ├── options_engine.py          — IV surface, VRP, VVIX, Skew, timing
│   ├── ml_signals.py              — Ensemble stacking (GBM+LR+RF)
│   ├── ml_regime_forecast.py      — Ensemble regime predictor + auto-retrain
│   ├── feature_engine.py          — 38+ features + interactions
│   ├── paper_trader.py            — Dual-source paper trading + slippage
│   ├── pnl_tracker.py             — P&L + factor attribution
│   ├── signal_decay.py            — IC decay curves (exp/Weibull)
│   ├── regime_alerts.py           — Markov transition + crisis probability
│   ├── attribution.py             — Brinson + factor decomposition
│   ├── market_microstructure.py   — Spread, impact, liquidity scoring
│   └── dss_backtest.py            — Param sweep + stress overlay
│
├── agents/                        — 19 autonomous agents (~33K LOC)
│   ├── orchestrator.py            — Master scheduler (19 tasks, parallel exec)
│   ├── auto_improve/engine.py     — Parameter tuning + promotion
│   ├── methodology/               — Strategy evaluation + governance
│   ├── optimizer/                  — Bayesian parameter search
│   ├── math/                      — Formula improvement via LLM
│   ├── architect/                  — Code improvement scanning
│   ├── risk_guardian/              — Risk veto + exposure limits
│   ├── regime_forecaster/          — ML regime prediction
│   ├── alpha_decay/                — Signal staleness monitoring
│   ├── data_scout/                 — External data quality scanning
│   ├── portfolio_construction/     — Portfolio assembly
│   └── execution/                  — Trade execution (IBKR)
│
├── ui/                            — Dashboard components
│   ├── analytics_tabs.py          — 15 tab builders (5,159 lines)
│   ├── tab_renderer.py            — Tab dispatch + TabContext
│   ├── components.py              — Shared KPI/chart/format library
│   └── tabs/__init__.py           — Tab package re-exports
│
├── db/                            — DuckDB persistence (schema v8)
│   ├── schema.py                  — 19 tables + migrations
│   ├── writer.py                  — Write methods (upsert pattern)
│   ├── reader.py                  — Read methods (latest semantics)
│   ├── repository.py              — Canonical data access + quality checks
│   └── connection.py              — Singleton thread-safe connection
│
├── data/                          — Data pipeline
│   └── pipeline.py                — FMP API + yfinance fallback
│
├── execution/                     — Trade execution
│   └── ibkr_gateway.py            — IBKR gateway + options builder + smart routing
│
├── config/
│   └── settings.py                — Pydantic settings (reads .env + .env.auto_improve)
│
├── reports/
│   └── dss_brief_generator.py     — PM brief with 10 sections
│
└── scripts/
    ├── run_all.py                 — Master pipeline (19 steps)
    └── daily_report.py            — Automated PM daily report
```

## Key Design Decisions

1. **Service Layer** — main.py delegates to EngineService (487 lines) which runs 19 analytics steps. No business logic in the dashboard.

2. **RunContext** — Every pipeline run gets a unique RunContext (run_id, uuid, regime, steps, timing). Persisted to DuckDB `analytics.run_contexts`.

3. **DuckDB as canonical store** — Append-only, idempotent writes (INSERT OR REPLACE). Schema versioned with migrations. Agent JSON files are shadowed to `analytics.agent_snapshots`.

4. **Momentum > Mean-Reversion** — PCA z-score mean-reverts in ~1 day (HL=1d) but strategies held 20-45d → negative Sharpe. RelativeMomentum (Moskowitz 1999) gives Sharpe 1.52.

5. **Auto-improve writes to .env.auto_improve** — Agents test parameter changes via backtest, promote winners to .env.auto_improve which settings.py reads on next load.

## Data Flow

```
FMP API / yfinance → Parquet files → DuckDB → QuantEngine → master_df
                                                    ↓
                                        EngineService (19 steps)
                                                    ↓
                                    ┌─── Stress/Risk/Correlation
                                    ├─── Signal Stack → Trade Tickets
                                    ├─── Paper Trader → P&L
                                    ├─── ML Models → Quality Scores
                                    └─── Data Quality → Repository
                                                    ↓
                                            Dashboard (18 tabs)
```

## Settings

Pydantic BaseSettings loads from `.env` + `.env.auto_improve`:
- `FMP_API_KEY` — Financial Modeling Prep API key
- `SLACK_WEBHOOK_URL` — Slack notifications (optional)
- `OPENAI_API_KEY` — GPT for math agent (optional)
- Momentum params: `MOMENTUM_LOOKBACK`, `MOMENTUM_TOP_N`, `MOMENTUM_REBAL_DAYS`
- DCC-GARCH: `USE_DCC_GARCH`, `DCC_A_PARAM`, `DCC_B_PARAM`

## Testing

```bash
python -m pytest tests/          # Run all tests
python -m pytest tests/test_portfolio_risk.py  # Specific test
```

## Agent Schedule (Orchestrator)

| Agent | Schedule | Depends On |
|-------|----------|-----------|
| data_refresh | every 6h | — |
| vix_monitor | every 30min | — |
| data_scout | 06:05 weekdays | — |
| methodology | 06:00 weekdays | data_refresh |
| pair_scan | 06:15 weekdays | data_refresh |
| regime_forecaster | every 2h market | — |
| optimizer | 07:00 weekdays | methodology |
| auto_improve | 07:30 weekdays | methodology, optimizer |
| alpha_decay | 08:00 weekdays | methodology |
| math | Monday 08:00 | — |
| architect | 09:00 weekdays | optimizer |
| weekly_backtest | Monday 09:00 | methodology |
| alpha_research | Monday 10:00 | weekly_backtest |
| portfolio_construction | 16:30 weekdays | methodology, optimizer |
| risk_guardian | 16:45 weekdays | portfolio_construction |
| paper_trading | 17:00 weekdays | data_refresh |
| morning_brief | 06:30 weekdays | methodology |
| evening_recap | 17:30 weekdays | paper_trading |
| health_check | every 15min | — |
