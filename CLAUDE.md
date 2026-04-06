# SRV Quantamental DSS — Architecture Guide

## System Purpose

Institutional-grade Decision Support System for a sector-focused hedge fund.

**Primary alpha source:** Sector momentum (Sharpe 1.92 after auto-tuning)
- Beta-to-SPY cross-sectional ranking (IC = +0.065, t = 2.58)
- Low idiosyncratic volatility anomaly (IC = -0.037, t = -1.82)
- IC-weighted composite with vol-scaled sizing

**Secondary:** Short Vol / Dispersion — sell implied correlation, profit from sector dispersion

**Key finding:** PCA residual mean-reversion is NOT valid for sector ETFs (Hurst ≈ 1.0, IC ≈ 0). Momentum-based strategies are the only empirically validated alpha source.

## Quick Start

```bash
# Pipeline (deterministic, with run manifest)
python -m services.pipeline                    # Full run with artifact tracking
python -m services.pipeline --dry-run          # Show config hash + stages
python -m services.pipeline --list-runs        # Compare previous runs

# Dashboard
python main.py                                 # → http://localhost:8050

# Agents (autonomous operation)
python agents/orchestrator.py                  # Daemon mode (20 tasks, 24/7)
python agents/orchestrator.py --once           # Single cycle
python agents/orchestrator.py --status         # Show status

# Scheduler setup (Windows — run once as admin)
setup_scheduler.bat                            # Registers daily tasks

# Paper Trading
python analytics/paper_trader.py --update      # Daily update
python analytics/paper_trader.py --status      # Show portfolio

# Validation
python -c "from analytics.validation import ValidationSuite; ..."
```

## Architecture (101K+ LOC, 193 files)

```
main.py (1,840 lines)
│  Dashboard: institutional design, 18 tabs grouped by workflow
│  Trading → Risk → Analytics → Research
│
├── services/                          — Service layer (clean separation)
│   ├── pipeline.py (363)             — Deterministic pipeline + run manifests
│   ├── engine_service.py (622)       — 20-step analytics orchestration
│   ├── run_context.py (106)          — Run lineage metadata
│   ├── data_loader.py (155)          — Centralized JSON/agent I/O
│   └── alerting.py (240)            — Unified alerts (Slack/file/bus)
│
├── analytics/                         — 45+ quant modules (~35K LOC)
│   ├── stat_arb.py (2,305)           — Core QuantEngine (PCA, z-scores)
│   ├── scoring.py (271)              — Typed facade over master_df
│   ├── alpha_sources.py (523)        — Empirically validated alpha signals
│   ├── validation.py (690)           — Institutional validation framework
│   ├── quant_review.py (357)         — Signal quality audit + methodology fixes
│   ├── regime_classifier.py (185)    — Market regime classification
│   ├── decision_engine.py (172)      — Score → PM decision conversion
│   ├── methodology_lab.py (1,623)    — 26 strategies incl. RelativeMomentum
│   ├── correlation_structure.py      — DCC-GARCH conditional correlation
│   ├── pair_scanner.py (726)         — Kalman + Johansen + basket scanner
│   ├── stress.py                     — Deterministic + Monte Carlo stress
│   ├── portfolio_risk.py             — VaR, CVaR, MCTR, iVaR, DaR
│   ├── risk_decomposition.py (369)   — Euler marginal risk + factor VaR
│   ├── tail_risk.py                  — EVT (GPD), Hill estimator
│   ├── optimizer.py (694)            — Risk-parity, MV, Black-Litterman
│   ├── leverage_engine.py (568)      — Kelly, GARCH vol, asymmetric DD
│   ├── market_microstructure.py (413)— Spread, impact, liquidity scoring
│   ├── trade_monitor.py (842)        — Trailing stops, Greeks, aging
│   ├── paper_trader.py (885)         — Dual-source (DSS + momentum) trading
│   └── ... (30+ more modules)
│
├── agents/                            — 20 autonomous agents (~33K LOC)
│   ├── orchestrator.py (1,282)       — Master scheduler with governance
│   ├── auto_improve/ (2,848)         — Parameter tuning + promotion
│   └── ... (11 agent directories)
│
├── ui/                                — Dashboard components
│   ├── analytics_tabs.py (5,220)     — 15 tab builders
│   ├── tab_renderer.py (566)         — Tab dispatch + TabContext
│   ├── components.py (234)           — Shared KPI/chart library
│   └── tabs/ (45)                    — Tab package re-exports
│
├── db/                                — DuckDB persistence (schema v8)
│   ├── repository.py (450)           — Canonical data access + quality
│   ├── schema.py                     — 19 tables + migrations
│   └── writer.py / reader.py         — Structured read/write
│
├── execution/
│   └── ibkr_gateway.py (1,484)       — IBKR + options builder + smart routing
│
├── tests/                             — 25 test files + conftest.py
│   ├── conftest.py                   — Shared fixtures (fake_prices, settings)
│   ├── test_new_analytics.py (461)   — 45+ tests for expanded modules
│   └── test_integration_pipeline.py  — 15 E2E tests
│
├── .github/workflows/ci.yml          — GitHub Actions CI
└── data/runs/                         — Pipeline artifact directories
```

## Key Design Decisions

1. **Service Layer** — main.py delegates to EngineService (20 steps). No business logic in dashboard.
2. **RunContext** — Every run gets UUID + config hash. Persisted to DuckDB `analytics.run_contexts`.
3. **Pipeline** — `services/pipeline.py` creates artifact directories with config snapshots + manifests.
4. **Typed Scoring** — `ScoringResult` wraps raw master_df with typed dataclasses.
5. **Validation Framework** — Purged walk-forward (21d purge + 5d embargo) with 4 tests: OOS, cost, regime, stability.
6. **Momentum > MR** — PCA residual z-score has zero predictive power (IC ≈ 0, Hurst ≈ 1.0). Beta-momentum (IC = 0.065) is the validated alpha.
7. **Auto-improve** — Agents test parameter changes via backtest, promote winners to `.env.auto_improve`.

## Pipeline Stages

```
data_ingest → quant_engine → scoring → signals → risk → stress →
correlation → options → portfolio → tracking → validation → artifacts
```

Each stage: timed, error-captured, persisted to RunContext + DuckDB.

## Agent Schedule (20 tasks)

| Time | Agent | Depends On |
|------|-------|-----------|
| every 6h | data_refresh | — |
| every 30m | vix_monitor | — |
| every 2h | regime_forecaster | — |
| 06:00 | methodology | data_refresh |
| 06:05 | data_scout | — |
| 06:15 | pair_scan | data_refresh |
| 06:30 | morning_brief | methodology |
| 07:00 | optimizer | methodology |
| 07:30 | auto_improve | methodology + optimizer |
| 08:00 | alpha_decay | methodology |
| Mon 08:00 | math | — |
| 09:00 | architect | optimizer |
| Mon 09:00 | weekly_backtest | methodology |
| Mon 10:00 | alpha_research | weekly_backtest |
| every 15m | health_check | — |
| 16:30 | portfolio_construction | methodology |
| 16:45 | risk_guardian | portfolio_construction |
| 17:00 | paper_trading | data_refresh |
| 17:30 | evening_recap | paper_trading |
| 18:00 | daily_report | evening_recap |

## Settings

Pydantic BaseSettings from `.env` + `.env.auto_improve`:
- `FMP_API_KEY` — Financial Modeling Prep (only data source)
- `SLACK_WEBHOOK_URL` — Notifications (optional)
- `MOMENTUM_LOOKBACK`, `MOMENTUM_TOP_N`, `MOMENTUM_REBAL_DAYS` — Auto-tuned
- `USE_DCC_GARCH`, `DCC_A_PARAM`, `DCC_B_PARAM` — Correlation model

## Testing

```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_new_analytics.py  # Expanded module tests
python tests/test_integration_pipeline.py     # Direct E2E execution
```
