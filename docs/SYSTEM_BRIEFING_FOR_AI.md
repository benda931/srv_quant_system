# SRV Quantamental DSS — Full System Briefing

## Purpose of This Document
This document is a complete technical briefing for an AI assistant (GPT/Claude) to fully understand the SRV Quant System — its architecture, purpose, capabilities, data flow, agent system, and current state. After reading this, you should be able to answer any question about the system, suggest improvements, debug issues, and contribute code.

---

## 1. What Is This System?

**SRV Quantamental DSS** (Decision Support System) is a professional-grade quantitative trading research and signal generation platform for a discretionary quant hedge fund.

**Core Thesis:** Short Volatility via Correlation and Dispersion Trades on S&P 500 sector ETFs.

**What it does:**
- Analyzes 11 S&P 500 sector ETFs (XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU) relative to SPY
- Detects statistical mispricings using PCA residuals, correlation structure analysis, and regime detection
- Generates trade signals with 4-layer conviction scoring
- Constructs trades (Sector RV, Dispersion, RV Spread)
- Monitors positions and generates exit signals
- Runs paper trading with realistic slippage and commissions
- Self-improves via 10 autonomous AI agents

**What it does NOT do (yet):**
- It does NOT execute live trades automatically (paper only)
- It does NOT provide financial advice
- It is NOT connected to a live brokerage (IBKR gateway exists but runs in dry-run mode)

---

## 2. Technology Stack

```
Language:        Python 3.11+
Web Framework:   Dash (Plotly) with dash-bootstrap-components
Database:        DuckDB (analytics) + SQLite (journal) + JSON (agent state)
Data Source:     Financial Modeling Prep (FMP) API
AI Integration:  OpenAI GPT-4o API + Anthropic Claude API
Optimization:    Optuna (Bayesian), scipy (SLSQP), sklearn (GBM, PCA)
Version Control: Git → GitHub (github.com/benda931/srv_quant_system)
```

---

## 3. Directory Structure

```
srv_quant_system/
├── analytics/              # Core quant engine (38 modules, ~25K LOC)
│   ├── stat_arb.py            # QuantEngine: PCA residuals, regime detection, conviction
│   ├── signal_stack.py        # 4-layer signal scoring (Distortion × Dislocation × MR × Safety)
│   ├── signal_mean_reversion.py  # Layer 3: OU, ADF, Hurst mean-reversion quality
│   ├── signal_regime_safety.py   # Layer 4: Regime safety gate (VIX, credit, correlation)
│   ├── correlation_engine.py     # Frobenius distortion, market-mode, CoC instability
│   ├── trade_structure.py     # Trade construction: Sector RV, Dispersion, RV Spread
│   ├── trade_monitor.py       # 6 exit signal types, trade health scoring
│   ├── paper_trader.py        # Paper portfolio with daily P&L tracking
│   ├── portfolio_risk.py      # VaR, CVaR, Ledoit-Wolf covariance, HHI
│   ├── tail_risk.py           # Kupiec VaR backtest, Cornish-Fisher ES
│   ├── leverage_engine.py     # Kelly criterion, vol targeting, DD deleverage
│   ├── multi_strategy.py      # MR + Dispersion + MomentumMR ensemble
│   ├── methodology_lab.py     # 19 strategy backtester with regime analysis
│   ├── bayesian_optimizer.py  # Optuna 3-objective (Sharpe/MaxDD/IC)
│   ├── feature_engine.py      # 200+ features: z-scores, momentum, vol, RSI, correlation
│   ├── alpha_model.py         # Walk-forward GBM with 35 features
│   ├── options_engine.py      # Black-Scholes, Greeks, implied correlation
│   ├── macro_calendar.py      # FOMC/CPI/NFP event impact scoring
│   ├── dispersion_backtest.py # Dispersion trade P&L simulation
│   ├── performance_metrics.py # Sortino, IR, Tracking Error, Omega
│   ├── stress.py              # 10 deterministic stress scenarios
│   └── backtest.py            # Walk-forward IC/Sharpe validation
│
├── agents/                 # 10 autonomous AI agents (~10.5K LOC)
│   ├── methodology/           # Strategy evaluation + walk-forward validation
│   ├── optimizer/             # Parameter tuning + Bayesian search
│   ├── math/                  # Formula improvement via dual LLM
│   ├── auto_improve/          # Orchestrator: full improvement cycle
│   ├── risk_guardian/         # Independent risk monitor with VETO power
│   ├── execution/             # Signal-to-trade with slippage model
│   ├── regime_forecaster/     # HMM regime prediction + leading indicators
│   ├── alpha_decay/           # Strategy death detection
│   ├── portfolio_construction/ # Mean-variance + risk parity optimization
│   ├── data_scout/            # FRED macro + anomaly detection
│   └── shared/                # Registry, scheduler, bus, GPT bridge
│
├── execution/              # IBKR gateway (dry-run mode)
│   ├── ibkr_gateway.py        # TWS/IB Gateway connection, order management
│   └── order_manager.py       # Order tracking, fill updates, daily summary
│
├── ui/                     # Dashboard (18 tabs)
│   ├── analytics_tabs.py      # All tab builders (~2,600 LOC)
│   ├── panels.py              # Reusable UI components
│   └── scanner_pro.py         # Sector scanner with WHY explanations
│
├── config/
│   └── settings.py            # ~120 Pydantic-validated parameters
│
├── data_ops/               # Data pipeline
│   ├── pipeline.py            # FMP API fetcher with retry/backoff
│   ├── orchestrator.py        # Data health scoring
│   ├── pre_write_validator.py # Schema + range validation before writes
│   └── freshness.py           # Staleness detection
│
├── db/
│   ├── schema.py              # DuckDB migrations (v1-v5)
│   ├── writer.py              # Atomic writes to DuckDB
│   ├── audit.py               # Immutable audit trail with SHA256 hash chain
│   └── connection.py          # Connection management
│
├── scripts/
│   ├── run_all.py             # 16-step pipeline orchestrator
│   ├── ensemble_backtest.py   # Multi-strategy comparison
│   └── claude_loop.py         # Claude API integration for agents
│
├── data_lake/parquet/      # Historical data (2016-2026)
│   ├── prices.parquet         # 2,622 days × 19 tickers
│   └── fundamentals.parquet   # PE, EPS, Market Cap
│
├── tests/                  # 408 tests across 25 test files
├── docs/                   # Architecture, data dictionary, runbook
├── main.py                 # Dash app entry point (~2,000 LOC)
└── README.md               # Project overview
```

---

## 4. Core Quant Engine

### 4.1 Data Flow

```
FMP API → prices.parquet (19 tickers, 10yr daily)
                ↓
         QuantEngine.run()
                ↓
    ┌───────────────────────┐
    │  PCA Residuals (OOS)  │  Out-of-sample rolling PCA on sector relative returns
    │  252d train window    │  3 components, residual = actual - reconstructed
    │  60d z-score window   │  z_t = (resid_t - μ) / σ
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │  Regime Detection     │  Based on VIX level + credit spread + correlation
    │  CALM / NORMAL /      │  VIX < 15 = CALM, < 22 = NORMAL
    │  TENSION / CRISIS     │  < 30 = TENSION, ≥ 30 = CRISIS
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │  4-Layer Signal Stack │
    │  S = S_dist × S_disloc│  Layer 1: Correlation distortion score
    │    × S_mr × S_safe    │  Layer 2: Residual dislocation (|z|/Z_cap)
    │                       │  Layer 3: Mean-reversion quality (OU + ADF + Hurst)
    │                       │  Layer 4: Regime safety gate
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │  master_df (133 cols) │  Per-sector: direction, conviction, z-score,
    │  11 rows (sectors)    │  half-life, fundamentals, attribution, Greeks
    └───────────────────────┘
```

### 4.2 Signal Stack Formula

```
Score_j = S^dist × S^disloc_j × S^mr_j × S^safe

Layer 1 (Distortion):   S^dist = σ(a1·z_Frobenius + a2·rank(market_mode) + a3·z_CoC)
Layer 2 (Dislocation):  S^disloc = min(1, |z_residual| / Z_cap)
Layer 3 (Mean-Rev):     S^mr = w1·OU_quality + w2·ADF_quality + w3·Hurst_quality
Layer 4 (Safety):       S^safe = f(VIX, credit_stress, correlation_spike, regime_prob)

Entry: Score_j ≥ 0.15 AND all gates pass
Exit:  z compressed OR time stop OR regime deterioration OR stop-loss
```

### 4.3 Regime-Adaptive Parameters

The system uses different parameters per regime:

| Parameter | CALM | NORMAL | TENSION | CRISIS |
|-----------|------|--------|---------|--------|
| z_entry   | 0.5  | 0.7    | 1.0     | ∞ (no entry) |
| sizing    | 1.3x | 1.0x   | 0.5x    | 0.0x |
| leverage  | 3.0x | 2.0x   | 1.2x    | 0.0x |

### 4.4 Alpha Research Results

Best strategy: **AlphaWhitelistMR** (Mean Reversion on XLC, XLF, XLI, XLP, XLU)

```
In-Sample Sharpe:   0.891 (1,837 days)
Out-of-Sample Sharpe: 0.885 (787 days)
OOS/IS Ratio:       0.99 (excellent stability)
Win Rate:           57.0%
Annualized Return:  6.62%
Max Drawdown:       -14.3%
```

Key insight: MR works best in CALM (Sharpe +0.66) and TENSION (Sharpe +0.68), but DESTROYS value in CRISIS (Sharpe -0.78). The regime kill-switch is essential.

---

## 5. The 10-Agent System

### Architecture

```
┌─────────────────────────────────────────────┐
│           Auto-Improve (Orchestrator)        │
│  Methodology → Optimizer → Math → Validate   │
└─────────┬───────────┬───────────┬───────────┘
          │           │           │
┌─────────▼─┐ ┌───────▼───┐ ┌────▼────────┐
│Methodology│ │ Optimizer │ │    Math     │
│1,685 LOC  │ │ 1,819 LOC │ │  695 LOC   │
└───────────┘ └───────────┘ └─────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│         Signal Stack + Feature Engine        │
└─────────┬───────────────────────┬───────────┘
          │                       │
┌─────────▼─────┐ ┌──────────────▼──────────┐
│Portfolio Const.│ │   Regime Forecaster    │
│  367 LOC      │ │      881 LOC           │
└───────┬───────┘ └────────────┬────────────┘
        │                      │
┌───────▼──────────────────────▼──────────────┐
│            Execution Agent (1,301 LOC)       │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│     Risk Guardian (1,154 LOC) — VETO POWER   │
└─────────────────────────────────────────────┘
        │                       │
┌───────▼───────┐ ┌─────────────▼─────────────┐
│ Alpha Decay   │ │      Data Scout           │
│  356 LOC      │ │       671 LOC             │
└───────────────┘ └───────────────────────────┘
```

### Daily Schedule

```
05:00  Data Scout          → Scans FRED macro, anomalies, correlation breaks
06:00  Methodology         → Evaluates all 19 strategies with walk-forward CV
06:30  Alpha Decay         → Checks strategy health (HEALTHY/EARLY_DECAY/DECAYING/DEAD)
07:00  Optimizer           → Bayesian search, sensitivity analysis, GPT brainstorming
07:30  Regime Forecaster   → HMM prediction, 10 leading indicators, transition matrix
08:00  Math (weekly)       → Formula improvements via Claude + GPT dual query
08:30  Portfolio Constr.   → Mean-variance + risk parity + conviction blend
09:00  Risk Guardian       → Pre-market risk check (VaR, exposure, Greeks, tail risk)
09:15  Execution           → Place paper trades with TWAP simulation
09:30+ Risk Guardian       → Continuous monitoring every 15 minutes
```

### Agent Communication

Agents communicate via:
1. **Agent Bus** (`agents/shared/agent_bus.py`) — pub/sub messaging
2. **Agent Registry** (`agents/shared/agent_registry.py`) — status tracking (IDLE/RUNNING/COMPLETED/FAILED/STALE)
3. **JSON files** — each agent saves output to its directory
4. **GPT API** — agents query GPT-4o for analysis and suggestions
5. **Claude API** — Math agent uses Claude for mathematical improvements

---

## 6. Dashboard (18 Tabs)

| Tab | Purpose |
|-----|---------|
| Overview | KPI cards, regime status, DSS snapshot, paper portfolio summary |
| DSS | Full Decision Support: safety score, signal stack, trade book, correlation, monitor |
| Scanner | 11 sector cards with conviction, direction, WHY explanations |
| Correlation | Correlation heatmap with annotations, delta tracking |
| Tear Sheet | Per-sector deep dive with PCA residual time series |
| Stress | 10 deterministic scenarios (rates shock, risk-off, stagflation, etc.) |
| Risk | Portfolio risk: VaR, CVaR, HHI, exposure limits |
| Corr&Vol | 3 correlation heatmaps (current, baseline, delta) + time series |
| P&L | P&L tracking |
| Backtest | Walk-forward IC/Sharpe with equity curve and drawdown chart |
| Decay | Signal decay analysis per sector |
| Regime | Regime timeline (Gantt-style) with component scores |
| Health | Data quality monitoring (freshness, NaN rates, validation) |
| Journal | PM trading journal (manual + auto-fill from paper trader) |
| Portfolio | Paper trading portfolio: positions, equity curve, closed trades |
| Methodology | Strategy comparison table (19 strategies), regime breakdown |
| Agents | Agent monitor: status, last run, total runs, parameter audit trail |
| ML Insights | Feature importance, regime forecast donut, model drift status |

---

## 7. Configuration (settings.py)

~120 parameters organized by category:

```python
# Universe
sector_tickers = {XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU}
macro_tickers = {VIX, TLT, HYG, IEF, GLD, TLT, UUP}

# PCA
pca_window = 252          # Rolling PCA train window
n_components = 3          # PCA components to keep
zscore_window = 60        # Z-score lookback

# Signal Stack
signal_a1_frob = 1.0      # Frobenius distortion weight
signal_a2_mode = 0.5      # Market-mode weight
signal_a3_coc = 0.3       # CoC instability weight
signal_z_cap = 3.0        # Z-score cap for dislocation
signal_entry_threshold = 0.15  # Minimum conviction for entry

# Regime
zscore_threshold_calm = 0.5
zscore_threshold_normal = 0.7
zscore_threshold_tension = 1.0
regime_conviction_scale_calm = 1.3
regime_conviction_scale_tension = 0.5

# Risk
max_leverage = 5.0
target_vol_annual = 0.10
kelly_fraction = 0.5
dd_deleverage_start = 0.02
dd_deleverage_full_stop = 0.12
max_single_name_weight = 0.20

# Alpha
mr_whitelist_sectors = [XLC, XLF, XLI, XLP, XLU]
non_whitelist_penalty = 0.3
```

---

## 8. Current Performance Metrics

```
Regime:            TENSION (VIX ~26, Crisis prob ~50%)
Avg Correlation:   0.31 (low → dispersion opportunity)
Signal Stack:      6/11 sectors pass entry threshold
Top Signal:        XLE SHORT (conviction 0.255)

Alpha (OOS):       Sharpe 0.885, WR 57%, Annual Return 6.6%
Risk:              Vol 4.1%, VaR95 0.42%, CVaR95 0.65%
Pipeline:          16 steps, ~25 seconds, 0 failures

Tests:             408 passing, 0 failures
Files:             162 Python files
LOC:               ~75,000
Agents:            10 (all operational)
```

---

## 9. Known Limitations and Gaps

1. **Methodology Lab Sharpe** — The lab's daily P&L accumulation method shows all strategies as negative Sharpe, while per-trade alpha research shows +0.885. These are different measurement methodologies.

2. **Agents Never Ran Continuously** — Agents work with `--once` but haven't been scheduled for continuous operation yet.

3. **No Live Execution** — IBKR gateway exists but is in dry-run mode. No live trading.

4. **Limited Universe** — 11 sector ETFs + SPY + 5 macro tickers. Not yet scalable to 100+ tickers.

5. **Single User** — No authentication, no multi-user access control.

6. **Options Data** — Black-Scholes engine exists but uses synthetic IV, not real options market data.

---

## 10. How to Run

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add: FMP_API_KEY=xxx, OPENAI_API_KEY=xxx

# Run dashboard
python main.py
# Open http://localhost:8050

# Run pipeline
python scripts/run_all.py

# Run agents
python -m agents.auto_improve --cycle        # Full improvement cycle
python -m agents.methodology --once          # Single methodology run
python -m agents.risk_guardian --once         # Risk check
python -m agents.data_scout --once           # External data scan

# Run tests
python -m pytest tests/ -q
```

---

## 11. Key Design Decisions

1. **Multiplicative Signal Stack** — Layers multiply (not add) so any zero kills the signal. This prevents "short vol at the cliff."

2. **Regime Kill-Switch** — CRISIS regime = 0x sizing. No exceptions. This saved Sharpe +0.78 in backtests.

3. **Mean-Reversion Whitelist** — Only XLC, XLF, XLI, XLP, XLU trade MR. Other sectors (XLK, XLE) are too momentum-driven.

4. **Risk Guardian Independence** — The risk agent has no dependencies on other agents and can halt everything. No override possible.

5. **Sandbox Before Promote** — Every parameter change is tested in a sandboxed methodology lab run before being promoted to production settings.

6. **Dual LLM** — Math agent queries both Claude and GPT, scores responses, picks the better one. Avoids single-model bias.

---

## 12. Repository

**GitHub:** https://github.com/benda931/srv_quant_system

**Branch:** master

**Latest stats:** 162 files, ~75K LOC, 408 tests, 10 agents
