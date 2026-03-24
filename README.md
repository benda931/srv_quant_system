# SRV Quantamental DSS

**Institutional-grade Sector Relative Value Decision Support System**
with Short Volatility / Dispersion Trade capabilities.

Discretionary quant PM decision support for 11 S&P 500 sector ETFs vs SPY.
Core focus: short vol via correlation/dispersion trades with a 4-layer signal stack.

> This system is a **decision support tool** -- it does NOT auto-trade.

---

## Quick Start

### Prerequisites

- Python 3.11+
- DuckDB (installed via pip)
- FMP API key ([financialmodelingprep.com](https://financialmodelingprep.com))

### Installation

```bash
# Clone and install dependencies
cd srv_quant_system
pip install -r requirements.txt

# Configure API key
echo "FMP_API_KEY=your_key_here" > .env
```

### Run the Dashboard

```bash
python main.py
```

The Dash dashboard launches at `http://127.0.0.1:8050` with Hebrew RTL UI.

### Run the Full Pipeline

```bash
python scripts/run_all.py                  # Normal daily run
python scripts/run_all.py --force-refresh  # Force FMP data re-fetch
python scripts/run_all.py --backtest       # Include walk-forward backtest
```

### Run Agents

```bash
python agents/orchestrator.py              # 24/7 daemon (all agents)
python agents/orchestrator.py --once       # Single full cycle
python agents/methodology/agent_methodology.py --once
python agents/math/agent_math.py --once
python agents/optimizer/agent_optimizer.py --once
```

---

## Architecture Overview

```
  FMP API
    |
    v
+-----------+     +-------------+     +--------------+     +----------------+
| DataLake  |---->| QuantEngine |---->| Signal Stack |---->| Trade          |
| Manager   |     | (stat_arb)  |     | (4 layers)   |     | Structure      |
| pipeline  |     |             |     |              |     | (sizing/legs)  |
+-----------+     +-----+-------+     +------+-------+     +-------+--------+
    |                   |                    |                      |
    v                   v                    v                      v
+----------+     +-----------+     +-----------------+     +--------------+
| DuckDB   |     | Backtest  |     | Trade Monitor   |     | Dashboard    |
| Parquet  |     | Stress    |     | (exit signals)  |     | (16 tabs)    |
+----------+     | Risk      |     +-----------------+     | Hebrew RTL   |
                 +-----------+                              +--------------+
                      |
                      v
               +-------------+
               | Agent System |
               | (3 agents + |
               |  orchestr.) |
               +-------------+
```

### Data Flow

1. **FMP API** ingestion via `DataLakeManager` (prices, fundamentals, holdings, macro)
2. **QuantEngine** computes PCA residuals, z-scores, regime classification
3. **Signal Stack** applies 4-layer scoring (distortion, dislocation, mean-reversion, regime safety)
4. **Trade Structure** constructs actionable trade tickets with legs, Greeks, sizing
5. **Trade Monitor** tracks open positions, generates exit signals
6. **Dashboard** presents everything in a 16-tab Hebrew RTL interface

---

## Module Listing

| Directory       | Purpose                                                    |
|-----------------|------------------------------------------------------------|
| `analytics/`    | Core quant engines: signal stack, trade structure, risk, backtest, options, stress |
| `agents/`       | AI agent system: methodology, math, optimizer + orchestrator |
| `config/`       | Pydantic settings -- single source of truth for all parameters |
| `data/`         | Data pipeline (FMP API ingestion)                          |
| `data_ops/`     | Data quality, freshness checks, health monitoring, PM journal |
| `db/`           | DuckDB connection, schema migrations, reader/writer, audit trail |
| `reports/`      | Daily brief and DSS brief generators                       |
| `scripts/`      | Pipeline orchestration (`run_all.py`), agent bus, Claude loop |
| `tests/`        | pytest test suite                                          |
| `ui/`           | Dash UI components, analytics tabs, scanner, journal panel |

### Key Analytics Modules

| Module                      | Class / Function             | Purpose                              |
|-----------------------------|------------------------------|--------------------------------------|
| `analytics/stat_arb.py`    | `QuantEngine`                | Core quant engine (PCA, z-scores, regime) |
| `analytics/signal_stack.py`| `SignalStackEngine`          | 4-layer signal scoring               |
| `analytics/signal_mean_reversion.py` | `compute_mean_reversion_score` | Layer 3: OU, ADF, Hurst scoring |
| `analytics/signal_regime_safety.py`  | `compute_regime_safety_score`  | Layer 4: regime gating + hard kills |
| `analytics/trade_structure.py` | `TradeStructureEngine`    | Trade construction + sizing          |
| `analytics/trade_monitor.py`  | `TradeMonitor`             | Exit signals + trade health          |
| `analytics/correlation_engine.py` | `CorrVolEngine`         | Implied correlation, variance decomp |
| `analytics/portfolio_risk.py` | `PortfolioRiskEngine`      | VaR, CVaR, MCTR                     |
| `analytics/stress.py`        | `StressEngine`              | 10 institutional stress scenarios    |
| `analytics/tail_risk.py`     | `compute_expected_shortfall` | ES (Basel FRTB), tail correlation   |
| `analytics/options_engine.py` | `OptionsEngine`            | BS Greeks, IV surface, implied corr  |
| `analytics/backtest.py`      | `run_backtest`              | Walk-forward backtesting             |
| `analytics/performance_metrics.py` | Various                | Sortino, Information Ratio, Omega, etc. |

---

## Configuration

All parameters live in `config/settings.py` as a Pydantic `Settings` class with validation.

### Key Parameter Groups

- **Universe**: 11 sector ETFs (XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU) + SPY
- **Core Windows**: `pca_window=252`, `zscore_window=60`, `corr_window=60`
- **Signal Stack Layer 1**: `signal_a1_frob=0.3`, `signal_a2_mode=0.2`, `signal_a3_coc=0.3`
- **Signal Stack Layer 3**: `signal_mr_w_hl=0.35`, `signal_mr_w_adf=0.40`, `signal_mr_w_hurst=0.25`
- **Signal Stack Layer 4**: `signal_vix_soft=21`, `signal_vix_hard=31`, `signal_vix_kill=45`
- **Trade Structure**: `trade_max_holding_days=25`, `trade_max_loss_pct=0.02`
- **Entry/Exit**: `signal_entry_threshold=0.05`, `monitor_z_compression_exit=0.75`

Parameters can be overridden via environment variables.

---

## Data Sources

**Primary**: [Financial Modeling Prep (FMP) API](https://financialmodelingprep.com)
- Historical daily prices (10 years)
- ETF holdings / sector weights
- Fundamental ratios (TTM)
- Batch quotes
- Macro indicators (VIX, TNX, DXY, HYG/IEF)

**Storage**:
- DuckDB (`data/srv_quant.duckdb`) -- 5 schemas, 8+ tables
- Parquet files (`data_lake/parquet/`) -- prices, fundamentals, weights
- SQLite (`data/pm_journal.db`) -- PM decision journal

---

## Dashboard Tabs (16)

1. Overview (sector KPIs, regime hero, market narrative)
2. DSS (Signal Stack table, Trade Book, Correlation Distortion, Trade Monitor)
3. Correlation / Volatility
4. Signal Decay
5. Regime Timeline
6. Stress Testing
7. Portfolio Risk
8. Backtest
9. Scanner (basic)
10. Scanner Pro
11. P&L Tracker
12. Portfolio
13. Methodology Lab
14. Daily Brief
15. Data Health
16. PM Journal

---

## Agent System

Three specialized AI agents coordinated by a master orchestrator:

| Agent          | Role                                  | Schedule          |
|----------------|---------------------------------------|-------------------|
| **Methodology** | Runs full pipeline, evaluates backtest quality, generates conclusions | Daily 06:00 weekdays |
| **Math**        | Analyzes scoring functions, proposes mathematical improvements | Weekly (Monday 08:00) |
| **Optimizer**   | Modifies parameters/code to improve performance, auto-reverts on failure | Daily 07:00 weekdays |
| **Orchestrator**| 24/7 daemon: schedules agents, monitors health, dispatches alerts | Continuous |

Communication via `AgentBus` (publish/subscribe). Optimization history tracked in `optimization_log`.

---

## API Keys Setup

Create a `.env` file in the project root:

```env
FMP_API_KEY=your_fmp_api_key_here
```

For the agent system (optional):
```env
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

Store agent-specific keys in `agents/credentials/api_keys.env`.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_signal_stack.py -v
pytest tests/test_portfolio_risk.py -v
pytest tests/test_trade_structure.py -v
pytest tests/test_options_engine.py -v
```

---

## License

Proprietary. All rights reserved.
