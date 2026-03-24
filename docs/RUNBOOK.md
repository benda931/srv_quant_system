# SRV Quantamental DSS -- Operational Runbook

## 1. Starting the Dashboard

```bash
cd srv_quant_system
python main.py
```

The Dash server starts at `http://127.0.0.1:8050`.

**Expected startup behavior**:
1. Loads settings from `config/settings.py` + `.env`
2. Connects to DuckDB (`data/srv_quant.duckdb`)
3. Runs `DataOrchestrator` to check data freshness
4. If data is stale (> 12h), fetches from FMP API
5. Runs `QuantEngine` to compute `master_df`
6. Builds all 16 dashboard tabs
7. Registers Dash callbacks

**Startup time**: 30-90 seconds (first run with data fetch), 10-20 seconds (cached).

**Verify**: Open browser to `http://127.0.0.1:8050`. The Overview tab should show
sector KPI cards and a regime hero badge.

---

## 2. Running the Pipeline

### Full daily pipeline

```bash
python scripts/run_all.py
```

Pipeline steps (in order):
1. Smart data refresh (skips if DB fresh < 12h)
2. QuantEngine (PCA, z-scores, regime)
3. Walk-forward backtest (optional, runs if stale > 7 days)
4. ML signal quality (optional)
5. Portfolio optimization (optional)
6. Stress tests + portfolio risk
7. Daily brief generation
8. DB audit write + data pruning

### With specific flags

```bash
python scripts/run_all.py --force-refresh   # Force re-fetch from FMP
python scripts/run_all.py --backtest        # Force backtest run
python scripts/run_all.py --no-ml           # Skip ML layer
```

### Verify pipeline success

Check `logs/srv_system.log` for step completion messages. Each step logs
`steps_ok` and `steps_failed` arrays.

---

## 3. Starting the Agent System

### Full orchestrator (24/7 daemon)

```bash
python agents/orchestrator.py
```

Schedule:
- Data refresh: every 6 hours
- VIX monitoring: every 30 minutes
- Methodology agent: 06:00 weekdays
- Optimizer agent: 07:00 weekdays
- Math agent: Monday 08:00
- Paper trader: 17:00 weekdays
- Health check: every 15 minutes
- Morning brief: 06:30 weekdays
- Evening recap: 17:30 weekdays

### Run specific agent once

```bash
python agents/orchestrator.py --run methodology
python agents/orchestrator.py --run optimizer
python agents/orchestrator.py --run math
```

### Run agent standalone

```bash
python agents/methodology/agent_methodology.py --once
python agents/math/agent_math.py --once
python agents/optimizer/agent_optimizer.py --once
```

### Check agent status

```bash
python agents/orchestrator.py --status
```

---

## 4. Troubleshooting Common Issues

### DuckDB locked by another process

**Symptom**: `DuckDB locked by another process` warning in logs.

**Cause**: Dashboard and pipeline running simultaneously.

**Fix**: The system auto-falls back to in-memory mode for the dashboard.
No action needed. To resolve permanently, stop the dashboard before running
the pipeline, or run the pipeline from the dashboard's built-in refresh.

### FMP API rate limit

**Symptom**: `429 Too Many Requests` in logs.

**Cause**: Exceeded FMP API rate limit.

**Fix**: Wait 60 seconds and retry. The pipeline has built-in retry logic.
Reduce `max_workers` in settings if persistent. Free-tier FMP keys have
lower rate limits.

### Empty master_df / no sectors loaded

**Symptom**: Dashboard shows blank Overview tab or 0 sectors.

**Cause**: Price data missing or stale.

**Fix**:
1. Check `data_lake/parquet/prices.parquet` exists and is recent
2. Run `python scripts/run_all.py --force-refresh`
3. Verify FMP_API_KEY in `.env` is valid

### Backtest returns NaN metrics

**Symptom**: IC, Sharpe, hit rate all NaN.

**Cause**: Insufficient price history for walk-forward windows.

**Fix**: Ensure at least 2 years of price data. Check `history_years` setting.
Run with `--force-refresh` to ensure full history is fetched.

### Agent fails with "No LLM API keys"

**Symptom**: Math or Optimizer agent logs `No LLM API keys available`.

**Cause**: Missing ANTHROPIC_API_KEY or OPENAI_API_KEY.

**Fix**: Add keys to `.env` or `agents/credentials/api_keys.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Import errors on startup

**Symptom**: `ModuleNotFoundError` for analytics modules.

**Fix**: Ensure you are running from the project root directory, and all
dependencies are installed:
```bash
pip install -r requirements.txt
```

### Log files growing too large

**Symptom**: `logs/` directory consuming excessive disk space.

**Fix**: Log rotation is configured with 50MB max per file and 5 backup files.
If old logs exist from before rotation was added, manually delete them:
```bash
rm logs/*.log.old
```

---

## 5. Refreshing Data

### Automatic refresh

The pipeline checks data freshness automatically. If price data in DuckDB is
older than `cache_max_age_hours` (default: 12), it triggers a full FMP fetch.

### Force refresh

```bash
python scripts/run_all.py --force-refresh
```

This re-fetches all data from FMP regardless of freshness:
- Historical prices (all tickers, full history)
- ETF holdings and sector weights
- Fundamental ratios (TTM)
- Macro indicators

### Refresh specific data

To refresh only prices without running the full pipeline:
```python
from data.pipeline import DataLakeManager
from config.settings import get_settings
dlm = DataLakeManager(get_settings())
dlm.refresh_prices(force=True)
```

---

## 6. Adding a New Sector

1. **Update settings** in `config/settings.py`:
   ```python
   sector_tickers: Dict[str, str] = Field(
       default_factory=lambda: {
           ...
           "New Sector": "XNEW",  # Add new ETF ticker
       }
   )
   ```

2. **Force data refresh** to fetch history for the new ticker:
   ```bash
   python scripts/run_all.py --force-refresh
   ```

3. **Verify** the new sector appears in the dashboard Overview tab.

4. **Run backtest** to establish baseline metrics with the new universe:
   ```bash
   python scripts/run_all.py --backtest
   ```

Note: Adding or removing sectors changes PCA decomposition and residual z-scores
for ALL sectors. Re-run the backtest to verify impact.

---

## 7. Calibrating Parameters

### Interactive calibration

Modify parameters in `config/settings.py` or via environment variables:

```bash
# Override via environment
export signal_entry_threshold=0.10
python main.py
```

### Key parameters to calibrate

| Parameter                | Default | Impact                              |
|--------------------------|---------|-------------------------------------|
| `signal_entry_threshold` | 0.05    | Higher = fewer but higher conviction entries |
| `signal_a1_frob`        | 0.3     | Layer 1: sensitivity to Frobenius distortion |
| `signal_vix_soft`       | 21.0    | Layer 4: VIX level where penalty starts |
| `signal_vix_kill`       | 45.0    | Layer 4: VIX hard kill threshold    |
| `trade_max_holding_days`| 25      | Maximum position hold period        |
| `trade_max_loss_pct`    | 0.02    | Stop-loss as % of notional          |
| `pca_window`            | 252     | PCA lookback window (trading days)  |
| `zscore_window`         | 60      | Z-score lookback window             |

### Calibration workflow

1. Change parameter(s) in `config/settings.py`
2. Run backtest: `python scripts/run_all.py --backtest`
3. Compare IC, Sharpe, hit rate vs previous backtest
4. Check regime-conditional metrics (especially TENSION and CRISIS)
5. If improved, keep changes. If degraded, revert.

### Automated calibration via Optimizer Agent

The Optimizer Agent can automatically tune parameters:
```bash
python agents/optimizer/agent_optimizer.py --once
```

It creates backups before changes and auto-reverts if tests fail. Results are
logged in `logs/agent_optimizer_*.json` and the optimization log.

---

## 8. Monitoring and Alerts

### Log locations

| Log file                    | Content                              |
|-----------------------------|--------------------------------------|
| `logs/srv_system.log`       | Main dashboard and pipeline          |
| `logs/orchestrator.log`     | Agent orchestrator                   |
| `logs/agent_methodology.log`| Methodology agent runs               |
| `logs/agent_math.log`       | Math agent proposals                 |
| `logs/agent_optimizer.log`  | Optimizer agent changes              |

### Data health

The Data Health tab in the dashboard shows:
- Freshness of each data source
- Row counts and quality metrics
- Missing data alerts

Programmatic check:
```python
from data_ops.status_report import DataHealthReport
report = DataHealthReport()
report.run()
```

### Audit trail

All signals, trades, and parameter changes are logged in the DuckDB `audit` schema
with SHA256 hash chains for tamper detection:
```python
from db.audit import AuditTrail
audit = AuditTrail()
audit.verify_chain()  # Returns True if chain is intact
```
