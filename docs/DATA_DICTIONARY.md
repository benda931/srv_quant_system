# SRV Quantamental DSS -- Data Dictionary

## 1. Parquet Files

### prices.parquet

**Location**: `data_lake/parquet/prices.parquet`
**Source**: FMP API `/api/v3/historical-price-full`
**Refresh**: Daily (12h cache)

| Column    | Type       | Description                                    |
|-----------|------------|------------------------------------------------|
| date      | datetime64 | Trading date (index)                           |
| XLC       | float64    | Communication Services ETF close price         |
| XLY       | float64    | Consumer Discretionary ETF close price         |
| XLP       | float64    | Consumer Staples ETF close price               |
| XLE       | float64    | Energy ETF close price                         |
| XLF       | float64    | Financials ETF close price                     |
| XLV       | float64    | Health Care ETF close price                    |
| XLI       | float64    | Industrials ETF close price                    |
| XLB       | float64    | Materials ETF close price                      |
| XLRE      | float64    | Real Estate ETF close price                    |
| XLK       | float64    | Technology ETF close price                     |
| XLU       | float64    | Utilities ETF close price                      |
| SPY       | float64    | S&P 500 ETF close price                        |
| ^VIX      | float64    | CBOE Volatility Index close                    |
| ^TNX      | float64    | 10-Year Treasury Yield                         |
| DX-Y.NYB  | float64    | US Dollar Index                                |
| HYG       | float64    | High Yield Corporate Bond ETF close            |
| IEF       | float64    | 7-10 Year Treasury Bond ETF close              |

**History**: 10 years (configurable via `history_years`)

---

### fundamentals.parquet

**Location**: `data_lake/parquet/fundamentals.parquet`
**Source**: FMP API `/stable/profile`, `/api/v3/ratios-ttm`, `/api/v3/analyst-estimates`
**Refresh**: Daily (12h cache)

| Column               | Type    | Description                          |
|----------------------|---------|--------------------------------------|
| symbol               | str     | Ticker (index level 1)               |
| snapshot_date        | date    | Date of snapshot (index level 2)     |
| pe_ttm               | float64 | Price/Earnings (trailing 12 months)  |
| pb                   | float64 | Price/Book                           |
| ps                   | float64 | Price/Sales                          |
| ev_ebitda            | float64 | Enterprise Value / EBITDA            |
| ev_ebit              | float64 | Enterprise Value / EBIT              |
| div_yield            | float64 | Dividend yield                       |
| fcf_yield            | float64 | Free cash flow yield                 |
| roe                  | float64 | Return on equity                     |
| roa                  | float64 | Return on assets                     |
| roic                 | float64 | Return on invested capital           |
| net_margin           | float64 | Net profit margin                    |
| gross_margin         | float64 | Gross profit margin                  |
| operating_margin     | float64 | Operating profit margin              |
| debt_equity          | float64 | Debt/Equity ratio                    |
| current_ratio        | float64 | Current ratio                        |
| interest_coverage    | float64 | Interest coverage ratio              |
| accrual_ratio        | float64 | Accrual quality ratio                |
| eps_growth_yoy       | float64 | EPS growth year-over-year            |
| revenue_growth_yoy   | float64 | Revenue growth YoY                   |
| ebitda_growth_yoy    | float64 | EBITDA growth YoY                    |
| forward_pe           | float64 | Forward P/E                          |
| ntm_eps_consensus    | float64 | Next 12-month EPS consensus          |
| avg_eps_surprise_pct | float64 | Average EPS surprise (%)             |
| enterprise_value     | float64 | Enterprise value                     |
| ebitda               | float64 | EBITDA                               |
| ocf                  | float64 | Operating cash flow                  |
| capex                | float64 | Capital expenditures                 |

---

### weights.parquet

**Location**: `data_lake/parquet/weights.parquet`
**Source**: FMP API ETF holdings endpoints
**Refresh**: Daily (12h cache)

| Column        | Type    | Description                            |
|---------------|---------|----------------------------------------|
| snapshot_date | date    | Date of snapshot (index)               |
| sector        | str     | Sector name                            |
| weight_pct    | float64 | Sector weight in SPY (percentage)      |

---

## 2. DuckDB Tables

**Database**: `data/srv_quant.duckdb`
**Engine**: DuckDB (columnar OLAP, MVCC concurrency)

### Schema: _meta

#### _meta.schema_migrations

| Column     | Type      | Description                     |
|------------|-----------|---------------------------------|
| version    | INTEGER   | Migration version (PK)          |
| name       | VARCHAR   | Migration name                  |
| applied_at | TIMESTAMP | When migration was applied      |

### Schema: market_data

#### market_data.prices

| Column      | Type        | Description                      |
|-------------|-------------|----------------------------------|
| date        | DATE        | Trading date (PK with ticker)    |
| ticker      | VARCHAR(20) | Ticker symbol (PK with date)     |
| close       | DOUBLE      | Closing price                    |
| inserted_at | TIMESTAMP   | Row insertion timestamp          |

### Schema: holdings

#### holdings.etf_holdings

| Column        | Type        | Description                        |
|---------------|-------------|------------------------------------|
| snapshot_date | DATE        | Snapshot date (PK with etf+asset)  |
| etf_symbol    | VARCHAR(20) | ETF ticker (PK)                    |
| asset         | VARCHAR(20) | Held asset ticker (PK)             |
| weight_pct    | DOUBLE      | Weight in ETF (%)                  |
| shares_number | DOUBLE      | Number of shares held              |
| market_value  | DOUBLE      | Market value of position           |
| inserted_at   | TIMESTAMP   | Row insertion timestamp            |

#### holdings.spy_sector_weights

| Column        | Type         | Description                       |
|---------------|--------------|-----------------------------------|
| snapshot_date | DATE         | Snapshot date (PK with sector)    |
| sector        | VARCHAR(100) | Sector name (PK)                  |
| weight_pct    | DOUBLE       | Sector weight in SPY (%)          |
| inserted_at   | TIMESTAMP    | Row insertion timestamp           |

### Schema: fundamentals

#### fundamentals.quotes

| Column        | Type        | Description                       |
|---------------|-------------|-----------------------------------|
| snapshot_date | DATE        | Quote date (PK with symbol)       |
| symbol        | VARCHAR(20) | Ticker symbol (PK)                |
| price         | DOUBLE      | Quote price                       |
| pe            | DOUBLE      | P/E ratio                         |
| eps           | DOUBLE      | Earnings per share                |
| market_cap    | DOUBLE      | Market capitalization             |
| inserted_at   | TIMESTAMP   | Row insertion timestamp           |

#### fundamentals.ratios

| Column               | Type        | Description                     |
|----------------------|-------------|---------------------------------|
| snapshot_date        | DATE        | Snapshot date (PK with symbol)  |
| symbol               | VARCHAR(20) | Ticker symbol (PK)              |
| pe_ttm               | DOUBLE      | P/E trailing 12 months          |
| pb                   | DOUBLE      | Price/Book                      |
| ps                   | DOUBLE      | Price/Sales                     |
| ev_ebitda            | DOUBLE      | EV/EBITDA                       |
| ev_ebit              | DOUBLE      | EV/EBIT                         |
| div_yield            | DOUBLE      | Dividend yield                  |
| fcf_yield            | DOUBLE      | Free cash flow yield            |
| roe                  | DOUBLE      | Return on equity                |
| roa                  | DOUBLE      | Return on assets                |
| roic                 | DOUBLE      | Return on invested capital      |
| net_margin           | DOUBLE      | Net margin                      |
| gross_margin         | DOUBLE      | Gross margin                    |
| operating_margin     | DOUBLE      | Operating margin                |
| debt_equity          | DOUBLE      | Debt/Equity                     |
| current_ratio        | DOUBLE      | Current ratio                   |
| interest_coverage    | DOUBLE      | Interest coverage               |
| accrual_ratio        | DOUBLE      | Accrual ratio                   |
| eps_growth_yoy       | DOUBLE      | EPS growth YoY                  |
| revenue_growth_yoy   | DOUBLE      | Revenue growth YoY              |
| ebitda_growth_yoy    | DOUBLE      | EBITDA growth YoY               |
| forward_pe           | DOUBLE      | Forward P/E                     |
| ntm_eps_consensus    | DOUBLE      | NTM EPS consensus               |
| avg_eps_surprise_pct | DOUBLE      | Avg EPS surprise %              |
| enterprise_value     | DOUBLE      | Enterprise value                |
| ebitda               | DOUBLE      | EBITDA                          |
| ocf                  | DOUBLE      | Operating cash flow             |
| capex                | DOUBLE      | Capital expenditures            |
| inserted_at          | TIMESTAMP   | Row insertion timestamp         |

### Schema: analytics

#### analytics.runs

| Column           | Type        | Description                       |
|------------------|-------------|-----------------------------------|
| run_id           | BIGINT      | Auto-incrementing run ID (PK)     |
| run_date         | DATE        | Date of analytics run             |
| run_started_at   | TIMESTAMP   | Run start time                    |
| run_finished_at  | TIMESTAMP   | Run end time                      |
| duration_ms      | INTEGER     | Run duration in milliseconds      |
| market_state     | VARCHAR(20) | Regime label (CALM/NORMAL/etc.)   |
| regime_alert     | VARCHAR(500)| Regime alert text                 |
| avg_corr         | DOUBLE      | Average pairwise correlation      |
| n_longs          | INTEGER     | Number of long signals            |
| n_shorts         | INTEGER     | Number of short signals           |
| avg_mc           | DOUBLE      | Average conviction score          |
| data_health_label| VARCHAR(20) | Data quality status               |
| inserted_at      | TIMESTAMP   | Row insertion timestamp           |

#### analytics.backtest_cache

| Column       | Type      | Description                        |
|--------------|-----------|------------------------------------|
| cache_date   | DATE      | Backtest date (PK)                 |
| ic_mean      | DOUBLE    | Information Coefficient mean       |
| ic_ir        | DOUBLE    | IC Information Ratio               |
| hit_rate     | DOUBLE    | Directional hit rate               |
| sharpe       | DOUBLE    | Sharpe ratio                       |
| max_drawdown | DOUBLE    | Maximum drawdown                   |
| n_walks      | INTEGER   | Number of walk-forward windows     |
| n_sectors    | INTEGER   | Number of sectors evaluated        |
| regime_json  | VARCHAR   | Regime breakdown (JSON)            |
| params_json  | VARCHAR   | Parameters snapshot (JSON)         |
| inserted_at  | TIMESTAMP | Row insertion timestamp            |

### Schema: audit

#### audit.signal_log

| Column           | Type      | Description                        |
|------------------|-----------|------------------------------------|
| seq_id           | BIGINT    | Auto-incrementing sequence (PK)    |
| timestamp        | TIMESTAMP | When signal was generated          |
| sector           | VARCHAR   | Sector ticker                      |
| direction        | VARCHAR   | LONG / SHORT                       |
| conviction_score | DOUBLE    | Combined conviction score          |
| distortion_score | DOUBLE    | Layer 1 score                      |
| dislocation_score| DOUBLE    | Layer 2 score                      |
| mr_score         | DOUBLE    | Layer 3 score                      |
| safety_score     | DOUBLE    | Layer 4 score                      |
| regime           | VARCHAR   | Current regime label               |
| prev_hash        | VARCHAR   | SHA256 of previous record          |
| record_hash      | VARCHAR   | SHA256 of this record              |

#### audit.trade_log

| Column      | Type      | Description                           |
|-------------|-----------|---------------------------------------|
| seq_id      | BIGINT    | Auto-incrementing sequence (PK)       |
| timestamp   | TIMESTAMP | When trade action occurred            |
| trade_id    | VARCHAR   | Unique trade identifier               |
| action      | VARCHAR   | ENTRY / EXIT / ADJUST / CANCEL        |
| details     | VARCHAR   | JSON-encoded trade details            |
| prev_hash   | VARCHAR   | SHA256 of previous record             |
| record_hash | VARCHAR   | SHA256 of this record                 |

#### audit.param_log

| Column      | Type      | Description                           |
|-------------|-----------|---------------------------------------|
| seq_id      | BIGINT    | Auto-incrementing sequence (PK)       |
| timestamp   | TIMESTAMP | When parameter was changed            |
| param_name  | VARCHAR   | Parameter name                        |
| old_value   | VARCHAR   | Previous value (string repr)          |
| new_value   | VARCHAR   | New value (string repr)               |
| changed_by  | VARCHAR   | Who made the change (agent/manual)    |
| prev_hash   | VARCHAR   | SHA256 of previous record             |
| record_hash | VARCHAR   | SHA256 of this record                 |

---

## 3. master_df (QuantEngine Output)

The `master_df` DataFrame is produced by `QuantEngine.calculate_conviction_score()`
and contains one row per sector with approximately 133 columns. Key column groups:

### Price / Return Columns
- `ticker`: Sector ETF ticker
- `close`: Latest close price
- `return_1d`, `return_5d`, `return_21d`, `return_63d`: Period returns
- `vol_20d`, `vol_60d`: Realized volatility

### PCA / Residual Columns
- `pca_residual`: PCA residual (distance from fair value)
- `residual_z`: Z-score of PCA residual
- `explained_var_ratio`: PCA explained variance ratio
- `market_mode_strength`: First principal component share (lambda_1/N)

### Signal Score Columns
- `distortion_score`: Layer 1 score [0, 1]
- `dislocation_score`: Layer 2 score [0, 1]
- `mean_reversion_score`: Layer 3 score [0, 1]
- `regime_safety_score`: Layer 4 score [0, 1]
- `conviction_score`: Combined score (product of 4 layers)

### Regime Columns
- `regime_label`: CALM / NORMAL / TENSION / CRISIS
- `vix_level`: Current VIX value
- `credit_z`: Credit spread z-score

### Fundamental Columns
- `pe_ttm`, `pb`, `ps`, `ev_ebitda`: Valuation ratios
- `roe`, `roa`, `roic`: Profitability metrics
- `debt_equity`, `current_ratio`: Leverage metrics
- `eps_growth_yoy`, `revenue_growth_yoy`: Growth metrics

### Attribution Score Columns
- `sds_score`: Statistical Dislocation Score
- `fjs_score`: Fundamental Justification Score
- `mss_score`: Market Structure Score
- `stf_score`: Short-Term Flow Score
- `mc_score`: Master Conviction Score

### ML Signal Columns (when ML layer is enabled)
- `ml_quality_pred`: ML-predicted signal quality
- `ml_regime_prob_*`: ML regime transition probabilities
- `ml_adjusted_conviction`: Quality-adjusted conviction
