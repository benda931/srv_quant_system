# SRV Quantamental DSS -- Architecture

## System Diagram

```
                           +------------------+
                           |   FMP REST API   |
                           | (prices, fund.,  |
                           |  holdings, macro)|
                           +--------+---------+
                                    |
                                    v
                      +-------------+--------------+
                      |    DataLakeManager          |
                      |    data/pipeline.py         |
                      | - Fetches prices (10yr)     |
                      | - ETF holdings / weights    |
                      | - Fundamental ratios (TTM)  |
                      | - Macro (VIX, TNX, DXY)     |
                      | - 12h cache freshness       |
                      +------+-------+------+-------+
                             |       |      |
                      +------+  +----+  +---+-------+
                      |         |       |           |
                      v         v       v           v
               +---------+ +-------+ +--------+ +----------+
               | DuckDB  | |Parquet| |DataOps | | SQLite   |
               | 5 schema| |prices | |quality | | PM       |
               | 8 tables| |fund.  | |health  | | Journal  |
               +---------+ |wgts   | |freshness |          |
                            +-------+ +--------+ +----------+
                                    |
                                    v
                      +-------------+--------------+
                      |     QuantEngine             |
                      |     analytics/stat_arb.py   |
                      | - PCA decomposition         |
                      | - Z-score computation       |
                      | - Regime classification     |
                      | - master_df (133 columns)   |
                      +-------------+--------------+
                                    |
                +-------------------+-------------------+
                |                   |                   |
                v                   v                   v
  +-------------+----+  +----------+--------+  +-------+----------+
  | CorrVolEngine    |  | ML Signal Layer   |  | Fundamentals     |
  | correlation_     |  | ml_signals.py     |  | Engine           |
  | engine.py        |  | ml_regime_        |  | fundamentals_    |
  | - Frobenius dist |  |   forecast.py     |  |   engine.py      |
  | - Implied corr   |  | ml_signal_        |  | - TTM ratios     |
  | - Variance decomp|  |   combiner.py     |  | - Factor scores  |
  +--------+---------+  +----------+--------+  +------------------+
           |                       |
           v                       v
  +--------+-------------------------------------------+
  |            Signal Stack Engine                      |
  |            analytics/signal_stack.py                |
  |                                                     |
  |  Layer 1: Distortion Score (S^dist)                |
  |    S^dist = sigma(a1*z_D + a2*rank(m_t) + a3*z_CoC)|
  |                                                     |
  |  Layer 2: Dislocation Score (S^disloc)             |
  |    S^disloc = min(1, |z| / Z_cap)                  |
  |                                                     |
  |  Layer 3: Mean-Reversion Score (S^mr)              |
  |    S^mr = w_hl*f_hl + w_adf*f_adf + w_hurst*f_hurst|
  |    (analytics/signal_mean_reversion.py)             |
  |                                                     |
  |  Layer 4: Regime Safety Score (S^safe)             |
  |    S^safe = prod(1 - w_i * P_i)                    |
  |    Hard kills: VIX>=45, credit z<=-3, etc.         |
  |    (analytics/signal_regime_safety.py)              |
  |                                                     |
  |  Combined: Score_j = S^dist * S^disloc * S^mr * S^safe |
  |  Entry: Score_j >= theta_enter AND all gates pass   |
  +------------------------+----------------------------+
                           |
                           v
  +------------------------+----------------------------+
  |       Trade Structure Engine                        |
  |       analytics/trade_structure.py                  |
  | - Sector RV: Long/Short ETF vs SPY (beta-neutral)  |
  | - Dispersion: Short SPY straddle + Long sector vol  |
  | - RV Spread: Pair trade (cointegration-based)       |
  | - Position sizing (regime-adjusted)                 |
  | - Greeks profile (delta, vega, gamma, theta)        |
  | - Exit conditions                                   |
  +------------------------+----------------------------+
                           |
              +------------+------------+
              |                         |
              v                         v
  +-----------+----------+   +----------+-----------+
  | Trade Monitor        |   | Dashboard            |
  | analytics/           |   | main.py              |
  |   trade_monitor.py   |   | 16 tabs, Hebrew RTL  |
  | - Exit signals:      |   | - Overview           |
  |   PROFIT_TAKE        |   | - DSS (primary tool) |
  |   STOP_LOSS          |   | - Correlation/Vol    |
  |   TIME_EXIT          |   | - Stress / Risk      |
  |   REGIME_EXIT        |   | - Backtest           |
  | - Trade health score |   | - P&L / Portfolio    |
  +----------------------+   | - Scanner Pro        |
                             | - Journal            |
                             +----------------------+
```

---

## Module Responsibilities

### analytics/stat_arb.py -- QuantEngine (Core)

The central computation engine. Loads price data, computes PCA decomposition over
a rolling window, derives sector residuals (distance from fair value), z-scores,
regime classification, and produces `master_df` -- a DataFrame with 133 columns
covering every sector's signal, risk, and fundamental metrics.

### analytics/signal_stack.py -- SignalStackEngine

Implements the 4-layer multiplicative scoring system that determines trade conviction.
Layers are evaluated sequentially; any hard-kill gate zeros the final score.

### analytics/signal_mean_reversion.py -- Layer 3

Evaluates mean-reversion quality using three sub-components:
- **OU half-life**: Ornstein-Uhlenbeck process calibration (sweet spot 5-90 days)
- **ADF stationarity**: Augmented Dickey-Fuller test for unit root rejection
- **Hurst exponent**: H < 0.5 confirms mean-reversion, H > 0.5 indicates trending

### analytics/signal_regime_safety.py -- Layer 4

Prevents short-vol entries during dangerous regimes. Four penalty sub-components
(VIX, credit, correlation, transition) are combined multiplicatively.
Hard kills zero the score when any threshold is breached.

### analytics/trade_structure.py -- Trade Construction

Converts high-conviction signals into actionable trade tickets with:
- Leg construction (instruments, direction, notional)
- Greeks profile (delta, vega, gamma, theta)
- Risk limits (max loss, stop-loss levels)
- Exit conditions (z-compression, time decay, regime change)

### analytics/trade_monitor.py -- Exit Management

Tracks open trades and emits exit signals:
- `PROFIT_TAKE`: z-score compressed >= 75% toward mean
- `STOP_LOSS`: z-score extended >= 150% (moved against position)
- `TIME_EXIT`: holding period >= 90% of maximum
- `REGIME_EXIT`: safety score collapsed or regime entered CRISIS

### analytics/correlation_engine.py -- CorrVolEngine

Computes Frobenius distortion, implied correlation (Cboe-style), variance
decomposition, and correlation-of-correlations instability metrics that feed
Layer 1 of the signal stack.

### analytics/portfolio_risk.py -- PortfolioRiskEngine

Parametric and historical risk decomposition: Ledoit-Wolf shrunk covariance,
VaR (95%), CVaR, Marginal Contribution to Risk (MCTR), factor VaR decomposition.

### analytics/tail_risk.py -- Expected Shortfall

Basel FRTB-compliant ES at 97.5% using three methods (parametric, historical,
Cornish-Fisher). Includes parametric correlation stress and tail-correlation
diagnostics. VaR backtesting via Kupiec POF and Christoffersen independence tests.

### analytics/stress.py -- StressEngine

10 institutional stress scenarios (e.g., 2008 crisis replay, vol spike, credit
crunch, correlation spike, rate shock). Each scenario shocks the covariance
matrix and re-computes portfolio impact.

### analytics/options_engine.py -- OptionsEngine

Black-Scholes pricing, Greeks computation, IV surface construction, Cboe-style
implied correlation index, dispersion index, and variance risk premium.

### analytics/backtest.py -- Walk-Forward Backtest

Expanding-window walk-forward framework. Computes IC, Sharpe, hit rate, max
drawdown, with regime-conditional breakdown.

---

## Signal Stack Formula (4 Layers)

### Layer 1: Distortion Score

```
S^dist = sigma(a1 * z_D + a2 * rank(m_t) + a3 * z_CoC + a4 * credit_z + a5 * vix_term)
```

- `z_D`: Frobenius distortion z-score (correlation matrix vs baseline)
- `rank(m_t)`: Market-mode share percentile (lambda_1 / N)
- `z_CoC`: Correlation-of-correlations instability z-score
- `credit_z`: HYG/IEF spread z-score (negative = stress)
- `vix_term`: VIX term structure slope (negative = backwardation)

Sigmoid maps the linear combination to [0, 1]. Higher = more correlation distortion.

### Layer 2: Dislocation Score

```
S^disloc = min(1, |z_t^(j)| / Z_cap)
```

Per-candidate residual z-score, capped at Z_cap (default 2.5). Measures how far
the candidate is dislocated from fair value.

### Layer 3: Mean-Reversion Score

```
S^mr = w_hl * f_hl(half_life) + w_adf * f_adf(p_value) + w_hurst * f_hurst(H)
```

- `f_hl`: Half-life quality (Gaussian bump, sweet spot 5-90 days)
- `f_adf`: ADF p-value quality (lower p = stronger stationarity evidence)
- `f_hurst`: Hurst exponent quality (H < 0.5 = mean-reverting)

Weights: w_hl=0.35, w_adf=0.40, w_hurst=0.25.

### Layer 4: Regime Safety Score

```
S^safe = (1 - w_vix * P_vix) * (1 - w_credit * P_credit) * (1 - w_corr * P_corr) * (1 - w_trans * P_trans)
```

Hard kills (S^safe = 0):
- VIX >= 45
- Credit spread z-score <= -3
- Correlation z-score >= 2.5
- Crisis probability >= 70%
- CRISIS regime label

### Combined Conviction

```
Score_j = S^dist * S^disloc * S^mr * S^safe
Entry: Score_j >= theta_enter (default 0.05) AND all layer gates pass
```

---

## Regime Classification Logic

The QuantEngine classifies market regimes into four states based on composite
stress indicators:

| Regime    | Conditions                                              |
|-----------|---------------------------------------------------------|
| CALM      | VIX < 15, credit normal, correlation below median       |
| NORMAL    | VIX 15-21, no stress indicators elevated                |
| TENSION   | VIX 21-31, OR credit stress, OR correlation spike       |
| CRISIS    | VIX >= 31, OR multiple stress indicators simultaneously |

Regime drives:
- Layer 4 safety gating (CRISIS = hard kill)
- Position sizing multiplier (reduced in TENSION)
- Stop-loss thresholds (tightened in TENSION)
- Backtest evaluation (regime-conditional metrics)

---

## Trade Types

### 1. Sector Relative-Value (sector_rv)

Existing core trade. Long dislocated sector ETF, short SPY (or vice versa).
Beta-neutral via PCA residual hedge ratio.

**Legs**: 2 (equity only)
**Edge source**: PCA residual mean-reversion
**Typical hold**: 5-25 days

### 2. Dispersion Trade (dispersion)

Short index volatility + long constituent volatility. Profits when realized
sector dispersion exceeds what implied correlation prices.

**Legs**: N+1 (short SPY straddle + long sector straddles)
**Edge source**: Correlation risk premium (implied corr > realized)
**P&L**: ~ sum(w_i^2 * sigma_i^2) - sigma_index^2
**Typical hold**: 30-45 days (option expiry aligned)

### 3. RV Spread (rv_spread)

Pair trade between two sectors using cointegration regression hedge ratio.
Long the cheap (dislocated) sector, short the rich sector.

**Legs**: 2 (equity, hedge-ratio sized)
**Edge source**: Cointegration mean-reversion
**Typical hold**: 10-25 days

---

## Agent Lifecycle

```
                    Orchestrator (24/7)
                         |
           +-------------+-------------+
           |             |             |
           v             v             v
    Methodology       Math         Optimizer
    (daily 06:00)   (Mon 08:00)   (daily 07:00)
           |             |             |
           v             v             v
    Run pipeline    Analyze        Read methodology
    Evaluate BT     scoring funcs  Read math proposals
    Publish to bus  Propose        Ask Claude/GPT
           |        improvements   Execute changes
           |             |         Run backtest
           |             v         Log results
           |        Save proposals      |
           |             |              |
           +-------> AgentBus <---------+
                    (pub/sub)
```

1. **Methodology Agent** runs first: executes the full pipeline, evaluates
   backtest quality, computes signal stack health, stress resilience, and
   publishes structured results to AgentBus.

2. **Math Agent** (weekly): reads current scoring functions, sends them to
   dual LLM bridge (Claude + GPT), receives mathematical improvement proposals,
   saves as JSON + Python files in `agents/math/math_proposals/`.

3. **Optimizer Agent** runs after methodology: reads methodology report + math
   proposals, loads all parameters with ranges, builds a system prompt for
   Claude, executes a multi-turn loop with `edit_param` / `edit_code` actions.
   Auto-backs-up before changes, runs tests, reverts on failure.

4. **Orchestrator** manages scheduling, health monitoring, VIX spike alerts,
   morning/evening briefs, and self-healing (restarts failed agents).
