"""
DatabaseWriter: replaces all DataFrame.to_parquet() calls.

Write strategy:
  - UPSERT via INSERT OR REPLACE (idempotent — safe to re-run daily)
  - Prices use long format internally; DatabaseReader pivots back to wide
  - All writes include snapshot_date for time-travel queries
  - Every call logs row counts for observability
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from db.connection import get_connection

logger = logging.getLogger(__name__)


def _sf(x: Any) -> Optional[float]:
    """Safe float — returns None for NaN / Inf / non-numeric."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


class DatabaseWriter:
    """Write market data, holdings, fundamentals, and analytics results to DuckDB."""

    def __init__(self, db_path: Path) -> None:
        self.conn = get_connection(db_path)

    # ── Prices ────────────────────────────────────────────────────────────────

    def write_prices(self, prices_df: pd.DataFrame) -> int:
        """
        Write prices wide DataFrame (DatetimeIndex × ticker columns) to market_data.prices.
        Upserts all rows — safe to call multiple times with overlapping data.
        Returns number of rows upserted.
        """
        if prices_df is None or prices_df.empty:
            logger.warning("write_prices: empty DataFrame — skipping")
            return 0

        df = prices_df.copy()
        df.index = pd.to_datetime(df.index).date
        df.index.name = "date"

        long = (
            df.reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="close")
            .dropna(subset=["close"])
        )
        long["date"] = pd.to_datetime(long["date"]).dt.date
        long["close"] = pd.to_numeric(long["close"], errors="coerce")
        long = long.dropna(subset=["close"])
        long = long[long["close"].apply(lambda x: math.isfinite(x))]

        if long.empty:
            return 0

        self.conn.register("_prices_staging", long)
        self.conn.execute("""
            INSERT OR REPLACE INTO market_data.prices (date, ticker, close)
            SELECT date, ticker, close FROM _prices_staging
        """)
        self.conn.unregister("_prices_staging")

        n = len(long)
        logger.info(
            "write_prices: upserted %d rows | %d tickers | %d–%s dates",
            n,
            long["ticker"].nunique(),
            long["date"].min().year if hasattr(long["date"].min(), "year") else "?",
            long["date"].max(),
        )
        return n

    # ── Holdings ──────────────────────────────────────────────────────────────

    def write_weights(self, weights_df: pd.DataFrame) -> int:
        """
        Write holdings DataFrame to holdings.etf_holdings + holdings.spy_sector_weights.
        Accepts the same format produced by DataLakeManager (record_type column).
        """
        if weights_df is None or weights_df.empty:
            logger.warning("write_weights: empty DataFrame — skipping")
            return 0

        snap_date = date.today()
        n_total = 0

        # ── ETF Holdings ──────────────────────────────────────────────────────
        if "record_type" in weights_df.columns:
            holdings_mask = weights_df["record_type"] == "holding"
            holdings_raw = weights_df[holdings_mask].copy()
        else:
            holdings_raw = weights_df.copy()

        if not holdings_raw.empty:
            # Normalise column names (pipeline uses camelCase)
            etf_col    = next((c for c in ["etfSymbol", "etf_symbol"] if c in holdings_raw.columns), None)
            asset_col  = next((c for c in ["asset"] if c in holdings_raw.columns), None)
            wt_col     = next((c for c in ["weightPercentage", "weight_pct"] if c in holdings_raw.columns), None)
            shr_col    = next((c for c in ["sharesNumber", "shares_number"] if c in holdings_raw.columns), None)
            mv_col     = next((c for c in ["marketValue", "market_value"] if c in holdings_raw.columns), None)

            h = pd.DataFrame({
                "snapshot_date": snap_date,
                "etf_symbol":    holdings_raw[etf_col]   if etf_col  else None,
                "asset":         holdings_raw[asset_col] if asset_col else None,
                "weight_pct":    pd.to_numeric(holdings_raw[wt_col],  errors="coerce") if wt_col  else None,
                "shares_number": pd.to_numeric(holdings_raw[shr_col], errors="coerce") if shr_col else None,
                "market_value":  pd.to_numeric(holdings_raw[mv_col],  errors="coerce") if mv_col  else None,
            }).dropna(subset=["etf_symbol", "asset"])

            if not h.empty:
                self.conn.register("_holdings_staging", h)
                self.conn.execute("""
                    INSERT OR REPLACE INTO holdings.etf_holdings
                        (snapshot_date, etf_symbol, asset, weight_pct, shares_number, market_value)
                    SELECT snapshot_date, etf_symbol, asset, weight_pct, shares_number, market_value
                    FROM _holdings_staging
                """)
                self.conn.unregister("_holdings_staging")
                n_total += len(h)

        # ── SPY Sector Weights ────────────────────────────────────────────────
        if "record_type" in weights_df.columns:
            spy_raw = weights_df[weights_df["record_type"] == "sector_weight"].copy()
        else:
            spy_raw = pd.DataFrame()

        if not spy_raw.empty:
            sector_col = next((c for c in ["sector", "etfSymbol"] if c in spy_raw.columns), None)
            wt_col     = next((c for c in ["weightPercentage", "weight_pct"] if c in spy_raw.columns), None)

            s = pd.DataFrame({
                "snapshot_date": snap_date,
                "sector":    spy_raw[sector_col] if sector_col else None,
                "weight_pct": pd.to_numeric(spy_raw[wt_col], errors="coerce") if wt_col else None,
            }).dropna(subset=["sector"])

            if not s.empty:
                self.conn.register("_spy_staging", s)
                self.conn.execute("""
                    INSERT OR REPLACE INTO holdings.spy_sector_weights
                        (snapshot_date, sector, weight_pct)
                    SELECT snapshot_date, sector, weight_pct FROM _spy_staging
                """)
                self.conn.unregister("_spy_staging")
                n_total += len(s)

        logger.info("write_weights: upserted %d total rows", n_total)
        return n_total

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def write_fundamentals(self, fund_df: pd.DataFrame) -> int:
        """Write batch-quote fundamentals (symbol, price, pe, eps, marketCap)."""
        if fund_df is None or fund_df.empty:
            return 0

        snap_date = date.today()
        col_map = {
            "symbol": "symbol", "price": "price",
            "pe": "pe", "eps": "eps", "marketCap": "market_cap",
        }
        available = {src: dst for src, dst in col_map.items() if src in fund_df.columns}
        df = fund_df[list(available.keys())].copy().rename(columns=available)
        df["snapshot_date"] = snap_date

        for c in ["price", "pe", "eps", "market_cap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["symbol"])
        if df.empty:
            return 0

        # Ensure all expected columns exist
        for c in ["price", "pe", "eps", "market_cap"]:
            if c not in df.columns:
                df[c] = None

        self.conn.register("_quotes_staging", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO fundamentals.quotes
                (snapshot_date, symbol, price, pe, eps, market_cap)
            SELECT snapshot_date, symbol, price, pe, eps, market_cap
            FROM _quotes_staging
        """)
        self.conn.unregister("_quotes_staging")
        logger.info("write_fundamentals: upserted %d rows", len(df))
        return len(df)

    def write_extended_fundamentals(self, ext_df: pd.DataFrame) -> int:
        """Write extended fundamentals (all TTM / estimate columns) to fundamentals.ratios."""
        if ext_df is None or ext_df.empty:
            return 0

        snap_date = date.today()

        # Map every source column the pipeline produces → DB column name
        col_map = {
            "symbol":                  "symbol",
            "pe":                      "pe_ttm",
            "priceToBookRatio":        "pb",
            "priceToSalesRatio":       "ps",
            "enterpriseValueMultiple": "ev_ebitda",
            "ev_ebit":                 "ev_ebit",
            "dividendYield":           "div_yield",
            "freeCashFlowYield":       "fcf_yield",
            "roe":                     "roe",
            "roa":                     "roa",
            "returnOnInvestedCapital": "roic",
            "netProfitMargin":         "net_margin",
            "grossProfitMargin":       "gross_margin",
            "operatingProfitMargin":   "operating_margin",
            "debtEquityRatio":         "debt_equity",
            "currentRatio":            "current_ratio",
            "interestCoverage":        "interest_coverage",
            "accrual_ratio":           "accrual_ratio",
            "epsGrowth":               "eps_growth_yoy",
            "revenueGrowth":           "revenue_growth_yoy",
            "ebitdaGrowth":            "ebitda_growth_yoy",
            "forwardPE":               "forward_pe",
            "ntmEpsConsensus":         "ntm_eps_consensus",
            "avg_eps_surprise_pct":    "avg_eps_surprise_pct",
            "enterpriseValue":         "enterprise_value",
            "ebitda":                  "ebitda",
            "operatingCashFlow":       "ocf",
            "capitalExpenditure":      "capex",
        }

        available = {src: dst for src, dst in col_map.items() if src in ext_df.columns}
        df = ext_df[list(available.keys())].copy().rename(columns=available)
        df["snapshot_date"] = snap_date

        for c in df.columns:
            if c not in ("symbol", "snapshot_date"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["symbol"])
        if df.empty:
            return 0

        db_cols = ["snapshot_date", "symbol"] + [v for v in available.values() if v != "symbol"]
        cols_sql = ", ".join(db_cols)

        self.conn.register("_ratios_staging", df[db_cols])
        self.conn.execute(f"""
            INSERT OR REPLACE INTO fundamentals.ratios ({cols_sql})
            SELECT {cols_sql} FROM _ratios_staging
        """)
        self.conn.unregister("_ratios_staging")
        logger.info("write_extended_fundamentals: upserted %d rows", len(df))
        return len(df)

    # ── Analytics audit trail ─────────────────────────────────────────────────

    def write_run(
        self,
        master_df: pd.DataFrame,
        started_at: datetime,
        finished_at: datetime,
        data_health_label: str = "UNKNOWN",
    ) -> int:
        """
        Persist one analytics run (signals + metadata) to the audit trail.
        Returns the auto-generated run_id.
        """
        if master_df is None or master_df.empty:
            return -1

        def _col0(name: str) -> Any:
            return master_df[name].iloc[0] if name in master_df.columns and len(master_df) else None

        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        self.conn.execute("""
            INSERT INTO analytics.runs
                (run_date, run_started_at, run_finished_at, duration_ms,
                 market_state, regime_alert, avg_corr,
                 n_longs, n_shorts, avg_mc, data_health_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            date.today(),
            started_at,
            finished_at,
            duration_ms,
            str(_col0("market_state") or ""),
            str(_col0("regime_alert") or ""),
            _sf(_col0("avg_corr_t")),
            int((master_df["direction"] == "LONG").sum())  if "direction" in master_df.columns else 0,
            int((master_df["direction"] == "SHORT").sum()) if "direction" in master_df.columns else 0,
            _sf(master_df["mc_score"].mean()) if "mc_score" in master_df.columns else None,
            data_health_label,
        ])

        run_id_row = self.conn.execute("SELECT max(run_id) FROM analytics.runs").fetchone()
        run_id = int(run_id_row[0]) if run_id_row and run_id_row[0] is not None else -1

        # ── Sector signals ────────────────────────────────────────────────────
        src_to_db = {
            "sector_ticker":     "sector_ticker",
            "sector_name":       "sector_name",
            "direction":         "direction",
            "pca_residual_z":    "pca_residual_z",
            "mc_score":          "mc_score",
            "conviction_score":  "conviction_score",
            "sds_score":         "sds_score",
            "fjs_score":         "fjs_score",
            "mss_score":         "mss_score",
            "stf_score":         "stf_score",
            "w_final":           "w_final",
            "hedge_ratio":       "hedge_ratio",
            "decision_label":    "decision_label",
            "interpretation":    "interpretation",
            "action_bias":       "action_bias",
            "risk_label":        "risk_label",
            "pm_note":           "pm_note",
            "fjs_engine_score":  "fjs_engine_score",
            "rel_pe_vs_spy":     "rel_pe_vs_spy",
            "half_life_days_est": "half_life_days",
            "beta_tnx_60d":      "beta_tnx_60d",
            "beta_dxy_60d":      "beta_dxy_60d",
            "beta_spy_delta":    "beta_spy_delta",
            "corr_to_spy_delta": "corr_to_spy_delta",
        }
        _str_cols = {"sector_ticker", "sector_name", "direction", "decision_label",
                     "interpretation", "action_bias", "risk_label", "pm_note"}

        rows = []
        for _, row in master_df.iterrows():
            rec: dict = {"run_id": run_id}
            for src, db in src_to_db.items():
                v = row.get(src)
                rec[db] = str(v) if db in _str_cols else _sf(v)
            rows.append(rec)

        if rows:
            sig_df = pd.DataFrame(rows)
            self.conn.register("_signals_staging", sig_df)
            cols_sql = ", ".join(sig_df.columns)
            self.conn.execute(f"""
                INSERT OR REPLACE INTO analytics.sector_signals ({cols_sql})
                SELECT {cols_sql} FROM _signals_staging
            """)
            self.conn.unregister("_signals_staging")

        logger.info(
            "write_run: run_id=%d | %d sectors | %s market_state | %dms",
            run_id, len(rows), str(_col0("market_state") or "?"), duration_ms,
        )
        return run_id

    # ── Backtest cache ────────────────────────────────────────────────────────

    def write_backtest_cache(self, result: "BacktestResult") -> None:
        """Cache backtest summary to avoid expensive re-runs."""
        import json
        regime_json = json.dumps({
            k: {"ic_mean": v.ic_mean, "sharpe": v.sharpe, "hit_rate": v.hit_rate}
            for k, v in result.regime_breakdown.items()
        })
        params_json = json.dumps({
            "train_window": result.train_window,
            "fwd_period": result.fwd_period,
            "step": result.step,
        })
        self.conn.execute("""
            INSERT OR REPLACE INTO analytics.backtest_cache
                (cache_date, ic_mean, ic_ir, hit_rate, sharpe, max_drawdown,
                 n_walks, n_sectors, regime_json, params_json)
            VALUES (today(), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [result.ic_mean, result.ic_ir, result.hit_rate, result.sharpe,
              result.max_drawdown, result.n_walks, result.n_sectors,
              regime_json, params_json])
        logger.info("write_backtest_cache: cached IC_mean=%.4f, Sharpe=%.2f",
                    result.ic_mean or 0, result.sharpe or 0)

    # ── ML predictions ────────────────────────────────────────────────────────

    def write_ml_predictions(self, run_id: int, predictions: dict, regime_ic: dict, model_type: str) -> None:
        """Write ML signal quality predictions to DB."""
        rows = [{"run_id": run_id, "sector_ticker": ticker,
                 "quality_score": float(q),
                 "regime_ic_mean": float(regime_ic.get(ticker, float("nan"))),
                 "model_type": model_type}
                for ticker, q in predictions.items()]
        if rows:
            df = pd.DataFrame(rows)
            self.conn.register("_ml_staging", df)
            self.conn.execute("""
                INSERT OR REPLACE INTO analytics.ml_predictions
                    (run_id, sector_ticker, quality_score, regime_ic_mean, model_type)
                SELECT run_id, sector_ticker, quality_score, regime_ic_mean, model_type
                FROM _ml_staging
            """)
            self.conn.unregister("_ml_staging")
        logger.info("write_ml_predictions: %d sectors", len(rows))

    # ── Optimization results ──────────────────────────────────────────────────

    def write_optimization_results(self, run_id: int, opt_df: pd.DataFrame) -> None:
        """Write portfolio optimizer output to DB."""
        if opt_df is None or opt_df.empty:
            return
        cols_needed = ["sector_ticker", "opt_weight", "risk_parity_w", "w_final"]
        available = [c for c in cols_needed if c in opt_df.columns]
        df = opt_df[available].copy()
        df["run_id"] = run_id
        if "w_final" in df.columns:
            df = df.rename(columns={"w_final": "signal_weight"})
        self.conn.register("_opt_staging", df)
        cols_sql = ", ".join(df.columns)
        self.conn.execute(f"""
            INSERT OR REPLACE INTO analytics.optimization_results ({cols_sql})
            SELECT {cols_sql} FROM _opt_staging
        """)
        self.conn.unregister("_opt_staging")
        logger.info("write_optimization_results: %d sectors", len(df))

    # ── Data pruning ──────────────────────────────────────────────────────────

    def prune_old_snapshots(self, snapshot_retain_days: int = 90, holdings_retain_days: int = 30) -> None:
        """
        Delete old snapshot rows from fundamentals and holdings to prevent data bloat.
        Prices are NEVER pruned (needed for full 10-year history).
        """
        from datetime import date, timedelta
        fund_cutoff = date.today() - timedelta(days=snapshot_retain_days)
        hold_cutoff = date.today() - timedelta(days=holdings_retain_days)

        for table, cutoff in [
            ("fundamentals.quotes",             fund_cutoff),
            ("fundamentals.ratios",             fund_cutoff),
            ("holdings.etf_holdings",           hold_cutoff),
            ("holdings.spy_sector_weights",     hold_cutoff),
        ]:
            try:
                self.conn.execute(
                    f"DELETE FROM {table} WHERE snapshot_date < ?", [cutoff]
                ).fetchone()
                logger.info("prune %s: deleted rows before %s", table, cutoff)
            except Exception as e:
                logger.warning("prune %s failed (non-fatal): %s", table, e)
