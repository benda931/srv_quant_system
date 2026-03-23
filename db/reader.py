"""
DatabaseReader: replaces pd.read_parquet() calls throughout the system.

Returns DataFrames in the EXACT same format QuantEngine expects:
  - read_prices()  → wide DataFrame: DatetimeIndex × ticker columns
  - read_weights() → long DataFrame: record_type, etfSymbol, asset, weightPercentage, ...
  - read_fundamentals() → symbol, price, pe, eps, marketCap
  - read_extended_fundamentals() → symbol + all TTM / estimate columns
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from db.connection import get_connection

logger = logging.getLogger(__name__)


class DatabaseReader:
    """Read market data from DuckDB, returning DataFrames compatible with QuantEngine."""

    def __init__(self, db_path: Path) -> None:
        self.conn = get_connection(db_path)

    # ── Freshness ─────────────────────────────────────────────────────────────

    def is_snapshot_fresh(self, max_age_hours: int = 12) -> bool:
        """True if prices contain data within max_age_hours (accounting for weekends)."""
        try:
            row = self.conn.execute("SELECT max(date) FROM market_data.prices").fetchone()
            if not row or row[0] is None:
                return False
            last = row[0]
            if not isinstance(last, date):
                last = date.fromisoformat(str(last))
            # Allow up to max_age_hours + 72h buffer for weekends / public holidays
            cutoff = date.today() - timedelta(hours=max_age_hours + 72)
            return last >= cutoff
        except Exception:
            return False

    def last_prices_date(self) -> Optional[date]:
        """Return the most recent price date in the database."""
        try:
            row = self.conn.execute("SELECT max(date) FROM market_data.prices").fetchone()
            if row and row[0] is not None:
                d = row[0]
                return d if isinstance(d, date) else date.fromisoformat(str(d))
        except Exception:
            pass
        return None

    # ── Market data ───────────────────────────────────────────────────────────

    def read_prices(self) -> pd.DataFrame:
        """
        Read prices as wide DataFrame: DatetimeIndex × ticker columns.
        Identical format to pd.read_parquet(prices_path).
        """
        df_long = self.conn.execute(
            "SELECT date, ticker, close FROM market_data.prices ORDER BY date ASC"
        ).df()

        if df_long.empty:
            logger.warning("read_prices: database is empty")
            return pd.DataFrame()

        df_wide = df_long.pivot(index="date", columns="ticker", values="close")
        df_wide.index = pd.to_datetime(df_wide.index)
        df_wide.columns.name = None
        df_wide = df_wide.sort_index()

        logger.debug("read_prices: %d dates × %d tickers", len(df_wide), len(df_wide.columns))
        return df_wide

    # ── Holdings ──────────────────────────────────────────────────────────────

    def read_weights(self) -> pd.DataFrame:
        """
        Read latest holdings snapshot as long DataFrame matching weights.parquet schema.
        Columns: record_type, etfSymbol, asset, weightPercentage, sharesNumber, marketValue, sector, updated
        """
        h_snap = self.conn.execute(
            "SELECT max(snapshot_date) FROM holdings.etf_holdings"
        ).fetchone()
        h_snap = h_snap[0] if h_snap and h_snap[0] is not None else None

        if h_snap is None:
            return pd.DataFrame()

        holdings = self.conn.execute("""
            SELECT
                'holding'       AS record_type,
                etf_symbol      AS etfSymbol,
                asset,
                weight_pct      AS weightPercentage,
                shares_number   AS sharesNumber,
                market_value    AS marketValue,
                NULL            AS sector,
                snapshot_date   AS updated
            FROM holdings.etf_holdings
            WHERE snapshot_date = ?
        """, [h_snap]).df()

        s_snap = self.conn.execute(
            "SELECT max(snapshot_date) FROM holdings.spy_sector_weights"
        ).fetchone()
        s_snap = s_snap[0] if s_snap and s_snap[0] is not None else None

        spy_df = pd.DataFrame()
        if s_snap is not None:
            spy_df = self.conn.execute("""
                SELECT
                    'sector_weight' AS record_type,
                    sector          AS etfSymbol,
                    sector          AS asset,
                    weight_pct      AS weightPercentage,
                    NULL            AS sharesNumber,
                    NULL            AS marketValue,
                    sector,
                    snapshot_date   AS updated
                FROM holdings.spy_sector_weights
                WHERE snapshot_date = ?
            """, [s_snap]).df()

        if not spy_df.empty:
            # Drop all-NA columns before concat to avoid FutureWarning
            spy_df = spy_df.dropna(axis=1, how="all")
            combined = pd.concat([holdings, spy_df], ignore_index=True)
        else:
            combined = holdings
        logger.debug("read_weights: %d rows", len(combined))
        return combined

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def read_fundamentals(self) -> pd.DataFrame:
        """
        Read latest fundamentals snapshot.
        Columns: symbol, price, pe, eps, marketCap
        """
        snap = self.conn.execute(
            "SELECT max(snapshot_date) FROM fundamentals.quotes"
        ).fetchone()
        snap = snap[0] if snap and snap[0] is not None else None

        if snap is None:
            return pd.DataFrame()

        df = self.conn.execute("""
            SELECT symbol, price, pe, eps, market_cap AS marketCap
            FROM fundamentals.quotes
            WHERE snapshot_date = ?
        """, [snap]).df()

        logger.debug("read_fundamentals: %d symbols", len(df))
        return df

    def read_extended_fundamentals(self) -> Optional[pd.DataFrame]:
        """
        Read latest extended fundamentals with column names matching pipeline output.
        Returns None if table is empty.
        """
        try:
            snap = self.conn.execute(
                "SELECT max(snapshot_date) FROM fundamentals.ratios"
            ).fetchone()
            snap = snap[0] if snap and snap[0] is not None else None

            if snap is None:
                return None

            df = self.conn.execute("""
                SELECT
                    symbol,
                    pe_ttm               AS pe,
                    pb                   AS priceToBookRatio,
                    ps                   AS priceToSalesRatio,
                    ev_ebitda            AS enterpriseValueMultiple,
                    ev_ebit,
                    div_yield            AS dividendYield,
                    fcf_yield            AS freeCashFlowYield,
                    roe,
                    roa,
                    roic                 AS returnOnInvestedCapital,
                    net_margin           AS netProfitMargin,
                    gross_margin         AS grossProfitMargin,
                    operating_margin     AS operatingProfitMargin,
                    debt_equity          AS debtEquityRatio,
                    current_ratio        AS currentRatio,
                    interest_coverage    AS interestCoverage,
                    accrual_ratio,
                    eps_growth_yoy       AS epsGrowth,
                    revenue_growth_yoy   AS revenueGrowth,
                    ebitda_growth_yoy    AS ebitdaGrowth,
                    forward_pe           AS forwardPE,
                    ntm_eps_consensus    AS ntmEpsConsensus,
                    avg_eps_surprise_pct,
                    enterprise_value     AS enterpriseValue,
                    ebitda,
                    ocf                  AS operatingCashFlow,
                    capex                AS capitalExpenditure
                FROM fundamentals.ratios
                WHERE snapshot_date = ?
            """, [snap]).df()

            return df if not df.empty else None
        except Exception as exc:
            logger.warning("read_extended_fundamentals failed: %s", exc)
            return None

    # ── Health / diagnostics ──────────────────────────────────────────────────

    def table_stats(self) -> dict:
        """Row counts + latest date for all main tables — used by health checks."""
        queries = {
            "prices":            "SELECT count(*), max(date)          FROM market_data.prices",
            "etf_holdings":      "SELECT count(*), max(snapshot_date) FROM holdings.etf_holdings",
            "spy_sector_weights":"SELECT count(*), max(snapshot_date) FROM holdings.spy_sector_weights",
            "quotes":            "SELECT count(*), max(snapshot_date) FROM fundamentals.quotes",
            "ratios":            "SELECT count(*), max(snapshot_date) FROM fundamentals.ratios",
            "runs":              "SELECT count(*), max(run_date)       FROM analytics.runs",
            "sector_signals":    "SELECT count(*), NULL               FROM analytics.sector_signals",
        }
        stats: dict = {}
        for name, sql in queries.items():
            try:
                row = self.conn.execute(sql).fetchone()
                stats[name] = {
                    "rows":   int(row[0]) if row and row[0] is not None else 0,
                    "latest": str(row[1]) if row and row[1] is not None else None,
                }
            except Exception:
                stats[name] = {"rows": 0, "latest": None}
        return stats

    def last_runs(self, n: int = 5) -> pd.DataFrame:
        """Return the N most recent analytics run records."""
        try:
            return self.conn.execute(f"""
                SELECT run_id, run_date, duration_ms, market_state,
                       avg_corr, n_longs, n_shorts, avg_mc, data_health_label
                FROM analytics.runs
                ORDER BY run_id DESC
                LIMIT {n}
            """).df()
        except Exception:
            return pd.DataFrame()
