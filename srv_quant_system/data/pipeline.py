"""
data/pipeline.py

Data ingestion & persistence layer (DataLakeManager).

Requirements met:
- Concurrent fetching (ThreadPoolExecutor) for prices / macro / credit.
- Live fundamentals via /api/v3/quote/{tickers}.
- ETF holdings via /api/v3/etf-holder/{symbol}.
- SPY sector weights via stable endpoint (fallback) for accurate dispersion weights:
  /stable/etf/sector-weightings?symbol=SPY   citeturn4search7
- Forward fill cleaning for macro holidays.
- Persist three layers to Parquet via pyarrow: prices, fundamentals, weights.

Notes on endpoints:
- Legacy historical prices for equities/ETFs: /api/v3/historical-price-full/{ticker}
  supports `from` date and returns `adjClose` typically  citeturn5search14
- Legacy index historical prices: /api/v3/historical-price-full/index/%5E{symbol}  citeturn5search0
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd
import requests

from config.settings import Settings


@dataclass(frozen=True)
class ParquetArtifacts:
    prices_path: Path
    fundamentals_path: Path
    weights_path: Path

    @staticmethod
    def from_settings(settings: Settings) -> "ParquetArtifacts":
        return ParquetArtifacts(
            prices_path=settings.parquet_dir / "prices.parquet",
            fundamentals_path=settings.parquet_dir / "fundamentals.parquet",
            weights_path=settings.parquet_dir / "weights.parquet",
        )


class DataLakeManager:
    """
    Institutional-grade ingestion manager for SRV Quantamental system.

    Design principles:
    - Deterministic, reproducible data snapshots.
    - Graceful degradation: partial data is allowed, but logged.
    - Explicit no-lookahead: raw data only; analytics performs windowing.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifacts = ParquetArtifacts.from_settings(settings)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()

    # -----------------------------
    # Cache management
    # -----------------------------
    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.settings.cache_max_age_hours <= 0:
            return False
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds <= self.settings.cache_max_age_hours * 3600

    def is_snapshot_fresh(self) -> bool:
        return all(
            self._is_cache_fresh(p)
            for p in [self.artifacts.prices_path, self.artifacts.fundamentals_path, self.artifacts.weights_path]
        )

    # -----------------------------
    # HTTP helpers
    # -----------------------------
    def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 6,
        backoff_base_s: float = 0.75,
    ) -> Any:
        """
        Robust GET with exponential backoff.

        Handles:
        - transient 429/5xx
        - network timeouts
        - JSON decode errors
        """
        params = dict(params or {})
        params.setdefault("apikey", self.settings.fmp_api_key)

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.request_timeout_s)
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_s = backoff_base_s * (2**attempt) + 0.05 * attempt
                    self.logger.warning(
                        "Transient HTTP %s for %s (attempt %s/%s). Sleeping %.2fs",
                        resp.status_code,
                        url,
                        attempt + 1,
                        max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON decode failed for {url}: {resp.text[:200]}") from e
            except Exception as e:
                last_err = e
                sleep_s = backoff_base_s * (2**attempt) + 0.05 * attempt
                self.logger.warning(
                    "Request failed for %s (attempt %s/%s): %s. Sleeping %.2fs",
                    url,
                    attempt + 1,
                    max_retries,
                    repr(e),
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Request failed after {max_retries} retries for {url}") from last_err

    # -----------------------------
    # FMP endpoints
    # -----------------------------
    def _url_api_v3(self, path: str) -> str:
        return f"{self.settings.fmp_base_url}{self.settings.fmp_api_v3_path}{path}"

    def _url_stable(self, path: str) -> str:
        return f"{self.settings.fmp_base_url}{self.settings.fmp_stable_path}{path}"

    def _historical_price_url(self, ticker: str) -> str:
        """
        Legacy logic + robustness:
        - Try /api/v3/historical-price-full/{ticker}
        - If ticker indicates an index (^...), FMP legacy supports
          /api/v3/historical-price-full/index/%5E...  citeturn5search0
        """
        if ticker.startswith("^"):
            # Legacy index endpoint expects URL-encoded caret symbols under /index/
            return self._url_api_v3(f"/historical-price-full/index/{quote(ticker)}")
        return self._url_api_v3(f"/historical-price-full/{quote(ticker)}")

    def fetch_price_history(
        self,
        ticker: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """
        Fetch historical daily adjusted close (if available) as a pandas Series.

        - Uses legacy endpoint required by spec.
        - Uses `from` parameter (supported by legacy endpoint; also common in FMP examples)  citeturn5search14
        """
        url = self._historical_price_url(ticker)
        params: Dict[str, Any] = {"from": start_date.isoformat()}
        if end_date is not None:
            params["to"] = end_date.isoformat()

        payload = self._request_json(url, params=params)

        # Expected structure: {"symbol": "...", "historical": [{...}, ...]}
        hist = None
        if isinstance(payload, dict):
            hist = payload.get("historical")
        elif isinstance(payload, list):
            # Some endpoints can return list; normalize
            hist = payload

        if not hist:
            # Robust fallback: try stable endpoint for index symbols if legacy returned empty
            # (not a placeholder; operational fallback)
            self.logger.warning("Empty history for %s on legacy endpoint. Attempting stable fallback.", ticker)
            stable_series = self._fetch_price_history_stable_fallback(ticker, start_date, end_date)
            if stable_series is not None and not stable_series.empty:
                return stable_series
            raise ValueError(f"No historical data returned for {ticker}")

        df = pd.DataFrame(hist)
        if df.empty or "date" not in df.columns:
            raise ValueError(f"Malformed historical response for {ticker}: missing 'date'")

        df["date"] = pd.to_datetime(df["date"], utc=False)
        df = df.sort_values("date")

        price_col = "adjClose" if "adjClose" in df.columns else "close"
        if price_col not in df.columns:
            raise ValueError(f"Malformed historical response for {ticker}: missing {price_col}")

        s = pd.to_numeric(df[price_col], errors="coerce")
        s.index = pd.to_datetime(df["date"])
        s.name = ticker
        s = s[~s.index.duplicated(keep="last")].dropna()
        return s

    def _fetch_price_history_stable_fallback(
        self, ticker: str, start_date: date, end_date: Optional[date]
    ) -> Optional[pd.Series]:
        """
        Stable fallback for edge symbols (indexes/ICE symbols).
        Uses stable historical full chart endpoint:
        /stable/historical-price-eod/full?symbol=...  citeturn5search1

        Note: This is a resilience improvement; the core system still uses legacy endpoints as requested.
        """
        try:
            # Works for both equities and indexes on stable API
            url = self._url_stable("/historical-price-eod/full")
            params: Dict[str, Any] = {"symbol": ticker, "from": start_date.isoformat()}
            if end_date is not None:
                params["to"] = end_date.isoformat()

            payload = self._request_json(url, params=params)
            if not isinstance(payload, list) or len(payload) == 0:
                return None

            df = pd.DataFrame(payload)
            if "date" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"], utc=False)
            df = df.sort_values("date")
            price_col = "adjClose" if "adjClose" in df.columns else "close"
            if price_col not in df.columns:
                return None
            s = pd.to_numeric(df[price_col], errors="coerce")
            s.index = pd.to_datetime(df["date"])
            s.name = ticker
            return s[~s.index.duplicated(keep="last")].dropna()
        except Exception as e:
            self.logger.warning("Stable fallback failed for %s: %s", ticker, repr(e))
            return None

    def fetch_etf_holdings(self, etf_symbol: str) -> pd.DataFrame:
        """
        Fetch ETF holdings (constituent weights) via legacy endpoint:
            /api/v3/etf-holder/{symbol}

        Practical note:
        - Field names across vendor versions may vary.
        - We normalize into:
            etfSymbol, asset, weightPercentage, sharesNumber, marketValue, updated

        (Even if some fields are missing, weightPercentage + asset is enough for sector valuation aggregation.)
        """
        url = self._url_api_v3(f"/etf-holder/{quote(etf_symbol)}")
        payload = self._request_json(url, params={})

        if not isinstance(payload, list) or len(payload) == 0:
            raise ValueError(f"No holdings returned for ETF {etf_symbol}")

        df = pd.DataFrame(payload)
        df["etfSymbol"] = etf_symbol

        # Normalize columns
        # Common keys seen in practice: asset, weightPercentage, sharesNumber, marketValue, updated
        for col in ["asset", "weightPercentage", "sharesNumber", "marketValue", "updated"]:
            if col not in df.columns:
                df[col] = None

        df["weightPercentage"] = pd.to_numeric(df["weightPercentage"], errors="coerce")
        df["sharesNumber"] = pd.to_numeric(df["sharesNumber"], errors="coerce")
        df["marketValue"] = pd.to_numeric(df["marketValue"], errors="coerce")

        # Basic sanitation
        df["asset"] = df["asset"].astype(str).str.strip()
        df = df[df["asset"].notna() & (df["asset"] != "None") & (df["asset"] != "nan")]

        df["record_type"] = "holding"
        return df[
            ["record_type", "etfSymbol", "asset", "weightPercentage", "sharesNumber", "marketValue", "updated"]
        ].copy()

    def fetch_spy_sector_weightings(self) -> pd.DataFrame:
        """
        Fetch SPY sector weights via FMP stable endpoint:
            /stable/etf/sector-weightings?symbol=SPY  citeturn4search7

        This is the cleanest way to get sector-level weights for realized dispersion using variance decomposition.
        """
        url = self._url_stable("/etf/sector-weightings")
        payload = self._request_json(url, params={"symbol": self.settings.spy_ticker})

        if not isinstance(payload, list) or len(payload) == 0:
            raise ValueError("No sector weighting returned for SPY")

        df = pd.DataFrame(payload)
        # stable response typically includes: sector, weightPercentage
        if "sector" not in df.columns or "weightPercentage" not in df.columns:
            raise ValueError("Malformed sector weighting payload for SPY")

        df["record_type"] = "sector_weight"
        df["etfSymbol"] = self.settings.spy_ticker
        df["sector"] = df["sector"].astype(str).str.strip()
        df["weightPercentage"] = pd.to_numeric(df["weightPercentage"], errors="coerce")
        df["updated"] = None
        df["asset"] = None
        df["sharesNumber"] = None
        df["marketValue"] = None
        return df[
            ["record_type", "etfSymbol", "sector", "weightPercentage", "asset", "sharesNumber", "marketValue", "updated"]
        ].copy()

    def fetch_quotes(self, symbols: Sequence[str]) -> pd.DataFrame:
        """
        Fetch quotes (fundamentals proxy) via legacy endpoint:
            /api/v3/quote/{tickers}

        We chunk requests to avoid URL length and reduce server strain.
        """
        symbols_clean: List[str] = [s.strip() for s in symbols if s and isinstance(s, str)]
        symbols_clean = [s for s in symbols_clean if not s.startswith("^")]  # quote endpoint often excludes indexes
        symbols_clean = list(dict.fromkeys(symbols_clean))  # de-duplicate while preserving order

        if not symbols_clean:
            return pd.DataFrame()

        out_rows: List[Dict[str, Any]] = []
        chunk = self.settings.quote_chunk_size

        for i in range(0, len(symbols_clean), chunk):
            part = symbols_clean[i : i + chunk]
            csv_symbols = ",".join(part)
            url = self._url_api_v3(f"/quote/{quote(csv_symbols, safe=',')}")
            try:
                payload = self._request_json(url, params={})
                if isinstance(payload, list):
                    out_rows.extend(payload)
                else:
                    self.logger.warning("Unexpected quote payload type for chunk %s..%s: %s", i, i + chunk, type(payload))
            except Exception as e:
                self.logger.exception("Quote fetch failed for chunk starting %s: %s", i, repr(e))

        df = pd.DataFrame(out_rows)
        if df.empty or "symbol" not in df.columns:
            return pd.DataFrame()

        # Normalize numeric fields we rely on
        for col in ["price", "pe", "eps", "marketCap", "sharesOutstanding"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["asof"] = datetime.utcnow().isoformat()
        return df

    # -----------------------------
    # Orchestration
    # -----------------------------
    def _fetch_prices_block(self, tickers: Sequence[str], start: date, end: Optional[date]) -> pd.DataFrame:
        """
        Fetch a block of tickers concurrently -> wide DataFrame indexed by date.
        """
        series_list: List[pd.Series] = []

        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, max(2, len(tickers)))) as ex:
            fut_map = {ex.submit(self.fetch_price_history, t, start, end): t for t in tickers}
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    s = fut.result()
                    series_list.append(s)
                except Exception as e:
                    self.logger.exception("Failed to fetch history for %s: %s", t, repr(e))

        if not series_list:
            return pd.DataFrame()

        df = pd.concat(series_list, axis=1).sort_index()
        return df

    def build_snapshot(self, force_refresh: bool = False) -> ParquetArtifacts:
        """
        Build or reuse a local Parquet snapshot (prices, fundamentals, weights).

        If cache is fresh and not forced -> reuse existing Parquet files.
        """
        if not force_refresh and self.is_snapshot_fresh():
            self.logger.info("Using fresh cached snapshot under %s", self.settings.parquet_dir)
            return self.artifacts

        self.logger.info("Building fresh snapshot...")

        today = date.today()
        start_date = today - timedelta(days=int(self.settings.history_years * 365.25))
        end_date = None  # keep open-ended (up to vendor latest)

        # --- Concurrent blocks: prices vs macro vs credit ---
        sector_and_spy = self.settings.sector_list() + [self.settings.spy_ticker]
        macro = list(self.settings.macro_tickers.values()) + list(self.settings.vol_tickers.values())
        credit = list(self.settings.credit_tickers.values())

        self.logger.info("Fetching price histories: %s tickers", len(sector_and_spy) + len(macro) + len(credit))
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                "equities": ex.submit(self._fetch_prices_block, sector_and_spy, start_date, end_date),
                "macro": ex.submit(self._fetch_prices_block, macro, start_date, end_date),
                "credit": ex.submit(self._fetch_prices_block, credit, start_date, end_date),
            }
            blocks: Dict[str, pd.DataFrame] = {k: fut.result() for k, fut in futs.items()}

        prices = pd.concat([blocks.get("equities", pd.DataFrame()), blocks.get("macro", pd.DataFrame()), blocks.get("credit", pd.DataFrame())], axis=1)

        if prices.empty:
            raise RuntimeError("Price snapshot is empty; cannot proceed.")

        prices = prices[~prices.index.duplicated(keep="last")].sort_index()

        # Cleaning: forward-fill to reduce NaNs across macro holidays / mismatched calendars.
        # We limit ffill to a reasonable bound to avoid dragging stale values for too long.
        prices = prices.ffill(limit=5)

        # Persist prices
        prices.to_parquet(self.artifacts.prices_path, engine="pyarrow", compression="snappy")
        self.logger.info("Saved prices to %s (rows=%s cols=%s)", self.artifacts.prices_path, len(prices), prices.shape[1])

        # --- ETF holdings weights (sector ETFs + SPY) ---
        weight_frames: List[pd.DataFrame] = []
        etfs_for_holdings = list(dict.fromkeys(self.settings.sector_list() + [self.settings.spy_ticker]))

        self.logger.info("Fetching ETF holdings for %s ETFs", len(etfs_for_holdings))
        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, max(2, len(etfs_for_holdings)))) as ex:
            fut_map = {ex.submit(self.fetch_etf_holdings, sym): sym for sym in etfs_for_holdings}
            for fut in as_completed(fut_map):
                sym = fut_map[fut]
                try:
                    weight_frames.append(fut.result())
                except Exception as e:
                    self.logger.exception("Failed to fetch holdings for %s: %s", sym, repr(e))

        # SPY sector weights (for dispersion)
        try:
            spy_sector_weights = self.fetch_spy_sector_weightings()
            weight_frames.append(spy_sector_weights)
        except Exception as e:
            self.logger.warning("SPY sector weights unavailable; analytics will fallback to equal sector weights. %s", repr(e))

        weights_df = pd.concat(weight_frames, axis=0, ignore_index=True) if weight_frames else pd.DataFrame()
        if weights_df.empty:
            self.logger.warning("Weights snapshot is empty; fundamentals aggregation may degrade.")
        else:
            weights_df.to_parquet(self.artifacts.weights_path, engine="pyarrow", compression="snappy")
            self.logger.info("Saved weights to %s (rows=%s)", self.artifacts.weights_path, len(weights_df))

        # --- Fundamentals (quotes) ---
        # Build a quote universe:
        # - sector ETFs + SPY
        # - all holdings in those ETFs (to compute sector-level valuation precisely via harmonic PE aggregation)
        quote_symbols: List[str] = list(dict.fromkeys(sector_and_spy))

        if not weights_df.empty and "record_type" in weights_df.columns:
            holding_assets = (
                weights_df.loc[weights_df["record_type"] == "holding", "asset"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            quote_symbols.extend(holding_assets)

        self.logger.info("Fetching fundamentals/quotes for %s unique symbols", len(set(quote_symbols)))
        fundamentals = self.fetch_quotes(quote_symbols)

        fundamentals.to_parquet(self.artifacts.fundamentals_path, engine="pyarrow", compression="snappy")
        self.logger.info(
            "Saved fundamentals to %s (rows=%s cols=%s)",
            self.artifacts.fundamentals_path,
            len(fundamentals),
            fundamentals.shape[1],
        )

        return self.artifacts
