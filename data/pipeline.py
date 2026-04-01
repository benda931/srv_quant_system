from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import pandas as pd
import requests

from config.settings import Settings

def _get_db_writer(db_path):
    """Lazy DuckDB writer — imported only when actually needed to avoid
    pyarrow/DuckDB GC conflicts at Python 3.13 interpreter shutdown."""
    try:
        from db.writer import DatabaseWriter  # noqa: PLC0415
        return DatabaseWriter(db_path)
    except Exception:
        return None


@dataclass(frozen=True)
class ParquetArtifacts:
    prices_path: Path
    fundamentals_path: Path
    weights_path: Path
    extended_fundamentals_path: Path

    @staticmethod
    def from_settings(settings: Settings) -> "ParquetArtifacts":
        return ParquetArtifacts(
            prices_path=settings.parquet_dir / "prices.parquet",
            fundamentals_path=settings.parquet_dir / "fundamentals.parquet",
            weights_path=settings.parquet_dir / "weights.parquet",
            extended_fundamentals_path=settings.parquet_dir / "extended_fundamentals.parquet",
        )


def _first_numeric(row: Dict[str, Any], keys: Sequence[str]) -> float:
    for k in keys:
        try:
            v = row.get(k)
            if v is None:
                continue
            x = float(v)
            if math.isfinite(x):
                return x
        except Exception:
            continue
    return float("nan")


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _chunks(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


class DataLakeManager:
    """
    Institutional-grade ingestion manager for the SRV Quantamental system.

    Responsibilities:
    - Fetch prices for sectors, SPY, macro, vol and credit instruments
    - Fetch ETF holdings for sector ETFs and SPY
    - Fetch SPY sector weights for dispersion analytics
    - Expand fundamentals universe from ETF-level to holdings-level
    - Persist prices / fundamentals / weights into Parquet
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifacts = ParquetArtifacts.from_settings(settings)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "srv-quant-system/1.0",
                "apikey": self.settings.fmp_api_key,
            }
        )

        pool_size = max(10, int(self.settings.max_workers) * 2)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,
            pool_block=True,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # =====================================================
    # Cache management
    # =====================================================
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
            for p in (
                self.artifacts.prices_path,
                self.artifacts.fundamentals_path,
                self.artifacts.weights_path,
                self.artifacts.extended_fundamentals_path,
            )
        )

    # =====================================================
    # URL helpers
    # =====================================================
    def _url_api_v3(self, path: str) -> str:
        return f"{self.settings.fmp_base_url}{self.settings.fmp_api_v3_path}{path}"

    def _url_stable(self, path: str) -> str:
        return f"{self.settings.fmp_base_url}{self.settings.fmp_stable_path}{path}"

    def _historical_price_url_legacy(self, ticker: str) -> str:
        if ticker.startswith("^"):
            return self._url_api_v3(f"/historical-price-full/index/{quote(ticker, safe='')}")
        return self._url_api_v3(f"/historical-price-full/{quote(ticker, safe='')}")

    # =====================================================
    # HTTP helper
    # =====================================================
    def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 6,
        backoff_base_s: float = 0.75,
    ) -> Any:
        """
        Robust GET with fail-fast behavior on 401/403 and exponential backoff
        on transient failures.
        """
        params = dict(params or {})
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.request_timeout_s)
                status = int(resp.status_code)

                if status in (401, 403):
                    snippet = (resp.text or "")[:250].replace("\n", " ").strip()
                    raise PermissionError(
                        f"HTTP {status} for {url}. "
                        f"This usually means auth/plan/legacy restriction. "
                        f"Response: {snippet}"
                    )

                if status in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = backoff_base_s * (2**attempt) + 0.05 * attempt
                    else:
                        sleep_s = backoff_base_s * (2**attempt) + 0.05 * attempt

                    self.logger.warning(
                        "Transient HTTP %s for %s (attempt %s/%s). Sleeping %.2fs",
                        status,
                        url,
                        attempt + 1,
                        max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue

                if status >= 400:
                    snippet = (resp.text or "")[:250].replace("\n", " ").strip()
                    # 4xx errors (except rate-limit 429) are client/plan errors — not retriable
                    raise PermissionError(f"HTTP {status} for {url}: {snippet}")

                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    snippet = (resp.text or "")[:250].replace("\n", " ").strip()
                    raise RuntimeError(f"JSON decode failed for {url}: {snippet}") from e

            except PermissionError:
                raise
            except Exception as e:
                last_err = e
                sleep_s = backoff_base_s * (2**attempt) + 0.05 * attempt
                self.logger.warning(
                    "Request failed for %s (attempt %s/%s): %s. Sleeping %.2fs",
                    url,
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Request failed after {max_retries} retries for {url}") from last_err

    # =====================================================
    # Prices
    # =====================================================
    def fetch_price_history(
        self,
        ticker: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """
        Stable-first historical fetch with legacy fallback.
        """
        stable_url = self._url_stable("/historical-price-eod/full")
        params: Dict[str, Any] = {"symbol": ticker, "from": start_date.isoformat()}
        if end_date is not None:
            params["to"] = end_date.isoformat()

        payload = self._request_json(stable_url, params=params)

        hist: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            hist = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("historical"), list):
                hist = payload["historical"]
            elif isinstance(payload.get("data"), list):
                hist = payload["data"]
            else:
                hist = [payload]

        if not hist:
            self.logger.warning("Empty stable history for %s. Trying legacy fallback.", ticker)
            legacy_url = self._historical_price_url_legacy(ticker)
            legacy_params: Dict[str, Any] = {"from": start_date.isoformat()}
            if end_date is not None:
                legacy_params["to"] = end_date.isoformat()
            payload2 = self._request_json(legacy_url, params=legacy_params)

            if isinstance(payload2, dict) and isinstance(payload2.get("historical"), list):
                hist = payload2["historical"]
            elif isinstance(payload2, list):
                hist = payload2
            else:
                hist = None

        if not hist:
            raise ValueError(f"No historical data returned for {ticker}")

        df = pd.DataFrame(hist)
        if df.empty or "date" not in df.columns:
            raise ValueError(f"Malformed historical response for {ticker}: missing 'date'")

        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        price_col = next((c for c in ("adjClose", "close", "price", "value") if c in df.columns), None)
        if price_col is None:
            raise ValueError(f"Malformed historical response for {ticker}: missing price column")

        s = pd.to_numeric(df[price_col], errors="coerce")
        s.index = pd.to_datetime(df["date"])
        s.name = ticker
        s = s[~s.index.duplicated(keep="last")].dropna()

        s = s[s.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            s = s[s.index <= pd.Timestamp(end_date)]

        return s

    def _fetch_prices_block(self, tickers: Sequence[str], start: date, end: Optional[date]) -> pd.DataFrame:
        series_list: List[pd.Series] = []

        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, max(2, len(tickers)))) as ex:
            fut_map = {ex.submit(self.fetch_price_history, t, start, end): t for t in tickers}
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    series_list.append(fut.result())
                except Exception as e:
                    self.logger.exception("Failed to fetch history for %s: %s", t, repr(e))

        if not series_list:
            self.logger.warning(
                "_fetch_prices_block: all %d ticker requests failed — returning empty DataFrame",
                len(fut_map),
            )
            return pd.DataFrame()

        return pd.concat(series_list, axis=1).sort_index()

    # =====================================================
    # Holdings / weights
    # =====================================================
    def fetch_etf_holdings(self, etf_symbol: str) -> pd.DataFrame:
        """
        Fetch ETF holdings and return canonical schema aligned with stat_arb.py:
        record_type='holding', etfSymbol, sector, weightPercentage, asset,
        sharesNumber, marketValue, updated
        """
        url = self._url_stable("/etf/holdings")
        payload = self._request_json(url, params={"symbol": etf_symbol})

        rows: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("holdings"), list):
                rows = payload["holdings"]
            elif isinstance(payload.get("data"), list):
                rows = payload["data"]

        if not rows:
            raise ValueError(f"No ETF holdings returned for {etf_symbol}")

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"Empty ETF holdings frame for {etf_symbol}")

        asset_col = next((c for c in ("asset", "symbol", "holding", "ticker") if c in df.columns), None)
        weight_col = next((c for c in ("weightPercentage", "weight", "weightPercent", "weight_percentage") if c in df.columns), None)
        shares_col = next((c for c in ("sharesNumber", "shares", "sharesHeld") if c in df.columns), None)
        mv_col = next((c for c in ("marketValue", "market_value", "value") if c in df.columns), None)
        upd_col = next((c for c in ("updated", "date", "reportDate") if c in df.columns), None)

        if asset_col is None:
            raise ValueError(f"ETF holdings response for {etf_symbol} missing holding symbol column")
        if weight_col is None:
            raise ValueError(f"ETF holdings response for {etf_symbol} missing weight column")

        out = pd.DataFrame(
            {
                "record_type": "holding",
                "etfSymbol": etf_symbol,
                "sector": None,
                "weightPercentage": pd.to_numeric(df[weight_col], errors="coerce"),
                "asset": df[asset_col].astype(str).str.strip(),
                "sharesNumber": pd.to_numeric(df[shares_col], errors="coerce") if shares_col else None,
                "marketValue": pd.to_numeric(df[mv_col], errors="coerce") if mv_col else None,
                "updated": df[upd_col] if upd_col else None,
            }
        )

        out = out.dropna(subset=["asset", "weightPercentage"])
        out = out[out["asset"] != ""]

        max_w = float(out["weightPercentage"].max()) if not out.empty else float("nan")
        if math.isfinite(max_w) and max_w <= 1.5:
            out["weightPercentage"] = out["weightPercentage"] * 100.0

        out = out.sort_values("weightPercentage", ascending=False).reset_index(drop=True)
        return out[
            ["record_type", "etfSymbol", "sector", "weightPercentage", "asset", "sharesNumber", "marketValue", "updated"]
        ].copy()

    def fetch_spy_sector_weightings(self) -> pd.DataFrame:
        """
        Fetch SPY sector weights in canonical schema for dispersion analytics.
        """
        url = self._url_stable("/etf/sector-weightings")
        payload = self._request_json(url, params={"symbol": self.settings.spy_ticker})

        rows: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("sectorWeightings"), list):
                rows = payload["sectorWeightings"]
            elif isinstance(payload.get("data"), list):
                rows = payload["data"]

        if not rows:
            raise ValueError("No sector weighting returned for SPY")

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Empty sector weighting payload for SPY")

        sector_col = next((c for c in ("sector", "name", "industry") if c in df.columns), None)
        weight_col = next((c for c in ("weightPercentage", "weight", "weightPercent", "weight_percentage") if c in df.columns), None)

        if sector_col is None or weight_col is None:
            raise ValueError("Malformed sector weighting payload for SPY")

        out = pd.DataFrame(
            {
                "record_type": "sector_weight",
                "etfSymbol": self.settings.spy_ticker,
                "sector": df[sector_col].astype(str).str.strip(),
                "weightPercentage": pd.to_numeric(df[weight_col], errors="coerce"),
                "asset": None,
                "sharesNumber": None,
                "marketValue": None,
                "updated": datetime.now(timezone.utc).date().isoformat(),
            }
        )

        out = out.dropna(subset=["sector", "weightPercentage"])

        max_w = float(out["weightPercentage"].max()) if not out.empty else float("nan")
        if math.isfinite(max_w) and max_w <= 1.5:
            out["weightPercentage"] = out["weightPercentage"] * 100.0

        return out[
            ["record_type", "etfSymbol", "sector", "weightPercentage", "asset", "sharesNumber", "marketValue", "updated"]
        ].copy()

    # =====================================================
    # Quotes / fundamentals
    # =====================================================
    def fetch_quotes(self, tickers: Sequence[str]) -> pd.DataFrame:
        """
        Stable-first quote / valuation fetch.

        Step 1:
        - /stable/batch-quote?symbols=...

        Step 2:
        - /stable/key-metrics-ttm?symbol=...
          fills missing PE / EPS
        """
        syms = [str(t).strip() for t in tickers if str(t).strip()]
        syms = [s for s in syms if not s.startswith("^")]
        syms_unique = _dedupe_preserve_order(syms)

        if not syms_unique:
            self.logger.warning("fetch_fundamentals_snapshot: ticker list empty after dedup/filter — returning empty")
            return pd.DataFrame(columns=["symbol", "price", "pe", "eps", "marketCap"])

        url = self._url_stable("/batch-quote")
        frames: List[pd.DataFrame] = []

        chunk_size = max(1, int(self.settings.quote_chunk_size))
        for chunk in _chunks(syms_unique, chunk_size):
            payload = self._request_json(url, params={"symbols": ",".join(chunk)})

            rows: Optional[List[Dict[str, Any]]] = None
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                if isinstance(payload.get("data"), list):
                    rows = payload["data"]
                elif isinstance(payload.get("quotes"), list):
                    rows = payload["quotes"]

            if not rows:
                self.logger.warning("Empty batch-quote payload for chunk size=%s", len(chunk))
                continue

            frames.append(pd.DataFrame(rows))

        quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if quotes.empty:
            self.logger.warning("Stable batch-quote returned empty result set.")
            return pd.DataFrame(columns=["symbol", "price", "pe", "eps", "marketCap"])

        sym_col = "symbol" if "symbol" in quotes.columns else ("ticker" if "ticker" in quotes.columns else None)
        if sym_col is None:
            raise ValueError("batch-quote response missing symbol column")

        quotes[sym_col] = quotes[sym_col].astype(str).str.strip()
        quotes = quotes.drop_duplicates(subset=[sym_col], keep="last").rename(columns={sym_col: "symbol"})

        if "price" not in quotes.columns:
            for c in ("lastPrice", "close", "last", "c"):
                if c in quotes.columns:
                    quotes["price"] = quotes[c]
                    break

        if "marketCap" not in quotes.columns:
            for c in ("mktCap", "marketcap", "marketCapitalization"):
                if c in quotes.columns:
                    quotes["marketCap"] = quotes[c]
                    break

        if "pe" not in quotes.columns:
            for c in ("pe", "peRatio", "peRatioTTM", "priceEarningsRatio", "priceEarningsRatioTTM"):
                if c in quotes.columns:
                    quotes["pe"] = quotes[c]
                    break

        if "eps" not in quotes.columns:
            for c in ("eps", "epsTTM", "netIncomePerShareTTM", "netIncomePerShare"):
                if c in quotes.columns:
                    quotes["eps"] = quotes[c]
                    break

        for c in ("price", "marketCap", "pe", "eps"):
            if c in quotes.columns:
                quotes[c] = pd.to_numeric(quotes[c], errors="coerce")
            else:
                quotes[c] = float("nan")

        need = quotes.loc[(quotes["pe"].isna()) | (quotes["pe"] <= 0) | (quotes["eps"].isna()), "symbol"].tolist()
        need = [s for s in need if s]

        if need:
            # /stable/key-metrics-ttm returns empty for all symbols on this plan.
            # /stable/ratios returns priceToEarningsRatio and netIncomePerShare
            # from the most recent annual filing — reliable for individual holdings.
            # ETFs (XLK, XLF, etc.) will still return empty; that is expected
            # and handled downstream via holdings-weighted PE aggregation.
            ratios_url = self._url_stable("/ratios")

            def _fetch_one(sym: str) -> Dict[str, Any]:
                payload_km = self._request_json(ratios_url, params={"symbol": sym, "limit": "1"})

                if isinstance(payload_km, list) and payload_km:
                    row = payload_km[0]
                elif isinstance(payload_km, dict):
                    row = payload_km
                else:
                    row = {}

                row = dict(row) if isinstance(row, dict) else {}
                row["symbol"] = sym
                row["pe_km"] = _first_numeric(
                    row,
                    keys=("priceToEarningsRatio", "peRatio", "peRatioTTM", "priceEarningsRatio", "pe"),
                )
                row["eps_km"] = _first_numeric(
                    row,
                    keys=("netIncomePerShare", "epsDiluted", "eps", "netIncomePerShareTTM", "epsTTM"),
                )
                return row

            km_rows: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(int(self.settings.max_workers), 16)) as ex:
                futs = {ex.submit(_fetch_one, sym): sym for sym in need}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        km_rows.append(fut.result())
                    except Exception as e:
                        self.logger.warning("key-metrics-ttm failed for %s: %s", sym, type(e).__name__)

            if km_rows:
                km_df = pd.DataFrame(km_rows)[["symbol", "pe_km", "eps_km"]].copy()
                km_df["pe_km"] = pd.to_numeric(km_df["pe_km"], errors="coerce")
                km_df["eps_km"] = pd.to_numeric(km_df["eps_km"], errors="coerce")

                quotes = quotes.merge(km_df, on="symbol", how="left")
                quotes["pe"] = quotes["pe"].where(quotes["pe"].notna() & (quotes["pe"] > 0), quotes["pe_km"])
                quotes["eps"] = quotes["eps"].where(quotes["eps"].notna(), quotes["eps_km"])
                quotes = quotes.drop(columns=["pe_km", "eps_km"], errors="ignore")

        out = quotes[["symbol", "price", "pe", "eps", "marketCap"]].copy()
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out = out[out["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="last")
        out = out.sort_values("symbol").reset_index(drop=True)
        return out


    # =====================================================
    # Extended fundamentals fetch (Phase 4 — institutional multiples)
    # =====================================================
    def fetch_extended_fundamentals(
        self,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        """
        Fetch extended fundamental data for holdings-level FJS computation.

        Fetches per-symbol:
        - ratios-ttm:              P/B, div yield, FCF yield, ROE, ROIC, margins
        - key-metrics-ttm:         ROIC, earnings yield, revenue per share
        - enterprise-values:       EV, EBITDA, market cap (for ratio-of-sums EV/EBITDA)
        - financial-statement-growth: EPS/revenue/EBITDA growth (NTM proxy)
        - analyst-estimates:       Forward P/E, NTM EPS consensus
        - cashflow-statement:      OCF + capex (for Owner Earnings / XLRE)
        - earnings-surprises:      beat/miss history (last 4 quarters)

        Returns consolidated DataFrame, one row per symbol.
        Missing fields are NaN — never raises on partial data.
        """
        syms = _dedupe_preserve_order([
            str(t).strip() for t in tickers
            if str(t).strip() and not str(t).strip().startswith("^")
        ])
        if not syms:
            self.logger.warning("fetch_extended_fundamentals: no valid tickers — returning empty")
            return pd.DataFrame(columns=["symbol"])

        self.logger.info("Fetching extended fundamentals for %s symbols", len(syms))

        results: Dict[str, Dict[str, Any]] = {s: {"symbol": s} for s in syms}
        # Tracks endpoint paths blocked by plan restriction (PermissionError on probe)
        _blocked: set = set()

        def _merge(sym: str, data: Dict[str, Any]) -> None:
            if sym in results:
                results[sym].update(data)

        def _probe_endpoint(path: str, probe_sym: str, params: Dict[str, Any]) -> bool:
            """Return True if endpoint is accessible; False if plan-blocked. Logs once."""
            if path in _blocked:
                return False
            try:
                self._request_json(self._url_stable(path), params={**params, "symbol": probe_sym})
                return True
            except PermissionError as exc:
                _blocked.add(path)
                self.logger.warning(
                    "Endpoint /stable%s not available on current FMP plan — skipping for all symbols. (%s)",
                    path, str(exc)[:120],
                )
                return False
            except Exception:
                return True  # transient error, not a plan block — proceed

        probe_sym = syms[0]

        # ── Probe all endpoints before bulk fetch ─────────────────────────────
        endpoints_available = {
            "/ratios-ttm":               _probe_endpoint("/ratios-ttm", probe_sym, {}),
            "/enterprise-values":        _probe_endpoint("/enterprise-values", probe_sym, {"limit": 1}),
            "/key-metrics-ttm":          _probe_endpoint("/key-metrics-ttm", probe_sym, {}),
            "/income-statement":         _probe_endpoint("/income-statement", probe_sym, {"limit": 1}),
            "/financial-statement-growth": _probe_endpoint("/financial-statement-growth", probe_sym, {"limit": 1}),
            "/analyst-estimates":        _probe_endpoint("/analyst-estimates", probe_sym, {"period": "annual", "limit": 2}),
            "/cash-flow-statement":      _probe_endpoint("/cash-flow-statement", probe_sym, {"limit": 1}),
            "/earnings-surprises":       _probe_endpoint("/earnings-surprises", probe_sym, {"limit": 8}),
        }
        skipped = [p for p, ok in endpoints_available.items() if not ok]
        if skipped:
            self.logger.info("Skipping %d unavailable endpoint(s): %s", len(skipped), skipped)

        # ── Ratios TTM ──────────────────────────────────────
        def _fetch_ratios(sym: str) -> None:
            if not endpoints_available["/ratios-ttm"]:
                return
            try:
                url = self._url_stable("/ratios-ttm")
                payload = self._request_json(url, params={"symbol": sym})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "pb":                   _first_numeric(row, ("priceToBookRatioTTM", "priceToBookRatio", "pb")),
                    "dividendYield":        _first_numeric(row, ("dividendYieldTTM", "dividendYield")),
                    "freeCashFlowYield":    _first_numeric(row, ("freeCashFlowYieldTTM", "freeCashFlowYield")),
                    "returnOnEquity":       _first_numeric(row, ("returnOnEquityTTM", "roe", "returnOnEquity")),
                    "returnOnInvestedCapital": _first_numeric(row, ("returnOnInvestedCapitalTTM", "roic")),
                    "grossProfitMargin":    _first_numeric(row, ("grossProfitMarginTTM", "grossProfitMargin")),
                    "netProfitMargin":      _first_numeric(row, ("netProfitMarginTTM", "netProfitMargin")),
                    "priceToSalesRatioTTM": _first_numeric(row, ("priceToSalesRatioTTM", "priceToSalesRatio")),
                    "forwardPE":            _first_numeric(row, ("priceEarningsRatioTTM", "forwardPE", "pe")),
                })
            except Exception as exc:
                self.logger.debug("ratios-ttm failed for %s: %s", sym, exc)

        # ── Enterprise values ────────────────────────────────
        def _fetch_ev(sym: str) -> None:
            if not endpoints_available["/enterprise-values"]:
                return
            try:
                url = self._url_stable("/enterprise-values")
                payload = self._request_json(url, params={"symbol": sym, "limit": 1})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "enterpriseValue": _first_numeric(row, ("enterpriseValue", "ev")),
                    "ebitda":          _first_numeric(row, ("ebitda", "EBITDA")),
                    "sharesOutstanding": _first_numeric(row, ("numberOfShares", "sharesOutstanding")),
                })
            except Exception as exc:
                self.logger.debug("enterprise-values failed for %s: %s", sym, exc)

        # ── Financial statement growth ───────────────────────
        def _fetch_growth(sym: str) -> None:
            if not endpoints_available["/financial-statement-growth"]:
                return
            try:
                url = self._url_stable("/financial-statement-growth")
                payload = self._request_json(url, params={"symbol": sym, "limit": 1})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "epsGrowthNTM":     _first_numeric(row, ("epsgrowth", "epsGrowth", "eps_growth")),
                    "revenueGrowthNTM": _first_numeric(row, ("revenuegrowth", "revenueGrowth")),
                    "ebitdaGrowthNTM":  _first_numeric(row, ("ebitdagrowth", "ebitdaGrowth")),
                })
            except Exception as exc:
                self.logger.debug("financial-statement-growth failed for %s: %s", sym, exc)

        # ── Analyst estimates (forward P/E, NTM EPS) ────────
        def _fetch_estimates(sym: str) -> None:
            if not endpoints_available["/analyst-estimates"]:
                return
            try:
                url = self._url_stable("/analyst-estimates")
                payload = self._request_json(url, params={"symbol": sym, "period": "annual", "limit": 2})
                rows = payload if isinstance(payload, list) else []
                if rows:
                    r = rows[0]
                    eps_est = _first_numeric(r, ("estimatedEpsAvg", "epsAvg", "estimatedEps"))
                    if len(rows) > 1:
                        eps_prev = _first_numeric(rows[1], ("estimatedEpsAvg", "epsAvg"))
                        if math.isfinite(eps_est) and math.isfinite(eps_prev) and abs(eps_prev) > 0.01:
                            _merge(sym, {
                                "epsGrowthNTM": (eps_est - eps_prev) / abs(eps_prev),
                                "ntmEpsConsensus": eps_est,
                            })
                    rev_3m = _first_numeric(r, ("estimatedEpsAvg", "revision3m"))
                    if math.isfinite(rev_3m):
                        _merge(sym, {"epsRevision3m": rev_3m})
            except Exception as exc:
                self.logger.debug("analyst-estimates failed for %s: %s", sym, exc)

        # ── Key metrics TTM (ROIC, earnings yield) ──────────────
        def _fetch_key_metrics(sym: str) -> None:
            if not endpoints_available["/key-metrics-ttm"]:
                return
            try:
                url = self._url_stable("/key-metrics-ttm")
                payload = self._request_json(url, params={"symbol": sym})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "returnOnInvestedCapital": _first_numeric(
                        row, ("roicTTM", "returnOnInvestedCapitalTTM", "roic")),
                    "earningsYieldTTM":        _first_numeric(
                        row, ("earningsYieldTTM", "earningsYield")),
                    "freeCashFlowYieldTTM":    _first_numeric(
                        row, ("freeCashFlowYieldTTM", "fcfYieldTTM")),
                    "revenuePerShareTTM":      _first_numeric(
                        row, ("revenuePerShareTTM", "revenuePerShare")),
                })
            except Exception as exc:
                self.logger.debug("key-metrics-ttm failed for %s: %s", sym, exc)

        # ── Income statement (net income for accrual ratio) ───
        def _fetch_income(sym: str) -> None:
            if not endpoints_available["/income-statement"]:
                return
            try:
                url = self._url_stable("/income-statement")
                payload = self._request_json(url, params={"symbol": sym, "limit": 1})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "netIncomeIS":   _first_numeric(row, ("netIncome", "netIncomeApplicableToCommonShares")),
                    "grossProfit":   _first_numeric(row, ("grossProfit",)),
                    "ebitdaIS":      _first_numeric(row, ("ebitda", "EBITDA")),
                    "totalRevenue":  _first_numeric(row, ("revenue", "totalRevenue")),
                })
            except Exception as exc:
                self.logger.debug("income-statement failed for %s: %s", sym, exc)

        # ── Cashflow statement (OCF + capex for Owner Earnings) ─
        def _fetch_cashflow(sym: str) -> None:
            if not endpoints_available["/cash-flow-statement"]:
                return
            try:
                url = self._url_stable("/cash-flow-statement")
                payload = self._request_json(url, params={"symbol": sym, "limit": 1})
                row = payload[0] if isinstance(payload, list) and payload else (
                      payload if isinstance(payload, dict) else {})
                _merge(sym, {
                    "operatingCashFlow":   _first_numeric(row, ("operatingCashFlow", "cashFlowFromOperations")),
                    "capitalExpenditures": _first_numeric(row, ("capitalExpenditure", "capitalExpenditures", "capex")),
                    "netIncome":           _first_numeric(row, ("netIncome",)),
                })
            except Exception as exc:
                self.logger.debug("cash-flow-statement failed for %s: %s", sym, exc)

        # ── Earnings surprises (beat/miss history) ───────────
        def _fetch_surprises(sym: str) -> None:
            if not endpoints_available["/earnings-surprises"]:
                return
            try:
                url = self._url_stable("/earnings-surprises")
                payload = self._request_json(url, params={"symbol": sym, "limit": 8})
                rows = payload if isinstance(payload, list) else []
                if rows:
                    beats = sum(
                        1 for r in rows[:4]
                        if _first_numeric(r, ("actualEarningResult", "actual")) >
                           _first_numeric(r, ("estimatedEarning", "estimated"))
                    )
                    _merge(sym, {"beat_rate_4q": beats / min(4, len(rows[:4]))})
            except Exception as exc:
                self.logger.debug("earnings-surprises failed for %s: %s", sym, exc)

        # ── Parallel fetch (only available endpoints) ─────────────────────────
        active_fetchers = [fn for fn, path in [
            (_fetch_ratios,      "/ratios-ttm"),
            (_fetch_ev,          "/enterprise-values"),
            (_fetch_key_metrics, "/key-metrics-ttm"),
            (_fetch_income,      "/income-statement"),
            (_fetch_growth,      "/financial-statement-growth"),
            (_fetch_estimates,   "/analyst-estimates"),
            (_fetch_cashflow,    "/cash-flow-statement"),
            (_fetch_surprises,   "/earnings-surprises"),
        ] if endpoints_available[path]]

        if not active_fetchers:
            self.logger.warning("All extended fundamentals endpoints unavailable — returning symbol stubs.")
        else:
            with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, 8)) as ex:
                futs = []
                for sym in syms:
                    for fn in active_fetchers:
                        futs.append(ex.submit(fn, sym))
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as exc:
                        self.logger.debug("Extended fetch error: %s", exc)

        df = pd.DataFrame(list(results.values()))
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        self.logger.info("Extended fundamentals: %d symbols, %d columns", len(df), df.shape[1])
        return df

    # =====================================================
    # Incremental price helpers
    # =====================================================
    _INCREMENTAL_THRESHOLD_DAYS: int = 5  # If prices are < 5 trading days stale, do incremental

    def _prices_last_date(self) -> Optional[date]:
        """Return latest date in the prices parquet, or None if it doesn't exist."""
        if not self.artifacts.prices_path.exists():
            return None
        try:
            idx = pd.read_parquet(
                self.artifacts.prices_path,
                engine="pyarrow",
                columns=[],
            ).index
            if idx.empty:
                return None
            last = pd.Timestamp(idx.max())
            return last.date()
        except Exception as e:
            self.logger.debug("_prices_last_date: could not read parquet index — %s", e)
            return None

    def _merge_incremental_prices(self, existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
        """Merge new rows into existing prices DataFrame, handling column superset."""
        if existing.empty:
            return new_rows
        if new_rows.empty:
            return existing
        combined = pd.concat([existing, new_rows], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.ffill(limit=5)
        return combined

    # =====================================================
    # Snapshot orchestration
    # =====================================================
    def build_snapshot(self, force_refresh: bool = False) -> ParquetArtifacts:
        if not force_refresh and self.is_snapshot_fresh():
            self.logger.info("Using fresh cached snapshot under %s", self.settings.parquet_dir)
            return self.artifacts

        _db_writer = _get_db_writer(self.settings.db_path)
        today = date.today()

        # ── Decide: incremental or full fetch ─────────────────────────────────
        last_price_date = self._prices_last_date()
        days_stale = (today - last_price_date).days if last_price_date else 9999
        do_incremental = (
            not force_refresh
            and last_price_date is not None
            and days_stale <= self._INCREMENTAL_THRESHOLD_DAYS
        )

        sector_and_spy = self.settings.sector_list() + [self.settings.spy_ticker]
        macro = list(self.settings.macro_tickers.values()) + list(self.settings.vol_tickers.values())
        credit = list(self.settings.credit_tickers.values())
        all_tickers = sector_and_spy + macro + credit

        if do_incremental:
            # ── INCREMENTAL: fetch only missing days ─────────────────────────
            incremental_start = last_price_date + timedelta(days=1)
            self.logger.info(
                "Incremental price refresh: %d tickers from %s (last=%s, stale=%dd)",
                len(all_tickers), incremental_start, last_price_date, days_stale,
            )
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {
                    "equities": ex.submit(self._fetch_prices_block, sector_and_spy,
                                          incremental_start, today),
                    "macro":    ex.submit(self._fetch_prices_block, macro,
                                          incremental_start, today),
                    "credit":   ex.submit(self._fetch_prices_block, credit,
                                          incremental_start, today),
                }
                new_blocks: Dict[str, pd.DataFrame] = {k: fut.result() for k, fut in futs.items()}

            new_rows = pd.concat(
                [new_blocks.get(k, pd.DataFrame()) for k in ("equities", "macro", "credit")],
                axis=1,
            )

            if new_rows.empty:
                self.logger.info(
                    "Incremental fetch returned no new rows (market may be closed). "
                    "Using existing parquet as-is."
                )
                # Touch parquet to refresh cache timestamp
                self.artifacts.prices_path.touch()
                prices = pd.read_parquet(self.artifacts.prices_path, engine="pyarrow")
            else:
                existing = pd.read_parquet(self.artifacts.prices_path, engine="pyarrow")
                prices = self._merge_incremental_prices(existing, new_rows)
                self.logger.info(
                    "Incremental merge: %d existing + %d new rows = %d total (%d cols)",
                    len(existing), len(new_rows), len(prices), prices.shape[1],
                )
        else:
            # ── FULL FETCH ───────────────────────────────────────────────────
            start_date = today - timedelta(days=int(self.settings.history_years * 365.25))
            self.logger.info(
                "Full price fetch: %d tickers from %s (stale=%dd)",
                len(all_tickers), start_date, days_stale,
            )
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {
                    "equities": ex.submit(self._fetch_prices_block, sector_and_spy, start_date, None),
                    "macro":    ex.submit(self._fetch_prices_block, macro, start_date, None),
                    "credit":   ex.submit(self._fetch_prices_block, credit, start_date, None),
                }
                blocks: Dict[str, pd.DataFrame] = {k: fut.result() for k, fut in futs.items()}

            prices = pd.concat(
                [blocks.get(k, pd.DataFrame()) for k in ("equities", "macro", "credit")],
                axis=1,
            )

        if prices.empty:
            raise RuntimeError("Price snapshot is empty; cannot proceed.")

        prices = prices[~prices.index.duplicated(keep="last")].sort_index()
        prices = prices.ffill(limit=5)

        self.artifacts.prices_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(self.artifacts.prices_path, engine="pyarrow", compression="snappy")
        self.logger.info(
            "Saved prices to %s (rows=%s cols=%s) [%s]",
            self.artifacts.prices_path, len(prices), prices.shape[1],
            "incremental" if do_incremental else "full",
        )
        if _db_writer is not None:
            try:
                # DuckDB write is always safe (INSERT OR REPLACE) —
                # on incremental run, only new rows land; on full run everything upserts.
                _db_writer.write_prices(prices)
            except Exception as _e:
                self.logger.warning("DB write_prices failed (non-fatal): %s", _e)

        weight_frames: List[pd.DataFrame] = []
        etfs_for_holdings = _dedupe_preserve_order(self.settings.sector_list() + [self.settings.spy_ticker])

        self.logger.info("Fetching ETF holdings for %s ETFs", len(etfs_for_holdings))
        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, max(2, len(etfs_for_holdings)))) as ex:
            fut_map = {ex.submit(self.fetch_etf_holdings, sym): sym for sym in etfs_for_holdings}
            for fut in as_completed(fut_map):
                sym = fut_map[fut]
                try:
                    df_h = fut.result()
                    if not df_h.empty:
                        weight_frames.append(df_h)
                except Exception as e:
                    self.logger.exception("Failed to fetch holdings for %s: %s", sym, repr(e))

        try:
            spy_sector_weights = self.fetch_spy_sector_weightings()
            if not spy_sector_weights.empty:
                weight_frames.append(spy_sector_weights)
        except Exception as e:
            self.logger.warning(
                "SPY sector weights unavailable; analytics will fallback to equal sector weights. %s",
                repr(e),
            )

        weights_df = pd.concat(weight_frames, axis=0, ignore_index=True) if weight_frames else pd.DataFrame()

        if weights_df.empty:
            self.logger.warning("Weights snapshot is empty; fundamentals aggregation may degrade.")
        else:
            required_cols = [
                "record_type",
                "etfSymbol",
                "sector",
                "weightPercentage",
                "asset",
                "sharesNumber",
                "marketValue",
                "updated",
            ]
            for col in required_cols:
                if col not in weights_df.columns:
                    weights_df[col] = None

            weights_df = weights_df[required_cols].copy()
            weights_df.to_parquet(self.artifacts.weights_path, engine="pyarrow", compression="snappy")
            self.logger.info("Saved weights to %s (rows=%s)", self.artifacts.weights_path, len(weights_df))
            if _db_writer is not None:
                try:
                    _db_writer.write_weights(weights_df)
                except Exception as _e:
                    self.logger.warning("DB write_weights failed (non-fatal): %s", _e)

        quote_symbols: List[str] = list(_dedupe_preserve_order(sector_and_spy))

        if not weights_df.empty and "record_type" in weights_df.columns:
            holding_assets = (
                weights_df.loc[weights_df["record_type"] == "holding", "asset"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

            cleaned_assets: List[str] = []
            for sym in holding_assets:
                if not sym:
                    continue
                if sym.startswith("^"):
                    continue
                if " " in sym:
                    continue
                if sym.upper() in {"USD", "CASH", "N/A", "NA", "NULL"}:
                    continue
                cleaned_assets.append(sym)

            quote_symbols.extend(cleaned_assets)

        quote_symbols = _dedupe_preserve_order(quote_symbols)

        self.logger.info("Fetching fundamentals/quotes for %s unique symbols", len(quote_symbols))
        fundamentals = self.fetch_quotes(quote_symbols)

        if fundamentals.empty:
            self.logger.warning("Fundamentals snapshot is empty.")
        else:
            fundamentals["symbol"] = fundamentals["symbol"].astype(str).str.strip()
            fundamentals = (
                fundamentals.drop_duplicates(subset=["symbol"], keep="last")
                .sort_values("symbol")
                .reset_index(drop=True)
            )

        fundamentals.to_parquet(self.artifacts.fundamentals_path, engine="pyarrow", compression="snappy")
        self.logger.info(
            "Saved fundamentals to %s (rows=%s cols=%s)",
            self.artifacts.fundamentals_path,
            len(fundamentals),
            fundamentals.shape[1],
        )
        if _db_writer is not None:
            try:
                _db_writer.write_fundamentals(fundamentals)
            except Exception as _e:
                self.logger.warning("DB write_fundamentals failed (non-fatal): %s", _e)

        # Extended fundamentals (institutional multiples for FJS engine)
        self.logger.info("Fetching extended fundamentals for institutional FJS engine...")
        try:
            ext_fundamentals = self.fetch_extended_fundamentals(quote_symbols)
            ext_path = self.artifacts.extended_fundamentals_path
            ext_path.parent.mkdir(parents=True, exist_ok=True)
            ext_fundamentals.to_parquet(ext_path, engine="pyarrow", compression="snappy")
            self.logger.info(
                "Saved extended_fundamentals to %s (rows=%s cols=%s)",
                ext_path, len(ext_fundamentals), ext_fundamentals.shape[1],
            )
            if _db_writer is not None:
                try:
                    _db_writer.write_extended_fundamentals(ext_fundamentals)
                except Exception as _e:
                    self.logger.warning("DB write_extended_fundamentals failed (non-fatal): %s", _e)
        except Exception as exc:
            self.logger.warning(
                "Extended fundamentals fetch failed (non-fatal): %s — FJS engine will use basic PE fallback",
                exc,
            )
            # Write a minimal placeholder so is_snapshot_fresh() doesn't force re-fetch
            placeholder = pd.DataFrame(columns=["symbol"])
            try:
                self.artifacts.extended_fundamentals_path.parent.mkdir(parents=True, exist_ok=True)
                placeholder.to_parquet(self.artifacts.extended_fundamentals_path, engine="pyarrow", compression="snappy")
            except Exception:
                pass

        # ── Post-fetch validation ─────────────────────────────────────────────
        self._validate_snapshot(prices, weights_df)

        return self.artifacts

    # =====================================================
    # Post-fetch data validation
    # =====================================================
    def _validate_snapshot(self, prices: pd.DataFrame, weights_df: pd.DataFrame) -> None:
        """
        Run lightweight data-quality checks after every ingestion.
        Logs WARNING for each issue found; never raises (non-blocking).

        Checks:
          1. Date continuity — gaps > 5 business days flag stale feed
          2. Price-spike detection — single-day moves > 50% flag bad data
          3. Holdings coverage — < 80% of sector ETFs with holdings flags universe gap
        """
        # ── 1. Date continuity ────────────────────────────────────────────────
        if not prices.empty and len(prices.index) > 1:
            bdays = pd.bdate_range(prices.index.min(), prices.index.max())
            actual_dates = set(prices.index.normalize())
            # Report contiguous gaps only (suppress isolated holidays)
            gap_runs: List[int] = []
            run = 0
            for d in bdays:
                if d in actual_dates:
                    if run > 5:
                        gap_runs.append(run)
                    run = 0
                else:
                    run += 1
            if run > 5:
                gap_runs.append(run)
            if gap_runs:
                self.logger.warning(
                    "DATA QUALITY: Price date gaps detected — %s gap(s) > 5 business days "
                    "(longest: %s days). Feed may be stale.",
                    len(gap_runs),
                    max(gap_runs),
                )
            else:
                self.logger.info("DATA QUALITY: Date continuity OK (no gaps > 5 business days).")

        # ── 2. Price-spike detection ──────────────────────────────────────────
        SPIKE_THRESHOLD = 0.50  # 50% single-day move
        if not prices.empty:
            pct_chg = prices.pct_change().abs()
            spikes = (pct_chg > SPIKE_THRESHOLD).stack()
            spikes = spikes[spikes]
            if not spikes.empty:
                flagged = [(str(date), str(col)) for (date, col) in spikes.index.tolist()[:10]]
                self.logger.warning(
                    "DATA QUALITY: %s price spike(s) > 50%% detected (first 10): %s",
                    len(spikes),
                    flagged,
                )
            else:
                self.logger.info("DATA QUALITY: No price spikes > 50%% detected.")

        # ── 3. Holdings coverage ──────────────────────────────────────────────
        sector_etfs = set(self.settings.sector_list())
        if not weights_df.empty and "record_type" in weights_df.columns and "etfSymbol" in weights_df.columns:
            covered = set(
                weights_df.loc[weights_df["record_type"] == "holding", "etfSymbol"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
            coverage = len(covered & sector_etfs) / max(len(sector_etfs), 1)
            if coverage < 0.80:
                missing = sorted(sector_etfs - covered)
                self.logger.warning(
                    "DATA QUALITY: Holdings coverage %.0f%% < 80%%. Missing ETFs: %s",
                    coverage * 100,
                    missing,
                )
            else:
                self.logger.info(
                    "DATA QUALITY: Holdings coverage OK (%.0f%% of sector ETFs).", coverage * 100
                )
        else:
            self.logger.warning("DATA QUALITY: weights_df empty — cannot assess holdings coverage.")