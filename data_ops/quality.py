"""
data_ops/quality.py

Coverage and completeness diagnostics per ETF and across the price universe.

Produces a CoverageReport with a scalar health_score (0.0 – 1.0).
No side effects. All functions are pure read-only inspections.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds
_MIN_PRICE_COV   = 0.80     # fraction of rows non-null before a ticker is "adequate"
_MIN_HOLD_WEIGHT = 50.0     # % weight sum before an ETF's holdings are "adequate"
_MIN_PE_COV      = 0.50     # fraction of holding rows with valid PE before ETF is "adequate"


# =========================================================================
# Data containers
# =========================================================================

@dataclass
class TickerPriceCoverage:
    ticker: str
    present: bool           # column exists in prices.parquet
    total_rows: int
    non_null_rows: int
    coverage_ratio: float   # non_null / total
    adequate: bool          # coverage_ratio >= _MIN_PRICE_COV
    warning: Optional[str]


@dataclass
class ETFFundamentalsCoverage:
    etf_ticker: str
    holdings_count: int
    pe_coverage_count: int
    pe_coverage_ratio: float    # holdings with valid PE / total holdings
    weight_covered_pct: float   # % AUM weight of holdings that have PE
    etf_level_fallback: bool    # True when no holdings rows available
    adequate: bool
    warning: Optional[str]


@dataclass
class HoldingsCoverage:
    etf_ticker: str
    holdings_row_count: int
    weight_sum_pct: float
    weight_sum_adequate: bool   # weight_sum_pct >= _MIN_HOLD_WEIGHT
    warning: Optional[str]


@dataclass(frozen=True)
class SPYSectorWeightCoverage:
    sector_count: int
    sectors_present: List[str]
    sectors_missing: List[str]
    adequate: bool
    warning: Optional[str]


@dataclass
class CoverageReport:
    """
    Aggregated coverage diagnostics across all three parquet artifacts.

    health_score (0.0 – 1.0) is a composite metric suitable for inclusion
    in the overall DataHealthReport score.
    """
    price_coverage: Dict[str, TickerPriceCoverage]
    fundamentals_coverage: Dict[str, ETFFundamentalsCoverage]
    holdings_coverage: Dict[str, HoldingsCoverage]
    spy_sector_weight: SPYSectorWeightCoverage

    missing_price_tickers: List[str]    # tickers absent from prices.parquet
    sparse_price_tickers: List[str]     # present but below _MIN_PRICE_COV
    fallback_etfs: List[str]            # ETFs using ETF-level PE fallback

    health_score: float                 # 0.0 – 1.0
    degraded: bool
    warnings: List[str] = field(default_factory=list)


# =========================================================================
# Helpers
# =========================================================================

def _r(n: float, d: float) -> float:
    """Safe ratio capped to [0, 1]."""
    if d <= 0 or not math.isfinite(d):
        return 0.0
    return max(0.0, min(1.0, n / d))


def _assess_price_coverage(
    df: pd.DataFrame,
    tickers: List[str],
) -> Dict[str, TickerPriceCoverage]:
    total = len(df)
    out: Dict[str, TickerPriceCoverage] = {}

    for t in tickers:
        if t not in df.columns:
            out[t] = TickerPriceCoverage(
                ticker=t, present=False, total_rows=total,
                non_null_rows=0, coverage_ratio=0.0, adequate=False,
                warning=f"{t} not found in prices.parquet",
            )
            continue

        nn    = int(df[t].notna().sum())
        ratio = _r(nn, total)
        ok    = ratio >= _MIN_PRICE_COV
        out[t] = TickerPriceCoverage(
            ticker=t, present=True, total_rows=total,
            non_null_rows=nn, coverage_ratio=ratio, adequate=ok,
            warning=(
                f"{t}: price coverage {ratio:.1%} ({nn}/{total} rows)"
                if not ok else None
            ),
        )

    return out


def _assess_holdings(
    weights_df: pd.DataFrame,
    etfs: List[str],
) -> Dict[str, HoldingsCoverage]:
    out: Dict[str, HoldingsCoverage] = {}

    if weights_df.empty or "record_type" not in weights_df.columns:
        for e in etfs:
            out[e] = HoldingsCoverage(e, 0, 0.0, False, "No holdings data in weights.parquet")
        return out

    hr = weights_df[weights_df["record_type"] == "holding"]

    for e in etfs:
        rows = hr[hr["etfSymbol"] == e] if "etfSymbol" in hr.columns else pd.DataFrame()

        if rows.empty or "weightPercentage" not in rows.columns:
            out[e] = HoldingsCoverage(e, 0, 0.0, False, f"{e}: no holding rows")
            continue

        w  = float(pd.to_numeric(rows["weightPercentage"], errors="coerce").sum())
        ok = w >= _MIN_HOLD_WEIGHT
        out[e] = HoldingsCoverage(
            etf_ticker=e,
            holdings_row_count=len(rows),
            weight_sum_pct=w,
            weight_sum_adequate=ok,
            warning=(
                f"{e}: holdings weight sum {w:.1f}% < {_MIN_HOLD_WEIGHT:.0f}%"
                if not ok else None
            ),
        )

    return out


def _assess_fundamentals(
    weights_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    etfs: List[str],
) -> Dict[str, ETFFundamentalsCoverage]:
    out: Dict[str, ETFFundamentalsCoverage] = {}

    # Build a lookup of valid PE by holding symbol
    pe_map: Dict[str, float] = {}
    if (not fund_df.empty
            and "symbol" in fund_df.columns
            and "pe" in fund_df.columns):
        raw = (
            fund_df.dropna(subset=["symbol"])
            .set_index("symbol")["pe"]
            .dropna()
            .to_dict()
        )
        pe_map = {k: v for k, v in raw.items()
                  if math.isfinite(float(v)) and float(v) > 0}

    ok_cols = all(c in weights_df.columns
                  for c in ["record_type", "etfSymbol", "asset", "weightPercentage"])

    for e in etfs:
        if weights_df.empty or not ok_cols:
            out[e] = ETFFundamentalsCoverage(
                e, 0, 0, 0.0, 0.0, etf_level_fallback=True, adequate=False,
                warning=f"{e}: ETF-level PE fallback (no holdings data)",
            )
            continue

        rows = weights_df[
            (weights_df["record_type"] == "holding") & (weights_df["etfSymbol"] == e)
        ].copy()

        if rows.empty:
            out[e] = ETFFundamentalsCoverage(
                e, 0, 0, 0.0, 0.0, etf_level_fallback=True, adequate=False,
                warning=f"{e}: no holding rows — ETF-level fallback",
            )
            continue

        rows["weightPercentage"] = pd.to_numeric(rows["weightPercentage"], errors="coerce")
        rows = rows.dropna(subset=["weightPercentage", "asset"])

        if rows.empty:
            out[e] = ETFFundamentalsCoverage(
                e, 0, 0, 0.0, 0.0, etf_level_fallback=True, adequate=False,
                warning=f"{e}: all-NaN weights — ETF-level fallback",
            )
            continue

        has_pe = rows["asset"].map(lambda s: s in pe_map)
        n      = len(rows)
        pe_n   = int(has_pe.sum())
        ratio  = _r(pe_n, n)

        tw     = float(rows["weightPercentage"].sum())
        cw_pct = (
            float(rows.loc[has_pe, "weightPercentage"].sum()) / tw * 100.0
            if tw > 0 else 0.0
        )

        ok = ratio >= _MIN_PE_COV
        out[e] = ETFFundamentalsCoverage(
            etf_ticker=e,
            holdings_count=n,
            pe_coverage_count=pe_n,
            pe_coverage_ratio=ratio,
            weight_covered_pct=cw_pct,
            etf_level_fallback=False,
            adequate=ok,
            warning=(
                f"{e}: PE coverage {ratio:.1%} ({cw_pct:.0f}% by weight)"
                if not ok else None
            ),
        )

    return out


def _assess_spy_weights(
    weights_df: pd.DataFrame,
    expected_sectors: List[str],
) -> SPYSectorWeightCoverage:
    if weights_df.empty or "record_type" not in weights_df.columns:
        return SPYSectorWeightCoverage(
            0, [], expected_sectors, False,
            "No sector_weight rows in weights.parquet",
        )

    sw = weights_df[weights_df["record_type"] == "sector_weight"]
    if sw.empty or "sector" not in sw.columns:
        return SPYSectorWeightCoverage(
            0, [], expected_sectors, False,
            "SPY sector_weight rows missing 'sector' column",
        )

    present = sw["sector"].dropna().astype(str).str.strip().unique().tolist()
    missing = [s for s in expected_sectors if s not in set(present)]

    return SPYSectorWeightCoverage(
        sector_count=len(present),
        sectors_present=present,
        sectors_missing=missing,
        adequate=not missing,
        warning=(
            f"SPY sector weights missing: {', '.join(missing)}"
            if missing else None
        ),
    )


def _health_score(
    pc: Dict[str, TickerPriceCoverage],
    hc: Dict[str, HoldingsCoverage],
    fc: Dict[str, ETFFundamentalsCoverage],
    sw: SPYSectorWeightCoverage,
) -> float:
    """
    Composite coverage score: 0.0 – 1.0.

    Weights:
      40% prices  — average non-null coverage ratio across all tickers
      25% holdings — average weight_sum / 100% across ETFs
      25% fundamentals — average PE coverage ratio (fallback ETFs score 0.2)
      10% SPY sector weights — binary
    """
    s = 0.0

    if pc:
        s += 0.40 * sum(r.coverage_ratio if r.present else 0.0
                        for r in pc.values()) / len(pc)

    if hc:
        s += 0.25 * min(1.0,
                        sum(min(1.0, r.weight_sum_pct / 100.0)
                            for r in hc.values()) / len(hc))

    if fc:
        s += 0.25 * sum(
            (r.pe_coverage_ratio if not r.etf_level_fallback else 0.2)
            for r in fc.values()
        ) / len(fc)

    s += 0.10 * (1.0 if sw.adequate else 0.0)

    return round(max(0.0, min(1.0, s)), 4)


# =========================================================================
# Main entry point
# =========================================================================

def assess_coverage(
    prices_path: Path,
    fundamentals_path: Path,
    weights_path: Path,
    sector_tickers: List[str],
    spy_ticker: str,
    expected_sectors: List[str],
    all_price_tickers: List[str],
) -> CoverageReport:
    """
    Audit coverage across all three parquet artifacts.

    Args:
        prices_path, fundamentals_path, weights_path: artifact paths
        sector_tickers: list of sector ETF tickers (e.g. ["XLC", "XLY", ...])
        spy_ticker: "SPY"
        expected_sectors: canonical sector names for SPY weight check
        all_price_tickers: all tickers expected in prices.parquet

    Returns:
        CoverageReport with health_score (0–1) and per-dimension diagnostics.
    """
    warnings: List[str] = []

    def _load(path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            warnings.append(f"{name}.parquet not found at {path}")
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            warnings.append(f"Failed to read {name}.parquet: {exc}")
            return pd.DataFrame()

    prices_df = _load(prices_path, "prices")
    fund_df   = _load(fundamentals_path, "fundamentals")
    weights_df = _load(weights_path, "weights")

    pc = _assess_price_coverage(prices_df, all_price_tickers)
    hc = _assess_holdings(weights_df, sector_tickers + [spy_ticker])
    fc = _assess_fundamentals(weights_df, fund_df, sector_tickers)
    sw = _assess_spy_weights(weights_df, expected_sectors)

    # Collect per-item warnings
    for d in (pc, hc, fc):
        for r in d.values():
            if r.warning:
                warnings.append(r.warning)
    if sw.warning:
        warnings.append(sw.warning)

    missing = [t for t, r in pc.items() if not r.present]
    sparse  = [t for t, r in pc.items() if r.present and not r.adequate]
    fallbk  = [e for e, r in fc.items() if r.etf_level_fallback]
    score   = _health_score(pc, hc, fc, sw)

    return CoverageReport(
        price_coverage=pc,
        fundamentals_coverage=fc,
        holdings_coverage=hc,
        spy_sector_weight=sw,
        missing_price_tickers=missing,
        sparse_price_tickers=sparse,
        fallback_etfs=fallbk,
        health_score=score,
        degraded=(score < 0.70 or len(missing) > 0),
        warnings=warnings,
    )
