"""
data_ops/freshness.py

Snapshot freshness diagnostics — age and state (FRESH / STALE / MISSING)
of each Parquet artifact, plus last available price date check.

No side effects. All functions are pure read-only inspections.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRESH   = "FRESH"
STALE   = "STALE"
MISSING = "MISSING"
UNKNOWN = "UNKNOWN"

# Calendar days gap between today and last price date before we raise a warning
_MAX_PRICE_GAP_DAYS = 5


# =========================================================================
# Data containers
# =========================================================================

@dataclass(frozen=True)
class ArtifactFreshness:
    """Freshness status for a single parquet artifact."""
    name: str
    path: Path
    exists: bool
    age_seconds: Optional[float]
    age_hours: Optional[float]
    mtime: Optional[datetime]
    state: str              # FRESH | STALE | MISSING | UNKNOWN
    warning: Optional[str]  # Human-readable reason if not FRESH, else None


@dataclass(frozen=True)
class PriceFreshness:
    """Recency of the most recent price date inside prices.parquet."""
    last_price_date: Optional[date]
    days_since_last_price: Optional[int]
    expected_trading_date: Optional[date]
    price_date_gap_ok: bool
    warning: Optional[str]


@dataclass
class FreshnessSummary:
    """
    Freshness audit across all three parquet artifacts.

    Consumers should check:
      - overall_state: FRESH | STALE | MISSING | UNKNOWN
      - degraded: True when any artifact is stale or missing
      - price_detail.last_price_date: most recent date in prices.parquet
    """
    as_of: datetime
    artifacts: Dict[str, ArtifactFreshness]     # keys: "prices", "fundamentals", "weights"
    price_detail: Optional[PriceFreshness]
    overall_state: str
    warnings: List[str] = field(default_factory=list)
    degraded: bool = False


# =========================================================================
# Helpers
# =========================================================================

def _last_expected_trading_date(today: Optional[date] = None) -> date:
    """Return the most recent expected trading day (Mon–Fri) on or before today."""
    d = today or date.today()
    if d.weekday() == 0:    # Monday
        return d - timedelta(days=3)
    if d.weekday() == 6:    # Sunday
        return d - timedelta(days=2)
    if d.weekday() == 5:    # Saturday
        return d - timedelta(days=1)
    return d


def _check_artifact(name: str, path: Path, max_age_hours: float) -> ArtifactFreshness:
    """Inspect a single parquet artifact on disk."""
    if not path.exists():
        return ArtifactFreshness(
            name=name, path=path, exists=False,
            age_seconds=None, age_hours=None, mtime=None,
            state=MISSING,
            warning=f"{name}.parquet not found at {path}",
        )

    try:
        stat  = path.stat()
        age_s = time.time() - stat.st_mtime
        age_h = age_s / 3600.0
        mtime = datetime.fromtimestamp(stat.st_mtime)
    except OSError as exc:
        return ArtifactFreshness(
            name=name, path=path, exists=True,
            age_seconds=None, age_hours=None, mtime=None,
            state=UNKNOWN,
            warning=f"Could not stat {name}.parquet: {exc}",
        )

    # max_age_hours == 0 means freshness checking is disabled
    if max_age_hours > 0 and age_h > max_age_hours:
        return ArtifactFreshness(
            name=name, path=path, exists=True,
            age_seconds=age_s, age_hours=age_h, mtime=mtime,
            state=STALE,
            warning=(
                f"{name}.parquet is {age_h:.1f}h old "
                f"(threshold: {max_age_hours:.1f}h). "
                f"Last modified: {mtime:%Y-%m-%d %H:%M}."
            ),
        )

    return ArtifactFreshness(
        name=name, path=path, exists=True,
        age_seconds=age_s, age_hours=age_h, mtime=mtime,
        state=FRESH, warning=None,
    )


def _check_price_freshness(prices_path: Path) -> PriceFreshness:
    """
    Read the last date index from prices.parquet and check for recency.

    Uses columns=[] to load only the index — avoids pulling all price data.
    """
    today    = date.today()
    expected = _last_expected_trading_date(today)

    if not prices_path.exists():
        return PriceFreshness(
            last_price_date=None, days_since_last_price=None,
            expected_trading_date=expected, price_date_gap_ok=False,
            warning="prices.parquet does not exist.",
        )

    try:
        # columns=[] reads only the index — fast path, no column data loaded.
        # Note: df.empty is True when rows*cols == 0, so check len(index) directly.
        df = pd.read_parquet(prices_path, columns=[])
        if len(df.index) == 0:
            return PriceFreshness(
                last_price_date=None, days_since_last_price=None,
                expected_trading_date=expected, price_date_gap_ok=False,
                warning="prices.parquet has no rows.",
            )
        last_dt = pd.Timestamp(df.index[-1]).date()
    except Exception as exc:
        return PriceFreshness(
            last_price_date=None, days_since_last_price=None,
            expected_trading_date=expected, price_date_gap_ok=False,
            warning=f"Failed to read last price date: {exc}",
        )

    gap    = (today - last_dt).days
    gap_ok = gap <= _MAX_PRICE_GAP_DAYS

    return PriceFreshness(
        last_price_date=last_dt,
        days_since_last_price=gap,
        expected_trading_date=expected,
        price_date_gap_ok=gap_ok,
        warning=(
            f"Last price date is {last_dt} ({gap}d ago). "
            f"Expected: {expected}."
            if not gap_ok else None
        ),
    )


# =========================================================================
# Main entry point
# =========================================================================

def assess_freshness(
    prices_path: Path,
    fundamentals_path: Path,
    weights_path: Path,
    max_age_hours: float = 12.0,
) -> FreshnessSummary:
    """
    Audit the freshness of all three parquet artifacts.

    Args:
        prices_path, fundamentals_path, weights_path: artifact file paths
        max_age_hours: staleness threshold. Pass 0 to disable.

    Returns:
        FreshnessSummary with per-artifact status and last price date detail.
    """
    as_of = datetime.now()

    artifacts: Dict[str, ArtifactFreshness] = {
        "prices":       _check_artifact("prices",       prices_path,       max_age_hours),
        "fundamentals": _check_artifact("fundamentals", fundamentals_path, max_age_hours),
        "weights":      _check_artifact("weights",      weights_path,      max_age_hours),
    }

    price_detail = _check_price_freshness(prices_path)

    warnings: List[str] = [a.warning for a in artifacts.values() if a.warning]
    if price_detail and price_detail.warning:
        warnings.append(price_detail.warning)

    states  = {a.state for a in artifacts.values()}
    price_lag = price_detail and not price_detail.price_date_gap_ok

    if MISSING in states or UNKNOWN in states:
        overall = MISSING
    elif STALE in states or price_lag:
        overall = STALE
    else:
        overall = FRESH

    return FreshnessSummary(
        as_of=as_of,
        artifacts=artifacts,
        price_detail=price_detail,
        overall_state=overall,
        warnings=warnings,
        degraded=(overall in (STALE, MISSING, UNKNOWN)),
    )
