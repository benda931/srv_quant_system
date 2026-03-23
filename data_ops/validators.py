"""
data_ops/validators.py

Structural integrity checks on pre-loaded DataFrames.

Produces a ValidationReport with ERROR / WARNING / INFO level issues,
each with a machine-readable code and a human-readable remediation hint.

ERROR   — system cannot produce reliable output
WARNING — output may be degraded
INFO    — noteworthy but not immediately actionable
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

ERROR   = "ERROR"
WARNING = "WARNING"
INFO    = "INFO"

_MIN_ROWS       = 252       # minimum rows for analytics to work
_MAX_NAN_FRAC   = 0.10      # fraction of all-NaN columns before escalating
_MAX_WEIGHT_PCT = 110.0     # ETF weight sum above this suggests double-counting


# =========================================================================
# Data containers
# =========================================================================

@dataclass(frozen=True)
class ValidationIssue:
    severity: str       # ERROR | WARNING | INFO
    source: str         # "prices" | "fundamentals" | "weights" | "orchestrator"
    code: str           # short snake_case identifier, e.g. "PRICES_EMPTY"
    message: str        # what is wrong
    remediation: str    # how to fix it


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    passed: bool = True     # False as soon as any ERROR is added

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ERROR:
            self.error_count += 1
            self.passed = False
        elif issue.severity == WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]


# =========================================================================
# Per-artifact check functions
# =========================================================================

def _check_prices(
    df: pd.DataFrame,
    sector_tickers: List[str],
) -> List[ValidationIssue]:
    out: List[ValidationIssue] = []

    if df.empty:
        return [ValidationIssue(
            ERROR, "prices", "PRICES_EMPTY",
            "prices.parquet is empty.",
            "Re-run DataLakeManager.build_snapshot(force_refresh=True).",
        )]

    if len(df) < _MIN_ROWS:
        out.append(ValidationIssue(
            ERROR, "prices", "PRICES_INSUFFICIENT_HISTORY",
            f"prices.parquet has {len(df)} rows (minimum required: {_MIN_ROWS}).",
            "Increase Settings.history_years or check ingestion date range.",
        ))

    missing_sectors = [t for t in sector_tickers if t not in df.columns]
    if missing_sectors:
        out.append(ValidationIssue(
            ERROR, "prices", "SECTOR_TICKERS_MISSING",
            f"Sector ETFs missing from prices.parquet: {', '.join(missing_sectors)}.",
            "Check fetch_price_history() for these tickers. Verify FMP subscription.",
        ))

    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        frac = len(all_nan_cols) / max(1, len(df.columns))
        sev  = ERROR if frac > _MAX_NAN_FRAC else WARNING
        sample = all_nan_cols[:6]
        tail   = "..." if len(all_nan_cols) > 6 else ""
        out.append(ValidationIssue(
            sev, "prices", "ALL_NAN_PRICE_COLUMNS",
            f"{len(all_nan_cols)} columns are entirely NaN: "
            f"{', '.join(sample)}{tail}.",
            "Check fetch_price_history() for these tickers.",
        ))

    # Flat / stale series check: annualised std < 0.5% is suspicious
    for t in sector_tickers:
        if t not in df.columns:
            continue
        s = df[t].dropna()
        if len(s) < 20:
            continue
        ann_std = float(s.pct_change().dropna().std()) * math.sqrt(252)
        if math.isfinite(ann_std) and ann_std < 0.005:
            out.append(ValidationIssue(
                WARNING, "prices", "FLAT_PRICE_SERIES",
                f"{t}: annualised return std is {ann_std:.4f} — series may be flat or stale.",
                f"Inspect {t} price history in prices.parquet.",
            ))

    return out


def _check_fundamentals(
    df: pd.DataFrame,
    etf_tickers: List[str],
) -> List[ValidationIssue]:
    if df.empty:
        return [ValidationIssue(
            WARNING, "fundamentals", "FUNDAMENTALS_EMPTY",
            "fundamentals.parquet is empty — all FJS scores will use ETF-level fallback.",
            "Re-run build_snapshot(force_refresh=True).",
        )]

    required_cols = ["symbol", "pe", "eps", "marketCap"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return [ValidationIssue(
            ERROR, "fundamentals", "FUNDAMENTALS_MISSING_COLUMNS",
            f"fundamentals.parquet is missing columns: {', '.join(missing_cols)}.",
            "Check fetch_quotes() schema mapping.",
        )]

    out: List[ValidationIssue] = []

    pe_valid = pd.to_numeric(df["pe"], errors="coerce")
    pe_valid = pe_valid[(pe_valid > 0.01) & (pe_valid < 500)]
    if len(pe_valid) == 0:
        out.append(ValidationIssue(
            ERROR, "fundamentals", "ALL_PE_INVALID",
            "No valid PE ratios in fundamentals.parquet — FJS will be zero for all sectors.",
            "Check key-metrics-ttm endpoint and batch-quote PE field mappings in pipeline.py.",
        ))

    for etf in etf_tickers:
        if df[df["symbol"] == etf].empty:
            out.append(ValidationIssue(
                WARNING, "fundamentals", "ETF_MISSING_FROM_FUNDAMENTALS",
                f"{etf} not found in fundamentals.parquet — no ETF-level fallback PE available.",
                f"Ensure {etf} is included in quote_symbols during build_snapshot().",
            ))

    return out


def _check_weights(
    df: pd.DataFrame,
    sector_tickers: List[str],
) -> List[ValidationIssue]:
    if df.empty:
        return [ValidationIssue(
            WARNING, "weights", "WEIGHTS_EMPTY",
            "weights.parquet is empty.",
            "Re-run build_snapshot(force_refresh=True).",
        )]

    if "record_type" not in df.columns:
        return [ValidationIssue(
            ERROR, "weights", "WEIGHTS_MISSING_RECORD_TYPE",
            "weights.parquet is missing the 'record_type' column.",
            "Check fetch_etf_holdings() output schema.",
        )]

    out: List[ValidationIssue] = []
    hr = df[df["record_type"] == "holding"]

    if hr.empty:
        out.append(ValidationIssue(
            WARNING, "weights", "NO_HOLDING_ROWS",
            "No 'holding' rows found in weights.parquet.",
            "Check fetch_etf_holdings() results in pipeline.py.",
        ))

    if (not hr.empty
            and "etfSymbol" in hr.columns
            and "weightPercentage" in hr.columns):
        for etf in sector_tickers:
            rows = hr[hr["etfSymbol"] == etf]
            if rows.empty:
                continue
            w = float(pd.to_numeric(rows["weightPercentage"], errors="coerce").sum())
            if w > _MAX_WEIGHT_PCT:
                out.append(ValidationIssue(
                    WARNING, "weights", "WEIGHT_SUM_EXCEEDS_THRESHOLD",
                    f"{etf}: holding weight sum is {w:.1f}% "
                    f"(threshold: {_MAX_WEIGHT_PCT:.0f}%) — possible double-counting.",
                    f"Check weight normalisation in fetch_etf_holdings() for {etf}.",
                ))

    sw = df[df["record_type"] == "sector_weight"]
    if sw.empty:
        out.append(ValidationIssue(
            WARNING, "weights", "SPY_SECTOR_WEIGHTS_MISSING",
            "No 'sector_weight' rows in weights.parquet — "
            "dispersion analytics will use equal-weight fallback.",
            "Check fetch_spy_sector_weightings() in pipeline.py.",
        ))

    return out


# =========================================================================
# Main entry point
# =========================================================================

def validate_all(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    sector_tickers: List[str],
    spy_ticker: str,
) -> ValidationReport:
    """
    Run all structural integrity checks and return a ValidationReport.

    All errors are captured — no function raises. A single ValidationReport
    is returned regardless of how many issues are found.
    """
    report = ValidationReport()

    for issue in _check_prices(prices_df, sector_tickers):
        report.add(issue)

    etf_tickers = sector_tickers + [spy_ticker]
    for issue in _check_fundamentals(fundamentals_df, etf_tickers):
        report.add(issue)

    for issue in _check_weights(weights_df, sector_tickers):
        report.add(issue)

    return report
