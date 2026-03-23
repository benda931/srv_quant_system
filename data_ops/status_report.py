"""
data_ops/status_report.py

Aggregates freshness + coverage + validation into a single DataHealthReport.

Scoring (health_score: 0.0 – 1.0):
    Freshness degraded penalty:  -0.15
    Validation error penalty:    -0.10 per error (max -0.30)
    Validation warning penalty:  -0.03 per warning (max -0.12)
    Base score:                  coverage.health_score

Health labels:
    HEALTHY   >= 0.75
    DEGRADED  >= 0.50
    CRITICAL   < 0.50
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_ops.freshness  import FreshnessSummary, assess_freshness
from data_ops.quality    import CoverageReport, assess_coverage
from data_ops.validators import ValidationReport, validate_all

logger = logging.getLogger(__name__)

_HEALTHY_THRESH  = 0.75
_DEGRADED_THRESH = 0.50


# =========================================================================
# Report dataclass
# =========================================================================

@dataclass
class DataHealthReport:
    """
    Complete data health report for the SRV DSS.

    Use to_ui_dict() to get a flat dict suitable for Dash callbacks.
    Use to_coverage_table() to get a DataFrame for display.
    """
    as_of: datetime
    health_score: float         # 0.0 – 1.0
    health_label: str           # "HEALTHY" | "DEGRADED" | "CRITICAL"
    degraded: bool

    freshness: FreshnessSummary
    coverage: CoverageReport
    validation: ValidationReport

    warnings: List[str] = field(default_factory=list)
    errors:   List[str] = field(default_factory=list)

    def to_ui_dict(self) -> Dict[str, Any]:
        """
        Flat dict for UI/Dash consumption.

        health_score is output as a 0–100 percentage for display.
        """
        fq = self.freshness
        cq = self.coverage

        p  = fq.artifacts.get("prices")
        f  = fq.artifacts.get("fundamentals")
        w  = fq.artifacts.get("weights")
        pd_ = fq.price_detail

        return {
            # ---- Summary ----
            "health_score":  round(self.health_score * 100, 1),
            "health_label":  self.health_label,
            "degraded":      self.degraded,
            "as_of":         self.as_of.strftime("%Y-%m-%d %H:%M"),

            # ---- Freshness ----
            "freshness_overall":      fq.overall_state,
            "prices_state":           p.state if p else "MISSING",
            "fundamentals_state":     f.state if f else "MISSING",
            "weights_state":          w.state if w else "MISSING",
            "prices_age_hours":       round(p.age_hours, 1) if (p and p.age_hours is not None) else None,
            "fundamentals_age_hours": round(f.age_hours, 1) if (f and f.age_hours is not None) else None,
            "weights_age_hours":      round(w.age_hours, 1) if (w and w.age_hours is not None) else None,
            "last_price_date":        str(pd_.last_price_date) if (pd_ and pd_.last_price_date) else "—",
            "days_since_last_price":  pd_.days_since_last_price if pd_ else None,
            "price_date_gap_ok":      pd_.price_date_gap_ok if pd_ else False,

            # ---- Coverage ----
            "missing_price_tickers":  cq.missing_price_tickers,
            "sparse_price_tickers":   cq.sparse_price_tickers,
            "fallback_etfs":          cq.fallback_etfs,
            "spy_sector_weights_ok":  cq.spy_sector_weight.adequate,
            "spy_sectors_missing":    cq.spy_sector_weight.sectors_missing,

            # ---- Validation ----
            "validation_passed":        self.validation.passed,
            "validation_error_count":   self.validation.error_count,
            "validation_warning_count": self.validation.warning_count,

            # ---- Issues ----
            "warnings":      self.warnings,
            "errors":        self.errors,
            "warning_count": len(self.warnings),
            "error_count":   len(self.errors),
        }

    def to_coverage_table(self) -> pd.DataFrame:
        """
        Per-ETF coverage summary DataFrame for tabular display.

        Only includes sector ETFs and SPY (not macro/credit/vol tickers).
        """
        rows = []
        # Iterate holdings_coverage keys — these are only sector ETFs + SPY
        for etf in self.coverage.holdings_coverage:
            pc = self.coverage.price_coverage.get(etf)
            hc = self.coverage.holdings_coverage.get(etf)
            fc = self.coverage.fundamentals_coverage.get(etf)

            price_str    = f"{pc.coverage_ratio:.0%}" if (pc and pc.present) else "MISSING"
            hold_rows    = hc.holdings_row_count if hc else 0
            weight_str   = f"{hc.weight_sum_pct:.0f}%" if hc else "—"
            pe_str       = f"{fc.pe_coverage_ratio:.0%}" if fc else "—"
            fallback_str = "YES" if (fc and fc.etf_level_fallback) else "no"

            ok = (
                (pc is None or pc.adequate)
                and (hc is None or hc.weight_sum_adequate)
                and (fc is None or fc.adequate)
            )
            rows.append({
                "ETF":           etf,
                "Price":         price_str,
                "Holdings rows": hold_rows,
                "Weight sum %":  weight_str,
                "PE coverage":   pe_str,
                "Fallback":      fallback_str,
                "Status":        "OK" if ok else "DEGRADED",
            })

        return pd.DataFrame(rows)


# =========================================================================
# Scoring
# =========================================================================

def _compute_score(
    freshness_degraded: bool,
    coverage_score: float,
    error_count: int,
    warning_count: int,
) -> float:
    s = coverage_score
    if freshness_degraded:
        s -= 0.15
    s -= min(0.30, error_count   * 0.10)
    s -= min(0.12, warning_count * 0.03)
    return round(max(0.0, min(1.0, s)), 4)


def _label(score: float) -> str:
    if score >= _HEALTHY_THRESH:
        return "HEALTHY"
    elif score >= _DEGRADED_THRESH:
        return "DEGRADED"
    else:
        return "CRITICAL"


def _dedup(items: List[str]) -> List[str]:
    """Return items with duplicates removed, preserving order."""
    seen: set = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =========================================================================
# Main entry point
# =========================================================================

def build_data_health_report(
    prices_path: Path,
    fundamentals_path: Path,
    weights_path: Path,
    max_age_hours: float,
    sector_tickers: List[str],
    spy_ticker: str,
    expected_sectors: List[str],
    all_price_tickers: List[str],
) -> DataHealthReport:
    """
    Build a complete DataHealthReport from artifact paths and settings params.

    This is the primary entry point for the data_ops layer. Call this from
    DataOrchestrator or directly from any consumer that needs a health summary.

    Does NOT re-fetch from FMP or mutate any files.
    """
    as_of = datetime.now()

    freshness = assess_freshness(
        prices_path=prices_path,
        fundamentals_path=fundamentals_path,
        weights_path=weights_path,
        max_age_hours=max_age_hours,
    )

    coverage = assess_coverage(
        prices_path=prices_path,
        fundamentals_path=fundamentals_path,
        weights_path=weights_path,
        sector_tickers=sector_tickers,
        spy_ticker=spy_ticker,
        expected_sectors=expected_sectors,
        all_price_tickers=all_price_tickers,
    )

    # Load DataFrames for structural validation
    def _safe_read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    validation = validate_all(
        prices_df=_safe_read(prices_path),
        fundamentals_df=_safe_read(fundamentals_path),
        weights_df=_safe_read(weights_path),
        sector_tickers=sector_tickers,
        spy_ticker=spy_ticker,
    )

    score = _compute_score(
        freshness_degraded=freshness.degraded,
        coverage_score=coverage.health_score,
        error_count=validation.error_count,
        warning_count=validation.warning_count,
    )
    label = _label(score)

    # Aggregate warnings from all three layers (deduplicated, ordered)
    raw_warnings = (
        [f"[FRESHNESS] {w}"  for w in freshness.warnings]
        + [f"[COVERAGE] {w}" for w in coverage.warnings]
        + [f"[VALIDATION:{i.code}] {i.message}"
           for i in validation.issues if i.severity == "WARNING"]
    )
    raw_errors = [
        f"[VALIDATION:{i.code}] {i.message}"
        for i in validation.issues if i.severity == "ERROR"
    ]

    return DataHealthReport(
        as_of=as_of,
        health_score=score,
        health_label=label,
        degraded=(score < _HEALTHY_THRESH),
        freshness=freshness,
        coverage=coverage,
        validation=validation,
        warnings=_dedup(raw_warnings),
        errors=_dedup(raw_errors),
    )
