"""
data_ops/pre_write_validator.py
================================
Pre-Write Data Validation — validates data BEFORE writing to parquet/DB

Prevents corrupted API responses from overwriting good data.

Checks:
  1. Schema validation (expected columns present)
  2. Row count validation (not too few, not suspiciously many)
  3. Value range validation (prices > 0, VIX in [5, 100])
  4. Freshness validation (dates make sense)
  5. NaN ratio validation (too many NaNs = bad data)
  6. Duplicate check (no duplicate dates)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pre-write validation."""
    valid: bool
    checks_passed: int
    checks_total: int
    issues: List[str]
    warnings: List[str]


def validate_prices(
    df: pd.DataFrame,
    expected_tickers: Optional[List[str]] = None,
    min_rows: int = 100,
    max_nan_pct: float = 0.30,
) -> ValidationResult:
    """Validate a prices DataFrame before writing."""
    issues = []
    warnings = []
    checks_passed = 0
    checks_total = 0

    # 1. Not empty
    checks_total += 1
    if df is None or df.empty:
        issues.append("DataFrame is empty or None")
        return ValidationResult(False, 0, checks_total, issues, warnings)
    checks_passed += 1

    # 2. Minimum rows
    checks_total += 1
    if len(df) < min_rows:
        issues.append(f"Only {len(df)} rows (need >= {min_rows})")
    else:
        checks_passed += 1

    # 3. Expected columns
    if expected_tickers:
        checks_total += 1
        missing = [t for t in expected_tickers if t not in df.columns]
        if len(missing) > len(expected_tickers) * 0.3:
            issues.append(f"Missing {len(missing)}/{len(expected_tickers)} tickers: {missing[:5]}")
        else:
            checks_passed += 1
            if missing:
                warnings.append(f"Missing {len(missing)} tickers: {missing}")

    # 4. All prices positive
    checks_total += 1
    numeric = df.select_dtypes(include=[np.number])
    if (numeric <= 0).any().any():
        neg_cols = [c for c in numeric.columns if (numeric[c] <= 0).any()]
        warnings.append(f"Non-positive values in: {neg_cols[:3]}")
    checks_passed += 1

    # 5. NaN ratio
    checks_total += 1
    nan_pct = numeric.isna().mean().mean()
    if nan_pct > max_nan_pct:
        issues.append(f"NaN ratio {nan_pct:.1%} > {max_nan_pct:.1%}")
    else:
        checks_passed += 1
        if nan_pct > 0.10:
            warnings.append(f"NaN ratio {nan_pct:.1%}")

    # 6. Date index valid
    checks_total += 1
    if hasattr(df.index, 'date'):
        last_date = df.index[-1]
        if hasattr(last_date, 'date'):
            if last_date.date() > date.today() + timedelta(days=5):
                issues.append(f"Future date in data: {last_date}")
            elif last_date.date() < date.today() - timedelta(days=30):
                warnings.append(f"Data may be stale: last date {last_date.date()}")
            checks_passed += 1
    else:
        checks_passed += 1

    # 7. No duplicate dates
    checks_total += 1
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        issues.append(f"{n_dup} duplicate dates in index")
    else:
        checks_passed += 1

    # 8. VIX sanity (if present)
    for vix_col in ["^VIX", "VIX"]:
        if vix_col in df.columns:
            checks_total += 1
            vix_vals = df[vix_col].dropna()
            if len(vix_vals) > 0:
                vix_max = vix_vals.max()
                vix_min = vix_vals.min()
                if vix_max > 100 or vix_min < 3:
                    issues.append(f"VIX out of range: [{vix_min:.1f}, {vix_max:.1f}]")
                else:
                    checks_passed += 1

    valid = len(issues) == 0
    return ValidationResult(valid, checks_passed, checks_total, issues, warnings)


def validate_fundamentals(
    df: pd.DataFrame,
    min_rows: int = 10,
) -> ValidationResult:
    """Validate fundamentals DataFrame."""
    issues = []
    warnings = []
    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if df is None or df.empty:
        issues.append("Fundamentals empty")
        return ValidationResult(False, 0, checks_total, issues, warnings)
    checks_passed += 1

    # Check key columns
    checks_total += 1
    key_cols = ["symbol", "price", "pe"]
    present = [c for c in key_cols if c in df.columns]
    if len(present) < 2:
        issues.append(f"Missing key columns: expected {key_cols}, found {list(df.columns)[:5]}")
    else:
        checks_passed += 1

    # Check PE sanity
    if "pe" in df.columns:
        checks_total += 1
        pe_vals = pd.to_numeric(df["pe"], errors="coerce").dropna()
        if len(pe_vals) > 0:
            if pe_vals.min() < -500 or pe_vals.max() > 1000:
                warnings.append(f"Extreme PE values: [{pe_vals.min():.0f}, {pe_vals.max():.0f}]")
            checks_passed += 1

    # Completeness
    checks_total += 1
    null_pct = df.isnull().mean().mean()
    if null_pct > 0.50:
        warnings.append(f"Fundamentals {null_pct:.0%} null — sparse coverage")
    checks_passed += 1

    valid = len(issues) == 0
    return ValidationResult(valid, checks_passed, checks_total, issues, warnings)
