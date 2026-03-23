"""tests/test_validators.py — ValidationReport with good / bad DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_ops.validators import (
    ERROR,
    WARNING,
    validate_all,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
SPY = "SPY"
_NROWS = 300   # > 252  threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def good_prices() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-02", periods=_NROWS, freq="B")
    data = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (_NROWS, len(SECTORS) + 1)), axis=0))
    return pd.DataFrame(data, index=idx, columns=SECTORS + [SPY])


@pytest.fixture()
def good_fundamentals() -> pd.DataFrame:
    rows = [{"symbol": t, "pe": 20.0, "eps": 5.0, "marketCap": 1e9} for t in SECTORS + [SPY]]
    return pd.DataFrame(rows)


@pytest.fixture()
def good_weights() -> pd.DataFrame:
    holdings = [
        {"record_type": "holding", "etfSymbol": t, "asset": "AAPL", "weightPercentage": 80.0}
        for t in SECTORS
    ]
    spy_weights = [
        {"record_type": "sector_weight", "sector": "Technology", "weightPercentage": 30.0}
    ]
    return pd.DataFrame(holdings + spy_weights)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_good_data_passes(good_prices, good_fundamentals, good_weights):
    report = validate_all(good_prices, good_fundamentals, good_weights, SECTORS, SPY)
    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert report.error_count == 0


# ---------------------------------------------------------------------------
# prices errors
# ---------------------------------------------------------------------------
def test_empty_prices_produces_error(good_fundamentals, good_weights):
    report = validate_all(pd.DataFrame(), good_fundamentals, good_weights, SECTORS, SPY)
    assert report.passed is False
    assert report.error_count >= 1
    assert any(i.code == "PRICES_EMPTY" for i in report.errors)


def test_insufficient_rows_produces_error(good_fundamentals, good_weights):
    short_df = pd.DataFrame(
        np.ones((100, len(SECTORS) + 1)) * 100.0,
        columns=SECTORS + [SPY],
    )
    report = validate_all(short_df, good_fundamentals, good_weights, SECTORS, SPY)
    assert report.passed is False
    codes = [i.code for i in report.errors]
    assert "PRICES_INSUFFICIENT_HISTORY" in codes


def test_missing_sector_ticker_produces_error(good_prices, good_fundamentals, good_weights):
    prices_no_xlk = good_prices.drop(columns=["XLK"])
    report = validate_all(prices_no_xlk, good_fundamentals, good_weights, SECTORS, SPY)
    assert report.passed is False
    assert any(i.code == "SECTOR_TICKERS_MISSING" for i in report.errors)


def test_all_nan_column_warning(good_prices, good_fundamentals, good_weights):
    prices_with_nan = good_prices.copy()
    prices_with_nan["XLU"] = np.nan
    report = validate_all(prices_with_nan, good_fundamentals, good_weights, SECTORS, SPY)
    # One all-NaN column in 12-column frame → fraction ~8.3% < 10% → WARNING not ERROR
    assert any(i.code == "ALL_NAN_PRICE_COLUMNS" for i in report.warnings)


# ---------------------------------------------------------------------------
# fundamentals / weights degraded paths
# ---------------------------------------------------------------------------
def test_empty_fundamentals_is_warning_not_error(good_prices, good_weights):
    report = validate_all(good_prices, pd.DataFrame(), good_weights, SECTORS, SPY)
    assert report.passed is True          # fundamentals empty is WARNING, not ERROR
    assert report.warning_count >= 1
    assert any(i.severity == WARNING for i in report.issues)


def test_empty_weights_is_warning_not_error(good_prices, good_fundamentals):
    report = validate_all(good_prices, good_fundamentals, pd.DataFrame(), SECTORS, SPY)
    assert report.passed is True
    assert report.warning_count >= 1


def test_error_count_and_warning_count_are_consistent(good_prices, good_fundamentals, good_weights):
    report = validate_all(good_prices, good_fundamentals, good_weights, SECTORS, SPY)
    assert report.error_count == len(report.errors)
    assert report.warning_count == len(report.warnings)


def test_severity_propagation():
    """Adding an ERROR flips passed to False; WARNING does not."""
    from data_ops.validators import ValidationIssue
    r = ValidationReport()
    assert r.passed is True
    r.add(ValidationIssue(WARNING, "prices", "TEST_WARN", "warn", "fix"))
    assert r.passed is True
    r.add(ValidationIssue(ERROR, "prices", "TEST_ERR", "err", "fix"))
    assert r.passed is False
    assert r.error_count == 1
    assert r.warning_count == 1
