"""tests/test_quality.py — CoverageReport health_score with synthetic DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_ops.quality import (
    SPYSectorWeightCoverage,
    _assess_price_coverage,
    _assess_holdings,
    _assess_spy_weights,
    _health_score,
    assess_coverage,
)

SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
SPY = "SPY"
NROWS = 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_prices(tickers=SECTORS, nrows=NROWS, nan_ticker=None) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    data = rng.uniform(90.0, 110.0, (nrows, len(tickers)))
    df = pd.DataFrame(data, columns=tickers)
    if nan_ticker and nan_ticker in df.columns:
        df[nan_ticker] = np.nan
    return df


def _make_weights(etfs=SECTORS, weight_pct=80.0) -> pd.DataFrame:
    rows = [
        {"record_type": "holding", "etfSymbol": e, "asset": f"A_{e}",
         "weightPercentage": weight_pct}
        for e in etfs
    ]
    rows.append({"record_type": "sector_weight", "sector": "Technology",
                 "weightPercentage": 30.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _assess_price_coverage
# ---------------------------------------------------------------------------
def test_price_coverage_all_present_and_adequate():
    df = _make_prices()
    cov = _assess_price_coverage(df, SECTORS)
    assert all(c.present for c in cov.values())
    assert all(c.adequate for c in cov.values())
    assert all(c.coverage_ratio == pytest.approx(1.0) for c in cov.values())


def test_price_coverage_missing_ticker():
    df = _make_prices(tickers=[t for t in SECTORS if t != "XLK"])
    cov = _assess_price_coverage(df, SECTORS)
    assert cov["XLK"].present is False
    assert cov["XLK"].adequate is False
    assert cov["XLK"].warning is not None


def test_price_coverage_sparse_ticker():
    df = _make_prices()
    # Make XLP only 50% populated — below 80% threshold
    df["XLP"] = np.where(np.arange(NROWS) % 2 == 0, df["XLP"], np.nan)
    cov = _assess_price_coverage(df, SECTORS)
    assert cov["XLP"].present is True
    assert cov["XLP"].adequate is False
    assert cov["XLP"].coverage_ratio < 0.80


def test_price_coverage_ratio_is_in_0_1():
    df = _make_prices()
    cov = _assess_price_coverage(df, SECTORS)
    for c in cov.values():
        assert 0.0 <= c.coverage_ratio <= 1.0


# ---------------------------------------------------------------------------
# _health_score
# ---------------------------------------------------------------------------
def test_health_score_full_data_near_one():
    df = _make_prices()
    wdf = _make_weights()
    pc = _assess_price_coverage(df, SECTORS)
    hc = _assess_holdings(wdf, SECTORS)
    sw = _assess_spy_weights(wdf, ["Technology"])
    # Build minimal fundamentals coverage: use holdings without fund_df (fallback)
    from data_ops.quality import ETFFundamentalsCoverage
    fc = {t: ETFFundamentalsCoverage(t, 10, 8, 0.8, 80.0, False, True, None) for t in SECTORS}
    score = _health_score(pc, hc, fc, sw)
    assert 0.0 <= score <= 1.0
    assert score > 0.70


def test_health_score_empty_price_coverage():
    from data_ops.quality import HoldingsCoverage
    pc = {}
    hc = {}
    fc = {}
    sw = SPYSectorWeightCoverage(0, [], ["Technology"], False, "missing")
    score = _health_score(pc, hc, fc, sw)
    assert score == pytest.approx(0.0, abs=0.01)


def test_health_score_all_missing_tickers():
    df = pd.DataFrame()     # empty → all tickers absent
    pc = _assess_price_coverage(df, SECTORS)
    from data_ops.quality import HoldingsCoverage, ETFFundamentalsCoverage
    hc = {t: HoldingsCoverage(t, 0, 0.0, False, "no data") for t in SECTORS}
    fc = {t: ETFFundamentalsCoverage(t, 0, 0, 0.0, 0.0, True, False, "fallback") for t in SECTORS}
    sw = SPYSectorWeightCoverage(0, [], list(SECTORS), False, "none")
    score = _health_score(pc, hc, fc, sw)
    assert score < 0.30


# ---------------------------------------------------------------------------
# assess_coverage — integrated (writes real parquet to tmp_path)
# ---------------------------------------------------------------------------
def test_assess_coverage_with_parquet_files(tmp_path):
    prices_df = _make_prices()
    prices_df.to_parquet(tmp_path / "prices.parquet")
    pd.DataFrame(
        [{"symbol": t, "pe": 20.0, "eps": 3.0, "marketCap": 1e9} for t in SECTORS]
    ).to_parquet(tmp_path / "fundamentals.parquet")
    _make_weights().to_parquet(tmp_path / "weights.parquet")

    from config.settings import Settings
    report = assess_coverage(
        prices_path=tmp_path / "prices.parquet",
        fundamentals_path=tmp_path / "fundamentals.parquet",
        weights_path=tmp_path / "weights.parquet",
        sector_tickers=SECTORS,
        spy_ticker=SPY,
        expected_sectors=["Technology"],
        all_price_tickers=SECTORS,
    )
    assert 0.0 <= report.health_score <= 1.0
    assert report.missing_price_tickers == []
    assert isinstance(report.degraded, bool)


def test_assess_coverage_missing_files_gives_low_score(tmp_path):
    report = assess_coverage(
        prices_path=tmp_path / "no_prices.parquet",
        fundamentals_path=tmp_path / "no_fund.parquet",
        weights_path=tmp_path / "no_weights.parquet",
        sector_tickers=SECTORS,
        spy_ticker=SPY,
        expected_sectors=["Technology"],
        all_price_tickers=SECTORS,
    )
    assert report.health_score < 0.30
    assert report.degraded is True
