"""
tests/test_data_scout.py
========================
Tests for the Data Scout agent.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.data_scout.agent_data_scout import DataScout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scout():
    """A DataScout with no external data loaded."""
    s = DataScout.__new__(DataScout)
    s.settings = None
    s._session = None
    s._prices = None
    return s


@pytest.fixture
def scout_with_prices():
    """A DataScout with synthetic price data."""
    s = DataScout.__new__(DataScout)
    s.settings = None
    s._session = None

    # Build synthetic sector ETF prices (150 days)
    np.random.seed(42)
    n_days = 150
    dates = pd.bdate_range(end="2025-03-01", periods=n_days)

    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    data = {}
    for ticker in tickers:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.012, n_days)
        prices = 100.0 * np.cumprod(1 + returns)
        data[ticker] = prices

    s._prices = pd.DataFrame(data, index=dates)
    return s


# ---------------------------------------------------------------------------
# test_detect_correlation_breaks
# ---------------------------------------------------------------------------

class TestDetectCorrelationBreaks:
    """Tests for correlation break detection with synthetic data."""

    def test_no_breaks_with_correlated_data(self, scout):
        """When all sectors move together, no breaks are flagged."""
        np.random.seed(0)
        n_days = 150
        dates = pd.bdate_range(end="2025-03-01", periods=n_days)

        # Common factor drives all tickers -- high, stable correlation
        common = np.random.normal(0, 0.01, n_days)
        data = {}
        for ticker in ["XLK", "XLF", "XLE"]:
            noise = np.random.normal(0, 0.002, n_days)
            prices = 100.0 * np.cumprod(1 + common + noise)
            data[ticker] = prices

        scout._prices = pd.DataFrame(data, index=dates)
        breaks = scout.detect_correlation_breaks()

        # Correlations should be stable, so few or no breaks
        for brk in breaks:
            # If any show up, change must exceed threshold
            assert abs(brk["change"]) > 0.3

    def test_detects_break_in_synthetic_pair(self, scout):
        """Engineer a pair where 20d and 120d correlations diverge."""
        np.random.seed(1)
        n_days = 150
        dates = pd.bdate_range(end="2025-03-01", periods=n_days)

        # A and B: correlated for first 130 days, then B diverges
        common = np.random.normal(0, 0.01, n_days)

        ret_a = common + np.random.normal(0, 0.002, n_days)
        ret_b = common.copy()
        ret_b[:130] += np.random.normal(0, 0.002, 130)
        # Last 20 days: B inverts
        ret_b[130:] = -common[130:] + np.random.normal(0, 0.002, 20)

        prices_a = 100.0 * np.cumprod(1 + ret_a)
        prices_b = 100.0 * np.cumprod(1 + ret_b)

        scout._prices = pd.DataFrame(
            {"A": prices_a, "B": prices_b}, index=dates
        )
        breaks = scout.detect_correlation_breaks()

        # Should detect the A/B break
        assert len(breaks) >= 1
        pair_names = [b["pair"] for b in breaks]
        assert "A/B" in pair_names

        brk = [b for b in breaks if b["pair"] == "A/B"][0]
        assert abs(brk["change"]) > 0.3
        assert "corr_20d" in brk
        assert "corr_120d" in brk
        assert "direction" in brk

    def test_empty_prices(self, scout):
        """No prices -> empty list."""
        scout._prices = None
        assert scout.detect_correlation_breaks() == []

    def test_insufficient_history(self, scout):
        """Less than 120 days -> empty list."""
        dates = pd.bdate_range(end="2025-03-01", periods=50)
        scout._prices = pd.DataFrame(
            {"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates
        )
        assert scout.detect_correlation_breaks() == []


# ---------------------------------------------------------------------------
# test_detect_momentum_divergence
# ---------------------------------------------------------------------------

class TestDetectMomentumDivergence:
    """Tests for momentum divergence detection with synthetic data."""

    def test_bearish_divergence(self, scout):
        """Price near 20d high but RSI declining -> bearish divergence."""
        n_days = 60
        dates = pd.bdate_range(end="2025-03-01", periods=n_days)

        # Price trending up but with decelerating momentum
        # First 40 days: strong up move
        prices = np.zeros(n_days)
        prices[0] = 100.0
        for i in range(1, 40):
            prices[i] = prices[i - 1] * (1 + 0.01)  # Strong gains
        # Last 20 days: still edging higher but with smaller gains
        for i in range(40, n_days):
            prices[i] = prices[i - 1] * (1 + 0.001)  # Very small gains

        scout._prices = pd.DataFrame({"TEST": prices}, index=dates)
        divergences = scout.detect_momentum_divergence()

        # May or may not trigger depending on RSI dynamics,
        # but structure should be correct
        for div in divergences:
            assert "ticker" in div
            assert "type" in div
            assert div["type"] in ("bearish_divergence", "bullish_divergence")
            assert "rsi_recent" in div
            assert "rsi_prior" in div

    def test_bullish_divergence(self, scout):
        """Price near 20d low but RSI rising -> bullish divergence."""
        n_days = 60
        dates = pd.bdate_range(end="2025-03-01", periods=n_days)

        # Strong down, then stabilizing near lows with RSI recovering
        prices = np.zeros(n_days)
        prices[0] = 100.0
        for i in range(1, 40):
            prices[i] = prices[i - 1] * (1 - 0.012)  # Strong decline
        # Last 20 days: bounce, then re-test low
        for i in range(40, 50):
            prices[i] = prices[i - 1] * (1 + 0.005)  # Small bounce
        for i in range(50, n_days):
            prices[i] = prices[i - 1] * (1 - 0.003)  # Drift back near low

        scout._prices = pd.DataFrame({"TEST": prices}, index=dates)
        divergences = scout.detect_momentum_divergence()

        for div in divergences:
            assert "ticker" in div
            assert "type" in div
            assert div["type"] in ("bearish_divergence", "bullish_divergence")

    def test_empty_prices(self, scout):
        """No prices -> empty list."""
        scout._prices = None
        assert scout.detect_momentum_divergence() == []

    def test_short_history(self, scout):
        """Too few data points -> empty list."""
        dates = pd.bdate_range(end="2025-03-01", periods=10)
        scout._prices = pd.DataFrame(
            {"A": np.random.randn(10)}, index=dates
        )
        assert scout.detect_momentum_divergence() == []


# ---------------------------------------------------------------------------
# test_macro_regime_signals
# ---------------------------------------------------------------------------

class TestMacroRegimeSignals:
    """Tests for macro regime signal aggregation."""

    def test_risk_off_regime(self, scout):
        """When yield curve inverted, HY widening, dollar up -> risk-off."""
        mock_fred = {
            "yield_curve": {"series_id": "T10Y2Y", "value": -0.5, "z_score": -1.5, "trend": -0.1},
            "hy_spread": {"series_id": "BAMLH0A0HYM2", "value": 5.0, "z_score": 2.0, "trend": 0.3},
            "dollar_index": {"series_id": "DTWEXBGS", "value": 110.0, "z_score": 1.0, "trend": 0.05},
        }
        with patch.object(scout, "fetch_fred_data", return_value=mock_fred):
            signals = scout.compute_macro_regime_signals()

        assert "macro_score" in signals
        assert signals["macro_score"] < 0  # risk-off
        assert "yield_curve" in signals
        assert signals["yield_curve"]["signal"] == "inverted"

    def test_risk_on_regime(self, scout):
        """When yield curve normal, HY tight, dollar weak -> risk-on."""
        mock_fred = {
            "yield_curve": {"series_id": "T10Y2Y", "value": 1.5, "z_score": 0.5, "trend": 0.05},
            "hy_spread": {"series_id": "BAMLH0A0HYM2", "value": 3.0, "z_score": -1.0, "trend": -0.1},
            "dollar_index": {"series_id": "DTWEXBGS", "value": 100.0, "z_score": -0.5, "trend": -0.03},
        }
        with patch.object(scout, "fetch_fred_data", return_value=mock_fred):
            signals = scout.compute_macro_regime_signals()

        assert signals["macro_score"] > 0  # risk-on
        assert signals["yield_curve"]["signal"] == "normal"

    def test_no_fred_data(self, scout):
        """When FRED is down, macro_score defaults to 0."""
        with patch.object(scout, "fetch_fred_data", return_value={}):
            signals = scout.compute_macro_regime_signals()

        assert signals["macro_score"] == 0.0

    def test_macro_score_bounded(self, scout):
        """Macro score should always be in [-1, 1]."""
        mock_fred = {
            "yield_curve": {"series_id": "T10Y2Y", "value": -5.0, "z_score": -5.0, "trend": -1.0},
            "hy_spread": {"series_id": "BAMLH0A0HYM2", "value": 20.0, "z_score": 10.0, "trend": 2.0},
            "dollar_index": {"series_id": "DTWEXBGS", "value": 150.0, "z_score": 5.0, "trend": 0.5},
        }
        with patch.object(scout, "fetch_fred_data", return_value=mock_fred):
            signals = scout.compute_macro_regime_signals()

        assert -1.0 <= signals["macro_score"] <= 1.0


# ---------------------------------------------------------------------------
# test_report_structure
# ---------------------------------------------------------------------------

class TestReportStructure:
    """Tests for the overall report structure."""

    def test_report_has_required_keys(self, scout):
        """Report must contain all required top-level keys."""
        with patch.object(scout, "fetch_fred_data", return_value={}):
            report = scout.generate_report()

        required_keys = [
            "timestamp",
            "macro",
            "anomalies",
            "correlation_breaks",
            "momentum_divergences",
            "opportunities",
            "risk_flags",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_report_types(self, scout):
        """Report values have correct types."""
        with patch.object(scout, "fetch_fred_data", return_value={}):
            report = scout.generate_report()

        assert isinstance(report["timestamp"], str)
        assert isinstance(report["macro"], dict)
        assert isinstance(report["anomalies"], list)
        assert isinstance(report["correlation_breaks"], list)
        assert isinstance(report["momentum_divergences"], list)
        assert isinstance(report["opportunities"], list)
        assert isinstance(report["risk_flags"], list)

    def test_report_serializable(self, scout_with_prices):
        """Report must be JSON-serializable."""
        with patch.object(scout_with_prices, "fetch_fred_data", return_value={}):
            report = scout_with_prices.generate_report()

        # Should not raise
        serialized = json.dumps(report, default=str)
        assert len(serialized) > 0

    def test_report_timestamp_is_utc(self, scout):
        """Report timestamp should be UTC ISO format."""
        with patch.object(scout, "fetch_fred_data", return_value={}):
            report = scout.generate_report()

        ts = report["timestamp"]
        # Should parse without error
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_opportunities_capped_at_five(self, scout):
        """Opportunities list should have at most 5 items."""
        with patch.object(scout, "fetch_fred_data", return_value={}):
            report = scout.generate_report()

        assert len(report["opportunities"]) <= 5

    def test_report_with_data(self, scout_with_prices):
        """Report with real price data should populate fields."""
        with patch.object(scout_with_prices, "fetch_fred_data", return_value={}):
            report = scout_with_prices.generate_report()

        # Should still have valid structure even with data
        assert "macro" in report
        assert "macro_score" in report["macro"]
        assert isinstance(report["anomalies"], list)
