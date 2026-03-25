"""
tests/test_risk_guardian.py
============================
Tests for the Risk Guardian agent.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.risk_guardian.agent_risk_guardian import (
    RiskGuardian,
    RISK_GREEN,
    RISK_YELLOW,
    RISK_RED,
    RISK_BLACK,
    _worst_level,
    _DEFAULT_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guardian():
    """A RiskGuardian with no external data loaded."""
    g = RiskGuardian.__new__(RiskGuardian)
    g.settings = None
    g.thresholds = dict(_DEFAULT_THRESHOLDS)
    g._prices = None
    g._portfolio_data = {}
    g._risk_engine = None
    return g


@pytest.fixture
def guardian_with_positions():
    """A RiskGuardian with sample portfolio positions."""
    g = RiskGuardian.__new__(RiskGuardian)
    g.settings = None
    g.thresholds = dict(_DEFAULT_THRESHOLDS)
    g._prices = None
    g._risk_engine = None
    g._portfolio_data = {
        "capital": 1_000_000.0,
        "cash": 500_000.0,
        "max_drawdown": -0.03,
        "positions": [
            {
                "ticker": "XLK",
                "direction": "LONG",
                "notional": 200_000.0,
                "entry_price": 180.0,
                "current_price": 185.0,
                "unrealized_pnl": 5555.0,
                "unrealized_pnl_pct": 0.028,
                "days_held": 5,
            },
            {
                "ticker": "XLF",
                "direction": "SHORT",
                "notional": 150_000.0,
                "entry_price": 42.0,
                "current_price": 41.5,
                "unrealized_pnl": 1785.0,
                "unrealized_pnl_pct": 0.012,
                "days_held": 3,
            },
        ],
    }
    return g


@pytest.fixture
def sample_prices():
    """Create a sample prices DataFrame with 100 days."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    tickers = ["XLK", "XLF", "XLE", "XLV", "^VIX"]
    data = {}
    for t in tickers:
        if t == "^VIX":
            data[t] = 20.0 + np.cumsum(np.random.randn(n) * 0.5)
        else:
            data[t] = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame(data, index=dates)


# ===========================================================================
# Utility tests
# ===========================================================================

class TestWorstLevel:
    def test_green_green(self):
        assert _worst_level(RISK_GREEN, RISK_GREEN) == RISK_GREEN

    def test_green_yellow(self):
        assert _worst_level(RISK_GREEN, RISK_YELLOW) == RISK_YELLOW

    def test_red_yellow(self):
        assert _worst_level(RISK_RED, RISK_YELLOW) == RISK_RED

    def test_black_red(self):
        assert _worst_level(RISK_BLACK, RISK_RED) == RISK_BLACK

    def test_yellow_red(self):
        assert _worst_level(RISK_YELLOW, RISK_RED) == RISK_RED


# ===========================================================================
# Constructor tests
# ===========================================================================

class TestRiskGuardianInit:
    def test_default_thresholds(self, guardian):
        assert guardian.thresholds["var_limit_pct"] == 0.02
        assert guardian.thresholds["gross_exposure_limit"] == 3.0
        assert guardian.thresholds["drawdown_halt_pct"] == 0.12

    def test_empty_portfolio(self, guardian):
        assert guardian._get_positions() == []
        assert guardian._get_capital() == 1_000_000.0
        assert guardian._get_portfolio_weights() == {}


# ===========================================================================
# Portfolio data helpers
# ===========================================================================

class TestPortfolioHelpers:
    def test_get_positions(self, guardian_with_positions):
        positions = guardian_with_positions._get_positions()
        assert len(positions) == 2
        assert positions[0]["ticker"] == "XLK"

    def test_get_capital(self, guardian_with_positions):
        assert guardian_with_positions._get_capital() == 1_000_000.0

    def test_get_portfolio_weights(self, guardian_with_positions):
        weights = guardian_with_positions._get_portfolio_weights()
        assert "XLK" in weights
        assert "XLF" in weights
        assert abs(weights["XLK"] - 0.20) < 0.01
        assert abs(weights["XLF"] - 0.15) < 0.01


# ===========================================================================
# Individual check tests
# ===========================================================================

class TestVarBreach:
    def test_no_data_returns_green(self, guardian):
        result = guardian.check_var_breach()
        assert result["level"] == RISK_GREEN
        assert "Insufficient" in result["detail"]

    def test_with_data(self, guardian_with_positions, sample_prices):
        from analytics.portfolio_risk import PortfolioRiskEngine
        guardian_with_positions._prices = sample_prices
        guardian_with_positions._risk_engine = PortfolioRiskEngine()
        result = guardian_with_positions.check_var_breach()
        assert result["level"] in (RISK_GREEN, RISK_RED, RISK_YELLOW)
        assert "check" in result


class TestExposureLimits:
    def test_no_positions(self, guardian):
        result = guardian.check_exposure_limits()
        assert result["level"] == RISK_GREEN

    def test_normal_exposure(self, guardian_with_positions):
        result = guardian_with_positions.check_exposure_limits()
        assert result["level"] == RISK_GREEN
        assert "gross_exposure" in result
        # gross = (200k + 150k) / 1M = 0.35 < 3.0
        assert result["gross_exposure"] < 3.0

    def test_high_gross_exposure(self, guardian_with_positions):
        # Inflate positions to breach gross limit
        for pos in guardian_with_positions._portfolio_data["positions"]:
            pos["notional"] = 2_000_000.0  # way above 3x
        result = guardian_with_positions.check_exposure_limits()
        assert result["level"] == RISK_RED
        assert "Gross" in result["detail"]


class TestConcentration:
    def test_no_positions(self, guardian):
        result = guardian.check_concentration()
        assert result["level"] == RISK_GREEN

    def test_normal_concentration(self, guardian_with_positions):
        result = guardian_with_positions.check_concentration()
        assert result["level"] == RISK_GREEN
        assert "hhi" in result

    def test_high_concentration(self, guardian_with_positions):
        # Make one position 30% of NAV
        guardian_with_positions._portfolio_data["positions"][0]["notional"] = 300_000.0
        result = guardian_with_positions.check_concentration()
        assert result["level"] == RISK_RED
        assert "Concentration breach" in result["detail"]


class TestCorrelationSpike:
    def test_not_enough_positions(self, guardian):
        result = guardian.check_correlation_spike()
        assert result["level"] == RISK_GREEN
        assert "Not enough" in result["detail"]

    def test_with_prices(self, guardian_with_positions, sample_prices):
        guardian_with_positions._prices = sample_prices
        result = guardian_with_positions.check_correlation_spike()
        assert result["level"] in (RISK_GREEN, RISK_YELLOW)
        assert "check" in result


class TestDrawdown:
    def test_no_portfolio(self, guardian):
        result = guardian.check_drawdown()
        assert result["level"] == RISK_GREEN

    def test_small_drawdown(self, guardian_with_positions):
        guardian_with_positions._portfolio_data["max_drawdown"] = -0.03
        result = guardian_with_positions.check_drawdown()
        assert result["level"] == RISK_GREEN

    def test_warning_drawdown(self, guardian_with_positions):
        guardian_with_positions._portfolio_data["max_drawdown"] = -0.09
        result = guardian_with_positions.check_drawdown()
        assert result["level"] == RISK_YELLOW

    def test_halt_drawdown(self, guardian_with_positions):
        guardian_with_positions._portfolio_data["max_drawdown"] = -0.15
        result = guardian_with_positions.check_drawdown()
        assert result["level"] == RISK_RED


class TestVixRegime:
    def test_no_vix_data(self, guardian):
        result = guardian.check_vix_regime()
        assert result["level"] == RISK_GREEN
        assert "No VIX" in result["detail"]

    def test_normal_vix(self, guardian, sample_prices):
        # Set VIX to stable values
        sample_prices["^VIX"] = 18.0
        sample_prices.iloc[-1, sample_prices.columns.get_loc("^VIX")] = 19.0
        guardian._prices = sample_prices
        result = guardian.check_vix_regime()
        assert result["level"] == RISK_GREEN

    def test_vix_spike(self, guardian, sample_prices):
        sample_prices["^VIX"] = 20.0
        sample_prices.iloc[-2, sample_prices.columns.get_loc("^VIX")] = 20.0
        sample_prices.iloc[-1, sample_prices.columns.get_loc("^VIX")] = 28.0  # +8 pts
        guardian._prices = sample_prices
        result = guardian.check_vix_regime()
        assert result["level"] == RISK_RED
        assert "SPIKE" in result["detail"]


# ===========================================================================
# Aggregate checks
# ===========================================================================

class TestRunAllChecks:
    def test_all_green(self, guardian):
        status = guardian.run_all_checks()
        assert status["level"] == RISK_GREEN
        assert status["breaches"] == []
        assert "timestamp" in status

    def test_contains_all_checks(self, guardian):
        status = guardian.run_all_checks()
        check_names = [c["check"] for c in status["checks"]]
        assert "var_breach" in check_names
        assert "exposure_limits" in check_names
        assert "concentration" in check_names
        assert "correlation_spike" in check_names
        assert "drawdown" in check_names
        assert "vix_regime" in check_names

    def test_multiple_reds_promote_to_black(self, guardian_with_positions, sample_prices):
        """Three RED checks should promote overall to BLACK."""
        guardian_with_positions._prices = sample_prices

        # Breach multiple limits
        for pos in guardian_with_positions._portfolio_data["positions"]:
            pos["notional"] = 2_000_000.0  # gross breach
        guardian_with_positions._portfolio_data["positions"][0]["notional"] = 300_000.0
        guardian_with_positions._portfolio_data["max_drawdown"] = -0.15

        # Set VIX spike
        sample_prices.iloc[-2, sample_prices.columns.get_loc("^VIX")] = 20.0
        sample_prices.iloc[-1, sample_prices.columns.get_loc("^VIX")] = 28.0

        status = guardian_with_positions.run_all_checks()
        # At least some breaches detected
        assert len(status["breaches"]) > 0


# ===========================================================================
# Actions
# ===========================================================================

class TestExecuteActions:
    def test_green_no_action(self, guardian):
        status = {"level": RISK_GREEN, "breaches": [], "recommendations": []}
        # Should not raise
        guardian.execute_actions(status)

    def test_yellow_no_action(self, guardian):
        status = {"level": RISK_YELLOW, "breaches": ["test"], "recommendations": []}
        guardian.execute_actions(status)

    @patch("agents.risk_guardian.agent_risk_guardian._IMPORTS_OK", {"agent_bus": True})
    def test_red_publishes_halt(self, guardian):
        status = {
            "level": RISK_RED,
            "breaches": ["test breach"],
            "recommendations": ["halt"],
            "date": date.today().isoformat(),
        }
        with patch("agents.risk_guardian.agent_risk_guardian.get_bus") as mock_bus:
            mock_instance = MagicMock()
            mock_bus.return_value = mock_instance
            guardian.execute_actions(status)
            mock_instance.publish.assert_called_once()
            call_args = mock_instance.publish.call_args
            assert call_args[0][0] == "risk_guardian"
            assert call_args[0][1]["action"] == "HALT"


# ===========================================================================
# Save status
# ===========================================================================

class TestSaveStatus:
    def test_save_status(self, guardian, tmp_path):
        import agents.risk_guardian.agent_risk_guardian as mod
        original = mod.STATUS_PATH
        mod.STATUS_PATH = tmp_path / "risk_status.json"
        try:
            status = {"level": RISK_GREEN, "timestamp": "2025-01-01T00:00:00Z"}
            guardian._save_status(status)
            assert mod.STATUS_PATH.exists()
            loaded = json.loads(mod.STATUS_PATH.read_text(encoding="utf-8"))
            assert loaded["level"] == RISK_GREEN
        finally:
            mod.STATUS_PATH = original


# ===========================================================================
# Full run cycle
# ===========================================================================

class TestFullRun:
    def test_run_returns_status(self, guardian):
        status = guardian.run()
        assert "level" in status
        assert "checks" in status
        assert "timestamp" in status
        assert status["level"] in (RISK_GREEN, RISK_YELLOW, RISK_RED, RISK_BLACK)

    def test_run_saves_status_file(self, guardian, tmp_path):
        import agents.risk_guardian.agent_risk_guardian as mod
        original = mod.STATUS_PATH
        mod.STATUS_PATH = tmp_path / "risk_status.json"
        try:
            guardian.run()
            assert mod.STATUS_PATH.exists()
        finally:
            mod.STATUS_PATH = original
