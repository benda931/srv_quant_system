"""
tests/test_execution_agent.py
==============================
Tests for the Execution Agent (signal-to-trade translator).
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

from agents.execution.agent_execution import (
    ExecutionAgent,
    _DEFAULT_SLIPPAGE_BPS,
    _DEFAULT_COMMISSION_PER_SHARE,
    _DEFAULT_CONVICTION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """An ExecutionAgent with no external dependencies."""
    a = ExecutionAgent.__new__(ExecutionAgent)
    a.settings = None
    a.slippage_bps = _DEFAULT_SLIPPAGE_BPS
    a.commission_per_share = _DEFAULT_COMMISSION_PER_SHARE
    a.conviction_threshold = _DEFAULT_CONVICTION_THRESHOLD
    a._leverage_engine = None
    a._execution_log = []
    return a


@pytest.fixture
def sample_master_df():
    """Sample master_df with conviction signals."""
    return pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLE", "XLV", "XLB", "XLU"],
        "conviction_score": [0.45, 0.32, 0.28, 0.18, 0.10, 0.05],
        "direction": ["LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"],
        "z_score": [1.2, -0.9, 0.8, -0.6, 0.3, -0.1],
        "regime": ["CALM", "CALM", "CALM", "CALM", "CALM", "CALM"],
    })


@pytest.fixture
def sample_prices():
    """Create a sample prices DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLB", "XLU", "^VIX"]
    data = {}
    for t in tickers:
        if t == "^VIX":
            data[t] = 20.0 + np.random.randn(n) * 2
        else:
            data[t] = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame(data, index=dates)


# ===========================================================================
# Constructor tests
# ===========================================================================

class TestExecutionAgentInit:
    def test_default_params(self, agent):
        assert agent.slippage_bps == 5
        assert agent.commission_per_share == 0.005
        assert agent.conviction_threshold == 0.15

    def test_custom_params(self):
        a = ExecutionAgent.__new__(ExecutionAgent)
        a.settings = None
        a.slippage_bps = 10
        a.commission_per_share = 0.01
        a.conviction_threshold = 0.20
        a._leverage_engine = None
        a._execution_log = []
        assert a.slippage_bps == 10
        assert a.commission_per_share == 0.01


# ===========================================================================
# Risk Guardian check
# ===========================================================================

class TestCheckRiskGuardian:
    def test_no_status_file_returns_true(self, agent):
        """If risk_status.json doesn't exist, assume GREEN."""
        with patch("agents.execution.agent_execution.RISK_STATUS_PATH") as mock_path:
            mock_path.exists.return_value = False
            assert agent.check_risk_guardian() is True

    def test_green_status_returns_true(self, agent, tmp_path):
        status_file = tmp_path / "risk_status.json"
        status_file.write_text(json.dumps({"level": "GREEN"}), encoding="utf-8")

        import agents.execution.agent_execution as mod
        original = mod.RISK_STATUS_PATH
        mod.RISK_STATUS_PATH = status_file
        try:
            assert agent.check_risk_guardian() is True
        finally:
            mod.RISK_STATUS_PATH = original

    def test_red_status_returns_false(self, agent, tmp_path):
        status_file = tmp_path / "risk_status.json"
        status_file.write_text(json.dumps({"level": "RED"}), encoding="utf-8")

        import agents.execution.agent_execution as mod
        original = mod.RISK_STATUS_PATH
        mod.RISK_STATUS_PATH = status_file
        try:
            assert agent.check_risk_guardian() is False
        finally:
            mod.RISK_STATUS_PATH = original

    def test_black_status_returns_false(self, agent, tmp_path):
        status_file = tmp_path / "risk_status.json"
        status_file.write_text(json.dumps({"level": "BLACK"}), encoding="utf-8")

        import agents.execution.agent_execution as mod
        original = mod.RISK_STATUS_PATH
        mod.RISK_STATUS_PATH = status_file
        try:
            assert agent.check_risk_guardian() is False
        finally:
            mod.RISK_STATUS_PATH = original


# ===========================================================================
# Signal filtering
# ===========================================================================

class TestFilterSignals:
    def test_empty_df(self, agent):
        assert agent.filter_signals(pd.DataFrame()) == []

    def test_filters_by_threshold(self, agent, sample_master_df):
        signals = agent.filter_signals(sample_master_df)
        # XLK(0.45), XLF(0.32), XLE(0.28), XLV(0.18) >= 0.15
        # XLB(0.10), XLU(0.05) < 0.15
        assert len(signals) == 4
        tickers = [s["ticker"] for s in signals]
        assert "XLK" in tickers
        assert "XLF" in tickers
        assert "XLE" in tickers
        assert "XLV" in tickers
        assert "XLB" not in tickers
        assert "XLU" not in tickers

    def test_filters_neutral(self, agent):
        df = pd.DataFrame({
            "ticker": ["XLK", "XLF"],
            "conviction_score": [0.50, 0.40],
            "direction": ["NEUTRAL", "LONG"],
        })
        signals = agent.filter_signals(df)
        assert len(signals) == 1
        assert signals[0]["ticker"] == "XLF"

    def test_sorted_by_conviction_desc(self, agent, sample_master_df):
        signals = agent.filter_signals(sample_master_df)
        convictions = [s["conviction"] for s in signals]
        assert convictions == sorted(convictions, reverse=True)

    def test_no_conviction_column(self, agent):
        df = pd.DataFrame({"ticker": ["XLK"], "foo": [1.0]})
        assert agent.filter_signals(df) == []

    def test_no_ticker_column(self, agent):
        df = pd.DataFrame({"conviction_score": [0.5], "direction": ["LONG"]})
        assert agent.filter_signals(df) == []

    def test_alternative_column_names(self, agent):
        """Test that alternative column names work (conviction, symbol)."""
        df = pd.DataFrame({
            "symbol": ["XLK", "XLF"],
            "conviction": [0.50, 0.40],
            "direction": ["LONG", "SHORT"],
        })
        signals = agent.filter_signals(df)
        assert len(signals) == 2


# ===========================================================================
# Position sizing
# ===========================================================================

class TestComputeSizing:
    def test_empty_signals(self, agent):
        assert agent.compute_sizing([]) == []

    def test_sizing_without_leverage(self, agent):
        signals = [
            {"ticker": "XLK", "direction": "LONG", "conviction": 0.40,
             "z_score": 1.0, "regime": "NORMAL"},
        ]
        sized = agent.compute_sizing(signals, regime="NORMAL")
        assert len(sized) == 1
        assert "notional" in sized[0]
        assert "weight" in sized[0]
        assert sized[0]["notional"] > 0

    def test_max_8_positions(self, agent):
        signals = [
            {"ticker": f"T{i}", "direction": "LONG", "conviction": 0.30,
             "z_score": 1.0, "regime": "NORMAL"}
            for i in range(15)
        ]
        sized = agent.compute_sizing(signals)
        assert len(sized) == 8

    def test_weight_capped(self, agent):
        signals = [
            {"ticker": "XLK", "direction": "LONG", "conviction": 0.99,
             "z_score": 1.0, "regime": "NORMAL"},
        ]
        sized = agent.compute_sizing(signals)
        assert sized[0]["weight"] <= 0.20


# ===========================================================================
# Slippage model
# ===========================================================================

class TestApplySlippage:
    def test_long_slippage_increases_price(self, agent):
        price = 100.0
        fill = agent.apply_slippage(price, "LONG")
        assert fill > price

    def test_short_slippage_decreases_price(self, agent):
        price = 100.0
        fill = agent.apply_slippage(price, "SHORT")
        assert fill < price

    def test_slippage_magnitude(self, agent):
        price = 100.0
        fill = agent.apply_slippage(price, "LONG")
        expected = price * (1.0 + 5 / 10_000)
        assert abs(fill - expected) < 0.001

    def test_buy_slippage(self, agent):
        fill = agent.apply_slippage(100.0, "BUY")
        assert fill > 100.0

    def test_sell_slippage(self, agent):
        fill = agent.apply_slippage(100.0, "SELL")
        assert fill < 100.0


# ===========================================================================
# Execution log saving
# ===========================================================================

class TestSaveExecutionLog:
    def test_save_creates_file(self, agent, tmp_path):
        import agents.execution.agent_execution as mod
        original = mod.EXECUTION_LOG_PATH
        mod.EXECUTION_LOG_PATH = tmp_path / "execution_log.json"
        try:
            agent._execution_log = [
                {"action": "ENTRY", "ticker": "XLK", "timestamp": "2025-01-01T00:00:00Z"}
            ]
            agent._save_execution_log()
            assert mod.EXECUTION_LOG_PATH.exists()
            loaded = json.loads(mod.EXECUTION_LOG_PATH.read_text(encoding="utf-8"))
            assert len(loaded) == 1
            assert loaded[0]["ticker"] == "XLK"
        finally:
            mod.EXECUTION_LOG_PATH = original

    def test_save_appends_to_existing(self, agent, tmp_path):
        import agents.execution.agent_execution as mod
        original = mod.EXECUTION_LOG_PATH
        log_path = tmp_path / "execution_log.json"
        log_path.write_text(json.dumps([{"action": "OLD"}]), encoding="utf-8")
        mod.EXECUTION_LOG_PATH = log_path
        try:
            agent._execution_log = [{"action": "NEW"}]
            agent._save_execution_log()
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            assert len(loaded) == 2
        finally:
            mod.EXECUTION_LOG_PATH = original

    def test_save_caps_at_500(self, agent, tmp_path):
        import agents.execution.agent_execution as mod
        original = mod.EXECUTION_LOG_PATH
        log_path = tmp_path / "execution_log.json"
        existing = [{"action": f"OLD_{i}"} for i in range(498)]
        log_path.write_text(json.dumps(existing), encoding="utf-8")
        mod.EXECUTION_LOG_PATH = log_path
        try:
            agent._execution_log = [{"action": "NEW_1"}, {"action": "NEW_2"}, {"action": "NEW_3"}]
            agent._save_execution_log()
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            assert len(loaded) == 500
        finally:
            mod.EXECUTION_LOG_PATH = original


# ===========================================================================
# Full run integration (mocked)
# ===========================================================================

class TestRunIntegration:
    @patch("agents.execution.agent_execution.RISK_STATUS_PATH")
    def test_run_halted_by_guardian(self, mock_path, agent, tmp_path):
        """When Risk Guardian says RED, run should halt gracefully."""
        status_file = tmp_path / "risk_status.json"
        status_file.write_text(json.dumps({"level": "RED"}), encoding="utf-8")

        import agents.execution.agent_execution as mod
        original = mod.RISK_STATUS_PATH
        mod.RISK_STATUS_PATH = status_file
        try:
            summary = agent.run()
            assert summary["risk_guardian_ok"] is False
            assert len(summary["entries"]) == 0
            assert len(summary["exits"]) == 0
        finally:
            mod.RISK_STATUS_PATH = original
