"""tests/test_leverage_engine.py — Leverage engine unit tests (20 tests)."""
from __future__ import annotations

import numpy as np
import pytest

from analytics.leverage_engine import (
    LeverageEngine,
    LeverageResult,
    PositionAllocation,
    kelly_fraction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> LeverageEngine:
    return LeverageEngine(base_capital=1_000_000, max_leverage=5.0)


@pytest.fixture
def simple_signals() -> list:
    return [
        {"symbol": "XLK", "conviction": 0.9, "sector": "Technology"},
        {"symbol": "XLF", "conviction": 0.7, "sector": "Financials"},
        {"symbol": "XLE", "conviction": 0.5, "sector": "Energy"},
        {"symbol": "XLV", "conviction": 0.3, "sector": "Health Care"},
    ]


@pytest.fixture
def cov_3x3() -> np.ndarray:
    """Simple 3x3 diagonal covariance (uncorrelated assets)."""
    return np.diag([0.04, 0.09, 0.16])  # vols: 20%, 30%, 40%


# ---------------------------------------------------------------------------
# Kelly Criterion tests
# ---------------------------------------------------------------------------

class TestKellyFraction:

    def test_positive_edge(self):
        """60% win rate, 1:1 payoff → positive Kelly."""
        f = kelly_fraction(0.60, 1.0, 1.0)
        assert f > 0
        # Full Kelly = (0.6*1 - 0.4)/1 = 0.2, half = 0.1
        assert abs(f - 0.10) < 1e-9

    def test_no_edge(self):
        """50% win rate, 1:1 payoff → zero Kelly."""
        f = kelly_fraction(0.50, 1.0, 1.0)
        assert f == 0.0

    def test_negative_edge(self):
        """40% win rate, 1:1 payoff → negative edge → floored at 0."""
        f = kelly_fraction(0.40, 1.0, 1.0)
        assert f == 0.0

    def test_high_payoff_ratio(self):
        """Low win rate but high payoff compensates."""
        f = kelly_fraction(0.30, 5.0, 1.0, kelly_frac=1.0)
        # full_kelly = (0.3*5 - 0.7)/5 = (1.5-0.7)/5 = 0.16
        assert abs(f - 0.16) < 1e-9

    def test_zero_avg_loss_returns_zero(self):
        f = kelly_fraction(0.6, 1.0, 0.0)
        assert f == 0.0

    def test_invalid_win_rate(self):
        assert kelly_fraction(0.0, 1.0, 1.0) == 0.0
        assert kelly_fraction(1.0, 1.0, 1.0) == 0.0
        assert kelly_fraction(-0.1, 1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Drawdown deleveraging tests
# ---------------------------------------------------------------------------

class TestDrawdownDeleverage:

    def test_no_drawdown(self, engine):
        assert engine.drawdown_deleverage(0.0) == 1.0

    def test_small_drawdown_below_start(self, engine):
        assert engine.drawdown_deleverage(0.01) == 1.0

    def test_at_start_boundary(self, engine):
        assert engine.drawdown_deleverage(0.02) == pytest.approx(1.0)

    def test_midrange_2_to_5(self, engine):
        """3.5% DD → halfway between 2% and 5%, should be ~75%."""
        mult = engine.drawdown_deleverage(0.035)
        assert 0.70 < mult < 0.80

    def test_at_5pct(self, engine):
        mult = engine.drawdown_deleverage(0.05)
        assert mult == pytest.approx(0.50, abs=1e-6)

    def test_at_8pct(self, engine):
        """At 8% DD, we enter emergency zone → 10% of leverage."""
        mult = engine.drawdown_deleverage(0.08)
        assert mult == pytest.approx(0.10, abs=1e-6)

    def test_emergency_zone(self, engine):
        """10% DD → between 8-12%, should be small but > 0."""
        mult = engine.drawdown_deleverage(0.10)
        assert 0.0 < mult < 0.10

    def test_full_stop(self, engine):
        assert engine.drawdown_deleverage(0.12) == 0.0

    def test_beyond_full_stop(self, engine):
        assert engine.drawdown_deleverage(0.20) == 0.0


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------

class TestVolTargetLeverage:

    def test_normal_vol(self, engine):
        """Portfolio vol = 10%, target = 10% → leverage = 1x."""
        lev = engine.vol_target_leverage(0.10)
        assert lev == pytest.approx(1.0)

    def test_low_vol_amplifies(self, engine):
        """Portfolio vol = 5%, target = 10% → leverage = 2x."""
        lev = engine.vol_target_leverage(0.05)
        assert lev == pytest.approx(2.0)

    def test_high_vol_reduces(self, engine):
        """Portfolio vol = 20%, target = 10% → leverage = 0.5x."""
        lev = engine.vol_target_leverage(0.20)
        assert lev == pytest.approx(0.5)

    def test_capped_at_max(self, engine):
        """Very low vol should not exceed max_leverage."""
        lev = engine.vol_target_leverage(0.01)
        assert lev == engine.max_leverage

    def test_zero_vol_returns_zero(self, engine):
        assert engine.vol_target_leverage(0.0) == 0.0


# ---------------------------------------------------------------------------
# Compute target leverage
# ---------------------------------------------------------------------------

class TestComputeTargetLeverage:

    def test_calm_high_sharpe(self, engine):
        result = engine.compute_target_leverage("CALM", 14.0, 0.0, 2.0)
        assert isinstance(result, LeverageResult)
        assert result.target_leverage > 0
        assert result.target_leverage <= engine.max_leverage
        assert "CALM" in result.reasoning

    def test_crisis_zeroes_leverage(self, engine):
        result = engine.compute_target_leverage("CRISIS", 40.0, 0.10, 1.5)
        assert result.target_leverage == 0.0

    def test_high_vix_dampens(self, engine):
        low_vix = engine.compute_target_leverage("CALM", 12.0, 0.0, 2.0)
        high_vix = engine.compute_target_leverage("CALM", 30.0, 0.0, 2.0)
        assert high_vix.target_leverage < low_vix.target_leverage

    def test_drawdown_dampens(self, engine):
        no_dd = engine.compute_target_leverage("CALM", 14.0, 0.0, 2.0)
        big_dd = engine.compute_target_leverage("CALM", 14.0, 0.06, 2.0)
        assert big_dd.target_leverage < no_dd.target_leverage

    def test_margin_dampens(self, engine):
        no_margin = engine.compute_target_leverage("CALM", 14.0, 0.0, 2.0, margin_used_pct=0.0)
        half_margin = engine.compute_target_leverage("CALM", 14.0, 0.0, 2.0, margin_used_pct=0.5)
        assert half_margin.target_leverage < no_margin.target_leverage

    def test_gross_notional_consistent(self, engine):
        result = engine.compute_target_leverage("NORMAL", 18.0, 0.01, 1.5)
        expected = result.target_leverage * engine.base_capital
        assert abs(result.target_gross_notional - expected) < 100.0

    def test_risk_budget_keys(self, engine):
        result = engine.compute_target_leverage("CALM", 14.0, 0.0, 2.0)
        expected_keys = {"sharpe_component", "regime_mult", "vix_mult", "dd_mult", "margin_mult"}
        assert set(result.risk_budget.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestComputePositionSizes:

    def test_basic_allocation(self, engine, simple_signals):
        positions = engine.compute_position_sizes(simple_signals, 2.0, 1_000_000)
        assert len(positions) == 4
        assert all(isinstance(p, PositionAllocation) for p in positions)

    def test_conviction_ordering(self, engine, simple_signals):
        """Higher conviction → larger allocation."""
        positions = engine.compute_position_sizes(simple_signals, 2.0, 1_000_000)
        notionals = [p.notional for p in positions]
        assert notionals[0] >= notionals[1] >= notionals[2] >= notionals[3]

    def test_single_name_cap(self, engine):
        """One dominant signal should be capped at 20%."""
        signals = [
            {"symbol": "XLK", "conviction": 10.0, "sector": "Technology"},
            {"symbol": "XLF", "conviction": 0.1, "sector": "Financials"},
        ]
        positions = engine.compute_position_sizes(signals, 2.0, 1_000_000)
        for p in positions:
            assert p.weight <= 0.20 + 1e-9

    def test_sector_concentration_cap(self, engine):
        """Same-sector signals should be capped at 40% total."""
        signals = [
            {"symbol": "AAPL", "conviction": 0.8, "sector": "Technology"},
            {"symbol": "MSFT", "conviction": 0.7, "sector": "Technology"},
            {"symbol": "GOOGL", "conviction": 0.6, "sector": "Technology"},
        ]
        positions = engine.compute_position_sizes(signals, 2.0, 1_000_000)
        tech_weight = sum(p.weight for p in positions if p.sector == "Technology")
        assert tech_weight <= 0.40 + 1e-9

    def test_empty_signals(self, engine):
        assert engine.compute_position_sizes([], 2.0, 1_000_000) == []

    def test_zero_leverage(self, engine, simple_signals):
        assert engine.compute_position_sizes(simple_signals, 0.0, 1_000_000) == []


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------

class TestRiskParity:

    def test_diagonal_cov_equal_risk(self, engine, cov_3x3):
        """With uncorrelated assets, lower-vol assets get more weight."""
        tickers = ["A", "B", "C"]
        weights = engine.risk_parity_weights(cov_3x3, tickers)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # A (lowest vol) should get the largest weight
        assert weights["A"] > weights["B"] > weights["C"]

    def test_equal_vol_equal_weight(self, engine):
        """With identical uncorrelated assets, weights should be equal."""
        cov = np.diag([0.04, 0.04, 0.04])
        tickers = ["X", "Y", "Z"]
        weights = engine.risk_parity_weights(cov, tickers)
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    def test_shape_mismatch_raises(self, engine):
        with pytest.raises(ValueError):
            engine.risk_parity_weights(np.eye(3), ["A", "B"])

    def test_empty_tickers(self, engine):
        assert engine.risk_parity_weights(np.array([[]]), []) == {}
