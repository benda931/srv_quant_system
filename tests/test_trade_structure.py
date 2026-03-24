"""tests/test_trade_structure.py — Trade construction, sizing, and constraint tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.trade_structure import (
    TradeStructureEngine,
    TradeTicket,
    TradeLeg,
    GreeksProfile,
    ExitConditions,
)
from analytics.signal_stack import SignalStackResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> TradeStructureEngine:
    """Trade structure engine with default settings."""
    return TradeStructureEngine(settings=None)


@pytest.fixture(scope="module")
def long_signal() -> SignalStackResult:
    """A mock long signal result."""
    return SignalStackResult(
        ticker="XLK",
        trade_type="sector",
        distortion_score=0.65,
        dislocation_score=0.70,
        mean_reversion_score=0.80,
        regime_safety_score=0.90,
        conviction_score=0.65 * 0.70 * 0.80 * 0.90,
        entry_threshold=0.05,
        passes_entry=True,
        direction="LONG",
        residual_z=-1.8,
        size_multiplier=0.85,
    )


@pytest.fixture(scope="module")
def short_signal() -> SignalStackResult:
    """A mock short signal result."""
    return SignalStackResult(
        ticker="XLE",
        trade_type="sector",
        distortion_score=0.55,
        dislocation_score=0.60,
        mean_reversion_score=0.75,
        regime_safety_score=0.95,
        conviction_score=0.55 * 0.60 * 0.75 * 0.95,
        entry_threshold=0.05,
        passes_entry=True,
        direction="SHORT",
        residual_z=1.5,
        size_multiplier=0.90,
    )


# ---------------------------------------------------------------------------
# Sector RV Construction
# ---------------------------------------------------------------------------

class TestSectorRVConstruction:
    """Tests for construct_sector_rv."""

    def test_sector_rv_construction(self, engine, long_signal):
        """Sector RV trade should produce a 2-leg structure."""
        ticket = engine.construct_sector_rv(
            long_signal,
            beta_spy=1.1,
            ewma_vol=0.22,
            half_life=20.0,
        )
        assert isinstance(ticket, TradeTicket)
        assert ticket.trade_type == "sector_rv"
        assert ticket.ticker == "XLK"
        assert ticket.direction == "LONG"
        assert ticket.n_legs == 2

    def test_sector_rv_legs(self, engine, long_signal):
        """Long signal: leg 1 BUY sector, leg 2 SELL SPY."""
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=1.0, ewma_vol=0.20,
        )
        legs = ticket.legs
        assert len(legs) == 2
        assert legs[0].instrument == "XLK"
        assert legs[0].direction == "BUY"
        assert legs[1].instrument == "SPY"
        assert legs[1].direction == "SELL"

    def test_sector_rv_short(self, engine, short_signal):
        """Short signal: leg 1 SELL sector, leg 2 BUY SPY."""
        ticket = engine.construct_sector_rv(
            short_signal, beta_spy=0.9, ewma_vol=0.25,
        )
        assert ticket.direction == "SHORT"
        assert ticket.legs[0].direction == "SELL"
        assert ticket.legs[1].direction == "BUY"


# ---------------------------------------------------------------------------
# Position Sizing Limits
# ---------------------------------------------------------------------------

class TestPositionSizingLimits:
    """Tests for position sizing constraints."""

    def test_position_sizing_limits(self, engine, long_signal):
        """Final weight must not exceed max_single_trade_weight."""
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=1.0, ewma_vol=0.20,
        )
        assert ticket.final_weight <= engine.max_single_trade_weight
        assert ticket.final_weight >= 0.0

    def test_sizing_regime_adjustment(self, engine):
        """Lower size_multiplier should reduce final_weight."""
        sig_high = SignalStackResult(
            ticker="XLF", trade_type="sector",
            distortion_score=0.8, dislocation_score=0.8,
            mean_reversion_score=0.8, regime_safety_score=0.95,
            conviction_score=0.8 ** 3 * 0.95,
            entry_threshold=0.05, passes_entry=True,
            direction="LONG", residual_z=-2.0,
            size_multiplier=1.0,
        )
        sig_low = SignalStackResult(
            ticker="XLF", trade_type="sector",
            distortion_score=0.8, dislocation_score=0.8,
            mean_reversion_score=0.8, regime_safety_score=0.95,
            conviction_score=0.8 ** 3 * 0.95,
            entry_threshold=0.05, passes_entry=True,
            direction="LONG", residual_z=-2.0,
            size_multiplier=0.3,
        )
        t_high = engine.construct_sector_rv(sig_high, beta_spy=1.0, ewma_vol=0.20)
        t_low = engine.construct_sector_rv(sig_low, beta_spy=1.0, ewma_vol=0.20)
        assert t_low.final_weight < t_high.final_weight

    def test_weight_non_negative(self, engine, long_signal):
        """No position weight should ever be negative."""
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=1.0, ewma_vol=0.20,
        )
        assert ticket.raw_weight >= 0.0
        assert ticket.final_weight >= 0.0
        for leg in ticket.legs:
            assert leg.notional_weight >= 0.0


# ---------------------------------------------------------------------------
# Net Exposure Constraint
# ---------------------------------------------------------------------------

class TestNetExposureConstraint:
    """Tests for net exposure properties of constructed trades."""

    def test_net_exposure_constraint(self, engine, long_signal):
        """
        Sector RV is designed to be approximately beta-neutral.
        The SPY hedge leg should approximately offset the sector leg
        (scaled by beta).
        """
        beta = 1.1
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=beta, ewma_vol=0.20,
        )
        sector_leg = ticket.legs[0]
        spy_leg = ticket.legs[1]
        # Net dollar exposure ≈ sector_weight - spy_weight
        net_dollar = sector_leg.notional_weight - spy_leg.notional_weight
        # For beta > 1, SPY leg > sector leg, so net should be negative
        # The key invariant: |net| < sector_weight (we are hedged, not naked)
        assert abs(net_dollar) < sector_leg.notional_weight + 0.01

    def test_greeks_near_zero_delta(self, engine, long_signal):
        """Beta-hedged trade should have small residual SPY delta."""
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=1.0, ewma_vol=0.20,
        )
        # With beta=1.0, delta_spy should be very close to zero
        # (the hedge perfectly offsets)
        assert abs(ticket.greeks.delta_spy) < 0.15

    def test_exit_conditions_present(self, engine, long_signal):
        """Every trade ticket must have valid exit conditions."""
        ticket = engine.construct_sector_rv(
            long_signal, beta_spy=1.0, ewma_vol=0.20, half_life=15.0,
        )
        ec = ticket.exit_conditions
        assert isinstance(ec, ExitConditions)
        assert ec.max_holding_days > 0
        assert ec.max_loss_pct > 0
        assert ec.profit_target_pct > 0
