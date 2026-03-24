"""
tests/test_execution.py
=======================
Tests for the IBKR execution layer: IBKRGateway, SignalExecutor, OrderManager.

All tests run in DRY-RUN mode (no IBKR connection required).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from execution.ibkr_gateway import IBKRGateway, SignalExecutor, set_audit
from execution.order_manager import OrderManager

# Disable DuckDB-based audit trail to prevent segfaults under pytest
set_audit(None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gateway():
    """A dry-run IBKRGateway with default settings."""
    gw = IBKRGateway(paper=True)
    # Ensure dry-run regardless of ib_insync availability
    gw._dry_run = True
    gw._connected = False
    return gw


@pytest.fixture
def order_manager(tmp_path):
    """OrderManager using a temporary state file."""
    state_file = str(tmp_path / "test_order_state.json")
    return OrderManager(state_file=state_file)


@pytest.fixture
def signal_executor(gateway):
    """A SignalExecutor wired to a dry-run gateway."""
    return SignalExecutor(gateway=gateway)


@pytest.fixture
def sample_master_df():
    """Sample master_df with conviction signals."""
    return pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLE", "XLV", "XLB", "XLU"],
        "conviction_score": [0.45, 0.32, 0.28, 0.18, 0.10, 0.05],
        "direction": ["LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"],
    })


# =========================================================================
# IBKRGateway — Construction and Properties
# =========================================================================

class TestGatewayInit:
    def test_default_is_paper(self):
        gw = IBKRGateway()
        assert gw.paper is True
        assert gw.port == 7497

    def test_paper_override_bad_port(self):
        """If paper=True but port is live, it should be overridden to 7497."""
        gw = IBKRGateway(paper=True, port=7496)
        assert gw.port == 7497

    def test_live_port_allowed_when_not_paper(self):
        gw = IBKRGateway(paper=False, port=7496)
        assert gw.port == 7496

    def test_dry_run_by_default(self, gateway):
        assert gateway.is_dry_run is True
        assert gateway.is_connected is False

    def test_connect_returns_false_in_dry_run(self, gateway):
        result = gateway.connect()
        assert result is False


# =========================================================================
# IBKRGateway — Pre-trade Risk Checks
# =========================================================================

class TestPreTradeCheck:
    def test_approved_small_order(self, gateway):
        result = gateway.pre_trade_check("XLK", 10, "BUY", price_estimate=150.0)
        assert result["approved"] is True
        assert result["reason"] == "OK"

    def test_rejected_exceeds_single_name(self, gateway):
        """Order notional > 20% of $200K = $40K should be rejected."""
        result = gateway.pre_trade_check("XLK", 500, "BUY", price_estimate=150.0)
        # 500 * 150 = $75K > $40K limit
        assert result["approved"] is False
        assert "single-name" in result["reason"].lower()

    def test_rejected_exceeds_gross_exposure(self, gateway):
        """Total gross exceeding $200K should be rejected."""
        # First put a large position in the book
        gateway._dry_positions["SPY"] = {
            "quantity": 400, "avg_cost": 450.0,
            "market_value": 180_000.0, "unrealized_pnl": 0.0,
        }
        # Now try to add $30K more (total = $210K > $200K)
        result = gateway.pre_trade_check("XLK", 200, "BUY", price_estimate=150.0)
        assert result["approved"] is False
        assert "gross exposure" in result["reason"].lower()

    def test_default_price_placeholder(self, gateway):
        """When price_estimate=0, uses $100 placeholder."""
        result = gateway.pre_trade_check("XLK", 10, "BUY", price_estimate=0)
        assert result["approved"] is True

    def test_warnings_for_many_sectors(self, gateway):
        """Should warn when holding 4+ sector ETFs."""
        for etf in ["XLF", "XLE", "XLV", "XLI"]:
            gateway._dry_positions[etf] = {
                "quantity": 10, "avg_cost": 50.0,
                "market_value": 500.0, "unrealized_pnl": 0.0,
            }
        result = gateway.pre_trade_check("XLK", 10, "BUY", price_estimate=100.0)
        assert result["approved"] is True
        assert any("crowding" in w.lower() for w in result["warnings"])


# =========================================================================
# IBKRGateway — Order Placement (dry-run)
# =========================================================================

class TestOrderPlacement:
    def test_market_order_fills_in_dry_run(self, gateway):
        result = gateway.place_market_order("XLK", 50, "BUY")
        assert result["status"] == "FILLED"
        assert result["dry_run"] is True
        assert result["order_id"] is not None
        assert result["symbol"] == "XLK"
        assert result["side"] == "BUY"
        assert result["quantity"] == 50

    def test_market_order_updates_position(self, gateway):
        gateway.place_market_order("XLK", 100, "BUY")
        pos = gateway.get_positions()
        assert "XLK" in pos
        assert pos["XLK"]["quantity"] == 100

    def test_sell_reduces_position(self, gateway):
        gateway.place_market_order("XLK", 100, "BUY")
        gateway.place_market_order("XLK", 50, "SELL")
        pos = gateway.get_positions()
        assert pos["XLK"]["quantity"] == 50

    def test_sell_all_removes_position(self, gateway):
        gateway.place_market_order("XLK", 100, "BUY")
        gateway.place_market_order("XLK", 100, "SELL")
        pos = gateway.get_positions()
        assert "XLK" not in pos

    def test_limit_order_is_submitted(self, gateway):
        result = gateway.place_limit_order("XLF", 30, "BUY", limit_price=42.50)
        assert result["status"] == "SUBMITTED"
        assert result["order_type"] == "LMT"
        assert result["limit_price"] == 42.50

    def test_rejected_order_has_risk_info(self, gateway):
        """Order that exceeds risk limits should be REJECTED with reason."""
        result = gateway.place_market_order("XLK", 5000, "BUY")
        assert result["status"] == "REJECTED"
        assert result["risk_check"]["approved"] is False

    def test_invalid_side_raises(self, gateway):
        with pytest.raises(AssertionError):
            gateway.place_market_order("XLK", 10, "HOLD")

    def test_zero_quantity_raises(self, gateway):
        with pytest.raises(AssertionError):
            gateway.place_market_order("XLK", 0, "BUY")


# =========================================================================
# IBKRGateway — RV Trade
# =========================================================================

class TestRVTrade:
    def test_rv_trade_long_with_hedge(self, gateway):
        result = gateway.place_rv_trade(
            sector_etf="XLK", spy_hedge=True, direction="LONG",
            notional=10_000, hedge_ratio=0.85,
        )
        assert "trade_id" in result
        assert result["trade_id"].startswith("RV-XLK-")
        assert result["sector_leg"]["status"] == "FILLED"
        assert result["sector_leg"]["side"] == "BUY"
        assert result["hedge_leg"] is not None
        assert result["hedge_leg"]["side"] == "SELL"  # hedge opposite

    def test_rv_trade_short_no_hedge(self, gateway):
        result = gateway.place_rv_trade(
            sector_etf="XLE", spy_hedge=False, direction="SHORT",
            notional=5_000,
        )
        assert result["sector_leg"]["side"] == "SELL"
        assert result["hedge_leg"] is None


# =========================================================================
# IBKRGateway — Cancel and Open Orders
# =========================================================================

class TestCancelOrders:
    def test_cancel_existing_order(self, gateway):
        result = gateway.place_limit_order("XLK", 10, "BUY", limit_price=100.0)
        oid = result["order_id"]
        assert gateway.cancel_order(oid) is True
        assert gateway._dry_orders[oid]["status"] == "CANCELLED"

    def test_cancel_nonexistent_order(self, gateway):
        assert gateway.cancel_order(999999) is False

    def test_get_open_orders(self, gateway):
        gateway.place_limit_order("XLK", 10, "BUY", limit_price=100.0)
        gateway.place_limit_order("XLF", 20, "SELL", limit_price=50.0)
        open_orders = gateway.get_open_orders()
        assert len(open_orders) == 2


# =========================================================================
# IBKRGateway — Reconciliation
# =========================================================================

class TestReconciliation:
    def test_matching_portfolios(self, gateway):
        gateway._dry_positions = {
            "XLK": {"quantity": 100, "avg_cost": 150.0, "market_value": 15000.0,
                     "unrealized_pnl": 0.0},
        }
        paper = {"XLK": {"quantity": 100, "avg_cost": 150.0}}
        result = gateway.reconcile_with_paper(paper)
        assert result["match"] is True
        assert result["discrepancies"] == []

    def test_quantity_mismatch(self, gateway):
        gateway._dry_positions = {
            "XLK": {"quantity": 100, "avg_cost": 150.0, "market_value": 15000.0,
                     "unrealized_pnl": 0.0},
        }
        paper = {"XLK": {"quantity": 80, "avg_cost": 150.0}}
        result = gateway.reconcile_with_paper(paper)
        assert result["match"] is False
        assert len(result["discrepancies"]) == 1
        assert result["discrepancies"][0]["diff"] == 20

    def test_ibkr_only_position(self, gateway):
        gateway._dry_positions = {
            "XLK": {"quantity": 50, "avg_cost": 150.0, "market_value": 7500.0,
                     "unrealized_pnl": 0.0},
        }
        result = gateway.reconcile_with_paper({})
        assert result["match"] is False
        assert len(result["ibkr_only"]) == 1

    def test_paper_only_position(self, gateway):
        paper = {"XLF": {"quantity": 30, "avg_cost": 40.0}}
        result = gateway.reconcile_with_paper(paper)
        assert result["match"] is False
        assert len(result["paper_only"]) == 1


# =========================================================================
# SignalExecutor
# =========================================================================

class TestSignalExecutor:
    def test_execute_signals_basic(self, signal_executor, sample_master_df):
        results = signal_executor.execute_signals(sample_master_df, regime_state="RISK_ON")
        # threshold=0.15, so XLK(0.45), XLF(0.32), XLE(0.28), XLV(0.18) qualify
        # top_n=5 but only 4 above threshold
        assert len(results) == 4
        assert all("trade_id" in r for r in results)

    def test_execute_signals_risk_off_scales_down(self, signal_executor, sample_master_df):
        results = signal_executor.execute_signals(sample_master_df, regime_state="RISK_OFF")
        assert len(results) > 0  # still executes, just smaller size

    def test_execute_signals_empty_df(self, signal_executor):
        results = signal_executor.execute_signals(pd.DataFrame(), regime_state="NEUTRAL")
        assert results == []

    def test_execute_signals_none_df(self, signal_executor):
        results = signal_executor.execute_signals(None, regime_state="NEUTRAL")
        assert results == []

    def test_execute_signals_missing_columns(self, signal_executor):
        bad_df = pd.DataFrame({"foo": [1, 2]})
        results = signal_executor.execute_signals(bad_df, regime_state="NEUTRAL")
        assert results == []

    def test_skips_existing_positions(self, signal_executor, sample_master_df):
        """Should skip tickers already in the portfolio."""
        signal_executor.gw._dry_positions["XLK"] = {
            "quantity": 50, "avg_cost": 150.0,
            "market_value": 7500.0, "unrealized_pnl": 0.0,
        }
        results = signal_executor.execute_signals(sample_master_df, regime_state="RISK_ON")
        tickers_traded = [
            r["sector_leg"]["symbol"] for r in results
        ]
        assert "XLK" not in tickers_traded

    def test_regime_scaling(self):
        assert SignalExecutor._regime_scaling("RISK_ON") == 1.0
        assert SignalExecutor._regime_scaling("RISK_OFF") == 0.40
        assert SignalExecutor._regime_scaling("CRISIS") == 0.25
        assert SignalExecutor._regime_scaling("NEUTRAL") == 0.70
        assert SignalExecutor._regime_scaling("SOMETHING_ELSE") == 0.60

    def test_execute_exits(self, signal_executor):
        # Set up a position to exit
        signal_executor.gw.place_market_order("XLK", 100, "BUY")

        exit_signals = [
            {"ticker": "XLK", "action": "EXIT", "reason": "stop_loss"},
        ]
        results = signal_executor.execute_exits(exit_signals)
        assert len(results) == 1
        assert results[0]["side"] == "SELL"
        assert results[0]["quantity"] == 100

    def test_execute_exits_reduce(self, signal_executor):
        signal_executor.gw.place_market_order("XLF", 200, "BUY")

        exit_signals = [
            {"ticker": "XLF", "action": "REDUCE", "reason": "take_profit",
             "reduce_pct": 0.5},
        ]
        results = signal_executor.execute_exits(exit_signals)
        assert len(results) == 1
        assert results[0]["quantity"] == 100

    def test_execute_exits_no_position(self, signal_executor):
        exit_signals = [
            {"ticker": "FAKE", "action": "EXIT", "reason": "test"},
        ]
        results = signal_executor.execute_exits(exit_signals)
        assert results == []

    def test_execute_exits_empty(self, signal_executor):
        assert signal_executor.execute_exits([]) == []


# =========================================================================
# OrderManager
# =========================================================================

class TestOrderManager:
    def test_log_order(self, order_manager):
        record = order_manager.log_order(
            order_id=1001, symbol="XLK", side="BUY", qty=50,
            price=155.0, status="FILLED",
        )
        assert record["order_id"] == "1001"
        assert record["symbol"] == "XLK"
        assert record["status"] == "FILLED"

    def test_persistence(self, tmp_path):
        state_file = str(tmp_path / "persist_test.json")
        om1 = OrderManager(state_file=state_file)
        om1.log_order(1, "XLK", "BUY", 10, 100.0, "FILLED")

        # New instance should load persisted data
        om2 = OrderManager(state_file=state_file)
        assert om2.get_order(1) is not None
        assert om2.get_order(1)["symbol"] == "XLK"

    def test_update_fill(self, order_manager):
        order_manager.log_order(2001, "XLF", "BUY", 100, 0.0, "SUBMITTED")
        record = order_manager.update_fill(2001, fill_price=42.0, fill_qty=50)
        assert record["status"] == "PARTIAL"

        record = order_manager.update_fill(2001, fill_price=42.10, fill_qty=50)
        assert record["status"] == "FILLED"
        # Weighted average: (42.0*50 + 42.10*50) / 100 = 42.05
        assert abs(record["price"] - 42.05) < 0.01

    def test_update_fill_not_found(self, order_manager):
        result = order_manager.update_fill(99999, 100.0, 10)
        assert result is None

    def test_update_status(self, order_manager):
        order_manager.log_order(3001, "XLE", "SELL", 20, 80.0, "SUBMITTED")
        record = order_manager.update_status(3001, "CANCELLED")
        assert record["status"] == "CANCELLED"

    def test_get_history(self, order_manager):
        for i in range(10):
            order_manager.log_order(
                4000 + i, f"SYM{i}", "BUY", 10, 100.0, "FILLED",
            )
        history = order_manager.get_history(last_n=5)
        assert len(history) == 5

    def test_get_open_orders(self, order_manager):
        order_manager.log_order(5001, "XLK", "BUY", 10, 100.0, "SUBMITTED")
        order_manager.log_order(5002, "XLF", "SELL", 20, 50.0, "FILLED")
        open_orders = order_manager.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0]["order_id"] == "5001"

    def test_daily_summary(self, order_manager):
        order_manager.log_order(6001, "XLK", "BUY", 100, 150.0, "FILLED")
        order_manager.log_order(6002, "XLF", "SELL", 50, 40.0, "FILLED")
        order_manager.log_order(6003, "XLE", "BUY", 10, 80.0, "REJECTED")

        summary = order_manager.daily_summary()
        assert summary["total_orders"] == 3
        assert summary["filled"] == 2
        assert summary["rejected"] == 1
        assert summary["buy_notional"] == 100 * 150.0
        assert summary["sell_notional"] == 50 * 40.0
        assert "XLK" in summary["symbols_traded"]

    def test_clear_state(self, order_manager):
        order_manager.log_order(7001, "XLK", "BUY", 10, 100.0, "FILLED")
        assert len(order_manager.get_history()) == 1
        order_manager.clear_state()
        assert len(order_manager.get_history()) == 0
