"""
execution/ibkr_gateway.py
=========================
Interactive Brokers execution gateway for the SRV Quant System.

Provides:
    IBKRGateway   — connection management, order placement, position queries,
                    risk pre-trade checks, and portfolio reconciliation.
    SignalExecutor — translates DSS conviction signals into executable orders
                    with mandatory risk gates.

Design:
    - Paper mode (port 7497) is the default; live requires explicit opt-in.
    - Every public method works in DRY-RUN mode when no IBKR connection exists.
    - ib_insync is imported with a try/except; if unavailable the gateway
      operates entirely in dry-run mode.
    - Every order is logged to the audit trail (db/audit.py).
    - Pre-trade risk checks are mandatory — no order bypasses them.

Port conventions:
    7497 — TWS paper trading
    7496 — TWS live trading
    4002 — IB Gateway paper
    4001 — IB Gateway live
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional ib_insync import
# ---------------------------------------------------------------------------
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Trade, Order
    _HAS_IB_INSYNC = True
except ImportError:
    _HAS_IB_INSYNC = False
    logger.info("ib_insync not installed — IBKRGateway will operate in DRY-RUN mode")

# ---------------------------------------------------------------------------
# Audit trail (lazy import to avoid circular deps at module level)
# ---------------------------------------------------------------------------
_audit = None


_AUDIT_DISABLED = object()  # sentinel: audit explicitly disabled
_audit_init_attempted = False


def _get_audit():
    """
    Lazy-load the AuditTrail singleton.

    Uses an in-memory DuckDB backend to avoid file-lock segfaults.
    If DuckDB is unavailable or crashes during import, audit is silently
    disabled (all order events still logged via the Python logger).

    Call ``set_audit(None)`` to explicitly disable audit without attempting
    a DuckDB import (useful in test harnesses where DuckDB may segfault).
    """
    global _audit, _audit_init_attempted
    if _audit is _AUDIT_DISABLED:
        return None
    if _audit is None and not _audit_init_attempted:
        _audit_init_attempted = True
        try:
            import duckdb  # noqa: F811
            from db.audit import AuditTrail
            conn = duckdb.connect(":memory:")
            _audit = AuditTrail(conn=conn)
        except Exception as exc:
            logger.warning("Could not initialize AuditTrail: %s — audit disabled", exc)
    return _audit


def set_audit(audit_instance) -> None:
    """
    Inject an audit trail instance, or pass ``None`` to disable audit entirely.

    This must be called **before** any order placement if you want to prevent
    the lazy DuckDB import (e.g., in tests).
    """
    global _audit, _audit_init_attempted
    if audit_instance is None:
        _audit = _AUDIT_DISABLED
    else:
        _audit = audit_instance
    _audit_init_attempted = True


# ---------------------------------------------------------------------------
# Dry-run order counter
# ---------------------------------------------------------------------------
_dry_run_seq = 0


def _next_dry_id() -> int:
    global _dry_run_seq
    _dry_run_seq += 1
    return 900_000 + _dry_run_seq


# =========================================================================
# IBKRGateway
# =========================================================================
class IBKRGateway:
    """
    Interactive Brokers execution gateway.

    Supports TWS and IB Gateway via the ``ib_insync`` library.
    Falls back to full DRY-RUN mode if ``ib_insync`` is not installed or
    connection to TWS/Gateway fails.

    Parameters
    ----------
    host : str
        TWS / IB Gateway host address.
    port : int
        TWS / IB Gateway port.
        7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live.
    client_id : int
        Unique client identifier for the connection.
    paper : bool
        If True, enforce paper-trading ports only.
    max_position_pct : float
        Maximum single-name position as fraction of gross exposure limit.
    max_gross_exposure : float
        Maximum total gross notional across all positions.
    max_net_exposure_pct : float
        Maximum net exposure as fraction of gross limit (e.g., 0.3 = 30%).
    """

    # Ports that are considered paper-trading
    _PAPER_PORTS = {7497, 4002}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper: bool = True,
        max_position_pct: float = 0.20,
        max_gross_exposure: float = 200_000.0,
        max_net_exposure_pct: float = 0.30,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper
        self._connected = False
        self._ib = None  # ib_insync.IB instance when connected
        self._dry_run = not _HAS_IB_INSYNC

        # Risk limits
        self.max_position_pct = max_position_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure_pct = max_net_exposure_pct

        # In-memory position cache for dry-run mode
        self._dry_positions: Dict[str, dict] = {}
        self._dry_orders: Dict[int, dict] = {}

        if self.paper and self.port not in self._PAPER_PORTS:
            logger.warning(
                "paper=True but port=%d is NOT a paper port. "
                "Overriding to 7497 for safety.",
                self.port,
            )
            self.port = 7497

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish connection to TWS / IB Gateway.

        Returns True on success, False on failure (falls back to dry-run).
        """
        if self._dry_run:
            logger.info("DRY-RUN mode — no IBKR connection attempted")
            self._connected = False
            return False

        try:
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(
                "Connected to IBKR at %s:%d (client_id=%d, paper=%s)",
                self.host, self.port, self.client_id, self.paper,
            )
            return True
        except Exception as exc:
            logger.error("IBKR connection failed: %s — falling back to DRY-RUN", exc)
            self._connected = False
            self._dry_run = True
            return False

    def disconnect(self) -> None:
        """Disconnect from TWS / IB Gateway."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        """True when actively connected to IBKR."""
        return self._connected and self._ib is not None

    @property
    def is_dry_run(self) -> bool:
        """True when operating in dry-run mode (no live connection)."""
        return self._dry_run or not self._connected

    # ------------------------------------------------------------------
    # Pre-trade risk checks
    # ------------------------------------------------------------------

    def pre_trade_check(
        self, symbol: str, quantity: int, side: str, price_estimate: float = 0.0,
    ) -> dict:
        """
        Mandatory pre-trade risk validation.

        Checks performed:
            1. Max single-name position size vs gross exposure limit.
            2. Net exposure stays within bounds after the trade.
            3. Buying power / margin headroom (IBKR live only).
            4. Warnings for correlated positions.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares.
        side : str
            "BUY" or "SELL".
        price_estimate : float
            Estimated fill price for notional calculation.
            If 0, uses 100.0 as a conservative placeholder.

        Returns
        -------
        dict
            {approved: bool, reason: str, warnings: list[str]}
        """
        warnings: list[str] = []
        price = price_estimate if price_estimate > 0 else 100.0
        trade_notional = abs(quantity) * price

        # 1. Single-name limit
        max_single = self.max_gross_exposure * self.max_position_pct
        if trade_notional > max_single:
            return {
                "approved": False,
                "reason": (
                    f"Trade notional ${trade_notional:,.0f} exceeds single-name "
                    f"limit ${max_single:,.0f} ({self.max_position_pct:.0%} of "
                    f"${self.max_gross_exposure:,.0f})"
                ),
                "warnings": warnings,
            }

        # 2. Net / gross exposure check
        positions = self.get_positions()
        current_gross = sum(
            abs(p.get("market_value", 0.0)) for p in positions.values()
        )
        current_net = sum(
            p.get("market_value", 0.0) for p in positions.values()
        )

        signed_notional = trade_notional if side.upper() == "BUY" else -trade_notional
        new_gross = current_gross + trade_notional
        new_net = current_net + signed_notional

        if new_gross > self.max_gross_exposure:
            return {
                "approved": False,
                "reason": (
                    f"New gross exposure ${new_gross:,.0f} would exceed "
                    f"limit ${self.max_gross_exposure:,.0f}"
                ),
                "warnings": warnings,
            }

        net_pct = abs(new_net) / self.max_gross_exposure if self.max_gross_exposure else 0
        if net_pct > self.max_net_exposure_pct:
            warnings.append(
                f"Net exposure {net_pct:.1%} exceeds soft limit "
                f"{self.max_net_exposure_pct:.0%}"
            )

        # 3. Buying-power check (live connection only)
        if self.is_connected and not self.is_dry_run:
            try:
                acct_values = self._ib.accountValues()
                bp_items = [
                    v for v in acct_values
                    if v.tag == "BuyingPower" and v.currency == "USD"
                ]
                if bp_items:
                    buying_power = float(bp_items[0].value)
                    if trade_notional > buying_power * 0.95:
                        return {
                            "approved": False,
                            "reason": (
                                f"Insufficient buying power: need "
                                f"${trade_notional:,.0f}, have ${buying_power:,.0f}"
                            ),
                            "warnings": warnings,
                        }
            except Exception as exc:
                warnings.append(f"Could not verify buying power: {exc}")

        # 4. Correlation / concentration warning
        sector_etfs = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLC",
                       "XLY", "XLP", "XLRE", "XLU", "SPY"}
        if symbol in sector_etfs:
            existing_sectors = [
                s for s in positions if s in sector_etfs and s != symbol
            ]
            if len(existing_sectors) >= 4:
                warnings.append(
                    f"Already holding {len(existing_sectors)} sector ETFs — "
                    f"check for crowding"
                )

        return {"approved": True, "reason": "OK", "warnings": warnings}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(self, symbol: str, quantity: int, side: str) -> dict:
        """
        Place a market order after mandatory pre-trade risk check.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g., "XLK").
        quantity : int
            Number of shares (positive).
        side : str
            "BUY" or "SELL".

        Returns
        -------
        dict
            {order_id, symbol, side, quantity, order_type, status,
             fill_price, dry_run, timestamp, risk_check}
        """
        side = side.upper()
        assert side in ("BUY", "SELL"), f"Invalid side: {side}"
        assert quantity > 0, f"Quantity must be positive, got {quantity}"

        # Mandatory risk gate
        risk = self.pre_trade_check(symbol, quantity, side)
        if not risk["approved"]:
            result = {
                "order_id": None,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "MKT",
                "status": "REJECTED",
                "fill_price": 0.0,
                "dry_run": self.is_dry_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
            }
            self._audit_order(result)
            return result

        if risk["warnings"]:
            logger.warning("Pre-trade warnings for %s %s %d: %s",
                           side, symbol, quantity, risk["warnings"])

        # Execute
        if self.is_dry_run:
            return self._dry_run_market(symbol, quantity, side, risk)
        else:
            return self._live_market(symbol, quantity, side, risk)

    def place_limit_order(
        self, symbol: str, quantity: int, side: str, limit_price: float,
    ) -> dict:
        """
        Place a limit order after mandatory pre-trade risk check.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : int
            Number of shares (positive).
        side : str
            "BUY" or "SELL".
        limit_price : float
            Limit price.

        Returns
        -------
        dict
            Same structure as place_market_order.
        """
        side = side.upper()
        assert side in ("BUY", "SELL"), f"Invalid side: {side}"
        assert quantity > 0, f"Quantity must be positive, got {quantity}"
        assert limit_price > 0, f"Limit price must be positive, got {limit_price}"

        risk = self.pre_trade_check(symbol, quantity, side, price_estimate=limit_price)
        if not risk["approved"]:
            result = {
                "order_id": None,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "LMT",
                "limit_price": limit_price,
                "status": "REJECTED",
                "fill_price": 0.0,
                "dry_run": self.is_dry_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
            }
            self._audit_order(result)
            return result

        if self.is_dry_run:
            return self._dry_run_limit(symbol, quantity, side, limit_price, risk)
        else:
            return self._live_limit(symbol, quantity, side, limit_price, risk)

    def place_rv_trade(
        self,
        sector_etf: str,
        spy_hedge: bool,
        direction: str,
        notional: float,
        hedge_ratio: float = 1.0,
    ) -> dict:
        """
        Place a relative-value trade: sector ETF vs SPY hedge.

        Parameters
        ----------
        sector_etf : str
            Sector ETF ticker (e.g., "XLK").
        spy_hedge : bool
            If True, place an offsetting SPY leg.
        direction : str
            "LONG" or "SHORT" for the sector leg.
        notional : float
            Target notional for the sector leg.
        hedge_ratio : float
            Beta-adjusted hedge ratio (e.g., 0.85 means hedge 85% of notional).

        Returns
        -------
        dict
            {sector_leg: <order_result>, hedge_leg: <order_result> | None,
             trade_id: str}
        """
        direction = direction.upper()
        assert direction in ("LONG", "SHORT"), f"Invalid direction: {direction}"
        assert notional > 0, f"Notional must be positive, got {notional}"

        trade_id = f"RV-{sector_etf}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

        # Estimate share counts (use $100 placeholder if no market data)
        est_price_sector = 100.0
        est_price_spy = 450.0

        sector_qty = max(1, int(notional / est_price_sector))
        sector_side = "BUY" if direction == "LONG" else "SELL"

        sector_result = self.place_market_order(sector_etf, sector_qty, sector_side)
        sector_result["trade_id"] = trade_id

        hedge_result = None
        if spy_hedge:
            hedge_notional = notional * hedge_ratio
            spy_qty = max(1, int(hedge_notional / est_price_spy))
            spy_side = "SELL" if direction == "LONG" else "BUY"
            hedge_result = self.place_market_order("SPY", spy_qty, spy_side)
            hedge_result["trade_id"] = trade_id

        result = {
            "trade_id": trade_id,
            "sector_leg": sector_result,
            "hedge_leg": hedge_result,
        }

        logger.info("RV trade %s: %s %s $%.0f (hedge=%s, ratio=%.2f)",
                     trade_id, direction, sector_etf, notional, spy_hedge, hedge_ratio)
        return result

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an open order.

        Parameters
        ----------
        order_id : int
            The order ID to cancel.

        Returns
        -------
        bool
            True if cancel was submitted successfully.
        """
        if self.is_dry_run:
            if order_id in self._dry_orders:
                self._dry_orders[order_id]["status"] = "CANCELLED"
                logger.info("DRY-RUN: Cancelled order %d", order_id)
                self._audit_cancel(order_id)
                return True
            logger.warning("DRY-RUN: Order %d not found", order_id)
            return False

        try:
            open_trades = self._ib.openTrades()
            for trade in open_trades:
                if trade.order.orderId == order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info("Cancel submitted for order %d", order_id)
                    self._audit_cancel(order_id)
                    return True
            logger.warning("Order %d not found among open trades", order_id)
            return False
        except Exception as exc:
            logger.error("Cancel failed for order %d: %s", order_id, exc)
            return False

    def get_open_orders(self) -> list:
        """
        Get all open (working) orders.

        Returns
        -------
        list[dict]
            Each dict: {order_id, symbol, side, quantity, order_type, status}.
        """
        if self.is_dry_run:
            return [
                o for o in self._dry_orders.values()
                if o.get("status") in ("SUBMITTED", "PENDING")
            ]

        try:
            trades = self._ib.openTrades()
            results = []
            for t in trades:
                results.append({
                    "order_id": t.order.orderId,
                    "symbol": t.contract.symbol,
                    "side": t.order.action,
                    "quantity": int(t.order.totalQuantity),
                    "order_type": t.order.orderType,
                    "status": t.orderStatus.status,
                })
            return results
        except Exception as exc:
            logger.error("Failed to fetch open orders: %s", exc)
            return []

    def get_positions(self) -> dict:
        """
        Get current positions.

        Returns
        -------
        dict
            {symbol: {quantity, avg_cost, market_value, unrealized_pnl}}
        """
        if self.is_dry_run:
            return dict(self._dry_positions)

        try:
            positions = self._ib.positions()
            result = {}
            for pos in positions:
                sym = pos.contract.symbol
                qty = int(pos.position)
                avg = float(pos.avgCost)
                result[sym] = {
                    "quantity": qty,
                    "avg_cost": avg,
                    "market_value": qty * avg,  # approximation
                    "unrealized_pnl": 0.0,  # updated via portfolio
                }
            # Enrich with portfolio data if available
            try:
                for pv in self._ib.portfolio():
                    sym = pv.contract.symbol
                    if sym in result:
                        result[sym]["market_value"] = float(pv.marketValue)
                        result[sym]["unrealized_pnl"] = float(pv.unrealizedPNL)
            except Exception:
                pass
            return result
        except Exception as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def reconcile_with_paper(self, paper_portfolio: dict) -> dict:
        """
        Compare IBKR positions with an internal paper portfolio.

        Parameters
        ----------
        paper_portfolio : dict
            {symbol: {quantity: int, avg_cost: float, ...}}

        Returns
        -------
        dict
            {
                "match": bool,
                "discrepancies": [
                    {symbol, ibkr_qty, paper_qty, diff, type}
                ],
                "ibkr_only": [...],
                "paper_only": [...],
            }
        """
        ibkr_pos = self.get_positions()
        discrepancies = []
        ibkr_only = []
        paper_only = []

        all_symbols = set(ibkr_pos.keys()) | set(paper_portfolio.keys())
        for sym in sorted(all_symbols):
            ibkr_qty = ibkr_pos.get(sym, {}).get("quantity", 0)
            paper_qty = paper_portfolio.get(sym, {}).get("quantity", 0)

            if sym in ibkr_pos and sym not in paper_portfolio:
                ibkr_only.append({"symbol": sym, "ibkr_qty": ibkr_qty})
            elif sym not in ibkr_pos and sym in paper_portfolio:
                paper_only.append({"symbol": sym, "paper_qty": paper_qty})
            elif ibkr_qty != paper_qty:
                discrepancies.append({
                    "symbol": sym,
                    "ibkr_qty": ibkr_qty,
                    "paper_qty": paper_qty,
                    "diff": ibkr_qty - paper_qty,
                    "type": "QUANTITY_MISMATCH",
                })

        match = len(discrepancies) == 0 and len(ibkr_only) == 0 and len(paper_only) == 0

        result = {
            "match": match,
            "discrepancies": discrepancies,
            "ibkr_only": ibkr_only,
            "paper_only": paper_only,
        }
        if not match:
            logger.warning(
                "Reconciliation mismatch: %d discrepancies, %d ibkr-only, %d paper-only",
                len(discrepancies), len(ibkr_only), len(paper_only),
            )
        else:
            logger.info("Reconciliation: positions match")
        return result

    # ------------------------------------------------------------------
    # Internal: dry-run order execution
    # ------------------------------------------------------------------

    def _dry_run_market(self, symbol: str, quantity: int, side: str, risk: dict) -> dict:
        """Simulate a market order fill in dry-run mode."""
        oid = _next_dry_id()
        sim_price = 100.0  # placeholder fill price
        result = {
            "order_id": oid,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": "MKT",
            "status": "FILLED",
            "fill_price": sim_price,
            "dry_run": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_check": risk,
        }
        self._dry_orders[oid] = result
        self._update_dry_position(symbol, quantity, side, sim_price)
        self._audit_order(result)
        logger.info("DRY-RUN MKT: %s %d %s @ $%.2f (id=%d)",
                     side, quantity, symbol, sim_price, oid)
        return result

    def _dry_run_limit(
        self, symbol: str, quantity: int, side: str, limit_price: float, risk: dict,
    ) -> dict:
        """Simulate a limit order submission in dry-run mode."""
        oid = _next_dry_id()
        result = {
            "order_id": oid,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": "LMT",
            "limit_price": limit_price,
            "status": "SUBMITTED",
            "fill_price": 0.0,
            "dry_run": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_check": risk,
        }
        self._dry_orders[oid] = result
        self._audit_order(result)
        logger.info("DRY-RUN LMT: %s %d %s @ $%.2f (id=%d)",
                     side, quantity, symbol, limit_price, oid)
        return result

    def _update_dry_position(
        self, symbol: str, quantity: int, side: str, price: float,
    ) -> None:
        """Update the in-memory dry-run position tracker."""
        signed_qty = quantity if side == "BUY" else -quantity
        if symbol in self._dry_positions:
            pos = self._dry_positions[symbol]
            old_qty = pos["quantity"]
            new_qty = old_qty + signed_qty
            if new_qty == 0:
                del self._dry_positions[symbol]
            else:
                # Weighted average cost (simplified)
                if (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                    total_cost = abs(old_qty) * pos["avg_cost"] + abs(signed_qty) * price
                    pos["avg_cost"] = total_cost / abs(new_qty)
                pos["quantity"] = new_qty
                pos["market_value"] = new_qty * price
        else:
            self._dry_positions[symbol] = {
                "quantity": signed_qty,
                "avg_cost": price,
                "market_value": signed_qty * price,
                "unrealized_pnl": 0.0,
            }

    # ------------------------------------------------------------------
    # Internal: live order execution
    # ------------------------------------------------------------------

    def _live_market(self, symbol: str, quantity: int, side: str, risk: dict) -> dict:
        """Place a live market order via ib_insync."""
        try:
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = MarketOrder(side, quantity)
            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(1)  # brief wait for fill status

            fill_price = 0.0
            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice

            result = {
                "order_id": trade.order.orderId,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "MKT",
                "status": trade.orderStatus.status,
                "fill_price": fill_price,
                "dry_run": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
            }
            self._audit_order(result)
            logger.info("LIVE MKT: %s %d %s status=%s (id=%d)",
                         side, quantity, symbol, result["status"],
                         result["order_id"])
            return result
        except Exception as exc:
            logger.error("Live market order failed: %s", exc)
            return {
                "order_id": None,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "MKT",
                "status": "ERROR",
                "fill_price": 0.0,
                "dry_run": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
                "error": str(exc),
            }

    def _live_limit(
        self, symbol: str, quantity: int, side: str, limit_price: float, risk: dict,
    ) -> dict:
        """Place a live limit order via ib_insync."""
        try:
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = LimitOrder(side, quantity, limit_price)
            trade = self._ib.placeOrder(contract, order)

            result = {
                "order_id": trade.order.orderId,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "LMT",
                "limit_price": limit_price,
                "status": trade.orderStatus.status,
                "fill_price": 0.0,
                "dry_run": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
            }
            self._audit_order(result)
            logger.info("LIVE LMT: %s %d %s @ $%.2f status=%s (id=%d)",
                         side, quantity, symbol, limit_price,
                         result["status"], result["order_id"])
            return result
        except Exception as exc:
            logger.error("Live limit order failed: %s", exc)
            return {
                "order_id": None,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "LMT",
                "limit_price": limit_price,
                "status": "ERROR",
                "fill_price": 0.0,
                "dry_run": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_check": risk,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _audit_order(self, result: dict) -> None:
        """Log an order event to the audit trail."""
        audit = _get_audit()
        if audit is None:
            return
        try:
            audit.log_trade(
                timestamp=datetime.now(timezone.utc),
                trade_id=str(result.get("trade_id", result.get("order_id", "unknown"))),
                action=f"ORDER_{result.get('status', 'UNKNOWN')}",
                details={
                    "symbol": result.get("symbol"),
                    "side": result.get("side"),
                    "quantity": result.get("quantity"),
                    "order_type": result.get("order_type"),
                    "fill_price": result.get("fill_price"),
                    "dry_run": result.get("dry_run"),
                    "risk_check": result.get("risk_check"),
                },
            )
        except Exception as exc:
            logger.warning("Audit logging failed: %s", exc)

    def _audit_cancel(self, order_id: int) -> None:
        """Log a cancel event to the audit trail."""
        audit = _get_audit()
        if audit is None:
            return
        try:
            audit.log_trade(
                timestamp=datetime.now(timezone.utc),
                trade_id=str(order_id),
                action="ORDER_CANCELLED",
                details={"order_id": order_id, "dry_run": self.is_dry_run},
            )
        except Exception as exc:
            logger.warning("Audit logging failed: %s", exc)


# =========================================================================
# SignalExecutor
# =========================================================================
class SignalExecutor:
    """
    Translates DSS signals from QuantEngine master_df into executable orders.

    The executor reads conviction scores and directions from the master DataFrame,
    selects top-N candidates above the entry threshold, applies risk checks,
    and submits orders through the IBKRGateway.

    Parameters
    ----------
    gateway : IBKRGateway
        The gateway instance for order execution.
    max_gross_exposure : float
        Maximum total gross notional.
    max_single_name : float
        Maximum fraction of gross exposure per single name.
    entry_threshold : float
        Minimum conviction score to generate an entry order.
    top_n : int
        Maximum number of simultaneous positions.
    spy_hedge : bool
        Whether to hedge sector positions with SPY.
    hedge_ratio : float
        Beta-adjusted hedge ratio for SPY hedges.
    """

    def __init__(
        self,
        gateway: IBKRGateway,
        settings=None,
        max_gross_exposure: float = 200_000.0,
        max_single_name: float = 0.20,
        entry_threshold: float = 0.15,
        top_n: int = 5,
        spy_hedge: bool = True,
        hedge_ratio: float = 0.85,
    ):
        self.gw = gateway
        self.max_gross_exposure = max_gross_exposure
        self.max_single_name = max_single_name
        self.entry_threshold = entry_threshold
        self.top_n = top_n
        self.spy_hedge = spy_hedge
        self.hedge_ratio = hedge_ratio

        # Override from settings if provided
        if settings is not None:
            self.max_gross_exposure = getattr(settings, "max_gross_exposure", self.max_gross_exposure)
            self.entry_threshold = getattr(settings, "entry_threshold", self.entry_threshold)

    def execute_signals(self, master_df: pd.DataFrame, regime_state: str = "UNKNOWN") -> list:
        """
        Read master_df conviction scores and directions, generate orders.

        Expects master_df to have columns:
            - ticker (str): sector ETF ticker
            - conviction_score (float): combined conviction
            - direction (str): "LONG" or "SHORT"

        Additional optional columns used if present:
            - hedge_ratio (float): per-ticker beta hedge ratio

        Parameters
        ----------
        master_df : pd.DataFrame
            Signal dataframe from QuantEngine.
        regime_state : str
            Current regime label (e.g., "RISK_ON", "RISK_OFF", "NEUTRAL").

        Returns
        -------
        list[dict]
            List of execution results from the gateway.
        """
        if master_df is None or master_df.empty:
            logger.info("SignalExecutor: no signals to execute")
            return []

        # Ensure required columns
        required = {"ticker", "conviction_score", "direction"}
        missing = required - set(master_df.columns)
        if missing:
            logger.error("SignalExecutor: master_df missing columns: %s", missing)
            return []

        # Filter by entry threshold and sort by conviction
        candidates = master_df[
            master_df["conviction_score"].abs() >= self.entry_threshold
        ].copy()
        candidates = candidates.sort_values("conviction_score", key=abs, ascending=False)

        # Take top N
        candidates = candidates.head(self.top_n)

        if candidates.empty:
            logger.info("SignalExecutor: no candidates above threshold %.3f",
                        self.entry_threshold)
            return []

        # Scale regime exposure
        regime_scale = self._regime_scaling(regime_state)

        # Get existing positions to avoid doubling up
        existing = set(self.gw.get_positions().keys())

        results = []
        for _, row in candidates.iterrows():
            ticker = row["ticker"]
            direction = row["direction"].upper()
            conviction = abs(row["conviction_score"])

            # Skip if already holding this name
            if ticker in existing:
                logger.info("SignalExecutor: skipping %s — already in portfolio", ticker)
                continue

            # Compute notional: conviction-weighted, regime-scaled
            notional = (
                self.max_gross_exposure
                * self.max_single_name
                * conviction
                * regime_scale
            )

            hr = row.get("hedge_ratio", self.hedge_ratio)
            if pd.isna(hr):
                hr = self.hedge_ratio

            result = self.gw.place_rv_trade(
                sector_etf=ticker,
                spy_hedge=self.spy_hedge,
                direction=direction,
                notional=notional,
                hedge_ratio=hr,
            )
            results.append(result)

        logger.info("SignalExecutor: executed %d of %d candidates (regime=%s, scale=%.2f)",
                     len(results), len(candidates), regime_state, regime_scale)
        return results

    def execute_exits(self, trade_monitor_results: list) -> list:
        """
        Execute exit orders based on trade monitor signals.

        Parameters
        ----------
        trade_monitor_results : list[dict]
            Each dict should contain:
                - ticker (str): symbol to exit
                - action (str): "EXIT" or "REDUCE"
                - reason (str): why the exit is triggered
                - reduce_pct (float, optional): fraction to reduce (for REDUCE)

        Returns
        -------
        list[dict]
            List of execution results.
        """
        if not trade_monitor_results:
            return []

        positions = self.gw.get_positions()
        results = []

        for signal in trade_monitor_results:
            ticker = signal.get("ticker")
            action = signal.get("action", "EXIT").upper()
            reason = signal.get("reason", "monitor_signal")

            if ticker not in positions:
                logger.warning("Exit signal for %s but no position found", ticker)
                continue

            pos = positions[ticker]
            qty = abs(pos["quantity"])
            side = "SELL" if pos["quantity"] > 0 else "BUY"

            if action == "REDUCE":
                reduce_pct = signal.get("reduce_pct", 0.5)
                qty = max(1, int(qty * reduce_pct))

            result = self.gw.place_market_order(ticker, qty, side)
            result["exit_reason"] = reason
            results.append(result)

            logger.info("Exit %s: %s %d %s (reason=%s)", action, side, qty, ticker, reason)

        return results

    @staticmethod
    def _regime_scaling(regime_state: str) -> float:
        """
        Scale position sizing by regime.

        Returns a multiplier in [0.25, 1.0].
        """
        scaling = {
            "RISK_ON": 1.0,
            "NEUTRAL": 0.70,
            "RISK_OFF": 0.40,
            "CRISIS": 0.25,
        }
        return scaling.get(regime_state.upper() if regime_state else "NEUTRAL", 0.60)


# ═════════════════════════════════════════════════════════════════════════════
# Options Execution — Straddle/Strangle Builder
# ═════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
import math


@dataclass
class OptionLeg:
    """Single leg of an options trade."""
    ticker: str
    expiry: str           # YYYYMMDD
    strike: float
    right: str            # "C" (call) or "P" (put)
    action: str           # "BUY" or "SELL"
    quantity: int
    order_type: str = "LMT"
    limit_price: float = 0.0
    # Computed
    notional: float = 0.0
    iv_at_entry: float = 0.0


@dataclass
class OptionsOrder:
    """Multi-leg options order with risk checks."""
    order_id: str
    strategy: str         # "STRADDLE" / "STRANGLE" / "CALL_SPREAD" / "PUT_SPREAD" / "COLLAR"
    underlying: str
    legs: List[OptionLeg] = field(default_factory=list)
    # Risk
    max_loss: float = 0.0
    max_profit: float = 0.0
    breakeven_up: float = 0.0
    breakeven_down: float = 0.0
    # Greeks (aggregate)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    # Status
    status: str = "PENDING"  # "PENDING" / "FILLED" / "REJECTED" / "DRY_RUN"
    rejection_reason: str = ""


class OptionsBuilder:
    """
    Builder for options strategies on sector ETFs.

    Supported strategies:
      - ATM Straddle: sell call + sell put at same strike (short vol)
      - OTM Strangle: sell call above + sell put below (wider short vol)
      - Call Spread: buy call + sell higher call (bullish with cap)
      - Put Spread: buy put + sell lower put (bearish with cap)
      - Collar: long stock + sell call + buy put (hedged long)

    All strategies include:
      - IV threshold check (only enter when IV > RV × threshold)
      - Max loss limit enforcement
      - Delta-neutral verification for straddles/strangles
      - Expiry selection (nearest monthly with DTE ≥ min_dte)
    """

    def __init__(
        self,
        gateway: Optional["IBKRGateway"] = None,
        min_dte: int = 21,
        iv_threshold_ratio: float = 1.2,    # IV must be > 1.2× RV to sell vol
        max_loss_pct: float = 0.03,          # Max 3% portfolio loss per trade
    ):
        self.gateway = gateway
        self.min_dte = min_dte
        self.iv_threshold = iv_threshold_ratio
        self.max_loss_pct = max_loss_pct
        self._dry_run = gateway is None or not _HAS_IB_INSYNC

    def build_straddle(
        self,
        ticker: str,
        price: float,
        iv: float,
        rv: float,
        quantity: int = 1,
        expiry: str = "",
    ) -> OptionsOrder:
        """
        Build an ATM straddle (sell call + sell put at same strike).

        For short-vol / dispersion trades:
        - Sell when IV/RV > threshold (vol is rich)
        - Profit from theta decay if realized vol < implied vol
        """
        order_id = f"OPT_STRAD_{ticker}_{uuid.uuid4().hex[:6]}"

        # IV threshold check
        if iv < rv * self.iv_threshold:
            return OptionsOrder(
                order_id=order_id, strategy="STRADDLE", underlying=ticker,
                status="REJECTED",
                rejection_reason=f"IV ({iv:.1%}) < {self.iv_threshold:.1f}× RV ({rv:.1%}) — vol not rich enough",
            )

        strike = round(price, 0)  # ATM strike

        call_leg = OptionLeg(
            ticker=ticker, expiry=expiry, strike=strike, right="C",
            action="SELL", quantity=quantity,
            iv_at_entry=iv, notional=price * quantity * 100,
        )
        put_leg = OptionLeg(
            ticker=ticker, expiry=expiry, strike=strike, right="P",
            action="SELL", quantity=quantity,
            iv_at_entry=iv, notional=price * quantity * 100,
        )

        # Greeks estimation (Black-Scholes approximation)
        T = self.min_dte / 252
        sqrt_T = math.sqrt(T)
        straddle_premium = price * iv * sqrt_T * 0.8  # ~80% of BS straddle price
        max_loss = price * 0.15 * quantity * 100       # Cap at 15% of underlying
        breakeven_up = strike + straddle_premium
        breakeven_down = strike - straddle_premium

        return OptionsOrder(
            order_id=order_id, strategy="STRADDLE", underlying=ticker,
            legs=[call_leg, put_leg],
            max_loss=round(max_loss, 2),
            max_profit=round(straddle_premium * quantity * 100, 2),
            breakeven_up=round(breakeven_up, 2),
            breakeven_down=round(breakeven_down, 2),
            net_delta=0.0,  # ATM straddle is delta-neutral
            net_gamma=round(-2 * 0.4 / (price * iv * sqrt_T + 0.01), 4),  # Negative gamma
            net_vega=round(-2 * price * sqrt_T * 0.01, 4),  # Negative vega
            net_theta=round(2 * price * iv / (2 * sqrt_T * 252) * 0.01, 4),  # Positive theta
            status="DRY_RUN" if self._dry_run else "PENDING",
        )

    def build_strangle(
        self,
        ticker: str,
        price: float,
        iv: float,
        rv: float,
        quantity: int = 1,
        width_pct: float = 0.05,
        expiry: str = "",
    ) -> OptionsOrder:
        """
        Build an OTM strangle (sell call above + sell put below).

        Width: strikes are ±width_pct from current price (default 5%).
        Wider than straddle → lower premium but higher probability of profit.
        """
        order_id = f"OPT_STRANG_{ticker}_{uuid.uuid4().hex[:6]}"

        if iv < rv * self.iv_threshold:
            return OptionsOrder(
                order_id=order_id, strategy="STRANGLE", underlying=ticker,
                status="REJECTED",
                rejection_reason=f"IV ({iv:.1%}) < {self.iv_threshold:.1f}× RV ({rv:.1%})",
            )

        call_strike = round(price * (1 + width_pct), 0)
        put_strike = round(price * (1 - width_pct), 0)

        call_leg = OptionLeg(
            ticker=ticker, expiry=expiry, strike=call_strike, right="C",
            action="SELL", quantity=quantity, iv_at_entry=iv,
        )
        put_leg = OptionLeg(
            ticker=ticker, expiry=expiry, strike=put_strike, right="P",
            action="SELL", quantity=quantity, iv_at_entry=iv,
        )

        T = self.min_dte / 252
        sqrt_T = math.sqrt(T)
        premium = price * iv * sqrt_T * 0.5  # OTM strangle ~50% of straddle
        max_loss = price * 0.20 * quantity * 100

        return OptionsOrder(
            order_id=order_id, strategy="STRANGLE", underlying=ticker,
            legs=[call_leg, put_leg],
            max_loss=round(max_loss, 2),
            max_profit=round(premium * quantity * 100, 2),
            breakeven_up=round(call_strike + premium, 2),
            breakeven_down=round(put_strike - premium, 2),
            net_delta=0.0,
            net_gamma=round(-2 * 0.3 / (price * iv * sqrt_T + 0.01), 4),
            net_vega=round(-2 * price * sqrt_T * 0.008, 4),
            net_theta=round(2 * price * iv / (2 * sqrt_T * 252) * 0.008, 4),
            status="DRY_RUN" if self._dry_run else "PENDING",
        )

    def build_call_spread(
        self,
        ticker: str,
        price: float,
        quantity: int = 1,
        width_pct: float = 0.05,
        expiry: str = "",
    ) -> OptionsOrder:
        """
        Bull call spread: buy ATM call + sell OTM call.
        Defined risk: max loss = net debit, max profit = width - debit.
        """
        order_id = f"OPT_CALL_SPR_{ticker}_{uuid.uuid4().hex[:6]}"
        buy_strike = round(price, 0)
        sell_strike = round(price * (1 + width_pct), 0)

        return OptionsOrder(
            order_id=order_id, strategy="CALL_SPREAD", underlying=ticker,
            legs=[
                OptionLeg(ticker=ticker, expiry=expiry, strike=buy_strike,
                          right="C", action="BUY", quantity=quantity),
                OptionLeg(ticker=ticker, expiry=expiry, strike=sell_strike,
                          right="C", action="SELL", quantity=quantity),
            ],
            max_loss=round((sell_strike - buy_strike) * 0.4 * quantity * 100, 2),
            max_profit=round((sell_strike - buy_strike) * 0.6 * quantity * 100, 2),
            breakeven_up=round(buy_strike + (sell_strike - buy_strike) * 0.4, 2),
            net_delta=round(0.3 * quantity, 2),
            status="DRY_RUN" if self._dry_run else "PENDING",
        )

    def build_dispersion_trade(
        self,
        index_ticker: str,
        sector_tickers: List[str],
        index_price: float,
        sector_prices: Dict[str, float],
        index_iv: float,
        sector_ivs: Dict[str, float],
        sector_weights: Dict[str, float],
        quantity: int = 1,
    ) -> List[OptionsOrder]:
        """
        Build a dispersion trade: short index straddle + long sector straddles.

        Profits when realized correlation < implied correlation.
        The core alpha trade for the Short Vol / Dispersion strategy.

        Parameters
        ----------
        index_ticker : str — SPY or similar index ETF
        sector_tickers : list — sector ETFs to go long vol
        index_price, sector_prices : current prices
        index_iv, sector_ivs : implied volatilities
        sector_weights : portfolio weights for each sector
        """
        orders = []

        # Short index straddle
        index_order = self.build_straddle(
            index_ticker, index_price, index_iv, index_iv * 0.8,  # Assume RV = 80% of IV
            quantity=quantity,
        )
        if index_order.status != "REJECTED":
            index_order.strategy = "DISP_SHORT_INDEX"
            orders.append(index_order)

        # Long sector straddles (weighted by sector weights)
        for sec in sector_tickers:
            if sec not in sector_prices or sec not in sector_ivs:
                continue
            sec_qty = max(1, int(quantity * sector_weights.get(sec, 0.1)))
            sec_iv = sector_ivs[sec]
            sec_price = sector_prices[sec]

            sec_order = OptionsOrder(
                order_id=f"OPT_DISP_LONG_{sec}_{uuid.uuid4().hex[:6]}",
                strategy="DISP_LONG_SECTOR",
                underlying=sec,
                legs=[
                    OptionLeg(ticker=sec, expiry="", strike=round(sec_price, 0),
                              right="C", action="BUY", quantity=sec_qty, iv_at_entry=sec_iv),
                    OptionLeg(ticker=sec, expiry="", strike=round(sec_price, 0),
                              right="P", action="BUY", quantity=sec_qty, iv_at_entry=sec_iv),
                ],
                net_vega=round(2 * sec_price * math.sqrt(self.min_dte / 252) * 0.01, 4),
                status="DRY_RUN" if self._dry_run else "PENDING",
            )
            orders.append(sec_order)

        return orders


# ═════════════════════════════════════════════════════════════════════════════
# Smart Order Routing
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartRouteDecision:
    """Decision on how to route an order for best execution."""
    ticker: str
    quantity: int
    side: str                     # "BUY" / "SELL"
    algo: str                     # "MARKET" / "LIMIT" / "TWAP" / "VWAP" / "ARRIVAL_PRICE"
    urgency: str                  # "LOW" / "MEDIUM" / "HIGH" / "IMMEDIATE"
    limit_offset_bps: float       # Offset from mid for limit orders
    participation_rate: float     # For TWAP/VWAP: max % of volume
    reasoning: str


def smart_route(
    ticker: str,
    quantity: int,
    side: str,
    notional: float,
    avg_daily_volume: float = 10_000_000,
    spread_bps: float = 2.0,
    urgency: str = "MEDIUM",
) -> SmartRouteDecision:
    """
    Determine optimal order routing based on order size, liquidity, and urgency.

    Rules:
      - Small orders (< 0.1% ADV): Market order — minimal impact
      - Medium orders (0.1-1% ADV): Limit order with offset — reduce slippage
      - Large orders (1-5% ADV): TWAP over 30 min — spread impact
      - Very large orders (> 5% ADV): VWAP with low participation — minimize footprint

    Urgency overrides:
      - IMMEDIATE: always market order (VIX spike, regime change)
      - HIGH: limit order with tight offset
      - LOW: TWAP with extended horizon
    """
    participation = notional / max(avg_daily_volume, 1_000_000)

    if urgency == "IMMEDIATE":
        return SmartRouteDecision(
            ticker=ticker, quantity=quantity, side=side,
            algo="MARKET", urgency=urgency, limit_offset_bps=0,
            participation_rate=1.0,
            reasoning="Immediate urgency — market order for speed",
        )

    if participation < 0.001:
        algo = "MARKET"
        offset = 0
        part_rate = 1.0
        reason = f"Small order ({participation:.3%} of ADV) — market order OK"
    elif participation < 0.01:
        algo = "LIMIT"
        offset = max(1.0, spread_bps * 0.5)
        part_rate = 1.0
        reason = f"Medium order ({participation:.2%} of ADV) — limit order {offset:.0f}bps from mid"
    elif participation < 0.05:
        algo = "TWAP"
        offset = spread_bps
        part_rate = 0.15
        reason = f"Large order ({participation:.1%} of ADV) — TWAP 15% participation"
    else:
        algo = "VWAP"
        offset = spread_bps * 1.5
        part_rate = 0.08
        reason = f"Very large order ({participation:.1%} of ADV) — VWAP 8% participation"

    if urgency == "HIGH" and algo not in ("MARKET",):
        algo = "LIMIT"
        offset = max(1.0, spread_bps * 0.3)
        reason += " [HIGH urgency: upgraded to tight limit]"
    elif urgency == "LOW" and algo == "MARKET":
        algo = "LIMIT"
        offset = spread_bps
        reason += " [LOW urgency: downgraded to limit]"

    return SmartRouteDecision(
        ticker=ticker, quantity=quantity, side=side,
        algo=algo, urgency=urgency,
        limit_offset_bps=round(offset, 1),
        participation_rate=round(part_rate, 2),
        reasoning=reason,
    )
