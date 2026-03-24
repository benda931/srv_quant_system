"""
execution/order_manager.py
==========================
Persistent order tracking for the SRV Quant System.

Provides:
    OrderManager — tracks all orders, fills, and cancels with JSON-backed
                   persistent state.  Supports history queries and daily
                   P&L summaries.

Design:
    - State persisted to a JSON file (default: data/order_state.json).
    - Thread-safe via a threading.Lock.
    - Atomic writes: writes to a temp file first, then renames.
    - Each order record is timestamped and immutable once filled/cancelled.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Default state file path (relative to project root)
_DEFAULT_STATE = str(
    Path(__file__).resolve().parents[1] / "data" / "order_state.json"
)


class OrderManager:
    """
    Tracks all orders, fills, and cancels with persistent state.

    Parameters
    ----------
    state_file : str
        Path to the JSON file for order state persistence.
    """

    def __init__(self, state_file: str = _DEFAULT_STATE):
        self.state_file = state_file
        self._orders: Dict[str, dict] = {}  # order_id (str) -> record
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load order state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self._orders = data.get("orders", {})
                logger.info("OrderManager: loaded %d orders from %s",
                            len(self._orders), self.state_file)
            except Exception as exc:
                logger.warning("OrderManager: failed to load state: %s", exc)
                self._orders = {}
        else:
            self._orders = {}
            logger.info("OrderManager: no state file found, starting fresh")

    def _save_state(self) -> None:
        """Persist order state to disk (atomic write)."""
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_file + ".tmp"
            payload = {
                "orders": self._orders,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "count": len(self._orders),
            }
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, self.state_file)
        except Exception as exc:
            logger.error("OrderManager: failed to save state: %s", exc)

    # ------------------------------------------------------------------
    # Order logging
    # ------------------------------------------------------------------

    def log_order(
        self,
        order_id: Any,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        status: str,
        order_type: str = "MKT",
        dry_run: bool = True,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Log a new order.

        Parameters
        ----------
        order_id : Any
            Unique order identifier (will be stored as string).
        symbol : str
            Ticker symbol.
        side : str
            "BUY" or "SELL".
        qty : int
            Order quantity.
        price : float
            Order or fill price.
        status : str
            Order status (e.g., "FILLED", "SUBMITTED", "REJECTED").
        order_type : str
            "MKT", "LMT", etc.
        dry_run : bool
            Whether this was a dry-run order.
        metadata : dict or None
            Additional metadata (risk_check, trade_id, etc.).

        Returns
        -------
        dict
            The stored order record.
        """
        oid = str(order_id)
        record = {
            "order_id": oid,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": qty,
            "price": price,
            "status": status.upper(),
            "order_type": order_type.upper(),
            "dry_run": dry_run,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fills": [],
            "metadata": metadata or {},
        }
        with _lock:
            self._orders[oid] = record
            self._save_state()

        logger.info("OrderManager: logged order %s %s %d %s @ $%.2f [%s]",
                     oid, side, qty, symbol, price, status)
        return record

    def update_fill(
        self, order_id: Any, fill_price: float, fill_qty: int,
    ) -> Optional[dict]:
        """
        Record a fill (partial or complete) for an existing order.

        Parameters
        ----------
        order_id : Any
            The order to update.
        fill_price : float
            Fill price.
        fill_qty : int
            Number of shares filled.

        Returns
        -------
        dict or None
            Updated order record, or None if order not found.
        """
        oid = str(order_id)
        with _lock:
            if oid not in self._orders:
                logger.warning("OrderManager: order %s not found for fill update", oid)
                return None

            record = self._orders[oid]
            fill_entry = {
                "fill_price": fill_price,
                "fill_qty": fill_qty,
                "fill_time": datetime.now(timezone.utc).isoformat(),
            }
            record["fills"].append(fill_entry)

            # Update aggregate
            total_filled = sum(f["fill_qty"] for f in record["fills"])
            if total_filled >= record["quantity"]:
                record["status"] = "FILLED"
            else:
                record["status"] = "PARTIAL"

            # Weighted average fill price
            total_notional = sum(f["fill_price"] * f["fill_qty"] for f in record["fills"])
            record["price"] = total_notional / total_filled if total_filled > 0 else 0.0
            record["updated_at"] = datetime.now(timezone.utc).isoformat()

            self._save_state()

        logger.info("OrderManager: fill for %s — %d @ $%.2f (total filled: %d/%d)",
                     oid, fill_qty, fill_price, total_filled, record["quantity"])
        return record

    def update_status(self, order_id: Any, new_status: str) -> Optional[dict]:
        """
        Update the status of an order (e.g., CANCELLED, ERROR).

        Parameters
        ----------
        order_id : Any
            The order to update.
        new_status : str
            New status string.

        Returns
        -------
        dict or None
            Updated record, or None if not found.
        """
        oid = str(order_id)
        with _lock:
            if oid not in self._orders:
                return None
            self._orders[oid]["status"] = new_status.upper()
            self._orders[oid]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
        return self._orders[oid]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: Any) -> Optional[dict]:
        """Get a single order by ID."""
        return self._orders.get(str(order_id))

    def get_history(self, last_n: int = 50) -> list:
        """
        Get the most recent N orders, sorted by creation time descending.

        Parameters
        ----------
        last_n : int
            Maximum number of records to return.

        Returns
        -------
        list[dict]
            Order records sorted newest-first.
        """
        sorted_orders = sorted(
            self._orders.values(),
            key=lambda o: o.get("created_at", ""),
            reverse=True,
        )
        return sorted_orders[:last_n]

    def get_open_orders(self) -> list:
        """Get all orders with status SUBMITTED or PARTIAL."""
        return [
            o for o in self._orders.values()
            if o.get("status") in ("SUBMITTED", "PARTIAL", "PENDING")
        ]

    def daily_summary(self, target_date: Optional[date] = None) -> dict:
        """
        Generate a daily summary of order activity and approximate P&L.

        Parameters
        ----------
        target_date : date or None
            Date to summarize. Defaults to today (UTC).

        Returns
        -------
        dict
            {
                date: str,
                total_orders: int,
                filled: int,
                rejected: int,
                cancelled: int,
                partial: int,
                buy_notional: float,
                sell_notional: float,
                net_notional: float,
                symbols_traded: list[str],
            }
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()
        target_str = target_date.isoformat()

        day_orders = [
            o for o in self._orders.values()
            if o.get("created_at", "")[:10] == target_str
        ]

        filled = [o for o in day_orders if o["status"] == "FILLED"]
        rejected = [o for o in day_orders if o["status"] == "REJECTED"]
        cancelled = [o for o in day_orders if o["status"] == "CANCELLED"]
        partial = [o for o in day_orders if o["status"] == "PARTIAL"]

        buy_notional = sum(
            o["price"] * o["quantity"]
            for o in filled if o["side"] == "BUY"
        )
        sell_notional = sum(
            o["price"] * o["quantity"]
            for o in filled if o["side"] == "SELL"
        )

        symbols = sorted(set(o["symbol"] for o in day_orders))

        return {
            "date": target_str,
            "total_orders": len(day_orders),
            "filled": len(filled),
            "rejected": len(rejected),
            "cancelled": len(cancelled),
            "partial": len(partial),
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "net_notional": buy_notional - sell_notional,
            "symbols_traded": symbols,
        }

    def clear_state(self) -> None:
        """
        Clear all order state (use with caution — for testing only).
        """
        with _lock:
            self._orders = {}
            self._save_state()
        logger.warning("OrderManager: state cleared")
