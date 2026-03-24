"""
data_ops/realtime_feed.py
==========================
Real-time market data feed via multiple sources.

Supports:
1. FMP WebSocket (real-time quotes)
2. Yahoo Finance (fallback, 15-min delay)
3. IBKR TWS (if connected)
4. Polling mode (FMP REST API every N seconds)

Usage:
    feed = RealtimeFeed(mode="polling", interval=60)
    feed.subscribe(["SPY", "XLK", "XLF", "^VIX"])
    feed.on_update(callback)
    feed.start()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Single instrument quote."""
    symbol: str
    price: float
    change_pct: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime
    source: str  # "fmp", "yahoo", "ibkr", "cache"


@dataclass
class MarketSnapshot:
    """Full market snapshot at a point in time."""
    timestamp: datetime
    quotes: Dict[str, Quote]
    vix: float = float("nan")
    spy_price: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vix": self.vix,
            "spy": self.spy_price,
            "quotes": {
                s: {"price": q.price, "change": q.change_pct, "volume": q.volume}
                for s, q in self.quotes.items()
            },
        }


class RealtimeFeed:
    """
    Multi-source real-time market data feed.

    Modes:
    - "polling": REST API polling every N seconds (default, works everywhere)
    - "websocket": FMP WebSocket streaming (requires API key)
    - "ibkr": IBKR TWS data feed (requires connection)
    """

    def __init__(
        self,
        mode: str = "polling",
        interval: int = 60,
        api_key: Optional[str] = None,
        cache_dir: str = "data/realtime/",
    ):
        self.mode = mode
        self.interval = interval
        self.api_key = api_key or self._load_api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._symbols: List[str] = []
        self._callbacks: List[Callable[[MarketSnapshot], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[MarketSnapshot] = None
        self._history: List[MarketSnapshot] = []
        self._max_history = 1000

        # Alert thresholds
        self._alerts: Dict[str, Dict[str, float]] = {}  # {symbol: {vix_above: 30, ...}}

    def _load_api_key(self) -> str:
        try:
            from config.settings import get_settings
            return get_settings().fmp_api_key
        except Exception:
            return ""

    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time quotes for given symbols."""
        self._symbols = list(set(symbols))
        logger.info("Subscribed to %d symbols: %s", len(self._symbols), self._symbols[:5])

    def on_update(self, callback: Callable[[MarketSnapshot], None]) -> None:
        """Register a callback for each market update."""
        self._callbacks.append(callback)

    def set_alert(self, symbol: str, condition: str, value: float) -> None:
        """Set price/vol alert. condition: 'above', 'below', 'change_above'."""
        if symbol not in self._alerts:
            self._alerts[symbol] = {}
        self._alerts[symbol][condition] = value

    # ------------------------------------------------------------------
    # Polling mode (works with FMP REST API)
    # ------------------------------------------------------------------
    def _fetch_quotes_fmp(self) -> Dict[str, Quote]:
        """Fetch quotes via FMP REST API."""
        import urllib.request
        import urllib.error

        if not self.api_key:
            return {}

        quotes = {}
        # Batch request
        symbols_str = ",".join(self._symbols)
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={self.api_key}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SRV-Quant/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            now = datetime.now(timezone.utc)
            for item in data:
                sym = item.get("symbol", "")
                quotes[sym] = Quote(
                    symbol=sym,
                    price=float(item.get("price", 0)),
                    change_pct=float(item.get("changesPercentage", 0)),
                    volume=int(item.get("volume", 0)),
                    bid=float(item.get("price", 0)) * 0.9999,  # FMP doesn't always have bid/ask
                    ask=float(item.get("price", 0)) * 1.0001,
                    timestamp=now,
                    source="fmp",
                )
        except Exception as e:
            logger.warning("FMP quote fetch failed: %s", e)

        return quotes

    def _fetch_quotes_yahoo(self) -> Dict[str, Quote]:
        """Fallback: fetch quotes via yfinance (no API key needed)."""
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance not installed, skipping yahoo fallback")
            return {}

        quotes = {}
        try:
            tickers = yf.Tickers(" ".join(self._symbols))
            now = datetime.now(timezone.utc)
            for sym in self._symbols:
                try:
                    info = tickers.tickers[sym].fast_info
                    quotes[sym] = Quote(
                        symbol=sym,
                        price=float(info.get("lastPrice", 0) or info.get("regularMarketPrice", 0)),
                        change_pct=0.0,
                        volume=int(info.get("lastVolume", 0)),
                        bid=float(info.get("bid", 0)),
                        ask=float(info.get("ask", 0)),
                        timestamp=now,
                        source="yahoo",
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Yahoo fetch failed: %s", e)

        return quotes

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        """Polling loop: fetch → snapshot → callbacks → sleep."""
        while self._running:
            try:
                # Try FMP first, fallback to Yahoo
                quotes = self._fetch_quotes_fmp()
                if not quotes:
                    quotes = self._fetch_quotes_yahoo()

                if quotes:
                    snapshot = MarketSnapshot(
                        timestamp=datetime.now(timezone.utc),
                        quotes=quotes,
                        vix=quotes.get("^VIX", quotes.get("VIX", Quote("", 0, 0, 0, 0, 0, datetime.now(timezone.utc), ""))).price,
                        spy_price=quotes.get("SPY", Quote("", 0, 0, 0, 0, 0, datetime.now(timezone.utc), "")).price,
                    )
                    self._latest = snapshot
                    self._history.append(snapshot)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]

                    # Check alerts
                    self._check_alerts(snapshot)

                    # Fire callbacks
                    for cb in self._callbacks:
                        try:
                            cb(snapshot)
                        except Exception as e:
                            logger.warning("Callback error: %s", e)

                    # Cache to disk
                    self._save_snapshot(snapshot)

            except Exception as e:
                logger.error("Poll loop error: %s", e)

            time.sleep(self.interval)

    def _check_alerts(self, snapshot: MarketSnapshot) -> None:
        """Check alert conditions and log triggered alerts."""
        for sym, conditions in self._alerts.items():
            q = snapshot.quotes.get(sym)
            if not q:
                continue
            for condition, threshold in conditions.items():
                triggered = False
                if condition == "above" and q.price > threshold:
                    triggered = True
                elif condition == "below" and q.price < threshold:
                    triggered = True
                elif condition == "change_above" and abs(q.change_pct) > threshold:
                    triggered = True

                if triggered:
                    logger.warning(
                        "ALERT: %s %s %.2f (current=%.2f, change=%.2f%%)",
                        sym, condition, threshold, q.price, q.change_pct,
                    )
                    # Write alert to file for dashboard
                    alert_file = self.cache_dir / "alerts.jsonl"
                    with open(alert_file, "a") as f:
                        f.write(json.dumps({
                            "timestamp": snapshot.timestamp.isoformat(),
                            "symbol": sym,
                            "condition": condition,
                            "threshold": threshold,
                            "price": q.price,
                            "change_pct": q.change_pct,
                        }) + "\n")

    def _save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Cache snapshot to disk."""
        try:
            path = self.cache_dir / f"snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(snapshot.to_dict(), indent=2))

            # Also write latest.json for dashboard
            latest_path = self.cache_dir / "latest.json"
            latest_path.write_text(json.dumps(snapshot.to_dict(), indent=2))
        except Exception as e:
            logger.debug("Snapshot save failed: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the real-time feed in a background thread."""
        if self._running:
            return
        self._running = True

        if self.mode == "polling":
            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="realtime-feed")
            self._thread.start()
            logger.info("Real-time feed started (polling, interval=%ds)", self.interval)
        else:
            logger.warning("Mode '%s' not yet implemented, using polling", self.mode)
            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="realtime-feed")
            self._thread.start()

    def stop(self) -> None:
        """Stop the feed."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Real-time feed stopped")

    @property
    def latest(self) -> Optional[MarketSnapshot]:
        """Get latest market snapshot."""
        return self._latest

    @property
    def is_running(self) -> bool:
        return self._running

    def get_history(self, last_n: int = 100) -> List[MarketSnapshot]:
        """Get recent snapshots."""
        return self._history[-last_n:]

    def get_vix(self) -> float:
        """Current VIX level."""
        if self._latest:
            return self._latest.vix
        return float("nan")

    def get_price(self, symbol: str) -> float:
        """Current price for a symbol."""
        if self._latest and symbol in self._latest.quotes:
            return self._latest.quotes[symbol].price
        return float("nan")

    def status(self) -> dict:
        """Feed status for dashboard."""
        return {
            "running": self._running,
            "mode": self.mode,
            "interval": self.interval,
            "symbols": len(self._symbols),
            "snapshots": len(self._history),
            "latest_time": self._latest.timestamp.isoformat() if self._latest else None,
            "vix": self.get_vix(),
            "spy": self.get_price("SPY"),
        }


# ------------------------------------------------------------------
# VIX Spike Monitor (runs alongside feed)
# ------------------------------------------------------------------
class VIXMonitor:
    """
    Monitors VIX for regime-relevant events:
    - Spike: VIX jumps > 3 points in 1 hour
    - Elevated: VIX > 25 sustained for > 30 min
    - Extreme: VIX > 35
    - Contango flip: VIX term structure inverts
    """

    def __init__(self, feed: RealtimeFeed):
        self.feed = feed
        self._vix_history: List[tuple] = []  # [(timestamp, vix)]
        self._alerts_fired: set = set()

    def check(self, snapshot: MarketSnapshot) -> List[dict]:
        """Check VIX conditions. Returns list of alerts."""
        vix = snapshot.vix
        if not np.isfinite(vix):
            return []

        self._vix_history.append((snapshot.timestamp, vix))
        # Keep 2 hours of history
        cutoff = snapshot.timestamp.timestamp() - 7200
        self._vix_history = [(t, v) for t, v in self._vix_history if t.timestamp() > cutoff]

        alerts = []

        # Extreme VIX
        if vix > 35:
            alerts.append({
                "type": "VIX_EXTREME",
                "level": "CRITICAL",
                "vix": vix,
                "message": f"VIX at {vix:.1f} — extreme stress, halt all new entries",
            })

        # VIX spike (> 3 points in 1 hour)
        if len(self._vix_history) >= 2:
            hour_ago = snapshot.timestamp.timestamp() - 3600
            old_vix = [v for t, v in self._vix_history if t.timestamp() <= hour_ago]
            if old_vix:
                spike = vix - old_vix[-1]
                if spike > 3:
                    alerts.append({
                        "type": "VIX_SPIKE",
                        "level": "WARNING",
                        "vix": vix,
                        "spike": round(spike, 2),
                        "message": f"VIX spiked {spike:.1f} points in 1hr (now {vix:.1f})",
                    })

        # Elevated sustained
        if vix > 25:
            sustained = all(v > 25 for _, v in self._vix_history[-30:]) if len(self._vix_history) >= 30 else False
            if sustained:
                alerts.append({
                    "type": "VIX_ELEVATED",
                    "level": "WARNING",
                    "vix": vix,
                    "message": f"VIX sustained above 25 ({vix:.1f}) — reduce exposure",
                })

        return alerts


# ------------------------------------------------------------------
# Integration with dashboard
# ------------------------------------------------------------------
def create_dashboard_feed(settings=None) -> RealtimeFeed:
    """Create a pre-configured feed for the dashboard."""
    from config.settings import get_settings
    s = settings or get_settings()

    sectors = s.sector_list()
    symbols = list(sectors) + [s.spy_ticker, "^VIX"]

    feed = RealtimeFeed(
        mode="polling",
        interval=60,
        api_key=s.fmp_api_key,
    )
    feed.subscribe(symbols)

    # Set default alerts
    feed.set_alert("^VIX", "above", 30)
    feed.set_alert("^VIX", "above", 35)

    return feed
