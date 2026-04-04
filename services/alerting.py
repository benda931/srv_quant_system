"""
services/alerting.py
======================
Unified Alert Routing Service for the SRV Quantamental DSS.

Centralizes all alert dispatch across channels:
  - Slack (webhook)
  - File (JSONL logs per day/level)
  - Agent Bus (pub/sub events)
  - Console (structured logging)

Alert levels: INFO / WARNING / CRITICAL / TRADE / REGIME

Usage:
    from services.alerting import AlertService, Alert
    alerts = AlertService(settings)
    alerts.send(Alert(level="CRITICAL", title="VIX Spike", message="VIX +5 to 30"))
    alerts.send_trade_alert("OPENED", "LONG XLK", conviction=0.85)
    alerts.send_regime_change("NORMAL", "TENSION")
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("alerting")


@dataclass
class Alert:
    """Structured alert message."""
    level: str                    # INFO / WARNING / CRITICAL / TRADE / REGIME
    title: str
    message: str
    source: str = "system"        # Which component generated this alert
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AlertService:
    """
    Unified alert routing to multiple channels.

    Channels:
      1. Slack: sends rich-formatted messages via webhook
      2. File: appends to data/alerts/YYYY-MM-DD_{level}.jsonl
      3. Bus: publishes to agent bus for inter-agent awareness
      4. Console: structured log at appropriate level
    """

    def __init__(self, settings=None):
        self._settings = settings
        self._webhook_url = ""
        self._alerts_dir = Path("data/alerts")

        if settings:
            self._webhook_url = getattr(settings, "slack_webhook_url", "")
            self._alerts_dir = settings.project_root / "data" / "alerts"

        self._alerts_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Alert] = []

    # ── Core dispatch ────────────────────────────────────────────────────

    def send(self, alert: Alert) -> bool:
        """
        Send an alert to all configured channels.
        Returns True if at least one channel succeeded.
        """
        success = False
        self._history.append(alert)

        # Console
        self._log_console(alert)

        # File
        try:
            self._write_file(alert)
            success = True
        except Exception as e:
            log.debug("Alert file write failed: %s", e)

        # Slack
        if self._webhook_url and alert.level in ("CRITICAL", "WARNING", "TRADE", "REGIME"):
            try:
                self._send_slack(alert)
                success = True
            except Exception as e:
                log.debug("Slack alert failed: %s", e)

        # Bus
        try:
            self._publish_bus(alert)
            success = True
        except Exception:
            pass

        return success

    # ── Convenience methods ──────────────────────────────────────────────

    def send_trade_alert(self, action: str, description: str,
                          conviction: float = 0.0, **kwargs) -> None:
        """Alert for trade events (OPENED, CLOSED, REBALANCED)."""
        self.send(Alert(
            level="TRADE",
            title=f"Trade {action}",
            message=f"{description} | Conviction: {conviction:.2f}",
            source="paper_trader",
            metadata={"action": action, "conviction": conviction, **kwargs},
        ))

    def send_regime_change(self, from_regime: str, to_regime: str,
                            confidence: float = 0.0) -> None:
        """Alert for regime transitions."""
        level = "CRITICAL" if to_regime == "CRISIS" else "WARNING" if to_regime == "TENSION" else "INFO"
        self.send(Alert(
            level=level,
            title=f"Regime Change: {from_regime} → {to_regime}",
            message=f"Market regime transitioned with {confidence:.0%} confidence",
            source="regime_forecaster",
            metadata={"from": from_regime, "to": to_regime, "confidence": confidence},
        ))

    def send_risk_alert(self, risk_type: str, details: str, severity: str = "WARNING") -> None:
        """Alert for risk events (drawdown, exposure, VIX spike)."""
        self.send(Alert(
            level=severity,
            title=f"Risk: {risk_type}",
            message=details,
            source="risk_guardian",
        ))

    def send_pipeline_complete(self, run_id: int, steps_ok: int, steps_fail: int,
                                 duration_s: float, regime: str) -> None:
        """Summary alert after pipeline completes."""
        level = "INFO" if steps_fail == 0 else "WARNING"
        self.send(Alert(
            level=level,
            title="Pipeline Complete",
            message=f"Run #{run_id} | {steps_ok}/{steps_ok+steps_fail} OK | {duration_s:.0f}s | {regime}",
            source="engine_service",
            metadata={"run_id": run_id, "steps_ok": steps_ok, "steps_fail": steps_fail},
        ))

    def send_auto_improve(self, param: str, old_val: Any, new_val: Any,
                            sharpe_delta: float, promoted: bool) -> None:
        """Alert for auto-improve parameter changes."""
        icon = "✅" if promoted else "❌"
        self.send(Alert(
            level="INFO" if promoted else "WARNING",
            title=f"Auto-Improve: {param}",
            message=f"{icon} {old_val} → {new_val} | Sharpe Δ={sharpe_delta:+.3f} | {'PROMOTED' if promoted else 'REJECTED'}",
            source="auto_improve",
            metadata={"param": param, "promoted": promoted, "delta": sharpe_delta},
        ))

    # ── Recent alerts ────────────────────────────────────────────────────

    def recent_alerts(self, n: int = 20, level: Optional[str] = None) -> List[Alert]:
        """Get most recent alerts, optionally filtered by level."""
        alerts = self._history[-n * 3:]  # Buffer for filtering
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts[-n:]

    def today_alert_count(self) -> Dict[str, int]:
        """Count today's alerts by level."""
        today = date.today().isoformat()
        counts = {"INFO": 0, "WARNING": 0, "CRITICAL": 0, "TRADE": 0, "REGIME": 0}
        for a in self._history:
            if a.timestamp.startswith(today):
                counts[a.level] = counts.get(a.level, 0) + 1
        return counts

    # ── Channel implementations ──────────────────────────────────────────

    def _log_console(self, alert: Alert) -> None:
        level_map = {
            "INFO": logging.INFO, "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL, "TRADE": logging.INFO,
            "REGIME": logging.WARNING,
        }
        log.log(
            level_map.get(alert.level, logging.INFO),
            "🔔 [%s] %s: %s", alert.level, alert.title, alert.message[:100],
        )

    def _write_file(self, alert: Alert) -> None:
        today = date.today().isoformat()
        filepath = self._alerts_dir / f"{today}_{alert.level.lower()}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": alert.timestamp,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "metadata": alert.metadata,
            }, default=str) + "\n")

    def _send_slack(self, alert: Alert) -> None:
        if not self._webhook_url:
            return

        icon = {
            "CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️",
            "TRADE": "📊", "REGIME": "🌡️",
        }.get(alert.level, "📋")

        text = f"{icon} *[{alert.level}] {alert.title}*\n{alert.message}"
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            self._webhook_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)

    def _publish_bus(self, alert: Alert) -> None:
        try:
            from scripts.agent_bus import AgentBus
            bus = AgentBus()
            bus.publish("alerts", {
                "ts": alert.timestamp,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message[:200],
                "source": alert.source,
            })
        except Exception:
            pass
