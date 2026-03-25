"""
Alpha Decay Monitor — Early warning system for strategy death.

Monitors rolling performance metrics and signals when alpha is decaying:
- Rolling IC (21d, 63d)
- Rolling Sharpe (63d)
- Rolling Win Rate (63d)
- Regime-conditional performance
- Consecutive losing periods

Decay levels: HEALTHY → EARLY_DECAY → DECAYING → DEAD
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_DIR / "alpha_decay.log", maxBytes=10_000_000, backupCount=3),
    ],
)
log = logging.getLogger("alpha_decay")

OUTPUT_DIR = ROOT / "agents" / "alpha_decay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "agents" / "methodology" / "reports"


class AlphaDecayMonitor:
    """Detects strategy decay before major losses."""

    SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
    MR_WHITELIST = {"XLC", "XLF", "XLI", "XLU"}

    def __init__(self, settings=None):
        self._settings = settings
        self._prices: Optional[pd.DataFrame] = None
        self._methodology_results: Optional[Dict] = None

    @property
    def settings(self):
        if self._settings is None:
            from config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            p = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if p.exists():
                self._prices = pd.read_parquet(p)
        return self._prices

    @property
    def methodology_results(self) -> Optional[Dict]:
        if self._methodology_results is None:
            self._methodology_results = self._load_latest_methodology()
        return self._methodology_results

    def _load_latest_methodology(self) -> Optional[Dict]:
        """Load most recent methodology lab results."""
        if not REPORTS_DIR.exists():
            return None
        files = sorted(REPORTS_DIR.glob("*_methodology_lab.json"), reverse=True)
        if not files:
            return None
        try:
            return json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── Rolling Metrics ────────────────────────────────────────────

    def compute_rolling_metrics(self, lookback: int = 63) -> Dict[str, Any]:
        """Compute rolling IC, Sharpe, Win Rate for the MR whitelist strategy."""
        prices = self.prices
        if prices is None or len(prices) < lookback + 60:
            return {"error": "Insufficient data"}

        spy = prices.get("SPY", pd.Series(dtype=float)).dropna()
        sectors = [s for s in self.MR_WHITELIST if s in prices.columns]
        if not sectors:
            return {"error": "No whitelist sectors"}

        # Build relative returns
        log_rel = np.log(prices[sectors].div(spy, axis=0))
        rets = log_rel.diff().dropna()

        # Z-scores
        z_lookback = 60
        z_scores = {}
        for s in sectors:
            z = (log_rel[s] - log_rel[s].rolling(z_lookback).mean()) / log_rel[s].rolling(z_lookback).std()
            z_scores[s] = z

        # Rolling IC: correlation between z-score and forward 5d return
        fwd_5d = rets.shift(-5).rolling(5).mean()
        n = len(rets)
        ic_series = []
        sharpe_series = []
        wr_series = []

        step = 5
        for end in range(lookback + z_lookback, n - 5, step):
            start = end - lookback
            period_ics = []
            period_returns = []

            for s in sectors:
                z = z_scores[s].iloc[start:end].dropna()
                fwd = fwd_5d[s].iloc[start:end].dropna()
                common = z.index.intersection(fwd.index)
                if len(common) >= 20:
                    ic = float(z.loc[common].corr(fwd.loc[common]))
                    if math.isfinite(ic):
                        period_ics.append(ic)

                # Simulated returns: direction opposite to z, weighted by |z|
                for i in range(max(0, len(z) - 5), len(z)):
                    idx = z.index[i] if i < len(z.index) else None
                    if idx is None:
                        continue
                    z_val = float(z.iloc[i]) if math.isfinite(float(z.iloc[i])) else 0
                    if abs(z_val) > 0.7 and idx in fwd.index:
                        direction = -1 if z_val > 0 else 1
                        ret = direction * float(fwd.loc[idx])
                        period_returns.append(ret)

            avg_ic = float(np.mean(period_ics)) if period_ics else 0
            ic_series.append(avg_ic)

            if period_returns:
                arr = np.array(period_returns)
                sh = float(arr.mean() / arr.std() * np.sqrt(252/5)) if arr.std() > 1e-10 else 0
                wr = float((arr > 0).mean())
            else:
                sh, wr = 0, 0.5

            sharpe_series.append(sh)
            wr_series.append(wr)

        return {
            "ic_series": ic_series,
            "sharpe_series": sharpe_series,
            "wr_series": wr_series,
            "ic_current": ic_series[-1] if ic_series else 0,
            "sharpe_current": sharpe_series[-1] if sharpe_series else 0,
            "wr_current": wr_series[-1] if wr_series else 0.5,
            "n_periods": len(ic_series),
        }

    def detect_decay_signals(self, metrics: Dict) -> Dict[str, Any]:
        """Analyze metrics for decay signals."""
        signals = {
            "ic_declining": False,
            "sharpe_declining": False,
            "wr_declining": False,
            "consecutive_negative_ic": 0,
            "consecutive_negative_sharpe": 0,
        }

        ic = metrics.get("ic_series", [])
        sharpe = metrics.get("sharpe_series", [])
        wr = metrics.get("wr_series", [])

        # IC trend: linear regression slope over last 10 periods
        if len(ic) >= 10:
            recent = np.array(ic[-10:])
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])
            signals["ic_slope"] = round(slope, 4)
            signals["ic_declining"] = slope < -0.001

        # Sharpe trend
        if len(sharpe) >= 10:
            recent = np.array(sharpe[-10:])
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])
            signals["sharpe_slope"] = round(slope, 4)
            signals["sharpe_declining"] = slope < -0.01

        # Win rate trend
        if len(wr) >= 10:
            recent = np.array(wr[-10:])
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])
            signals["wr_slope"] = round(slope, 4)
            signals["wr_declining"] = slope < -0.005

        # Consecutive negative periods
        for i in range(len(ic) - 1, -1, -1):
            if ic[i] < 0:
                signals["consecutive_negative_ic"] += 1
            else:
                break

        for i in range(len(sharpe) - 1, -1, -1):
            if sharpe[i] < 0:
                signals["consecutive_negative_sharpe"] += 1
            else:
                break

        return signals

    def classify_decay(self, decay_signals: Dict) -> str:
        """Classify overall decay level."""
        neg_ic = decay_signals.get("consecutive_negative_ic", 0)
        neg_sharpe = decay_signals.get("consecutive_negative_sharpe", 0)
        declining_count = sum([
            decay_signals.get("ic_declining", False),
            decay_signals.get("sharpe_declining", False),
            decay_signals.get("wr_declining", False),
        ])

        if neg_sharpe >= 6 or neg_ic >= 6:
            return "DEAD"
        if neg_sharpe >= 3 or neg_ic >= 3 or declining_count >= 2:
            return "DECAYING"
        if declining_count >= 1 or neg_sharpe >= 1:
            return "EARLY_DECAY"
        return "HEALTHY"

    def generate_recommendations(self, decay_level: str, metrics: Dict) -> List[str]:
        """Generate actionable recommendations based on decay level."""
        recs = []
        if decay_level == "HEALTHY":
            recs.append("Continue current strategy — all metrics within normal range")
        elif decay_level == "EARLY_DECAY":
            recs.append("Tighten z-entry threshold from 0.7 to 0.9")
            recs.append("Reduce position sizes by 20%")
            recs.append("Monitor daily for further deterioration")
        elif decay_level == "DECAYING":
            recs.append("⚠️ ALERT: Strategy is decaying — switch to defensive params")
            recs.append("Reduce max positions from 4 to 2")
            recs.append("Increase VIX kill threshold from 32 to 28")
            recs.append("Notify Optimizer agent for parameter re-calibration")
        elif decay_level == "DEAD":
            recs.append("🚨 CRITICAL: Strategy appears dead — recommend DISABLE")
            recs.append("Close all open positions from this strategy")
            recs.append("Notify all agents to switch to alternative strategy")
            recs.append("Run full re-calibration cycle")

        # Add methodology comparison if available
        if self.methodology_results:
            strategies = self.methodology_results.get("strategies", {})
            best = max(strategies.items(), key=lambda x: x[1].get("sharpe", -999), default=None)
            if best:
                recs.append(f"Best alternative strategy: {best[0]} (Sharpe={best[1].get('sharpe', 0):.3f})")

        return recs

    # ── Run ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Full cycle: compute metrics → detect decay → classify → recommend."""
        log.info("=" * 60)
        log.info("ALPHA DECAY MONITOR — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        metrics = self.compute_rolling_metrics(lookback=63)
        if "error" in metrics:
            log.warning("Cannot compute metrics: %s", metrics["error"])
            result = {"status": "ERROR", "error": metrics["error"]}
            (OUTPUT_DIR / "decay_status.json").write_text(json.dumps(result, indent=2))
            return result

        decay_signals = self.detect_decay_signals(metrics)
        decay_level = self.classify_decay(decay_signals)
        recommendations = self.generate_recommendations(decay_level, metrics)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decay_level": decay_level,
            "metrics": {
                "ic_current": metrics["ic_current"],
                "sharpe_current": metrics["sharpe_current"],
                "wr_current": metrics["wr_current"],
                "n_periods": metrics["n_periods"],
            },
            "signals": decay_signals,
            "recommendations": recommendations,
        }

        log.info("Decay level: %s", decay_level)
        log.info("  IC=%.4f, Sharpe=%.3f, WR=%.1f%%",
                 metrics["ic_current"], metrics["sharpe_current"], metrics["wr_current"]*100)
        log.info("  IC declining: %s, Sharpe declining: %s",
                 decay_signals.get("ic_declining"), decay_signals.get("sharpe_declining"))
        log.info("  Consecutive negative IC: %d, Sharpe: %d",
                 decay_signals.get("consecutive_negative_ic", 0),
                 decay_signals.get("consecutive_negative_sharpe", 0))
        for rec in recommendations:
            log.info("  → %s", rec)

        # Save
        (OUTPUT_DIR / "decay_status.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

        # Publish to bus
        try:
            from agents.shared.agent_bus import get_bus
            bus = get_bus()
            bus.publish("alpha_decay", result)
        except Exception:
            pass

        # Update registry
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            reg = get_registry()
            reg.heartbeat("alpha_decay", AgentStatus.COMPLETED)
        except Exception:
            pass

        return result


def main():
    parser = argparse.ArgumentParser(description="Alpha Decay Monitor Agent")
    parser.add_argument("--once", action="store_true", help="Run one decay analysis")
    args = parser.parse_args()

    if args.once:
        monitor = AlphaDecayMonitor()
        result = monitor.run()
        print(f"\nDecay Level: {result.get('decay_level', 'UNKNOWN')}")
    else:
        print("Usage: python -m agents.alpha_decay --once")


if __name__ == "__main__":
    main()
