"""
Regime Forecaster Agent — Predicts market regime transitions.

Multi-signal approach:
1. VIX level + term structure proxy
2. Credit spread momentum (HYG-IEF)
3. Correlation regime (avg sector corr trend)
4. Momentum breadth (% sectors above 50d MA)
5. Volatility clustering (EWMA vol trend)

Outputs: P(CALM), P(NORMAL), P(TENSION), P(CRISIS)
Updates regime_safety_score for the signal stack.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
        RotatingFileHandler(LOG_DIR / "regime_forecaster.log", maxBytes=10_000_000, backupCount=3),
    ],
)
log = logging.getLogger("regime_forecaster")

OUTPUT_DIR = ROOT / "agents" / "regime_forecaster"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _sigmoid(x: float) -> float:
    if x > 20: return 1.0
    if x < -20: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe(val) -> float:
    v = float(val) if val is not None else 0.0
    return v if math.isfinite(v) else 0.0


class RegimeForecaster:
    """Predicts regime transitions using multi-signal approach."""

    SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

    def __init__(self, settings=None):
        self._settings = settings
        self._prices: Optional[pd.DataFrame] = None
        self.forecast_history: list = []
        self._load_history()

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
            else:
                raise FileNotFoundError(f"No prices at {p}")
        return self._prices

    def _load_history(self):
        hist_path = OUTPUT_DIR / "forecast_history.json"
        if hist_path.exists():
            try:
                self.forecast_history = json.loads(hist_path.read_text())[-365:]
            except Exception:
                self.forecast_history = []

    # ── Feature Blocks ────────────────────────────────────────────

    def compute_vix_features(self) -> Dict[str, float]:
        """VIX level, percentile, momentum, volatility-of-vol."""
        vix = self.prices.get("^VIX", pd.Series(dtype=float)).dropna()
        if len(vix) < 60:
            return {"vix": 20, "vix_pct": 0.5, "vix_5d_chg": 0, "vix_vol": 0.2}

        current = float(vix.iloc[-1])
        pct_252 = float((vix.iloc[-252:] <= current).mean()) if len(vix) >= 252 else 0.5
        chg_5d = float(vix.iloc[-1] - vix.iloc[-6]) if len(vix) >= 6 else 0
        vol_vix = float(vix.pct_change().iloc[-60:].std() * np.sqrt(252))

        return {
            "vix": round(current, 2),
            "vix_pct": round(pct_252, 3),
            "vix_5d_chg": round(chg_5d, 2),
            "vix_vol": round(vol_vix, 3),
        }

    def compute_credit_features(self) -> Dict[str, float]:
        """Credit spread momentum via HYG-IEF."""
        hyg = self.prices.get("HYG", pd.Series(dtype=float)).dropna()
        ief = self.prices.get("IEF", pd.Series(dtype=float)).dropna()
        if len(hyg) < 60 or len(ief) < 60:
            return {"credit_spread": 0, "credit_z": 0, "credit_mom": 0}

        spread = np.log(hyg) - np.log(ief)
        spread = spread.dropna()
        if len(spread) < 60:
            return {"credit_spread": 0, "credit_z": 0, "credit_mom": 0}

        current = float(spread.iloc[-1])
        mu = float(spread.iloc[-252:].mean()) if len(spread) >= 252 else float(spread.mean())
        sigma = float(spread.iloc[-252:].std()) if len(spread) >= 252 else float(spread.std())
        z = (current - mu) / sigma if sigma > 1e-10 else 0
        mom = float(spread.iloc[-1] - spread.iloc[-21]) if len(spread) >= 21 else 0

        return {
            "credit_spread": round(current, 4),
            "credit_z": round(z, 3),
            "credit_mom": round(mom, 4),
        }

    def compute_correlation_features(self) -> Dict[str, float]:
        """Average sector correlation, rate of change, dispersion."""
        avail = [s for s in self.SECTORS if s in self.prices.columns]
        if len(avail) < 5:
            return {"avg_corr": 0.3, "corr_chg": 0, "dispersion": 0.01}

        rets = np.log(self.prices[avail] / self.prices[avail].shift(1)).dropna()
        if len(rets) < 60:
            return {"avg_corr": 0.3, "corr_chg": 0, "dispersion": 0.01}

        # Current 60d correlation
        corr_now = rets.iloc[-60:].corr().values
        n = corr_now.shape[0]
        iu = np.triu_indices(n, k=1)
        avg_now = float(np.nanmean(corr_now[iu]))

        # 60d ago correlation
        if len(rets) >= 120:
            corr_old = rets.iloc[-120:-60].corr().values
            avg_old = float(np.nanmean(corr_old[iu]))
        else:
            avg_old = avg_now

        # Dispersion = cross-sectional vol of returns
        dispersion = float(rets.iloc[-20:].std().mean())

        return {
            "avg_corr": round(avg_now, 3),
            "corr_chg": round(avg_now - avg_old, 4),
            "dispersion": round(dispersion, 4),
        }

    def compute_breadth_features(self) -> Dict[str, float]:
        """Momentum breadth: % of sectors above 50d/200d MA."""
        avail = [s for s in self.SECTORS if s in self.prices.columns]
        if not avail:
            return {"breadth_50d": 0.5, "breadth_200d": 0.5}

        above_50 = 0
        above_200 = 0
        for s in avail:
            px = self.prices[s].dropna()
            if len(px) >= 50:
                above_50 += 1 if float(px.iloc[-1]) > float(px.iloc[-50:].mean()) else 0
            if len(px) >= 200:
                above_200 += 1 if float(px.iloc[-1]) > float(px.iloc[-200:].mean()) else 0

        return {
            "breadth_50d": round(above_50 / len(avail), 3),
            "breadth_200d": round(above_200 / max(1, len([s for s in avail if len(self.prices[s].dropna()) >= 200])), 3),
        }

    def compute_volatility_features(self) -> Dict[str, float]:
        """Realized vol, EWMA vol trend."""
        spy = self.prices.get("SPY", pd.Series(dtype=float)).dropna()
        if len(spy) < 60:
            return {"realized_vol": 0.15, "vol_trend": 0, "vol_of_vol": 0}

        rets = np.log(spy / spy.shift(1)).dropna()
        rv_20 = float(rets.iloc[-20:].std() * np.sqrt(252))
        rv_60 = float(rets.iloc[-60:].std() * np.sqrt(252))

        # Vol trend: positive = vol increasing
        vol_trend = rv_20 - rv_60

        # Vol of vol
        rolling_vol = rets.rolling(20).std() * np.sqrt(252)
        vov = float(rolling_vol.iloc[-60:].std()) if len(rolling_vol.dropna()) >= 60 else 0

        return {
            "realized_vol": round(rv_20, 4),
            "vol_trend": round(vol_trend, 4),
            "vol_of_vol": round(vov, 4),
        }

    # ── Forecasting ───────────────────────────────────────────────

    def forecast_regime(self) -> Dict[str, Any]:
        """Combine all features into regime probabilities."""
        vix_f = self.compute_vix_features()
        credit_f = self.compute_credit_features()
        corr_f = self.compute_correlation_features()
        breadth_f = self.compute_breadth_features()
        vol_f = self.compute_volatility_features()

        # Crisis score: higher = more likely crisis
        crisis_score = (
            0.30 * _safe(vix_f["vix_pct"])           # VIX percentile
            + 0.20 * max(0, -_safe(credit_f["credit_z"]))  # Credit stress (negative z = stress)
            + 0.15 * _safe(corr_f["avg_corr"])        # High correlation = crisis
            + 0.15 * (1 - _safe(breadth_f["breadth_50d"]))  # Low breadth = crisis
            + 0.10 * max(0, _safe(vol_f["vol_trend"]))  # Rising vol
            + 0.10 * max(0, _safe(vix_f["vix_5d_chg"]) / 10)  # VIX spike
        )

        # Tension score
        tension_score = (
            0.25 * max(0, _safe(vix_f["vix_pct"]) - 0.5)
            + 0.25 * _safe(corr_f["avg_corr"])
            + 0.25 * max(0, _safe(vol_f["realized_vol"]) - 0.15) * 5
            + 0.25 * max(0, -_safe(breadth_f["breadth_50d"]) + 0.6)
        )

        # Convert to probabilities
        p_crisis = _sigmoid(crisis_score * 5 - 2.5)  # Centered around 0.5 input
        p_tension = _sigmoid(tension_score * 4 - 1.5) * (1 - p_crisis)
        p_calm = max(0, (1 - p_crisis - p_tension) * _safe(breadth_f["breadth_50d"]))
        p_normal = max(0, 1 - p_crisis - p_tension - p_calm)

        # Normalize
        total = p_crisis + p_tension + p_calm + p_normal
        if total > 0:
            p_crisis /= total
            p_tension /= total
            p_calm /= total
            p_normal /= total

        # Most likely regime
        probs = {"CALM": round(p_calm, 3), "NORMAL": round(p_normal, 3),
                 "TENSION": round(p_tension, 3), "CRISIS": round(p_crisis, 3)}
        predicted = max(probs, key=probs.get)

        # Transition probability: how likely to change in next 5 days
        transition_prob = self.compute_transition_probability(vix_f, corr_f, vol_f)

        # Safety score for signal stack (0 = danger, 1 = safe)
        safety = max(0, 1 - p_crisis * 2 - p_tension * 0.5)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probabilities": probs,
            "predicted_regime": predicted,
            "transition_probability": round(transition_prob, 3),
            "regime_safety_score": round(safety, 3),
            "features": {**vix_f, **credit_f, **corr_f, **breadth_f, **vol_f},
        }

    def compute_transition_probability(self, vix_f, corr_f, vol_f) -> float:
        """Probability of regime change in next 5 days."""
        # High vol-of-vol + rapid VIX change + correlation shift = transition likely
        score = (
            abs(_safe(vix_f["vix_5d_chg"])) / 5  # VIX moved a lot
            + _safe(vol_f["vol_of_vol"]) * 3       # Unstable vol
            + abs(_safe(corr_f["corr_chg"])) * 10  # Correlation shifting
        )
        return min(1.0, _sigmoid(score * 2 - 1))

    # ── Run ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Full cycle: features → forecast → save → publish."""
        log.info("=" * 60)
        log.info("REGIME FORECASTER — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        forecast = self.forecast_regime()

        probs = forecast["probabilities"]
        log.info("Forecast: %s (P=%.1f%%)", forecast["predicted_regime"],
                 probs[forecast["predicted_regime"]] * 100)
        log.info("  CALM=%.1f%% NORMAL=%.1f%% TENSION=%.1f%% CRISIS=%.1f%%",
                 probs["CALM"]*100, probs["NORMAL"]*100, probs["TENSION"]*100, probs["CRISIS"]*100)
        log.info("  Transition prob: %.1f%%, Safety score: %.3f",
                 forecast["transition_probability"] * 100, forecast["regime_safety_score"])

        # Save current forecast
        (OUTPUT_DIR / "regime_forecast.json").write_text(
            json.dumps(forecast, indent=2, default=str), encoding="utf-8"
        )

        # Append to history
        self.forecast_history.append(forecast)
        self.forecast_history = self.forecast_history[-365:]
        (OUTPUT_DIR / "forecast_history.json").write_text(
            json.dumps(self.forecast_history, indent=2, default=str), encoding="utf-8"
        )

        # Publish to agent bus
        try:
            from agents.shared.agent_bus import get_bus
            bus = get_bus()
            bus.publish("regime_forecaster", forecast)
        except Exception:
            pass

        # Update registry
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            reg = get_registry()
            reg.heartbeat("regime_forecaster", AgentStatus.COMPLETED)
        except Exception:
            pass

        log.info("Forecast saved and published.")
        return forecast


def main():
    parser = argparse.ArgumentParser(description="Regime Forecaster Agent")
    parser.add_argument("--once", action="store_true", help="Run one forecast cycle")
    args = parser.parse_args()

    if args.once:
        forecaster = RegimeForecaster()
        result = forecaster.run()
        print(json.dumps(result["probabilities"], indent=2))
    else:
        print("Usage: python -m agents.regime_forecaster --once")


if __name__ == "__main__":
    main()
