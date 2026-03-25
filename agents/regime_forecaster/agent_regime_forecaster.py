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

    # ── Professional-grade analytics ──────────────────────────────

    def fit_hmm(self, n_states: int = 4) -> dict:
        """
        Fit a Hidden Markov Model to multi-feature regime data.

        Uses hmmlearn if available; otherwise falls back to a simple
        k-means + Viterbi approximation. States map to
        CALM / NORMAL / TENSION / CRISIS.

        Parameters
        ----------
        n_states : int
            Number of hidden states (default: 4).

        Returns
        -------
        dict
            current_state, state_names, transition_matrix, state_probabilities,
            method (hmm/kmeans_fallback), detail
        """
        result: Dict[str, Any] = {
            "current_state": None,
            "state_names": ["CALM", "NORMAL", "TENSION", "CRISIS"],
            "transition_matrix": None,
            "state_probabilities": None,
            "method": None,
            "detail": "",
        }
        try:
            # Build feature matrix from VIX, credit, correlation
            vix_f = self.compute_vix_features()
            credit_f = self.compute_credit_features()
            corr_f = self.compute_correlation_features()

            # Need historical feature series
            vix_col = self.prices.get("^VIX", pd.Series(dtype=float)).dropna()
            if len(vix_col) < 100:
                result["detail"] = "Insufficient data for HMM (need 100+ observations)"
                return result

            # Build feature matrix: VIX level, VIX pct change, credit proxy, avg corr proxy
            vix_vals = vix_col.values[-252:]
            vix_pct = np.diff(vix_vals) / (vix_vals[:-1] + 1e-10)

            # Credit spread proxy
            hyg = self.prices.get("HYG", pd.Series(dtype=float)).dropna()
            ief = self.prices.get("IEF", pd.Series(dtype=float)).dropna()
            if len(hyg) >= len(vix_pct) + 1 and len(ief) >= len(vix_pct) + 1:
                credit_spread = (np.log(hyg.values[-len(vix_pct)-1:]) -
                                 np.log(ief.values[-len(vix_pct)-1:]))
                credit_chg = np.diff(credit_spread)
            else:
                credit_chg = np.zeros(len(vix_pct))

            n_obs = min(len(vix_pct), len(credit_chg))
            features = np.column_stack([
                vix_vals[-n_obs:] if len(vix_vals) > n_obs else vix_vals[:n_obs],
                vix_pct[-n_obs:],
                credit_chg[-n_obs:],
            ])

            # Standardize
            mu = features.mean(axis=0)
            sigma = features.std(axis=0) + 1e-10
            features_std = (features - mu) / sigma

            # Try hmmlearn
            try:
                from hmmlearn.hmm import GaussianHMM
                model = GaussianHMM(
                    n_components=n_states, covariance_type="full",
                    n_iter=100, random_state=42, verbose=False,
                )
                model.fit(features_std)
                states = model.predict(features_std)
                trans_mat = model.transmat_
                state_probs = model.predict_proba(features_std)[-1]
                result["method"] = "hmmlearn"
            except ImportError:
                log.info("hmmlearn not available, using k-means fallback")
                # K-means fallback
                from numpy.linalg import norm

                # Simple k-means
                np.random.seed(42)
                centroids = features_std[np.random.choice(len(features_std), n_states, replace=False)]
                for _ in range(50):
                    dists = np.array([norm(features_std - c, axis=1) for c in centroids])
                    labels = dists.argmin(axis=0)
                    for k in range(n_states):
                        if np.sum(labels == k) > 0:
                            centroids[k] = features_std[labels == k].mean(axis=0)
                states = labels

                # Compute transition matrix empirically
                trans_mat = np.zeros((n_states, n_states))
                for i in range(len(states) - 1):
                    trans_mat[states[i], states[i+1]] += 1
                row_sums = trans_mat.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                trans_mat = trans_mat / row_sums

                # State probabilities from last observation distance
                last_dists = np.array([norm(features_std[-1] - c) for c in centroids])
                inv_dists = 1.0 / (last_dists + 1e-10)
                state_probs = inv_dists / inv_dists.sum()
                result["method"] = "kmeans_fallback"

            # Map states to regime names by VIX level in each cluster
            state_vix_means = []
            for s in range(n_states):
                mask = states == s
                if mask.any():
                    state_vix_means.append(features[mask, 0].mean())
                else:
                    state_vix_means.append(0)
            # Sort by VIX level: lowest VIX = CALM, highest = CRISIS
            state_order = np.argsort(state_vix_means)
            name_map = {}
            regime_names = ["CALM", "NORMAL", "TENSION", "CRISIS"]
            for rank, state_idx in enumerate(state_order):
                name_map[int(state_idx)] = regime_names[min(rank, 3)]

            current_raw_state = int(states[-1])
            current_state = name_map.get(current_raw_state, "NORMAL")

            # Reorder transition matrix and probabilities
            ordered_probs = {name_map.get(i, f"S{i}"): round(float(state_probs[i]), 3)
                            for i in range(n_states)}

            result.update({
                "current_state": current_state,
                "transition_matrix": [[round(float(x), 4) for x in row] for row in trans_mat],
                "state_probabilities": ordered_probs,
                "detail": (
                    f"HMM fit ({result['method']}): current={current_state}, "
                    f"probs={ordered_probs}"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("HMM fitting failed: %s", exc)
            result["detail"] = f"HMM error: {exc}"

        return result

    def compute_transition_matrix(self) -> pd.DataFrame:
        """
        Compute empirical regime transition probability matrix from history.

        Uses forecast_history to compute P(regime_t+1 | regime_t),
        e.g., P(CRISIS | TENSION) = 0.15.

        Returns
        -------
        pd.DataFrame
            4x4 transition matrix with regimes as index and columns.
        """
        try:
            regimes = ["CALM", "NORMAL", "TENSION", "CRISIS"]
            if len(self.forecast_history) < 10:
                # Return uniform priors
                n = len(regimes)
                uniform = pd.DataFrame(
                    1.0 / n, index=regimes, columns=regimes
                )
                log.info("Insufficient history for transition matrix, returning uniform priors")
                return uniform

            # Extract predicted regimes from history
            predicted = [
                h.get("predicted_regime", "NORMAL")
                for h in self.forecast_history
                if h.get("predicted_regime") in regimes
            ]

            if len(predicted) < 10:
                n = len(regimes)
                return pd.DataFrame(1.0 / n, index=regimes, columns=regimes)

            # Count transitions
            counts = pd.DataFrame(0.0, index=regimes, columns=regimes)
            for i in range(len(predicted) - 1):
                from_r = predicted[i]
                to_r = predicted[i + 1]
                counts.loc[from_r, to_r] += 1

            # Normalize rows (add Laplace smoothing to avoid zeros)
            counts += 0.1  # Laplace smoothing
            row_sums = counts.sum(axis=1)
            trans_matrix = counts.div(row_sums, axis=0)

            log.info(
                "Transition matrix computed from %d observations:\n%s",
                len(predicted), trans_matrix.round(3).to_string(),
            )
            return trans_matrix

        except Exception as exc:
            log.error("Transition matrix computation failed: %s", exc)
            regimes = ["CALM", "NORMAL", "TENSION", "CRISIS"]
            return pd.DataFrame(0.25, index=regimes, columns=regimes)

    def compute_leading_indicators(self) -> dict:
        """
        Compute 10 leading indicators for regime forecasting.

        Each indicator returns value, z-score, and signal (bullish/bearish/neutral).

        Returns
        -------
        dict
            indicators (list of dicts), n_bullish, n_bearish, n_neutral,
            composite_score, detail
        """
        result: Dict[str, Any] = {
            "indicators": [],
            "n_bullish": 0,
            "n_bearish": 0,
            "n_neutral": 0,
            "composite_score": 0.0,
            "detail": "",
        }
        try:
            indicators = []

            def _classify(z: float) -> str:
                if z > 1.0:
                    return "bearish"
                elif z < -1.0:
                    return "bullish"
                return "neutral"

            def _inv_classify(z: float) -> str:
                """Inverse classification -- higher z is bullish."""
                if z > 1.0:
                    return "bullish"
                elif z < -1.0:
                    return "bearish"
                return "neutral"

            def _z(series: np.ndarray, lookback: int = 252) -> float:
                """Z-score of last value vs lookback window."""
                s = series[-lookback:] if len(series) > lookback else series
                mu = float(np.nanmean(s))
                sigma = float(np.nanstd(s))
                if sigma < 1e-10:
                    return 0.0
                return float((s[-1] - mu) / sigma)

            prices = self.prices

            # 1. VIX term structure slope (VIX level vs its 60d MA as proxy)
            vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()
            if len(vix) >= 60:
                vix_arr = vix.values
                vix_slope = float(vix_arr[-1] - np.mean(vix_arr[-60:]))
                z1 = _z(vix_arr - pd.Series(vix_arr).rolling(60).mean().dropna().values)
                indicators.append({
                    "name": "VIX term structure slope",
                    "value": round(vix_slope, 3),
                    "z_score": round(z1, 3),
                    "signal": _classify(z1),
                })

            # 2. Credit momentum (5d change in HYG-IEF)
            hyg = prices.get("HYG", pd.Series(dtype=float)).dropna()
            ief = prices.get("IEF", pd.Series(dtype=float)).dropna()
            if len(hyg) >= 60 and len(ief) >= 60:
                spread = np.log(hyg.values) - np.log(ief.values)
                credit_mom = float(spread[-1] - spread[-6]) if len(spread) >= 6 else 0
                z2 = _z(np.diff(spread, 5) if len(spread) > 5 else np.array([0]))
                indicators.append({
                    "name": "Credit momentum (5d)",
                    "value": round(credit_mom, 5),
                    "z_score": round(z2, 3),
                    "signal": _inv_classify(z2),
                })

            # 3. Yield curve momentum (TLT if available)
            tlt = prices.get("TLT", pd.Series(dtype=float)).dropna()
            if len(tlt) >= 60:
                tlt_mom = float(tlt.values[-1] / tlt.values[-21] - 1) if len(tlt) >= 21 else 0
                z3 = _z(pd.Series(tlt.values).pct_change(21).dropna().values)
                indicators.append({
                    "name": "Yield curve momentum (TLT 21d)",
                    "value": round(tlt_mom, 5),
                    "z_score": round(z3, 3),
                    "signal": _inv_classify(z3),
                })

            # 4. Dollar momentum (UUP if available)
            uup = prices.get("UUP", pd.Series(dtype=float)).dropna()
            if len(uup) >= 60:
                uup_mom = float(uup.values[-1] / uup.values[-21] - 1) if len(uup) >= 21 else 0
                z4 = _z(pd.Series(uup.values).pct_change(21).dropna().values)
                indicators.append({
                    "name": "Dollar momentum (UUP 21d)",
                    "value": round(uup_mom, 5),
                    "z_score": round(z4, 3),
                    "signal": _classify(z4),  # Strong dollar = bearish for equities
                })

            # 5. Momentum breadth change (delta % above 50d MA)
            avail_sectors = [s for s in self.SECTORS if s in prices.columns]
            if len(avail_sectors) >= 5:
                def _breadth_at(offset: int) -> float:
                    count = 0
                    for s in avail_sectors:
                        px = prices[s].dropna()
                        if len(px) >= 50 + offset:
                            idx = -1 - offset
                            if float(px.iloc[idx]) > float(px.iloc[idx-50:idx].mean()):
                                count += 1
                    return count / len(avail_sectors)
                breadth_now = _breadth_at(0)
                breadth_5d = _breadth_at(5) if all(len(prices[s].dropna()) >= 56 for s in avail_sectors) else breadth_now
                breadth_chg = breadth_now - breadth_5d
                indicators.append({
                    "name": "Momentum breadth change (5d)",
                    "value": round(breadth_chg, 4),
                    "z_score": round(breadth_chg * 10, 3),  # scale for signal
                    "signal": _inv_classify(breadth_chg * 10),
                })

            # 6. Correlation acceleration (2nd derivative of avg corr)
            corr_f = self.compute_correlation_features()
            corr_accel = _safe(corr_f.get("corr_chg", 0))
            indicators.append({
                "name": "Correlation acceleration",
                "value": round(corr_accel, 5),
                "z_score": round(corr_accel * 20, 3),
                "signal": _classify(corr_accel * 20),
            })

            # 7. Variance Risk Premium proxy (realized vol vs VIX)
            spy = prices.get("SPY", pd.Series(dtype=float)).dropna()
            if len(spy) >= 60 and len(vix) >= 1:
                rv = float(np.log(spy / spy.shift(1)).dropna().iloc[-20:].std() * np.sqrt(252) * 100)
                iv = float(vix.iloc[-1])
                vrp = iv - rv
                indicators.append({
                    "name": "VRP (implied - realized vol)",
                    "value": round(vrp, 2),
                    "z_score": round(vrp / 5, 3),  # normalize
                    "signal": _classify(vrp / 5),
                })

            # 8. Put-call ratio proxy (VIX vs realized vol)
            if len(spy) >= 60 and len(vix) >= 1:
                pcr_proxy = float(vix.iloc[-1]) / max(rv, 1)
                indicators.append({
                    "name": "Put-call ratio proxy",
                    "value": round(pcr_proxy, 3),
                    "z_score": round((pcr_proxy - 1.0) * 3, 3),
                    "signal": _classify((pcr_proxy - 1.0) * 3),
                })

            # 9. High-yield spread acceleration
            if len(hyg) >= 60 and len(ief) >= 60:
                spread_arr = np.log(hyg.values) - np.log(ief.values)
                if len(spread_arr) >= 21:
                    spread_chg_5d = spread_arr[-1] - spread_arr[-6]
                    spread_chg_21d = spread_arr[-1] - spread_arr[-22]
                    accel = spread_chg_5d - (spread_chg_21d / 4)
                    indicators.append({
                        "name": "HY spread acceleration",
                        "value": round(float(accel), 5),
                        "z_score": round(float(accel) * 200, 3),
                        "signal": _inv_classify(float(accel) * 200),
                    })

            # 10. Sector dispersion trend
            if len(avail_sectors) >= 5:
                rets = np.log(prices[avail_sectors] / prices[avail_sectors].shift(1)).dropna()
                if len(rets) >= 40:
                    disp_20 = float(rets.iloc[-20:].std().mean())
                    disp_40 = float(rets.iloc[-40:-20].std().mean()) if len(rets) >= 40 else disp_20
                    disp_trend = disp_20 - disp_40
                    indicators.append({
                        "name": "Sector dispersion trend",
                        "value": round(disp_trend, 5),
                        "z_score": round(disp_trend * 100, 3),
                        "signal": _classify(disp_trend * 100),
                    })

            # Aggregate signals
            n_bullish = sum(1 for ind in indicators if ind["signal"] == "bullish")
            n_bearish = sum(1 for ind in indicators if ind["signal"] == "bearish")
            n_neutral = len(indicators) - n_bullish - n_bearish
            composite = (n_bullish - n_bearish) / max(len(indicators), 1)

            result.update({
                "indicators": indicators,
                "n_bullish": n_bullish,
                "n_bearish": n_bearish,
                "n_neutral": n_neutral,
                "composite_score": round(composite, 3),
                "detail": (
                    f"Leading indicators: {len(indicators)} computed, "
                    f"{n_bullish} bullish, {n_bearish} bearish, {n_neutral} neutral, "
                    f"composite={composite:.3f}"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("Leading indicators computation failed: %s", exc)
            result["detail"] = f"Leading indicators error: {exc}"

        return result

    def track_forecast_accuracy(self) -> dict:
        """
        Evaluate historical forecast accuracy against actual regime outcomes.

        Loads forecast_history.json, compares predicted vs next-day actual
        regime, and computes accuracy, hit rate per regime, and Brier score.

        Returns
        -------
        dict
            accuracy, hit_rate_per_regime, brier_score, n_forecasts, detail
        """
        result: Dict[str, Any] = {
            "accuracy": None,
            "hit_rate_per_regime": {},
            "brier_score": None,
            "n_forecasts": 0,
            "detail": "",
        }
        try:
            hist_path = OUTPUT_DIR / "forecast_history.json"
            if not hist_path.exists() or len(self.forecast_history) < 5:
                result["detail"] = "Insufficient forecast history for accuracy tracking"
                return result

            history = self.forecast_history
            regimes = ["CALM", "NORMAL", "TENSION", "CRISIS"]

            # Compare forecast[t] vs actual regime at t+1
            correct = 0
            total = 0
            per_regime_correct: Dict[str, int] = {r: 0 for r in regimes}
            per_regime_total: Dict[str, int] = {r: 0 for r in regimes}
            brier_sum = 0.0

            for i in range(len(history) - 1):
                pred = history[i].get("predicted_regime")
                actual = history[i + 1].get("predicted_regime")
                probs = history[i].get("probabilities", {})

                if pred not in regimes or actual not in regimes:
                    continue

                total += 1
                per_regime_total[pred] = per_regime_total.get(pred, 0) + 1

                if pred == actual:
                    correct += 1
                    per_regime_correct[pred] = per_regime_correct.get(pred, 0) + 1

                # Brier score: sum of (prob_i - indicator_i)^2 for each regime
                for r in regimes:
                    p = probs.get(r, 0.25)
                    actual_ind = 1.0 if r == actual else 0.0
                    brier_sum += (p - actual_ind) ** 2

            if total < 3:
                result["detail"] = "Too few comparable forecasts for accuracy"
                return result

            accuracy = correct / total
            brier_score = brier_sum / (total * len(regimes))

            hit_rate = {}
            for r in regimes:
                if per_regime_total[r] > 0:
                    hit_rate[r] = round(per_regime_correct[r] / per_regime_total[r], 3)
                else:
                    hit_rate[r] = None

            result.update({
                "accuracy": round(accuracy, 4),
                "hit_rate_per_regime": hit_rate,
                "brier_score": round(brier_score, 4),
                "n_forecasts": total,
                "detail": (
                    f"Forecast accuracy: {accuracy:.1%} over {total} forecasts, "
                    f"Brier={brier_score:.4f}, hit rates={hit_rate}"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("Forecast accuracy tracking failed: %s", exc)
            result["detail"] = f"Accuracy tracking error: {exc}"

        return result

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

        # Run advanced analytics
        try:
            hmm_result = self.fit_hmm()
            forecast["hmm"] = hmm_result
            log.info("  HMM: %s", hmm_result.get("detail", "N/A"))
        except Exception as exc:
            log.warning("HMM fitting skipped: %s", exc)

        try:
            trans_mat = self.compute_transition_matrix()
            forecast["empirical_transition_matrix"] = trans_mat.to_dict()
            log.info("  Transition matrix computed")
        except Exception as exc:
            log.warning("Transition matrix skipped: %s", exc)

        try:
            leading = self.compute_leading_indicators()
            forecast["leading_indicators"] = leading
            log.info("  Leading indicators: %s", leading.get("detail", "N/A"))
        except Exception as exc:
            log.warning("Leading indicators skipped: %s", exc)

        try:
            accuracy = self.track_forecast_accuracy()
            forecast["forecast_accuracy"] = accuracy
            log.info("  Accuracy: %s", accuracy.get("detail", "N/A"))
        except Exception as exc:
            log.warning("Accuracy tracking skipped: %s", exc)

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
