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

    # ── Institutional Regime Intelligence Engine ─────────────────

    # 1. Multi-Horizon Forecast
    def forecast_multi_horizon(self) -> Dict:
        """
        Multi-horizon regime forecast: 1d, 5d, 20d.

        Uses current regime probabilities, transition matrix, and feature
        momentum to extrapolate regime probabilities at each horizon.
        """
        try:
            forecast = self.forecast_regime()
            probs = forecast["probabilities"]
            regimes = ["CALM", "NORMAL", "TENSION", "CRISIS"]
            prob_vec = np.array([probs.get(r, 0.25) for r in regimes])

            # Get transition matrix
            trans_df = self.compute_transition_matrix()
            trans_mat = trans_df.values  # 4x4

            # Feature momentum adjustments
            vix_f = self.compute_vix_features()
            vol_f = self.compute_volatility_features()
            vix_trend = _safe(vix_f.get("vix_5d_chg", 0))
            vol_trend = _safe(vol_f.get("vol_trend", 0))

            # Momentum bias: positive = trending toward stress
            momentum_bias = np.clip(vix_trend / 20.0 + vol_trend * 2.0, -0.15, 0.15)

            horizons = {}
            for horizon, label in [(1, "horizon_1d"), (5, "horizon_5d"), (20, "horizon_20d")]:
                # Matrix power for multi-step transition
                mat_power = np.linalg.matrix_power(trans_mat, horizon)
                projected = prob_vec @ mat_power

                # Apply momentum bias scaled by horizon
                bias_scale = min(horizon / 5.0, 1.0)
                bias_vec = np.array([-0.3, -0.1, 0.1, 0.3]) * momentum_bias * bias_scale
                projected = projected + bias_vec
                projected = np.clip(projected, 0.01, 1.0)
                projected /= projected.sum()

                # Transition risk: probability of NOT staying in current regime
                current_idx = regimes.index(forecast["predicted_regime"])
                stay_prob = mat_power[current_idx, current_idx]
                transition_risk = 1.0 - stay_prob

                # Confidence decreases with horizon
                base_confidence = max(probs.values())
                confidence = base_confidence * (1.0 - 0.02 * horizon)

                horizons[label] = {
                    "probabilities": {r: round(float(projected[i]), 3) for i, r in enumerate(regimes)},
                    "most_likely": regimes[int(np.argmax(projected))],
                    "transition_prob": round(float(transition_risk), 3),
                    "confidence": round(max(0.1, float(confidence)), 3),
                }

            log.info("Multi-horizon forecast computed: 1d=%s, 5d=%s, 20d=%s",
                     horizons["horizon_1d"]["most_likely"],
                     horizons["horizon_5d"]["most_likely"],
                     horizons["horizon_20d"]["most_likely"])
            return horizons
        except Exception as exc:
            log.error("Multi-horizon forecast failed: %s", exc)
            default_h = {
                "probabilities": {"CALM": 0.25, "NORMAL": 0.25, "TENSION": 0.25, "CRISIS": 0.25},
                "most_likely": "NORMAL", "transition_prob": 0.5, "confidence": 0.25,
            }
            return {"horizon_1d": default_h, "horizon_5d": default_h, "horizon_20d": default_h}

    # 2. False Transition Filter
    def filter_false_transitions(self, raw_forecast: Dict) -> Dict:
        """
        Filter false regime transitions by requiring multi-signal confirmation.

        Transition states: STABLE / EARLY_WARNING / TRANSITION_RISK /
                           TRANSITION_CONFIRMED / STABILIZING
        """
        try:
            features = raw_forecast.get("features", {})
            probs = raw_forecast.get("probabilities", {})
            predicted = raw_forecast.get("predicted_regime", "NORMAL")

            # Define feature families and their stress signals
            family_signals = {
                "volatility": (
                    _safe(features.get("vix_pct", 0.5)) > 0.7
                    or _safe(features.get("vol_trend", 0)) > 0.02
                ),
                "credit": (
                    _safe(features.get("credit_z", 0)) < -1.0
                    or _safe(features.get("credit_mom", 0)) < -0.005
                ),
                "correlation": (
                    _safe(features.get("avg_corr", 0.3)) > 0.6
                    or _safe(features.get("corr_chg", 0)) > 0.05
                ),
                "breadth": (
                    _safe(features.get("breadth_50d", 0.5)) < 0.4
                ),
                "vol_clustering": (
                    _safe(features.get("vol_of_vol", 0)) > 0.3
                ),
            }
            n_stressed = sum(1 for v in family_signals.values() if v)

            # Track consecutive elevated risk days from history
            consecutive_elevated = 0
            for h in reversed(self.forecast_history[-30:]):
                tp = _safe(h.get("transition_probability", 0))
                if tp > 0.4:
                    consecutive_elevated += 1
                else:
                    break

            # Determine previous regime
            prev_regime = "NORMAL"
            if len(self.forecast_history) >= 2:
                prev_regime = self.forecast_history[-2].get("predicted_regime", "NORMAL")

            regime_changed = predicted != prev_regime
            transition_prob = _safe(raw_forecast.get("transition_probability", 0))

            # Classify transition state
            if not regime_changed and transition_prob < 0.3 and n_stressed < 2:
                state = "STABLE"
            elif not regime_changed and (n_stressed >= 1 or transition_prob >= 0.3):
                state = "EARLY_WARNING"
            elif regime_changed and n_stressed < 3:
                # Single-signal spike: don't confirm yet
                state = "TRANSITION_RISK"
            elif regime_changed and n_stressed >= 3:
                if consecutive_elevated >= 2:
                    state = "TRANSITION_CONFIRMED"
                else:
                    state = "TRANSITION_RISK"
            elif not regime_changed and consecutive_elevated >= 3 and n_stressed < 2:
                state = "STABILIZING"
            else:
                state = "EARLY_WARNING"

            result = {
                "transition_state": state,
                "n_stressed_families": n_stressed,
                "stressed_families": [k for k, v in family_signals.items() if v],
                "consecutive_elevated_risk_days": consecutive_elevated,
                "regime_changed_raw": regime_changed,
                "confirmed": state == "TRANSITION_CONFIRMED",
                "previous_regime": prev_regime,
            }
            log.info("Transition filter: state=%s, stressed=%d/5, consecutive=%d",
                     state, n_stressed, consecutive_elevated)
            return result
        except Exception as exc:
            log.error("False transition filter failed: %s", exc)
            return {
                "transition_state": "EARLY_WARNING",
                "n_stressed_families": 0, "stressed_families": [],
                "consecutive_elevated_risk_days": 0, "regime_changed_raw": False,
                "confirmed": False, "previous_regime": "NORMAL",
            }

    # 3. Regime Persistence Engine
    def estimate_persistence(self) -> Dict:
        """
        Estimate regime persistence: age, maturity, survival probabilities.
        """
        try:
            regimes_list = ["CALM", "NORMAL", "TENSION", "CRISIS"]
            current_regime = "NORMAL"
            if self.forecast_history:
                current_regime = self.forecast_history[-1].get("predicted_regime", "NORMAL")

            # Compute regime age (days in current regime)
            regime_age = 0
            for h in reversed(self.forecast_history):
                if h.get("predicted_regime") == current_regime:
                    regime_age += 1
                else:
                    break

            # Maturity state
            if regime_age < 5:
                maturity = "EMERGING"
            elif regime_age < 20:
                maturity = "ESTABLISHED"
            elif regime_age < 60:
                maturity = "MATURE"
            else:
                maturity = "EXHAUSTING"

            # Persistence probabilities from transition matrix
            trans_df = self.compute_transition_matrix()
            if current_regime in trans_df.index:
                stay_1d = float(trans_df.loc[current_regime, current_regime])
            else:
                stay_1d = 0.7

            # P(stay for n days) = stay_1d^n (geometric approximation)
            # Adjust for regime age — older regimes have slightly lower persistence
            age_decay = 1.0 - min(regime_age / 200.0, 0.15)
            adj_stay = stay_1d * age_decay

            persistence_5d = round(max(0.01, adj_stay ** 5), 3)
            persistence_20d = round(max(0.001, adj_stay ** 20), 3)

            # Historical regime durations
            durations = []
            if len(self.forecast_history) >= 5:
                current_run = 1
                for i in range(1, len(self.forecast_history)):
                    if self.forecast_history[i].get("predicted_regime") == \
                       self.forecast_history[i - 1].get("predicted_regime"):
                        current_run += 1
                    else:
                        durations.append(current_run)
                        current_run = 1
                durations.append(current_run)

            avg_duration = round(float(np.mean(durations)), 1) if durations else 10.0
            median_duration = round(float(np.median(durations)), 1) if durations else 8.0

            result = {
                "current_regime": current_regime,
                "regime_age": regime_age,
                "maturity_state": maturity,
                "persistence_probability_5d": persistence_5d,
                "persistence_probability_20d": persistence_20d,
                "daily_stay_probability": round(adj_stay, 3),
                "avg_regime_duration": avg_duration,
                "median_regime_duration": median_duration,
            }
            log.info("Persistence: regime=%s, age=%dd, maturity=%s, P(5d)=%.1f%%, P(20d)=%.1f%%",
                     current_regime, regime_age, maturity,
                     persistence_5d * 100, persistence_20d * 100)
            return result
        except Exception as exc:
            log.error("Persistence estimation failed: %s", exc)
            return {
                "current_regime": "NORMAL", "regime_age": 0,
                "maturity_state": "EMERGING",
                "persistence_probability_5d": 0.5,
                "persistence_probability_20d": 0.25,
                "daily_stay_probability": 0.7,
                "avg_regime_duration": 10.0, "median_regime_duration": 8.0,
            }

    # 4. Evidence Quality & Confidence Scoring
    def compute_forecast_confidence(self, features: Dict, model_outputs: Dict) -> Dict:
        """
        Score forecast confidence based on evidence quality, model agreement,
        feature conflict, and classification stability.
        """
        try:
            # Evidence quality: what fraction of features are available and non-zero?
            expected_features = [
                "vix", "vix_pct", "credit_z", "credit_mom", "avg_corr",
                "breadth_50d", "realized_vol", "vol_trend", "vol_of_vol",
            ]
            available = sum(1 for f in expected_features
                          if f in features and features[f] != 0)
            evidence_quality = round(available / len(expected_features), 3)

            # Model disagreement: compare forecast_regime probs vs HMM probs
            forecast_probs = model_outputs.get("probabilities", {})
            hmm_probs = {}
            hmm_data = model_outputs.get("hmm", {})
            if isinstance(hmm_data, dict):
                hmm_probs = hmm_data.get("state_probabilities", {})

            if forecast_probs and hmm_probs:
                regimes = ["CALM", "NORMAL", "TENSION", "CRISIS"]
                diffs = []
                for r in regimes:
                    p1 = forecast_probs.get(r, 0.25)
                    p2 = hmm_probs.get(r, 0.25)
                    diffs.append(abs(p1 - p2))
                model_disagreement = round(min(1.0, float(np.mean(diffs)) * 4), 3)
            else:
                model_disagreement = 0.5  # Unknown without both models

            # Feature conflict: do features point in opposite directions?
            stress_signals = []
            calm_signals = []
            vix_pct = _safe(features.get("vix_pct", 0.5))
            if vix_pct > 0.7:
                stress_signals.append("vix_high")
            elif vix_pct < 0.3:
                calm_signals.append("vix_low")

            credit_z = _safe(features.get("credit_z", 0))
            if credit_z < -1.0:
                stress_signals.append("credit_stress")
            elif credit_z > 0.5:
                calm_signals.append("credit_healthy")

            breadth = _safe(features.get("breadth_50d", 0.5))
            if breadth < 0.4:
                stress_signals.append("breadth_weak")
            elif breadth > 0.7:
                calm_signals.append("breadth_strong")

            vol_trend = _safe(features.get("vol_trend", 0))
            if vol_trend > 0.02:
                stress_signals.append("vol_rising")
            elif vol_trend < -0.02:
                calm_signals.append("vol_falling")

            # Conflict = both stress and calm signals present
            if stress_signals and calm_signals:
                conflict_ratio = min(len(stress_signals), len(calm_signals)) / \
                                 max(len(stress_signals), len(calm_signals))
                feature_conflict = round(conflict_ratio, 3)
            else:
                feature_conflict = 0.0

            # Forecast confidence: weighted average, capped by evidence quality
            raw_confidence = (
                0.40 * (1.0 - model_disagreement)
                + 0.30 * (1.0 - feature_conflict)
                + 0.30 * evidence_quality
            )
            forecast_confidence = round(min(raw_confidence, evidence_quality + 0.1), 3)

            # Stability score: how likely is classification to flip tomorrow?
            max_prob = max(forecast_probs.values()) if forecast_probs else 0.25
            second_prob = sorted(forecast_probs.values(), reverse=True)[1] if len(forecast_probs) >= 2 else 0.25
            margin = max_prob - second_prob
            stability = round(min(1.0, margin * 3 + 0.2), 3)

            result = {
                "evidence_quality_score": evidence_quality,
                "model_disagreement_score": model_disagreement,
                "feature_conflict_score": feature_conflict,
                "forecast_confidence": forecast_confidence,
                "stability_score": stability,
                "stress_signals": stress_signals,
                "calm_signals": calm_signals,
            }
            log.info("Confidence: evidence=%.2f, disagreement=%.2f, conflict=%.2f, "
                     "confidence=%.2f, stability=%.2f",
                     evidence_quality, model_disagreement, feature_conflict,
                     forecast_confidence, stability)
            return result
        except Exception as exc:
            log.error("Forecast confidence scoring failed: %s", exc)
            return {
                "evidence_quality_score": 0.5, "model_disagreement_score": 0.5,
                "feature_conflict_score": 0.0, "forecast_confidence": 0.5,
                "stability_score": 0.5, "stress_signals": [], "calm_signals": [],
            }

    # 5. Institutional Safety Scores
    def compute_safety_scores(self, forecast: Dict) -> Dict:
        """
        Compute multi-dimensional safety scores for downstream agents.
        """
        try:
            probs = forecast.get("probabilities", {})
            p_crisis = _safe(probs.get("CRISIS", 0))
            p_tension = _safe(probs.get("TENSION", 0))
            p_calm = _safe(probs.get("CALM", 0))
            transition_prob = _safe(forecast.get("transition_probability", 0))

            # Original regime safety (backward compatible)
            regime_safety = _safe(forecast.get("regime_safety_score", 0.5))

            # Regime risk score (inverse of safety)
            regime_risk = round(1.0 - regime_safety, 3)

            # Execution safety: can we trade safely?
            # Penalize high vol, crisis, and transition risk
            execution_safety = round(max(0.0, min(1.0,
                1.0 - p_crisis * 2.5 - p_tension * 0.8 - transition_prob * 0.5
            )), 3)

            # Portfolio safety: should we be defensive?
            # More conservative than execution
            portfolio_safety = round(max(0.0, min(1.0,
                1.0 - p_crisis * 3.0 - p_tension * 1.0 - transition_prob * 0.3
            )), 3)

            # Research stability: is regime stable enough for research/optimization?
            # Penalize transitions heavily — unstable regimes make backtests unreliable
            research_stability = round(max(0.0, min(1.0,
                p_calm * 0.5 + (1.0 - transition_prob) * 0.5
            )), 3)

            result = {
                "regime_safety_score": regime_safety,
                "regime_risk_score": regime_risk,
                "execution_safety_score": execution_safety,
                "portfolio_safety_score": portfolio_safety,
                "research_stability_score": research_stability,
            }
            log.info("Safety scores: regime=%.2f, risk=%.2f, exec=%.2f, "
                     "portfolio=%.2f, research=%.2f",
                     regime_safety, regime_risk, execution_safety,
                     portfolio_safety, research_stability)
            return result
        except Exception as exc:
            log.error("Safety score computation failed: %s", exc)
            return {
                "regime_safety_score": 0.5, "regime_risk_score": 0.5,
                "execution_safety_score": 0.5, "portfolio_safety_score": 0.5,
                "research_stability_score": 0.5,
            }

    # 6. Downstream Action Engine
    def generate_downstream_actions(self, forecast: Dict, safety: Dict) -> Dict:
        """
        Generate concrete action directives for each downstream agent
        based on current regime and safety scores.

        Logic: CALM->normal, NORMAL->cautious, TENSION->defensive, CRISIS->halt
        """
        try:
            predicted = forecast.get("predicted_regime", "NORMAL")
            transition_prob = _safe(forecast.get("transition_probability", 0))
            exec_safety = _safe(safety.get("execution_safety_score", 0.5))
            portfolio_safety = _safe(safety.get("portfolio_safety_score", 0.5))
            research_stability = _safe(safety.get("research_stability_score", 0.5))

            def _urgency(regime: str, transition_p: float) -> str:
                if regime == "CRISIS":
                    return "CRITICAL"
                if regime == "TENSION" and transition_p > 0.5:
                    return "HIGH"
                if regime == "TENSION":
                    return "MEDIUM"
                if regime == "NORMAL" and transition_p > 0.4:
                    return "MEDIUM"
                return "LOW"

            urgency = _urgency(predicted, transition_prob)

            ACTIONS = {
                "CALM": {
                    "methodology": {"action": "normal_research_operations", "urgency": "LOW"},
                    "optimizer": {"action": "full_optimization_enabled", "urgency": "LOW"},
                    "alpha_decay": {"note": "decay_rates_nominal"},
                    "portfolio_construction": {"action": "normal_gross_exposure", "urgency": "LOW"},
                    "risk_guardian": {"action": "standard_monitoring", "urgency": "LOW"},
                    "execution": {"action": "normal_urgency_tight_slippage", "urgency": "LOW"},
                    "auto_improve": {"action": "normal_promotion_cycles", "urgency": "LOW"},
                },
                "NORMAL": {
                    "methodology": {"action": "standard_research_caution", "urgency": "LOW"},
                    "optimizer": {"action": "prefer_diversified_portfolios", "urgency": "LOW"},
                    "alpha_decay": {"note": "monitor_for_regime_driven_decay"},
                    "portfolio_construction": {"action": "standard_gross_exposure", "urgency": "LOW"},
                    "risk_guardian": {"action": "standard_monitoring", "urgency": "LOW"},
                    "execution": {"action": "normal_urgency_standard_slippage", "urgency": "LOW"},
                    "auto_improve": {"action": "normal_promotion_cycles", "urgency": "LOW"},
                },
                "TENSION": {
                    "methodology": {"action": "emphasize_stress_regime_testing", "urgency": "MEDIUM"},
                    "optimizer": {"action": "prefer_all_weather", "urgency": "MEDIUM"},
                    "alpha_decay": {"note": "observed_decay_may_be_regime_suppression"},
                    "portfolio_construction": {"action": "reduce_gross_to_1x", "urgency": "HIGH"},
                    "risk_guardian": {"action": "activate_elevated_monitoring", "urgency": "HIGH"},
                    "execution": {"action": "slow_urgency_widen_slippage", "urgency": "MEDIUM"},
                    "auto_improve": {"action": "pause_promotion_cycles", "urgency": "MEDIUM"},
                },
                "CRISIS": {
                    "methodology": {"action": "halt_new_research_validate_existing", "urgency": "CRITICAL"},
                    "optimizer": {"action": "defensive_only_minimize_drawdown", "urgency": "CRITICAL"},
                    "alpha_decay": {"note": "all_decay_suspended_crisis_override"},
                    "portfolio_construction": {"action": "max_defensive_reduce_gross_to_0.5x", "urgency": "CRITICAL"},
                    "risk_guardian": {"action": "crisis_mode_continuous_monitoring", "urgency": "CRITICAL"},
                    "execution": {"action": "halt_new_orders_unwind_only", "urgency": "CRITICAL"},
                    "auto_improve": {"action": "freeze_all_changes", "urgency": "CRITICAL"},
                },
            }

            actions = ACTIONS.get(predicted, ACTIONS["NORMAL"])

            # Override urgency if transition risk is high
            if urgency in ("HIGH", "CRITICAL"):
                for agent_key in actions:
                    if isinstance(actions[agent_key], dict) and "urgency" in actions[agent_key]:
                        current = actions[agent_key]["urgency"]
                        if urgency == "CRITICAL" or (urgency == "HIGH" and current == "LOW"):
                            actions[agent_key]["urgency"] = urgency

            actions["_meta"] = {
                "regime": predicted,
                "overall_urgency": urgency,
                "transition_probability": round(transition_prob, 3),
            }
            log.info("Downstream actions: regime=%s, urgency=%s", predicted, urgency)
            return actions
        except Exception as exc:
            log.error("Downstream action generation failed: %s", exc)
            return {
                "methodology": {"action": "standard_research_caution", "urgency": "MEDIUM"},
                "optimizer": {"action": "prefer_diversified_portfolios", "urgency": "MEDIUM"},
                "alpha_decay": {"note": "unknown_regime"},
                "portfolio_construction": {"action": "reduce_gross_to_1x", "urgency": "MEDIUM"},
                "risk_guardian": {"action": "activate_elevated_monitoring", "urgency": "MEDIUM"},
                "execution": {"action": "slow_urgency_widen_slippage", "urgency": "MEDIUM"},
                "auto_improve": {"action": "pause_promotion_cycles", "urgency": "MEDIUM"},
                "_meta": {"regime": "NORMAL", "overall_urgency": "MEDIUM", "transition_probability": 0.5},
            }

    # 7. Enhanced Leading Indicators (augmentation — called after base)
    def enhance_leading_indicators(self, leading: Dict, features: Dict) -> Dict:
        """
        Augment leading indicators with regime implications, net pressures,
        and importance ranking.
        """
        try:
            indicators = leading.get("indicators", [])

            # Add regime_implication per indicator
            regime_map = {
                "bullish": {"CALM": "supportive", "NORMAL": "supportive",
                            "TENSION": "counter_signal", "CRISIS": "counter_signal"},
                "bearish": {"CALM": "counter_signal", "NORMAL": "neutral",
                            "TENSION": "supportive", "CRISIS": "supportive"},
                "neutral": {"CALM": "neutral", "NORMAL": "neutral",
                            "TENSION": "neutral", "CRISIS": "neutral"},
            }
            for ind in indicators:
                signal = ind.get("signal", "neutral")
                ind["regime_implication"] = regime_map.get(signal, regime_map["neutral"])

            # Net risk-off pressure
            risk_off_indicators = [
                ind for ind in indicators
                if ind.get("signal") == "bearish"
            ]
            net_risk_off = round(sum(abs(ind.get("z_score", 0)) for ind in risk_off_indicators), 3)

            # Net stabilization pressure
            stabilization_indicators = [
                ind for ind in indicators
                if ind.get("signal") == "bullish"
            ]
            net_stabilization = round(sum(abs(ind.get("z_score", 0)) for ind in stabilization_indicators), 3)

            # Net fragility pressure (hidden stress: high vol-of-vol, dispersion trend, correlation accel)
            fragility_names = {"Correlation acceleration", "Sector dispersion trend", "VRP (implied - realized vol)"}
            fragility_indicators = [
                ind for ind in indicators
                if ind.get("name") in fragility_names and ind.get("signal") == "bearish"
            ]
            net_fragility = round(sum(abs(ind.get("z_score", 0)) for ind in fragility_indicators), 3)

            # Rank by absolute z-score (importance)
            ranked = sorted(indicators, key=lambda x: abs(x.get("z_score", 0)), reverse=True)

            leading["indicators"] = ranked
            leading["net_risk_off_pressure"] = net_risk_off
            leading["net_stabilization_pressure"] = net_stabilization
            leading["net_fragility_pressure"] = net_fragility
            leading["indicator_ranking"] = [
                {"name": ind["name"], "abs_z": round(abs(ind.get("z_score", 0)), 3)}
                for ind in ranked[:5]
            ]

            log.info("Enhanced indicators: risk_off=%.2f, stabilization=%.2f, fragility=%.2f",
                     net_risk_off, net_stabilization, net_fragility)
            return leading
        except Exception as exc:
            log.error("Enhanced leading indicators failed: %s", exc)
            return leading

    # 8. Machine Summary
    def build_machine_summary(self, forecast: Dict, confidence: Dict,
                               transition_filter: Dict, persistence: Dict,
                               safety: Dict, leading: Dict,
                               downstream: Dict) -> Dict:
        """
        Build a compact machine-readable summary for downstream consumption.
        """
        try:
            predicted = forecast.get("predicted_regime", "NORMAL")
            probs = forecast.get("probabilities", {})
            features = forecast.get("features", {})

            # Top risk factors
            top_risks = []
            vix_pct = _safe(features.get("vix_pct", 0.5))
            if vix_pct > 0.7:
                top_risks.append(f"VIX at {int(vix_pct * 100)}th percentile")
            credit_z = _safe(features.get("credit_z", 0))
            if credit_z < -1.0:
                top_risks.append("credit spread widening")
            breadth = _safe(features.get("breadth_50d", 0.5))
            if breadth < 0.4:
                top_risks.append(f"weak breadth ({breadth:.0%} above 50d)")
            vol_trend = _safe(features.get("vol_trend", 0))
            if vol_trend > 0.02:
                top_risks.append("rising volatility trend")
            avg_corr = _safe(features.get("avg_corr", 0.3))
            if avg_corr > 0.6:
                top_risks.append(f"elevated correlations ({avg_corr:.2f})")
            if not top_risks:
                top_risks.append("no major risk factors detected")

            # Key instruction
            instructions = {
                "CALM": "normal operations, full risk budget available",
                "NORMAL": "standard positioning, monitor transition indicators",
                "TENSION": "reduce exposure, favor defensive positioning",
                "CRISIS": "halt new positions, maximize hedges, unwind risk",
            }

            overall_urgency = downstream.get("_meta", {}).get("overall_urgency", "LOW")

            horizon_5d = forecast.get("multi_horizon", {}).get("horizon_5d", {})
            transition_risk_5d = horizon_5d.get("transition_prob", 0.5) if horizon_5d else \
                _safe(forecast.get("transition_probability", 0.5))

            summary = {
                "predicted_regime": predicted,
                "confidence": _safe(confidence.get("forecast_confidence", 0.5)),
                "transition_state": transition_filter.get("transition_state", "STABLE"),
                "transition_risk_5d": round(float(transition_risk_5d), 3),
                "persistence": persistence.get("maturity_state", "EMERGING"),
                "regime_age_days": persistence.get("regime_age", 0),
                "safety_score": _safe(safety.get("regime_safety_score", 0.5)),
                "top_risk_factors": top_risks[:5],
                "downstream_urgency": overall_urgency,
                "key_instruction": instructions.get(predicted, instructions["NORMAL"]),
            }
            log.info("Machine summary: regime=%s, confidence=%.2f, state=%s, urgency=%s",
                     predicted, summary["confidence"],
                     summary["transition_state"], overall_urgency)
            return summary
        except Exception as exc:
            log.error("Machine summary build failed: %s", exc)
            return {
                "predicted_regime": "NORMAL", "confidence": 0.5,
                "transition_state": "EARLY_WARNING", "transition_risk_5d": 0.5,
                "persistence": "EMERGING", "regime_age_days": 0,
                "safety_score": 0.5, "top_risk_factors": ["computation_error"],
                "downstream_urgency": "MEDIUM",
                "key_instruction": "reduce exposure, monitor closely",
            }

    # 9. Forecast History Analysis
    def get_transition_alerts(self, n: int = 10) -> list:
        """Return the most recent n transition state changes from history."""
        try:
            alerts = []
            history = self.forecast_history
            for i in range(1, min(len(history), n + 50)):
                prev = history[-(i + 1)] if i + 1 <= len(history) else None
                curr = history[-i]
                if prev is None:
                    continue
                prev_regime = prev.get("predicted_regime", "NORMAL")
                curr_regime = curr.get("predicted_regime", "NORMAL")
                if prev_regime != curr_regime:
                    alerts.append({
                        "timestamp": curr.get("timestamp", ""),
                        "from_regime": prev_regime,
                        "to_regime": curr_regime,
                        "transition_probability": curr.get("transition_probability", 0),
                    })
                    if len(alerts) >= n:
                        break
            return list(reversed(alerts))
        except Exception as exc:
            log.error("Transition alerts failed: %s", exc)
            return []

    def get_confidence_drift(self, n: int = 20) -> list:
        """Return rolling confidence trend from recent history."""
        try:
            history = self.forecast_history[-n:]
            drift = []
            for h in history:
                probs = h.get("probabilities", {})
                max_prob = max(probs.values()) if probs else 0.25
                drift.append({
                    "timestamp": h.get("timestamp", ""),
                    "regime": h.get("predicted_regime", "NORMAL"),
                    "max_probability": round(max_prob, 3),
                    "transition_probability": round(_safe(h.get("transition_probability", 0)), 3),
                })
            return drift
        except Exception as exc:
            log.error("Confidence drift failed: %s", exc)
            return []

    def get_regime_persistence_history(self, n: int = 20) -> list:
        """Return how long each of the last n regimes lasted."""
        try:
            if not self.forecast_history:
                return []
            runs = []
            current_regime = self.forecast_history[0].get("predicted_regime", "NORMAL")
            current_start = self.forecast_history[0].get("timestamp", "")
            current_count = 1
            for h in self.forecast_history[1:]:
                r = h.get("predicted_regime", "NORMAL")
                if r == current_regime:
                    current_count += 1
                else:
                    runs.append({
                        "regime": current_regime,
                        "duration_days": current_count,
                        "start": current_start,
                    })
                    current_regime = r
                    current_start = h.get("timestamp", "")
                    current_count = 1
            runs.append({
                "regime": current_regime,
                "duration_days": current_count,
                "start": current_start,
            })
            return runs[-n:]
        except Exception as exc:
            log.error("Regime persistence history failed: %s", exc)
            return []

    def get_false_transition_events(self) -> list:
        """Identify transitions that reverted within 3 days."""
        try:
            false_transitions = []
            history = self.forecast_history
            if len(history) < 5:
                return []
            for i in range(1, len(history) - 3):
                prev = history[i - 1].get("predicted_regime", "NORMAL")
                curr = history[i].get("predicted_regime", "NORMAL")
                if prev != curr:
                    # Check if it reverts within 3 days
                    for j in range(1, 4):
                        if i + j < len(history):
                            future = history[i + j].get("predicted_regime", "NORMAL")
                            if future == prev:
                                false_transitions.append({
                                    "timestamp": history[i].get("timestamp", ""),
                                    "from_regime": prev,
                                    "false_regime": curr,
                                    "reverted_after_days": j,
                                })
                                break
            return false_transitions
        except Exception as exc:
            log.error("False transition events failed: %s", exc)
            return []

    # ── Run ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Full cycle: features -> forecast -> institutional analytics -> save -> publish."""
        log.info("=" * 60)
        log.info("REGIME INTELLIGENCE ENGINE — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        forecast = self.forecast_regime()

        probs = forecast["probabilities"]
        log.info("Forecast: %s (P=%.1f%%)", forecast["predicted_regime"],
                 probs[forecast["predicted_regime"]] * 100)
        log.info("  CALM=%.1f%% NORMAL=%.1f%% TENSION=%.1f%% CRISIS=%.1f%%",
                 probs["CALM"]*100, probs["NORMAL"]*100, probs["TENSION"]*100, probs["CRISIS"]*100)
        log.info("  Transition prob: %.1f%%, Safety score: %.3f",
                 forecast["transition_probability"] * 100, forecast["regime_safety_score"])

        # ── Existing advanced analytics ───────────────────────────
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
            leading = {}

        try:
            accuracy = self.track_forecast_accuracy()
            forecast["forecast_accuracy"] = accuracy
            log.info("  Accuracy: %s", accuracy.get("detail", "N/A"))
        except Exception as exc:
            log.warning("Accuracy tracking skipped: %s", exc)

        # ── Institutional Intelligence Layer ──────────────────────

        # 1. Multi-Horizon Forecast
        try:
            multi_horizon = self.forecast_multi_horizon()
            forecast["multi_horizon"] = multi_horizon
            log.info("  Multi-horizon forecast computed")
        except Exception as exc:
            log.warning("Multi-horizon forecast skipped: %s", exc)

        # 2. False Transition Filter
        try:
            transition_filter = self.filter_false_transitions(forecast)
            forecast["transition_filter"] = transition_filter
            log.info("  Transition filter: %s", transition_filter.get("transition_state", "N/A"))
        except Exception as exc:
            log.warning("Transition filter skipped: %s", exc)
            transition_filter = {"transition_state": "EARLY_WARNING"}

        # 3. Regime Persistence
        try:
            persistence = self.estimate_persistence()
            forecast["persistence"] = persistence
            log.info("  Persistence: %s (age=%dd)",
                     persistence.get("maturity_state", "N/A"),
                     persistence.get("regime_age", 0))
        except Exception as exc:
            log.warning("Persistence estimation skipped: %s", exc)
            persistence = {"maturity_state": "EMERGING", "regime_age": 0}

        # 4. Evidence Quality & Confidence
        try:
            confidence = self.compute_forecast_confidence(
                forecast.get("features", {}), forecast
            )
            forecast["confidence"] = confidence
            log.info("  Confidence: %.2f", confidence.get("forecast_confidence", 0))
        except Exception as exc:
            log.warning("Confidence scoring skipped: %s", exc)
            confidence = {"forecast_confidence": 0.5}

        # 5. Institutional Safety Scores
        try:
            safety = self.compute_safety_scores(forecast)
            forecast["safety_scores"] = safety
            log.info("  Safety: exec=%.2f, portfolio=%.2f, research=%.2f",
                     safety.get("execution_safety_score", 0),
                     safety.get("portfolio_safety_score", 0),
                     safety.get("research_stability_score", 0))
        except Exception as exc:
            log.warning("Safety scores skipped: %s", exc)
            safety = {"regime_safety_score": forecast.get("regime_safety_score", 0.5)}

        # 6. Downstream Actions
        try:
            downstream = self.generate_downstream_actions(forecast, safety)
            forecast["downstream_actions"] = downstream
            log.info("  Downstream: urgency=%s",
                     downstream.get("_meta", {}).get("overall_urgency", "N/A"))
        except Exception as exc:
            log.warning("Downstream actions skipped: %s", exc)
            downstream = {"_meta": {"overall_urgency": "MEDIUM"}}

        # 7. Enhanced Leading Indicators
        try:
            if leading:
                enhanced_leading = self.enhance_leading_indicators(leading, forecast.get("features", {}))
                forecast["leading_indicators"] = enhanced_leading
                log.info("  Enhanced indicators: risk_off=%.2f, stabilization=%.2f",
                         enhanced_leading.get("net_risk_off_pressure", 0),
                         enhanced_leading.get("net_stabilization_pressure", 0))
        except Exception as exc:
            log.warning("Enhanced leading indicators skipped: %s", exc)

        # 8. Machine Summary
        try:
            machine_summary = self.build_machine_summary(
                forecast, confidence, transition_filter, persistence,
                safety, leading, downstream
            )
            forecast["machine_summary"] = machine_summary
            log.info("  Machine summary: regime=%s, confidence=%.2f, urgency=%s",
                     machine_summary.get("predicted_regime", "N/A"),
                     machine_summary.get("confidence", 0),
                     machine_summary.get("downstream_urgency", "N/A"))
        except Exception as exc:
            log.warning("Machine summary skipped: %s", exc)

        # 9. History Analysis (attach to forecast for record)
        try:
            forecast["history_analysis"] = {
                "transition_alerts": self.get_transition_alerts(5),
                "confidence_drift": self.get_confidence_drift(10),
                "regime_persistence_history": self.get_regime_persistence_history(10),
                "false_transition_events": self.get_false_transition_events(),
            }
            log.info("  History analysis attached")
        except Exception as exc:
            log.warning("History analysis skipped: %s", exc)

        # ── Save & Publish ────────────────────────────────────────
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

        log.info("Regime Intelligence Engine complete — forecast saved and published.")
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
