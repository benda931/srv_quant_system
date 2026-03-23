"""
analytics/ml_regime_forecast.py
================================
ML-Based Regime Transition Forecasting

Predicts 5-day-ahead regime probabilities using:
  - VIX level + change + z-score
  - Credit spread z-score
  - Average correlation + change
  - Market-mode share + change
  - Dispersion ratio
  - Yield curve slope (TNX)
  - Dollar momentum (DXY)

Uses GradientBoosting classifier trained on historical regime labels.

Output: probability distribution over {CALM, NORMAL, TENSION, CRISIS}
for the next 5 trading days.

Ref: Hidden Markov Model for regime detection (Hamilton, 1989)
Ref: Gradient Boosting for classification (Friedman, 2001)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RegimeForecast:
    """5-day-ahead regime probability forecast."""
    current_regime: str
    forecast_probabilities: Dict[str, float]  # {CALM: 0.1, NORMAL: 0.3, TENSION: 0.4, CRISIS: 0.2}
    most_likely_regime: str
    transition_probability: float  # Probability of regime change
    confidence: float              # Model confidence (0-1)
    features_used: Dict[str, float]


def compute_regime_features(
    prices: pd.DataFrame,
    idx: int,
    sectors: list,
    settings=None,
) -> Dict[str, float]:
    """Compute regime features at a given date index."""
    features = {}

    # VIX features
    vix_col = "^VIX" if "^VIX" in prices.columns else "VIX" if "VIX" in prices.columns else None
    if vix_col and idx >= 20:
        vix = prices[vix_col].iloc[:idx+1].dropna()
        if len(vix) >= 20:
            features["vix_level"] = float(vix.iloc[-1])
            features["vix_change_5d"] = float(vix.iloc[-1] - vix.iloc[-5]) if len(vix) > 5 else 0
            mu = float(vix.tail(60).mean())
            sd = float(vix.tail(60).std())
            features["vix_z"] = (float(vix.iloc[-1]) - mu) / sd if sd > 0 else 0

    # Credit spread
    if "HYG" in prices.columns and "IEF" in prices.columns and idx >= 60:
        spread = np.log(prices["HYG"] / prices["IEF"]).iloc[:idx+1].dropna()
        if len(spread) >= 60:
            mu = float(spread.tail(60).mean())
            sd = float(spread.tail(60).std())
            features["credit_z"] = (float(spread.iloc[-1]) - mu) / sd if sd > 0 else 0

    # Correlation features
    avail = [s for s in sectors if s in prices.columns]
    if len(avail) >= 5 and idx >= 60:
        log_rets = np.log(prices[avail] / prices[avail].shift(1)).iloc[:idx+1].dropna()
        if len(log_rets) >= 20:
            C = log_rets.tail(20).corr()
            iu = np.triu_indices(len(avail), k=1)
            features["avg_corr"] = float(np.mean(C.values[iu]))

            if len(log_rets) >= 40:
                C_prev = log_rets.iloc[-40:-20].corr()
                features["corr_change"] = features["avg_corr"] - float(np.mean(C_prev.values[iu]))

    # Market mode (eigenvalue)
    if "avg_corr" in features:
        features["mode_proxy"] = features["avg_corr"]  # Simplified

    # Yield curve
    if "^TNX" in prices.columns and idx >= 5:
        tnx = prices["^TNX"].iloc[:idx+1].dropna()
        if len(tnx) >= 5:
            features["yield_level"] = float(tnx.iloc[-1])
            features["yield_change_5d"] = float(tnx.iloc[-1] - tnx.iloc[-5])

    # Dollar
    dxy_col = "DX-Y.NYB" if "DX-Y.NYB" in prices.columns else None
    if dxy_col and idx >= 20:
        dxy = prices[dxy_col].iloc[:idx+1].dropna()
        if len(dxy) >= 20:
            features["dxy_momentum"] = float(np.log(dxy.iloc[-1] / dxy.iloc[-20]))

    # Dispersion
    if len(avail) >= 5 and idx >= 20:
        log_rets2 = np.log(prices[avail] / prices[avail].shift(1)).iloc[:idx+1].dropna()
        if len(log_rets2) >= 20:
            features["dispersion"] = float(log_rets2.tail(20).std().mean())

    return features


def classify_regime(features: Dict[str, float]) -> str:
    """Rule-based regime classification (used as label for training)."""
    vix = features.get("vix_level", 18)
    corr = features.get("avg_corr", 0.3)
    credit = features.get("credit_z", 0)

    if vix > 35 or corr > 0.75 or credit < -2.5:
        return "CRISIS"
    if vix > 25 or corr > 0.60 or credit < -1.0:
        return "TENSION"
    if vix > 18 or corr > 0.45:
        return "NORMAL"
    return "CALM"


def train_regime_forecaster(
    prices: pd.DataFrame,
    sectors: list,
    forecast_horizon: int = 5,
) -> Optional[object]:
    """
    Train a GradientBoosting classifier to predict regime 5 days ahead.

    Returns trained model or None if insufficient data.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        log.warning("sklearn not available — regime forecasting disabled")
        return None

    n = len(prices)
    min_start = 300

    if n < min_start + forecast_horizon + 50:
        return None

    # Build feature matrix
    feature_names = ["vix_level", "vix_change_5d", "vix_z", "credit_z",
                     "avg_corr", "corr_change", "yield_level", "yield_change_5d",
                     "dxy_momentum", "dispersion"]

    X_rows = []
    y_rows = []
    dates = []

    for idx in range(min_start, n - forecast_horizon):
        features = compute_regime_features(prices, idx, sectors)
        if len(features) < 5:
            continue

        # Label = regime at t+horizon
        future_features = compute_regime_features(prices, idx + forecast_horizon, sectors)
        label = classify_regime(future_features)

        row = [features.get(f, 0.0) for f in feature_names]
        X_rows.append(row)
        y_rows.append(label)
        dates.append(prices.index[idx])

    if len(X_rows) < 100:
        return None

    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(y_rows)

    # Train with last 80% as train, 20% as validation
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    log.info("Regime forecaster trained: %d samples, val accuracy=%.1f%%, classes=%s",
             len(X_train), val_acc * 100, list(le.classes_))

    return {"model": model, "encoder": le, "feature_names": feature_names, "val_accuracy": val_acc}


def forecast_regime(
    prices: pd.DataFrame,
    sectors: list,
    trained_model: Optional[dict] = None,
) -> RegimeForecast:
    """
    Predict regime probabilities for next 5 days.
    """
    # Compute current features
    idx = len(prices) - 1
    features = compute_regime_features(prices, idx, sectors)
    current_regime = classify_regime(features)

    if trained_model is None:
        # Fallback: heuristic forecast
        probs = {"CALM": 0.25, "NORMAL": 0.25, "TENSION": 0.25, "CRISIS": 0.25}
        # Bias toward current regime
        probs[current_regime] += 0.30
        total = sum(probs.values())
        probs = {k: round(v / total, 3) for k, v in probs.items()}

        return RegimeForecast(
            current_regime=current_regime,
            forecast_probabilities=probs,
            most_likely_regime=current_regime,
            transition_probability=round(1.0 - probs[current_regime], 3),
            confidence=0.3,
            features_used=features,
        )

    model = trained_model["model"]
    le = trained_model["encoder"]
    feature_names = trained_model["feature_names"]

    X = np.array([[features.get(f, 0.0) for f in feature_names]])
    proba = model.predict_proba(X)[0]

    probs = {}
    for i, cls in enumerate(le.classes_):
        probs[cls] = round(float(proba[i]), 3)

    most_likely = max(probs, key=probs.get)
    transition_prob = round(1.0 - probs.get(current_regime, 0), 3)

    return RegimeForecast(
        current_regime=current_regime,
        forecast_probabilities=probs,
        most_likely_regime=most_likely,
        transition_probability=transition_prob,
        confidence=round(trained_model.get("val_accuracy", 0.5), 3),
        features_used=features,
    )
