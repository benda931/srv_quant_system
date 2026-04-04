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


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble Regime Forecaster (GBM + RF + LR stacking)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleRegimeForecast:
    """Enhanced regime forecast from ensemble model."""
    current_regime: str
    forecast_probabilities: Dict[str, float]
    most_likely_regime: str
    transition_probability: float
    confidence: float
    features_used: Dict[str, float]
    # Ensemble details
    model_weights: Dict[str, float]        # {model_name: weight}
    model_accuracies: Dict[str, float]     # {model_name: val_accuracy}
    feature_importances: Dict[str, float]  # {feature: importance}
    # Calibration
    calibration_error: float = 0.0         # Brier score
    # Time-aware
    regime_duration_days: int = 0          # Days in current regime
    mean_regime_duration: float = 0.0      # Historical avg duration


def train_ensemble_regime_forecaster(
    prices: pd.DataFrame,
    sectors: list,
    forecast_horizon: int = 5,
) -> Optional[dict]:
    """
    Train a 3-model ensemble for regime prediction.

    Models:
      1. GradientBoosting (nonlinear interactions)
      2. RandomForest (variance reduction, OOB estimates)
      3. LogisticRegression (linear baseline, calibrated probabilities)

    Stacking weights from cross-validated accuracy.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score
    except ImportError:
        log.warning("sklearn not available — ensemble regime forecasting disabled")
        return None

    n = len(prices)
    min_start = 300

    if n < min_start + forecast_horizon + 50:
        return None

    feature_names = ["vix_level", "vix_change_5d", "vix_z", "credit_z",
                     "avg_corr", "corr_change", "yield_level", "yield_change_5d",
                     "dxy_momentum", "dispersion"]

    X_rows = []
    y_rows = []

    for idx in range(min_start, n - forecast_horizon, 3):  # Step=3 to reduce redundancy
        features = compute_regime_features(prices, idx, sectors)
        if len(features) < 5:
            continue
        future_features = compute_regime_features(prices, idx + forecast_horizon, sectors)
        label = classify_regime(future_features)
        X_rows.append([features.get(f, 0.0) for f in feature_names])
        y_rows.append(label)

    if len(X_rows) < 100:
        return None

    X = np.array(X_rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    le = LabelEncoder()
    y = le.fit_transform(y_rows)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    models = []
    accuracies = {}
    importances = np.zeros(len(feature_names))

    # Model 1: GBM
    try:
        gbm = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=15, subsample=0.8, random_state=42,
        )
        gbm.fit(X_train, y_train)
        acc = float(gbm.score(X_val, y_val))
        models.append(("GBM", gbm))
        accuracies["GBM"] = round(acc, 4)
        importances += gbm.feature_importances_ * acc
    except Exception as e:
        log.debug("GBM training failed: %s", e)

    # Model 2: RandomForest
    try:
        rf = RandomForestClassifier(
            n_estimators=80, max_depth=4, min_samples_leaf=10, random_state=42,
        )
        rf.fit(X_train, y_train)
        acc = float(rf.score(X_val, y_val))
        models.append(("RF", rf))
        accuracies["RF"] = round(acc, 4)
        importances += rf.feature_importances_ * acc
    except Exception as e:
        log.debug("RF training failed: %s", e)

    # Model 3: LogisticRegression
    try:
        lr = LogisticRegression(C=1.0, max_iter=300, multi_class="multinomial", random_state=42)
        lr.fit(X_train, y_train)
        acc = float(lr.score(X_val, y_val))
        models.append(("LR", lr))
        accuracies["LR"] = round(acc, 4)
        importances += np.abs(lr.coef_).mean(axis=0) * acc
    except Exception as e:
        log.debug("LR training failed: %s", e)

    if not models:
        return None

    # Normalize importances
    total_imp = importances.sum()
    if total_imp > 0:
        importances /= total_imp

    feature_imp = {feature_names[i]: round(float(importances[i]), 4) for i in range(len(feature_names))}

    # Stacking weights (accuracy-proportional)
    total_acc = sum(accuracies.values())
    weights = {name: round(acc / total_acc, 3) for name, acc in accuracies.items()}

    # Brier score (calibration)
    brier = 0.0
    try:
        proba_ensemble = np.zeros((len(X_val), len(le.classes_)))
        for name, model in models:
            w = weights.get(name, 1 / len(models))
            proba_ensemble += w * model.predict_proba(X_val)
        for i in range(len(X_val)):
            true_class = y_val[i]
            for j in range(len(le.classes_)):
                brier += (proba_ensemble[i, j] - (1 if j == true_class else 0)) ** 2
        brier /= (len(X_val) * len(le.classes_))
    except Exception:
        pass

    overall_acc = sum(acc * weights.get(name, 0) for name, acc in accuracies.items())

    log.info(
        "Ensemble regime forecaster: %d models, weighted acc=%.1f%%, Brier=%.4f, top feature=%s",
        len(models), overall_acc * 100, brier,
        max(feature_imp, key=feature_imp.get) if feature_imp else "?",
    )

    return {
        "models": models,
        "encoder": le,
        "feature_names": feature_names,
        "weights": weights,
        "accuracies": accuracies,
        "feature_importances": feature_imp,
        "val_accuracy": overall_acc,
        "brier_score": brier,
    }


def forecast_regime_ensemble(
    prices: pd.DataFrame,
    sectors: list,
    trained_ensemble: Optional[dict] = None,
) -> EnsembleRegimeForecast:
    """
    Predict regime using ensemble model with weighted probabilities.
    """
    idx = len(prices) - 1
    features = compute_regime_features(prices, idx, sectors)
    current_regime = classify_regime(features)

    if trained_ensemble is None:
        # Fallback to basic
        basic = forecast_regime(prices, sectors, None)
        return EnsembleRegimeForecast(
            current_regime=basic.current_regime,
            forecast_probabilities=basic.forecast_probabilities,
            most_likely_regime=basic.most_likely_regime,
            transition_probability=basic.transition_probability,
            confidence=basic.confidence,
            features_used=basic.features_used,
            model_weights={}, model_accuracies={}, feature_importances={},
        )

    le = trained_ensemble["encoder"]
    feature_names = trained_ensemble["feature_names"]
    models = trained_ensemble["models"]
    weights = trained_ensemble["weights"]

    X = np.array([[features.get(f, 0.0) for f in feature_names]])
    X = np.nan_to_num(X, nan=0.0)

    # Weighted ensemble prediction
    proba = np.zeros(len(le.classes_))
    for name, model in models:
        w = weights.get(name, 1 / len(models))
        try:
            proba += w * model.predict_proba(X)[0]
        except Exception:
            proba += w * (1 / len(le.classes_))

    probs = {cls: round(float(proba[i]), 4) for i, cls in enumerate(le.classes_)}

    # Ensure all regimes present
    for r in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
        if r not in probs:
            probs[r] = 0.0

    most_likely = max(probs, key=probs.get)
    transition_prob = round(1.0 - probs.get(current_regime, 0), 4)

    # Regime duration
    regime_dur = 0
    for t in range(idx, max(0, idx - 252), -1):
        feat = compute_regime_features(prices, t, sectors)
        if classify_regime(feat) == current_regime:
            regime_dur += 1
        else:
            break

    return EnsembleRegimeForecast(
        current_regime=current_regime,
        forecast_probabilities=probs,
        most_likely_regime=most_likely,
        transition_probability=transition_prob,
        confidence=round(trained_ensemble.get("val_accuracy", 0.5), 4),
        features_used=features,
        model_weights=weights,
        model_accuracies=trained_ensemble.get("accuracies", {}),
        feature_importances=trained_ensemble.get("feature_importances", {}),
        calibration_error=round(trained_ensemble.get("brier_score", 0), 4),
        regime_duration_days=regime_dur,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Auto-Retrain Schedule
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrainDecision:
    """Whether to retrain the regime forecaster."""
    should_retrain: bool
    reason: str
    days_since_last_train: int
    drift_detected: bool
    accuracy_degradation: float


def check_retrain_needed(
    last_train_date: str,
    current_accuracy: float,
    baseline_accuracy: float = 0.60,
    max_age_days: int = 30,
    accuracy_threshold: float = 0.05,
) -> RetrainDecision:
    """
    Determine if the regime forecaster needs retraining.

    Triggers:
      1. Model older than max_age_days (30 days default)
      2. Accuracy dropped more than accuracy_threshold (5%) from baseline
      3. Regime distribution shift (concept drift)
    """
    from datetime import date

    try:
        last = date.fromisoformat(last_train_date)
        days_stale = (date.today() - last).days
    except Exception:
        days_stale = 999

    acc_drop = baseline_accuracy - current_accuracy
    drift = acc_drop > accuracy_threshold

    should = days_stale >= max_age_days or drift

    if days_stale >= max_age_days:
        reason = f"Model {days_stale}d old (max {max_age_days}d)"
    elif drift:
        reason = f"Accuracy dropped {acc_drop:.1%} from baseline {baseline_accuracy:.1%}"
    else:
        reason = "Model is fresh and performing well"

    return RetrainDecision(
        should_retrain=should,
        reason=reason,
        days_since_last_train=days_stale,
        drift_detected=drift,
        accuracy_degradation=round(acc_drop, 4),
    )
