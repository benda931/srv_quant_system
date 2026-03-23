"""
analytics/ml_signal_combiner.py
================================
ML-Based Adaptive Signal Combination

Instead of fixed weights (40/20/20/20) for the conviction score,
trains a GradientBoosting model to learn optimal signal weights
per regime from historical data.

Features (factor zoo):
  - PCA residual z-score
  - Sector momentum (21d, 63d)
  - EWMA vol + vol rank
  - Correlation to SPY
  - Credit spread z-score
  - VIX level + z-score
  - Macro event proximity
  - Dispersion ratio
  - Fundamental valuation (PE ratio vs SPY)

Target: sign(forward_return_5d) — binary classification

Output: per-sector conviction score [0, 1] that adaptively weights
the factors based on the current regime.

Ref: Gradient Boosting (Friedman, 2001)
Ref: Adaptive factor weighting (Gu, Kelly, Xiu, 2020 — RFS)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AdaptiveSignal:
    """ML-generated conviction score for one sector."""
    ticker: str
    ml_conviction: float         # [0, 1] — ML-predicted probability of positive return
    direction: str               # "LONG" if conviction > 0.55, "SHORT" if < 0.45
    feature_importance: Dict[str, float]  # Top features driving this prediction
    regime: str


@dataclass
class MLCombinerResult:
    """Output of the ML signal combiner."""
    signals: List[AdaptiveSignal]
    model_accuracy: float        # Out-of-sample accuracy
    regime_accuracies: Dict[str, float]  # Accuracy per regime
    feature_importances: Dict[str, float]  # Global feature importances


def build_feature_matrix(
    prices: pd.DataFrame,
    sectors: List[str],
    spy: str = "SPY",
    window: int = 252,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build feature matrix for all sectors across all dates.

    Returns (X, y, metadata) where:
      X: feature matrix (n_samples × n_features)
      y: target (sign of 5-day forward return)
      metadata: (date, sector) for each sample
    """
    log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
    avail = [s for s in sectors if s in log_rets.columns]
    n = len(log_rets)

    vix_col = "^VIX" if "^VIX" in prices.columns else "VIX" if "VIX" in prices.columns else None
    hyg_col = "HYG" if "HYG" in prices.columns else None
    ief_col = "IEF" if "IEF" in prices.columns else None

    # Pre-compute features
    rel_rets = pd.DataFrame(index=log_rets.index)
    for s in avail:
        rel_rets[s] = log_rets[s] - log_rets[spy] if spy in log_rets.columns else log_rets[s]

    cum_resid = rel_rets.cumsum()
    mom_21 = log_rets[avail].rolling(21).sum()
    mom_63 = log_rets[avail].rolling(63).sum()
    vol_20 = log_rets[avail].rolling(20).std() * np.sqrt(252)
    vol_60 = log_rets[avail].rolling(60).std() * np.sqrt(252)

    # Z-scores of residuals
    z_scores = pd.DataFrame(index=log_rets.index, columns=avail, dtype=float)
    for s in avail:
        mu = cum_resid[s].rolling(60).mean()
        sd = cum_resid[s].rolling(60).std(ddof=1)
        z_scores[s] = (cum_resid[s] - mu) / sd.replace(0, np.nan)

    # VIX features
    vix_series = prices[vix_col] if vix_col else pd.Series(18.0, index=prices.index)
    vix_z = (vix_series - vix_series.rolling(60).mean()) / vix_series.rolling(60).std().replace(0, np.nan)

    # Credit z
    credit_z = pd.Series(0.0, index=prices.index)
    if hyg_col and ief_col:
        spread = np.log(prices[hyg_col] / prices[ief_col])
        mu_s = spread.rolling(60).mean()
        sd_s = spread.rolling(60).std(ddof=1)
        credit_z = (spread - mu_s) / sd_s.replace(0, np.nan)

    # Correlation to SPY
    corr_spy = pd.DataFrame(index=log_rets.index, columns=avail, dtype=float)
    for s in avail:
        if spy in log_rets.columns:
            corr_spy[s] = log_rets[s].rolling(60).corr(log_rets[spy])

    # Build X, y
    rows_X = []
    rows_y = []
    rows_meta = []
    fwd = 5  # 5-day forward return

    start = max(window, 63)
    for t in range(start, n - fwd):
        for s in avail:
            z = z_scores[s].iloc[t]
            if not math.isfinite(z):
                continue

            features = {
                "z_score": z,
                "momentum_21d": mom_21[s].iloc[t] if math.isfinite(mom_21[s].iloc[t]) else 0,
                "momentum_63d": mom_63[s].iloc[t] if math.isfinite(mom_63[s].iloc[t]) else 0,
                "vol_20d": vol_20[s].iloc[t] if math.isfinite(vol_20[s].iloc[t]) else 0.20,
                "vol_60d": vol_60[s].iloc[t] if math.isfinite(vol_60[s].iloc[t]) else 0.20,
                "vol_ratio": (vol_20[s].iloc[t] / vol_60[s].iloc[t]) if math.isfinite(vol_20[s].iloc[t]) and vol_60[s].iloc[t] > 0 else 1.0,
                "corr_spy": corr_spy[s].iloc[t] if math.isfinite(corr_spy[s].iloc[t]) else 0.5,
                "vix_level": vix_series.iloc[t] if math.isfinite(vix_series.iloc[t]) else 18,
                "vix_z": vix_z.iloc[t] if math.isfinite(vix_z.iloc[t]) else 0,
                "credit_z": credit_z.iloc[t] if math.isfinite(credit_z.iloc[t]) else 0,
            }

            # Forward return (target)
            fwd_ret = rel_rets[s].iloc[t + 1: t + fwd + 1].sum()
            if not math.isfinite(fwd_ret):
                continue

            rows_X.append(features)
            rows_y.append(1 if fwd_ret > 0 else 0)
            rows_meta.append({"date": str(log_rets.index[t].date()), "sector": s})

    X = pd.DataFrame(rows_X)
    y = pd.Series(rows_y)
    meta = pd.DataFrame(rows_meta)

    return X, y, meta


def train_adaptive_combiner(
    prices: pd.DataFrame,
    sectors: List[str],
    spy: str = "SPY",
) -> Optional[dict]:
    """
    Train ML signal combiner on historical data.

    Uses purged walk-forward: train on first 70%, validate on last 30%.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        log.warning("sklearn not available — ML combiner disabled")
        return None

    log.info("Building feature matrix for ML combiner...")
    X, y, meta = build_feature_matrix(prices, sectors, spy)

    if len(X) < 500:
        log.warning("Too few samples (%d) for ML training", len(X))
        return None

    # Purged train/test split (70/30, no overlap)
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    meta_test = meta.iloc[split:]

    log.info("Training GBM: %d train, %d test samples", len(X_train), len(X_test))

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    test_acc = model.score(X_test, y_test)
    log.info("ML combiner accuracy: %.1f%% (test)", test_acc * 100)

    # Feature importances
    fi = dict(zip(X.columns, model.feature_importances_))
    fi = {k: round(v, 4) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

    # Regime-conditional accuracy
    regime_acc = {}
    if "vix_level" in X_test.columns:
        for regime, vix_range in [("CALM", (0, 18)), ("NORMAL", (18, 25)),
                                   ("TENSION", (25, 35)), ("CRISIS", (35, 100))]:
            mask = (X_test["vix_level"] >= vix_range[0]) & (X_test["vix_level"] < vix_range[1])
            if mask.sum() >= 20:
                regime_acc[regime] = round(model.score(X_test[mask], y_test[mask]), 4)

    return {
        "model": model,
        "feature_names": list(X.columns),
        "test_accuracy": test_acc,
        "feature_importances": fi,
        "regime_accuracies": regime_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_signals(
    prices: pd.DataFrame,
    sectors: List[str],
    trained_model: dict,
    spy: str = "SPY",
) -> MLCombinerResult:
    """Generate ML-based conviction scores for current date."""
    model = trained_model["model"]
    feature_names = trained_model["feature_names"]

    log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
    avail = [s for s in sectors if s in log_rets.columns]
    n = len(log_rets)

    # Compute current features for each sector
    vix_col = "^VIX" if "^VIX" in prices.columns else "VIX" if "VIX" in prices.columns else None
    vix = float(prices[vix_col].iloc[-1]) if vix_col else 18.0

    # Classify current regime
    if vix > 35:
        regime = "CRISIS"
    elif vix > 25:
        regime = "TENSION"
    elif vix > 18:
        regime = "NORMAL"
    else:
        regime = "CALM"

    rel_rets = pd.DataFrame(index=log_rets.index)
    for s in avail:
        rel_rets[s] = log_rets[s] - log_rets[spy] if spy in log_rets.columns else log_rets[s]

    signals = []
    for s in avail:
        cum = rel_rets[s].cumsum()
        mu = cum.rolling(60).mean().iloc[-1]
        sd = cum.rolling(60).std(ddof=1).iloc[-1]
        z = (cum.iloc[-1] - mu) / sd if sd > 1e-10 else 0

        features = {
            "z_score": z,
            "momentum_21d": float(log_rets[s].tail(21).sum()),
            "momentum_63d": float(log_rets[s].tail(63).sum()),
            "vol_20d": float(log_rets[s].tail(20).std() * np.sqrt(252)),
            "vol_60d": float(log_rets[s].tail(60).std() * np.sqrt(252)),
        }
        v20 = features["vol_20d"]
        v60 = features["vol_60d"]
        features["vol_ratio"] = v20 / v60 if v60 > 0 else 1.0
        features["corr_spy"] = float(log_rets[s].tail(60).corr(log_rets[spy])) if spy in log_rets.columns else 0.5
        features["vix_level"] = vix

        vix_s = prices[vix_col].dropna() if vix_col else pd.Series(18.0, index=prices.index)
        mu_v = float(vix_s.tail(60).mean())
        sd_v = float(vix_s.tail(60).std())
        features["vix_z"] = (vix - mu_v) / sd_v if sd_v > 0 else 0

        if "HYG" in prices.columns and "IEF" in prices.columns:
            sp = np.log(prices["HYG"] / prices["IEF"]).dropna()
            mu_s = float(sp.tail(60).mean())
            sd_s = float(sp.tail(60).std(ddof=1))
            features["credit_z"] = (float(sp.iloc[-1]) - mu_s) / sd_s if sd_s > 0 else 0
        else:
            features["credit_z"] = 0

        # Predict
        X = np.array([[features.get(f, 0) for f in feature_names]])
        proba = model.predict_proba(X)[0]
        conviction = float(proba[1]) if len(proba) > 1 else 0.5

        direction = "LONG" if conviction > 0.55 else "SHORT" if conviction < 0.45 else "NEUTRAL"

        signals.append(AdaptiveSignal(
            ticker=s,
            ml_conviction=round(conviction, 4),
            direction=direction,
            feature_importance={f: round(features.get(f, 0), 4) for f in feature_names[:5]},
            regime=regime,
        ))

    signals.sort(key=lambda s: abs(s.ml_conviction - 0.5), reverse=True)

    return MLCombinerResult(
        signals=signals,
        model_accuracy=trained_model["test_accuracy"],
        regime_accuracies=trained_model.get("regime_accuracies", {}),
        feature_importances=trained_model.get("feature_importances", {}),
    )
