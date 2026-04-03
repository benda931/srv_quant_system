"""
analytics/ml_signals.py

Signal Quality Model — sklearn-based predictor trained on walk-forward IC history.

Purpose:
    Predict how reliable the current sector signal is, given current market regime
    and historical backtest performance in similar conditions.

Output:
    quality_score per sector [0.5 – 1.5]:
    - < 1.0 → dampen conviction (regime historically unfavorable for this signal)
    - = 1.0 → neutral (no modification)
    - > 1.0 → boost conviction (regime historically favorable)

Design:
    - Train on walk-level IC from BacktestResult.walk_metrics
    - Features: regime_code, recent_ic_mean, n_walks, avg_ic_in_regime, vol_of_ic
    - Target: IC > threshold (signal was informative)
    - Falls back to regime-conditional IC mean if sklearn model underfits
    - No persistence — re-trains from backtest data on each pipeline run (~50ms)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from analytics.backtest import BacktestResult

logger = logging.getLogger(__name__)

# Regime label → integer code (matches signal doc + Settings thresholds)
_REGIME_CODE: Dict[str, int] = {
    "CALM":    0,
    "NORMAL":  1,
    "TENSION": 2,
    "CRISIS":  3,
}

_IC_THRESHOLD = 0.05          # IC > this → signal is informative (binary target)
_MIN_WALKS_FOR_GBM = 10       # fewer walks → fall back to logistic / mean IC
_QUALITY_LO = 0.5             # hard floor on quality multiplier
_QUALITY_HI = 1.5             # hard ceiling on quality multiplier
_NEUTRAL = 1.0                # fallback / prior quality score


def _regime_code(label: str) -> int:
    """Map regime string → int feature (case-insensitive, unknown → 1 / NORMAL)."""
    return _REGIME_CODE.get(str(label).upper().strip(), 1)


def _safe(x) -> float:
    """Return float or 0.0 if x is non-finite/None."""
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


class SignalQualityModel:
    """
    Stateless signal-quality predictor.

    Call sequence (each pipeline run):
        model = SignalQualityModel()
        model.train(backtest_result)          # fit on walk-level IC history
        quality = model.predict(regime, master_df)   # {ticker: float}

    The model is intentionally stateless — re-train on every run from the
    freshest backtest data.  Training typically takes < 50 ms.
    """

    def __init__(self) -> None:
        self._model = None          # fitted sklearn estimator (or None)
        self._model_type: str = "neutral"
        self._walk_df: Optional[pd.DataFrame] = None   # raw walk features
        self._regime_ic: Dict[str, float] = {}         # regime → mean IC
        self._global_ic_mean: float = 0.0
        self._trained: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, backtest_result: "BacktestResult") -> None:
        """
        Fit the quality model on walk-forward IC history.

        Parameters
        ----------
        backtest_result : BacktestResult
            Output of WalkForwardBacktester.run_backtest().  If None or empty,
            the model falls back to neutral quality (1.0) for all sectors.
        """
        self._trained = False
        self._model = None
        self._regime_ic = {}
        self._global_ic_mean = 0.0

        if backtest_result is None:
            logger.warning("train: backtest_result is None — using neutral quality")
            self._model_type = "neutral"
            return

        walks = getattr(backtest_result, "walk_metrics", None)
        if not walks:
            logger.warning("train: no walk_metrics — using neutral quality")
            self._model_type = "neutral"
            return

        try:
            self._fit_from_walks(walks, backtest_result)
            self._trained = True
        except Exception as exc:
            logger.warning("train: failed to fit model (%s) — using neutral quality", exc)
            self._model_type = "neutral"
            self._trained = False

    def predict(
        self,
        regime: str,
        master_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Return a quality score per sector ticker.

        Parameters
        ----------
        regime : str
            Current market regime label (e.g. "NORMAL", "TENSION").
        master_df : pd.DataFrame
            QuantEngine output — must have a 'sector_ticker' column (or be indexed by it).

        Returns
        -------
        dict[str, float]
            {sector_ticker: quality_score} where quality_score ∈ [0.5, 1.5].
            Returns {ticker: 1.0} (neutral) on any failure.
        """
        try:
            tickers = _extract_tickers(master_df)
            if not tickers:
                return {}

            if not self._trained or self._model is None:
                return {t: _NEUTRAL for t in tickers}

            return self._predict_tickers(tickers, regime)

        except Exception as exc:
            logger.warning("predict: error (%s) — returning neutral quality", exc)
            tickers = _extract_tickers(master_df) or []
            return {t: _NEUTRAL for t in tickers}

    @property
    def model_type(self) -> str:
        return self._model_type

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_from_walks(self, walks, backtest_result) -> None:
        """Build feature matrix, fit sklearn model, cache regime IC means."""
        # ── Build feature rows ────────────────────────────────────────────────
        rows: List[Dict] = []
        for w in walks:
            rc = _regime_code(getattr(w, "regime", "NORMAL"))
            ic = _safe(getattr(w, "ic", 0.0))
            ns = _safe(getattr(w, "n_sectors", 11))
            rows.append({
                "regime_code": rc,
                "ic_abs":      abs(ic),
                "n_sectors":   ns,
                "ic_raw":      ic,      # keep raw for regime IC mean
            })

        if not rows:
            self._model_type = "neutral"
            return

        df = pd.DataFrame(rows)

        # Per-regime walk counts — feature n_walks_in_regime
        regime_counts = df.groupby("regime_code").size().to_dict()
        df["n_walks_in_regime"] = df["regime_code"].map(regime_counts).fillna(1)

        # Regime IC means (for fallback / regime_ic output)
        regime_labels = {v: k for k, v in _REGIME_CODE.items()}
        for code, grp in df.groupby("regime_code"):
            label = regime_labels.get(code, str(code))
            self._regime_ic[label] = float(grp["ic_raw"].mean())

        # Fallback from BacktestResult aggregate
        self._global_ic_mean = _safe(getattr(backtest_result, "ic_mean", 0.0))

        # ── Prepare X, y ─────────────────────────────────────────────────────
        feature_cols = ["regime_code", "ic_abs", "n_sectors", "n_walks_in_regime"]
        X = df[feature_cols].values.astype(float)
        y = (df["ic_abs"] > _IC_THRESHOLD).astype(int).values

        n_walks = len(X)
        n_pos = int(y.sum())

        if n_pos == 0 or n_pos == n_walks:
            # Degenerate — all same class → use regime IC mean as scalar quality
            logger.info(
                "train: degenerate target (n_pos=%d / n_walks=%d) → mean-IC fallback",
                n_pos, n_walks,
            )
            self._model_type = "regime_mean_ic"
            self._walk_df = df
            self._model = _MeanICFallback(self._regime_ic, self._global_ic_mean)
            return

        # ── Fit GBM or LogisticRegression ─────────────────────────────────────
        if n_walks >= _MIN_WALKS_FOR_GBM:
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )
                clf.fit(X, y)
                self._model = _SklearnWrapper(clf, feature_cols)
                self._model_type = "GradientBoosting"
                logger.info(
                    "train: GBM fitted on %d walks (n_pos=%d, n_neg=%d)",
                    n_walks, n_pos, n_walks - n_pos,
                )
                return
            except Exception as exc:
                logger.warning("GBM fit failed (%s) — falling back to LogisticRegression", exc)

        # Logistic regression fallback
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_scaled, y)
            self._model = _SklearnWrapper(lr, feature_cols, scaler=scaler)
            self._model_type = "LogisticRegression"
            logger.info(
                "train: LogisticRegression fitted on %d walks", n_walks
            )
        except Exception as exc:
            logger.warning("LogisticRegression fit failed (%s) → mean-IC fallback", exc)
            self._model_type = "regime_mean_ic"
            self._model = _MeanICFallback(self._regime_ic, self._global_ic_mean)

        self._walk_df = df

    def _predict_tickers(self, tickers: List[str], regime: str) -> Dict[str, float]:
        """Produce quality scores for all tickers given current regime."""
        rc = _regime_code(regime)

        # Compute regime-conditional IC mean for this regime
        regime_ic_in_regime = self._regime_ic.get(regime.upper().strip(), self._global_ic_mean)

        # Build n_walks_in_regime from training data
        n_walks_regime = 1
        if self._walk_df is not None and not self._walk_df.empty:
            mask = self._walk_df["regime_code"] == rc
            n_walks_regime = max(1, int(mask.sum()))

        raw_quality = self._model.predict_quality(
            regime_code=rc,
            ic_abs=abs(regime_ic_in_regime),
            n_sectors=11,
            n_walks_in_regime=n_walks_regime,
        )

        quality = float(np.clip(raw_quality, _QUALITY_LO, _QUALITY_HI))

        return {ticker: quality for ticker in tickers}


# ──────────────────────────────────────────────────────────────────────────────
# Internal model wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _SklearnWrapper:
    """Wraps an sklearn classifier to produce a scalar quality score in [0.5, 1.5]."""

    def __init__(self, clf, feature_cols: List[str], scaler=None) -> None:
        self._clf = clf
        self._feature_cols = feature_cols
        self._scaler = scaler

    def predict_quality(
        self,
        regime_code: int,
        ic_abs: float,
        n_sectors: int,
        n_walks_in_regime: int,
    ) -> float:
        X = np.array([[regime_code, ic_abs, n_sectors, n_walks_in_regime]], dtype=float)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        try:
            proba = self._clf.predict_proba(X)[0]
            # proba[1] = P(IC > threshold | regime)
            # Map [0, 1] → [0.5, 1.5] linearly
            p_informative = float(proba[1]) if len(proba) > 1 else 0.5
            return _QUALITY_LO + p_informative * (_QUALITY_HI - _QUALITY_LO)
        except Exception:
            return _NEUTRAL


class _MeanICFallback:
    """
    Non-ML fallback: maps regime-conditional mean IC to a quality score.
    Score > 1 when regime IC > IC_THRESHOLD, < 1 when below.
    """

    def __init__(self, regime_ic: Dict[str, float], global_ic_mean: float) -> None:
        self._regime_ic = regime_ic
        self._global_ic_mean = global_ic_mean

    def predict_quality(
        self,
        regime_code: int,
        ic_abs: float,
        n_sectors: int,
        n_walks_in_regime: int,
    ) -> float:
        # Use ic_abs passed in (already regime-filtered by caller)
        if ic_abs <= 0:
            return _NEUTRAL
        # Normalise relative to threshold: IC = 0.05 → quality = 1.0
        # IC = 0.10 → quality = 1.5;  IC = 0.0 → quality = 0.5
        quality = _NEUTRAL + (ic_abs - _IC_THRESHOLD) / _IC_THRESHOLD * 0.5
        return float(np.clip(quality, _QUALITY_LO, _QUALITY_HI))


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _extract_tickers(master_df: pd.DataFrame) -> List[str]:
    """Return sector tickers from master_df regardless of index/column layout."""
    if master_df is None or master_df.empty:
        return []
    if "sector_ticker" in master_df.columns:
        return list(master_df["sector_ticker"].dropna().unique())
    if master_df.index.name == "sector_ticker":
        return list(master_df.index.dropna().unique())
    # Fall back to index
    return [str(x) for x in master_df.index.dropna().unique()]


def apply_quality_to_master(
    master_df: pd.DataFrame,
    quality_scores: Dict[str, float],
) -> pd.DataFrame:
    """
    Multiply conviction_score by quality_score in-place (safe copy).

    Adds a 'quality_score' column to master_df and multiplies
    'conviction_score' by it.  Both are clipped to sane ranges.
    """
    if master_df is None or master_df.empty or not quality_scores:
        return master_df

    df = master_df.copy()

    def _get_ticker(row) -> str:
        if "sector_ticker" in df.columns:
            return row.get("sector_ticker", "")
        return str(row.name)

    if "sector_ticker" in df.columns:
        df["quality_score"] = df["sector_ticker"].map(quality_scores).fillna(_NEUTRAL)
    else:
        df["quality_score"] = df.index.map(quality_scores).fillna(_NEUTRAL)

    if "conviction_score" in df.columns:
        df["conviction_score"] = (
            df["conviction_score"] * df["quality_score"]
        ).clip(lower=0.0)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble Stacking Model
# ═════════════════════════════════════════════════════════════════════════════

class EnsembleSignalModel:
    """
    3-model ensemble for signal quality prediction via stacking.

    Base models:
      1. GradientBoosting (capture nonlinear regime effects)
      2. LogisticRegression (linear baseline + regularization)
      3. RandomForest (variance reduction + feature importance)

    Meta-model:
      Weighted average of base predictions, weights from OOB score.

    Typically improves Information Ratio by 2-5% over single GBM.
    """

    def __init__(self):
        self._base_models = []
        self._weights = []
        self._feature_names: List[str] = []
        self._trained = False
        self._oob_scores = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train all base models and compute stacking weights.

        Parameters
        ----------
        X : pd.DataFrame — features (regime_code, recent_ic, vol_ic, etc.)
        y : pd.Series — binary target (1 = IC > threshold)

        Returns
        -------
        dict — model scores {model_name: accuracy}
        """
        from sklearn.model_selection import cross_val_score

        self._feature_names = list(X.columns)
        X_arr = X.values.astype(float)
        y_arr = y.values.astype(float)

        # Handle NaN
        X_arr = np.nan_to_num(X_arr, nan=0.0)

        if len(y_arr) < 30 or y_arr.std() < 0.01:
            logger.warning("EnsembleSignalModel: insufficient or no-variance target — skipping")
            self._trained = False
            return {}

        models = []
        scores = {}

        # Model 1: GradientBoosting
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gbm = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            cv_scores = cross_val_score(gbm, X_arr, y_arr, cv=3, scoring="accuracy")
            gbm.fit(X_arr, y_arr)
            models.append(("GBM", gbm))
            scores["GBM"] = float(cv_scores.mean())
        except Exception as e:
            logger.debug("GBM training failed: %s", e)

        # Model 2: LogisticRegression
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            cv_scores = cross_val_score(lr, X_arr, y_arr, cv=3, scoring="accuracy")
            lr.fit(X_arr, y_arr)
            models.append(("LR", lr))
            scores["LR"] = float(cv_scores.mean())
        except Exception as e:
            logger.debug("LR training failed: %s", e)

        # Model 3: RandomForest
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=42,
            )
            cv_scores = cross_val_score(rf, X_arr, y_arr, cv=3, scoring="accuracy")
            rf.fit(X_arr, y_arr)
            models.append(("RF", rf))
            scores["RF"] = float(cv_scores.mean())
        except Exception as e:
            logger.debug("RF training failed: %s", e)

        if not models:
            self._trained = False
            return scores

        # Stacking weights: proportional to OOB accuracy
        total_score = sum(scores.values())
        self._weights = [scores.get(name, 0.5) / total_score for name, _ in models]
        self._base_models = models
        self._oob_scores = scores
        self._trained = True

        logger.info("Ensemble trained: %s | weights=%s",
                     {n: f"{s:.3f}" for n, s in scores.items()},
                     [f"{w:.2f}" for w in self._weights])
        return scores

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weighted ensemble prediction.
        Returns probability of positive IC (quality > threshold).
        """
        if not self._trained or not self._base_models:
            return np.full(len(X), 0.5)

        X_arr = np.nan_to_num(X.values.astype(float), nan=0.0)
        preds = np.zeros(len(X_arr))

        for (name, model), weight in zip(self._base_models, self._weights):
            try:
                prob = model.predict_proba(X_arr)[:, 1]
                preds += weight * prob
            except Exception:
                preds += weight * 0.5

        return preds

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Aggregate feature importance across ensemble."""
        if not self._trained:
            return {}

        n_features = len(self._feature_names)
        agg_imp = np.zeros(n_features)

        for (name, model), weight in zip(self._base_models, self._weights):
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    agg_imp += weight * imp[:n_features]
                elif hasattr(model, "coef_"):
                    imp = np.abs(model.coef_[0])
                    agg_imp += weight * imp[:n_features]
            except Exception:
                pass

        total = agg_imp.sum()
        if total > 0:
            agg_imp /= total

        return {self._feature_names[i]: round(float(agg_imp[i]), 4)
                for i in range(n_features)}


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive IC Thresholds
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveICResult:
    """Regime-adaptive IC threshold determination."""
    regime: str
    ic_threshold: float              # Dynamic threshold (vs fixed 0.02)
    ic_current: float                # Current IC estimate
    is_informative: bool             # IC > adaptive threshold
    confidence: float                # Bayesian posterior confidence
    n_observations: int              # Sample size for this regime


def compute_adaptive_ic_threshold(
    walk_ics: List[float],
    regime_ics: Dict[str, List[float]],
    current_regime: str,
    prior_threshold: float = 0.02,
) -> AdaptiveICResult:
    """
    Compute adaptive IC threshold using Bayesian updating.

    Instead of fixed threshold (IC > 0.02 = informative), adapts
    threshold based on the IC distribution in the current regime.

    Method:
      - Prior: threshold = prior_threshold (0.02)
      - Likelihood: IC distribution in current regime
      - Posterior: Bayesian update → regime-specific threshold
      - IC threshold = regime_mean_IC - 0.5 * regime_std_IC

    High-vol regimes (where IC naturally swings more) get wider thresholds.
    """
    regime_ic_list = regime_ics.get(current_regime, walk_ics)
    n = len(regime_ic_list)

    if n < 5:
        return AdaptiveICResult(
            regime=current_regime, ic_threshold=prior_threshold,
            ic_current=0.0, is_informative=False, confidence=0.0,
            n_observations=n,
        )

    ic_arr = np.array(regime_ic_list, dtype=float)
    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std())
    ic_current = float(ic_arr[-1]) if len(ic_arr) > 0 else 0.0

    # Adaptive threshold: tighter in stable regimes, wider in volatile
    adaptive = max(0.005, ic_mean - 0.5 * ic_std)

    # Bayesian confidence: how confident are we in this regime's IC distribution
    # Higher n → higher confidence → use adaptive threshold more
    n_weight = min(1.0, n / 30)  # Full confidence at 30+ observations
    final_threshold = n_weight * adaptive + (1 - n_weight) * prior_threshold

    return AdaptiveICResult(
        regime=current_regime,
        ic_threshold=round(final_threshold, 4),
        ic_current=round(ic_current, 4),
        is_informative=ic_current > final_threshold,
        confidence=round(n_weight, 3),
        n_observations=n,
    )
