"""
analytics/ml_ensemble.py
==========================
3-Model Stacking Ensemble for sector signal prediction.

Models:
  1. LightGBM -- gradient boosting (non-linear, high capacity)
  2. Ridge Regression -- L2-regularized linear (low variance baseline)
  3. Random Forest -- bagged trees (variance reduction)

Meta-learner: Logistic Regression on OOS predictions from each model.
Output: conviction score [0, 1] per sector.

Anti-overfitting:
  - Early stopping (patience=10) on LightGBM
  - L2 regularization (alpha=1.0) on Ridge
  - Max depth=4, min_samples_leaf=50 on RF
  - Feature subsampling (0.7 per split)
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Optional: LightGBM (falls back to sklearn GradientBoosting)
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
LGBM_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

RIDGE_PARAMS: Dict[str, Any] = {
    "alpha": 1.0,
}

RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 50,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}

N_FOLDS = 5


def _clean_array(X: np.ndarray) -> np.ndarray:
    """Forward-fill then zero-fill any remaining NaNs."""
    if isinstance(X, pd.DataFrame):
        X = X.ffill().fillna(0.0).values
    else:
        X = np.array(X, dtype=np.float64)
        # Column-wise ffill equivalent for raw arrays
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                idx = np.where(~mask, np.arange(len(mask)), 0)
                np.maximum.accumulate(idx, out=idx)
                X[:, col] = X[idx, col]
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _clean_target(y: np.ndarray) -> np.ndarray:
    """Ensure binary target with NaN -> 0."""
    y = np.asarray(y, dtype=np.float64)
    np.nan_to_num(y, copy=False, nan=0.0)
    return (y > 0).astype(np.int32)


# ===================================================================
# EnsembleModel
# ===================================================================
class EnsembleModel:
    """
    Stacking ensemble: LightGBM + Ridge + RandomForest.

    Meta-learner: Logistic Regression trained on out-of-sample
    predictions from each base model (K-fold stacking).
    """

    def __init__(
        self,
        lgbm_params: Optional[Dict] = None,
        ridge_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        n_folds: int = N_FOLDS,
    ) -> None:
        self.lgbm_params = {**LGBM_PARAMS, **(lgbm_params or {})}
        self.ridge_params = {**RIDGE_PARAMS, **(ridge_params or {})}
        self.rf_params = {**RF_PARAMS, **(rf_params or {})}
        self.n_folds = n_folds

        # Models (populated after fit)
        self._lgbm: Any = None
        self._ridge: Any = None
        self._rf: Any = None
        self._meta: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None

        # Feature names (for importance)
        self._feature_names: List[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Internal: build base models
    # ------------------------------------------------------------------
    def _make_lgbm(self) -> Any:
        """Create LightGBM or fallback GradientBoosting classifier."""
        if _HAS_LGBM:
            params = {k: v for k, v in self.lgbm_params.items()
                      if k not in ("objective", "metric")}
            return lgb.LGBMClassifier(**params)
        else:
            logger.info("lightgbm not installed; falling back to sklearn GradientBoosting")
            return GradientBoostingClassifier(
                n_estimators=self.lgbm_params.get("n_estimators", 200),
                max_depth=self.lgbm_params.get("max_depth", 4),
                learning_rate=self.lgbm_params.get("learning_rate", 0.05),
                subsample=self.lgbm_params.get("subsample", 0.8),
                min_samples_leaf=self.lgbm_params.get("min_child_samples", 50),
                random_state=42,
            )

    def _make_ridge(self) -> RidgeClassifier:
        return RidgeClassifier(alpha=self.ridge_params["alpha"])

    def _make_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.rf_params)

    # ------------------------------------------------------------------
    # Internal: predict proba helper (Ridge has no predict_proba)
    # ------------------------------------------------------------------
    @staticmethod
    def _base_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for any base model."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Handle single-class edge case
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]
        # RidgeClassifier: use decision function -> sigmoid
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> "EnsembleModel":
        """
        Train all 3 base models via K-fold stacking, then fit meta-learner.

        Parameters
        ----------
        X_train : array-like (N, D)
        y_train : array-like (N,) binary or continuous (>0 -> 1)
        X_val   : optional validation set (used for LightGBM early stopping)
        y_val   : optional validation labels
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        X = _clean_array(X_train)
        y = _clean_target(y_train)

        n_samples, n_features = X.shape
        if n_samples < self.n_folds * 2:
            logger.warning(
                "Too few samples (%d) for %d-fold stacking; using 2 folds",
                n_samples, self.n_folds,
            )
            self.n_folds = max(2, min(self.n_folds, n_samples // 2))

        # Scale features for Ridge
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # ----- K-fold OOS predictions for stacking -----
        oos_preds = np.zeros((n_samples, 3), dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_v = X[train_idx], X[val_idx]
            X_tr_s, X_v_s = X_scaled[train_idx], X_scaled[val_idx]
            y_tr = y[train_idx]

            # Train fold models
            fold_lgbm = self._make_lgbm()
            fold_ridge = self._make_ridge()
            fold_rf = self._make_rf()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # LightGBM with early stopping
                if _HAS_LGBM and X_val is not None and y_val is not None:
                    X_val_c = _clean_array(X_val)
                    y_val_c = _clean_target(y_val)
                    fold_lgbm.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val_c, y_val_c)],
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                    )
                else:
                    fold_lgbm.fit(X_tr, y_tr)

                fold_ridge.fit(X_tr_s, y_tr)
                fold_rf.fit(X_tr, y_tr)

            # OOS predictions for this fold
            oos_preds[val_idx, 0] = self._base_predict_proba(fold_lgbm, X_v)
            oos_preds[val_idx, 1] = self._base_predict_proba(fold_ridge, X_v_s)
            oos_preds[val_idx, 2] = self._base_predict_proba(fold_rf, X_v)

        # ----- Train final base models on full data -----
        self._lgbm = self._make_lgbm()
        self._ridge = self._make_ridge()
        self._rf = self._make_rf()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _HAS_LGBM and X_val is not None and y_val is not None:
                X_val_c = _clean_array(X_val)
                y_val_c = _clean_target(y_val)
                self._lgbm.fit(
                    X, y,
                    eval_set=[(X_val_c, y_val_c)],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                )
            else:
                self._lgbm.fit(X, y)

            self._ridge.fit(X_scaled, y)
            self._rf.fit(X, y)

        # ----- Train meta-learner on OOS predictions -----
        self._meta = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, random_state=42,
        )
        self._meta.fit(oos_preds, y)

        self._is_fitted = True
        logger.info(
            "EnsembleModel fitted: %d samples, %d features, %d folds",
            n_samples, n_features, self.n_folds,
        )
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return P(positive return) per row as array of shape (N,)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_c = _clean_array(X)
        X_scaled = self._scaler.transform(X_c)  # type: ignore[union-attr]

        p_lgbm = self._base_predict_proba(self._lgbm, X_c)
        p_ridge = self._base_predict_proba(self._ridge, X_scaled)
        p_rf = self._base_predict_proba(self._rf, X_c)

        stacked = np.column_stack([p_lgbm, p_ridge, p_rf])
        return self._meta.predict_proba(stacked)[:, 1]  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return binary predictions (threshold = 0.5)."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int32)

    # ------------------------------------------------------------------
    # feature_importance
    # ------------------------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """
        Averaged feature importance across all base models.

        LightGBM/RF: uses built-in feature_importances_.
        Ridge: absolute value of coefficients (normalized).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n = len(self._feature_names)
        combined = np.zeros(n, dtype=np.float64)

        # LightGBM / GradientBoosting
        lgbm_imp = np.array(self._lgbm.feature_importances_, dtype=np.float64)
        lgbm_total = lgbm_imp.sum()
        if lgbm_total > 0:
            lgbm_imp /= lgbm_total
        combined += lgbm_imp

        # Ridge: absolute coefficient values, normalized
        ridge_imp = np.abs(self._ridge.coef_).flatten()
        ridge_total = ridge_imp.sum()
        if ridge_total > 0:
            ridge_imp /= ridge_total
        combined += ridge_imp

        # Random Forest
        rf_imp = np.array(self._rf.feature_importances_, dtype=np.float64)
        rf_total = rf_imp.sum()
        if rf_total > 0:
            rf_imp /= rf_total
        combined += rf_imp

        # Average across 3 models
        combined /= 3.0

        return {name: float(combined[i]) for i, name in enumerate(self._feature_names)}

    # ------------------------------------------------------------------
    # get_model_weights
    # ------------------------------------------------------------------
    def get_model_weights(self) -> Dict[str, float]:
        """
        Meta-learner coefficients: how much weight each base model gets.

        Returns dict with keys 'lgbm', 'ridge', 'rf'.
        """
        if not self._is_fitted or self._meta is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        coefs = self._meta.coef_[0]
        return {
            "lgbm": float(coefs[0]),
            "ridge": float(coefs[1]),
            "rf": float(coefs[2]),
        }
