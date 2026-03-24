"""
analytics/ml_cross_validation.py
==================================
Purged K-Fold Cross-Validation for financial time series.
Ref: de Prado (2018) -- Advances in Financial Machine Learning, Ch. 7

Prevents information leakage in time-series ML by:
  1. Purging: removing training samples whose labels overlap with the test period
  2. Embargo: adding a gap between train/test to prevent serial correlation leakage

Usage:
    from analytics.ml_cross_validation import PurgedKFold, cross_val_score_purged, nested_cv

    cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    scores = cross_val_score_purged(model, X, y, cv=cv, scoring='sharpe')
    nested = nested_cv(model, X, y, inner_cv=cv, outer_cv=cv)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.base import BaseEstimator, clone
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.metrics import make_scorer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    # Provide stubs so the module can be imported without sklearn
    class BaseCrossValidator:  # type: ignore[no-redef]
        pass
    class BaseEstimator:  # type: ignore[no-redef]
        pass
    def clone(est):  # type: ignore[no-redef]
        raise ImportError("sklearn is required: pip install scikit-learn")
    def make_scorer(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("sklearn is required: pip install scikit-learn")


# ---------------------------------------------------------------------------
# PurgedKFold: sklearn-compatible time-series cross-validator
# ---------------------------------------------------------------------------

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for financial time series.

    Unlike standard KFold, this splitter:
      - Splits data chronologically (no shuffling)
      - Purges training observations that overlap temporally with the test set
      - Adds an embargo gap after the test set to prevent leakage from
        autocorrelated features/labels

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    embargo_pct : float
        Fraction of total samples to use as embargo gap after each test fold
        (default 0.02 = 2%).
    purge_pct : float
        Fraction of total samples to purge from the training set before the
        test fold starts (default 0.0). Set > 0 if labels have a lookahead
        window (e.g., forward returns computed over N days).

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        purge_pct: float = 0.0,
    ):
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for PurgedKFold. "
                "Install with: pip install scikit-learn"
            )
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
        groups=None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test index arrays for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If a DataFrame with DatetimeIndex, indices are
            used for temporal ordering.
        y : array-like, optional
            Target variable (not used for splitting, but accepted for
            sklearn API compatibility).
        groups : ignored

        Yields
        ------
        train_idx : np.ndarray
            Integer indices into X for the training set.
        test_idx : np.ndarray
            Integer indices into X for the test set.
        """
        n_samples = len(X)
        if n_samples < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} with "
                f"n_samples={n_samples}."
            )

        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Test fold boundaries (chronological, non-overlapping)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_start = current
            test_end = current + fold_size

            test_idx = indices[test_start:test_end]

            # Training indices: everything before (test_start - purge) and
            # after (test_end + embargo), excluding the purge/embargo zones
            purge_start = max(0, test_start - purge_size)
            embargo_end = min(n_samples, test_end + embargo_size)

            train_before = indices[:purge_start]
            train_after = indices[embargo_end:]
            train_idx = np.concatenate([train_before, train_after])

            if len(train_idx) == 0:
                logger.warning(
                    "Fold at [%d:%d] has empty training set after purge/embargo. "
                    "Consider reducing embargo_pct or purge_pct.",
                    test_start, test_end,
                )
                current = test_end
                continue

            yield train_idx, test_idx
            current = test_end


# ---------------------------------------------------------------------------
# Scoring functions for financial ML
# ---------------------------------------------------------------------------

def _sharpe_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Sharpe-like score: mean(y_pred * y_true) / std(y_pred * y_true).

    Treats y_pred as position weights and y_true as forward returns.
    Annualised assuming daily frequency (sqrt(252)).
    """
    pnl = np.asarray(y_pred, dtype=float) * np.asarray(y_true, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 2:
        return 0.0
    std = float(np.std(pnl, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(pnl) / std * np.sqrt(252))


def _get_scorer(scoring: str) -> Callable:
    """Map scoring name to a callable (y_true, y_pred) -> float."""
    if scoring == "sharpe":
        return _sharpe_score
    if scoring == "accuracy":
        def accuracy(y_true, y_pred):
            y_t = np.asarray(y_true, dtype=float)
            y_p = np.asarray(y_pred, dtype=float)
            valid = np.isfinite(y_t) & np.isfinite(y_p)
            if valid.sum() == 0:
                return 0.0
            return float(np.mean(np.sign(y_t[valid]) == np.sign(y_p[valid])))
        return accuracy
    if scoring == "mse":
        def mse(y_true, y_pred):
            diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
            diff = diff[np.isfinite(diff)]
            return -float(np.mean(diff ** 2)) if len(diff) > 0 else 0.0
        return mse
    if scoring == "ic":
        def ic(y_true, y_pred):
            from scipy.stats import spearmanr
            y_t = np.asarray(y_true, dtype=float)
            y_p = np.asarray(y_pred, dtype=float)
            valid = np.isfinite(y_t) & np.isfinite(y_p)
            if valid.sum() < 3:
                return 0.0
            rho, _ = spearmanr(y_t[valid], y_p[valid])
            return float(rho) if math.isfinite(rho) else 0.0
        return ic
    raise ValueError(f"Unknown scoring method: {scoring}. Use 'sharpe', 'accuracy', 'mse', or 'ic'.")


# ---------------------------------------------------------------------------
# cross_val_score_purged: convenience wrapper
# ---------------------------------------------------------------------------

def cross_val_score_purged(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: Optional[PurgedKFold] = None,
    scoring: str = "sharpe",
) -> np.ndarray:
    """
    Evaluate a model using purged K-fold cross-validation.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must implement fit(X, y) and predict(X).
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target variable (e.g., forward returns).
    cv : PurgedKFold, optional
        Cross-validator (default: PurgedKFold(n_splits=5, embargo_pct=0.02)).
    scoring : str
        Scoring metric: 'sharpe', 'accuracy', 'mse', or 'ic'.

    Returns
    -------
    np.ndarray
        Array of scores, one per fold.
    """
    if cv is None:
        cv = PurgedKFold(n_splits=5, embargo_pct=0.02)

    scorer = _get_scorer(scoring)
    scores = []

    # Handle NaN in features/target
    X_arr = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X
    y_arr = np.asarray(y, dtype=float) if not isinstance(y, pd.Series) else y

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr)):
        try:
            if isinstance(X_arr, pd.DataFrame):
                X_train = X_arr.iloc[train_idx].copy()
                X_test = X_arr.iloc[test_idx].copy()
            else:
                X_train = X_arr[train_idx].copy()
                X_test = X_arr[test_idx].copy()

            if isinstance(y_arr, pd.Series):
                y_train = y_arr.iloc[train_idx].copy()
                y_test = y_arr.iloc[test_idx].copy()
            else:
                y_train = y_arr[train_idx].copy()
                y_test = y_arr[test_idx].copy()

            # Handle NaN: fill with 0 for training stability
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.ffill().fillna(0)
                X_test = X_test.ffill().fillna(0)
            else:
                X_train = np.nan_to_num(X_train, nan=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0)

            if isinstance(y_train, pd.Series):
                y_train = y_train.ffill().fillna(0)
            else:
                y_train = np.nan_to_num(y_train, nan=0.0)

            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            y_test_arr = np.asarray(y_test, dtype=float)
            score = scorer(y_test_arr, y_pred)
            scores.append(score)
            logger.debug("Fold %d: score=%.4f", fold_idx, score)

        except Exception as e:
            logger.warning("Fold %d failed: %s", fold_idx, e)
            scores.append(float("nan"))

    return np.array(scores, dtype=float)


# ---------------------------------------------------------------------------
# nested_cv: nested cross-validation
# ---------------------------------------------------------------------------

def nested_cv(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    inner_cv: Optional[PurgedKFold] = None,
    outer_cv: Optional[PurgedKFold] = None,
    scoring: str = "sharpe",
    param_grid: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """
    Nested cross-validation with purged K-fold.

    Inner loop: hyperparameter tuning (grid search over param_grid).
    Outer loop: unbiased performance estimation.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must implement fit(X, y), predict(X), and support set_params().
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target variable.
    inner_cv : PurgedKFold, optional
        Inner cross-validator for tuning (default: 3-fold).
    outer_cv : PurgedKFold, optional
        Outer cross-validator for evaluation (default: 5-fold).
    scoring : str
        Scoring metric (default: 'sharpe').
    param_grid : dict, optional
        Parameter grid for inner tuning. Keys are model param names,
        values are lists of values to try. If None, no tuning is performed.

    Returns
    -------
    dict
        {
            'mean_score': float,
            'std_score': float,
            'fold_scores': list[float],
            'best_params_per_fold': list[dict],
        }
    """
    if inner_cv is None:
        inner_cv = PurgedKFold(n_splits=3, embargo_pct=0.02)
    if outer_cv is None:
        outer_cv = PurgedKFold(n_splits=5, embargo_pct=0.02)

    scorer = _get_scorer(scoring)
    fold_scores: List[float] = []
    best_params_per_fold: List[Dict[str, Any]] = []

    X_arr = X if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    y_arr = y if isinstance(y, pd.Series) else np.asarray(y, dtype=float)

    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_arr, y_arr)):
        logger.info("Outer fold %d/%d", outer_idx + 1, outer_cv.n_splits)

        if isinstance(X_arr, pd.DataFrame):
            X_outer_train = X_arr.iloc[outer_train_idx]
            X_outer_test = X_arr.iloc[outer_test_idx]
        else:
            X_outer_train = X_arr[outer_train_idx]
            X_outer_test = X_arr[outer_test_idx]

        if isinstance(y_arr, pd.Series):
            y_outer_train = y_arr.iloc[outer_train_idx]
            y_outer_test = y_arr.iloc[outer_test_idx]
        else:
            y_outer_train = y_arr[outer_train_idx]
            y_outer_test = y_arr[outer_test_idx]

        # Inner loop: find best params
        best_inner_score = -np.inf
        best_params: Dict[str, Any] = {}

        if param_grid:
            # Generate all param combinations
            from itertools import product
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            for combo in product(*values):
                params = dict(zip(keys, combo))
                inner_scores = []

                for inner_train_idx, inner_test_idx in inner_cv.split(
                    X_outer_train, y_outer_train
                ):
                    try:
                        if isinstance(X_outer_train, pd.DataFrame):
                            X_in_train = X_outer_train.iloc[inner_train_idx].ffill().fillna(0)
                            X_in_test = X_outer_train.iloc[inner_test_idx].ffill().fillna(0)
                        else:
                            X_in_train = np.nan_to_num(X_outer_train[inner_train_idx], nan=0.0)
                            X_in_test = np.nan_to_num(X_outer_train[inner_test_idx], nan=0.0)

                        if isinstance(y_outer_train, pd.Series):
                            y_in_train = y_outer_train.iloc[inner_train_idx].ffill().fillna(0)
                            y_in_test = y_outer_train.iloc[inner_test_idx]
                        else:
                            y_in_train = np.nan_to_num(y_outer_train[inner_train_idx], nan=0.0)
                            y_in_test = y_outer_train[inner_test_idx]

                        m = clone(model)
                        m.set_params(**params)
                        m.fit(X_in_train, y_in_train)
                        y_pred = m.predict(X_in_test)
                        s = scorer(np.asarray(y_in_test, dtype=float), y_pred)
                        inner_scores.append(s)
                    except Exception as e:
                        logger.debug("Inner fold failed: %s", e)
                        inner_scores.append(float("nan"))

                valid_scores = [s for s in inner_scores if math.isfinite(s)]
                avg_score = float(np.mean(valid_scores)) if valid_scores else -np.inf
                if avg_score > best_inner_score:
                    best_inner_score = avg_score
                    best_params = params.copy()

        # Train final model on full outer training set with best params
        try:
            final_model = clone(model)
            if best_params:
                final_model.set_params(**best_params)

            if isinstance(X_outer_train, pd.DataFrame):
                X_fit = X_outer_train.ffill().fillna(0)
            else:
                X_fit = np.nan_to_num(X_outer_train, nan=0.0)

            if isinstance(y_outer_train, pd.Series):
                y_fit = y_outer_train.ffill().fillna(0)
            else:
                y_fit = np.nan_to_num(y_outer_train, nan=0.0)

            if isinstance(X_outer_test, pd.DataFrame):
                X_eval = X_outer_test.ffill().fillna(0)
            else:
                X_eval = np.nan_to_num(X_outer_test, nan=0.0)

            final_model.fit(X_fit, y_fit)
            y_pred = final_model.predict(X_eval)
            outer_score = scorer(np.asarray(y_outer_test, dtype=float), y_pred)
            fold_scores.append(outer_score)
            best_params_per_fold.append(best_params)
            logger.info(
                "Outer fold %d: score=%.4f, best_params=%s",
                outer_idx, outer_score, best_params,
            )
        except Exception as e:
            logger.warning("Outer fold %d failed: %s", outer_idx, e)
            fold_scores.append(float("nan"))
            best_params_per_fold.append(best_params)

    valid_scores = [s for s in fold_scores if math.isfinite(s)]
    mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")
    std_score = float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else float("nan")

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_scores": fold_scores,
        "best_params_per_fold": best_params_per_fold,
        "scoring": scoring,
        "n_outer_folds": outer_cv.n_splits,
        "n_inner_folds": inner_cv.n_splits,
    }
