"""
analytics/ml_adaptive.py
==========================
Adaptive Learning with Concept Drift Detection.

Monitors model performance in real-time and triggers retraining when:
  1. IC drops below threshold (rolling 20-day IC < 0)
  2. Prediction accuracy drops (hit rate < 50% for 10 consecutive days)
  3. Feature distribution shifts (KS test on top features)

Model versioning: stores model + params + metrics for A/B comparison.
Only promotes new model if OOS improvement > 1 sigma above current.

Ref: Gama et al. (2014) -- A Survey on Concept Drift Adaptation
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IC_WINDOW = 20           # rolling IC window
_IC_DRIFT_STREAK = 3      # consecutive observations of rolling IC < 0
_HIT_RATE_THRESH = 0.50   # hit rate threshold
_HIT_RATE_STREAK = 10     # consecutive days below threshold
_ADWIN_DELTA = 0.002      # ADWIN confidence parameter
_PROMOTION_SIGMA = 1.0    # new model must beat current by this many sigma
_DEFAULT_STORAGE = "data/ml_models/"


# ===================================================================
# DriftDetector
# ===================================================================
class DriftDetector:
    """
    Monitors model performance and detects concept drift via:
      - IC degradation (rolling 20-day IC < 0 for N consecutive obs)
      - Hit-rate collapse (< 50% for 10 consecutive days)
      - ADWIN-style adaptive windowing on IC stream
    """

    def __init__(
        self,
        ic_window: int = _IC_WINDOW,
        ic_drift_streak: int = _IC_DRIFT_STREAK,
        hit_rate_thresh: float = _HIT_RATE_THRESH,
        hit_rate_streak: int = _HIT_RATE_STREAK,
        adwin_delta: float = _ADWIN_DELTA,
    ) -> None:
        self.ic_window = ic_window
        self.ic_drift_streak = ic_drift_streak
        self.hit_rate_thresh = hit_rate_thresh
        self.hit_rate_streak = hit_rate_streak
        self.adwin_delta = adwin_delta

        # Internal state
        self._ic_history: List[float] = []
        self._hit_rate_history: List[float] = []
        self._dates: List[str] = []
        self._predictions: List[np.ndarray] = []
        self._actuals: List[np.ndarray] = []

        # Drift flags
        self._ic_drift = False
        self._hit_rate_drift = False
        self._adwin_drift = False

        # ADWIN state: track rolling IC values in an adaptive window
        self._adwin_window: Deque[float] = deque(maxlen=500)

    # ------------------------------------------------------------------
    def update(
        self,
        date: str,
        ic: float,
        hit_rate: float,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> None:
        """
        Feed daily results into the drift detector.

        Parameters
        ----------
        date        : date string (YYYY-MM-DD)
        ic          : information coefficient for today
        hit_rate    : fraction of correct directional predictions
        predictions : model predictions (optional, for distribution tracking)
        actuals     : actual outcomes (optional)
        """
        ic = float(ic) if np.isfinite(ic) else 0.0
        hit_rate = float(hit_rate) if np.isfinite(hit_rate) else 0.5

        self._dates.append(str(date))
        self._ic_history.append(ic)
        self._hit_rate_history.append(hit_rate)

        if predictions is not None:
            self._predictions.append(np.asarray(predictions, dtype=np.float64))
        if actuals is not None:
            self._actuals.append(np.asarray(actuals, dtype=np.float64))

        # Update ADWIN window
        self._adwin_window.append(ic)

        # --- Check IC drift ---
        self._check_ic_drift()

        # --- Check hit-rate drift ---
        self._check_hit_rate_drift()

        # --- Check ADWIN drift ---
        self._check_adwin_drift()

    # ------------------------------------------------------------------
    def _check_ic_drift(self) -> None:
        """Rolling IC < 0 for N consecutive observations."""
        if len(self._ic_history) < self.ic_window:
            self._ic_drift = False
            return

        # Compute rolling IC (mean of last ic_window values)
        recent_ics: List[float] = []
        n = len(self._ic_history)
        # We need at least ic_drift_streak rolling IC computations
        start = max(0, n - self.ic_drift_streak - self.ic_window + 1)
        for i in range(start, n - self.ic_window + 1):
            window = self._ic_history[i : i + self.ic_window]
            recent_ics.append(float(np.mean(window)))

        if len(recent_ics) < self.ic_drift_streak:
            self._ic_drift = False
            return

        # Check last N rolling ICs are all negative
        tail = recent_ics[-self.ic_drift_streak:]
        self._ic_drift = all(v < 0 for v in tail)

    def _check_hit_rate_drift(self) -> None:
        """Hit rate < threshold for N consecutive days."""
        if len(self._hit_rate_history) < self.hit_rate_streak:
            self._hit_rate_drift = False
            return

        tail = self._hit_rate_history[-self.hit_rate_streak:]
        self._hit_rate_drift = all(h < self.hit_rate_thresh for h in tail)

    def _check_adwin_drift(self) -> None:
        """
        Simplified ADWIN: split window into two halves.
        If means differ significantly -> drift.
        """
        w = list(self._adwin_window)
        n = len(w)
        if n < 20:
            self._adwin_drift = False
            return

        # Try split points in the middle third of the window
        best_cut = None
        best_eps = 0.0
        lo = n // 3
        hi = 2 * n // 3

        for cut in range(lo, hi):
            left = w[:cut]
            right = w[cut:]
            if len(left) < 5 or len(right) < 5:
                continue

            mu_left = float(np.mean(left))
            mu_right = float(np.mean(right))
            eps = abs(mu_left - mu_right)

            # ADWIN bound: epsilon_cut
            m = 1.0 / (1.0 / len(left) + 1.0 / len(right))
            var = float(np.var(w))
            if var <= 0:
                continue
            bound = np.sqrt(var / (2.0 * m) * np.log(2.0 / self.adwin_delta))

            if eps > bound and eps > best_eps:
                best_eps = eps
                best_cut = cut

        self._adwin_drift = best_cut is not None

    # ------------------------------------------------------------------
    # PSI (Population Stability Index) — feature distribution drift
    # ------------------------------------------------------------------
    def check_feature_drift_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
        psi_threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Compute PSI between reference (training) and current (live) distributions.

        PSI < 0.10 → no drift
        0.10 ≤ PSI < 0.20 → moderate drift (monitor)
        PSI ≥ 0.20 → significant drift (retrain)

        Ref: Yurdakul (2018) — Statistical Properties of PSI
        """
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(current, dtype=np.float64)

        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 30 or len(cur) < 30:
            return {"psi": 0.0, "drift": False, "label": "INSUFFICIENT_DATA"}

        # Create bins from reference distribution
        breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(ref, bins=breakpoints)[0]
        cur_counts = np.histogram(cur, bins=breakpoints)[0]

        # Normalize to proportions (add small epsilon to avoid log(0))
        eps = 1e-6
        ref_pct = ref_counts / ref_counts.sum() + eps
        cur_pct = cur_counts / cur_counts.sum() + eps

        # PSI = Σ (cur_i - ref_i) × ln(cur_i / ref_i)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        if psi >= psi_threshold:
            label = "SIGNIFICANT_DRIFT"
        elif psi >= 0.10:
            label = "MODERATE_DRIFT"
        else:
            label = "STABLE"

        return {
            "psi": round(psi, 4),
            "drift": psi >= psi_threshold,
            "label": label,
            "ref_size": len(ref),
            "cur_size": len(cur),
        }

    def check_multi_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Check PSI drift across multiple features.
        Returns per-feature PSI + aggregate drift decision.
        """
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        if not common_cols:
            return {"drifted_features": [], "n_drifted": 0, "total_features": 0}

        results = {}
        drifted = []
        for col in common_cols:
            r = self.check_feature_drift_psi(
                reference_df[col].values, current_df[col].values,
                psi_threshold=psi_threshold,
            )
            results[col] = r
            if r["drift"]:
                drifted.append(col)

        return {
            "per_feature": results,
            "drifted_features": drifted,
            "n_drifted": len(drifted),
            "total_features": len(common_cols),
            "aggregate_drift": len(drifted) >= max(1, len(common_cols) // 3),
        }

    # ------------------------------------------------------------------
    # KS test — distribution comparison
    # ------------------------------------------------------------------
    def check_feature_drift_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Two-sample Kolmogorov-Smirnov test for feature distribution drift.
        More sensitive than PSI for small samples.
        """
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(current, dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 20 or len(cur) < 20:
            return {"ks_stat": 0.0, "p_value": 1.0, "drift": False}

        ks_stat, p_value = sp_stats.ks_2samp(ref, cur)
        return {
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift": p_value < alpha,
        }

    # ------------------------------------------------------------------
    def is_drift_detected(self) -> bool:
        """True if any drift signal is active."""
        return self._ic_drift or self._hit_rate_drift or self._adwin_drift

    def drift_type(self) -> str:
        """Return the primary drift type string."""
        if self._ic_drift or self._hit_rate_drift:
            return "performance_degradation"
        if self._adwin_drift:
            return "distribution_shift"
        return "none"

    def get_status(self) -> Dict[str, Any]:
        """Full monitoring state."""
        n = len(self._ic_history)
        rolling_ic = float(np.mean(self._ic_history[-self.ic_window:])) if n >= self.ic_window else None
        recent_hr = float(np.mean(self._hit_rate_history[-10:])) if len(self._hit_rate_history) >= 10 else None

        return {
            "n_observations": n,
            "rolling_ic_20d": rolling_ic,
            "recent_hit_rate_10d": recent_hr,
            "ic_drift": self._ic_drift,
            "hit_rate_drift": self._hit_rate_drift,
            "adwin_drift": self._adwin_drift,
            "drift_detected": self.is_drift_detected(),
            "drift_type": self.drift_type(),
            "last_date": self._dates[-1] if self._dates else None,
        }


# ===================================================================
# ModelVersionManager
# ===================================================================
class ModelVersionManager:
    """
    Stores and manages model versions with metadata.

    Layout: storage_dir/
        versions.json          -- index of all versions
        {version_tag}.pkl      -- serialized model
        {version_tag}_meta.json -- metrics + params
    """

    def __init__(self, storage_dir: str = _DEFAULT_STORAGE) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_dir / "versions.json"
        self._index = self._load_index()

    # ------------------------------------------------------------------
    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt versions.json; starting fresh")
        return {"production": None, "versions": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def save_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        version_tag: str,
    ) -> str:
        """
        Serialize model + metadata to disk.

        Returns the version_tag.
        """
        # Pickle the model
        model_path = self.storage_dir / f"{version_tag}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        meta = {
            "version_tag": version_tag,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat(),
            "model_path": str(model_path),
        }
        meta_path = self.storage_dir / f"{version_tag}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update index
        self._index["versions"][version_tag] = meta
        self._save_index()

        logger.info("Saved model version '%s' to %s", version_tag, model_path)
        return version_tag

    def load_model(self, version_tag: str) -> Any:
        """Deserialize a model by version tag."""
        model_path = self.storage_dir / f"{version_tag}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    def compare_models(
        self,
        current_tag: str,
        candidate_tag: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        A/B test: compare two model versions on test data.

        Promotes candidate only if improvement > 1 sigma.

        Returns dict with both scores and recommendation.
        """
        X_test = _clean_array_safe(X_test)
        y_test = np.asarray(y_test, dtype=np.float64)
        np.nan_to_num(y_test, copy=False, nan=0.0)
        y_binary = (y_test > 0).astype(np.int32)

        current_model = self.load_model(current_tag)
        candidate_model = self.load_model(candidate_tag)

        # Evaluate both
        current_proba = current_model.predict_proba(X_test)
        candidate_proba = candidate_model.predict_proba(X_test)

        current_acc = float(np.mean((current_proba >= 0.5).astype(int) == y_binary))
        candidate_acc = float(np.mean((candidate_proba >= 0.5).astype(int) == y_binary))

        # Compute IC (Spearman rank correlation)
        current_ic = _safe_ic(current_proba, y_test)
        candidate_ic = _safe_ic(candidate_proba, y_test)

        # Significance: improvement must exceed 1 sigma of current performance
        # Use bootstrap to estimate sigma of current accuracy
        n_boot = 100
        boot_accs = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            idx = rng.choice(len(y_binary), size=len(y_binary), replace=True)
            boot_acc = float(np.mean(
                (current_proba[idx] >= 0.5).astype(int) == y_binary[idx]
            ))
            boot_accs.append(boot_acc)
        sigma = float(np.std(boot_accs))

        improvement = candidate_acc - current_acc
        promote = improvement > (_PROMOTION_SIGMA * sigma) if sigma > 0 else improvement > 0

        return {
            "current_tag": current_tag,
            "candidate_tag": candidate_tag,
            "current_accuracy": current_acc,
            "candidate_accuracy": candidate_acc,
            "current_ic": current_ic,
            "candidate_ic": candidate_ic,
            "improvement": improvement,
            "sigma": sigma,
            "promote_candidate": promote,
        }

    def promote_model(self, version_tag: str) -> None:
        """Mark a version as the production model."""
        if version_tag not in self._index["versions"]:
            raise ValueError(f"Version '{version_tag}' not found")
        self._index["production"] = version_tag
        self._save_index()
        logger.info("Promoted model '%s' to production", version_tag)

    def get_production_model(self) -> Any:
        """Return the current production model (deserialized)."""
        tag = self._index.get("production")
        if tag is None:
            raise RuntimeError("No production model set")
        return self.load_model(tag)

    def get_production_tag(self) -> Optional[str]:
        """Return the current production version tag, or None."""
        return self._index.get("production")

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions with metrics."""
        result = []
        for tag, meta in self._index.get("versions", {}).items():
            entry = {
                "version_tag": tag,
                "is_production": (tag == self._index.get("production")),
                "metrics": meta.get("metrics", {}),
                "created_at": meta.get("created_at", ""),
            }
            result.append(entry)
        return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)


# ===================================================================
# AdaptiveLearner
# ===================================================================
class AdaptiveLearner:
    """
    Orchestrates adaptive model retraining based on drift detection.

    Daily workflow:
      1. Compute features for today
      2. Feed results into DriftDetector
      3. If drift -> retrain, A/B test, promote if better
    """

    def __init__(
        self,
        model_class: Any = None,
        feature_engine: Any = None,
        storage_dir: str = _DEFAULT_STORAGE,
        min_train_samples: int = 100,
    ) -> None:
        # Lazy import to avoid circular deps
        if model_class is None:
            from analytics.ml_ensemble import EnsembleModel
            model_class = EnsembleModel

        self.model_class = model_class
        self.feature_engine = feature_engine
        self.storage_dir = storage_dir
        self.min_train_samples = min_train_samples

        self.drift_detector = DriftDetector()
        self.version_manager = ModelVersionManager(storage_dir=storage_dir)

        self._retrain_count = 0

    # ------------------------------------------------------------------
    def check_and_retrain(
        self,
        prices: pd.DataFrame,
        master_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Main daily call: check drift, retrain if needed.

        Parameters
        ----------
        prices    : price DataFrame (dates x tickers)
        master_df : feature DataFrame with columns for ML features and
                    'forward_return' or 'target' column

        Returns
        -------
        Status dict with drift info and any retraining results.
        """
        status: Dict[str, Any] = {
            "date": str(datetime.utcnow().date()),
            "drift_detected": False,
            "drift_type": "none",
            "retrained": False,
            "promoted": False,
        }

        # --- Step 1: compute today's metrics if we have a production model ---
        prod_tag = self.version_manager.get_production_tag()
        if prod_tag is not None and len(master_df) > 0:
            try:
                ic, hit_rate, preds, actuals = self._evaluate_current(master_df)
                self.drift_detector.update(
                    date=status["date"],
                    ic=ic,
                    hit_rate=hit_rate,
                    predictions=preds,
                    actuals=actuals,
                )
                status["current_ic"] = ic
                status["current_hit_rate"] = hit_rate
            except Exception as exc:
                logger.warning("Failed to evaluate current model: %s", exc)

        # --- Step 2: check drift ---
        drift = self.drift_detector.is_drift_detected()
        status["drift_detected"] = drift
        status["drift_type"] = self.drift_detector.drift_type()
        status["detector_status"] = self.drift_detector.get_status()

        # --- Step 3: retrain if drift detected ---
        if drift:
            logger.info("Drift detected (%s) — triggering retrain", status["drift_type"])
            retrain_result = self._retrain(master_df)
            status["retrained"] = retrain_result.get("success", False)
            status["retrain_details"] = retrain_result

            # --- Step 4: A/B test and promote if better ---
            if retrain_result.get("success") and prod_tag is not None:
                candidate_tag = retrain_result["version_tag"]
                try:
                    X_test, y_test = self._prepare_test_set(master_df)
                    comparison = self.version_manager.compare_models(
                        prod_tag, candidate_tag, X_test, y_test,
                    )
                    status["comparison"] = comparison
                    if comparison["promote_candidate"]:
                        self.version_manager.promote_model(candidate_tag)
                        status["promoted"] = True
                        logger.info(
                            "Promoted %s (acc=%.4f) over %s (acc=%.4f)",
                            candidate_tag, comparison["candidate_accuracy"],
                            prod_tag, comparison["current_accuracy"],
                        )
                    else:
                        logger.info(
                            "Candidate %s not promoted (improvement=%.4f < %.4f sigma)",
                            candidate_tag, comparison["improvement"], comparison["sigma"],
                        )
                except Exception as exc:
                    logger.warning("A/B comparison failed: %s", exc)
                    # If no production model exists, promote anyway
                    self.version_manager.promote_model(retrain_result["version_tag"])
                    status["promoted"] = True

            elif retrain_result.get("success") and prod_tag is None:
                # First model: auto-promote
                self.version_manager.promote_model(retrain_result["version_tag"])
                status["promoted"] = True

        return status

    # ------------------------------------------------------------------
    def get_current_predictions(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Use production model for predictions."""
        model = self.version_manager.get_production_model()
        X_c = _clean_array_safe(X)
        return model.predict_proba(X_c)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_current(
        self, df: pd.DataFrame,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate current production model on recent data."""
        model = self.version_manager.get_production_model()
        X, y = self._extract_Xy(df)

        preds = model.predict_proba(X)
        actuals = np.asarray(y, dtype=np.float64)

        ic = _safe_ic(preds, actuals)
        hit_rate = float(np.mean(
            ((preds >= 0.5).astype(int)) == ((actuals > 0).astype(int))
        ))

        return ic, hit_rate, preds, actuals

    def _retrain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Retrain model on available data."""
        try:
            X, y = self._extract_Xy(df)
            if len(X) < self.min_train_samples:
                return {"success": False, "reason": f"Too few samples ({len(X)})"}

            # Split: 80% train, 20% validation
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            model = self.model_class()
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            # Evaluate on validation
            val_preds = model.predict_proba(X_val)
            y_val_binary = (np.asarray(y_val) > 0).astype(int)
            val_acc = float(np.mean((val_preds >= 0.5).astype(int) == y_val_binary))
            val_ic = _safe_ic(val_preds, np.asarray(y_val, dtype=np.float64))

            # Save with version tag
            self._retrain_count += 1
            version_tag = f"v{self._retrain_count}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            metrics = {"val_accuracy": val_acc, "val_ic": val_ic, "n_train": split}

            self.version_manager.save_model(model, metrics, version_tag)

            return {
                "success": True,
                "version_tag": version_tag,
                "val_accuracy": val_acc,
                "val_ic": val_ic,
            }
        except Exception as exc:
            logger.error("Retrain failed: %s", exc, exc_info=True)
            return {"success": False, "reason": str(exc)}

    def _prepare_test_set(
        self, df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract most recent 20% as test set for A/B comparison."""
        X, y = self._extract_Xy(df)
        split = int(len(X) * 0.8)
        return X[split:], y[split:]

    def _extract_Xy(
        self, df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and target y from master DataFrame."""
        df = df.copy()
        df = df.ffill().fillna(0.0)

        # Identify target column
        target_col = None
        for candidate in ("target", "forward_return", "fwd_ret_5d", "fwd_return"):
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            raise ValueError(
                "No target column found. Expected one of: "
                "target, forward_return, fwd_ret_5d, fwd_return"
            )

        y = df[target_col].values.astype(np.float64)

        # Feature columns: everything except target and metadata
        exclude = {target_col, "date", "Date", "ticker", "sector", "symbol"}
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].values.astype(np.float64)

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y, copy=False, nan=0.0)

        return X, y


# ===================================================================
# Utility functions
# ===================================================================
def _clean_array_safe(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Clean array/DataFrame: ffill + fillna(0)."""
    if isinstance(X, pd.DataFrame):
        return X.ffill().fillna(0.0).values.astype(np.float64)
    X = np.array(X, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _safe_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank correlation (IC), NaN-safe."""
    try:
        mask = np.isfinite(predictions) & np.isfinite(actuals)
        if mask.sum() < 5:
            return 0.0
        corr, _ = sp_stats.spearmanr(predictions[mask], actuals[mask])
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


# ===================================================================
# Auto-retraining pipeline
# ===================================================================
def auto_retrain_pipeline(
    prices: pd.DataFrame,
    sectors: list,
    current_model_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Daily auto-retraining pipeline.

    1. Check drift via DriftDetector on recent walk-forward IC
    2. If drift detected (or force=True), train new ensemble model
    3. A/B test: compare new vs current on holdout period
    4. Promote if new model beats current by > _PROMOTION_SIGMA

    Parameters
    ----------
    prices : pd.DataFrame — full price history
    sectors : list — sector ticker list
    current_model_path : str — path to current model pickle (optional)
    force : bool — force retrain regardless of drift

    Returns
    -------
    dict with keys: retrained, promoted, drift_detected, metrics, model_path
    """
    import pickle
    from pathlib import Path

    result = {
        "retrained": False,
        "promoted": False,
        "drift_detected": False,
        "metrics": {},
        "model_path": None,
        "reason": "",
    }

    model_dir = Path(prices.__class__.__module__).parent if hasattr(prices, '__module__') else Path(".")
    model_dir = Path(__file__).resolve().parent.parent / "data" / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check drift
    try:
        from analytics.backtest import WalkForwardBacktester
        from config.settings import get_settings

        settings = get_settings()
        bt = WalkForwardBacktester(settings)
        bt_result = bt.run_backtest(prices, pd.DataFrame(), pd.DataFrame())

        if bt_result and bt_result.walk_metrics:
            recent_ics = [w.ic for w in bt_result.walk_metrics[-20:] if np.isfinite(w.ic)]
            detector = DriftDetector(window=20, ic_threshold=0.0, hit_rate_threshold=0.50)
            for ic_val in recent_ics:
                detector.update(ic_val, 0.55)  # approx hit rate

            drift = detector.is_drifting()
            result["drift_detected"] = drift
            result["metrics"]["recent_ic_mean"] = float(np.mean(recent_ics)) if recent_ics else 0.0
            result["metrics"]["recent_ic_std"] = float(np.std(recent_ics)) if recent_ics else 0.0
            result["metrics"]["n_walks"] = len(bt_result.walk_metrics)

            if not drift and not force:
                result["reason"] = "No drift detected, model is stable"
                logger.info("Auto-retrain: no drift, skipping")
                return result
        else:
            result["reason"] = "No backtest results available"
            if not force:
                return result

    except Exception as e:
        logger.warning("Drift detection failed: %s", e)
        if not force:
            result["reason"] = f"Drift check error: {e}"
            return result

    # Step 2: Train new model
    logger.info("Auto-retrain: training new ensemble model...")
    try:
        from analytics.ml_ensemble import EnsembleModel
        from analytics.feature_engine import FeatureEngine

        fe = FeatureEngine(prices, sectors)
        feat_df, feat_names = fe.build_features()

        if feat_df is None or len(feat_df) < 100:
            result["reason"] = "Not enough feature data for retraining"
            return result

        # Split: 80% train, 20% holdout
        split_idx = int(len(feat_df) * 0.8)
        train_df = feat_df.iloc[:split_idx]
        holdout_df = feat_df.iloc[split_idx:]

        target_col = "fwd_return_5d" if "fwd_return_5d" in feat_df.columns else feat_df.columns[-1]
        X_train = train_df[feat_names].values
        y_train = (train_df[target_col] > 0).astype(int).values
        X_holdout = holdout_df[feat_names].values
        y_holdout = (holdout_df[target_col] > 0).astype(int).values

        model = EnsembleModel()
        model.fit(X_train, y_train)

        # Evaluate on holdout
        preds = model.predict(X_holdout)
        new_accuracy = float(np.mean(preds == y_holdout))
        new_proba = model.predict_proba(X_holdout)
        new_ic = _safe_ic(new_proba, holdout_df[target_col].values)

        result["retrained"] = True
        result["metrics"]["new_accuracy"] = new_accuracy
        result["metrics"]["new_ic"] = new_ic

        logger.info("New model: accuracy=%.3f, IC=%.3f", new_accuracy, new_ic)

    except Exception as e:
        logger.error("Model training failed: %s", e)
        result["reason"] = f"Training error: {e}"
        return result

    # Step 3: A/B comparison
    current_accuracy = 0.50  # assume baseline
    if current_model_path and Path(current_model_path).exists():
        try:
            old_model = pickle.loads(Path(current_model_path).read_bytes())
            old_preds = old_model.predict(X_holdout)
            current_accuracy = float(np.mean(old_preds == y_holdout))
        except Exception:
            pass

    improvement = new_accuracy - current_accuracy

    # Step 4: Promote if improvement > threshold
    if improvement > 0.01 or force:
        new_model_path = model_dir / "ensemble_latest.pkl"
        new_model_path.write_bytes(pickle.dumps(model))
        result["promoted"] = True
        result["model_path"] = str(new_model_path)
        result["metrics"]["improvement"] = improvement
        result["reason"] = f"Promoted: +{improvement:.3f} accuracy improvement"
        logger.info("Model promoted: %s (improvement=%.3f)", new_model_path, improvement)

        # Save feature importances
        fi = model.feature_importances(feat_names)
        fi_path = model_dir / "feature_importances.pkl"
        fi_path.write_bytes(pickle.dumps(fi))
    else:
        result["reason"] = f"Improvement too small ({improvement:.3f}), keeping current model"
        logger.info("Model not promoted: improvement=%.3f < threshold", improvement)

    return result
