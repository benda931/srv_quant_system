"""
analytics/alpha_model.py
========================
Walk-forward GradientBoosting model for sector selection.

Train on expanding window, predict 5-day forward relative return.
Target: sign(sector_return - spy_return) over next 5 days.

Usage:
    from analytics.alpha_model import WalkForwardAlphaModel
    from analytics.feature_engine import FeatureEngine

    engine = FeatureEngine(prices, sectors)
    model = WalkForwardAlphaModel(prices, sectors, engine)
    results = model.run()
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WalkForwardAlphaModel
# ---------------------------------------------------------------------------

class WalkForwardAlphaModel:
    """
    Walk-forward GradientBoosting model for sector selection.

    Train on expanding window, predict 5-day forward relative return.
    Target: sign(sector_return - spy_return) over next 5 days.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide daily price panel with DatetimeIndex.
    sectors : list[str]
        List of sector tickers.
    feature_engine : FeatureEngine
        Pre-configured FeatureEngine instance.
    train_start : int
        Minimum number of rows before first train (default 252 = ~1 year).
    retrain_freq : int
        Retrain every N trading days (default 21 = ~1 month).
    hold_period : int
        Forward return horizon in days (default 5 = 1 week).
    max_positions : int
        Maximum number of positions to hold (default 4).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        sectors: List[str],
        feature_engine,
        train_start: int = 252,
        retrain_freq: int = 21,
        hold_period: int = 5,
        max_positions: int = 4,
    ):
        self.prices = prices.copy()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index()
        self.sectors = [s for s in sectors if s in self.prices.columns]
        self.feature_engine = feature_engine
        self.train_start = train_start
        self.retrain_freq = retrain_freq
        self.hold_period = hold_period
        self.max_positions = max_positions

        # Pre-compute features and returns
        self.features = feature_engine.compute_all_features()
        self.log_returns = np.log(self.prices / self.prices.shift(1))
        self.log_returns = self.log_returns.replace([np.inf, -np.inf], np.nan)

        spy_col = feature_engine.spy_ticker
        self.spy_returns = self.log_returns[spy_col] if spy_col in self.log_returns.columns else None

    # -----------------------------------------------------------------------
    # Target construction
    # -----------------------------------------------------------------------

    def _build_target(self) -> pd.DataFrame:
        """
        Forward 5d relative return per sector.

        Returns DataFrame with columns = sectors, index = dates.
        Values are the cumulative relative return over the next `hold_period` days.
        """
        targets = pd.DataFrame(index=self.prices.index)
        for s in self.sectors:
            fwd = self.log_returns[s].rolling(self.hold_period).sum().shift(-self.hold_period)
            if self.spy_returns is not None:
                spy_fwd = self.spy_returns.rolling(self.hold_period).sum().shift(-self.hold_period)
                targets[s] = fwd - spy_fwd
            else:
                targets[s] = fwd
        return targets

    # -----------------------------------------------------------------------
    # Feature matrix for a single date slice
    # -----------------------------------------------------------------------

    def _get_feature_matrix(self, date_mask) -> pd.DataFrame:
        """
        Stack feature matrices for all sectors on the given date mask.
        Returns DataFrame with columns = feature names, plus 'sector' and 'date'.
        """
        rows = []
        for s in self.sectors:
            if s not in self.features.columns.get_level_values(0):
                continue
            sect_feat = self.features[s].loc[date_mask].copy()
            sect_feat["sector"] = s
            sect_feat["date"] = sect_feat.index
            rows.append(sect_feat)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    # -----------------------------------------------------------------------
    # Walk-forward engine
    # -----------------------------------------------------------------------

    def _train_predict_walk_forward(self) -> pd.DataFrame:
        """
        Walk-forward: at each retrain point:
        1. Train GBM on all data up to t
        2. Predict for next retrain_freq days
        3. Track predictions and actuals

        Returns DataFrame with columns:
        date, sector, prediction, actual, signal_direction
        """
        targets = self._build_target()
        dates = self.features.index
        n_dates = len(dates)

        feature_cols = [c for c in self.features[self.sectors[0]].columns]

        all_predictions = []

        # Retrain points
        retrain_points = list(range(self.train_start, n_dates - self.hold_period, self.retrain_freq))

        for t_idx in retrain_points:
            # --- Build training data up to t_idx ---
            train_dates = dates[:t_idx]
            train_rows = []
            train_targets = []

            for s in self.sectors:
                if s not in self.features.columns.get_level_values(0):
                    continue
                sf = self.features[s].loc[train_dates].copy()
                tgt = targets[s].loc[train_dates]
                valid = sf.notna().all(axis=1) & tgt.notna()
                sf = sf.loc[valid]
                tgt = tgt.loc[valid]
                if len(sf) == 0:
                    continue
                train_rows.append(sf[feature_cols])
                train_targets.append(tgt)

            if not train_rows:
                continue

            X_train = pd.concat(train_rows, ignore_index=True).fillna(0)
            y_train_raw = pd.concat(train_targets, ignore_index=True).fillna(0)

            # Classification target: sign of forward relative return
            y_train_cls = (y_train_raw > 0).astype(int)

            if len(X_train) < 50 or y_train_cls.nunique() < 2:
                continue

            # --- Feature selection: top 50 by GBM importance ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quick_gbm = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )
                quick_gbm.fit(X_train.values, y_train_cls.values)

            importances = pd.Series(quick_gbm.feature_importances_, index=feature_cols)
            top_features = list(importances.nlargest(min(50, len(feature_cols))).index)

            X_train_sel = X_train[top_features]

            # --- Train classifier ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=20, random_state=42,
                )
                clf.fit(X_train_sel.values, y_train_cls.values)

            # --- Train regressor for magnitude ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=20, random_state=42,
                )
                reg.fit(X_train_sel.values, y_train_raw.values)

            # --- Predict for next retrain_freq days ---
            pred_end = min(t_idx + self.retrain_freq, n_dates - self.hold_period)
            pred_dates = dates[t_idx:pred_end]

            for s in self.sectors:
                if s not in self.features.columns.get_level_values(0):
                    continue
                sf = self.features[s].loc[pred_dates][top_features].fillna(0)
                tgt = targets[s].loc[pred_dates]

                if len(sf) == 0:
                    continue

                prob = clf.predict_proba(sf.values)
                # probability of class 1 (positive return)
                prob_up = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
                magnitude = reg.predict(sf.values)

                # Combined signal: probability-weighted magnitude
                combined = prob_up * np.abs(magnitude)
                direction = (prob_up > 0.5).astype(int) * 2 - 1  # +1 or -1

                for i, d in enumerate(pred_dates):
                    actual_val = tgt.loc[d] if d in tgt.index and pd.notna(tgt.loc[d]) else np.nan
                    all_predictions.append({
                        "date": d,
                        "sector": s,
                        "prediction": float(combined[i]),
                        "prob_up": float(prob_up[i]),
                        "magnitude": float(magnitude[i]),
                        "actual": float(actual_val) if pd.notna(actual_val) else np.nan,
                        "signal_direction": int(direction[i]),
                    })

            # Store feature importance from last train
            self._last_feature_importance = dict(zip(
                top_features,
                clf.feature_importances_.tolist(),
            ))
            self._last_clf = clf
            self._last_reg = reg
            self._last_top_features = top_features

        if not all_predictions:
            return pd.DataFrame(columns=["date", "sector", "prediction", "actual", "signal_direction"])

        return pd.DataFrame(all_predictions)

    # -----------------------------------------------------------------------
    # Main runner
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run full walk-forward and return:
        - sharpe: Sharpe ratio of the strategy
        - ic_mean: Mean IC (rank correlation of prediction vs actual)
        - hit_rate: % of correct direction predictions
        - feature_importance: dict of feature name -> importance
        - equity_curve: pd.Series of cumulative returns
        - predictions: full DataFrame of predictions
        """
        predictions = self._train_predict_walk_forward()

        if predictions.empty:
            return {
                "sharpe": 0.0,
                "ic_mean": 0.0,
                "hit_rate": 0.0,
                "feature_importance": {},
                "equity_curve": pd.Series(dtype=float),
                "predictions": predictions,
            }

        # --- Compute strategy returns ---
        # For each date, pick top max_positions sectors by prediction score
        valid = predictions.dropna(subset=["actual"]).copy()

        if valid.empty:
            return {
                "sharpe": 0.0,
                "ic_mean": 0.0,
                "hit_rate": 0.0,
                "feature_importance": getattr(self, "_last_feature_importance", {}),
                "equity_curve": pd.Series(dtype=float),
                "predictions": predictions,
            }

        daily_returns = []
        dates_used = sorted(valid["date"].unique())

        for d in dates_used:
            day_preds = valid[valid["date"] == d].copy()
            if day_preds.empty:
                continue

            # Top sectors by prediction score
            day_preds = day_preds.sort_values("prediction", ascending=False)
            top = day_preds.head(self.max_positions)

            # Equal-weight the top picks, scaled by confidence
            total_conf = top["prob_up"].sum()
            if total_conf > 0:
                weights = top["prob_up"] / total_conf
            else:
                weights = pd.Series(1.0 / len(top), index=top.index)

            # Strategy return: weighted average of actual returns
            day_ret = (top["actual"].values * weights.values).sum()
            daily_returns.append({"date": d, "return": day_ret})

        if not daily_returns:
            return {
                "sharpe": 0.0,
                "ic_mean": 0.0,
                "hit_rate": 0.0,
                "feature_importance": getattr(self, "_last_feature_importance", {}),
                "equity_curve": pd.Series(dtype=float),
                "predictions": predictions,
            }

        ret_df = pd.DataFrame(daily_returns).set_index("date")["return"]

        # --- Sharpe ratio (annualized) ---
        # Returns are already per hold_period, so annualize accordingly
        periods_per_year = 252 / self.hold_period
        mean_ret = ret_df.mean()
        std_ret = ret_df.std()
        sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        # --- Information Coefficient (rank correlation per date) ---
        ic_values = []
        for d in dates_used:
            day_data = valid[valid["date"] == d]
            if len(day_data) >= 3:
                ic = day_data["prediction"].corr(day_data["actual"], method="spearman")
                if pd.notna(ic):
                    ic_values.append(ic)
        ic_mean = float(np.mean(ic_values)) if ic_values else 0.0

        # --- Hit rate ---
        valid_for_hr = valid.dropna(subset=["actual"])
        correct = ((valid_for_hr["signal_direction"] > 0) & (valid_for_hr["actual"] > 0)) | \
                  ((valid_for_hr["signal_direction"] < 0) & (valid_for_hr["actual"] < 0))
        hit_rate = float(correct.mean()) if len(correct) > 0 else 0.0

        # --- Equity curve ---
        equity = (1 + ret_df).cumprod()

        return {
            "sharpe": round(sharpe, 4),
            "ic_mean": round(ic_mean, 4),
            "hit_rate": round(hit_rate, 4),
            "feature_importance": getattr(self, "_last_feature_importance", {}),
            "equity_curve": equity,
            "predictions": predictions,
        }

    # -----------------------------------------------------------------------
    # Live signals
    # -----------------------------------------------------------------------

    def get_current_signals(self) -> dict:
        """
        Return current live signals: {sector: score}
        Trained on ALL available data, predict today.
        """
        targets = self._build_target()
        feature_cols = list(self.features[self.sectors[0]].columns)
        dates = self.features.index

        # Build full training set
        train_rows = []
        train_targets = []
        for s in self.sectors:
            if s not in self.features.columns.get_level_values(0):
                continue
            sf = self.features[s].copy()
            tgt = targets[s]
            valid = sf.notna().all(axis=1) & tgt.notna()
            sf = sf.loc[valid]
            tgt = tgt.loc[valid]
            if len(sf) == 0:
                continue
            train_rows.append(sf[feature_cols])
            train_targets.append(tgt)

        if not train_rows:
            return {s: 0.0 for s in self.sectors}

        X_train = pd.concat(train_rows, ignore_index=True).fillna(0)
        y_train = pd.concat(train_targets, ignore_index=True).fillna(0)
        y_cls = (y_train > 0).astype(int)

        if len(X_train) < 50 or y_cls.nunique() < 2:
            return {s: 0.0 for s in self.sectors}

        # Feature selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quick_gbm = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            quick_gbm.fit(X_train.values, y_cls.values)

        importances = pd.Series(quick_gbm.feature_importances_, index=feature_cols)
        top_features = list(importances.nlargest(min(50, len(feature_cols))).index)

        X_sel = X_train[top_features]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )
            clf.fit(X_sel.values, y_cls.values)

        # Predict for today (last date)
        signals = {}
        last_date = dates[-1]
        for s in self.sectors:
            if s not in self.features.columns.get_level_values(0):
                signals[s] = 0.0
                continue
            sf = self.features[s].loc[[last_date]][top_features].fillna(0)
            prob = clf.predict_proba(sf.values)
            prob_up = float(prob[0, 1]) if prob.shape[1] > 1 else float(prob[0, 0])
            # Score: centered probability (0.5 = neutral)
            signals[s] = round(prob_up - 0.5, 4)

        return signals
