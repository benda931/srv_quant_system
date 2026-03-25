"""
analytics/feature_engine.py
=============================
30+ Alpha Features + Feature Selection (SHAP, Permutation, RFE).

Comprehensive feature generation and selection pipeline for the SRV
Quant System. Generates ~26 features per sector covering momentum,
volatility, mean-reversion, cross-sectional, macro, correlation,
technical, and calendar domains.

Usage:
    from analytics.feature_engine import FeatureEngine

    engine = FeatureEngine(prices, sectors=['XLK', 'XLF', ...], spy_ticker='SPY')
    features = engine.compute_all_features()
    top_features = engine.select_features(X, y, method='shap', top_k=15)
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Generate alpha features from a price panel for sector rotation strategies.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide daily price panel with DatetimeIndex. Must contain columns for
        sector tickers, SPY, and optionally VIX (^VIX), HYG, IEF, DXY.
    sectors : list[str]
        List of sector ticker symbols (e.g., ['XLK', 'XLF', ...]).
    spy_ticker : str
        SPY ticker column name (default 'SPY').
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        sectors: List[str],
        spy_ticker: str = "SPY",
    ):
        self.prices = prices.copy()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index()
        self.sectors = [s for s in sectors if s in self.prices.columns]
        self.spy_ticker = spy_ticker

        if not self.sectors:
            raise ValueError("No valid sector tickers found in prices columns.")

        # Pre-compute log returns
        self.log_returns = np.log(self.prices / self.prices.shift(1))
        self.log_returns = self.log_returns.replace([np.inf, -np.inf], np.nan)

        # Relative returns (vs SPY)
        self.rel_returns = pd.DataFrame(index=self.log_returns.index)
        if spy_ticker in self.log_returns.columns:
            for s in self.sectors:
                self.rel_returns[s] = self.log_returns[s] - self.log_returns[spy_ticker]
        else:
            self.rel_returns = self.log_returns[self.sectors].copy()

    def compute_all_features(self) -> pd.DataFrame:
        """
        Generate all alpha features for each sector.

        Returns a DataFrame with MultiIndex columns: (sector, feature_name).
        Rows are dates. NaN values are forward-filled then zero-filled.

        Feature categories (~38 per sector):
          - Momentum: 5d/10d/21d/63d/126d relative returns (5)
          - Volatility: realized 10d/20d/60d + vol_ratio + vol_rank (5)
          - Mean-reversion: z-score 20d/60d/120d (3)
          - Statistical: hurst, adf, skew, kurtosis, autocorr, max_dd (6)
          - Capture: upside/downside capture vs SPY (2)
          - Cross-sectional: rank, dist_median, z_cross_sector_rank, momentum_breadth (4)
          - Macro: VIX z, credit z, yield curve, DXY mom (4)
          - Macro interaction: vix_x_zscore, credit_x_vol (2)
          - Correlation: rolling corr vs SPY, sector dispersion, avg corr (3)
          - Technical: RSI-14, Bollinger Band width (2)
          - Calendar: month, day_of_week (2)
        """
        features = {}

        for s in self.sectors:
            prefix = s
            feat = pd.DataFrame(index=self.log_returns.index)

            # ----- Momentum (5 features) -----
            for window in [5, 10, 21, 63, 126]:
                feat[f"mom_{window}d"] = self.rel_returns[s].rolling(window).sum()

            # ----- Volatility (5 features) -----
            for window in [10, 20, 60]:
                feat[f"vol_{window}d"] = (
                    self.log_returns[s].rolling(window).std() * np.sqrt(252)
                )

            # Vol ratio: short-term / long-term
            vol_10 = self.log_returns[s].rolling(10).std()
            vol_60 = self.log_returns[s].rolling(60).std()
            feat["vol_ratio"] = vol_10 / vol_60.replace(0, np.nan)

            # Vol rank: percentile rank of 20d vol over trailing 252d
            vol_20 = self.log_returns[s].rolling(20).std()
            feat["vol_rank"] = vol_20.rolling(252).apply(
                lambda x: float(np.sum(x <= x.iloc[-1]) / len(x))
                if len(x) > 0 else np.nan,
                raw=False,
            )

            # ----- Mean-Reversion z-scores (3 features) -----
            cum_rel = self.rel_returns[s].cumsum()
            for window in [20, 60, 120]:
                mu = cum_rel.rolling(window).mean()
                sd = cum_rel.rolling(window).std(ddof=1)
                feat[f"zscore_{window}d"] = (cum_rel - mu) / sd.replace(0, np.nan)

            # ----- Cross-Sectional (2 features) -----
            # Computed after the sector loop (needs all sectors)
            # Placeholder columns
            feat["cs_rank"] = np.nan
            feat["cs_dist_median"] = np.nan

            # ----- Technical: RSI-14 (1 feature) -----
            delta = self.prices[s].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(span=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            feat["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

            # ----- Technical: Bollinger Band width (1 feature) -----
            sma_20 = self.prices[s].rolling(20).mean()
            std_20 = self.prices[s].rolling(20).std()
            feat["bb_width"] = (2 * std_20) / sma_20.replace(0, np.nan)

            # ----- Correlation: rolling corr vs SPY (1 feature) -----
            if self.spy_ticker in self.log_returns.columns:
                feat["corr_spy_60d"] = (
                    self.log_returns[s]
                    .rolling(60)
                    .corr(self.log_returns[self.spy_ticker])
                )
            else:
                feat["corr_spy_60d"] = np.nan

            # ----- Statistical features (6 features) -----
            # Hurst exponent (rolling 120d) via R/S analysis
            feat["hurst_exp"] = self._rolling_hurst(self.log_returns[s], window=120)

            # ADF test statistic (rolling 120d) — simplified OLS version
            feat["adf_stat"] = self._rolling_adf(self.log_returns[s].cumsum(), window=120)

            # Rolling 60d skewness and kurtosis
            feat["skew_60d"] = self.log_returns[s].rolling(60).skew()
            feat["kurt_60d"] = self.log_returns[s].rolling(60).kurt()

            # 5-day return autocorrelation (rolling 120d)
            ret_5d = self.log_returns[s].rolling(5).sum()
            feat["autocorr_5d"] = ret_5d.rolling(120).apply(
                lambda x: pd.Series(x).autocorr(lag=1)
                if len(x) > 5 else np.nan,
                raw=False,
            )

            # Maximum drawdown over last 60 days
            feat["max_dd_60d"] = self._rolling_max_drawdown(self.prices[s], window=60)

            # ----- Capture ratios (2 features) -----
            if self.spy_ticker in self.log_returns.columns:
                feat["upside_capture"] = self._rolling_capture(
                    self.log_returns[s], self.log_returns[self.spy_ticker],
                    window=60, side="up"
                )
                feat["downside_capture"] = self._rolling_capture(
                    self.log_returns[s], self.log_returns[self.spy_ticker],
                    window=60, side="down"
                )
            else:
                feat["upside_capture"] = np.nan
                feat["downside_capture"] = np.nan

            features[s] = feat

        # ----- Cross-Sectional features (computed across all sectors) -----
        # 21d momentum for ranking
        mom_21d = pd.DataFrame(index=self.log_returns.index)
        for s in self.sectors:
            mom_21d[s] = self.rel_returns[s].rolling(21).sum()

        for s in self.sectors:
            # Rank: percentile rank among all sectors
            rank_series = mom_21d.rank(axis=1, pct=True)[s]
            features[s]["cs_rank"] = rank_series

            # Distance from cross-sectional median
            cs_median = mom_21d.median(axis=1)
            features[s]["cs_dist_median"] = mom_21d[s] - cs_median

        # ----- Cross-Sectional: z_cross_sector_rank (1 feature per sector) -----
        zscore_60d_all = pd.DataFrame(index=self.log_returns.index)
        for s in self.sectors:
            zscore_60d_all[s] = features[s]["zscore_60d"]
        zscore_60d_rank = zscore_60d_all.rank(axis=1, pct=True)
        for s in self.sectors:
            features[s]["z_cross_sector_rank"] = zscore_60d_rank[s]

        # ----- Cross-Sectional: momentum_breadth (same for all sectors) -----
        momentum_breadth = (mom_21d > 0).sum(axis=1) / max(len(self.sectors), 1)

        # ----- Sector Dispersion (1 feature, same for all sectors) -----
        sector_disp = self.log_returns[self.sectors].rolling(20).std().mean(axis=1)

        # ----- Average Correlation (1 feature, same for all sectors) -----
        avg_corr = pd.Series(np.nan, index=self.log_returns.index)
        n_sectors = len(self.sectors)
        if n_sectors >= 2:
            for i in range(60, len(self.log_returns)):
                window_data = self.log_returns[self.sectors].iloc[i - 60:i]
                C = window_data.corr()
                iu = np.triu_indices(n_sectors, k=1)
                vals = C.values[iu]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    avg_corr.iloc[i] = float(np.mean(vals))

        # ----- Macro features (4 features, same for all sectors) -----
        macro_feat = self._compute_macro_features()

        # ----- Calendar features (2 features, same for all sectors) -----
        cal_month = pd.Series(
            self.prices.index.month, index=self.prices.index, dtype=float
        )
        cal_dow = pd.Series(
            self.prices.index.dayofweek, index=self.prices.index, dtype=float
        )

        # Merge shared features into each sector
        for s in self.sectors:
            features[s]["sector_dispersion"] = sector_disp
            features[s]["avg_corr"] = avg_corr
            features[s]["momentum_breadth"] = momentum_breadth
            features[s]["month"] = cal_month
            features[s]["day_of_week"] = cal_dow
            for col in macro_feat.columns:
                features[s][col] = macro_feat[col]

            # ----- Macro interaction features (2 features) -----
            features[s]["vix_x_zscore"] = (
                macro_feat["vix_z"] * features[s]["zscore_60d"]
            )
            features[s]["credit_x_vol"] = (
                macro_feat["credit_z"] * features[s]["vol_ratio"]
            )

        # Build final MultiIndex DataFrame
        panels = {}
        for s in self.sectors:
            for col in features[s].columns:
                panels[(s, col)] = features[s][col]

        result = pd.DataFrame(panels)
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["sector", "feature"]
        )

        # Handle NaN: forward-fill then zero-fill
        result = result.ffill().fillna(0)

        logger.info(
            "Features computed: %d dates x %d sector-features (%d sectors x %d features)",
            len(result),
            result.shape[1],
            len(self.sectors),
            result.shape[1] // max(len(self.sectors), 1),
        )

        return result

    # ---------------------------------------------------------------------------
    # Statistical feature helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _rolling_hurst(series: pd.Series, window: int = 120) -> pd.Series:
        """
        Rolling Hurst exponent via R/S analysis.
        H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
        """
        def _hurst_rs(x):
            """Compute Hurst exponent for a single window using R/S analysis."""
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            n = len(x)
            if n < 20:
                return np.nan
            # Use sub-series lengths
            max_k = min(n // 2, 64)
            if max_k < 8:
                return np.nan
            log_rs = []
            log_n = []
            for k in [8, 16, 32, 64]:
                if k > max_k:
                    break
                n_sub = n // k
                if n_sub < 1:
                    continue
                rs_vals = []
                for i in range(n_sub):
                    sub = x[i * k:(i + 1) * k]
                    m = np.mean(sub)
                    y = np.cumsum(sub - m)
                    r = np.max(y) - np.min(y)
                    s = np.std(sub, ddof=1)
                    if s > 0:
                        rs_vals.append(r / s)
                if rs_vals:
                    log_rs.append(np.log(np.mean(rs_vals)))
                    log_n.append(np.log(k))
            if len(log_rs) < 2:
                return np.nan
            # OLS fit: log(R/S) = H * log(n) + c
            log_n = np.array(log_n)
            log_rs = np.array(log_rs)
            slope = np.polyfit(log_n, log_rs, 1)[0]
            return np.clip(slope, 0.0, 1.0)

        return series.rolling(window).apply(_hurst_rs, raw=True)

    @staticmethod
    def _rolling_adf(series: pd.Series, window: int = 120) -> pd.Series:
        """
        Simplified rolling ADF test statistic.
        OLS regression: diff(x) = alpha + beta * lag(x) + epsilon.
        Returns t-stat on beta. More negative → more stationary.
        """
        def _adf_simple(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            n = len(x)
            if n < 20:
                return np.nan
            dx = np.diff(x)
            lag_x = x[:-1]
            n_obs = len(dx)
            # OLS: dx = a + b * lag_x
            X = np.column_stack([np.ones(n_obs), lag_x])
            try:
                beta = np.linalg.lstsq(X, dx, rcond=None)[0]
                resid = dx - X @ beta
                sse = np.sum(resid ** 2)
                mse = sse / max(n_obs - 2, 1)
                var_beta = mse * np.linalg.inv(X.T @ X)
                se_b = np.sqrt(max(var_beta[1, 1], 1e-16))
                return beta[1] / se_b
            except (np.linalg.LinAlgError, ValueError):
                return np.nan

        return series.rolling(window).apply(_adf_simple, raw=True)

    @staticmethod
    def _rolling_max_drawdown(price_series: pd.Series, window: int = 60) -> pd.Series:
        """Rolling maximum drawdown (negative value, 0 = no drawdown)."""
        def _max_dd(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 2 or x[0] <= 0:
                return np.nan
            cummax = np.maximum.accumulate(x)
            dd = (x - cummax) / np.where(cummax > 0, cummax, 1.0)
            return float(np.min(dd))

        return price_series.rolling(window).apply(_max_dd, raw=True)

    @staticmethod
    def _rolling_capture(
        sector_ret: pd.Series,
        bench_ret: pd.Series,
        window: int = 60,
        side: str = "up",
    ) -> pd.Series:
        """
        Rolling upside or downside capture ratio vs benchmark.
        upside_capture = mean(sector_ret | bench_ret > 0) / mean(bench_ret | bench_ret > 0)
        """
        result = pd.Series(np.nan, index=sector_ret.index)
        s_vals = sector_ret.values
        b_vals = bench_ret.values
        for i in range(window, len(s_vals)):
            s_w = s_vals[i - window:i]
            b_w = b_vals[i - window:i]
            mask = np.isfinite(s_w) & np.isfinite(b_w)
            s_w = s_w[mask]
            b_w = b_w[mask]
            if side == "up":
                filt = b_w > 0
            else:
                filt = b_w < 0
            if np.sum(filt) < 5:
                continue
            bench_mean = np.mean(b_w[filt])
            if abs(bench_mean) < 1e-10:
                continue
            result.iloc[i] = np.mean(s_w[filt]) / bench_mean
        return result

    def _compute_macro_features(self) -> pd.DataFrame:
        """
        Compute macro features: VIX z-score, credit z (HYG-IEF),
        yield curve proxy, DXY momentum.
        """
        idx = self.prices.index
        macro = pd.DataFrame(index=idx)

        # VIX z-score (60d)
        vix_col = None
        for candidate in ["^VIX", "VIX"]:
            if candidate in self.prices.columns:
                vix_col = candidate
                break

        if vix_col:
            vix = self.prices[vix_col]
            mu = vix.rolling(60).mean()
            sd = vix.rolling(60).std(ddof=1)
            macro["vix_z"] = (vix - mu) / sd.replace(0, np.nan)
        else:
            macro["vix_z"] = 0.0

        # Credit z: log(HYG) - log(IEF), z-scored over 60d
        hyg_col = "HYG" if "HYG" in self.prices.columns else None
        ief_col = "IEF" if "IEF" in self.prices.columns else None

        if hyg_col and ief_col:
            spread = np.log(self.prices[hyg_col]) - np.log(self.prices[ief_col])
            mu = spread.rolling(60).mean()
            sd = spread.rolling(60).std(ddof=1)
            macro["credit_z"] = (spread - mu) / sd.replace(0, np.nan)
        else:
            macro["credit_z"] = 0.0

        # Yield curve proxy: if ^TNX available, use its 60d z-score
        tnx_col = None
        for candidate in ["^TNX", "TNX"]:
            if candidate in self.prices.columns:
                tnx_col = candidate
                break

        if tnx_col:
            tnx = self.prices[tnx_col]
            mu = tnx.rolling(60).mean()
            sd = tnx.rolling(60).std(ddof=1)
            macro["yield_curve_z"] = (tnx - mu) / sd.replace(0, np.nan)
        else:
            macro["yield_curve_z"] = 0.0

        # DXY momentum: 21d return of DXY index
        dxy_col = None
        for candidate in ["DX-Y.NYB", "DXY", "^DXY"]:
            if candidate in self.prices.columns:
                dxy_col = candidate
                break

        if dxy_col:
            dxy_ret = np.log(self.prices[dxy_col] / self.prices[dxy_col].shift(1))
            macro["dxy_mom"] = dxy_ret.rolling(21).sum()
        else:
            macro["dxy_mom"] = 0.0

        return macro

    # ---------------------------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------------------------

    def select_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: str = "shap",
        top_k: int = 15,
    ) -> List[str]:
        """
        Select top features using the specified method.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (single-level columns: feature names).
        y : pd.Series or np.ndarray
            Target variable (e.g., forward returns).
        method : str
            Selection method: 'shap', 'permutation', 'rfe', or 'correlation'.
        top_k : int
            Number of top features to return.

        Returns
        -------
        list[str]
            Names of the top-k features.
        """
        # Clean inputs
        X_clean = X.copy()
        y_clean = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y.copy()

        # Align and drop NaN
        mask = X_clean.notna().all(axis=1) & y_clean.notna()
        X_clean = X_clean.loc[mask].ffill().fillna(0)
        y_clean = y_clean.loc[mask].ffill().fillna(0)

        if len(X_clean) < 50:
            logger.warning(
                "Too few samples (%d) for feature selection. "
                "Returning all features.", len(X_clean)
            )
            return list(X_clean.columns[:top_k])

        method = method.lower()
        if method == "shap":
            return self._select_shap(X_clean, y_clean, top_k)
        elif method == "permutation":
            return self._select_permutation(X_clean, y_clean, top_k)
        elif method == "rfe":
            return self._select_rfe(X_clean, y_clean, top_k)
        elif method == "correlation":
            return self._select_correlation(X_clean, y_clean, top_k)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'shap', 'permutation', 'rfe', or 'correlation'."
            )

    def _get_tree_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit and return a tree-based model (LightGBM preferred, sklearn fallback)."""
        if _HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        elif _HAS_SKLEARN:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ImportError(
                "Either lightgbm or scikit-learn is required for tree-based "
                "feature selection. Install with: pip install lightgbm scikit-learn"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.values, y.values)
        return model

    def _select_shap(
        self, X: pd.DataFrame, y: pd.Series, top_k: int
    ) -> List[str]:
        """Select features using SHAP TreeExplainer."""
        if not _HAS_SHAP:
            logger.warning(
                "shap not installed. Falling back to permutation importance."
            )
            return self._select_permutation(X, y, top_k)

        model = self._get_tree_model(X, y)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.values)

            # Mean absolute SHAP value per feature
            importance = np.abs(shap_values).mean(axis=0)
            importance_series = pd.Series(importance, index=X.columns)
            importance_series = importance_series.sort_values(ascending=False)

            selected = list(importance_series.head(top_k).index)
            logger.info("SHAP selected %d features", len(selected))
            return selected

        except Exception as e:
            logger.warning("SHAP failed: %s. Falling back to permutation.", e)
            return self._select_permutation(X, y, top_k)

    def _select_permutation(
        self, X: pd.DataFrame, y: pd.Series, top_k: int
    ) -> List[str]:
        """Select features using sklearn permutation importance."""
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for permutation feature selection. "
                "Install with: pip install scikit-learn"
            )

        model = self._get_tree_model(X, y)

        result = permutation_importance(
            model, X.values, y.values,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        importance = pd.Series(result.importances_mean, index=X.columns)
        importance = importance.sort_values(ascending=False)
        selected = list(importance.head(top_k).index)
        logger.info("Permutation selected %d features", len(selected))
        return selected

    def _select_rfe(
        self, X: pd.DataFrame, y: pd.Series, top_k: int
    ) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for RFE. "
                "Install with: pip install scikit-learn"
            )

        if _HAS_LGB:
            base_model = lgb.LGBMRegressor(
                n_estimators=50, max_depth=4, verbose=-1, random_state=42,
            )
        else:
            base_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=4, random_state=42,
            )

        n_features = min(top_k, X.shape[1])
        rfe = RFE(
            estimator=base_model,
            n_features_to_select=n_features,
            step=max(1, (X.shape[1] - n_features) // 5),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfe.fit(X.values, y.values)

        mask = rfe.support_
        selected = list(X.columns[mask])

        # Sort by ranking
        ranking = pd.Series(rfe.ranking_, index=X.columns)
        selected = list(ranking[mask].sort_values().index)

        logger.info("RFE selected %d features", len(selected))
        return selected

    def _select_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int,
        threshold: float = 0.9,
    ) -> List[str]:
        """
        Select features by removing highly correlated pairs (|corr| > threshold),
        then ranking by absolute correlation with target.

        Parameters
        ----------
        threshold : float
            Drop one feature from each pair with |corr| > threshold (default 0.9).
        """
        # Step 1: Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > threshold].tolist()
            if high_corr:
                # Drop the feature with lower correlation to target
                target_corr = X[high_corr + [col]].corrwith(y).abs()
                worst = target_corr.idxmin()
                to_drop.add(worst)

        remaining = [c for c in X.columns if c not in to_drop]
        logger.info(
            "Correlation filter: dropped %d features (|corr| > %.2f), %d remaining",
            len(to_drop), threshold, len(remaining),
        )

        # Step 2: Rank remaining by absolute correlation with target
        target_corr = X[remaining].corrwith(y).abs()
        target_corr = target_corr.sort_values(ascending=False)
        selected = list(target_corr.head(top_k).index)

        logger.info("Correlation selected %d features", len(selected))
        return selected

    # ---------------------------------------------------------------------------
    # Utility: flatten multi-index features for a single sector
    # ---------------------------------------------------------------------------

    @staticmethod
    def flatten_for_sector(
        features: pd.DataFrame, sector: str
    ) -> pd.DataFrame:
        """
        Extract features for a single sector from the MultiIndex DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Output of compute_all_features() with MultiIndex columns.
        sector : str
            Sector ticker to extract.

        Returns
        -------
        pd.DataFrame
            Single-level columns with feature names.
        """
        if isinstance(features.columns, pd.MultiIndex):
            if sector in features.columns.get_level_values(0):
                return features[sector].copy()
        return features.copy()
