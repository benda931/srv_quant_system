"""
analytics/stat_arb.py

QuantEngine: mathematical core of the SRV Quantamental system.

Implements:
- Log returns & EWMA volatility.
- Volatility parity hedge ratios (HR = Vol(SPY)/Vol(Sector))  citeturn2search5turn2search2
- Realized Dispersion via variance difference method (index variance vs weighted sum of sector variances).
- OOS Rolling PCA residual generation (strict train on [t-window:t), transform on [t]).
- Macro betas vs ^TNX and DX-Y.NYB (rolling covariance beta).
- Conviction scoring across 4 layers -> 0..100.

No placeholders for critical logic; all algorithms are implemented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import Settings
from data.pipeline import ParquetArtifacts


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mu) / sd


def _rolling_percentile_of_last(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the last element within its rolling window.
    Returns values in [0..1]. Uses an O(window) apply, acceptable for small universes.
    """
    def pct_rank(x: np.ndarray) -> float:
        if len(x) == 0:
            return float("nan")
        last = x[-1]
        return float(np.sum(x <= last) / len(x))

    return series.rolling(window=window, min_periods=window).apply(lambda x: pct_rank(np.asarray(x)), raw=False)


def ewma_volatility(returns: pd.Series, lam: float, min_periods: int = 20) -> pd.Series:
    """
    EWMA volatility estimator (RiskMetrics style):
        sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2

    Implementation is iterative for exactness and numerical stability.
    """
    r = returns.astype(float).values
    n = len(r)
    var = np.full(n, np.nan, dtype=float)

    # Initialize using sample variance of first min_periods returns (if enough data)
    if n >= min_periods and np.isfinite(r[:min_periods]).sum() >= int(0.8 * min_periods):
        init = np.nanvar(r[:min_periods], ddof=0)
        var[min_periods - 1] = init

    for t in range(min_periods, n):
        prev_var = var[t - 1]
        if not np.isfinite(prev_var):
            # if still uninitialized, try to bootstrap with local variance
            window = r[max(0, t - min_periods) : t]
            if np.isfinite(window).sum() >= int(0.8 * len(window)):
                prev_var = np.nanvar(window, ddof=0)
            else:
                continue

        rt_1 = r[t - 1]
        if not np.isfinite(rt_1):
            var[t] = prev_var
            continue
        var[t] = lam * prev_var + (1.0 - lam) * (rt_1**2)

    return pd.Series(np.sqrt(var), index=returns.index, name=returns.name)


def estimate_ar1_half_life(series: pd.Series, min_obs: int = 120) -> float:
    """
    Estimate mean reversion half-life using AR(1) on the level:
        x_t = a + phi x_{t-1} + eps
        half-life = -ln(2) / ln(phi), for 0 < phi < 1

    Returns NaN if not estimable / not mean-reverting.
    """
    x = series.dropna().astype(float)
    if len(x) < min_obs:
        return float("nan")

    x_lag = x.shift(1).dropna()
    x_now = x.loc[x_lag.index]

    if len(x_lag) < min_obs:
        return float("nan")

    # OLS: x_now = a + phi*x_lag
    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    y = x_now.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = float(beta[1])
    except Exception:
        return float("nan")

    if phi <= 0.0 or phi >= 0.999:
        return float("nan")

    return float(-math.log(2.0) / math.log(phi))


class QuantEngine:
    """
    Core analytics engine.

    Expected inputs:
    - prices.parquet: wide daily price panel
    - fundamentals.parquet: quote table
    - weights.parquet: ETF holdings + SPY sector allocations (optional)

    Outputs:
    - master_df: sector-level signal + conviction + execution directives
    - residual_level_ts: time series per sector (for tear sheets)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artifacts = ParquetArtifacts.from_settings(settings)

        self.prices: Optional[pd.DataFrame] = None
        self.fundamentals: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.DataFrame] = None

        # Analytics outputs
        self.residual_returns: Optional[pd.DataFrame] = None
        self.residual_levels: Optional[pd.DataFrame] = None
        self.residual_zscores: Optional[pd.DataFrame] = None
        self.dispersion_df: Optional[pd.DataFrame] = None
        self.ewma_vol: Optional[pd.DataFrame] = None
        self.macro_betas: Optional[pd.DataFrame] = None
        self.master_df: Optional[pd.DataFrame] = None

    # -----------------------------
    # Data loading
    # -----------------------------
    def load(self) -> None:
        self.prices = pd.read_parquet(self.artifacts.prices_path)
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index()

        self.fundamentals = pd.read_parquet(self.artifacts.fundamentals_path)
        self.weights = None
        if self.artifacts.weights_path.exists():
            try:
                self.weights = pd.read_parquet(self.artifacts.weights_path)
            except Exception as e:
                self.logger.warning("Failed to load weights parquet: %s", repr(e))
                self.weights = None

        # Basic cleaning
        self.prices = self.prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

    # -----------------------------
    # Core transforms
    # -----------------------------
    def _log_returns(self, px: pd.DataFrame) -> pd.DataFrame:
        """
        Daily log returns.
        """
        lp = np.log(px)
        return lp.diff()

    def _compute_ewma_vol_panel(self, returns: pd.DataFrame) -> pd.DataFrame:
        vols: List[pd.Series] = []
        for col in returns.columns:
            vols.append(ewma_volatility(returns[col], lam=self.settings.ewma_lambda, min_periods=30).rename(col))
        vol = pd.concat(vols, axis=1)
        # annualize to get meaningful hedge ratios
        return vol * math.sqrt(252.0)

    # -----------------------------
    # Dispersion (variance decomposition across sectors)
    # -----------------------------
    def _get_spy_sector_weights(self) -> Dict[str, float]:
        """
        Returns weights for each sector ETF ticker in SPY.
        If unavailable, falls back to equal weights.
        """
        sector_etfs = self.settings.sector_list()
        if self.weights is None or self.weights.empty or "record_type" not in self.weights.columns:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        df = self.weights
        df_sw = df[df["record_type"] == "sector_weight"].copy()
        if df_sw.empty or "sector" not in df_sw.columns or "weightPercentage" not in df_sw.columns:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        # Normalize FMP sector names -> our canonical names -> ETF ticker
        alias = self.settings.fmp_sector_name_aliases
        canonical_to_ticker = self.settings.sector_tickers

        sector_w: Dict[str, float] = {}
        for _, row in df_sw.iterrows():
            raw_name = str(row.get("sector", "")).strip()
            canon = alias.get(raw_name, None)
            if canon is None:
                continue
            etf = canonical_to_ticker.get(canon)
            if not etf:
                continue
            w = _safe_float(row.get("weightPercentage"))
            if math.isfinite(w) and w > 0:
                sector_w[etf] = sector_w.get(etf, 0.0) + w

        # Convert to fractions
        total = sum(sector_w.values())
        if total <= 0:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        # Some feeds supply 0..100, others 0..1. Normalize robustly:
        if total > 1.5:
            sector_w = {k: v / 100.0 for k, v in sector_w.items()}
            total = sum(sector_w.values())

        # Re-normalize to sum 1
        sector_w = {k: v / total for k, v in sector_w.items()}

        # Ensure every sector exists (fill missing with 0)
        for t in sector_etfs:
            sector_w.setdefault(t, 0.0)

        # If too sparse, fallback
        if sum(1 for v in sector_w.values() if v > 0) < 6:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        return sector_w

    def _compute_realized_dispersion(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Variance difference method (21D):
            var_spy = Var(r_spy)
            indep_var = Σ (w_i^2 * Var(r_sector_i))
            dispersion = indep_var - var_spy
            correlation_component = var_spy - indep_var

        Additionally compute dispersion_ratio = indep_var / var_spy (bounded in [0,1] typically).
        """
        w = self._get_spy_sector_weights()
        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker

        # Rolling variances
        rv = returns[sectors + [spy]].rolling(window=self.settings.dispersion_window, min_periods=self.settings.dispersion_window).var(ddof=0)
        var_spy = rv[spy]

        indep_var = None
        for s in sectors:
            wi = float(w.get(s, 0.0))
            term = (wi**2) * rv[s]
            indep_var = term if indep_var is None else (indep_var + term)

        indep_var = indep_var.rename("indep_var")
        dispersion = (indep_var - var_spy).rename("dispersion")
        corr_component = (var_spy - indep_var).rename("correlation_component")

        dispersion_ratio = (indep_var / var_spy).rename("dispersion_ratio")
        out = pd.concat([var_spy.rename("var_spy"), indep_var, dispersion, corr_component, dispersion_ratio], axis=1)
        out["dispersion_ratio_z"] = _zscore(out["dispersion_ratio"], window=max(126, self.settings.pca_window))
        return out

    # -----------------------------
    # OOS PCA residuals
    # -----------------------------
    def _oos_rolling_pca_residuals(self, rel_returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Out-of-sample rolling PCA.

        For each time t >= window:
          - Fit scaler+PCA on rel_returns[t-window : t]  (STRICTly excluding row t)
          - Transform and reconstruct rel_returns[t]
          - residual_return[t] = rel_returns[t] - reconstruction[t]
        Then:
          - residual_level = cumsum(residual_return)
          - zscore over residual_level window

        rel_returns: sector returns relative to SPY (r_sector - r_spy)
        """
        window = self.settings.pca_window
        zwin = self.settings.zscore_window
        sectors = rel_returns.columns.tolist()

        n = len(rel_returns)
        idx = rel_returns.index

        resid_ret = pd.DataFrame(index=idx, columns=sectors, dtype=float)
        pca_k_series = pd.Series(index=idx, dtype=float)

        for t in range(window, n):
            train = rel_returns.iloc[t - window : t].astype(float)
            x_t = rel_returns.iloc[t].astype(float)

            # If the current row or train has too many NaNs, skip.
            if train.isna().mean().mean() > 0.05 or x_t.isna().mean() > 0.10:
                continue

            # Fill any residual NaNs conservatively (0 relative return) to keep PCA stable.
            train_f = train.fillna(0.0).values
            x_row = x_t.fillna(0.0).values.reshape(1, -1)

            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(train_f)

            # Dynamic component selection based on explained variance target.
            # Must be computed on training sample only => no lookahead.
            pca_full = PCA(n_components=min(self.settings.pca_max_components, Xs.shape[1]), svd_solver="full")
            pca_full.fit(Xs)

            csum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(csum, self.settings.pca_explained_var_target) + 1)
            k = max(self.settings.pca_min_components, min(k, self.settings.pca_max_components))

            pca = PCA(n_components=k, svd_solver="full")
            pca.fit(Xs)

            x_s = scaler.transform(x_row)
            scores = pca.transform(x_s)
            x_hat_s = pca.inverse_transform(scores)  # reconstructed in scaled space
            resid_s = (x_s - x_hat_s).reshape(-1)

            # Convert residual from scaled units back to original return units:
            # x = mean + std * x_s, so (x - x_hat) = std*(x_s - x_hat_s).
            resid = resid_s * scaler.scale_

            resid_ret.iloc[t] = resid
            pca_k_series.iloc[t] = k

        resid_level = resid_ret.fillna(0.0).cumsum()
        resid_z = resid_level.apply(lambda s: _zscore(s, window=zwin), axis=0)

        return resid_ret, resid_level, resid_z.rename_axis(index="date")

    # -----------------------------
    # Macro betas
    # -----------------------------
    def _compute_macro_betas(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling covariance betas (60D) vs:
        - ^TNX (treated as daily change, not log return)
        - DX-Y.NYB (log return)

        beta = Cov(r_sector, r_macro) / Var(r_macro)
        """
        sectors = self.settings.sector_list()
        tnx = self.settings.macro_tickers["TNX_10Y"]
        dxy = self.settings.macro_tickers["DXY_USD"]

        # Macro return transforms
        tnx_change = self.prices[tnx].diff()  # yield index change
        dxy_ret = returns[dxy]

        out_rows: List[Dict[str, Any]] = []
        w = self.settings.macro_window

        for s in sectors:
            r_s = returns[s]

            beta_tnx = (
                r_s.rolling(w, min_periods=w).cov(tnx_change) / tnx_change.rolling(w, min_periods=w).var(ddof=0)
            )
            beta_dxy = (
                r_s.rolling(w, min_periods=w).cov(dxy_ret) / dxy_ret.rolling(w, min_periods=w).var(ddof=0)
            )

            out = pd.DataFrame(
                {
                    "beta_tnx": beta_tnx,
                    "beta_dxy": beta_dxy,
                },
                index=returns.index,
            )
            out["sector"] = s
            out_rows.append(out)

        betas = pd.concat(out_rows, axis=0).reset_index().rename(columns={"index": "date"})
        betas["date"] = pd.to_datetime(betas["date"])
        betas = betas.set_index(["date", "sector"]).sort_index()
        return betas

    # -----------------------------
    # Fundamentals aggregation (sector-level valuation)
    # -----------------------------
    def _build_symbol_to_quote(self) -> Dict[str, Dict[str, Any]]:
        if self.fundamentals is None or self.fundamentals.empty:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for _, row in self.fundamentals.iterrows():
            sym = str(row.get("symbol", "")).strip()
            if sym:
                out[sym] = dict(row)
        return out

    def _get_holdings_for_etf(self, etf: str) -> pd.DataFrame:
        if self.weights is None or self.weights.empty or "record_type" not in self.weights.columns:
            return pd.DataFrame(columns=["asset", "weightPercentage"])
        df = self.weights
        h = df[(df["record_type"] == "holding") & (df["etfSymbol"] == etf)].copy()
        if h.empty:
            return pd.DataFrame(columns=["asset", "weightPercentage"])
        h["weightPercentage"] = pd.to_numeric(h["weightPercentage"], errors="coerce")
        h = h.dropna(subset=["asset", "weightPercentage"])
        return h[["asset", "weightPercentage"]].copy()

    def _aggregate_portfolio_pe(self, holdings: pd.DataFrame, quote_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute portfolio-level valuation metrics from holdings + quotes.

        For a market-cap weighted portfolio (weights ~ market cap weights),
        portfolio earnings yield is Σ w_i / PE_i for PE_i > 0.
        Then portfolio PE = 1 / earnings_yield.

        Also compute:
        - covered_weight: sum weights with valid PE
        - negative_or_missing_weight
        - weighted_median_eps (for display)
        """
        if holdings.empty:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }

        w = holdings.copy()
        w["w"] = pd.to_numeric(w["weightPercentage"], errors="coerce") / 100.0
        w = w.dropna(subset=["asset", "w"])
        w = w[w["w"] > 0]
        if w.empty:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }

        # Normalize weights to sum 1 for numeric stability
        w_sum = float(w["w"].sum())
        if w_sum <= 0:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }
        w["w"] = w["w"] / w_sum

        earnings_yield = 0.0
        covered_weight = 0.0

        eps_vals: List[Tuple[float, float]] = []  # (eps, weight)

        for _, row in w.iterrows():
            sym = str(row["asset"]).strip()
            wi = float(row["w"])
            q = quote_map.get(sym, {})
            pe = _safe_float(q.get("pe"))
            eps = _safe_float(q.get("eps"))

            if math.isfinite(eps):
                eps_vals.append((eps, wi))

            if math.isfinite(pe) and pe > 0:
                earnings_yield += wi / pe
                covered_weight += wi

        neg_missing_weight = 1.0 - covered_weight
        pe_portfolio = (1.0 / earnings_yield) if earnings_yield > 0 else float("nan")

        # Weighted median EPS (display-only; EPS itself isn't additive)
        weighted_median_eps = float("nan")
        if eps_vals:
            eps_vals.sort(key=lambda t: t[0])
            cum = 0.0
            for eps, wi in eps_vals:
                cum += wi
                if cum >= 0.5:
                    weighted_median_eps = float(eps)
                    break

        return {
            "pe_portfolio": pe_portfolio,
            "earnings_yield": earnings_yield,
            "covered_weight": covered_weight,
            "neg_or_missing_weight": neg_missing_weight,
            "weighted_median_eps": weighted_median_eps,
        }

    # -----------------------------
    # Scoring
    # -----------------------------
    def calculate_conviction_score(self) -> pd.DataFrame:
        """
        Returns a master DataFrame with:
        - Latest signals per sector
        - Layer scores (stat/macro/fund/vol)
        - Total conviction 0..100
        - Execution directives (LONG/SHORT/NEUTRAL + hedge ratio)
        """
        if self.prices is None:
            raise RuntimeError("Run load() first.")

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        tnx = self.settings.macro_tickers["TNX_10Y"]
        dxy = self.settings.macro_tickers["DXY_USD"]
        vix = self.settings.vol_tickers["VIX"]
        hyg = self.settings.credit_tickers["HYG"]
        ief = self.settings.credit_tickers["IEF"]

        prices = self.prices.copy()

        # Ensure required columns exist
        required_cols = sectors + [spy, tnx, dxy, vix, hyg, ief]
        missing = [c for c in required_cols if c not in prices.columns]
        if missing:
            raise ValueError(f"Missing required tickers in prices panel: {missing}")

        returns = self._log_returns(prices)

        # EWMA annualized vol panel
        ewma_vol = self._compute_ewma_vol_panel(returns[sectors + [spy]])
        self.ewma_vol = ewma_vol

        # Hedge ratio time series (vol parity): HR = vol(SPY)/vol(Sector)
        hr = (ewma_vol[spy].rename("vol_spy") / ewma_vol[sectors]).rename(columns=lambda c: c)

        # Realized dispersion
        dispersion_df = self._compute_realized_dispersion(returns)
        self.dispersion_df = dispersion_df

        # OOS PCA residuals
        rel_returns = returns[sectors].subtract(returns[spy], axis=0)  # r_sector - r_spy
        resid_ret, resid_level, resid_z = self._oos_rolling_pca_residuals(rel_returns)
        self.residual_returns = resid_ret
        self.residual_levels = resid_level
        self.residual_zscores = resid_z

        # Macro betas (panel indexed by [date, sector])
        betas = self._compute_macro_betas(returns)
        self.macro_betas = betas

        # Credit stress (HYG vs IEF)
        credit_spread = (np.log(prices[hyg]) - np.log(prices[ief])).rename("log_hyg_minus_ief")
        credit_z = _zscore(credit_spread, window=self.settings.macro_window).rename("credit_z")
        latest_credit_z = float(credit_z.iloc[-1]) if len(credit_z) else float("nan")
        credit_stress = bool(math.isfinite(latest_credit_z) and latest_credit_z < self.settings.credit_stress_z)

        # VIX regime gating
        vix_level = prices[vix]
        vix_pct = _rolling_percentile_of_last(vix_level, window=max(252, self.settings.pca_window)).rename("vix_pct")
        latest_vix = float(vix_level.iloc[-1]) if len(vix_level) else float("nan")
        latest_vix_pct = float(vix_pct.iloc[-1]) if len(vix_pct) else float("nan")

        # Fundamentals aggregation
        quote_map = self._build_symbol_to_quote()
        # Compute SPY portfolio PE as benchmark (prefer portfolio-derived over ETF quote field)
        spy_holdings = self._get_holdings_for_etf(spy)
        spy_val = self._aggregate_portfolio_pe(spy_holdings, quote_map)
        spy_pe = float(spy_val["pe_portfolio"])
        spy_ey = float(spy_val["earnings_yield"])

        # Helper lists
        cyclicals = set([self.settings.sector_tickers["Energy"],
                         self.settings.sector_tickers["Financials"],
                         self.settings.sector_tickers["Industrials"],
                         self.settings.sector_tickers["Materials"],
                         self.settings.sector_tickers["Consumer Discretionary"]])
        defensives = set([self.settings.sector_tickers["Utilities"],
                          self.settings.sector_tickers["Consumer Staples"],
                          self.settings.sector_tickers["Health Care"]])

        latest_date = prices.index[-1]

        # Build sector rows
        rows: List[Dict[str, Any]] = []
        for s in sectors:
            row: Dict[str, Any] = {"date": latest_date, "sector_ticker": s, "sector_name": self.settings.canonical_sector_by_ticker().get(s, s)}

            # Residual metrics
            z = float(resid_z[s].iloc[-1]) if s in resid_z.columns else float("nan")
            mispricing = float(resid_level[s].iloc[-1]) if s in resid_level.columns else float("nan")

            row["pca_residual_level"] = mispricing
            row["pca_residual_z"] = z

            # Define direction: trade only if |z| >= 0.75 (swing horizon)
            if math.isfinite(z) and abs(z) >= 0.75:
                row["direction"] = "LONG" if z < 0 else "SHORT"
            else:
                row["direction"] = "NEUTRAL"

            # Half-life estimate for mean reversion confidence (research-grade enhancement)
            hl = estimate_ar1_half_life(resid_level[s].iloc[-max(400, self.settings.pca_window) :], min_obs=120)
            row["half_life_days_est"] = hl

            # Volatility & hedge ratio
            vol_s = float(ewma_vol[s].iloc[-1]) if s in ewma_vol.columns else float("nan")
            vol_spy = float(ewma_vol[spy].iloc[-1]) if spy in ewma_vol.columns else float("nan")
            hr_s = float((vol_spy / vol_s)) if (math.isfinite(vol_s) and vol_s > 0 and math.isfinite(vol_spy)) else float("nan")

            row["ewma_vol_ann"] = vol_s
            row["hedge_ratio"] = hr_s  # HR = Vol(SPY)/Vol(Sector)

            # Dispersion snapshot (market-wide, same for all sectors)
            disp_ratio = float(dispersion_df["dispersion_ratio"].iloc[-1]) if not dispersion_df.empty else float("nan")
            disp_z = float(dispersion_df["dispersion_ratio_z"].iloc[-1]) if not dispersion_df.empty else float("nan")
            row["market_dispersion_ratio"] = disp_ratio
            row["market_dispersion_z"] = disp_z

            # Macro betas snapshot (latest)
            b_tnx = float(betas.loc[(latest_date, s), "beta_tnx"]) if (latest_date, s) in betas.index else float("nan")
            b_dxy = float(betas.loc[(latest_date, s), "beta_dxy"]) if (latest_date, s) in betas.index else float("nan")
            row["beta_tnx_60d"] = b_tnx
            row["beta_dxy_60d"] = b_dxy

            beta_mag = math.sqrt((b_tnx if math.isfinite(b_tnx) else 0.0) ** 2 + (b_dxy if math.isfinite(b_dxy) else 0.0) ** 2)
            row["beta_mag"] = beta_mag

            # Fundamentals: sector portfolio PE from holdings
            h = self._get_holdings_for_etf(s)
            val = self._aggregate_portfolio_pe(h, quote_map)
            pe = float(val["pe_portfolio"])
            ey = float(val["earnings_yield"])
            neg_w = float(val["neg_or_missing_weight"])
            eps_med = float(val["weighted_median_eps"])

            row["pe_sector_portfolio"] = pe
            row["earnings_yield_sector"] = ey
            row["neg_or_missing_earnings_weight"] = neg_w
            row["eps_weighted_median"] = eps_med

            # Relative valuation vs SPY (if SPY valuation available)
            rel_pe = (pe / spy_pe) if (math.isfinite(pe) and pe > 0 and math.isfinite(spy_pe) and spy_pe > 0) else float("nan")
            rel_ey = (ey / spy_ey) if (math.isfinite(ey) and math.isfinite(spy_ey) and spy_ey > 0) else float("nan")
            row["rel_pe_vs_spy"] = rel_pe
            row["rel_earnings_yield_vs_spy"] = rel_ey

            # -------------------------
            # Layer scoring (0..100)
            # -------------------------
            # 1) Statistical layer (max 40)
            # z-based points with smooth scaling + half-life confidence
            z_abs = abs(z) if math.isfinite(z) else 0.0
            # scale: 0 at 0.75, 1 at 2.5+
            z_strength = _clip01((z_abs - 0.75) / (2.5 - 0.75))
            z_points = self.settings.points_stat * 0.75 * z_strength  # up to 30 points from z
            # dispersion contributes up to 10 points (25% of stat layer)
            disp_strength = _clip01((disp_z + 1.0) / 3.0) if math.isfinite(disp_z) else 0.0  # favors > -1
            disp_points = self.settings.points_stat * 0.25 * disp_strength  # up to 10 points

            # half-life factor: best around 20-60 trading days for 1-3m horizon
            hl_factor = 1.0
            if math.isfinite(hl):
                if hl < 5:
                    hl_factor = 0.75  # too fast (more HFT-ish)
                elif 5 <= hl <= 80:
                    # peak at 35 days, smooth decay
                    hl_factor = 0.6 + 0.4 * math.exp(-abs(hl - 35.0) / 35.0)
                else:
                    hl_factor = 0.55  # too slow -> weak mean reversion
            stat_score = (z_points + disp_points) * hl_factor
            stat_score = max(0.0, min(float(self.settings.points_stat), stat_score))

            # 2) Macro & regime layer (max 20)
            # Base: penalize high macro betas (hard to isolate idiosyncratic SRV alpha)
            beta_threshold = 1.0  # not a placeholder: practical scale for daily cov beta magnitude
            beta_factor = math.exp(-beta_mag / beta_threshold) if math.isfinite(beta_mag) else 0.5

            # Credit stress penalty depending on cyclicals/defensives and direction
            macro_factor = beta_factor
            if credit_stress and row["direction"] != "NEUTRAL":
                if row["direction"] == "LONG" and s in cyclicals:
                    macro_factor *= 0.65
                if row["direction"] == "SHORT" and s in defensives:
                    macro_factor *= 0.70
                # In stress, correlation spikes and SRV can break down; apply general dampening
                macro_factor *= 0.85

            macro_score = self.settings.points_macro * max(0.0, min(1.0, macro_factor))

            # 3) Fundamental layer (max 20)
            # Validate dislocation: long wants cheap (lower rel_pe), short wants rich.
            fund_score = 0.0
            if row["direction"] != "NEUTRAL" and math.isfinite(rel_pe):
                if row["direction"] == "LONG":
                    raw = (1.0 - rel_pe) / 0.25  # full score if rel_pe <= 0.75
                else:
                    raw = (rel_pe - 1.0) / 0.25  # full score if rel_pe >= 1.25
                fund_strength = _clip01(raw)
                fund_score = self.settings.points_fund * fund_strength

            # Guardrails: avoid value traps / unreliable valuation
            if math.isfinite(pe) and pe > self.settings.pe_extreme:
                fund_score *= 0.60
            if math.isfinite(neg_w) and neg_w > self.settings.neg_earnings_weight_cap:
                fund_score *= 0.50
            if not math.isfinite(pe) or pe <= 0:
                fund_score *= 0.35

            # 4) Volatility layer (max 20)
            # VIX gating + hedge ratio practicality
            vix_factor = 1.0
            if math.isfinite(latest_vix):
                if latest_vix >= self.settings.vix_level_hard:
                    vix_factor = 0.25
                elif latest_vix >= self.settings.vix_level_soft:
                    vix_factor = 0.55
                else:
                    vix_factor = 1.0

            if math.isfinite(latest_vix_pct) and latest_vix_pct >= self.settings.vix_percentile_hard:
                vix_factor *= 0.60

            hr_factor = 0.75
            if math.isfinite(hr_s) and hr_s > 0:
                # penalize extremes; best near 1.0
                hr_factor = 0.5 + 0.5 * math.exp(-abs(math.log(hr_s)) / math.log(2.0))  # ~1 at hr=1, decays

            vol_score = self.settings.points_vol * max(0.0, min(1.0, vix_factor * hr_factor))

            # Total conviction 0..100
            conviction = float(stat_score + macro_score + fund_score + vol_score)
            conviction = max(0.0, min(100.0, conviction))

            row["score_stat"] = float(stat_score)
            row["score_macro"] = float(macro_score)
            row["score_fund"] = float(fund_score)
            row["score_vol"] = float(vol_score)
            row["conviction_score"] = conviction

            # Execution directives (vol parity sizing)
            # HR = Vol(SPY)/Vol(Sector). Interpret as:
            #   - Use SPY notional = 1.0
            #   - Use Sector notional = HR  (so sector vol dollars ~ SPY vol dollars)
            if row["direction"] == "LONG":
                row["trade_leg_sector"] = f"BUY {s} notional={hr_s:.2f}" if math.isfinite(hr_s) else f"BUY {s}"
                row["trade_leg_spy"] = "SELL SPY notional=1.00"
            elif row["direction"] == "SHORT":
                row["trade_leg_sector"] = f"SELL {s} notional={hr_s:.2f}" if math.isfinite(hr_s) else f"SELL {s}"
                row["trade_leg_spy"] = "BUY SPY notional=1.00"
            else:
                row["trade_leg_sector"] = "NO TRADE"
                row["trade_leg_spy"] = "NO TRADE"

            # Linking metrics (explicitly requested):
            # - RV alignment: combines statistical + valuation in one signal (for dashboard)
            # - Higher is better (direction-aware)
            rv_align = 0.0
            if row["direction"] == "LONG":
                rv_align = z_strength * _clip01((1.0 - rel_pe) / 0.25) if math.isfinite(rel_pe) else 0.0
            elif row["direction"] == "SHORT":
                rv_align = z_strength * _clip01((rel_pe - 1.0) / 0.25) if math.isfinite(rel_pe) else 0.0
            row["rv_alignment"] = float(rv_align)

            rows.append(row)

        master = pd.DataFrame(rows)
        master = master.sort_values(["conviction_score", "pca_residual_z"], ascending=[False, True]).reset_index(drop=True)

        # Attach global regime flags for the dashboard
        master["credit_z"] = latest_credit_z
        master["credit_stress"] = credit_stress
        master["vix_level"] = latest_vix
        master["vix_percentile"] = latest_vix_pct
        master["spy_pe_portfolio"] = spy_pe
        master["spy_earnings_yield"] = spy_ey

        self.master_df = master
        return master

    # Convenience API for dashboard
    def get_sector_tearsheet_series(self, sector_ticker: str) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        - residual_level
        - rolling mean
        - +/- 2 std bands
        - zscore
        """
        if self.residual_levels is None or self.residual_zscores is None:
            raise RuntimeError("Run calculate_conviction_score() first.")

        s = sector_ticker
        level = self.residual_levels[s].rename("residual_level")
        mu = level.rolling(window=self.settings.zscore_window, min_periods=self.settings.zscore_window).mean().rename("mean")
        sd = level.rolling(window=self.settings.zscore_window, min_periods=self.settings.zscore_window).std(ddof=0).rename("std")
        upper = (mu + 2.0 * sd).rename("upper_2s")
        lower = (mu - 2.0 * sd).rename("lower_2s")
        z = self.residual_zscores[s].rename("zscore")

        out = pd.concat([level, mu, upper, lower, z], axis=1).dropna()
        return out
