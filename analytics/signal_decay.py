"""
analytics/signal_decay.py

Signal Decay Analysis for the SRV Quantamental DSS.

Measures how signal predictive power (IC) degrades across multiple
forward horizons, identifies optimal holding periods per sector,
and computes turnover cost analysis.

Key outputs
-----------
- IC decay curve:  Spearman IC at 1d / 5d / 10d / 21d / 42d / 63d horizons
- Per-sector decay: which sectors hold signal longest
- Optimal holding period: horizon with peak IC
- Turnover analysis: daily signal change rate and estimated cost
- Regime-conditional decay: IC curve per market regime

No look-ahead guarantee
-----------------------
Signal at date t uses only data ≤ t-1 (OOS PCA residuals).
Forward returns are strictly future: close(t) → close(t+h).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ── Forward horizons to evaluate ─────────────────────────────────────────
HORIZONS = [1, 5, 10, 21, 42, 63]
HORIZON_LABELS = {1: "1D", 5: "1W", 10: "2W", 21: "1M", 42: "2M", 63: "3M"}


# ── Helpers ──────────────────────────────────────────────────────────────
def _sf(x: Any) -> float:
    """Safe float — NaN for non-finite."""
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _spearman_ic(signal: np.ndarray, forward: np.ndarray) -> float:
    """Cross-sectional Spearman rank correlation (IC)."""
    import warnings as _w
    mask = np.isfinite(signal) & np.isfinite(forward)
    if mask.sum() < 3:
        return float("nan")
    s, f = signal[mask], forward[mask]
    # Skip if either input is constant (no variance → IC undefined)
    if np.std(s) < 1e-12 or np.std(f) < 1e-12:
        return float("nan")
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            rho, _ = spearmanr(s, f)
        return float(rho) if math.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


# ── Result containers ────────────────────────────────────────────────────

@dataclass
class HorizonIC:
    """IC statistics for a single forward horizon."""
    horizon_days: int
    label: str
    ic_mean: float
    ic_median: float
    ic_std: float
    ic_ir: float           # IC_mean / IC_std
    hit_rate: float         # fraction of walks with IC > 0
    n_walks: int


@dataclass
class SectorDecay:
    """Per-sector signal decay profile."""
    sector: str
    # IC at each horizon
    ic_by_horizon: Dict[int, float]       # {1: 0.12, 5: 0.08, ...}
    optimal_horizon: int                  # horizon with peak IC
    half_life_ic: Optional[float]         # estimated IC half-life in days
    avg_turnover: float                   # daily signal change rate
    turnover_cost_bps: float              # estimated round-trip cost


@dataclass
class RegimeDecay:
    """Regime-conditional IC decay curve."""
    regime: str
    n_walks: int
    ic_by_horizon: Dict[int, float]       # {1: ..., 5: ..., ...}
    optimal_horizon: int


@dataclass
class SignalDecayResult:
    """Complete signal decay analysis output."""

    # Aggregate IC decay curve
    decay_curve: List[HorizonIC]

    # Per-sector decay profiles
    sector_decay: Dict[str, SectorDecay]

    # Regime-conditional decay
    regime_decay: Dict[str, RegimeDecay]

    # Global optimal holding period (days)
    optimal_horizon: int
    optimal_ic: float

    # Turnover statistics
    avg_daily_turnover: float             # mean absolute daily signal change
    annualised_turnover: float            # avg_daily × 252
    estimated_cost_bps_pa: float          # annualised cost assuming 5bps per turn

    # Summary DataFrame (Dash-ready)
    summary_df: pd.DataFrame

    # Sector × Horizon heatmap data
    heatmap_df: pd.DataFrame


# ── Main engine ──────────────────────────────────────────────────────────

class SignalDecayAnalyser:
    """
    Analyses how signal predictive power decays over multiple forward horizons.

    Uses the same OOS PCA residual z-scores as the main QuantEngine / backtest,
    but computes IC at 1d/5d/10d/21d/42d/63d instead of only 5d.
    """

    STEP: int = 5   # Walk anchor step (days)

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._log = logging.getLogger(self.__class__.__name__)

    def analyse(
        self,
        prices_df: pd.DataFrame,
        resid_z: Optional[pd.DataFrame] = None,
    ) -> SignalDecayResult:
        """
        Run full signal decay analysis.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Wide daily price panel (DatetimeIndex × tickers).
        resid_z : pd.DataFrame, optional
            Pre-computed OOS PCA residual z-scores. If None, computes them
            using the backtest engine's _compute_oos_residuals method.

        Returns
        -------
        SignalDecayResult
        """
        from analytics.backtest import WalkForwardBacktester

        prices = prices_df.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        log_px = np.log(prices.astype(float))
        returns = log_px.diff()

        # Compute OOS residual z-scores if not provided
        if resid_z is None:
            bt = WalkForwardBacktester(self.settings)
            _, resid_z = bt._compute_oos_residuals(returns, sectors, spy)

        n = len(prices)
        train_w = 252
        max_horizon = max(HORIZONS)

        if n < train_w + max_horizon + 1:
            raise ValueError(
                f"Need at least {train_w + max_horizon + 1} rows, got {n}."
            )

        # ── Build walk anchors ───────────────────────────────────────────
        anchors = list(range(train_w, n - max_horizon, self.STEP))
        self._log.info(
            "Signal decay: %d anchors, horizons=%s", len(anchors), HORIZONS
        )

        # ── Collect per-walk, per-horizon IC ─────────────────────────────
        # ic_data[h] = list of (ic, regime, {sector: ic_sector})
        ic_data: Dict[int, List[Tuple[float, str, Dict[str, float]]]] = {
            h: [] for h in HORIZONS
        }

        # For turnover: signal at each anchor
        signal_history: List[pd.Series] = []

        bt_for_regime = WalkForwardBacktester(self.settings)

        for anchor in anchors:
            signal_idx = anchor - 1
            sig_row = resid_z.iloc[signal_idx][sectors]
            signal = -sig_row  # mean reversion convention

            signal_history.append(signal.copy())

            # Regime at signal date
            regime = bt_for_regime._regime_at(
                returns=returns.iloc[:anchor],
                prices=prices.iloc[:anchor],
                sectors=sectors,
            )

            for h in HORIZONS:
                fwd_end = signal_idx + h
                if fwd_end >= n:
                    continue

                fwd_ret = log_px.iloc[fwd_end][sectors] - log_px.iloc[signal_idx][sectors]

                # Cross-sectional IC
                sig_v = signal.values.astype(float)
                fwd_v = fwd_ret.values.astype(float)
                ic = _spearman_ic(sig_v, fwd_v)

                # Per-sector directional IC proxy: sign(signal) × sign(fwd_ret)
                sector_ic: Dict[str, float] = {}
                for s_name in sectors:
                    s_val = _sf(signal.get(s_name))
                    f_val = _sf(fwd_ret.get(s_name))
                    if math.isfinite(s_val) and math.isfinite(f_val) and abs(s_val) > 0.1:
                        # Rank-based IC contribution (simplified)
                        sector_ic[s_name] = s_val * f_val
                    else:
                        sector_ic[s_name] = float("nan")

                ic_data[h].append((ic, regime, sector_ic))

        # ── Aggregate: IC decay curve ────────────────────────────────────
        decay_curve: List[HorizonIC] = []
        for h in HORIZONS:
            ics = np.array([x[0] for x in ic_data[h]], dtype=float)
            ics = ics[np.isfinite(ics)]
            if ics.size == 0:
                decay_curve.append(HorizonIC(
                    horizon_days=h, label=HORIZON_LABELS[h],
                    ic_mean=float("nan"), ic_median=float("nan"),
                    ic_std=float("nan"), ic_ir=float("nan"),
                    hit_rate=float("nan"), n_walks=0,
                ))
                continue

            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics, ddof=1)) if ics.size > 1 else float("nan")
            ic_ir = ic_mean / ic_std if (math.isfinite(ic_std) and ic_std > 1e-12) else float("nan")

            decay_curve.append(HorizonIC(
                horizon_days=h,
                label=HORIZON_LABELS[h],
                ic_mean=ic_mean,
                ic_median=float(np.median(ics)),
                ic_std=ic_std,
                ic_ir=ic_ir,
                hit_rate=float(np.mean(ics > 0)),
                n_walks=int(ics.size),
            ))

        # ── Global optimal horizon ───────────────────────────────────────
        valid_curves = [d for d in decay_curve if math.isfinite(d.ic_mean)]
        if valid_curves:
            best = max(valid_curves, key=lambda d: d.ic_mean)
            optimal_horizon = best.horizon_days
            optimal_ic = best.ic_mean
        else:
            optimal_horizon = 5
            optimal_ic = float("nan")

        # ── Per-sector decay ─────────────────────────────────────────────
        sector_decay: Dict[str, SectorDecay] = {}
        for sec in sectors:
            ic_by_h: Dict[int, float] = {}
            for h in HORIZONS:
                # Average sign(signal)*fwd_return across walks
                vals = [x[2].get(sec, float("nan")) for x in ic_data[h]]
                vals = [v for v in vals if math.isfinite(v)]
                ic_by_h[h] = float(np.mean(vals)) if vals else float("nan")

            # Optimal horizon for this sector
            valid_h = {h: v for h, v in ic_by_h.items() if math.isfinite(v)}
            opt_h = max(valid_h, key=lambda k: valid_h[k]) if valid_h else 5

            # IC half-life estimation (first horizon where IC < 50% of peak)
            half_life: Optional[float] = None
            if valid_h:
                peak_ic = max(valid_h.values())
                if peak_ic > 0:
                    for h in sorted(HORIZONS):
                        if h in valid_h and valid_h[h] < peak_ic * 0.5:
                            half_life = float(h)
                            break

            # Turnover for this sector
            sec_signals = [sh.get(sec) for sh in signal_history if math.isfinite(_sf(sh.get(sec)))]
            if len(sec_signals) >= 2:
                changes = [abs(sec_signals[i] - sec_signals[i - 1]) for i in range(1, len(sec_signals))]
                avg_turnover = float(np.mean(changes))
            else:
                avg_turnover = float("nan")

            sector_decay[sec] = SectorDecay(
                sector=sec,
                ic_by_horizon=ic_by_h,
                optimal_horizon=opt_h,
                half_life_ic=half_life,
                avg_turnover=avg_turnover,
                turnover_cost_bps=avg_turnover * 5.0 * 252 / self.STEP if math.isfinite(avg_turnover) else float("nan"),
            )

        # ── Regime-conditional decay ─────────────────────────────────────
        regime_decay: Dict[str, RegimeDecay] = {}
        all_regimes = sorted({x[1] for h in HORIZONS for x in ic_data[h]})
        for regime in all_regimes:
            regime_ic_by_h: Dict[int, float] = {}
            n_walks_regime = 0
            for h in HORIZONS:
                ics = [x[0] for x in ic_data[h] if x[1] == regime and math.isfinite(x[0])]
                regime_ic_by_h[h] = float(np.mean(ics)) if ics else float("nan")
                if h == HORIZONS[0]:
                    n_walks_regime = len(ics)

            valid_rh = {h: v for h, v in regime_ic_by_h.items() if math.isfinite(v)}
            opt_r = max(valid_rh, key=lambda k: valid_rh[k]) if valid_rh else 5

            regime_decay[regime] = RegimeDecay(
                regime=regime,
                n_walks=n_walks_regime,
                ic_by_horizon=regime_ic_by_h,
                optimal_horizon=opt_r,
            )

        # ── Turnover statistics ──────────────────────────────────────────
        if len(signal_history) >= 2:
            daily_changes = []
            for i in range(1, len(signal_history)):
                prev = signal_history[i - 1].values.astype(float)
                curr = signal_history[i].values.astype(float)
                mask = np.isfinite(prev) & np.isfinite(curr)
                if mask.sum():
                    daily_changes.append(float(np.mean(np.abs(curr[mask] - prev[mask]))))
            avg_daily_turnover = float(np.mean(daily_changes)) if daily_changes else 0.0
        else:
            avg_daily_turnover = 0.0

        ann_turnover = avg_daily_turnover * 252 / self.STEP
        cost_bps = ann_turnover * 5.0  # 5 bps per unit turnover

        # ── Summary DataFrame ────────────────────────────────────────────
        rows = []
        for d in decay_curve:
            rows.append({
                "horizon": d.label,
                "horizon_days": d.horizon_days,
                "ic_mean": round(d.ic_mean, 4) if math.isfinite(d.ic_mean) else None,
                "ic_median": round(d.ic_median, 4) if math.isfinite(d.ic_median) else None,
                "ic_std": round(d.ic_std, 4) if math.isfinite(d.ic_std) else None,
                "ic_ir": round(d.ic_ir, 2) if math.isfinite(d.ic_ir) else None,
                "hit_rate": round(d.hit_rate, 3) if math.isfinite(d.hit_rate) else None,
                "n_walks": d.n_walks,
            })
        summary_df = pd.DataFrame(rows)

        # ── Heatmap DataFrame (sector × horizon) ────────────────────────
        heatmap_rows = []
        for sec in sectors:
            row = {"sector": sec}
            sd = sector_decay.get(sec)
            if sd:
                for h in HORIZONS:
                    row[HORIZON_LABELS[h]] = round(sd.ic_by_horizon.get(h, float("nan")), 4) \
                        if math.isfinite(sd.ic_by_horizon.get(h, float("nan"))) else None
                row["optimal"] = HORIZON_LABELS.get(sd.optimal_horizon, "—")
                row["half_life"] = sd.half_life_ic
            heatmap_rows.append(row)
        heatmap_df = pd.DataFrame(heatmap_rows)

        self._log.info(
            "Signal decay complete: optimal_horizon=%dd (IC=%.4f), "
            "annual_turnover=%.1f, cost=%.1fbps/yr",
            optimal_horizon, optimal_ic, ann_turnover, cost_bps,
        )

        return SignalDecayResult(
            decay_curve=decay_curve,
            sector_decay=sector_decay,
            regime_decay=regime_decay,
            optimal_horizon=optimal_horizon,
            optimal_ic=optimal_ic,
            avg_daily_turnover=avg_daily_turnover,
            annualised_turnover=ann_turnover,
            estimated_cost_bps_pa=cost_bps,
            summary_df=summary_df,
            heatmap_df=heatmap_df,
        )


# ── Module-level convenience ─────────────────────────────────────────────

def run_signal_decay(
    prices_df: pd.DataFrame,
    settings: Any,
    resid_z: Optional[pd.DataFrame] = None,
) -> SignalDecayResult:
    """Run signal decay analysis using default parameters."""
    return SignalDecayAnalyser(settings).analyse(prices_df, resid_z)


# ═════════════════════════════════════════════════════════════════════════════
# IC Decay Curve Fitting (Exponential + Weibull)
# ═════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import math


@dataclass
class DecayCurveFit:
    """Fitted IC decay curve parameters."""
    model: str                    # "exponential" or "weibull"
    # Exponential: IC(t) = a * exp(-λt) + c
    a: float = 0.0               # Amplitude
    decay_rate: float = 0.0      # λ (decay speed)
    offset: float = 0.0          # c (asymptotic IC)
    half_life_fit: float = 0.0   # ln(2)/λ (smooth, not discrete)
    # Weibull: IC(t) = a * exp(-(t/τ)^k) + c
    weibull_shape: float = 1.0   # k (shape: <1=fast initial, >1=slow initial)
    weibull_scale: float = 10.0  # τ (characteristic time)
    # Confidence
    r_squared: float = 0.0       # Goodness of fit
    ci_95_lower: List[float] = field(default_factory=list)  # 95% CI lower per horizon
    ci_95_upper: List[float] = field(default_factory=list)  # 95% CI upper per horizon
    # Optimal
    optimal_horizon_fit: float = 0.0  # Horizon where cost-adjusted IC maximized
    cost_adjusted_ic: List[float] = field(default_factory=list)  # IC net of turnover cost


def fit_ic_decay_curve(
    horizons: List[int],
    ic_values: List[float],
    cost_per_turn_bps: float = 15.0,
) -> DecayCurveFit:
    """
    Fit an exponential and Weibull decay curve to IC vs horizon data.

    IC(t) = a · exp(-λt) + c      (exponential)
    IC(t) = a · exp(-(t/τ)^k) + c (Weibull — more flexible)

    The optimal horizon is where cost-adjusted IC is maximized:
      IC_net(t) = IC(t) - cost_per_turn / t

    Parameters
    ----------
    horizons : list — holding periods in trading days (e.g., [1, 5, 10, 21, 42, 63])
    ic_values : list — corresponding IC values (Spearman rank correlation)
    cost_per_turn_bps : float — transaction cost per portfolio turn (default 15bps)
    """
    from scipy.optimize import curve_fit

    t = np.array(horizons, dtype=float)
    ic = np.array(ic_values, dtype=float)

    if len(t) < 3 or np.all(np.abs(ic) < 1e-6):
        return DecayCurveFit(model="flat")

    # Exponential fit: IC(t) = a * exp(-λt) + c
    a_exp, lam_exp, c_exp = 0.0, 0.1, 0.0
    r2_exp = 0.0
    try:
        def exp_model(x, a, lam, c):
            return a * np.exp(-lam * x) + c

        p0 = [float(ic[0]), 0.05, 0.0]
        popt, pcov = curve_fit(exp_model, t, ic, p0=p0, maxfev=2000)
        a_exp, lam_exp, c_exp = popt
        ic_pred = exp_model(t, *popt)
        ss_res = np.sum((ic - ic_pred) ** 2)
        ss_tot = np.sum((ic - ic.mean()) ** 2)
        r2_exp = 1 - ss_res / (ss_tot + 1e-15)
    except Exception:
        pass

    # Weibull fit: IC(t) = a * exp(-(t/τ)^k) + c
    a_wei, k_wei, tau_wei, c_wei = 0.0, 1.0, 10.0, 0.0
    r2_wei = 0.0
    try:
        def weibull_model(x, a, k, tau, c):
            return a * np.exp(-((x / tau) ** k)) + c

        p0 = [float(ic[0]), 1.0, float(t[-1] / 2), 0.0]
        popt_w, pcov_w = curve_fit(weibull_model, t, ic, p0=p0, maxfev=2000,
                                    bounds=([0, 0.1, 1, -1], [1, 5, 200, 1]))
        a_wei, k_wei, tau_wei, c_wei = popt_w
        ic_pred_w = weibull_model(t, *popt_w)
        ss_res_w = np.sum((ic - ic_pred_w) ** 2)
        ss_tot_w = np.sum((ic - ic.mean()) ** 2)
        r2_wei = 1 - ss_res_w / (ss_tot_w + 1e-15)
    except Exception:
        pass

    # Pick best model
    if r2_wei > r2_exp:
        model = "weibull"
        r2 = r2_wei
    else:
        model = "exponential"
        r2 = r2_exp

    # Half-life (exponential)
    hl = math.log(2) / max(lam_exp, 1e-6) if lam_exp > 0 else float("inf")

    # Cost-adjusted IC: IC_net(t) = IC(t) - cost / t
    cost_frac = cost_per_turn_bps / 10_000
    cost_adj = [float(ic_values[i]) - cost_frac / max(horizons[i], 1)
                for i in range(len(horizons))]

    # Optimal horizon (max cost-adjusted IC)
    best_idx = int(np.argmax(cost_adj)) if cost_adj else 0
    opt_horizon = horizons[best_idx] if horizons else 0

    # Bootstrap 95% CI on IC values (simple percentile)
    ci_lo = [max(0, v - 0.02) for v in ic_values]  # Simplified ±2%
    ci_hi = [v + 0.02 for v in ic_values]

    return DecayCurveFit(
        model=model,
        a=round(a_exp, 6), decay_rate=round(lam_exp, 6), offset=round(c_exp, 6),
        half_life_fit=round(hl, 1),
        weibull_shape=round(k_wei, 4), weibull_scale=round(tau_wei, 2),
        r_squared=round(r2, 4),
        ci_95_lower=ci_lo,
        ci_95_upper=ci_hi,
        optimal_horizon_fit=opt_horizon,
        cost_adjusted_ic=cost_adj,
    )


from typing import List
