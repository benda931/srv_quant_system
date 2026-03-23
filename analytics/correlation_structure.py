"""
analytics/correlation_structure.py
===================================
Correlation Structure Measurement Engine

Implements the full measurement layer from the research brief:
"Short Volatility via Correlation and Dispersion Trades"

Produces reproducible, parameterized features for signals, risk, and monitoring:

1. ΔC = C_short - C_base  (correlation matrix distortion)
2. Frobenius distortion score D_t = ||ΔC ⊙ M||_F  (off-diagonal norm)
3. Eigenvalue / market-mode metrics:
   - Market-mode share: m_t = λ1 / N
   - Herfindahl concentration: H_t = Σ(λ_k / Σλ_j)²
   - Eigenvector loadings per sector
4. Sector contributions to distortion:
   - Within-sector: D_g^within = ||ΔC[S_g, S_g] ⊙ M||_F
   - Cross-sector:  D_{g,h}^cross = ||ΔC[S_g, S_h]||_F
   - Normalized: π_g^within, π_{g,h}^cross
5. Correlation-of-correlations (CoC) stability:
   - CoC_t = 1 - Corr(u(C_t^s), u(C_{t-1}^s))
   - Higher = structure changing rapidly

All calculations are explicit, parameterized, and logged.

Ref: Cboe Implied Correlation Index methodology (variance decomposition)
Ref: Random Matrix / PCA cross-correlations (Laloux et al., cond-mat/0108023)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrelationStructureSnapshot:
    """
    Complete measurement engine output at time t.
    All fields are NaN-safe scalars or DataFrames.
    """
    # ── Correlation matrices ─────────────────────────────────────────────────
    C_short: pd.DataFrame          # Short-window correlation matrix
    C_base: pd.DataFrame           # Base-window correlation matrix
    delta_C: pd.DataFrame          # ΔC = C_short - C_base

    # ── Frobenius distortion ─────────────────────────────────────────────────
    frob_distortion: float         # D_t = ||ΔC ⊙ M||_F (off-diagonal)
    frob_distortion_normalized: float  # D_t / √(N*(N-1)) for cross-universe comparability
    frob_distortion_z: float       # z-score of D_t vs rolling history

    # ── Eigenvalue / market-mode ─────────────────────────────────────────────
    eigenvalues: np.ndarray        # Sorted descending
    eigenvectors: np.ndarray       # Columns = eigenvectors (descending order)
    market_mode_share: float       # m_t = λ1 / N
    herfindahl: float              # H_t = Σ(λ_k / Σλ_j)²
    market_mode_loadings: Dict[str, float]  # Per-sector loadings of 1st eigenvector
    n_significant_eigenvalues: int # Eigenvalues above random-matrix noise threshold

    # ── Sector distortion contributions ──────────────────────────────────────
    sector_within_contrib: Dict[str, float]       # π_g^within (normalized)
    sector_cross_contrib: Dict[Tuple[str, str], float]  # π_{g,h}^cross (normalized)
    sector_within_raw: Dict[str, float]            # D_g^within (raw Frobenius)
    sector_cross_raw: Dict[Tuple[str, str], float] # D_{g,h}^cross (raw Frobenius)
    top_distortion_drivers: List[Dict]             # Sorted list of top contributors

    # ── Correlation-of-correlations stability ────────────────────────────────
    coc_instability: float         # CoC_t = 1 - Corr(u(C_t^s), u(C_{t-1}^s))
    coc_instability_z: float       # z-score vs rolling history

    # ── Average correlation metrics ──────────────────────────────────────────
    avg_corr_short: float          # Mean off-diagonal of C_short
    avg_corr_base: float           # Mean off-diagonal of C_base
    avg_corr_delta: float          # Δ average correlation

    # ── Metadata ─────────────────────────────────────────────────────────────
    as_of_date: Optional[str]      # Date of computation
    universe_size: int             # N assets
    short_window: int              # W_s used
    base_window: int               # W_b used


# ─────────────────────────────────────────────────────────────────────────────
# Time series result for rolling computation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrelationStructureTimeSeries:
    """Rolling measurement engine outputs across dates."""
    frob_distortion: pd.Series
    frob_distortion_z: pd.Series
    market_mode_share: pd.Series
    herfindahl: pd.Series
    coc_instability: pd.Series
    coc_instability_z: pd.Series
    avg_corr_short: pd.Series
    avg_corr_base: pd.Series
    # Sector contributions over time (Dict[sector_name, pd.Series])
    sector_within_ts: Dict[str, pd.Series] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation functions (stateless, testable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_corr_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix from returns DataFrame.

    Parameters
    ----------
    returns_df : DataFrame[T_window × N] — daily log returns

    Returns
    -------
    pd.DataFrame : N × N correlation matrix
    """
    C = returns_df.corr()
    # Force symmetry and fill diagonal
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C.values, 1.0)
    return C


def frobenius_offdiag(mat: pd.DataFrame) -> float:
    """
    Compute Frobenius norm of off-diagonal elements.

    D_t = ||M ⊙ ΔC||_F  where M_{ij} = 1 for i ≠ j, 0 otherwise

    Parameters
    ----------
    mat : N × N matrix (DataFrame or ndarray)

    Returns
    -------
    float : Off-diagonal Frobenius norm
    """
    vals = np.asarray(mat, dtype=float)
    n = vals.shape[0]
    if n < 2:
        return 0.0
    off = vals.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.sqrt((off ** 2).sum()))


def frobenius_block(mat: pd.DataFrame, rows: list, cols: list) -> float:
    """
    Compute Frobenius norm of a sub-block of the matrix.
    For within-sector blocks, excludes diagonal.

    Parameters
    ----------
    mat  : Full ΔC matrix
    rows : Row indices (sector tickers)
    cols : Column indices (sector tickers)

    Returns
    -------
    float : Frobenius norm of sub-block
    """
    try:
        block = np.asarray(mat.loc[rows, cols], dtype=float)
    except (KeyError, IndexError):
        return 0.0
    # If this is a within-sector block (rows == cols), mask diagonal
    if rows == cols:
        block = block.copy()
        np.fill_diagonal(block, 0.0)
    return float(np.sqrt((block ** 2).sum()))


def eigen_decomposition(C: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Eigenvalue decomposition of correlation matrix.

    Uses eigh for symmetric PSD stability.
    Ref: Random-matrix/PCA (Laloux et al., cond-mat/0108023)

    Parameters
    ----------
    C : N × N correlation matrix (DataFrame)

    Returns
    -------
    eigenvalues  : ndarray, sorted descending
    eigenvectors : ndarray, columns sorted by eigenvalue (descending)
    metrics      : dict with market_mode_share, herfindahl, n_significant
    """
    vals_arr = np.asarray(C, dtype=float)
    # Ensure symmetry
    vals_arr = 0.5 * (vals_arr + vals_arr.T)
    vals_arr = np.nan_to_num(vals_arr, nan=0.0)

    evals, evecs = np.linalg.eigh(vals_arr)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Clip negative eigenvalues (numerical noise)
    evals = np.clip(evals, 0.0, None)

    n = len(evals)
    total = evals.sum()

    if total < 1e-10:
        return evals, evecs, {
            "market_mode_share": float("nan"),
            "herfindahl": float("nan"),
            "n_significant": 0,
        }

    # Market-mode share: m_t = λ1 / N
    market_mode_share = float(evals[0] / n)

    # Herfindahl concentration: H_t = Σ(λ_k / Σλ_j)²
    shares = evals / total
    herfindahl = float((shares ** 2).sum())

    # Random-matrix noise threshold (Marchenko-Pastur upper edge)
    # For N assets and T observations, λ_max_noise ≈ (1 + √(N/T))²
    # Use conservative N_significant = eigenvalues above 1 + 2*sqrt(N/T)
    # With T unknown here, use simple heuristic: above 1/N * total * 2
    threshold = total / n * 2.0
    n_significant = int(np.sum(evals > threshold))

    return evals, evecs, {
        "market_mode_share": market_mode_share,
        "herfindahl": herfindahl,
        "n_significant": max(1, n_significant),
    }


def vectorize_upper_triangle(C: pd.DataFrame) -> np.ndarray:
    """
    Vectorize upper triangle (excluding diagonal) of correlation matrix.
    Used for correlation-of-correlations computation.

    Parameters
    ----------
    C : N × N correlation matrix

    Returns
    -------
    ndarray : vector of N*(N-1)/2 off-diagonal elements
    """
    vals = np.asarray(C, dtype=float)
    n = vals.shape[0]
    idx = np.triu_indices(n, k=1)
    return vals[idx]


def compute_coc_instability(C_current: pd.DataFrame, C_previous: pd.DataFrame) -> float:
    """
    Correlation-of-correlations instability.

    CoC_t = 1 - Corr(u(C_t^s), u(C_{t-1}^s))

    Higher value = correlation structure changing rapidly day-to-day.

    Parameters
    ----------
    C_current  : Current short-window correlation matrix
    C_previous : Previous short-window correlation matrix

    Returns
    -------
    float : CoC instability ∈ [0, 2] (0 = identical structure, 2 = perfect anticorrelation)
    """
    u_curr = vectorize_upper_triangle(C_current)
    u_prev = vectorize_upper_triangle(C_previous)

    if len(u_curr) < 3 or len(u_prev) < 3:
        return float("nan")

    # Remove NaN pairs
    mask = np.isfinite(u_curr) & np.isfinite(u_prev)
    if mask.sum() < 3:
        return float("nan")

    corr = np.corrcoef(u_curr[mask], u_prev[mask])[0, 1]
    if not np.isfinite(corr):
        return float("nan")

    return float(1.0 - corr)


def compute_sector_distortion_contributions(
    delta_C: pd.DataFrame,
    sector_groups: Dict[str, List[str]],
    total_distortion_sq: float,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float],
           Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Compute blockwise distortion contributions per sector.

    Within-sector: D_g^within = ||ΔC[S_g, S_g] ⊙ M||_F
    Cross-sector:  D_{g,h}^cross = ||ΔC[S_g, S_h]||_F
    Normalized:    π_g^within = (D_g^within)² / D_t²

    Parameters
    ----------
    delta_C            : ΔC matrix
    sector_groups      : {sector_name: [ticker1, ticker2, ...]}
    total_distortion_sq: D_t² for normalization

    Returns
    -------
    within_normalized : {sector: π_g^within}
    cross_normalized  : {(sector_a, sector_b): π_{g,h}^cross}
    within_raw        : {sector: D_g^within}
    cross_raw         : {(sector_a, sector_b): D_{g,h}^cross}
    """
    within_raw = {}
    cross_raw = {}
    within_norm = {}
    cross_norm = {}

    available_cols = set(delta_C.columns)
    sector_names = sorted(sector_groups.keys())

    # Within-sector contributions
    for g in sector_names:
        tickers = [t for t in sector_groups[g] if t in available_cols]
        if len(tickers) < 2:
            within_raw[g] = 0.0
            within_norm[g] = 0.0
            continue
        d = frobenius_block(delta_C, tickers, tickers)
        within_raw[g] = d
        within_norm[g] = (d ** 2) / total_distortion_sq if total_distortion_sq > 1e-12 else 0.0

    # Cross-sector contributions
    for i, g in enumerate(sector_names):
        tickers_g = [t for t in sector_groups[g] if t in available_cols]
        for h in sector_names[i + 1:]:
            tickers_h = [t for t in sector_groups[h] if t in available_cols]
            if not tickers_g or not tickers_h:
                cross_raw[(g, h)] = 0.0
                cross_norm[(g, h)] = 0.0
                continue
            d = frobenius_block(delta_C, tickers_g, tickers_h)
            cross_raw[(g, h)] = d
            cross_norm[(g, h)] = (d ** 2) / total_distortion_sq if total_distortion_sq > 1e-12 else 0.0

    return within_norm, cross_norm, within_raw, cross_raw


def avg_offdiag_corr(C: pd.DataFrame) -> float:
    """Average off-diagonal correlation."""
    vals = np.asarray(C, dtype=float)
    n = vals.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    offdiag = vals[mask]
    offdiag = offdiag[np.isfinite(offdiag)]
    if len(offdiag) == 0:
        return float("nan")
    return float(offdiag.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Measurement Engine class
# ─────────────────────────────────────────────────────────────────────────────
class CorrelationStructureEngine:
    """
    Measurement engine for correlation structure distortions.

    Usage:
        engine = CorrelationStructureEngine()
        snapshot = engine.compute_snapshot(returns, sector_groups, settings)
        ts = engine.compute_timeseries(returns, sector_groups, settings)

    All parameters are read from Settings for reproducibility.
    """

    def compute_snapshot(
        self,
        returns: pd.DataFrame,
        sector_groups: Dict[str, List[str]],
        W_s: int = 60,
        W_b: int = 252,
        distortion_z_lookback: int = 252,
        coc_z_lookback: int = 252,
        *,
        returns_prev_window: Optional[pd.DataFrame] = None,
        distortion_history: Optional[pd.Series] = None,
        coc_history: Optional[pd.Series] = None,
    ) -> CorrelationStructureSnapshot:
        """
        Compute full measurement engine snapshot at the latest date.

        Parameters
        ----------
        returns           : DataFrame indexed by date, columns = asset tickers (log returns)
        sector_groups     : {sector_name: [ticker1, ...]} — sector → constituent mapping
        W_s               : Short correlation window (days)
        W_b               : Base correlation window (days)
        distortion_z_lookback : Lookback for z-scoring Frobenius distortion
        coc_z_lookback    : Lookback for z-scoring CoC
        returns_prev_window : Returns for previous short window (for CoC). If None, auto-computed.
        distortion_history : Pre-computed rolling D_t series (for z-score). If None, skipped.
        coc_history        : Pre-computed rolling CoC series (for z-score). If None, skipped.

        Returns
        -------
        CorrelationStructureSnapshot
        """
        returns = returns.dropna(how="all")
        if len(returns) < W_b:
            raise ValueError(f"Need at least {W_b} observations, got {len(returns)}")

        # Identify available assets
        assets = [c for c in returns.columns if returns[c].notna().sum() >= W_s]
        returns = returns[assets]
        n = len(assets)

        # ── Step 1: Correlation matrices ─────────────────────────────────
        R_short = returns.iloc[-W_s:]
        R_base = returns.iloc[-W_b:]

        C_short = compute_corr_matrix(R_short)
        C_base = compute_corr_matrix(R_base)
        delta_C = C_short - C_base

        # ── Step 2: Frobenius distortion ─────────────────────────────────
        D_t = frobenius_offdiag(delta_C)
        D_t_norm = D_t / math.sqrt(n * (n - 1)) if n > 1 else 0.0

        # Z-score of distortion
        if distortion_history is not None and len(distortion_history) >= 20:
            recent = distortion_history.tail(distortion_z_lookback).dropna()
            mu_d = float(recent.mean())
            sd_d = float(recent.std(ddof=1))
            D_t_z = (D_t - mu_d) / sd_d if sd_d > 1e-10 else 0.0
        else:
            D_t_z = float("nan")

        # ── Step 3: Eigenvalue decomposition ─────────────────────────────
        evals, evecs, eig_metrics = eigen_decomposition(C_short)

        # Market-mode loadings per sector
        v1 = evecs[:, 0]  # First eigenvector
        mm_loadings = {}
        for i, asset in enumerate(assets):
            mm_loadings[asset] = round(float(v1[i]), 4)

        # ── Step 4: Sector distortion contributions ──────────────────────
        total_D_sq = D_t ** 2
        within_norm, cross_norm, within_raw, cross_raw = (
            compute_sector_distortion_contributions(delta_C, sector_groups, total_D_sq)
        )

        # Build top distortion drivers list
        top_drivers = self._build_top_drivers(
            within_norm, cross_norm, within_raw, cross_raw
        )

        # ── Step 5: Correlation-of-correlations stability ────────────────
        if returns_prev_window is not None:
            C_prev = compute_corr_matrix(returns_prev_window[assets])
        elif len(returns) > W_s:
            R_prev = returns.iloc[-(W_s + 1):-1]
            if len(R_prev) >= W_s:
                C_prev = compute_corr_matrix(R_prev)
            else:
                C_prev = None
        else:
            C_prev = None

        if C_prev is not None:
            coc = compute_coc_instability(C_short, C_prev)
        else:
            coc = float("nan")

        # Z-score of CoC
        if coc_history is not None and len(coc_history) >= 20:
            recent_coc = coc_history.tail(coc_z_lookback).dropna()
            mu_c = float(recent_coc.mean())
            sd_c = float(recent_coc.std(ddof=1))
            coc_z = (coc - mu_c) / sd_c if sd_c > 1e-10 else 0.0
        else:
            coc_z = float("nan")

        # ── Step 6: Average correlations ─────────────────────────────────
        avg_short = avg_offdiag_corr(C_short)
        avg_base = avg_offdiag_corr(C_base)
        avg_delta = avg_short - avg_base if (
            math.isfinite(avg_short) and math.isfinite(avg_base)
        ) else float("nan")

        # ── Assemble result ──────────────────────────────────────────────
        as_of = str(returns.index[-1]) if hasattr(returns.index[-1], "strftime") else str(returns.index[-1])

        return CorrelationStructureSnapshot(
            C_short=C_short,
            C_base=C_base,
            delta_C=delta_C,
            frob_distortion=round(D_t, 6),
            frob_distortion_normalized=round(D_t_norm, 6),
            frob_distortion_z=round(D_t_z, 4) if math.isfinite(D_t_z) else float("nan"),
            eigenvalues=evals,
            eigenvectors=evecs,
            market_mode_share=round(eig_metrics["market_mode_share"], 4),
            herfindahl=round(eig_metrics["herfindahl"], 4),
            market_mode_loadings=mm_loadings,
            n_significant_eigenvalues=eig_metrics["n_significant"],
            sector_within_contrib=within_norm,
            sector_cross_contrib=cross_norm,
            sector_within_raw=within_raw,
            sector_cross_raw=cross_raw,
            top_distortion_drivers=top_drivers,
            coc_instability=round(coc, 6) if math.isfinite(coc) else float("nan"),
            coc_instability_z=round(coc_z, 4) if math.isfinite(coc_z) else float("nan"),
            avg_corr_short=round(avg_short, 4) if math.isfinite(avg_short) else float("nan"),
            avg_corr_base=round(avg_base, 4) if math.isfinite(avg_base) else float("nan"),
            avg_corr_delta=round(avg_delta, 4) if math.isfinite(avg_delta) else float("nan"),
            as_of_date=as_of,
            universe_size=n,
            short_window=W_s,
            base_window=W_b,
        )

    def compute_timeseries(
        self,
        returns: pd.DataFrame,
        sector_groups: Dict[str, List[str]],
        W_s: int = 60,
        W_b: int = 252,
        distortion_z_lookback: int = 252,
        coc_z_lookback: int = 252,
        start_offset: Optional[int] = None,
    ) -> CorrelationStructureTimeSeries:
        """
        Compute rolling measurement engine outputs across dates.

        Parameters
        ----------
        returns       : Full returns DataFrame
        sector_groups : Sector groupings
        W_s, W_b      : Correlation windows
        start_offset  : Start computing from this index (default: W_b)

        Returns
        -------
        CorrelationStructureTimeSeries
        """
        returns = returns.dropna(how="all")
        assets = [c for c in returns.columns if returns[c].notna().sum() >= W_b]
        returns = returns[assets]

        if start_offset is None:
            start_offset = W_b

        dates = returns.index[start_offset:]
        n_dates = len(dates)

        # Pre-allocate Series
        frob_vals = pd.Series(index=dates, dtype=float)
        mm_vals = pd.Series(index=dates, dtype=float)
        hhi_vals = pd.Series(index=dates, dtype=float)
        coc_vals = pd.Series(index=dates, dtype=float)
        avg_s_vals = pd.Series(index=dates, dtype=float)
        avg_b_vals = pd.Series(index=dates, dtype=float)
        sector_within_ts = {g: pd.Series(index=dates, dtype=float) for g in sector_groups}

        C_prev = None
        log.info("Computing correlation structure timeseries: %d dates", n_dates)

        for i, t in enumerate(dates):
            t_idx = returns.index.get_loc(t)

            # Windows
            R_s = returns.iloc[t_idx - W_s + 1: t_idx + 1]
            R_b = returns.iloc[t_idx - W_b + 1: t_idx + 1]

            if len(R_s) < W_s or len(R_b) < W_b:
                continue

            C_s = compute_corr_matrix(R_s)
            C_b = compute_corr_matrix(R_b)
            dC = C_s - C_b

            # Frobenius distortion
            D_t = frobenius_offdiag(dC)
            frob_vals.iloc[i] = D_t

            # Eigen metrics
            _, _, eig_m = eigen_decomposition(C_s)
            mm_vals.iloc[i] = eig_m["market_mode_share"]
            hhi_vals.iloc[i] = eig_m["herfindahl"]

            # Average correlations
            avg_s_vals.iloc[i] = avg_offdiag_corr(C_s)
            avg_b_vals.iloc[i] = avg_offdiag_corr(C_b)

            # CoC instability
            if C_prev is not None:
                coc_vals.iloc[i] = compute_coc_instability(C_s, C_prev)
            C_prev = C_s

            # Sector within contributions
            total_D_sq = D_t ** 2
            for g, tickers in sector_groups.items():
                avail = [t for t in tickers if t in assets]
                if len(avail) >= 2:
                    d_within = frobenius_block(dC, avail, avail)
                    sector_within_ts[g].iloc[i] = (
                        (d_within ** 2) / total_D_sq if total_D_sq > 1e-12 else 0.0
                    )

            if (i + 1) % 50 == 0:
                log.debug("  ... processed %d / %d dates", i + 1, n_dates)

        # Z-scores
        frob_z = self._rolling_zscore(frob_vals, distortion_z_lookback)
        coc_z = self._rolling_zscore(coc_vals, coc_z_lookback)

        return CorrelationStructureTimeSeries(
            frob_distortion=frob_vals,
            frob_distortion_z=frob_z,
            market_mode_share=mm_vals,
            herfindahl=hhi_vals,
            coc_instability=coc_vals,
            coc_instability_z=coc_z,
            avg_corr_short=avg_s_vals,
            avg_corr_base=avg_b_vals,
            sector_within_ts=sector_within_ts,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Rolling z-score of a series."""
        mu = series.rolling(window=window, min_periods=20).mean()
        sd = series.rolling(window=window, min_periods=20).std(ddof=1)
        z = (series - mu) / sd.replace(0.0, np.nan)
        return z

    def compute_snapshot_with_zscore(
        self,
        returns: pd.DataFrame,
        sector_groups: Dict[str, List[str]],
        W_s: int = 60,
        W_b: int = 252,
        distortion_z_lookback: int = 252,
        coc_z_lookback: int = 252,
        history_step: int = 5,
    ) -> CorrelationStructureSnapshot:
        """
        Compute snapshot WITH proper z-scores by building a lightweight
        rolling history of Frobenius distortion and CoC instability.

        This avoids the NaN z-scores from compute_snapshot() when no
        history series is provided.

        Parameters
        ----------
        history_step : Sample every N-th date for the rolling history
                       to keep computation time manageable (default: 5 = weekly)
        """
        returns = returns.dropna(how="all")
        assets = [c for c in returns.columns if returns[c].notna().sum() >= W_b]
        returns = returns[assets]

        if len(returns) < W_b + distortion_z_lookback:
            # Not enough data for history — fall back to basic snapshot
            return self.compute_snapshot(
                returns, sector_groups, W_s, W_b,
                distortion_z_lookback, coc_z_lookback,
            )

        # Build lightweight rolling history
        start_idx = W_b
        end_idx = len(returns)
        sample_indices = list(range(start_idx, end_idx, history_step))
        # Always include the last date
        if sample_indices[-1] != end_idx - 1:
            sample_indices.append(end_idx - 1)

        frob_hist = []
        coc_hist = []
        C_prev = None

        for idx in sample_indices:
            R_s = returns.iloc[max(0, idx - W_s + 1): idx + 1]
            R_b = returns.iloc[max(0, idx - W_b + 1): idx + 1]
            if len(R_s) < W_s or len(R_b) < W_b:
                continue

            C_s = compute_corr_matrix(R_s)
            C_b = compute_corr_matrix(R_b)
            dC = C_s - C_b
            D_t = frobenius_offdiag(dC)
            frob_hist.append(D_t)

            if C_prev is not None:
                coc_hist.append(compute_coc_instability(C_s, C_prev))
            C_prev = C_s

        # Convert to Series for z-score use
        frob_series = pd.Series(frob_hist) if frob_hist else None
        coc_series = pd.Series(coc_hist) if coc_hist else None

        # Now compute the proper snapshot with history
        return self.compute_snapshot(
            returns, sector_groups, W_s, W_b,
            distortion_z_lookback, coc_z_lookback,
            distortion_history=frob_series,
            coc_history=coc_series,
        )

    @staticmethod
    def _build_top_drivers(
        within_norm: Dict[str, float],
        cross_norm: Dict[Tuple[str, str], float],
        within_raw: Dict[str, float],
        cross_raw: Dict[Tuple[str, str], float],
    ) -> List[Dict]:
        """Build sorted list of top distortion contributors."""
        drivers = []
        for g, pi in within_norm.items():
            drivers.append({
                "type": "within",
                "sector": g,
                "contribution": round(pi, 4),
                "raw_distortion": round(within_raw.get(g, 0.0), 6),
            })
        for (g, h), pi in cross_norm.items():
            drivers.append({
                "type": "cross",
                "sector_a": g,
                "sector_b": h,
                "contribution": round(pi, 4),
                "raw_distortion": round(cross_raw.get((g, h), 0.0), 6),
            })
        drivers.sort(key=lambda x: x["contribution"], reverse=True)
        return drivers[:15]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build sector_groups from Settings
# ─────────────────────────────────────────────────────────────────────────────
def build_sector_groups_from_settings(settings) -> Dict[str, List[str]]:
    """
    Build sector_groups dict from Settings.

    For ETF-level universe (each sector = 1 ETF ticker),
    returns {sector_name: [ticker]}.
    For dispersion at constituent level, this should be extended
    to map sectors → constituent tickers.
    """
    return {
        sector_name: [ticker]
        for sector_name, ticker in settings.sector_tickers.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compact summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────
def measurement_summary(snap: CorrelationStructureSnapshot) -> dict:
    """Compact serializable summary for Claude / AgentBus / daily brief."""
    return {
        "as_of": snap.as_of_date,
        "universe_size": snap.universe_size,
        "windows": {"short": snap.short_window, "base": snap.base_window},
        "frob_distortion": snap.frob_distortion,
        "frob_distortion_normalized": snap.frob_distortion_normalized,
        "frob_distortion_z": snap.frob_distortion_z if math.isfinite(snap.frob_distortion_z) else None,
        "market_mode_share": snap.market_mode_share,
        "herfindahl": snap.herfindahl,
        "n_significant_eigenvalues": snap.n_significant_eigenvalues,
        "coc_instability": snap.coc_instability if math.isfinite(snap.coc_instability) else None,
        "coc_instability_z": snap.coc_instability_z if math.isfinite(snap.coc_instability_z) else None,
        "avg_corr_short": snap.avg_corr_short if math.isfinite(snap.avg_corr_short) else None,
        "avg_corr_base": snap.avg_corr_base if math.isfinite(snap.avg_corr_base) else None,
        "avg_corr_delta": snap.avg_corr_delta if math.isfinite(snap.avg_corr_delta) else None,
        "top_distortion_drivers": [
            f"{d.get('sector', d.get('sector_a', ''))}"
            + (f"-{d['sector_b']}" if "sector_b" in d else "")
            + f" ({d['type']}): π={d['contribution']:.3f}"
            for d in snap.top_distortion_drivers[:5]
        ],
    }
