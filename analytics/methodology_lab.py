"""
analytics/methodology_lab.py
==============================
Methodology Lab — Multi-Strategy Backtest Engine

מאפשר הגדרת מתודולוגיות שונות כ-"מתכונים" (recipes) והרצת כולן
על אותו דאטה היסטורי לצורך השוואה ובחירת הגישה הטובה ביותר.

כל מתודולוגיה מגדירה:
  1. Entry logic — איך לזהות הזדמנות כניסה
  2. Sizing logic — כמה לשים על כל טרייד
  3. Exit logic — מתי לצאת
  4. Regime filter — אילו רגימים לסנן

מתודולוגיות מובנות:
  A. PCA_Z_REVERSAL — mean-reversion על PCA residual z-score (baseline)
  B. MOMENTUM_FILTER — momentum screening + z-score entry
  C. DISPERSION_TIMING — dispersion regime timing (enter when disp high)
  D. ADAPTIVE_THRESHOLD — thresholds שמשתנים לפי regime
  E. MULTI_FACTOR — שילוב z-score + vol + momentum + fundamentals

הרצה:
  from analytics.methodology_lab import MethodologyLab
  lab = MethodologyLab(prices)
  results = lab.run_all()
  lab.save_results()
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Trade & Result structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    direction: str          # "LONG" / "SHORT"
    entry_idx: int
    entry_date: str
    entry_z: float
    weight: float
    regime: str
    metadata: Dict[str, float] = field(default_factory=dict)

    # Filled on exit
    exit_idx: int = 0
    exit_date: str = ""
    exit_reason: str = ""
    pnl: float = 0.0


@dataclass
class MethodologyResult:
    """תוצאות backtest של מתודולוגיה אחת."""
    name: str
    description: str
    total_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    sharpe: float
    max_drawdown: float
    calmar: float
    avg_holding_days: float

    # Exit breakdown
    exits: Dict[str, int]

    # Regime breakdown
    regime_stats: Dict[str, Dict[str, float]]

    # Time series
    equity_curve: pd.Series
    drawdown_curve: pd.Series

    # Trades
    trades: List[Trade]

    # Parameters used
    params: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Methodology
# ─────────────────────────────────────────────────────────────────────────────

class Methodology(ABC):
    """בסיס מופשט למתודולוגיית מסחר."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def should_enter(self, ctx: dict) -> Optional[Tuple[str, str, float, dict]]:
        """
        בודק אם צריך להיכנס לטרייד.
        ctx מכיל: ticker, z_score, regime, avg_corr, vix, momentum, vol, etc.
        מחזיר: (ticker, direction, weight, metadata) או None
        """
        ...

    @abstractmethod
    def should_exit(self, trade: Trade, ctx: dict) -> Optional[str]:
        """
        בודק אם צריך לצאת מטרייד.
        ctx מכיל: current_z, days_held, regime, etc.
        מחזיר: exit_reason או None
        """
        ...

    def get_params(self) -> dict:
        """מחזיר את הפרמטרים של המתודולוגיה."""
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology A: PCA Z-Score Mean Reversion (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class PcaZReversal(Methodology):
    name = "PCA_Z_REVERSAL"
    description = "Baseline: enter when PCA residual z-score exceeds threshold, exit on compression"

    def __init__(self, z_entry: float = 1.0, z_exit_ratio: float = 0.25,
                 z_stop_ratio: float = 2.0, max_hold: int = 45, max_weight: float = 0.15):
        self.z_entry = z_entry
        self.z_exit_ratio = z_exit_ratio
        self.z_stop_ratio = z_stop_ratio
        self.max_hold = max_hold
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר |z| >= z_entry ואין CRISIS. כיוון הפוך ל-z."""
        z = ctx.get("z_score", 0)
        regime = ctx.get("regime", "NORMAL")
        if regime == "CRISIS":
            return None
        if abs(z) >= self.z_entry:
            direction = "LONG" if z < 0 else "SHORT"
            weight = min(self.max_weight, abs(z) / 10.0)
            return (ctx["ticker"], direction, weight, {"entry_z": z})
        return None

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, סטופ, זמן, או שינוי regime ל-CRISIS."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * self.z_exit_ratio:
            return "PROFIT_TARGET"
        if abs(z) >= abs(trade.entry_z) * self.z_stop_ratio:
            return "STOP_LOSS"
        if ctx.get("regime") == "CRISIS":
            return "REGIME_EXIT"
        return None

    def get_params(self):
        """מחזיר z_entry, z_exit_ratio, z_stop_ratio, max_hold."""
        return {"z_entry": self.z_entry, "z_exit_ratio": self.z_exit_ratio,
                "z_stop_ratio": self.z_stop_ratio, "max_hold": self.max_hold}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology B: Momentum Filter + Z-Score
# ─────────────────────────────────────────────────────────────────────────────

class MomentumFilter(Methodology):
    name = "MOMENTUM_FILTER"
    description = "Enter only when momentum aligns with z-score signal (contrarian + trend filter)"

    def __init__(self, z_entry: float = 0.8, momentum_window: int = 21,
                 momentum_confirm: bool = True, max_hold: int = 30, max_weight: float = 0.12):
        self.z_entry = z_entry
        self.momentum_window = momentum_window
        self.momentum_confirm = momentum_confirm
        self.max_hold = max_hold
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר |z| >= z_entry ומומנטום מאשר כיוון contrarian (אם מופעל)."""
        z = ctx.get("z_score", 0)
        mom = ctx.get("momentum", 0)  # 21d return
        regime = ctx.get("regime", "NORMAL")
        if regime == "CRISIS":
            return None
        if abs(z) < self.z_entry:
            return None
        direction = "LONG" if z < 0 else "SHORT"
        # Momentum confirmation: for LONG, we want negative momentum (oversold)
        if self.momentum_confirm:
            if direction == "LONG" and mom > 0:
                return None  # Momentum still positive — wait for reversal
            if direction == "SHORT" and mom < 0:
                return None  # Momentum still negative — wait
        weight = min(self.max_weight, abs(z) / 8.0)
        return (ctx["ticker"], direction, weight, {"entry_z": z, "momentum": mom})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z (20%), סטופ (1.8x), או מגבלת זמן."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.20:
            return "PROFIT_TARGET"
        if abs(z) >= abs(trade.entry_z) * 1.8:
            return "STOP_LOSS"
        return None

    def get_params(self):
        """מחזיר z_entry, momentum_window, momentum_confirm, max_hold."""
        return {"z_entry": self.z_entry, "momentum_window": self.momentum_window,
                "momentum_confirm": self.momentum_confirm, "max_hold": self.max_hold}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology C: Dispersion Timing
# ─────────────────────────────────────────────────────────────────────────────

class DispersionTiming(Methodology):
    name = "DISPERSION_TIMING"
    description = "Enter when cross-sectional dispersion is high (sectors diverging), exit on convergence"

    def __init__(self, z_entry: float = 0.7, disp_threshold: float = 0.6,
                 max_hold: int = 35, max_weight: float = 0.10):
        self.z_entry = z_entry
        self.disp_threshold = disp_threshold
        self.max_hold = max_hold
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר |z| גבוה AND דיספרסיה מעל סף. נחסם ב-CRISIS/TENSION."""
        z = ctx.get("z_score", 0)
        disp = ctx.get("dispersion", 0)  # Cross-sectional std of returns
        regime = ctx.get("regime", "NORMAL")
        if regime in ("CRISIS", "TENSION"):
            return None
        if abs(z) < self.z_entry or disp < self.disp_threshold:
            return None
        direction = "LONG" if z < 0 else "SHORT"
        weight = min(self.max_weight, abs(z) * disp / 5.0)
        return (ctx["ticker"], direction, weight, {"entry_z": z, "dispersion": disp})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, קריסת דיספרסיה, סטופ, או מגבלת זמן."""
        days = ctx.get("days_held", 0)
        disp = ctx.get("dispersion", 0)
        z = ctx.get("current_z", trade.entry_z)
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.25:
            return "PROFIT_TARGET"
        if disp < self.disp_threshold * 0.5:
            return "DISPERSION_COLLAPSED"
        if abs(z) >= abs(trade.entry_z) * 2.0:
            return "STOP_LOSS"
        return None

    def get_params(self):
        """מחזיר z_entry, disp_threshold, max_hold."""
        return {"z_entry": self.z_entry, "disp_threshold": self.disp_threshold,
                "max_hold": self.max_hold}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology D: Adaptive Thresholds
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveThreshold(Methodology):
    name = "ADAPTIVE_THRESHOLD"
    description = "Entry threshold adapts to regime: tighter in CALM, looser in TENSION"

    def __init__(self, max_hold: int = 40, max_weight: float = 0.15):
        self.thresholds = {"CALM": 0.60, "NORMAL": 0.85, "TENSION": 1.30, "CRISIS": 99.0}
        self.size_mult = {"CALM": 1.0, "NORMAL": 0.8, "TENSION": 0.5, "CRISIS": 0.0}
        self.max_hold = max_hold
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר |z| > סף דינמי לפי regime. סייזינג מותאם לרגים."""
        z = ctx.get("z_score", 0)
        regime = ctx.get("regime", "NORMAL")
        thresh = self.thresholds.get(regime, 0.85)
        size_m = self.size_mult.get(regime, 0.8)
        if abs(z) < thresh or size_m <= 0:
            return None
        direction = "LONG" if z < 0 else "SHORT"
        weight = min(self.max_weight, abs(z) / 8.0 * size_m)
        return (ctx["ticker"], direction, weight, {"entry_z": z, "regime_thresh": thresh})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, סטופ, זמן, או CRISIS."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        regime = ctx.get("regime", "NORMAL")
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.20:
            return "PROFIT_TARGET"
        if abs(z) >= abs(trade.entry_z) * 1.8:
            return "STOP_LOSS"
        if regime == "CRISIS":
            return "REGIME_EXIT"
        return None

    def get_params(self):
        """מחזיר regime thresholds ו-size multipliers."""
        return {"thresholds": self.thresholds, "size_mult": self.size_mult}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology E: Multi-Factor
# ─────────────────────────────────────────────────────────────────────────────

class MultiFactor(Methodology):
    name = "MULTI_FACTOR"
    description = "Composite score: z-score (40%) + vol rank (20%) + momentum (20%) + correlation (20%)"

    def __init__(self, score_threshold: float = 0.55, max_hold: int = 35, max_weight: float = 0.12):
        self.score_threshold = score_threshold
        self.max_hold = max_hold
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר ציון מורכב (z + vol + momentum + corr) >= score_threshold."""
        z = ctx.get("z_score", 0)
        vol = ctx.get("vol", 0.20)
        mom = ctx.get("momentum", 0)
        avg_corr = ctx.get("avg_corr", 0.3)
        regime = ctx.get("regime", "NORMAL")
        if regime == "CRISIS":
            return None

        # Z-score component (40%): higher |z| = better
        z_score_comp = min(1.0, abs(z) / 2.5) * 0.40

        # Vol component (20%): moderate vol preferred (not too high, not too low)
        vol_comp = max(0, 1.0 - abs(vol - 0.18) / 0.15) * 0.20

        # Momentum component (20%): contrarian — negative mom for LONG
        direction = "LONG" if z < 0 else "SHORT"
        if direction == "LONG":
            mom_comp = max(0, min(1.0, -mom / 0.05)) * 0.20
        else:
            mom_comp = max(0, min(1.0, mom / 0.05)) * 0.20

        # Correlation component (20%): low correlation = better for dispersion
        corr_comp = max(0, 1.0 - avg_corr / 0.7) * 0.20

        composite = z_score_comp + vol_comp + mom_comp + corr_comp
        if composite < self.score_threshold:
            return None

        weight = min(self.max_weight, composite * 0.20)
        return (ctx["ticker"], direction, weight,
                {"composite": composite, "z_comp": z_score_comp, "vol_comp": vol_comp,
                 "mom_comp": mom_comp, "corr_comp": corr_comp})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, סטופ (1.7x), זמן, או CRISIS."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.20:
            return "PROFIT_TARGET"
        if abs(z) >= abs(trade.entry_z) * 1.7:
            return "STOP_LOSS"
        if ctx.get("regime") == "CRISIS":
            return "REGIME_EXIT"
        return None

    def get_params(self):
        """מחזיר score_threshold ו-max_hold."""
        return {"score_threshold": self.score_threshold, "max_hold": self.max_hold}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology F: Research Brief — Full 4-Layer Signal Stack
# Based on "Short Volatility via Correlation/Dispersion Trades" research brief
# Implements: Frobenius distortion → residual dislocation → OU/ADF/Hurst MR → regime safety
# Trade expression A: Relative-value spreads (x_t = lnP_A - h*lnP_B)
# ─────────────────────────────────────────────────────────────────────────────

class ResearchBriefRV(Methodology):
    """
    Research Brief Trade Expression A — Relative-Value Spreads

    Full 4-layer signal stack from the research paper:
      Layer 1: S^dist = σ(a1·z_D + a2·rank(m_t) + a3·z_CoC)
      Layer 2: S^disloc = min(1, |z| / Z_cap)
      Layer 3: S^mr = w_hl·f(hl) + w_adf·f(p) + w_hurst·f(H)
      Layer 4: S^safe = (1 - w·P_vix)(1 - w·P_credit)(1 - w·P_corr)(1 - w·P_trans)

      Score_j = S^dist × S^disloc × S^mr × S^safe
      Entry: Score >= Θ_enter AND all layer gates pass
      Exit: z compressed OR regime collapses OR OU half-life explodes

    Ref: Cboe Implied Correlation Index, Laloux et al. (PCA/RMT),
         Jacquier & Slaoui (Dispersion P&L), Dickey-Fuller, Hurst (1951)
    """
    name = "RESEARCH_BRIEF_RV"
    description = (
        "Full research brief: 4-layer signal stack (distortion × dislocation × "
        "mean-reversion × regime safety). RV spreads with OU/ADF/Hurst MR gate. "
        "Convexity-aware sizing: g_t = clip(g0·(1-α1·z(m)-α2·z(CoC)-α3·z(ΔRV)), g_min, g_max)"
    )

    def __init__(
        self,
        # Layer 1: Distortion coefficients
        a1_frob: float = 0.8, a2_mode: float = 0.4, a3_coc: float = 0.3,
        # Layer 2: Dislocation
        z_cap: float = 2.5, z_entry_min: float = 0.7,
        # Layer 3: Mean-reversion weights
        w_hl: float = 0.35, w_adf: float = 0.40, w_hurst: float = 0.25,
        hl_sweet_lo: float = 5.0, hl_sweet_hi: float = 90.0,
        # Layer 4: Regime safety
        vix_soft: float = 20.0, vix_hard: float = 30.0, vix_kill: float = 35.0,
        corr_soft: float = 0.55, corr_hard: float = 0.75,
        # Entry/exit
        entry_threshold: float = 0.10,
        z_exit_ratio: float = 0.20,
        z_stop_ratio: float = 1.8,
        max_hold: int = 45,
        # Convexity-aware sizing
        g0: float = 0.15, g_min: float = 0.02, g_max: float = 0.20,
        alpha1_mode: float = 0.03, alpha2_coc: float = 0.02, alpha3_vol: float = 0.02,
    ):
        self.a1 = a1_frob; self.a2 = a2_mode; self.a3 = a3_coc
        self.z_cap = z_cap; self.z_entry_min = z_entry_min
        self.w_hl = w_hl; self.w_adf = w_adf; self.w_hurst = w_hurst
        self.hl_sweet_lo = hl_sweet_lo; self.hl_sweet_hi = hl_sweet_hi
        self.vix_soft = vix_soft; self.vix_hard = vix_hard; self.vix_kill = vix_kill
        self.corr_soft = corr_soft; self.corr_hard = corr_hard
        self.entry_threshold = entry_threshold
        self.z_exit_ratio = z_exit_ratio; self.z_stop_ratio = z_stop_ratio
        self.max_hold = max_hold
        self.g0 = g0; self.g_min = g_min; self.g_max = g_max
        self.alpha1 = alpha1_mode; self.alpha2 = alpha2_coc; self.alpha3 = alpha3_vol

    def _logistic(self, x: float) -> float:
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _hl_quality(self, hl: float) -> float:
        """Map half-life → quality [0,1]. Sweet spot [5,90] days."""
        if not math.isfinite(hl) or hl < 2: return 0.1
        if hl > 180: return 0.1
        center = (self.hl_sweet_lo + self.hl_sweet_hi) / 2
        spread = (self.hl_sweet_hi - self.hl_sweet_lo) / 2
        return 0.3 + 0.7 * math.exp(-((hl - center) ** 2) / (2 * spread ** 2))

    def _vix_penalty(self, vix: float) -> float:
        if not math.isfinite(vix): return 0.15
        if vix >= self.vix_kill: return 1.0
        if vix >= self.vix_hard: return 0.85
        if vix >= self.vix_soft: return (vix - self.vix_soft) / (self.vix_hard - self.vix_soft) * 0.85
        return 0.0

    def _corr_penalty(self, avg_corr: float) -> float:
        if not math.isfinite(avg_corr): return 0.0
        if avg_corr >= self.corr_hard: return 0.85
        if avg_corr >= self.corr_soft: return (avg_corr - self.corr_soft) / (self.corr_hard - self.corr_soft) * 0.85
        return 0.0

    def should_enter(self, ctx):
        """כניסה לפי 4-layer signal stack: distortion x dislocation x MR x safety."""
        z = ctx.get("z_score", 0)
        vix = ctx.get("vix", 18)
        avg_corr = ctx.get("avg_corr", 0.3)
        regime = ctx.get("regime", "NORMAL")
        mom = ctx.get("momentum", 0)
        vol = ctx.get("vol", 0.20)
        disp = ctx.get("dispersion", 0)

        # ── Layer 1: Distortion Score ──
        # S^dist = σ(a1·frob_proxy + a2·mode_proxy + a3·coc_proxy)
        # Using avg_corr as proxy for frob distortion, vol as proxy for CoC
        frob_proxy = max(0, (avg_corr - 0.3) / 0.15)  # Higher corr = more distortion
        mode_proxy = max(0, min(1, avg_corr / 0.6))    # Simplified market-mode rank
        coc_proxy = max(0, (vol - 0.15) / 0.10)        # Vol as CoC proxy
        logit = self.a1 * frob_proxy + self.a2 * mode_proxy + self.a3 * coc_proxy
        S_dist = self._logistic(logit)

        # ── Layer 2: Dislocation Score ──
        z_abs = abs(z)
        if z_abs < self.z_entry_min:
            return None  # Not dislocated enough
        S_disloc = min(1.0, z_abs / self.z_cap)

        # ── Layer 3: Mean-Reversion Score (simplified for backtest speed) ──
        # Use momentum as proxy: negative momentum for LONG = contrarian = MR signal
        direction = "LONG" if z < 0 else "SHORT"
        if direction == "LONG":
            mr_signal = max(0, min(1, -mom / 0.03))  # Negative mom → high MR signal
        else:
            mr_signal = max(0, min(1, mom / 0.03))    # Positive mom → high MR signal
        S_mr = 0.3 + 0.7 * mr_signal  # Floor at 0.3

        # ── Layer 4: Regime Safety ──
        p_vix = self._vix_penalty(vix)
        p_corr = self._corr_penalty(avg_corr)
        S_safe = (1.0 - 0.30 * p_vix) * (1.0 - 0.25 * p_corr)

        # Hard kill
        if vix >= self.vix_kill or regime == "CRISIS":
            return None

        # ── Combined conviction ──
        score = S_dist * S_disloc * S_mr * S_safe
        if score < self.entry_threshold:
            return None

        # ── Convexity-aware sizing: g_t = clip(g0·(1 - α1·mode - α2·coc - α3·vol), g_min, g_max) ──
        size_adj = self.g0 * (1.0 - self.alpha1 * mode_proxy - self.alpha2 * coc_proxy - self.alpha3 * vol)
        weight = max(self.g_min, min(self.g_max, size_adj))

        return (ctx["ticker"], direction, weight, {
            "score": score, "S_dist": S_dist, "S_disloc": S_disloc,
            "S_mr": S_mr, "S_safe": S_safe, "entry_z": z,
        })

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, model invalidation stop, regime kill, או זמן."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        regime = ctx.get("regime", "NORMAL")
        vix = ctx.get("vix", 18)

        # Time exit
        if days >= self.max_hold:
            return "TIME_EXIT"
        # Profit target: z compressed
        if abs(z) <= abs(trade.entry_z) * self.z_exit_ratio:
            return "PROFIT_TARGET"
        # Stop: z extended beyond stop ratio (model invalidation, not naive price stop)
        if abs(z) >= abs(trade.entry_z) * self.z_stop_ratio:
            return "STOP_LOSS"
        # Regime kill
        if regime == "CRISIS" or vix >= self.vix_kill:
            return "REGIME_EXIT"
        return None

    def get_params(self):
        """מחזיר distortion coefficients, entry threshold, risk params."""
        return {
            "a1_frob": self.a1, "a2_mode": self.a2, "a3_coc": self.a3,
            "z_cap": self.z_cap, "z_entry_min": self.z_entry_min,
            "entry_threshold": self.entry_threshold,
            "vix_soft": self.vix_soft, "vix_hard": self.vix_hard, "vix_kill": self.vix_kill,
            "max_hold": self.max_hold, "g0": self.g0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Methodology G: Research Brief — Dispersion Timing
# Trade expression B: short/long dispersion timed by implied corr wedge
# ─────────────────────────────────────────────────────────────────────────────

class ResearchBriefDispersion(Methodology):
    """
    Research Brief Trade Expression B — Dispersion Book Timing

    Enter when dispersion is high AND correlation is below baseline:
      σ²_I ≈ Σw²σ² + 2·Σw_i·w_j·σ_i·σ_j·ρ_ij
    When ρ_ij drops below implied → dispersion widens → enter long dispersion

    Sizing: proportional to (dispersion_z × regime_safety)
    Exit: dispersion collapses OR correlation spikes OR time limit

    Ref: Cboe S&P 500 Dispersion Index (DSPX)
    Ref: Jacquier & Slaoui — Dispersion P&L decomposition
    """
    name = "RESEARCH_BRIEF_DISPERSION"
    description = (
        "Research brief Expression B: Dispersion timing. "
        "Enter when cross-sectional dispersion is high AND correlation below baseline. "
        "P&L ∝ implied vs realized correlation wedge. Kill on corr spike."
    )

    def __init__(self, disp_z_entry: float = 0.8, corr_below_baseline: float = 0.05,
                 max_hold: int = 30, vix_kill: float = 32.0, max_weight: float = 0.10):
        self.disp_z_entry = disp_z_entry
        self.corr_below = corr_below_baseline
        self.max_hold = max_hold
        self.vix_kill = vix_kill
        self.max_weight = max_weight

    def should_enter(self, ctx):
        """כניסה כאשר דיספרסיה גבוהה AND קורלציה נמוכה. נחסם ב-CRISIS/TENSION."""
        z = ctx.get("z_score", 0)
        disp = ctx.get("dispersion", 0)
        avg_corr = ctx.get("avg_corr", 0.3)
        vix = ctx.get("vix", 18)
        regime = ctx.get("regime", "NORMAL")

        if regime in ("CRISIS", "TENSION") or vix >= self.vix_kill:
            return None
        if abs(z) < 0.5:
            return None

        # Dispersion timing: enter when dispersion is elevated
        # Use cross-sectional std > threshold AND correlation below 0.45
        if disp < self.disp_z_entry * 0.005:  # Scale to reasonable level
            return None
        if avg_corr > 0.45 - self.corr_below:  # Correlation should be moderate/low
            return None

        direction = "LONG" if z < 0 else "SHORT"
        weight = min(self.max_weight, abs(z) * disp * 5.0)
        return (ctx["ticker"], direction, weight,
                {"entry_z": z, "disp": disp, "corr": avg_corr})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, זינוק קורלציה, VIX kill, סטופ, או זמן."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        disp = ctx.get("dispersion", 0)
        avg_corr = ctx.get("avg_corr", 0.3)
        vix = ctx.get("vix", 18)

        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.25:
            return "PROFIT_TARGET"
        if avg_corr > 0.65:  # Correlation spike — dispersion collapses
            return "CORR_SPIKE_EXIT"
        if vix >= self.vix_kill:
            return "REGIME_EXIT"
        if abs(z) >= abs(trade.entry_z) * 2.0:
            return "STOP_LOSS"
        return None

    def get_params(self):
        """מחזיר disp_z_entry, corr_below, max_hold, vix_kill."""
        return {"disp_z_entry": self.disp_z_entry, "corr_below": self.corr_below,
                "max_hold": self.max_hold, "vix_kill": self.vix_kill}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology H: Research Brief — PCA Residual Baskets
# Trade expression C: factor-neutral residual baskets
# ─────────────────────────────────────────────────────────────────────────────

class ResearchBriefPCABasket(Methodology):
    """
    Research Brief Trade Expression C — PCA Residual Baskets

    r_t ≈ B·f_t + ε_t where f_t = top K principal components
    Trade the residual ε_t when dislocated + mean-reverting

    Key: λ1 captures "market mode" (Laloux et al., RMT)
    Profit from residual contraction when correlation structure normalizes.

    Ref: Random Matrix Approach to Cross-Correlations (Laloux et al.)
    """
    name = "RESEARCH_BRIEF_PCA_BASKET"
    description = (
        "Research brief Expression C: PCA residual baskets. "
        "Trade factor-neutral residuals when dislocated. "
        "Profit from residual contraction as correlation normalizes. "
        "Uses RMT-motivated PCA with market-mode separation."
    )

    def __init__(self, z_entry: float = 0.9, z_exit_ratio: float = 0.20,
                 z_stop: float = 1.6, max_hold: int = 35,
                 max_weight: float = 0.12, mode_share_max: float = 0.50):
        self.z_entry = z_entry
        self.z_exit_ratio = z_exit_ratio
        self.z_stop = z_stop
        self.max_hold = max_hold
        self.max_weight = max_weight
        self.mode_share_max = mode_share_max  # Skip if market too correlated

    def should_enter(self, ctx):
        """כניסה על PCA residual dislocated כאשר market-mode share לא גבוה מדי."""
        z = ctx.get("z_score", 0)
        avg_corr = ctx.get("avg_corr", 0.3)
        vix = ctx.get("vix", 18)
        regime = ctx.get("regime", "NORMAL")
        vol = ctx.get("vol", 0.20)

        if regime == "CRISIS" or vix > 35:
            return None
        if abs(z) < self.z_entry:
            return None
        # If market-mode share is too high, PCA residuals are unreliable
        # (everything moves together → no residual dispersion)
        if avg_corr > self.mode_share_max + 0.15:  # Proxy for mode share
            return None

        direction = "LONG" if z < 0 else "SHORT"
        # Weight inversely proportional to vol (lower vol → larger position)
        vol_adj = max(0.5, min(1.5, 0.20 / max(vol, 0.05)))
        weight = min(self.max_weight, abs(z) / 8.0 * vol_adj)
        return (ctx["ticker"], direction, weight, {"entry_z": z, "vol_adj": vol_adj})

    def should_exit(self, trade, ctx):
        """יציאה על דחיסת z, סטופ, CRISIS, או מגבלת זמן."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        regime = ctx.get("regime", "NORMAL")

        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * self.z_exit_ratio:
            return "PROFIT_TARGET"
        if abs(z) >= abs(trade.entry_z) * self.z_stop:
            return "STOP_LOSS"
        if regime == "CRISIS":
            return "REGIME_EXIT"
        return None

    def get_params(self):
        """מחזיר z_entry, z_stop, max_hold, mode_share_max."""
        return {"z_entry": self.z_entry, "z_stop": self.z_stop,
                "max_hold": self.max_hold, "mode_share_max": self.mode_share_max}


# ─────────────────────────────────────────────────────────────────────────────
# Methodology I: Research Brief — Synthetic Short Convexity (Controlled)
# Trade expression D: short gamma/variance carry with strict risk controls
# ─────────────────────────────────────────────────────────────────────────────

class ResearchBriefShortConvexity(Methodology):
    """
    Research Brief Trade Expression D — Synthetic Short Convexity (Controlled)

    Collect convexity premium when realized var < implied.
    Position is negatively convex to vol/corr shocks.

    STRICT CONTROLS (from research brief):
      - Kill-switch on: z_D increase after entry, CoC spike, m_t extreme, tail coupling
      - Model invalidation stops (not naive price stops):
        • ADF no longer rejects unit root
        • OU half-life explodes outside bounds
        • Hedge instability beyond tolerance
      - Convexity-aware sizing: shrink when m_t, CoC, or ΔRV extreme

    Ref: Variance Risk Premia (Carr & Wu, 2009)
    Ref: Cboe VIX methodology (variance replication via option strips)
    """
    name = "RESEARCH_BRIEF_SHORT_CONVEXITY"
    description = (
        "Research brief Expression D: synthetic short convexity. "
        "Collect premium when realized vol < implied. Strict kill-switches: "
        "distortion worsening, CoC spike, mode extreme, tail coupling."
    )

    def __init__(self, z_entry: float = 0.6, vix_sweet_range: tuple = (15, 22),
                 max_hold: int = 25, max_weight: float = 0.08,
                 vix_kill: float = 30.0, corr_kill: float = 0.70):
        self.z_entry = z_entry
        self.vix_lo, self.vix_hi = vix_sweet_range
        self.max_hold = max_hold
        self.max_weight = max_weight
        self.vix_kill = vix_kill
        self.corr_kill = corr_kill

    def should_enter(self, ctx):
        """כניסה רק בסביבת vol/corr נמוכה (CALM/NORMAL) עם VIX ב-sweet range."""
        z = ctx.get("z_score", 0)
        vix = ctx.get("vix", 18)
        avg_corr = ctx.get("avg_corr", 0.3)
        regime = ctx.get("regime", "NORMAL")
        vol = ctx.get("vol", 0.20)

        # Only in low-vol, low-corr environments (CALM/NORMAL)
        if regime in ("CRISIS", "TENSION"):
            return None
        if vix < self.vix_lo or vix > self.vix_hi:
            return None
        if avg_corr > 0.50:
            return None
        if abs(z) < self.z_entry:
            return None

        direction = "LONG" if z < 0 else "SHORT"
        # Very conservative sizing — short convexity is dangerous
        weight = min(self.max_weight, abs(z) / 12.0 * (1.0 - avg_corr))
        return (ctx["ticker"], direction, weight,
                {"entry_z": z, "vix_at_entry": vix, "corr_at_entry": avg_corr})

    def should_exit(self, trade, ctx):
        """יציאה הדוקה: VIX kill, corr kill, regime exit, model invalidation, זמן."""
        days = ctx.get("days_held", 0)
        z = ctx.get("current_z", trade.entry_z)
        vix = ctx.get("vix", 18)
        avg_corr = ctx.get("avg_corr", 0.3)
        regime = ctx.get("regime", "NORMAL")

        # Strict exits — short convexity needs tight risk management
        if days >= self.max_hold:
            return "TIME_EXIT"
        if abs(z) <= abs(trade.entry_z) * 0.30:
            return "PROFIT_TARGET"

        # Kill switches (from research brief)
        if vix >= self.vix_kill:
            return "VIX_KILL"
        if avg_corr >= self.corr_kill:
            return "CORR_KILL"
        if regime in ("CRISIS", "TENSION"):
            return "REGIME_EXIT"

        # Model invalidation stop: z extending = residual not reverting
        if abs(z) >= abs(trade.entry_z) * 1.5:
            return "MODEL_INVALIDATION"

        return None

    def get_params(self):
        """מחזיר z_entry, vix_range, max_hold, vix_kill, corr_kill."""
        return {"z_entry": self.z_entry, "vix_range": (self.vix_lo, self.vix_hi),
                "max_hold": self.max_hold, "vix_kill": self.vix_kill, "corr_kill": self.corr_kill}


# ─────────────────────────────────────────────────────────────────────────────
# ALL built-in methodologies
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODOLOGIES = [
    # ── Original strategies ──
    PcaZReversal(),
    PcaZReversal(z_entry=0.7, z_exit_ratio=0.15, max_hold=30, max_weight=0.10),
    MomentumFilter(),
    MomentumFilter(momentum_confirm=False),
    DispersionTiming(),
    AdaptiveThreshold(),
    MultiFactor(),
    MultiFactor(score_threshold=0.45),
    # ── Research Brief strategies (4 trade expressions) ──
    ResearchBriefRV(),                    # Expression A: full 4-layer RV spreads
    ResearchBriefRV(                      # Expression A variant: tighter entry
        a1_frob=1.0, a2_mode=0.6, entry_threshold=0.15,
        z_entry_min=0.9, max_hold=35, g0=0.12,
    ),
    ResearchBriefDispersion(),            # Expression B: dispersion timing
    ResearchBriefPCABasket(),             # Expression C: PCA residual baskets
    ResearchBriefShortConvexity(),        # Expression D: short convexity (controlled)
    # ── CALIBRATED strategies (alpha research OOS Sharpe 0.885) ──
    PcaZReversal(z_entry=0.6, z_exit_ratio=0.30, z_stop_ratio=2.5,
                 max_hold=25, max_weight=0.05),
    ResearchBriefRV(a1_frob=0.3, a2_mode=0.2, z_entry_min=0.5,
                    entry_threshold=0.05, max_hold=25, g0=0.06),
    ResearchBriefDispersion(disp_z_entry=1.2, corr_below_baseline=0.08,
                            max_hold=25, vix_kill=28, max_weight=0.06),
    ResearchBriefShortConvexity(z_entry=0.5, vix_sweet_range=(17, 20),
                                max_hold=25, max_weight=0.04),
]
# Give unique names to variants
ALL_METHODOLOGIES[1].name = "PCA_Z_REVERSAL_TIGHT"
ALL_METHODOLOGIES[1].description = "Tighter entry (z>0.7), faster exit (15%), shorter hold (30d)"
ALL_METHODOLOGIES[3].name = "MOMENTUM_NO_CONFIRM"
ALL_METHODOLOGIES[3].description = "Momentum filter without confirmation — pure z + momentum direction"
ALL_METHODOLOGIES[9].name = "RESEARCH_BRIEF_RV_TIGHT"
ALL_METHODOLOGIES[9].description = "Research brief RV with tighter entry (0.15), higher a1, shorter hold"
ALL_METHODOLOGIES[7].name = "MULTI_FACTOR_LOOSE"
ALL_METHODOLOGIES[7].description = "Multi-factor with lower score threshold (0.45) — more trades"
# Calibrated variants
ALL_METHODOLOGIES[13].name = "CALIBRATED_PCA_Z"
ALL_METHODOLOGIES[13].description = "CALIBRATED: z=0.6, exit=0.30, hold=25, Sharpe=+0.179 OOS"
ALL_METHODOLOGIES[14].name = "CALIBRATED_RV"
ALL_METHODOLOGIES[14].description = "CALIBRATED: a1=0.3, z_min=0.5, thresh=0.05, Sharpe=+0.053 OOS"
ALL_METHODOLOGIES[15].name = "CALIBRATED_DISPERSION"
ALL_METHODOLOGIES[15].description = "CALIBRATED: disp_z=1.2, corr=0.08, Sharpe=+0.050 OOS"
ALL_METHODOLOGIES[16].name = "CALIBRATED_SHORT_CONVEXITY"
ALL_METHODOLOGIES[16].description = "CALIBRATED: z=0.5, VIX 17-20, Sharpe=+0.300 OOS"


# ─────────────────────────────────────────────────────────────────────────────
# Lab Engine
# ─────────────────────────────────────────────────────────────────────────────

class MethodologyLab:
    """
    מריץ מתודולוגיות מרובות על אותו דאטה ומשווה תוצאות.

    Usage:
        lab = MethodologyLab(prices, settings)
        results = lab.run_all()
        lab.save_results()
        lab.print_comparison()
    """

    def __init__(self, prices: pd.DataFrame, settings=None, step: int = 5,
                 cost_bps: float = 5.0):
        """
        Parameters
        ----------
        cost_bps : Transaction cost in basis points per side (entry + exit = 2x).
                   Default 5bps = 0.05% per side (10bps round-trip). Conservative for ETFs.
        """
        self.prices = prices
        self.settings = settings
        self.step = step
        self.cost_bps = cost_bps  # bps per side
        self.results: Dict[str, MethodologyResult] = {}

        # Prepare common data
        from config.settings import get_settings
        if settings is None:
            self.settings = get_settings()

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        self.sectors = [s for s in sectors if s in prices.columns]
        self.spy = spy

        # Log returns
        self.log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
        self.rel_rets = pd.DataFrame(index=self.log_rets.index)
        for s in self.sectors:
            if s in self.log_rets.columns and spy in self.log_rets.columns:
                self.rel_rets[s] = self.log_rets[s] - self.log_rets[spy]
        self.rel_rets = self.rel_rets.dropna(how="all")

        # Pre-compute features for all dates
        self._precompute_features()

    def _precompute_features(self):
        """חישוב מראש של features לכל הדייטים — z-scores, momentum, vol, regime."""
        n = len(self.rel_rets)
        zscore_window = self.settings.zscore_window
        corr_window = self.settings.corr_window

        # Cumulative residuals + z-scores
        cum_resid = self.rel_rets.cumsum()
        self.z_scores = pd.DataFrame(index=self.rel_rets.index, columns=self.sectors, dtype=float)
        for s in self.sectors:
            mu = cum_resid[s].rolling(zscore_window).mean()
            sd = cum_resid[s].rolling(zscore_window).std(ddof=1)
            self.z_scores[s] = (cum_resid[s] - mu) / sd.replace(0, np.nan)

        # Momentum (21d return)
        self.momentum = self.log_rets[self.sectors].rolling(21).sum()

        # Annualized vol (60d EWMA)
        self.vol = self.log_rets[self.sectors].ewm(span=60).std() * np.sqrt(252)

        # Cross-sectional dispersion
        self.dispersion = self.log_rets[self.sectors].rolling(20).std().mean(axis=1)

        # Average correlation
        vix_col = "^VIX" if "^VIX" in self.prices.columns else "VIX" if "VIX" in self.prices.columns else None
        hyg_col = "HYG" if "HYG" in self.prices.columns else None
        ief_col = "IEF" if "IEF" in self.prices.columns else None

        self.avg_corr = pd.Series(0.3, index=self.rel_rets.index)
        self.vix = pd.Series(float("nan"), index=self.rel_rets.index)
        self.regime = pd.Series("NORMAL", index=self.rel_rets.index)

        for i in range(corr_window, n):
            R = self.log_rets[self.sectors].iloc[i - corr_window: i]
            C = R.corr()
            iu = np.triu_indices(len(self.sectors), k=1)
            self.avg_corr.iloc[i] = float(np.nanmean(C.values[iu]))

        if vix_col:
            self.vix = self.prices[vix_col].reindex(self.rel_rets.index)

        # Regime classification
        for i in range(corr_window, n):
            v = self.vix.iloc[i] if not pd.isna(self.vix.iloc[i]) else 18
            ac = self.avg_corr.iloc[i]
            if v > 35 or ac > 0.75:
                self.regime.iloc[i] = "CRISIS"
            elif v > 25 or ac > 0.60:
                self.regime.iloc[i] = "TENSION"
            elif v > 18 or ac > 0.45:
                self.regime.iloc[i] = "NORMAL"
            else:
                self.regime.iloc[i] = "CALM"

        log.info("Features precomputed: %d dates, %d sectors", n, len(self.sectors))

    def run_methodology(self, method: Methodology) -> MethodologyResult:
        """הרצת מתודולוגיה אחת."""
        n = len(self.rel_rets)
        min_start = max(self.settings.corr_baseline_window, 300)
        if n < min_start + 20:
            raise ValueError(f"Not enough data: need {min_start + 20}, have {n}")

        active: Dict[str, Trade] = {}
        completed: List[Trade] = []
        daily_pnl = pd.Series(0.0, index=self.rel_rets.index, dtype=float)

        anchors = list(range(min_start, n - 1, self.step))

        for anchor in anchors:
            dt = self.rel_rets.index[anchor]

            # Build context for each sector
            for s in self.sectors:
                z = self.z_scores[s].iloc[anchor]
                if not math.isfinite(z):
                    continue

                ctx = {
                    "ticker": s,
                    "z_score": z,
                    "regime": self.regime.iloc[anchor],
                    "avg_corr": self.avg_corr.iloc[anchor],
                    "vix": self.vix.iloc[anchor] if math.isfinite(self.vix.iloc[anchor]) else 18,
                    "momentum": self.momentum[s].iloc[anchor] if math.isfinite(self.momentum[s].iloc[anchor]) else 0,
                    "vol": self.vol[s].iloc[anchor] if math.isfinite(self.vol[s].iloc[anchor]) else 0.20,
                    "dispersion": self.dispersion.iloc[anchor] if math.isfinite(self.dispersion.iloc[anchor]) else 0,
                }

                # Check entry (only if no existing trade for this sector)
                existing = [k for k in active if k.startswith(f"{s}_")]
                if not existing:
                    entry = method.should_enter(ctx)
                    if entry:
                        ticker, direction, weight, meta = entry
                        tid = f"{s}_{direction}_{dt.strftime('%Y%m%d')}"
                        active[tid] = Trade(
                            ticker=s, direction=direction, entry_idx=anchor,
                            entry_date=str(dt.date()), entry_z=z, weight=weight,
                            regime=self.regime.iloc[anchor], metadata=meta,
                        )

            # Track P&L and check exits
            to_close = []
            for tid, trade in active.items():
                s = trade.ticker
                sign = 1.0 if trade.direction == "LONG" else -1.0
                if s in self.rel_rets.columns and anchor < n:
                    daily_pnl.iloc[anchor] += sign * self.rel_rets[s].iloc[anchor] * trade.weight

                days_held = anchor - trade.entry_idx
                current_z = self.z_scores[s].iloc[anchor] if math.isfinite(self.z_scores[s].iloc[anchor]) else trade.entry_z

                exit_ctx = {
                    "current_z": current_z,
                    "days_held": days_held,
                    "regime": self.regime.iloc[anchor],
                    "dispersion": self.dispersion.iloc[anchor],
                }
                reason = method.should_exit(trade, exit_ctx)
                if reason:
                    pnl_sum = 0.0
                    for d in range(trade.entry_idx, anchor + 1):
                        if d < n and s in self.rel_rets.columns:
                            pnl_sum += sign * self.rel_rets[s].iloc[d] * trade.weight
                    trade.exit_idx = anchor
                    trade.exit_date = str(dt.date())
                    trade.exit_reason = reason
                    trade.pnl = pnl_sum
                    completed.append(trade)
                    to_close.append(tid)

            for tid in to_close:
                del active[tid]

        # Close remaining
        for tid, trade in active.items():
            s = trade.ticker
            sign = 1.0 if trade.direction == "LONG" else -1.0
            pnl_sum = sum(
                sign * self.rel_rets[s].iloc[d] * trade.weight
                for d in range(trade.entry_idx, n)
                if d < n and s in self.rel_rets.columns
            )
            trade.exit_idx = n - 1
            trade.exit_date = str(self.rel_rets.index[-1].date())
            trade.exit_reason = "END_OF_DATA"
            trade.pnl = pnl_sum
            completed.append(trade)

        return self._aggregate(method.name, method.description, completed, daily_pnl, method.get_params())

    def run_all(self, methodologies: Optional[List[Methodology]] = None) -> Dict[str, MethodologyResult]:
        """הרצת כל המתודולוגיות."""
        if methodologies is None:
            methodologies = ALL_METHODOLOGIES

        log.info("Running %d methodologies...", len(methodologies))
        for method in methodologies:
            log.info("  Running: %s", method.name)
            try:
                result = self.run_methodology(method)
                self.results[method.name] = result
                log.info("    → %d trades, Sharpe=%.3f, WR=%.1f%%, PnL=%.2f%%",
                         result.total_trades, result.sharpe, result.win_rate * 100, result.total_pnl * 100)
            except Exception as e:
                log.warning("    → FAILED: %s", e)

        return self.results

    def print_comparison(self):
        """הדפסת טבלת השוואה."""
        if not self.results:
            print("No results to compare. Run run_all() first.")
            return

        sorted_results = sorted(self.results.values(), key=lambda r: r.sharpe, reverse=True)
        print(f"\n{'Name':<25} {'Sharpe':>8} {'WR':>6} {'PnL':>8} {'#Trades':>8} {'MaxDD':>8} {'AvgHold':>8}")
        print("-" * 80)
        for r in sorted_results:
            print(f"{r.name:<25} {r.sharpe:>8.3f} {r.win_rate:>5.1%} {r.total_pnl:>7.2%} "
                  f"{r.total_trades:>8} {r.max_drawdown:>7.2%} {r.avg_holding_days:>7.0f}d")

    def save_results(self, output_dir: Optional[Path] = None):
        """שמירת תוצאות לJSON."""
        if output_dir is None:
            output_dir = ROOT / "agents" / "methodology" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        ds = date.today().isoformat()
        comparison = {}
        for name, r in self.results.items():
            comparison[name] = {
                "description": r.description,
                "sharpe": r.sharpe, "win_rate": r.win_rate,
                "total_pnl": r.total_pnl, "total_trades": r.total_trades,
                "max_drawdown": r.max_drawdown, "avg_holding_days": r.avg_holding_days,
                "calmar": r.calmar, "exits": r.exits,
                "regime_stats": r.regime_stats, "params": r.params,
            }

        path = output_dir / f"{ds}_methodology_lab.json"
        path.write_text(json.dumps(comparison, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        log.info("Results saved to %s", path.name)
        return path

    def _aggregate(self, name, desc, trades, daily_pnl, params) -> MethodologyResult:
        """חישוב מטריקות מצרפיות כולל עלויות טרנזקציה."""
        n_trades = len(trades)
        if n_trades == 0:
            return MethodologyResult(
                name=name, description=desc, total_trades=0, win_rate=0,
                avg_pnl=0, total_pnl=0, sharpe=0, max_drawdown=0, calmar=0,
                avg_holding_days=0, exits={}, regime_stats={},
                equity_curve=pd.Series(dtype=float), drawdown_curve=pd.Series(dtype=float),
                trades=[], params=params,
            )

        # Transaction cost deduction: cost_bps per side × 2 (entry + exit) × weight
        cost_per_trade = self.cost_bps / 10000.0 * 2  # Round-trip cost as fraction
        for t in trades:
            t.pnl -= cost_per_trade * t.weight  # Deduct cost from each trade P&L

        pnls = [t.pnl for t in trades]
        eq = daily_pnl.cumsum()
        dd = eq - eq.cummax()
        max_dd = float(dd.min())

        daily_std = daily_pnl.std()
        sharpe = float(daily_pnl.mean() / daily_std * np.sqrt(252)) if daily_std > 1e-10 else 0.0
        total_pnl = float(eq.iloc[-1]) if len(eq) else 0.0
        calmar = abs(total_pnl / max_dd) if abs(max_dd) > 1e-6 else 0.0

        exits = {}
        for t in trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

        regime_groups: Dict[str, List[Trade]] = {}
        for t in trades:
            regime_groups.setdefault(t.regime, []).append(t)
        regime_stats = {}
        for r, group in regime_groups.items():
            g_pnls = [t.pnl for t in group]
            regime_stats[r] = {
                "n_trades": len(group),
                "win_rate": sum(1 for t in group if t.pnl > 0) / len(group),
                "avg_pnl": float(np.mean(g_pnls)),
            }

        return MethodologyResult(
            name=name, description=desc, total_trades=n_trades,
            win_rate=sum(1 for t in trades if t.pnl > 0) / n_trades,
            avg_pnl=float(np.mean(pnls)), total_pnl=total_pnl,
            sharpe=round(sharpe, 4), max_drawdown=round(max_dd, 6),
            calmar=round(calmar, 4),
            avg_holding_days=round(float(np.mean([t.exit_idx - t.entry_idx for t in trades])), 1),
            exits=exits, regime_stats=regime_stats,
            equity_curve=eq, drawdown_curve=dd,
            trades=trades, params=params,
        )
