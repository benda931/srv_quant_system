"""
Risk Guardian Agent -- Independent risk monitor with veto power.

The guardian runs on its own schedule, monitors portfolio risk metrics,
and can HALT all trading if limits are breached. No other agent can
override the guardian's decisions.

Monitors:
- Portfolio VaR/CVaR breach (daily 95% VaR > 2% of NAV)
- Gross/net exposure limits (gross > 3x, net > 40%)
- Single-name concentration (> 25% of NAV)
- Correlation spike between positions (avg pair corr > 0.7)
- VIX regime change (VIX jumps > 5 pts intraday)
- Drawdown limit (portfolio DD > 8% -> reduce, > 12% -> halt)
- P&L velocity (losing > 1% in < 1 hour)

Actions:
- GREEN: All clear, log status
- YELLOW: Warning, notify via bus + log
- RED: HALT trading, close risky positions, alert
- BLACK: Emergency -- close ALL positions immediately

Output: agents/risk_guardian/risk_status.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# -- Root path ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
sys.path.insert(0, str(ROOT))

# -- Logging ------------------------------------------------------------------
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            _LOG_DIR / "agent_risk_guardian.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("agent_risk_guardian")

# -- Status output path -------------------------------------------------------
STATUS_PATH = Path(__file__).resolve().parent / "risk_status.json"

# -- Safe imports with fallbacks -----------------------------------------------
_IMPORTS_OK: Dict[str, bool] = {}

try:
    from analytics.portfolio_risk import PortfolioRiskEngine
    _IMPORTS_OK["portfolio_risk"] = True
except ImportError as e:
    log.warning("Could not import PortfolioRiskEngine: %s", e)
    _IMPORTS_OK["portfolio_risk"] = False

try:
    from analytics.paper_trader import PaperTrader, PORTFOLIO_PATH
    _IMPORTS_OK["paper_trader"] = True
except ImportError as e:
    log.warning("Could not import PaperTrader: %s", e)
    _IMPORTS_OK["paper_trader"] = False

try:
    from config.settings import get_settings, Settings
    _IMPORTS_OK["settings"] = True
except ImportError as e:
    log.warning("Could not import settings: %s", e)
    _IMPORTS_OK["settings"] = False

try:
    from agents.shared.agent_registry import get_registry, AgentStatus
    _IMPORTS_OK["registry"] = True
except ImportError as e:
    log.warning("Could not import agent_registry: %s", e)
    _IMPORTS_OK["registry"] = False

try:
    from scripts.agent_bus import get_bus
    _IMPORTS_OK["agent_bus"] = True
except ImportError as e:
    log.warning("Could not import agent_bus: %s", e)
    _IMPORTS_OK["agent_bus"] = False


# =============================================================================
# Risk level enum
# =============================================================================
RISK_GREEN = "GREEN"
RISK_YELLOW = "YELLOW"
RISK_RED = "RED"
RISK_BLACK = "BLACK"

_LEVEL_PRIORITY = {RISK_GREEN: 0, RISK_YELLOW: 1, RISK_RED: 2, RISK_BLACK: 3}

# =============================================================================
# Institutional Risk States — CRO Desk
# =============================================================================
RISK_STATES = [
    "SAFE", "CAUTION", "FRAGILE", "ELEVATED_RISK",
    "HALT_REQUIRED", "EMERGENCY", "INSUFFICIENT_EVIDENCE",
]
RISK_STATE_TO_LEGACY = {
    "SAFE": "GREEN",
    "CAUTION": "YELLOW",
    "FRAGILE": "YELLOW",
    "ELEVATED_RISK": "RED",
    "HALT_REQUIRED": "RED",
    "EMERGENCY": "BLACK",
    "INSUFFICIENT_EVIDENCE": "YELLOW",
}

# Institutional risk history path
_RISK_HISTORY_PATH = Path(__file__).resolve().parent / "risk_history.json"


def _worst_level(a: str, b: str) -> str:
    """Return the more severe risk level."""
    return a if _LEVEL_PRIORITY.get(a, 0) >= _LEVEL_PRIORITY.get(b, 0) else b


# =============================================================================
# Default risk thresholds
# =============================================================================
_DEFAULT_THRESHOLDS = {
    "var_limit_pct": 0.02,          # daily 95% VaR > 2% NAV
    "gross_exposure_limit": 3.0,    # gross > 3x
    "net_exposure_limit": 0.40,     # net > 40%
    "concentration_limit": 0.25,    # single-name > 25% NAV
    "correlation_spike": 0.70,      # avg pairwise corr > 0.7
    "vix_jump_pts": 5.0,            # VIX jumps > 5 pts intraday
    "drawdown_warn_pct": 0.08,      # DD > 8% -> YELLOW/reduce
    "drawdown_halt_pct": 0.12,      # DD > 12% -> RED/halt
    "pnl_velocity_pct": 0.01,       # losing > 1% in < 1 hour
}


# =============================================================================
# RiskGuardian
# =============================================================================
class RiskGuardian:
    """Independent risk monitor with veto power over all trading agents."""

    def __init__(self, settings: Optional[Any] = None) -> None:
        if settings is None and _IMPORTS_OK.get("settings"):
            try:
                self.settings = get_settings()
            except Exception:
                self.settings = None
        else:
            self.settings = settings

        self.thresholds = dict(_DEFAULT_THRESHOLDS)
        self._prices: Optional[pd.DataFrame] = None
        self._portfolio_data: Optional[dict] = None
        self._risk_engine: Optional[Any] = None

        if _IMPORTS_OK.get("portfolio_risk"):
            try:
                self._risk_engine = PortfolioRiskEngine()
            except Exception as exc:
                log.warning("Failed to initialize PortfolioRiskEngine: %s", exc)

        # Load paper portfolio data
        self._load_portfolio()
        # Load prices
        self._load_prices()

    # -----------------------------------------------------------------
    # Data loading helpers
    # -----------------------------------------------------------------

    def _load_portfolio(self) -> None:
        """Load the paper portfolio state from disk."""
        try:
            portfolio_path = ROOT / "data" / "paper_portfolio.json"
            if portfolio_path.exists():
                self._portfolio_data = json.loads(
                    portfolio_path.read_text(encoding="utf-8")
                )
                log.info(
                    "Loaded paper portfolio: %d positions",
                    len(self._portfolio_data.get("positions", [])),
                )
            else:
                log.info("No paper portfolio found at %s", portfolio_path)
                self._portfolio_data = {}
        except Exception as exc:
            log.warning("Failed to load paper portfolio: %s", exc)
            self._portfolio_data = {}

    def _load_prices(self) -> None:
        """Load latest prices from parquet if available."""
        try:
            prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if prices_path.exists():
                self._prices = pd.read_parquet(prices_path)
                log.info("Loaded prices: %s rows x %s columns", *self._prices.shape)
            else:
                log.info("No prices parquet found at %s", prices_path)
        except Exception as exc:
            log.warning("Failed to load prices: %s", exc)

    def _get_positions(self) -> List[dict]:
        """Return current open positions from paper portfolio."""
        if not self._portfolio_data:
            return []
        return self._portfolio_data.get("positions", [])

    def _get_capital(self) -> float:
        """Return portfolio capital (NAV)."""
        if not self._portfolio_data:
            return 1_000_000.0
        return float(self._portfolio_data.get("capital", 1_000_000.0))

    def _get_portfolio_weights(self) -> Dict[str, float]:
        """Derive weight dict from open positions."""
        positions = self._get_positions()
        capital = self._get_capital()
        if not positions or capital <= 0:
            return {}
        weights: Dict[str, float] = {}
        for pos in positions:
            ticker = pos.get("ticker", "")
            notional = float(pos.get("notional", 0))
            if ticker and notional:
                weights[ticker] = notional / capital
        return weights

    # -----------------------------------------------------------------
    # Individual risk checks
    # -----------------------------------------------------------------

    def check_var_breach(self) -> dict:
        """Compute VaR from portfolio_risk module and check against limit."""
        result = {"check": "var_breach", "level": RISK_GREEN, "detail": ""}

        weights = self._get_portfolio_weights()
        if not weights or self._risk_engine is None or self._prices is None:
            result["detail"] = "Insufficient data for VaR computation"
            return result

        try:
            # Filter prices to tickers in weights
            available = [t for t in weights if t in self._prices.columns]
            if not available:
                result["detail"] = "No matching price columns for positions"
                return result

            w_avail = {t: weights[t] for t in available}
            log_rets = np.log(self._prices[available] / self._prices[available].shift(1)).iloc[1:]
            cov = self._risk_engine.compute_cov(log_rets)
            var_95 = self._risk_engine.compute_var(w_avail, cov, confidence=0.95, horizon=1)

            limit = self.thresholds["var_limit_pct"]
            result["var_95_1d"] = round(var_95, 6)
            result["limit"] = limit

            if var_95 > limit:
                result["level"] = RISK_RED
                result["detail"] = (
                    f"VaR breach: daily 95%% VaR = {var_95:.4%} > limit {limit:.2%}"
                )
                log.warning(result["detail"])
            else:
                result["detail"] = f"VaR OK: {var_95:.4%} <= {limit:.2%}"
        except Exception as exc:
            log.error("VaR check failed: %s", exc)
            result["detail"] = f"VaR check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    def check_exposure_limits(self) -> dict:
        """Check gross/net exposure against limits."""
        result = {"check": "exposure_limits", "level": RISK_GREEN, "detail": ""}

        positions = self._get_positions()
        capital = self._get_capital()
        if not positions or capital <= 0:
            result["detail"] = "No positions or zero capital"
            return result

        try:
            long_notional = sum(
                float(p.get("notional", 0))
                for p in positions if p.get("direction") == "LONG"
            )
            short_notional = sum(
                float(p.get("notional", 0))
                for p in positions if p.get("direction") == "SHORT"
            )

            gross_exposure = (long_notional + short_notional) / capital
            net_exposure = abs(long_notional - short_notional) / capital

            result["gross_exposure"] = round(gross_exposure, 4)
            result["net_exposure"] = round(net_exposure, 4)

            gross_limit = self.thresholds["gross_exposure_limit"]
            net_limit = self.thresholds["net_exposure_limit"]

            breaches = []
            if gross_exposure > gross_limit:
                breaches.append(f"Gross={gross_exposure:.2f}x > {gross_limit:.1f}x")
                result["level"] = RISK_RED
            if net_exposure > net_limit:
                breaches.append(f"Net={net_exposure:.2%} > {net_limit:.0%}")
                result["level"] = _worst_level(result["level"], RISK_YELLOW)

            if breaches:
                result["detail"] = "Exposure breach: " + "; ".join(breaches)
                log.warning(result["detail"])
            else:
                result["detail"] = (
                    f"Exposure OK: gross={gross_exposure:.2f}x, net={net_exposure:.2%}"
                )
        except Exception as exc:
            log.error("Exposure check failed: %s", exc)
            result["detail"] = f"Exposure check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    def check_concentration(self) -> dict:
        """Check HHI and single-name concentration."""
        result = {"check": "concentration", "level": RISK_GREEN, "detail": ""}

        weights = self._get_portfolio_weights()
        if not weights:
            result["detail"] = "No positions for concentration check"
            return result

        try:
            abs_weights = [abs(w) for w in weights.values()]
            hhi = sum(w ** 2 for w in abs_weights)
            max_weight = max(abs_weights) if abs_weights else 0.0
            limit = self.thresholds["concentration_limit"]

            result["hhi"] = round(hhi, 4)
            result["max_single_weight"] = round(max_weight, 4)
            result["limit"] = limit

            if max_weight > limit:
                max_ticker = max(weights.keys(), key=lambda t: abs(weights[t]))
                result["level"] = RISK_RED
                result["detail"] = (
                    f"Concentration breach: {max_ticker} = {max_weight:.2%} > {limit:.0%}"
                )
                log.warning(result["detail"])
            else:
                result["detail"] = (
                    f"Concentration OK: max={max_weight:.2%}, HHI={hhi:.4f}"
                )
        except Exception as exc:
            log.error("Concentration check failed: %s", exc)
            result["detail"] = f"Concentration check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    def check_correlation_spike(self) -> dict:
        """Check average pairwise correlation between open positions."""
        result = {"check": "correlation_spike", "level": RISK_GREEN, "detail": ""}

        weights = self._get_portfolio_weights()
        if len(weights) < 2 or self._prices is None:
            result["detail"] = "Not enough positions for correlation check"
            return result

        try:
            tickers = [t for t in weights if t in self._prices.columns]
            if len(tickers) < 2:
                result["detail"] = "Not enough price data for correlation check"
                return result

            # Use last 60 days of returns
            rets = np.log(
                self._prices[tickers] / self._prices[tickers].shift(1)
            ).iloc[1:].tail(60).dropna(how="all")

            if len(rets) < 20:
                result["detail"] = "Insufficient return history for correlation"
                return result

            corr_matrix = rets.corr()
            n = len(tickers)
            # Average off-diagonal correlation
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = float(corr_matrix.values[mask].mean())

            limit = self.thresholds["correlation_spike"]
            result["avg_pairwise_corr"] = round(avg_corr, 4)
            result["limit"] = limit

            if avg_corr > limit:
                result["level"] = RISK_YELLOW
                result["detail"] = (
                    f"Correlation spike: avg pair corr = {avg_corr:.3f} > {limit:.2f}"
                )
                log.warning(result["detail"])
            else:
                result["detail"] = f"Correlation OK: avg={avg_corr:.3f} <= {limit:.2f}"
        except Exception as exc:
            log.error("Correlation check failed: %s", exc)
            result["detail"] = f"Correlation check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    def check_drawdown(self) -> dict:
        """Check portfolio drawdown vs limits."""
        result = {"check": "drawdown", "level": RISK_GREEN, "detail": ""}

        if not self._portfolio_data:
            result["detail"] = "No portfolio data for drawdown check"
            return result

        try:
            max_dd = abs(float(self._portfolio_data.get("max_drawdown", 0.0)))
            warn_limit = self.thresholds["drawdown_warn_pct"]
            halt_limit = self.thresholds["drawdown_halt_pct"]

            result["max_drawdown"] = round(max_dd, 6)
            result["warn_limit"] = warn_limit
            result["halt_limit"] = halt_limit

            if max_dd > halt_limit:
                result["level"] = RISK_RED
                result["detail"] = (
                    f"Drawdown HALT: DD={max_dd:.2%} > halt limit {halt_limit:.0%}"
                )
                log.warning(result["detail"])
            elif max_dd > warn_limit:
                result["level"] = RISK_YELLOW
                result["detail"] = (
                    f"Drawdown WARNING: DD={max_dd:.2%} > warn limit {warn_limit:.0%}"
                )
                log.warning(result["detail"])
            else:
                result["detail"] = f"Drawdown OK: {max_dd:.2%} <= {warn_limit:.0%}"
        except Exception as exc:
            log.error("Drawdown check failed: %s", exc)
            result["detail"] = f"Drawdown check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    def check_vix_regime(self) -> dict:
        """Check VIX level and rate of change."""
        result = {"check": "vix_regime", "level": RISK_GREEN, "detail": ""}

        if self._prices is None or "^VIX" not in self._prices.columns:
            result["detail"] = "No VIX data available"
            return result

        try:
            vix_series = self._prices["^VIX"].dropna()
            if len(vix_series) < 2:
                result["detail"] = "Insufficient VIX history"
                return result

            current_vix = float(vix_series.iloc[-1])
            prev_vix = float(vix_series.iloc[-2])
            vix_change = current_vix - prev_vix
            jump_limit = self.thresholds["vix_jump_pts"]

            result["current_vix"] = round(current_vix, 2)
            result["vix_1d_change"] = round(vix_change, 2)
            result["jump_limit"] = jump_limit

            if vix_change > jump_limit:
                result["level"] = RISK_RED
                result["detail"] = (
                    f"VIX SPIKE: VIX={current_vix:.1f}, change=+{vix_change:.1f} pts "
                    f"> {jump_limit:.0f} pts"
                )
                log.warning(result["detail"])
            elif current_vix > 35:
                result["level"] = RISK_YELLOW
                result["detail"] = f"VIX elevated: {current_vix:.1f}"
                log.warning(result["detail"])
            else:
                result["detail"] = (
                    f"VIX OK: {current_vix:.1f} (1d change: {vix_change:+.1f})"
                )
        except Exception as exc:
            log.error("VIX check failed: %s", exc)
            result["detail"] = f"VIX check error: {exc}"
            result["level"] = RISK_YELLOW

        return result

    # -----------------------------------------------------------------
    # Professional-grade risk analytics
    # -----------------------------------------------------------------

    def stress_test_correlation(self) -> dict:
        """
        Simulate crisis scenario where all pairwise correlations spike to 0.9.

        Computes portfolio VaR under both normal and stressed correlation
        matrices and returns the stress multiplier.

        Returns
        -------
        dict
            current_var, stressed_var, multiplier, detail
        """
        result = {
            "check": "correlation_stress_test",
            "current_var": None,
            "stressed_var": None,
            "multiplier": None,
            "detail": "",
        }
        try:
            weights = self._get_portfolio_weights()
            if not weights or self._prices is None:
                result["detail"] = "Insufficient data for correlation stress test"
                return result

            tickers = [t for t in weights if t in self._prices.columns]
            if len(tickers) < 2:
                result["detail"] = "Need at least 2 positions for stress test"
                return result

            log_rets = np.log(
                self._prices[tickers] / self._prices[tickers].shift(1)
            ).iloc[1:].dropna()

            if len(log_rets) < 30:
                result["detail"] = "Insufficient return history for stress test"
                return result

            w = np.array([weights[t] for t in tickers])
            vols = log_rets.std().values

            # Current covariance matrix
            cov_current = log_rets.cov().values
            current_var = float(np.sqrt(w @ cov_current @ w) * 1.645)  # 95% VaR

            # Stressed covariance: set all correlations to 0.9
            n = len(tickers)
            stressed_corr = np.full((n, n), 0.9)
            np.fill_diagonal(stressed_corr, 1.0)
            stressed_cov = np.outer(vols, vols) * stressed_corr
            stressed_var = float(np.sqrt(w @ stressed_cov @ w) * 1.645)

            multiplier = stressed_var / current_var if current_var > 1e-10 else 1.0

            result["current_var"] = round(current_var, 6)
            result["stressed_var"] = round(stressed_var, 6)
            result["multiplier"] = round(multiplier, 2)
            result["detail"] = (
                f"Stress test: current VaR={current_var:.4%}, "
                f"stressed VaR={stressed_var:.4%}, multiplier={multiplier:.1f}x"
            )
            log.info(result["detail"])
        except Exception as exc:
            log.error("Correlation stress test failed: %s", exc)
            result["detail"] = f"Stress test error: {exc}"

        return result

    def compute_tail_risk(self) -> dict:
        """
        Fit Generalized Pareto Distribution to the worst 5% of portfolio returns.

        Computes Expected Shortfall at 99% and 99.5%, and tail loss probabilities
        for drawdowns > 3%, > 5%, > 10%.

        Returns
        -------
        dict
            es_99, es_995, prob_loss_3pct, prob_loss_5pct, prob_loss_10pct,
            gpd_shape, gpd_scale, detail
        """
        result = {
            "check": "tail_risk_evt",
            "es_99": None,
            "es_995": None,
            "prob_loss_3pct": None,
            "prob_loss_5pct": None,
            "prob_loss_10pct": None,
            "gpd_shape": None,
            "gpd_scale": None,
            "detail": "",
        }
        try:
            weights = self._get_portfolio_weights()
            if not weights or self._prices is None:
                result["detail"] = "Insufficient data for tail risk analysis"
                return result

            tickers = [t for t in weights if t in self._prices.columns]
            if not tickers:
                result["detail"] = "No matching tickers for tail risk"
                return result

            log_rets = np.log(
                self._prices[tickers] / self._prices[tickers].shift(1)
            ).iloc[1:].dropna()

            if len(log_rets) < 60:
                result["detail"] = "Insufficient history for EVT analysis"
                return result

            # Portfolio returns
            w = np.array([weights[t] for t in tickers])
            port_rets = log_rets.values @ w
            losses = -port_rets[port_rets < 0]

            if len(losses) < 20:
                result["detail"] = "Not enough loss observations for EVT"
                return result

            # Threshold: 95th percentile of losses (worst 5%)
            threshold = float(np.percentile(losses, 95))
            exceedances = losses[losses > threshold] - threshold

            if len(exceedances) < 5:
                result["detail"] = "Too few tail exceedances for GPD fit"
                return result

            # Fit GPD using scipy
            shape, loc, scale = scipy_stats.genpareto.fit(exceedances, floc=0)

            # Expected shortfall computation via GPD
            n_total = len(port_rets)
            n_exceed = len(exceedances)
            exceed_prob = n_exceed / len(losses)

            # ES at given confidence level using GPD
            def _gpd_es(confidence: float) -> float:
                """Compute ES at given confidence from fitted GPD."""
                p = 1 - confidence
                # VaR from GPD
                if abs(shape) < 1e-10:
                    var_gpd = threshold + scale * np.log(exceed_prob / p)
                else:
                    var_gpd = threshold + (scale / shape) * (
                        (exceed_prob / p) ** shape - 1
                    )
                # ES = VaR / (1 - shape) + (scale - shape * threshold) / (1 - shape)
                if shape < 1.0:
                    es = var_gpd / (1 - shape) + (scale - shape * threshold) / (1 - shape)
                else:
                    es = var_gpd * 1.5  # fallback for extreme shape
                return float(es)

            es_99 = _gpd_es(0.99)
            es_995 = _gpd_es(0.995)

            # Tail loss probabilities from empirical distribution
            all_losses = -port_rets
            prob_3 = float(np.mean(all_losses > 0.03))
            prob_5 = float(np.mean(all_losses > 0.05))
            prob_10 = float(np.mean(all_losses > 0.10))

            result.update({
                "es_99": round(es_99, 6),
                "es_995": round(es_995, 6),
                "prob_loss_3pct": round(prob_3, 6),
                "prob_loss_5pct": round(prob_5, 6),
                "prob_loss_10pct": round(prob_10, 6),
                "gpd_shape": round(float(shape), 4),
                "gpd_scale": round(float(scale), 6),
                "detail": (
                    f"Tail risk: ES99={es_99:.4%}, ES99.5={es_995:.4%}, "
                    f"P(loss>3%)={prob_3:.4%}, P(loss>5%)={prob_5:.4%}, "
                    f"GPD shape={shape:.3f}"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("Tail risk analysis failed: %s", exc)
            result["detail"] = f"Tail risk error: {exc}"

        return result

    def monitor_portfolio_greeks(self) -> dict:
        """
        Estimate portfolio delta, beta, and gamma (convexity) vs SPY.

        Uses OLS regression of portfolio returns on SPY returns. Gamma is
        estimated from a quadratic regression term.

        Returns
        -------
        dict
            delta, beta, gamma, risk_flag (bool), detail
        """
        result = {
            "check": "portfolio_greeks",
            "delta": None,
            "beta": None,
            "gamma": None,
            "risk_flag": False,
            "detail": "",
        }
        try:
            weights = self._get_portfolio_weights()
            if not weights or self._prices is None:
                result["detail"] = "Insufficient data for portfolio greeks"
                return result

            if "SPY" not in self._prices.columns:
                result["detail"] = "SPY not available for greeks computation"
                return result

            tickers = [t for t in weights if t in self._prices.columns]
            if not tickers:
                result["detail"] = "No matching tickers for greeks"
                return result

            log_rets = np.log(
                self._prices[tickers + ["SPY"]] / self._prices[tickers + ["SPY"]].shift(1)
            ).iloc[1:].dropna()

            if len(log_rets) < 30:
                result["detail"] = "Insufficient history for greeks estimation"
                return result

            w = np.array([weights[t] for t in tickers])
            port_rets = log_rets[tickers].values @ w
            spy_rets = log_rets["SPY"].values

            # Linear regression: port_ret = alpha + beta * spy_ret
            # Use numpy polyfit for simplicity
            beta_coeffs = np.polyfit(spy_rets, port_rets, 1)
            beta = float(beta_coeffs[0])
            delta = beta  # For equity portfolios, delta ~ beta

            # Quadratic regression for gamma (convexity)
            quad_coeffs = np.polyfit(spy_rets, port_rets, 2)
            gamma = float(quad_coeffs[0])  # coefficient of x^2

            risk_flag = abs(beta) > 1.5

            result.update({
                "delta": round(delta, 4),
                "beta": round(beta, 4),
                "gamma": round(gamma, 4),
                "risk_flag": risk_flag,
                "detail": (
                    f"Portfolio greeks: delta={delta:.3f}, beta={beta:.3f}, "
                    f"gamma={gamma:.3f}"
                    + (f" [WARNING: |beta|={abs(beta):.2f} > 1.5]" if risk_flag else "")
                ),
            })
            log.info(result["detail"])

            if risk_flag:
                log.warning("Portfolio beta exceeds safe bounds: %.3f", beta)

        except Exception as exc:
            log.error("Portfolio greeks computation failed: %s", exc)
            result["detail"] = f"Greeks error: {exc}"

        return result

    def check_liquidity_risk(self) -> dict:
        """
        Estimate liquidity risk for each position.

        Uses price volatility as a proxy for average daily volume (ADV).
        Flags positions that are > 5% of estimated ADV or would take > 3 days
        to unwind.

        Returns
        -------
        dict
            positions_at_risk (list), n_flagged, detail
        """
        result = {
            "check": "liquidity_risk",
            "positions_at_risk": [],
            "n_flagged": 0,
            "detail": "",
        }
        try:
            positions = self._get_positions()
            if not positions or self._prices is None:
                result["detail"] = "No positions or price data for liquidity check"
                return result

            flagged = []
            for pos in positions:
                ticker = pos.get("ticker", "")
                notional = float(pos.get("notional", 0))
                if ticker not in self._prices.columns or notional <= 0:
                    continue

                px = self._prices[ticker].dropna()
                if len(px) < 30:
                    continue

                current_price = float(px.iloc[-1])
                # Estimate ADV from realized volatility and price level
                # ADV ~ price * daily_vol * empirical_volume_factor
                # We use vol as proxy: higher vol ETFs tend to have higher turnover
                daily_vol = float(px.pct_change().iloc[-20:].std())
                # Conservative ADV estimate: assume turnover ~ 1% of market cap proxy
                # For sector ETFs, rough ADV in dollar terms
                estimated_adv_dollars = current_price * max(daily_vol, 0.005) * 1e6

                pct_of_adv = notional / estimated_adv_dollars if estimated_adv_dollars > 0 else 1.0
                days_to_unwind = notional / (estimated_adv_dollars * 0.10) if estimated_adv_dollars > 0 else 99

                if pct_of_adv > 0.05 or days_to_unwind > 3:
                    flagged.append({
                        "ticker": ticker,
                        "notional": round(notional, 0),
                        "estimated_adv": round(estimated_adv_dollars, 0),
                        "pct_of_adv": round(pct_of_adv, 4),
                        "days_to_unwind": round(days_to_unwind, 1),
                        "flag_reason": (
                            f"{'ADV>' if pct_of_adv > 0.05 else ''}"
                            f"{'5%' if pct_of_adv > 0.05 else ''}"
                            f"{', ' if pct_of_adv > 0.05 and days_to_unwind > 3 else ''}"
                            f"{'unwind>' if days_to_unwind > 3 else ''}"
                            f"{'3d' if days_to_unwind > 3 else ''}"
                        ),
                    })

            result["positions_at_risk"] = flagged
            result["n_flagged"] = len(flagged)
            if flagged:
                tickers_str = ", ".join(f["ticker"] for f in flagged)
                result["detail"] = f"Liquidity risk: {len(flagged)} positions flagged ({tickers_str})"
                log.warning(result["detail"])
            else:
                result["detail"] = "Liquidity OK: all positions within ADV limits"
                log.info(result["detail"])
        except Exception as exc:
            log.error("Liquidity risk check failed: %s", exc)
            result["detail"] = f"Liquidity check error: {exc}"

        return result

    def format_alert(self, status: dict) -> str:
        """
        Create a formatted alert message suitable for Telegram/email.

        Includes risk level, breaches, recommendations, and portfolio snapshot
        in both Hebrew and English.

        Parameters
        ----------
        status : dict
            The risk status dict from run_all_checks().

        Returns
        -------
        str
            Formatted multi-line alert message.
        """
        try:
            level = status.get("level", "UNKNOWN")
            ts = status.get("timestamp", datetime.now(timezone.utc).isoformat())
            breaches = status.get("breaches", [])
            recommendations = status.get("recommendations", [])
            n_pos = status.get("n_positions", 0)
            capital = status.get("capital", 0)

            level_emoji_map = {
                "GREEN": "V",
                "YELLOW": "!",
                "RED": "X",
                "BLACK": "XXX",
            }
            level_icon = level_emoji_map.get(level, "?")

            # Hebrew level names
            level_heb = {
                "GREEN": "ירוק - תקין",
                "YELLOW": "צהוב - אזהרה",
                "RED": "אדום - עצור מסחר",
                "BLACK": "שחור - חירום מוחלט",
            }

            lines = [
                f"=== SRV Risk Guardian Alert ===",
                f"Status: [{level_icon}] {level}",
                f"סטטוס: {level_heb.get(level, level)}",
                f"Time: {ts[:19]}",
                f"",
                f"--- Portfolio Snapshot ---",
                f"Positions: {n_pos}",
                f"Capital: ${capital:,.0f}",
                f"",
            ]

            if breaches:
                lines.append("--- Breaches / הפרות ---")
                for i, b in enumerate(breaches, 1):
                    lines.append(f"  {i}. {b}")
                lines.append("")

            if recommendations:
                lines.append("--- Recommendations / המלצות ---")
                for r in recommendations:
                    lines.append(f"  - {r}")
                lines.append("")

            # Add advanced analytics summary if available
            checks = status.get("checks", [])
            advanced = [c for c in checks if c.get("check") in (
                "correlation_stress_test", "tail_risk_evt",
                "portfolio_greeks", "liquidity_risk",
            )]
            if advanced:
                lines.append("--- Advanced Analytics ---")
                for c in advanced:
                    lines.append(f"  [{c['check']}] {c.get('detail', 'N/A')}")
                lines.append("")

            lines.append("=== End Alert ===")
            alert = "\n".join(lines)
            log.info("Alert formatted: %d lines, level=%s", len(lines), level)
            return alert
        except Exception as exc:
            log.error("Alert formatting failed: %s", exc)
            return f"RISK ALERT: {status.get('level', 'UNKNOWN')} (formatting error: {exc})"

    # =================================================================
    # INSTITUTIONAL CRO DESK — New methods below
    # =================================================================

    # -----------------------------------------------------------------
    # 2. RiskInputAssembler
    # -----------------------------------------------------------------

    def assemble_risk_inputs(self) -> Dict:
        """
        Load all upstream agent outputs needed for institutional risk assessment.

        Loads:
          - Portfolio Construction weights
          - Regime Forecast
          - Alpha Decay status
          - Methodology machine_summary (latest report)
          - Optimizer machine_summary

        Returns dict with available_inputs list and each loaded payload.
        """
        inputs: Dict[str, Any] = {"available_inputs": []}

        # Portfolio Construction weights
        try:
            pw_path = ROOT / "agents" / "portfolio_construction" / "portfolio_weights.json"
            if pw_path.exists():
                inputs["portfolio_weights"] = json.loads(pw_path.read_text(encoding="utf-8"))
                inputs["available_inputs"].append("portfolio_weights")
        except Exception as exc:
            log.debug("Could not load portfolio_weights: %s", exc)

        # Regime Forecast
        try:
            rf_path = ROOT / "agents" / "regime_forecaster" / "regime_forecast.json"
            if rf_path.exists():
                inputs["regime_forecast"] = json.loads(rf_path.read_text(encoding="utf-8"))
                inputs["available_inputs"].append("regime_forecast")
        except Exception as exc:
            log.debug("Could not load regime_forecast: %s", exc)

        # Alpha Decay
        try:
            ad_path = ROOT / "agents" / "alpha_decay" / "decay_status.json"
            if ad_path.exists():
                inputs["decay_status"] = json.loads(ad_path.read_text(encoding="utf-8"))
                inputs["available_inputs"].append("decay_status")
        except Exception as exc:
            log.debug("Could not load decay_status: %s", exc)

        # Methodology machine_summary — find latest report
        try:
            reports_dir = ROOT / "agents" / "methodology" / "reports"
            if reports_dir.exists():
                report_files = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                for rp in report_files:
                    try:
                        data = json.loads(rp.read_text(encoding="utf-8"))
                        ms = data.get("machine_summary") or data.get("machine_summary_v3")
                        if ms:
                            inputs["methodology_machine_summary"] = ms
                            inputs["available_inputs"].append("methodology_machine_summary")
                            break
                    except Exception:
                        continue
        except Exception as exc:
            log.debug("Could not load methodology machine_summary: %s", exc)

        # Optimizer machine_summary — via bus or file
        try:
            if _IMPORTS_OK.get("agent_bus"):
                bus = get_bus()
                opt_report = bus.latest("agent_optimizer")
                if isinstance(opt_report, dict):
                    opt_ms = opt_report.get("machine_summary", {})
                    if opt_ms:
                        inputs["optimizer_machine_summary"] = opt_ms
                        inputs["available_inputs"].append("optimizer_machine_summary")
        except Exception as exc:
            log.debug("Could not load optimizer machine_summary: %s", exc)

        log.info("Risk inputs assembled: %s", inputs["available_inputs"])
        return inputs

    # -----------------------------------------------------------------
    # 3. DataSufficiencyEngine
    # -----------------------------------------------------------------

    def check_data_sufficiency(self, risk_inputs: Dict) -> Dict:
        """
        Check whether we have enough data to make risk decisions.

        Returns evidence_quality_score (0-1), missing_inputs, stale_data_flags,
        sufficient (bool). If not sufficient, risk_state = INSUFFICIENT_EVIDENCE.
        """
        missing: List[str] = []
        stale_flags: List[str] = []
        checks_passed = 0
        total_checks = 5

        # 1. Positions exist?
        positions = self._get_positions()
        if positions:
            checks_passed += 1
        else:
            missing.append("positions")

        # 2. Prices fresh?
        if self._prices is not None and len(self._prices) > 0:
            try:
                last_date = pd.Timestamp(self._prices.index[-1])
                now = pd.Timestamp.now(tz="UTC")
                if hasattr(last_date, "tz") and last_date.tz is None:
                    last_date = last_date.tz_localize("UTC")
                days_stale = (now - last_date).days
                if days_stale <= 3:
                    checks_passed += 1
                else:
                    stale_flags.append(f"prices_stale_{days_stale}d")
            except Exception:
                checks_passed += 1  # benefit of doubt
        else:
            missing.append("prices")

        # 3. Regime inputs available?
        if "regime_forecast" in risk_inputs.get("available_inputs", []):
            checks_passed += 1
        else:
            missing.append("regime_forecast")

        # 4. Decay available?
        if "decay_status" in risk_inputs.get("available_inputs", []):
            checks_passed += 1
        else:
            missing.append("decay_status")

        # 5. Allocation available?
        if "portfolio_weights" in risk_inputs.get("available_inputs", []):
            checks_passed += 1
        else:
            missing.append("portfolio_weights")

        evidence_quality = checks_passed / total_checks if total_checks > 0 else 0.0
        sufficient = evidence_quality >= 0.6 and "positions" not in missing

        result = {
            "evidence_quality_score": round(evidence_quality, 2),
            "missing_inputs": missing,
            "stale_data_flags": stale_flags,
            "sufficient": sufficient,
        }
        if not sufficient:
            result["risk_state"] = "INSUFFICIENT_EVIDENCE"
        log.info("Data sufficiency: score=%.2f, sufficient=%s, missing=%s",
                 evidence_quality, sufficient, missing)
        return result

    # -----------------------------------------------------------------
    # 4. FragilityEngine
    # -----------------------------------------------------------------

    def detect_fragility(self, risk_inputs: Dict) -> Dict:
        """
        Forward-looking fragility detection.

        Checks:
          - Rising correlation + high concentration -> fragility
          - Regime transition_risk > 30% + gross > 1.5x -> fragility
          - Multiple sleeves in EARLY_DECAY/STRUCTURAL_DECAY with combined weight > 30%
          - Proposed turnover spike into TENSION/CRISIS -> fragility

        Returns fragility_score (0-1), fragility_drivers list,
        fragility_state (STABLE/BUILDING/CRITICAL).
        """
        drivers: List[str] = []
        score_components: List[float] = []

        # 1. Correlation + concentration
        try:
            corr_check = self.check_correlation_spike()
            conc_check = self.check_concentration()
            avg_corr = corr_check.get("avg_pairwise_corr", 0.0)
            max_weight = conc_check.get("max_single_weight", 0.0)
            if avg_corr > 0.5 and max_weight > 0.15:
                component = min(1.0, (avg_corr - 0.3) * 1.5 + (max_weight - 0.1) * 2.0)
                score_components.append(component)
                drivers.append(f"corr_concentration: avg_corr={avg_corr:.3f}, max_wt={max_weight:.2%}")
        except Exception as exc:
            log.debug("Fragility corr+conc check: %s", exc)

        # 2. Regime transition risk + gross exposure
        try:
            regime = risk_inputs.get("regime_forecast", {})
            transition_prob = float(regime.get("transition_probability", 0.0))
            exp_check = self.check_exposure_limits()
            gross = exp_check.get("gross_exposure", 0.0)
            if transition_prob > 0.30 and gross > 1.5:
                component = min(1.0, (transition_prob - 0.2) * 2.0 + (gross - 1.0) * 0.5)
                score_components.append(component)
                drivers.append(f"regime_transition_risk: trans_prob={transition_prob:.2f}, gross={gross:.2f}x")
        except Exception as exc:
            log.debug("Fragility regime check: %s", exc)

        # 3. Decay sleeves with high combined weight
        try:
            decay = risk_inputs.get("decay_status", {})
            decay_level = decay.get("decay_level", "HEALTHY")
            pw = risk_inputs.get("portfolio_weights", {})
            proposed_weights = pw.get("weights", {})
            if decay_level in ("EARLY_DECAY", "STRUCTURAL_DECAY"):
                total_weight = sum(abs(v) for v in proposed_weights.values())
                if total_weight > 0.30:
                    component = min(1.0, total_weight * 1.5)
                    score_components.append(component)
                    drivers.append(f"decay_sleeves: level={decay_level}, combined_weight={total_weight:.2%}")
        except Exception as exc:
            log.debug("Fragility decay check: %s", exc)

        # 4. Proposed turnover into bad regime
        try:
            regime = risk_inputs.get("regime_forecast", {})
            predicted = regime.get("predicted_regime", "NORMAL")
            if predicted in ("TENSION", "CRISIS"):
                pw = risk_inputs.get("portfolio_weights", {})
                proposed_weights = pw.get("weights", {})
                current_weights = self._get_portfolio_weights()
                if proposed_weights and current_weights:
                    all_tickers = set(list(proposed_weights.keys()) + list(current_weights.keys()))
                    turnover = sum(
                        abs(proposed_weights.get(t, 0) - current_weights.get(t, 0))
                        for t in all_tickers
                    )
                    if turnover > 0.30:
                        component = min(1.0, turnover * 1.2)
                        score_components.append(component)
                        drivers.append(f"turnover_in_{predicted}: turnover={turnover:.2%}")
        except Exception as exc:
            log.debug("Fragility turnover check: %s", exc)

        fragility_score = float(np.mean(score_components)) if score_components else 0.0
        fragility_score = min(1.0, max(0.0, fragility_score))

        if fragility_score >= 0.7:
            fragility_state = "CRITICAL"
        elif fragility_score >= 0.35:
            fragility_state = "BUILDING"
        else:
            fragility_state = "STABLE"

        result = {
            "fragility_score": round(fragility_score, 4),
            "fragility_drivers": drivers,
            "fragility_state": fragility_state,
        }
        log.info("Fragility: score=%.3f, state=%s, drivers=%d",
                 fragility_score, fragility_state, len(drivers))
        return result

    # -----------------------------------------------------------------
    # 5. Proposed Portfolio Risk Assessment
    # -----------------------------------------------------------------

    def assess_proposed_portfolio(self, proposed_weights: Dict[str, float],
                                  risk_inputs: Dict) -> Dict:
        """
        Assess whether a proposed portfolio allocation is acceptable.

        Checks gross/net limits, concentration, governance compliance,
        overweighting decaying sleeves, and turnover for current regime.
        """
        reject_reasons: List[str] = []
        proposed_risk_score = 0.0

        if not proposed_weights:
            return {
                "acceptable": True,
                "haircut_required": False,
                "reject_reasons": ["no_proposed_weights"],
                "proposed_risk_score": 0.0,
            }

        # Gross / net
        long_sum = sum(v for v in proposed_weights.values() if v > 0)
        short_sum = sum(abs(v) for v in proposed_weights.values() if v < 0)
        gross = long_sum + short_sum
        net = abs(long_sum - short_sum)

        gross_limit = self.thresholds["gross_exposure_limit"]
        net_limit = self.thresholds["net_exposure_limit"]
        conc_limit = self.thresholds["concentration_limit"]

        if gross > gross_limit:
            reject_reasons.append(f"proposed_gross={gross:.2f}x > {gross_limit:.1f}x")
            proposed_risk_score += 0.3
        if net > net_limit:
            reject_reasons.append(f"proposed_net={net:.2%} > {net_limit:.0%}")
            proposed_risk_score += 0.15

        # Concentration
        max_wt = max(abs(v) for v in proposed_weights.values()) if proposed_weights else 0.0
        if max_wt > conc_limit:
            max_ticker = max(proposed_weights.keys(), key=lambda t: abs(proposed_weights[t]))
            reject_reasons.append(f"concentration: {max_ticker}={max_wt:.2%} > {conc_limit:.0%}")
            proposed_risk_score += 0.2

        # Overweight decaying sleeves?
        decay = risk_inputs.get("decay_status", {})
        decay_level = decay.get("decay_level", "HEALTHY")
        if decay_level in ("EARLY_DECAY", "STRUCTURAL_DECAY"):
            total_proposed = sum(abs(v) for v in proposed_weights.values())
            if total_proposed > 0.50:
                reject_reasons.append(f"overweight_decay: total={total_proposed:.2%}, decay={decay_level}")
                proposed_risk_score += 0.15

        # Turnover for regime
        regime = risk_inputs.get("regime_forecast", {})
        predicted = regime.get("predicted_regime", "NORMAL")
        current_weights = self._get_portfolio_weights()
        if current_weights and predicted in ("TENSION", "CRISIS"):
            all_tickers = set(list(proposed_weights.keys()) + list(current_weights.keys()))
            turnover = sum(
                abs(proposed_weights.get(t, 0) - current_weights.get(t, 0))
                for t in all_tickers
            )
            if turnover > 0.40:
                reject_reasons.append(
                    f"high_turnover_in_{predicted}: turnover={turnover:.2%}"
                )
                proposed_risk_score += 0.2

        proposed_risk_score = min(1.0, proposed_risk_score)
        acceptable = proposed_risk_score < 0.4
        haircut_required = 0.2 <= proposed_risk_score < 0.4

        result = {
            "acceptable": acceptable,
            "haircut_required": haircut_required,
            "reject_reasons": reject_reasons,
            "proposed_risk_score": round(proposed_risk_score, 4),
            "proposed_gross": round(gross, 4),
            "proposed_net": round(net, 4),
            "proposed_max_weight": round(max_wt, 4),
        }
        log.info("Proposed portfolio assessment: acceptable=%s, risk_score=%.3f",
                 acceptable, proposed_risk_score)
        return result

    # -----------------------------------------------------------------
    # 6. Sleeve-Level Risk Analysis
    # -----------------------------------------------------------------

    def analyze_sleeve_risk(self, risk_inputs: Dict) -> List[Dict]:
        """
        For each position/sleeve, compute risk contribution, decay health,
        regime mismatch, governance quality, and recommended action.

        Returns list sorted by risk_contribution desc.
        """
        positions = self._get_positions()
        capital = self._get_capital()
        if not positions or capital <= 0:
            return []

        decay = risk_inputs.get("decay_status", {})
        decay_level = decay.get("decay_level", "HEALTHY")
        regime = risk_inputs.get("regime_forecast", {})
        predicted_regime = regime.get("predicted_regime", "NORMAL")
        methodology_ms = risk_inputs.get("methodology_machine_summary", {})

        sleeves: List[Dict] = []
        total_notional = sum(abs(float(p.get("notional", 0))) for p in positions)

        for pos in positions:
            ticker = pos.get("ticker", "UNKNOWN")
            notional = abs(float(pos.get("notional", 0)))
            direction = pos.get("direction", "LONG")
            capital_weight = notional / capital if capital > 0 else 0.0

            # Risk contribution (simple proportional)
            risk_contribution = notional / total_notional if total_notional > 0 else 0.0

            # Decay health
            if decay_level in ("STRUCTURAL_DECAY",):
                decay_health = "POOR"
            elif decay_level in ("EARLY_DECAY",):
                decay_health = "DEGRADING"
            else:
                decay_health = "HEALTHY"

            # Regime mismatch: long in CRISIS is riskier
            regime_mismatch = False
            if predicted_regime in ("CRISIS",) and direction == "LONG" and capital_weight > 0.10:
                regime_mismatch = True
            elif predicted_regime in ("TENSION",) and capital_weight > 0.20:
                regime_mismatch = True

            # Governance quality — check methodology
            governance_quality = "approved"
            if methodology_ms:
                disabled = methodology_ms.get("strategies_to_disable", [])
                if ticker in disabled:
                    governance_quality = "rejected"

            # Determine action
            if governance_quality == "rejected":
                action = "DISABLE"
            elif decay_health == "POOR" and regime_mismatch:
                action = "CUT"
            elif decay_health == "DEGRADING" or regime_mismatch:
                action = "HAIRCUT"
            else:
                action = "KEEP"

            sleeves.append({
                "ticker": ticker,
                "direction": direction,
                "capital_weight": round(capital_weight, 4),
                "risk_contribution": round(risk_contribution, 4),
                "decay_health": decay_health,
                "regime_mismatch": regime_mismatch,
                "governance_quality": governance_quality,
                "action": action,
            })

        sleeves.sort(key=lambda s: s["risk_contribution"], reverse=True)
        log.info("Sleeve risk analysis: %d sleeves, actions=%s",
                 len(sleeves), [s["action"] for s in sleeves])
        return sleeves

    # -----------------------------------------------------------------
    # 7. Veto Engine
    # -----------------------------------------------------------------

    def compute_veto_status(self, all_checks: Dict, fragility: Dict,
                            proposed_assessment: Dict) -> Dict:
        """
        Compute veto status — determines what actions are allowed/blocked.

        Rules:
          - BLACK/EMERGENCY -> emergency_unwind + can't allocate + can't execute
          - RED/HALT_REQUIRED -> can't allocate, can't execute new, must reduce
          - FRAGILE with fragility > 0.7 -> can't allocate new
          - INSUFFICIENT_EVIDENCE -> can't allocate (be conservative)
        """
        legacy_level = all_checks.get("level", RISK_GREEN)
        fragility_score = fragility.get("fragility_score", 0.0)
        fragility_state = fragility.get("fragility_state", "STABLE")

        can_allocate = True
        can_execute = True
        must_reduce = False
        emergency_unwind = False
        veto_reasons: List[str] = []
        veto_scope = "none"
        vetoed_sleeves: List[str] = []

        # BLACK / EMERGENCY
        if legacy_level == RISK_BLACK:
            emergency_unwind = True
            can_allocate = False
            can_execute = False
            must_reduce = True
            veto_reasons.append("BLACK_LEVEL: emergency unwind required")
            veto_scope = "portfolio_wide"

        # RED / HALT_REQUIRED
        elif legacy_level == RISK_RED:
            can_allocate = False
            can_execute = False
            must_reduce = True
            veto_reasons.append("RED_LEVEL: halt new trades, reduce risk")
            veto_scope = "portfolio_wide"

        # FRAGILE with high score
        if fragility_score > 0.7:
            can_allocate = False
            veto_reasons.append(f"FRAGILE: fragility_score={fragility_score:.2f} > 0.7")
            if veto_scope == "none":
                veto_scope = "regime_specific"

        # INSUFFICIENT_EVIDENCE from proposed assessment or data sufficiency
        if not proposed_assessment.get("acceptable", True) and proposed_assessment.get("reject_reasons"):
            can_allocate = False
            veto_reasons.extend([f"proposed_rejected: {r}" for r in proposed_assessment["reject_reasons"]])
            if veto_scope == "none":
                veto_scope = "sleeve_specific"

        # Collect vetoed sleeves from proposed assessment
        if proposed_assessment.get("reject_reasons"):
            for reason in proposed_assessment["reject_reasons"]:
                if "concentration" in reason:
                    # Extract ticker from reason string
                    parts = reason.split(":")
                    if len(parts) > 1:
                        ticker_part = parts[1].strip().split("=")[0].strip()
                        if ticker_part:
                            vetoed_sleeves.append(ticker_part)

        result = {
            "can_allocate_new_risk": can_allocate,
            "can_execute_new_trades": can_execute,
            "must_reduce_existing_risk": must_reduce,
            "emergency_unwind_required": emergency_unwind,
            "veto_reasons": veto_reasons,
            "veto_scope": veto_scope if veto_reasons else "none",
            "vetoed_sleeves": vetoed_sleeves,
        }
        log.info("Veto status: allocate=%s, execute=%s, reduce=%s, emergency=%s",
                 can_allocate, can_execute, must_reduce, emergency_unwind)
        return result

    # -----------------------------------------------------------------
    # 8. Deterministic Action Engine
    # -----------------------------------------------------------------

    def generate_risk_actions(self, risk_state: str, checks: Dict,
                              fragility: Dict, veto: Dict) -> List[Dict]:
        """
        Generate deterministic risk actions based on current state.

        Actions: APPROVE_CURRENT, APPROVE_PROPOSED, HAIRCUT_PROPOSED,
        REDUCE_GROSS, REDUCE_NET, CAP_CONCENTRATION, DISABLE_SLEEVE,
        REGIME_BLOCK, FREEZE_NEW_RISK, HALT_NEW_TRADES, FORCE_DELEVER,
        EMERGENCY_UNWIND
        """
        actions: List[Dict] = []
        legacy = RISK_STATE_TO_LEGACY.get(risk_state, "YELLOW")
        fragility_score = fragility.get("fragility_score", 0.0)

        if risk_state == "EMERGENCY" or legacy == RISK_BLACK:
            actions.append({
                "action": "EMERGENCY_UNWIND",
                "target": "portfolio_wide",
                "urgency": "IMMEDIATE",
                "rationale": "BLACK/EMERGENCY state requires full unwind",
                "confidence": 1.0,
                "downstream_agents": ["execution", "portfolio_construction"],
            })
            return actions

        if risk_state in ("HALT_REQUIRED",) or legacy == RISK_RED:
            actions.append({
                "action": "HALT_NEW_TRADES",
                "target": "portfolio_wide",
                "urgency": "HIGH",
                "rationale": f"Risk state {risk_state}: no new trades allowed",
                "confidence": 0.95,
                "downstream_agents": ["execution"],
            })
            actions.append({
                "action": "FORCE_DELEVER",
                "target": "portfolio_wide",
                "urgency": "HIGH",
                "rationale": "RED-level requires deleveraging",
                "confidence": 0.90,
                "downstream_agents": ["execution", "portfolio_construction"],
            })

        if fragility_score > 0.7:
            actions.append({
                "action": "FREEZE_NEW_RISK",
                "target": "portfolio_wide",
                "urgency": "HIGH",
                "rationale": f"Fragility critical: {fragility_score:.2f}",
                "confidence": 0.85,
                "downstream_agents": ["optimizer", "portfolio_construction"],
            })

        if fragility_score > 0.35:
            actions.append({
                "action": "REDUCE_GROSS",
                "target": "portfolio_wide",
                "urgency": "MEDIUM",
                "rationale": f"Fragility building: {fragility_score:.2f}",
                "confidence": 0.75,
                "downstream_agents": ["portfolio_construction"],
            })

        if risk_state == "INSUFFICIENT_EVIDENCE":
            actions.append({
                "action": "FREEZE_NEW_RISK",
                "target": "portfolio_wide",
                "urgency": "MEDIUM",
                "rationale": "Insufficient evidence — conservative stance",
                "confidence": 0.80,
                "downstream_agents": ["optimizer", "portfolio_construction", "execution"],
            })

        # Check individual breaches
        for check in checks.get("checks", []):
            if check.get("level") == RISK_RED and check.get("check") == "concentration":
                actions.append({
                    "action": "CAP_CONCENTRATION",
                    "target": check.get("detail", "unknown"),
                    "urgency": "HIGH",
                    "rationale": check.get("detail", "concentration breach"),
                    "confidence": 0.90,
                    "downstream_agents": ["portfolio_construction"],
                })

        # If nothing flagged, approve current
        if not actions and risk_state in ("SAFE", "CAUTION"):
            actions.append({
                "action": "APPROVE_CURRENT",
                "target": "portfolio_wide",
                "urgency": "LOW",
                "rationale": f"Risk state {risk_state}: portfolio within limits",
                "confidence": 0.90,
                "downstream_agents": [],
            })

        log.info("Generated %d risk actions for state %s", len(actions), risk_state)
        return actions

    # -----------------------------------------------------------------
    # 9. Scenario Stress Engine
    # -----------------------------------------------------------------

    def run_stress_scenarios(self) -> Dict:
        """
        Run 6 stress scenarios:
          1. Crisis correlation convergence (all corr -> 0.9)
          2. Volatility shock (vol x 2)
          3. Gap-down (-5% SPY)
          4. Liquidity deterioration (ADV x 0.3)
          5. Regime transition stress (instant CRISIS)
          6. Combined (corr + vol + liquidity)

        Returns dict of scenario results.
        """
        scenarios: Dict[str, Any] = {}
        weights = self._get_portfolio_weights()

        if not weights or self._prices is None:
            return {"detail": "Insufficient data for stress scenarios", "scenarios": {}}

        tickers = [t for t in weights if t in self._prices.columns]
        if len(tickers) < 1:
            return {"detail": "No matching tickers for stress", "scenarios": {}}

        try:
            log_rets = np.log(
                self._prices[tickers] / self._prices[tickers].shift(1)
            ).iloc[1:].dropna()
            if len(log_rets) < 30:
                return {"detail": "Insufficient history for stress scenarios", "scenarios": {}}

            w = np.array([weights[t] for t in tickers])
            vols = log_rets.std().values
            cov_normal = log_rets.cov().values
            capital = self._get_capital()

            def _portfolio_var(cov: np.ndarray) -> float:
                return float(np.sqrt(w @ cov @ w) * 1.645)

            normal_var = _portfolio_var(cov_normal)

            # 1. Crisis correlation convergence
            n = len(tickers)
            stressed_corr = np.full((n, n), 0.9)
            np.fill_diagonal(stressed_corr, 1.0)
            cov_crisis_corr = np.outer(vols, vols) * stressed_corr
            crisis_corr_var = _portfolio_var(cov_crisis_corr)
            scenarios["crisis_correlation"] = {
                "estimated_loss": round(crisis_corr_var * capital, 2),
                "stressed_var": round(crisis_corr_var, 6),
                "multiplier": round(crisis_corr_var / normal_var if normal_var > 1e-10 else 1.0, 2),
                "top_drivers": tickers[:3],
            }

            # 2. Volatility shock (vol x 2)
            cov_vol_shock = np.outer(vols * 2, vols * 2) * log_rets.corr().values
            vol_shock_var = _portfolio_var(cov_vol_shock)
            scenarios["volatility_shock"] = {
                "estimated_loss": round(vol_shock_var * capital, 2),
                "stressed_var": round(vol_shock_var, 6),
                "multiplier": round(vol_shock_var / normal_var if normal_var > 1e-10 else 1.0, 2),
                "top_drivers": tickers[:3],
            }

            # 3. Gap-down (-5% SPY)
            port_beta = 1.0
            if "SPY" in self._prices.columns:
                spy_rets = np.log(self._prices["SPY"] / self._prices["SPY"].shift(1)).iloc[1:].dropna()
                port_rets_hist = log_rets.values @ w
                if len(spy_rets) >= 30 and len(port_rets_hist) >= 30:
                    min_len = min(len(spy_rets), len(port_rets_hist))
                    cov_spy = np.cov(port_rets_hist[-min_len:], spy_rets.values[-min_len:])
                    if cov_spy[1, 1] > 1e-10:
                        port_beta = float(cov_spy[0, 1] / cov_spy[1, 1])
            gap_loss = abs(0.05 * port_beta * capital * sum(abs(ww) for ww in w))
            scenarios["gap_down_5pct"] = {
                "estimated_loss": round(gap_loss, 2),
                "portfolio_beta": round(port_beta, 3),
                "top_drivers": tickers[:3],
            }

            # 4. Liquidity deterioration (ADV x 0.3)
            liquidity_impact = 0.0
            for t in tickers:
                pos_notional = abs(weights[t]) * capital
                px = self._prices[t].dropna()
                if len(px) > 20:
                    daily_vol = float(px.pct_change().iloc[-20:].std())
                    est_adv = float(px.iloc[-1]) * max(daily_vol, 0.005) * 1e6
                    reduced_adv = est_adv * 0.3
                    if reduced_adv > 0:
                        days_to_unwind = pos_notional / (reduced_adv * 0.10)
                        if days_to_unwind > 5:
                            liquidity_impact += pos_notional * 0.02  # slippage estimate
            scenarios["liquidity_deterioration"] = {
                "estimated_loss": round(liquidity_impact, 2),
                "impact_pct": round(liquidity_impact / capital if capital > 0 else 0.0, 4),
                "top_drivers": tickers[:3],
            }

            # 5. Regime transition stress (instant CRISIS)
            crisis_var = _portfolio_var(cov_crisis_corr)  # Use crisis corr as proxy
            crisis_loss = crisis_var * capital * 1.5  # amplified
            scenarios["regime_transition"] = {
                "estimated_loss": round(crisis_loss, 2),
                "stressed_var": round(crisis_var * 1.5, 6),
                "top_drivers": tickers[:3],
            }

            # 6. Combined
            cov_combined = np.outer(vols * 2, vols * 2) * stressed_corr
            combined_var = _portfolio_var(cov_combined)
            combined_loss = combined_var * capital + liquidity_impact
            scenarios["combined"] = {
                "estimated_loss": round(combined_loss, 2),
                "stressed_var": round(combined_var, 6),
                "multiplier": round(combined_var / normal_var if normal_var > 1e-10 else 1.0, 2),
                "top_drivers": tickers[:3],
            }

        except Exception as exc:
            log.error("Stress scenarios failed: %s", exc)
            return {"detail": f"Stress scenario error: {exc}", "scenarios": scenarios}

        log.info("Stress scenarios completed: %d scenarios", len(scenarios))
        return {"scenarios": scenarios, "normal_var": round(normal_var, 6)}

    # -----------------------------------------------------------------
    # 10. Risk Scoring Framework
    # -----------------------------------------------------------------

    def compute_risk_scores(self, all_checks: Dict, fragility: Dict,
                            scenarios: Dict) -> Dict:
        """
        Compute composite risk scores and classify final risk_state.

        Scores: market_risk, fragility, concentration, liquidity,
        governance, regime, implementation, overall.
        """
        checks = all_checks.get("checks", [])

        def _check_level_score(check_name: str) -> float:
            for c in checks:
                if c.get("check") == check_name:
                    lvl = c.get("level", RISK_GREEN)
                    return {RISK_GREEN: 0.0, RISK_YELLOW: 0.4, RISK_RED: 0.8, RISK_BLACK: 1.0}.get(lvl, 0.0)
            return 0.0

        market_risk = max(
            _check_level_score("var_breach"),
            _check_level_score("drawdown"),
            _check_level_score("vix_regime"),
        )
        concentration_risk = _check_level_score("concentration")
        liquidity_risk = _check_level_score("liquidity_risk")
        fragility_score = fragility.get("fragility_score", 0.0)

        # Governance risk from methodology
        governance_risk = 0.0  # default: no governance issues

        # Regime risk
        regime_risk = max(
            _check_level_score("vix_regime"),
            _check_level_score("correlation_spike"),
        )

        # Implementation risk from stress scenarios
        implementation_risk = 0.0
        scenario_data = scenarios.get("scenarios", {})
        if scenario_data:
            worst_multiplier = max(
                s.get("multiplier", 1.0) for s in scenario_data.values()
                if isinstance(s, dict) and "multiplier" in s
            ) if any("multiplier" in s for s in scenario_data.values() if isinstance(s, dict)) else 1.0
            implementation_risk = min(1.0, (worst_multiplier - 1.0) * 0.3)

        # Weighted overall
        weights_map = {
            "market_risk": 0.25,
            "fragility": 0.20,
            "concentration": 0.15,
            "liquidity": 0.10,
            "governance": 0.10,
            "regime": 0.10,
            "implementation": 0.10,
        }
        overall = (
            market_risk * weights_map["market_risk"]
            + fragility_score * weights_map["fragility"]
            + concentration_risk * weights_map["concentration"]
            + liquidity_risk * weights_map["liquidity"]
            + governance_risk * weights_map["governance"]
            + regime_risk * weights_map["regime"]
            + implementation_risk * weights_map["implementation"]
        )
        overall = min(1.0, max(0.0, overall))

        # Classify risk state from overall score
        if overall >= 0.85:
            risk_state = "EMERGENCY"
        elif overall >= 0.65:
            risk_state = "HALT_REQUIRED"
        elif overall >= 0.50:
            risk_state = "ELEVATED_RISK"
        elif overall >= 0.35:
            risk_state = "FRAGILE"
        elif overall >= 0.15:
            risk_state = "CAUTION"
        else:
            risk_state = "SAFE"

        result = {
            "market_risk_score": round(market_risk, 4),
            "fragility_score": round(fragility_score, 4),
            "concentration_risk_score": round(concentration_risk, 4),
            "liquidity_risk_score": round(liquidity_risk, 4),
            "governance_risk_score": round(governance_risk, 4),
            "regime_risk_score": round(regime_risk, 4),
            "implementation_risk_score": round(implementation_risk, 4),
            "overall_risk_score": round(overall, 4),
            "risk_state": risk_state,
        }
        log.info("Risk scores: overall=%.3f -> state=%s", overall, risk_state)
        return result

    # -----------------------------------------------------------------
    # 11. Risk History / Escalation
    # -----------------------------------------------------------------

    def track_escalation(self, risk_state: str) -> Dict:
        """
        Track risk state history and escalation patterns.

        Appends to risk_history.json (capped at 365 entries).
        Returns consecutive_elevated_days, escalation_events, recovery_events.
        """
        history: List[Dict] = []
        try:
            if _RISK_HISTORY_PATH.exists():
                history = json.loads(_RISK_HISTORY_PATH.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []

        entry = {
            "date": date.today().isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_state": risk_state,
            "legacy_level": RISK_STATE_TO_LEGACY.get(risk_state, "YELLOW"),
        }
        history.append(entry)

        # Cap at 365
        if len(history) > 365:
            history = history[-365:]

        # Save
        try:
            _RISK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _RISK_HISTORY_PATH.write_text(
                json.dumps(history, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to save risk history: %s", exc)

        # Compute escalation metrics
        elevated_states = {"ELEVATED_RISK", "HALT_REQUIRED", "EMERGENCY"}
        consecutive_elevated = 0
        for h in reversed(history):
            if h.get("risk_state") in elevated_states:
                consecutive_elevated += 1
            else:
                break

        escalation_events = 0
        recovery_events = 0
        for i in range(1, len(history)):
            prev_state = history[i - 1].get("risk_state", "SAFE")
            curr_state = history[i].get("risk_state", "SAFE")
            prev_elevated = prev_state in elevated_states
            curr_elevated = curr_state in elevated_states
            if not prev_elevated and curr_elevated:
                escalation_events += 1
            elif prev_elevated and not curr_elevated:
                recovery_events += 1

        result = {
            "consecutive_elevated_days": consecutive_elevated,
            "escalation_events": escalation_events,
            "recovery_events": recovery_events,
            "history_length": len(history),
        }
        log.info("Escalation tracking: consecutive_elevated=%d, escalations=%d, recoveries=%d",
                 consecutive_elevated, escalation_events, recovery_events)
        return result

    def get_recent_status_history(self, n: int = 30) -> List[Dict]:
        """Return last n entries from risk history."""
        try:
            if _RISK_HISTORY_PATH.exists():
                history = json.loads(_RISK_HISTORY_PATH.read_text(encoding="utf-8"))
                if isinstance(history, list):
                    return history[-n:]
        except Exception:
            pass
        return []

    def time_in_elevated_risk(self) -> Dict:
        """Compute time spent in elevated risk states."""
        history = self.get_recent_status_history(365)
        elevated_states = {"ELEVATED_RISK", "HALT_REQUIRED", "EMERGENCY"}
        total = len(history)
        elevated = sum(1 for h in history if h.get("risk_state") in elevated_states)
        return {
            "total_days": total,
            "elevated_days": elevated,
            "elevated_pct": round(elevated / total, 4) if total > 0 else 0.0,
        }

    def repeated_breach_summary(self) -> Dict:
        """Summarize repeated breaches from history."""
        history = self.get_recent_status_history(90)
        state_counts: Dict[str, int] = {}
        for h in history:
            st = h.get("risk_state", "SAFE")
            state_counts[st] = state_counts.get(st, 0) + 1
        return {"state_distribution_90d": state_counts, "total_entries": len(history)}

    # -----------------------------------------------------------------
    # 12. Enhanced machine_summary builder
    # -----------------------------------------------------------------

    def _build_institutional_machine_summary(
        self,
        legacy_level: str,
        risk_state: str,
        risk_scores: Dict,
        fragility: Dict,
        veto: Dict,
        risk_inputs: Dict,
        sleeve_risk: List[Dict],
        data_sufficiency: Dict,
    ) -> Dict:
        """Build the enhanced machine_summary for downstream agents."""
        regime = risk_inputs.get("regime_forecast", {})
        exp_check = self.check_exposure_limits()

        # Top risk drivers
        top_drivers: List[str] = []
        if fragility.get("fragility_drivers"):
            top_drivers.extend(fragility["fragility_drivers"][:3])
        if veto.get("veto_reasons"):
            top_drivers.extend(veto["veto_reasons"][:2])
        if not top_drivers:
            top_drivers.append("none")

        # Top flagged sleeves
        flagged_sleeves: List[Dict] = []
        for s in sleeve_risk[:3]:
            if s.get("action") in ("CUT", "DISABLE", "HAIRCUT"):
                flagged_sleeves.append({
                    s["ticker"]: f"{s.get('decay_health', 'UNKNOWN')}, {s['capital_weight']:.0%} weight"
                })

        # Transition state
        transition_prob = regime.get("transition_probability", 0.0)
        if transition_prob > 0.5:
            transition_state = "ACTIVE_TRANSITION"
        elif transition_prob > 0.3:
            transition_state = "EARLY_WARNING"
        else:
            transition_state = "STABLE"

        # Implementation caution level
        overall = risk_scores.get("overall_risk_score", 0.0)
        if overall >= 0.6:
            caution = "CRITICAL"
        elif overall >= 0.4:
            caution = "HIGH"
        elif overall >= 0.2:
            caution = "MODERATE"
        else:
            caution = "LOW"

        return {
            "legacy_level": legacy_level,
            "risk_state": risk_state,
            "overall_risk_score": risk_scores.get("overall_risk_score", 0.0),
            "fragility_score": fragility.get("fragility_score", 0.0),
            "can_allocate_new_risk": veto.get("can_allocate_new_risk", True),
            "can_execute_new_trades": veto.get("can_execute_new_trades", True),
            "must_reduce_existing_risk": veto.get("must_reduce_existing_risk", False),
            "emergency_unwind_required": veto.get("emergency_unwind_required", False),
            "active_regime": regime.get("predicted_regime", "UNKNOWN"),
            "transition_state": transition_state,
            "gross_exposure": exp_check.get("gross_exposure", 0.0),
            "net_exposure": exp_check.get("net_exposure", 0.0),
            "top_risk_drivers": top_drivers,
            "top_flagged_sleeves": flagged_sleeves,
            "missing_critical_inputs": data_sufficiency.get("missing_inputs", []),
            "implementation_caution_level": caution,
        }

    # -----------------------------------------------------------------
    # Aggregate checks (original)
    # -----------------------------------------------------------------

    def run_all_checks(self) -> dict:
        """
        Run all risk checks and return a consolidated status dict.

        Returns
        -------
        dict
            Keys: level (GREEN/YELLOW/RED/BLACK), breaches (list),
            recommendations (list), checks (list of individual results),
            timestamp.
        """
        checks = [
            self.check_var_breach(),
            self.check_exposure_limits(),
            self.check_concentration(),
            self.check_correlation_spike(),
            self.check_drawdown(),
            self.check_vix_regime(),
            self.stress_test_correlation(),
            self.compute_tail_risk(),
            self.monitor_portfolio_greeks(),
            self.check_liquidity_risk(),
        ]

        # Determine overall level
        overall_level = RISK_GREEN
        breaches: List[str] = []
        recommendations: List[str] = []

        for check in checks:
            level = check.get("level", RISK_GREEN)
            overall_level = _worst_level(overall_level, level)
            if level != RISK_GREEN:
                breaches.append(check["detail"])

        # Generate recommendations based on level
        if overall_level == RISK_RED:
            recommendations.append("HALT all new trades immediately")
            recommendations.append("Review and close positions breaching limits")
        elif overall_level == RISK_YELLOW:
            recommendations.append("Reduce position sizes by 50%")
            recommendations.append("Tighten stop-losses")
        elif overall_level == RISK_BLACK:
            recommendations.append("EMERGENCY: Close ALL positions immediately")

        # Promote RED to BLACK if multiple severe breaches
        red_count = sum(
            1 for c in checks if c.get("level") == RISK_RED
        )
        if red_count >= 3:
            overall_level = RISK_BLACK
            recommendations.insert(0, "EMERGENCY: Multiple severe breaches detected")

        status = {
            "level": overall_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "date": date.today().isoformat(),
            "breaches": breaches,
            "recommendations": recommendations,
            "checks": checks,
            "n_positions": len(self._get_positions()),
            "capital": self._get_capital(),
        }

        log.info(
            "Risk status: %s | %d breaches | %d positions",
            overall_level, len(breaches), status["n_positions"],
        )

        return status

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------

    def execute_actions(self, status: dict) -> None:
        """
        Execute actions based on risk level.

        - RED: publish halt to agent_bus
        - BLACK: close all positions in paper_trader
        """
        level = status.get("level", RISK_GREEN)

        if level in (RISK_RED, RISK_BLACK):
            # Publish halt signal to bus
            if _IMPORTS_OK.get("agent_bus"):
                try:
                    bus = get_bus()
                    bus.publish("risk_guardian", {
                        "action": "HALT" if level == RISK_RED else "EMERGENCY_CLOSE",
                        "level": level,
                        "breaches": status.get("breaches", []),
                        "recommendations": status.get("recommendations", []),
                        "date": status.get("date", date.today().isoformat()),
                    })
                    log.info("Published %s signal to agent_bus", level)
                except Exception as exc:
                    log.error("Failed to publish to agent_bus: %s", exc)

        if level == RISK_BLACK:
            # Emergency: attempt to close all positions via PaperTrader
            if _IMPORTS_OK.get("paper_trader"):
                try:
                    trader = PaperTrader(settings=self.settings)
                    trader.load()
                    n_positions = len(trader.portfolio.positions)
                    if n_positions > 0:
                        # Force close all positions
                        trader.portfolio.positions.clear()
                        trader.save()
                        log.warning(
                            "BLACK ALERT: Closed all %d positions in paper portfolio",
                            n_positions,
                        )
                except Exception as exc:
                    log.error("Failed to close positions in BLACK mode: %s", exc)

        if level == RISK_GREEN:
            log.info("All risk checks passed -- GREEN status")
        elif level == RISK_YELLOW:
            log.warning("Risk warnings detected -- YELLOW status")

    # -----------------------------------------------------------------
    # Save status
    # -----------------------------------------------------------------

    def _save_status(self, status: dict) -> None:
        """Save risk status to JSON file."""
        try:
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATUS_PATH.write_text(
                json.dumps(status, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("Saved risk status to %s", STATUS_PATH)
        except Exception as exc:
            log.error("Failed to save risk status: %s", exc)

    # -----------------------------------------------------------------
    # Full run cycle
    # -----------------------------------------------------------------

    def run(self) -> dict:
        """
        Full cycle: checks -> actions -> save status -> registry heartbeat.

        Returns the status dict.
        """
        # Registry heartbeat: RUNNING
        if _IMPORTS_OK.get("registry"):
            try:
                registry = get_registry()
                registry.register("risk_guardian", role="independent risk monitor with veto power")
                registry.heartbeat("risk_guardian", AgentStatus.RUNNING)
            except Exception as exc:
                log.warning("Registry heartbeat failed: %s", exc)

        # Run all checks
        status = self.run_all_checks()

        # Execute actions based on level
        self.execute_actions(status)

        # Format and log alert
        alert_msg = self.format_alert(status)
        status["alert_message"] = alert_msg
        if status["level"] != RISK_GREEN:
            log.warning("Alert:\n%s", alert_msg)

        # =============================================================
        # INSTITUTIONAL CRO DESK — additional pipeline steps
        # =============================================================
        try:
            # Step CRO-1: Assemble risk inputs
            risk_inputs = self.assemble_risk_inputs()
            status["risk_inputs_available"] = risk_inputs.get("available_inputs", [])

            # Step CRO-2: Check data sufficiency
            data_sufficiency = self.check_data_sufficiency(risk_inputs)
            status["data_sufficiency"] = data_sufficiency

            # Step CRO-3: Detect fragility
            fragility = self.detect_fragility(risk_inputs)
            status["fragility"] = fragility

            # Step CRO-4: Assess proposed portfolio (if available)
            proposed_assessment: Dict[str, Any] = {}
            pw = risk_inputs.get("portfolio_weights", {})
            proposed_weights = pw.get("weights", {}) if isinstance(pw, dict) else {}
            if proposed_weights:
                proposed_assessment = self.assess_proposed_portfolio(proposed_weights, risk_inputs)
            status["proposed_assessment"] = proposed_assessment

            # Step CRO-5: Analyze sleeve risk
            sleeve_risk = self.analyze_sleeve_risk(risk_inputs)
            status["sleeve_risk"] = sleeve_risk

            # Step CRO-6: Run stress scenarios
            scenarios = self.run_stress_scenarios()
            status["stress_scenarios"] = scenarios

            # Step CRO-7: Compute risk scores + classify institutional state
            risk_scores = self.compute_risk_scores(status, fragility, scenarios)
            status["risk_scores"] = risk_scores
            institutional_risk_state = risk_scores.get("risk_state", "CAUTION")

            # Override with INSUFFICIENT_EVIDENCE if data not sufficient
            if not data_sufficiency.get("sufficient", True):
                institutional_risk_state = "INSUFFICIENT_EVIDENCE"
            status["institutional_risk_state"] = institutional_risk_state

            # Step CRO-8: Compute veto status
            veto = self.compute_veto_status(status, fragility, proposed_assessment)
            # Also apply INSUFFICIENT_EVIDENCE veto
            if institutional_risk_state == "INSUFFICIENT_EVIDENCE":
                veto["can_allocate_new_risk"] = False
                if "INSUFFICIENT_EVIDENCE: conservative stance" not in veto.get("veto_reasons", []):
                    veto["veto_reasons"].append("INSUFFICIENT_EVIDENCE: conservative stance")
            status["veto"] = veto

            # Step CRO-9: Generate risk actions
            risk_actions = self.generate_risk_actions(
                institutional_risk_state, status, fragility, veto,
            )
            status["risk_actions"] = risk_actions

            # Step CRO-10: Track escalation
            escalation = self.track_escalation(institutional_risk_state)
            status["escalation"] = escalation

            # Step CRO-11: Build machine_summary
            machine_summary = self._build_institutional_machine_summary(
                legacy_level=status["level"],
                risk_state=institutional_risk_state,
                risk_scores=risk_scores,
                fragility=fragility,
                veto=veto,
                risk_inputs=risk_inputs,
                sleeve_risk=sleeve_risk,
                data_sufficiency=data_sufficiency,
            )
            status["machine_summary"] = machine_summary
            log.info(
                "CRO desk complete: state=%s, overall=%.3f, allocate=%s",
                institutional_risk_state,
                risk_scores.get("overall_risk_score", 0.0),
                veto.get("can_allocate_new_risk", True),
            )

        except Exception as exc:
            log.error("Institutional CRO pipeline failed (legacy checks preserved): %s", exc)
            status["cro_error"] = str(exc)

        # Save status to JSON
        self._save_status(status)

        # Registry heartbeat: COMPLETED or FAILED
        if _IMPORTS_OK.get("registry"):
            try:
                registry = get_registry()
                if status["level"] in (RISK_RED, RISK_BLACK):
                    registry.heartbeat(
                        "risk_guardian", AgentStatus.COMPLETED,
                        error=f"Risk level: {status['level']}",
                    )
                else:
                    registry.heartbeat("risk_guardian", AgentStatus.COMPLETED)
            except Exception as exc:
                log.warning("Registry heartbeat failed: %s", exc)

        return status


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Risk Guardian Agent")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between runs in loop mode (default: 300)",
    )
    args = parser.parse_args()

    guardian = RiskGuardian()

    if args.once:
        status = guardian.run()
        log.info("Risk Guardian completed: level=%s", status["level"])
        return

    # Loop mode
    log.info("Risk Guardian starting in loop mode (interval=%ds)", args.interval)
    while True:
        try:
            status = guardian.run()
            log.info("Cycle complete: level=%s", status["level"])
        except Exception as exc:
            log.error("Risk Guardian cycle failed: %s", exc)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
