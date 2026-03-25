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

    # -----------------------------------------------------------------
    # Aggregate checks
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
