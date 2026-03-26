"""
Data Scout Agent -- Scans external data sources for signals and anomalies.

Expands the information edge by fetching macro data from FRED,
detecting volume anomalies, correlation breaks, and momentum divergences
in sector ETFs.

Output: agents/data_scout/scout_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
            _LOG_DIR / "agent_data_scout.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("agent_data_scout")

# -- Report output path -------------------------------------------------------
REPORT_PATH = Path(__file__).resolve().parent / "scout_report.json"

# -- Safe imports with fallbacks -----------------------------------------------
_IMPORTS_OK: Dict[str, bool] = {}

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

try:
    import requests
    _IMPORTS_OK["requests"] = True
except ImportError as e:
    log.warning("Could not import requests: %s", e)
    _IMPORTS_OK["requests"] = False


# -- FRED series config -------------------------------------------------------
FRED_SERIES = {
    "T10Y2Y": "yield_curve",
    "BAMLH0A0HYM2": "hy_spread",
    "DTWEXBGS": "dollar_index",
    "UNRATE": "unemployment",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# -- History path for persistence engine --------------------------------------
HISTORY_PATH = Path(__file__).resolve().parent / "scout_history.json"


# =============================================================================
# ResearchLead
# =============================================================================
@dataclass
class ResearchLead:
    """Classified research finding from the Research Intelligence Desk."""

    lead_id: str
    category: str  # MACRO_STRESS / CREDIT_FRAGILITY / CORRELATION_BREAK /
    # DISPERSION_EVENT / MOMENTUM_DIVERGENCE / RELATIVE_VALUE_SETUP /
    # REGIME_TRANSITION_PRECURSOR / DEFENSIVE_ROTATION / RISK_OFF_ACCELERATION
    subcategory: str
    tickers: List[str]
    summary: str
    evidence: List[str]
    severity_score: float  # 0-1
    confidence_score: float  # 0-1
    persistence_score: float  # 0-1, based on history
    actionability_score: float  # 0-1
    regime_relevance: str  # HIGH / MEDIUM / LOW
    downstream_targets: List[str] = field(
        default_factory=list
    )  # e.g. ["methodology", "regime_forecaster", "risk_guardian"]
    recommended_followup: str = ""
    aging_days: int = 0  # 0 = new, >0 = recurring
    priority_score: float = 0.0  # computed composite


# =============================================================================
# DataScout
# =============================================================================
class DataScout:
    """Scans external data sources for signals and anomalies."""

    def __init__(self, settings: Optional[Any] = None) -> None:
        if settings is None and _IMPORTS_OK.get("settings"):
            try:
                self.settings = get_settings()
            except Exception:
                self.settings = None
        else:
            self.settings = settings

        self._session: Optional[Any] = None
        if _IMPORTS_OK.get("requests"):
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "SRV-Quant-DataScout/1.0",
            })

        self._prices: Optional[pd.DataFrame] = None
        self._load_prices()

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # FRED macro data
    # -----------------------------------------------------------------
    def fetch_fred_data(self) -> dict:
        """
        Fetch key macro indicators from FRED API (free CSV endpoint).

        Returns dict of {series_name: {value, z_score, trend}}.
        """
        result: Dict[str, Dict[str, Any]] = {}

        if not _IMPORTS_OK.get("requests") or self._session is None:
            log.warning("requests library not available, skipping FRED fetch")
            return result

        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%d")

        for series_id, series_name in FRED_SERIES.items():
            try:
                url = FRED_CSV_URL
                params = {
                    "id": series_id,
                    "cosd": start_date,
                    "coed": end_date,
                }
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()

                df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
                # FRED uses "." for missing values
                col = [c for c in df.columns if c != "DATE"][0]
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=[col])

                if df.empty:
                    log.warning("FRED series %s returned no data", series_id)
                    continue

                latest_value = float(df[col].iloc[-1])

                # z-score vs trailing 252 business days (~1yr)
                history = df[col].values
                if len(history) >= 20:
                    mean_val = float(np.mean(history))
                    std_val = float(np.std(history))
                    z_score = (latest_value - mean_val) / std_val if std_val > 0 else 0.0
                else:
                    z_score = 0.0

                # Trend: compare last value to 3-month-ago value
                lookback = min(63, len(history) - 1)
                if lookback > 0:
                    older_value = float(history[-(lookback + 1)])
                    if older_value != 0:
                        trend = (latest_value - older_value) / abs(older_value)
                    else:
                        trend = 0.0
                else:
                    trend = 0.0

                result[series_name] = {
                    "series_id": series_id,
                    "value": round(latest_value, 4),
                    "z_score": round(z_score, 3),
                    "trend": round(trend, 4),
                }
                log.info("FRED %s (%s): value=%.4f z=%.3f trend=%.4f",
                         series_id, series_name, latest_value, z_score, trend)

            except Exception as exc:
                log.warning("FRED fetch failed for %s: %s", series_id, exc)
                continue

        return result

    # -----------------------------------------------------------------
    # Volume anomalies
    # -----------------------------------------------------------------
    def detect_volume_anomalies(self) -> list:
        """
        Check for unusual volume in sector ETFs.

        If volume data not available, estimate from price moves:
        flag days where |return| > 3 sigma.
        """
        anomalies: List[Dict[str, Any]] = []

        if self._prices is None or self._prices.empty:
            log.info("No price data available for volume anomaly detection")
            return anomalies

        for ticker in self._prices.columns:
            try:
                series = self._prices[ticker].dropna()
                if len(series) < 60:
                    continue

                returns = series.pct_change().dropna()
                if len(returns) < 60:
                    continue

                rolling_mean = returns.rolling(60).mean()
                rolling_std = returns.rolling(60).std()

                # Check last 5 trading days
                for i in range(-1, -min(6, len(returns)), -1):
                    ret = returns.iloc[i]
                    mu = rolling_mean.iloc[i]
                    sigma = rolling_std.iloc[i]

                    if sigma is None or np.isnan(sigma) or sigma == 0:
                        continue

                    z = (ret - mu) / sigma

                    if abs(z) > 3.0:
                        dt = returns.index[i]
                        date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                        anomaly_type = "extreme_positive" if z > 0 else "extreme_negative"
                        anomalies.append({
                            "ticker": ticker,
                            "date": date_str,
                            "anomaly_type": anomaly_type,
                            "magnitude": round(float(z), 2),
                            "return_pct": round(float(ret) * 100, 2),
                        })

            except Exception as exc:
                log.warning("Volume anomaly check failed for %s: %s", ticker, exc)

        anomalies.sort(key=lambda x: abs(x.get("magnitude", 0)), reverse=True)
        return anomalies

    # -----------------------------------------------------------------
    # Correlation breaks
    # -----------------------------------------------------------------
    def detect_correlation_breaks(self) -> list:
        """
        Find pairs where correlation changed dramatically.

        Compute 20d vs 120d correlation for all sector pairs.
        Flag pairs where |corr_20d - corr_120d| > 0.3.
        """
        breaks: List[Dict[str, Any]] = []

        if self._prices is None or self._prices.empty:
            log.info("No price data for correlation break detection")
            return breaks

        returns = self._prices.pct_change().dropna()
        if len(returns) < 120:
            log.info("Insufficient history for correlation breaks (need 120 days)")
            return breaks

        tickers = [c for c in returns.columns if not returns[c].isna().all()]
        if len(tickers) < 2:
            return breaks

        returns_subset = returns[tickers]

        try:
            corr_20d = returns_subset.tail(20).corr()
            corr_120d = returns_subset.tail(120).corr()
        except Exception as exc:
            log.warning("Correlation computation failed: %s", exc)
            return breaks

        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                try:
                    c20 = corr_20d.loc[t1, t2]
                    c120 = corr_120d.loc[t1, t2]
                    if np.isnan(c20) or np.isnan(c120):
                        continue
                    diff = c20 - c120
                    if abs(diff) > 0.3:
                        breaks.append({
                            "pair": f"{t1}/{t2}",
                            "corr_20d": round(float(c20), 3),
                            "corr_120d": round(float(c120), 3),
                            "change": round(float(diff), 3),
                            "direction": "decorrelating" if diff < 0 else "converging",
                        })
                except Exception:
                    continue

        breaks.sort(key=lambda x: abs(x.get("change", 0)), reverse=True)
        return breaks

    # -----------------------------------------------------------------
    # Momentum divergence
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI for a price series."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_momentum_divergence(self) -> list:
        """
        Find sectors where price and momentum diverge.

        - Price making new 20d high but RSI declining (bearish divergence)
        - Price making new 20d low but RSI rising (bullish divergence)
        """
        divergences: List[Dict[str, Any]] = []

        if self._prices is None or self._prices.empty:
            log.info("No price data for momentum divergence detection")
            return divergences

        for ticker in self._prices.columns:
            try:
                series = self._prices[ticker].dropna()
                if len(series) < 40:
                    continue

                rsi = self._compute_rsi(series)
                if rsi.dropna().empty:
                    continue

                # Last 20 days
                recent_prices = series.tail(20)
                recent_rsi = rsi.tail(20).dropna()
                if len(recent_rsi) < 10:
                    continue

                current_price = float(series.iloc[-1])
                high_20d = float(recent_prices.max())
                low_20d = float(recent_prices.min())

                # RSI trend: compare last 5d avg to previous 5d avg
                rsi_last5 = float(recent_rsi.tail(5).mean())
                rsi_prev5 = float(recent_rsi.iloc[-10:-5].mean()) if len(recent_rsi) >= 10 else rsi_last5

                # Bearish divergence: new 20d high but RSI declining
                if current_price >= high_20d * 0.99 and rsi_last5 < rsi_prev5 - 3:
                    divergences.append({
                        "ticker": ticker,
                        "type": "bearish_divergence",
                        "price": round(current_price, 2),
                        "high_20d": round(high_20d, 2),
                        "rsi_recent": round(rsi_last5, 1),
                        "rsi_prior": round(rsi_prev5, 1),
                    })

                # Bullish divergence: new 20d low but RSI rising
                if current_price <= low_20d * 1.01 and rsi_last5 > rsi_prev5 + 3:
                    divergences.append({
                        "ticker": ticker,
                        "type": "bullish_divergence",
                        "price": round(current_price, 2),
                        "low_20d": round(low_20d, 2),
                        "rsi_recent": round(rsi_last5, 1),
                        "rsi_prior": round(rsi_prev5, 1),
                    })

            except Exception as exc:
                log.warning("Momentum divergence check failed for %s: %s", ticker, exc)

        return divergences

    # -----------------------------------------------------------------
    # Macro regime signals
    # -----------------------------------------------------------------
    def compute_macro_regime_signals(self) -> dict:
        """
        Aggregate macro signals into a single macro_score.

        Yield curve: inverted = recession risk
        HY spread: widening = credit stress
        Dollar: strengthening = risk-off
        Combine into macro_score (-1 to +1, negative = risk-off)
        """
        fred_data = self.fetch_fred_data()
        signals: Dict[str, Any] = {}
        component_scores: List[float] = []

        # Yield curve signal
        yc = fred_data.get("yield_curve", {})
        if yc:
            value = yc.get("value", 0)
            # Negative yield curve = inverted = risk-off
            if value < 0:
                yc_score = max(-1.0, value / 1.0)  # scale: -1% -> -1.0
            else:
                yc_score = min(1.0, value / 2.0)   # scale: +2% -> +1.0
            signals["yield_curve"] = {
                "value": value,
                "z_score": yc.get("z_score", 0),
                "signal": "inverted" if value < 0 else "normal",
                "score": round(yc_score, 3),
            }
            component_scores.append(yc_score)

        # HY spread signal
        hy = fred_data.get("hy_spread", {})
        if hy:
            z = hy.get("z_score", 0)
            # High z-score (widening) = credit stress = risk-off
            hy_score = max(-1.0, min(1.0, -z / 2.0))
            signals["hy_spread"] = {
                "value": hy.get("value", 0),
                "z_score": z,
                "signal": "stress" if z > 1 else "normal",
                "score": round(hy_score, 3),
            }
            component_scores.append(hy_score)

        # Dollar signal
        dollar = fred_data.get("dollar_index", {})
        if dollar:
            trend = dollar.get("trend", 0)
            # Strengthening dollar = risk-off
            d_score = max(-1.0, min(1.0, -trend * 10))
            signals["dollar"] = {
                "value": dollar.get("value", 0),
                "z_score": dollar.get("z_score", 0),
                "trend": trend,
                "signal": "risk_off" if trend > 0.02 else "neutral",
                "score": round(d_score, 3),
            }
            component_scores.append(d_score)

        # Aggregate macro score
        if component_scores:
            macro_score = float(np.mean(component_scores))
        else:
            macro_score = 0.0

        signals["macro_score"] = round(max(-1.0, min(1.0, macro_score)), 3)

        return signals

    # =================================================================
    # RESEARCH INTELLIGENCE DESK — New Methods
    # =================================================================

    # -----------------------------------------------------------------
    # 2. Event Classification Engine
    # -----------------------------------------------------------------
    def classify_events(
        self,
        anomalies: list,
        corr_breaks: list,
        divergences: list,
        macro: dict,
    ) -> List[ResearchLead]:
        """
        Map raw detections into professional research event classes.

        Each raw finding becomes a classified ResearchLead with initial scores.
        Cross-domain patterns are combined into higher-level categories.
        """
        leads: List[ResearchLead] = []
        macro_score = macro.get("macro_score", 0.0)
        hy_info = macro.get("hy_spread", {})
        yc_info = macro.get("yield_curve", {})
        is_macro_stress = macro_score < -0.3
        is_credit_stress = hy_info.get("signal") == "stress"
        is_yc_inverted = yc_info.get("signal") == "inverted"

        # --- Identify defensive / cyclical tickers for rotation detection ---
        defensive_tickers = {"XLU", "XLP", "XLV", "XLRE", "TLT", "GLD"}
        cyclical_tickers = {"XLK", "XLF", "XLI", "XLE", "XLY", "XLC"}

        anomaly_tickers = {a["ticker"] for a in anomalies}
        neg_anomaly_tickers = {
            a["ticker"] for a in anomalies if a.get("anomaly_type") == "extreme_negative"
        }
        pos_anomaly_tickers = {
            a["ticker"] for a in anomalies if a.get("anomaly_type") == "extreme_positive"
        }
        divergence_tickers = {d["ticker"] for d in divergences}
        corr_break_pairs = {b["pair"] for b in corr_breaks}

        # ---------- RISK_OFF_ACCELERATION ----------
        # Multiple extreme negative moves clustered together
        if len(neg_anomaly_tickers) >= 3:
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("RISK_OFF_ACCELERATION", sorted(neg_anomaly_tickers)),
                category="RISK_OFF_ACCELERATION",
                subcategory="clustered_extreme_moves",
                tickers=sorted(neg_anomaly_tickers),
                summary=f"Risk-off acceleration: {len(neg_anomaly_tickers)} sectors with extreme negative moves",
                evidence=[
                    f"{a['ticker']}: {a['return_pct']:.1f}% (z={a['magnitude']:.1f})"
                    for a in anomalies if a["anomaly_type"] == "extreme_negative"
                ][:5],
                severity_score=min(1.0, 0.5 + 0.1 * len(neg_anomaly_tickers)),
                confidence_score=min(1.0, 0.4 + 0.15 * len(neg_anomaly_tickers)),
                persistence_score=0.3,
                actionability_score=0.8,
                regime_relevance="HIGH",
                downstream_targets=["risk_guardian", "regime_forecaster", "execution"],
                recommended_followup="Immediate portfolio risk check; consider hedges",
            ))

        # ---------- MACRO_STRESS ----------
        # Volume anomaly (negative) + macro stress environment
        if is_macro_stress and neg_anomaly_tickers:
            for anom in anomalies:
                if anom.get("anomaly_type") != "extreme_negative":
                    continue
                leads.append(ResearchLead(
                    lead_id=self._make_lead_id("MACRO_STRESS", [anom["ticker"]]),
                    category="MACRO_STRESS",
                    subcategory="anomaly_in_stress_regime",
                    tickers=[anom["ticker"]],
                    summary=f"Macro-stress amplified move in {anom['ticker']}: {anom['return_pct']:.1f}%",
                    evidence=[
                        f"Macro score: {macro_score:.3f}",
                        f"Return z-score: {anom['magnitude']:.2f}",
                    ],
                    severity_score=min(1.0, abs(anom["magnitude"]) / 5.0),
                    confidence_score=0.6,
                    persistence_score=0.3,
                    actionability_score=0.6,
                    regime_relevance="HIGH",
                    downstream_targets=["risk_guardian", "regime_forecaster"],
                    recommended_followup="Monitor for contagion to other sectors",
                ))

        # ---------- CREDIT_FRAGILITY ----------
        # HY spread widening + defensive rotation signals
        defensive_positive = pos_anomaly_tickers & defensive_tickers
        cyclical_negative = neg_anomaly_tickers & cyclical_tickers
        if is_credit_stress or (defensive_positive and cyclical_negative):
            evidence_list = []
            if is_credit_stress:
                evidence_list.append(f"HY spread z-score: {hy_info.get('z_score', 0):.2f}")
            if defensive_positive:
                evidence_list.append(f"Defensive strength: {', '.join(sorted(defensive_positive))}")
            if cyclical_negative:
                evidence_list.append(f"Cyclical weakness: {', '.join(sorted(cyclical_negative))}")
            all_tickers = sorted(
                (defensive_positive | cyclical_negative)
                or ({hy_info.get("series_id", "HY")} if is_credit_stress else set())
            )
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("CREDIT_FRAGILITY", all_tickers),
                category="CREDIT_FRAGILITY",
                subcategory="hy_widening_with_rotation" if defensive_positive else "hy_spread_stress",
                tickers=all_tickers if all_tickers else ["HY"],
                summary="Credit fragility detected: HY stress with defensive rotation",
                evidence=evidence_list,
                severity_score=min(1.0, 0.5 + abs(hy_info.get("z_score", 0)) * 0.15),
                confidence_score=0.55 if is_credit_stress else 0.35,
                persistence_score=0.3,
                actionability_score=0.7,
                regime_relevance="HIGH",
                downstream_targets=["risk_guardian", "portfolio_construction", "regime_forecaster"],
                recommended_followup="Check credit exposure; consider HY hedges",
            ))

        # ---------- DEFENSIVE_ROTATION ----------
        if defensive_positive and not cyclical_negative:
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("DEFENSIVE_ROTATION", sorted(defensive_positive)),
                category="DEFENSIVE_ROTATION",
                subcategory="defensive_strength",
                tickers=sorted(defensive_positive),
                summary=f"Defensive rotation: strength in {', '.join(sorted(defensive_positive))}",
                evidence=[
                    f"{a['ticker']}: +{a['return_pct']:.1f}%"
                    for a in anomalies
                    if a["ticker"] in defensive_positive
                ],
                severity_score=0.4,
                confidence_score=0.45,
                persistence_score=0.3,
                actionability_score=0.5,
                regime_relevance="MEDIUM",
                downstream_targets=["regime_forecaster", "portfolio_construction"],
                recommended_followup="Monitor if rotation broadens to full risk-off",
            ))

        # ---------- REGIME_TRANSITION_PRECURSOR ----------
        if is_yc_inverted and is_macro_stress:
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("REGIME_TRANSITION_PRECURSOR", ["MACRO"]),
                category="REGIME_TRANSITION_PRECURSOR",
                subcategory="yield_curve_plus_macro_stress",
                tickers=["MACRO"],
                summary=f"Regime transition precursor: inverted YC + macro stress ({macro_score:.3f})",
                evidence=[
                    f"Yield curve: {yc_info.get('value', 0):.3f}",
                    f"Macro score: {macro_score:.3f}",
                ],
                severity_score=0.7,
                confidence_score=0.55,
                persistence_score=0.3,
                actionability_score=0.6,
                regime_relevance="HIGH",
                downstream_targets=["regime_forecaster", "methodology", "optimizer"],
                recommended_followup="Prepare regime-shift playbook; stress test current positions",
            ))

        # ---------- CORRELATION_BREAK ----------
        for brk in corr_breaks:
            pair_tickers = brk["pair"].split("/")
            # Check if also has momentum divergence in either leg
            has_div = any(t in divergence_tickers for t in pair_tickers)
            cat = "RELATIVE_VALUE_SETUP" if has_div else "CORRELATION_BREAK"
            subcat = "corr_break_with_divergence" if has_div else "structural_decorrelation"
            leads.append(ResearchLead(
                lead_id=self._make_lead_id(cat, pair_tickers),
                category=cat,
                subcategory=subcat,
                tickers=pair_tickers,
                summary=f"{cat.replace('_', ' ').title()}: {brk['pair']} "
                        f"({brk['direction']}, delta={brk['change']:.3f})",
                evidence=[
                    f"Corr 20d: {brk['corr_20d']:.3f}",
                    f"Corr 120d: {brk['corr_120d']:.3f}",
                    f"Direction: {brk['direction']}",
                ],
                severity_score=min(1.0, abs(brk["change"]) / 0.6),
                confidence_score=0.5 + (0.15 if has_div else 0.0),
                persistence_score=0.3,
                actionability_score=0.65 if has_div else 0.45,
                regime_relevance="MEDIUM",
                downstream_targets=["methodology", "portfolio_construction"],
                recommended_followup="Investigate pair dynamics; check for structural shift",
            ))

        # ---------- MOMENTUM_DIVERGENCE ----------
        for div in divergences:
            # Skip tickers already captured as part of RELATIVE_VALUE_SETUP
            already_rv = any(
                ld.category == "RELATIVE_VALUE_SETUP" and div["ticker"] in ld.tickers
                for ld in leads
            )
            if already_rv:
                continue
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("MOMENTUM_DIVERGENCE", [div["ticker"]]),
                category="MOMENTUM_DIVERGENCE",
                subcategory=div["type"],
                tickers=[div["ticker"]],
                summary=f"Momentum divergence ({div['type']}): {div['ticker']} "
                        f"RSI {div['rsi_prior']:.0f}->{div['rsi_recent']:.0f}",
                evidence=[
                    f"Type: {div['type']}",
                    f"RSI prior: {div['rsi_prior']:.1f}",
                    f"RSI recent: {div['rsi_recent']:.1f}",
                ],
                severity_score=0.4,
                confidence_score=0.35,
                persistence_score=0.3,
                actionability_score=0.4,
                regime_relevance="LOW",
                downstream_targets=["methodology"],
                recommended_followup="Monitor for confirmation or reversal",
            ))

        # ---------- DISPERSION_EVENT ----------
        # If we see both positive and negative anomalies across sectors
        if pos_anomaly_tickers and neg_anomaly_tickers:
            spread_tickers = sorted(pos_anomaly_tickers | neg_anomaly_tickers)
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("DISPERSION_EVENT", spread_tickers),
                category="DISPERSION_EVENT",
                subcategory="sector_divergence",
                tickers=spread_tickers,
                summary=f"Sector dispersion: {len(pos_anomaly_tickers)} up / "
                        f"{len(neg_anomaly_tickers)} down extreme moves",
                evidence=[
                    f"Positive extremes: {', '.join(sorted(pos_anomaly_tickers))}",
                    f"Negative extremes: {', '.join(sorted(neg_anomaly_tickers))}",
                ],
                severity_score=0.5,
                confidence_score=0.5,
                persistence_score=0.3,
                actionability_score=0.6,
                regime_relevance="MEDIUM",
                downstream_targets=["methodology", "portfolio_construction", "optimizer"],
                recommended_followup="Evaluate sector allocation; look for pair trades",
            ))

        # ---------- Isolated single-domain anomalies (LOW confidence) ----------
        for anom in anomalies:
            ticker = anom["ticker"]
            already_captured = any(ticker in ld.tickers for ld in leads)
            if already_captured:
                continue
            leads.append(ResearchLead(
                lead_id=self._make_lead_id("MOMENTUM_DIVERGENCE", [ticker, anom.get("date", "")]),
                category=("MACRO_STRESS" if anom["anomaly_type"] == "extreme_negative"
                          else "MOMENTUM_DIVERGENCE"),
                subcategory="isolated_anomaly",
                tickers=[ticker],
                summary=f"Isolated anomaly: {ticker} {anom['return_pct']:.1f}% (z={anom['magnitude']:.1f})",
                evidence=[f"Return: {anom['return_pct']:.1f}%", f"Z-score: {anom['magnitude']:.2f}"],
                severity_score=min(1.0, abs(anom["magnitude"]) / 6.0),
                confidence_score=0.25,  # single-domain, low confidence
                persistence_score=0.3,
                actionability_score=0.2,
                regime_relevance="LOW",
                downstream_targets=["methodology"],
                recommended_followup="Watch for cross-confirmation before acting",
            ))

        return leads

    @staticmethod
    def _make_lead_id(category: str, tickers: List[str]) -> str:
        """Deterministic lead ID from category + tickers for persistence tracking."""
        raw = f"{category}:{'|'.join(sorted(tickers))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    # -----------------------------------------------------------------
    # 3. Cross-Confirmation Logic
    # -----------------------------------------------------------------
    def cross_confirm(self, leads: List[ResearchLead]) -> List[ResearchLead]:
        """
        Boost or penalize leads based on cross-domain confirmation.

        Rules:
        - 2+ detection domains supporting a lead -> confidence +0.2
        - Macro + correlation + momentum aligned -> severity +0.3
        - Single-domain only -> cap confidence at 0.4
        - Contradictory signals (stress + breadth healthy) -> confidence -0.2
        """
        # Build domain presence maps
        domain_tickers: Dict[str, set] = {
            "anomaly": set(),
            "correlation": set(),
            "divergence": set(),
            "macro": set(),
        }

        for ld in leads:
            cat = ld.category
            for t in ld.tickers:
                if cat in ("MACRO_STRESS", "REGIME_TRANSITION_PRECURSOR", "CREDIT_FRAGILITY"):
                    domain_tickers["macro"].add(t)
                if cat in ("CORRELATION_BREAK", "RELATIVE_VALUE_SETUP", "DISPERSION_EVENT"):
                    domain_tickers["correlation"].add(t)
                if cat in ("MOMENTUM_DIVERGENCE",):
                    domain_tickers["divergence"].add(t)
                if cat in ("RISK_OFF_ACCELERATION", "DEFENSIVE_ROTATION"):
                    domain_tickers["anomaly"].add(t)

        # Check for healthy-breadth contradiction: if most anomalies are positive
        # but we have stress leads, that's contradictory
        has_stress_leads = any(
            ld.category in ("MACRO_STRESS", "CREDIT_FRAGILITY", "RISK_OFF_ACCELERATION")
            for ld in leads
        )
        has_breadth_healthy = any(
            ld.category == "DEFENSIVE_ROTATION" for ld in leads
        ) and not has_stress_leads

        for ld in leads:
            ticker_set = set(ld.tickers)

            # Count how many domains this lead's tickers appear in
            domain_count = sum(
                1 for domain_set in domain_tickers.values()
                if ticker_set & domain_set
            )

            # Multi-domain boost
            if domain_count >= 2:
                ld.confidence_score = min(1.0, ld.confidence_score + 0.2)

            # Triple-alignment boost
            if (domain_tickers["macro"] & ticker_set
                    and domain_tickers["correlation"] & ticker_set
                    and domain_tickers["divergence"] & ticker_set):
                ld.severity_score = min(1.0, ld.severity_score + 0.3)

            # Single-domain cap
            if domain_count <= 1 and ld.subcategory == "isolated_anomaly":
                ld.confidence_score = min(ld.confidence_score, 0.4)

            # Contradiction penalty
            if has_breadth_healthy and ld.category in (
                "MACRO_STRESS", "CREDIT_FRAGILITY", "RISK_OFF_ACCELERATION"
            ):
                ld.confidence_score = max(0.0, ld.confidence_score - 0.2)

        return leads

    # -----------------------------------------------------------------
    # 4. Persistence Engine
    # -----------------------------------------------------------------
    def compute_persistence(
        self, leads: List[ResearchLead], history: Optional[List[Dict]] = None
    ) -> List[ResearchLead]:
        """
        Update persistence_score based on how often similar leads recurred.

        Loads previous scout reports from scout_history.json.
        - Recurring 3+ times -> persistence_score = 0.8+
        - New (first time)   -> persistence_score = 0.3
        - Was present but now resolved -> mark as RESOLVED (not returned)
        """
        if history is None:
            history = self._load_history()

        # Build occurrence count per lead_id
        occurrence_count: Dict[str, int] = {}
        first_seen: Dict[str, str] = {}
        for entry in history:
            for past_lead in entry.get("leads", []):
                lid = past_lead.get("lead_id", "")
                occurrence_count[lid] = occurrence_count.get(lid, 0) + 1
                if lid not in first_seen:
                    first_seen[lid] = entry.get("timestamp", "")

        for ld in leads:
            count = occurrence_count.get(ld.lead_id, 0)
            if count >= 3:
                ld.persistence_score = min(1.0, 0.8 + 0.05 * (count - 3))
                ld.aging_days = count
            elif count >= 1:
                ld.persistence_score = 0.3 + 0.15 * count
                ld.aging_days = count
            else:
                ld.persistence_score = 0.3
                ld.aging_days = 0

        return leads

    def _load_history(self) -> List[Dict]:
        """Load scout_history.json if it exists."""
        if HISTORY_PATH.exists():
            try:
                data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as exc:
                log.warning("Failed to load scout history: %s", exc)
        return []

    # -----------------------------------------------------------------
    # 5. Priority Scoring
    # -----------------------------------------------------------------
    def score_and_rank(self, leads: List[ResearchLead]) -> List[ResearchLead]:
        """
        Compute composite priority score and sort descending.

        priority = 0.25*severity + 0.20*confidence + 0.20*persistence
                 + 0.20*actionability + 0.15*regime_relevance_numeric

        Top 5 = actionable, next 10 = watchlist, rest = archive.
        """
        regime_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.2}

        for ld in leads:
            regime_num = regime_map.get(ld.regime_relevance, 0.2)
            ld.priority_score = round(
                0.25 * ld.severity_score
                + 0.20 * ld.confidence_score
                + 0.20 * ld.persistence_score
                + 0.20 * ld.actionability_score
                + 0.15 * regime_num,
                4,
            )

        leads.sort(key=lambda x: x.priority_score, reverse=True)
        return leads

    # -----------------------------------------------------------------
    # 6. Opportunity Discovery Engine
    # -----------------------------------------------------------------
    def discover_opportunities(self, leads: List[ResearchLead]) -> List[Dict]:
        """
        Extract structured trading opportunities from high-priority research leads.

        Opportunity types: RELATIVE_VALUE_SETUP, DISPERSION_EVENT,
        DEFENSIVE_ROTATION, MEAN_REVERSION.
        """
        opportunities: List[Dict[str, Any]] = []

        for ld in leads:
            if ld.priority_score < 0.3:
                continue

            opp: Optional[Dict[str, Any]] = None

            if ld.category == "RELATIVE_VALUE_SETUP" and len(ld.tickers) >= 2:
                opp = {
                    "opportunity_id": f"RV_{ld.lead_id[:8]}",
                    "category": "RELATIVE_VALUE_SETUP",
                    "instruments": ld.tickers,
                    "setup": f"Pair divergence: {'/'.join(ld.tickers)}",
                    "evidence": ld.evidence,
                    "confidence": round(ld.confidence_score, 3),
                    "urgency": "HIGH" if ld.severity_score > 0.6 else "MEDIUM",
                    "expiry_estimate": "5-15 trading days",
                }

            elif ld.category == "DISPERSION_EVENT":
                opp = {
                    "opportunity_id": f"DISP_{ld.lead_id[:8]}",
                    "category": "DISPERSION_EVENT",
                    "instruments": ld.tickers,
                    "setup": ld.summary,
                    "evidence": ld.evidence,
                    "confidence": round(ld.confidence_score, 3),
                    "urgency": "MEDIUM",
                    "expiry_estimate": "5-20 trading days",
                }

            elif ld.category == "DEFENSIVE_ROTATION":
                opp = {
                    "opportunity_id": f"DEF_{ld.lead_id[:8]}",
                    "category": "DEFENSIVE_ROTATION",
                    "instruments": ld.tickers,
                    "setup": f"Defensive strengthening: {', '.join(ld.tickers)}",
                    "evidence": ld.evidence,
                    "confidence": round(ld.confidence_score, 3),
                    "urgency": "LOW",
                    "expiry_estimate": "10-30 trading days",
                }

            elif ld.category == "MOMENTUM_DIVERGENCE" and ld.subcategory == "bullish_divergence":
                opp = {
                    "opportunity_id": f"MR_{ld.lead_id[:8]}",
                    "category": "MEAN_REVERSION",
                    "instruments": ld.tickers,
                    "setup": f"Bullish divergence mean-reversion: {', '.join(ld.tickers)}",
                    "evidence": ld.evidence,
                    "confidence": round(ld.confidence_score, 3),
                    "urgency": "MEDIUM",
                    "expiry_estimate": "5-10 trading days",
                }

            if opp is not None:
                opportunities.append(opp)

        return opportunities[:15]

    # -----------------------------------------------------------------
    # 7. Risk Lead Generation
    # -----------------------------------------------------------------
    def generate_risk_leads(
        self, leads: List[ResearchLead], macro: dict
    ) -> List[Dict]:
        """
        Separate risk-focused findings for downstream risk consumers.

        Categories: CREDIT_FRAGILITY, CORRELATION_CLUSTERING,
        DEFENSIVE_FAILURE, LIQUIDITY_WARNING.
        """
        risk_leads: List[Dict[str, Any]] = []

        for ld in leads:
            if ld.category == "CREDIT_FRAGILITY":
                risk_leads.append({
                    "risk_id": f"RISK_CF_{ld.lead_id[:8]}",
                    "category": "CREDIT_FRAGILITY",
                    "severity": round(ld.severity_score, 3),
                    "instruments": ld.tickers,
                    "warning": ld.summary,
                    "downstream_target": "risk_guardian",
                })

            elif ld.category in ("CORRELATION_BREAK", "RELATIVE_VALUE_SETUP"):
                risk_leads.append({
                    "risk_id": f"RISK_CC_{ld.lead_id[:8]}",
                    "category": "CORRELATION_CLUSTERING",
                    "severity": round(ld.severity_score, 3),
                    "instruments": ld.tickers,
                    "warning": f"Correlation regime shift: {ld.summary}",
                    "downstream_target": "portfolio_construction",
                })

            elif ld.category == "RISK_OFF_ACCELERATION":
                risk_leads.append({
                    "risk_id": f"RISK_LW_{ld.lead_id[:8]}",
                    "category": "LIQUIDITY_WARNING",
                    "severity": round(ld.severity_score, 3),
                    "instruments": ld.tickers,
                    "warning": f"Clustered sell-off may impair liquidity: {ld.summary}",
                    "downstream_target": "execution",
                })

        # DEFENSIVE_FAILURE: if macro is stressed but no defensive rotation detected
        has_defensive = any(ld.category == "DEFENSIVE_ROTATION" for ld in leads)
        macro_score = macro.get("macro_score", 0.0)
        if macro_score < -0.3 and not has_defensive:
            risk_leads.append({
                "risk_id": "RISK_DF_macro_no_defense",
                "category": "DEFENSIVE_FAILURE",
                "severity": 0.6,
                "instruments": ["MACRO"],
                "warning": "Macro stress without defensive rotation — hedges may be ineffective",
                "downstream_target": "regime_forecaster",
            })

        return risk_leads

    # -----------------------------------------------------------------
    # 8. Downstream Action Builder
    # -----------------------------------------------------------------
    def build_downstream_actions(
        self,
        leads: List[ResearchLead],
        opportunities: List[Dict],
        risk_leads: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Build structured action recommendations for each downstream agent.
        """
        actions: Dict[str, Dict[str, Any]] = {
            "methodology": {"new_hypotheses": [], "sectors_to_revalidate": []},
            "regime_forecaster": {"transition_precursors": [], "macro_stress_themes": []},
            "alpha_decay": {"external_pressure_note": ""},
            "portfolio_construction": {"sectors_to_deprioritize": [], "concentration_warnings": []},
            "risk_guardian": {"fragility_warnings": [], "correlation_risk": []},
            "optimizer": {"new_feature_ideas": [], "regime_specific_hints": []},
            "execution": {"caution_instruments": [], "urgency_flags": []},
        }

        for ld in leads:
            if ld.priority_score < 0.2:
                continue

            if "methodology" in ld.downstream_targets:
                if ld.category in ("RELATIVE_VALUE_SETUP", "DISPERSION_EVENT"):
                    actions["methodology"]["new_hypotheses"].append(
                        f"{ld.category}: {ld.summary}"
                    )
                for t in ld.tickers:
                    if t not in actions["methodology"]["sectors_to_revalidate"] and t != "MACRO":
                        actions["methodology"]["sectors_to_revalidate"].append(t)

            if "regime_forecaster" in ld.downstream_targets:
                if ld.category == "REGIME_TRANSITION_PRECURSOR":
                    actions["regime_forecaster"]["transition_precursors"].append(ld.summary)
                if ld.category in ("MACRO_STRESS", "CREDIT_FRAGILITY"):
                    actions["regime_forecaster"]["macro_stress_themes"].append(ld.summary)

            if "risk_guardian" in ld.downstream_targets:
                if ld.category == "CREDIT_FRAGILITY":
                    actions["risk_guardian"]["fragility_warnings"].append(ld.summary)
                if ld.category in ("CORRELATION_BREAK", "RELATIVE_VALUE_SETUP"):
                    actions["risk_guardian"]["correlation_risk"].append(ld.summary)

            if "execution" in ld.downstream_targets:
                for t in ld.tickers:
                    if t not in actions["execution"]["caution_instruments"] and t != "MACRO":
                        actions["execution"]["caution_instruments"].append(t)
                if ld.severity_score > 0.6:
                    actions["execution"]["urgency_flags"].append(ld.summary)

            if "portfolio_construction" in ld.downstream_targets:
                if ld.severity_score > 0.5:
                    for t in ld.tickers:
                        if t not in actions["portfolio_construction"]["sectors_to_deprioritize"]:
                            actions["portfolio_construction"]["sectors_to_deprioritize"].append(t)
                if ld.category in ("CORRELATION_BREAK", "DISPERSION_EVENT"):
                    actions["portfolio_construction"]["concentration_warnings"].append(ld.summary)

            if "optimizer" in ld.downstream_targets:
                if ld.category == "REGIME_TRANSITION_PRECURSOR":
                    actions["optimizer"]["regime_specific_hints"].append(ld.summary)
                if ld.category in ("DISPERSION_EVENT", "RELATIVE_VALUE_SETUP"):
                    actions["optimizer"]["new_feature_ideas"].append(
                        f"Feature idea from {ld.category}: {', '.join(ld.tickers)}"
                    )

        # Alpha decay external pressure note
        macro_leads = [ld for ld in leads if ld.category == "MACRO_STRESS"]
        if macro_leads:
            actions["alpha_decay"]["external_pressure_note"] = (
                "Observed sector weakness may be macro-driven, not alpha decay"
            )

        # Deduplicate lists
        for agent_key in actions:
            for field_key in actions[agent_key]:
                val = actions[agent_key][field_key]
                if isinstance(val, list):
                    seen: List[str] = []
                    for item in val:
                        if item not in seen:
                            seen.append(item)
                    actions[agent_key][field_key] = seen

        return actions

    # -----------------------------------------------------------------
    # 9. Macro Intelligence Engine
    # -----------------------------------------------------------------
    def compute_macro_intelligence(self) -> Dict[str, Any]:
        """
        Deep macro analysis beyond the existing macro_score.

        Returns per-domain detail (level, z_score, momentum, acceleration,
        direction), macro themes, and aggregate risk/transition/confidence
        scores.
        """
        fred_data = self.fetch_fred_data()
        domains: Dict[str, Dict[str, Any]] = {}

        for series_name, data in fred_data.items():
            z = data.get("z_score", 0.0)
            trend = data.get("trend", 0.0)
            # Acceleration proxy: sign(trend) — positive trend = accelerating
            accel = 1.0 if trend > 0.01 else (-1.0 if trend < -0.01 else 0.0)
            if abs(trend) > 0:
                direction = "rising" if trend > 0 else "falling"
            else:
                direction = "flat"

            domains[series_name] = {
                "level": data.get("value", 0.0),
                "z_score": round(z, 3),
                "momentum": round(trend, 4),
                "acceleration": round(accel, 2),
                "direction": direction,
            }

        # --- Macro themes ---
        themes: Dict[str, float] = {}

        yc = fred_data.get("yield_curve", {})
        hy = fred_data.get("hy_spread", {})
        dollar = fred_data.get("dollar_index", {})

        # Rates pressure: yield curve flattening / inverting
        yc_val = yc.get("value", 0)
        themes["rates_pressure"] = round(max(0.0, min(1.0, -yc_val / 1.0 + 0.5)), 3)

        # Credit stress: HY spread z-score
        hy_z = hy.get("z_score", 0)
        themes["credit_stress"] = round(max(0.0, min(1.0, hy_z / 2.0)), 3)

        # Dollar tightening: dollar strength
        d_trend = dollar.get("trend", 0)
        themes["dollar_tightening"] = round(max(0.0, min(1.0, d_trend * 10)), 3)

        # Growth scare: inverted YC + widening spreads
        growth_scare = 0.0
        if yc_val < 0:
            growth_scare += 0.4
        if hy_z > 1.0:
            growth_scare += 0.3
        if d_trend > 0.02:
            growth_scare += 0.2
        themes["growth_scare"] = round(min(1.0, growth_scare), 3)

        # Stabilization: opposite of stress
        themes["stabilization"] = round(max(0.0, 1.0 - themes["growth_scare"]), 3)

        # --- Aggregate scores ---
        stress_vals = [themes["rates_pressure"], themes["credit_stress"],
                       themes["dollar_tightening"], themes["growth_scare"]]
        macro_risk_score = round(float(np.mean(stress_vals)) if stress_vals else 0.0, 3)
        macro_transition_pressure = round(
            min(1.0, themes["rates_pressure"] * 0.4
                + themes["credit_stress"] * 0.3
                + themes["growth_scare"] * 0.3),
            3,
        )
        # Confidence: higher when we have more data
        data_coverage = len(fred_data) / max(len(FRED_SERIES), 1)
        macro_confidence = round(min(1.0, data_coverage), 3)

        return {
            "domains": domains,
            "themes": themes,
            "macro_risk_score": macro_risk_score,
            "macro_transition_pressure": macro_transition_pressure,
            "macro_confidence": macro_confidence,
        }

    # -----------------------------------------------------------------
    # 10. Watchlist / History Tracker
    # -----------------------------------------------------------------
    def update_watchlist(self, leads: List[ResearchLead]) -> Dict[str, Any]:
        """
        Save current leads to scout_history.json (cap 365 entries).

        Tracks recurring leads, aging, and resolution.
        """
        history = self._load_history()

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leads": [asdict(ld) for ld in leads],
        }
        history.append(entry)

        # Cap at 365 entries
        if len(history) > 365:
            history = history[-365:]

        try:
            HISTORY_PATH.write_text(
                json.dumps(history, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Updated scout history (%d entries)", len(history))
        except Exception as exc:
            log.warning("Failed to save scout history: %s", exc)

        return {
            "total_entries": len(history),
            "current_leads": len(leads),
            "timestamp": entry["timestamp"],
        }

    def get_open_watchlist(self) -> List[Dict]:
        """Return active unresolved leads from the most recent history entry."""
        history = self._load_history()
        if not history:
            return []
        latest = history[-1]
        return latest.get("leads", [])

    def get_recurring_events(self, n: int = 10) -> List[Dict]:
        """Return leads that appeared n+ times across history."""
        history = self._load_history()
        occurrence_count: Dict[str, Dict] = {}

        for entry in history:
            for past_lead in entry.get("leads", []):
                lid = past_lead.get("lead_id", "")
                if lid not in occurrence_count:
                    occurrence_count[lid] = {
                        "lead_id": lid,
                        "category": past_lead.get("category", ""),
                        "summary": past_lead.get("summary", ""),
                        "tickers": past_lead.get("tickers", []),
                        "count": 0,
                    }
                occurrence_count[lid]["count"] += 1

        recurring = [v for v in occurrence_count.values() if v["count"] >= n]
        recurring.sort(key=lambda x: x["count"], reverse=True)
        return recurring

    # -----------------------------------------------------------------
    # 11. Machine Summary Builder
    # -----------------------------------------------------------------
    def build_machine_summary(
        self,
        leads: List[ResearchLead],
        risk_leads: List[Dict],
        opportunities: List[Dict],
        macro_intel: Dict,
        watchlist_info: Dict,
    ) -> Dict[str, Any]:
        """
        Build compact machine-readable summary for downstream consumption.
        """
        high_conf = [ld for ld in leads if ld.confidence_score >= 0.6]
        low_conf = [ld for ld in leads if ld.confidence_score < 0.4]

        # Collect unique downstream targets from top leads
        key_targets: List[str] = []
        for ld in leads[:5]:
            for t in ld.downstream_targets:
                if t not in key_targets:
                    key_targets.append(t)

        # Implementation caution level
        macro_risk = macro_intel.get("macro_risk_score", 0.0)
        if macro_risk > 0.7:
            caution = "HIGH"
        elif macro_risk > 0.4:
            caution = "MEDIUM"
        else:
            caution = "LOW"

        return {
            "macro_risk_score": macro_intel.get("macro_risk_score", 0.0),
            "macro_transition_pressure": macro_intel.get("macro_transition_pressure", 0.0),
            "top_research_leads": [
                {"lead": f"{ld.category}: {', '.join(ld.tickers)}", "priority": ld.priority_score}
                for ld in leads[:5]
            ],
            "top_risk_leads": [
                {"risk": rl.get("category", ""), "severity": rl.get("severity", 0)}
                for rl in risk_leads[:5]
            ],
            "top_opportunities": [
                {"opportunity": op.get("category", ""), "confidence": op.get("confidence", 0)}
                for op in opportunities[:5]
            ],
            "watchlist_count": watchlist_info.get("current_leads", 0),
            "high_confidence_count": len(high_conf),
            "low_confidence_count": len(low_conf),
            "key_downstream_targets": key_targets,
            "implementation_caution": caution,
        }

    # -----------------------------------------------------------------
    # Report generation (original)
    # -----------------------------------------------------------------
    def generate_report(self) -> dict:
        """Combine all findings into a comprehensive scout report."""
        log.info("Generating Data Scout report...")

        macro = self.compute_macro_regime_signals()
        anomalies = self.detect_volume_anomalies()
        corr_breaks = self.detect_correlation_breaks()
        divergences = self.detect_momentum_divergence()

        # Build top opportunities
        opportunities: List[str] = []

        # From correlation breaks
        for brk in corr_breaks[:3]:
            opportunities.append(
                f"Correlation break: {brk['pair']} ({brk['direction']}, "
                f"20d={brk['corr_20d']:.2f} vs 120d={brk['corr_120d']:.2f})"
            )

        # From momentum divergences
        for div in divergences[:3]:
            direction = "bullish" if div["type"] == "bullish_divergence" else "bearish"
            opportunities.append(
                f"Momentum divergence ({direction}): {div['ticker']} "
                f"RSI {div['rsi_prior']:.0f}->{div['rsi_recent']:.0f}"
            )

        # Trim to top 5
        opportunities = opportunities[:5]

        # Build risk flags
        risk_flags: List[str] = []

        macro_score = macro.get("macro_score", 0)
        if macro_score < -0.3:
            risk_flags.append(f"Macro environment risk-off (score={macro_score:.2f})")

        yc_info = macro.get("yield_curve", {})
        if yc_info.get("signal") == "inverted":
            risk_flags.append("Yield curve inverted -- recession risk elevated")

        hy_info = macro.get("hy_spread", {})
        if hy_info.get("signal") == "stress":
            risk_flags.append("HY spreads widening -- credit stress detected")

        for anom in anomalies[:3]:
            if anom["anomaly_type"] == "extreme_negative":
                risk_flags.append(
                    f"Extreme negative move in {anom['ticker']}: "
                    f"{anom['return_pct']:.1f}% (z={anom['magnitude']:.1f})"
                )

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "macro": macro,
            "anomalies": anomalies,
            "correlation_breaks": corr_breaks,
            "momentum_divergences": divergences,
            "opportunities": opportunities,
            "risk_flags": risk_flags,
        }

        return report

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    def _save_report(self, report: dict) -> None:
        """Save report to JSON file."""
        try:
            REPORT_PATH.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Scout report saved to %s", REPORT_PATH)
        except Exception as exc:
            log.warning("Failed to save scout report: %s", exc)

    # -----------------------------------------------------------------
    # Publish to bus
    # -----------------------------------------------------------------
    def _publish_to_bus(self, report: dict) -> None:
        """Publish report summary to agent bus."""
        if not _IMPORTS_OK.get("agent_bus"):
            return
        try:
            bus = get_bus()
            bus.publish("data_scout", {
                "event": "scout_report",
                "macro_score": report.get("macro", {}).get("macro_score", 0),
                "n_anomalies": len(report.get("anomalies", [])),
                "n_corr_breaks": len(report.get("correlation_breaks", [])),
                "n_divergences": len(report.get("momentum_divergences", [])),
                "risk_flags": report.get("risk_flags", []),
                "opportunities": report.get("opportunities", []),
            })
            log.info("Published scout report to agent bus")
        except Exception as exc:
            log.warning("Failed to publish to agent bus: %s", exc)

    # -----------------------------------------------------------------
    # Full run cycle
    # -----------------------------------------------------------------
    def run(self) -> dict:
        """
        Full cycle: fetch -> detect -> report -> save -> publish to bus.

        Returns the report dict.
        """
        # Registry heartbeat: RUNNING
        if _IMPORTS_OK.get("registry"):
            try:
                registry = get_registry()
                registry.register("data_scout", role="external data scanner for signals and anomalies")
                registry.heartbeat("data_scout", AgentStatus.RUNNING)
            except Exception as exc:
                log.warning("Registry heartbeat failed: %s", exc)

        try:
            report = self.generate_report()
            self._save_report(report)
            self._publish_to_bus(report)

            # ---- Research Intelligence Desk pipeline ----
            try:
                # 1. Classify events into ResearchLeads
                leads = self.classify_events(
                    anomalies=report.get("anomalies", []),
                    corr_breaks=report.get("correlation_breaks", []),
                    divergences=report.get("momentum_divergences", []),
                    macro=report.get("macro", {}),
                )
                log.info("Classified %d research leads", len(leads))

                # 2. Cross-confirm
                leads = self.cross_confirm(leads)

                # 3. Compute persistence from history
                leads = self.compute_persistence(leads)

                # 4. Score and rank
                leads = self.score_and_rank(leads)
                log.info(
                    "Ranked %d leads, top priority=%.4f",
                    len(leads),
                    leads[0].priority_score if leads else 0.0,
                )

                # 5. Discover opportunities
                opportunities = self.discover_opportunities(leads)

                # 6. Generate risk leads
                risk_leads = self.generate_risk_leads(leads, report.get("macro", {}))

                # 7. Build downstream actions
                downstream_actions = self.build_downstream_actions(
                    leads, opportunities, risk_leads,
                )

                # 8. Compute macro intelligence
                macro_intel = self.compute_macro_intelligence()

                # 9. Update watchlist
                watchlist_info = self.update_watchlist(leads)

                # 10. Build machine summary
                machine_summary = self.build_machine_summary(
                    leads, risk_leads, opportunities, macro_intel, watchlist_info,
                )

                # Enrich report with Research Intelligence Desk outputs
                report["research_leads"] = [asdict(ld) for ld in leads]
                report["research_opportunities"] = opportunities
                report["risk_leads"] = risk_leads
                report["downstream_actions"] = downstream_actions
                report["macro_intelligence"] = macro_intel
                report["watchlist_info"] = watchlist_info
                report["machine_summary"] = machine_summary

                # Re-save the enriched report
                self._save_report(report)

                # Publish enriched summary to bus
                if _IMPORTS_OK.get("agent_bus"):
                    try:
                        bus = get_bus()
                        bus.publish("data_scout_intelligence", {
                            "event": "research_intelligence_report",
                            "machine_summary": machine_summary,
                        })
                    except Exception as exc:
                        log.warning("Failed to publish intelligence to bus: %s", exc)

                log.info(
                    "Research Intelligence Desk complete: %d leads, %d opportunities, "
                    "%d risk leads, macro_risk=%.3f",
                    len(leads),
                    len(opportunities),
                    len(risk_leads),
                    macro_intel.get("macro_risk_score", 0.0),
                )

            except Exception as intel_exc:
                log.warning("Research Intelligence pipeline failed (base report OK): %s", intel_exc)

            # Registry heartbeat: COMPLETED
            if _IMPORTS_OK.get("registry"):
                try:
                    registry = get_registry()
                    registry.heartbeat("data_scout", AgentStatus.COMPLETED)
                except Exception as exc:
                    log.warning("Registry heartbeat failed: %s", exc)

            log.info(
                "Data Scout run complete: macro_score=%.3f, %d anomalies, "
                "%d corr breaks, %d divergences",
                report.get("macro", {}).get("macro_score", 0),
                len(report.get("anomalies", [])),
                len(report.get("correlation_breaks", [])),
                len(report.get("momentum_divergences", [])),
            )
            return report

        except Exception as exc:
            log.error("Data Scout run failed: %s", exc)
            # Registry heartbeat: FAILED
            if _IMPORTS_OK.get("registry"):
                try:
                    registry = get_registry()
                    registry.heartbeat("data_scout", AgentStatus.FAILED, error=str(exc))
                except Exception:
                    pass
            raise


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Data Scout Agent")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--interval", type=int, default=600,
        help="Seconds between runs in loop mode (default: 600)",
    )
    args = parser.parse_args()

    scout = DataScout()

    if args.once:
        report = scout.run()
        log.info(
            "Data Scout completed: macro_score=%.3f",
            report.get("macro", {}).get("macro_score", 0),
        )
        return

    # Loop mode
    log.info("Data Scout starting in loop mode (interval=%ds)", args.interval)
    while True:
        try:
            report = scout.run()
            log.info(
                "Cycle complete: macro_score=%.3f",
                report.get("macro", {}).get("macro_score", 0),
            )
        except Exception as exc:
            log.error("Data Scout cycle failed: %s", exc)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
