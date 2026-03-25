"""
Data Scout Agent -- Scans external data sources for signals and anomalies.

Expands the information edge by fetching macro data from FRED,
detecting volume anomalies, correlation breaks, and momentum divergences
in sector ETFs.

Output: agents/data_scout/scout_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

    # -----------------------------------------------------------------
    # Report generation
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
