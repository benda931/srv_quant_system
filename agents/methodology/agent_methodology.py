"""
agents/methodology/agent_methodology.py
-----------------------------------------
סוכן מתודולוגיה — מריץ את הפייפליין המלא, מעריך תוצאות,
מפרסם ל-AgentBus ושומר דוח מפורט.

Methodology Agent — runs the full trading methodology pipeline,
evaluates backtest quality, signal stack health, and stress resilience.
Publishes structured results and tracks performance over time.

CLI:
    python agents/methodology/agent_methodology.py --once
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── נתיב שורש — 3 רמות מעלה: agents/methodology/agent_methodology.py → ROOT ──
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings, Settings

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "agent_methodology.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_methodology")

# ── ייבוא מודולי המערכת — עם טיפול בשגיאות ──────────────────────────────────
_IMPORTS_OK: Dict[str, bool] = {}

try:
    from scripts.run_all import run_pipeline
    _IMPORTS_OK["run_pipeline"] = True
except ImportError as e:
    log.warning("Could not import run_pipeline: %s", e)
    _IMPORTS_OK["run_pipeline"] = False

try:
    from analytics.stat_arb import QuantEngine
    _IMPORTS_OK["QuantEngine"] = True
except ImportError as e:
    log.warning("Could not import QuantEngine: %s", e)
    _IMPORTS_OK["QuantEngine"] = False

try:
    from analytics.backtest import run_backtest as run_wf_backtest, BacktestResult
    _IMPORTS_OK["backtest"] = True
except ImportError as e:
    log.warning("Could not import backtest: %s", e)
    _IMPORTS_OK["backtest"] = False

try:
    from analytics.signal_stack import SignalStackEngine, signal_stack_summary
    _IMPORTS_OK["signal_stack"] = True
except ImportError as e:
    log.warning("Could not import signal_stack: %s", e)
    _IMPORTS_OK["signal_stack"] = False

try:
    from analytics.signal_regime_safety import compute_regime_safety_score
    _IMPORTS_OK["regime_safety"] = True
except ImportError as e:
    log.warning("Could not import signal_regime_safety: %s", e)
    _IMPORTS_OK["regime_safety"] = False

try:
    from analytics.stress import StressEngine
    _IMPORTS_OK["stress"] = True
except ImportError as e:
    log.warning("Could not import StressEngine: %s", e)
    _IMPORTS_OK["stress"] = False

try:
    from analytics.methodology_lab import MethodologyLab, ALL_METHODOLOGIES
    _IMPORTS_OK["methodology_lab"] = True
except ImportError as e:
    log.warning("Could not import methodology_lab: %s", e)
    _IMPORTS_OK["methodology_lab"] = False

try:
    from analytics.tail_risk import compute_expected_shortfall, parametric_correlation_stress, tail_correlation_diagnostic
    _IMPORTS_OK["tail_risk"] = True
except ImportError as e:
    log.warning("Could not import tail_risk: %s", e)
    _IMPORTS_OK["tail_risk"] = False

try:
    from scripts.agent_bus import get_bus
    _IMPORTS_OK["agent_bus"] = True
except ImportError as e:
    log.warning("Could not import agent_bus: %s", e)
    _IMPORTS_OK["agent_bus"] = False

try:
    from agents.shared.agent_registry import get_registry, AgentStatus
    _IMPORTS_OK["registry"] = True
except ImportError as e:
    log.warning("Could not import agent_registry: %s", e)
    _IMPORTS_OK["registry"] = False

# ── Portfolio Tracker ────────────────────────────────────────────────────────
from agents.methodology.methodology_portfolio import MethodologyPortfolio

# ── תיקיית דוחות ─────────────────────────────────────────────────────────────
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# עזרים
# =============================================================================

def _sf(x: Any, default: float = 0.0) -> float:
    """Safe float — מחזיר default אם הערך לא תקין."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _pct_delta(old: float, new: float) -> str:
    """פורמט שינוי אחוזי: '+8.3%' / '-2.1%'."""
    if abs(old) < 1e-12:
        return "N/A"
    delta = (new - old) / abs(old) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _regime_breakdown_to_dict(breakdown: dict) -> dict:
    """ממיר RegimeBreakdown dataclasses ל-dict פשוט."""
    result = {}
    for regime, rb in breakdown.items():
        if hasattr(rb, "__dict__"):
            result[regime] = {
                "regime": getattr(rb, "regime", regime),
                "n_walks": getattr(rb, "n_walks", 0),
                "ic_mean": round(_sf(getattr(rb, "ic_mean", 0)), 4),
                "ic_ir": round(_sf(getattr(rb, "ic_ir", 0)), 4),
                "hit_rate": round(_sf(getattr(rb, "hit_rate", 0)), 4),
                "sharpe": round(_sf(getattr(rb, "sharpe", 0)), 4),
            }
        elif isinstance(rb, dict):
            result[regime] = rb
        else:
            result[regime] = {"raw": str(rb)}
    return result


# =============================================================================
# גוף הסוכן
# =============================================================================

def run(once: bool = False) -> dict:
    """
    הרצת הסוכן — pipeline מלא, backtest, signal stack, stress, דוח.

    Parameters
    ----------
    once : bool
        True = ריצה חד-פעמית (ללא loop).

    Returns
    -------
    dict — דוח מפורט של הריצה.
    """
    started_at = datetime.now(timezone.utc)
    today = date.today().isoformat()
    settings = get_settings()
    portfolio = MethodologyPortfolio()

    # רישום ב-registry
    if _IMPORTS_OK.get("registry"):
        registry = get_registry()
        registry.register("agent_methodology", role="methodology evaluation & benchmarking")
        registry.heartbeat("agent_methodology", AgentStatus.RUNNING)

    log.info("=" * 70)
    log.info("Methodology Agent — %s", today)
    log.info("=" * 70)

    report: Dict[str, Any] = {
        "run_date": today,
        "started_at": started_at.isoformat(),
        "metrics": {},
        "regime_breakdown": {},
        "signal_stack": {},
        "stress_summary": {},
        "conclusions": [],
        "recommendations": [],
        "errors": [],
        "parameters_snapshot": {},
    }

    # ── שלב 1: הרצת Pipeline מלא ────────────────────────────────────────────
    pipeline_result = None
    if _IMPORTS_OK.get("run_pipeline"):
        log.info("[1/6] Running full pipeline (with backtest)...")
        try:
            pipeline_result = run_pipeline(
                force_refresh=False,
                run_backtest=True,
                run_ml=True,
                run_optimizer=True,
            )
            log.info(
                "  Pipeline: steps_ok=%s, steps_failed=%s",
                pipeline_result.get("steps_ok", []),
                pipeline_result.get("steps_failed", []),
            )
        except Exception as exc:
            log.exception("Pipeline failed")
            report["errors"].append(f"Pipeline: {exc}")
    else:
        log.warning("[1/6] run_pipeline not available — skipping")
        report["errors"].append("run_pipeline import failed")

    # ── שלב 2: QuantEngine → master_df ──────────────────────────────────────
    master_df = None
    engine = None
    if _IMPORTS_OK.get("QuantEngine"):
        log.info("[2/6] Running QuantEngine...")
        try:
            engine = QuantEngine(settings)
            engine.load()
            master_df = engine.calculate_conviction_score()
            n_sectors = len(master_df)
            log.info("  QuantEngine: %d sectors loaded", n_sectors)

            # שמירת סנאפשוט פרמטרים
            report["parameters_snapshot"] = {
                "pca_window": settings.pca_window,
                "zscore_window": settings.zscore_window,
                "n_sectors": n_sectors,
                "regime": str(getattr(engine, "regime_label", "UNKNOWN")),
            }
        except Exception as exc:
            log.exception("QuantEngine failed")
            report["errors"].append(f"QuantEngine: {exc}")
    else:
        log.warning("[2/6] QuantEngine not available — skipping")
        report["errors"].append("QuantEngine import failed")

    # ── שלב 3: Walk-Forward Backtest ────────────────────────────────────────
    backtest_result: Optional[Any] = None
    if _IMPORTS_OK.get("backtest") and engine is not None:
        log.info("[3/6] Running walk-forward backtest...")
        try:
            # שימוש בנתונים ישירות מה-engine (לא מ-artifacts)
            prices_df = getattr(engine, "prices", None)
            fundamentals_df = getattr(engine, "fundamentals", None)
            weights_df = getattr(engine, "weights", None)

            if prices_df is not None and not prices_df.empty:
                    backtest_result = run_wf_backtest(
                        prices_df=prices_df,
                        fundamentals_df=fundamentals_df if fundamentals_df is not None else prices_df.iloc[:0],
                        weights_df=weights_df if weights_df is not None else prices_df.iloc[:0],
                        settings=settings,
                    )

                    # מדדים מרכזיים מה-backtest
                    report["metrics"] = {
                        "ic_mean": round(_sf(backtest_result.ic_mean), 4),
                        "ic_ir": round(_sf(backtest_result.ic_ir), 4),
                        "sharpe": round(_sf(backtest_result.sharpe), 4),
                        "hit_rate": round(_sf(backtest_result.hit_rate), 4),
                        "max_dd": round(_sf(backtest_result.max_drawdown), 4),
                        "n_walks": backtest_result.n_walks,
                    }

                    # פירוט לפי רג'ים
                    report["regime_breakdown"] = _regime_breakdown_to_dict(
                        backtest_result.regime_breakdown
                    )

                    log.info(
                        "  Backtest: IC=%.4f, Sharpe=%.2f, hit=%.1f%%, DD=%.1f%%",
                        _sf(backtest_result.ic_mean),
                        _sf(backtest_result.sharpe),
                        _sf(backtest_result.hit_rate) * 100,
                        _sf(backtest_result.max_drawdown) * 100,
                    )
            else:
                report["errors"].append("Backtest: prices_df not available from engine")
        except Exception as exc:
            log.exception("Backtest failed")
            report["errors"].append(f"Backtest: {exc}")
    else:
        log.warning("[3/6] Backtest not available — skipping")
        if not _IMPORTS_OK.get("backtest"):
            report["errors"].append("backtest import failed")

    # ── שלב 4: Signal Stack ─────────────────────────────────────────────────
    signals: List[Any] = []
    if _IMPORTS_OK.get("signal_stack") and master_df is not None:
        log.info("[4/6] Computing Signal Stack metrics...")
        try:
            stack_engine = SignalStackEngine(settings)

            # חישוב Distortion parameters מהנתונים הזמינים
            frob_z = _sf(getattr(engine, "distortion_z", None), 0.0) if engine else 0.0
            mode_share = _sf(getattr(engine, "market_mode_strength", None), 0.3) if engine else 0.3
            coc_z = _sf(getattr(engine, "coc_instability_z", None), 0.0) if engine else 0.0

            signals = stack_engine.score_from_master_df(
                frob_distortion_z=frob_z,
                market_mode_share=mode_share,
                coc_instability_z=coc_z,
                master_df=master_df,
            )

            summary = signal_stack_summary(signals)
            report["signal_stack"] = {
                "n_total": len(signals),
                "n_passing": sum(1 for s in signals if s.passes_entry),
                "top_conviction": round(signals[0].conviction_score, 4) if signals else 0,
                "distortion_score": round(signals[0].distortion_score, 4) if signals else 0,
                "top_candidates": summary.get("top_candidates", [])[:5],
            }

            log.info(
                "  Signal Stack: %d total, %d passing, top=%.4f",
                len(signals),
                report["signal_stack"]["n_passing"],
                report["signal_stack"]["top_conviction"],
            )
        except Exception as exc:
            log.exception("Signal Stack failed")
            report["errors"].append(f"Signal Stack: {exc}")
    else:
        log.warning("[4/6] Signal Stack not available — skipping")

    # ── שלב 5: Stress Tests ─────────────────────────────────────────────────
    if _IMPORTS_OK.get("stress") and master_df is not None:
        log.info("[5/6] Running Stress Tests...")
        try:
            stress_results = StressEngine().run_all(master_df, settings)
            if stress_results:
                worst = stress_results[0]
                report["stress_summary"] = {
                    "n_scenarios": len(stress_results),
                    "worst_scenario": worst.scenario_name,
                    "worst_pnl_pct": round(_sf(worst.portfolio_pnl_estimate) * 100, 2),
                    "worst_reliability": round(_sf(getattr(worst, "signal_reliability_score", 0)), 4),
                }
                log.info(
                    "  Stress: worst=%s (%.1f%%)",
                    worst.scenario_name,
                    _sf(worst.portfolio_pnl_estimate) * 100,
                )
        except Exception as exc:
            log.exception("Stress Tests failed")
            report["errors"].append(f"Stress: {exc}")
    else:
        log.warning("[5/6] Stress Tests not available — skipping")

    # ── שלב 6: Methodology Lab — all strategies comparison ──────────────────
    if _IMPORTS_OK.get("methodology_lab") and engine is not None and engine.prices is not None:
        log.info("[6/8] Running Methodology Lab (all strategies)...")
        try:
            lab = MethodologyLab(engine.prices, settings, step=10)
            lab_results = lab.run_all()
            lab.save_results()
            # Summarize
            sorted_methods = sorted(lab_results.values(), key=lambda r: r.sharpe, reverse=True)
            report["methodology_lab"] = {
                "n_methodologies": len(sorted_methods),
                "best_name": sorted_methods[0].name if sorted_methods else "N/A",
                "best_sharpe": sorted_methods[0].sharpe if sorted_methods else 0,
                "ranking": [
                    {"name": m.name, "sharpe": m.sharpe, "win_rate": m.win_rate,
                     "total_pnl": m.total_pnl, "total_trades": m.total_trades}
                    for m in sorted_methods[:8]
                ],
            }
            log.info("  Lab: %d strategies, best=%s (Sharpe=%.3f)",
                     len(sorted_methods), sorted_methods[0].name if sorted_methods else "N/A",
                     sorted_methods[0].sharpe if sorted_methods else 0)
        except Exception as exc:
            log.exception("Methodology Lab failed")
            report["errors"].append(f"Methodology Lab: {exc}")
    else:
        log.warning("[6/8] Methodology Lab not available — skipping")

    # ── שלב 7: Tail Risk — ES + Correlation Stress ──────────────────────────
    if _IMPORTS_OK.get("tail_risk") and engine is not None and engine.prices is not None:
        log.info("[7/8] Computing tail risk metrics...")
        try:
            log_rets = np.log(engine.prices / engine.prices.shift(1)).dropna()
            _sectors_tr = settings.sector_list()
            eq_weights = {s: 1.0/len(_sectors_tr) for s in _sectors_tr if s in log_rets.columns}

            es = compute_expected_shortfall(log_rets, eq_weights, confidence=0.975)
            corr_stress = parametric_correlation_stress(log_rets, eq_weights)
            tail_diag = tail_correlation_diagnostic(log_rets, _sectors_tr)

            report["tail_risk"] = {
                "es_97_5_1d": round(es.es_pct, 6),
                "var_97_5_1d": round(es.var_pct, 6),
                "es_var_ratio": round(es.es_to_var_ratio, 4),
                "skewness": es.skewness,
                "kurtosis": es.kurtosis,
                "corr_stress_panic_vol_increase": round(corr_stress[-1].vol_increase_pct, 4) if corr_stress else 0,
                "corr_stress_panic_disp_change": round(corr_stress[-1].dispersion_change_pct, 4) if corr_stress else 0,
                "tail_corr": tail_diag.get("tail_corr", 0),
                "tail_ratio": tail_diag.get("tail_ratio", 1.0),
                "panic_coupling": tail_diag.get("panic_coupling", False),
            }
            log.info("  ES: %.4f%%, ES/VaR=%.2f, skew=%.2f, kurt=%.1f, tail_ratio=%.2f",
                     es.es_pct * 100, es.es_to_var_ratio, es.skewness, es.kurtosis,
                     tail_diag.get("tail_ratio", 1.0))
        except Exception as exc:
            log.exception("Tail risk computation failed")
            report["errors"].append(f"Tail Risk: {exc}")
    else:
        log.warning("[7/8] Tail risk not available — skipping")

    # ── שלב 8: ניתוח מסקנות והמלצות ────────────────────────────────────────
    log.info("[8/8] Generating conclusions and recommendations...")
    try:
        _generate_conclusions(report, portfolio)
    except Exception as exc:
        log.exception("Conclusion generation failed")
        report["errors"].append(f"Conclusions: {exc}")

    # ── סיום: שמירה, פרסום, עדכון portfolio ──────────────────────────────────
    finished_at = datetime.now(timezone.utc)
    report["finished_at"] = finished_at.isoformat()
    report["elapsed_seconds"] = round(
        (finished_at - started_at).total_seconds(), 1
    )

    # שמירת דוח JSON
    report_path = REPORTS_DIR / f"{today}.json"
    try:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Report saved: %s", report_path)
    except Exception as exc:
        log.exception("Failed to save report")

    # פרסום ל-AgentBus
    if _IMPORTS_OK.get("agent_bus"):
        try:
            bus = get_bus()
            bus_payload = {
                "status": "ok" if not report["errors"] else "partial",
                "run_date": today,
                "metrics": report["metrics"],
                "signal_stack_passing": report.get("signal_stack", {}).get("n_passing", 0),
                "signal_stack_top": report.get("signal_stack", {}).get("top_conviction", 0),
                "conclusions_count": len(report["conclusions"]),
                "errors_count": len(report["errors"]),
            }
            bus.publish("agent_methodology", bus_payload)
            log.info("Published to AgentBus")
        except Exception as exc:
            log.exception("AgentBus publish failed")

    # עדכון portfolio tracker
    try:
        portfolio.add_run(report)
        log.info("Portfolio tracker updated")
    except Exception as exc:
        log.exception("Portfolio tracker update failed")

    # עדכון registry — סיום
    if _IMPORTS_OK.get("registry"):
        status = AgentStatus.COMPLETED if not report["errors"] else AgentStatus.FAILED
        registry = get_registry()
        registry.heartbeat("agent_methodology", status,
                           error="; ".join(report["errors"][:3]) if report["errors"] else None)

    log.info(
        "Methodology Agent completed in %.1fs — %d conclusions, %d errors",
        report["elapsed_seconds"],
        len(report["conclusions"]),
        len(report["errors"]),
    )
    return report


# =============================================================================
# מנוע מסקנות והמלצות
# =============================================================================

def _generate_conclusions(report: dict, portfolio: MethodologyPortfolio) -> None:
    """
    מייצר מסקנות אוטומטיות מהשוואת ריצה נוכחית מול היסטוריה.

    מוסיף ל-report["conclusions"] ו-report["recommendations"].
    """
    conclusions: List[str] = []
    recommendations: List[str] = []

    metrics = report.get("metrics", {})

    # ── השוואה מול ריצה קודמת ────────────────────────────────────────────────
    comparison = portfolio.compare_vs_previous(report)
    if comparison.get("has_previous"):
        deltas = comparison.get("deltas", {})

        # IC trend
        ic_delta = deltas.get("ic_mean", {})
        if ic_delta:
            old_ic = ic_delta["old"]
            new_ic = ic_delta["new"]
            pct = _pct_delta(old_ic, new_ic)
            conclusions.append(
                f"IC changed from {old_ic:.4f} to {new_ic:.4f} ({pct})"
            )

        # Sharpe trend
        sharpe_delta = deltas.get("sharpe", {})
        if sharpe_delta:
            old_s = sharpe_delta["old"]
            new_s = sharpe_delta["new"]
            pct = _pct_delta(old_s, new_s)
            conclusions.append(
                f"Sharpe changed from {old_s:.2f} to {new_s:.2f} ({pct})"
            )
            if new_s < 0.5 and old_s >= 0.5:
                recommendations.append(
                    "Sharpe dropped below 0.5 — consider reducing position sizes"
                )

        # Hit rate trend
        hr_delta = deltas.get("hit_rate", {})
        if hr_delta:
            old_hr = hr_delta["old"]
            new_hr = hr_delta["new"]
            pct = _pct_delta(old_hr, new_hr)
            conclusions.append(
                f"Hit rate changed from {old_hr:.1%} to {new_hr:.1%} ({pct})"
            )
            if new_hr < 0.50:
                recommendations.append(
                    "Hit rate below 50% — signal quality degraded, review entry thresholds"
                )

        # Max drawdown
        dd_delta = deltas.get("max_dd", {})
        if dd_delta:
            old_dd = dd_delta["old"]
            new_dd = dd_delta["new"]
            if new_dd > old_dd * 1.25:
                conclusions.append(
                    f"Max drawdown increased from {old_dd:.1%} to {new_dd:.1%} — risk elevated"
                )
                recommendations.append(
                    "Drawdown expanding — tighten stop-losses and review regime gates"
                )

    # ── ניתוח רג'ים חלש ──────────────────────────────────────────────────────
    regime_breakdown = report.get("regime_breakdown", {})
    for regime, data in regime_breakdown.items():
        if isinstance(data, dict):
            hr = data.get("hit_rate", 1.0)
            if hr < 0.45:
                conclusions.append(
                    f"{regime} regime hit_rate={hr:.1%} — below threshold"
                )

    # ── מגמות ארוכות טווח ────────────────────────────────────────────────────
    for metric_name in ("ic_mean", "sharpe", "hit_rate"):
        trend = portfolio.get_trend(metric_name, n=10)
        direction = trend.get("direction", "insufficient_data")
        if direction == "declining":
            delta_pct = trend.get("delta_pct", 0)
            conclusions.append(
                f"{metric_name} declining over last 10 runs ({delta_pct:+.1f}%)"
            )
            recommendations.append(
                f"Persistent {metric_name} decline — investigate parameter drift or regime shift"
            )
        elif direction == "improving":
            delta_pct = trend.get("delta_pct", 0)
            conclusions.append(
                f"{metric_name} improving over last 10 runs ({delta_pct:+.1f}%)"
            )

    # ── בדיקת רג'ים חלש בהיסטוריה ───────────────────────────────────────────
    worst = portfolio.worst_regime()
    if worst and worst.get("consecutive_below_50", 0) >= 3:
        regime = worst["regime"]
        consec = worst["consecutive_below_50"]
        avg_hr = worst["avg_hit_rate"]
        conclusions.append(
            f"{regime} regime consistently weakest: {consec} consecutive runs below 50% "
            f"(avg hit_rate={avg_hr:.1%})"
        )
        recommendations.append(
            f"Consider regime-specific parameter tuning for {regime} or disabling entries in that regime"
        )

    # ── Signal Stack המלצות ──────────────────────────────────────────────────
    signal_stack = report.get("signal_stack", {})
    n_passing = signal_stack.get("n_passing", 0)
    n_total = signal_stack.get("n_total", 0)
    if n_total > 0 and n_passing == 0:
        conclusions.append("No signals passing entry threshold — all gates blocked")
        recommendations.append("Review regime safety score and entry threshold calibration")
    elif n_total > 0:
        pass_rate = n_passing / n_total
        if pass_rate > 0.7:
            conclusions.append(
                f"High signal pass rate ({pass_rate:.0%}) — possible threshold too loose"
            )
            recommendations.append("Consider tightening entry_threshold to improve selectivity")

    # ── Stress recommendations ───────────────────────────────────────────────
    stress = report.get("stress_summary", {})
    worst_pnl = stress.get("worst_pnl_pct", 0)
    if worst_pnl < -5.0:
        worst_name = stress.get("worst_scenario", "?")
        conclusions.append(
            f"Worst stress scenario '{worst_name}' shows {worst_pnl:.1f}% loss"
        )
        recommendations.append("Stress loss exceeds 5% — review tail hedging or position sizing")

    # ── כתיבה לדוח ───────────────────────────────────────────────────────────
    if not conclusions:
        conclusions.append("No significant changes detected — methodology performing within norms")

    report["conclusions"] = conclusions
    report["recommendations"] = recommendations


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRV Methodology Agent")
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Run once (default behavior)",
    )
    args = parser.parse_args()

    result = run(once=args.once)

    # סיכום מהיר ל-stdout
    print("\n" + "=" * 60)
    print(f"Methodology Report — {result['run_date']}")
    print("=" * 60)

    m = result.get("metrics", {})
    if m:
        print(f"  IC Mean:     {m.get('ic_mean', 'N/A')}")
        print(f"  IC IR:       {m.get('ic_ir', 'N/A')}")
        print(f"  Sharpe:      {m.get('sharpe', 'N/A')}")
        print(f"  Hit Rate:    {m.get('hit_rate', 'N/A')}")
        print(f"  Max DD:      {m.get('max_dd', 'N/A')}")

    ss = result.get("signal_stack", {})
    if ss:
        print(f"\n  Signals:     {ss.get('n_passing', 0)}/{ss.get('n_total', 0)} passing")
        print(f"  Top Conv:    {ss.get('top_conviction', 0)}")

    if result.get("conclusions"):
        print("\n  Conclusions:")
        for c in result["conclusions"]:
            print(f"    - {c}")

    if result.get("recommendations"):
        print("\n  Recommendations:")
        for r in result["recommendations"]:
            print(f"    - {r}")

    if result.get("errors"):
        print(f"\n  Errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"    ! {e}")

    print(f"\n  Elapsed: {result.get('elapsed_seconds', 0)}s")

    sys.exit(0)
