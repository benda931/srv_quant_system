"""
scripts/agent_daily_pipeline.py
--------------------------------
סוכן 1: Pipeline יומי אוטומטי

ROLE:    מרענן נתונים, מחשב master_df, stress, risk, ומייצר Daily Brief.
         מפרסם תוצאות ל-AgentBus. אם יש vol breach — שולח alert ל-Claude.

INPUTS:  FMP REST API (prices, holdings, fundamentals)
         data_lake/parquet/ cache

OUTPUTS: data_lake/parquet/prices.parquet (מעודכן)
         reports/output/<ts>_brief.txt + .json
         logs/pipeline_history.jsonl
         → agent_bus["agent_daily_pipeline"]

הרצה: python scripts/agent_daily_pipeline.py
       python scripts/agent_daily_pipeline.py --force-refresh
       python scripts/agent_daily_pipeline.py --no-claude  # ללא alert loop
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.pipeline import DataLakeManager
from data_ops.orchestrator import DataOrchestrator
from analytics.stat_arb import QuantEngine
from analytics.stress import StressEngine
from analytics.portfolio_risk import PortfolioRiskEngine
from reports.daily_report import DailyBriefGenerator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "pipeline_runs.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_daily_pipeline")


def run(force_refresh: bool = False) -> dict:
    """Run the full daily pipeline. Returns a status dict."""
    started_at = datetime.now(timezone.utc)
    status: dict = {"started_at": started_at.isoformat(), "steps": {}}

    settings = get_settings()
    log.info("=" * 60)
    log.info("Daily Pipeline — %s", started_at.strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 60)

    # Step 1 ── Data refresh
    log.info("[1/5] Data refresh (force=%s)...", force_refresh)
    try:
        orchestrator = DataOrchestrator(settings)
        data_state = orchestrator.run(force_refresh=force_refresh)
        health_label = data_state.health.health_label
        status["steps"]["data"] = {"status": "ok", "health": health_label}
        log.info("  → Data: %s", health_label)
    except Exception as e:
        log.exception("Data refresh failed")
        status["steps"]["data"] = {"status": "error", "error": str(e)}
        return _finish(status, started_at, success=False)

    # Step 2 ── QuantEngine
    log.info("[2/5] Running QuantEngine...")
    try:
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()
        n_longs  = (master_df["direction"] == "LONG").sum()  if "direction" in master_df.columns else 0
        n_shorts = (master_df["direction"] == "SHORT").sum() if "direction" in master_df.columns else 0
        status["steps"]["engine"] = {"status": "ok", "longs": int(n_longs), "shorts": int(n_shorts)}
        log.info("  → Engine: %d LONG, %d SHORT", n_longs, n_shorts)
    except Exception as e:
        log.exception("QuantEngine failed")
        status["steps"]["engine"] = {"status": "error", "error": str(e)}
        return _finish(status, started_at, success=False)

    # Step 3 ── Stress Tests
    log.info("[3/5] Running Stress Tests...")
    stress_results = None
    try:
        stress_results = StressEngine().run_all(master_df, settings)
        worst = stress_results[0]
        status["steps"]["stress"] = {
            "status": "ok",
            "worst_scenario": worst.scenario_name,
            "worst_pnl_pct": round(worst.portfolio_pnl_estimate * 100, 2),
        }
        log.info("  → Stress: worst=%s (%.1f%%)", worst.scenario_name, worst.portfolio_pnl_estimate * 100)
    except Exception as e:
        log.exception("Stress Tests failed")
        status["steps"]["stress"] = {"status": "error", "error": str(e)}

    # Step 4 ── Portfolio Risk
    log.info("[4/5] Computing Portfolio Risk...")
    risk_report = None
    try:
        prices_df = data_state.artifacts.prices
        weights = {
            row["sector_ticker"]: float(row["w_final"])
            for _, row in master_df.iterrows()
            if row.get("direction") in ("LONG", "SHORT")
        }
        if weights:
            risk_report = PortfolioRiskEngine().full_risk_report(weights, prices_df, settings)
            status["steps"]["risk"] = {
                "status": "ok",
                "vol_ann_pct": round((risk_report.portfolio_vol_ann or 0) * 100, 2),
                "var_95_pct":  round((risk_report.var_95_1d  or 0) * 100, 3),
                "vol_breach":  risk_report.vol_target_breach,
            }
            log.info(
                "  → Risk: vol=%.1f%%, VaR=%.2f%%, breach=%s",
                (risk_report.portfolio_vol_ann or 0) * 100,
                (risk_report.var_95_1d or 0) * 100,
                risk_report.vol_target_breach,
            )
        else:
            status["steps"]["risk"] = {"status": "skipped", "reason": "no tradable positions"}
    except Exception as e:
        log.exception("Portfolio Risk failed")
        status["steps"]["risk"] = {"status": "error", "error": str(e)}

    # Step 5 ── Daily Brief
    log.info("[5/5] Generating Daily Brief...")
    try:
        out_dir = settings.project_root / "reports" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        from data_ops.journal import open_journal
        journal_db = settings.project_root / "data" / "pm_journal.db"
        journal = open_journal(journal_db)

        gen = DailyBriefGenerator()
        txt_path, json_path = gen.generate(
            master_df=master_df,
            risk_report=risk_report,
            stress_results=stress_results,
            journal=journal,
            settings=settings,
            output_dir=out_dir,
        )
        status["steps"]["brief"] = {"status": "ok", "txt": str(txt_path), "json": str(json_path)}
        log.info("  → Brief: %s", txt_path.name)
    except Exception as e:
        log.exception("Daily Brief failed")
        status["steps"]["brief"] = {"status": "error", "error": str(e)}

    return _finish(status, started_at, success=True)


def _finish(status: dict, started_at: datetime, success: bool) -> dict:
    finished_at = datetime.now(timezone.utc)
    elapsed = (finished_at - started_at).total_seconds()
    status["finished_at"] = finished_at.isoformat()
    status["elapsed_seconds"] = round(elapsed, 1)
    status["success"] = success

    # Append to run history
    run_log = ROOT / "logs" / "pipeline_history.jsonl"
    try:
        with open(run_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(status) + "\n")
    except Exception:
        pass

    log.info("Pipeline %s in %.1fs", "completed" if success else "FAILED", elapsed)
    return status


_ALERT_SYSTEM_PROMPT = """אתה Pipeline Alert AI של מערכת SRV Quantamental DSS.
קיבלת דוח pipeline יומי עם breach. תפקידך לנתח את הבעיה ולתת המלצות ל-PM.
ענה בצורה תמציתית ומעשית. אל תריץ פעולות — רק ניתוח והמלצה.
שלח done לסיום."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRV Daily Pipeline Agent")
    parser.add_argument("--force-refresh", action="store_true", help="Force FMP data refresh regardless of cache")
    parser.add_argument("--no-claude",     action="store_true", help="Skip Claude alert loop even on breach")
    args = parser.parse_args()

    result = run(force_refresh=args.force_refresh)

    # ── Publish to AgentBus ───────────────────────────────────────────────────
    from scripts.agent_bus import get_bus
    bus = get_bus()
    bus.publish("agent_daily_pipeline", {
        "status": "ok" if result.get("success") else "failed",
        "success": result.get("success"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "steps": result.get("steps", {}),
    })

    # ── Claude alert on critical events ───────────────────────────────────────
    risk_step = result.get("steps", {}).get("risk", {})
    vol_breach = risk_step.get("vol_breach", False)
    if vol_breach and not args.no_claude:
        import json as _json
        alert_msg = (
            "## Pipeline Alert — Vol Breach Detected\n\n"
            f"Pipeline completed with a volatility breach.\n\n"
            f"```json\n{_json.dumps(result, ensure_ascii=False, indent=2, default=str)}\n```\n\n"
            "אנא נתח את ה-breach ותן המלצות לPM."
        )
        from scripts.claude_loop import run_agent_loop
        run_agent_loop(
            agent_name="pipeline_alert",
            system_prompt=_ALERT_SYSTEM_PROMPT,
            initial_message=alert_msg,
            max_turns=3,
        )

    sys.exit(0 if result.get("success") else 1)
