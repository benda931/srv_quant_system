"""
scripts/agent_improve_system.py
---------------------------------
סוכן 2: שיפור אוטומטי של המערכת על-פי תוצאות הבאקטסט

ROLE:    מנתח תוצאות walk-forward backtest, מזהה חולשות (IC נמוך, Hit Rate ירוד
         לפי משטר), שולח ניתוח מלא ל-Claude API, ו-Claude מחזיר פקודות
         לשיפור פרמטרים — עם backup אוטומטי ורברט אם tests נכשלים.

INPUTS:  logs/last_backtest.json  (cache)
         config/settings.py       (לקריאה ועריכה)
         analytics/stat_arb.py    (לקריאה)

OUTPUTS: logs/agent_improve_system_<ts>.json  (loop log)
         logs/last_backtest.json              (cache עדכני)
         scripts/improvement_prompts/<ts>.txt (preview בלבד)
         → agent_bus["agent_improve_system"]

הרצה:
  python scripts/agent_improve_system.py --run-backtest
  python scripts/agent_improve_system.py                   # משתמש בcache
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "improve_system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_improve_system")


# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """אתה System Improvement AI של מערכת SRV Quantamental DSS.

תפקידך: לנתח תוצאות backtest, לזהות חולשות אמיתיות, ולשפר פרמטרים
בצורה מתמטית מוצדקת — תוך שמירה קפדנית על כל הinterfaces הקיימים.

**כלים זמינים (שלח כ-JSON actions):**
```json
{"actions": [
  {"type": "read_file",  "file": "config/settings.py"},
  {"type": "read_file",  "file": "analytics/stat_arb.py"},
  {"type": "edit_param", "file": "config/settings.py", "param": "pca_window", "value": 180},
  {"type": "run_tests",  "path": "tests/"},
  {"type": "log",        "message": "הסבר מתמטי לשינוי"},
  {"type": "done",       "summary": "סיכום: מה שונה ולמה"}
]}
```

**כללי ברזל:**
1. לפני כל edit_param — קרא את הקובץ עם read_file
2. כל שינוי פרמטר עובר tests אוטומטית (backup+revert מובנה)
3. שנה מקסימום 2-3 פרמטרים בסשן אחד
4. הסבר מתמטי חובה לכל שינוי
5. אל תשנה ממשקי פונקציות — רק ערכים ב-settings.py
6. IC improvement in the weakest regime is the primary objective"""


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest() -> dict:
    """Run walk-forward backtest and return results as a serializable dict."""
    from config.settings import get_settings
    from data_ops.orchestrator import DataOrchestrator
    from analytics.backtest import WalkForwardBacktester

    settings = get_settings()
    log.info("Loading data for backtest...")
    orchestrator = DataOrchestrator(settings)
    data_state = orchestrator.run(force_refresh=False)
    prices_df = data_state.artifacts.prices

    log.info("Running walk-forward backtest (this may take 2-4 minutes)...")
    bt = WalkForwardBacktester(settings)
    result = bt.run_backtest(prices_df, prices_df, prices_df)

    regime_bd = result.regime_breakdown
    regime_dict: dict = {}
    for reg in ["calm", "normal", "tension", "crisis"]:
        rd = getattr(regime_bd, reg, None)
        if rd:
            regime_dict[reg.upper()] = {
                "ic_mean":  round(float(rd.ic_mean  or 0), 4),
                "hit_rate": round(float(rd.hit_rate or 0), 4),
                "sharpe":   round(float(rd.sharpe   or 0), 4),
                "n_walks":  int(rd.n_walks or 0),
            }

    return {
        "ic_mean":      round(float(result.ic_mean      or 0), 4),
        "ic_ir":        round(float(result.ic_ir         or 0), 4),
        "hit_rate":     round(float(result.hit_rate      or 0), 4),
        "sharpe":       round(float(result.sharpe        or 0), 4),
        "max_drawdown": round(float(result.max_drawdown  or 0), 4),
        "n_walks":      int(result.n_walks or 0),
        "n_sectors":    int(result.n_sectors or 0),
        "regime_breakdown": regime_dict,
    }


def build_improvement_message(bt: dict) -> str:
    """Build the initial message to Claude with backtest analysis."""
    regime_lines = []
    for reg, rd in bt.get("regime_breakdown", {}).items():
        ic   = rd.get("ic_mean", 0)
        hr   = rd.get("hit_rate", 0)
        sh   = rd.get("sharpe", 0)
        n    = rd.get("n_walks", 0)
        flag = "⚠ WEAK" if ic < 0.03 or hr < 0.5 else "✓ OK"
        regime_lines.append(f"  {reg}: IC={ic:.3f}, HitRate={hr:.1%}, Sharpe={sh:.2f}, n={n} {flag}")

    weakest = min(
        bt.get("regime_breakdown", {}).items(),
        key=lambda kv: kv[1].get("ic_mean", 0),
        default=("N/A", {}),
    )

    return f"""## Walk-Forward Backtest Results
Timestamp: {datetime.now(timezone.utc).isoformat()}

### Overall Performance
- IC Mean:       {bt['ic_mean']:.4f}  (target: >0.05)
- IC IR:         {bt['ic_ir']:.3f}   (target: >0.5)
- Hit Rate:      {bt['hit_rate']:.1%} (target: >55%)
- Sharpe:        {bt['sharpe']:.3f}  (target: >1.0)
- Max Drawdown:  {bt['max_drawdown']:.1%}
- Walk windows:  {bt['n_walks']}

### Regime Breakdown
{chr(10).join(regime_lines)}

### Weakest Regime: **{weakest[0]}**

---
**Task:** אנא שפר את המערכת:
1. קרא תחילה את config/settings.py ואת analytics/stat_arb.py
2. זהה את 2-3 הפרמטרים הכי משפיעים על המשטר החלש ({weakest[0]})
3. שנה אותם עם edit_param (tests יורצו אוטומטית, revert אוטומטי אם נכשל)
4. דווח על מה שונה ולמה"""


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRV System Improvement Agent")
    parser.add_argument("--run-backtest", action="store_true", help="Run backtest before improving")
    parser.add_argument("--no-claude",   action="store_true", help="Print analysis only, no AI loop")
    args = parser.parse_args()

    # ── Load or run backtest ──────────────────────────────────────────────────
    cache_path = ROOT / "logs" / "last_backtest.json"
    if args.run_backtest or not cache_path.exists():
        log.info("Running backtest...")
        bt_results = run_backtest()
        cache_path.write_text(json.dumps(bt_results, indent=2), encoding="utf-8")
        log.info("Backtest cached → %s", cache_path)
    else:
        log.info("Loading cached backtest from %s", cache_path)
        bt_results = json.loads(cache_path.read_text(encoding="utf-8"))

    log.info("Backtest summary: IC=%.3f, HitRate=%.1f%%, Sharpe=%.2f",
             bt_results["ic_mean"], bt_results["hit_rate"] * 100, bt_results["sharpe"])

    # ── Publish to bus ────────────────────────────────────────────────────────
    from scripts.agent_bus import get_bus
    bus = get_bus()
    bus.publish("agent_improve_system", {
        "status": "running",
        "bt_ic": bt_results["ic_mean"],
        "bt_sharpe": bt_results["sharpe"],
        "bt_hit_rate": bt_results["hit_rate"],
    })

    initial_msg = build_improvement_message(bt_results)

    if args.no_claude:
        print(initial_msg)
        sys.exit(0)

    # ── Save prompt preview ───────────────────────────────────────────────────
    prompt_dir = ROOT / "scripts" / "improvement_prompts"
    prompt_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (prompt_dir / f"improve_{ts}.txt").write_text(initial_msg, encoding="utf-8")

    # ── Run Claude feedback loop ──────────────────────────────────────────────
    from scripts.claude_loop import run_agent_loop
    result = run_agent_loop(
        agent_name="improve_system",
        system_prompt=SYSTEM_PROMPT,
        initial_message=initial_msg,
        max_turns=8,
    )

    # ── Update bus with outcome ───────────────────────────────────────────────
    bus.publish("agent_improve_system", {
        "status": "completed",
        "turns": result["turns"],
        "bt_ic": bt_results["ic_mean"],
        "bt_sharpe": bt_results["sharpe"],
    })
    log.info("Improvement agent completed in %d turns.", result["turns"])
