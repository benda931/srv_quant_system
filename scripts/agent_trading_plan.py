"""
scripts/agent_trading_plan.py
------------------------------
סוכן 5: תוכנית מסחר — Master Orchestrator

ROLE:    נקודת הכניסה המרכזית. אוסף state מלא, קורא את AgentBus לסנאפשוט
         מכל הסוכנים, שולח ל-Claude תמונה כוללת, ו-Claude מחזיר תוכנית מסחר
         + פקודות להפעלת סוכנים נוספים.

INPUTS:  QuantEngine state (master_df, signals, regime)
         StressEngine / PortfolioRiskEngine results
         agent_bus (סנאפשוט של כל הסוכנים)
         logs/last_backtest.json
         data/pm_journal.db

OUTPUTS: תוכנית מסחר מלאה ב-log
         → agent_bus["agent_trading_plan"]
         → מפעיל סוכנים אחרים לפי הצורך

מרכז את כל המידע ממנועי המערכת, מגבש תוכנית מסחר מלאה,
ומפעיל את הסוכנים האחרים לפי הצורך.

הסוכן:
  1. אוסף state מלא: master_df, stress, risk, backtest cache, journal
  2. שולח ל-Claude תמונה כוללת
  3. Claude מחזיר תוכנית מסחר מובנית + פקודות לסוכנים
  4. מבצע את הפקודות (מריץ optimizer, pipeline, improve וכו')
  5. ממשיך בלולאה עד "done"

הרצה:
  python scripts/agent_trading_plan.py
  python scripts/agent_trading_plan.py --full   # כולל הרצת optimizer
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
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
        logging.FileHandler(ROOT / "logs" / "trading_plan.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_trading_plan")


# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """אתה Trading Plan AI של מערכת SRV Quantamental DSS.

תפקידך: לנתח את מצב המערכת המלא, לגבש תוכנית מסחר ברורה ל-PM,
ולהפעיל את הסוכנים האחרים לפי הצורך.

**סוכנים זמינים שאתה יכול להפעיל:**
```json
{"type": "run_script", "script": "scripts/agent_daily_pipeline.py", "args": ["--force-refresh"]}
{"type": "run_script", "script": "scripts/agent_portfolio_optimizer.py", "args": ["--no-claude", "--objective", "max_sharpe"]}
{"type": "run_script", "script": "scripts/agent_improve_system.py", "args": ["--run-backtest"]}
{"type": "run_script", "script": "scripts/agent_daily_pipeline.py"}
```

**תוכנית מסחר — פורמט נדרש בתשובה:**
כאשר אתה מגבש תוכנית, כלול:
1. סיכום משטר (1-2 שורות)
2. רשימת עסקאות מוצעות (sector, כיוון, גודל, נימוק)
3. רמות סיכון ועצירה
4. איזה סוכנים להפעיל ובאיזה סדר

**כלל ברזל:** כל שינוי פרמטר עובר tests. לא משנים ממשקים קיימים.
לא מבצעים trades בפועל — זו מערכת DSS בלבד."""


# ─────────────────────────────────────────────────────────────────────────────
def _collect_system_state(full: bool = False) -> dict:
    """Collect the complete system state into a single dict."""
    state: dict = {"collected_at": datetime.now(timezone.utc).isoformat()}

    # ── Master DF ─────────────────────────────────────────────────────────────
    log.info("Loading QuantEngine state...")
    try:
        from config.settings import get_settings
        from data_ops.orchestrator import DataOrchestrator
        from analytics.stat_arb import QuantEngine

        settings = get_settings()
        orchestrator = DataOrchestrator(settings)
        data_state = orchestrator.run(force_refresh=False)
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()

        row0 = master_df.iloc[0].to_dict() if len(master_df) else {}

        def _sf(x):
            try:
                v = float(x)
                return round(v, 4) if v == v else None
            except Exception:
                return None

        state["regime"] = {
            "market_state":           str(row0.get("market_state", "?")),
            "avg_corr":               _sf(row0.get("avg_corr_t")),
            "mode_strength":          _sf(row0.get("market_mode_strength")),
            "corr_distortion":        _sf(row0.get("corr_matrix_dist_t")),
            "transition_probability": _sf(row0.get("transition_probability")),
            "crisis_probability":     _sf(row0.get("crisis_probability")),
            "execution_regime":       str(row0.get("execution_regime", "?")),
            "regime_alert":           str(row0.get("regime_alert", "")),
        }

        signals = []
        for _, r in master_df.iterrows():
            if r.get("direction") in ("LONG", "SHORT"):
                signals.append({
                    "sector":     str(r.get("sector_ticker", "?")),
                    "name":       str(r.get("sector_name", "?")),
                    "direction":  str(r.get("direction")),
                    "mc":         _sf(r.get("mc_score")),
                    "conviction": _sf(r.get("conviction_score")),
                    "z_score":    _sf(r.get("pca_residual_z")),
                    "rel_pe":     _sf(r.get("rel_pe_vs_spy")),
                    "w_final":    _sf(r.get("w_final")),
                    "decision":   str(r.get("decision_label", "?")),
                    "sds":        _sf(r.get("sds_score")),
                    "fjs":        _sf(r.get("fjs_score")),
                    "mss":        _sf(r.get("mss_score")),
                    "pm_note":    str(r.get("pm_note", ""))[:80],
                })
        state["signals"] = signals
        state["n_longs"]  = sum(1 for s in signals if s["direction"] == "LONG")
        state["n_shorts"] = sum(1 for s in signals if s["direction"] == "SHORT")
        state["data_health"] = data_state.health.health_label

        log.info("State loaded: %d signals (%d L / %d S), regime=%s",
                 len(signals), state["n_longs"], state["n_shorts"], state["regime"]["market_state"])

    except Exception as e:
        log.exception("Failed to load master_df")
        state["engine_error"] = str(e)
        master_df = None
        settings = None
        data_state = None

    # ── Stress Tests ─────────────────────────────────────────────────────────
    if master_df is not None:
        try:
            from analytics.stress import StressEngine
            stress = StressEngine().run_all(master_df, settings)
            state["stress"] = {
                "worst": {"scenario": stress[0].scenario_name, "pnl_pct": round(stress[0].portfolio_pnl_estimate * 100, 2)},
                "best":  {"scenario": stress[-1].scenario_name, "pnl_pct": round(stress[-1].portfolio_pnl_estimate * 100, 2)},
                "n_negative": sum(1 for r in stress if r.portfolio_pnl_estimate < 0),
                "scenarios_gt_minus5pct": [r.scenario_name for r in stress if r.portfolio_pnl_estimate < -0.05],
            }
        except Exception as e:
            state["stress_error"] = str(e)

    # ── Portfolio Risk ────────────────────────────────────────────────────────
    if master_df is not None and data_state is not None:
        try:
            from analytics.portfolio_risk import PortfolioRiskEngine
            prices_df = data_state.artifacts.prices
            weights = {
                row["sector_ticker"]: float(row["w_final"])
                for _, row in master_df.iterrows()
                if row.get("direction") in ("LONG", "SHORT")
            }
            if weights:
                rr = PortfolioRiskEngine().full_risk_report(weights, prices_df, settings)
                state["risk"] = {
                    "vol_ann_pct":    round((rr.portfolio_vol_ann or 0) * 100, 2),
                    "var_95_1d_pct":  round((rr.var_95_1d          or 0) * 100, 3),
                    "cvar_95_1d_pct": round((rr.cvar_95_1d         or 0) * 100, 3),
                    "hhi":            round(rr.concentration_hhi   or 0, 4),
                    "vol_breach":     bool(rr.vol_target_breach),
                    "weight_breach":  bool(rr.max_weight_breach),
                }
        except Exception as e:
            state["risk_error"] = str(e)

    # ── Cached Backtest ────────────────────────────────────────────────────────
    bt_cache = ROOT / "logs" / "last_backtest.json"
    if bt_cache.exists():
        try:
            state["backtest"] = json.loads(bt_cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ── Journal stats ─────────────────────────────────────────────────────────
    try:
        from data_ops.journal import open_journal
        journal_db = ROOT / "data" / "pm_journal.db"
        if journal_db.exists():
            j = open_journal(journal_db)
            stats = j.get_stats()
            state["journal"] = {
                "decisions_30d":  int(stats.get("decisions_30d",  0)),
                "override_rate":  round(float(stats.get("override_rate", 0)), 3),
                "pm_accuracy":    round(float(stats.get("pm_accuracy",   0)), 3),
            }
    except Exception:
        pass

    # ── Correlation Volatility Analysis ──────────────────────────────────────
    if master_df is not None and settings is not None:
        try:
            from analytics.correlation_engine import CorrVolEngine, corr_vol_summary
            engine = QuantEngine(settings)
            engine.load()
            engine.calculate_conviction_score()   # populates corr_metrics + dispersion_df
            corr_analysis = CorrVolEngine().run(engine, master_df, settings)
            state["corr_vol"] = corr_vol_summary(corr_analysis)
            log.info(
                "Corr-Vol: implied=%.3f, short_vol_score=%.0f (%s)",
                corr_analysis.implied_corr,
                corr_analysis.short_vol_score,
                corr_analysis.short_vol_label,
            )
        except Exception as e:
            state["corr_vol_error"] = str(e)

    # ── AgentBus snapshot ────────────────────────────────────────────────────
    try:
        from scripts.agent_bus import get_bus
        bus_snapshot = get_bus().all_latest()
        state["agent_bus"] = {
            name: {k: v for k, v in (entry or {}).items() if k not in ("ts",)}
            for name, entry in bus_snapshot.items()
        }
    except Exception:
        pass

    # ── Latest brief snippet ──────────────────────────────────────────────────
    import glob as _glob
    briefs = sorted(_glob.glob(str(ROOT / "reports" / "output" / "*_brief.txt")))
    if briefs:
        try:
            state["latest_brief_snippet"] = open(briefs[-1], encoding="utf-8").read()[:800]
        except Exception:
            pass

    # ── Run Optimizer if --full ────────────────────────────────────────────────
    if full and master_df is not None:
        try:
            log.info("Running portfolio optimizer (--full)...")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "agent_portfolio_optimizer.py"),
                 "--no-claude", "--objective", "max_sharpe"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                state["optimizer"] = {"status": "ok", "output": result.stdout[-1000:]}
            else:
                state["optimizer"] = {"status": "error", "stderr": result.stderr[-500:]}
        except Exception as e:
            state["optimizer"] = {"status": "error", "error": str(e)}

    return state


def _build_initial_message(state: dict) -> str:
    """Build the initial message to Claude with full system state."""
    regime = state.get("regime", {})
    signals = state.get("signals", [])
    stress  = state.get("stress",  {})
    risk    = state.get("risk",    {})
    bt      = state.get("backtest",{})

    sig_lines = "\n".join(
        f"  {s['direction']:5} {s['sector']:5} ({s['name'][:15]:15}) | "
        f"MC={s['mc'] or 0:.0f} | Z={s['z_score'] or 0:+.2f} | "
        f"PE_rel={s['rel_pe'] or 0:.2f} | w={s['w_final'] or 0:.3f} | "
        f"{s['decision']}"
        for s in signals
    )

    bt_summary = ""
    if bt:
        bt_summary = (
            f"IC={bt.get('ic_mean', '?'):.3f}, "
            f"HitRate={bt.get('hit_rate', 0):.1%}, "
            f"Sharpe={bt.get('sharpe', '?'):.2f}"
        )

    return f"""## SRV Quantamental DSS — System State Snapshot
Timestamp: {state['collected_at']}

### Market Regime
- State: **{regime.get('market_state', '?')}** | Alert: {regime.get('regime_alert', '')}
- Avg Corr: {regime.get('avg_corr', '?')} | Mode Strength: {regime.get('mode_strength', '?')}
- Transition Prob: {regime.get('transition_probability', '?')} | Crisis Prob: {regime.get('crisis_probability', '?')}
- Execution Regime: {regime.get('execution_regime', '?')}
- Data Health: {state.get('data_health', '?')}

### Current Signals ({state.get('n_longs', 0)} Long / {state.get('n_shorts', 0)} Short)
{sig_lines if sig_lines else '  (אין אותות)'}

### Stress Testing
- Worst: {stress.get('worst', {}).get('scenario', '?')} ({stress.get('worst', {}).get('pnl_pct', '?')}%)
- Best:  {stress.get('best',  {}).get('scenario', '?')} ({stress.get('best',  {}).get('pnl_pct', '?')}%)
- Negative scenarios: {stress.get('n_negative', '?')}/10
- Scenarios >-5%: {stress.get('scenarios_gt_minus5pct', [])}

### Portfolio Risk
- Vol (ann): {risk.get('vol_ann_pct', '?')}% | VaR 95%: {risk.get('var_95_1d_pct', '?')}% | CVaR: {risk.get('cvar_95_1d_pct', '?')}%
- Vol breach: {risk.get('vol_breach', '?')} | Weight breach: {risk.get('weight_breach', '?')}

### Backtest (cached)
{bt_summary if bt_summary else '(לא הורץ עדיין)'}

### Correlation & Volatility Pricing
{json.dumps(state.get('corr_vol', {}), ensure_ascii=False, indent=2)}

### PM Journal
{json.dumps(state.get('journal', {}), ensure_ascii=False)}

### Agent Bus (סנאפשוטים אחרונים)
{json.dumps(state.get('agent_bus', {}), ensure_ascii=False, indent=2)}

---
**בקשה מהסוכן:** אנא גבש תוכנית מסחר מלאה לפי המידע הנ"ל:
1. מה לעשות עכשיו? (קנה/מכור/המתן — עם גדלי פוזיציות ספציפיים)
2. אילו סקטורים הכי מעניינים ולמה? (עם Z-score, MC, PE_rel)
3. מה הסיכונים המרכזיים שצריך לנהל?
4. האם יש הזדמנות Short Vol / Dispersion? (לפי corr_vol section)
   - אם short_vol_score > 50: הסבר את ה-trade — מה למכור, כמה, ואיך לגדר
   - אם implied_corr > fair_value_corr: ציין את פרמיית הקורלציה
   - זהה אילו עיוותי זוגות הכי אטרקטיביים לexploit
5. האם להפעיל סוכנים נוספים? (optimizer, improve, pipeline)
6. האם הפרמטרים צריכים עדכון?

שלח תוכנית מובנית ואז actions לביצוע."""


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Plan Master Agent")
    parser.add_argument("--full", action="store_true", help="Also run portfolio optimizer")
    parser.add_argument("--no-claude", action="store_true", help="Print state only, no AI loop")
    args = parser.parse_args()

    log.info("Collecting system state...")
    state = _collect_system_state(full=args.full)

    if args.no_claude:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        sys.exit(0)

    initial_msg = _build_initial_message(state)

    # ── Publish start to bus ──────────────────────────────────────────────────
    from scripts.agent_bus import get_bus
    bus = get_bus()
    bus.publish("agent_trading_plan", {
        "status": "running",
        "regime": state.get("regime", {}).get("market_state", "?"),
        "n_longs": state.get("n_longs", 0),
        "n_shorts": state.get("n_shorts", 0),
    })

    from scripts.claude_loop import run_agent_loop
    result = run_agent_loop(
        agent_name="trading_plan",
        system_prompt=SYSTEM_PROMPT,
        initial_message=initial_msg,
        max_turns=10,
    )

    bus.publish("agent_trading_plan", {
        "status": "completed",
        "turns": result["turns"],
        "regime": state.get("regime", {}).get("market_state", "?"),
        "n_longs": state.get("n_longs", 0),
        "n_shorts": state.get("n_shorts", 0),
    })
