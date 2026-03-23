"""
agents/optimizer/agent_optimizer.py
-------------------------------------
סוכן האופטימיזציה — Optimizer Agent

תפקיד:
  1. קורא דוחות methodology מה-AgentBus
  2. קורא הצעות Math Agent (אם קיימות)
  3. מנתח חולשות: משטר חלש ביותר, מגמות IC, בריאות trade monitor
  4. שולח ניתוח ל-Claude עם כל הפרמטרים + מטריקות נוכחיים
  5. Claude מציע 1-3 שינויים (פרמטרים ו/או קוד)
  6. מבצע שינויים עם backup אוטומטי
  7. מריץ methodology מחדש לאימות שיפור
  8. רושם תוצאות ב-optimization_log
  9. מפרסם ל-bus

הרצה:
  python agents/optimizer/agent_optimizer.py --once
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── נתיבי שורש ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── לוגים ────────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "agent_optimizer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_optimizer")

# ── Imports פנימיים ─────────────────────────────────────────────────────────
from scripts.agent_bus import get_bus
from scripts.claude_loop import execute_actions, _extract_json_block, send_to_claude
from agents.optimizer.optimization_log import get_optimization_log
from agents.shared.agent_registry import get_registry, AgentStatus

# ── GPT Fallback — when Claude is unavailable, use GPT via LLM bridge ──
def _query_gpt_fallback(prompt: str) -> str:
    """Query GPT when Claude API is not configured."""
    try:
        from agents.math.llm_bridge import DualLLMBridge
        bridge = DualLLMBridge()
        if bridge.has_gpt:
            return bridge.query_gpt(prompt, "optimization")
        elif bridge.has_claude:
            return bridge.query_claude(prompt, "optimization")
    except Exception as e:
        log.warning("GPT fallback failed: %s", e)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# עזרים — קריאת מצב המערכת
# ─────────────────────────────────────────────────────────────────────────────
def _load_settings_snapshot() -> dict[str, Any]:
    """
    קורא את כל הפרמטרים מ-settings.py עם ערכים נוכחיים, טווחים, וסוגים.
    מחזיר dict מובנה לשימוש ב-system prompt.
    """
    try:
        from config.settings import get_settings, Settings
        settings = get_settings()

        # חילוץ כל השדות עם metadata
        params: dict[str, dict] = {}
        for name, field_info in Settings.model_fields.items():
            # דילוג על שדות לא-מספריים שאינם רלוונטיים לאופטימיזציה
            val = getattr(settings, name, None)
            if isinstance(val, (int, float, bool)):
                meta: dict[str, Any] = {
                    "value": val,
                    "type": type(val).__name__,
                }
                # חילוץ ge/le מה-metadata אם קיים
                for constraint in field_info.metadata:
                    if hasattr(constraint, "ge"):
                        meta["min"] = constraint.ge
                    if hasattr(constraint, "le"):
                        meta["max"] = constraint.le
                params[name] = meta

        return params
    except Exception as exc:
        log.warning("Failed to load settings snapshot: %s", exc)
        return {}


def _load_backtest_cache() -> Optional[dict]:
    """טוען backtest cache אם קיים."""
    cache_path = ROOT / "logs" / "last_backtest.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _load_math_proposals() -> list[dict]:
    """
    קורא הצעות Math Agent מתיקיית math_proposals ומה-bus.
    מחזיר רשימת הצעות.
    """
    proposals = []

    # מתיקייה
    proposals_dir = ROOT / "agents" / "math" / "math_proposals"
    if proposals_dir.exists():
        for f in sorted(proposals_dir.glob("*.py"))[-5:]:  # 5 אחרונות
            try:
                content = f.read_text(encoding="utf-8")
                proposals.append({
                    "source": "file",
                    "filename": f.name,
                    "content": content[:3000],  # חיתוך למניעת overflow
                })
            except Exception:
                continue

    # מה-bus
    bus = get_bus()
    math_latest = bus.latest("agent_math")
    if math_latest and "proposals" in math_latest:
        for p in math_latest["proposals"]:
            proposals.append({
                "source": "bus",
                **p,
            })

    return proposals


def _run_backtest() -> dict:
    """מריץ backtest ומחזיר תוצאות."""
    try:
        from config.settings import get_settings
        from data_ops.orchestrator import DataOrchestrator
        from analytics.backtest import WalkForwardBacktester

        settings = get_settings()
        orchestrator = DataOrchestrator(settings)
        data_state = orchestrator.run(force_refresh=False)
        prices_df = data_state.artifacts.prices

        bt = WalkForwardBacktester(settings)
        result = bt.run_backtest(prices_df, prices_df, prices_df)

        regime_bd = result.regime_breakdown
        regime_dict: dict = {}
        for reg in ["calm", "normal", "tension", "crisis"]:
            rd = getattr(regime_bd, reg, None)
            if rd:
                regime_dict[reg.upper()] = {
                    "ic_mean": round(float(rd.ic_mean or 0), 4),
                    "hit_rate": round(float(rd.hit_rate or 0), 4),
                    "sharpe": round(float(rd.sharpe or 0), 4),
                    "n_walks": int(rd.n_walks or 0),
                }

        return {
            "ic_mean": round(float(result.ic_mean or 0), 4),
            "ic_ir": round(float(result.ic_ir or 0), 4),
            "hit_rate": round(float(result.hit_rate or 0), 4),
            "sharpe": round(float(result.sharpe or 0), 4),
            "max_drawdown": round(float(result.max_drawdown or 0), 4),
            "n_walks": int(result.n_walks or 0),
            "regime_breakdown": regime_dict,
        }
    except Exception as exc:
        log.error("Backtest failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# edit_code — שינוי קוד עם backup, tests, ו-revert
# ─────────────────────────────────────────────────────────────────────────────
def _execute_edit_code(action: dict) -> dict:
    """
    מבצע שינוי קוד בפונקציה ספציפית:
      1. יוצר backup לקובץ היעד
      2. מחליף את הפונקציה (old_code → new_code)
      3. מריץ pytest
      4. אם tests עוברים → שומר; אם נכשלים → מחזיר לbackup

    פרמטרים נדרשים ב-action:
      target_file  — נתיב יחסי ל-ROOT (e.g., "analytics/signal_stack.py")
      function_name — שם הפונקציה להחלפה
      new_code     — הקוד החדש (כולל def ו-docstring)
    """
    target_rel = action.get("target_file", "")
    func_name = action.get("function_name", "")
    new_code = action.get("new_code", "")

    if not target_rel or not func_name or not new_code:
        return {
            "action": "edit_code",
            "outcome": "error",
            "error": "missing target_file, function_name, or new_code",
        }

    target_path = ROOT / target_rel

    if not target_path.exists():
        return {
            "action": "edit_code",
            "outcome": "error",
            "error": f"file not found: {target_rel}",
        }

    # 1. Backup — שמירת גיבוי
    backup_path = target_path.with_suffix(target_path.suffix + ".optbak")
    try:
        shutil.copy2(target_path, backup_path)
        log.info("Backup created: %s", backup_path.name)
    except Exception as exc:
        return {"action": "edit_code", "outcome": "error", "error": f"backup failed: {exc}"}

    # 2. קריאת הקוד הנוכחי וזיהוי הפונקציה
    try:
        original_text = target_path.read_text(encoding="utf-8")

        # חיפוש הפונקציה — regex שתופס def function_name עד הפונקציה הבאה
        # תומך בפונקציות עם דקורטורים
        pattern = re.compile(
            rf"(^(?:@\w+.*\n)*def\s+{re.escape(func_name)}\s*\(.*?(?:^def\s|\Z))",
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(original_text)

        if not match:
            # ניסיון חלופי — חיפוש פשוט יותר
            # מחפש def func_name עד def הבא או סוף קובץ
            lines = original_text.split("\n")
            start_idx = None
            end_idx = len(lines)

            for i, line in enumerate(lines):
                if re.match(rf"def\s+{re.escape(func_name)}\s*\(", line):
                    start_idx = i
                elif start_idx is not None and i > start_idx and re.match(r"(?:def\s|class\s)", line):
                    end_idx = i
                    break

            if start_idx is None:
                backup_path.unlink(missing_ok=True)
                return {
                    "action": "edit_code",
                    "outcome": "error",
                    "error": f"function '{func_name}' not found in {target_rel}",
                }

            # החלפת הפונקציה
            new_lines = lines[:start_idx] + new_code.rstrip().split("\n") + lines[end_idx:]
            new_text = "\n".join(new_lines)
        else:
            # החלפה באמצעות regex match
            matched_text = match.group(0)
            # הסרת ה-def הבא שנתפס בסוף (אם קיים)
            if matched_text.rstrip().endswith("def"):
                matched_text = matched_text[:matched_text.rfind("\ndef") + 1]

            new_text = original_text.replace(matched_text, new_code.rstrip() + "\n\n")

        target_path.write_text(new_text, encoding="utf-8")
        log.info("Code replaced: %s in %s", func_name, target_rel)

    except Exception as exc:
        shutil.copy2(backup_path, target_path)
        backup_path.unlink(missing_ok=True)
        return {"action": "edit_code", "outcome": "error", "error": f"code replacement failed: {exc}"}

    # 3. הרצת Tests — אימות תקינות
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        tests_passed = result.returncode == 0 and "passed" in result.stdout

        if tests_passed:
            log.info("Tests PASSED after code edit: %s.%s", target_rel, func_name)
            backup_path.unlink(missing_ok=True)
            return {
                "action": "edit_code",
                "outcome": "success",
                "function": func_name,
                "file": target_rel,
                "test_output": result.stdout[-500:],
            }
        else:
            # 4. Revert — tests נכשלו, מחזירים לגיבוי
            log.warning("Tests FAILED after code edit — reverting %s", target_rel)
            shutil.copy2(backup_path, target_path)
            backup_path.unlink(missing_ok=True)
            return {
                "action": "edit_code",
                "outcome": "reverted",
                "function": func_name,
                "file": target_rel,
                "revert_reason": "tests_failed",
                "test_output": result.stdout[-500:] + "\n" + result.stderr[-300:],
            }

    except Exception as exc:
        # Revert בגלל שגיאת הרצת tests
        log.error("Test runner error — reverting: %s", exc)
        shutil.copy2(backup_path, target_path)
        backup_path.unlink(missing_ok=True)
        return {
            "action": "edit_code",
            "outcome": "reverted",
            "function": func_name,
            "file": target_rel,
            "revert_reason": f"test_runner_error: {exc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Extended action executor — מרחיב את claude_loop עם edit_code
# ─────────────────────────────────────────────────────────────────────────────
def execute_extended_actions(actions: list[dict]) -> list[dict]:
    """
    מרחיב את execute_actions הרגיל עם תמיכה ב-edit_code.
    כל שאר הפעולות מועברות ל-executor הרגיל.
    """
    results = []
    regular_actions = []

    for action in actions:
        if action.get("type") == "edit_code":
            result = _execute_edit_code(action)
            results.append(result)
        else:
            regular_actions.append(action)

    # הפעלת שאר הפעולות דרך ה-executor הרגיל
    if regular_actions:
        regular_results = execute_actions(regular_actions)
        results.extend(regular_results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — הנחיות ל-Claude
# ─────────────────────────────────────────────────────────────────────────────
def _build_system_prompt(
    params: dict,
    metrics: Optional[dict],
    trend: list[dict],
    math_proposals: list[dict],
) -> str:
    """בונה system prompt מלא עם כל ההקשר."""

    # פירוט פרמטרים — כל פרמטר עם ערך נוכחי וטווח
    param_lines = []
    for name, meta in sorted(params.items()):
        val = meta.get("value", "?")
        pmin = meta.get("min", "")
        pmax = meta.get("max", "")
        range_str = f"  [{pmin} .. {pmax}]" if pmin != "" and pmax != "" else ""
        param_lines.append(f"  {name}: {val}{range_str}")
    params_block = "\n".join(param_lines) if param_lines else "  (unavailable)"

    # מטריקות נוכחיות
    if metrics:
        regime_lines = []
        for reg, rd in metrics.get("regime_breakdown", {}).items():
            ic = rd.get("ic_mean", 0)
            hr = rd.get("hit_rate", 0)
            sh = rd.get("sharpe", 0)
            regime_lines.append(f"    {reg}: IC={ic:.4f}, HitRate={hr:.1%}, Sharpe={sh:.2f}")

        metrics_block = textwrap.dedent(f"""\
            Overall: IC={metrics.get('ic_mean', 0):.4f}, Sharpe={metrics.get('sharpe', 0):.3f}, HitRate={metrics.get('hit_rate', 0):.1%}, MaxDD={metrics.get('max_drawdown', 0):.1%}
            Regime breakdown:
            {chr(10).join(regime_lines)}""")
    else:
        metrics_block = "  (no backtest data available)"

    # טרנד היסטורי
    if trend:
        trend_lines = [
            f"    {t.get('timestamp', '?')[:10]}: {t.get('outcome', '?')} "
            f"Δsharpe={t.get('delta_sharpe', 'N/A')} Δic={t.get('delta_ic', 'N/A')}"
            for t in trend
        ]
        trend_block = "\n".join(trend_lines)
    else:
        trend_block = "  (no history yet)"

    # הצעות Math Agent
    if math_proposals:
        proposals_lines = []
        for p in math_proposals[:3]:  # מקסימום 3
            proposals_lines.append(
                f"  - {p.get('filename', p.get('function_name', '?'))}: "
                f"{p.get('content', '')[:500]}"
            )
        proposals_block = "\n".join(proposals_lines)
    else:
        proposals_block = "  (none)"

    return textwrap.dedent(f"""\
        You are the Optimizer Agent for the SRV Quantamental DSS.
        Your role: improve system performance by modifying parameters and code.

        ## Architecture
        4-layer signal stack for short vol/dispersion:
          Layer 1: Distortion Score — S^dist = sigma(a1*z_D + a2*rank(m_t) + a3*z_CoC)
          Layer 2: Dislocation Score — S^disloc = min(1, |z|/Z_cap) per candidate
          Layer 3: Mean-Reversion Score — S^mr = w_hl*f_hl + w_adf*f_adf + w_hurst*f_hurst
          Layer 4: Regime Safety — S^safe = prod(1 - w_i * P_i) with hard kills
          Combined: Score_j = S^dist * S^disloc * S^mr * S^safe

        ## Current Parameters (all ~80 tunable params with ranges)
        {params_block}

        ## Current Backtest Metrics
        {metrics_block}

        ## Historical Trend (last 5 optimization runs)
        {trend_block}

        ## Math Agent Proposals
        {math_proposals_block}

        ## Available Actions
        ```json
        {{"actions": [
          {{"type": "read_file",  "file": "config/settings.py"}},
          {{"type": "edit_param", "file": "config/settings.py", "param": "pca_window", "value": 180}},
          {{"type": "edit_code",  "target_file": "analytics/signal_stack.py",
            "function_name": "compute_distortion_score",
            "new_code": "def compute_distortion_score(...):\\n    ..."}},
          {{"type": "run_tests",  "path": "tests/"}},
          {{"type": "log",        "message": "explanation"}},
          {{"type": "done",       "summary": "what changed and why"}}
        ]}}
        ```

        ## Rules
        1. Propose 1-3 specific changes, each as an action block
        2. Always read the target file before editing
        3. Every change must have mathematical justification
        4. Parameter changes: use edit_param (auto backup + test + revert)
        5. Code changes: use edit_code (auto backup + test + revert)
        6. Primary objective: improve IC in the weakest regime
        7. Secondary: improve overall Sharpe
        8. Never break existing interfaces — only modify formulas/values
        9. If math agent proposals exist, evaluate and adopt the best one
        """).replace("{math_proposals_block}", proposals_block)


# ─────────────────────────────────────────────────────────────────────────────
# Custom agent loop — מורחב עם edit_code
# ─────────────────────────────────────────────────────────────────────────────
def _run_optimizer_loop(
    system_prompt: str,
    initial_message: str,
    max_turns: int = 8,
) -> dict:
    """
    לולאת סוכן מותאמת — כמו run_agent_loop אבל עם execute_extended_actions.
    """
    log.info("=" * 55)
    log.info("Optimizer Agent Loop")
    log.info("=" * 55)

    messages: list[dict] = []
    turn = 0
    loop_log: list[dict] = []
    current_message = initial_message

    while turn < max_turns:
        turn += 1
        log.info("[Turn %d/%d] Sending to Claude...", turn, max_turns)

        # Try Claude first, fall back to GPT if Claude unavailable
        try:
            reply, messages = send_to_claude(system_prompt, messages, current_message)
            log.info("[Turn %d] Claude replied (%d chars)", turn, len(reply))
        except Exception as claude_err:
            log.warning("[Turn %d] Claude failed (%s) — trying GPT fallback", turn, claude_err)
            reply = _query_gpt_fallback(f"{system_prompt}\n\n{current_message}")
            if reply:
                messages.append({"role": "user", "content": current_message})
                messages.append({"role": "assistant", "content": reply})
                log.info("[Turn %d] GPT replied (%d chars)", turn, len(reply))
            else:
                log.error("[Turn %d] Both Claude and GPT failed — aborting", turn)
                break

        loop_log.append({
            "turn": turn,
            "claude_reply_preview": reply[:300],
        })

        # חילוץ וביצוע פעולות
        action_block = _extract_json_block(reply)
        if action_block and "actions" in action_block:
            actions = action_block["actions"]
            log.info("[Turn %d] Executing %d actions (extended)...", turn, len(actions))

            # שימוש ב-executor המורחב שתומך ב-edit_code
            results = execute_extended_actions(actions)
            loop_log[-1]["actions_executed"] = len(actions)
            loop_log[-1]["results"] = results

            # בדיקה אם done
            if any(r.get("action") == "done" for r in results):
                log.info("Optimizer completed (done action received).")
                break

            # החזרת תוצאות ל-Claude
            current_message = (
                f"פעולות בוצעו. תוצאות:\n```json\n"
                f"{json.dumps(results, ensure_ascii=False, indent=2)}\n```\n"
                'האם יש פעולות נוספות? אם הכל הושלם, שלח {"actions": [{"type": "done", "summary": "..."}]}'
            )
        else:
            current_message = (
                "האם יש פעולות ספציפיות שתרצה לבצע? "
                "שלח בלוק JSON עם actions, או done אם הסתיים."
            )
            loop_log[-1]["actions_executed"] = 0

        if turn >= max_turns:
            log.warning("Optimizer reached max_turns=%d — stopping.", max_turns)

    # שמירת לוג
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"agent_optimizer_{ts}.json"
    log_save = [
        {k: v for k, v in entry.items() if k != "claude_reply_full"}
        for entry in loop_log
    ]
    log_path.write_text(
        json.dumps({"agent": "optimizer", "turns": turn, "log": log_save},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Loop log saved → %s", log_path.name)

    return {"agent": "optimizer", "turns": turn, "log": loop_log, "log_path": str(log_path)}


# ─────────────────────────────────────────────────────────────────────────────
# ניתוח חולשות
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_weaknesses(metrics: dict) -> str:
    """מנתח חולשות מהמטריקות ובונה הודעה ראשונית ל-Claude."""

    lines = ["## Current System Analysis\n"]

    # מטריקות כלליות
    lines.append(f"Overall: IC={metrics.get('ic_mean', 0):.4f}, "
                 f"Sharpe={metrics.get('sharpe', 0):.3f}, "
                 f"HitRate={metrics.get('hit_rate', 0):.1%}")

    # זיהוי משטר חלש ביותר
    regime_bd = metrics.get("regime_breakdown", {})
    if regime_bd:
        lines.append("\n### Regime Performance:")
        weakest_regime = None
        weakest_ic = float("inf")

        for reg, rd in regime_bd.items():
            ic = rd.get("ic_mean", 0)
            hr = rd.get("hit_rate", 0)
            sh = rd.get("sharpe", 0)
            flag = " ← WEAKEST" if ic < weakest_ic else ""
            if ic < weakest_ic:
                weakest_ic = ic
                weakest_regime = reg
            lines.append(f"  {reg}: IC={ic:.4f}, HitRate={hr:.1%}, Sharpe={sh:.2f}{flag}")

        if weakest_regime:
            lines.append(f"\n### Primary Target: {weakest_regime} regime (IC={weakest_ic:.4f})")

    # בדיקת חולשות ספציפיות
    issues = []
    ic = metrics.get("ic_mean", 0)
    sharpe = metrics.get("sharpe", 0)
    hr = metrics.get("hit_rate", 0)

    if ic < 0.03:
        issues.append(f"IC very low ({ic:.4f} < 0.03 target)")
    if sharpe < 0.5:
        issues.append(f"Sharpe below acceptable ({sharpe:.3f} < 0.5)")
    if hr < 0.52:
        issues.append(f"Hit rate barely above random ({hr:.1%} < 52%)")

    if issues:
        lines.append("\n### Identified Issues:")
        for issue in issues:
            lines.append(f"  - {issue}")

    lines.append("\n---")
    lines.append("**Task:** Propose 1-3 specific changes to improve the weakest regime.")
    lines.append("Start by reading relevant files, then propose edit_param or edit_code actions.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────
def run(once: bool = False) -> None:
    """
    הרצת סוכן האופטימיזציה.
    once=True → הרצה בודדת; False → לולאה אינסופית (כל 4 שעות)
    """
    registry = get_registry()
    registry.register("agent_optimizer", role="parameter & code optimization")

    bus = get_bus()
    opt_log = get_optimization_log()

    while True:
        try:
            registry.heartbeat("agent_optimizer", AgentStatus.RUNNING)
            log.info("=" * 60)
            log.info("Optimizer Agent — starting run")
            log.info("=" * 60)

            # 1. קריאת דוח methodology מה-bus
            methodology_report = bus.latest("agent_methodology")
            log.info("Methodology report: %s", "found" if methodology_report else "none")

            # 2. קריאת הצעות Math Agent
            math_proposals = _load_math_proposals()
            log.info("Math proposals: %d found", len(math_proposals))

            # 3. טעינת מטריקות נוכחיות (מ-cache או bus)
            metrics = _load_backtest_cache()
            if not metrics:
                # ניסיון לקרוא מ-bus
                improve_report = bus.latest("agent_improve_system")
                if improve_report:
                    metrics = {
                        "ic_mean": improve_report.get("bt_ic", 0),
                        "sharpe": improve_report.get("bt_sharpe", 0),
                        "hit_rate": improve_report.get("bt_hit_rate", 0),
                        "regime_breakdown": {},
                    }

            if not metrics:
                log.warning("No backtest metrics available — running backtest...")
                metrics = _run_backtest()
                if not metrics:
                    log.error("Cannot proceed without metrics. Aborting.")
                    registry.heartbeat("agent_optimizer", AgentStatus.FAILED,
                                       error="no backtest metrics")
                    if once:
                        return
                    time.sleep(14400)
                    continue

            # שמירת מטריקות before
            before_metrics = {
                "ic": metrics.get("ic_mean", 0),
                "sharpe": metrics.get("sharpe", 0),
                "hit_rate": metrics.get("hit_rate", 0),
            }

            # 4. טעינת snapshot של פרמטרים
            params = _load_settings_snapshot()
            log.info("Loaded %d parameters", len(params))

            # 5. טרנד היסטורי
            trend = opt_log.recent_trend(n=5)

            # 6. בניית system prompt
            system_prompt = _build_system_prompt(params, metrics, trend, math_proposals)

            # 7. ניתוח חולשות — הודעה ראשונית
            initial_message = _analyze_weaknesses(metrics)

            # 8. הפעלת לולאת Claude עם executor מורחב
            result = _run_optimizer_loop(
                system_prompt=system_prompt,
                initial_message=initial_message,
                max_turns=8,
            )

            # 9. הרצת backtest אחרי שינויים — השוואה
            log.info("Running post-optimization backtest...")
            after_bt = _run_backtest()
            after_metrics = {
                "ic": after_bt.get("ic_mean", 0),
                "sharpe": after_bt.get("sharpe", 0),
                "hit_rate": after_bt.get("hit_rate", 0),
            } if after_bt else before_metrics

            # חישוב דלתות
            delta_sharpe = after_metrics["sharpe"] - before_metrics["sharpe"]
            delta_ic = after_metrics["ic"] - before_metrics["ic"]
            outcome = "improved" if delta_sharpe > 0 or delta_ic > 0 else "no_improvement"

            log.info(
                "Optimization result: %s (Δsharpe=%+.4f, Δic=%+.4f)",
                outcome, delta_sharpe, delta_ic,
            )

            # 10. רישום ב-optimization log
            opt_log.log_attempt({
                "agent_source": "optimizer",
                "change_type": "mixed",
                "target_file": "various",
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "outcome": outcome,
                "delta_sharpe": round(delta_sharpe, 6),
                "delta_ic": round(delta_ic, 6),
                "turns": result["turns"],
            })

            # 11. פרסום ל-bus
            bus.publish("agent_optimizer", {
                "status": "completed",
                "outcome": outcome,
                "before_ic": before_metrics["ic"],
                "after_ic": after_metrics["ic"],
                "delta_sharpe": round(delta_sharpe, 6),
                "delta_ic": round(delta_ic, 6),
                "turns": result["turns"],
                "math_proposals_used": len(math_proposals),
            })

            registry.heartbeat("agent_optimizer", AgentStatus.COMPLETED)
            log.info("Optimizer run completed successfully.")

        except Exception as exc:
            log.exception("Optimizer agent error: %s", exc)
            registry.heartbeat("agent_optimizer", AgentStatus.FAILED, error=str(exc))
            bus.publish("agent_optimizer", {
                "status": "failed",
                "error": str(exc),
            })

        if once:
            break

        # המתנה בין ריצות — 4 שעות
        log.info("Sleeping 4 hours until next optimization run...")
        time.sleep(14400)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(once="--once" in sys.argv)
