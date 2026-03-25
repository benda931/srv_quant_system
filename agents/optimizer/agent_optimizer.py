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

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "agent_optimizer.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
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
    """Run backtest with timeout — prefer dispersion backtest (faster, more relevant)."""
    # ── Primary: Dispersion Backtest (fast, strategy-relevant) ────────────
    try:
        import pandas as pd
        from analytics.dispersion_backtest import DispersionBacktester
        prices = pd.read_parquet(str(ROOT / "data_lake" / "parquet" / "prices.parquet"))
        bt = DispersionBacktester(
            prices, hold_period=15, z_entry=0.6, z_exit=0.2,
            max_positions=3, lookback=30,
        )
        result = bt.run()
        return {
            "sharpe": result.sharpe,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "avg_hold": result.avg_holding_days,
            "source": "dispersion_backtest",
        }
    except Exception as e:
        log.warning("Dispersion backtest failed: %s, trying walk-forward...", e)

    # ── Fallback: Walk-Forward Backtest with 60s timeout ──────────────────
    try:
        import concurrent.futures

        def _walk_forward_backtest() -> dict:
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
                "source": "walk_forward",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_walk_forward_backtest)
            try:
                return future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                log.error("Walk-forward backtest timed out (60s limit)")
                return {"sharpe": 0, "error": "walk_forward_timeout"}
    except Exception as exc:
        log.error("Backtest failed: %s", exc)
        return {"sharpe": 0, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter bounds validation — verify changes are within settings.py Field bounds
# ─────────────────────────────────────────────────────────────────────────────
def _validate_param_bounds(param_name: str, new_value: float) -> dict:
    """
    Validate a parameter change against settings.py Field bounds (ge/le constraints).
    Returns: {"valid": bool, "reason": str, "clamped_value": float}
    """
    try:
        from config.settings import Settings
        field_info = Settings.model_fields.get(param_name)
        if field_info is None:
            return {"valid": False, "reason": f"Unknown parameter: {param_name}", "clamped_value": new_value}

        # Extract bounds from field metadata
        lower = None
        upper = None
        for constraint in field_info.metadata:
            if hasattr(constraint, "ge"):
                lower = constraint.ge
            if hasattr(constraint, "gt"):
                lower = constraint.gt
            if hasattr(constraint, "le"):
                upper = constraint.le
            if hasattr(constraint, "lt"):
                upper = constraint.lt

        clamped = new_value
        reasons = []

        if lower is not None and new_value < lower:
            clamped = lower
            reasons.append(f"below min {lower}")
        if upper is not None and new_value > upper:
            clamped = upper
            reasons.append(f"above max {upper}")

        if reasons:
            return {
                "valid": False,
                "reason": f"{param_name}={new_value} out of bounds ({', '.join(reasons)}). Clamped to {clamped}",
                "clamped_value": clamped,
            }

        return {"valid": True, "reason": "within bounds", "clamped_value": new_value}
    except Exception as exc:
        log.warning("Bounds check failed for %s: %s", param_name, exc)
        return {"valid": True, "reason": f"bounds check error: {exc}", "clamped_value": new_value}


def _validate_suggestion(param_name: str, new_value: float, current_metrics: dict) -> dict:
    """
    Test a parameter change via quick backtest before applying.
    Tests: the suggested value, a halfway value, to find the best option.
    Returns: {approved, best_value, sharpe_before, sharpe_after, delta}
    """
    from config.settings import get_settings
    settings = get_settings()
    old_value = getattr(settings, param_name, None)

    if old_value is None:
        return {"approved": False, "reason": f"Unknown parameter: {param_name}"}

    # Bounds check first
    bounds = _validate_param_bounds(param_name, new_value)
    if not bounds["valid"]:
        log.warning("Bounds violation: %s — clamping", bounds["reason"])
        new_value = bounds["clamped_value"]

    # Test candidates: suggested value and halfway point
    candidates = [new_value]
    halfway = (float(old_value) + float(new_value)) / 2.0
    if abs(halfway - new_value) > 1e-9:
        candidates.append(halfway)

    best_sharpe = current_metrics.get("sharpe", 0)
    best_value = old_value

    for candidate in candidates:
        try:
            # Temporarily set parameter
            setattr(settings, param_name, type(old_value)(candidate))

            # Quick backtest
            result = _run_backtest()
            candidate_sharpe = result.get("sharpe", 0)

            log.info(
                "  Validation: %s=%s -> Sharpe=%.4f (current=%.4f)",
                param_name, candidate, candidate_sharpe, best_sharpe,
            )

            if candidate_sharpe > best_sharpe:
                best_sharpe = candidate_sharpe
                best_value = candidate
        except Exception as exc:
            log.warning("  Validation backtest failed for %s=%s: %s", param_name, candidate, exc)

    # Restore original value
    setattr(settings, param_name, old_value)

    current_sharpe = current_metrics.get("sharpe", 0)
    improved = best_sharpe > current_sharpe
    return {
        "approved": improved,
        "best_value": best_value,
        "original_value": old_value,
        "suggested_value": new_value,
        "sharpe_before": current_sharpe,
        "sharpe_after": best_sharpe,
        "delta": best_sharpe - current_sharpe,
        "bounds_check": bounds,
    }


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
def execute_extended_actions(actions: list[dict], current_metrics: dict | None = None) -> list[dict]:
    """
    מרחיב את execute_actions הרגיל עם תמיכה ב-edit_code + parameter validation.
    כל שאר הפעולות מועברות ל-executor הרגיל.
    """
    results = []
    regular_actions = []

    for action in actions:
        if action.get("type") == "edit_code":
            result = _execute_edit_code(action)
            results.append(result)
        elif action.get("type") == "edit_param" and current_metrics:
            # Validate parameter change before applying
            param_name = action.get("param", "")
            new_value = action.get("value")
            if param_name and new_value is not None:
                # Bounds check
                bounds = _validate_param_bounds(param_name, float(new_value))
                if not bounds["valid"]:
                    log.warning("Bounds violation for %s: %s", param_name, bounds["reason"])
                    action["value"] = bounds["clamped_value"]
                    results.append({
                        "action": "bounds_check",
                        "param": param_name,
                        "original_suggestion": new_value,
                        "clamped_to": bounds["clamped_value"],
                        "reason": bounds["reason"],
                    })

                # Quick validation backtest
                log.info("Validating %s=%s via quick backtest...", param_name, action["value"])
                validation = _validate_suggestion(param_name, float(action["value"]), current_metrics)
                results.append({
                    "action": "validation",
                    "param": param_name,
                    "approved": validation["approved"],
                    "best_value": validation.get("best_value"),
                    "sharpe_delta": validation.get("delta", 0),
                })

                if validation["approved"]:
                    # Use the best value found (might differ from suggestion)
                    action["value"] = validation["best_value"]
                    regular_actions.append(action)
                else:
                    log.warning(
                        "Validation REJECTED %s=%s (Sharpe delta=%.4f)",
                        param_name, action["value"], validation.get("delta", 0),
                    )
            else:
                regular_actions.append(action)
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
        {{math_proposals_block}}

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
        10. Parameter changes are validated via quick backtest before being applied
        11. All parameter values must be within the Field bounds defined in settings.py
        12. Focus on the SINGLE highest-impact change first — don't scatter

        ## Parameter Change Guidelines
        - When changing a parameter, explain WHY the new value is better (math/data reason)
        - Prefer small incremental changes (10-20%) over large jumps
        - If Sharpe < 0.3: focus on IC/signal quality parameters
        - If Sharpe 0.3-0.8: focus on regime-specific tuning
        - If HitRate < 50%: tighten entry thresholds or MR whitelist
        - If MaxDD > 5%: reduce leverage or add deleverage triggers
        """).replace("{math_proposals_block}", proposals_block)


# ─────────────────────────────────────────────────────────────────────────────
# Custom agent loop — מורחב עם edit_code
# ─────────────────────────────────────────────────────────────────────────────
def _run_optimizer_loop(
    system_prompt: str,
    initial_message: str,
    max_turns: int = 8,
    current_metrics: dict | None = None,
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
            log.info("[Turn %d] Executing %d actions (extended + validated)...", turn, len(actions))

            # שימוש ב-executor המורחב שתומך ב-edit_code + parameter validation
            results = execute_extended_actions(actions, current_metrics=current_metrics)
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

    # Dispersion backtest metrics (if available from bus)
    bus = get_bus()
    meth_report = bus.latest("agent_methodology")
    if isinstance(meth_report, dict):
        disp = meth_report.get("dispersion_backtest", {})
        if disp:
            lines.append(f"\n### Dispersion Backtest:")
            lines.append(f"  Sharpe={disp.get('sharpe', 'N/A')}, "
                         f"WR={disp.get('win_rate', 'N/A')}, "
                         f"P&L={disp.get('total_pnl', 'N/A')}")

        # Smart conclusions from methodology agent
        smart = meth_report.get("smart_conclusions", [])
        if smart:
            lines.append("\n### PM-Grade Analysis (from Methodology Agent):")
            for sc in smart[:3]:
                lines.append(f"  [{sc.get('priority')}] {sc.get('finding', '')}")
                if sc.get("action"):
                    lines.append(f"    -> Action: {sc.get('action')}")
                if sc.get("expected_impact"):
                    lines.append(f"    -> Expected: {sc.get('expected_impact')}")

    # Specific parameter guidance based on issues
    lines.append("\n### Recommended Focus:")
    if ic < 0.02:
        lines.append("  IC critically low — prioritize signal_a1_frob, zscore_window, pca_window")
    if sharpe < 0.3:
        lines.append("  Sharpe critically low — consider reducing trade_max_holding_days or max_leverage")
    if hr < 0.50:
        lines.append("  Hit rate below breakeven — raise signal_entry_threshold")

    lines.append("\n---")
    lines.append("**Task:** Propose 1-3 specific changes to improve the weakest regime.")
    lines.append("Start by reading relevant files, then propose edit_param or edit_code actions.")
    lines.append("Each change will be validated via quick backtest before applying.")

    return "\n".join(lines)


# =============================================================================
# Professional-Grade OptimizerAgent — Bayesian search, joint optimization,
# sensitivity analysis, auto-revert, GPT strategy brainstorming
# =============================================================================

class OptimizerAgent:
    """
    Hedge-fund professional optimization engine.

    Adds institutional-grade parameter optimization: Bayesian search,
    multi-parameter joint optimization, sensitivity analysis, automatic
    degradation revert, and GPT strategy brainstorming.
    """

    def __init__(self):
        """
        Initialize the OptimizerAgent.

        Loads current settings, backtest cache, and optimization log.
        """
        try:
            from config.settings import get_settings, Settings
            self.settings = get_settings()
            self.Settings = Settings
            self.opt_log = get_optimization_log()
            self._backup_settings: dict = {}
            log.info("OptimizerAgent initialized")
        except Exception as exc:
            log.error("OptimizerAgent init failed: %s", exc)
            from config.settings import get_settings, Settings
            self.settings = get_settings()
            self.Settings = Settings
            self.opt_log = None

    def _get_param_bounds(self, param_name: str) -> tuple:
        """Get (min, max, current) for a parameter from Settings field metadata."""
        try:
            field_info = self.Settings.model_fields.get(param_name)
            if field_info is None:
                return (None, None, None)

            current = getattr(self.settings, param_name, None)
            lower, upper = None, None
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    lower = constraint.ge
                if hasattr(constraint, "gt"):
                    lower = constraint.gt
                if hasattr(constraint, "le"):
                    upper = constraint.le
                if hasattr(constraint, "lt"):
                    upper = constraint.lt

            return (lower, upper, current)
        except Exception:
            return (None, None, getattr(self.settings, param_name, None))

    def _quick_backtest_sharpe(self) -> float:
        """Run a quick backtest and return Sharpe ratio."""
        try:
            result = _run_backtest()
            return float(result.get("sharpe", 0))
        except Exception as exc:
            log.warning("Quick backtest failed: %s", exc)
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # A. Bayesian Parameter Search
    # ─────────────────────────────────────────────────────────────────────
    def bayesian_search(
        self, param_name: str, n_trials: int = 20
    ) -> dict:
        """
        Bayesian-like parameter optimization using expected improvement.

        Uses Optuna if available, otherwise falls back to a manual
        surrogate-based search that starts from the current value and
        intelligently explores the neighborhood.

        Parameters
        ----------
        param_name : str
            The parameter to optimize (must exist in Settings).
        n_trials : int
            Number of evaluation trials (default 20).

        Returns
        -------
        dict
            best_value, best_sharpe, improvement_curve, n_trials_run.
        """
        try:
            lower, upper, current = self._get_param_bounds(param_name)
            if current is None:
                return {"error": f"parameter '{param_name}' not found in Settings"}

            current = float(current)
            if lower is None:
                lower = current * 0.3 if current > 0 else current - abs(current) * 2
            if upper is None:
                upper = current * 3.0 if current > 0 else current + abs(current) * 2
            lower, upper = float(lower), float(upper)

            original_value = current
            best_sharpe = self._quick_backtest_sharpe()
            best_value = current
            improvement_curve = [{"trial": 0, "value": current, "sharpe": best_sharpe}]

            # Try Optuna first
            optuna_available = False
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                optuna_available = True
            except ImportError:
                pass

            if optuna_available:
                import optuna

                def objective(trial):
                    val = trial.suggest_float(param_name, lower, upper)
                    setattr(self.settings, param_name, type(original_value)(val))
                    sharpe = self._quick_backtest_sharpe()
                    return sharpe

                study = optuna.create_study(direction="maximize")
                # Seed with current value
                study.enqueue_trial({param_name: current})
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                for i, trial in enumerate(study.trials):
                    improvement_curve.append({
                        "trial": i + 1,
                        "value": round(trial.params.get(param_name, current), 6),
                        "sharpe": round(trial.value if trial.value else 0, 4),
                    })

                best_value = study.best_params.get(param_name, current)
                best_sharpe = study.best_value or 0

            else:
                # Manual Bayesian-like search: Latin hypercube + neighborhood refinement
                import numpy as np
                rng = np.random.default_rng(42)

                # Phase 1: Broad exploration — Latin hypercube
                n_explore = max(n_trials // 2, 5)
                explore_vals = np.linspace(lower, upper, n_explore)
                rng.shuffle(explore_vals)

                observed_x = [current]
                observed_y = [best_sharpe]

                for i, val in enumerate(explore_vals):
                    try:
                        setattr(self.settings, param_name, type(original_value)(val))
                        sharpe = self._quick_backtest_sharpe()
                        observed_x.append(float(val))
                        observed_y.append(sharpe)
                        improvement_curve.append({
                            "trial": i + 1, "value": round(float(val), 6), "sharpe": round(sharpe, 4),
                        })
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_value = float(val)
                    except Exception:
                        continue

                # Phase 2: Refinement around best region
                n_refine = n_trials - n_explore
                if n_refine > 0 and best_value is not None:
                    radius = (upper - lower) * 0.1
                    for i in range(n_refine):
                        try:
                            val = float(rng.normal(best_value, radius))
                            val = max(lower, min(upper, val))
                            setattr(self.settings, param_name, type(original_value)(val))
                            sharpe = self._quick_backtest_sharpe()
                            improvement_curve.append({
                                "trial": n_explore + i + 1,
                                "value": round(val, 6),
                                "sharpe": round(sharpe, 4),
                            })
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_value = val
                        except Exception:
                            continue

            # Restore original value (caller decides whether to apply best)
            setattr(self.settings, param_name, type(original_value)(original_value))

            result = {
                "param_name": param_name,
                "original_value": round(original_value, 6),
                "best_value": round(float(best_value), 6),
                "original_sharpe": round(improvement_curve[0]["sharpe"], 4),
                "best_sharpe": round(float(best_sharpe), 4),
                "improvement": round(float(best_sharpe) - improvement_curve[0]["sharpe"], 4),
                "n_trials_run": len(improvement_curve) - 1,
                "improvement_curve": improvement_curve,
                "used_optuna": optuna_available,
            }

            log.info(
                "Bayesian search %s: %.4f -> %.4f (Sharpe %.3f -> %.3f, %+.3f)",
                param_name, original_value, best_value,
                improvement_curve[0]["sharpe"], best_sharpe,
                float(best_sharpe) - improvement_curve[0]["sharpe"],
            )

            return result

        except Exception as exc:
            log.exception("Bayesian search failed for %s", param_name)
            # Restore original
            try:
                setattr(self.settings, param_name, original_value)
            except Exception:
                pass
            return {"error": str(exc), "param_name": param_name}

    # ─────────────────────────────────────────────────────────────────────
    # B. Multi-Parameter Joint Optimization
    # ─────────────────────────────────────────────────────────────────────
    def joint_optimize(
        self, params_to_tune: list, n_trials: int = 30
    ) -> dict:
        """
        Optimize multiple parameters simultaneously using random search
        with Latin hypercube sampling.

        Parameters
        ----------
        params_to_tune : list of str
            Parameter names to optimize jointly.
        n_trials : int
            Number of joint evaluation trials (default 30).

        Returns
        -------
        dict
            best_params, best_sharpe, pareto_front (top combos sorted by Sharpe),
            trials log.
        """
        try:
            import numpy as np

            # Collect bounds and originals
            param_info = {}
            originals = {}
            for p in params_to_tune:
                lo, hi, cur = self._get_param_bounds(p)
                if cur is None:
                    return {"error": f"parameter '{p}' not found"}
                cur = float(cur)
                if lo is None:
                    lo = cur * 0.5 if cur > 0 else cur - abs(cur)
                if hi is None:
                    hi = cur * 2.0 if cur > 0 else cur + abs(cur)
                param_info[p] = {"lower": float(lo), "upper": float(hi), "current": cur}
                originals[p] = cur

            # Latin hypercube sampling
            rng = np.random.default_rng(42)
            n_params = len(params_to_tune)
            trials_log = []

            # Generate LHS samples
            for trial_idx in range(n_trials):
                try:
                    combo = {}
                    for p in params_to_tune:
                        info = param_info[p]
                        # Stratified random sample with some bias toward current
                        if trial_idx == 0:
                            val = info["current"]  # First trial: baseline
                        else:
                            val = float(rng.uniform(info["lower"], info["upper"]))
                        combo[p] = val
                        setattr(self.settings, p, type(originals[p])(val))

                    sharpe = self._quick_backtest_sharpe()
                    trials_log.append({
                        "trial": trial_idx,
                        "params": {k: round(v, 6) for k, v in combo.items()},
                        "sharpe": round(sharpe, 4),
                    })
                except Exception as trial_exc:
                    log.debug("Joint trial %d failed: %s", trial_idx, trial_exc)

            # Restore originals
            for p, orig in originals.items():
                setattr(self.settings, p, type(orig)(orig))

            if not trials_log:
                return {"error": "no valid trials completed"}

            # Sort by Sharpe — Pareto front (top 5)
            trials_sorted = sorted(trials_log, key=lambda t: t["sharpe"], reverse=True)
            best = trials_sorted[0]
            pareto = trials_sorted[:5]

            baseline_sharpe = next(
                (t["sharpe"] for t in trials_log if t["trial"] == 0), 0.0
            )

            result = {
                "params_tuned": params_to_tune,
                "n_trials": len(trials_log),
                "baseline_sharpe": baseline_sharpe,
                "best_params": best["params"],
                "best_sharpe": best["sharpe"],
                "improvement": round(best["sharpe"] - baseline_sharpe, 4),
                "pareto_front": pareto,
            }

            log.info(
                "Joint optimization (%s): best Sharpe=%.3f (%+.3f), %d trials",
                ", ".join(params_to_tune), best["sharpe"],
                best["sharpe"] - baseline_sharpe, len(trials_log),
            )

            return result

        except Exception as exc:
            log.exception("Joint optimization failed")
            # Restore originals
            try:
                for p, orig in originals.items():
                    setattr(self.settings, p, type(orig)(orig))
            except Exception:
                pass
            return {"error": str(exc)}

    # ─────────────────────────────────────────────────────────────────────
    # C. Parameter Sensitivity Analysis
    # ─────────────────────────────────────────────────────────────────────
    def sensitivity_analysis(
        self, param_name: str, n_points: int = 10
    ) -> dict:
        """
        Sweep a parameter from min to max and record performance metrics
        at each point. Computes gradient (dSharpe/dParam), optimal region,
        and stability assessment.

        Parameters
        ----------
        param_name : str
            Parameter to analyze.
        n_points : int
            Number of sweep points (default 10).

        Returns
        -------
        dict
            sweep_results (list), gradient, optimal_region, stability_score.
        """
        try:
            import numpy as np

            lower, upper, current = self._get_param_bounds(param_name)
            if current is None:
                return {"error": f"parameter '{param_name}' not found"}

            current = float(current)
            if lower is None:
                lower = current * 0.3 if current > 0 else current - abs(current) * 2
            if upper is None:
                upper = current * 3.0 if current > 0 else current + abs(current) * 2
            lower, upper = float(lower), float(upper)

            original_value = current
            sweep_values = np.linspace(lower, upper, n_points)
            sweep_results = []

            for val in sweep_values:
                try:
                    setattr(self.settings, param_name, type(original_value)(val))
                    result = _run_backtest()
                    sweep_results.append({
                        "value": round(float(val), 6),
                        "sharpe": round(float(result.get("sharpe", 0)), 4),
                        "win_rate": round(float(result.get("win_rate", result.get("hit_rate", 0))), 4),
                        "max_drawdown": round(float(result.get("max_drawdown", 0)), 4),
                    })
                except Exception as sweep_exc:
                    log.debug("Sensitivity sweep at %s=%.4f failed: %s", param_name, val, sweep_exc)

            # Restore
            setattr(self.settings, param_name, type(original_value)(original_value))

            if len(sweep_results) < 3:
                return {"error": "insufficient sweep data", "param_name": param_name}

            sharpes = np.array([s["sharpe"] for s in sweep_results])
            values = np.array([s["value"] for s in sweep_results])

            # Gradient: average dSharpe/dParam
            if len(values) > 1:
                d_sharpe = np.diff(sharpes)
                d_param = np.diff(values)
                gradients = d_sharpe / np.where(np.abs(d_param) > 1e-12, d_param, 1e-12)
                avg_gradient = float(np.mean(gradients))
            else:
                avg_gradient = 0.0

            # Optimal region: contiguous region where Sharpe is within 90% of peak
            peak_sharpe = float(np.max(sharpes))
            threshold = peak_sharpe * 0.9 if peak_sharpe > 0 else peak_sharpe * 1.1
            in_optimal = sharpes >= threshold if peak_sharpe > 0 else sharpes <= threshold
            optimal_indices = np.where(in_optimal)[0]
            if len(optimal_indices) > 0:
                optimal_region = {
                    "lower": round(float(values[optimal_indices[0]]), 6),
                    "upper": round(float(values[optimal_indices[-1]]), 6),
                    "best_value": round(float(values[np.argmax(sharpes)]), 6),
                    "best_sharpe": round(peak_sharpe, 4),
                }
            else:
                optimal_region = {"lower": lower, "upper": upper, "best_value": current, "best_sharpe": 0}

            # Stability: std of Sharpe across sweep (lower = more stable)
            stability = round(float(np.std(sharpes)), 4)
            stability_score = (
                "STABLE" if stability < 0.1
                else "MODERATE" if stability < 0.3
                else "SENSITIVE"
            )

            result = {
                "param_name": param_name,
                "n_points": len(sweep_results),
                "current_value": round(original_value, 6),
                "sweep_results": sweep_results,
                "gradient": round(avg_gradient, 6),
                "optimal_region": optimal_region,
                "stability": stability,
                "stability_score": stability_score,
            }

            log.info(
                "Sensitivity %s: gradient=%.4f, optimal=[%.3f, %.3f], stability=%s",
                param_name, avg_gradient,
                optimal_region["lower"], optimal_region["upper"],
                stability_score,
            )

            return result

        except Exception as exc:
            log.exception("Sensitivity analysis failed for %s", param_name)
            try:
                setattr(self.settings, param_name, original_value)
            except Exception:
                pass
            return {"error": str(exc), "param_name": param_name}

    # ─────────────────────────────────────────────────────────────────────
    # D. Automatic Revert on Degradation
    # ─────────────────────────────────────────────────────────────────────
    def check_and_revert(self) -> dict:
        """
        Compare current performance vs last promoted performance.
        If Sharpe dropped > 0.1 or win-rate dropped > 5%, automatically
        revert to backup settings and log the revert action.

        Returns
        -------
        dict
            action_taken (str: 'reverted' or 'no_action'), before/after metrics,
            reverted_params (list if reverted).
        """
        try:
            # Load last promoted metrics from optimization log
            if self.opt_log is None:
                return {"action_taken": "no_action", "reason": "optimization log not available"}

            trend = self.opt_log.recent_trend(n=1)
            if not trend:
                return {"action_taken": "no_action", "reason": "no previous optimization records"}

            last_record = trend[-1]
            promoted_metrics = last_record.get("after_metrics", last_record.get("before_metrics", {}))
            promoted_sharpe = float(promoted_metrics.get("sharpe", promoted_metrics.get("ic", 0)))
            promoted_wr = float(promoted_metrics.get("hit_rate", promoted_metrics.get("win_rate", 0)))

            # Current performance
            current_bt = _run_backtest()
            current_sharpe = float(current_bt.get("sharpe", 0))
            current_wr = float(current_bt.get("win_rate", current_bt.get("hit_rate", 0)))

            sharpe_drop = promoted_sharpe - current_sharpe
            wr_drop = (promoted_wr - current_wr) * 100  # percentage points

            needs_revert = sharpe_drop > 0.1 or wr_drop > 5.0

            result = {
                "promoted_sharpe": round(promoted_sharpe, 4),
                "current_sharpe": round(current_sharpe, 4),
                "sharpe_drop": round(sharpe_drop, 4),
                "promoted_wr": round(promoted_wr, 4),
                "current_wr": round(current_wr, 4),
                "wr_drop_pct": round(wr_drop, 2),
            }

            if needs_revert:
                log.warning(
                    "DEGRADATION DETECTED: Sharpe %.3f->%.3f (drop=%.3f), WR drop=%.1f%%",
                    promoted_sharpe, current_sharpe, sharpe_drop, wr_drop,
                )

                # Load backup settings
                backup_path = ROOT / "config" / "settings.py.optbak"
                reverted_params = []

                if backup_path.exists():
                    try:
                        import importlib
                        # Reload settings from backup
                        import shutil
                        settings_path = ROOT / "config" / "settings.py"
                        shutil.copy2(backup_path, settings_path)
                        log.info("Reverted settings.py from backup")
                        reverted_params.append("settings.py (full revert from backup)")
                    except Exception as rev_exc:
                        log.warning("File revert failed: %s", rev_exc)
                else:
                    log.info("No backup file found — logging revert recommendation only")
                    reverted_params.append("NO_BACKUP_AVAILABLE — manual review required")

                # Log the revert
                if self.opt_log:
                    self.opt_log.log_attempt({
                        "agent_source": "optimizer_revert",
                        "change_type": "auto_revert",
                        "target_file": "config/settings.py",
                        "before_metrics": {"sharpe": current_sharpe, "hit_rate": current_wr},
                        "after_metrics": {"sharpe": promoted_sharpe, "hit_rate": promoted_wr},
                        "outcome": "reverted",
                        "delta_sharpe": round(sharpe_drop, 6),
                        "reason": f"Sharpe drop={sharpe_drop:.3f}, WR drop={wr_drop:.1f}%",
                    })

                result["action_taken"] = "reverted"
                result["reverted_params"] = reverted_params
                result["reason"] = f"Sharpe drop={sharpe_drop:.3f}, WR drop={wr_drop:.1f}%"

            else:
                result["action_taken"] = "no_action"
                result["reason"] = "performance within acceptable bounds"

            log.info("Check-and-revert: %s", result["action_taken"])
            return result

        except Exception as exc:
            log.exception("Check-and-revert failed")
            return {"action_taken": "error", "error": str(exc)}

    # ─────────────────────────────────────────────────────────────────────
    # E. GPT Strategy Brainstorming
    # ─────────────────────────────────────────────────────────────────────
    def brainstorm_with_gpt(self, context: dict) -> dict:
        """
        Send current performance, regime, and weaknesses to GPT and ask
        for NEW strategy ideas (not just parameter changes). Parses the
        response into actionable strategy blueprints and saves them.

        Parameters
        ----------
        context : dict
            Current system context with keys: metrics, regime, weaknesses,
            existing_strategies.

        Returns
        -------
        dict
            strategy_ideas (list of dicts), raw_response, saved_path.
        """
        try:
            from agents.math.llm_bridge import DualLLMBridge

            bridge = DualLLMBridge()
            if not bridge.has_gpt and not bridge.has_claude:
                return {"error": "no LLM available for brainstorming"}

            metrics = context.get("metrics", {})
            regime = context.get("regime", "UNKNOWN")
            weaknesses = context.get("weaknesses", "none identified")
            existing = context.get("existing_strategies", [])

            prompt = (
                f"You are a senior quant researcher at a systematic hedge fund.\n\n"
                f"CURRENT REGIME: {regime}\n"
                f"METRICS: Sharpe={metrics.get('sharpe', 'N/A')}, "
                f"WR={metrics.get('hit_rate', metrics.get('win_rate', 'N/A'))}, "
                f"IC={metrics.get('ic_mean', metrics.get('ic', 'N/A'))}\n"
                f"WEAKNESSES: {weaknesses}\n"
                f"EXISTING STRATEGIES: {', '.join(existing[:10]) if existing else 'N/A'}\n\n"
                f"Propose 3 NEW trading strategies (not parameter changes) that would:\n"
                f"1. Diversify from existing strategies\n"
                f"2. Perform well in the current regime ({regime})\n"
                f"3. Be implementable with sector ETF data\n\n"
                f"For EACH strategy, provide:\n"
                f"- NAME: (short unique name)\n"
                f"- SIGNAL: (how to generate entry/exit signals)\n"
                f"- EDGE: (why this should work, economic intuition)\n"
                f"- PARAMS: (key parameters with suggested starting values)\n"
                f"- EXPECTED: (estimated Sharpe range)\n"
            )

            if bridge.has_gpt:
                response = bridge.query_gpt(prompt, "strategy_brainstorm")
            else:
                response = bridge.query_claude(prompt, "strategy_brainstorm")

            if not response:
                return {"error": "empty LLM response"}

            # Parse strategy ideas
            ideas = []
            current_idea: dict = {}
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    if current_idea and current_idea.get("name"):
                        ideas.append(current_idea)
                        current_idea = {}
                    continue
                line_upper = line.upper()
                if "NAME:" in line_upper or line_upper.startswith("NAME"):
                    if current_idea and current_idea.get("name"):
                        ideas.append(current_idea)
                    current_idea = {"name": line.split(":", 1)[-1].strip()}
                elif "SIGNAL:" in line_upper:
                    current_idea["signal"] = line.split(":", 1)[-1].strip()
                elif "EDGE:" in line_upper:
                    current_idea["edge"] = line.split(":", 1)[-1].strip()
                elif "PARAMS:" in line_upper:
                    current_idea["params"] = line.split(":", 1)[-1].strip()
                elif "EXPECTED:" in line_upper:
                    current_idea["expected_sharpe"] = line.split(":", 1)[-1].strip()

            if current_idea and current_idea.get("name"):
                ideas.append(current_idea)

            # Save to strategy_ideas directory
            ideas_dir = Path(__file__).resolve().parent / "strategy_ideas"
            ideas_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            save_path = ideas_dir / f"brainstorm_{ts}.json"
            save_data = {
                "timestamp": ts,
                "regime": regime,
                "metrics": metrics,
                "ideas": ideas,
                "raw_response": response,
            }
            save_path.write_text(
                json.dumps(save_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = {
                "strategy_ideas": ideas[:3],
                "n_ideas": len(ideas[:3]),
                "raw_response": response[:500],
                "saved_path": str(save_path),
                "regime": regime,
            }

            log.info(
                "GPT brainstorm: %d strategy ideas for regime %s, saved to %s",
                len(ideas[:3]), regime, save_path.name,
            )

            return result

        except Exception as exc:
            log.exception("GPT strategy brainstorming failed")
            return {"error": str(exc)}

    # ─────────────────────────────────────────────────────────────────────
    # Run all professional optimizations
    # ─────────────────────────────────────────────────────────────────────
    def run_professional_optimization(self, metrics: dict) -> dict:
        """
        Run full professional-grade optimization suite.

        Executes: degradation check, sensitivity analysis on key params,
        Bayesian search on most sensitive param, brainstorming.

        Parameters
        ----------
        metrics : dict
            Current backtest metrics.

        Returns
        -------
        dict
            Results from all professional optimization steps.
        """
        try:
            log.info("Running professional-grade optimization suite...")
            results: dict = {}

            # Step 1: Check for degradation / auto-revert
            log.info("  [OPT-PRO 1/4] Checking for degradation...")
            revert_result = self.check_and_revert()
            results["degradation_check"] = revert_result

            if revert_result.get("action_taken") == "reverted":
                log.warning("  Settings reverted due to degradation — re-running backtest")
                metrics = _run_backtest()

            # Step 2: Sensitivity analysis on key parameters
            log.info("  [OPT-PRO 2/4] Sensitivity analysis on key params...")
            key_params = ["pca_window", "zscore_window", "signal_entry_threshold"]
            sensitivity_results = {}
            for param in key_params:
                try:
                    if hasattr(self.settings, param):
                        sa = self.sensitivity_analysis(param, n_points=7)
                        sensitivity_results[param] = sa
                except Exception as sa_exc:
                    log.debug("Sensitivity for %s failed: %s", param, sa_exc)
            results["sensitivity"] = sensitivity_results

            # Step 3: Bayesian search on most sensitive parameter
            log.info("  [OPT-PRO 3/4] Bayesian search on most sensitive param...")
            most_sensitive = None
            max_gradient = 0
            for param, sa in sensitivity_results.items():
                if isinstance(sa, dict) and "gradient" in sa:
                    if abs(sa["gradient"]) > abs(max_gradient):
                        max_gradient = sa["gradient"]
                        most_sensitive = param

            if most_sensitive:
                bayesian_result = self.bayesian_search(most_sensitive, n_trials=15)
                results["bayesian_search"] = bayesian_result
            else:
                results["bayesian_search"] = {"skipped": "no sensitive parameter found"}

            # Step 4: GPT brainstorming
            log.info("  [OPT-PRO 4/4] GPT strategy brainstorming...")
            bus = get_bus()
            meth_report = bus.latest("agent_methodology") or {}
            regime = "UNKNOWN"
            if isinstance(meth_report, dict):
                regime = meth_report.get("parameters_snapshot", {}).get("regime", "UNKNOWN")

            existing_strategies = []
            if isinstance(meth_report, dict):
                lab_data = meth_report.get("methodology_lab", {})
                if isinstance(lab_data, dict):
                    existing_strategies = [
                        r.get("name", "") for r in lab_data.get("ranking", [])
                    ]

            weaknesses = []
            sharpe = metrics.get("sharpe", 0)
            if sharpe < 0.5:
                weaknesses.append(f"Low Sharpe ({sharpe})")
            wr = metrics.get("hit_rate", metrics.get("win_rate", 0))
            if wr < 0.52:
                weaknesses.append(f"Low win rate ({wr})")

            brainstorm = self.brainstorm_with_gpt({
                "metrics": metrics,
                "regime": regime,
                "weaknesses": "; ".join(weaknesses) if weaknesses else "none critical",
                "existing_strategies": existing_strategies,
            })
            results["brainstorm"] = brainstorm

            log.info("Professional optimization suite complete")
            return results

        except Exception as exc:
            log.exception("Professional optimization suite failed")
            return {"error": str(exc)}


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

            # 8. הפעלת לולאת Claude עם executor מורחב + parameter validation
            result = _run_optimizer_loop(
                system_prompt=system_prompt,
                initial_message=initial_message,
                max_turns=8,
                current_metrics=metrics,
            )

            # 8.5 Professional-grade optimization suite
            log.info("Running professional-grade optimization suite...")
            try:
                pro_optimizer = OptimizerAgent()
                pro_results = pro_optimizer.run_professional_optimization(metrics)
                log.info("Professional optimization: %d results", len(pro_results))
            except Exception as pro_exc:
                log.warning("Professional optimization failed: %s", pro_exc)
                pro_results = {"error": str(pro_exc)}

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
                "professional_optimization": {
                    k: v for k, v in pro_results.items()
                    if k not in ("brainstorm",)  # exclude large raw responses
                } if isinstance(pro_results, dict) else {},
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
