"""
scripts/claude_loop.py
-----------------------
מנגנון הלולאה המרכזי — Claude API Feedback Loop

כל סוכן שקורא למודול זה יכול:
  1. לשלוח תוצאות ל-Claude ולקבל משימות המשך
  2. לנהל היסטוריית שיחה (multi-turn)
  3. לפענח פקודות מובנות בתשובת Claude (JSON block)
  4. לבצע פקודות אוטומטית עם בטיחות מובנית

פורמט פקודה שClaude יכול לשלוח בתשובה:
  ```json
  {
    "actions": [
      {"type": "run_script", "script": "scripts/agent_daily_pipeline.py", "args": ["--force-refresh"]},
      {"type": "read_file",  "file": "config/settings.py"},
      {"type": "edit_param", "file": "config/settings.py", "param": "pca_window", "value": 180},
      {"type": "run_tests",  "path": "tests/"},
      {"type": "log",        "message": "הערה: הגדלתי חלון PCA מ-252 ל-180 בגלל IC נמוך ב-TENSION"},
      {"type": "done",       "summary": "כל המשימות הושלמו"}
    ]
  }
  ```

בטיחות edit_param:
  - יוצר backup לפני כל שינוי
  - מריץ tests אוטומטית לאחר שינוי
  - מחזיר לbackup אם tests נכשלים
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("claude_loop")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_MODEL = "claude-sonnet-4-6"
_MAX_TURNS = 8          # מקסימום סבבים בלולאה אחת
_MAX_TOKENS = 4096
_READ_FILE_MAX_CHARS = 8000   # חיתוך קריאת קובץ לClaude


def _get_client():
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or _load_env_key()
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY לא מוגדר ב-.env")
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("pip install anthropic")


def _load_env_key() -> Optional[str]:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[-1].strip().strip('"').strip("'")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: send one message, get response
# ─────────────────────────────────────────────────────────────────────────────
def send_to_claude(
    system_prompt: str,
    messages: list[dict],
    user_content: str,
) -> tuple[str, list[dict]]:
    """
    Send user_content to Claude and return (response_text, updated_messages).
    Manages the conversation history.
    """
    client = _get_client()
    messages = list(messages)  # copy
    messages.append({"role": "user", "content": user_content})

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    )
    reply = response.content[0].text
    messages.append({"role": "assistant", "content": reply})
    return reply, messages


# ─────────────────────────────────────────────────────────────────────────────
# Action executor
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json_block(text: str) -> Optional[dict]:
    """Extract first ```json ... ``` block from text."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def execute_actions(actions: list[dict]) -> list[dict]:
    """
    Execute a list of actions returned by Claude.
    Returns results list for the next Claude turn.

    Supported action types:
      run_script  — run a Python script via subprocess
      read_file   — read a file and return its contents
      edit_param  — safely edit a single parameter in a settings file
                    (creates backup, runs tests, auto-reverts on failure)
      run_tests   — run pytest
      log         — write a note to the log
      done        — signal completion
    """
    results = []
    for action in actions:
        atype = action.get("type", "")
        log.info("Executing action: %s", atype)

        # ── run_script ────────────────────────────────────────────────────────
        if atype == "run_script":
            script = ROOT / action.get("script", "")
            args   = action.get("args", [])
            try:
                result = subprocess.run(
                    [sys.executable, str(script)] + args,
                    cwd=str(ROOT), capture_output=True, text=True, timeout=600,
                )
                outcome = "success" if result.returncode == 0 else "failed"
                results.append({
                    "action": atype, "script": str(script.name),
                    "outcome": outcome, "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-500:] if result.returncode != 0 else "",
                })
            except Exception as e:
                results.append({"action": atype, "script": str(script), "outcome": "error", "error": str(e)})

        # ── read_file ─────────────────────────────────────────────────────────
        elif atype == "read_file":
            file_rel = action.get("file", "")
            try:
                target = ROOT / file_rel
                content = target.read_text(encoding="utf-8")
                if len(content) > _READ_FILE_MAX_CHARS:
                    content = content[:_READ_FILE_MAX_CHARS] + f"\n... (truncated at {_READ_FILE_MAX_CHARS} chars)"
                results.append({
                    "action": atype, "file": file_rel,
                    "outcome": "success", "content": content,
                })
            except FileNotFoundError:
                results.append({"action": atype, "file": file_rel, "outcome": "error", "error": "file not found"})
            except Exception as e:
                results.append({"action": atype, "file": file_rel, "outcome": "error", "error": str(e)})

        # ── edit_param ────────────────────────────────────────────────────────
        elif atype == "edit_param":
            file_rel = action.get("file", "")
            param    = action.get("param", "")
            value    = action.get("value")
            run_tests_after = action.get("run_tests", True)  # default: always test
            try:
                edited, test_result = _safe_edit_param(ROOT / file_rel, param, value, run_tests=run_tests_after)
                results.append({
                    "action": atype, "param": param, "value": value,
                    "outcome": "success" if edited else "reverted",
                    "test_result": test_result,
                })
            except Exception as e:
                results.append({"action": atype, "param": param, "outcome": "error", "error": str(e)})

        # ── run_tests ─────────────────────────────────────────────────────────
        elif atype == "run_tests":
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", action.get("path", "tests/"), "-q"],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=120,
                )
                passed = "passed" in result.stdout
                results.append({
                    "action": atype, "outcome": "passed" if passed else "failed",
                    "output": result.stdout[-1000:],
                })
            except Exception as e:
                results.append({"action": atype, "outcome": "error", "error": str(e)})

        # ── log ───────────────────────────────────────────────────────────────
        elif atype == "log":
            msg = action.get("message", "")
            log.info("[Claude note] %s", msg)
            results.append({"action": atype, "outcome": "logged"})

        # ── done ──────────────────────────────────────────────────────────────
        elif atype == "done":
            results.append({"action": "done", "summary": action.get("summary", ""), "outcome": "complete"})

        else:
            results.append({"action": atype, "outcome": "unknown_action"})

    return results


def _safe_edit_param(settings_path: Path, param: str, value: Any, run_tests: bool = True) -> tuple[bool, str]:
    """
    Safely edit a single numeric/bool parameter in settings.py.

    Steps:
      1. Create a .bak backup of the file
      2. Apply the regex substitution
      3. If run_tests=True, run pytest:
         - Pass  → keep changes, delete backup, return (True, "passed")
         - Fail  → restore backup, return (False, "reverted: tests failed")
      4. If run_tests=False → keep changes, return (True, "no tests run")

    Raises ValueError if param is not found in the file.
    """
    text = settings_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"(\b{re.escape(param)}\b\s*:\s*\w+\s*=\s*)([^\n#]+)")
    new_text, count = pattern.subn(rf"\g<1>{value}", text)
    if count == 0:
        raise ValueError(f"Parameter '{param}' not found in {settings_path}")

    # 1. Backup
    backup = settings_path.with_suffix(".py.bak")
    shutil.copy2(settings_path, backup)
    log.info("Backup created: %s", backup.name)

    # 2. Apply change
    settings_path.write_text(new_text, encoding="utf-8")
    log.info("Edited %s: %s = %s", settings_path.name, param, value)

    if not run_tests:
        backup.unlink(missing_ok=True)
        return True, "no tests run"

    # 3. Test
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and "passed" in result.stdout:
            log.info("Tests passed after editing %s=%s", param, value)
            backup.unlink(missing_ok=True)
            return True, f"tests passed: {result.stdout.strip().splitlines()[-1]}"
        else:
            log.warning("Tests FAILED after editing %s=%s — reverting!", param, value)
            shutil.copy2(backup, settings_path)
            backup.unlink(missing_ok=True)
            return False, f"reverted: tests failed\n{result.stdout[-500:]}"
    except Exception as e:
        log.error("Test run errored — reverting: %s", e)
        shutil.copy2(backup, settings_path)
        backup.unlink(missing_ok=True)
        return False, f"reverted: test runner error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# High-level: run a full agent loop
# ─────────────────────────────────────────────────────────────────────────────
def run_agent_loop(
    agent_name: str,
    system_prompt: str,
    initial_message: str,
    max_turns: int = _MAX_TURNS,
) -> dict:
    """
    Run a full multi-turn agent loop with Claude.

    Claude can respond with:
      - Free-text analysis
      - A ```json {"actions": [...]} ``` block to execute actions

    The loop continues until Claude sends {"type": "done"} or max_turns reached.
    """
    log.info("=" * 55)
    log.info("Agent Loop: %s", agent_name)
    log.info("=" * 55)

    messages: list[dict] = []
    turn = 0
    loop_log = []

    current_message = initial_message

    while turn < max_turns:
        turn += 1
        log.info("[Turn %d/%d] Sending to Claude...", turn, max_turns)

        reply, messages = send_to_claude(system_prompt, messages, current_message)
        log.info("[Turn %d] Claude replied (%d chars)", turn, len(reply))

        # Log the full reply (truncated for storage)
        loop_log.append({
            "turn": turn,
            "claude_reply_preview": reply[:300],
            "claude_reply_full": reply,
        })

        # Extract and execute actions
        action_block = _extract_json_block(reply)
        if action_block and "actions" in action_block:
            actions = action_block["actions"]
            log.info("[Turn %d] Executing %d actions...", turn, len(actions))
            results = execute_actions(actions)
            loop_log[-1]["actions_executed"] = len(actions)
            loop_log[-1]["results"] = results

            # Check if done
            if any(r.get("action") == "done" for r in results):
                log.info("Agent %s completed (done action received).", agent_name)
                break

            # Feed results back to Claude
            current_message = (
                f"פעולות בוצעו. תוצאות:\n```json\n{json.dumps(results, ensure_ascii=False, indent=2)}\n```\n"
                "האם יש פעולות נוספות? אם הכל הושלם, שלח {\"actions\": [{\"type\": \"done\", \"summary\": \"...\"}]}"
            )
        else:
            # No actions — ask if there's more to do
            current_message = (
                "האם יש פעולות ספציפיות שתרצה לבצע? "
                "אם כן, שלח בלוק JSON עם actions. אם הסתיים, שלח done."
            )
            loop_log[-1]["actions_executed"] = 0

        if turn >= max_turns:
            log.warning("Agent %s reached max_turns=%d — stopping.", agent_name, max_turns)

    # Save loop log
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"agent_{agent_name}_{ts}.json"
    # Save without full replies to keep file small; full replies are in the preview
    log_save = [
        {k: v for k, v in entry.items() if k != "claude_reply_full"}
        for entry in loop_log
    ]
    log_path.write_text(
        json.dumps({"agent": agent_name, "turns": turn, "log": log_save}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Loop log saved → %s", log_path.name)

    return {"agent": agent_name, "turns": turn, "log": loop_log, "log_path": str(log_path)}
