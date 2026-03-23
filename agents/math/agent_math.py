"""
agents/math/agent_math.py
---------------------------
סוכן מתמטי — Math Agent

תפקיד:
  1. סוקר פונקציות ניקוד (scoring) ומציע שיפורים מתמטיים
  2. משתמש בגשר LLM כפול (Claude + GPT) לניתוח מתמטי
  3. שומר הצעות לתיקייה agents/math/math_proposals/
  4. מפרסם ל-bus לשימוש ה-Optimizer Agent

תחומי מיקוד (בסדר עדיפות):
  1. compute_distortion_score()               — analytics/signal_stack.py
  2. _half_life_quality()                      — analytics/signal_mean_reversion.py
  3. _adf_quality()                            — analytics/signal_mean_reversion.py
  4. _hurst_quality()                          — analytics/signal_mean_reversion.py
  5. compute_regime_safety_score()             — analytics/signal_regime_safety.py
  6. compute_statistical_dislocation_score()   — analytics/attribution.py

הרצה:
  python agents/math/agent_math.py --once
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        logging.FileHandler(LOG_DIR / "agent_math.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_math")

# ── Imports פנימיים ─────────────────────────────────────────────────────────
from scripts.agent_bus import AgentBus, get_bus
from agents.math.llm_bridge import DualLLMBridge
from agents.optimizer.optimization_log import get_optimization_log
from agents.shared.agent_registry import get_registry, AgentStatus

# ── תיקיית הצעות ────────────────────────────────────────────────────────────
PROPOSALS_DIR = ROOT / "agents" / "math" / "math_proposals"
PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# טעינת API keys מ-.env
# ─────────────────────────────────────────────────────────────────────────────
def _load_keys() -> dict:
    """טוען מפתחות API מקבצי .env ומגדיר ב-environment."""
    keys = {}
    for env_file in [ROOT / ".env", ROOT / "agents" / "credentials" / "api_keys.env"]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip("'\"")
                    if v:
                        keys[k.strip()] = v
                        os.environ.setdefault(k.strip(), v)
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# יעדי שיפור מתמטי — כל פונקציה עם הקשר מלא
# ─────────────────────────────────────────────────────────────────────────────
MATH_TARGETS = [
    {
        "priority": 1,
        "name": "distortion_score_logistic",
        "file": "analytics/signal_stack.py",
        "function": "compute_distortion_score",
        "description": (
            "Layer 1: Distortion Score — logistic combination of Frobenius distortion z-score, "
            "market-mode share rank, and CoC instability z-score. "
            "Formula: S^dist = sigma(a1*z_D + a2*rank(m_t) + a3*z_CoC). "
            "Current coefficients: a1=1.0, a2=0.5, a3=0.3."
        ),
        "math_context": (
            "The logistic sigmoid maps a linear combination to [0,1]. "
            "Key question: are the coefficients optimal? Should the combination be "
            "nonlinear (e.g., interaction terms, polynomial features)? "
            "Is sigmoid the right link function vs. probit or tanh?"
        ),
        "metric_sensitivity": "Affects Layer 1 distortion score -> directly impacts conviction threshold",
    },
    {
        "priority": 2,
        "name": "half_life_quality",
        "file": "analytics/signal_mean_reversion.py",
        "function": "_half_life_quality",
        "description": (
            "Layer 3 sub-component: maps OU half-life (days) to quality score [0,1]. "
            "Sweet spot: [5, 90] days. Too short = noise, too long = no reversion."
        ),
        "math_context": (
            "Current uses Gaussian-like bump with exponential tails. "
            "Consider: beta distribution, truncated Gaussian, or asymmetric logistic. "
            "The ideal mapping should peak around 20-30 days and decay smoothly."
        ),
        "metric_sensitivity": "Affects Layer 3 mean-reversion score -> trade filtering",
    },
    {
        "priority": 3,
        "name": "adf_quality",
        "file": "analytics/signal_mean_reversion.py",
        "function": "_adf_quality",
        "description": (
            "Layer 3 sub-component: maps ADF p-value to quality score [0,1]. "
            "Lower p-value = stronger evidence of stationarity. "
            "Current: piecewise linear from p<0.01->1.0 to p>0.30->~0."
        ),
        "math_context": (
            "Current mapping is piecewise linear. "
            "Consider: exponential decay, CDF-based mapping, or logistic function "
            "with inflection at p=0.05. The mapping should be convex: "
            "marginal improvement matters more at low p-values."
        ),
        "metric_sensitivity": "Affects Layer 3 -> how strongly stationarity evidence boosts conviction",
    },
    {
        "priority": 4,
        "name": "hurst_quality",
        "file": "analytics/signal_mean_reversion.py",
        "function": "_hurst_quality",
        "description": (
            "Layer 3 sub-component: maps Hurst exponent H to quality score [0,1]. "
            "H < 0.5 = mean-reverting (good), H = 0.5 = random walk, H > 0.5 = trending. "
            "Current: piecewise linear decay from H=0.30->1.0 to H>0.60->~0."
        ),
        "math_context": (
            "Ideal mapping: monotonically decreasing from H=0 to H=1, "
            "with inflection near H=0.5. Consider: logistic centered at 0.5, "
            "or a beta CDF. Should strongly penalize H > 0.6."
        ),
        "metric_sensitivity": "Affects Layer 3 -> differentiating MR from random walk from trend",
    },
    {
        "priority": 5,
        "name": "regime_penalty_functions",
        "file": "analytics/signal_regime_safety.py",
        "function": "compute_regime_safety_score",
        "description": (
            "Layer 4: Regime Safety Score — multiplicative combination of "
            "VIX, credit, correlation, and transition penalties. "
            "S^safe = prod(1 - w_i * P_i). Hard kills zero the score. "
            "All penalty sub-functions (_vix_penalty, _credit_penalty, etc.) use linear ramps."
        ),
        "math_context": (
            "Multiplicative structure means any single large penalty dominates. "
            "Consider: is linear ramp optimal? Would sigmoid ramps be smoother? "
            "Should penalty weights adapt to the current regime? "
            "Is multiplicative aggregation robust vs geometric mean or weighted minimum?"
        ),
        "metric_sensitivity": "Affects Layer 4 -> regime gating, position sizing, hard kills",
    },
    {
        "priority": 6,
        "name": "statistical_dislocation_score",
        "file": "analytics/attribution.py",
        "function": "compute_statistical_dislocation_score",
        "description": (
            "SDS = (0.75*z_strength + 0.25*disp_strength) * hl_quality. "
            "z_strength from z-score in [z_lo, z_hi], hl_quality from half-life. "
            "Parameters: sds_z_lo=0.75 (entry threshold), sds_z_hi=2.75 (full score)."
        ),
        "math_context": (
            "Linear interpolation between lo and hi may miss nonlinear effects. "
            "Consider: sigmoid mapping, power transformation, "
            "or regime-dependent thresholds."
        ),
        "metric_sensitivity": "Core attribution scoring -> conviction generation",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# עזרים — קריאת קוד פונקציה מקובץ
# ─────────────────────────────────────────────────────────────────────────────
def _read_function_code(file_rel: str, function_name: str) -> str:
    """
    קריאת קוד פונקציה מקובץ — תומך בשמות מרובים מופרדים ב-" / ".
    מחזיר את הקוד המלא של כל הפונקציות שנמצאו.
    """
    target = ROOT / file_rel
    if not target.exists():
        return f"# File not found: {file_rel}"

    content = target.read_text(encoding="utf-8")
    lines = content.split("\n")

    # תמיכה בפונקציות מרובות (e.g., "_vix_penalty / _credit_penalty")
    func_names = [fn.strip() for fn in function_name.split(" / ")]
    results = []

    for fname in func_names:
        in_func = False
        func_lines: list[str] = []
        indent = 0

        for line in lines:
            if f"def {fname}(" in line:
                in_func = True
                indent = len(line) - len(line.lstrip())
                func_lines = [line]
            elif in_func:
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#"):
                    func_lines.append(line)
                elif len(line) - len(line.lstrip()) <= indent and stripped:
                    # שורה ברמת הזחה זהה או נמוכה — סוף הפונקציה
                    if stripped.startswith("def ") or stripped.startswith("class "):
                        break
                    break
                else:
                    func_lines.append(line)

        if func_lines:
            results.append("\n".join(func_lines[:80]))  # חיתוך ל-80 שורות

    return "\n\n".join(results) if results else f"# Function '{function_name}' not found"


# ─────────────────────────────────────────────────────────────────────────────
# בניית prompt מתמטי ל-LLM
# ─────────────────────────────────────────────────────────────────────────────
def _build_math_prompt(target: dict, code: str, metrics: dict) -> str:
    """בונה prompt מובנה לניתוח מתמטי של פונקציה."""
    return f"""You are a quantitative mathematics expert specializing in financial signal processing.

## Target Function
**Name:** {target['function']}
**File:** {target['file']}
**Description:** {target['description']}
**Mathematical Context:** {target.get('math_context', '')}
**Metric sensitivity:** {target['metric_sensitivity']}

## Current Implementation
```python
{code}
```

## Current System Performance
{json.dumps(metrics, indent=2, default=str)}

## Task
Analyze the current mathematical formula and propose an improvement. Consider:
1. Is the functional form optimal? (linear vs sigmoid vs exponential vs polynomial)
2. Are the coefficients/thresholds well-calibrated?
3. Could a different parameterization improve regime-conditional performance?
4. Are there edge cases or numerical stability issues?
5. Does the function behave well asymptotically?

## Requirements
- The replacement MUST have the EXACT same function signature
- Must return the same type
- Must handle edge cases (NaN, inf, empty input)
- Include docstring with mathematical formula
- Use numpy where appropriate for vectorized operations
- Preserve any logging calls

Respond with:
1. A brief mathematical analysis (2-3 sentences)
2. Your proposed improvement as a complete Python function
3. Expected impact on performance

Format your response as:
```json
{{
    "analysis": "...",
    "proposed_code": "def function_name(...):\\n    ...",
    "expected_impact": "...",
    "confidence": 0.7
}}
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# חילוץ JSON מתשובת LLM
# ─────────────────────────────────────────────────────────────────────────────
def _extract_proposal_json(text: str) -> Optional[dict]:
    """מחלץ בלוק JSON מתשובת LLM."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _extract_python_code(text: str) -> Optional[str]:
    """מחלץ בלוק קוד Python מתשובת LLM — בוחר הארוך ביותר."""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return None
    return max(blocks, key=len).strip()


# ─────────────────────────────────────────────────────────────────────────────
# שמירת הצעה לקובץ
# ─────────────────────────────────────────────────────────────────────────────
def _save_proposal(target_name: str, proposal: dict) -> Path:
    """שמירת הצעה לתיקיית math_proposals — JSON עם metadata."""
    filename = f"{date.today().isoformat()}_{target_name}.json"
    path = PROPOSALS_DIR / filename
    path.write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _save_proposal_code(function_name: str, code: str, analysis: str, source: str) -> Path:
    """שמירת הצעת קוד כקובץ Python עם metadata ב-docstring."""
    date_str = date.today().isoformat()
    filename = f"{date_str}_{function_name}.py"
    filepath = PROPOSALS_DIR / filename

    header = f'''"""
Math Agent Proposal — {function_name}
Source: {source}
Generated: {datetime.now(timezone.utc).isoformat()}

Analysis Summary:
{analysis[:500]}
"""

'''
    filepath.write_text(header + code, encoding="utf-8")
    log.info("Proposal code saved: %s", filepath.name)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# זיהוי יעדים עדיפותיים לפי ביצועים
# ─────────────────────────────────────────────────────────────────────────────
def _prioritize_targets(metrics: dict, opt_history: list[dict]) -> list[dict]:
    """
    בוחר יעדי אופטימיזציה לפי מטריקות ביצועים.
    מחזיר 3 יעדים מסוננים.
    """
    targets = list(MATH_TARGETS)

    # אם יש מטריקות — העדפת יעדים קשורים למשטר החלש
    regime_bd = metrics.get("regime_breakdown", metrics.get("regime", {}))
    if regime_bd:
        # זיהוי משטר חלש ביותר
        weakest = None
        weakest_ic = float("inf")
        for reg, rd in regime_bd.items():
            ic = rd.get("ic_mean", 0) if isinstance(rd, dict) else 0
            if ic < weakest_ic:
                weakest_ic = ic
                weakest = reg

        if weakest and weakest in ("TENSION", "CRISIS"):
            # הקפצת regime safety לראש
            targets.sort(
                key=lambda t: 0 if "regime" in t["name"] else t["priority"]
            )
            log.info("Prioritizing regime safety targets (weakest: %s, IC=%.4f)",
                     weakest, weakest_ic)

    # סינון יעדים שכבר נותחו ב-24 שעות האחרונות
    recent_analyzed = set()
    for entry in opt_history[-10:]:
        if entry.get("agent_source") == "math" and entry.get("change_type") == "proposal":
            recent_analyzed.add(entry.get("param_name", ""))

    filtered = [t for t in targets if t["function"] not in recent_analyzed]
    if not filtered:
        filtered = targets  # אם הכל כבר נותח — חוזרים לכולם

    return filtered[:3]  # מקסימום 3 יעדים לריצה


# ─────────────────────────────────────────────────────────────────────────────
# Main run — לולאת הסוכן
# ─────────────────────────────────────────────────────────────────────────────
def run(once: bool = False) -> None:
    """
    הפעלת סוכן המתמטיקה.
    once=True → הרצה בודדת; False → לולאה אינסופית (כל 6 שעות)
    """
    _load_keys()

    registry = get_registry()
    registry.register("agent_math", role="mathematical formula optimization")

    bus = get_bus()
    opt_log = get_optimization_log()

    while True:
        try:
            registry.heartbeat("agent_math", AgentStatus.RUNNING)
            log.info("=" * 60)
            log.info("Math Agent — starting run")
            log.info("=" * 60)

            # 1. אתחול LLM bridge — טיפול חינני במפתחות חסרים
            try:
                bridge = DualLLMBridge()
                has_llm = bool(bridge.claude_client or bridge.openai_client)
            except Exception as exc:
                log.warning("LLM bridge init failed: %s — generating heuristic proposals", exc)
                bridge = None
                has_llm = False

            if not has_llm:
                log.warning("No LLM API keys available — proposals will be heuristic only")

            # 2. קריאת מטריקות מ-bus ומ-cache
            methodology = bus.latest("agent_methodology")
            metrics: dict = {}

            # ניסיון טעינה מ-cache
            cache_path = ROOT / "logs" / "last_backtest.json"
            if cache_path.exists():
                try:
                    metrics = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # fallback — מדוח methodology
            if not metrics and methodology:
                metrics = {
                    **(methodology.get("metrics", {})),
                    "regime_breakdown": methodology.get("regime_breakdown", {}),
                }

            # fallback — מ-improve_system
            if not metrics:
                improve_report = bus.latest("agent_improve_system")
                if improve_report:
                    metrics = {
                        "ic_mean": improve_report.get("bt_ic", 0),
                        "sharpe": improve_report.get("bt_sharpe", 0),
                        "hit_rate": improve_report.get("bt_hit_rate", 0),
                        "regime_breakdown": {},
                    }

            # 3. קריאת היסטוריית אופטימיזציה — לזיהוי טרנדים
            opt_history = opt_log.get_history(n=20)

            # 4. בחירת יעדים לניתוח
            targets = _prioritize_targets(metrics, opt_history)
            log.info("Selected %d targets: %s",
                     len(targets), [t["function"] for t in targets])

            # 5. ניתוח כל יעד
            proposals: list[dict] = []

            for target in targets:
                log.info("Analyzing: %s (%s)", target["name"], target["function"])

                # קריאת קוד נוכחי
                code = _read_function_code(target["file"], target["function"])

                if has_llm and bridge:
                    # שליחה ל-LLM כפול
                    prompt = _build_math_prompt(target, code, metrics)
                    try:
                        result = bridge.query_both(prompt, context="mathematical analysis")
                        best_source = result.get("best", "none")
                        best_text = result.get(best_source, "")

                        log.info("  LLM response: best=%s, reasoning=%s",
                                 best_source, result.get("reasoning", "?")[:80])

                        # חילוץ הצעה מובנית (JSON)
                        proposal_data = _extract_proposal_json(best_text)
                        if not proposal_data:
                            proposal_data = {
                                "analysis": best_text[:500],
                                "proposed_code": _extract_python_code(best_text) or "",
                                "expected_impact": "See analysis",
                                "confidence": 0.5,
                            }

                        proposal = {
                            "target": target["name"],
                            "file": target["file"],
                            "function": target["function"],
                            "current_code": code[:500],
                            "provider": best_source,
                            **proposal_data,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        proposals.append(proposal)

                        # שמירת JSON
                        path = _save_proposal(target["name"], proposal)
                        log.info("  Proposal saved: %s", path.name)

                        # שמירת קוד Python אם קיים
                        proposed_code = proposal_data.get("proposed_code", "")
                        if proposed_code and "def " in proposed_code:
                            _save_proposal_code(
                                function_name=target["function"],
                                code=proposed_code,
                                analysis=proposal_data.get("analysis", ""),
                                source=best_source,
                            )

                    except Exception as exc:
                        log.warning("  LLM query failed for %s: %s", target["name"], exc)
                        # יוצר הצעה ללא LLM
                        proposal = {
                            "target": target["name"],
                            "file": target["file"],
                            "function": target["function"],
                            "analysis": f"LLM query failed: {exc}",
                            "proposed_code": "",
                            "expected_impact": "Requires manual analysis",
                            "confidence": 0.1,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        proposals.append(proposal)
                        _save_proposal(target["name"], proposal)
                else:
                    # הצעה היוריסטית — בלי LLM
                    proposal = {
                        "target": target["name"],
                        "file": target["file"],
                        "function": target["function"],
                        "analysis": "No LLM available — generating heuristic suggestion",
                        "proposed_code": "",
                        "expected_impact": "Requires LLM for specific code proposals",
                        "confidence": 0.3,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    proposals.append(proposal)
                    _save_proposal(target["name"], proposal)

                # רישום ב-optimization log — כהצעה (לא כשינוי שבוצע)
                opt_log.log_attempt({
                    "agent_source": "math",
                    "change_type": "proposal",
                    "target_file": target["file"],
                    "param_name": target["function"],
                    "outcome": "proposed",
                    "delta_sharpe": None,
                    "delta_ic": None,
                })

            # 6. פרסום ל-bus — כל ההצעות
            bus.publish("agent_math", {
                "status": "completed",
                "proposals_count": len(proposals),
                "proposals": [
                    {
                        "target": p["target"],
                        "function": p["function"],
                        "function_name": p["function"],
                        "confidence": p.get("confidence", 0),
                        "analysis": p.get("analysis", "")[:200],
                        "has_code": bool(p.get("proposed_code")),
                        "content": p.get("proposed_code", "")[:300],
                    }
                    for p in proposals
                ],
                "targets_analyzed": [t["function"] for t in targets],
                "metrics_context": {
                    "ic_mean": metrics.get("ic_mean"),
                    "sharpe": metrics.get("sharpe"),
                },
            })

            registry.heartbeat("agent_math", AgentStatus.COMPLETED)
            log.info("Math Agent completed: %d proposals generated", len(proposals))

        except Exception as exc:
            log.exception("Math Agent error: %s", exc)
            registry.heartbeat("agent_math", AgentStatus.FAILED, error=str(exc))
            bus.publish("agent_math", {
                "status": "failed",
                "error": str(exc),
            })

        if once:
            break

        # המתנה בין ריצות — 6 שעות
        log.info("Sleeping 6 hours until next math analysis run...")
        time.sleep(21600)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(once="--once" in sys.argv)
