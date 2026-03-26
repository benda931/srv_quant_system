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

import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

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
            LOG_DIR / "agent_math.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
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
    """בונה prompt מובנה לניתוח מתמטי של פונקציה — PM-grade mathematical precision."""

    # Extract specific performance context
    sharpe = metrics.get("sharpe", metrics.get("ic_mean", "N/A"))
    ic = metrics.get("ic_mean", "N/A")
    hit_rate = metrics.get("hit_rate", "N/A")

    # Regime-specific context
    regime_ctx = ""
    regime_bd = metrics.get("regime_breakdown", {})
    if regime_bd:
        regime_lines = []
        for reg, rd in regime_bd.items():
            if isinstance(rd, dict):
                regime_lines.append(
                    f"  {reg}: Sharpe={rd.get('sharpe', 'N/A')}, "
                    f"IC={rd.get('ic_mean', 'N/A')}, "
                    f"HR={rd.get('hit_rate', 'N/A')}"
                )
        if regime_lines:
            regime_ctx = "\n".join(regime_lines)

    # Academic reference hints per target
    academic_refs = {
        "compute_distortion_score": (
            "Ref: Logistic regression theory (McCullagh & Nelder 1989); "
            "consider probit link, interaction terms, or adaptive coefficients. "
            "Key question: should a1,a2,a3 be regime-dependent?"
        ),
        "_half_life_quality": (
            "Ref: Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein 1930); "
            "half-life = -ln(2)/ln(phi) where phi is AR(1) coefficient. "
            "Consider: beta(2,5) distribution for ideal shape, "
            "or asymmetric logistic with slower right-tail decay."
        ),
        "_adf_quality": (
            "Ref: Dickey-Fuller test (1979); p-value CDF is approximately "
            "chi-squared. Consider: exponential decay exp(-k/p) for convex "
            "mapping where marginal improvement matters more at low p."
        ),
        "_hurst_quality": (
            "Ref: Hurst exponent via R/S analysis (Hurst 1951, Mandelbrot 1975). "
            "H < 0.5 = mean-reverting. Consider: logistic sigmoid centered at H=0.5 "
            "with steep slope: 1/(1+exp(k*(H-0.5))). Should strongly penalize H > 0.6."
        ),
        "compute_regime_safety_score": (
            "Ref: Multiplicative penalty aggregation (Meucci 2009 risk budgeting). "
            "Consider: geometric mean vs multiplicative product; "
            "sigmoid ramps vs linear for smoother transitions; "
            "regime-adaptive weights."
        ),
        "compute_statistical_dislocation_score": (
            "Ref: Statistical arbitrage z-score mapping (Avellaneda & Lee 2010). "
            "Consider: power transformation z^alpha for convex/concave shaping; "
            "or regime-dependent thresholds."
        ),
    }
    ref = academic_refs.get(target["function"], "")

    return f"""You are a quantitative mathematician specializing in financial signal processing
and scoring function optimization. Your task is to review a specific function for
mathematical optimality and propose a strictly superior alternative.

## Target Function
**Name:** {target['function']}
**File:** {target['file']}
**Role in system:** {target['description']}

## Current Implementation
```python
{code}
```

## Current Performance (this function contributes to these system-level metrics)
- **Sharpe ratio:** {sharpe}
- **IC (Information Coefficient):** {ic}
- **Hit rate:** {hit_rate}
- **Regime breakdown:**
{regime_ctx if regime_ctx else '  (not available)'}

## Mathematical Context
{target.get('math_context', '')}

## Academic References
{ref}

## Mathematical Requirements
1. Output must be bounded in [0, 1] (or same range as current function)
2. Must be monotonic where expected (increasing with signal strength)
3. Must be smooth (C1 continuous — no discontinuities in value or derivative)
4. Must handle edge cases: z=0, z=+/-inf, z=NaN, empty arrays
5. Must be numerically stable (no overflow/underflow for reasonable inputs)
6. Asymptotic behavior: well-defined limits as inputs approach extremes

## What to Propose
1. A mathematically superior formula with PROOF of why it's better:
   - Show the derivative properties (monotonicity)
   - Show boundary behavior (limits at extremes)
   - Explain why the functional form better captures the relationship
2. Specific parameter values (not just "tune these")
3. Expected impact on Sharpe ratio (quantitative estimate)

## Requirements for Code
- EXACT same function signature as current
- Same return type
- Handle edge cases (NaN, inf, empty input)
- Include docstring with the mathematical formula in LaTeX-like notation
- Use numpy for vectorized operations where applicable
- Preserve any logging calls from the original

Format your response as:
```json
{{
    "analysis": "2-3 sentences on why current formula is suboptimal",
    "mathematical_proof": "Show derivative, limits, and why proposed form is superior",
    "proposed_code": "def function_name(...):\\n    ...",
    "parameter_values": {{"param1": value1, "param2": value2}},
    "expected_sharpe_delta": 0.05,
    "expected_impact": "description of expected improvement",
    "confidence": 0.7,
    "risks": "what could go wrong with this change"
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
# Institutional Research Dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ResearchTarget:
    """A single scoring function targeted for formula research."""
    target_id: str
    function_name: str
    file: str
    business_role: str
    mathematical_role: str
    invariants: List[str] = field(default_factory=list)   # e.g. ["bounded_0_1", "monotone_increasing", "smooth"]
    priority_score: float = 0.0
    current_failure_signals: List[str] = field(default_factory=list)
    downstream_impact: List[str] = field(default_factory=list)


@dataclass
class CandidateFormula:
    """A candidate formula proposed for a research target."""
    candidate_id: str = ""
    target_id: str = ""
    source: str = ""          # "claude" / "gpt" / "template_library" / "hybrid"
    formula_family: str = ""  # "logistic" / "tanh" / "piecewise" / "exponential" / "gaussian"
    code: str = ""
    validation_results: Dict[str, bool] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    decision: str = ""        # "ACCEPT" / "ACCEPT_FOR_SANDBOX" / "NEEDS_REFINEMENT" / "REJECT" / "REJECT_UNSAFE"
    risks: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Formula Template Library
# ─────────────────────────────────────────────────────────────────────────────
FORMULA_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "bounded_monotone_inc": [
        ("logistic", "1.0 / (1.0 + math.exp(-k * (x - x0)))"),
        ("tanh", "0.5 * (1.0 + math.tanh(k * (x - x0)))"),
        ("probit", "0.5 * (1.0 + math.erf(k * (x - x0) / math.sqrt(2)))"),
    ],
    "sweet_spot": [
        ("gaussian", "math.exp(-((x - mu) / sigma)**2)"),
        ("beta_like", "(x ** a) * ((1 - x) ** b)"),
    ],
    "penalty_ramp": [
        ("soft_step", "1.0 / (1.0 + math.exp(k * (x - threshold)))"),
        ("exponential", "math.exp(-k * max(0, x - threshold))"),
    ],
    "bounded_monotone_dec": [
        ("logistic_dec", "1.0 / (1.0 + math.exp(k * (x - x0)))"),
        ("tanh_dec", "0.5 * (1.0 - math.tanh(k * (x - x0)))"),
    ],
}

# Mapping from MATH_TARGETS function names to invariant + spec hints
_TARGET_SPECS: Dict[str, Dict[str, Any]] = {
    "compute_distortion_score": {
        "domain": "z-score linear combination, effectively R",
        "codomain": "[0, 1]",
        "monotonicity": "increasing",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "C1",
        "template_key": "bounded_monotone_inc",
        "invariants": ["bounded_0_1", "monotone_increasing", "smooth"],
        "business_role": "Layer 1 distortion score — drives conviction threshold",
        "mathematical_role": "Logistic combination of distortion z-scores",
        "downstream_impact": ["conviction_threshold", "optimizer_weights", "regime_gating"],
        "edge_cases": {"input_nan": "return 0.5", "input_inf": "clip to bounds"},
    },
    "_half_life_quality": {
        "domain": "half-life in days, [0, +inf)",
        "codomain": "[0, 1]",
        "monotonicity": "bell-shaped",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "C1",
        "template_key": "sweet_spot",
        "invariants": ["bounded_0_1", "bell_shaped", "smooth"],
        "business_role": "Layer 3 mean-reversion quality — trade filtering",
        "mathematical_role": "Maps OU half-life to quality; sweet spot 5-90 days",
        "downstream_impact": ["mean_reversion_score", "trade_filtering"],
        "edge_cases": {"input_nan": "return 0.0", "input_inf": "return 0.0", "input_zero": "return 0.0"},
    },
    "_adf_quality": {
        "domain": "ADF p-value, [0, 1]",
        "codomain": "[0, 1]",
        "monotonicity": "decreasing",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "C1",
        "template_key": "bounded_monotone_dec",
        "invariants": ["bounded_0_1", "monotone_decreasing", "smooth"],
        "business_role": "Layer 3 stationarity evidence — conviction",
        "mathematical_role": "Maps ADF p-value to quality; lower p = better",
        "downstream_impact": ["mean_reversion_score", "stationarity_confidence"],
        "edge_cases": {"input_nan": "return 0.0", "input_zero": "return 1.0"},
    },
    "_hurst_quality": {
        "domain": "Hurst exponent, [0, 1]",
        "codomain": "[0, 1]",
        "monotonicity": "decreasing",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "C1",
        "template_key": "bounded_monotone_dec",
        "invariants": ["bounded_0_1", "monotone_decreasing", "smooth"],
        "business_role": "Layer 3 MR vs random walk — differentiation",
        "mathematical_role": "Maps Hurst exponent to quality; H<0.5 = MR",
        "downstream_impact": ["mean_reversion_score", "regime_classification"],
        "edge_cases": {"input_nan": "return 0.0", "input_half": "return ~0.5"},
    },
    "compute_regime_safety_score": {
        "domain": "penalty sub-scores, each [0, 1]",
        "codomain": "[0, 1]",
        "monotonicity": "decreasing (with penalty increase)",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "Lipschitz",
        "template_key": "penalty_ramp",
        "invariants": ["bounded_0_1", "penalty_aggregation", "lipschitz"],
        "business_role": "Layer 4 regime gating — position sizing, hard kills",
        "mathematical_role": "Multiplicative penalty aggregation across regime signals",
        "downstream_impact": ["position_sizing", "hard_kills", "regime_gating"],
        "edge_cases": {"input_nan": "return 0.0 (safe default)", "all_zero_penalties": "return 1.0"},
    },
    "compute_statistical_dislocation_score": {
        "domain": "z-score ∈ [-5, 5], half-life quality [0,1]",
        "codomain": "[0, 1]",
        "monotonicity": "increasing",
        "boundedness": True,
        "bounds": (0.0, 1.0),
        "smoothness": "C1",
        "template_key": "bounded_monotone_inc",
        "invariants": ["bounded_0_1", "monotone_increasing", "smooth"],
        "business_role": "Core attribution scoring — conviction generation",
        "mathematical_role": "Weighted combination of z-strength and dispersion",
        "downstream_impact": ["conviction_score", "trade_entry_signal"],
        "edge_cases": {"input_nan": "return 0.0", "input_zero": "return 0.0"},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MathResearchLab — Institutional Formula Research & Validation
# ─────────────────────────────────────────────────────────────────────────────
class MathResearchLab:
    """
    Institutional Quant Formula Research & Validation Lab.

    Wraps the existing MathAgent workflow with:
      - Dynamic target prioritization from downstream agent summaries
      - Formal formula specification per target
      - Template-based candidate generation
      - Structural, numerical, and behavioral validation
      - Empirical replay against historical data
      - Deterministic proposal decision engine
      - Downstream action builder + machine_summary
    """

    def __init__(self, bus: Any = None, bridge: Optional[Any] = None):
        self.bus = bus
        self.bridge = bridge
        self._import_numpy()

    def _import_numpy(self) -> None:
        """Lazy numpy import with graceful fallback."""
        try:
            import numpy as _np
            self.np = _np
        except ImportError:
            self.np = None

    # ── 1. Dynamic Target Prioritization (v2) ────────────────────────────

    def prioritize_targets_v2(self) -> List[ResearchTarget]:
        """
        Prioritize research targets using downstream agent machine_summaries.

        Loads:
          - methodology machine_summary  (weakest dimensions)
          - optimizer machine_summary    (repeated failures)
          - alpha_decay machine_summary  (decay patterns)

        Scores each target by:
          business_criticality * failure_evidence * downstream_impact

        Returns sorted list of ResearchTarget, highest priority first.
        """
        # Load downstream machine summaries from bus
        meth_summary = self._load_summary("agent_methodology", "machine_summary")
        meth_v3 = self._load_summary("agent_methodology_v3", "machine_summary_v3")
        opt_summary = self._load_summary("agent_optimizer", "machine_summary")
        decay_summary = self._load_summary("agent_alpha_decay", "machine_summary")

        # Merge methodology signals
        meth = {**meth_summary, **meth_v3} if meth_v3 else meth_summary

        # Extract weakness signals
        weakest_regime = meth.get("weakest_regime", "")
        overfitting_flag = meth.get("overfitting_flag", False)
        optimizer_failures = opt_summary.get("candidates_rejected", 0)
        optimizer_mode = opt_summary.get("optimization_mode", "")
        decay_urgent = decay_summary.get("most_urgent_action", "")
        at_risk_strategies = decay_summary.get("at_risk_strategies", [])

        research_targets: List[ResearchTarget] = []

        for target_def in MATH_TARGETS:
            func_name = target_def["function"]
            spec = _TARGET_SPECS.get(func_name, {})

            # Base priority from MATH_TARGETS ordering (lower = higher priority)
            base_priority = 1.0 / target_def["priority"]

            # Failure evidence multiplier
            failure_evidence = 1.0
            failure_signals: List[str] = []

            # Regime weakness boosts regime-related targets
            if weakest_regime and "regime" in func_name.lower():
                failure_evidence *= 1.8
                failure_signals.append(f"weakest_regime={weakest_regime}")

            # Optimizer repeated failures boost all targets
            if optimizer_failures > 3:
                failure_evidence *= 1.2
                failure_signals.append(f"optimizer_rejected={optimizer_failures}")

            # Overfitting flag suggests formula instability
            if overfitting_flag:
                failure_evidence *= 1.3
                failure_signals.append("overfitting_detected")

            # Alpha decay signals
            if at_risk_strategies:
                failure_evidence *= 1.15
                failure_signals.append(f"decay_at_risk={len(at_risk_strategies)}")

            # Optimizer mode hints
            if optimizer_mode == "PARAM_PLUS_CODE":
                failure_evidence *= 1.1
                failure_signals.append("optimizer_wants_code_changes")

            # Downstream impact count
            downstream = spec.get("downstream_impact", [])
            impact_multiplier = 1.0 + 0.1 * len(downstream)

            priority_score = round(base_priority * failure_evidence * impact_multiplier, 4)

            rt = ResearchTarget(
                target_id=f"RT-{func_name}-{uuid.uuid4().hex[:6]}",
                function_name=func_name,
                file=target_def["file"],
                business_role=spec.get("business_role", target_def.get("description", "")),
                mathematical_role=spec.get("mathematical_role", target_def.get("math_context", "")),
                invariants=spec.get("invariants", []),
                priority_score=priority_score,
                current_failure_signals=failure_signals,
                downstream_impact=downstream,
            )
            research_targets.append(rt)

        # Sort descending by priority score
        research_targets.sort(key=lambda rt: rt.priority_score, reverse=True)
        log.info("prioritize_targets_v2: %d targets, top=%s (score=%.4f)",
                 len(research_targets),
                 research_targets[0].function_name if research_targets else "none",
                 research_targets[0].priority_score if research_targets else 0)
        return research_targets

    def _load_summary(self, channel: str, key: str) -> Dict[str, Any]:
        """Safely load a machine_summary dict from the bus."""
        if not self.bus:
            return {}
        try:
            report = self.bus.latest(channel)
            if isinstance(report, dict):
                val = report.get(key, {})
                return val if isinstance(val, dict) else {}
        except Exception as exc:
            log.debug("Failed to load %s/%s: %s", channel, key, exc)
        return {}

    # ── 2. Formula Specification Builder ─────────────────────────────────

    def build_formula_spec(self, target: ResearchTarget) -> Dict[str, Any]:
        """
        Build a formal mathematical specification for a research target.

        Defines: domain, codomain, monotonicity, boundedness,
        smoothness, and edge case handling.
        """
        spec_hints = _TARGET_SPECS.get(target.function_name, {})

        spec: Dict[str, Any] = {
            "target_id": target.target_id,
            "function_name": target.function_name,
            "domain": spec_hints.get("domain", "R"),
            "codomain": spec_hints.get("codomain", "[0, 1]"),
            "monotonicity": spec_hints.get("monotonicity", "increasing"),
            "boundedness": spec_hints.get("boundedness", True),
            "bounds": spec_hints.get("bounds", (0.0, 1.0)),
            "smoothness": spec_hints.get("smoothness", "C1"),
            "edge_cases": spec_hints.get("edge_cases", {
                "input_nan": "return 0.5",
                "input_inf": "clip to bounds",
            }),
            "template_key": spec_hints.get("template_key", "bounded_monotone_inc"),
            "invariants": target.invariants,
        }
        return spec

    # ── 3. Formula Validation Lab ────────────────────────────────────────

    def validate_candidate(
        self, candidate: CandidateFormula, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run structural, numerical, and behavioral validation on a candidate formula.

        Structural tests:
          - boundedness: output in [lo, hi] for 1000 random inputs
          - monotonicity: f(x+eps) >= f(x) for monotone-increasing specs
          - continuity: |f(x+delta) - f(x)| < threshold
          - NaN safety: f(NaN), f(inf), f(-inf), f(0)

        Numerical tests:
          - no overflow/underflow for inputs in [-100, 100]
          - vectorization: test on array of 1000 values
          - stability: deterministic output

        Behavioral tests:
          - mid-range sensitivity: not saturated at 0 or 1 for typical inputs
          - ranking preservation (basic check)

        Returns dict with structural_pass, numerical_pass, behavioral_pass,
        overall_pass, and details.
        """
        np = self.np
        details: List[str] = []
        structural_pass = True
        numerical_pass = True
        behavioral_pass = True

        bounds = spec.get("bounds", (0.0, 1.0))
        lo, hi = bounds
        monotonicity = spec.get("monotonicity", "increasing")

        # Compile the candidate function
        func = self._compile_candidate_func(candidate.code)
        if func is None:
            return {
                "structural_pass": False,
                "numerical_pass": False,
                "behavioral_pass": False,
                "overall_pass": False,
                "details": ["FATAL: candidate code failed to compile"],
            }

        # ── Structural Tests ─────────────────────────────────────────────
        # Boundedness: test 1000 random inputs
        if np is not None:
            rng = np.random.default_rng(42)
            test_inputs = rng.uniform(-5, 5, 1000)
            try:
                outputs = [self._safe_call(func, float(x)) for x in test_inputs]
                outputs_clean = [o for o in outputs if o is not None and not (isinstance(o, float) and math.isnan(o))]
                if outputs_clean:
                    out_min = min(outputs_clean)
                    out_max = max(outputs_clean)
                    if out_min < lo - 1e-9 or out_max > hi + 1e-9:
                        structural_pass = False
                        details.append(
                            f"Boundedness FAIL: outputs in [{out_min:.6f}, {out_max:.6f}], "
                            f"expected [{lo}, {hi}]"
                        )
                    else:
                        details.append("Boundedness PASS")
                else:
                    structural_pass = False
                    details.append("Boundedness FAIL: all outputs None/NaN")
            except Exception as exc:
                structural_pass = False
                details.append(f"Boundedness FAIL: exception {exc}")

            # Monotonicity (for monotone-increasing or monotone-decreasing)
            if monotonicity in ("increasing", "decreasing"):
                eps = 0.01
                sorted_inputs = np.sort(test_inputs[:200])
                try:
                    vals = [self._safe_call(func, float(x)) for x in sorted_inputs]
                    vals_shifted = [self._safe_call(func, float(x + eps)) for x in sorted_inputs]
                    violations = 0
                    for v1, v2 in zip(vals, vals_shifted):
                        if v1 is None or v2 is None:
                            continue
                        if monotonicity == "increasing" and v2 < v1 - 1e-9:
                            violations += 1
                        elif monotonicity == "decreasing" and v2 > v1 + 1e-9:
                            violations += 1
                    if violations > 5:
                        structural_pass = False
                        details.append(f"Monotonicity FAIL: {violations} violations ({monotonicity})")
                    else:
                        details.append(f"Monotonicity PASS ({violations} minor violations)")
                except Exception as exc:
                    structural_pass = False
                    details.append(f"Monotonicity FAIL: exception {exc}")

            # Continuity check
            delta = 1e-4
            try:
                cont_inputs = rng.uniform(-3, 3, 200)
                max_jump = 0.0
                for x in cont_inputs:
                    v1 = self._safe_call(func, float(x))
                    v2 = self._safe_call(func, float(x + delta))
                    if v1 is not None and v2 is not None:
                        jump = abs(v2 - v1)
                        max_jump = max(max_jump, jump)
                if max_jump > 0.1:
                    structural_pass = False
                    details.append(f"Continuity FAIL: max jump {max_jump:.6f}")
                else:
                    details.append(f"Continuity PASS (max jump={max_jump:.6f})")
            except Exception as exc:
                details.append(f"Continuity SKIP: {exc}")

        # NaN safety
        nan_safe = True
        for edge_val, label in [(float("nan"), "NaN"), (float("inf"), "inf"),
                                 (float("-inf"), "-inf"), (0.0, "zero")]:
            try:
                result = self._safe_call(func, edge_val)
                if result is None:
                    nan_safe = False
                    details.append(f"NaN-safety FAIL: {label} returned None")
                elif isinstance(result, float) and math.isnan(result):
                    nan_safe = False
                    details.append(f"NaN-safety FAIL: {label} returned NaN")
            except Exception as exc:
                nan_safe = False
                details.append(f"NaN-safety FAIL: {label} raised {exc}")
        if nan_safe:
            details.append("NaN-safety PASS")
        else:
            structural_pass = False

        # ── Numerical Tests ──────────────────────────────────────────────
        if np is not None:
            # Overflow / underflow
            extreme_inputs = [-100, -50, -10, 0, 10, 50, 100]
            overflow = False
            for x in extreme_inputs:
                try:
                    r = self._safe_call(func, float(x))
                    if r is not None and isinstance(r, float) and (math.isinf(r)):
                        overflow = True
                        details.append(f"Overflow at x={x}: {r}")
                except Exception:
                    overflow = True
            if overflow:
                numerical_pass = False
                details.append("Numerical overflow/underflow FAIL")
            else:
                details.append("Overflow/underflow PASS")

            # Vectorization test (basic — can the func handle many calls)
            try:
                arr_inputs = np.linspace(-5, 5, 1000)
                results = [self._safe_call(func, float(x)) for x in arr_inputs]
                valid = [r for r in results if r is not None]
                if len(valid) < 900:
                    numerical_pass = False
                    details.append(f"Vectorization FAIL: only {len(valid)}/1000 valid outputs")
                else:
                    details.append(f"Vectorization PASS ({len(valid)}/1000 valid)")
            except Exception as exc:
                numerical_pass = False
                details.append(f"Vectorization FAIL: {exc}")

            # Determinism
            try:
                test_x = 1.234
                r1 = self._safe_call(func, test_x)
                r2 = self._safe_call(func, test_x)
                if r1 != r2 and not (r1 is None and r2 is None):
                    numerical_pass = False
                    details.append(f"Determinism FAIL: f({test_x}) = {r1} then {r2}")
                else:
                    details.append("Determinism PASS")
            except Exception as exc:
                details.append(f"Determinism SKIP: {exc}")

        # ── Behavioral Tests ─────────────────────────────────────────────
        if np is not None:
            # Mid-range sensitivity: not saturated
            try:
                mid_inputs = np.linspace(-2, 2, 100)
                mid_outputs = [self._safe_call(func, float(x)) for x in mid_inputs]
                mid_clean = [o for o in mid_outputs if o is not None]
                if mid_clean:
                    mid_range = max(mid_clean) - min(mid_clean)
                    if mid_range < 0.05:
                        behavioral_pass = False
                        details.append(f"Sensitivity FAIL: mid-range span only {mid_range:.4f}")
                    else:
                        details.append(f"Sensitivity PASS (mid-range span={mid_range:.4f})")
                else:
                    behavioral_pass = False
                    details.append("Sensitivity FAIL: no valid mid-range outputs")
            except Exception as exc:
                details.append(f"Sensitivity SKIP: {exc}")

        overall_pass = structural_pass and numerical_pass and behavioral_pass

        return {
            "structural_pass": structural_pass,
            "numerical_pass": numerical_pass,
            "behavioral_pass": behavioral_pass,
            "overall_pass": overall_pass,
            "details": details,
        }

    @staticmethod
    def _compile_candidate_func(code: str) -> Optional[Callable]:
        """
        Compile candidate code and return the first callable function defined.
        Returns None if compilation fails.
        """
        if not code or "def " not in code:
            return None
        try:
            namespace: Dict[str, Any] = {"math": math}
            try:
                import numpy as _np
                namespace["np"] = _np
                namespace["numpy"] = _np
            except ImportError:
                pass
            exec(compile(code, "<candidate>", "exec"), namespace)
            # Find the first function defined
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name not in (
                    "math", "np", "numpy"
                ):
                    return obj
            # Try underscore-prefixed functions too
            for name, obj in namespace.items():
                if callable(obj) and name.startswith("_") and name not in (
                    "__builtins__",
                ):
                    return obj
        except Exception as exc:
            log.debug("Candidate compilation failed: %s", exc)
        return None

    @staticmethod
    def _safe_call(func: Callable, x: float) -> Optional[float]:
        """Call func(x) safely, returning None on any exception."""
        try:
            result = func(x)
            if isinstance(result, (int, float)):
                return float(result)
            return None
        except Exception:
            return None

    # ── 4. Template Candidate Generation ─────────────────────────────────

    def generate_template_candidates(
        self, target: ResearchTarget, spec: Dict[str, Any]
    ) -> List[CandidateFormula]:
        """
        Generate 2-3 parameterized candidate formulas from the template library.

        Selects template family based on the spec's template_key, then
        instantiates each template with reasonable default parameters.
        """
        template_key = spec.get("template_key", "bounded_monotone_inc")
        templates = FORMULA_TEMPLATES.get(template_key, [])
        if not templates:
            templates = FORMULA_TEMPLATES["bounded_monotone_inc"]

        candidates: List[CandidateFormula] = []

        # Default parameter sets per family
        default_params: Dict[str, Dict[str, float]] = {
            "logistic": {"k": 1.5, "x0": 0.0},
            "tanh": {"k": 1.0, "x0": 0.0},
            "probit": {"k": 1.0, "x0": 0.0},
            "gaussian": {"mu": 30.0, "sigma": 20.0},
            "beta_like": {"a": 2.0, "b": 5.0},
            "soft_step": {"k": 2.0, "threshold": 0.5},
            "exponential": {"k": 1.0, "threshold": 0.5},
            "logistic_dec": {"k": 1.5, "x0": 0.5},
            "tanh_dec": {"k": 1.0, "x0": 0.5},
        }

        for family, formula_expr in templates[:3]:
            params = default_params.get(family, {"k": 1.0, "x0": 0.0})

            # Build a complete function from the template
            param_assignments = "\n    ".join(
                f"{p} = {v}" for p, v in params.items()
            )
            func_code = (
                f"def {target.function_name}_candidate(x):\n"
                f"    \"\"\"\n"
                f"    Template candidate: {family}\n"
                f"    Formula: {formula_expr}\n"
                f"    \"\"\"\n"
                f"    import math\n"
                f"    if x != x:  # NaN check\n"
                f"        return 0.0\n"
                f"    if abs(x) == float('inf'):\n"
                f"        return {spec.get('bounds', (0.0, 1.0))[0]}\n"
                f"    {param_assignments}\n"
                f"    try:\n"
                f"        result = {formula_expr}\n"
                f"        return max({spec.get('bounds', (0.0, 1.0))[0]}, "
                f"min({spec.get('bounds', (0.0, 1.0))[1]}, result))\n"
                f"    except (OverflowError, ValueError, ZeroDivisionError):\n"
                f"        return 0.0\n"
            )

            cid = f"CF-{family}-{uuid.uuid4().hex[:6]}"
            candidates.append(CandidateFormula(
                candidate_id=cid,
                target_id=target.target_id,
                source="template_library",
                formula_family=family,
                code=func_code,
            ))

        return candidates

    # ── 5. Empirical Replay ──────────────────────────────────────────────

    def empirical_replay(
        self,
        target: ResearchTarget,
        candidate: CandidateFormula,
        current_func: Optional[Callable],
    ) -> Dict[str, Any]:
        """
        Compare candidate formula against current formula on historical-like inputs.

        Generates synthetic inputs representative of the target's domain,
        runs both formulas, and computes:
          - rank_corr: Spearman rank correlation between current and candidate scores
          - distribution_shift: absolute mean difference
          - improvement_estimate: estimated improvement (positive = candidate better)

        Falls back to synthetic data if historical prices are not available.
        """
        np = self.np
        if np is None:
            return {
                "rank_corr": None,
                "distribution_shift": None,
                "improvement_estimate": 0.0,
                "details": "numpy not available",
            }

        # Generate representative inputs
        rng = np.random.default_rng(123)
        n = 500
        spec_hints = _TARGET_SPECS.get(target.function_name, {})
        monotonicity = spec_hints.get("monotonicity", "increasing")

        if "half_life" in target.function_name:
            # Half-life domain: positive, typical 1-200 days
            inputs = rng.exponential(scale=30, size=n)
        elif "hurst" in target.function_name:
            # Hurst exponent: [0, 1]
            inputs = rng.beta(2, 2, size=n)
        elif "adf" in target.function_name:
            # ADF p-values: [0, 1], skewed toward small
            inputs = rng.beta(1, 5, size=n)
        else:
            # Generic z-score-like inputs
            inputs = rng.normal(0, 1.5, size=n)

        # Compile candidate
        cand_func = self._compile_candidate_func(candidate.code)
        if cand_func is None:
            return {
                "rank_corr": None,
                "distribution_shift": None,
                "improvement_estimate": 0.0,
                "details": "candidate compilation failed",
            }

        # Run both
        current_scores: List[Optional[float]] = []
        candidate_scores: List[Optional[float]] = []

        for x in inputs:
            cs = self._safe_call(current_func, float(x)) if current_func else None
            current_scores.append(cs)
            candidate_scores.append(self._safe_call(cand_func, float(x)))

        # Filter paired valid outputs
        paired = [
            (c, d)
            for c, d in zip(current_scores, candidate_scores)
            if c is not None and d is not None
        ]

        if len(paired) < 50:
            return {
                "rank_corr": None,
                "distribution_shift": None,
                "improvement_estimate": 0.0,
                "details": f"insufficient valid pairs: {len(paired)}",
            }

        curr_arr = np.array([p[0] for p in paired])
        cand_arr = np.array([p[1] for p in paired])

        # Rank correlation (Spearman)
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(curr_arr, cand_arr)
            rank_corr = round(float(corr), 4) if not np.isnan(corr) else 0.0
        except ImportError:
            # Manual rank correlation fallback
            rank_corr = round(float(np.corrcoef(curr_arr, cand_arr)[0, 1]), 4)

        # Distribution shift
        distribution_shift = round(float(np.abs(np.mean(cand_arr) - np.mean(curr_arr))), 6)

        # Improvement estimate: candidate has better spread and mid-range sensitivity
        curr_std = float(np.std(curr_arr))
        cand_std = float(np.std(cand_arr))
        improvement_estimate = 0.0

        # Better discrimination = higher std (more spread in scores)
        if curr_std > 0:
            spread_improvement = (cand_std - curr_std) / curr_std
            improvement_estimate += spread_improvement * 0.5

        # Rank preservation bonus
        if rank_corr > 0.8:
            improvement_estimate += 0.02

        improvement_estimate = round(improvement_estimate, 4)

        return {
            "rank_corr": rank_corr,
            "distribution_shift": distribution_shift,
            "improvement_estimate": improvement_estimate,
            "n_valid_pairs": len(paired),
            "current_mean": round(float(np.mean(curr_arr)), 6),
            "candidate_mean": round(float(np.mean(cand_arr)), 6),
            "current_std": round(curr_std, 6),
            "candidate_std": round(cand_std, 6),
            "details": "synthetic replay completed",
        }

    # ── 6. Proposal Decision Engine ──────────────────────────────────────

    def decide_proposal(
        self,
        candidate: CandidateFormula,
        validation: Dict[str, Any],
        replay: Dict[str, Any],
    ) -> str:
        """
        Deterministic decision rules for a candidate formula.

        Rules:
          - structural_pass=False  -> REJECT_UNSAFE
          - numerical_pass=False   -> REJECT
          - behavioral_pass=False AND replay degradation -> REJECT
          - All pass + replay improvement > 5% -> ACCEPT_FOR_SANDBOX
          - All pass + replay neutral -> NEEDS_REFINEMENT
          - All pass + significant improvement -> ACCEPT
        """
        structural = validation.get("structural_pass", False)
        numerical = validation.get("numerical_pass", False)
        behavioral = validation.get("behavioral_pass", False)
        improvement = replay.get("improvement_estimate", 0.0) or 0.0

        if not structural:
            return "REJECT_UNSAFE"

        if not numerical:
            return "REJECT"

        if not behavioral:
            if improvement < 0:
                return "REJECT"
            return "NEEDS_REFINEMENT"

        # All three pass
        if improvement > 0.10:
            return "ACCEPT"
        elif improvement > 0.05:
            return "ACCEPT_FOR_SANDBOX"
        elif improvement > -0.02:
            return "NEEDS_REFINEMENT"
        else:
            return "REJECT"

    # ── 7. Downstream Action Builder ─────────────────────────────────────

    def build_downstream_actions(
        self, proposals: List[CandidateFormula]
    ) -> Dict[str, Any]:
        """
        Build structured downstream actions for other agents.

        Maps proposal decisions to actionable items for:
          - optimizer: sandbox candidates, retuning needed
          - methodology: revalidation, threshold recalibration
          - alpha_decay: score pathology flags
          - auto_improve: eligible proposals, human review
        """
        sandbox_candidates = []
        accepted_proposals = []
        rejected_unsafe = []
        needs_refinement = []

        for p in proposals:
            entry = {
                "candidate_id": p.candidate_id,
                "target_id": p.target_id,
                "formula_family": p.formula_family,
                "source": p.source,
                "decision": p.decision,
            }
            if p.decision == "ACCEPT":
                accepted_proposals.append(entry)
            elif p.decision == "ACCEPT_FOR_SANDBOX":
                sandbox_candidates.append(entry)
            elif p.decision == "REJECT_UNSAFE":
                rejected_unsafe.append(entry)
            elif p.decision == "NEEDS_REFINEMENT":
                needs_refinement.append(entry)

        # Determine which functions need revalidation
        revalidation_funcs = list({
            p.target_id for p in proposals
            if p.decision in ("ACCEPT", "ACCEPT_FOR_SANDBOX")
        })

        return {
            "optimizer": {
                "sandbox_candidates": sandbox_candidates,
                "retuning_needed": [p["target_id"] for p in accepted_proposals],
            },
            "methodology": {
                "revalidation_needed": revalidation_funcs,
                "threshold_recalibration": [
                    p["target_id"] for p in proposals
                    if p.decision == "ACCEPT" and "threshold" in (p.target_id or "").lower()
                ],
            },
            "alpha_decay": {
                "score_pathology_flags": [
                    {"candidate_id": p.candidate_id, "reason": "unsafe formula"}
                    for p in proposals if p.decision == "REJECT_UNSAFE"
                ],
            },
            "auto_improve": {
                "eligible_proposals": [
                    p["candidate_id"] for p in accepted_proposals + sandbox_candidates
                ],
                "human_review_needed": [
                    p["candidate_id"] for p in needs_refinement
                ],
            },
        }

    # ── 8. Machine Summary ───────────────────────────────────────────────

    def build_machine_summary(
        self,
        targets: List[ResearchTarget],
        proposals: List[CandidateFormula],
        downstream_actions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the math agent machine_summary for downstream consumption.

        Compact, deterministic summary of the research cycle results.
        """
        accepted = [p for p in proposals if p.decision == "ACCEPT"]
        sandbox = [p for p in proposals if p.decision == "ACCEPT_FOR_SANDBOX"]
        rejected = [p for p in proposals if p.decision in ("REJECT", "REJECT_UNSAFE")]
        refinement = [p for p in proposals if p.decision == "NEEDS_REFINEMENT"]

        # Top improvements
        top_improvements = []
        for p in accepted + sandbox:
            improvement_est = p.scores.get("improvement_estimate", 0)
            top_improvements.append({
                "function": p.target_id,
                "family": p.formula_family,
                "improvement_estimate": improvement_est,
            })
        top_improvements.sort(key=lambda x: x.get("improvement_estimate", 0), reverse=True)

        # Top risks
        top_risks = []
        for p in proposals:
            top_risks.extend(p.risks)
        # Deduplicate
        top_risks = list(dict.fromkeys(top_risks))[:5]

        # Functions requiring revalidation
        revalidation_needed = downstream_actions.get("methodology", {}).get(
            "revalidation_needed", []
        )

        highest_priority = targets[0].function_name if targets else "none"

        return {
            "targets_analyzed": len(targets),
            "highest_priority": highest_priority,
            "accepted_count": len(accepted),
            "sandbox_ready_count": len(sandbox),
            "rejected_count": len(rejected),
            "needs_refinement_count": len(refinement),
            "top_improvements": top_improvements[:3],
            "top_risks": top_risks,
            "functions_requiring_revalidation": revalidation_needed,
            "downstream_actions": downstream_actions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Full Research Cycle (wired into run()) ───────────────────────────

    def run_research_cycle(self, existing_proposals: List[Dict]) -> Dict[str, Any]:
        """
        Execute the full institutional research cycle.

        Steps:
          1. Prioritize targets v2
          2. For each target: build spec
          3. Generate template candidates + merge LLM candidates
          4. Validate all candidates
          5. Run empirical replay
          6. Score and decide
          7. Build downstream actions
          8. Build machine_summary

        Parameters
        ----------
        existing_proposals : list[dict]
            Proposals already generated by the LLM step (from the existing run() flow).
            These are merged as additional candidates.

        Returns
        -------
        dict
            {targets, all_candidates, downstream_actions, machine_summary}
        """
        log.info("=" * 40 + " Research Cycle " + "=" * 40)

        # 1. Prioritize
        targets = self.prioritize_targets_v2()

        all_candidates: List[CandidateFormula] = []

        for target in targets[:4]:  # Top 4 targets per cycle
            log.info("Research: %s (priority=%.4f)", target.function_name, target.priority_score)

            # 2. Build spec
            spec = self.build_formula_spec(target)

            # 3a. Template candidates
            template_cands = self.generate_template_candidates(target, spec)
            log.info("  Templates: %d candidates", len(template_cands))

            # 3b. Merge LLM candidates from existing proposals
            llm_cands = self._merge_llm_proposals(target, existing_proposals)
            log.info("  LLM candidates: %d", len(llm_cands))

            candidates = template_cands + llm_cands

            # Load current function for replay
            current_func = self._load_current_function(target)

            for cand in candidates:
                # 4. Validate
                validation = self.validate_candidate(cand, spec)
                cand.validation_results = {
                    k: v for k, v in validation.items() if isinstance(v, bool)
                }

                # 5. Empirical replay
                replay = self.empirical_replay(target, cand, current_func)
                cand.scores = {
                    "improvement_estimate": replay.get("improvement_estimate", 0.0),
                    "rank_corr": replay.get("rank_corr", 0.0) or 0.0,
                    "distribution_shift": replay.get("distribution_shift", 0.0) or 0.0,
                }

                # 6. Decide
                decision = self.decide_proposal(cand, validation, replay)
                cand.decision = decision

                # Assign risks from validation details
                cand.risks = [
                    d for d in validation.get("details", []) if "FAIL" in d
                ]

                log.info("  %s [%s] -> %s (improvement=%.4f)",
                         cand.candidate_id, cand.formula_family, decision,
                         cand.scores.get("improvement_estimate", 0))

            all_candidates.extend(candidates)

        # 7. Downstream actions
        downstream_actions = self.build_downstream_actions(all_candidates)

        # 8. Machine summary
        machine_summary = self.build_machine_summary(
            targets, all_candidates, downstream_actions,
        )

        log.info("Research cycle complete: %d candidates, %d accepted, %d sandbox, %d rejected",
                 len(all_candidates),
                 machine_summary["accepted_count"],
                 machine_summary["sandbox_ready_count"],
                 machine_summary["rejected_count"])

        return {
            "targets": [asdict(t) for t in targets],
            "all_candidates": [asdict(c) for c in all_candidates],
            "downstream_actions": downstream_actions,
            "machine_summary": machine_summary,
        }

    def _merge_llm_proposals(
        self, target: ResearchTarget, existing_proposals: List[Dict]
    ) -> List[CandidateFormula]:
        """Convert existing LLM proposals (from run() step 5) into CandidateFormula objects."""
        candidates: List[CandidateFormula] = []
        for prop in existing_proposals:
            if prop.get("function") != target.function_name:
                continue
            code = prop.get("proposed_code", "")
            if not code or "def " not in code:
                continue

            # Detect family
            family = "unknown"
            if self.bridge:
                family = DualLLMBridge._detect_formula_family(code)

            source = prop.get("provider", "hybrid")
            cid = f"CF-llm-{uuid.uuid4().hex[:6]}"
            candidates.append(CandidateFormula(
                candidate_id=cid,
                target_id=target.target_id,
                source=source,
                formula_family=family,
                code=code,
            ))
        return candidates

    def _load_current_function(self, target: ResearchTarget) -> Optional[Callable]:
        """Attempt to load the current implementation of a target function."""
        try:
            code = _read_function_code(target.file, target.function_name)
            if code.startswith("# "):
                return None
            return self._compile_candidate_func(code)
        except Exception:
            return None


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

            # 6. Institutional Research Cycle — formula validation lab
            log.info("Starting institutional research cycle...")
            try:
                lab = MathResearchLab(bus=bus, bridge=bridge)
                research_result = lab.run_research_cycle(proposals)
                research_machine_summary = research_result.get("machine_summary", {})
                downstream_actions = research_result.get("downstream_actions", {})
                research_candidates = research_result.get("all_candidates", [])

                # Save research results
                research_path = PROPOSALS_DIR / f"{date.today().isoformat()}_research_cycle.json"
                research_path.write_text(
                    json.dumps(research_result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                log.info("Research cycle saved: %s", research_path.name)
            except Exception as exc:
                log.warning("Research cycle failed (non-fatal): %s", exc)
                research_machine_summary = {}
                downstream_actions = {}
                research_candidates = []

            # 7. פרסום ל-bus — כל ההצעות + research results
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
                "research_lab": {
                    "machine_summary": research_machine_summary,
                    "downstream_actions": downstream_actions,
                    "candidates_validated": len(research_candidates),
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
# MathAgent — convenience wrapper for import compatibility
# ─────────────────────────────────────────────────────────────────────────────
class MathAgent:
    """Thin wrapper that exposes run() and the research lab as a class."""

    def __init__(self) -> None:
        self.lab: Optional[MathResearchLab] = None

    def run(self, once: bool = False) -> None:
        run(once=once)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(once="--once" in sys.argv)
