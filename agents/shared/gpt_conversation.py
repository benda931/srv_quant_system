"""
agents/shared/gpt_conversation.py
====================================
Institutional LLM Interaction Governance Layer

Provides two tiers:
  1. Legacy GPTConversation class — focused multi-step GPT queries (unchanged).
  2. Governance extensions — InteractionMode, InteractionRecord, PromptContract,
     ResponseValidator, StructuredParser, governed_query, retry policy,
     token/cost governance, interaction history, agent profiles, and
     machine_summary.

All LLM outputs are tagged ``advisory_only=True``.  No LLM response may
promote itself to an authoritative decision; every response passes through
validation and is recorded for auditability.

Usage (legacy, still works):
  from agents.shared.gpt_conversation import GPTConversation
  conv = GPTConversation()
  analysis = conv.full_analysis(metrics, params)

Usage (governed):
  conv = GPTConversation()
  result = conv.governed_query(prompt, mode="formula_proposal",
                               agent_name="math", correlation_id="run-42")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════
#  Institutional LLM Interaction Governance — enums, config, dataclasses
# ═══════════════════════════════════════════════════════════════════════

class InteractionMode(str, Enum):
    """Typed modes that govern every LLM call's temperature, budget, retries."""
    RESEARCH_COMMENTARY = "research_commentary"
    IDEA_GENERATION = "idea_generation"
    FORMULA_PROPOSAL = "formula_proposal"
    CODE_PATCH_IDEA = "code_patch_idea"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    ROOT_CAUSE_SUMMARY = "root_cause_summary"
    PM_SUMMARY = "pm_summary"
    STRUCTURED_JSON = "structured_json"


MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "research_commentary":   {"max_tokens": 1000, "temperature": 0.7, "max_retries": 2, "timeout": 30},
    "idea_generation":       {"max_tokens": 800,  "temperature": 0.8, "max_retries": 2, "timeout": 30},
    "formula_proposal":      {"max_tokens": 1500, "temperature": 0.3, "max_retries": 3, "timeout": 45},
    "code_patch_idea":       {"max_tokens": 2000, "temperature": 0.2, "max_retries": 2, "timeout": 60},
    "architecture_analysis": {"max_tokens": 1500, "temperature": 0.5, "max_retries": 2, "timeout": 45},
    "root_cause_summary":    {"max_tokens": 1000, "temperature": 0.4, "max_retries": 2, "timeout": 30},
    "pm_summary":            {"max_tokens": 1000, "temperature": 0.5, "max_retries": 2, "timeout": 30},
    "structured_json":       {"max_tokens": 1000, "temperature": 0.1, "max_retries": 3, "timeout": 30},
}

AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "math":         {"default_mode": "formula_proposal",      "max_tokens": 1500, "temperature": 0.3},
    "architect":    {"default_mode": "architecture_analysis",  "max_tokens": 1500, "temperature": 0.5},
    "optimizer":    {"default_mode": "idea_generation",        "max_tokens": 800,  "temperature": 0.7},
    "methodology":  {"default_mode": "pm_summary",             "max_tokens": 1000, "temperature": 0.6},
    "auto_improve": {"default_mode": "idea_generation",        "max_tokens": 800,  "temperature": 0.7},
    "data_scout":   {"default_mode": "research_commentary",    "max_tokens": 600,  "temperature": 0.6},
}

# Backoff schedule for retries (seconds)
_BACKOFF_SCHEDULE = [2, 5, 15]

# Maximum interaction records kept in memory
_MAX_HISTORY = 500


@dataclass
class InteractionRecord:
    """Immutable audit record for every governed LLM interaction."""
    interaction_id: str
    timestamp: str
    requesting_agent: str
    task_type: str
    model_name: str
    prompt_hash: str
    correlation_id: str
    token_usage: Dict[str, int]
    latency_ms: int
    retry_count: int
    parse_status: str       # OK | MALFORMED | REPAIR_SUCCEEDED | FAILED
    validation_status: str  # VALID | INVALID | PARTIAL
    advisory_only: bool     # always True
    response_summary: str
    failure_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_gpt_client():
    """Get OpenAI client with API key from .env or credentials."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        for env_file in [ROOT / ".env", ROOT / "agents" / "credentials" / "api_keys.env"]:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENAI_API_KEY") and "=" in line:
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                        if api_key:
                            os.environ["OPENAI_API_KEY"] = api_key
                            break
    if not api_key:
        return None
    try:
        import openai
        return openai.OpenAI(api_key=api_key)
    except Exception:
        return None


class GPTConversation:
    """
    Multi-step focused GPT conversation for agent intelligence.

    Each method sends a SHORT, focused query and gets a SHORT, focused response.
    Conversation history is maintained for context.
    """

    def __init__(self, system_role: str = "quantitative analyst"):
        self.client = _get_gpt_client()
        self.history: List[Dict[str, str]] = []
        self.system_prompt = (
            f"You are a {system_role} for a hedge fund DSS. "
            "Give SHORT, SPECIFIC answers. No general advice. "
            "When suggesting parameter changes, give EXACT values. "
            "When analyzing, cite SPECIFIC numbers from the data. "
            "Max 150 words per response."
        )

    @property
    def available(self) -> bool:
        return self.client is not None

    def _query(self, user_msg: str) -> str:
        """Send one focused query to GPT."""
        if not self.client:
            return ""

        self.history.append({"role": "user", "content": user_msg})

        try:
            messages = [{"role": "system", "content": self.system_prompt}] + self.history[-8:]  # Keep last 4 turns
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                temperature=0.2,
            )
            reply = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            log.info("GPT: %s", reply[:80])
            return reply
        except Exception as e:
            log.warning("GPT query failed: %s", e)
            return ""

    # ── Focused analysis steps ────────────────────────────────

    def diagnose_performance(self, metrics: Dict) -> str:
        """Step 1: What's the main problem?"""
        return self._query(
            f"Backtest: Sharpe={metrics.get('sharpe', 'N/A')}, "
            f"WR={metrics.get('win_rate', 'N/A')}, "
            f"MaxDD={metrics.get('max_dd', 'N/A')}, "
            f"Trades={metrics.get('trades', 'N/A')}. "
            f"Regime breakdown: {json.dumps(metrics.get('regime_breakdown', {}), default=str)}. "
            f"What is the SINGLE biggest problem?"
        )

    def suggest_parameter(self, diagnosis: str, current_params: Dict) -> str:
        """Step 2: What specific parameter should change?"""
        # Only send 5 most relevant params
        key_params = {k: v for k, v in list(current_params.items())[:8]}
        return self._query(
            f"Problem: {diagnosis[:200]}. "
            f"Key params: {json.dumps(key_params)}. "
            f"Which ONE parameter should change, to what value, and why?"
        )

    def estimate_impact(self, suggestion: str) -> str:
        """Step 3: What's the expected impact?"""
        return self._query(
            f"Suggestion: {suggestion[:200]}. "
            f"Estimate: 1) Sharpe change, 2) WR change, 3) risk to watch for."
        )

    def prioritize_actions(self, all_suggestions: List[str]) -> str:
        """Step 4: What to do first?"""
        items = "\n".join(f"- {s[:100]}" for s in all_suggestions[:5])
        return self._query(
            f"These improvements were suggested:\n{items}\n"
            f"Rank them 1-3 by expected impact. Be specific."
        )

    def analyze_regime_shift(self, current: str, forecast: Dict) -> str:
        """Quick regime analysis."""
        return self._query(
            f"Current regime: {current}. "
            f"5-day forecast: {json.dumps(forecast, default=str)}. "
            f"What should we do with positions? Specific sizing advice."
        )

    def evaluate_trade(self, ticker: str, direction: str, z: float, conviction: float) -> str:
        """Quick trade evaluation."""
        return self._query(
            f"Trade: {direction} {ticker}, z={z:+.2f}, conviction={conviction:.3f}. "
            f"Is this a good entry? What's the key risk?"
        )

    def review_exit(self, ticker: str, pnl_pct: float, days_held: int, reason: str) -> str:
        """Quick exit review."""
        return self._query(
            f"Exiting {ticker}: P&L={pnl_pct:+.2%}, held {days_held}d, reason={reason}. "
            f"Was this the right exit? Should we have held longer or exited earlier?"
        )

    # ── Full analysis pipeline ────────────────────────────────

    def full_analysis(self, metrics: Dict, params: Dict) -> Dict[str, str]:
        """Run the complete 4-step analysis pipeline."""
        results = {}

        results["diagnosis"] = self.diagnose_performance(metrics)
        if results["diagnosis"]:
            results["suggestion"] = self.suggest_parameter(results["diagnosis"], params)
        if results.get("suggestion"):
            results["impact"] = self.estimate_impact(results["suggestion"])

        return results

    def get_conversation_log(self) -> List[Dict]:
        """Return the full conversation history for logging."""
        return list(self.history)

    # ═══════════════════════════════════════════════════════════════════
    #  Institutional LLM Interaction Governance Layer
    # ═══════════════════════════════════════════════════════════════════

    def _ensure_governance_state(self) -> None:
        """Lazily initialise governance bookkeeping (backward compat)."""
        if not hasattr(self, "_interaction_history"):
            self._interaction_history: List[InteractionRecord] = []

    # ── 1. Prompt Contract ─────────────────────────────────────────────

    def build_prompt_contract(
        self,
        mode: str,
        agent_name: str,
        evidence: str,
        constraints: Optional[List[str]] = None,
    ) -> str:
        """Build a governed system+user prompt for *mode* / *agent_name*.

        Returns the full user prompt string.  The caller is expected to use
        ``_governed_system_prompt`` as the system message.
        """
        cfg = MODE_CONFIG.get(mode, MODE_CONFIG["research_commentary"])
        profile = AGENT_PROFILES.get(agent_name, {})

        task_instructions = {
            "research_commentary": "Provide concise research commentary on the evidence below.",
            "idea_generation": "Generate actionable improvement ideas based on the evidence.",
            "formula_proposal": "Propose a mathematical formula or quantitative rule. Include a Python code block.",
            "code_patch_idea": "Suggest a code-level change. Include a Python code block with the patch.",
            "architecture_analysis": "Analyse the system architecture implied by the evidence.",
            "root_cause_summary": "Identify root causes for the issues described in the evidence.",
            "pm_summary": "Summarise the project/methodology status for a programme manager.",
            "structured_json": "Return your answer as a single valid JSON object inside a ```json``` block.",
        }

        constraint_block = "\n".join(
            f"- {c}" for c in (constraints or [
                "Do not claim authority over trading decisions.",
                "Do not make promotion or demotion decisions.",
                "All outputs are advisory only.",
            ])
        )

        truncated_evidence = self.truncate_evidence(evidence, max_tokens=cfg["max_tokens"] * 2)

        prompt = (
            f"## Task ({mode})\n"
            f"{task_instructions.get(mode, task_instructions['research_commentary'])}\n\n"
            f"## Evidence\n{truncated_evidence}\n\n"
            f"## Constraints\n{constraint_block}\n\n"
            "## Format\n"
            "Be concise. Stay under 300 words unless the task requires code."
        )
        return prompt

    @property
    def _governed_system_prompt(self) -> str:
        return (
            "You are a quantitative research advisor. "
            "Your output is advisory only — it never constitutes an autonomous decision. "
            "Give SHORT, SPECIFIC answers grounded in the evidence provided. "
            "Cite numbers. No vague generalities."
        )

    # ── 2. Response Validator ──────────────────────────────────────────

    def validate_response(
        self,
        response_text: str,
        mode: str,
        expected_schema: Optional[Dict[str, type]] = None,
    ) -> Dict[str, Any]:
        """Validate an LLM response against mode-specific rules.

        Returns ``{valid, parsed, issues, repair_attempted}``.
        """
        issues: List[str] = []
        parsed: Any = None
        repair_attempted = False

        # Empty check
        if not response_text or not response_text.strip():
            return {"valid": False, "parsed": None, "issues": ["Empty response"], "repair_attempted": False}

        cfg = MODE_CONFIG.get(mode, MODE_CONFIG["research_commentary"])
        max_chars = cfg["max_tokens"] * 5  # rough char estimate

        if len(response_text) > max_chars:
            issues.append(f"Response length {len(response_text)} exceeds budget ~{max_chars}")

        # Mode-specific checks
        if mode == "structured_json":
            parsed_result = self.parse_structured_response(response_text, required_fields=expected_schema)
            if "error" in parsed_result:
                issues.append(f"JSON parse failed: {parsed_result['error']}")
                repair_attempted = True
            else:
                parsed = parsed_result
                # Validate schema fields if provided
                if expected_schema:
                    for field_name in expected_schema:
                        if field_name not in parsed:
                            issues.append(f"Missing required field: {field_name}")

        elif mode == "formula_proposal":
            if "```" not in response_text and "def " not in response_text:
                issues.append("Formula proposal should contain a Python code block")

        elif mode == "code_patch_idea":
            if "```" not in response_text and "def " not in response_text:
                issues.append("Code patch idea should contain a code block")

        valid = len(issues) == 0
        status = "VALID" if valid else ("PARTIAL" if parsed is not None else "INVALID")
        return {
            "valid": valid,
            "parsed": parsed,
            "issues": issues,
            "repair_attempted": repair_attempted,
            "validation_status": status,
        }

    # ── 3. Structured Parser ──────────────────────────────────────────

    def parse_structured_response(
        self,
        response_text: str,
        required_fields: Optional[Dict[str, type]] = None,
    ) -> Dict[str, Any]:
        """Extract JSON from an LLM response.

        Looks for ``json ... `` fenced blocks first, then tries raw parse.
        Returns the parsed dict or ``{"error": "...", "raw": response_text}``.
        """
        # Try fenced JSON block
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        raw_json = match.group(1) if match else response_text.strip()

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            # Attempt brace-extraction fallback
            brace_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    return {"error": str(exc), "raw": response_text}
            else:
                return {"error": str(exc), "raw": response_text}

        if not isinstance(parsed, dict):
            return {"error": "Parsed value is not a dict", "raw": response_text}

        if required_fields:
            missing = [f for f in required_fields if f not in parsed]
            if missing:
                return {"error": f"Missing fields: {missing}", "raw": response_text, **parsed}

        return parsed

    # ── 4. Token / Cost Governance ────────────────────────────────────

    def estimate_token_cost(self, prompt: str, mode: str) -> Dict[str, Any]:
        """Rough token estimate and budget check for *prompt* under *mode*."""
        estimated = len(prompt) // 4
        cfg = MODE_CONFIG.get(mode, MODE_CONFIG["research_commentary"])
        budget = cfg["max_tokens"]
        return {
            "estimated_tokens": estimated,
            "within_budget": estimated <= budget * 3,  # prompt can be ~3x completion budget
            "budget_limit": budget,
        }

    def truncate_evidence(self, evidence: str, max_tokens: int = 2000) -> str:
        """Smart-truncate *evidence* to fit within *max_tokens* (approx).

        Preserves structure: keeps the most recent lines when truncation is
        needed (financial data is typically appended chronologically).
        """
        max_chars = max_tokens * 4
        if len(evidence) <= max_chars:
            return evidence

        lines = evidence.splitlines()
        # Keep header (first 5 lines) + tail (most recent data)
        header = lines[:5]
        budget_remaining = max_chars - sum(len(l) for l in header) - 40
        tail: List[str] = []
        for line in reversed(lines[5:]):
            if budget_remaining - len(line) < 0:
                break
            tail.insert(0, line)
            budget_remaining -= len(line) + 1

        return "\n".join(header + ["... [truncated] ..."] + tail)

    # ── 5. Retry Policy ───────────────────────────────────────────────

    def _retry_with_policy(
        self,
        func: Callable[[], Any],
        mode: str,
    ) -> Any:
        """Execute *func* with mode-governed retry + exponential backoff.

        On structured-JSON malformed output, injects one repair retry.
        Never infinite — bounded by MODE_CONFIG max_retries.
        """
        cfg = MODE_CONFIG.get(mode, MODE_CONFIG["research_commentary"])
        max_retries = cfg.get("max_retries", 2)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
                    log.warning(
                        "Governed retry %d/%d after %.1fs — %s",
                        attempt + 1, max_retries, backoff, exc,
                    )
                    time.sleep(backoff)

        # All retries exhausted
        return None

    # ── 6. Governed Query — the main institutional method ─────────────

    def governed_query(
        self,
        prompt: str,
        mode: str,
        agent_name: str,
        correlation_id: str = "",
        expected_schema: Optional[Dict[str, type]] = None,
    ) -> Dict[str, Any]:
        """Institutional governed LLM query.

        1. Build prompt contract
        2. Apply mode config (temperature, max_tokens, timeout)
        3. Send to GPT with retry policy
        4. Parse response
        5. Validate response
        6. Record interaction
        7. Return: {response, parsed, record, advisory_only}
        """
        self._ensure_governance_state()

        interaction_id = str(uuid.uuid4())
        ts_start = time.time()
        prompt_hash = hashlib.sha256(prompt[:500].encode("utf-8", errors="replace")).hexdigest()

        # Resolve mode string
        mode_str = mode.value if isinstance(mode, InteractionMode) else str(mode)
        cfg = MODE_CONFIG.get(mode_str, MODE_CONFIG["research_commentary"])
        profile = AGENT_PROFILES.get(agent_name, {})

        # Merge profile overrides
        temperature = profile.get("temperature", cfg["temperature"])
        max_tokens = profile.get("max_tokens", cfg["max_tokens"])

        # Build prompt contract
        contract_prompt = self.build_prompt_contract(mode_str, agent_name, prompt)

        # Token budget check
        cost_est = self.estimate_token_cost(contract_prompt, mode_str)
        if not cost_est["within_budget"]:
            log.warning(
                "Prompt exceeds token budget (%d est vs %d limit) — truncating",
                cost_est["estimated_tokens"], cost_est["budget_limit"],
            )
            contract_prompt = self.build_prompt_contract(
                mode_str, agent_name, self.truncate_evidence(prompt, max_tokens)
            )

        # Prepare the API call
        response_text = ""
        token_usage = {"prompt": 0, "completion": 0, "total": 0}
        retry_count = 0
        model_name = "gpt-4o"
        parse_status = "OK"
        failure_reason = ""

        def _call_llm() -> str:
            nonlocal token_usage, retry_count
            if not self.client:
                raise RuntimeError("OpenAI client not available")
            messages = [
                {"role": "system", "content": self._governed_system_prompt},
                {"role": "user", "content": contract_prompt},
            ]
            resp = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=cfg.get("timeout", 30),
            )
            usage = resp.usage
            if usage:
                token_usage = {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                }
            return resp.choices[0].message.content.strip()

        # Execute with retry policy
        result = self._retry_with_policy(_call_llm, mode_str)

        if result is None:
            response_text = ""
            parse_status = "FAILED"
            failure_reason = "All retries exhausted"
        else:
            response_text = result

        # Validate
        validation = self.validate_response(response_text, mode_str, expected_schema)
        parsed = validation.get("parsed")

        # If structured_json failed validation, attempt one repair retry
        if mode_str == "structured_json" and not validation["valid"] and response_text:
            repair_prompt = (
                f"Your previous response was not valid JSON. "
                f"Issues: {validation['issues']}. "
                f"Please reformat your answer as a single valid JSON object "
                f"inside a ```json``` block. Original response:\n{response_text[:500]}"
            )
            try:
                if self.client:
                    repair_resp = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self._governed_system_prompt},
                            {"role": "user", "content": repair_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    repair_text = repair_resp.choices[0].message.content.strip()
                    repair_validation = self.validate_response(repair_text, mode_str, expected_schema)
                    if repair_validation["valid"]:
                        response_text = repair_text
                        parsed = repair_validation.get("parsed")
                        parse_status = "REPAIR_SUCCEEDED"
                        validation = repair_validation
                        retry_count += 1
                    else:
                        parse_status = "MALFORMED"
                else:
                    parse_status = "MALFORMED"
            except Exception as exc:
                log.warning("Repair retry failed: %s", exc)
                parse_status = "MALFORMED"
        elif not validation["valid"] and response_text:
            parse_status = "MALFORMED"
            failure_reason = "; ".join(validation.get("issues", []))

        latency_ms = int((time.time() - ts_start) * 1000)
        validation_status = validation.get("validation_status", "VALID" if validation["valid"] else "INVALID")

        record = InteractionRecord(
            interaction_id=interaction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            requesting_agent=agent_name,
            task_type=mode_str,
            model_name=model_name,
            prompt_hash=prompt_hash,
            correlation_id=correlation_id or "",
            token_usage=token_usage,
            latency_ms=latency_ms,
            retry_count=retry_count,
            parse_status=parse_status,
            validation_status=validation_status,
            advisory_only=True,
            response_summary=response_text[:100] if response_text else "",
            failure_reason=failure_reason,
        )

        # Store record (capped)
        self._interaction_history.append(record)
        if len(self._interaction_history) > _MAX_HISTORY:
            self._interaction_history = self._interaction_history[-_MAX_HISTORY:]

        log.info(
            "Governed query [%s/%s] — %s — %dms — %d tokens",
            agent_name, mode_str, parse_status, latency_ms, token_usage.get("total", 0),
        )

        return {
            "response": response_text,
            "parsed": parsed,
            "record": record,
            "advisory_only": True,
        }

    # ── 7. Interaction History ────────────────────────────────────────

    def get_recent_interactions(self, n: int = 20) -> List[InteractionRecord]:
        """Return the *n* most recent interaction records."""
        self._ensure_governance_state()
        return list(self._interaction_history[-n:])

    def get_failure_summary(self) -> Dict[str, int]:
        """Count interaction failures by type."""
        self._ensure_governance_state()
        summary: Dict[str, int] = {}
        for rec in self._interaction_history:
            if rec.parse_status != "OK":
                summary[rec.parse_status] = summary.get(rec.parse_status, 0) + 1
        return summary

    def get_usage_summary(self) -> Dict[str, Any]:
        """Aggregate token usage and cost estimate across all recorded interactions."""
        self._ensure_governance_state()
        total_tokens = 0
        by_agent: Dict[str, int] = {}
        for rec in self._interaction_history:
            t = rec.token_usage.get("total", 0)
            total_tokens += t
            by_agent[rec.requesting_agent] = by_agent.get(rec.requesting_agent, 0) + t
        # GPT-4o pricing rough estimate: $5/1M input, $15/1M output — average ~$10/1M
        estimated_cost = total_tokens * 10.0 / 1_000_000
        return {
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "by_agent": by_agent,
            "interaction_count": len(self._interaction_history),
        }

    def get_malformed_count(self) -> int:
        """Return total count of MALFORMED parse results."""
        self._ensure_governance_state()
        return sum(1 for r in self._interaction_history if r.parse_status == "MALFORMED")

    # ── 8. Machine Summary ────────────────────────────────────────────

    def compute_machine_summary(self) -> Dict[str, Any]:
        """Full governance dashboard for machine consumption."""
        self._ensure_governance_state()
        records = self._interaction_history
        total = len(records)
        successful = sum(1 for r in records if r.parse_status == "OK")
        failed = sum(1 for r in records if r.parse_status == "FAILED")
        malformed = sum(1 for r in records if r.parse_status in ("MALFORMED",))
        repair_ok = sum(1 for r in records if r.parse_status == "REPAIR_SUCCEEDED")
        total_tokens = sum(r.token_usage.get("total", 0) for r in records)
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]
        avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

        by_agent: Dict[str, int] = {}
        by_mode: Dict[str, int] = {}
        for r in records:
            by_agent[r.requesting_agent] = by_agent.get(r.requesting_agent, 0) + 1
            by_mode[r.task_type] = by_mode.get(r.task_type, 0) + 1

        return {
            "total_interactions": total,
            "successful": successful,
            "failed": failed,
            "malformed": malformed,
            "repair_succeeded": repair_ok,
            "total_tokens_used": total_tokens,
            "estimated_cost_usd": round(total_tokens * 10.0 / 1_000_000, 4),
            "avg_latency_ms": avg_latency,
            "by_agent": by_agent,
            "by_mode": by_mode,
        }
