"""
agents/shared/gpt_conversation.py
====================================
Focused GPT Conversation Manager

Instead of sending one giant prompt to GPT, breaks the analysis into
small focused queries that build on each other:

  Step 1: "Here's the current performance. What's the main problem?"
  Step 2: "The problem is X. What specific parameter should change?"
  Step 3: "You suggest changing Y. What value, and what's the expected impact?"
  Step 4: "Summarize: what to do, in order of priority."

Each step is a focused ~200-word query that gets a focused ~200-word response.
This produces better results than a single 2000-word dump.

Usage:
  from agents.shared.gpt_conversation import GPTConversation
  conv = GPTConversation()
  analysis = conv.analyze_performance(metrics, regime_data)
  suggestion = conv.suggest_improvement(analysis, current_params)
  action = conv.propose_action(suggestion)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent


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
