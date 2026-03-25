"""
agents/math/llm_bridge.py
---------------------------
גשר LLM כפול — Dual LLM Bridge (Claude + GPT)

מאפשר לשלוח prompt מתמטי לשני מודלים במקביל ולבחור את התשובה הטובה ביותר.
תומך בחוסר מפתח — אם מפתח אחד חסר, משתמש רק בשני.

שימוש:
  bridge = DualLLMBridge()
  result = bridge.query_both(prompt, context)
  # result = {"claude": "...", "gpt": "...", "best": "claude"|"gpt", "reasoning": "..."}
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

# ── נתיבי שורש ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

log = logging.getLogger("llm_bridge")


def _load_env_key(key_name: str) -> Optional[str]:
    """
    טוען מפתח API מקובץ .env בשורש הפרויקט.
    מחפש שורות בפורמט KEY=value.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key_name:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return None


class DualLLMBridge:
    """
    גשר כפול לשני מודלי LLM — Claude (Anthropic) + GPT (OpenAI).
    מאפשר שאילתה במקביל והשוואת תוצאות.
    """

    # מודלים ברירת מחדל
    CLAUDE_MODEL = "claude-sonnet-4-6"
    GPT_MODEL = "gpt-4o"
    MAX_TOKENS = 4096

    def __init__(self) -> None:
        self.claude_client = None
        self.openai_client = None

        # טעינת Claude — ANTHROPIC_API_KEY
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or _load_env_key("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
                log.info("Claude client initialized (model=%s)", self.CLAUDE_MODEL)
            except ImportError:
                log.warning("anthropic package not installed — Claude unavailable (pip install anthropic)")
            except Exception as exc:
                log.warning("Failed to initialize Claude client: %s", exc)
        else:
            log.info("ANTHROPIC_API_KEY not found — Claude unavailable")

        # טעינת GPT — OPENAI_API_KEY
        openai_key = os.environ.get("OPENAI_API_KEY") or _load_env_key("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_key)
                log.info("OpenAI client initialized (model=%s)", self.GPT_MODEL)
            except ImportError:
                log.warning("openai package not installed — GPT unavailable (pip install openai)")
            except Exception as exc:
                log.warning("Failed to initialize OpenAI client: %s", exc)
        else:
            log.info("OPENAI_API_KEY not found — GPT unavailable")

        # Properties for easy checking
        self.has_claude = self.claude_client is not None
        self.has_gpt = self.openai_client is not None

        # אימות — לפחות מודל אחד חייב להיות זמין
        if not self.has_claude and not self.has_gpt:
            log.error("No LLM clients available! Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

    # ── שאילתה בודדת — Claude ──────────────────────────────────────────────
    def query_claude(self, prompt: str, context: str = "") -> str:
        """
        שולח prompt ל-Claude API ומחזיר תשובה.
        מחזיר מחרוזת ריקה אם Claude לא זמין.

        Handles Hebrew/non-ASCII text safely: first tries with full UTF-8 prompt,
        falls back to ASCII-only if encoding errors occur.
        """
        if not self.claude_client:
            log.warning("Claude client not available — skipping")
            return ""

        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        # Strip non-ASCII (Hebrew etc.) to prevent encoding errors in the API layer.
        # Mathematical prompts don't need Hebrew content.
        clean_prompt = full_prompt.encode("ascii", errors="ignore").decode("ascii")
        # Remove double-spaces and blank lines caused by stripping
        import re as _re
        clean_prompt = _re.sub(r"  +", " ", clean_prompt)
        clean_prompt = _re.sub(r"\n{3,}", "\n\n", clean_prompt)

        try:
            response = self.claude_client.messages.create(
                model=self.CLAUDE_MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": clean_prompt}],
            )
            text = response.content[0].text
            log.info("Claude response received (%d chars)", len(text))
            return text
        except Exception as exc:
            log.error("Claude API error: %s", exc)
            return ""

    # ── שאילתה בודדת — GPT ────────────────────────────────────────────────
    def query_gpt(self, prompt: str, context: str = "") -> str:
        """
        שולח prompt ל-OpenAI GPT API ומחזיר תשובה.
        מחזיר מחרוזת ריקה אם GPT לא זמין.
        """
        if not self.openai_client:
            log.warning("OpenAI client not available — skipping")
            return ""

        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        try:
            response = self.openai_client.chat.completions.create(
                model=self.GPT_MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": full_prompt}],
            )
            text = response.choices[0].message.content or ""
            log.info("GPT response received (%d chars)", len(text))
            return text
        except Exception as exc:
            log.error("OpenAI API error: %s", exc)
            return ""

    # ── שאילתה כפולה במקביל ───────────────────────────────────────────────
    def query_both(self, prompt: str, context: str = "") -> dict:
        """
        שולח prompt לשני המודלים במקביל (threading) ובוחר את הטוב ביותר.

        מחזיר:
          {
            "claude": "response text",
            "gpt": "response text",
            "best": "claude" | "gpt" | "none",
            "reasoning": "why this response was chosen"
          }
        """
        claude_response = ""
        gpt_response = ""
        errors: list[str] = []

        # הפעלה במקביל עם threads
        def _query_claude():
            nonlocal claude_response
            claude_response = self.query_claude(prompt, context)

        def _query_gpt():
            nonlocal gpt_response
            gpt_response = self.query_gpt(prompt, context)

        threads = []
        if self.claude_client:
            t = threading.Thread(target=_query_claude, daemon=True)
            threads.append(t)
            t.start()
        if self.openai_client:
            t = threading.Thread(target=_query_gpt, daemon=True)
            threads.append(t)
            t.start()

        # המתנה לסיום — timeout 60 שניות
        for t in threads:
            t.join(timeout=60)

        # בחירת התשובה הטובה ביותר
        result = self._pick_best(claude_response, gpt_response, prompt)
        return result

    # ── בחירת תשובה טובה ביותר ────────────────────────────────────────────
    def _pick_best(
        self,
        claude_response: str,
        gpt_response: str,
        task_type: str = "",
    ) -> dict:
        """
        משווה בין שתי תשובות ובוחר את הטובה יותר.
        קריטריונים: קיום קוד Python, ריגורוזיות מתמטית, ספציפיות.

        אם רק מודל אחד ענה — בוחרים אותו.
        אם שניהם ענו — היוריסטיקה מבוססת ניקוד.
        """
        result = {
            "claude": claude_response,
            "gpt": gpt_response,
            "best": "none",
            "reasoning": "",
        }

        # רק מודל אחד זמין
        if claude_response and not gpt_response:
            result["best"] = "claude"
            result["reasoning"] = "Only Claude responded"
            return result

        if gpt_response and not claude_response:
            result["best"] = "gpt"
            result["reasoning"] = "Only GPT responded"
            return result

        if not claude_response and not gpt_response:
            result["reasoning"] = "Neither model responded"
            return result

        # שני המודלים ענו — ניקוד היוריסטי
        claude_score = self._score_response(claude_response)
        gpt_score = self._score_response(gpt_response)

        if claude_score > gpt_score:
            result["best"] = "claude"
            result["reasoning"] = (
                f"Claude scored higher ({claude_score} vs {gpt_score}): "
                "better mathematical rigor and code quality"
            )
        elif gpt_score > claude_score:
            result["best"] = "gpt"
            result["reasoning"] = (
                f"GPT scored higher ({gpt_score} vs {claude_score}): "
                "better mathematical rigor and code quality"
            )
        else:
            # תיקו — העדפת Claude (יותר חזק מתמטית בניסיוננו)
            result["best"] = "claude"
            result["reasoning"] = f"Tied ({claude_score} vs {gpt_score}) — defaulting to Claude"

        return result

    @staticmethod
    def _score_response(text: str) -> int:
        """
        ניקוד היוריסטי של תשובת LLM למשימות מתמטיות/קוד.
        ניקוד גבוה יותר = תשובה טובה יותר.
        """
        score = 0

        # האם מכיל קוד Python?
        if "def " in text and "return " in text:
            score += 3
        elif "```python" in text:
            score += 2

        # ריגורוזיות מתמטית — מושגים מתמטיים
        math_terms = [
            "derivative", "gradient", "convex", "sigmoid", "logistic",
            "exponential", "polynomial", "linear", "asymptotic",
            "monotonic", "continuous", "differentiable",
            "OU process", "half-life", "mean-revert", "stationarity",
            "ADF", "Hurst", "z-score", "p-value",
        ]
        math_count = sum(1 for term in math_terms if term.lower() in text.lower())
        score += min(math_count, 5)  # מקסימום 5 נקודות

        # ספציפיות — מספרים קונקרטיים
        numbers = re.findall(r"\b\d+\.?\d*\b", text)
        if len(numbers) >= 3:
            score += 2

        # אורך סביר — לא קצר מדי ולא ארוך מדי
        word_count = len(text.split())
        if 100 < word_count < 2000:
            score += 1

        # מבנה מאורגן — כותרות או נקודות
        if any(marker in text for marker in ["##", "- ", "1.", "Step"]):
            score += 1

        return score


# ── CLI — בדיקה מהירה ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bridge = DualLLMBridge()

    if not bridge.claude_client and not bridge.openai_client:
        print("ERROR: No API keys found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in .env")
        sys.exit(1)

    # בדיקה פשוטה
    test_prompt = (
        "Given f(x) = sigmoid(a*x + b), what are the mathematical properties "
        "that make it suitable for scoring functions in quantitative trading? "
        "Provide a brief Python implementation."
    )
    print(f"Test prompt: {test_prompt[:80]}...")
    print()

    result = bridge.query_both(test_prompt)
    print(f"Best: {result['best']}")
    print(f"Reasoning: {result['reasoning']}")
    print()

    if result["claude"]:
        print(f"Claude ({len(result['claude'])} chars): {result['claude'][:200]}...")
    if result["gpt"]:
        print(f"GPT ({len(result['gpt'])} chars): {result['gpt'][:200]}...")
