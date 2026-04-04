"""
services/data_loader.py
========================
Data loading service — centralizes all JSON/agent output loading
that was previously scattered and duplicated in main.py.

Eliminates:
  - 5 duplicate function definitions in main.py
  - Inline JSON loading for 10+ agent files
  - Ad-hoc momentum/methodology/improvement data prep

Usage:
    from services.data_loader import DataLoader
    loader = DataLoader(settings)
    agent_data = loader.load_agent_outputs()
    momentum = loader.compute_momentum_ranking(engine.prices)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("data_loader")


class DataLoader:
    """Centralized data loading for dashboard and agents."""

    def __init__(self, settings):
        self.settings = settings
        self.root = settings.project_root

    # ── Core JSON loader ─────────────────────────────────────────────────

    def load_json(self, path) -> Optional[Dict]:
        """Load a JSON file safely — returns None on any error."""
        try:
            p = Path(path)
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    # ── Agent outputs (10+ JSON files) ───────────────────────────────────

    def load_agent_outputs(self) -> Dict[str, Any]:
        """
        Load all agent JSON outputs in one call.
        Returns dict with named agent data.
        """
        return {
            "registry": self.load_json(self.root / "logs" / "agent_registry.json"),
            "decay": self.load_json(self.root / "agents" / "alpha_decay" / "decay_status.json"),
            "regime": self.load_json(self.root / "agents" / "regime_forecaster" / "regime_forecast.json"),
            "risk": self.load_json(self.root / "agents" / "risk_guardian" / "risk_status.json"),
            "scout": self.load_json(self.root / "agents" / "data_scout" / "scout_report.json"),
            "portfolio": self.load_json(self.root / "agents" / "portfolio_construction" / "portfolio_weights.json"),
            "auto_improve": self.load_json(self.root / "agents" / "auto_improve" / "machine_summary.json"),
            "optimizer": self.load_json(self.root / "agents" / "optimizer" / "optimization_history.json"),
            "architect": self.load_json(self.root / "agents" / "architect" / "improvement_history.json"),
            "ensemble": self.load_json(self.root / "data" / "ensemble_results.json"),
        }

    # ── Improvement log ──────────────────────────────────────────────────

    def load_improvement_log(self) -> Optional[Dict]:
        return self.load_json(self.root / "agents" / "auto_improve" / "improvement_log.json")

    # ── Methodology lab results ──────────────────────────────────────────

    def load_methodology_results(self) -> Optional[Dict]:
        """Load latest methodology lab report for strategy ranking."""
        try:
            reports = sorted(
                (self.root / "agents" / "methodology" / "reports").glob("*methodology_lab*"),
                reverse=True,
            )
            if not reports:
                return None
            data = self.load_json(reports[0])
            if data and "results" in data:
                return data["results"]
            if data and isinstance(data, dict):
                if any(isinstance(v, dict) and "sharpe" in v for v in data.values()):
                    return data
        except Exception:
            pass
        return None

    # ── Alpha research ───────────────────────────────────────────────────

    def load_alpha_research(self) -> Optional[Dict]:
        """Load latest alpha research report."""
        try:
            reports = sorted(
                (self.root / "agents" / "methodology" / "reports").glob("*alpha_research*"),
                reverse=True,
            )
            if reports:
                return self.load_json(reports[0])
        except Exception:
            pass
        return None

    # ── Daily brief ──────────────────────────────────────────────────────

    def load_daily_brief(self) -> str:
        """Load latest daily brief text."""
        try:
            brief_dir = self.root / "reports" / "output"
            if brief_dir.exists():
                files = sorted(brief_dir.glob("*.txt"), reverse=True)
                if files:
                    return files[0].read_text(encoding="utf-8")
        except Exception:
            pass
        return ""

    # ── Momentum ranking ─────────────────────────────────────────────────

    def compute_momentum_ranking(self, prices: pd.DataFrame) -> Optional[List[Dict]]:
        """
        Cross-sectional sector momentum ranking vs SPY.
        Returns sorted list of {ticker, momentum_21d, momentum_42d, vol}.
        """
        try:
            sectors = [s for s in self.settings.sector_list() if s in prices.columns]
            if len(sectors) < 5 or len(prices) < 60:
                return None

            spy = self.settings.spy_ticker
            log_rets = np.log(prices[sectors] / prices[sectors].shift(1)).dropna()
            spy_ret = (
                np.log(prices[spy] / prices[spy].shift(1)).dropna()
                if spy in prices.columns
                else pd.Series(0, index=log_rets.index)
            )

            # Use simple returns (not log) for more intuitive display
            simple_rets = prices[sectors].pct_change().dropna()
            spy_simple = prices[spy].pct_change().dropna() if spy in prices.columns else pd.Series(0, index=simple_rets.index)

            ranking = []
            for s in sectors:
                if len(simple_rets) >= 21:
                    # Simple return over 21 days, relative to SPY
                    sec_21 = float((1 + simple_rets[s].iloc[-21:]).prod() - 1)
                    spy_21 = float((1 + spy_simple.iloc[-21:]).prod() - 1)
                    m21 = sec_21 - spy_21
                else:
                    m21 = 0
                if len(simple_rets) >= 42:
                    sec_42 = float((1 + simple_rets[s].iloc[-42:]).prod() - 1)
                    spy_42 = float((1 + spy_simple.iloc[-42:]).prod() - 1)
                    m42 = sec_42 - spy_42
                else:
                    m42 = 0
                vol = float(log_rets[s].iloc[-60:].std() * np.sqrt(252)) if len(log_rets) >= 60 else 0.15
                ranking.append({"ticker": s, "momentum_21d": m21, "momentum_42d": m42, "vol": vol})

            ranking.sort(key=lambda x: x["momentum_21d"], reverse=True)
            return ranking
        except Exception:
            return None
