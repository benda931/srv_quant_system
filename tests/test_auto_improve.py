"""tests/test_auto_improve.py — Tests for the auto-improvement feedback loop."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixture: settings with test API key
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
    from config.settings import Settings
    return Settings()


@pytest.fixture()
def improver(monkeypatch, settings):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
    from agents.auto_improve import AutoImprover
    return AutoImprover(settings=settings, prices=None, dry_run=True)


@pytest.fixture()
def sample_metrics():
    return {
        "best_name": "PCA_Z_REVERSAL",
        "best_sharpe": -0.286,
        "best_win_rate": 0.553,
        "best_max_dd": -0.105,
        "best_pnl": -0.069,
        "best_trades": 445,
        "best_avg_hold": 33.3,
        "regime_breakdown": {
            "CALM": {"sharpe": 0.66, "win_rate": 0.58, "trades": 120},
            "NORMAL": {"sharpe": 0.23, "win_rate": 0.54, "trades": 180},
            "TENSION": {"sharpe": 0.68, "win_rate": 0.59, "trades": 80},
            "CRISIS": {"sharpe": -0.78, "win_rate": 0.42, "trades": 65},
        },
        "all_results": {
            "PCA_Z_REVERSAL": {
                "sharpe": -0.286, "win_rate": 0.553,
                "total_pnl": -0.069, "max_drawdown": -0.105,
                "total_trades": 445, "params": {"z_entry": 1.0},
            },
            "ADAPTIVE_THRESHOLD": {
                "sharpe": -0.237, "win_rate": 0.549,
                "total_pnl": -0.059, "max_drawdown": -0.095,
                "total_trades": 512, "params": {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------
class TestImports:
    def test_import_auto_improve(self):
        from agents.auto_improve import AutoImprover
        assert AutoImprover is not None

    def test_import_constants(self):
        from agents.auto_improve import (
            SHARPE_PROMOTION_THRESHOLD,
            MAX_SUGGESTIONS_PER_CYCLE,
            MAX_GPT_CALLS_PER_CYCLE,
            TUNABLE_PARAMS,
        )
        assert SHARPE_PROMOTION_THRESHOLD == 0.03
        assert MAX_SUGGESTIONS_PER_CYCLE == 5
        assert MAX_GPT_CALLS_PER_CYCLE == 3
        assert len(TUNABLE_PARAMS) > 10

    def test_import_helpers(self):
        from agents.auto_improve import (
            _load_improvement_log,
            _save_improvement_log,
            show_status,
        )
        assert callable(_load_improvement_log)
        assert callable(show_status)


# ---------------------------------------------------------------------------
# AutoImprover initialization
# ---------------------------------------------------------------------------
class TestAutoImproverInit:
    def test_init_defaults(self, improver):
        assert improver.dry_run is True
        assert improver._prices is None
        assert improver._gpt is None
        assert improver._gpt_calls_used == 0

    def test_current_params(self, improver):
        params = improver.current_params()
        assert isinstance(params, dict)
        assert "regime_z_calm" in params
        assert "regime_size_calm" in params
        assert "signal_vix_hard" in params
        assert len(params) > 10


# ---------------------------------------------------------------------------
# Weakness identification
# ---------------------------------------------------------------------------
class TestIdentifyWeaknesses:
    def test_finds_negative_sharpe(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        types = [w["type"] for w in weaknesses]
        assert "negative_sharpe" in types

    def test_finds_weak_regime(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        types = [w["type"] for w in weaknesses]
        assert "weak_regime" in types
        crisis_w = [w for w in weaknesses if w["type"] == "weak_regime"
                    and w["context"].get("regime") == "CRISIS"]
        assert len(crisis_w) == 1
        assert crisis_w[0]["context"]["sharpe"] == -0.78

    def test_finds_all_negative(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        types = [w["type"] for w in weaknesses]
        assert "all_negative" in types

    def test_no_weaknesses_for_good_metrics(self, improver):
        good = {
            "best_name": "GOOD",
            "best_sharpe": 1.5,
            "best_win_rate": 0.62,
            "best_max_dd": -0.08,
            "regime_breakdown": {
                "CALM": {"sharpe": 1.2, "win_rate": 0.60, "trades": 100},
            },
            "all_results": {
                "GOOD": {"sharpe": 1.5, "win_rate": 0.62, "total_pnl": 0.15,
                          "max_drawdown": -0.08, "total_trades": 200, "params": {}},
            },
        }
        weaknesses = improver.identify_weaknesses(good)
        assert len(weaknesses) == 0

    def test_empty_metrics_returns_empty(self, improver):
        assert improver.identify_weaknesses({}) == []

    def test_severity_ordering(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        if len(weaknesses) >= 2:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            severities = [severity_order.get(w["severity"], 99) for w in weaknesses]
            assert severities == sorted(severities)


# ---------------------------------------------------------------------------
# Parameter suggestion generation
# ---------------------------------------------------------------------------
class TestGenerateSuggestions:
    def test_generates_suggestions_for_weaknesses(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        suggestions = improver.generate_parameter_suggestions(weaknesses)
        assert len(suggestions) > 0
        assert len(suggestions) <= 5  # MAX_SUGGESTIONS_PER_CYCLE

    def test_suggestion_structure(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        suggestions = improver.generate_parameter_suggestions(weaknesses)
        for s in suggestions:
            assert "param" in s
            assert "current" in s
            assert "proposed" in s
            assert "reason" in s
            assert "source" in s
            assert s["current"] != s["proposed"]

    def test_no_duplicate_params(self, improver, sample_metrics):
        weaknesses = improver.identify_weaknesses(sample_metrics)
        suggestions = improver.generate_parameter_suggestions(weaknesses)
        params = [s["param"] for s in suggestions]
        assert len(params) == len(set(params))

    def test_weak_regime_reduces_sizing(self, improver):
        weaknesses = [{
            "type": "weak_regime",
            "description": "CRISIS bad",
            "severity": "high",
            "context": {"regime": "CRISIS", "sharpe": -0.78, "trades": 50, "win_rate": 0.4},
        }]
        suggestions = improver.generate_parameter_suggestions(weaknesses)
        params = [s["param"] for s in suggestions]
        # For CRISIS (already disabled), fallback to VIX threshold tightening
        # For other regimes, would reduce sizing/z/conviction directly
        assert len(suggestions) > 0
        assert ("regime_size_crisis" in params or "regime_z_crisis" in params or
                "regime_conviction_scale_crisis" in params or "signal_vix_hard" in params)

    def test_weak_regime_tension_reduces_sizing(self, improver):
        weaknesses = [{
            "type": "weak_regime",
            "description": "TENSION bad",
            "severity": "medium",
            "context": {"regime": "TENSION", "sharpe": -0.35, "trades": 50, "win_rate": 0.45},
        }]
        suggestions = improver.generate_parameter_suggestions(weaknesses)
        params = [s["param"] for s in suggestions]
        assert "regime_size_tension" in params or "regime_z_tension" in params or \
               "regime_conviction_scale_tension" in params

    def test_empty_weaknesses_no_suggestions(self, improver):
        suggestions = improver.generate_parameter_suggestions([])
        assert suggestions == []


# ---------------------------------------------------------------------------
# GPT response parsing
# ---------------------------------------------------------------------------
class TestGPTParsing:
    def test_parse_equals_format(self, improver):
        response = "I recommend regime_z_calm = 0.9 and signal_vix_hard = 35.0"
        params = improver.current_params()
        parsed = improver._parse_gpt_suggestions(response, params)
        param_names = [p["param"] for p in parsed]
        assert "regime_z_calm" in param_names or "signal_vix_hard" in param_names

    def test_parse_to_format(self, improver):
        response = "Change regime_z_calm to 0.9 for better performance."
        params = improver.current_params()
        parsed = improver._parse_gpt_suggestions(response, params)
        param_names = [p["param"] for p in parsed]
        assert "regime_z_calm" in param_names

    def test_parse_from_to_format(self, improver):
        response = "Adjust regime_size_tension from 0.6 to 0.4"
        params = improver.current_params()
        parsed = improver._parse_gpt_suggestions(response, params)
        param_names = [p["param"] for p in parsed]
        assert "regime_size_tension" in param_names

    def test_rejects_out_of_range(self, improver):
        response = "Set regime_z_calm = 50.0"  # way out of range
        params = improver.current_params()
        parsed = improver._parse_gpt_suggestions(response, params)
        assert len(parsed) == 0

    def test_rejects_same_value(self, improver):
        current = improver.current_params()
        current_val = current.get("regime_z_calm", 0.7)
        response = f"Keep regime_z_calm = {current_val}"
        parsed = improver._parse_gpt_suggestions(response, current)
        # Should not suggest changing to the same value
        calm_suggestions = [p for p in parsed if p["param"] == "regime_z_calm"]
        assert len(calm_suggestions) == 0

    def test_empty_response(self, improver):
        parsed = improver._parse_gpt_suggestions("", improver.current_params())
        assert parsed == []


# ---------------------------------------------------------------------------
# Improvement log I/O
# ---------------------------------------------------------------------------
class TestImprovementLog:
    def test_load_empty_log(self, tmp_path, monkeypatch):
        from agents.auto_improve import engine as mod
        monkeypatch.setattr(mod, "IMPROVEMENT_LOG_PATH", tmp_path / "test_log.json")
        data = mod._load_improvement_log()
        assert data == {"cycles": []}

    def test_save_and_load_log(self, tmp_path, monkeypatch):
        from agents.auto_improve import engine as mod
        log_path = tmp_path / "test_log.json"
        monkeypatch.setattr(mod, "IMPROVEMENT_LOG_PATH", log_path)

        data = {"cycles": [{"timestamp": "2026-01-01", "promoted": 1}]}
        mod._save_improvement_log(data)

        loaded = mod._load_improvement_log()
        assert len(loaded["cycles"]) == 1
        assert loaded["cycles"][0]["promoted"] == 1

    def test_log_survives_corruption(self, tmp_path, monkeypatch):
        from agents.auto_improve import engine as mod
        log_path = tmp_path / "test_log.json"
        log_path.write_text("NOT JSON", encoding="utf-8")
        monkeypatch.setattr(mod, "IMPROVEMENT_LOG_PATH", log_path)
        data = mod._load_improvement_log()
        assert data == {"cycles": []}


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------
class TestPromotion:
    def test_promote_requires_passing_test(self, improver):
        suggestion = {"param": "regime_z_calm", "current": 0.7, "proposed": 0.9}
        result = {"passed": False, "delta": 0.01}
        assert improver.promote_if_better(suggestion, result) is False

    def test_dry_run_blocks_promotion(self, improver):
        suggestion = {"param": "regime_z_calm", "current": 0.7, "proposed": 0.9}
        result = {"passed": True, "delta": 0.05}
        # dry_run=True, so should not promote
        assert improver.promote_if_better(suggestion, result) is False

    def test_promotion_writes_env_file(self, settings, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
        from agents.auto_improve.engine import AutoImprover
        from agents.auto_improve import engine as mod

        # Override ROOT to tmp for env file writing
        env_file = tmp_path / ".env.auto_improve"
        monkeypatch.setattr(mod, "ROOT", tmp_path)

        imp = AutoImprover(settings=settings, dry_run=False)
        suggestion = {"param": "regime_z_calm", "current": 0.7, "proposed": 0.9}
        result = {"passed": True, "delta": 0.05}

        promoted = imp.promote_if_better(suggestion, result)
        assert promoted is True
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "REGIME_Z_CALM=0.9" in content


# ---------------------------------------------------------------------------
# TUNABLE_PARAMS sanity
# ---------------------------------------------------------------------------
class TestTunableParams:
    def test_all_params_exist_in_settings(self, settings):
        from agents.auto_improve import TUNABLE_PARAMS
        for field_name, _mn, _mx, _step, _desc in TUNABLE_PARAMS:
            assert hasattr(settings, field_name), f"Settings missing field: {field_name}"

    def test_ranges_are_valid(self):
        from agents.auto_improve import TUNABLE_PARAMS
        for field_name, mn, mx, step, desc in TUNABLE_PARAMS:
            assert mn < mx, f"{field_name}: min ({mn}) >= max ({mx})"
            assert step > 0, f"{field_name}: step must be positive"
            assert len(desc) > 0, f"{field_name}: missing description"

    def test_current_values_within_ranges(self, settings):
        from agents.auto_improve import TUNABLE_PARAMS
        for field_name, mn, mx, _step, _desc in TUNABLE_PARAMS:
            value = getattr(settings, field_name)
            # Allow some tolerance for values at boundaries
            assert mn - 0.01 <= value <= mx + 0.01, \
                f"{field_name}={value} outside range [{mn}, {mx}]"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
class TestCLI:
    def test_parse_cycle(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["auto_improve.py", "--cycle"])
        from agents.auto_improve import parse_args
        args = parse_args()
        assert args.cycle is True

    def test_parse_status(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["auto_improve.py", "--status"])
        from agents.auto_improve import parse_args
        args = parse_args()
        assert args.status is True

    def test_parse_dry_run(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["auto_improve.py", "--dry-run"])
        from agents.auto_improve import parse_args
        args = parse_args()
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# run_agents.py integration — verify auto_improve is in the cycle
# ---------------------------------------------------------------------------
class TestRunAgentsIntegration:
    def test_run_cycle_includes_auto_improve(self):
        """Verify run_cycle function references auto_improve."""
        import inspect
        from agents.run_agents import run_cycle
        source = inspect.getsource(run_cycle)
        assert "auto_improve" in source
        assert "Auto-Improve" in source

    def test_run_cycle_returns_auto_improve_key(self):
        """Verify the return dict structure includes auto_improve."""
        import inspect
        from agents.run_agents import run_cycle
        source = inspect.getsource(run_cycle)
        assert '"auto_improve"' in source
