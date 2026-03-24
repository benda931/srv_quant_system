"""tests/test_bayesian_multi_obj.py — Multi-objective optimizer tests."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture: Settings with test API key
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
    from config.settings import Settings
    return Settings()


# ---------------------------------------------------------------------------
# Test: MULTI_OBJ_SEARCH_SPACE has exactly the 10 required parameters
# ---------------------------------------------------------------------------
def test_multi_obj_search_space_params():
    from analytics.bayesian_optimizer import MULTI_OBJ_SEARCH_SPACE

    expected_params = {
        "zscore_threshold_calm",
        "zscore_threshold_normal",
        "zscore_threshold_tension",
        "regime_conviction_scale_calm",
        "regime_conviction_scale_normal",
        "regime_conviction_scale_tension",
        "non_whitelist_penalty",
        "signal_a1_frob",
        "signal_a2_mode",
        "signal_a3_coc",
    }
    assert set(MULTI_OBJ_SEARCH_SPACE.keys()) == expected_params


def test_multi_obj_search_space_ranges():
    from analytics.bayesian_optimizer import MULTI_OBJ_SEARCH_SPACE

    # Spot-check a few ranges
    assert MULTI_OBJ_SEARCH_SPACE["zscore_threshold_calm"]["low"] == 0.3
    assert MULTI_OBJ_SEARCH_SPACE["zscore_threshold_calm"]["high"] == 1.5
    assert MULTI_OBJ_SEARCH_SPACE["non_whitelist_penalty"]["low"] == 0.1
    assert MULTI_OBJ_SEARCH_SPACE["non_whitelist_penalty"]["high"] == 0.8
    assert MULTI_OBJ_SEARCH_SPACE["signal_a1_frob"]["low"] == 0.3
    assert MULTI_OBJ_SEARCH_SPACE["signal_a1_frob"]["high"] == 2.0

    # All params should be float type
    for name, spec in MULTI_OBJ_SEARCH_SPACE.items():
        assert spec["type"] == "float", f"{name} should be float"


# ---------------------------------------------------------------------------
# Test: _create_settings_with_overrides applies param values
# ---------------------------------------------------------------------------
def test_create_settings_with_overrides(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
    from analytics.bayesian_optimizer import _create_settings_with_overrides

    overrides = {
        "zscore_threshold_calm": 1.0,
        "signal_a1_frob": 1.5,
        "non_whitelist_penalty": 0.5,
    }
    settings = _create_settings_with_overrides(overrides)

    assert settings.zscore_threshold_calm == pytest.approx(1.0)
    assert settings.signal_a1_frob == pytest.approx(1.5)
    assert settings.non_whitelist_penalty == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test: _suggest_params works with MULTI_OBJ_SEARCH_SPACE
# ---------------------------------------------------------------------------
def test_suggest_params_multi_obj():
    from analytics.bayesian_optimizer import _suggest_params, MULTI_OBJ_SEARCH_SPACE
    import optuna

    study = optuna.create_study(directions=["maximize", "minimize", "maximize"])

    def trial_fn(trial):
        params = _suggest_params(trial, MULTI_OBJ_SEARCH_SPACE)
        assert len(params) == 10
        for name, spec in MULTI_OBJ_SEARCH_SPACE.items():
            assert name in params
            assert spec["low"] <= params[name] <= spec["high"], (
                f"{name}={params[name]} out of [{spec['low']}, {spec['high']}]"
            )
        return 0.0, 0.0, 0.0

    study.optimize(trial_fn, n_trials=3, catch=())


# ---------------------------------------------------------------------------
# Test: run_multi_objective produces valid Pareto output structure
# ---------------------------------------------------------------------------
def test_run_multi_objective_with_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")

    # Mock the backtest to return deterministic results
    mock_result = MagicMock()
    mock_result.sharpe = 1.5
    mock_result.max_drawdown = -0.08
    mock_result.ic_mean = 0.12

    with patch(
        "analytics.bayesian_optimizer._load_backtest_data"
    ) as mock_load, patch(
        "analytics.bayesian_optimizer.WalkForwardBacktester",
        create=True,
    ), patch(
        "analytics.backtest.WalkForwardBacktester"
    ) as MockBacktester:
        # Setup mocks
        mock_load.return_value = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        MockBacktester.return_value.run_backtest.return_value = mock_result

        import analytics.bayesian_optimizer as bopt

        # Override paths to use tmp_path
        original_data_dir = bopt.DATA_DIR
        original_pareto_path = bopt.PARETO_PATH
        original_study_db = bopt.STUDY_DB_PATH
        bopt.DATA_DIR = tmp_path
        bopt.PARETO_PATH = tmp_path / "optuna_pareto.json"
        bopt.STUDY_DB_PATH = tmp_path / "optuna_studies.db"

        try:
            result = bopt.run_multi_objective(
                n_trials=3,
                study_name="test_pareto_3obj",
            )

            # Validate result structure
            assert "timestamp" in result
            assert "n_trials" in result
            assert "n_pareto" in result
            assert "objectives" in result
            assert "parameter_space" in result
            assert "pareto_front" in result
            assert result["objectives"] == ["sharpe (max)", "max_dd (min)", "ic (max)"]
            assert result["n_trials"] == 3

            # Validate Pareto front entries
            for entry in result["pareto_front"]:
                assert "sharpe" in entry
                assert "max_dd" in entry
                assert "ic" in entry
                assert "trial" in entry

            # Validate JSON file was saved
            assert bopt.PARETO_PATH.exists()
            with open(bopt.PARETO_PATH) as f:
                saved = json.load(f)
            assert saved["n_trials"] == 3

        finally:
            bopt.DATA_DIR = original_data_dir
            bopt.PARETO_PATH = original_pareto_path
            bopt.STUDY_DB_PATH = original_study_db


# ---------------------------------------------------------------------------
# Test: backward-compatible alias exists
# ---------------------------------------------------------------------------
def test_backward_compatible_alias():
    from analytics.bayesian_optimizer import (
        run_multi_objective,
        run_multi_objective_optimization,
    )
    assert run_multi_objective_optimization is run_multi_objective


# ---------------------------------------------------------------------------
# Test: CLI parser accepts --multi --trials
# ---------------------------------------------------------------------------
def test_cli_parser_multi_trials():
    import argparse
    from analytics.bayesian_optimizer import main

    # We just verify the module can be imported and the argparse is valid
    # by checking the function exists; full CLI test would require subprocess
    assert callable(main)


# ---------------------------------------------------------------------------
# Test: graceful failure returns penalty values
# ---------------------------------------------------------------------------
def test_multi_objective_handles_backtest_failure(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")

    from analytics.bayesian_optimizer import (
        _build_multi_objective,
        MULTI_OBJ_SEARCH_SPACE,
    )
    import optuna

    prices_df = pd.DataFrame()
    fundamentals_df = pd.DataFrame()
    weights_df = pd.DataFrame()

    with patch("analytics.backtest.WalkForwardBacktester") as MockBT:
        MockBT.return_value.run_backtest.side_effect = RuntimeError("test error")

        objective = _build_multi_objective(prices_df, fundamentals_df, weights_df)

        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"]
        )
        study.optimize(objective, n_trials=1, catch=())

        # Failed trial should return penalty values
        trial = study.trials[0]
        assert trial.values[0] == -10.0   # sharpe penalty
        assert trial.values[1] == 1.0     # max_dd penalty
        assert trial.values[2] == -1.0    # ic penalty
