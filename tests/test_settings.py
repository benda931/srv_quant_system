"""tests/test_settings.py — Settings instantiation & validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key_pytest")
    from config.settings import Settings
    return Settings()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_instantiation_with_api_key(settings):
    from config.settings import Settings
    assert isinstance(settings, Settings)
    assert settings.fmp_api_key == "test_key_pytest"


def test_default_vol_target(settings):
    assert settings.target_portfolio_vol == pytest.approx(0.12)
    assert 0.02 <= settings.target_portfolio_vol <= 0.60


def test_default_max_single_name_weight(settings):
    assert settings.max_single_name_weight == pytest.approx(0.20)


def test_layer_weights_sum_to_one(settings):
    total = (
        settings.weight_stat
        + settings.weight_macro
        + settings.weight_fund
        + settings.weight_vol
    )
    assert abs(total - 1.0) < 1e-9


def test_layer_points_sum_to_100(settings):
    total = (
        settings.points_stat
        + settings.points_macro
        + settings.points_fund
        + settings.points_vol
    )
    assert total == 100


def test_sector_list_has_11_tickers(settings):
    sectors = settings.sector_list()
    assert len(sectors) == 11
    assert "XLK" in sectors
    assert "XLU" in sectors


def test_vix_thresholds_ordered(settings):
    assert settings.vix_level_soft < settings.vix_level_hard


def test_corr_regime_thresholds_ordered(settings):
    assert settings.calm_avg_corr_max < settings.tension_avg_corr_min
    assert settings.tension_avg_corr_min < settings.crisis_avg_corr_min


def test_invalid_layer_weight_sum_raises(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    with pytest.raises(ValidationError):
        # sum = 0.50 + 0.20 + 0.20 + 0.20 = 1.10  → validation error
        Settings(weight_stat=0.50, weight_macro=0.20, weight_fund=0.20, weight_vol=0.20)


def test_pca_window_default(settings):
    assert settings.pca_window == 252


def test_ewma_lambda_range(settings):
    assert 0.80 <= settings.ewma_lambda <= 0.99
