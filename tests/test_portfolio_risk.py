"""tests/test_portfolio_risk.py — VaR/CVaR/MCTR shapes, math invariants, breach flags."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.portfolio_risk import PortfolioRiskEngine, RiskReport

TICKERS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
N = len(TICKERS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mat = rng.normal(0.0, 0.01, (500, N))
    dates = pd.date_range("2022-01-03", periods=500, freq="B")
    return pd.DataFrame(mat, index=dates, columns=TICKERS)


@pytest.fixture(scope="module")
def prices_df(returns_df) -> pd.DataFrame:
    return 100.0 * np.exp(returns_df.cumsum())


@pytest.fixture(scope="module")
def equal_weights() -> dict:
    return {t: 1.0 / N for t in TICKERS}


@pytest.fixture(scope="module")
def engine() -> PortfolioRiskEngine:
    return PortfolioRiskEngine()


@pytest.fixture(scope="module")
def cov(engine, returns_df) -> np.ndarray:
    return engine.compute_cov(returns_df)


# ---------------------------------------------------------------------------
# compute_cov
# ---------------------------------------------------------------------------
def test_cov_shape(cov):
    assert cov.shape == (N, N)


def test_cov_is_symmetric(cov):
    assert np.allclose(cov, cov.T, atol=1e-10)


def test_cov_is_positive_semi_definite(cov):
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-8


# ---------------------------------------------------------------------------
# compute_var
# ---------------------------------------------------------------------------
def test_var_95_is_positive(engine, equal_weights, cov):
    var = engine.compute_var(equal_weights, cov, confidence=0.95)
    assert var > 0.0


def test_var_99_exceeds_95(engine, equal_weights, cov):
    var95 = engine.compute_var(equal_weights, cov, confidence=0.95)
    var99 = engine.compute_var(equal_weights, cov, confidence=0.99)
    assert var99 > var95


def test_var_horizon_scaling(engine, equal_weights, cov):
    var1 = engine.compute_var(equal_weights, cov, confidence=0.95, horizon=1)
    var5 = engine.compute_var(equal_weights, cov, confidence=0.95, horizon=5)
    assert var5 == pytest.approx(var1 * math.sqrt(5), rel=1e-6)


# ---------------------------------------------------------------------------
# compute_cvar
# ---------------------------------------------------------------------------
def test_cvar_is_positive(engine, equal_weights, returns_df):
    cvar = engine.compute_cvar(equal_weights, returns_df)
    assert cvar > 0.0


def test_cvar_finite(engine, equal_weights, returns_df):
    cvar = engine.compute_cvar(equal_weights, returns_df)
    assert math.isfinite(cvar)


def test_cvar_geq_var_approx(engine, equal_weights, cov, returns_df):
    var95 = engine.compute_var(equal_weights, cov, confidence=0.95)
    cvar95 = engine.compute_cvar(equal_weights, returns_df, confidence=0.95)
    # CVaR ≥ VaR by definition; allow some slack for normal vs. empirical
    assert cvar95 >= var95 * 0.70


# ---------------------------------------------------------------------------
# compute_mctr
# ---------------------------------------------------------------------------
def test_mctr_shape(engine, equal_weights, cov):
    mctr = engine.compute_mctr(equal_weights, cov)
    assert mctr.shape == (N,)
    assert list(mctr.index) == TICKERS


def test_mctr_euler_identity(engine, equal_weights, cov):
    """Euler's theorem: Σ w_i * MCTR_i = σ_p."""
    mctr = engine.compute_mctr(equal_weights, cov)
    w = np.array(list(equal_weights.values()))
    port_std = float((w @ cov @ w) ** 0.5)
    euler = float((w * mctr.values).sum())
    assert euler == pytest.approx(port_std, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_risk_budget
# ---------------------------------------------------------------------------
def test_risk_budget_sums_to_one(engine, equal_weights, cov):
    rb = engine.compute_risk_budget(equal_weights, cov)
    assert rb.sum() == pytest.approx(1.0, abs=1e-9)


def test_risk_budget_non_negative_long_only(engine, equal_weights, cov):
    rb = engine.compute_risk_budget(equal_weights, cov)
    assert (rb >= 0).all()


# ---------------------------------------------------------------------------
# full_risk_report
# ---------------------------------------------------------------------------
def test_full_report_returns_risk_report(engine, equal_weights, prices_df, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    s = Settings()
    report = engine.full_risk_report(equal_weights, prices_df, s)
    assert isinstance(report, RiskReport)


def test_full_report_hhi_equal_weights(engine, equal_weights, prices_df, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    report = engine.full_risk_report(equal_weights, prices_df, Settings())
    assert report.concentration_hhi == pytest.approx(1.0 / N, rel=1e-5)


def test_full_report_vol_positive(engine, equal_weights, prices_df, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    report = engine.full_risk_report(equal_weights, prices_df, Settings())
    assert report.portfolio_vol_ann > 0.0


def test_vol_target_breach_detected(engine, prices_df, monkeypatch):
    """Concentrated position in one high-vol ticker triggers breach."""
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    # Construct a very high-vol returns series: 5% daily vol → ~79% annual
    rng = np.random.default_rng(7)
    n_days = 300
    hi_vol_rets = rng.normal(0.0, 0.05, (n_days, N))
    hi_prices = 100.0 * np.exp(np.cumsum(hi_vol_rets, axis=0))
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B")
    hi_prices_df = pd.DataFrame(hi_prices, index=dates, columns=TICKERS)
    weights = {t: 1.0 / N for t in TICKERS}
    s = Settings()
    report = engine.full_risk_report(weights, hi_prices_df, s)
    assert report.vol_target_breach is True


def test_max_weight_breach_detected(engine, prices_df, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")
    from config.settings import Settings
    s = Settings()   # max_single_name_weight = 0.20
    # One sector at 50% → breach
    concentrated = {TICKERS[0]: 0.50}
    concentrated.update({t: 0.05 for t in TICKERS[1:]})
    report = engine.full_risk_report(concentrated, prices_df, s)
    assert report.max_weight_breach is True
