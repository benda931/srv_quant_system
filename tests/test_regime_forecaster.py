"""Tests for agents.regime_forecaster."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_prices():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
    macro = ["SPY", "^VIX", "HYG", "IEF", "TLT", "GLD", "UUP"]
    data = {}
    for t in sectors + ["SPY"]:
        data[t] = np.exp(np.cumsum(np.random.randn(n) * 0.01)) * 100
    data["^VIX"] = 18 + np.cumsum(np.random.randn(n) * 0.3)
    data["^VIX"] = np.clip(data["^VIX"], 10, 80)
    for t in ["HYG", "IEF", "TLT", "GLD", "UUP"]:
        data[t] = np.exp(np.cumsum(np.random.randn(n) * 0.005)) * 100
    return pd.DataFrame(data, index=dates)


class TestRegimeForecaster:
    def test_import(self):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        assert RegimeForecaster is not None

    def test_vix_features(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        f = rf.compute_vix_features()
        assert "vix" in f
        assert "vix_pct" in f
        assert 0 <= f["vix_pct"] <= 1

    def test_credit_features(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        f = rf.compute_credit_features()
        assert "credit_z" in f
        assert np.isfinite(f["credit_z"])

    def test_correlation_features(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        f = rf.compute_correlation_features()
        assert "avg_corr" in f
        assert -1 <= f["avg_corr"] <= 1

    def test_breadth_features(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        f = rf.compute_breadth_features()
        assert "breadth_50d" in f
        assert 0 <= f["breadth_50d"] <= 1

    def test_forecast_regime(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        forecast = rf.forecast_regime()
        probs = forecast["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert forecast["predicted_regime"] in ("CALM", "NORMAL", "TENSION", "CRISIS")
        assert 0 <= forecast["regime_safety_score"] <= 1

    def test_transition_probability(self, mock_prices):
        from agents.regime_forecaster.agent_regime_forecaster import RegimeForecaster
        rf = RegimeForecaster()
        rf._prices = mock_prices
        forecast = rf.forecast_regime()
        assert 0 <= forecast["transition_probability"] <= 1


class TestAlphaDecayMonitor:
    def test_import(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        assert AlphaDecayMonitor is not None

    def test_classify_healthy(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        m = AlphaDecayMonitor()
        signals = {"ic_declining": False, "sharpe_declining": False,
                   "wr_declining": False, "consecutive_negative_ic": 0,
                   "consecutive_negative_sharpe": 0}
        assert m.classify_decay(signals) == "HEALTHY"

    def test_classify_early_decay(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        m = AlphaDecayMonitor()
        signals = {"ic_declining": True, "sharpe_declining": False,
                   "wr_declining": False, "consecutive_negative_ic": 0,
                   "consecutive_negative_sharpe": 0}
        assert m.classify_decay(signals) == "EARLY_DECAY"

    def test_classify_decaying(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        m = AlphaDecayMonitor()
        signals = {"ic_declining": True, "sharpe_declining": True,
                   "wr_declining": False, "consecutive_negative_ic": 3,
                   "consecutive_negative_sharpe": 3}
        assert m.classify_decay(signals) == "DECAYING"

    def test_classify_dead(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        m = AlphaDecayMonitor()
        signals = {"ic_declining": True, "sharpe_declining": True,
                   "wr_declining": True, "consecutive_negative_ic": 7,
                   "consecutive_negative_sharpe": 7}
        assert m.classify_decay(signals) == "DEAD"

    def test_recommendations(self):
        from agents.alpha_decay.agent_alpha_decay import AlphaDecayMonitor
        m = AlphaDecayMonitor()
        recs = m.generate_recommendations("DECAYING", {})
        assert len(recs) > 0
        assert any("ALERT" in r or "defensive" in r for r in recs)
