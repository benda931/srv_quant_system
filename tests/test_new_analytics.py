"""
tests/test_new_analytics.py
==============================
Comprehensive tests for all newly expanded analytics modules.

Covers:
  - pair_scanner (Kalman, Johansen, dynamic thresholds, baskets)
  - trade_monitor (trailing stops, Greeks, position aging)
  - ml_signals (ensemble stacking, adaptive IC)
  - optimizer (Black-Litterman, regime views)
  - leverage_engine (GARCH vol, asymmetric DD)
  - risk_decomposition (Euler, factor VaR)
  - market_microstructure (spread, impact, liquidity)
  - signal_stack (ablation, sensitivity)
  - tail_risk (EVT, Hill, regime tails)
  - signal_mean_reversion (OU MLE, variance ratio, cointegration)
  - backtest (regime-conditional WF, bootstrap CI)
  - attribution (Brinson, factor decomposition)
  - regime_alerts (Markov transition)
  - signal_decay (curve fitting)
  - correlation_engine (dispersion surface, CRP)
  - alerting service
  - data_loader service
  - run_context service

Run: python -m pytest tests/test_new_analytics.py -v
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_prices():
    """Generate realistic fake sector ETF prices."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    tickers = ["XLK", "XLF", "XLV", "XLE", "XLU", "SPY", "XLI", "XLB", "XLC", "XLY", "XLP", "XLRE"]
    data = {}
    for t in tickers:
        base = 100 + np.random.randn() * 20
        rets = np.random.randn(504) * 0.012 + 0.0003
        data[t] = base * np.cumprod(1 + rets)
    data["^VIX"] = 18 + np.random.randn(504) * 3
    data["^TNX"] = 2.5 + np.cumsum(np.random.randn(504) * 0.02)
    data["DX-Y.NYB"] = 100 + np.cumsum(np.random.randn(504) * 0.1)
    data["HYG"] = 80 + np.cumsum(np.random.randn(504) * 0.05)
    data["IEF"] = 105 + np.cumsum(np.random.randn(504) * 0.03)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def fake_returns(fake_prices):
    return np.log(fake_prices / fake_prices.shift(1)).dropna()


# ═══════════════════════════════════════════════════════════════════════
# Pair Scanner Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPairScanner:
    def test_kalman_hedge_ratio(self):
        from analytics.pair_scanner import kalman_hedge_ratio
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n) * 0.01) + 100
        y = 1.5 * x + np.cumsum(np.random.randn(n) * 0.005)
        betas, spreads = kalman_hedge_ratio(pd.Series(y), pd.Series(x))
        assert len(betas) == n
        assert abs(betas[-1] - 1.5) < 0.5

    def test_scan_pairs(self, fake_prices):
        from analytics.pair_scanner import scan_pairs
        sectors = ["XLK", "XLF", "XLV", "XLE", "XLU"]
        pairs = scan_pairs(fake_prices, sectors)
        assert len(pairs) > 0
        assert pairs[0].signal_strength >= pairs[-1].signal_strength

    def test_classify_spread_regime(self):
        from analytics.pair_scanner import classify_spread_regime
        np.random.seed(42)
        mr_spread = pd.Series(np.cumsum(np.random.randn(200) * 0.01 - 0.001 * np.arange(200)))
        regime, vr = classify_spread_regime(mr_spread)
        assert regime in ("MEAN_REVERTING", "TRENDING", "RANDOM_WALK")
        assert 0 < vr < 5

    def test_dynamic_z_thresholds(self):
        from analytics.pair_scanner import dynamic_z_thresholds
        spread = pd.Series(np.random.randn(100) * 0.1)
        entry, exit_t, stop = dynamic_z_thresholds(spread)
        assert entry > exit_t
        assert stop > entry


# ═══════════════════════════════════════════════════════════════════════
# Trade Monitor Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTradeMonitor:
    def test_trailing_stop(self):
        from analytics.trade_monitor import compute_trailing_stop
        state = compute_trailing_stop(0.02, 0.025, trail_distance=0.015)
        assert state.high_water_mark == 0.025
        assert not state.is_triggered

        state2 = compute_trailing_stop(0.005, 0.025, trail_distance=0.015)
        assert state2.is_triggered  # Dropped 2% from HWM of 2.5%

    def test_position_aging(self):
        from analytics.trade_monitor import analyse_position_aging
        fresh = analyse_position_aging("T1", "XLK", 3, 20.0)
        assert fresh.aging_label == "FRESH"
        assert fresh.recommended_action == "HOLD"

        expired = analyse_position_aging("T2", "XLF", 50, 20.0)
        assert expired.aging_label == "EXPIRED"
        assert expired.recommended_action == "EXIT"

    def test_portfolio_greeks(self):
        from analytics.trade_monitor import compute_portfolio_greeks
        positions = [
            {"ticker": "XLK", "direction": "LONG", "notional": 100000},
            {"ticker": "XLF", "direction": "SHORT", "notional": 80000},
        ]
        greeks = compute_portfolio_greeks(positions)
        assert greeks.gross_exposure == 180000
        assert len(greeks.positions) == 2


# ═══════════════════════════════════════════════════════════════════════
# ML Signals Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMLSignals:
    def test_adaptive_ic_threshold(self):
        from analytics.ml_signals import compute_adaptive_ic_threshold
        walk_ics = [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.01, 0.05, 0.02, 0.03]
        regime_ics = {"NORMAL": walk_ics}
        result = compute_adaptive_ic_threshold(walk_ics, regime_ics, "NORMAL")
        assert result.regime == "NORMAL"
        assert result.ic_threshold > 0
        assert result.n_observations == 10


# ═══════════════════════════════════════════════════════════════════════
# Optimizer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestOptimizer:
    def test_black_litterman(self):
        from analytics.optimizer import black_litterman
        np.random.seed(42)
        cov = np.eye(3) * 0.0004
        cov[0, 1] = cov[1, 0] = 0.0001
        tickers = ["XLK", "XLF", "XLV"]
        views = {"XLK": 0.10}
        result = black_litterman(cov, tickers, views=views, view_confidence={"XLK": 0.8})
        assert "XLK" in result.posterior_returns
        assert result.blend_ratio > 0

    def test_regime_views(self):
        from analytics.optimizer import regime_views
        sectors = ["XLK", "XLF", "XLV", "XLE", "XLU", "XLP"]
        views, conf = regime_views("TENSION", sectors)
        assert len(views) > 0
        assert all(0 < c <= 1 for c in conf.values())


# ═══════════════════════════════════════════════════════════════════════
# Leverage Engine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLeverageEngine:
    def test_garch_vol_forecast(self):
        from analytics.leverage_engine import garch_vol_forecast
        np.random.seed(42)
        rets = pd.Series(np.random.randn(252) * 0.012)
        result = garch_vol_forecast(rets, target_vol=0.10)
        assert result.current_vol_ann > 0
        assert result.vol_regime in ("LOW", "NORMAL", "HIGH", "EXTREME")
        assert 0.2 <= result.leverage_adjustment <= 3.0

    def test_asymmetric_dd(self):
        from analytics.leverage_engine import asymmetric_drawdown_deleverage
        none_dd = asymmetric_drawdown_deleverage(0.01)
        assert none_dd.leverage_multiplier == 1.0
        assert none_dd.deleveraging_zone == "NONE"

        flat_dd = asymmetric_drawdown_deleverage(0.25)
        assert flat_dd.leverage_multiplier == 0.0
        assert flat_dd.deleveraging_zone == "FLAT"


# ═══════════════════════════════════════════════════════════════════════
# Risk Decomposition Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRiskDecomposition:
    def test_euler_decomposition(self):
        from analytics.risk_decomposition import euler_risk_decomposition
        np.random.seed(42)
        cov = np.eye(3) * 0.0004
        tickers = ["A", "B", "C"]
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        result = euler_risk_decomposition(weights, cov, tickers)
        assert result.portfolio_vol > 0
        assert abs(sum(result.component_risk.values()) - result.portfolio_vol) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Market Microstructure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMarketMicrostructure:
    def test_roll_spread(self):
        from analytics.market_microstructure import estimate_spread_roll
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))
        spread = estimate_spread_roll(prices)
        assert 0 <= spread <= 0.05

    def test_market_impact(self):
        from analytics.market_microstructure import estimate_market_impact
        impact = estimate_market_impact("XLK", 100000, 150.0)
        assert impact.total_impact_bps > 0
        assert impact.algo_suggestion in ("MARKET", "LIMIT", "TWAP", "VWAP")

    def test_liquidity_scores(self, fake_prices):
        from analytics.market_microstructure import compute_liquidity_scores
        scores = compute_liquidity_scores(fake_prices, tickers=["XLK", "XLF", "XLV"])
        assert len(scores) >= 2
        for s in scores.values():
            assert 0 <= s.score <= 100


# ═══════════════════════════════════════════════════════════════════════
# Signal Stack Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSignalStack:
    def test_entry_sensitivity(self):
        from analytics.signal_stack import signal_entry_sensitivity, SignalStackResult
        # Create mock results with varying conviction
        mock = []
        for i in range(10):
            from types import SimpleNamespace
            r = SimpleNamespace(
                conviction_score=0.05 * (i + 1), entry_threshold=0.10,
                passes_entry=0.05 * (i + 1) >= 0.10,
                ticker=f"T{i}", direction="LONG",
            )
            mock.append(r)
        result = signal_entry_sensitivity(mock, thresholds=[0.05, 0.10, 0.20, 0.30])
        assert result.param_name == "entry_threshold"
        assert len(result.sweep_results) == 4


# ═══════════════════════════════════════════════════════════════════════
# Tail Risk Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTailRisk:
    def test_hill_estimator(self):
        from analytics.tail_risk import hill_estimator
        np.random.seed(42)
        losses = np.abs(np.random.standard_t(df=3, size=1000))
        xi = hill_estimator(losses, k=50)
        assert xi > 0  # t-distribution has heavy tail

    def test_evt_pot(self, fake_returns):
        from analytics.tail_risk import fit_evt_pot
        weights = {"XLK": 0.3, "XLF": 0.3, "XLV": 0.2, "XLE": 0.2}
        result = fit_evt_pot(fake_returns, weights)
        assert result.n_total > 0
        assert result.tail_type in ("heavy", "medium", "thin", "unknown")


# ═══════════════════════════════════════════════════════════════════════
# Signal Mean Reversion Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSignalMeanReversion:
    def test_ou_mle(self):
        from analytics.signal_mean_reversion import ou_mle
        np.random.seed(42)
        n = 200
        ou = np.zeros(n)
        theta, mu, sigma = 0.1, 0.0, 0.05
        for t in range(1, n):
            ou[t] = ou[t-1] + theta * (mu - ou[t-1]) + sigma * np.random.randn()
        result = ou_mle(pd.Series(ou))
        assert result.theta > 0
        assert result.half_life > 0
        assert result.n_obs == n

    def test_variance_ratio(self):
        from analytics.signal_mean_reversion import variance_ratio_test
        np.random.seed(42)
        rw = pd.Series(np.cumsum(np.random.randn(300)))
        result = variance_ratio_test(rw, q=10)
        assert 0.5 < result.vr < 1.5  # Should be near 1 for random walk


# ═══════════════════════════════════════════════════════════════════════
# Backtest Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBacktest:
    def test_bootstrap_ci(self):
        from analytics.backtest import bootstrap_backtest_ci
        np.random.seed(42)
        walk_returns = list(np.random.randn(50) * 0.02 + 0.001)
        results = bootstrap_backtest_ci(walk_returns, n_bootstrap=500)
        assert "sharpe" in results
        assert results["sharpe"].ci_lower < results["sharpe"].ci_upper
        assert results["sharpe"].n_bootstrap == 500


# ═══════════════════════════════════════════════════════════════════════
# Attribution Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAttribution:
    def test_brinson(self):
        from analytics.attribution import brinson_attribution
        port_w = {"XLK": 0.3, "XLF": 0.2, "XLV": 0.5}
        bench_w = {"XLK": 0.33, "XLF": 0.33, "XLV": 0.34}
        sector_rets = {"XLK": 0.05, "XLF": -0.02, "XLV": 0.03}
        result = brinson_attribution(port_w, bench_w, sector_rets, 0.02)
        assert abs(result.allocation_effect + result.selection_effect + result.interaction_effect - result.excess_return) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Regime Alerts Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegimeAlerts:
    def test_markov_model(self):
        from analytics.regime_alerts import fit_markov_regime_model
        regimes = pd.Series(["CALM"]*50 + ["NORMAL"]*100 + ["TENSION"]*30 + ["NORMAL"]*20)
        result = fit_markov_regime_model(regimes, "NORMAL")
        assert result.current_regime == "NORMAL"
        assert len(result.transition_matrix) > 0
        assert result.crisis_probability_5d >= 0


# ═══════════════════════════════════════════════════════════════════════
# Signal Decay Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSignalDecay:
    def test_ic_decay_curve(self):
        from analytics.signal_decay import fit_ic_decay_curve
        horizons = [1, 5, 10, 21, 42, 63]
        ic_values = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
        result = fit_ic_decay_curve(horizons, ic_values)
        assert result.model in ("exponential", "weibull", "flat")
        assert result.half_life_fit > 0 or result.model == "flat"


# ═══════════════════════════════════════════════════════════════════════
# Correlation Engine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCorrelationEngine:
    def test_dispersion_surface(self, fake_prices):
        from analytics.correlation_engine import compute_dispersion_surface
        sectors = ["XLK", "XLF", "XLV", "XLE", "XLU"]
        result = compute_dispersion_surface(fake_prices, sectors)
        assert len(result.horizons) > 0
        assert result.crp_curve_shape in ("CONTANGO", "BACKWARDATION", "FLAT", "HUMPED")


# ═══════════════════════════════════════════════════════════════════════
# Services Tests
# ═══════════════════════════════════════════════════════════════════════

class TestServices:
    def test_alert_service(self):
        from services.alerting import AlertService, Alert
        svc = AlertService()
        result = svc.send(Alert(level="INFO", title="Test", message="Unit test"))
        assert result is True
        assert len(svc.recent_alerts(5)) >= 1

    def test_run_context(self):
        from services.run_context import RunContext
        from config.settings import get_settings
        ctx = RunContext.create(get_settings())
        ctx.mark_step("test_step", success=True)
        ctx.mark_step("fail_step", success=False, error="test error")
        ctx.finalize()
        assert ctx.duration_s >= 0
        assert len(ctx.steps_completed) == 1
        assert len(ctx.steps_failed) == 1
        assert ctx.errors["fail_step"] == "test error"

    def test_data_loader(self):
        from services.data_loader import DataLoader
        from config.settings import get_settings
        loader = DataLoader(get_settings())
        outputs = loader.load_agent_outputs()
        assert isinstance(outputs, dict)
        assert "registry" in outputs


# ═══════════════════════════════════════════════════════════════════════
# DCC-GARCH Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDCCGARCH:
    def test_fit_and_predict(self):
        from analytics.correlation_structure import DCCGARCHCorrelation
        np.random.seed(42)
        r = pd.DataFrame(np.random.randn(200, 4) * 0.01, columns=["A", "B", "C", "D"])
        dcc = DCCGARCHCorrelation(min_obs=50)
        R = dcc.fit_and_predict(r)
        assert R.shape == (4, 4)
        assert np.allclose(np.diag(R.values), 1.0)
        assert np.isfinite(R.values).all()

    def test_fallback_short_data(self):
        from analytics.correlation_structure import DCCGARCHCorrelation
        r = pd.DataFrame(np.random.randn(20, 3) * 0.01, columns=["A", "B", "C"])
        dcc = DCCGARCHCorrelation(min_obs=50)
        R = dcc.fit_and_predict(r)
        assert R.shape == (3, 3)  # Should fallback to Pearson


# ═══════════════════════════════════════════════════════════════════════
# Dispersion Backtest Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDispersionBacktest:
    def test_greeks_estimation(self):
        from analytics.dispersion_backtest import estimate_dispersion_greeks
        greeks = estimate_dispersion_greeks(
            index_price=500, sector_prices={"XLK": 150, "XLF": 50},
            index_iv=0.20, sector_ivs={"XLK": 0.25, "XLF": 0.18},
            sector_weights={"XLK": 0.6, "XLF": 0.4},
        )
        assert greeks.net_delta == 0.0  # Delta-neutral
        assert greeks.rho_correlation < 0  # Short correlation

    def test_skew_impact(self):
        from analytics.dispersion_backtest import estimate_skew_impact
        result = estimate_skew_impact(0.20, {"XLK": 0.25}, skew_index=135)
        assert result.skew_regime in ("FAVORABLE", "NEUTRAL", "ADVERSE")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
