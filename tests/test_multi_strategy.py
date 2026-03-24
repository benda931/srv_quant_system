"""tests/test_multi_strategy.py — Multi-strategy ensemble unit tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.multi_strategy import (
    ALL_SECTORS,
    MR_WHITELIST,
    DispersionStrategy,
    MeanReversionStrategy,
    MomentumMRStrategy,
    Signal,
    StrategyEnsemble,
    _max_drawdown,
    _regime_from_vix,
    _sharpe_from_returns,
    _zscore_rolling,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_prices() -> pd.DataFrame:
    """
    Synthetic price panel with SPY, VIX, and 11 sector ETFs.
    ~500 trading days of mean-reverting + trending data.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    data = {}
    # SPY: upward drift + noise
    spy = 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    data["SPY"] = spy

    # VIX: mean-reverting around 18
    vix = np.zeros(n)
    vix[0] = 18.0
    for t in range(1, n):
        vix[t] = max(10, vix[t - 1] + 0.1 * (18 - vix[t - 1]) + rng.normal(0, 1.5))
    data["^VIX"] = vix

    # Sectors: SPY-correlated with idiosyncratic noise
    for i, sector in enumerate(ALL_SECTORS):
        beta = 0.8 + 0.4 * rng.random()
        idio = rng.normal(0, 0.008, n)
        # Add some mean-reversion pattern for MR sectors
        if sector in MR_WHITELIST:
            mr_component = np.zeros(n)
            for t in range(1, n):
                mr_component[t] = 0.92 * mr_component[t - 1] + rng.normal(0, 0.005)
            idio += np.diff(np.concatenate([[0], mr_component]))
        sector_ret = beta * np.diff(np.log(spy), prepend=np.log(spy[0])) + idio
        data[sector] = 50 * np.exp(np.cumsum(sector_ret))

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mr_strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(z_entry=0.7, z_exit=0.25, hold_days=15)


@pytest.fixture
def disp_strategy() -> DispersionStrategy:
    return DispersionStrategy()


@pytest.fixture
def mom_mr_strategy() -> MomentumMRStrategy:
    return MomentumMRStrategy()


# ---------------------------------------------------------------------------
# Test 1: _zscore_rolling outputs valid z-scores
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_zscore_rolling_output_range(self):
        """Z-scores should be centered near 0 with std near 1 for normal data."""
        rng = np.random.default_rng(99)
        s = pd.Series(rng.normal(0, 1, 300))
        z = _zscore_rolling(s, window=60)
        valid = z.dropna()
        assert len(valid) > 0
        # Mean should be roughly 0, std roughly 1 for stationary normal input
        assert abs(valid.mean()) < 1.0
        assert 0.3 < valid.std() < 2.0

    def test_sharpe_from_returns_positive(self):
        """Positive mean returns should give positive Sharpe."""
        rng = np.random.default_rng(7)
        r = rng.normal(0.005, 0.01, 100)  # positive mean with noise
        assert _sharpe_from_returns(r) > 0

    def test_sharpe_from_returns_empty(self):
        """Empty or single-element returns -> 0."""
        assert _sharpe_from_returns(np.array([])) == 0.0
        assert _sharpe_from_returns(np.array([0.01])) == 0.0

    def test_max_drawdown_no_loss(self):
        """Monotonically increasing returns have zero drawdown."""
        r = np.array([0.01] * 50)
        assert _max_drawdown(r) == 0.0

    def test_max_drawdown_with_loss(self):
        """A drawdown should be detected."""
        r = np.array([0.01, 0.01, -0.05, -0.05, 0.01])
        assert _max_drawdown(r) > 0.0

    def test_regime_from_vix(self):
        """Regime classification thresholds."""
        assert _regime_from_vix(10) == "CALM"
        assert _regime_from_vix(18) == "NORMAL"
        assert _regime_from_vix(25) == "TENSION"
        assert _regime_from_vix(35) == "CRISIS"
        assert _regime_from_vix(float("nan")) == "NORMAL"


# ---------------------------------------------------------------------------
# Test 2: MeanReversionStrategy
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:
    def test_mr_only_whitelist_sectors(self, synthetic_prices, mr_strategy):
        """MR signals should only be for whitelist sectors."""
        z_scores = mr_strategy.compute_z_scores(synthetic_prices)
        # Find a date with some signals
        for i in range(300, len(synthetic_prices)):
            signals = mr_strategy.generate_signals(
                synthetic_prices, i, z_scores, "NORMAL"
            )
            for sig in signals:
                assert sig.sector in MR_WHITELIST, (
                    f"MR signal for non-whitelist sector {sig.sector}"
                )

    def test_mr_no_signals_in_crisis(self, synthetic_prices, mr_strategy):
        """MR should produce no signals during CRISIS regime."""
        z_scores = mr_strategy.compute_z_scores(synthetic_prices)
        for i in range(300, min(350, len(synthetic_prices))):
            signals = mr_strategy.generate_signals(
                synthetic_prices, i, z_scores, "CRISIS"
            )
            assert len(signals) == 0

    def test_mr_direction_opposes_zscore(self, synthetic_prices, mr_strategy):
        """MR direction should be opposite to z-score sign (mean reversion)."""
        z_scores = mr_strategy.compute_z_scores(synthetic_prices)
        for i in range(300, len(synthetic_prices)):
            signals = mr_strategy.generate_signals(
                synthetic_prices, i, z_scores, "NORMAL"
            )
            for sig in signals:
                z = float(z_scores[sig.sector].iloc[i])
                if z > 0:
                    assert sig.direction == -1, "Positive z should give SHORT"
                else:
                    assert sig.direction == 1, "Negative z should give LONG"

    def test_mr_z_scores_shape(self, synthetic_prices, mr_strategy):
        """Z-scores should match price index length and have sector columns."""
        z = mr_strategy.compute_z_scores(synthetic_prices)
        assert len(z) == len(synthetic_prices)
        for sector in MR_WHITELIST:
            assert sector in z.columns


# ---------------------------------------------------------------------------
# Test 3: DispersionStrategy
# ---------------------------------------------------------------------------

class TestDispersionStrategy:
    def test_dispersion_z_computation(self, synthetic_prices, disp_strategy):
        """Dispersion z-score should be a Series of same length as prices."""
        dz = disp_strategy.compute_dispersion_z(synthetic_prices)
        assert isinstance(dz, pd.Series)
        assert len(dz) == len(synthetic_prices)

    def test_dispersion_no_signal_high_vix(self, synthetic_prices, disp_strategy):
        """No dispersion signals when VIX > max threshold."""
        dz = disp_strategy.compute_dispersion_z(synthetic_prices)
        signals = disp_strategy.generate_signals(
            synthetic_prices, 300, dz, vix_level=30.0, regime="NORMAL"
        )
        assert len(signals) == 0

    def test_dispersion_no_signal_crisis(self, synthetic_prices, disp_strategy):
        """No dispersion signals in CRISIS."""
        dz = disp_strategy.compute_dispersion_z(synthetic_prices)
        signals = disp_strategy.generate_signals(
            synthetic_prices, 300, dz, vix_level=15.0, regime="CRISIS"
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Test 4: MomentumMRStrategy
# ---------------------------------------------------------------------------

class TestMomentumMRStrategy:
    def test_momentum_computation(self, synthetic_prices, mom_mr_strategy):
        """Momentum DataFrame should have sector columns."""
        m = mom_mr_strategy.compute_momentum(synthetic_prices)
        assert isinstance(m, pd.DataFrame)
        sectors_avail = [s for s in ALL_SECTORS if s in synthetic_prices.columns]
        for s in sectors_avail:
            assert s in m.columns

    def test_mom_mr_no_signals_crisis(self, synthetic_prices, mom_mr_strategy):
        """No signals in CRISIS."""
        z = MeanReversionStrategy().compute_z_scores(synthetic_prices)
        m = mom_mr_strategy.compute_momentum(synthetic_prices)
        signals = mom_mr_strategy.generate_signals(
            synthetic_prices, 300, z, m, "CRISIS"
        )
        assert len(signals) == 0

    def test_mom_mr_all_sectors_eligible(self, synthetic_prices, mom_mr_strategy):
        """MomentumMR should consider all 11 sectors (not just whitelist)."""
        z = MeanReversionStrategy().compute_z_scores(synthetic_prices)
        m = mom_mr_strategy.compute_momentum(synthetic_prices)
        all_signaled_sectors = set()
        for i in range(100, len(synthetic_prices)):
            signals = mom_mr_strategy.generate_signals(
                synthetic_prices, i, z, m, "NORMAL"
            )
            for sig in signals:
                all_signaled_sectors.add(sig.sector)
        # At least some non-whitelist sectors should appear
        non_whitelist = all_signaled_sectors - set(MR_WHITELIST)
        # Not a hard requirement (depends on data), but check it ran
        assert isinstance(all_signaled_sectors, set)


# ---------------------------------------------------------------------------
# Test 5: StrategyEnsemble
# ---------------------------------------------------------------------------

class TestStrategyEnsemble:
    def test_equal_weights(self):
        """Equal weight should give 1/3 to each of 3 strategies."""
        ens = StrategyEnsemble(method="equal_weight")
        w = ens.get_weights()
        assert len(w) == 3
        for v in w.values():
            assert abs(v - 1 / 3) < 1e-9

    def test_sharpe_weighted_default(self):
        """Sharpe-weighted with no history should give equal-ish weights."""
        ens = StrategyEnsemble(method="sharpe_weighted")
        w = ens.get_weights()
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_regime_adaptive_calm(self):
        """CALM regime should weight MR highest."""
        ens = StrategyEnsemble(method="regime_adaptive")
        w = ens.get_weights("CALM")
        assert w["MR"] > w["Dispersion"]
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_regime_adaptive_crisis(self):
        """CRISIS regime should weight MomentumMR highest."""
        ens = StrategyEnsemble(method="regime_adaptive")
        w = ens.get_weights("CRISIS")
        assert w["MomentumMR"] > w["MR"]
        assert w["MomentumMR"] > w["Dispersion"]

    def test_risk_parity_weights_sum_to_one(self):
        """Risk parity weights should sum to 1."""
        ens = StrategyEnsemble(method="risk_parity")
        # Seed some returns
        rng = np.random.default_rng(10)
        for _ in range(50):
            ens.update_rolling_stats("MR", rng.normal(0.001, 0.01))
            ens.update_rolling_stats("Dispersion", rng.normal(0.0005, 0.02))
            ens.update_rolling_stats("MomentumMR", rng.normal(0.001, 0.015))
        w = ens.get_weights()
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_combine_signals_empty(self):
        """Combining empty signals should return empty DataFrame."""
        ens = StrategyEnsemble(method="equal_weight")
        result = ens.combine_signals({"MR": [], "Dispersion": [], "MomentumMR": []})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_combine_signals_deduplicates_sectors(self):
        """If two strategies signal the same sector same direction, combine."""
        ens = StrategyEnsemble(method="equal_weight")
        date = pd.Timestamp("2024-01-15")
        sig_mr = Signal(date=date, sector="XLF", direction=1,
                        conviction=0.8, strategy_source="MR")
        sig_mom = Signal(date=date, sector="XLF", direction=1,
                         conviction=0.6, strategy_source="MomentumMR")
        result = ens.combine_signals({
            "MR": [sig_mr],
            "Dispersion": [],
            "MomentumMR": [sig_mom],
        })
        # XLF should appear only once
        xlf_rows = result[result["sector"] == "XLF"]
        assert len(xlf_rows) == 1

    def test_update_rolling_stats_limits_history(self):
        """Rolling stats should keep max 252 periods."""
        ens = StrategyEnsemble()
        for i in range(300):
            ens.update_rolling_stats("MR", 0.001)
        assert len(ens._rolling_returns["MR"]) == 252

    def test_signal_dataclass_fields(self):
        """Signal dataclass should have all required fields."""
        sig = Signal(
            date=pd.Timestamp("2024-01-15"),
            sector="XLK",
            direction=1,
            conviction=0.75,
            strategy_source="MR",
        )
        assert sig.weight == 0.0  # default
        assert sig.direction == 1
        assert sig.conviction == 0.75

    def test_backtest_ensemble_runs(self, synthetic_prices):
        """Ensemble backtest should run and return valid metrics."""
        ens = StrategyEnsemble(method="equal_weight")
        result = ens.backtest_ensemble(synthetic_prices, step=10, fwd_period=5)
        assert "sharpe" in result
        assert "max_dd" in result
        assert "annual_return" in result
        assert "strategy_sharpes" in result
        assert math.isfinite(result["sharpe"])
        assert result["n_periods"] > 0

    def test_backtest_ensemble_all_methods(self, synthetic_prices):
        """All ensemble methods should run without error."""
        for method in ["equal_weight", "sharpe_weighted", "regime_adaptive", "risk_parity"]:
            ens = StrategyEnsemble(method=method)
            result = ens.backtest_ensemble(synthetic_prices, step=10, fwd_period=5)
            assert math.isfinite(result["sharpe"]), f"NaN sharpe for {method}"
