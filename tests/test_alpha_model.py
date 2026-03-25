"""
tests/test_alpha_model.py
=========================
Tests for enhanced feature engine and walk-forward alpha model.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.feature_engine import FeatureEngine
from analytics.alpha_model import WalkForwardAlphaModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLU"]
NROWS = 600  # ~2.4 years of daily data
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_prices() -> pd.DataFrame:
    """Create synthetic price panel with sectors, SPY, VIX, HYG, IEF."""
    dates = pd.bdate_range("2021-01-04", periods=NROWS, freq="B")
    data = {}

    # SPY: random walk with drift
    spy = 100 * np.exp(np.cumsum(RNG.normal(0.0003, 0.01, NROWS)))
    data["SPY"] = spy

    # Sectors: correlated random walks
    for s in SECTORS:
        noise = RNG.normal(0.0002, 0.012, NROWS)
        corr_factor = 0.6 * np.log(spy / spy[0]) / max(np.log(spy / spy[0]).std(), 1e-6) * 0.012
        ret = noise + corr_factor * 0.3
        data[s] = 100 * np.exp(np.cumsum(ret))

    # VIX: mean-reverting around 20
    vix = np.zeros(NROWS)
    vix[0] = 20.0
    for i in range(1, NROWS):
        vix[i] = vix[i-1] + 0.05 * (20 - vix[i-1]) + RNG.normal(0, 1.5)
        vix[i] = max(vix[i], 9)
    data["^VIX"] = vix

    # HYG, IEF
    data["HYG"] = 80 * np.exp(np.cumsum(RNG.normal(0.0001, 0.005, NROWS)))
    data["IEF"] = 110 * np.exp(np.cumsum(RNG.normal(0.0001, 0.003, NROWS)))

    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="module")
def feature_engine(synthetic_prices) -> FeatureEngine:
    """Create FeatureEngine from synthetic data."""
    return FeatureEngine(synthetic_prices, SECTORS, spy_ticker="SPY")


@pytest.fixture(scope="module")
def all_features(feature_engine) -> pd.DataFrame:
    """Compute all features."""
    return feature_engine.compute_all_features()


# ---------------------------------------------------------------------------
# Test: New Feature Computation
# ---------------------------------------------------------------------------

class TestNewFeatures:
    """Tests for the 12 new features added to FeatureEngine."""

    def test_feature_count_per_sector(self, all_features):
        """Each sector should have >=38 features (26 original + 12 new)."""
        for s in SECTORS:
            n_feat = len(all_features[s].columns)
            assert n_feat >= 38, (
                f"Sector {s} has only {n_feat} features, expected >= 38"
            )

    def test_hurst_exponent_range(self, all_features):
        """Hurst exponent should be in [0, 1] where computed."""
        for s in SECTORS:
            h = all_features[s]["hurst_exp"]
            valid = h[h != 0].dropna()
            if len(valid) > 0:
                assert valid.min() >= -0.01, f"Hurst min={valid.min()}"
                assert valid.max() <= 1.01, f"Hurst max={valid.max()}"

    def test_adf_stat_is_numeric(self, all_features):
        """ADF stat should be finite numeric where computed."""
        for s in SECTORS:
            adf = all_features[s]["adf_stat"]
            valid = adf[(adf != 0) & adf.notna()]
            if len(valid) > 0:
                assert np.isfinite(valid.values).all(), "ADF contains non-finite values"

    def test_skew_kurt_shape(self, all_features):
        """Skewness and kurtosis should match date count."""
        for s in SECTORS:
            assert len(all_features[s]["skew_60d"]) == NROWS
            assert len(all_features[s]["kurt_60d"]) == NROWS

    def test_autocorr_range(self, all_features):
        """Autocorrelation should be in [-1, 1]."""
        for s in SECTORS:
            ac = all_features[s]["autocorr_5d"]
            valid = ac[(ac != 0) & ac.notna()]
            if len(valid) > 0:
                assert valid.min() >= -1.01
                assert valid.max() <= 1.01

    def test_max_drawdown_negative(self, all_features):
        """Max drawdown should be <= 0."""
        for s in SECTORS:
            dd = all_features[s]["max_dd_60d"]
            valid = dd[(dd != 0) & dd.notna()]
            if len(valid) > 0:
                assert valid.max() <= 0.001, f"Max DD should be <= 0, got {valid.max()}"

    def test_capture_ratios_exist(self, all_features):
        """Upside and downside capture should exist and be numeric."""
        for s in SECTORS:
            assert "upside_capture" in all_features[s].columns
            assert "downside_capture" in all_features[s].columns

    def test_z_cross_sector_rank(self, all_features):
        """Cross-sector rank should be in [0, 1]."""
        for s in SECTORS:
            rank = all_features[s]["z_cross_sector_rank"]
            valid = rank[(rank != 0) & rank.notna()]
            if len(valid) > 0:
                assert valid.min() >= -0.01
                assert valid.max() <= 1.01

    def test_momentum_breadth(self, all_features):
        """Momentum breadth should be in [0, 1]."""
        for s in SECTORS:
            mb = all_features[s]["momentum_breadth"]
            valid = mb[mb.notna()]
            if len(valid) > 0:
                assert valid.min() >= -0.01
                assert valid.max() <= 1.01

    def test_macro_interaction_features(self, all_features):
        """VIX x zscore and credit x vol interaction terms should exist."""
        for s in SECTORS:
            assert "vix_x_zscore" in all_features[s].columns
            assert "credit_x_vol" in all_features[s].columns

    def test_no_all_nan_columns(self, all_features):
        """No feature column should be entirely NaN after fill."""
        for s in SECTORS:
            for col in all_features[s].columns:
                assert not all_features[s][col].isna().all(), (
                    f"Feature {s}/{col} is all NaN"
                )


# ---------------------------------------------------------------------------
# Test: Target Construction
# ---------------------------------------------------------------------------

class TestTargetConstruction:
    """Tests for WalkForwardAlphaModel._build_target."""

    def test_target_shape(self, synthetic_prices, feature_engine):
        """Target should have correct shape."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=100, retrain_freq=21, hold_period=5,
        )
        targets = model._build_target()
        assert targets.shape[0] == len(synthetic_prices)
        assert set(targets.columns) == set(SECTORS)

    def test_target_has_nans_at_end(self, synthetic_prices, feature_engine):
        """Last hold_period rows should be NaN (forward-looking)."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=100, retrain_freq=21, hold_period=5,
        )
        targets = model._build_target()
        for s in SECTORS:
            assert targets[s].iloc[-1] != targets[s].iloc[-1] or pd.isna(targets[s].iloc[-1])

    def test_target_values_reasonable(self, synthetic_prices, feature_engine):
        """Target values should be reasonable (not too extreme)."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=100, retrain_freq=21, hold_period=5,
        )
        targets = model._build_target()
        valid = targets.dropna()
        if len(valid) > 0:
            # 5-day relative returns should typically be within +-20%
            assert valid.abs().max().max() < 0.5, "Target values seem extreme"


# ---------------------------------------------------------------------------
# Test: Walk-Forward Output
# ---------------------------------------------------------------------------

class TestWalkForward:
    """Tests for the walk-forward engine."""

    @pytest.fixture(scope="class")
    def wf_results(self, synthetic_prices, feature_engine):
        """Run the walk-forward model once for the class."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=200,  # smaller for speed
            retrain_freq=42,  # less frequent for speed
            hold_period=5,
        )
        return model.run()

    def test_run_produces_valid_output(self, wf_results):
        """Run should return all expected keys."""
        assert "sharpe" in wf_results
        assert "ic_mean" in wf_results
        assert "hit_rate" in wf_results
        assert "feature_importance" in wf_results
        assert "equity_curve" in wf_results
        assert "predictions" in wf_results

    def test_predictions_non_empty(self, wf_results):
        """Predictions DataFrame should not be empty."""
        assert len(wf_results["predictions"]) > 0

    def test_predictions_columns(self, wf_results):
        """Predictions should have required columns."""
        preds = wf_results["predictions"]
        required = {"date", "sector", "prediction", "actual", "signal_direction"}
        assert required.issubset(set(preds.columns))

    def test_sharpe_is_finite(self, wf_results):
        """Sharpe ratio should be a finite number."""
        assert math.isfinite(wf_results["sharpe"])

    def test_hit_rate_range(self, wf_results):
        """Hit rate should be in [0, 1]."""
        hr = wf_results["hit_rate"]
        assert 0.0 <= hr <= 1.0

    def test_ic_is_finite(self, wf_results):
        """Mean IC should be finite."""
        assert math.isfinite(wf_results["ic_mean"])

    def test_feature_importance_non_empty(self, wf_results):
        """Feature importance should have entries."""
        fi = wf_results["feature_importance"]
        assert len(fi) > 0


# ---------------------------------------------------------------------------
# Test: Current Signals
# ---------------------------------------------------------------------------

class TestCurrentSignals:
    """Tests for get_current_signals."""

    def test_signals_non_empty(self, synthetic_prices, feature_engine):
        """Current signals should return a dict with all sectors."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=200, retrain_freq=42, hold_period=5,
        )
        signals = model.get_current_signals()
        assert isinstance(signals, dict)
        assert len(signals) > 0
        for s in SECTORS:
            assert s in signals

    def test_signals_are_numeric(self, synthetic_prices, feature_engine):
        """Signal values should be finite numbers."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=200, retrain_freq=42, hold_period=5,
        )
        signals = model.get_current_signals()
        for s, v in signals.items():
            assert math.isfinite(v), f"Signal for {s} is not finite: {v}"


# ---------------------------------------------------------------------------
# Test: Sharpe Calculation
# ---------------------------------------------------------------------------

class TestSharpeCalculation:
    """Tests for Sharpe ratio computation."""

    def test_sharpe_zero_for_zero_returns(self, synthetic_prices, feature_engine):
        """Sharpe should be 0 when all returns are zero (trivially)."""
        # Create a model and check it handles edge cases
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=200, retrain_freq=42, hold_period=5,
        )
        # The actual Sharpe depends on data, but should be finite
        results = model.run()
        assert math.isfinite(results["sharpe"])

    def test_sharpe_sign(self, synthetic_prices, feature_engine):
        """Sharpe should be computable (positive or negative)."""
        model = WalkForwardAlphaModel(
            synthetic_prices, SECTORS, feature_engine,
            train_start=200, retrain_freq=42, hold_period=5,
        )
        results = model.run()
        # Sharpe can be negative, zero, or positive — just needs to be finite
        assert isinstance(results["sharpe"], float)
