"""Tests for analytics.feature_engine."""
import numpy as np
import pandas as pd
import pytest
from analytics.feature_engine import FeatureEngine


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    sectors = ["XLK", "XLF", "XLV", "XLU"]
    macro = ["SPY", "^VIX", "HYG", "IEF", "TLT", "GLD", "UUP"]
    all_tickers = sectors + macro
    data = {}
    for t in all_tickers:
        if t == "^VIX":
            data[t] = 18 + np.cumsum(np.random.randn(n) * 0.5)
        else:
            data[t] = np.exp(np.cumsum(np.random.randn(n) * 0.01)) * 100
    return pd.DataFrame(data, index=dates)


class TestFeatureEngine:
    def test_init(self, sample_prices):
        fe = FeatureEngine(sample_prices, ["XLK", "XLF", "XLV", "XLU"])
        assert fe is not None

    def test_compute_all_features(self, sample_prices):
        fe = FeatureEngine(sample_prices, ["XLK", "XLF", "XLV", "XLU"])
        features = fe.compute_all_features()
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert features.shape[1] > 20  # Should have many features

    def test_feature_count_per_sector(self, sample_prices):
        fe = FeatureEngine(sample_prices, ["XLK", "XLF", "XLV", "XLU"])
        features = fe.compute_all_features()
        # Multi-index columns: (sector, feature_name)
        if isinstance(features.columns, pd.MultiIndex):
            sectors_in_features = set(features.columns.get_level_values(0))
            assert len(sectors_in_features) == 4

    def test_no_all_nan_columns(self, sample_prices):
        fe = FeatureEngine(sample_prices, ["XLK", "XLF"])
        features = fe.compute_all_features()
        all_nan = features.isna().all()
        assert all_nan.sum() == 0, f"All-NaN columns: {all_nan[all_nan].index.tolist()}"

    def test_select_features(self, sample_prices):
        fe = FeatureEngine(sample_prices, ["XLK", "XLF"])
        features = fe.compute_all_features()
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = ["_".join(str(c) for c in col) for col in features.columns]
        target = pd.Series(np.random.randn(len(features)), index=features.index)
        common = features.dropna().index.intersection(target.dropna().index)
        if len(common) > 50:
            selected = fe.select_features(features.loc[common], target.loc[common], top_k=5)
            assert len(selected) <= 5
