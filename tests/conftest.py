"""
tests/conftest.py
==================
Shared pytest fixtures for the SRV Quantamental DSS test suite.

Provides:
  - fake_prices: 504 days of realistic sector ETF prices
  - fake_returns: log returns from fake_prices
  - settings: get_settings() with proper sys.path
  - engine: loaded QuantEngine (expensive — use sparingly)
"""
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settings():
    """Load settings (session-scoped — one per test run)."""
    from config.settings import get_settings
    return get_settings()


@pytest.fixture
def fake_prices():
    """Generate realistic fake sector ETF prices (504 trading days)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    tickers = ["XLK", "XLF", "XLV", "XLE", "XLU", "SPY",
               "XLI", "XLB", "XLC", "XLY", "XLP", "XLRE"]

    data = {}
    for t in tickers:
        base = 100 + np.random.randn() * 20
        rets = np.random.randn(504) * 0.012 + 0.0003
        data[t] = base * np.cumprod(1 + rets)

    data["^VIX"] = np.clip(18 + np.cumsum(np.random.randn(504) * 0.5), 10, 80)
    data["^TNX"] = np.clip(2.5 + np.cumsum(np.random.randn(504) * 0.02), 0.5, 6)
    data["DX-Y.NYB"] = 100 + np.cumsum(np.random.randn(504) * 0.1)
    data["HYG"] = 80 + np.cumsum(np.random.randn(504) * 0.05)
    data["IEF"] = 105 + np.cumsum(np.random.randn(504) * 0.03)

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def fake_returns(fake_prices):
    """Log returns from fake_prices."""
    return np.log(fake_prices / fake_prices.shift(1)).dropna()


@pytest.fixture
def sector_list():
    """Standard 11 sector tickers."""
    return ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV",
            "XLI", "XLB", "XLRE", "XLK", "XLU"]
