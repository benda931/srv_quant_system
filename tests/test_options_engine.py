"""tests/test_options_engine.py — Black-Scholes, Greeks, and implied correlation tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from analytics.options_engine import (
    bs_call,
    bs_put,
    bs_straddle,
    bs_d1,
    bs_d2,
    compute_greeks,
)


# ---------------------------------------------------------------------------
# Constants for tests
# ---------------------------------------------------------------------------
S = 100.0    # Spot price
K = 100.0    # ATM strike
T = 30 / 365  # 30 days to expiry
R = 0.05     # Risk-free rate
SIGMA = 0.20  # 20% annual vol

# Tolerance for float comparisons
TOL = 1e-6


# ---------------------------------------------------------------------------
# Black-Scholes Put-Call Parity
# ---------------------------------------------------------------------------

class TestBSCallPutParity:
    """Verify put-call parity: C - P = S - K * exp(-rT)."""

    def test_bs_call_put_parity_atm(self):
        """ATM put-call parity."""
        call = bs_call(S, K, T, R, SIGMA)
        put = bs_put(S, K, T, R, SIGMA)
        expected = S - K * math.exp(-R * T)
        assert abs((call - put) - expected) < 1e-8, (
            f"Put-call parity violated: C-P={call-put:.8f}, S-Ke^(-rT)={expected:.8f}"
        )

    def test_bs_call_put_parity_otm(self):
        """OTM put-call parity (K > S)."""
        K_otm = 110.0
        call = bs_call(S, K_otm, T, R, SIGMA)
        put = bs_put(S, K_otm, T, R, SIGMA)
        expected = S - K_otm * math.exp(-R * T)
        assert abs((call - put) - expected) < 1e-8

    def test_bs_call_put_parity_itm(self):
        """ITM put-call parity (K < S)."""
        K_itm = 90.0
        call = bs_call(S, K_itm, T, R, SIGMA)
        put = bs_put(S, K_itm, T, R, SIGMA)
        expected = S - K_itm * math.exp(-R * T)
        assert abs((call - put) - expected) < 1e-8

    def test_bs_call_put_parity_various_vols(self):
        """Parity must hold for any volatility."""
        for sigma in [0.05, 0.10, 0.20, 0.40, 0.80, 1.50]:
            call = bs_call(S, K, T, R, sigma)
            put = bs_put(S, K, T, R, sigma)
            expected = S - K * math.exp(-R * T)
            assert abs((call - put) - expected) < 1e-7, (
                f"Parity violated for sigma={sigma}"
            )

    def test_bs_call_put_parity_long_expiry(self):
        """Parity for long-dated option (1 year)."""
        T_1y = 1.0
        call = bs_call(S, K, T_1y, R, SIGMA)
        put = bs_put(S, K, T_1y, R, SIGMA)
        expected = S - K * math.exp(-R * T_1y)
        assert abs((call - put) - expected) < 1e-8

    def test_bs_expired(self):
        """At expiry, call = max(S-K, 0), put = max(K-S, 0)."""
        assert bs_call(105, 100, 0, R, SIGMA) == 5.0
        assert bs_call(95, 100, 0, R, SIGMA) == 0.0
        assert bs_put(95, 100, 0, R, SIGMA) == 5.0
        assert bs_put(105, 100, 0, R, SIGMA) == 0.0


# ---------------------------------------------------------------------------
# Greeks for ATM Straddle
# ---------------------------------------------------------------------------

class TestGreeksATMStraddle:
    """Test Greeks for ATM straddle positions."""

    def test_greeks_atm_straddle(self):
        """ATM straddle should have near-zero delta."""
        greeks = compute_greeks(S, SIGMA, T, R)
        # ATM straddle delta should be very close to 0
        assert abs(greeks["delta"]) < 0.15, (
            f"ATM straddle delta={greeks['delta']:.6f}, expected near 0"
        )

    def test_gamma_positive(self):
        """Straddle always has positive gamma (long convexity)."""
        greeks = compute_greeks(S, SIGMA, T, R)
        assert greeks["gamma"] > 0

    def test_vega_positive(self):
        """Long straddle has positive vega (benefits from vol increase)."""
        greeks = compute_greeks(S, SIGMA, T, R)
        assert greeks["vega"] > 0

    def test_theta_negative(self):
        """Long straddle has negative theta (time decay costs money)."""
        greeks = compute_greeks(S, SIGMA, T, R)
        assert greeks["theta"] < 0

    def test_greeks_zero_vol(self):
        """Greeks should return zeros for zero volatility."""
        greeks = compute_greeks(S, 0.0, T, R)
        assert greeks["delta"] == 0
        assert greeks["gamma"] == 0
        assert greeks["vega"] == 0
        assert greeks["theta"] == 0

    def test_greeks_zero_dte(self):
        """Greeks should return zeros at expiry."""
        greeks = compute_greeks(S, SIGMA, 0.0, R)
        assert greeks["delta"] == 0
        assert greeks["gamma"] == 0

    def test_gamma_increases_near_expiry(self):
        """ATM gamma should increase as expiry approaches."""
        gamma_30d = compute_greeks(S, SIGMA, 30 / 365, R)["gamma"]
        gamma_5d = compute_greeks(S, SIGMA, 5 / 365, R)["gamma"]
        assert gamma_5d > gamma_30d

    def test_vega_decreases_near_expiry(self):
        """Vega should decrease as expiry approaches (less vol sensitivity)."""
        vega_90d = compute_greeks(S, SIGMA, 90 / 365, R)["vega"]
        vega_5d = compute_greeks(S, SIGMA, 5 / 365, R)["vega"]
        assert vega_90d > vega_5d


# ---------------------------------------------------------------------------
# Implied Correlation Range
# ---------------------------------------------------------------------------

class TestImpliedCorrelationRange:
    """Tests for implied correlation index boundaries."""

    def test_implied_corr_formula_bounds(self):
        """
        Implied correlation is computed from index vs sector variances:
          rho_impl = (sigma_I^2 - sum(w_i^2 * sigma_i^2)) / (2 * sum(w_i*w_j*sigma_i*sigma_j))

        For uniform weights and equal vols, rho = (N*sigma^2 - N*(1/N)^2*sigma^2) / ...
        The result should be in [-1, 1] for well-formed inputs.
        """
        # Simulate: 4 sectors, equal weight, equal vol
        N = 4
        w = np.array([1 / N] * N)
        sigma_sectors = np.array([0.20] * N)
        sigma_index = 0.18  # Index vol < avg sector vol (positive correlation)

        # Numerator: sigma_I^2 - sum(w_i^2 * sigma_i^2)
        num = sigma_index ** 2 - np.sum(w ** 2 * sigma_sectors ** 2)

        # Denominator: 2 * sum over i<j of w_i*w_j*sigma_i*sigma_j
        denom = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                denom += w[i] * w[j] * sigma_sectors[i] * sigma_sectors[j]
        denom *= 2.0

        if denom > 1e-12:
            rho_impl = num / denom
        else:
            rho_impl = 0.0

        # Implied correlation should be in [-1, 1]
        assert -1.0 <= rho_impl <= 1.0, f"Implied corr {rho_impl} out of range"

    def test_perfect_corr_all_equal(self):
        """If all sectors are perfectly correlated, implied corr ~ 1."""
        N = 4
        w = np.array([1 / N] * N)
        sigma = 0.20
        sigma_sectors = np.array([sigma] * N)
        # If perfectly correlated, sigma_I = sum(w_i * sigma_i) = sigma
        sigma_index = sigma  # Perfect correlation case

        num = sigma_index ** 2 - np.sum(w ** 2 * sigma_sectors ** 2)
        denom = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                denom += w[i] * w[j] * sigma_sectors[i] * sigma_sectors[j]
        denom *= 2.0

        if denom > 1e-12:
            rho_impl = num / denom
        else:
            rho_impl = 0.0

        assert abs(rho_impl - 1.0) < 0.05, f"Expected ~1.0, got {rho_impl}"

    def test_straddle_price_positive(self):
        """Straddle price must always be positive (call + put > 0 when T > 0)."""
        for sigma in [0.05, 0.10, 0.20, 0.50, 1.0]:
            price = bs_straddle(S, K, T, R, sigma)
            assert price > 0, f"Straddle price {price} <= 0 for sigma={sigma}"

    def test_straddle_increases_with_vol(self):
        """Straddle value should increase with implied volatility."""
        strad_lo = bs_straddle(S, K, T, R, 0.10)
        strad_hi = bs_straddle(S, K, T, R, 0.40)
        assert strad_hi > strad_lo
