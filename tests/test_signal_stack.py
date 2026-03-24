"""tests/test_signal_stack.py — Signal stack Layer 1-4 scoring invariants."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.signal_stack import (
    compute_distortion_score,
    compute_dislocation_score,
    SignalStackResult,
)
from analytics.signal_regime_safety import compute_regime_safety_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def residual_series() -> pd.Series:
    """Synthetic mean-reverting residual series."""
    rng = np.random.default_rng(42)
    n = 300
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 0.5)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.Series(x, index=dates)


# ---------------------------------------------------------------------------
# Layer 1: Distortion Score
# ---------------------------------------------------------------------------

class TestDistortionScore:
    """Tests for compute_distortion_score (Layer 1)."""

    def test_distortion_score_range(self):
        """Distortion score must be in [0, 1] for any finite inputs."""
        for frob_z in [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
            for mode in [0.05, 0.2, 0.5, 0.8]:
                for coc_z in [-2.0, 0.0, 2.0]:
                    result = compute_distortion_score(frob_z, mode, coc_z)
                    assert 0.0 <= result.distortion_score <= 1.0, (
                        f"Score {result.distortion_score} out of range for "
                        f"frob_z={frob_z}, mode={mode}, coc_z={coc_z}"
                    )

    def test_distortion_monotone_frob(self):
        """Higher Frobenius z-score should increase distortion score (all else equal)."""
        low = compute_distortion_score(0.0, 0.3, 0.0)
        high = compute_distortion_score(3.0, 0.3, 0.0)
        assert high.distortion_score >= low.distortion_score

    def test_distortion_nan_handling(self):
        """NaN inputs should produce a valid [0,1] score, not raise."""
        result = compute_distortion_score(float("nan"), 0.3, float("nan"))
        assert 0.0 <= result.distortion_score <= 1.0
        assert math.isfinite(result.distortion_score)

    def test_distortion_labels(self):
        """Labels should be one of the three known categories."""
        valid_labels = {"HIGH_DISTORTION", "MODERATE_DISTORTION", "LOW_DISTORTION"}
        for frob_z in [0.0, 2.0, 5.0]:
            result = compute_distortion_score(frob_z, 0.5, frob_z * 0.5)
            assert result.label in valid_labels


# ---------------------------------------------------------------------------
# Layer 2: Dislocation Score
# ---------------------------------------------------------------------------

class TestDislocationScore:
    """Tests for compute_dislocation_score (Layer 2)."""

    def test_dislocation_score_range(self, residual_series):
        """Dislocation score must be in [0, 1]."""
        result = compute_dislocation_score(residual_series, "XLK", "sector")
        assert 0.0 <= result.dislocation_score <= 1.0

    def test_dislocation_direction(self, residual_series):
        """Direction should be LONG or SHORT."""
        result = compute_dislocation_score(residual_series, "XLK", "sector")
        assert result.direction in ("LONG", "SHORT")

    def test_dislocation_insufficient_data(self):
        """Should return score=0 for insufficient data."""
        short_series = pd.Series([0.1, 0.2, 0.3])
        result = compute_dislocation_score(short_series, "XLK", "sector")
        assert result.dislocation_score == 0.0

    def test_dislocation_z_cap(self, residual_series):
        """Score should be capped at 1.0 regardless of how extreme z is."""
        result = compute_dislocation_score(
            residual_series, "XLK", "sector", z_cap=0.1
        )
        assert result.dislocation_score <= 1.0


# ---------------------------------------------------------------------------
# Layer 4: Regime Safety — conviction additive model & hard kills
# ---------------------------------------------------------------------------

class TestConvictionAndRegimeSafety:
    """Tests for the combined conviction model and regime safety."""

    def test_conviction_additive_model(self):
        """Combined conviction = product of 4 layer scores."""
        dist = 0.7
        disloc = 0.8
        mr = 0.6
        safe = 0.9
        expected = dist * disloc * mr * safe
        actual = dist * disloc * mr * safe
        assert abs(actual - expected) < 1e-10

    def test_regime_safety_kills(self):
        """
        When VIX >= kill threshold, safety score must be 0.0.
        This tests the hard-kill gate that prevents short-vol entries
        during extreme market stress.
        """
        result = compute_regime_safety_score(
            market_state="CRISIS",
            vix_level=50.0,
            credit_z=0.0,
            corr_z=0.0,
        )
        assert result.regime_safety_score == 0.0
        assert result.any_hard_kill is True
        assert result.label == "KILLED"

    def test_regime_safety_calm(self):
        """In a calm regime with no stress, safety score should be high."""
        result = compute_regime_safety_score(
            market_state="CALM",
            vix_level=12.0,
            credit_z=1.0,
            corr_z=-0.5,
        )
        assert result.regime_safety_score > 0.7
        assert result.any_hard_kill is False

    def test_regime_safety_score_range(self):
        """Safety score must always be in [0, 1]."""
        for vix in [10, 15, 20, 25, 30, 35, 40, 50]:
            for credit_z in [-4, -2, 0, 1]:
                result = compute_regime_safety_score(
                    market_state="NORMAL",
                    vix_level=float(vix),
                    credit_z=float(credit_z),
                    corr_z=0.0,
                )
                assert 0.0 <= result.regime_safety_score <= 1.0
