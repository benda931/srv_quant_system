"""tests/test_freshness.py — ArtifactFreshness with mock paths, age, state."""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data_ops.freshness import (
    FRESH,
    MISSING,
    STALE,
    ArtifactFreshness,
    FreshnessSummary,
    _check_artifact,
    _last_expected_trading_date,
    assess_freshness,
)


# ---------------------------------------------------------------------------
# _last_expected_trading_date
# ---------------------------------------------------------------------------
def test_weekday_returns_same_day():
    monday = date(2026, 3, 16)   # known Monday
    assert _last_expected_trading_date(monday) == monday - timedelta(days=3)


def test_tuesday_to_friday_returns_same_day():
    for offset in range(1, 5):   # Tue–Fri
        d = date(2026, 3, 17) + timedelta(days=offset - 1)
        result = _last_expected_trading_date(d)
        assert result == d


def test_saturday_returns_friday():
    saturday = date(2026, 3, 21)   # known Saturday
    assert _last_expected_trading_date(saturday) == date(2026, 3, 20)


def test_sunday_returns_friday():
    sunday = date(2026, 3, 22)
    assert _last_expected_trading_date(sunday) == date(2026, 3, 20)


# ---------------------------------------------------------------------------
# _check_artifact — MISSING
# ---------------------------------------------------------------------------
def test_missing_artifact_has_missing_state(tmp_path):
    fake = tmp_path / "nonexistent.parquet"
    result = _check_artifact("prices", fake, max_age_hours=12.0)
    assert result.state == MISSING
    assert result.exists is False
    assert result.age_hours is None
    assert result.warning is not None


# ---------------------------------------------------------------------------
# _check_artifact — FRESH
# ---------------------------------------------------------------------------
def test_fresh_artifact_state(tmp_path):
    p = tmp_path / "prices.parquet"
    p.write_bytes(b"dummy")
    result = _check_artifact("prices", p, max_age_hours=24.0)
    assert result.state == FRESH
    assert result.exists is True
    assert result.age_hours >= 0.0
    assert result.warning is None


def test_age_hours_is_non_negative(tmp_path):
    p = tmp_path / "w.parquet"
    p.write_bytes(b"x")
    result = _check_artifact("weights", p, max_age_hours=12.0)
    assert isinstance(result.age_hours, float)
    assert result.age_hours >= 0.0


# ---------------------------------------------------------------------------
# _check_artifact — STALE
# ---------------------------------------------------------------------------
def test_stale_when_max_age_exceeded(tmp_path):
    import os, time as _time
    p = tmp_path / "fund.parquet"
    p.write_bytes(b"data")
    # Backdate mtime by 25 hours so any threshold < 25 h classifies as STALE
    old_mtime = _time.time() - 25 * 3600
    os.utime(p, (old_mtime, old_mtime))
    result = _check_artifact("fundamentals", p, max_age_hours=12.0)
    assert result.state == STALE
    assert result.warning is not None


def test_max_age_zero_disables_staleness_check(tmp_path):
    p = tmp_path / "p.parquet"
    p.write_bytes(b"x")
    result = _check_artifact("prices", p, max_age_hours=0.0)
    assert result.state == FRESH


# ---------------------------------------------------------------------------
# assess_freshness — integration
# ---------------------------------------------------------------------------
def test_all_missing_gives_overall_missing(tmp_path):
    summary = assess_freshness(
        tmp_path / "a.parquet",
        tmp_path / "b.parquet",
        tmp_path / "c.parquet",
        max_age_hours=12.0,
    )
    assert summary.overall_state == MISSING
    assert summary.degraded is True


def test_all_fresh_gives_overall_fresh(tmp_path, monkeypatch):
    for name in ("prices.parquet", "fundamentals.parquet", "weights.parquet"):
        p = tmp_path / name
        # Write a minimal parquet with a recent date index
        df = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-03-21"]))
        df.to_parquet(p)
    # Monkeypatch date.today so price-gap check passes
    import data_ops.freshness as freshness_mod
    monkeypatch.setattr(freshness_mod, "_MAX_PRICE_GAP_DAYS", 9999)
    summary = assess_freshness(
        tmp_path / "prices.parquet",
        tmp_path / "fundamentals.parquet",
        tmp_path / "weights.parquet",
        max_age_hours=24.0,
    )
    assert summary.overall_state == FRESH
    assert summary.degraded is False


def test_summary_has_three_artifact_keys(tmp_path):
    summary = assess_freshness(
        tmp_path / "p.parquet",
        tmp_path / "f.parquet",
        tmp_path / "w.parquet",
    )
    assert set(summary.artifacts.keys()) == {"prices", "fundamentals", "weights"}
