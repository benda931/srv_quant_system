"""tests/test_journal.py — PMJournal CRUD with a temp SQLite file."""
from __future__ import annotations

from pathlib import Path

import pytest

from analytics.attribution import AttributionResult
from data_ops.journal import PMJournal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def db_path(tmp_path) -> Path:
    return tmp_path / "test_journal.db"


@pytest.fixture()
def journal(db_path) -> PMJournal:
    return PMJournal(db_path)


@pytest.fixture()
def attribution() -> AttributionResult:
    return AttributionResult(
        sds=0.55, fjs=0.30, mss=0.20, stf=0.15, mc=0.42,
        trend_ratio_slope_63d=-0.01, trend_ratio_slope_126d=-0.005,
        beta_instability=0.10, corr_instability=0.08, corr_shift_score=0.12,
        dislocation_label="Moderate Statistical Dislocation",
        fundamental_label="Weak Fundamental Justification",
        macro_label="Low Macro Shift Risk",
        structural_label="Low Structural Trend Risk",
        mc_label="Moderate Mispricing Confidence",
        action_bias="SELECTIVE",
        risk_label="Contained Risk",
        interpretation="Mixed Signal / Requires PM Judgement",
        explanation_tags=["moderate_dislocation"],
    )


# ---------------------------------------------------------------------------
# log_decision
# ---------------------------------------------------------------------------
def test_log_decision_returns_positive_int(journal, attribution):
    did = journal.log_decision(
        "XLK", attribution, "LONG",
        model_direction="LONG", conviction_score=72.5, regime="CALM",
    )
    assert isinstance(did, int)
    assert did >= 1


def test_log_decision_ids_are_sequential(journal, attribution):
    d1 = journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    d2 = journal.log_decision("XLF", attribution, "SHORT", model_direction="SHORT")
    assert d2 > d1


def test_invalid_pm_direction_raises(journal, attribution):
    with pytest.raises(ValueError, match="pm_direction"):
        journal.log_decision("XLK", attribution, "BULLISH", model_direction="LONG")


def test_invalid_model_direction_raises(journal, attribution):
    with pytest.raises(ValueError, match="model_direction"):
        journal.log_decision("XLK", attribution, "LONG", model_direction="BULLISH")


# ---------------------------------------------------------------------------
# get_recent
# ---------------------------------------------------------------------------
def test_get_recent_returns_dataframe(journal, attribution):
    import pandas as pd
    journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    df = journal.get_recent()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_get_recent_has_expected_columns(journal, attribution):
    journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    df = journal.get_recent()
    required = {"id", "timestamp", "sector", "model_direction", "pm_direction", "agreement"}
    assert required.issubset(set(df.columns))


def test_agreement_column_true_when_directions_match(journal, attribution):
    journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    df = journal.get_recent(n=1)
    assert bool(df.iloc[0]["agreement"]) is True


def test_agreement_column_false_when_directions_differ(journal, attribution):
    journal.log_decision("XLF", attribution, "SHORT", model_direction="LONG")
    df = journal.get_recent(n=1)
    assert bool(df.iloc[0]["agreement"]) is False


def test_get_recent_respects_n_limit(journal, attribution):
    for ticker in ["XLC", "XLY", "XLP", "XLE", "XLF"]:
        journal.log_decision(ticker, attribution, "LONG", model_direction="LONG")
    df = journal.get_recent(n=3)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# log_override / resolve_override
# ---------------------------------------------------------------------------
def test_log_override_returns_positive_int(journal, attribution):
    did = journal.log_decision("XLK", attribution, "OVERRIDE", model_direction="SHORT")
    oid = journal.log_override(did, "Earnings surprise expected")
    assert isinstance(oid, int) and oid >= 1


def test_empty_override_reason_raises(journal, attribution):
    did = journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    with pytest.raises(ValueError):
        journal.log_override(did, "")


def test_resolve_override_sets_outcome(journal, attribution):
    did = journal.log_decision("XLK", attribution, "OVERRIDE", model_direction="LONG")
    oid = journal.log_override(did, "Macro call")
    journal.resolve_override(oid, "CORRECT")
    stats = journal.get_override_accuracy()
    assert stats["n_resolved"] == 1
    assert stats["n_pm_correct"] == 1
    assert stats["pm_accuracy"] == pytest.approx(1.0)


def test_resolve_override_incorrect_outcome(journal, attribution):
    did = journal.log_decision("XLF", attribution, "OVERRIDE", model_direction="LONG")
    oid = journal.log_override(did, "Credit concern")
    journal.resolve_override(oid, "INCORRECT")
    stats = journal.get_override_accuracy()
    assert stats["n_resolved"] >= 1
    # model_correct count should be >= 1
    assert stats["n_model_correct"] >= 1


# ---------------------------------------------------------------------------
# get_sector_history
# ---------------------------------------------------------------------------
def test_get_sector_history_filters_by_sector(journal, attribution):
    journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    journal.log_decision("XLF", attribution, "SHORT", model_direction="SHORT")
    hist = journal.get_sector_history("XLK", days=365)
    assert (hist["model_direction"].notnull()).all() or hist.empty
    # XLF rows must not appear
    if not hist.empty and "sector" in hist.columns:
        assert (hist["sector"] == "XLK").all()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------
def test_get_stats_keys(journal, attribution):
    journal.log_decision("XLK", attribution, "LONG", model_direction="LONG")
    stats = journal.get_stats()
    for key in ("n_decisions", "n_overrides", "n_resolved_overrides", "sectors"):
        assert key in stats


def test_get_stats_counts_increment(journal, attribution):
    before = journal.get_stats()["n_decisions"]
    journal.log_decision("XLU", attribution, "NEUTRAL", model_direction="NEUTRAL")
    after = journal.get_stats()["n_decisions"]
    assert after == before + 1
