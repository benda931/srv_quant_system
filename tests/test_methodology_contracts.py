"""
tests/test_methodology_contracts.py
====================================
Deterministic tests for the 5 locked methodology contracts:
1. JSON report schema validation
2. APPROVED / CONDITIONAL / REJECTED decision contract
3. Promotion gate criteria
4. Experiment lineage versioning
5. machine_summary stable contract
"""
import json
import pytest
from agents.methodology.report_schema import (
    REPORT_SCHEMA_VERSION,
    VALID_PROMOTION_DECISIONS,
    VALID_CLASSIFICATIONS,
    GOVERNANCE_REQUIRED_KEYS,
    PromotionCriteria,
    DEFAULT_PROMOTION_CRITERIA,
    evaluate_promotion,
    MachineSummary,
    validate_report,
    validate_machine_summary,
)


# =====================================================================
# 1. PROMOTION DECISIONS — only 3 valid values
# =====================================================================

class TestPromotionDecisionContract:
    """The system ONLY produces APPROVED, CONDITIONAL, or REJECTED."""

    def test_valid_decisions_are_exactly_three(self):
        assert VALID_PROMOTION_DECISIONS == {"APPROVED", "CONDITIONAL", "REJECTED"}

    def test_approved_requires_all_hard_gates_pass(self):
        """All hard criteria pass, no warnings → APPROVED."""
        decision, fails, warns = evaluate_promotion(
            net_sharpe=0.5,
            max_drawdown=-0.10,
            total_trades=200,
            positive_regimes=3,
            deflated_sharpe=0.2,
            tail_risk_score=0.6,
            cost_drag_pct=15.0,
            stability_score=0.7,
        )
        assert decision == "APPROVED"
        assert fails == []
        assert warns == []

    def test_conditional_on_soft_warning_only(self):
        """All hard pass, but tail_risk warning → CONDITIONAL."""
        decision, fails, warns = evaluate_promotion(
            net_sharpe=0.5,
            max_drawdown=-0.10,
            total_trades=200,
            positive_regimes=3,
            deflated_sharpe=0.2,
            tail_risk_score=0.2,   # Below 0.3 → warning
            cost_drag_pct=15.0,
            stability_score=0.7,
        )
        assert decision == "CONDITIONAL"
        assert len(fails) == 0
        assert len(warns) >= 1
        assert any("tail_risk" in w for w in warns)

    def test_rejected_on_any_hard_fail(self):
        """net_sharpe below minimum → REJECTED regardless of other metrics."""
        decision, fails, warns = evaluate_promotion(
            net_sharpe=0.1,   # Below 0.3
            max_drawdown=-0.05,
            total_trades=500,
            positive_regimes=4,
            deflated_sharpe=0.5,
            tail_risk_score=0.8,
            cost_drag_pct=5.0,
            stability_score=0.9,
        )
        assert decision == "REJECTED"
        assert len(fails) >= 1
        assert any("net_sharpe" in f for f in fails)

    def test_rejected_on_multiple_hard_fails(self):
        """Multiple hard failures → all listed in fail_reasons."""
        decision, fails, warns = evaluate_promotion(
            net_sharpe=0.1,       # fail
            max_drawdown=-0.25,   # fail
            total_trades=50,      # fail
            positive_regimes=1,   # fail
            deflated_sharpe=-0.5, # fail
            tail_risk_score=0.1,  # warning
            cost_drag_pct=50.0,   # fail
            stability_score=0.2,  # warning
        )
        assert decision == "REJECTED"
        assert len(fails) >= 5  # at least 5 hard fails


# =====================================================================
# 2. PROMOTION GATE — each criterion tested individually
# =====================================================================

class TestPromotionGateCriteria:
    """Each gate criterion is tested in isolation."""

    BASE = dict(
        net_sharpe=0.5,
        max_drawdown=-0.10,
        total_trades=200,
        positive_regimes=3,
        deflated_sharpe=0.2,
        tail_risk_score=0.6,
        cost_drag_pct=15.0,
        stability_score=0.7,
    )

    def _with(self, **overrides):
        params = {**self.BASE, **overrides}
        return evaluate_promotion(**params)

    def test_gate_net_sharpe(self):
        d, f, w = self._with(net_sharpe=0.29)
        assert d == "REJECTED"
        assert any("net_sharpe" in r for r in f)

    def test_gate_net_sharpe_boundary(self):
        d, f, w = self._with(net_sharpe=0.30)
        assert d in ("APPROVED", "CONDITIONAL")  # 0.30 >= 0.3

    def test_gate_max_drawdown(self):
        d, f, w = self._with(max_drawdown=-0.16)
        assert d == "REJECTED"
        assert any("max_drawdown" in r for r in f)

    def test_gate_min_trades(self):
        d, f, w = self._with(total_trades=99)
        assert d == "REJECTED"
        assert any("total_trades" in r for r in f)

    def test_gate_regime_consistency(self):
        d, f, w = self._with(positive_regimes=1)
        assert d == "REJECTED"
        assert any("regime" in r for r in f)

    def test_gate_deflated_sharpe(self):
        d, f, w = self._with(deflated_sharpe=-0.1)
        assert d == "REJECTED"
        assert any("deflated_sharpe" in r or "overfit" in r for r in f)

    def test_gate_cost_drag(self):
        d, f, w = self._with(cost_drag_pct=35.0)
        assert d == "REJECTED"
        assert any("cost_drag" in r for r in f)

    def test_gate_tail_risk_warning_not_fail(self):
        """tail_risk is a WARNING, not a hard fail."""
        d, f, w = self._with(tail_risk_score=0.1)
        assert d == "CONDITIONAL"  # warning, not rejected
        assert len(f) == 0
        assert any("tail_risk" in r for r in w)

    def test_gate_stability_warning_not_fail(self):
        """stability is a WARNING, not a hard fail."""
        d, f, w = self._with(stability_score=0.3)
        assert d == "CONDITIONAL"
        assert len(f) == 0
        assert any("stability" in r for r in w)

    def test_custom_criteria(self):
        """Custom criteria override defaults."""
        strict = PromotionCriteria(min_net_sharpe=1.0, min_trades=500)
        d, f, w = evaluate_promotion(
            net_sharpe=0.8, max_drawdown=-0.05, total_trades=300,
            positive_regimes=4, deflated_sharpe=0.5,
            tail_risk_score=0.8, cost_drag_pct=5.0, stability_score=0.9,
            criteria=strict,
        )
        assert d == "REJECTED"
        assert any("net_sharpe" in r for r in f)
        assert any("total_trades" in r for r in f)


# =====================================================================
# 3. MACHINE_SUMMARY — stable contract
# =====================================================================

class TestMachineSummaryContract:
    """machine_summary must have all required fields with correct types."""

    def test_default_construction(self):
        ms = MachineSummary()
        d = ms.to_dict()
        assert d["schema_version"] == REPORT_SCHEMA_VERSION
        assert d["n_strategies"] == 0
        assert d["best_strategy_decision"] == "REJECTED"
        assert d["validation_complete"] is False

    def test_full_construction(self):
        ms = MachineSummary(
            schema_version=REPORT_SCHEMA_VERSION,
            methodology_version="2.0",
            experiment_id="abc123",
            timestamp="2026-03-25T10:00:00Z",
            n_strategies=19,
            n_approved=2,
            n_conditional=5,
            n_rejected=12,
            best_strategy_name="ALPHA_WHITELIST_MR",
            best_strategy_decision="CONDITIONAL",
            best_net_sharpe=0.45,
            best_gross_sharpe=0.52,
            best_deflated_sharpe=0.31,
            best_hit_rate=0.57,
            best_max_drawdown=-0.12,
            best_total_trades=272,
            best_classification="CORE",
            current_regime="TENSION",
            best_per_regime={"CALM": "ALPHA_WHITELIST_MR", "TENSION": "ALPHA_WHITELIST_MR"},
            validation_complete=True,
            governance_complete=True,
            overfitting_flag=False,
            cost_drag_flag=False,
            mean_robustness=0.6,
            mean_stability=0.55,
            final_rankings={"ALPHA_WHITELIST_MR": 1, "PCA_Z_REVERSAL": 2},
            optimizer_should_tune=True,
            strategies_to_disable=["DISPERSION_TIMING"],
            strategies_to_observe=["MULTI_FACTOR"],
        )
        d = ms.to_dict()

        # Type checks
        assert isinstance(d["schema_version"], str)
        assert isinstance(d["n_strategies"], int)
        assert isinstance(d["n_approved"], int)
        assert isinstance(d["best_net_sharpe"], float)
        assert isinstance(d["validation_complete"], bool)
        assert isinstance(d["final_rankings"], dict)
        assert isinstance(d["strategies_to_disable"], list)
        assert isinstance(d["optimizer_should_tune"], bool)

        # Value checks
        assert d["best_strategy_decision"] in VALID_PROMOTION_DECISIONS
        assert d["best_classification"] in VALID_CLASSIFICATIONS

    def test_validate_machine_summary_valid(self):
        ms = MachineSummary(
            experiment_id="x", n_strategies=5, n_approved=1, n_rejected=3,
            best_strategy_name="A", best_strategy_decision="APPROVED",
            best_net_sharpe=0.5, validation_complete=True, governance_complete=True,
        )
        errors = validate_machine_summary(ms.to_dict())
        assert errors == []

    def test_validate_machine_summary_missing_field(self):
        d = {"schema_version": "2.0"}  # missing most fields
        errors = validate_machine_summary(d)
        assert len(errors) >= 5

    def test_validate_machine_summary_bad_decision(self):
        ms = MachineSummary(
            experiment_id="x", n_strategies=5,
            best_strategy_name="A", best_strategy_decision="MAYBE",
            best_net_sharpe=0.5, validation_complete=True, governance_complete=True,
        )
        d = ms.to_dict()
        errors = validate_machine_summary(d)
        assert any("INVALID" in e for e in errors)

    def test_downstream_agent_can_read_action_signals(self):
        """Optimizer and Auto-Improve read these fields."""
        ms = MachineSummary(
            optimizer_should_tune=True,
            strategies_to_disable=["DEAD_STRATEGY"],
            strategies_to_observe=["EXPERIMENTAL_ONE"],
        )
        d = ms.to_dict()
        assert d["optimizer_should_tune"] is True
        assert "DEAD_STRATEGY" in d["strategies_to_disable"]
        assert "EXPERIMENTAL_ONE" in d["strategies_to_observe"]


# =====================================================================
# 4. JSON REPORT SCHEMA — structure validation
# =====================================================================

class TestReportSchemaValidation:
    """Full report must pass schema validation."""

    def _minimal_valid_report(self) -> dict:
        return {
            "governance": {
                "experiment_id": "abc123",
                "data_fingerprint": "sha256:...",
                "settings_fingerprint": "sha256:...",
                "methodology_version": "2.0",
                "run_mode": "daily",
                "validation_status": "COMPLETE",
                "promotion_readiness": "CONDITIONAL",
                "fail_reasons": [],
                "timestamp": "2026-03-25T10:00:00Z",
            },
            "machine_summary": MachineSummary(
                experiment_id="abc123",
                n_strategies=2,
                best_strategy_name="A",
                best_strategy_decision="APPROVED",
                best_net_sharpe=0.5,
                validation_complete=True,
                governance_complete=True,
            ).to_dict(),
            "methodology_scorecards": [
                {
                    "name": "A",
                    "promotion_decision": "APPROVED",
                    "classification": "CORE",
                },
            ],
            "approval_matrix": {
                "A": {"decision": "APPROVED"},
            },
        }

    def test_valid_report_passes(self):
        report = self._minimal_valid_report()
        errors = validate_report(report)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_governance_fails(self):
        report = self._minimal_valid_report()
        del report["governance"]
        errors = validate_report(report)
        assert any("governance" in e for e in errors)

    def test_missing_machine_summary_fails(self):
        report = self._minimal_valid_report()
        del report["machine_summary"]
        errors = validate_report(report)
        assert any("machine_summary" in e for e in errors)

    def test_invalid_scorecard_decision_fails(self):
        report = self._minimal_valid_report()
        report["methodology_scorecards"][0]["promotion_decision"] = "MAYBE"
        errors = validate_report(report)
        assert any("promotion_decision" in e for e in errors)

    def test_invalid_classification_fails(self):
        report = self._minimal_valid_report()
        report["methodology_scorecards"][0]["classification"] = "UNKNOWN"
        errors = validate_report(report)
        assert any("classification" in e for e in errors)

    def test_invalid_approval_matrix_decision_fails(self):
        report = self._minimal_valid_report()
        report["approval_matrix"]["A"]["decision"] = "YES"
        errors = validate_report(report)
        assert any("approval_matrix" in e for e in errors)

    def test_governance_missing_keys_fails(self):
        report = self._minimal_valid_report()
        del report["governance"]["experiment_id"]
        errors = validate_report(report)
        assert any("governance missing" in e for e in errors)


# =====================================================================
# 5. EXPERIMENT LINEAGE — versioning
# =====================================================================

class TestExperimentLineage:
    """Experiment lineage must produce unique, traceable IDs."""

    def test_governance_record_has_experiment_id(self):
        from agents.methodology.agent_methodology import GovernanceEngine
        from config.settings import get_settings
        ge = GovernanceEngine(get_settings())
        record = ge.create_governance_record(run_mode="daily")
        assert "experiment_id" in record
        assert len(record["experiment_id"]) >= 8
        assert record["run_mode"] == "daily"

    def test_two_runs_produce_different_ids(self):
        from agents.methodology.agent_methodology import GovernanceEngine
        from config.settings import get_settings
        ge1 = GovernanceEngine(get_settings())
        ge2 = GovernanceEngine(get_settings())
        assert ge1.experiment_id != ge2.experiment_id

    def test_data_fingerprint_is_deterministic(self):
        """Same data → same fingerprint."""
        from agents.methodology.agent_methodology import GovernanceEngine
        from config.settings import get_settings
        s = get_settings()
        ge1 = GovernanceEngine(s)
        ge2 = GovernanceEngine(s)
        assert ge1.data_fingerprint == ge2.data_fingerprint

    def test_settings_fingerprint_is_deterministic(self):
        """Same settings → same fingerprint."""
        from agents.methodology.agent_methodology import GovernanceEngine
        from config.settings import get_settings
        s = get_settings()
        ge1 = GovernanceEngine(s)
        ge2 = GovernanceEngine(s)
        assert ge1.settings_fingerprint == ge2.settings_fingerprint

    def test_portfolio_stores_experiment_lineage(self):
        """MethodologyPortfolio tracks experiment lineage."""
        from agents.methodology.methodology_portfolio import MethodologyPortfolio
        mp = MethodologyPortfolio()
        lineage = mp.get_experiment_lineage(n=5)
        assert isinstance(lineage, list)


# =====================================================================
# 6. PROMOTION CRITERIA — immutability
# =====================================================================

class TestPromotionCriteriaImmutable:
    """PromotionCriteria is frozen — cannot be changed at runtime."""

    def test_default_values(self):
        c = DEFAULT_PROMOTION_CRITERIA
        assert c.min_net_sharpe == 0.3
        assert c.max_drawdown_pct == 0.15
        assert c.min_trades == 100
        assert c.min_positive_regimes == 2
        assert c.min_deflated_sharpe == 0.0
        assert c.max_cost_drag_pct == 30.0

    def test_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_PROMOTION_CRITERIA.min_net_sharpe = 0.0
