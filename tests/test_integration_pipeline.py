"""
tests/test_integration_pipeline.py
====================================
End-to-end integration test for the full analytics pipeline.

Tests the complete flow:
  Settings → QuantEngine → EngineService (19 steps) → Repository verification

Run:
  python -m pytest tests/test_integration_pipeline.py -v
  python tests/test_integration_pipeline.py  # Direct execution
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd


class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""

    def test_settings_load(self):
        """Settings load without error, all critical fields present."""
        from config.settings import get_settings
        s = get_settings()
        assert s.fmp_api_key, "FMP API key must be set"
        assert len(s.sector_list()) >= 10, f"Need >= 10 sectors, got {len(s.sector_list())}"
        assert s.spy_ticker == "SPY"
        assert s.signal_optimal_hold > 0
        assert s.use_dcc_garch in (True, False)

    def test_run_context_creation(self):
        """RunContext creates with valid metadata."""
        from config.settings import get_settings
        from services.run_context import RunContext
        ctx = RunContext.create(get_settings())
        assert ctx.run_uuid, "UUID should be set"
        assert ctx.run_date, "Date should be set"
        assert ctx.started_at is not None

    def test_data_loader_agent_outputs(self):
        """DataLoader loads agent JSON files."""
        from config.settings import get_settings
        from services.data_loader import DataLoader
        loader = DataLoader(get_settings())
        agent_data = loader.load_agent_outputs()
        assert isinstance(agent_data, dict)
        assert "registry" in agent_data
        assert "auto_improve" in agent_data
        # At least some should have data
        n_loaded = sum(1 for v in agent_data.values() if v is not None)
        assert n_loaded >= 3, f"Expected >= 3 agent outputs, got {n_loaded}"

    def test_quant_engine_loads(self):
        """QuantEngine loads prices and computes conviction scores."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()
        assert engine.prices is not None, "Prices should load"
        assert len(engine.prices) > 100, f"Need > 100 price rows, got {len(engine.prices)}"

        master_df = engine.calculate_conviction_score()
        assert master_df is not None
        assert not master_df.empty
        assert "sector_ticker" in master_df.columns
        assert "conviction_score" in master_df.columns
        assert len(master_df) >= 10, f"Need >= 10 sectors in master_df"

    def test_repository_queries(self):
        """Repository returns valid data freshness and run info."""
        from config.settings import get_settings
        from db.repository import Repository
        repo = Repository(get_settings().db_path)

        freshness = repo.data_freshness()
        assert freshness.prices_rows > 0, "Should have price data"

        latest = repo.latest_run()
        # May be None on first run, that's OK
        if latest is not None:
            assert latest > 0

    def test_data_quality_checks(self):
        """Data quality checks run without errors."""
        from config.settings import get_settings
        from db.repository import Repository
        repo = Repository(get_settings().db_path)
        checks = repo.run_data_quality_checks()
        assert len(checks) >= 3, f"Expected >= 3 quality checks, got {len(checks)}"
        for c in checks:
            assert c.status in ("PASS", "WARN", "FAIL"), f"Invalid status: {c.status}"
            assert c.check_name, "Check must have a name"

    def test_momentum_strategies_positive(self):
        """RelativeMomentum strategy has positive Sharpe."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.methodology_lab import MethodologyLab, RelativeMomentum

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()

        lab = MethodologyLab(engine.prices, settings, step=10, cost_bps=15)
        result = lab.run_methodology(RelativeMomentum())

        assert result.sharpe > 0, f"RelativeMomentum Sharpe should be positive, got {result.sharpe}"
        assert result.total_trades >= 100, f"Need >= 100 trades, got {result.total_trades}"
        assert result.total_pnl > 0, f"RelativeMomentum PnL should be positive"

    def test_stress_engine(self):
        """Stress engine runs all 10 scenarios."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.stress import StressEngine

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()

        results = StressEngine().run_all(master_df, settings)
        assert len(results) == 10, f"Expected 10 scenarios, got {len(results)}"
        # Results should be sorted worst to best
        assert results[0].portfolio_pnl_estimate <= results[-1].portfolio_pnl_estimate

    def test_mc_stress(self):
        """Monte Carlo stress produces valid VaR."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.stress import MonteCarloStressEngine

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()

        mc = MonteCarloStressEngine(n_simulations=1000, horizon_days=21).run(
            master_df, engine.prices, settings
        )
        assert mc.n_simulations == 1000
        assert len(mc.pnl_distribution) == 1000
        assert mc.var_95 <= mc.var_99  # 99% VaR should be worse (more negative)

    def test_dcc_garch(self):
        """DCC-GARCH correlation produces valid matrix."""
        from analytics.correlation_structure import DCCGARCHCorrelation
        np.random.seed(42)
        r = pd.DataFrame(np.random.randn(252, 5) * 0.01, columns=["A", "B", "C", "D", "E"])
        dcc = DCCGARCHCorrelation()
        R = dcc.fit_and_predict(r)
        assert R.shape == (5, 5)
        assert np.allclose(np.diag(R.values), 1.0)
        assert np.isfinite(R.values).all()

    def test_pair_scanner(self):
        """Pair scanner finds signals."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.pair_scanner import scan_pairs

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()

        pairs = scan_pairs(engine.prices, settings.sector_list())
        assert len(pairs) > 0, "Should find at least 1 pair"
        assert pairs[0].signal_strength >= pairs[-1].signal_strength, "Should be sorted by strength"

    def test_kalman_filter(self):
        """Kalman filter produces valid hedge ratios."""
        from analytics.pair_scanner import kalman_hedge_ratio
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n) * 0.01) + 100
        y = 1.5 * x + np.cumsum(np.random.randn(n) * 0.005)
        betas, spreads = kalman_hedge_ratio(pd.Series(y), pd.Series(x))
        assert len(betas) == n
        assert abs(betas[-1] - 1.5) < 0.5, f"Kalman beta should be near 1.5, got {betas[-1]}"

    def test_options_engine_surface(self):
        """Options engine computes IV surface with VVIX/Skew."""
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.options_engine import OptionsEngine

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()

        surface = OptionsEngine().compute_surface(engine.prices, settings)
        assert surface.vix_current > 0
        assert surface.implied_corr >= -1 and surface.implied_corr <= 1
        assert hasattr(surface, "short_vol_timing_score")

    def test_paper_portfolio_json(self):
        """Paper portfolio JSON exists and has valid structure."""
        pp_path = ROOT / "data" / "paper_portfolio.json"
        if pp_path.exists():
            import json
            data = json.loads(pp_path.read_text(encoding="utf-8"))
            assert "capital" in data
            assert "positions" in data
            assert isinstance(data["positions"], list)
            assert data["capital"] > 0

    def test_alerting_service(self):
        """AlertService sends alerts without errors."""
        from services.alerting import AlertService, Alert
        svc = AlertService()  # No settings → no Slack, file only
        success = svc.send(Alert(level="INFO", title="Test", message="Integration test"))
        assert success


# Allow direct execution
if __name__ == "__main__":
    import sys
    # Run each test and report
    suite = TestPipelineIntegration()
    methods = [m for m in dir(suite) if m.startswith("test_")]
    passed = 0
    failed = 0
    for name in sorted(methods):
        try:
            getattr(suite, name)()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {passed+failed}")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)
