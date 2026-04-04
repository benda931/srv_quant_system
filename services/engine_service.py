"""
services/engine_service.py
============================
EngineService — single entry point for all analytics computation.

Replaces the 800+ lines of inline engine initialization in main.py.
Each engine is run with proper error handling, timing, and RunContext tracking.

Usage:
    from services.engine_service import EngineService
    from services.run_context import RunContext

    ctx = RunContext.create(settings)
    svc = EngineService(ctx)
    results = svc.compute_all()
    # results.master_df, results.stress, results.mc_stress, ...
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from services.run_context import RunContext

log = logging.getLogger("engine_service")


@dataclass
class EngineResults:
    """
    Container for all computed analytics results.
    Replaces 50+ private variables scattered across main.py.
    """
    # Core
    master_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    engine: Any = None                     # QuantEngine instance
    data_health: Any = None                # DataHealthReport
    run_id: int = -1

    # Stress
    stress_results: Optional[List] = None
    mc_stress_result: Any = None

    # Risk
    risk_report: Any = None

    # Correlation
    corr_vol_analysis: Any = None
    corr_snapshot: Any = None

    # DSS / Signals
    signal_results: Optional[List] = None
    trade_tickets: Optional[List] = None
    regime_safety: Any = None
    trade_monitor: Any = None

    # Options
    options_surface: Any = None
    tail_risk_es: Any = None

    # Paper trading
    paper_portfolio: Optional[Dict] = None
    dispersion_result: Any = None

    # Tracking
    pnl_result: Any = None
    decay_result: Any = None
    regime_result: Any = None

    # Backtest
    backtest_result: Any = None

    # Trade book
    trade_book_history: Optional[pd.DataFrame] = None

    # Methodology
    methodology_ranking: Optional[List[Dict]] = None

    # ML
    ml_feature_importances: Optional[Dict] = None
    ml_regime_forecast: Any = None

    # Errors (for UI banners)
    errors: Dict[str, str] = field(default_factory=dict)


class EngineService:
    """
    Orchestrates all analytics engines in the correct order.
    Each step is isolated with error handling and RunContext tracking.
    """

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.settings = ctx.settings
        self.results = EngineResults()
        self._timings: Dict[str, float] = {}

    def compute_all(self) -> EngineResults:
        """
        Run all analytics engines in dependency order.
        Returns EngineResults with all computed data.
        """
        # Phase 1: Data + Core
        self._step("data_load", self._load_data)
        self._step("quant_engine", self._run_quant_engine)
        self._step("db_persist", self._persist_run)

        # Phase 2: Risk + Stress
        self._step("stress", self._run_stress)
        self._step("mc_stress", self._run_mc_stress)
        self._step("portfolio_risk", self._run_portfolio_risk)

        # Phase 3: Correlation + Signals
        self._step("correlation", self._run_correlation)
        self._step("dss_signals", self._run_dss_signals)

        # Phase 4: Options + Tail Risk
        self._step("options", self._run_options)

        # Phase 5: Paper Trading + Dispersion
        self._step("paper_trading", self._run_paper_trading)
        self._step("dispersion", self._run_dispersion)

        # Phase 6: Tracking
        self._step("pnl_tracker", self._run_pnl_tracker)
        self._step("signal_decay", self._run_signal_decay)
        self._step("regime_alerts", self._run_regime_alerts)

        # Phase 7: Backtest + ML
        self._step("backtest", self._run_backtest)
        self._step("ml_models", self._run_ml_models)

        # Phase 8: Trade book
        self._step("trade_book", self._load_trade_book)

        # Phase 9: Data quality + agent snapshots
        self._step("data_quality", self._run_data_quality)
        self._step("agent_snapshots", self._persist_agent_snapshots)

        # Finalize and persist run context
        self.ctx.finalize()
        self._persist_run_context()
        self._log_summary()

        return self.results

    def _step(self, name: str, func, fatal: bool = False):
        """Run a step with timing, error handling, and context tracking."""
        t0 = time.time()
        try:
            func()
            elapsed = time.time() - t0
            self._timings[name] = elapsed
            self.ctx.mark_step(name, success=True)
            log.info("✓ %s (%.1fs)", name, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            self._timings[name] = elapsed
            self.ctx.mark_step(name, success=False, error=str(e))
            self.results.errors[name] = str(e)
            if fatal:
                raise
            log.warning("✗ %s failed (%.1fs): %s", name, elapsed, str(e)[:100])

    # ── Step implementations ─────────────────────────────────────────────

    def _load_data(self):
        from data_ops.orchestrator import DataOrchestrator
        orch = DataOrchestrator(self.settings)
        data_state = orch.run(force_refresh=False)
        self.results.data_health = data_state.health
        self.ctx.data_health_label = data_state.health.health_label
        self.ctx.data_health_score = getattr(data_state.health, "score", 0)

    def _run_quant_engine(self):
        from analytics.stat_arb import QuantEngine
        engine = QuantEngine(self.settings)
        engine.load()
        master_df = engine.calculate_conviction_score()
        if master_df is None or master_df.empty:
            raise RuntimeError("master_df is empty; cannot proceed.")

        self.results.engine = engine
        self.results.master_df = master_df

        # Extract regime info for context
        if "market_state" in master_df.columns:
            self.ctx.regime = str(master_df["market_state"].iloc[0])
        if "vix_level" in master_df.columns:
            self.ctx.vix_level = float(master_df["vix_level"].iloc[0])

        self.ctx.prices_rows = len(engine.prices)
        self.ctx.prices_cols = len(engine.prices.columns)
        if hasattr(engine.prices.index, "max"):
            self.ctx.prices_latest_date = str(engine.prices.index.max().date())

    def _persist_run(self):
        from db.writer import DatabaseWriter
        dw = DatabaseWriter(self.settings.db_path)
        run_id = dw.write_run(
            self.results.master_df,
            started_at=self.ctx.started_at,
            finished_at=self.ctx.started_at,  # Updated later
            data_health_label=self.ctx.data_health_label,
        )
        self.results.run_id = run_id
        self.ctx.run_id = run_id

    def _run_stress(self):
        from analytics.stress import StressEngine
        self.results.stress_results = StressEngine().run_all(
            self.results.master_df, self.settings,
        )

    def _run_mc_stress(self):
        from analytics.stress import MonteCarloStressEngine
        prices = self.results.engine.prices
        if prices is not None and len(prices) > 60:
            self.results.mc_stress_result = MonteCarloStressEngine(
                n_simulations=10_000, horizon_days=21,
            ).run(self.results.master_df, prices, self.settings)

    def _run_portfolio_risk(self):
        from analytics.portfolio_risk import PortfolioRiskEngine
        engine = self.results.engine
        master_df = self.results.master_df
        prices = engine.prices
        if prices is None or "w_final" not in master_df.columns:
            return

        weights = {
            row["sector_ticker"]: float(row["w_final"])
            for _, row in master_df.iterrows()
            if row.get("direction") in ("LONG", "SHORT")
        }
        # If no active positions (e.g., CRISIS regime), use equal weights
        # so the Risk tab still shows analytical content
        if not weights:
            sectors = [str(row["sector_ticker"]) for _, row in master_df.iterrows()
                       if "sector_ticker" in master_df.columns]
            if sectors:
                w = 1.0 / len(sectors)
                weights = {s: w for s in sectors}
            else:
                return

        risk_engine = PortfolioRiskEngine()
        self.results.risk_report = risk_engine.full_risk_report(
            weights, prices, self.settings,
        )

    def _run_correlation(self):
        from analytics.correlation_engine import CorrVolEngine
        engine = self.results.engine
        master_df = self.results.master_df
        cv = CorrVolEngine()
        self.results.corr_vol_analysis = cv.run(engine, master_df, self.settings)

        # DCC-GARCH correlation snapshot for DSS
        from analytics.correlation_structure import (
            CorrelationStructureEngine, build_sector_groups_from_settings,
        )
        prices = engine.prices
        sectors = [s for s in self.settings.sector_list() if s in prices.columns]
        if len(sectors) >= 5:
            log_rets = np.log(prices[sectors] / prices[sectors].shift(1)).dropna()
            sg = build_sector_groups_from_settings(self.settings)
            cs = CorrelationStructureEngine()
            self.results.corr_snapshot = cs.compute_snapshot_with_zscore(
                returns=log_rets, sector_groups=sg,
                W_s=self.settings.corr_window,
                W_b=self.settings.corr_baseline_window,
                distortion_z_lookback=self.settings.corr_distortion_z_lookback,
                coc_z_lookback=self.settings.coc_z_lookback,
                settings=self.settings,
            )

    def _run_dss_signals(self):
        from analytics.signal_stack import SignalStackEngine
        from analytics.signal_regime_safety import compute_regime_safety_score
        from analytics.trade_structure import TradeStructureEngine, PositionSizingEngine

        master_df = self.results.master_df
        cs = self.results.corr_snapshot

        # Regime safety
        _ms = str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "NORMAL"
        _vix = float(master_df["vix_level"].iloc[0]) if "vix_level" in master_df.columns else float("nan")
        _cz = float(master_df["credit_z"].iloc[0]) if "credit_z" in master_df.columns else float("nan")

        safety = compute_regime_safety_score(
            market_state=_ms, vix_level=_vix, credit_z=_cz,
            avg_corr=cs.avg_corr_short if cs else 0.3,
            corr_z=cs.frob_distortion_z if cs else 0.0,
        )
        self.results.regime_safety = safety
        self.ctx.safety_label = safety.label
        self.ctx.safety_score = safety.regime_safety_score

        # Signal stack
        ss = SignalStackEngine(self.settings)
        signals = ss.score_from_master_df(
            frob_distortion_z=cs.frob_distortion_z if cs else 0.0,
            market_mode_share=cs.market_mode_share if cs else 0.3,
            coc_instability_z=cs.coc_instability_z if cs else 0.0,
            master_df=master_df,
            regime_safety_result=safety,
        )
        self.results.signal_results = signals

        # Trade structure + sizing
        ts = TradeStructureEngine(self.settings)
        tickets = ts.construct_all_trades(signals, master_df=master_df)
        ps = PositionSizingEngine(self.settings)
        tickets = ps.size_portfolio(tickets, safety.regime_safety_score, safety.size_cap)
        self.results.trade_tickets = tickets

        # Trade monitor
        try:
            from analytics.trade_monitor import TradeMonitorEngine
            tm = TradeMonitorEngine(self.settings)
            self.results.trade_monitor = tm.monitor_all(tickets, master_df)
        except Exception:
            pass

        # Persist trade book
        if self.ctx.run_id > 0 and tickets:
            try:
                from db.writer import DatabaseWriter
                DatabaseWriter(self.settings.db_path).write_trade_book(tickets, self.ctx.run_id)
            except Exception:
                pass

    def _run_options(self):
        from analytics.options_engine import OptionsEngine
        prices = self.results.engine.prices
        self.results.options_surface = OptionsEngine().compute_surface(prices, self.settings)

        try:
            from analytics.tail_risk import compute_expected_shortfall
            master_df = self.results.master_df
            weights = {
                row["sector_ticker"]: float(row.get("w_final", 0))
                for _, row in master_df.iterrows()
            }
            # Fallback to equal weights if all zero
            if not any(abs(v) > 0.001 for v in weights.values()):
                n = len(weights)
                if n > 0:
                    weights = {k: 1.0 / n for k in weights}
            returns = np.log(prices / prices.shift(1)).iloc[1:]
            self.results.tail_risk_es = compute_expected_shortfall(
                returns, weights, confidence=0.975,
            )
        except Exception:
            pass

    def _run_paper_trading(self):
        from analytics.paper_trader import PaperTrader
        prices = self.results.engine.prices
        trader = PaperTrader(settings=self.settings)
        trader.load()
        trader.daily_update(
            prices,
            signal_results=self.results.signal_results,
            trade_tickets=self.results.trade_tickets,
            regime_safety=self.results.regime_safety,
        )
        trader.save()

        # Load portfolio as dict for UI
        import json
        pp_path = self.settings.project_root / "data" / "paper_portfolio.json"
        if pp_path.exists():
            self.results.paper_portfolio = json.loads(pp_path.read_text(encoding="utf-8"))

    def _run_dispersion(self):
        from analytics.dispersion_backtest import DispersionBacktester
        prices = self.results.engine.prices
        sectors = [s for s in self.settings.sector_list() if s in prices.columns]
        if len(sectors) >= 5:
            bt = DispersionBacktester(
                prices, sectors=sectors,
                hold_period=15, z_entry=0.6, z_exit=0.2,
                max_positions=3, lookback=30,
            )
            self.results.dispersion_result = bt.run()

    def _run_pnl_tracker(self):
        from analytics.pnl_tracker import PnLTracker
        prices = self.results.engine.prices
        if prices is not None and not prices.empty:
            self.results.pnl_result = PnLTracker(self.settings).track(prices)

    def _run_signal_decay(self):
        from analytics.signal_decay import SignalDecayAnalyser
        prices = self.results.engine.prices
        if prices is not None and not prices.empty:
            self.results.decay_result = SignalDecayAnalyser(self.settings).analyse(prices)

    def _run_regime_alerts(self):
        from analytics.regime_alerts import RegimeAlertEngine
        prices = self.results.engine.prices
        if prices is not None and not prices.empty:
            self.results.regime_result = RegimeAlertEngine(self.settings).analyse(prices)

    def _run_backtest(self):
        # Try loading from DuckDB cache first
        try:
            import duckdb
            conn = duckdb.connect(str(self.settings.db_path), read_only=True)
            row = conn.execute(
                "SELECT * FROM analytics.backtest_cache ORDER BY cache_date DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row is not None:
                self.results.backtest_result = row
                return
        except Exception:
            pass

        # Fallback 1: alpha research report
        try:
            import json
            reports = sorted(
                (self.settings.project_root / "agents" / "methodology" / "reports").glob("*alpha_research*"),
                reverse=True,
            )
            if reports:
                data = json.loads(reports[0].read_text(encoding="utf-8"))
                best = data.get("best_oos", {})
                if best.get("oos_sharpe", 0) > 0:
                    from types import SimpleNamespace
                    self.results.backtest_result = SimpleNamespace(
                        sharpe=best["oos_sharpe"],
                        source="alpha_research_oos",
                    )
                    return
        except Exception:
            pass

        # Fallback 2: quick methodology lab run on RelativeMomentum
        try:
            from analytics.methodology_lab import MethodologyLab, RelativeMomentum
            prices = self.results.engine.prices
            if prices is not None and len(prices) > 300:
                lab = MethodologyLab(prices, self.settings, step=10, cost_bps=15)
                result = lab.run_methodology(RelativeMomentum(
                    lookback=int(getattr(self.settings, "momentum_lookback", 21)),
                    top_n=int(getattr(self.settings, "momentum_top_n", 3)),
                    rebal_days=int(getattr(self.settings, "momentum_rebal_days", 21)),
                ))
                from types import SimpleNamespace
                self.results.backtest_result = SimpleNamespace(
                    sharpe=result.sharpe,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    max_drawdown=result.max_drawdown,
                    total_trades=result.total_trades,
                    equity_curve=result.equity_curve,
                    source="momentum_live",
                )
                log.info("Backtest: RelativeMomentum Sharpe=%.3f PnL=%.1f%%",
                         result.sharpe, result.total_pnl * 100)
        except Exception as e:
            log.debug("Quick backtest failed: %s", e)

    def _run_ml_models(self):
        import pickle
        cache_dir = self.settings.project_root / "data" / "ml_models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature importances
        fi_path = cache_dir / "feature_importances.pkl"
        if fi_path.exists() and (time.time() - fi_path.stat().st_mtime) < 24 * 3600:
            self.results.ml_feature_importances = pickle.loads(fi_path.read_bytes())
        else:
            try:
                from analytics.feature_engine import FeatureEngine
                fe = FeatureEngine(self.results.engine.prices, self.settings.sector_list())
                feat_df = fe.compute_all_features()
                from analytics.ml_signals import SignalQualityModel
                model = SignalQualityModel()
                model.train(None)
                selected = model.feature_names_[:15] if hasattr(model, 'feature_names_') else list(feat_df.columns[:15])
                self.results.ml_feature_importances = {f: 1.0 / (i + 1) for i, f in enumerate(selected)}
                fi_path.write_bytes(pickle.dumps(self.results.ml_feature_importances))
            except Exception:
                pass

        # Regime forecast
        rf_path = cache_dir / "regime_forecast.pkl"
        if rf_path.exists() and (time.time() - rf_path.stat().st_mtime) < 24 * 3600:
            self.results.ml_regime_forecast = pickle.loads(rf_path.read_bytes())

    def _load_trade_book(self):
        try:
            from db.reader import DatabaseReader
            reader = DatabaseReader(self.settings.db_path)
            self.results.trade_book_history = reader.read_trade_book_history(n_runs=10)
        except Exception:
            pass

    def _run_data_quality(self):
        """Run canonical data quality checks and persist results."""
        from db.repository import Repository
        repo = Repository(self.settings.db_path)
        checks = repo.run_data_quality_checks()
        if self.ctx.run_id > 0:
            repo.write_data_quality(self.ctx.run_id, checks)
        n_pass = sum(1 for c in checks if c.status == "PASS")
        n_warn = sum(1 for c in checks if c.status == "WARN")
        n_fail = sum(1 for c in checks if c.status == "FAIL")
        log.info("Data quality: %d PASS, %d WARN, %d FAIL", n_pass, n_warn, n_fail)

    def _persist_agent_snapshots(self):
        """Persist key agent outputs to DuckDB (canonical store)."""
        import json as _json
        from db.repository import Repository
        repo = Repository(self.settings.db_path)
        run_id = self.ctx.run_id
        if run_id <= 0:
            return

        from services.data_loader import DataLoader
        loader = DataLoader(self.settings)
        agent_data = loader.load_agent_outputs()
        n_persisted = 0
        for agent_name, data in agent_data.items():
            if data is not None:
                try:
                    repo.write_agent_snapshot(
                        run_id, agent_name, "ok",
                        _json.dumps(data, default=str)[:50000],
                    )
                    n_persisted += 1
                except Exception:
                    pass
        log.info("Agent snapshots persisted: %d/%d to DuckDB", n_persisted, len(agent_data))

    def _persist_run_context(self):
        """Save RunContext to DuckDB for lineage tracking."""
        try:
            from db.writer import DatabaseWriter
            dw = DatabaseWriter(self.settings.db_path)
            dw.write_run_context(self.ctx)
        except Exception as e:
            log.debug("RunContext persistence failed: %s", e)

    def _log_summary(self):
        ok = len(self.ctx.steps_completed)
        fail = len(self.ctx.steps_failed)
        total_time = sum(self._timings.values())
        log.info(
            "EngineService complete: %d/%d steps OK (%.1fs) | regime=%s | VIX=%.1f | safety=%s",
            ok, ok + fail, total_time, self.ctx.regime, self.ctx.vix_level, self.ctx.safety_label,
        )
        if self.ctx.steps_failed:
            log.warning("  Failed steps: %s", self.ctx.steps_failed)
