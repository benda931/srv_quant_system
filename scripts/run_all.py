"""
scripts/run_all.py
------------------
Master automated pipeline — wires all engines in sequence.

Execution order (each step non-fatal unless marked REQUIRED):
  [REQUIRED] Step 1: Smart data refresh
             - Checks DB freshness (DatabaseReader.is_snapshot_fresh)
             - Only re-fetches if data is stale (> cache_max_age_hours)
             - Incremental: short window if prices are < 3 days old
  [REQUIRED] Step 2: QuantEngine
             - load() + calculate_conviction_score()
             - Writes run to analytics.runs + analytics.sector_signals
  [OPTIONAL] Step 3: Walk-forward backtest
             - Only runs if --backtest flag OR last backtest > 7 days ago (DB check)
             - Saves result to analytics.backtest_cache
  [OPTIONAL] Step 4: ML signal quality
             - Trains SignalQualityModel on backtest walk data
             - Applies quality multiplier to conviction scores
  [OPTIONAL] Step 5: Portfolio optimization
             - Runs risk-parity optimizer
             - Adds opt_weight column to master_df
  [OPTIONAL] Step 6: Stress + Portfolio Risk
  [OPTIONAL] Step 7: Daily brief
  [ALWAYS]   Step 8: DB audit write + data pruning
             - write_run() to DuckDB
             - Prune old snapshots (keep 90d fundamentals, 30d holdings)

Smart data refresh logic:
  - If DB is fresh (< max_age_hours): skip all FMP fetching entirely
  - If DB is stale: run full DataOrchestrator.run()
  - This avoids redundant API calls on dashboard restarts

Usage:
  python scripts/run_all.py                  # Normal daily run
  python scripts/run_all.py --force-refresh  # Force FMP re-fetch
  python scripts/run_all.py --backtest       # Include full backtest (slower)
  python scripts/run_all.py --no-ml          # Skip ML layer
  python scripts/run_all.py --alpha-research # Force OOS alpha research (weekly auto)
"""
from __future__ import annotations

# DuckDB must be imported before pandas/pyarrow — see main.py for explanation.
import duckdb as _duckdb_init  # noqa: F401

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ── Bootstrap sys.path so imports work from any cwd ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_runs.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=handlers)

logger = logging.getLogger("run_all")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: check if backtest cache is fresh
# ──────────────────────────────────────────────────────────────────────────────

def _is_backtest_fresh(db_conn, max_age_days: int = 7) -> bool:
    """
    Return True if analytics.backtest_cache has a record within the last
    max_age_days days.  Returns False on any DB error (first run safe).
    """
    try:
        row = db_conn.execute(
            "SELECT max(cache_date) FROM analytics.backtest_cache"
        ).fetchone()
        if not row or row[0] is None:
            return False
        last = row[0]
        if not isinstance(last, date):
            last = date.fromisoformat(str(last))
        age = (date.today() - last).days
        return age < max_age_days
    except Exception as exc:
        logger.debug("_is_backtest_fresh: DB check failed (%s) — treating as stale", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Helper: prune old snapshot data
# ──────────────────────────────────────────────────────────────────────────────

def _prune_old_data(
    db_conn,
    snapshot_retain_days: int = 90,
    holdings_retain_days: int = 30,
) -> None:
    """
    Delete old snapshot rows from fundamentals and holdings.
    Prices are NEVER pruned (needed for full 10-year history).
    Delegates to DatabaseWriter.prune_old_snapshots for DRY execution.
    """
    from datetime import timedelta
    fund_cutoff = date.today() - timedelta(days=snapshot_retain_days)
    hold_cutoff = date.today() - timedelta(days=holdings_retain_days)

    for table, cutoff in [
        ("fundamentals.quotes",         fund_cutoff),
        ("fundamentals.ratios",         fund_cutoff),
        ("holdings.etf_holdings",       hold_cutoff),
        ("holdings.spy_sector_weights", hold_cutoff),
    ]:
        try:
            db_conn.execute(f"DELETE FROM {table} WHERE snapshot_date < ?", [cutoff])
            logger.info("prune %s: removed rows before %s", table, cutoff)
        except Exception as exc:
            logger.warning("prune %s: non-fatal error (%s)", table, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Slack dispatch
# ──────────────────────────────────────────────────────────────────────────────

def _dispatch_slack(webhook_url: str, text: str) -> bool:
    """
    POST a plain-text message to a Slack Incoming Webhook.
    Returns True on HTTP 200, False on any error (non-fatal).
    """
    if not webhook_url:
        return False
    try:
        import urllib.request
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("Slack dispatch failed (non-fatal): %s", exc)
        return False


def _build_pipeline_slack_message(status: Dict[str, Any]) -> str:
    """Build a concise Slack summary message from the pipeline status dict."""
    run_id     = status.get("run_id", "?")
    ok_steps   = status.get("steps_ok", [])
    fail_steps = status.get("steps_failed", [])
    dur        = status.get("duration_s", 0)
    regime     = status.get("regime_safety_label") or status.get("current_regime") or status.get("regime", "Unknown")
    n_active   = status.get("trade_n_active", "?")
    dt_str     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    icon = ":white_check_mark:" if not fail_steps else ":warning:"
    lines = [
        f"{icon} *SRV DSS Pipeline* — {dt_str}",
        f"Run ID `{run_id}` | Duration `{dur:.0f}s`",
        f"Regime: `{regime}` | Active trades: `{n_active}`",
        f"Steps OK: `{', '.join(ok_steps) or 'none'}`",
    ]
    if fail_steps:
        lines.append(f":x: Failed: `{', '.join(fail_steps)}`")
    if status.get("dss_brief_path"):
        lines.append(f"DSS brief: `{status['dss_brief_path']}`")
    elif status.get("brief_path"):
        lines.append(f"Brief: `{status['brief_path']}`")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    force_refresh: bool = False,
    run_backtest: bool = False,
    run_ml: bool = True,
    run_optimizer: bool = True,
    run_alpha_research: bool = False,
) -> Dict[str, Any]:
    """
    Execute all pipeline steps in sequence.

    Parameters
    ----------
    force_refresh : bool
        Force FMP data re-fetch even if DB is fresh.
    run_backtest : bool
        Force walk-forward backtest (otherwise only runs if cache is stale).
    run_ml : bool
        Enable ML signal quality layer.
    run_optimizer : bool
        Enable portfolio optimisation layer.

    Returns
    -------
    dict
        Status summary with keys: run_id, steps_ok, steps_failed, duration_s,
        n_sectors, regime, ic_mean, opt_weights_computed.
    """
    settings = get_settings()
    _configure_logging(settings.log_dir)

    started_at = datetime.now(timezone.utc)
    status: Dict[str, Any] = {
        "run_id": -1,
        "steps_ok": [],
        "steps_failed": [],
        "duration_s": 0.0,
        "n_sectors": 0,
        "regime": "UNKNOWN",
        "ic_mean": None,
        "opt_weights_computed": False,
        "started_at": started_at.isoformat(),
    }

    logger.info("=" * 70)
    logger.info("SRV Quantamental DSS — Daily Pipeline run_all.py")
    logger.info("force_refresh=%s  run_backtest=%s  run_ml=%s", force_refresh, run_backtest, run_ml)
    logger.info("=" * 70)

    # ── Ensure DB + schema exist (first-run safe) ─────────────────────────────
    db_conn = None
    try:
        from db.connection import get_connection
        from db.schema import SchemaManager
        db_conn = get_connection(settings.db_path)
        SchemaManager(db_conn).apply_migrations()
        logger.info("Step 0: Schema OK (version %s)", SchemaManager(db_conn).current_version())
    except Exception as exc:
        logger.error("Step 0: Schema init failed — %s", exc)
        # Continue; DB writes later will fail gracefully

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 [REQUIRED]: Smart data refresh
    # ──────────────────────────────────────────────────────────────────────────
    data_state = None
    try:
        from db.reader import DatabaseReader
        from data_ops.orchestrator import DataOrchestrator

        # Check DB freshness
        is_fresh = False
        if not force_refresh and db_conn is not None:
            try:
                reader = DatabaseReader(settings.db_path)
                is_fresh = reader.is_snapshot_fresh(settings.cache_max_age_hours)
            except Exception:
                is_fresh = False

        if is_fresh:
            logger.info("Step 1: DB is fresh — skipping FMP fetch")
            # Still build a minimal DataState so QuantEngine can load from DB
            orchestrator = DataOrchestrator(settings)
            data_state = orchestrator.run(force_refresh=False)
        else:
            logger.info("Step 1: Data stale — running full DataOrchestrator.run()")
            orchestrator = DataOrchestrator(settings)
            data_state = orchestrator.run(force_refresh=force_refresh)

        # ── Backfill DuckDB from parquet if DB is empty ───────────────────────
        # build_snapshot() is skipped when parquet is fresh, so DB may never
        # get initial data.  If prices table is empty, seed it from artifacts.
        try:
            from db.reader import DatabaseReader as _DBR
            from db.writer import DatabaseWriter as _DBW
            import pandas as _pd
            _stats = _DBR(settings.db_path).table_stats()
            if _stats.get("prices", {}).get("rows", 0) == 0:
                logger.info("Step 1: DB empty — backfilling from parquet artifacts")
                _dw = _DBW(settings.db_path)
                _dw.write_prices(_pd.read_parquet(data_state.artifacts.prices_path))
                _dw.write_fundamentals(_pd.read_parquet(data_state.artifacts.fundamentals_path))
                _dw.write_weights(_pd.read_parquet(data_state.artifacts.weights_path))
                logger.info("Step 1: DB backfill complete")
        except Exception as _bfe:
            logger.warning("Step 1: DB backfill failed (non-fatal) — %s", _bfe)

        status["steps_ok"].append("data_refresh")
        logger.info("Step 1 OK — health=%s", getattr(data_state.health, "health_label", "?"))
    except Exception as exc:
        logger.error("Step 1 FAILED (data refresh) — %s", exc, exc_info=True)
        status["steps_failed"].append("data_refresh")
        # Cannot continue without data
        status["duration_s"] = (datetime.now(timezone.utc) - started_at).total_seconds()
        return status

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 [REQUIRED]: QuantEngine
    # ──────────────────────────────────────────────────────────────────────────
    master_df = None
    engine = None
    run_id = -1
    try:
        from analytics.stat_arb import QuantEngine
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()

        if master_df is None or master_df.empty:
            raise ValueError("QuantEngine returned empty master_df")

        status["n_sectors"] = len(master_df)
        if "market_state" in master_df.columns:
            status["regime"] = str(master_df["market_state"].iloc[0])

        # Write audit run immediately
        if db_conn is not None:
            try:
                from db.writer import DatabaseWriter
                dw = DatabaseWriter(settings.db_path)
                run_id = dw.write_run(
                    master_df,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    data_health_label=getattr(data_state.health, "health_label", "OK"),
                )
                status["run_id"] = run_id
                logger.info("Step 2: write_run → run_id=%d", run_id)
            except Exception as dbe:
                logger.warning("Step 2: write_run failed (non-fatal) — %s", dbe)

        status["steps_ok"].append("quant_engine")
        logger.info("Step 2 OK — %d sectors, regime=%s", len(master_df), status["regime"])
    except Exception as exc:
        logger.error("Step 2 FAILED (QuantEngine) — %s", exc, exc_info=True)
        status["steps_failed"].append("quant_engine")
        status["duration_s"] = (datetime.now(timezone.utc) - started_at).total_seconds()
        return status

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2b [OPTIONAL]: Correlation Structure Measurement Engine
    # ──────────────────────────────────────────────────────────────────────────
    corr_structure_snapshot = None
    try:
        from analytics.correlation_structure import (
            CorrelationStructureEngine,
            build_sector_groups_from_settings,
            measurement_summary,
        )
        import pandas as pd
        import numpy as np

        # Get prices for correlation computation
        _prices_for_corr = engine.prices
        if _prices_for_corr is not None and not _prices_for_corr.empty:
            sectors = settings.sector_list()
            avail = [s for s in sectors if s in _prices_for_corr.columns]
            if len(avail) >= 5:
                log_rets = np.log(_prices_for_corr[avail] / _prices_for_corr[avail].shift(1)).dropna()

                sector_groups = build_sector_groups_from_settings(settings)
                cs_engine = CorrelationStructureEngine()
                corr_structure_snapshot = cs_engine.compute_snapshot_with_zscore(
                    returns=log_rets,
                    sector_groups=sector_groups,
                    W_s=settings.corr_window,
                    W_b=settings.corr_baseline_window,
                    distortion_z_lookback=settings.corr_distortion_z_lookback,
                    coc_z_lookback=settings.coc_z_lookback,
                    settings=settings,
                )

                _cs_summary = measurement_summary(corr_structure_snapshot)
                logger.info(
                    "Step 2b OK — Corr structure: D=%.4f (z=%s), m=%.3f, CoC=%s, avg_ρ=%.3f",
                    corr_structure_snapshot.frob_distortion,
                    f"{corr_structure_snapshot.frob_distortion_z:.2f}" if math.isfinite(corr_structure_snapshot.frob_distortion_z) else "N/A",
                    corr_structure_snapshot.market_mode_share,
                    f"{corr_structure_snapshot.coc_instability:.4f}" if math.isfinite(corr_structure_snapshot.coc_instability) else "N/A",
                    corr_structure_snapshot.avg_corr_short if math.isfinite(corr_structure_snapshot.avg_corr_short) else 0,
                )
                status["steps_ok"].append("corr_structure")
                status["frob_distortion"] = corr_structure_snapshot.frob_distortion
                status["market_mode_share"] = corr_structure_snapshot.market_mode_share
                status["coc_instability"] = (
                    corr_structure_snapshot.coc_instability
                    if math.isfinite(corr_structure_snapshot.coc_instability) else None
                )
            else:
                logger.warning("Step 2b: Not enough sectors for correlation structure")
                status["steps_failed"].append("corr_structure")
        else:
            logger.warning("Step 2b: No prices available for correlation structure")
            status["steps_failed"].append("corr_structure")
    except Exception as exc:
        logger.warning("Step 2b: Correlation structure engine failed (non-fatal) — %s", exc, exc_info=True)
        status["steps_failed"].append("corr_structure")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2c: Signal Stack (all 4 layers: Distortion + Dislocation + MR + Safety)
    # ──────────────────────────────────────────────────────────────────────────
    signal_stack_results = None
    try:
        from analytics.signal_stack import SignalStackEngine, signal_stack_summary
        from analytics.signal_regime_safety import compute_regime_safety_score

        if corr_structure_snapshot is not None and master_df is not None:
            ss_engine = SignalStackEngine(settings)

            # Layer 4: Regime Safety from RegimeMetrics + corr snapshot
            _vix = float(master_df["vix_level"].iloc[0]) if "vix_level" in master_df.columns else float("nan")
            _cz = float(master_df["credit_z"].iloc[0]) if "credit_z" in master_df.columns else float("nan")
            _ms = str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "NORMAL"
            _ts = float(master_df["regime_transition_score"].iloc[0]) if "regime_transition_score" in master_df.columns else float("nan")
            _cp = float(master_df["crisis_probability"].iloc[0]) if "crisis_probability" in master_df.columns else float("nan")

            regime_safety = compute_regime_safety_score(
                market_state=_ms,
                vix_level=_vix,
                credit_z=_cz,
                avg_corr=corr_structure_snapshot.avg_corr_short,
                corr_z=corr_structure_snapshot.frob_distortion_z,
                transition_score=_ts,
                crisis_probability=_cp,
            )

            signal_stack_results = ss_engine.score_from_master_df(
                frob_distortion_z=corr_structure_snapshot.frob_distortion_z,
                market_mode_share=corr_structure_snapshot.market_mode_share,
                coc_instability_z=corr_structure_snapshot.coc_instability_z,
                master_df=master_df,
                regime_safety_result=regime_safety,
            )

            n_passing = sum(1 for r in signal_stack_results if r.passes_entry)
            top = signal_stack_results[0] if signal_stack_results else None
            logger.info(
                "Step 2c OK — Signal stack: %d candidates, %d passing, "
                "distortion=%.3f, safety=%.3f (%s), top=%s (conv=%.3f, dir=%s)",
                len(signal_stack_results), n_passing,
                top.distortion_score if top else 0,
                regime_safety.regime_safety_score, regime_safety.label,
                top.ticker if top else "N/A",
                top.conviction_score if top else 0,
                top.direction if top else "N/A",
            )
            if regime_safety.alerts:
                for alert in regime_safety.alerts:
                    logger.warning("  Regime alert: %s", alert)
            status["steps_ok"].append("signal_stack")
            status["signal_n_passing"] = n_passing
            status["regime_safety_label"] = regime_safety.label
        else:
            logger.info("Step 2c: Skipped — no corr_structure_snapshot or master_df")
            status["steps_ok"].append("signal_stack_skipped")
    except Exception as exc:
        logger.warning("Step 2c: Signal stack failed (non-fatal) — %s", exc, exc_info=True)
        status["steps_failed"].append("signal_stack")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2d: Trade Structure & Sizing
    # ──────────────────────────────────────────────────────────────────────────
    trade_tickets = None
    try:
        from analytics.trade_structure import TradeStructureEngine, PositionSizingEngine, trade_book_summary

        if signal_stack_results and master_df is not None:
            ts_engine = TradeStructureEngine(settings)
            trade_tickets = ts_engine.construct_all_trades(
                signal_stack_results,
                master_df=master_df,
            )

            # Portfolio-level sizing
            ps_engine = PositionSizingEngine(settings)
            _rs = regime_safety.regime_safety_score if "regime_safety" in dir() else 1.0
            _sc = regime_safety.size_cap if "regime_safety" in dir() else 1.0
            trade_tickets = ps_engine.size_portfolio(trade_tickets, _rs, _sc)

            n_active = sum(1 for t in trade_tickets if t.is_active)
            gross = sum(t.final_weight for t in trade_tickets if t.is_active)
            logger.info(
                "Step 2d OK — Trade book: %d tickets, %d active, gross=%.3f",
                len(trade_tickets), n_active, gross,
            )

            # ── Persist trade tickets to DuckDB ──────────────────────────────
            if db_conn is not None and run_id > 0:
                try:
                    from db.writer import DatabaseWriter
                    DatabaseWriter(settings.db_path).write_trade_book(trade_tickets, run_id)
                except Exception as _tbe:
                    logger.warning("Step 2d: write_trade_book failed (non-fatal) — %s", _tbe)

            status["steps_ok"].append("trade_structure")
            status["trade_n_active"] = n_active
        else:
            logger.info("Step 2d: Skipped — no signal results or master_df")
            status["steps_ok"].append("trade_structure_skipped")
    except Exception as exc:
        logger.warning("Step 2d: Trade structure failed (non-fatal) — %s", exc, exc_info=True)
        status["steps_failed"].append("trade_structure")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 [OPTIONAL]: Walk-forward backtest
    # ──────────────────────────────────────────────────────────────────────────
    backtest_result = None
    should_backtest = run_backtest
    if not should_backtest and db_conn is not None:
        should_backtest = not _is_backtest_fresh(db_conn, max_age_days=7)
        if should_backtest:
            logger.info("Step 3: Backtest cache stale — running backtest")
        else:
            logger.info("Step 3: Backtest cache is fresh — skipping")

    if should_backtest:
        try:
            from analytics.backtest import run_backtest as _run_backtest
            from db.reader import DatabaseReader
            import pandas as pd

            reader = DatabaseReader(settings.db_path)
            prices_df  = reader.read_prices()
            fund_df    = reader.read_fundamentals()
            weights_df = reader.read_weights()

            # Parquet fallback when DB not yet populated
            if prices_df is None or prices_df.empty:
                prices_df = pd.read_parquet(data_state.artifacts.prices_path)
            if fund_df is None or fund_df.empty:
                fund_df = pd.read_parquet(data_state.artifacts.fundamentals_path)
            if weights_df is None or weights_df.empty:
                weights_df = pd.read_parquet(data_state.artifacts.weights_path)

            logger.info("Step 3: Running walk-forward backtest...")
            bt_start = time.time()
            backtest_result = _run_backtest(prices_df, fund_df, weights_df, settings)
            bt_elapsed = time.time() - bt_start

            status["ic_mean"] = getattr(backtest_result, "ic_mean", None)
            logger.info(
                "Step 3 OK — IC_mean=%.4f, Sharpe=%.2f, n_walks=%d (%.1fs)",
                backtest_result.ic_mean or 0,
                backtest_result.sharpe or 0,
                backtest_result.n_walks or 0,
                bt_elapsed,
            )

            # Cache result
            if db_conn is not None:
                try:
                    from db.writer import DatabaseWriter
                    DatabaseWriter(settings.db_path).write_backtest_cache(backtest_result)
                except Exception as dbe:
                    logger.warning("Step 3: write_backtest_cache failed (non-fatal) — %s", dbe)

            status["steps_ok"].append("backtest")
        except Exception as exc:
            logger.warning("Step 3: Backtest failed (non-fatal) — %s", exc, exc_info=True)
            status["steps_failed"].append("backtest")
    else:
        status["steps_ok"].append("backtest_skipped")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3b [OPTIONAL]: Alpha Research (weekly OOS validation — non-fatal)
    # Runs only when --alpha-research flag is set OR last report is > 7 days old
    # ──────────────────────────────────────────────────────────────────────────
    _run_alpha = run_alpha_research
    if not _run_alpha:
        # Check for stale report (> 7 days)
        import glob as _glob
        _report_dir = ROOT / "agents" / "methodology" / "reports"
        _reports = sorted(_glob.glob(str(_report_dir / "*alpha_research*.json")))
        if _reports:
            try:
                _last_report_date = date.fromisoformat(_reports[-1].split("\\")[-1].split("/")[-1][:10])
                _run_alpha = (date.today() - _last_report_date).days > 7
            except Exception:
                _run_alpha = True
        else:
            _run_alpha = True  # No report yet — run now

    if _run_alpha:
        try:
            from analytics.alpha_research import run_alpha_research
            from db.reader import DatabaseReader as _AlphaDBR
            _alpha_prices = _AlphaDBR(settings.db_path).read_prices()
            if _alpha_prices is not None and len(_alpha_prices) >= 600:
                _alpha_report = run_alpha_research(
                    _alpha_prices[settings.sector_list()].dropna(how="all"),
                    settings,
                    include_gpt=False,   # GPT optional; skip in automated pipeline
                )
                status["steps_ok"].append("alpha_research")
                status["alpha_oos_sharpe"] = _alpha_report.ensemble_sharpe
                logger.info(
                    "Step 3b OK — OOS ensemble Sharpe=%.3f | recs=%d",
                    _alpha_report.ensemble_sharpe, len(_alpha_report.recommendations),
                )
            else:
                logger.info("Step 3b: Insufficient price history for alpha research — skipping")
                status["steps_ok"].append("alpha_research_skipped")
        except Exception as exc:
            logger.warning("Step 3b: Alpha research failed (non-fatal) — %s", exc)
            status["steps_failed"].append("alpha_research")
    else:
        logger.debug("Step 3b: Alpha research report is fresh — skipping")
        status["steps_ok"].append("alpha_research_fresh")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 [OPTIONAL]: ML signal quality
    # ──────────────────────────────────────────────────────────────────────────
    quality_scores: Dict[str, float] = {}
    if run_ml:
        try:
            from analytics.ml_signals import SignalQualityModel, apply_quality_to_master

            ml_model = SignalQualityModel()
            ml_model.train(backtest_result)  # backtest_result may be None → neutral
            quality_scores = ml_model.predict(status["regime"], master_df)

            if quality_scores:
                master_df = apply_quality_to_master(master_df, quality_scores)
                logger.info(
                    "Step 4 OK — ML model=%s, %d quality scores, "
                    "avg_quality=%.3f",
                    ml_model.model_type,
                    len(quality_scores),
                    sum(quality_scores.values()) / max(len(quality_scores), 1),
                )

                # Persist ML predictions
                if run_id > 0 and db_conn is not None:
                    try:
                        regime_ic = getattr(ml_model, "_regime_ic", {})
                        from db.writer import DatabaseWriter
                        DatabaseWriter(settings.db_path).write_ml_predictions(
                            run_id=run_id,
                            predictions=quality_scores,
                            regime_ic={t: regime_ic.get(status["regime"], 0.0) for t in quality_scores},
                            model_type=ml_model.model_type,
                        )
                    except Exception as dbe:
                        logger.warning("Step 4: write_ml_predictions failed (non-fatal) — %s", dbe)

            status["steps_ok"].append("ml_signals")
        except Exception as exc:
            logger.warning("Step 4: ML signals failed (non-fatal) — %s", exc, exc_info=True)
            status["steps_failed"].append("ml_signals")
    else:
        logger.info("Step 4: ML signals skipped (--no-ml)")
        status["steps_ok"].append("ml_skipped")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 [OPTIONAL]: Portfolio optimisation
    # ──────────────────────────────────────────────────────────────────────────
    if run_optimizer:
        try:
            from analytics.optimizer import PortfolioOptimizer
            from db.reader import DatabaseReader

            reader = DatabaseReader(settings.db_path)
            prices_df = reader.read_prices()

            optimizer = PortfolioOptimizer()
            master_df = optimizer.apply_to_master_df(master_df, prices_df, settings)

            if "opt_weight" in master_df.columns:
                gross = master_df["opt_weight"].abs().sum()
                logger.info("Step 5 OK — gross_opt_exposure=%.2f", gross)
                status["opt_weights_computed"] = True

                # Persist optimizer output
                if run_id > 0 and db_conn is not None:
                    try:
                        from db.writer import DatabaseWriter
                        DatabaseWriter(settings.db_path).write_optimization_results(
                            run_id=run_id,
                            opt_df=master_df,
                        )
                    except Exception as dbe:
                        logger.warning("Step 5: write_optimization_results failed (non-fatal) — %s", dbe)

            status["steps_ok"].append("optimizer")
        except Exception as exc:
            logger.warning("Step 5: Optimizer failed (non-fatal) — %s", exc, exc_info=True)
            status["steps_failed"].append("optimizer")
    else:
        status["steps_ok"].append("optimizer_skipped")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 [OPTIONAL]: Stress + Portfolio Risk
    # ──────────────────────────────────────────────────────────────────────────
    stress_results = None
    risk_report = None
    try:
        from analytics.stress import StressEngine
        stress_results = StressEngine().run_all(master_df, settings)
        logger.info("Step 6a OK — stress: %d scenarios", len(stress_results) if stress_results else 0)
        status["steps_ok"].append("stress")
    except Exception as exc:
        logger.warning("Step 6a: Stress engine failed (non-fatal) — %s", exc)
        status["steps_failed"].append("stress")

    try:
        from analytics.portfolio_risk import PortfolioRiskEngine
        import pandas as pd

        # Read prices: DB first, parquet fallback
        prices_df_risk: Optional[pd.DataFrame] = None
        try:
            from db.reader import DatabaseReader
            _r = DatabaseReader(settings.db_path)
            _p = _r.read_prices()
            if _p is not None and not _p.empty:
                prices_df_risk = _p
        except Exception:
            pass
        if prices_df_risk is None or prices_df_risk.empty:
            prices_df_risk = pd.read_parquet(data_state.artifacts.prices_path)

        # Build weights dict from master_df (opt_weight preferred, else w_final, else equal)
        wt_col = next((c for c in ("opt_weight", "w_final") if c in master_df.columns), None)
        if wt_col and "sector_ticker" in master_df.columns:
            weights_dict = dict(zip(master_df["sector_ticker"], master_df[wt_col]))
        elif "sector_ticker" in master_df.columns:
            n = len(master_df)
            weights_dict = {t: 1.0 / n for t in master_df["sector_ticker"]}
        else:
            weights_dict = {}

        if weights_dict:
            risk_report = PortfolioRiskEngine().full_risk_report(weights_dict, prices_df_risk, settings)
            logger.info("Step 6b OK — portfolio risk computed (vol=%.2f%%)",
                        risk_report.portfolio_vol_ann * 100)
            status["steps_ok"].append("portfolio_risk")
        else:
            logger.warning("Step 6b: No weights available — skipping risk")
            status["steps_failed"].append("portfolio_risk")
    except Exception as exc:
        logger.warning("Step 6b: Portfolio risk failed (non-fatal) — %s", exc)
        status["steps_failed"].append("portfolio_risk")

    # ──────────────────────────────────────────────────────────────────────────
    # Ensure prices_df is available for Steps 6c-6e (it may not be set if
    # both the backtest and optimizer steps were skipped/failed).
    # ──────────────────────────────────────────────────────────────────────────
    try:
        prices_df  # noqa: test if bound
    except NameError:
        import pandas as pd
        try:
            from db.reader import DatabaseReader
            _r2 = DatabaseReader(settings.db_path)
            prices_df = _r2.read_prices()
            if prices_df is None or prices_df.empty:
                prices_df = pd.read_parquet(data_state.artifacts.prices_path)
        except Exception:
            prices_df = pd.read_parquet(data_state.artifacts.prices_path)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6c [OPTIONAL]: P&L Tracking
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from analytics.pnl_tracker import PnLTracker
        _pnl = PnLTracker(settings).track(prices_df, lookback_days=252)
        logger.info(
            "Step 6c OK — P&L tracker: total=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%, HR=%.1f%%",
            _pnl.total_pnl * 100, _pnl.sharpe, _pnl.max_drawdown * 100, _pnl.hit_rate * 100,
        )
        status["steps_ok"].append("pnl_tracker")
        status["pnl_total_pct"] = round(_pnl.total_pnl * 100, 2)
        status["pnl_sharpe"] = round(_pnl.sharpe, 2)
    except Exception as exc:
        logger.warning("Step 6c: P&L tracker failed (non-fatal) — %s", exc)
        status["steps_failed"].append("pnl_tracker")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6d [OPTIONAL]: Signal Decay Analysis (compute-intensive)
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from analytics.signal_decay import SignalDecayAnalyser
        _decay = SignalDecayAnalyser(settings).analyse(prices_df)
        logger.info(
            "Step 6d OK — signal decay: optimal_horizon=%dd, IC=%.4f, turnover=%.1f/yr",
            _decay.optimal_horizon, _decay.optimal_ic, _decay.annualised_turnover,
        )
        status["steps_ok"].append("signal_decay")
        status["optimal_horizon"] = _decay.optimal_horizon
        status["decay_cost_bps"] = round(_decay.estimated_cost_bps_pa, 1)
    except Exception as exc:
        logger.warning("Step 6d: Signal decay failed (non-fatal) — %s", exc)
        status["steps_failed"].append("signal_decay")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6e [OPTIONAL]: Regime Transition Alerts
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from analytics.regime_alerts import RegimeAlertEngine
        _regime_eng = RegimeAlertEngine(settings)
        _regime_result = _regime_eng.analyse(prices_df)
        # Save alerts JSON for external consumers
        _regime_eng.save_alerts(_regime_result, ROOT / "logs" / "regime_alerts.json")
        n_transitions = len(_regime_result.transitions)
        n_warnings = len(_regime_result.warnings)
        logger.info(
            "Step 6e OK — regime alerts: current=%s, transitions=%d, warnings=%d",
            _regime_result.current_regime, n_transitions, n_warnings,
        )
        status["steps_ok"].append("regime_alerts")
        status["current_regime"] = _regime_result.current_regime
        if _regime_result.active_alerts:
            for _alert in _regime_result.active_alerts:
                logger.info("  ALERT [%s]: %s", _alert.get("level", "?"), _alert.get("message", ""))
    except Exception as exc:
        logger.warning("Step 6e: Regime alerts failed (non-fatal) — %s", exc)
        status["steps_failed"].append("regime_alerts")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 [OPTIONAL]: Daily brief
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from reports.daily_report import DailyBriefGenerator
        output_dir = ROOT / "reports" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        brief_gen = DailyBriefGenerator()
        txt_path, json_path = brief_gen.generate(
            master_df=master_df,
            risk_report=risk_report,
            stress_results=stress_results or [],
            journal=None,
            settings=settings,
            output_dir=output_dir,
        )
        logger.info("Step 7 OK — brief saved: %s", txt_path)
        status["steps_ok"].append("daily_brief")
        status["brief_path"] = str(txt_path)
    except Exception as exc:
        logger.warning("Step 7: Daily brief failed (non-fatal) — %s", exc)
        status["steps_failed"].append("daily_brief")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7b: DSS Brief (Short Vol / Dispersion specific)
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from reports.dss_brief_generator import DSSBriefGenerator
        dss_output_dir = ROOT / "reports" / "output"
        dss_output_dir.mkdir(parents=True, exist_ok=True)

        _dss_regime = locals().get("regime_safety")
        _dss_signals = signal_stack_results
        _dss_tickets = trade_tickets

        if _dss_signals or _dss_tickets:
            dss_gen = DSSBriefGenerator()
            dss_txt, dss_json = dss_gen.generate(
                signal_results=_dss_signals,
                trade_tickets=_dss_tickets,
                regime_safety=_dss_regime,
                settings=settings,
                output_dir=dss_output_dir,
            )
            logger.info("Step 7b OK — DSS brief saved: %s", dss_txt)
            status["steps_ok"].append("dss_brief")
            status["dss_brief_path"] = str(dss_txt)
        else:
            logger.info("Step 7b: Skipped — no DSS data")
            status["steps_ok"].append("dss_brief_skipped")
    except Exception as exc:
        logger.warning("Step 7b: DSS brief failed (non-fatal) — %s", exc)
        status["steps_failed"].append("dss_brief")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7c: Slack dispatch (non-fatal)
    # ──────────────────────────────────────────────────────────────────────────
    _webhook = getattr(settings, "slack_webhook_url", "")
    _notify_ok   = getattr(settings, "notify_on_pipeline_complete", True)
    _notify_fail = getattr(settings, "notify_on_pipeline_failure", True)
    _has_failures = bool(status.get("steps_failed"))

    if _webhook and ((_notify_ok and not _has_failures) or (_notify_fail and _has_failures)):
        try:
            status["duration_s"] = (datetime.now(timezone.utc) - started_at).total_seconds()
            _msg = _build_pipeline_slack_message(status)
            _sent = _dispatch_slack(_webhook, _msg)
            if _sent:
                logger.info("Step 7c OK — Slack notification dispatched")
                status["steps_ok"].append("slack_dispatch")
            else:
                logger.warning("Step 7c: Slack dispatch returned non-200 (non-fatal)")
                status["steps_failed"].append("slack_dispatch")
        except Exception as exc:
            logger.warning("Step 7c: Slack dispatch failed (non-fatal) — %s", exc)
            status["steps_failed"].append("slack_dispatch")
    else:
        if not _webhook:
            logger.debug("Step 7c: Slack webhook not configured — skipping")
        status["steps_ok"].append("slack_skipped")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 8 [ALWAYS]: DB audit write + data pruning
    # ──────────────────────────────────────────────────────────────────────────
    try:
        if db_conn is not None:
            _prune_old_data(db_conn, snapshot_retain_days=90, holdings_retain_days=30)
            logger.info("Step 8 OK — old snapshots pruned")
        status["steps_ok"].append("db_prune")
    except Exception as exc:
        logger.warning("Step 8: Data pruning failed (non-fatal) — %s", exc)
        status["steps_failed"].append("db_prune")

    # ── Final summary ─────────────────────────────────────────────────────────
    status["duration_s"] = (datetime.now(timezone.utc) - started_at).total_seconds()
    status["finished_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 70)
    logger.info(
        "Pipeline COMPLETE — run_id=%d | %.1fs | OK=%s | FAIL=%s",
        status["run_id"],
        status["duration_s"],
        ",".join(status["steps_ok"]) or "none",
        ",".join(status["steps_failed"]) or "none",
    )
    logger.info("=" * 70)

    # Write status JSON for external monitoring
    try:
        status_path = ROOT / "logs" / "last_run_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, default=str)
    except Exception:
        pass

    return status


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="SRV Quantamental DSS — Master Daily Pipeline",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force FMP data re-fetch even if DB is fresh",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Force walk-forward backtest (otherwise runs only if cache > 7 days old)",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Skip ML signal quality layer",
    )
    parser.add_argument(
        "--alpha-research",
        action="store_true",
        help="Force alpha research OOS validation (otherwise runs only if report > 7 days old)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(
        force_refresh=args.force_refresh,
        run_backtest=args.backtest,
        run_ml=not args.no_ml,
        run_alpha_research=args.alpha_research,
    )
    failed = result.get("steps_failed", [])
    # Exit code 0 if REQUIRED steps succeeded, 1 if any required step failed
    required_failed = [s for s in failed if s in ("data_refresh", "quant_engine")]
    sys.exit(1 if required_failed else 0)
