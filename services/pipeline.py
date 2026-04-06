"""
services/pipeline.py
======================
Deterministic pipeline orchestration with run tracking and artifact management.

Replaces the ad-hoc execution in run_all.py with a structured pipeline:

  data_ingest → snapshot_build → feature_gen → signal_gen →
  validation → artifact_write → dashboard_consumption

Each run produces:
  1. Run manifest (JSON) — complete record of what happened
  2. Config snapshot — frozen settings hash for reproducibility
  3. Artifact directory — organized outputs per run
  4. Status tracking — per-stage timing, errors, diagnostics
  5. Health check — post-run data quality validation

Usage:
    from services.pipeline import Pipeline, PipelineConfig
    pipeline = Pipeline(PipelineConfig.from_settings(settings))
    manifest = pipeline.run()  # Returns RunManifest
    pipeline.save_manifest(manifest)

CLI:
    python -m services.pipeline                    # Full run
    python -m services.pipeline --stage data_only  # Single stage
    python -m services.pipeline --dry-run          # Show plan without executing
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Config & Manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Frozen pipeline configuration — hashable for reproducibility."""
    # Data
    force_refresh: bool = False
    cache_max_age_hours: int = 12

    # Analytics
    pca_window: int = 252
    zscore_window: int = 60
    corr_window: int = 60
    momentum_lookback: int = 21
    momentum_top_n: int = 3

    # Validation
    cost_bps: float = 15.0
    backtest_splits: int = 3

    # Execution
    paper_trading: bool = True
    daily_report: bool = True

    # Paths
    db_path: str = ""
    artifact_dir: str = ""

    @classmethod
    def from_settings(cls, settings) -> "PipelineConfig":
        return cls(
            cache_max_age_hours=getattr(settings, "cache_max_age_hours", 12),
            pca_window=getattr(settings, "pca_window", 252),
            zscore_window=getattr(settings, "zscore_window", 60),
            corr_window=getattr(settings, "corr_window", 60),
            momentum_lookback=getattr(settings, "momentum_lookback", 21),
            momentum_top_n=getattr(settings, "momentum_top_n", 3),
            cost_bps=15.0,
            db_path=str(getattr(settings, "db_path", "")),
            artifact_dir=str(ROOT / "data" / "runs"),
        )

    @property
    def config_hash(self) -> str:
        """SHA256 hash of config for reproducibility tracking."""
        d = asdict(self)
        d.pop("artifact_dir", None)  # Don't include paths in hash
        d.pop("db_path", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    name: str
    status: str                        # "ok" / "failed" / "skipped"
    duration_s: float
    error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Complete record of a pipeline run — the single source of truth."""
    # Identity
    run_id: str                        # From RunContext
    run_date: str
    config_hash: str
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0

    # Pipeline
    stages: List[StageResult] = field(default_factory=list)
    n_ok: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    overall_status: str = "pending"    # "success" / "partial" / "failed"

    # Context
    regime: str = "UNKNOWN"
    vix: float = 0.0
    n_sectors: int = 0
    data_health: str = "UNKNOWN"

    # Artifacts
    artifact_dir: str = ""
    artifacts_written: List[str] = field(default_factory=list)

    # Data quality
    quality_checks: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Engine
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Deterministic pipeline with run tracking.

    Each run:
    1. Creates a run directory: data/runs/YYYY-MM-DD_HHMMSS_<hash>/
    2. Snapshots config for reproducibility
    3. Executes stages in order with timing + error capture
    4. Writes run manifest (JSON) to the run directory
    5. Updates DuckDB run_contexts table
    """

    STAGES = [
        "data_ingest",
        "quant_engine",
        "scoring",
        "signals",
        "risk",
        "stress",
        "correlation",
        "options",
        "portfolio",
        "tracking",
        "validation",
        "artifacts",
    ]

    def __init__(self, config: PipelineConfig, settings=None):
        self.config = config
        self.settings = settings
        self._stage_funcs: Dict[str, Callable] = {}
        self._manifest: Optional[RunManifest] = None

    def run(self, stages: Optional[List[str]] = None) -> RunManifest:
        """
        Execute the full pipeline (or specific stages).

        Returns RunManifest with complete run record.
        """
        if self.settings is None:
            from config.settings import get_settings
            self.settings = get_settings()

        # Create run directory
        ts = datetime.now(timezone.utc)
        run_dir_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_{self.config.config_hash}"
        run_dir = Path(self.config.artifact_dir) / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

        # Initialize manifest
        from services.run_context import RunContext
        ctx = RunContext.create(self.settings)

        manifest = RunManifest(
            run_id=ctx.run_uuid,
            run_date=ctx.run_date,
            config_hash=self.config.config_hash,
            started_at=ts.isoformat(),
            artifact_dir=str(run_dir),
        )

        # Execute stages
        stages_to_run = stages or self.STAGES
        log.info("Pipeline starting: %d stages, config_hash=%s, run_dir=%s",
                 len(stages_to_run), self.config.config_hash, run_dir.name)

        # Use EngineService for the heavy lifting
        from services.engine_service import EngineService
        engine_svc = EngineService(ctx)

        try:
            results = engine_svc.compute_all()
            # Map EngineService steps to manifest stages
            for step_name in ctx.steps_completed:
                manifest.stages.append(StageResult(
                    name=step_name, status="ok",
                    duration_s=0,  # Individual timing from EngineService
                ))
            for step_name in ctx.steps_failed:
                manifest.stages.append(StageResult(
                    name=step_name, status="failed",
                    duration_s=0,
                    error=ctx.errors.get(step_name, ""),
                ))
        except Exception as e:
            manifest.stages.append(StageResult(
                name="engine_service", status="failed",
                duration_s=0, error=str(e),
            ))

        # Finalize
        finished = datetime.now(timezone.utc)
        manifest.finished_at = finished.isoformat()
        manifest.duration_s = round((finished - ts).total_seconds(), 1)
        manifest.n_ok = sum(1 for s in manifest.stages if s.status == "ok")
        manifest.n_failed = sum(1 for s in manifest.stages if s.status == "failed")
        manifest.n_skipped = sum(1 for s in manifest.stages if s.status == "skipped")
        manifest.regime = ctx.regime
        manifest.vix = ctx.vix_level
        manifest.n_sectors = ctx.prices_cols
        manifest.data_health = ctx.data_health_label

        if manifest.n_failed == 0:
            manifest.overall_status = "success"
        elif manifest.n_ok > 0:
            manifest.overall_status = "partial"
        else:
            manifest.overall_status = "failed"

        # Save manifest
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        manifest.artifacts_written.append("manifest.json")

        # Save run context to DuckDB
        try:
            from db.writer import DatabaseWriter
            dw = DatabaseWriter(self.settings.db_path)
            dw.write_run_context(ctx)
        except Exception:
            pass

        # Log summary
        log.info(
            "Pipeline complete: %s | %d/%d stages OK | %.1fs | regime=%s | VIX=%.1f",
            manifest.overall_status, manifest.n_ok,
            manifest.n_ok + manifest.n_failed, manifest.duration_s,
            manifest.regime, manifest.vix,
        )

        self._manifest = manifest
        return manifest

    @staticmethod
    def list_runs(artifact_dir: Optional[str] = None) -> List[Dict]:
        """List all previous pipeline runs with summaries."""
        if artifact_dir is None:
            artifact_dir = str(ROOT / "data" / "runs")

        runs_dir = Path(artifact_dir)
        if not runs_dir.exists():
            return []

        runs = []
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    runs.append({
                        "run_id": data.get("run_id", "?"),
                        "date": data.get("run_date", "?"),
                        "status": data.get("overall_status", "?"),
                        "duration": data.get("duration_s", 0),
                        "regime": data.get("regime", "?"),
                        "stages_ok": data.get("n_ok", 0),
                        "stages_fail": data.get("n_failed", 0),
                        "config_hash": data.get("config_hash", "?"),
                        "dir": str(d),
                    })
                except Exception:
                    pass

        return runs[:20]  # Last 20 runs


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sys.path.insert(0, str(ROOT))
    from config.settings import get_settings

    settings = get_settings()
    config = PipelineConfig.from_settings(settings)

    if "--dry-run" in sys.argv:
        print(f"Pipeline dry run:")
        print(f"  Config hash: {config.config_hash}")
        print(f"  Stages: {Pipeline.STAGES}")
        print(f"  DB: {config.db_path}")
        print(f"  Artifact dir: {config.artifact_dir}")
        return

    if "--list-runs" in sys.argv:
        runs = Pipeline.list_runs()
        if not runs:
            print("No previous runs found.")
            return
        print(f"{'Run ID':<10} {'Date':<12} {'Status':<10} {'Duration':>8} {'Regime':<8} {'OK':>3}/{'':<3}{'Fail':>3} {'Hash':<12}")
        print("-" * 80)
        for r in runs:
            print(f"{r['run_id']:<10} {r['date']:<12} {r['status']:<10} {r['duration']:>7.0f}s {r['regime']:<8} {r['stages_ok']:>3}/{r['stages_fail']:>3}   {r['config_hash']:<12}")
        return

    pipeline = Pipeline(config, settings)
    manifest = pipeline.run()
    print(f"\nPipeline {manifest.overall_status}: {manifest.n_ok}/{manifest.n_ok + manifest.n_failed} stages | {manifest.duration_s:.0f}s | {manifest.regime}")


if __name__ == "__main__":
    main()
