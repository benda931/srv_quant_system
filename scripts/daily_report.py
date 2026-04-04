"""
scripts/daily_report.py
=========================
Automated Daily PM Report Generator

Produces a comprehensive end-of-day report for the PM including:
  1. Market regime + VIX status
  2. Portfolio P&L (paper trading)
  3. Signal stack: active trades + pending signals
  4. Strategy performance: best/worst strategies
  5. Risk summary: VaR, exposure, concentration
  6. Auto-improve activity: what agents changed
  7. Action items: required PM decisions

Delivery channels:
  - Slack (via AlertService webhook)
  - File (reports/output/daily_report_YYYY-MM-DD.txt)
  - DuckDB (analytics.daily_reports table — future)

Usage:
  python scripts/daily_report.py           # Generate + deliver
  python scripts/daily_report.py --preview # Preview without sending
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
log = logging.getLogger("daily_report")


def generate_daily_report(preview: bool = False) -> str:
    """
    Generate the daily PM report.

    Loads all available data and produces a structured text report.
    """
    from config.settings import get_settings
    settings = get_settings()
    today = date.today().isoformat()

    lines = [
        f"{'='*70}",
        f"  SRV QUANTAMENTAL DSS — DAILY PM REPORT",
        f"  {today} | Generated at {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
        f"{'='*70}",
        "",
    ]

    # ── 1. Market Regime ────────────────────────────────────────────
    lines.append("[1] MARKET REGIME")
    try:
        from services.data_loader import DataLoader
        loader = DataLoader(settings)
        regime_data = loader.load_json(settings.project_root / "agents" / "regime_forecaster" / "regime_forecast.json")
        if regime_data:
            regime = regime_data.get("predicted_regime", "UNKNOWN")
            confidence = regime_data.get("confidence", 0)
            vix = regime_data.get("features", {}).get("vix_level", 0)
            lines.append(f"    Regime:      {regime}")
            lines.append(f"    Confidence:  {confidence:.0%}")
            lines.append(f"    VIX:         {vix:.1f}")
        else:
            lines.append("    No regime data available")
    except Exception as e:
        lines.append(f"    Error: {e}")

    # ── 2. Portfolio P&L ────────────────────────────────────────────
    lines.append("")
    lines.append("[2] PAPER PORTFOLIO")
    try:
        pp_path = settings.project_root / "data" / "paper_portfolio.json"
        if pp_path.exists():
            pp = json.loads(pp_path.read_text(encoding="utf-8"))
            pnl = pp.get("total_pnl", 0)
            pnl_pct = pp.get("total_pnl_pct", 0)
            n_pos = len(pp.get("positions", []))
            wr = pp.get("win_rate", 0)
            sharpe = pp.get("sharpe", 0)
            lines.append(f"    P&L:         ${pnl:+,.0f} ({pnl_pct:+.2%})")
            lines.append(f"    Positions:   {n_pos}")
            lines.append(f"    Win Rate:    {wr:.0%}")
            lines.append(f"    Sharpe:      {sharpe:.2f}")

            # Top positions
            positions = pp.get("positions", [])
            if positions:
                lines.append(f"    Open positions:")
                for pos in sorted(positions, key=lambda x: -abs(x.get("unrealized_pnl", 0)))[:5]:
                    src = "MOM" if pos.get("signal_source") == "momentum" else "DSS"
                    pnl_pos = pos.get("unrealized_pnl", 0)
                    pnl_pct_pos = pos.get("unrealized_pnl_pct", 0)
                    lines.append(f"      [{src}] {pos['direction']:<5} {pos['ticker']:<5} ${pnl_pos:+,.0f} ({pnl_pct_pos:+.1%}) {pos.get('days_held', 0)}d")
        else:
            lines.append("    No paper portfolio data")
    except Exception as e:
        lines.append(f"    Error: {e}")

    # ── 3. Strategy Performance ─────────────────────────────────────
    lines.append("")
    lines.append("[3] STRATEGY PERFORMANCE")
    try:
        from services.data_loader import DataLoader
        loader = DataLoader(settings)
        meth = loader.load_methodology_results()
        if meth:
            ranked = sorted(meth.items(), key=lambda x: x[1].get("sharpe", 0) if isinstance(x[1], dict) else 0, reverse=True)
            positive = sum(1 for _, v in ranked if isinstance(v, dict) and v.get("sharpe", 0) > 0)
            lines.append(f"    Total strategies: {len(ranked)} ({positive} positive Sharpe)")
            for name, data in ranked[:5]:
                if isinstance(data, dict):
                    s = data.get("sharpe", 0)
                    wr = data.get("win_rate", 0)
                    pnl = data.get("total_pnl", 0)
                    icon = "✓" if s > 0 else "✗"
                    lines.append(f"    {icon} {name:<30} Sharpe={s:>7.3f} WR={wr:>5.1%} PnL={pnl:>7.1%}")
        else:
            lines.append("    No methodology results available")
    except Exception as e:
        lines.append(f"    Error: {e}")

    # ── 4. Risk Summary ─────────────────────────────────────────────
    lines.append("")
    lines.append("[4] RISK SUMMARY")
    try:
        from db.repository import Repository
        repo = Repository(settings.db_path)
        freshness = repo.data_freshness()
        lines.append(f"    Data fresh:  {'Yes' if freshness.is_fresh else 'NO — STALE'}")
        lines.append(f"    Last run:    #{freshness.last_successful_run_id} ({freshness.last_successful_date})")
        lines.append(f"    Prices:      {freshness.prices_rows:,} rows, latest {freshness.prices_latest}")

        # Quality checks
        checks = repo.run_data_quality_checks()
        n_pass = sum(1 for c in checks if c.status == "PASS")
        n_fail = sum(1 for c in checks if c.status == "FAIL")
        lines.append(f"    Quality:     {n_pass} PASS, {n_fail} FAIL")
        for c in checks:
            if c.status != "PASS":
                lines.append(f"      ⚠ {c.check_name}: {c.message}")
    except Exception as e:
        lines.append(f"    Error: {e}")

    # ── 5. Auto-Improve Activity ────────────────────────────────────
    lines.append("")
    lines.append("[5] AUTO-IMPROVE ACTIVITY")
    try:
        from services.data_loader import DataLoader
        loader = DataLoader(settings)
        imp_log = loader.load_improvement_log()
        if imp_log:
            cycles = imp_log.get("cycles", [])
            recent = cycles[-3:] if cycles else []
            total_promoted = sum(c.get("promoted", 0) for c in cycles)
            lines.append(f"    Total cycles: {len(cycles)} | Total promoted: {total_promoted}")
            for cycle in recent:
                ts = cycle.get("timestamp", "")[:16]
                promoted = cycle.get("promoted", 0)
                tested = cycle.get("suggestions_tested", 0)
                lines.append(f"      {ts}: {tested} tested, {promoted} promoted")
        else:
            lines.append("    No improvement log available")
    except Exception as e:
        lines.append(f"    Error: {e}")

    # ── 6. Agent Status ─────────────────────────────────────────────
    lines.append("")
    lines.append("[6] AGENT STATUS")
    try:
        ms = json.loads((settings.project_root / "agents" / "auto_improve" / "machine_summary.json").read_text(encoding="utf-8"))
        health = ms.get("system_health_score", 0)
        status = ms.get("stuck_state", "UNKNOWN")
        lines.append(f"    Health:      {health:.0%}")
        lines.append(f"    Status:      {status}")
        lines.append(f"    Bottleneck:  {ms.get('diagnosed_bottleneck', 'N/A')}")
    except Exception:
        lines.append("    No machine summary available")

    # ── 7. Action Items ─────────────────────────────────────────────
    lines.append("")
    lines.append("[7] ACTION ITEMS")
    actions = []

    # Check for regime warnings
    try:
        if regime_data and regime_data.get("predicted_regime") in ("TENSION", "CRISIS"):
            actions.append(f"[P1] Regime is {regime_data['predicted_regime']} — review position sizing")
    except Exception:
        pass

    # Check for stale data
    try:
        if freshness and not freshness.is_fresh:
            actions.append("[P1] Data is STALE — run pipeline: python scripts/run_all.py --force-refresh")
    except Exception:
        pass

    # Check paper portfolio
    try:
        if pp and pnl_pct < -0.03:
            actions.append(f"[P1] Portfolio drawdown {pnl_pct:.1%} — consider reducing exposure")
    except Exception:
        pass

    if not actions:
        actions.append("No action required — system is operating normally")

    for a in actions:
        lines.append(f"    {a}")

    lines.extend(["", f"{'='*70}", f"  End of Daily Report — {today}", f"{'='*70}", ""])

    report = "\n".join(lines)

    # Save to file
    output_dir = settings.project_root / "reports" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"daily_report_{today}.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", report_path.name)

    # Deliver via Slack (unless preview mode)
    if not preview:
        try:
            from services.alerting import AlertService, Alert
            alerts = AlertService(settings)
            # Send summary to Slack
            summary_lines = [l for l in lines if l.strip() and not l.startswith("=")][:15]
            alerts.send(Alert(
                level="INFO",
                title=f"Daily PM Report — {today}",
                message="\n".join(summary_lines),
                source="daily_report",
            ))
        except Exception as e:
            log.warning("Slack delivery failed: %s", e)

    return report


if __name__ == "__main__":
    preview = "--preview" in sys.argv
    report = generate_daily_report(preview=preview)
    print(report)
