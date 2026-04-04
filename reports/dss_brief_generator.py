"""
reports/dss_brief_generator.py
================================
DSS Brief Generator — extends the daily PM brief with Short-Vol /
Dispersion trade-specific sections.

Produces a professional hedge-fund-grade daily brief that includes:

  1. Regime Safety Assessment — current state, penalties, kill switches
  2. Correlation Distortion — Frobenius, market-mode, CoC metrics
  3. Signal Stack Summary — top candidates across all 4 layers
  4. Trade Book — active positions, legs, Greeks, weights
  5. Trade Monitor — health status, exit signals, urgent actions
  6. Action Items — prioritized PM todo list

Outputs:
  reports/output/YYYY-MM-DD_dss_brief.txt
  reports/output/YYYY-MM-DD_dss_brief.json
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _fmt(x: float, fmt: str = "{:.2f}") -> str:
    return fmt.format(x) if math.isfinite(x) else "N/A"


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%" if math.isfinite(x) else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

class DSSBriefGenerator:
    """
    Generate the DSS daily brief (TXT + JSON).

    Usage::

        gen = DSSBriefGenerator()
        txt_path, json_path = gen.generate(
            signal_results=...,
            trade_tickets=...,
            regime_safety=...,
            corr_snapshot=...,
            monitor_summary=...,
            settings=settings,
        )
    """

    def generate(
        self,
        signal_results: Optional[list] = None,
        trade_tickets: Optional[list] = None,
        regime_safety: Any = None,
        corr_snapshot: Any = None,
        monitor_summary: Any = None,
        options_surface: Any = None,
        settings: Any = None,
        output_dir: Optional[Path] = None,
        run_date: Optional[date] = None,
    ) -> Tuple[Path, Path]:
        """Generate DSS brief files. Returns (txt_path, json_path)."""
        if run_date is None:
            run_date = date.today()

        if output_dir is None and settings is not None:
            output_dir = Path(settings.project_root) / "reports" / "output"
        elif output_dir is None:
            output_dir = Path("reports/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(
            signal_results, trade_tickets, regime_safety,
            corr_snapshot, monitor_summary, options_surface, run_date,
        )

        txt_content = self._render_txt(payload, run_date)
        json_content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

        ds = run_date.strftime("%Y-%m-%d")
        txt_path = output_dir / f"{ds}_dss_brief.txt"
        json_path = output_dir / f"{ds}_dss_brief.json"

        txt_path.write_text(txt_content, encoding="utf-8")
        json_path.write_text(json_content, encoding="utf-8")

        logger.info("DSS brief written: %s | %s", txt_path.name, json_path.name)
        return txt_path, json_path

    # ─────────────────────────────────────────────────────────────────────
    # Payload construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_payload(self, signal_results, trade_tickets, regime_safety,
                       corr_snapshot, monitor_summary, options_surface, run_date) -> dict:
        payload = {
            "date": run_date.strftime("%Y-%m-%d"),
            "regime_safety": self._regime_safety_section(regime_safety),
            "correlation": self._correlation_section(corr_snapshot),
            "options": self._options_section(options_surface),
            "signal_stack": self._signal_stack_section(signal_results),
            "trade_book": self._trade_book_section(trade_tickets),
            "monitor": self._monitor_section(monitor_summary),
            "action_items": self._action_items(signal_results, trade_tickets,
                                                regime_safety, monitor_summary),
        }
        return payload

    def _options_section(self, surface) -> dict:
        if surface is None:
            return {"available": False}
        greeks_top = []
        for s, g in sorted(surface.sector_greeks.items(), key=lambda x: -x[1].iv)[:5]:
            greeks_top.append({"ticker": s, "iv": g.iv, "rv_60d": g.rv_60d, "vrp": g.vrp,
                              "theta": g.theta, "iv_rank": g.iv_rank_252d})
        return {
            "available": True,
            "vix": surface.vix_current,
            "vix_20d_avg": surface.vix_20d_avg,
            "term_slope": surface.term_slope,
            "implied_corr": surface.implied_corr,
            "dispersion_index": surface.dispersion_index,
            "vrp_index": surface.vrp_index,
            "vrp_sectors_avg": surface.vrp_sectors_avg,
            "top_iv_sectors": greeks_top,
        }

    def _regime_safety_section(self, rs) -> dict:
        if rs is None:
            return {"available": False}
        return {
            "available": True,
            "score": round(rs.regime_safety_score, 4),
            "label": rs.label,
            "market_state": rs.market_state,
            "size_cap": round(rs.size_cap, 4),
            "penalties": {
                "vix": round(rs.vix_penalty, 4),
                "credit": round(rs.credit_penalty, 4),
                "correlation": round(rs.corr_penalty, 4),
                "transition": round(rs.transition_penalty, 4),
            },
            "hard_kills": {k: v for k, v in rs.hard_kills.items() if v},
            "any_hard_kill": rs.any_hard_kill,
            "alerts": list(rs.alerts),
        }

    def _correlation_section(self, cs) -> dict:
        if cs is None:
            return {"available": False}
        return {
            "available": True,
            "frob_distortion_z": round(_sf(cs.frob_distortion_z), 4),
            "market_mode_share": round(_sf(cs.market_mode_share), 4),
            "coc_instability_z": round(_sf(cs.coc_instability_z), 4),
            "avg_corr_current": round(_sf(getattr(cs, "avg_corr_short", getattr(cs, "avg_corr_current", 0))), 4),
        }

    def _signal_stack_section(self, results) -> dict:
        if not results:
            return {"available": False, "candidates": []}

        passing = [r for r in results if r.passes_entry]
        return {
            "available": True,
            "n_total": len(results),
            "n_passing": len(passing),
            "distortion_score": results[0].distortion_score,
            "candidates": [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "conviction": r.conviction_score,
                    "residual_z": r.residual_z,
                    "distortion": r.distortion_score,
                    "dislocation": r.dislocation_score,
                    "mean_reversion": r.mean_reversion_score,
                    "regime_safety": r.regime_safety_score,
                    "passes_entry": r.passes_entry,
                }
                for r in results[:10]
            ],
        }

    def _trade_book_section(self, tickets) -> dict:
        if not tickets:
            return {"available": False, "trades": []}

        active = [t for t in tickets if t.is_active]
        gross = sum(t.final_weight for t in active)
        net_delta = sum(t.greeks.delta_spy for t in active)
        net_vega = sum(t.greeks.vega_net for t in active)

        return {
            "available": True,
            "n_total": len(tickets),
            "n_active": len(active),
            "gross_exposure": round(gross, 4),
            "net_delta_spy": round(net_delta, 6),
            "net_vega": round(net_vega, 6),
            "trades": [
                {
                    "id": t.trade_id,
                    "type": t.trade_type,
                    "ticker": t.ticker,
                    "direction": t.direction,
                    "weight": round(t.final_weight, 4),
                    "conviction": round(t.conviction_score, 3),
                    "entry_z": round(t.entry_z, 2),
                    "n_legs": t.n_legs,
                    "legs": [l.description for l in t.legs],
                    "exit": t.exit_conditions.description,
                    "pm_note": t.pm_note,
                }
                for t in active[:8]
            ],
        }

    def _monitor_section(self, ms) -> dict:
        if ms is None:
            return {"available": False}
        return {
            "available": True,
            "n_trades": ms.n_trades,
            "n_healthy": ms.n_healthy,
            "n_aging": ms.n_aging,
            "n_at_risk": ms.n_at_risk,
            "n_critical": ms.n_critical,
            "n_exit_signals": ms.n_exit_signals,
            "avg_health": round(ms.avg_health, 4),
            "avg_z_compression": round(ms.avg_z_compression, 4),
            "total_pnl_proxy": round(ms.total_pnl_proxy, 6),
            "urgent_exits": [
                {
                    "ticker": r.ticker,
                    "action": r.recommended_action,
                    "health": r.health_label,
                    "signal": r.primary_signal.signal_type if r.primary_signal else "N/A",
                    "reason": r.primary_signal.reason if r.primary_signal else "",
                    "urgency": r.primary_signal.urgency if r.primary_signal else "",
                }
                for r in ms.urgent_exits[:5]
            ],
            "pm_summary": ms.pm_summary,
        }

    def _action_items(self, signal_results, trade_tickets, regime_safety, monitor_summary) -> list:
        """Generate prioritized action items for the PM."""
        items = []

        # Priority 1: Regime kills
        if regime_safety and regime_safety.any_hard_kill:
            items.append({
                "priority": 1,
                "category": "REGIME",
                "action": "FLATTEN ALL POSITIONS — hard kill triggered",
                "urgency": "IMMEDIATE",
                "detail": f"Safety={regime_safety.regime_safety_score:.0%}, kills: {list(k for k,v in regime_safety.hard_kills.items() if v)}",
            })

        # Priority 2: Urgent exits
        if monitor_summary and monitor_summary.urgent_exits:
            for r in monitor_summary.urgent_exits[:3]:
                ps = r.primary_signal
                items.append({
                    "priority": 2,
                    "category": "EXIT",
                    "action": f"{r.recommended_action} {r.ticker}",
                    "urgency": ps.urgency if ps else "END_OF_DAY",
                    "detail": ps.reason if ps else "",
                })

        # Priority 3: New entries
        if signal_results:
            new_entries = [r for r in signal_results if r.passes_entry]
            # Only flag entries that don't already have tickets
            existing_tickers = set()
            if trade_tickets:
                existing_tickers = {t.ticker for t in trade_tickets if t.is_active}
            new_only = [r for r in new_entries if r.ticker not in existing_tickers]
            for r in new_only[:3]:
                items.append({
                    "priority": 3,
                    "category": "ENTRY",
                    "action": f"EVALUATE {r.direction} {r.ticker} (conv={r.conviction_score:.2f})",
                    "urgency": "NEXT_SESSION",
                    "detail": f"Z={r.residual_z:+.2f}, MR={r.mean_reversion_score:.2f}",
                })

        # Priority 4: Aging trades
        if monitor_summary:
            aging = [r for r in monitor_summary.trade_reports
                     if r.health_label in ("AGING", "AT_RISK") and not r.should_exit]
            for r in aging[:2]:
                items.append({
                    "priority": 4,
                    "category": "MONITOR",
                    "action": f"REVIEW {r.ticker} — {r.health_label}",
                    "urgency": "MONITOR",
                    "detail": f"Day {r.days_held}/{r.max_holding_days}, z compress {r.z_compression_pct:.0%}",
                })

        # Priority 5: Regime warnings
        if regime_safety and regime_safety.label in ("CAUTION", "DANGER"):
            items.append({
                "priority": 5,
                "category": "REGIME",
                "action": f"REDUCE GROSS — regime {regime_safety.label}",
                "urgency": "END_OF_DAY",
                "detail": f"Safety={regime_safety.regime_safety_score:.0%}, cap={regime_safety.size_cap:.0%}",
            })

        items.sort(key=lambda x: x["priority"])
        return items

    # ─────────────────────────────────────────────────────────────────────
    # Text rendering
    # ─────────────────────────────────────────────────────────────────────

    def _render_txt(self, payload: dict, run_date: date) -> str:
        L: list = []
        ds = run_date.strftime("%Y-%m-%d")

        L.append(f"{'='*70}")
        L.append(f"  SRV DSS — SHORT VOL / DISPERSION TRADE BRIEF — {ds}")
        L.append(f"{'='*70}")
        L.append("")

        # ── 1. Regime Safety ─────────────────────────────────
        rs = payload["regime_safety"]
        if rs.get("available"):
            L.append(f"[1] REGIME SAFETY: {rs['label']} ({_pct(rs['score'])})")
            L.append(f"    Market State: {rs['market_state']} | Size Cap: {_pct(rs['size_cap'])}")
            L.append(f"    Penalties: VIX={_pct(rs['penalties']['vix'])} "
                     f"Credit={_pct(rs['penalties']['credit'])} "
                     f"Corr={_pct(rs['penalties']['correlation'])} "
                     f"Trans={_pct(rs['penalties']['transition'])}")
            if rs["any_hard_kill"]:
                kills = ", ".join(rs["hard_kills"].keys())
                L.append(f"    *** HARD KILL ACTIVE: {kills} ***")
            for a in rs.get("alerts", []):
                L.append(f"    ! {a}")
        else:
            L.append("[1] REGIME SAFETY: not available")
        L.append("")

        # ── 2. Correlation Structure ─────────────────────────
        corr = payload["correlation"]
        if corr.get("available"):
            L.append(f"[2] CORRELATION DISTORTION:")
            L.append(f"    Frobenius Z: {corr['frob_distortion_z']:+.2f} "
                     f"| Market Mode: {_pct(corr['market_mode_share'])} "
                     f"| Avg Corr: {corr['avg_corr_current']:.3f} "
                     f"| CoC Z: {corr['coc_instability_z']:+.2f}")
        else:
            L.append("[2] CORRELATION DISTORTION: not available")
        L.append("")

        # ── 2b. Options Analytics ────────────────────────────
        opts = payload.get("options", {})
        if opts.get("available"):
            ts_label = "contango" if opts["term_slope"] > 0 else "backwardation"
            L.append(f"[2b] OPTIONS ANALYTICS:")
            L.append(f"    VIX: {opts['vix']:.1f} (20d avg: {opts['vix_20d_avg']:.1f}, "
                     f"term: {opts['term_slope']:+.2f}pp {ts_label})")
            L.append(f"    Implied Corr: {opts['implied_corr']:.3f} | "
                     f"Dispersion: {opts['dispersion_index']:.2f}% | "
                     f"VRP(index): {opts['vrp_index']:+.4f}")
            if opts.get("top_iv_sectors"):
                L.append(f"    Top IV: " + ", ".join(
                    f"{g['ticker']}={g['iv']:.0%}" for g in opts["top_iv_sectors"][:3]
                ))
        L.append("")

        # ── 3. Signal Stack ──────────────────────────────────
        ss = payload["signal_stack"]
        if ss.get("available"):
            L.append(f"[3] SIGNAL STACK: {ss['n_passing']}/{ss['n_total']} passing entry "
                     f"(distortion={_pct(ss['distortion_score'])})")
            L.append(f"    {'Ticker':<6} {'Dir':<6} {'Z':>6} {'Conv':>6} "
                     f"{'Dist':>6} {'Disloc':>6} {'MR':>6} {'Safe':>6} {'Entry':>5}")
            L.append(f"    {'-'*60}")
            for c in ss["candidates"][:8]:
                entry_mark = " YES" if c["passes_entry"] else "  no"
                L.append(
                    f"    {c['ticker']:<6} {c['direction']:<6} "
                    f"{c['residual_z']:>+5.2f} {c['conviction']:.3f} "
                    f"{c['distortion']:.3f} {c['dislocation']:.3f} "
                    f"{c['mean_reversion']:.3f} {c['regime_safety']:.3f} "
                    f"{entry_mark}"
                )
        else:
            L.append("[3] SIGNAL STACK: no candidates")
        L.append("")

        # ── 4. Trade Book ────────────────────────────────────
        tb = payload["trade_book"]
        if tb.get("available") and tb["n_active"] > 0:
            L.append(f"[4] TRADE BOOK: {tb['n_active']} active | "
                     f"Gross={_pct(tb['gross_exposure'])} "
                     f"| Δ SPY={tb['net_delta_spy']:+.4f} "
                     f"| Vega={tb['net_vega']:+.4f}")
            for t in tb["trades"][:6]:
                L.append(f"    {t['id']}: {t['direction']} {t['ticker']} "
                         f"w={_pct(t['weight'])} conv={t['conviction']:.2f} "
                         f"z={t['entry_z']:+.2f}")
                for leg in t["legs"]:
                    L.append(f"      • {leg}")
                L.append(f"      {t['exit']}")
        else:
            L.append("[4] TRADE BOOK: no active trades")
        L.append("")

        # ── 5. Trade Monitor ─────────────────────────────────
        mon = payload["monitor"]
        if mon.get("available"):
            L.append(f"[5] TRADE MONITOR: {mon['n_trades']} trades | "
                     f"Health: {mon['n_healthy']} healthy, "
                     f"{mon['n_aging']} aging, "
                     f"{mon['n_at_risk']} at-risk, "
                     f"{mon['n_critical']} critical")
            L.append(f"    Avg Health: {_pct(mon['avg_health'])} "
                     f"| Avg Z-Compression: {_pct(mon['avg_z_compression'])} "
                     f"| Exit Signals: {mon['n_exit_signals']}")
            if mon["urgent_exits"]:
                L.append("    URGENT EXITS:")
                for ue in mon["urgent_exits"]:
                    L.append(f"      ⚠ {ue['ticker']}: {ue['action']} "
                             f"({ue['signal']}, {ue['urgency']}) — {ue['reason'][:60]}")
        else:
            L.append("[5] TRADE MONITOR: not available")
        L.append("")

        # ── 6. Action Items ──────────────────────────────────
        actions = payload["action_items"]
        if actions:
            L.append(f"[6] ACTION ITEMS ({len(actions)}):")
            for a in actions:
                L.append(f"    P{a['priority']} [{a['category']:<7}] [{a['urgency']:<12}] {a['action']}")
                if a.get("detail"):
                    L.append(f"       → {a['detail']}")
        else:
            L.append("[6] ACTION ITEMS: none — all clear")

        # ── [7] P&L ATTRIBUTION (if available) ─────────────────────────
        if kwargs.get("pnl_attribution"):
            attr = kwargs["pnl_attribution"]
            L.append("")
            L.append("[7] P&L ATTRIBUTION")
            L.append(f"    {'Factor':<25} {'Contribution':>12}")
            L.append(f"    {'-'*40}")
            for factor, value in attr.items():
                sign = "+" if value >= 0 else ""
                L.append(f"    {factor:<25} {sign}{value*100:.2f}%")

        # ── [8] HEDGE RECOMMENDATIONS ────────────────────────────────────
        if kwargs.get("hedge_recommendations"):
            hedges = kwargs["hedge_recommendations"]
            L.append("")
            L.append("[8] HEDGE RECOMMENDATIONS")
            for h in hedges[:5]:
                L.append(f"    • {h}")

        # ── [9] MOMENTUM RANKING ─────────────────────────────────────────
        if kwargs.get("momentum_ranking"):
            ranking = kwargs["momentum_ranking"]
            L.append("")
            L.append("[9] MOMENTUM RANKING (21d vs SPY)")
            L.append(f"    {'Rank':<6} {'Sector':<8} {'Mom 21d':>10} {'Signal':>10}")
            L.append(f"    {'-'*40}")
            for i, item in enumerate(ranking[:11]):
                signal = "LONG" if i < 3 else ("SHORT" if i >= len(ranking) - 3 else "—")
                L.append(f"    #{i+1:<4} {item['ticker']:<8} {item['momentum_21d']:>+9.2%}  {signal:>8}")

        # ── [10] SHORT-VOL TIMING ────────────────────────────────────────
        if kwargs.get("options_surface"):
            os_ = kwargs["options_surface"]
            sv_score = getattr(os_, "short_vol_timing_score", 0)
            sv_label = getattr(os_, "short_vol_timing_label", "")
            vvix = getattr(os_, "vvix_current", 0)
            skew = getattr(os_, "skew_current", 100)
            if sv_score > 0:
                L.append("")
                L.append("[10] SHORT-VOL TIMING")
                L.append(f"    Score:  {sv_score:.0f}/100 → {sv_label}")
                L.append(f"    VVIX:   {vvix:.1f} ({getattr(os_, 'vvix_signal', '')})")
                L.append(f"    Skew:   {skew:.0f} ({getattr(os_, 'skew_signal', '')})")
                L.append(f"    VRP:    {getattr(os_, 'vrp_index', 0):+.4f}")
                if sv_score >= 65:
                    L.append(f"    ✅ FAVORABLE for short vol entry")
                elif sv_score <= 35:
                    L.append(f"    ⚠️ AVOID short vol — conditions unfavorable")

        L.append("")
        L.append(f"{'='*70}")
        L.append(f"  End of DSS Brief — {ds}")
        L.append(f"{'='*70}")
        L.append("")
        return "\n".join(L)
