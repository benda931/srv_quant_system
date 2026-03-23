"""
reports/daily_report.py

DailyBriefGenerator — produces a human-readable TXT brief and a
machine-readable JSON snapshot of the SRV DSS state as of the current
run date.

Outputs (both are idempotent — re-running on the same date overwrites)
-----------------------------------------------------------------------
  reports/output/YYYY-MM-DD_brief.txt
  reports/output/YYYY-MM-DD_brief.json

No external dependencies beyond stdlib + pandas.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import Settings

if TYPE_CHECKING:
    from analytics.portfolio_risk import RiskReport
    from analytics.stress import StressResult
    from data_ops.journal import PMJournal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_TOP_N: int = 5                  # max opportunities shown per direction
_STRESS_THRESHOLD: float = -0.005  # show stress flags worse than -0.5 %
_CONV_CHANGE_MIN: int = 5        # min conviction delta to count as "changed"
_DATE_FMT: str = "%Y-%m-%d"
_PREV_SEARCH_DAYS: int = 5       # look back up to N calendar days for prior brief


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sf(x: Any, default: float = float("nan")) -> float:
    """Safe float coercion; returns ``default`` for None / NaN / inf."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _fmt_score(x: float) -> str:
    """Format a 0-1 score as an integer string, e.g. 0.32 → '32'."""
    if not math.isfinite(x):
        return "--"
    return str(int(round(x * 100)))


def _fmt_pct(x: Optional[float], decimals: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return "N/A"
    return f"{x:.{decimals}f}%"


def _first_valid(df: pd.DataFrame, col: str, default: Any = None) -> Any:
    """Return the first non-null value in ``df[col]``, or ``default``."""
    if col not in df.columns:
        return default
    vals = df[col].dropna()
    return default if vals.empty else vals.iloc[0]


def _vix_regime_label(vix: float, settings: Settings) -> str:
    if not math.isfinite(vix):
        return "UNKNOWN"
    if vix >= settings.vix_level_hard:
        return "CRISIS"
    if vix >= settings.vix_level_soft:
        return "ELEVATED"
    return "CALM"


def _prev_json_path(output_dir: Path, today: date) -> Optional[Path]:
    """Walk back up to ``_PREV_SEARCH_DAYS`` days to find the most recent brief JSON."""
    for delta in range(1, _PREV_SEARCH_DAYS + 1):
        p = output_dir / f"{(today - timedelta(days=delta)).strftime(_DATE_FMT)}_brief.json"
        if p.exists():
            return p
    return None


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars / NaN to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    # numpy scalar → Python scalar
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 8)
    return obj


# ---------------------------------------------------------------------------
# DailyBriefGenerator
# ---------------------------------------------------------------------------

class DailyBriefGenerator:
    """
    Assembles the daily PM brief from all live DSS data sources.

    Usage
    -----
    ::

        generator = DailyBriefGenerator()
        txt_path, json_path = generator.generate(
            master_df, risk_report, stress_results, journal, settings, output_dir
        )

    The caller is responsible for supplying pre-computed inputs.  This class
    performs no analytics — it only formats and persists.
    """

    def generate(
        self,
        master_df: pd.DataFrame,
        risk_report: "RiskReport",
        stress_results: List["StressResult"],
        journal: Optional["PMJournal"],
        settings: Settings,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Generate the daily brief TXT and JSON files.

        Parameters
        ----------
        master_df : pd.DataFrame
            Current output of ``QuantEngine.calculate_conviction_score()``.
            Expected columns: sector_ticker, direction, conviction_score,
            sds_score, fjs_score, mss_score, stf_score, mc_score_raw,
            market_state, vix_level, avg_corr_t, market_mode_strength,
            credit_z, decision_label, pm_note, action_bias, w_final.
        risk_report : RiskReport
            Output of ``PortfolioRiskEngine.full_risk_report()``.
        stress_results : List[StressResult]
            Output of ``StressEngine.run_all()``, sorted worst → best.
        journal : PMJournal or None
            PM decision journal; pass ``None`` to omit the PM section.
        settings : Settings
            Validated settings instance.
        output_dir : Path, optional
            Directory for output files.  Defaults to
            ``<project_root>/reports/output/``.

        Returns
        -------
        tuple[Path, Path]
            ``(txt_path, json_path)``
        """
        run_date = self._infer_date(master_df)
        output_dir = self._resolve_output_dir(output_dir, settings)

        # Normalise master_df so sector_ticker is always a column
        df = self._normalise_df(master_df)

        # --- build structured payload ------------------------------------------
        regime       = self._extract_regime(df, settings)
        top_longs, top_shorts = self._top_opportunities(df)
        risk         = self._risk_section(risk_report, df) if risk_report is not None else {}
        stress_flags = self._stress_flags(stress_results)
        signal_chgs  = self._signal_changes(df, run_date, output_dir)
        pm_stats     = self._pm_stats(journal)

        payload: Dict[str, Any] = {
            "date":           run_date.strftime(_DATE_FMT),
            "regime":         regime,
            "top_longs":      top_longs,
            "top_shorts":     top_shorts,
            "risk":           risk,
            "stress_flags":   stress_flags,
            "signal_changes": signal_chgs,
            "pm_stats":       pm_stats,
        }

        # --- render & write ----------------------------------------------------
        txt_content  = self._render_txt(payload, run_date)
        json_content = json.dumps(_clean_for_json(payload), ensure_ascii=False, indent=2)

        date_str  = run_date.strftime(_DATE_FMT)
        txt_path  = output_dir / f"{date_str}_brief.txt"
        json_path = output_dir / f"{date_str}_brief.json"

        txt_path.write_text(txt_content, encoding="utf-8")
        json_path.write_text(json_content, encoding="utf-8")

        logger.info(
            "Daily brief written: %s | %s  [%d opps | %d stress flags | %d changes]",
            txt_path.name, json_path.name,
            len(top_longs) + len(top_shorts),
            len(stress_flags),
            len(signal_chgs),
        )
        return txt_path, json_path

    # ------------------------------------------------------------------
    # Internals: data extraction
    # ------------------------------------------------------------------

    def _infer_date(self, master_df: pd.DataFrame) -> date:
        """Use the DatetimeIndex tail date when available; fall back to today."""
        try:
            if isinstance(master_df.index, pd.DatetimeIndex) and not master_df.index.empty:
                return pd.Timestamp(master_df.index[-1]).date()
        except Exception:
            pass
        return date.today()

    def _resolve_output_dir(self, output_dir: Optional[Path], settings: Settings) -> Path:
        if output_dir is not None:
            d = Path(output_dir)
        else:
            d = settings.project_root / "reports" / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _normalise_df(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure sector_ticker is a regular column (not only in the index)."""
        df = master_df.copy()
        if "sector_ticker" not in df.columns:
            if df.index.name == "sector_ticker":
                df = df.reset_index()
            elif isinstance(df.index, pd.MultiIndex) and "sector_ticker" in df.index.names:
                df = df.reset_index(level="sector_ticker")
        return df

    def _extract_regime(self, df: pd.DataFrame, settings: Settings) -> Dict[str, Any]:
        market_state = str(_first_valid(df, "market_state", "UNKNOWN")).upper()
        avg_corr     = _sf(_first_valid(df, "avg_corr_t",         float("nan")))
        mode_str     = _sf(_first_valid(df, "market_mode_strength", float("nan")))
        vix          = _sf(_first_valid(df, "vix_level",           float("nan")))
        credit_z     = _sf(_first_valid(df, "credit_z",            float("nan")))

        return {
            "label":         market_state,
            "avg_corr":      round(avg_corr, 4)   if math.isfinite(avg_corr)  else None,
            "mode_strength": round(mode_str, 4)   if math.isfinite(mode_str)  else None,
            "vix":           round(vix, 2)         if math.isfinite(vix)       else None,
            "vix_regime":    _vix_regime_label(vix, settings),
            "credit_z":      round(credit_z, 3)   if math.isfinite(credit_z)  else None,
        }

    def _top_opportunities(
        self, df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return top-N LONG and top-N SHORT opportunities sorted by conviction."""
        if df.empty or "direction" not in df.columns:
            return [], []

        def _to_dict(row: pd.Series) -> Dict[str, Any]:
            note = str(row.get("pm_note", "") or "").strip()
            if not note:
                note = str(row.get("action_bias", "") or "").strip()
            return {
                "ticker":         str(row.get("sector_ticker", "?")),
                "conviction":     int(round(_sf(row.get("conviction_score", 0), 0))),
                "direction":      str(row.get("direction", "NEUTRAL")).upper(),
                "sds":            round(_sf(row.get("sds_score",   0), 0), 3),
                "fjs":            round(_sf(row.get("fjs_score",   0), 0), 3),
                "mss":            round(_sf(row.get("mss_score",   0), 0), 3),
                "stf":            round(_sf(row.get("stf_score",   0), 0), 3),
                "mc":             round(_sf(row.get("mc_score_raw", 0), 0), 3),
                "decision_label": str(row.get("decision_label", "WATCH")).upper(),
                "action_bias":    str(row.get("action_bias",    "")).upper(),
                "weight":         round(_sf(row.get("w_final", 0), 0), 4),
                "pm_note":        note[:120],
            }

        sort_col = "conviction_score" if "conviction_score" in df.columns else None

        longs  = df[df["direction"] == "LONG"].copy()
        shorts = df[df["direction"] == "SHORT"].copy()
        if sort_col:
            longs  = longs.sort_values(sort_col, ascending=False)
            shorts = shorts.sort_values(sort_col, ascending=False)

        return (
            [_to_dict(r) for _, r in longs.head(_TOP_N).iterrows()],
            [_to_dict(r) for _, r in shorts.head(_TOP_N).iterrows()],
        )

    def _risk_section(
        self, risk_report: "RiskReport", df: pd.DataFrame
    ) -> Dict[str, Any]:
        # Identify the highest-weighted sector
        max_ticker = self._max_weight_ticker(risk_report, df)
        max_wt     = _sf(risk_report.max_sector_weight, float("nan"))

        return {
            "portfolio_vol_ann_pct": round(risk_report.portfolio_vol_ann * 100, 2),
            "var_95_1d_pct":         round(-risk_report.var_95_1d * 100, 3),
            "cvar_95_1d_pct":        (
                round(-risk_report.cvar_95_1d * 100, 3)
                if math.isfinite(risk_report.cvar_95_1d) else None
            ),
            "hhi":                   round(risk_report.concentration_hhi, 4),
            "max_sector":            max_ticker,
            "max_sector_weight_pct": round(max_wt * 100, 2) if math.isfinite(max_wt) else None,
            "vol_target_breach":     bool(risk_report.vol_target_breach),
            "max_weight_breach":     bool(risk_report.max_weight_breach),
        }

    def _max_weight_ticker(self, risk_report: "RiskReport", df: pd.DataFrame) -> str:
        """Best-effort identification of the largest-weight sector."""
        # 1. Try the risk_budget_series index (highest risk contributor)
        try:
            rb = risk_report.risk_budget_series
            if rb is not None and not rb.empty:
                return str(rb.idxmax())
        except Exception:
            pass
        # 2. Fall back to master_df w_final
        try:
            if "sector_ticker" in df.columns and "w_final" in df.columns:
                idx = df["w_final"].abs().idxmax()
                return str(df.loc[idx, "sector_ticker"])
        except Exception:
            pass
        return "?"

    def _stress_flags(
        self, stress_results: List["StressResult"]
    ) -> List[Dict[str, Any]]:
        """Return all scenarios with PnL below the threshold, in received order."""
        flags = []
        for r in stress_results:
            pnl = _sf(r.portfolio_pnl_estimate, 0.0)
            if pnl < _STRESS_THRESHOLD:
                flags.append({
                    "scenario":    r.scenario_name,
                    "pnl_pct":     round(pnl * 100, 2),
                    "worst_sector": r.worst_sector,
                    "best_sector":  r.best_sector,
                    "reliability":  round(_sf(r.signal_reliability_score, 0.0), 3),
                    "regime":       r.regime_label,
                })
        return flags   # StressEngine.run_all already sorts worst → best

    def _signal_changes(
        self,
        df: pd.DataFrame,
        run_date: date,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Compare today's per-sector direction and conviction with the prior brief.

        Returns a list of detected changes, sorted by |conviction_change| desc.
        """
        if "sector_ticker" not in df.columns or "direction" not in df.columns:
            return []

        prev_path = _prev_json_path(output_dir, run_date)
        if prev_path is None:
            logger.debug("No previous brief JSON found — signal change detection skipped.")
            return []

        try:
            prev_payload: Dict[str, Any] = json.loads(
                prev_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning("Could not parse previous brief JSON (%s): %s", prev_path.name, exc)
            return []

        # Previous state: ticker → {direction, conviction}
        prev_state: Dict[str, Dict[str, Any]] = {}
        for entry in prev_payload.get("top_longs", []) + prev_payload.get("top_shorts", []):
            t = str(entry.get("ticker", ""))
            if t:
                prev_state[t] = {
                    "direction":  str(entry.get("direction", "NEUTRAL")).upper(),
                    "conviction": _sf(entry.get("conviction", 0), 0.0),
                }

        changes: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            ticker    = str(row.get("sector_ticker", ""))
            cur_dir   = str(row.get("direction", "NEUTRAL")).upper()
            cur_conv  = _sf(row.get("conviction_score", 0), 0.0)
            prev      = prev_state.get(ticker, {"direction": "NEUTRAL", "conviction": 0.0})
            prev_dir  = str(prev["direction"])
            prev_conv = _sf(prev["conviction"], 0.0)
            delta     = cur_conv - prev_conv

            if cur_dir != prev_dir or abs(delta) >= _CONV_CHANGE_MIN:
                changes.append({
                    "ticker":            ticker,
                    "from_direction":    prev_dir,
                    "to_direction":      cur_dir,
                    "conviction_before": int(round(prev_conv)),
                    "conviction_after":  int(round(cur_conv)),
                    "conviction_change": int(round(delta)),
                })

        changes.sort(key=lambda c: abs(c["conviction_change"]), reverse=True)
        return changes

    def _pm_stats(self, journal: Optional["PMJournal"]) -> Dict[str, Any]:
        """Extract PM override accuracy stats from the journal."""
        if journal is None:
            return {"available": False}
        try:
            acc   = journal.get_override_accuracy()
            stats = journal.get_stats()
            pm_acc = acc.get("pm_accuracy")
            return {
                "available":      True,
                "n_decisions":    int(stats.get("n_decisions", 0)),
                "n_overrides":    int(acc.get("n_disagreements", 0)),
                "n_resolved":     int(acc.get("n_resolved", 0)),
                "n_pm_correct":   int(acc.get("n_pm_correct", 0)),
                "n_model_correct": int(acc.get("n_model_correct", 0)),
                "pm_accuracy":    round(pm_acc, 4) if pm_acc is not None else None,
                "model_accuracy": (
                    round(acc["model_accuracy"], 4)
                    if acc.get("model_accuracy") is not None else None
                ),
            }
        except Exception as exc:
            logger.warning("Could not read PM journal stats: %s", exc)
            return {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals: text rendering
    # ------------------------------------------------------------------

    def _render_txt(self, payload: Dict[str, Any], run_date: date) -> str:
        lines: List[str] = []
        date_str = run_date.strftime(_DATE_FMT)

        # ── Header ──────────────────────────────────────────────────────
        lines.append(f"=== SRV QUANTAMENTAL DSS — DAILY BRIEF {date_str} ===")

        # ── Regime ──────────────────────────────────────────────────────
        regime   = payload["regime"]
        avg_corr = regime.get("avg_corr")
        mode_str = regime.get("mode_strength")
        lines.append(
            f"REGIME: {regime['label']}"
            f" | AVG CORR: {f'{avg_corr:.2f}' if avg_corr is not None else 'N/A'}"
            f" | MODE STRENGTH: {f'{mode_str:.2f}' if mode_str is not None else 'N/A'}"
        )
        vix = regime.get("vix")
        lines.append(
            f"VIX: {f'{vix:.1f}' if vix is not None else 'N/A'} ({regime['vix_regime']})"
        )

        # ── Top opportunities ────────────────────────────────────────────
        lines.append("---")
        lines.append("TOP OPPORTUNITIES (by conviction):")
        all_opps = sorted(
            payload["top_longs"] + payload["top_shorts"],
            key=lambda x: x["conviction"],
            reverse=True,
        )
        if all_opps:
            for opp in all_opps:
                direction = opp["direction"]
                ticker    = opp["ticker"]
                conv      = opp["conviction"]
                sds_s     = _fmt_score(opp["sds"])
                fjs_s     = _fmt_score(opp["fjs"])
                mss_s     = _fmt_score(opp["mss"])
                mc_s      = _fmt_score(opp["mc"])
                note      = (opp.get("pm_note") or "")[:80].strip()
                lines.append(
                    f"  [{direction:<5}] {ticker:<4} — conviction {conv:>3}"
                    f" | SDS:{sds_s} FJS:{fjs_s} MSS:{mss_s} MC:{mc_s}"
                    + (f" | {note}" if note else "")
                )
        else:
            lines.append("  (no directional opportunities)")

        # ── Risk summary ─────────────────────────────────────────────────
        lines.append("---")
        lines.append("RISK SUMMARY:")
        risk = payload["risk"]
        vol  = risk.get("portfolio_vol_ann_pct")
        var_ = risk.get("var_95_1d_pct")
        cvar = risk.get("cvar_95_1d_pct")
        msec = risk.get("max_sector", "?")
        mwt  = risk.get("max_sector_weight_pct")
        lines.append(f"  Portfolio Vol (ann): {_fmt_pct(vol)}")
        lines.append(f"  1D VaR 95%:  {_fmt_pct(var_, decimals=2)}")
        lines.append(f"  1D CVaR 95%: {_fmt_pct(cvar, decimals=2)}")
        lines.append(f"  Max Sector: {msec} {_fmt_pct(mwt)}")
        if risk.get("vol_target_breach"):
            lines.append("  *** VOL TARGET BREACH ***")
        if risk.get("max_weight_breach"):
            lines.append("  *** MAX WEIGHT BREACH ***")

        # ── Stress flags ─────────────────────────────────────────────────
        lines.append("---")
        sf = payload["stress_flags"]
        if sf:
            parts = [f"[{f['scenario']}: {f['pnl_pct']:.1f}%]" for f in sf]
            lines.append(f"STRESS FLAGS: {' '.join(parts)}")
        else:
            lines.append("STRESS FLAGS: none above threshold")

        # ── Signal changes ───────────────────────────────────────────────
        lines.append("---")
        chgs = payload["signal_changes"]
        if chgs:
            lines.append("SIGNALS CHANGED VS YESTERDAY:")
            for ch in chgs[:10]:
                delta = ch["conviction_change"]
                sign  = "+" if delta >= 0 else ""
                lines.append(
                    f"  {ch['ticker']}: {ch['from_direction']} → {ch['to_direction']}"
                    f" (conviction {sign}{delta})"
                )
        else:
            lines.append("SIGNALS CHANGED VS YESTERDAY: none detected")

        # ── PM overrides ─────────────────────────────────────────────────
        lines.append("---")
        pm = payload["pm_stats"]
        if pm.get("available"):
            n_total = pm.get("n_overrides", 0)
            n_res   = pm.get("n_resolved", 0)
            acc     = pm.get("pm_accuracy")
            acc_str = f"{acc * 100:.0f}%" if acc is not None else "N/A"
            lines.append(
                f"PM OVERRIDES (last 7d): {n_total} total, {n_res} resolved,"
                f" accuracy {acc_str}"
            )
        else:
            lines.append("PM OVERRIDES: journal not available")

        lines.append("")  # trailing newline
        return "\n".join(lines)
