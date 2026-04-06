from __future__ import annotations

# ── DuckDB must be imported before pandas/pyarrow to avoid Python 3.13 GC
# conflict: both register Arrow allocator destructors; DuckDB must register
# FIRST so it is destroyed LAST at interpreter shutdown (LIFO atexit order).
import duckdb as _duckdb_init  # noqa: F401

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.stat_arb import QuantEngine
from config.settings import get_settings
from data.pipeline import DataLakeManager
from data_ops.orchestrator import DataOrchestrator
from data_ops.health_panel import build_health_tab
from data_ops.status_report import DataHealthReport
from ui.panels import (
    bar_figure,
    build_action_plan,
    build_correlation_panel,
    build_correlation_summary,
    build_market_narrative,
    build_opportunities_section,
    build_overview_panel,
    build_regime_hero,
    build_scanner_panel,
    build_scanner_table,
    build_scatter,
    build_stat_analysis_panel,
    build_tearsheet_panel,
    format_float,
    heatmap_fig,
    kpi_card,
    line_figure,
)
from ui.scanner_pro import build_scanner_pro
from ui.journal_panel import build_journal_tab
from ui.analytics_tabs import (
    build_stress_tab, build_risk_tab, build_backtest_tab,
    build_daily_brief_panel, build_corr_vol_tab,
    build_signal_decay_tab, build_regime_timeline_tab,
    build_pnl_tracker_tab, build_dss_tab,
    build_portfolio_tab, build_methodology_tab,
    build_ml_insights_tab,
    build_agent_monitor_tab,
    build_optimization_tab,
)
from data_ops.journal import PMJournal, open_journal


# ==========================================================
# UI styles
# ==========================================================
RTL_STYLE = {
    "direction": "rtl",
    "textAlign": "right",
}

CARD_STYLE = {
    "direction": "rtl",
    "textAlign": "right",
    "height": "100%",
}

TEXT_MUTED_STYLE = {
    "direction": "rtl",
    "textAlign": "right",
    "color": "#A9A9A9",
}

APP_CONTAINER_STYLE = {
    "direction": "rtl",
    "textAlign": "right",
    "paddingBottom": "24px",
}


# ==========================================================
# Logging / primitive helpers
# ==========================================================
def configure_logging(log_dir: Path) -> None:
    from logging.handlers import RotatingFileHandler
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "srv_system.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
            ),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "—") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def coerce_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    try:
        return bool(x)
    except Exception:
        return default


# ==========================================================
# Reusable UI blocks
# ==========================================================
def make_info_alert(title: str, body: str, color: str = "secondary") -> dbc.Alert:
    return dbc.Alert(
        [
            html.Div(title, className="fw-bold mb-1", style=RTL_STYLE),
            html.Div(body, style=RTL_STYLE),
        ],
        color=color,
        className="mb-2",
        style=RTL_STYLE,
    )


def make_section_intro(title: str, body: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="mb-2", style=RTL_STYLE),
                html.Div(body, style=TEXT_MUTED_STYLE),
            ]
        ),
        className="mb-3",
        style=CARD_STYLE,
    )


def build_methodology_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("הסבר מתודולוגי על המערכת", style=RTL_STYLE),
                close_button=True,
            ),
            dbc.ModalBody(
                [
                    html.Div(
                        "המערכת היא מערכת DSS כמותית־פונדמנטלית לספר Sector Relative Value. "
                        "היא אינה מבצעת מסחר אוטומטי, אלא מבודדת עיוותים סטטיסטיים, בודקת האם הם מוסברים על ידי מאקרו, "
                        "פונדמנטלס, תנודתיות או שינוי מבני, ואז מייצרת הקשר החלטה, MC, משקולות book והמלצת execution regime.",
                        className="mb-3",
                        style=RTL_STYLE,
                    ),
                    html.H6("שכבות העבודה", style=RTL_STYLE),
                    html.Ul(
                        [
                            html.Li(
                                "שכבת מחירים ונתוני holdings — אוספת ETF sectors, SPY, מאקרו, קרדיט ונתוני החזקות.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Statistical Dislocation — מחשבת OOS PCA residuals, Z-score ו-half-life.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Macro / Credit / Vol — מודדת חשיפות ל-TNX, DXY, VIX ו-credit stress.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Correlation Structure — מודדת average correlation, market mode ו-distortion.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Regime Engine — מסווגת את מצב השוק ל-CALM / NORMAL / TENSION / CRISIS.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Fundamentals — מבצעת אגרגציית holdings-weighted valuation מול SPY.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Attribution — מפרידה בין mispricing, macro repricing ו-structural shift.",
                                style=RTL_STYLE,
                            ),
                            html.Li(
                                "שכבת Book Construction — בונה הצעת משקולות market-neutral עם MC scaling.",
                                style=RTL_STYLE,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.H6("פירוש MC", style=RTL_STYLE),
                    html.Div(
                        "MC = Mispricing Confidence. "
                        "זהו מדד שמנסה לאמוד עד כמה הסטייה נראית כמו mispricing נקי ולא כמו תמחור נכון של שינוי פונדמנטלי, מאקרו או מבנה שוק.",
                        className="mb-3",
                        style=RTL_STYLE,
                    ),
                    html.H6("פירוש Regime", style=RTL_STYLE),
                    html.Div(
                        "ה-Regime Engine מתרגם את מצב השוק לסביבת החלטה. "
                        "ב-CALM/NORMAL אפשר להיות פתוחים יותר ל-mean reversion. "
                        "ב-TENSION צריך להיות סלקטיביים וזהירים יותר. "
                        "ב-CRISIS המערכת צריכה להעדיף הגנה, צמצום או עצירה.",
                        style=RTL_STYLE,
                    ),
                ],
                style=RTL_STYLE,
            ),
            dbc.ModalFooter(
                dbc.Button("סגור", id="close-methodology-modal", color="secondary"),
                style=RTL_STYLE,
            ),
        ],
        id="methodology-modal",
        size="xl",
        is_open=False,
        scrollable=True,
        centered=True,
    )


def build_global_explanation_banner() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("מה המערכת עושה בפועל", className="fw-bold mb-2", style=RTL_STYLE),
                html.Div(
                    "המערכת מדרגת את 11 סקטורי ה-S&P 500 מול SPY, מחשבת סטיות סטטיסטיות Out-of-Sample, "
                    "בודקת האם התנועה מוסברת על ידי מאקרו, אשראי, תנודתיות, fundamentals או שינוי במבנה הקורלציות, "
                    "ואז מייצרת MC, context של regime, והצעת book weights לספר Relative Value דלתא־1.",
                    style=RTL_STYLE,
                ),
            ]
        ),
        className="mb-3",
        style=CARD_STYLE,
    )


def build_overview_explainer() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                make_info_alert(
                    "פיזור שוק (Dispersion)",
                    "מודד עד כמה הסקטורים זזים באופן שונה אחד מהשני. "
                    "כשפיזור גבוה, יש יותר קרקע ל-cross-sectional mean reversion. "
                    "כשפיזור נמוך, השוק יותר נשלט על ידי market mode רחב.",
                    color="info",
                ),
                md=4,
            ),
            dbc.Col(
                make_info_alert(
                    "Market Regime",
                    "זהו סיכום משטר השוק הכולל. "
                    "CALM/NORMAL תומכים יותר ב-mean reversion. "
                    "TENSION אומר שהשוק עובר חוסר יציבות ולכן צריך להיות סלקטיבי יותר. "
                    "CRISIS אומר להעדיף הגנה, צמצום או הימנעות.",
                    color="warning",
                ),
                md=4,
            ),
            dbc.Col(
                make_info_alert(
                    "Top Long / Short",
                    "אלו המועמדים המובילים לפי המנוע. "
                    "הם לא בהכרח הזולים או היקרים ביותר, אלא אלו עם יחס טוב יותר בין dislocation, MC והקשר regime נוכחי.",
                    color="primary",
                ),
                md=4,
            ),
        ],
        className="mb-2",
    )


def build_regime_explainer() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                make_info_alert(
                    "Market State",
                    "CALM / NORMAL / TENSION / CRISIS. "
                    "זהו סיכום משטר השוק הנוכחי על בסיס VIX, credit stress, correlation structure ו-transition risk.",
                    color="primary",
                ),
                md=4,
            ),
            dbc.Col(
                make_info_alert(
                    "Transition Risk",
                    "מודד האם השוק רק לא נוח, או עובר שבירה מבנית. "
                    "ציון גבוה כאן אומר שהסביבה פחות תומכת ב-mean reversion אגרסיבי.",
                    color="warning",
                ),
                md=4,
            ),
            dbc.Col(
                make_info_alert(
                    "Crisis Probability",
                    "אומדן תפעולי לסיכון שהשוק נכנס למצב שבו trades יחסיים נהיים פחות נקיים ויותר רגישים ל-market mode רחב.",
                    color="danger",
                ),
                md=4,
            ),
        ],
        className="mb-3",
    )


def build_book_explainer() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                make_info_alert(
                    "Synthetic Greeks",
                    "הגריקים כאן הם operational proxies לספר דלתא־1: "
                    "Δ_SPY, Δ_TNX, Δ_DXY מראים חשיפות בטא; "
                    "Γ_synth מודד כמה הספר פגיע להמשך האצה בעיוותים; "
                    "Vega_synth מודד רגישות לסביבת תנודתיות; "
                    "ρ_mode ו-ρ_dist מודדים תלות ב-market mode ובשינוי מבנה הקורלציות.",
                    color="secondary",
                ),
                md=6,
            ),
            dbc.Col(
                make_info_alert(
                    "Execution Regime",
                    "זהו סטטוס תפעולי של הספר: OK / CAUTION / REDUCE / STOP. "
                    "הוא מושפע מ-VIX, credit stress, correlation distortion, gamma intensity ו-MC ממוצע.",
                    color="secondary",
                ),
                md=6,
            ),
        ],
        className="mb-3",
    )


def build_scanner_explainer() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                make_info_alert(
                    "איך לקרוא את ה-Scanner",
                    "הטבלה מדרגת את כל הסקטורים במבט רוחבי. "
                    "היא משלבת סטייה סטטיסטית, מאקרו, fundamentals, correlation structure, regime ו-MC. "
                    "המטרה היא לאתר מועמדים מעניינים ל-long/short ולבחור את אלו שהאות בהם נראה פחות מוסבר ויותר reversible.",
                    color="dark",
                ),
                md=6,
            ),
            dbc.Col(
                make_info_alert(
                    "Decision Layer",
                    "מעבר ל-MC ול-interpretation, המנוע מסווג כל סקטור ל-ENTER / WATCH / REDUCE / AVOID, "
                    "ומוסיף size bucket, entry quality ו-PM note תפעולי.",
                    color="success",
                ),
                md=6,
            ),
            dbc.Col(
                make_info_alert(
                    "Interpretation / Action Bias",
                    "מעבר ל-MC, המנוע מסווג כל סקטור לפי interpretation, action bias ו-risk label. "
                    "כך ה-PM רואה לא רק מספרים אלא גם את ההיגיון: clean mispricing, macro repricing, structural shift או mixed signal.",
                    color="success",
                ),
                md=6,
            ),
        ],
        className="mb-3",
    )


def build_correlation_explainer() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                make_info_alert(
                    "Correlation Matrix",
                    "המטריצה הנוכחית מראה איך הסקטורים נעים אחד ביחס לשני כרגע. "
                    "מטריצה גבוהה ואחידה יותר מצביעה על שוק שנשלט על ידי market mode רחב.",
                    color="dark",
                ),
                md=3,
            ),
            dbc.Col(
                make_info_alert(
                    "Distortion ΔC",
                    "ΔC מראה איך מבנה הקורלציות השתנה מול baseline ארוך יותר. "
                    "שינוי חד כאן מרמז שהשוק לא רק זז, אלא משנה מבנה פנימי — וזה מסוכן לספרי mean reversion.",
                    color="warning",
                ),
                md=3,
            ),
            dbc.Col(
                make_info_alert(
                    "Market Mode Strength",
                    "מודד כמה התנועה של השוק נשלטת על ידי פקטור מרכזי אחד. "
                    "ככל שה-mode strength עולה, trades idiosyncratic פחות נקיים כי הכל נגרר אחרי תמה מרכזית.",
                    color="info",
                ),
                md=3,
            ),
            dbc.Col(
                make_info_alert(
                    "Sector Contribution",
                    "הגרף מראה איזה סקטורים תורמים יותר לשבירת מבנה הקורלציות. "
                    "כך ניתן להבין האם distortion הוא רחב או מונע על ידי pockets נקודתיים.",
                    color="primary",
                ),
                md=3,
            ),
        ],
        className="mb-3",
    )


def build_tearsheet_explainer() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("איך לקרוא את ה-Tear Sheet", className="fw-bold mb-2", style=RTL_STYLE),
                html.Div(
                    "זהו מסך עומק לסקטור בודד. "
                    "הוא משלב residual history, fundamentals, attribution, regime ומצב הספר. "
                    "המטרה היא להבין אם האות נראה כמו overshoot mean-reverting או כמו שינוי מבני אמיתי.",
                    className="mb-3",
                    style=RTL_STYLE,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            make_info_alert(
                                "Macro Card",
                                "מציג רגישות לריבית, דולר ויציבות בטא/קורלציה מול SPY. "
                                "אם המדדים כאן קיצוניים, ייתכן שהמהלך הוא macro repricing ולא mispricing טהור.",
                                color="secondary",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            make_info_alert(
                                "Fundamental Card",
                                "מציג valuation holdings-weighted מול SPY, איכות כיסוי ונתוני earnings. "
                                "אם valuation מצדיק את המהלך, confidence ב-mean reversion יורד.",
                                color="secondary",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            make_info_alert(
                                "Attribution Card",
                                "SDS מודד עוצמת dislocation; FJS מודד הצדקה פונדמנטלית; MSS מודד סיכון מאקרו; STF מודד מגמה מבנית; MC מסכם עד כמה הסטייה נראית clean mispricing.",
                                color="secondary",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            make_info_alert(
                                "Execution / Book Card",
                                "מציג direction, hedge ratio, משקל סופי, regime context, מצב ספר וגריקים סינתטיים. "
                                "זהו החיבור בין האות התיאורטי לבין ניהול ספר בפועל.",
                                color="secondary",
                            ),
                            md=3,
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3",
        style=CARD_STYLE,
    )


# ==========================================================
# Data health banner (shown at top of Overview when degraded)
# ==========================================================
_HEALTH_COLOR = {"HEALTHY": "success", "DEGRADED": "warning", "CRITICAL": "danger"}


def build_health_overview_banner(health: DataHealthReport) -> Any:
    """
    Returns a compact alert at the top of the Overview tab when data is not HEALTHY.
    Returns an empty div when HEALTHY so no space is consumed.

    Intent: the PM should immediately know if analytics are running on degraded data
    before reading any signal or making any decision.
    """
    if not health.degraded:
        return html.Div()

    color = _HEALTH_COLOR.get(health.health_label, "warning")
    score_pct = round(health.health_score * 100, 1)

    # Build the body: errors first, then top warnings (capped at 3 total)
    issues = health.errors[:3] + health.warnings[:max(0, 3 - len(health.errors[:3]))]
    # Strip the [SOURCE:CODE] prefix for readability
    def _strip_prefix(msg: str) -> str:
        if msg.startswith("[") and "]" in msg:
            return msg[msg.index("]") + 2:]
        return msg

    issue_items = [
        html.Li(_strip_prefix(i), style={"fontSize": "12px"})
        for i in issues
    ]
    n_remaining = (len(health.errors) + len(health.warnings)) - len(issues)

    return dbc.Alert(
        [
            html.Div(
                [
                    html.Strong(f"Data Health: {health.health_label} ({score_pct}%) — "),
                    html.Span(
                        "analytics may be operating on incomplete or stale data. "
                        "Open the Data Health tab for full diagnostics.",
                        style={"fontSize": "13px"},
                    ),
                ],
                className="mb-1",
            ),
            html.Ul(
                issue_items
                + ([html.Li(f"… and {n_remaining} more", style={"fontSize": "12px", "color": "#888"})]
                   if n_remaining > 0 else []),
                className="mb-0 ps-3",
            ),
        ],
        color=color,
        className="mb-2 py-2",
    )


# ==========================================================
# KPI / summary builders
# ==========================================================
def build_overview_kpi_rows(
    master_df: pd.DataFrame,
    settings,
    health: DataHealthReport,
) -> Tuple[dbc.Row, dbc.Row]:
    def col0(name: str) -> Any:
        return master_df[name].iloc[0] if (name in master_df.columns and len(master_df)) else None

    tradable = (
        master_df[master_df["direction"].isin(["LONG", "SHORT"])].copy()
        if "direction" in master_df.columns
        else pd.DataFrame()
    )
    top_long = tradable[tradable["direction"] == "LONG"].head(1) if not tradable.empty else pd.DataFrame()
    top_short = tradable[tradable["direction"] == "SHORT"].head(1) if not tradable.empty else pd.DataFrame()

    long_txt = (
        f"LONG {top_long['sector_name'].iloc[0]} ({top_long['sector_ticker'].iloc[0]}) "
        f"{safe_str(top_long['decision_label'].iloc[0])} | "
        f"MC={format_float(top_long['mc_score'].iloc[0], '{:.1f}')}"
        if not top_long.empty
        else "LONG —"
    )
    short_txt = (
        f"SHORT {top_short['sector_name'].iloc[0]} ({top_short['sector_ticker'].iloc[0]}) "
        f"{safe_str(top_short['decision_label'].iloc[0])} | "
        f"MC={format_float(top_short['mc_score'].iloc[0], '{:.1f}')}"
        if not top_short.empty
        else "SHORT —"
    )

    # Data Health card — score, label, and most actionable context line
    _h_score  = round(health.health_score * 100, 1)
    _h_color  = _HEALTH_COLOR.get(health.health_label, "secondary")
    _h_last   = (
        str(health.freshness.price_detail.last_price_date)
        if (health.freshness.price_detail and health.freshness.price_detail.last_price_date)
        else "—"
    )
    if health.health_label == "HEALTHY":
        _h_subtitle = f"{_h_score}% | last price: {_h_last}"
    elif health.errors:
        # Show the first error stripped of its [PREFIX] tag
        _first_issue = health.errors[0]
        if _first_issue.startswith("[") and "]" in _first_issue:
            _first_issue = _first_issue[_first_issue.index("]") + 2:]
        _h_subtitle = f"{_h_score}% | {_first_issue[:60]}…" if len(_first_issue) > 60 else f"{_h_score}% | {_first_issue}"
    else:
        _h_subtitle = f"{_h_score}% | {len(health.warnings)} warning(s) — see Data Health tab"

    cards_top = dbc.Row(
        [
            dbc.Col(
                kpi_card(
                    "Market State",
                    safe_str(col0("market_state")),
                    f"Bias {safe_str(col0('state_bias'))} | Alert {safe_str(col0('regime_alert'))}",
                    color="primary",
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Data Health",
                    health.health_label,
                    _h_subtitle,
                    color=_h_color,
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Top Long / Short",
                    "Top Candidates",
                    f"{long_txt} | {short_txt}",
                    color="primary",
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Transition / Crisis Risk",
                    (
                        f"T {format_float(col0('transition_probability'), '{:.2f}')} | "
                        f"C {format_float(col0('crisis_probability'), '{:.2f}')}"
                    ),
                    (
                        f"RegimeScore {format_float(col0('regime_transition_score'), '{:.2f}')} | "
                        f"CorrScore {format_float(col0('regime_corr_score'), '{:.2f}')}"
                    ),
                    color="warning",
                ),
                md=3,
            ),
        ],
        className="mt-2",
    )

    cards_bottom = dbc.Row(
        [
            dbc.Col(
                kpi_card(
                    "Portfolio Δ",
                    (
                        f"SPY {format_float(col0('delta_spy_P'), '{:.3f}')} | "
                        f"TNX {format_float(col0('delta_tnx_P'), '{:.3f}')} | "
                        f"DXY {format_float(col0('delta_dxy_P'), '{:.3f}')}"
                    ),
                    (
                        f"Gross {format_float(col0('gross_exposure'), '{:.2f}')} | "
                        f"Net {format_float(col0('net_exposure'), '{:.2f}')}"
                    ),
                    color="secondary",
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Synthetic Greeks",
                    (
                        f"Γ {format_float(col0('gamma_synth_P'), '{:.2f}')} | "
                        f"Vega {format_float(col0('vega_synth_P'), '{:.2f}')}"
                    ),
                    (
                        f"ρ_mode {format_float(col0('rho_mode_P'), '{:.2f}')} | "
                        f"ρ_dist {format_float(col0('rho_dist_P'), '{:.2f}')}"
                    ),
                    color="secondary",
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Correlation Regime",
                    (
                        f"Mode {format_float(col0('market_mode_strength'), '{:.2f}')} | "
                        f"D {format_float(col0('corr_matrix_dist_t'), '{:.2f}')}"
                    ),
                    (
                        f"AvgCorr {format_float(col0('avg_corr_t'), '{:.2f}')} | "
                        f"ΔAvgCorr {format_float(col0('avg_corr_delta'), '{:.2f}')} | "
                        f"ΔD {format_float(col0('delta_corr_dist'), '{:.2f}')}"
                    ),
                    color="secondary",
                ),
                md=3,
            ),
            dbc.Col(
                kpi_card(
                    "Execution / Risk",
                    f"PortVol {format_float(col0('port_vol'), '{:.2%}')}",
                    (
                        f"OffDiagShare {format_float(col0('port_offdiag_share'), '{:.2f}')} | "
                        f"{safe_str(col0('execution_regime'))}"
                    ),
                    color="secondary",
                ),
                md=3,
            ),
        ],
        className="mt-2",
    )

    return cards_top, cards_bottom


def build_engine_outputs(engine: QuantEngine, master_df: pd.DataFrame) -> Dict[str, Any]:
    table_df, table = build_scanner_table(master_df)
    scatter_fig = build_scatter(master_df)

    corr_fig = heatmap_fig(engine.corr_matrix_current, "Correlation Matrix (Current)")
    delta_corr_fig = heatmap_fig(engine.corr_matrix_delta, "Correlation Distortion (ΔC)")

    corr_ts = engine.get_correlation_regime_timeseries()
    corr_ts_fig = line_figure(
        corr_ts.reset_index().rename(columns={"index": "date"}),
        x="date",
        y_cols=["avg_corr_t", "distortion_t", "market_mode_strength"],
        title="Correlation Regime Time Series",
    )

    contrib_cols = ["sector_ticker", "sector_corr_dist_contrib"]
    contrib_df = (
        master_df[contrib_cols].copy()
        if all(c in master_df.columns for c in contrib_cols)
        else pd.DataFrame(columns=contrib_cols)
    )

    contrib_fig = bar_figure(
        contrib_df["sector_ticker"] if not contrib_df.empty else [],
        contrib_df["sector_corr_dist_contrib"] if not contrib_df.empty else [],
        "Sector Contribution to Correlation Distortion",
    )

    return {
        "table_df": table_df,
        "table": table,
        "scatter_fig": scatter_fig,
        "corr_fig": corr_fig,
        "delta_corr_fig": delta_corr_fig,
        "corr_ts_fig": corr_ts_fig,
        "contrib_fig": contrib_fig,
    }


# ==========================================================
# Dash app factory
# ==========================================================
def build_app() -> dash.Dash:
    settings = get_settings()
    configure_logging(settings.log_dir)
    logger = logging.getLogger("main")

    # Auto-run pipeline if data is stale (non-blocking for UI — uses cached data if fresh)
    try:
        from db.reader import DatabaseReader
        _db_reader = DatabaseReader(settings.db_path)
        if not _db_reader.is_snapshot_fresh(settings.cache_max_age_hours):
            logger.info("Data stale — triggering auto pipeline run...")
            import subprocess, sys
            subprocess.Popen(
                [sys.executable, str(settings.project_root / "scripts" / "run_all.py")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as _e:
        logger.debug("Auto-pipeline trigger skipped: %s", _e)

    # ══════════════════════════════════════════════════════════════════════
    # SERVICE LAYER: All analytics computation delegated to EngineService
    # This replaces 800+ lines of inline engine initialization.
    # ══════════════════════════════════════════════════════════════════════
    from services.run_context import RunContext
    from services.engine_service import EngineService

    ctx = RunContext.create(settings)
    svc = EngineService(ctx)
    _results = svc.compute_all()

    # Unpack results into the variable names that callbacks depend on
    engine = _results.engine
    master_df = _results.master_df
    _health = _results.data_health

    if master_df is None or master_df.empty:
        raise RuntimeError("master_df is empty; cannot build dashboard.")

    # Map EngineResults → callback closure variables
    _startup_run_id = _results.run_id
    _stress_results = _results.stress_results
    _mc_stress_result = _results.mc_stress_result
    _risk_report = _results.risk_report
    _corr_vol_analysis = _results.corr_vol_analysis
    _dss_corr_snapshot = _results.corr_snapshot
    _dss_signal_results = _results.signal_results
    _dss_trade_tickets = _results.trade_tickets
    _dss_regime_safety = _results.regime_safety
    _dss_monitor_summary = _results.trade_monitor
    _options_surface = _results.options_surface
    _tail_risk_es = _results.tail_risk_es
    _paper_portfolio = _results.paper_portfolio
    _dispersion_result = _results.dispersion_result
    _pnl_result_cached = _results.pnl_result
    _decay_result_cached = _results.decay_result
    _regime_result_cached = _results.regime_result
    _backtest_cached = _results.backtest_result
    _dss_trade_book_history = _results.trade_book_history
    _methodology_ranking = _results.methodology_ranking
    _ml_feature_importances = _results.ml_feature_importances
    _ml_regime_forecast = _results.ml_regime_forecast
    _engine_errors = _results.errors

    logger.info(
        "EngineService: %d/%d steps OK in %.1fs | run_id=%d | regime=%s",
        len(ctx.steps_completed), len(ctx.steps_completed) + len(ctx.steps_failed),
        ctx.duration_s, ctx.run_id, ctx.regime,
    )

    # ── Data loading via service layer ───────────────────────────────────────
    from services.data_loader import DataLoader
    _loader = DataLoader(settings)

    _brief_txt = _loader.load_daily_brief()
    _ml_signals_result = None
    _ml_drift_status = None

    # _engine_error_banner defined below (near callbacks)

    # Delegate closures to DataLoader methods
    def _load_json_safe(path):
        return _loader.load_json(path)

    def _load_improvement_log():
        return _loader.load_improvement_log()

    def _load_methodology_results():
        return _loader.load_methodology_results()

    def _compute_momentum_ranking():
        return _loader.compute_momentum_ranking(engine.prices) if engine else None

    # Load all agent outputs in one call
    _agent_data = _loader.load_agent_outputs()
    _agent_registry_data = _agent_data["registry"]
    _decay_data = _agent_data["decay"]
    _regime_agent_data = _agent_data["regime"]
    _risk_agent_data = _agent_data["risk"]
    _scout_data = _agent_data["scout"]
    _portfolio_alloc = _agent_data["portfolio"]
    _auto_improve_data = _agent_data["auto_improve"]
    _optimizer_data = _agent_data["optimizer"]
    _architect_data = _agent_data["architect"]
    _ensemble_results = _agent_data["ensemble"]

    if _regime_agent_data and not _ml_regime_forecast:
        _ml_regime_forecast = _regime_agent_data

    _n_loaded = sum(1 for v in _agent_data.values() if v is not None)
    logger.info("Agent outputs loaded: %d/%d JSON files via DataLoader", _n_loaded, len(_agent_data))

    # ── Build UI components from results ────────────────────────────────────
    cards_top, cards_bottom = build_overview_kpi_rows(master_df, settings, _health)
    ui_outputs = build_engine_outputs(engine, master_df)

    # Load latest alpha research report
    _alpha_research_data = None
    try:
        _alpha_research_reports = sorted(
            (settings.project_root / "agents" / "methodology" / "reports").glob("*alpha_research*"),
            reverse=True,
        )
        if _alpha_research_reports:
            _alpha_research_data = _load_json_safe(str(_alpha_research_reports[0]))
    except Exception:
        pass

    # Load latest methodology governance report
    _methodology_gov = None
    try:
        _method_gov_reports = sorted(
            (settings.project_root / "agents" / "methodology" / "reports").glob("2026-*.json"),
            reverse=True,
        )
        # Prefer the governance report (not the _methodology_lab or _alpha_research variants)
        for _mgr in _method_gov_reports:
            if "_methodology_lab" not in _mgr.name and "_alpha_research" not in _mgr.name and "calibration_" not in _mgr.name:
                _methodology_gov = _load_json_safe(str(_mgr))
                break
        if _methodology_gov is None and _method_gov_reports:
            _methodology_gov = _load_json_safe(str(_method_gov_reports[0]))
    except Exception:
        pass

    # Also use agent regime data for ML insights if the ML pipeline didn't produce it
    if _ml_regime_forecast is None and _regime_agent_data:
        _ml_regime_forecast = _regime_agent_data

    logger.info(
        "Agent outputs loaded: registry=%s, decay=%s, regime=%s, risk=%s, scout=%s, portfolio=%s, methodology_gov=%s",
        "yes" if _agent_registry_data else "no",
        "yes" if _decay_data else "no",
        "yes" if _regime_agent_data else "no",
        "yes" if _risk_agent_data else "no",
        "yes" if _scout_data else "no",
        "yes" if _portfolio_alloc else "no",
        "yes" if _methodology_gov else "no",
    )

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        compress=True,
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    app.title = "SRV Quantamental DSS"

    # Institutional design system CSS
    app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  /* ══════════════════════════════════════════════════════════════
     SRV Quantamental DSS — Institutional Design System
     Color semantics: green=long/safe, red=short/danger,
     amber=caution, cyan=info, white=primary text
     ══════════════════════════════════════════════════════════════ */

  /* ── Typography ─────────────────────────────────────────────── */
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, monospace;
    font-size: 13px;
    letter-spacing: 0.01em;
    -webkit-font-smoothing: antialiased;
  }
  .h1,.h2,.h3,.h4,.h5,.h6,h1,h2,h3,h4,h5,h6 {
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  /* ── Navigation ─────────────────────────────────────────────── */
  .nav-tabs {
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    scrollbar-width: thin;
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 0;
    gap: 2px;
  }
  .nav-tabs .nav-link {
    font-size: 0.78rem;
    font-weight: 500;
    padding: 8px 14px;
    color: #8a8a9a;
    border: none;
    border-bottom: 2px solid transparent;
    transition: all 0.15s ease;
    letter-spacing: 0.02em;
  }
  .nav-tabs .nav-link:hover {
    color: #ccc;
    border-bottom-color: #444;
  }
  .nav-tabs .nav-link.active {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
    font-weight: 600;
  }

  /* ── Cards ──────────────────────────────────────────────────── */
  .card {
    border-radius: 4px;
    border-color: #1e1e2e;
    transition: box-shadow 0.15s ease;
  }
  .card:hover {
    box-shadow: 0 1px 6px rgba(0,212,255,0.08);
  }
  .card-header {
    background: #16162a !important;
    border-bottom: 1px solid #2a2a3a;
    padding: 8px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .card-body {
    padding: 12px;
  }

  /* ── Tables ─────────────────────────────────────────────────── */
  .table {
    font-size: 11.5px;
    margin-bottom: 0;
  }
  .table thead th {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8a8a9a;
    font-weight: 600;
    border-bottom: 1px solid #2a2a3a;
    padding: 6px 8px;
  }
  .table td {
    padding: 5px 8px;
    border-color: #1a1a2e;
    vertical-align: middle;
  }
  .table-hover tbody tr:hover {
    background-color: rgba(0,212,255,0.04) !important;
  }

  /* ── KPI Cards (override heavy borders) ─────────────────────── */
  .card[class*="border-"] {
    border-width: 1px !important;
  }
  .card[style*="borderTop: 3px"] {
    border-top-width: 2px !important;
  }

  /* ── Badges ─────────────────────────────────────────────────── */
  .badge {
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 3px 8px;
  }

  /* ── Charts ─────────────────────────────────────────────────── */
  .js-plotly-plot .plotly .modebar {
    opacity: 0;
    transition: opacity 0.2s;
  }
  .js-plotly-plot:hover .plotly .modebar {
    opacity: 0.5;
  }

  /* ── Alerts ─────────────────────────────────────────────────── */
  .alert {
    border-radius: 4px;
    font-size: 12px;
    padding: 8px 12px;
  }

  /* ── Scrollbar ──────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #111; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #555; }

  /* ── Loading spinner ────────────────────────────────────────── */
  ._dash-loading { opacity: 0.5; }

  /* ── Responsive ─────────────────────────────────────────────── */
  @media (max-width: 768px) {
    .container-fluid { padding: 6px !important; }
    .card-body { padding: 8px !important; }
    .h4 { font-size: 0.95rem !important; }
    .h5 { font-size: 0.82rem !important; }
    .nav-tabs .nav-link { font-size: 0.68rem !important; padding: 5px 8px !important; }
    table { font-size: 10px !important; }
    .col, .col-md-6, .col-md-7, .col-md-8, .col-md-5, .col-md-4 {
      flex: 0 0 100% !important; max-width: 100% !important;
    }
  }
  @media (max-width: 1200px) {
    .nav-tabs .nav-link { font-size: 0.72rem !important; padding: 6px 10px !important; }
  }

  /* ── Utility ────────────────────────────────────────────────── */
  .text-profit { color: #00d4aa !important; }
  .text-loss { color: #ff4757 !important; }
  .text-caution { color: #ffa502 !important; }
  .text-info-dim { color: #5f9ea0 !important; }
  .border-subtle { border-color: #2a2a3a !important; }
  html { scroll-behavior: smooth; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''

    app.layout = dbc.Container(
        fluid=True,
        style=APP_CONTAINER_STYLE,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div([
                                html.Span("SRV", style={"color": "#00d4ff", "fontWeight": "700", "fontSize": "1.4rem", "letterSpacing": "-0.02em"}),
                                html.Span(" Quantamental DSS", style={"color": "#888", "fontWeight": "400", "fontSize": "1.1rem", "marginLeft": "4px"}),
                            ], className="mt-3 mb-0"),
                            html.Div(
                                "Sector Relative Value — Decision Support System",
                                style={"fontSize": "11px", "color": "#555", "letterSpacing": "0.05em", "textTransform": "uppercase"},
                            ),
                        ],
                        md=9,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "Daily Brief",
                                id="open-brief-modal",
                                color="outline-info",
                                size="sm",
                                className="mt-3 me-2",
                                disabled=not bool(_brief_txt),
                            ),
                            dbc.Button(
                                "הסבר מתודולוגי מלא",
                                id="open-methodology-modal",
                                color="primary",
                                className="mt-3",
                            ),
                        ],
                        md=3,
                        style={"textAlign": "left"},
                    ),
                ]
            ),
            build_global_explanation_banner(),
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-overview",
                children=[
                    # ── Trading ──
                    dbc.Tab(label="Overview",       tab_id="tab-overview"),
                    dbc.Tab(label="DSS",            tab_id="tab-dss"),
                    dbc.Tab(label="Scanner",        tab_id="tab-scanner"),
                    dbc.Tab(label="Portfolio",      tab_id="tab-portfolio"),
                    # ── Risk ──
                    dbc.Tab(label="Risk",           tab_id="tab-risk"),
                    dbc.Tab(label="Stress",         tab_id="tab-stress"),
                    dbc.Tab(label="Regime",         tab_id="tab-regime"),
                    # ── Analytics ──
                    dbc.Tab(label="P&L",            tab_id="tab-pnl"),
                    dbc.Tab(label="Backtest",       tab_id="tab-backtest"),
                    dbc.Tab(label="Correlation",    tab_id="tab-correlation"),
                    dbc.Tab(label="Corr&Vol",       tab_id="tab-corrvol"),
                    dbc.Tab(label="Decay",          tab_id="tab-decay"),
                    dbc.Tab(label="Tear Sheet",     tab_id="tab-tearsheet"),
                    # ── Research ──
                    dbc.Tab(label="Methodology",    tab_id="tab-methodology"),
                    dbc.Tab(label="Optimization",   tab_id="tab-optimization"),
                    dbc.Tab(label="ML",             tab_id="tab-ml"),
                    dbc.Tab(label="Agents",         tab_id="tab-agents"),
                    dbc.Tab(label="Health",         tab_id="tab-health"),
                ],
                className="mt-2",
                style={"flexWrap": "wrap", "overflow": "visible"},
            ),
            html.Div(id="tab-content", className="mt-3"),
            dcc.Store(id="master-store", data=master_df.to_dict("records")),
            dcc.Store(id="table-store", data=ui_outputs["table_df"].to_dict("records")),
            dcc.Store(id="backtest-store", data=None),
            dcc.Store(id="run-context-store", data=ctx.summary if ctx else {}),
            dcc.Interval(id="auto-refresh-interval", interval=5 * 60 * 1000, n_intervals=0),
            # Daily brief modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Daily Brief — Morning Report"), close_button=True),
                    dbc.ModalBody(id="brief-modal-body"),
                    dbc.ModalFooter(dbc.Button("סגור", id="close-brief-modal", color="secondary")),
                ],
                id="brief-modal",
                size="xl",
                scrollable=True,
                is_open=False,
            ),
            build_methodology_modal(),
            # ── Disclaimer footer ──
            html.Hr(style={"borderColor": "#333", "marginTop": "30px"}),
            html.Div(
                "DSS בלבד — אינו מהווה ייעוץ השקעות | Not financial advice | "
                "ביצועי עבר אינם ערובה לתוצאות עתידיות",
                style={
                    "textAlign": "center", "color": "#666",
                    "fontSize": "11px", "paddingBottom": "15px",
                },
            ),
        ],
    )

    # ======================================================
    # Modal toggle
    # ======================================================
    @app.callback(
        Output("methodology-modal", "is_open"),
        Input("open-methodology-modal", "n_clicks"),
        Input("close-methodology-modal", "n_clicks"),
        State("methodology-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_methodology_modal(
        open_clicks: Optional[int],
        close_clicks: Optional[int],
        is_open: bool,
    ) -> bool:
        return not is_open

    # ======================================================
    # Daily Brief modal
    # ======================================================
    @app.callback(
        Output("brief-modal", "is_open"),
        Output("brief-modal-body", "children"),
        Input("open-brief-modal", "n_clicks"),
        Input("close-brief-modal", "n_clicks"),
        State("brief-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_brief_modal(
        open_clicks: Optional[int],
        close_clicks: Optional[int],
        is_open: bool,
    ):
        from dash import ctx
        if ctx.triggered_id == "close-brief-modal":
            return False, dash.no_update
        if ctx.triggered_id == "open-brief-modal":
            if _brief_txt:
                lines = _brief_txt.split("\n")
                content = [html.Pre(_brief_txt, style={"fontSize": "13px", "whiteSpace": "pre-wrap", "direction": "ltr", "textAlign": "left"})]
            else:
                content = [html.Div("אין Daily Brief זמין. הרץ את agent_daily_pipeline.py כדי לייצר.", style={"textAlign": "right"})]
            return True, content
        return is_open, dash.no_update

    # ======================================================
    # Helper: engine error banner
    # ======================================================
    def _engine_error_banner(engine_key: str, engine_label: str) -> Optional[dbc.Alert]:
        """Return a visible error alert if an engine failed at startup, else None."""
        err = _engine_errors.get(engine_key)
        if not err:
            return None
        return dbc.Alert(
            [
                html.Strong(f"⚠️ {engine_label} נכשל בהפעלה — "),
                html.Span("הנתונים בלשונית זו אינם זמינים. "),
                html.Code(err[:200], style={"fontSize": "11px"}),
            ],
            color="danger",
            className="mb-3",
            style={"textAlign": "right", "direction": "rtl"},
        )

    # ======================================================

    # ======================================================
    # Tab rendering via TabContext + tab_renderer.py
    # ======================================================
    from ui.tab_renderer import TabContext, render_tab_content

    _tab_ctx = TabContext(
        master_df=master_df, settings=settings, engine=engine,
        signal_results=_dss_signal_results, trade_tickets=_dss_trade_tickets,
        regime_safety=_dss_regime_safety, corr_snapshot=_dss_corr_snapshot,
        monitor_summary=_dss_monitor_summary, trade_book_history=_dss_trade_book_history,
        methodology_ranking=_methodology_ranking,
        options_surface=_options_surface, tail_risk_es=_tail_risk_es,
        stress_results=_stress_results, mc_stress_result=_mc_stress_result,
        risk_report=_risk_report, corr_vol_analysis=_corr_vol_analysis,
        pnl_result=_pnl_result_cached, decay_result=_decay_result_cached,
        regime_result=_regime_result_cached, backtest_result=_backtest_cached,
        paper_portfolio=_paper_portfolio, dispersion_result=_dispersion_result,
        ml_feature_importances=_ml_feature_importances,
        ml_regime_forecast=_ml_regime_forecast,
        ml_signals_result=_ml_signals_result, ml_drift_status=_ml_drift_status,
        ensemble_results=_ensemble_results,
        agent_registry=_agent_registry_data, decay_data=_decay_data,
        regime_agent_data=_regime_agent_data, risk_agent_data=_risk_agent_data,
        scout_data=_scout_data, portfolio_alloc=_portfolio_alloc,
        auto_improve_data=_auto_improve_data, optimizer_data=_optimizer_data,
        architect_data=_architect_data,
        errors=_engine_errors, brief_txt=_brief_txt, data_health=_health,
        compute_momentum_ranking=_compute_momentum_ranking,
        load_improvement_log=_load_improvement_log,
        load_methodology_results=_load_methodology_results,
        load_json_safe=_load_json_safe,
        engine_error_banner=_engine_error_banner,
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "active_tab"),
    )
    def render_tab(active_tab: str):
        logger.debug("render_tab: %s", active_tab)
        if not active_tab:
            active_tab = "tab-overview"
        return render_tab_content(active_tab, _tab_ctx)

    # Tear sheet callback
    # ======================================================
    @app.callback(
        Output("residual-xray", "figure"),
        Output("card-macro", "children"),
        Output("card-fund", "children"),
        Output("card-attrib", "children"),
        Output("card-exec", "children"),
        Input("tearsheet-sector-dropdown", "value"),
        Input("main-tabs", "active_tab"),
        State("master-store", "data"),
        prevent_initial_call=False,
    )
    def update_tearsheet(
        sector_ticker: Optional[str],
        active_tab: Optional[str],
        master_store: Optional[List[Dict[str, Any]]],
    ):
        _dark = dict(template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", height=520)
        _empty = go.Figure().update_layout(**_dark, title="בחר סקטור")
        _empty_ret = (_empty, "—", "—", "—", "—")

        # Only render when tear sheet tab is active
        if active_tab != "tab-tearsheet":
            return _empty_ret

        # Default to XLK if dropdown hasn't loaded yet
        if not sector_ticker:
            sector_ticker = "XLK"

        if not master_store:
            fig = go.Figure()
            fig.update_layout(**_dark, title="בחר סקטור")
            return fig, "—", "—", "—", "—"

        full_df = pd.DataFrame(master_store)
        row_full = full_df.loc[full_df["sector_ticker"] == sector_ticker]
        if row_full.empty:
            fig = go.Figure()
            fig.update_layout(**_dark, title="סקטור לא נמצא")
            return fig, "—", "—", "—", "—"

        row = row_full.iloc[0].to_dict()
        sector_name = row.get("sector_name", sector_ticker)

        try:
            ts = engine.get_sector_tearsheet_series(sector_ticker)
        except Exception as _ts_err:
            logger.warning("Tear sheet data failed for %s: %s", sector_ticker, _ts_err)
            fig = go.Figure()
            fig.update_layout(**_dark, title=f"שגיאה בטעינת {sector_ticker}")
            return fig, "—", "—", "—", "—"

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.30, 0.25],
            vertical_spacing=0.04,
            subplot_titles=[
                f"{sector_name} ({sector_ticker}) — OOS PCA Residual X-Ray",
                "Cumulative Alpha",
                "Z-Score History",
            ],
        )

        # Row 1: Residual level + bands
        fig.add_trace(go.Scatter(x=ts.index, y=ts["residual_level"], name="Residual Level",
                                 mode="lines", line=dict(color="#4da6ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=ts["upper_2s"], name="+2\u03c3",
                                 mode="lines", line=dict(color="#ff6666", dash="dash", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=ts["lower_2s"], name="-2\u03c3",
                                 mode="lines", line=dict(color="#ff6666", dash="dash", width=1)), row=1, col=1)

        # Row 2: Cumulative alpha (cumulative sum of residual level changes)
        if "residual_level" in ts.columns and len(ts) > 1:
            _resid_diff = ts["residual_level"].diff().fillna(0)
            _cum_alpha = _resid_diff.cumsum()
            fig.add_trace(go.Scatter(x=ts.index, y=_cum_alpha, name="Cum. Alpha",
                                     mode="lines", fill="tozeroy",
                                     line=dict(color="#20c997", width=1.5),
                                     fillcolor="rgba(32,201,151,0.15)"), row=2, col=1)

        # Row 3: Z-score history with threshold lines
        if "zscore" in ts.columns:
            fig.add_trace(go.Scatter(x=ts.index, y=ts["zscore"], name="Z-Score",
                                     mode="lines", line=dict(color="#ffc107", width=1.5)), row=3, col=1)
            fig.add_hline(y=2.0, line_dash="dot", line_color="#ff4444", row=3, col=1)
            fig.add_hline(y=-2.0, line_dash="dot", line_color="#ff4444", row=3, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="#555", row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=680,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=30, r=20, t=60, b=30),
        )

        def m0(name: str) -> Any:
            return row_full[name].iloc[0] if (name in row_full.columns and len(row_full)) else None

        macro_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("מאקרו ויציבות מבנית", className="fw-bold mb-2", style=RTL_STYLE),
                    html.Div(
                        "כאן נבדקת הרגישות של הסקטור לריבית, דולר ושינויי בטא/קורלציה מול SPY. "
                        "אם המדדים כאן קיצוניים, יש סיכוי גבוה יותר שמדובר ב-macro repricing ולא ב-mean reversion נקי.",
                        className="text-muted mb-3",
                        style=RTL_STYLE,
                    ),
                    html.Div(f"β(TNX): {format_float(row.get('beta_tnx_60d'), '{:.3f}')}", style=RTL_STYLE),
                    html.Div(f"β(DXY): {format_float(row.get('beta_dxy_60d'), '{:.3f}')}", style=RTL_STYLE),
                    html.Div(f"β_SPY Δ: {format_float(row.get('beta_spy_delta'), '{:.3f}')}", style=RTL_STYLE),
                    html.Div(f"Corr_SPY Δ: {format_float(row.get('corr_to_spy_delta'), '{:.3f}')}", style=RTL_STYLE),
                    html.Div(f"MSS: {format_float(row.get('mss_score'), '{:.3f}')}", className="mt-2 fw-bold", style=RTL_STYLE),
                ]
            ),
            style=CARD_STYLE,
        )

        fund_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("שכבת fundamentals", className="fw-bold mb-2", style=RTL_STYLE),
                    html.Div(
                        "הכרטיס מציג valuation holdings-weighted, השוואה מול SPY ואיכות כיסוי הנתונים. "
                        "אם valuation תומך בכיוון המחיר, confidence ב-mispricing יורד.",
                        className="text-muted mb-3",
                        style=RTL_STYLE,
                    ),
                    html.Div(f"P/E סקטור: {format_float(row.get('pe_sector_portfolio'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(f"Relative P/E vs SPY: {format_float(row.get('rel_pe_vs_spy'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(f"Coverage Weight: {format_float(row.get('fund_covered_weight'), '{:.1%}')}", style=RTL_STYLE),
                    html.Div(
                        f"Negative / Missing Earnings Weight: {format_float(row.get('neg_or_missing_earnings_weight'), '{:.1%}')}",
                        style=RTL_STYLE,
                    ),
                    html.Div(f"Source: {safe_str(row.get('fund_source'))}", style=RTL_STYLE),
                    html.Div(f"FJS: {format_float(row.get('fjs_score'), '{:.3f}')}", className="mt-2 fw-bold", style=RTL_STYLE),
                ]
            ),
            style=CARD_STYLE,
        )

        def _attrib_bar(label: str, val: Any, max_v: float, color: str) -> html.Div:
            try:
                pct = min(100.0, max(0.0, float(val or 0) / max_v * 100))
            except Exception:
                pct = 0.0
            return html.Div(
                [
                    html.Div(
                        [
                            html.Span(label, style={"width": "32px", "fontSize": "11px", "color": "#aaa"}),
                            dbc.Progress(value=pct, color=color, style={"height": "8px", "flex": "1"}),
                            html.Span(format_float(val, "{:.1f}"), style={"width": "34px", "fontSize": "11px", "textAlign": "right"}),
                        ],
                        className="d-flex align-items-center gap-2",
                    )
                ],
                className="mb-2",
            )

        mc_val = row.get("mc_score", 0)
        mc_pct = min(100.0, max(0.0, float(mc_val or 0)))
        direction_ts = str(row.get("direction", "NEUTRAL"))
        mc_color = "success" if direction_ts == "LONG" else "danger" if direction_ts == "SHORT" else "secondary"

        attrib_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("Attribution", className="fw-bold mb-3", style=RTL_STYLE),
                    # MC gauge
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("MC", style={"fontSize": "12px", "color": "#aaa", "width": "32px"}),
                                    dbc.Progress(value=mc_pct, color=mc_color, style={"height": "12px", "flex": "1"}),
                                    html.Span(
                                        format_float(mc_val, "{:.0f}"),
                                        style={"fontSize": "14px", "fontWeight": "700", "width": "34px", "textAlign": "right"},
                                    ),
                                ],
                                className="d-flex align-items-center gap-2",
                            )
                        ],
                        className="mb-3",
                    ),
                    _attrib_bar("SDS", row.get("sds_score"), 1.0, "info"),
                    _attrib_bar("FJS", row.get("fjs_score"), 1.0, "success"),
                    _attrib_bar("MSS", row.get("mss_score"), 1.0, "warning"),
                    _attrib_bar("STF", row.get("stf_score"), 1.0, "danger"),
                    html.Hr(className="my-2"),
                    dbc.Row(
                        [
                            dbc.Col([html.Small("פירוש", className="text-muted"), html.Div(safe_str(row.get("interpretation")), style={"fontSize": "12px"})], md=6),
                            dbc.Col([html.Small("Action Bias", className="text-muted"), html.Div(safe_str(row.get("action_bias")), style={"fontSize": "12px"})], md=6),
                        ],
                        className="mb-1",
                    ),
                    dbc.Row(
                        [
                            dbc.Col([html.Small("החלטה", className="text-muted"), dbc.Badge(safe_str(row.get("decision_label")), color="light", text_color="dark")], md=6),
                            dbc.Col([html.Small("Risk Label", className="text-muted"), html.Div(safe_str(row.get("risk_label")), style={"fontSize": "12px"})], md=6),
                        ],
                        className="mb-1",
                    ),
                    html.Div(safe_str(row.get("pm_note")), className="text-muted mt-2", style={"fontSize": "11px", "fontStyle": "italic", "textAlign": "right"}),
                ]
            ),
            style=CARD_STYLE,
        )

        exec_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("ביצוע וניהול ספר", className="fw-bold mb-2", style=RTL_STYLE),
                    html.Div(
                        "זהו התרגום של האות למחלקת book construction. "
                        "המערכת משתמשת ב-MC, vol-scaling, beta-neutralization, regime context ו-risk-targeting כדי להציע משקל סופי וניהול סיכונים.",
                        className="text-muted mb-3",
                        style=RTL_STYLE,
                    ),
                    html.Div(f"Direction: {safe_str(row.get('direction'))}", style=RTL_STYLE),
                    html.Div(f"Conviction: {format_float(row.get('conviction_score'), '{:.1f}')}", style=RTL_STYLE),
                    html.Div(f"MC Score: {format_float(row.get('mc_score'), '{:.1f}')}", style=RTL_STYLE),
                    html.Div(f"Hedge Ratio: {format_float(row.get('hedge_ratio'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(f"w_final: {format_float(row.get('w_final'), '{:.4f}')}", style=RTL_STYLE),
                    html.Div(f"Market State: {safe_str(row.get('market_state'))}", style=RTL_STYLE),
                    html.Div(f"State Bias: {safe_str(row.get('state_bias'))}", style=RTL_STYLE),
                    html.Div(f"Regime Alert: {safe_str(row.get('regime_alert'))}", style=RTL_STYLE),
                    html.Div(
                        f"Transition Score: {format_float(row.get('regime_transition_score'), '{:.2f}')}",
                        style=RTL_STYLE,
                    ),
                    html.Div(
                        f"Crisis Probability: {format_float(row.get('crisis_probability'), '{:.2f}')}",
                        style=RTL_STYLE,
                    ),
                    html.Div(f"Decision: {safe_str(row.get('decision_label'))}", style=RTL_STYLE),
                    html.Div(f"Size Bucket: {safe_str(row.get('size_bucket'))}", style=RTL_STYLE),
                    html.Div(f"Risk Override: {safe_str(row.get('risk_override'))}", style=RTL_STYLE),
                    html.Div(f"Interpretation: {safe_str(row.get('interpretation'))}", style=RTL_STYLE),
                    html.Div(f"Action Bias: {safe_str(row.get('action_bias'))}", style=RTL_STYLE),
                    html.Div(f"Risk Label: {safe_str(row.get('risk_label'))}", style=RTL_STYLE),
                    html.Hr(className="my-2"),
                    html.Div(f"Δ_SPY_P {format_float(m0('delta_spy_P'), '{:.3f}')}", style=RTL_STYLE),
                    html.Div(f"Γ_P {format_float(m0('gamma_synth_P'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(f"ρ_mode_P {format_float(m0('rho_mode_P'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(f"ρ_dist_P {format_float(m0('rho_dist_P'), '{:.2f}')}", style=RTL_STYLE),
                    html.Div(
                        f"Execution Regime: {safe_str(m0('execution_regime'))}",
                        className="mt-2 fw-bold",
                        style=RTL_STYLE,
                    ),
                ]
            ),
            style=CARD_STYLE,
        )

        return fig, macro_card, fund_card, attrib_card, exec_card

    # ======================================================
    # Journal: log new decision
    # ======================================================
    @app.callback(
        Output("journal-submit-feedback", "children"),
        Input("journal-submit-btn", "n_clicks"),
        State("journal-sector-dropdown", "value"),
        State("journal-pm-direction", "value"),
        State("journal-model-direction", "value"),
        State("journal-conviction-input", "value"),
        State("journal-notes-input", "value"),
        prevent_initial_call=True,
    )
    def log_journal_decision(
        n_clicks: Optional[int],
        sector: Optional[str],
        pm_direction: Optional[str],
        model_direction: Optional[str],
        conviction_score: Optional[float],
        notes: Optional[str],
    ) -> Any:
        if not sector or not pm_direction or not model_direction:
            return dbc.Alert("יש לבחור סקטור וכיוון לפני שמירה.", color="warning", dismissable=True)
        try:
            from analytics.attribution import AttributionResult
            _blank = AttributionResult(
                sds=0.0, fjs=0.0, mss=0.0, stf=0.0, mc=0.0,
                trend_ratio_slope_63d=0.0, trend_ratio_slope_126d=0.0,
                beta_instability=0.0, corr_instability=0.0, corr_shift_score=0.0,
                dislocation_label="—", fundamental_label="—", macro_label="—",
                structural_label="—", mc_label="—", action_bias="NEUTRAL",
                risk_label="—", interpretation="Manual entry", explanation_tags=[],
            )
            did = journal.log_decision(
                sector=sector,
                attribution_result=_blank,
                pm_direction=pm_direction,
                model_direction=model_direction,
                conviction_score=float(conviction_score) if conviction_score is not None else float("nan"),
                notes=notes or "",
            )
            return dbc.Alert(f"ההחלטה נשמרה (ID={did}). רענן את הטאב לעדכון.", color="success", dismissable=True)
        except Exception as exc:
            logger.exception("Failed to log journal decision")
            return dbc.Alert(f"שגיאה: {exc}", color="danger", dismissable=True)

    # ======================================================
    # Backtest: run on-demand
    # ======================================================
    @app.callback(
        Output("backtest-output", "children"),
        Output("backtest-status", "children"),
        Input("run-backtest-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_backtest_callback(n_clicks: Optional[int]) -> tuple:
        if not n_clicks:
            return html.Div(), html.Div()
        try:
            from analytics.backtest import WalkForwardBacktester
            try:
                prices_df = engine.prices if engine is not None else None
                if prices_df is None or prices_df.empty:
                    prices_df = pd.read_parquet(settings.project_root / "data_lake" / "parquet" / "prices.parquet")
            except Exception:
                from db.reader import DatabaseReader as _DBR
                prices_df = _DBR(settings.db_path).read_prices()
            if prices_df is None or prices_df.empty:
                return html.Div(), dbc.Alert("אין נתוני מחירים להרצת הבאקטסט.", color="danger")
            if prices_df is None or prices_df.empty:
                return html.Div(), dbc.Alert("אין נתוני מחירים להרצת הבאקטסט.", color="danger")
            bt = WalkForwardBacktester(settings)
            result = bt.run_backtest(prices_df, pd.DataFrame(), pd.DataFrame())
            return build_backtest_tab(result), dbc.Alert("הבאקטסט הסתיים בהצלחה.", color="success", duration=4000)
        except Exception as exc:
            logger.exception("Backtest failed")
            return html.Div(), dbc.Alert(f"שגיאה בבאקטסט: {exc}", color="danger")

    # ======================================================
    # Correlation heatmap window selector
    # ======================================================
    @app.callback(
        Output("corr-heatmap-current", "figure"),
        Output("corr-heatmap-baseline", "figure"),
        Output("corr-heatmap-delta", "figure"),
        Input("corr-window-select", "value"),
        prevent_initial_call=True,
    )
    def update_corr_heatmaps(window_str):
        """Recalculate correlation heatmaps with selected rolling window."""
        try:
            window = int(window_str) if window_str else 60
            _sectors = settings.sector_list()
            _p = engine.prices[_sectors] if engine and engine.prices is not None else None
            if _p is None or _p.empty:
                raise dash.exceptions.PreventUpdate

            _rets = _p.pct_change().dropna()
            _corr_current = _rets.tail(window).corr()
            _corr_baseline = _rets.tail(252).corr()
            _corr_delta = _corr_current - _corr_baseline

            def _hm(df, title, colorscale, zmid=None):
                if df is None or df.empty:
                    fig = go.Figure()
                    fig.add_annotation(text="N/A", xref="paper", yref="paper",
                                       x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(template="plotly_dark",
                                      paper_bgcolor="#1a1a2e", height=280)
                    return fig
                z = df.values.round(3).tolist()
                labels = list(df.columns)
                kw = dict(zmid=zmid) if zmid is not None else {}
                fig = go.Figure(go.Heatmap(
                    z=z, x=labels, y=labels,
                    colorscale=colorscale, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in z],
                    texttemplate="%{text}", textfont=dict(size=9),
                    showscale=True, **kw,
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#16213e",
                    margin=dict(l=5, r=5, t=30, b=5), height=280,
                    title=dict(text=title, font=dict(size=11)),
                )
                return fig

            fig_c = _hm(_corr_current, f"C_t — Rolling {window}d", "RdYlGn")
            fig_b = _hm(_corr_baseline, "C_b — Baseline 252d", "RdYlGn")
            fig_d = _hm(_corr_delta, f"Delta ({window}d - 252d)", "RdBu", zmid=0)
            return fig_c, fig_b, fig_d
        except Exception:
            raise dash.exceptions.PreventUpdate

    # ======================================================
    # Auto-refresh: regime status + VIX on interval tick
    # ======================================================
    @app.callback(
        Output("overview-regime-container", "children"),
        Output("overview-vix-display", "children"),
        Input("auto-refresh-interval", "n_intervals"),
        State("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def auto_refresh_overview(n_intervals, active_tab):
        """
        Refresh regime hero + VIX display every 5 minutes.
        Reads fresh prices from parquet for live VIX/SPY updates.
        """
        if active_tab != "tab-overview":
            raise dash.exceptions.PreventUpdate
        try:
            import datetime as _dt

            # Try reading fresh VIX from parquet (updates from pipeline)
            _vix_val = float("nan")
            _spy_val = float("nan")
            _spy_chg = float("nan")
            try:
                _fresh_prices = pd.read_parquet(
                    settings.project_root / "data_lake" / "parquet" / "prices.parquet"
                )
                _vix_col = next((c for c in _fresh_prices.columns if "VIX" in c.upper()), None)
                if _vix_col:
                    _vix_s = _fresh_prices[_vix_col].dropna()
                    if len(_vix_s) >= 2:
                        _vix_val = float(_vix_s.iloc[-1])
                if "SPY" in _fresh_prices.columns:
                    _spy_s = _fresh_prices["SPY"].dropna()
                    if len(_spy_s) >= 2:
                        _spy_val = float(_spy_s.iloc[-1])
                        _spy_chg = float(_spy_s.iloc[-1] / _spy_s.iloc[-2] - 1)
            except Exception:
                pass

            # Fallback to startup master_df
            if not np.isfinite(_vix_val):
                _vix_val = float(master_df["vix_level"].iloc[0]) if "vix_level" in master_df.columns else float("nan")

            _vix_pct = float(master_df["vix_percentile"].iloc[0]) if "vix_percentile" in master_df.columns else float("nan")
            _ms_val = str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "—"

            # Regime classification from live VIX
            if np.isfinite(_vix_val):
                if _vix_val > 32:
                    _ms_val = "CRISIS"
                elif _vix_val > 21:
                    _ms_val = "TENSION"
                elif _vix_val > 16:
                    _ms_val = "NORMAL"
                else:
                    _ms_val = "CALM"

            _row0 = master_df.iloc[0].to_dict() if len(master_df) else {}
            _row0["market_state"] = _ms_val
            if np.isfinite(_vix_val):
                _row0["vix_level"] = _vix_val
            regime_component = build_regime_hero(_row0)

            _ts_now = _dt.datetime.now().strftime("%H:%M:%S")
            _regime_color = {"CALM": "success", "NORMAL": "info", "TENSION": "warning", "CRISIS": "danger"}.get(_ms_val, "secondary")

            _spy_str = f"SPY: ${_spy_val:.2f} ({_spy_chg:+.2%})" if np.isfinite(_spy_val) else ""
            vix_component = dbc.Alert(
                [
                    html.Span(f"🔄 {_ts_now} | ", className="text-muted small"),
                    dbc.Badge(_ms_val, color=_regime_color, className="me-2", style={"fontSize": "10px"}),
                    html.Span(f"VIX: {_vix_val:.1f}" if np.isfinite(_vix_val) else "VIX: —",
                              className="small fw-bold me-2"),
                    html.Span(f"({_vix_pct:.0%}ile)" if np.isfinite(_vix_pct) else "",
                              className="text-muted small me-3"),
                    html.Span(_spy_str, className="small me-2"),
                ],
                color="dark",
                className="py-1 px-3 mb-2",
                style={"fontSize": "11px", "border": "1px solid #333"},
                dismissable=True,
            )
            return regime_component, vix_component
        except Exception:
            raise dash.exceptions.PreventUpdate

    logger.info("Dashboard ready. Starting server...")
    return app


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    app = build_app()
    try:
        app.run(debug=False, host="0.0.0.0", port=8050)
    except Exception:
        app.run_server(debug=False, host="0.0.0.0", port=8050)