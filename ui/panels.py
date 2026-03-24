from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table import DataTable


def format_float(x: Any, fmt: str = "{:.2f}") -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        if not (v == v):
            return "—"
        return fmt.format(v)
    except Exception:
        return "—"


def kpi_card(title: str, value: str, subtitle: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.Div(value, className="h3 mb-0"),
                html.Div(subtitle, className="text-muted small mt-1"),
            ]
        ),
        className=f"border-{color}",
    )


# ── Design system constants ─────────────────────────────────────────────────
_S = {
    "label":  {"fontSize": "11px", "color": "#888", "textTransform": "uppercase", "letterSpacing": "0.5px"},
    "value":  {"fontSize": "15px", "fontWeight": "700", "color": "#e8eaf6"},
    "value_sm": {"fontSize": "13px", "fontWeight": "600", "color": "#e8eaf6"},
    "muted":  {"fontSize": "12px", "color": "#888"},
    "explain":{"fontSize": "12px", "color": "#9eaabf", "lineHeight": "1.6"},
    "card_pad": {"padding": "16px"},
}

_REGIME_BADGE: Dict[str, str] = {
    "CALM": "success",
    "NORMAL": "primary",
    "TENSION": "warning",
    "CRISIS": "danger",
}
_DIR_BADGE: Dict[str, str] = {
    "LONG": "success",
    "SHORT": "danger",
    "NEUTRAL": "secondary",
}
_REGIME_EXPLAIN: Dict[str, str] = {
    "CALM":    "קורלציות נמוכות, פיזור בין סקטורים — סביבה אידיאלית לאסטרטגיות Relative Value",
    "NORMAL":  "קורלציות ממוצעות, משטר יציב — מסחר כרגיל עם גדלים סטנדרטיים",
    "TENSION": "קורלציות מוגברות, סיכון מעבר לCRISIS — צמצם גדלים ב-30-40%, הוסף גידור",
    "CRISIS":  "קורלציות גבוהות מאוד, כל הסקטורים נופלים יחד — מינימום חשיפה, הגנות בלבד",
}


def _regime_color(state: str) -> str:
    return _REGIME_BADGE.get(str(state).upper(), "secondary")


def _val_color(val: Any, good_high: bool = True, threshold_warn: float = 0.0, threshold_bad: float = 0.0) -> str:
    """Return bootstrap color class based on value direction."""
    try:
        v = float(val or 0)
        if good_high:
            if v >= threshold_warn:
                return "text-success"
            if v >= threshold_bad:
                return "text-warning"
            return "text-danger"
        else:
            if v <= threshold_warn:
                return "text-success"
            if v <= threshold_bad:
                return "text-warning"
            return "text-danger"
    except Exception:
        return "text-muted"


def _section_header(icon: str, title: str, explanation: str, color: str = "primary") -> html.Div:
    """Consistent section header: icon + title + 1-line explanation."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon + " ", style={"fontSize": "18px"}),
                    html.Span(title, style={"fontSize": "16px", "fontWeight": "700", "color": "#e8eaf6"}),
                ],
                className="mb-1",
            ),
            html.Div(explanation, style=_S["explain"]),
        ],
        style={
            "borderRight": f"4px solid var(--bs-{color})",
            "paddingRight": "12px",
            "marginBottom": "16px",
            "marginTop": "4px",
        },
    )


def _score_bar(label: str, val: Any, max_val: float, color: str) -> html.Div:
    try:
        pct = min(100.0, max(0.0, float(val or 0) / max_val * 100))
    except Exception:
        pct = 0.0
    return html.Div(
        [
            html.Span(label, style={"fontSize": "11px", "width": "32px", "display": "inline-block", "color": "#aaa"}),
            dbc.Progress(value=pct, color=color, style={"height": "8px", "flex": "1", "display": "inline-flex"}),
            html.Span(
                format_float(val, "{:.0f}"),
                style={"fontSize": "11px", "width": "28px", "textAlign": "right", "color": "#ccc", "fontWeight": "600"},
            ),
        ],
        className="d-flex align-items-center gap-2 mb-1",
    )


def build_market_narrative(master_df: pd.DataFrame) -> dbc.Card:
    """
    One-paragraph plain-language summary of the current market situation.
    Sits at the very top of the Overview tab — the PM reads this first.
    """
    if master_df is None or master_df.empty:
        return html.Div()

    row0 = master_df.iloc[0].to_dict()
    state = str(row0.get("market_state", "—")).upper()
    alert = str(row0.get("regime_alert", "") or "")
    exec_r = str(row0.get("execution_regime", "—"))
    avg_corr = row0.get("avg_corr_t")
    trans_prob = row0.get("transition_probability")
    crisis_prob = row0.get("crisis_probability")
    n_longs  = int((master_df["direction"] == "LONG").sum())  if "direction" in master_df.columns else 0
    n_shorts = int((master_df["direction"] == "SHORT").sum()) if "direction" in master_df.columns else 0
    n_neutral = len(master_df) - n_longs - n_shorts

    color = _regime_color(state)
    explain = _REGIME_EXPLAIN.get(state, "")

    lines = []
    lines.append(
        f"המשטר הנוכחי הוא {state} — {explain}."
    )
    if avg_corr is not None:
        try:
            ac = float(avg_corr)
            level = "גבוהה" if ac > 0.60 else "בינונית" if ac > 0.40 else "נמוכה"
            lines.append(f"קורלציה ממוצעת: {ac:.2f} ({level}) — {'סקטורים נעים יחד, קשה לבדל' if ac > 0.60 else 'יש פיזור בין סקטורים, מתאים ל-RV'}.")
        except Exception:
            pass
    if trans_prob is not None:
        try:
            tp = float(trans_prob)
            if tp > 0.5:
                lines.append(f"⚠️ הסתברות מעבר משטר: {tp:.0%} — שים לב לשינוי צפוי!")
        except Exception:
            pass
    lines.append(
        f"ספר נוכחי: {n_longs} Long | {n_shorts} Short | {n_neutral} Neutral. "
        f"משטר ביצוע: {exec_r}."
    )
    if alert and alert not in ("—", ""):
        lines.append(f"🔔 {alert}")

    narrative = " ".join(lines)

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(state, color=color, className="fs-6 px-3 py-2 me-3"),
                    ], width="auto", className="d-flex align-items-start pt-1"),
                    dbc.Col([
                        html.Div("שורה תחתונה — מצב השוק עכשיו", style={**_S["label"], "marginBottom": "6px"}),
                        html.Div(narrative, style={**_S["explain"], "fontSize": "13px", "lineHeight": "1.8"}),
                    ]),
                ]),
            ],
            style=_S["card_pad"],
        ),
        className=f"border-{color} mb-3",
        style={"borderTop": f"4px solid var(--bs-{color})", "borderWidth": "1px"},
    )


def build_regime_hero(row: Dict[str, Any]) -> dbc.Card:
    """Full-width regime dashboard — 8 metrics with colored indicators and tooltips."""
    state = str(row.get("market_state", "—")).upper()
    color = _regime_color(state)
    exec_regime = str(row.get("execution_regime", "—"))

    def _m(label: str, value: str, tip: str, vc: str = "#e8eaf6",
            pv: Optional[float] = None, pc: str = "secondary") -> dbc.Col:
        inner: list = [
            html.Div(label, style=_S["label"]),
            html.Div(value, style={"fontSize": "16px", "fontWeight": "700", "color": vc, "margin": "3px 0"}),
        ]
        if pv is not None:
            inner.append(dbc.Progress(
                value=min(100.0, max(0.0, pv)),
                color=pc, style={"height": "4px", "opacity": "0.75"},
            ))
        inner.append(html.Div(tip, style={"fontSize": "10px", "color": "#555", "marginTop": "4px", "lineHeight": "1.3"}))
        return dbc.Col(html.Div(inner, style={"padding": "10px 14px", "borderRight": "1px solid #22223a"}))

    def _sf(x, fmt="{:.3f}"):
        try:
            v = float(x)
            return fmt.format(v) if v == v else "—"
        except Exception:
            return "—"

    ac  = _sf(row.get("avg_corr_t"))
    acf = float(row.get("avg_corr_t") or 0)
    ms  = _sf(row.get("market_mode_strength"))
    msf = float(row.get("market_mode_strength") or 0)
    cd  = _sf(row.get("corr_matrix_dist_t"))
    cdf = float(row.get("corr_matrix_dist_t") or 0)
    ts  = _sf(row.get("transition_probability"), "{:.1%}")
    tsf = float(row.get("transition_probability") or 0)
    cp  = _sf(row.get("crisis_probability"), "{:.1%}")
    cpf = float(row.get("crisis_probability") or 0)
    dz  = _sf(row.get("dispersion_ratio_z"), "{:+.2f}")
    dzf = float(row.get("dispersion_ratio_z") or 0)
    acd = _sf(row.get("avg_corr_delta"), "{:+.3f}")
    acdf = float(row.get("avg_corr_delta") or 0)
    vp  = _sf(row.get("vix_percentile"), "{:.0%}")
    vpf = float(row.get("vix_percentile") or 0)

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Badge(state, color=color, className="fs-5 px-3 py-2"),
                    html.Span(
                        f"  {_REGIME_EXPLAIN.get(state, '')}",
                        style={"fontSize": "12px", "color": "#9eaabf", "marginRight": "8px"},
                    ),
                ], className="d-flex align-items-center"),
                dbc.Col([
                    html.Span("ביצוע: ", style=_S["label"]),
                    dbc.Badge(exec_regime, color="info" if exec_regime != "HALTED" else "danger",
                              className="ms-1 fs-6"),
                ], width="auto", className="d-flex align-items-center"),
            ], className="mb-3 align-items-center"),
            dbc.Row([
                _m("Avg Corr (60d)", ac,
                   "ממוצע קורלציות. >0.60 = משטר מתוח",
                   "#ff9944" if acf > 0.60 else "#20c997" if acf < 0.45 else "#e8eaf6",
                   acf * 100, "danger" if acf > 0.60 else "success" if acf < 0.45 else "warning"),
                _m("Mode Strength", ms,
                   "חוזק גורם שיטתי (PC1). >0.30 = שוק נע כיחידה",
                   "#ff9944" if msf > 0.30 else "#e8eaf6",
                   msf / 0.5 * 100, "warning" if msf > 0.30 else "info"),
                _m("Corr Distortion", cd,
                   "סטיית מבנה קורלציות מ-baseline. >0.20 = לא נורמלי",
                   "#ff6666" if cdf > 0.20 else "#e8eaf6",
                   cdf / 0.5 * 100, "danger" if cdf > 0.20 else "secondary"),
                _m("Transition Prob", ts,
                   "הסתברות מעבר משטר. >45% = שים לב",
                   "#ffcc44" if tsf > 0.45 else "#e8eaf6",
                   tsf * 100, "warning" if tsf > 0.45 else "secondary"),
                _m("Crisis Prob", cp,
                   "הסתברות כניסה ל-CRISIS. >15% = מסוכן",
                   "#ff4444" if cpf > 0.15 else "#e8eaf6",
                   cpf * 100, "danger" if cpf > 0.15 else "secondary"),
                _m("Corr Δ (vs 252d)", acd,
                   "שינוי קורלציה מ-baseline. חיובי = מעלה",
                   "#ff9944" if acdf > 0.05 else "#20c997" if acdf < -0.05 else "#e8eaf6"),
                _m("Dispersion Z", dz,
                   "פיזור סקטורים. שלילי = נעים יחד (high corr regime)",
                   "#20c997" if dzf > 1 else "#ffcc44" if dzf < -1 else "#e8eaf6"),
                _m("VIX Pct", vp,
                   "דרגת VIX ביחס ל-5 שנים. >80% = תנודתיות היסטורית גבוהה",
                   "#ff4444" if vpf > 0.80 else "#ffcc44" if vpf > 0.60 else "#20c997",
                   vpf * 100, "danger" if vpf > 0.80 else "warning" if vpf > 0.60 else "success"),
            ], className="g-0"),
        ], style=_S["card_pad"]),
        className=f"border-{color} mb-3",
        style={"borderTop": f"4px solid var(--bs-{color})", "borderWidth": "1px"},
    )


def build_sector_opportunity_card(row: Dict[str, Any]) -> dbc.Card:
    """Spacious sector card with attribution breakdown and key stats."""
    direction = str(row.get("direction", "NEUTRAL")).upper()
    sector_name = str(row.get("sector_name", "—"))
    ticker = str(row.get("sector_ticker", "—"))
    mc = row.get("mc_score", 0)
    conviction = row.get("conviction_score", 0)
    decision = str(row.get("decision_label", "—"))
    size_bucket = str(row.get("size_bucket", "—"))
    pm_note = str(row.get("pm_note", "") or "")
    z_score = row.get("pca_residual_z")
    rel_pe = row.get("rel_pe_vs_spy")
    w_final = row.get("w_final", 0)

    dir_color = _DIR_BADGE.get(direction, "secondary")
    mc_pct = min(100.0, max(0.0, float(mc or 0)))

    def _num(label: str, val: Any, fmt: str, tip: str, vc: str = "#e8eaf6") -> html.Div:
        return html.Div([
            html.Div(label, style=_S["label"]),
            html.Div(format_float(val, fmt), style={"fontSize": "15px", "fontWeight": "700", "color": vc}),
            html.Div(tip, style={"fontSize": "10px", "color": "#555", "lineHeight": "1.2"}),
        ], style={"padding": "4px 0"})

    # Z-score coloring: negative=cheap(good for long), positive=expensive(good for short)
    zf = float(z_score or 0)
    z_color = "#20c997" if (direction == "LONG" and zf < -0.5) or (direction == "SHORT" and zf > 0.5) else "#ffcc44" if abs(zf) < 0.5 else "#ff9944"

    # PE coloring: low PE good for long, high PE good for short
    pef = float(rel_pe or 0)
    pe_color = "#20c997" if (direction == "LONG" and pef < 0) or (direction == "SHORT" and pef > 0) else "#ffcc44"

    return dbc.Card([
        dbc.CardBody([
            # Header
            dbc.Row([
                dbc.Col(dbc.Badge(direction, color=dir_color, className="fs-6 px-3 py-1"), width="auto"),
                dbc.Col([
                    html.Div(sector_name, style={"fontSize": "14px", "fontWeight": "700", "color": "#e8eaf6"}),
                    html.Div(ticker, style={"fontSize": "11px", "color": "#888"}),
                ]),
                dbc.Col(
                    html.Div([
                        html.Div("משקל", style=_S["label"]),
                        html.Div(f"{float(w_final or 0)*100:.1f}%",
                                 style={"fontSize": "14px", "fontWeight": "700",
                                        "color": "#20c997" if direction=="LONG" else "#dc3545"}),
                    ]),
                    width="auto",
                ),
            ], className="mb-3 align-items-center g-2"),

            # MC conviction bar
            html.Div([
                html.Div([
                    html.Span("Mispricing Confidence", style=_S["label"]),
                    html.Span(f" {format_float(mc, '{:.0f}')} / 100",
                              style={"fontSize": "13px", "fontWeight": "700", "marginRight": "6px"}),
                ], className="d-flex justify-content-between mb-1"),
                dbc.Progress(value=mc_pct, color=dir_color, style={"height": "10px"},
                             className="mb-1"),
                html.Div("ציון 0-100: עד כמה האות מהימן ומבוסס על מכלול הנתונים",
                         style={"fontSize": "10px", "color": "#555"}),
            ], className="mb-3"),

            # Attribution breakdown
            html.Div("Attribution — פירוט מקורות האות:", style={**_S["label"], "marginBottom": "6px"}),
            _score_bar("SDS", row.get("sds_score"), 1.0, "info"),
            html.Div("SDS: Statistical Dislocation Score — עוצמת הסטייה הסטטיסטית (PCA residual)",
                     style={"fontSize": "10px", "color": "#444", "marginBottom": "3px", "paddingRight": "34px"}),
            _score_bar("FJS", row.get("fjs_score"), 1.0, "success" if direction == "LONG" else "danger"),
            html.Div("FJS: Fundamental Justification Score — עד כמה הסטייה מוסברת ע\"י ערך פונדמנטלי",
                     style={"fontSize": "10px", "color": "#444", "marginBottom": "3px", "paddingRight": "34px"}),
            _score_bar("MSS", row.get("mss_score"), 1.0, "warning"),
            html.Div("MSS: Macro Shift Score — סיכון מאקרו ושינוי מבנה קורלציות",
                     style={"fontSize": "10px", "color": "#444", "marginBottom": "3px", "paddingRight": "34px"}),
            _score_bar("STF", row.get("stf_score"), 1.0, "secondary"),
            html.Div("STF: Structural Trend Filter — מגמה מבנית בתמחור היחסי (מנטרל mean reversion)",
                     style={"fontSize": "10px", "color": "#444", "marginBottom": "8px", "paddingRight": "34px"}),

            html.Hr(style={"borderColor": "#252540", "margin": "8px 0"}),

            # Key stats row
            dbc.Row([
                dbc.Col(_num("Z-Score (OOS PCA)", z_score, "{:.2f}",
                             "סטיות תקן מהממוצע. <-1 = זול, >+1 = יקר", z_color)),
                dbc.Col(_num("P/E rel to SPY", rel_pe, "{:.2f}",
                             "P/E יחסי. שלילי = זול יחסית לשוק", pe_color)),
                dbc.Col(_num("Conviction", conviction, "{:.0f}",
                             "ציון כולל 0-100 של הממשל")),
            ], className="g-2 mb-2"),

            # Decision + size
            dbc.Row([
                dbc.Col([
                    dbc.Badge(decision, color="light", text_color="dark", className="me-1"),
                    dbc.Badge(size_bucket, color="secondary"),
                ]),
            ], className="mb-2"),

            # PM note
            html.Div(
                f"💬 {pm_note[:90]}{'…' if len(pm_note) > 90 else ''}" if pm_note else "",
                style={"fontSize": "11px", "color": "#7a8bbf", "fontStyle": "italic",
                       "borderRight": "3px solid #333", "paddingRight": "8px"},
            ),
        ], style={"padding": "14px"}),
    ],
        className=f"border-{dir_color} h-100",
        style={"borderTop": f"3px solid var(--bs-{dir_color})", "borderWidth": "1px"},
    )


def build_opportunities_section(master_df: pd.DataFrame) -> html.Div:
    """Top 3 longs + top 3 shorts with section header."""
    if master_df is None or master_df.empty:
        return html.Div()

    longs  = master_df[master_df["direction"] == "LONG"].head(3)  if "direction" in master_df.columns else pd.DataFrame()
    shorts = master_df[master_df["direction"] == "SHORT"].head(3) if "direction" in master_df.columns else pd.DataFrame()

    def _side(df_sub: pd.DataFrame, label: str, icon: str, color: str) -> dbc.Col:
        hdr = html.Div([
            html.Span(icon + " ", style={"fontSize": "16px"}),
            html.Span(label, style={"fontSize": "14px", "fontWeight": "700",
                                    "color": f"var(--bs-{color})"}),
        ], className="mb-2")
        if df_sub.empty:
            return dbc.Col([hdr, html.Div("אין מועמדים", style=_S["muted"])], md=6)
        cards = [dbc.Col(build_sector_opportunity_card(r.to_dict()), md=4) for _, r in df_sub.iterrows()]
        return dbc.Col([hdr, dbc.Row(cards, className="g-2")], md=6)

    return html.Div([
        _section_header("💡", "הזדמנויות מסחר", "סקטורים עם האות החזק ביותר — מוצגים לפי כיוון וחוזק MC", "info"),
        dbc.Row([
            _side(longs,  "מועמדים Long — קנייה", "▲", "success"),
            _side(shorts, "מועמדים Short — מכירה", "▼", "danger"),
        ], className="mb-3 g-3"),
    ])


# Scanner columns shown by default (ordered, with Hebrew labels)
_SCANNER_COLS: List[Tuple[str, str]] = [
    ("sector_name",           "סקטור"),
    ("direction",             "כיוון"),
    ("mc_score",              "MC"),
    ("conviction_score",      "קונביקשן"),
    ("market_state",          "משטר"),
    ("decision_label",        "החלטה"),
    ("pca_residual_z",        "Z-Score"),
    ("rel_pe_vs_spy",         "P/E rel"),
    ("fjs_engine_score",      "FJS"),
    ("fjs_multiple_used",     "Multiple"),
    ("sds_score",             "SDS"),
    ("mss_score",             "MSS"),
    ("stf_score",             "STF"),
    ("beta_tnx_60d",          "β ריבית"),
    ("size_bucket",           "גודל"),
    ("pm_note",               "הערת PM"),
    ("interpretation",        "פרשנות"),
]


def heatmap_fig(mat: Optional[pd.DataFrame], title: str) -> go.Figure:
    if mat is None or mat.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=title, height=500)
        return fig

    z_vals = mat.values.round(3)
    labels = list(mat.columns)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=labels,
            y=list(mat.index),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=True,
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        title=dict(text=title, font=dict(size=13)),
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def build_scanner_table(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, DataTable]:
    # Build display DataFrame with key columns and Hebrew headers
    col_map = {raw: label for raw, label in _SCANNER_COLS if raw in master_df.columns}
    df = master_df[list(col_map.keys())].copy()

    # Always carry sector_ticker in data (needed by tearsheet callback for lookup)
    if "sector_ticker" in master_df.columns:
        df["sector_ticker"] = master_df["sector_ticker"].values

    round_map = {
        "mc_score": 1,
        "conviction_score": 1,
        "pca_residual_z": 2,
        "rel_pe_vs_spy": 2,
        "fjs_engine_score": 2,
        "sds_score": 2,
        "mss_score": 2,
        "stf_score": 2,
        "beta_tnx_60d": 3,
    }
    for c, nd in round_map.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(nd)

    # Rename to Hebrew for display; keep raw key as "id" for callbacks
    display_columns = [{"name": col_map[c], "id": c} for c in col_map]

    table = DataTable(
        id="signals-table",
        columns=display_columns,
        data=df.to_dict("records"),
        hidden_columns=["sector_ticker"] if "sector_ticker" in df.columns else [],
        page_size=12,
        sort_action="native",
        filter_action="native",
        row_selectable="single",
        selected_rows=[0] if len(df) else [],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#111111",
            "color": "#eaeaea",
            "border": "1px solid #2a2a2a",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "12px",
            "padding": "7px 10px",
            "textAlign": "center",
            "minWidth": "70px",
            "maxWidth": "200px",
            "whiteSpace": "normal",
        },
        style_header={
            "backgroundColor": "#1a1a2e",
            "fontWeight": "bold",
            "fontSize": "12px",
            "border": "1px solid #333",
            "color": "#c8d0e0",
        },
        style_data_conditional=[
            # Direction coloring
            {"if": {"filter_query": "{direction} = 'LONG'"},  "backgroundColor": "rgba(32, 201, 151, 0.10)", "borderLeft": "3px solid #20c997"},
            {"if": {"filter_query": "{direction} = 'SHORT'"}, "backgroundColor": "rgba(220, 53, 69, 0.10)",  "borderLeft": "3px solid #dc3545"},
            # MC highlight
            {"if": {"filter_query": "{mc_score} >= 70"}, "fontWeight": "700"},
            # Market state right border
            {"if": {"filter_query": "{market_state} = 'CALM'"},    "borderRight": "3px solid #20c997"},
            {"if": {"filter_query": "{market_state} = 'NORMAL'"},  "borderRight": "3px solid #0d6efd"},
            {"if": {"filter_query": "{market_state} = 'TENSION'"}, "borderRight": "3px solid #ffc107"},
            {"if": {"filter_query": "{market_state} = 'CRISIS'"},  "borderRight": "3px solid #dc3545"},
            # Decision background tint
            {"if": {"filter_query": "{decision_label} contains 'ENTER'"},  "backgroundColor": "rgba(32, 201, 151, 0.12)"},
            {"if": {"filter_query": "{decision_label} contains 'WATCH'"},  "backgroundColor": "rgba(13, 110, 253, 0.10)"},
            {"if": {"filter_query": "{decision_label} contains 'REDUCE'"}, "backgroundColor": "rgba(255, 193, 7, 0.10)"},
            {"if": {"filter_query": "{decision_label} contains 'AVOID'"},  "backgroundColor": "rgba(220, 53, 69, 0.10)"},
        ],
    )
    return df, table

def build_action_plan(master_df: pd.DataFrame) -> html.Div:
    """
    Actionable trade table — מה לקנות/למכור, כמה, ולמה.
    כולל: כיוון, MC, Z-Score, PE rel, גודל מוצע, החלטה, הסבר.
    """
    if master_df is None or master_df.empty:
        return html.Div()

    tradable = (master_df[master_df["direction"].isin(["LONG", "SHORT"])].copy()
                if "direction" in master_df.columns else pd.DataFrame())
    neutral_count = len(master_df) - len(tradable) if "direction" in master_df.columns else 0

    n_long  = int((tradable["direction"] == "LONG").sum())  if not tradable.empty else 0
    n_short = int((tradable["direction"] == "SHORT").sum()) if not tradable.empty else 0

    def _row(r: dict) -> html.Tr:
        direction = str(r.get("direction", ""))
        is_long   = direction == "LONG"
        dir_color = "#20c997" if is_long else "#dc3545"
        bg        = "rgba(32,201,151,0.06)" if is_long else "rgba(220,53,69,0.06)"
        mc        = float(r.get("mc_score", 0) or 0)
        mc_pct    = min(100.0, max(0.0, mc))
        z         = float(r.get("pca_residual_z", 0) or 0)
        pe        = float(r.get("rel_pe_vs_spy", 0) or 0)
        w         = float(r.get("w_final", 0) or 0) * 100
        conv      = float(r.get("conviction_score", 0) or 0)
        decision  = str(r.get("decision_label", "—"))
        pm_note   = str(r.get("pm_note", "") or "")[:55]

        z_col  = "#20c997" if (is_long and z < -0.5) or (not is_long and z > 0.5) else "#ffcc44" if abs(z) < 0.5 else "#ff9944"
        pe_col = "#20c997" if (is_long and pe < 0)   or (not is_long and pe > 0)  else "#ffcc44"

        return html.Tr([
            html.Td([
                html.Strong(r.get("sector_name", "—"),
                            style={"fontSize": "13px", "color": "#e8eaf6"}),
                html.Span(f" ({r.get('sector_ticker','—')})",
                          style={"fontSize": "11px", "color": "#888"}),
            ], style={"textAlign": "right", "paddingRight": "12px"}),
            html.Td(
                dbc.Badge("▲ קנה" if is_long else "▼ מכור",
                          color="success" if is_long else "danger",
                          className="fs-6 px-3"),
                className="text-center",
            ),
            html.Td([
                dbc.Progress(value=mc_pct, color="success" if is_long else "danger",
                             style={"height": "8px", "marginBottom": "3px"}),
                html.Span(f"{mc:.0f}", style={"fontSize": "12px", "fontWeight": "700",
                                               "color": dir_color}),
            ], style={"minWidth": "110px", "verticalAlign": "middle"}),
            html.Td(
                html.Span(f"{z:+.2f}", style={"fontSize": "13px", "fontWeight": "700", "color": z_col}),
                className="text-center",
            ),
            html.Td(
                html.Span(f"{pe:+.2f}", style={"fontSize": "13px", "fontWeight": "700", "color": pe_col}),
                className="text-center",
            ),
            html.Td(
                html.Strong(f"{w:.1f}%", style={"fontSize": "14px", "color": dir_color}),
                className="text-center",
            ),
            html.Td(
                html.Span(f"{conv:.0f}", style={"fontSize": "13px", "color": "#9eaabf"}),
                className="text-center",
            ),
            html.Td(
                dbc.Badge(decision, color="light", text_color="dark", style={"fontSize": "10px"}),
                className="text-center",
            ),
            html.Td(
                html.Span(pm_note, style={"fontSize": "11px", "color": "#7a8bbf", "fontStyle": "italic"}),
                style={"textAlign": "right", "maxWidth": "200px"},
            ),
        ], style={"backgroundColor": bg, "lineHeight": "2.2"})

    rows = [_row(r.to_dict()) for _, r in tradable.iterrows()] if not tradable.empty else []

    if not rows:
        rows = [html.Tr(html.Td("אין עסקאות פעילות כרגע", colSpan=9,
                                 className="text-center text-muted py-3"))]

    th_style = {"backgroundColor": "#12122a", "fontSize": "11px", "color": "#888",
                "textTransform": "uppercase", "letterSpacing": "0.5px",
                "padding": "10px 8px", "fontWeight": "600", "borderBottom": "2px solid #252550"}

    return html.Div([
        _section_header("📋", "תוכנית פעולה",
                        "כל הסקטורים עם אות — מה לקנות, כמה, ולמה. "
                        f"סיכום: {n_long} Long | {n_short} Short | {neutral_count} Neutral",
                        "primary"),
        dbc.Card(dbc.CardBody([
            html.Div(
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("סקטור", style={**th_style, "textAlign": "right"}),
                        html.Th("פעולה", style={**th_style, "textAlign": "center"}),
                        html.Th("MC 0-100", style={**th_style, "textAlign": "center",
                                                   "minWidth": "110px"}),
                        html.Th("Z-Score", style={**th_style, "textAlign": "center"},
                                title="OOS PCA residual z-score. שלילי=זול, חיובי=יקר"),
                        html.Th("P/E rel", style={**th_style, "textAlign": "center"},
                                title="P/E יחסי לSPY. שלילי = זול יחסית"),
                        html.Th("משקל %", style={**th_style, "textAlign": "center"},
                                title="w_final: גודל פוזיציה מוצע לאחר Vol-scaling"),
                        html.Th("קונביקשן", style={**th_style, "textAlign": "center"}),
                        html.Th("החלטה", style={**th_style, "textAlign": "center"}),
                        html.Th("הערת PM", style={**th_style, "textAlign": "right"}),
                    ])),
                    html.Tbody(rows),
                ], className="table table-dark mb-0",
                   style={"fontSize": "13px", "borderCollapse": "separate",
                          "borderSpacing": "0 2px"}),
                style={"overflowX": "auto"},
            ),
            html.Div([
                html.Span("💡 ", style={"fontSize": "13px"}),
                html.Span("MC", style={"fontWeight": "700", "color": "#4da6ff"}),
                html.Span(" = Mispricing Confidence 0-100 | ", style=_S["muted"]),
                html.Span("Z-Score", style={"fontWeight": "700", "color": "#4da6ff"}),
                html.Span(" = OOS PCA residual (שלילי=זול, חיובי=יקר) | ", style=_S["muted"]),
                html.Span("משקל", style={"fontWeight": "700", "color": "#4da6ff"}),
                html.Span(" = w_final לאחר vol-scaling, beta-neutral ו-regime sizing. "
                          "הפעולה הסופית בשיקול דעת ה-PM.", style=_S["muted"]),
            ], style={"marginTop": "10px", "padding": "8px 12px",
                      "backgroundColor": "#12122a", "borderRadius": "6px",
                      "fontSize": "11px", "lineHeight": "1.8"}),
        ], style=_S["card_pad"]), className="mb-3"),
    ])


def build_correlation_summary(master_df: pd.DataFrame, engine_corr: Optional[pd.DataFrame] = None) -> dbc.Card:
    """Compact correlation/relationship summary card."""
    if master_df is None or master_df.empty:
        return dbc.Card(dbc.CardBody("אין נתונים"))

    row0 = master_df.iloc[0].to_dict()

    avg_corr = row0.get("avg_corr_t")
    mode_strength = row0.get("market_mode_strength")
    corr_dist = row0.get("corr_matrix_dist_t")
    avg_corr_delta = row0.get("avg_corr_delta")

    try:
        avg_corr_pct = min(100.0, max(0.0, float(avg_corr or 0) * 100))
        mode_pct     = min(100.0, max(0.0, float(mode_strength or 0) / 0.5 * 100))
        dist_pct     = min(100.0, max(0.0, float(corr_dist or 0) / 0.5 * 100))
    except Exception:
        avg_corr_pct = mode_pct = dist_pct = 0.0

    # Sector-level correlation to market mode
    corr_rows = []
    if "sector_ticker" in master_df.columns and "market_mode_loading" in master_df.columns:
        for _, r in master_df.iterrows():
            loading = r.get("market_mode_loading", 0)
            try:
                loading_pct = min(100.0, max(0.0, abs(float(loading or 0)) / 0.5 * 100))
                bar_color = "warning" if loading_pct > 60 else "info"
            except Exception:
                loading_pct = 0.0
                bar_color = "secondary"
            corr_rows.append(
                html.Tr(
                    [
                        html.Td(str(r.get("sector_ticker", "—")), style={"fontSize": "11px", "textAlign": "right"}),
                        html.Td(
                            dbc.Progress(value=loading_pct, color=bar_color, style={"height": "7px"}),
                            style={"minWidth": "100px"},
                        ),
                        html.Td(format_float(loading, "{:.3f}"), style={"fontSize": "11px", "textAlign": "center"}),
                        html.Td(
                            format_float(r.get("sector_corr_dist_contrib"), "{:.3f}"),
                            style={"fontSize": "11px", "textAlign": "center"},
                        ),
                    ]
                )
            )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("מבנה קורלציות וקשרים בין סקטורים", style={"textAlign": "right"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("ממוצע קורלציה", style={"fontSize": "10px", "color": "#888", "textAlign": "center"}),
                                dbc.Progress(value=avg_corr_pct, color="warning" if avg_corr_pct > 60 else "success", style={"height": "8px", "marginBottom": "3px"}),
                                html.Div(format_float(avg_corr, "{:.2f}"), style={"fontSize": "12px", "fontWeight": "600", "textAlign": "center"}),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Div("עוצמת Market Mode", style={"fontSize": "10px", "color": "#888", "textAlign": "center"}),
                                dbc.Progress(value=mode_pct, color="warning" if mode_pct > 60 else "info", style={"height": "8px", "marginBottom": "3px"}),
                                html.Div(format_float(mode_strength, "{:.2f}"), style={"fontSize": "12px", "fontWeight": "600", "textAlign": "center"}),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Div("עיוות מבנה (ΔC)", style={"fontSize": "10px", "color": "#888", "textAlign": "center"}),
                                dbc.Progress(value=dist_pct, color="danger" if dist_pct > 60 else "secondary", style={"height": "8px", "marginBottom": "3px"}),
                                html.Div(format_float(corr_dist, "{:.2f}"), style={"fontSize": "12px", "fontWeight": "600", "textAlign": "center"}),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div("חשיפת כל סקטור ל-Market Mode (loading) ותרומתו לעיוות הקורלציות:", style={"fontSize": "11px", "color": "#aaa", "textAlign": "right", "marginBottom": "6px"}),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("סקטור", style={"textAlign": "right", "fontSize": "10px"}),
                                    html.Th("חשיפה ל-Mode", style={"fontSize": "10px"}),
                                    html.Th("Loading", style={"textAlign": "center", "fontSize": "10px"}),
                                    html.Th("תרומה לעיוות", style={"textAlign": "center", "fontSize": "10px"}),
                                ],
                                style={"backgroundColor": "#1a1a2e"},
                            )
                        ),
                        html.Tbody(corr_rows),
                    ],
                    className="table table-dark table-sm mb-0",
                    style={"fontSize": "11px"},
                ),
            ]
        ),
        className="mb-3",
    )


def build_stat_analysis_panel(master_df: pd.DataFrame) -> html.Div:
    """
    Statistical analysis panel: Z-scores, beta exposures, and MC confidence per sector.
    Each chart has a plain-language explanation of what it measures and how to use it.
    """
    if master_df is None or master_df.empty:
        return html.Div()

    df = master_df.copy()
    tickers = df["sector_ticker"].tolist() if "sector_ticker" in df.columns else []
    z_scores  = pd.to_numeric(df.get("pca_residual_z",  pd.Series(dtype=float)), errors="coerce").tolist()
    betas_tnx = pd.to_numeric(df.get("beta_tnx_60d",    pd.Series(dtype=float)), errors="coerce").tolist()
    betas_dxy = pd.to_numeric(df.get("beta_dxy_60d",    pd.Series(dtype=float)), errors="coerce").tolist()
    mc_scores  = pd.to_numeric(df.get("mc_score",       pd.Series(dtype=float)), errors="coerce").tolist()

    dir_colors = []
    for _, r in df.iterrows():
        d = str(r.get("direction", "NEUTRAL"))
        dir_colors.append("#20c997" if d == "LONG" else "#dc3545" if d == "SHORT" else "#6c757d")

    # ── Z-Score chart ────────────────────────────────────────────────────────
    z_fig = go.Figure()
    z_fig.add_trace(go.Bar(
        x=tickers, y=z_scores,
        marker_color=dir_colors,
        name="Z-Score",
        hovertemplate="%{x}: %{y:.2f}σ<extra></extra>",
    ))
    z_fig.add_hline(y=2,  line_dash="dash", line_color="#ffc107", annotation_text="+2σ סף LONG")
    z_fig.add_hline(y=-2, line_dash="dash", line_color="#ffc107", annotation_text="-2σ סף SHORT")
    z_fig.add_hline(y=0,  line_color="#444")
    z_fig.update_layout(
        template="plotly_dark", height=300,
        title=dict(text="OOS PCA Z-Score — סטייה מהתמחור ההוגן", font=dict(size=13)),
        margin=dict(l=10, r=10, t=44, b=30), showlegend=False,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(title="σ", titlefont=dict(size=11)),
    )

    # ── Beta chart (TNX + DXY) ───────────────────────────────────────────────
    beta_fig = go.Figure()
    beta_fig.add_trace(go.Bar(x=tickers, y=betas_tnx, name="β ריבית (TNX 10Y)", marker_color="#0dcaf0",
                              hovertemplate="%{x} TNX: %{y:.2f}<extra></extra>"))
    beta_fig.add_trace(go.Bar(x=tickers, y=betas_dxy, name="β דולר (DXY)", marker_color="#fd7e14",
                              hovertemplate="%{x} DXY: %{y:.2f}<extra></extra>"))
    beta_fig.add_hline(y=0, line_color="#444")
    beta_fig.update_layout(
        template="plotly_dark", height=300,
        title=dict(text="חשיפות מאקרו — ריבית (TNX) ודולר (DXY)", font=dict(size=13)),
        barmode="group", margin=dict(l=10, r=10, t=44, b=30),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(title="β", titlefont=dict(size=11)),
    )

    # ── MC Score chart ───────────────────────────────────────────────────────
    mc_fig = go.Figure()
    mc_fig.add_trace(go.Bar(
        x=tickers, y=mc_scores,
        marker_color=dir_colors,
        name="MC Score",
        hovertemplate="%{x}: %{y:.0f}/100<extra></extra>",
    ))
    mc_fig.add_hline(y=60, line_dash="dash", line_color="#20c997",
                     annotation_text="60 — סף High Conviction", annotation_font_size=10)
    mc_fig.add_hline(y=40, line_dash="dot",  line_color="#6c757d",
                     annotation_text="40 — גבול פעולה", annotation_font_size=10)
    mc_fig.update_layout(
        template="plotly_dark", height=300,
        title=dict(text="Mispricing Confidence (MC) — רמת ביטחון בסיגנל", font=dict(size=13)),
        margin=dict(l=10, r=10, t=44, b=30), showlegend=False,
        yaxis_range=[0, 100],
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(title="MC (0-100)", titlefont=dict(size=11)),
    )

    def _chart_card(fig, explain_text: str) -> dbc.Card:
        return dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    html.Div(explain_text, style={**_S["explain"], "marginTop": "8px", "textAlign": "right"}),
                ],
                style={"padding": "12px"},
            ),
            className="mb-3",
            style={"borderColor": "#2a2a3e"},
        )

    return html.Div(
        [
            _section_header(
                "📐", "ניתוח סטטיסטי",
                "שלושה מימדים: עד כמה הסקטור חרג מהתמחור ההוגן (Z), מה חשיפתו לריבית ודולר, ועד כמה הסיגנל מהימן (MC)",
                "#0dcaf0",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _chart_card(
                            z_fig,
                            "Z-Score מחושב כשארית OOS מתוך PCA walk-forward. "
                            "מעל +2σ = הסקטור יקר ביחס לגורמים — סיגנל SHORT. "
                            "מתחת ל-2σ- = הסקטור זול — סיגנל LONG. "
                            "ערכים בין ±1σ הם רעש ואין לסחור בהם.",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _chart_card(
                            beta_fig,
                            "β חיובי ל-TNX = הסקטור מרוויח כשהריבית עולה (בנקים, ביטוח). "
                            "β שלילי = הסקטור נפגע מעלייה בריבית (utilities, נדל\"ן). "
                            "β ל-DXY: שלילי = הסקטור נהנה מדולר חלש (אנרגיה, חומרים).",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        _chart_card(
                            mc_fig,
                            "MC = Mispricing Confidence. ציון 0-100 המשלב עוצמת Z-Score, "
                            "עקביות הסיגנל לאורך זמן, וגודל העיוות ביחס להיסטוריה. "
                            "מעל 60 = High Conviction (סחר בגודל מלא). "
                            "40-60 = פעיל אך צמצם גודל. מתחת ל-40 = המתן.",
                        ),
                        md=4,
                    ),
                ],
            ),
        ]
    )


def build_scatter(master_df: pd.DataFrame) -> go.Figure:
    color_col = "market_state" if "market_state" in master_df.columns else ("direction" if "direction" in master_df.columns else None)
    size_col = "mc_score" if "mc_score" in master_df.columns else None

    hover_cols = [
        c for c in [
            "sector_ticker",
            "conviction_score",
            "mc_score",
            "rel_pe_vs_spy",
            "market_state",
            "regime_transition_score",
            "crisis_probability",
        ]
        if c in master_df.columns
    ]

    fig = px.scatter(
        master_df,
        x="pca_residual_z",
        y="hedge_ratio",
        color=color_col,
        size=size_col,
        hover_name="sector_name" if "sector_name" in master_df.columns else None,
        hover_data=hover_cols,
        title="Cross-Sectional Scanner: Z-Score vs Hedge Ratio",
    )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(zeroline=True)
    return fig

def build_overview_panel(cards_top: dbc.Row, cards_bottom: dbc.Row, summary_table: pd.DataFrame) -> dbc.Container:
    cols = [
        c for c in [
            "sector_name",
            "direction",
            "market_state",
            "conviction_score",
            "mc_score",
            "regime_transition_score",
            "pca_residual_z",
            "rel_pe_vs_spy",
            "w_final",
        ]
        if c in summary_table.columns
    ]

    tbl = DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=summary_table[cols].head(11).to_dict("records"),
        page_size=11,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#111111",
            "color": "#eaeaea",
            "border": "1px solid #333",
            "fontFamily": "monospace",
            "fontSize": "12px",
            "padding": "6px",
        },
        style_header={
            "backgroundColor": "#1f1f1f",
            "fontWeight": "bold",
            "border": "1px solid #444",
        },
    )

    return dbc.Container(
        fluid=True,
        children=[
            cards_top,
            cards_bottom,
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Book Snapshot", className="mb-2"),
                                    html.Div(
                                        "Top cross-sectional view with regime context, MC and portfolio sizing.",
                                        className="text-muted small mb-2",
                                    ),
                                    tbl,
                                ]
                            )
                        ),
                        md=12,
                        className="mt-3",
                    )
                ]
            ),
        ],
    )


def build_scanner_panel(table: DataTable, scatter_fig: go.Figure) -> dbc.Container:
    return dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Cross-Sectional Table", className="mb-2"),
                                    html.Div("Select a row to drive the tear sheet.", className="text-muted small mb-2"),
                                    table,
                                ]
                            )
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Cross-Sectional Scatter", className="mb-2"),
                                    html.Div("Point size reflects MC score.", className="text-muted small mb-2"),
                                    dcc.Graph(id="scatter-z-hr", figure=scatter_fig),
                                ]
                            )
                        ),
                        md=5,
                    ),
                ],
                className="mt-3",
            )
        ],
    )


def build_correlation_panel(
    corr_fig: go.Figure,
    delta_corr_fig: go.Figure,
    corr_ts_fig: go.Figure,
    contrib_fig: go.Figure,
) -> dbc.Container:
    return dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="corr-heatmap", figure=corr_fig)])), md=6, className="mt-3"),
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="delta-corr-heatmap", figure=delta_corr_fig)])), md=6, className="mt-3"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="corr-timeseries", figure=corr_ts_fig)])), md=6, className="mt-3"),
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="corr-contrib", figure=contrib_fig)])), md=6, className="mt-3"),
                ]
            ),
        ],
    )


def build_tearsheet_panel() -> dbc.Container:
    # Sector dropdown for selection (replaces signals-table dependency)
    sector_options = [
        {"label": t, "value": t}
        for t in ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
    ]
    return dbc.Container(
        fluid=True,
        children=[
            dbc.Row([
                dbc.Col([
                    html.Label("בחר סקטור:", className="text-muted mb-1",
                               style={"fontSize": "12px"}),
                    dcc.Dropdown(
                        id="tearsheet-sector-dropdown",
                        options=sector_options,
                        value="XLK",
                        clearable=False,
                        style={"backgroundColor": "#1a1a2e", "color": "#fff"},
                    ),
                ], md=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(dbc.Card(dbc.CardBody(id="card-macro")), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody(id="card-fund")), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody(id="card-attrib")), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody(id="card-exec")), md=3),
                            ], className="mb-3"),
                            dcc.Graph(id="residual-xray"),
                        ])
                    ),
                    md=12,
                ),
            ]),
        ],
    )


def line_figure(df: pd.DataFrame, x: str, y_cols: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for c in y_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[c], mode="lines", name=c))
    fig.update_layout(template="plotly_dark", height=440, title=title, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def bar_figure(x, y, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(template="plotly_dark", height=440, title=title, margin=dict(l=20, r=20, t=55, b=20))
    return fig