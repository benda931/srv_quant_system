"""
ui/scanner_pro.py
─────────────────
Professional Hedge-Fund Grade Scanner — SRV Quantamental DSS.

The Scanner is the PM's command center. It answers 6 questions at a glance:

  1. WHERE are we?     → Market regime context bar
  2. WHAT to trade?    → Signal cards grid, sorted by actionability
  3. HOW confident?    → Attribution breakdown per sector
  4. HOW MUCH?         → Position sizing with portfolio impact
  5. WHAT are risks?   → Risk overlay + Greeks
  6. WHY this trade?   → Statistical + fundamental justification

Design principles:
  - Hebrew RTL throughout
  - Dark institutional theme
  - Visual hierarchy: actionable first, risks prominent
  - At-a-glance: PM should understand the book in <5 seconds
  - No click required for key info — everything visible on scanner
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

# ── Design tokens ────────────────────────────────────────────────────────
_BG = "#0d0d1a"
_CARD_BG = "#111126"
_SURFACE = "#1a1a2e"
_BORDER = "#252540"
_TEXT = "#e8eaf6"
_TEXT_DIM = "#8892b0"
_TEXT_MUTED = "#5a6480"
_ACCENT_BLUE = "#00b4d8"
_ACCENT_GREEN = "#20c997"
_ACCENT_RED = "#dc3545"
_ACCENT_GOLD = "#ffd700"
_ACCENT_ORANGE = "#ff9800"

_RTL = {"direction": "rtl", "textAlign": "right"}

_REGIME_COLORS = {"CALM": "#4caf50", "NORMAL": "#2196f3", "TENSION": "#ff9800", "CRISIS": "#f44336"}
_REGIME_HEB = {"CALM": "רגוע", "NORMAL": "רגיל", "TENSION": "מתח", "CRISIS": "משבר"}
_DIR_COLORS = {"LONG": _ACCENT_GREEN, "SHORT": _ACCENT_RED, "NEUTRAL": _TEXT_MUTED}
_DIR_HEB = {"LONG": "▲ לונג", "SHORT": "▼ שורט", "NEUTRAL": "— ניטרלי"}

_DECISION_COLORS = {
    "ENTER": _ACCENT_GREEN, "WATCH": _ACCENT_BLUE,
    "REDUCE": _ACCENT_ORANGE, "AVOID": _ACCENT_RED,
}


def _ff(x: Any, fmt: str = "{:.2f}") -> str:
    try:
        v = float(x)
        return fmt.format(v) if math.isfinite(v) else "—"
    except Exception:
        return "—"


def _sf(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _build_decision_why(
    direction: str, decision: str, conviction: float, z: float, row: dict,
) -> list:
    """Return a list with 0-or-1 html.Div explaining WHY the decision is AVOID / NEUTRAL."""
    reasons: list = []
    regime = str(row.get("market_state", "")).upper()
    _CONV_THRESHOLD = 30  # conviction below this → not actionable

    if decision in ("AVOID", "REDUCE") or direction == "NEUTRAL":
        if regime == "CRISIS":
            reasons.append("regime CRISIS \u2014 \u05d0\u05d9\u05df \u05de\u05e1\u05d7\u05e8")
        if direction == "NEUTRAL":
            reasons.append(f"\u05d0\u05d9\u05df \u05e1\u05d8\u05d9\u05d9\u05d4 \u05de\u05e1\u05e4\u05e7\u05ea (|z|={abs(z):.2f})")
        if conviction < _CONV_THRESHOLD:
            reasons.append(f"conviction \u05e0\u05de\u05d5\u05da ({conviction:.0f} < {_CONV_THRESHOLD})")

    if not reasons:
        return []

    return [html.Div(
        " | ".join(reasons),
        style={"fontSize": "9px", "color": _ACCENT_ORANGE, "lineHeight": "1.3",
               "marginBottom": "4px", "paddingRight": "2px", **_RTL},
    )]


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1: MARKET REGIME COMMAND BAR
# ═════════════════════════════════════════════════════════════════════════

def _build_regime_bar(row0: dict) -> html.Div:
    """Top-level market regime context bar — always visible."""
    state = str(row0.get("market_state", "UNKNOWN"))
    color = _REGIME_COLORS.get(state, "#666")
    crisis_p = _sf(row0.get("crisis_probability"))
    trans_p = _sf(row0.get("transition_probability"))
    vix = _sf(row0.get("vix_level"))
    avg_corr = _sf(row0.get("avg_corr_t"))
    mode = _sf(row0.get("market_mode_strength"))
    exec_regime = str(row0.get("execution_regime", "—"))

    def _metric(label, value, fmt="{:.2f}", warn_above=None):
        v = _sf(value)
        v_str = fmt.format(v) if math.isfinite(v) else "—"
        c = _TEXT
        if warn_above and math.isfinite(v) and v > warn_above:
            c = _ACCENT_ORANGE if v < warn_above * 1.3 else _ACCENT_RED
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": _TEXT_MUTED,
                                   "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(v_str, style={"fontSize": "16px", "fontWeight": "700", "color": c}),
        ], style={"textAlign": "center", "minWidth": "65px"})

    return html.Div([
        # Regime badge
        html.Div([
            html.Div(_REGIME_HEB.get(state, state),
                     style={"fontSize": "18px", "fontWeight": "800", "color": "white"}),
            html.Div(state, style={"fontSize": "10px", "color": "rgba(255,255,255,0.6)"}),
        ], style={"backgroundColor": color, "padding": "8px 20px", "borderRadius": "8px",
                  "textAlign": "center", "minWidth": "90px"}),

        html.Div(style={"width": "1px", "backgroundColor": _BORDER, "margin": "0 12px"}),

        # Metrics row
        _metric("VIX", vix, "{:.1f}", warn_above=25),
        _metric("קורלציה", avg_corr, "{:.2f}", warn_above=0.55),
        _metric("מרקט מוד", mode, "{:.3f}", warn_above=0.28),
        _metric("מעבר", trans_p, "{:.0%}", warn_above=0.45),

        html.Div(style={"width": "1px", "backgroundColor": _BORDER, "margin": "0 12px"}),

        # Crisis gauge
        html.Div([
            html.Div("הסתברות משבר", style={"fontSize": "9px", "color": _TEXT_MUTED, "textTransform": "uppercase"}),
            html.Div([
                dbc.Progress(
                    value=crisis_p * 100, max=100,
                    color="danger" if crisis_p > 0.6 else "warning" if crisis_p > 0.3 else "success",
                    style={"height": "6px", "width": "80px", "backgroundColor": "#1a1a2e"},
                ),
                html.Span(f"{crisis_p:.0%}",
                          style={"fontSize": "14px", "fontWeight": "700", "marginRight": "6px",
                                 "color": _ACCENT_RED if crisis_p > 0.6 else _ACCENT_ORANGE if crisis_p > 0.3 else _ACCENT_GREEN}),
            ], className="d-flex align-items-center gap-2"),
        ]),

        html.Div(style={"width": "1px", "backgroundColor": _BORDER, "margin": "0 12px"}),

        _metric("ביצוע", exec_regime, "{}"),

    ], className="d-flex align-items-center gap-3",
       style={"backgroundColor": _SURFACE, "padding": "10px 16px", "borderRadius": "8px",
              "border": f"1px solid {_BORDER}", "marginBottom": "12px", "overflowX": "auto"})


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2: PORTFOLIO EXPOSURE SNAPSHOT
# ═════════════════════════════════════════════════════════════════════════

def _build_portfolio_bar(master_df: pd.DataFrame) -> html.Div:
    """Portfolio-level exposure bar — gross/net, Greeks, risk."""
    if master_df is None or master_df.empty:
        return html.Div()

    row0 = master_df.iloc[0]

    def _exp(label, val, fmt="{:+.2f}", color=None):
        v_str = fmt.format(_sf(val))
        c = color or _TEXT
        return html.Div([
            html.Span(label, style={"fontSize": "9px", "color": _TEXT_MUTED, "marginLeft": "4px"}),
            html.Span(v_str, style={"fontSize": "13px", "fontWeight": "700", "color": c}),
        ], style={"textAlign": "center"})

    n_long = int((master_df["direction"] == "LONG").sum()) if "direction" in master_df.columns else 0
    n_short = int((master_df["direction"] == "SHORT").sum()) if "direction" in master_df.columns else 0
    gross = _sf(row0.get("gross_exposure"))
    net = _sf(row0.get("net_exposure"))
    d_spy = _sf(row0.get("delta_spy_P"))
    d_tnx = _sf(row0.get("delta_tnx_P"))
    gamma = _sf(row0.get("gamma_synth_P"))
    vega = _sf(row0.get("vega_synth_P"))
    port_vol = _sf(row0.get("port_vol"))

    return html.Div([
        html.Div([
            html.Span(f"{n_long}", style={"color": _ACCENT_GREEN, "fontWeight": "800", "fontSize": "16px"}),
            html.Span(" L ", style={"color": _TEXT_MUTED, "fontSize": "11px"}),
            html.Span(f"{n_short}", style={"color": _ACCENT_RED, "fontWeight": "800", "fontSize": "16px"}),
            html.Span(" S", style={"color": _TEXT_MUTED, "fontSize": "11px"}),
        ]),
        _exp("Gross", gross, "{:.1%}"),
        _exp("Net", net, "{:+.1%}", _ACCENT_GREEN if abs(net) < 0.05 else _ACCENT_ORANGE),
        html.Div(style={"width": "1px", "backgroundColor": _BORDER}),
        _exp("Δ SPY", d_spy, "{:+.3f}"),
        _exp("Δ TNX", d_tnx, "{:+.3f}"),
        _exp("Γ", gamma, "{:.3f}"),
        _exp("Vega", vega, "{:.3f}"),
        html.Div(style={"width": "1px", "backgroundColor": _BORDER}),
        _exp("Vol", port_vol, "{:.1%}" if port_vol < 1 else "{:.1f}"),
    ], className="d-flex align-items-center gap-3 justify-content-center",
       style={"backgroundColor": _CARD_BG, "padding": "8px 16px", "borderRadius": "6px",
              "border": f"1px solid {_BORDER}", "marginBottom": "12px"})


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3: SIGNAL CARDS GRID
# ═════════════════════════════════════════════════════════════════════════

def _conviction_ring(score: float, size: int = 48) -> html.Div:
    """Circular conviction gauge 0-100."""
    pct = max(0, min(100, _sf(score)))
    if pct >= 70:
        color = _ACCENT_GREEN
    elif pct >= 40:
        color = _ACCENT_BLUE
    else:
        color = _TEXT_MUTED

    # CSS conic gradient ring
    return html.Div(
        html.Div(f"{pct:.0f}", style={
            "fontSize": f"{size // 4}px", "fontWeight": "800", "color": color,
            "lineHeight": f"{size - 8}px", "textAlign": "center",
        }),
        style={
            "width": f"{size}px", "height": f"{size}px",
            "borderRadius": "50%",
            "background": f"conic-gradient({color} {pct * 3.6}deg, {_BORDER} {pct * 3.6}deg)",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "padding": "4px",
        },
    )


def _mini_bar(label: str, val: float, color: str, max_val: float = 1.0) -> html.Div:
    """Tiny attribution bar."""
    pct = max(0, min(100, _sf(val) / max_val * 100))
    return html.Div([
        html.Span(label, style={"fontSize": "8px", "color": _TEXT_MUTED, "width": "24px",
                                "display": "inline-block", "textAlign": "left"}),
        html.Div(style={
            "height": "4px", "width": f"{pct}%", "backgroundColor": color,
            "borderRadius": "2px", "flex": "1", "maxWidth": "60px",
        }),
    ], className="d-flex align-items-center gap-1", style={"marginBottom": "1px"})


def _build_sector_card(row: dict, rank: int) -> dbc.Col:
    """Single sector signal card — rich, compact, institutional."""
    direction = str(row.get("direction", "NEUTRAL"))
    decision = str(row.get("decision_label", "—"))
    sector_name = str(row.get("sector_name", "—"))
    ticker = str(row.get("sector_ticker", "—"))
    conviction = _sf(row.get("conviction_score"))
    mc = _sf(row.get("mc_score"))
    z = _sf(row.get("pca_residual_z"))
    w_final = _sf(row.get("w_final"))
    pe_rel = _sf(row.get("rel_pe_vs_spy"))
    size_bucket = str(row.get("size_bucket", "—"))
    entry_quality = str(row.get("entry_quality", "—"))
    interpretation = str(row.get("interpretation", ""))[:80]
    action_bias = str(row.get("action_bias", "—"))
    pm_note = str(row.get("pm_note", ""))[:60]
    half_life = _sf(row.get("half_life_days_est"))
    risk_label = str(row.get("risk_label", "—"))

    # Attribution scores
    sds = _sf(row.get("sds_score"))
    fjs = _sf(row.get("fjs_score"))
    mss = _sf(row.get("mss_score"))
    stf = _sf(row.get("stf_score"))

    # Colors
    dir_color = _DIR_COLORS.get(direction, _TEXT_MUTED)
    dec_color = _DECISION_COLORS.get(decision.split()[0] if decision else "", _TEXT_MUTED)
    border_color = dir_color if direction != "NEUTRAL" else _BORDER

    # Card border intensity based on conviction
    border_width = "3px" if conviction >= 60 else "2px" if conviction >= 30 else "1px"

    card_content = html.Div([
        # ── Header: Sector name + Direction badge ────────────────────────
        html.Div([
            html.Div([
                html.Span(f"#{rank}", style={"fontSize": "10px", "color": _TEXT_MUTED, "marginLeft": "6px"}),
                html.Strong(sector_name, style={"fontSize": "14px", "color": _TEXT}),
                html.Span(f" {ticker}", style={"fontSize": "11px", "color": _TEXT_MUTED}),
            ]),
            html.Div([
                html.Span(
                    _DIR_HEB.get(direction, direction),
                    style={"fontSize": "11px", "fontWeight": "700", "color": "white",
                           "backgroundColor": dir_color, "padding": "2px 8px",
                           "borderRadius": "4px"},
                ),
            ]),
        ], className="d-flex justify-content-between align-items-center", style={"marginBottom": "8px"}),

        # ── Main metrics row: Conviction ring + MC + Z-Score + Weight ────
        html.Div([
            # Conviction ring
            html.Div([
                _conviction_ring(conviction, 50),
                html.Div("קונביקשן", style={"fontSize": "8px", "color": _TEXT_MUTED, "textAlign": "center"}),
            ]),

            # MC gauge
            html.Div([
                html.Div(f"{mc:.0f}", style={"fontSize": "20px", "fontWeight": "800",
                                              "color": _ACCENT_GREEN if mc >= 50 else _ACCENT_BLUE if mc >= 20 else _TEXT_MUTED}),
                html.Div("MC", style={"fontSize": "8px", "color": _TEXT_MUTED}),
            ], style={"textAlign": "center"}),

            # Z-Score
            html.Div([
                html.Div(f"{z:+.2f}", style={
                    "fontSize": "16px", "fontWeight": "700",
                    "color": _ACCENT_GREEN if (direction == "LONG" and z < -0.5) or (direction == "SHORT" and z > 0.5) else _ACCENT_ORANGE,
                }),
                html.Div("Z-Score", style={"fontSize": "8px", "color": _TEXT_MUTED}),
            ], style={"textAlign": "center"}),

            # Weight
            html.Div([
                html.Div(f"{w_final * 100:+.1f}%", style={
                    "fontSize": "16px", "fontWeight": "700", "color": dir_color,
                }),
                html.Div("משקל", style={"fontSize": "8px", "color": _TEXT_MUTED}),
            ], style={"textAlign": "center"}),
        ], className="d-flex justify-content-around align-items-center",
           style={"marginBottom": "8px", "padding": "4px 0"}),

        # ── Attribution mini-bars ────────────────────────────────────────
        html.Div([
            _mini_bar("SDS", sds, _ACCENT_BLUE),
            _mini_bar("FJS", fjs, _ACCENT_GREEN if direction == "LONG" else _ACCENT_RED),
            _mini_bar("MSS", mss, _ACCENT_ORANGE),
            _mini_bar("STF", stf, _TEXT_MUTED),
        ], style={"marginBottom": "6px"}),

        # ── Decision + Sizing row ────────────────────────────────────────
        html.Div([
            html.Span(decision,
                      style={"fontSize": "10px", "fontWeight": "700", "color": "white",
                             "backgroundColor": dec_color, "padding": "1px 6px",
                             "borderRadius": "3px"}),
            html.Span(f"P/E rel: {pe_rel:+.2f}" if math.isfinite(pe_rel) else "",
                      style={"fontSize": "10px", "color": _TEXT_MUTED, "marginRight": "8px"}),
            html.Span(f"גודל: {size_bucket}", style={"fontSize": "10px", "color": _TEXT_DIM}),
            html.Span(f"HL: {half_life:.0f}d" if math.isfinite(half_life) and half_life > 0 else "",
                      style={"fontSize": "10px", "color": _TEXT_MUTED, "marginRight": "8px"}),
        ], className="d-flex flex-wrap gap-2 align-items-center", style={"marginBottom": "4px"}),

        # ── WHY explanation for AVOID / NEUTRAL decisions ─────────────
        *(_build_decision_why(direction, decision, conviction, z, row)),

        # ── Risk label ───────────────────────────────────────────────────
        html.Div([
            html.Span(risk_label,
                      style={"fontSize": "9px",
                             "color": _ACCENT_RED if "Elevated" in risk_label else _ACCENT_ORANGE if "Moderate" in risk_label else _TEXT_MUTED}),
        ], style={"marginBottom": "4px"}) if risk_label and risk_label != "—" else html.Div(),

        # ── Interpretation + PM note ─────────────────────────────────────
        html.Div(interpretation,
                 style={"fontSize": "10px", "color": _TEXT_DIM, "lineHeight": "1.4",
                        "borderTop": f"1px solid {_BORDER}", "paddingTop": "4px",
                        **_RTL}) if interpretation else html.Div(),
        html.Div(pm_note,
                 style={"fontSize": "9px", "color": "#6a79a0", "fontStyle": "italic",
                        "marginTop": "2px", **_RTL}) if pm_note else html.Div(),

    ], style={"padding": "12px"})

    return dbc.Col(
        html.Div(card_content, style={
            "backgroundColor": _CARD_BG,
            "border": f"{border_width} solid {border_color}",
            "borderRadius": "8px",
            "height": "100%",
            "transition": "border-color 0.2s",
        }),
        xs=12, sm=6, lg=4, xl=3,
        className="mb-3",
    )


# ═════════════════════════════════════════════════════════════════════════
# SECTION 4: CROSS-SECTIONAL INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════

def _build_cross_section(master_df: pd.DataFrame) -> html.Div:
    """Cross-sectional scatter + conviction distribution."""
    if master_df is None or master_df.empty:
        return html.Div()

    # ── Scatter: Z-Score (x) vs MC (y), size=conviction, color=direction ─
    fig_scatter = go.Figure()

    for direction, color in [("LONG", _ACCENT_GREEN), ("SHORT", _ACCENT_RED), ("NEUTRAL", _TEXT_MUTED)]:
        mask = master_df["direction"] == direction
        subset = master_df[mask]
        if subset.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=subset["pca_residual_z"],
            y=subset["mc_score"],
            mode="markers+text",
            name=_DIR_HEB.get(direction, direction),
            text=subset["sector_name"].str[:8] if "sector_name" in subset.columns else None,
            textposition="top center",
            textfont=dict(size=9, color=_TEXT_DIM),
            marker=dict(
                size=subset["conviction_score"].clip(10, 80) / 3 + 5,
                color=color,
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Z-Score: %{x:.2f}<br>"
                "MC: %{y:.0f}<br>"
                "<extra></extra>"
            ),
        ))

    fig_scatter.add_vline(x=0, line_dash="dot", line_color=_TEXT_MUTED, opacity=0.3)
    fig_scatter.add_hline(y=50, line_dash="dot", line_color=_ACCENT_BLUE, opacity=0.2)
    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_CARD_BG,
        title=dict(text="Z-Score vs MC — מפת הזדמנויות", font=dict(size=12)),
        xaxis=dict(title="PCA Residual Z-Score", zeroline=True),
        yaxis=dict(title="Mispricing Confidence (0-100)"),
        height=300, margin=dict(l=50, r=20, t=35, b=40),
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10)),
    )

    # ── Conviction bar chart ─────────────────────────────────────────────
    sorted_df = master_df.sort_values("conviction_score", ascending=True)
    directions = sorted_df["direction"].tolist() if "direction" in sorted_df.columns else []
    colors = [_DIR_COLORS.get(d, _TEXT_MUTED) for d in directions]

    fig_conv = go.Figure(data=go.Bar(
        y=sorted_df["sector_name"] if "sector_name" in sorted_df.columns else sorted_df.index,
        x=sorted_df["conviction_score"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in sorted_df["conviction_score"]],
        textposition="auto",
        textfont=dict(size=10, color="white"),
    ))
    fig_conv.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_CARD_BG,
        title=dict(text="דירוג קונביקשן", font=dict(size=12)),
        xaxis=dict(title="Conviction Score (0-100)", range=[0, 100]),
        height=300, margin=dict(l=90, r=20, t=35, b=40),
    )

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}),
        ), style={"backgroundColor": _CARD_BG, "border": f"1px solid {_BORDER}"}), width=7),
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig_conv, config={"displayModeBar": False}),
        ), style={"backgroundColor": _CARD_BG, "border": f"1px solid {_BORDER}"}), width=5),
    ], className="mb-3 g-2")


# ═════════════════════════════════════════════════════════════════════════
# SECTION 5: TRADE PLAN TABLE (institutional format)
# ═════════════════════════════════════════════════════════════════════════

def _build_trade_plan(master_df: pd.DataFrame) -> html.Div:
    """Professional trade plan — actionable signals only."""
    if master_df is None or master_df.empty:
        return html.Div()

    tradable = master_df[master_df["direction"].isin(["LONG", "SHORT"])].copy() \
        if "direction" in master_df.columns else pd.DataFrame()

    if tradable.empty:
        return html.Div(
            dbc.Alert("אין אותות פעילים כרגע — כל הסקטורים NEUTRAL.", color="secondary"),
        )

    tradable = tradable.sort_values("conviction_score", ascending=False)

    rows = []
    for _, r in tradable.iterrows():
        d = r.to_dict()
        direction = str(d.get("direction", ""))
        is_long = direction == "LONG"
        dc = _ACCENT_GREEN if is_long else _ACCENT_RED
        mc = _sf(d.get("mc_score"))
        z = _sf(d.get("pca_residual_z"))
        w = _sf(d.get("w_final")) * 100
        conv = _sf(d.get("conviction_score"))
        pe = _sf(d.get("rel_pe_vs_spy"))
        decision = str(d.get("decision_label", "—"))
        note = str(d.get("pm_note", ""))[:50]
        sds = _sf(d.get("sds_score"))
        fjs = _sf(d.get("fjs_score"))

        # Score composition mini visual
        stat_pts = _sf(d.get("score_stat"))
        macro_pts = _sf(d.get("score_macro"))
        fund_pts = _sf(d.get("score_fund"))
        vol_pts = _sf(d.get("score_vol"))

        score_visual = html.Div([
            html.Div(style={"width": f"{stat_pts}%", "height": "4px", "backgroundColor": _ACCENT_BLUE, "display": "inline-block"}),
            html.Div(style={"width": f"{macro_pts}%", "height": "4px", "backgroundColor": _ACCENT_ORANGE, "display": "inline-block"}),
            html.Div(style={"width": f"{fund_pts}%", "height": "4px", "backgroundColor": _ACCENT_GREEN, "display": "inline-block"}),
            html.Div(style={"width": f"{vol_pts}%", "height": "4px", "backgroundColor": "#9c27b0", "display": "inline-block"}),
        ], style={"width": "100px", "borderRadius": "2px", "overflow": "hidden", "display": "inline-flex"})

        rows.append(html.Tr([
            html.Td(html.Div([
                html.Strong(str(d.get("sector_name", "—")), style={"color": _TEXT}),
                html.Div(str(d.get("sector_ticker", "")), style={"fontSize": "10px", "color": _TEXT_MUTED}),
            ]), style={**_RTL}),
            html.Td(html.Span(
                "▲ לונג" if is_long else "▼ שורט",
                style={"color": "white", "backgroundColor": dc, "padding": "2px 8px",
                       "borderRadius": "4px", "fontSize": "11px", "fontWeight": "700"},
            ), className="text-center"),
            html.Td(html.Div([
                html.Div(f"{w:+.1f}%", style={"fontSize": "15px", "fontWeight": "800", "color": dc}),
            ]), className="text-center"),
            html.Td(html.Div([
                html.Div(f"{conv:.0f}", style={"fontSize": "14px", "fontWeight": "700", "color": _TEXT}),
                score_visual,
            ]), className="text-center"),
            html.Td(f"{mc:.0f}", className="text-center",
                     style={"fontWeight": "700", "color": _ACCENT_GREEN if mc >= 50 else _TEXT_DIM}),
            html.Td(f"{z:+.2f}", className="text-center",
                     style={"fontWeight": "600", "fontSize": "13px"}),
            html.Td(f"{pe:+.2f}" if math.isfinite(pe) else "—", className="text-center",
                     style={"color": _ACCENT_GREEN if (is_long and pe < 0) or (not is_long and pe > 0) else _TEXT_DIM}),
            html.Td(html.Span(decision,
                              style={"fontSize": "10px", "fontWeight": "600",
                                     "color": _DECISION_COLORS.get(decision.split()[0] if decision else "", _TEXT_MUTED)}),
                     className="text-center"),
            html.Td(note, style={"fontSize": "10px", "color": _TEXT_DIM, "maxWidth": "180px", **_RTL}),
        ], style={"borderBottom": f"1px solid {_BORDER}"}))

    th_style = {"backgroundColor": "#0a0a1a", "fontSize": "10px", "color": _TEXT_MUTED,
                "textTransform": "uppercase", "letterSpacing": "0.5px",
                "padding": "8px 6px", "fontWeight": "600", "borderBottom": f"2px solid {_BORDER}"}

    return html.Div([
        html.Div([
            html.Span("📋 ", style={"fontSize": "16px"}),
            html.Strong("תוכנית פעולה", style={"fontSize": "15px", "color": _TEXT}),
            html.Span(f" — {len(tradable)} סקטורים פעילים",
                      style={"fontSize": "12px", "color": _TEXT_MUTED, "marginRight": "8px"}),
        ], style={"marginBottom": "8px", **_RTL}),
        html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th("סקטור", style={**th_style, **_RTL}),
                    html.Th("כיוון", style={**th_style, "textAlign": "center"}),
                    html.Th("משקל", style={**th_style, "textAlign": "center"}),
                    html.Th("קונביקשן", style={**th_style, "textAlign": "center"}),
                    html.Th("MC", style={**th_style, "textAlign": "center"}),
                    html.Th("Z", style={**th_style, "textAlign": "center"}),
                    html.Th("P/E rel", style={**th_style, "textAlign": "center"}),
                    html.Th("החלטה", style={**th_style, "textAlign": "center"}),
                    html.Th("הערה", style={**th_style, **_RTL}),
                ])),
                html.Tbody(rows),
            ], className="table mb-0", style={"fontSize": "12px", "color": _TEXT}),
            style={"overflowX": "auto"},
        ),
        # Legend
        html.Div([
            html.Span("קונביקשן = ", style={"fontSize": "9px", "color": _TEXT_MUTED}),
            html.Span("■", style={"color": _ACCENT_BLUE, "fontSize": "9px"}), html.Span(" סטטיסטי ", style={"fontSize": "9px", "color": _TEXT_MUTED}),
            html.Span("■", style={"color": _ACCENT_ORANGE, "fontSize": "9px"}), html.Span(" מאקרו ", style={"fontSize": "9px", "color": _TEXT_MUTED}),
            html.Span("■", style={"color": _ACCENT_GREEN, "fontSize": "9px"}), html.Span(" פנדמנטלי ", style={"fontSize": "9px", "color": _TEXT_MUTED}),
            html.Span("■", style={"color": "#9c27b0", "fontSize": "9px"}), html.Span(" תנודתיות", style={"fontSize": "9px", "color": _TEXT_MUTED}),
        ], className="mt-1"),
    ], style={"backgroundColor": _CARD_BG, "padding": "12px", "borderRadius": "8px",
              "border": f"1px solid {_BORDER}"})


# ═════════════════════════════════════════════════════════════════════════
# SECTION 6: VISION STATEMENT
# ═════════════════════════════════════════════════════════════════════════

def _build_vision_footer() -> html.Div:
    """The SRV DSS vision statement."""
    return html.Div([
        html.Div([
            html.Div("SRV Quantamental Decision Support System",
                     style={"fontSize": "11px", "fontWeight": "700", "color": _ACCENT_BLUE,
                            "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.Div(
                "מערכת תמיכה בהחלטות לניהול תיקי סקטורים מוסדי — "
                "משלבת ניתוח סטטיסטי (PCA mean-reversion), ניתוח פנדמנטלי (FJS multi-factor), "
                "מודל מאקרו (regime-aware), ניהול סיכונים (VaR/CVaR/Greeks) "
                "ואופטימיזציה (risk-parity / mean-variance) "
                "לייצור ספר Delta-1 market-neutral sector rotation.",
                style={"fontSize": "10px", "color": _TEXT_MUTED, "lineHeight": "1.5", "marginTop": "4px", **_RTL},
            ),
        ], style={"textAlign": "center", "padding": "8px 0"}),
    ], style={"borderTop": f"1px solid {_BORDER}", "marginTop": "16px", "paddingTop": "8px"})


# ═════════════════════════════════════════════════════════════════════════
# MASTER BUILDER: build_scanner_pro()
# ═════════════════════════════════════════════════════════════════════════

def build_scanner_pro(master_df: pd.DataFrame) -> html.Div:
    """
    Build the complete professional Scanner panel.

    The Scanner is the PM's command center for the SRV DSS.
    It provides a complete institutional-grade view of the sector rotation book,
    answering all 6 questions at a glance.
    """
    if master_df is None or master_df.empty:
        return html.Div(dbc.Alert("אין נתונים זמינים.", color="warning"))

    row0 = master_df.iloc[0].to_dict()

    # Sort sectors: ENTER first, then by conviction desc
    decision_order = {"ENTER": 0, "WATCH": 1, "REDUCE": 2, "AVOID": 3}
    sort_df = master_df.copy()
    if "decision_label" in sort_df.columns:
        sort_df["_dec_rank"] = sort_df["decision_label"].apply(
            lambda x: decision_order.get(str(x).split()[0] if x else "", 9)
        )
        sort_df = sort_df.sort_values(["_dec_rank", "conviction_score"], ascending=[True, False])

    # Build signal cards
    signal_cards = []
    for rank, (_, row) in enumerate(sort_df.iterrows(), 1):
        signal_cards.append(_build_sector_card(row.to_dict(), rank))

    return html.Div([
        # ── Header ───────────────────────────────────────────────────────
        html.Div([
            html.H4("🔍 Scanner — מרכז פיקוד", style={"color": _TEXT, "fontWeight": "800", "marginBottom": "2px"}),
            html.Div("סריקת סקטורים רוחבית | כל המידע הדרוש לקבלת החלטה — במבט אחד",
                     style={"fontSize": "12px", "color": _TEXT_MUTED, **_RTL}),
        ], style={"marginBottom": "12px", **_RTL}),

        # ── Section 1: Regime bar ────────────────────────────────────────
        _build_regime_bar(row0),

        # ── Section 2: Portfolio exposure ─────────────────────────────────
        _build_portfolio_bar(master_df),

        # ── Section 3: Signal cards grid ─────────────────────────────────
        html.Div([
            html.Div([
                html.Span("📡 ", style={"fontSize": "14px"}),
                html.Strong("אותות סקטוריים", style={"fontSize": "14px", "color": _TEXT}),
                html.Span(f" — {len(sort_df)} סקטורים, מדורגים לפי פעילות",
                          style={"fontSize": "11px", "color": _TEXT_MUTED, "marginRight": "8px"}),
            ], style={**_RTL, "marginBottom": "8px"}),
            dbc.Row(signal_cards, className="g-2"),
        ], style={"marginBottom": "16px"}),

        # ── Section 4: Cross-sectional intelligence ──────────────────────
        _build_cross_section(master_df),

        # ── Section 5: Trade plan ────────────────────────────────────────
        _build_trade_plan(master_df),

        # ── Vision footer ────────────────────────────────────────────────
        _build_vision_footer(),

    ], style={"padding": "8px", "backgroundColor": _BG, "minHeight": "100vh"})
