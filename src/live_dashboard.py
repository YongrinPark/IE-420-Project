"""
live_dashboard.py
=================
Professional real-time VaR web dashboard built with Plotly Dash +
Bootstrap (DARKLY theme).

The page auto-refreshes every 60 seconds via dcc.Interval.
All data comes from the LiveVaREngine that runs in a background thread.

Launch
------
    python run_live.py
Then open  http://127.0.0.1:8050  in any browser.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from src.live_data import market_status_label, is_market_open

# Colour palette (works on dark background)
_CLR = {
    "green":   "#00d97e",
    "red":     "#e63757",
    "yellow":  "#f6c343",
    "blue":    "#2c7be5",
    "purple":  "#6f42c1",
    "orange":  "#fd7e14",
    "gray":    "#adb5bd",
    "bg":      "#12263f",    # card background
    "panel":   "#152e4d",    # slightly lighter panel
}


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def _metric_card(title: str, value_id: str, sub_id: str = None, color: str = "white") -> dbc.Card:
    """Compact metric card with optional subtitle."""
    children = [
        html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem", "letterSpacing": "0.05em"}),
        html.H4(id=value_id, children="—", style={"color": color, "fontWeight": "700"}),
    ]
    if sub_id:
        children.append(html.Small(id=sub_id, children="", className="text-muted"))

    return dbc.Card(
        dbc.CardBody(children, className="py-2 px-3"),
        className="h-100",
        style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"},
    )


def _price_card(ticker: str) -> dbc.Card:
    """Individual stock price card."""
    return dbc.Card(
        dbc.CardBody([
            html.H6(ticker, className="mb-1", style={"color": _CLR["gray"], "fontSize": "0.8rem"}),
            html.H5(id=f"price-{ticker}", children="—", style={"fontWeight": "700", "marginBottom": "2px"}),
            html.Small(id=f"chg-{ticker}", children="", style={"fontSize": "0.75rem"}),
        ], className="py-2 px-3 text-center"),
        style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"},
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_live_app(engine) -> dash.Dash:
    """
    Create and return the Dash application bound to `engine`.

    Parameters
    ----------
    engine : LiveVaREngine
        Running engine instance (must be started before calling this).
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="IE 420 VaR Live Monitor",
        suppress_callback_exceptions=True,
    )

    tickers = config.TICKERS

    # ── Layout ────────────────────────────────────────────────────────
    app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#0b1e35", "minHeight": "100vh"}, children=[

        # Auto-refresh interval
        dcc.Interval(id="interval", interval=config.LIVE_INTERVAL_SEC * 1_000, n_intervals=0),

        # ── NAVBAR ───────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Span("📊", style={"fontSize": "1.4rem"}), width="auto"),
                    dbc.Col(dbc.NavbarBrand("IE 420  |  Real-Time Portfolio VaR Monitor",
                                            className="ms-2 fw-bold"), width="auto"),
                ], align="center"),
                dbc.Row([
                    dbc.Col(dbc.Badge(id="market-badge", pill=True,
                                     style={"fontSize": "0.8rem", "padding": "6px 14px"}), width="auto"),
                    dbc.Col(html.Small(id="last-update", className="text-muted ms-3"), width="auto"),
                    dbc.Col(dbc.Badge(id="source-badge", color="info", pill=True,
                                     style={"fontSize": "0.7rem"}), width="auto"),
                ], align="center", className="ms-auto"),
            ], fluid=True),
            color="dark", dark=True, sticky="top",
            style={"borderBottom": f"2px solid {_CLR['blue']}"},
        ),

        html.Br(),

        # ── ROW 1: Stock price cards ──────────────────────────────────
        dbc.Row(
            [dbc.Col(_price_card(t), xs=6, sm=4, md=2) for t in tickers],
            className="g-2 mb-3",
        ),

        # ── ROW 2: Portfolio summary cards ───────────────────────────
        dbc.Row([
            dbc.Col(_metric_card("Portfolio Value",   "port-value",   "port-pnl",   _CLR["blue"]),   md=3),
            dbc.Col(_metric_card("VaR 99% (Rolling)", "var99-rolling", "var99-pct99", _CLR["red"]),   md=3),
            dbc.Col(_metric_card("VaR 99% (EWMA)",    "var99-ewma",   "var99-pct-e", _CLR["orange"]), md=3),
            dbc.Col(_metric_card("VaR 99% (MC 10k)",  "var99-mc",     "var99-pct-m", _CLR["purple"]), md=3),
        ], className="g-2 mb-3"),

        # ── ROW 3: Charts ─────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📈  Intraday Portfolio Value",
                                   style={"backgroundColor": _CLR["panel"], "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="chart-value", style={"height": "300px"},
                                          config={"displayModeBar": False}),
                                 className="p-0"),
                ], style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"}),
                md=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("🛡️  VaR Estimates (Today's Session)",
                                   style={"backgroundColor": _CLR["panel"], "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="chart-var", style={"height": "300px"},
                                          config={"displayModeBar": False}),
                                 className="p-0"),
                ], style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"}),
                md=6,
            ),
        ], className="g-2 mb-3"),

        # ── ROW 4: Latency chart + model comparison table ─────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("⚡  Computation Latency per Update",
                                   style={"backgroundColor": _CLR["panel"], "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="chart-latency", style={"height": "260px"},
                                          config={"displayModeBar": False}),
                                 className="p-0"),
                ], style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"}),
                md=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📋  Live Method Comparison",
                                   style={"backgroundColor": _CLR["panel"], "fontWeight": "600"}),
                    dbc.CardBody(id="comparison-table", className="p-2"),
                ], style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"}),
                md=6,
            ),
        ], className="g-2 mb-3"),

        # ── ROW 5: Weight drift ───────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("⚖️  Dynamic Portfolio Weights (drift from 20% equal-weight)",
                                   style={"backgroundColor": _CLR["panel"], "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="chart-weights", style={"height": "220px"},
                                          config={"displayModeBar": False}),
                                 className="p-0"),
                ], style={"backgroundColor": _CLR["bg"], "border": "1px solid #1e3a5f"}),
            ),
        ], className="g-2 mb-4"),

        # ── Footer ────────────────────────────────────────────────────
        html.Hr(style={"borderColor": "#1e3a5f"}),
        dbc.Row(dbc.Col(
            html.Small([
                "Data: ",
                html.Span(id="footer-source", children="—"),
                "  |  Refresh: every 60 s  |  "
                "Confidence: 95% & 99%  |  "
                "Covariance: Rolling-252d & EWMA(λ=0.94)  |  "
                "Monte Carlo: 10,000 paths (every 5 min)  |  "
                "IE 420 Final Project"
            ], className="text-muted"),
            className="text-center",
        ), className="mb-3"),

    ])

    # ── Callbacks ─────────────────────────────────────────────────────

    @app.callback(
        [
            # Market / update status
            Output("market-badge",  "children"),
            Output("market-badge",  "color"),
            Output("last-update",   "children"),
            Output("source-badge",  "children"),
            Output("footer-source", "children"),
            # Price cards
            *[Output(f"price-{t}", "children") for t in tickers],
            *[Output(f"chg-{t}",   "children") for t in tickers],
            *[Output(f"chg-{t}",   "style")    for t in tickers],
            # Portfolio metric cards
            Output("port-value",    "children"),
            Output("port-pnl",      "children"),
            Output("var99-rolling", "children"),
            Output("var99-pct99",   "children"),
            Output("var99-ewma",    "children"),
            Output("var99-pct-e",   "children"),
            Output("var99-mc",      "children"),
            Output("var99-pct-m",   "children"),
            # Charts
            Output("chart-value",   "figure"),
            Output("chart-var",     "figure"),
            Output("chart-latency", "figure"),
            Output("chart-weights", "figure"),
            Output("comparison-table", "children"),
        ],
        [Input("interval", "n_intervals")],
    )
    def refresh(_n):
        latest  = engine.get_latest()
        history = engine.get_history()

        # ── Market status ─────────────────────────────────────────────
        mstatus, mcolor = market_status_label()
        last_ts = (
            latest["timestamp"].strftime("Last update: %H:%M:%S")
            if latest.get("timestamp") else "Waiting for first update…"
        )
        src       = latest.get("source") or "—"
        src_label = f"source: {src}"

        # ── Price cards ───────────────────────────────────────────────
        details   = latest.get("details", {})
        prices_out, chg_out, chg_style_out = [], [], []
        for t in tickers:
            d = details.get(t, {})
            p   = d.get("price")
            chg = d.get("change")
            pct = d.get("change_pct")

            prices_out.append(f"${p:,.2f}" if p else "—")

            if chg is not None:
                arrow = "▲" if chg >= 0 else "▼"
                color = _CLR["green"] if chg >= 0 else _CLR["red"]
                chg_out.append(f"{arrow} {abs(chg):.2f}  ({pct:+.2f}%)")
                chg_style_out.append({"color": color, "fontSize": "0.75rem"})
            else:
                chg_out.append("")
                chg_style_out.append({})

        # ── Portfolio metric cards ────────────────────────────────────
        pv     = latest.get("portfolio_value")
        pv_str = f"${pv:>12,.2f}" if pv else "—"

        # P&L relative to $100,000 initial capital
        pnl     = (pv - config.INITIAL_CAPITAL) if pv else None
        pnl_str = (
            f"{'▲' if pnl >= 0 else '▼'} ${abs(pnl):,.2f} "
            f"({'+'if pnl>=0 else ''}{pnl/config.INITIAL_CAPITAL*100:.2f}%)"
        ) if pnl is not None else ""

        def _var_str(key):
            v = latest.get(key)
            return f"${v:,.2f}" if v else "—"

        def _var_pct(key, base="portfolio_value"):
            v  = latest.get(key)
            bv = latest.get(base)
            if v and bv:
                return f"{v/bv*100:.2f}% of portfolio"
            return ""

        # ── Charts ────────────────────────────────────────────────────
        chart_value   = _build_value_chart(history)
        chart_var     = _build_var_chart(history)
        chart_latency = _build_latency_chart(history)
        chart_weights = _build_weights_chart(history, tickers)
        comp_table    = _build_comparison_table(latest)

        return (
            mstatus, mcolor, last_ts, src_label, src,
            *prices_out, *chg_out, *chg_style_out,
            pv_str, pnl_str,
            _var_str("var_99_rolling"), _var_pct("var_99_rolling"),
            _var_str("var_99_ewma"),    _var_pct("var_99_ewma"),
            _var_str("var_99_mc"),      _var_pct("var_99_mc"),
            chart_value, chart_var, chart_latency, chart_weights,
            comp_table,
        )

    return app


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#adb5bd", size=11),
    margin=dict(l=50, r=20, t=20, b=40),
    xaxis=dict(gridcolor="#1e3a5f", showgrid=True),
    yaxis=dict(gridcolor="#1e3a5f", showgrid=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def _empty_fig(msg="No data yet — waiting for first update"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="#adb5bd", size=13))
    fig.update_layout(**_LAYOUT_BASE)
    return fig


def _build_value_chart(history: pd.DataFrame) -> go.Figure:
    if history.empty or "portfolio_value" not in history.columns:
        return _empty_fig()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history["portfolio_value"],
        mode="lines", name="Portfolio Value",
        line=dict(color=_CLR["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(44,123,229,0.08)",
    ))
    # Initial capital reference line
    fig.add_hline(y=config.INITIAL_CAPITAL, line_dash="dot",
                  line_color=_CLR["gray"], opacity=0.5,
                  annotation_text="Initial $100k", annotation_position="bottom right")
    fig.update_layout(**_LAYOUT_BASE, yaxis_title="Value ($)")
    return fig


def _build_var_chart(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return _empty_fig()

    fig = go.Figure()
    for col, name, color, dash in [
        ("var_99_rolling", "99% VaR Rolling", _CLR["red"],    "solid"),
        ("var_99_ewma",    "99% VaR EWMA",    _CLR["orange"], "dash"),
        ("var_95_rolling", "95% VaR Rolling", _CLR["purple"], "dot"),
        ("var_99_mc",      "99% VaR MC",      _CLR["green"],  "longdash"),
    ]:
        sub = history[col].dropna() if col in history.columns else pd.Series()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub,
            mode="lines", name=name,
            line=dict(color=color, width=1.5, dash=dash),
        ))
    fig.update_layout(**_LAYOUT_BASE, yaxis_title="VaR ($)")
    return fig


def _build_latency_chart(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return _empty_fig()

    fig = go.Figure()
    for col, name, color in [
        ("latency_rolling_ms", "Rolling (ms)", _CLR["blue"]),
        ("latency_ewma_ms",    "EWMA (ms)",    _CLR["orange"]),
        ("latency_mc_ms",      "MC (ms)",      _CLR["purple"]),
    ]:
        sub = history[col].dropna() if col in history.columns else pd.Series()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub,
            mode="lines+markers", name=name,
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
        ))
    fig.update_layout(**_LAYOUT_BASE, yaxis_title="Latency (ms)")
    return fig


def _build_weights_chart(history: pd.DataFrame, tickers: list) -> go.Figure:
    """Stacked area chart showing dynamic weight drift throughout the session."""
    if history.empty:
        return _empty_fig()

    # Compute weights from prices if stored, else derive from history cols
    weight_cols = [c for c in history.columns if c.startswith("weight_") or c in tickers]
    if not weight_cols:
        return _empty_fig("Weight data not available in history")

    colors = [_CLR["blue"], _CLR["green"], _CLR["orange"], _CLR["purple"], _CLR["red"]]
    fig = go.Figure()

    for i, t in enumerate(tickers):
        col = f"weight_{t}" if f"weight_{t}" in history.columns else None
        if col is None:
            continue
        sub = history[col].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub * 100,
            mode="lines", name=t,
            line=dict(color=colors[i % len(colors)], width=1.5),
            stackgroup="weights",
        ))

    fig.add_hline(y=20, line_dash="dot", line_color=_CLR["gray"],
                  opacity=0.4, annotation_text="Equal weight (20%)")
    fig.update_layout(**_LAYOUT_BASE, yaxis_title="Weight (%)")
    return fig


def _build_comparison_table(latest: dict):
    """Bootstrap-styled comparison table for the latest snapshot."""
    pv = latest.get("portfolio_value") or 1

    rows = []
    for label, v99_key, v95_key, lat_key in [
        ("Rolling Cov",  "var_99_rolling", "var_95_rolling", "latency_rolling_ms"),
        ("EWMA Cov",     "var_99_ewma",    "var_95_ewma",    "latency_ewma_ms"),
        ("Monte Carlo",  "var_99_mc",      "var_95_mc",      "latency_mc_ms"),
    ]:
        v99 = latest.get(v99_key)
        v95 = latest.get(v95_key)
        lat = latest.get(lat_key)
        rows.append({
            "Method":       label,
            "VaR 95% ($)":  f"${v95:,.2f}" if v95 else "—",
            "VaR 99% ($)":  f"${v99:,.2f}" if v99 else "—",
            "% of Portfolio": f"{v99/pv*100:.2f}%" if v99 else "—",
            "Latency":      f"{lat:.3f} ms" if lat else "—",
        })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": _CLR["bg"],
            "color": "white",
            "textAlign": "center",
            "padding": "8px",
            "border": f"1px solid #1e3a5f",
            "fontSize": "0.83rem",
        },
        style_header={
            "backgroundColor": _CLR["panel"],
            "fontWeight": "700",
            "border": f"1px solid #1e3a5f",
        },
        style_data_conditional=[
            {"if": {"row_index": 0}, "borderLeft": f"3px solid {_CLR['blue']}"},
            {"if": {"row_index": 1}, "borderLeft": f"3px solid {_CLR['orange']}"},
            {"if": {"row_index": 2}, "borderLeft": f"3px solid {_CLR['purple']}"},
        ],
    )
