"""
dashboard.py
============
Plotly/Dash interactive dashboard for the IE 420 VaR Monitoring System.

Launch
------
    python src/dashboard.py

Then open http://127.0.0.1:8050 in your browser.

Prerequisites
-------------
Run the full pipeline first so these CSV files exist:
    data/processed/var_results.csv
    data/processed/mc_var_results.csv      (optional)
    data/processed/backtest_series.csv
    results/tables/latency_summary.csv
    results/tables/backtesting_summary.csv
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(filename: str, directory: str = None) -> pd.DataFrame:
    directory = directory or config.DATA_PROCESSED
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_layout() -> html.Div:
    return html.Div(
        style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f8f9fa", "padding": "20px"},
        children=[
            # ---- Header --------------------------------------------------
            html.H1(
                "Real-Time Portfolio VaR Monitor",
                style={"textAlign": "center", "color": "#1a1a2e", "marginBottom": "5px"},
            ),
            html.P(
                "IE 420 Final Project  |  AAPL · MSFT · NVDA · SPY · QQQ",
                style={"textAlign": "center", "color": "#555", "marginBottom": "30px"},
            ),

            # ---- Refresh button ------------------------------------------
            html.Div(
                html.Button(
                    "Refresh Data",
                    id="refresh-btn",
                    n_clicks=0,
                    style={
                        "padding": "8px 20px",
                        "backgroundColor": "#0d6efd",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
                style={"textAlign": "center", "marginBottom": "20px"},
            ),

            # ---- Row 1: Portfolio Value ----------------------------------
            html.Div(
                dcc.Graph(id="portfolio-value-chart"),
                style={"backgroundColor": "white", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px",
                       "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),

            # ---- Row 2: VaR Comparison ----------------------------------
            html.Div(
                dcc.Graph(id="var-comparison-chart"),
                style={"backgroundColor": "white", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px",
                       "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),

            # ---- Row 3: P&L vs VaR + Exceedances -----------------------
            html.Div(
                dcc.Graph(id="pnl-var-chart"),
                style={"backgroundColor": "white", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px",
                       "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),

            # ---- Row 4: Latency -----------------------------------------
            html.Div(
                dcc.Graph(id="latency-chart"),
                style={"backgroundColor": "white", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px",
                       "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),

            # ---- Row 5: Summary Table -----------------------------------
            html.Div(
                children=[
                    html.H3("Summary Statistics", style={"color": "#1a1a2e"}),
                    html.Div(id="summary-table"),
                ],
                style={"backgroundColor": "white", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px",
                       "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            ),

            dcc.Store(id="dummy-trigger"),
        ],
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> dash.Dash:
    app = dash.Dash(__name__, title="IE 420 VaR Monitor")
    app.layout = build_layout()

    # ---- Callbacks -------------------------------------------------------

    @app.callback(
        [
            Output("portfolio-value-chart", "figure"),
            Output("var-comparison-chart",  "figure"),
            Output("pnl-var-chart",          "figure"),
            Output("latency-chart",          "figure"),
            Output("summary-table",          "children"),
        ],
        [Input("refresh-btn", "n_clicks")],
    )
    def update_all(_n):
        var_df  = _load("var_results.csv")
        back_df = _load("backtest_series.csv")
        lat_df  = _load("latency_summary.csv", config.RESULTS_TABLES)
        bt_sum  = _load("backtesting_summary.csv", config.RESULTS_TABLES)
        mc_df   = _load("mc_var_results.csv")

        # ---- 1. Portfolio value ----------------------------------------
        fig_pv = go.Figure()
        if not var_df.empty and "portfolio_value" in var_df.columns:
            fig_pv.add_trace(go.Scatter(
                x=var_df.index, y=var_df["portfolio_value"],
                mode="lines", name="Portfolio Value",
                line=dict(color="#0d6efd", width=2),
            ))
        fig_pv.update_layout(
            title="Intraday Portfolio Value",
            yaxis_title="Value ($)", xaxis_title="Time",
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )

        # ---- 2. VaR comparison -----------------------------------------
        fig_var = go.Figure()
        if not var_df.empty:
            for col, name, color, dash_style in [
                ("var_99_rolling", "99% VaR Rolling",    "#dc3545", "solid"),
                ("var_99_ewma",    "99% VaR EWMA",       "#fd7e14", "dash"),
                ("var_95_rolling", "95% VaR Rolling",    "#6f42c1", "dot"),
                ("var_95_ewma",    "95% VaR EWMA",       "#20c997", "dashdot"),
            ]:
                if col in var_df.columns:
                    fig_var.add_trace(go.Scatter(
                        x=var_df.index, y=var_df[col],
                        mode="lines", name=name,
                        line=dict(color=color, dash=dash_style),
                    ))
            if not mc_df.empty and "var_99_mc" in mc_df.columns:
                fig_var.add_trace(go.Scatter(
                    x=mc_df.index, y=mc_df["var_99_mc"],
                    mode="lines", name="99% VaR MC",
                    line=dict(color="#198754", dash="longdash"),
                ))
        fig_var.update_layout(
            title="VaR Estimates: Rolling vs EWMA vs Monte Carlo",
            yaxis_title="VaR ($)", xaxis_title="Time",
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )

        # ---- 3. P&L vs VaR (backtesting) --------------------------------
        fig_pnl = go.Figure()
        if not back_df.empty:
            if "actual_loss" in back_df.columns:
                fig_pnl.add_trace(go.Scatter(
                    x=back_df.index, y=back_df["actual_loss"],
                    mode="lines", name="Actual Daily Loss",
                    line=dict(color="#0d6efd"),
                ))
            if "var_99" in back_df.columns:
                fig_pnl.add_trace(go.Scatter(
                    x=back_df.index, y=back_df["var_99"],
                    mode="lines", name="99% VaR Threshold",
                    line=dict(color="#dc3545", dash="dash"),
                ))
            breach_col = "breach_99"
            if breach_col in back_df.columns:
                breaches = back_df[back_df[breach_col]]
                fig_pnl.add_trace(go.Scatter(
                    x=breaches.index, y=breaches["actual_loss"],
                    mode="markers", name=f"VaR Breach ({len(breaches)})",
                    marker=dict(color="red", size=8, symbol="x"),
                ))
        fig_pnl.update_layout(
            title="Actual P&L vs 99% VaR Threshold (Backtesting)",
            yaxis_title="Dollar Loss ($)", xaxis_title="Date",
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )

        # ---- 4. Latency --------------------------------------------------
        fig_lat = go.Figure()
        if not lat_df.empty and "method" in lat_df.columns:
            fig_lat.add_trace(go.Bar(
                x=lat_df["method"], y=lat_df["mean_ms"],
                name="Mean Latency",
                marker_color=["#0d6efd", "#dc3545", "#fd7e14", "#198754", "#6f42c1"][
                    :len(lat_df)
                ],
                error_y=dict(type="data", array=lat_df.get("std_ms", pd.Series()).tolist()),
            ))
        fig_lat.update_layout(
            title="Computation Latency per VaR Method",
            yaxis_title="Latency (ms)", xaxis_title="Method",
            plot_bgcolor="white", paper_bgcolor="white",
        )

        # ---- 5. Summary table -------------------------------------------
        rows = []
        if not lat_df.empty:
            rows.append({"Metric": "Avg Latency – Rolling (ms)",
                         "Value": f"{lat_df[lat_df['method']=='Parametric_Rolling']['mean_ms'].values[0]:.4f}"
                         if "Parametric_Rolling" in lat_df["method"].values else "—"})
            rows.append({"Metric": "Avg Latency – EWMA (ms)",
                         "Value": f"{lat_df[lat_df['method']=='Parametric_EWMA']['mean_ms'].values[0]:.4f}"
                         if "Parametric_EWMA" in lat_df["method"].values else "—"})

        if not bt_sum.empty:
            for _, r in bt_sum.iterrows():
                rows += [
                    {"Metric": "Backtesting Observations (T)", "Value": str(r.get("T", "—"))},
                    {"Metric": "VaR Breaches",                 "Value": str(r.get("Breaches", "—"))},
                    {"Metric": "Expected Breaches",            "Value": str(r.get("Expected Breaches", "—"))},
                    {"Metric": "Kupiec p-value",               "Value": str(r.get("p-value", "—"))},
                    {"Metric": "Reject H0 at 5%?",             "Value": str(r.get("Reject H0 (5%)", "—"))},
                ]

        if not rows:
            rows = [{"Metric": "No data", "Value": "Run the pipeline first"}]

        table = dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in ["Metric", "Value"]],
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
            ],
        )

        return fig_pv, fig_var, fig_pnl, fig_lat, table

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    print("[dashboard] Starting at http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
