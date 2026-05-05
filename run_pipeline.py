"""
run_pipeline.py
===============
End-to-end pipeline runner for the IE 420 VaR Monitoring System.

Steps
-----
1.  Download daily prices
2.  Download intraday 1-minute prices
3.  Compute daily log returns
4.  Initialise portfolio
5.  Estimate covariance (Rolling + EWMA)
6.  Run parametric VaR intraday replay
7.  Run Monte Carlo VaR benchmark (all path counts)
8.  Run intraday MC VaR replay (10k paths)
9.  Run backtesting + Kupiec test
10. Aggregate latency records and save summary
11. Generate matplotlib figures
12. Run report_generator to copy assets into report/

Run from the project root:
    python run_pipeline.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.data_loader    import download_daily_prices, download_intraday_prices, load_saved_data
from src.preprocess     import compute_log_returns, save_processed_data
from src.portfolio      import initialize_portfolio, compute_portfolio_series
from src.covariance     import rolling_covariance, ewma_covariance, compare_covariance_matrices
from src.parametric_var import run_parametric_var_replay
from src.monte_carlo_var import run_monte_carlo_experiment, run_mc_var_replay
from src.backtesting    import summarize_backtest_results
from src.latency        import record_latency, summarize_latency
from src.report_generator import run_all as generate_report_assets


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# Step 1 & 2 – Download data
# ===========================================================================
separator("STEP 1 – Download Daily Prices")
try:
    daily = download_daily_prices(save=True)
except Exception as e:
    print(f"Download failed ({e}). Trying to load from disk …")
    daily = load_saved_data("daily_prices.csv", config.DATA_RAW)

separator("STEP 2 – Download Intraday Prices")
try:
    intraday = download_intraday_prices(save=True)
except Exception as e:
    print(f"Download failed ({e}). Trying to load from disk …")
    intraday = load_saved_data("intraday_1min_prices.csv", config.DATA_RAW)

print(f"\nDaily   shape : {daily.shape}")
print(f"Intraday shape: {intraday.shape}")

# ===========================================================================
# Step 3 – Log returns
# ===========================================================================
separator("STEP 3 – Compute Log Returns")
log_returns = compute_log_returns(daily)
save_processed_data(log_returns, "daily_log_returns.csv")
print(f"Log returns shape: {log_returns.shape}")

# ===========================================================================
# Step 4 – Portfolio initialisation
# ===========================================================================
separator("STEP 4 – Initialise Portfolio")
portfolio = initialize_portfolio(intraday.iloc[0])
shares    = portfolio["shares"]

# ===========================================================================
# Step 5 – Covariance estimation
# ===========================================================================
separator("STEP 5 – Covariance Estimation")
cov_rolling = rolling_covariance(log_returns)
cov_ewma    = ewma_covariance(log_returns)
compare_covariance_matrices(cov_rolling, cov_ewma)

# ===========================================================================
# Step 6 – Parametric VaR replay
# ===========================================================================
separator("STEP 6 – Parametric VaR Intraday Replay")
var_results = run_parametric_var_replay(shares, intraday, cov_rolling, cov_ewma, save=True)

# Collect per-bar latency records for the summary
latency_records = []
for _, row in var_results.iterrows():
    record_latency(latency_records, "Parametric_Rolling", row["latency_rolling_ms"])
    record_latency(latency_records, "Parametric_EWMA",    row["latency_ewma_ms"])

# ===========================================================================
# Step 7 – Monte Carlo VaR benchmark (snapshot at first bar)
# ===========================================================================
separator("STEP 7 – Monte Carlo VaR Benchmark")
mc_bench = run_monte_carlo_experiment(
    shares         = shares,
    current_prices = intraday.iloc[0],
    cov_matrix     = cov_rolling,
    path_counts    = config.MC_PATHS,
    save           = True,
)
for _, row in mc_bench.iterrows():
    record_latency(
        latency_records,
        f"MC_{int(row['n_paths'])}",
        row["latency_ms"],
        n_paths=int(row["n_paths"]),
    )

# ===========================================================================
# Step 8 – MC VaR intraday replay (10k paths)
# ===========================================================================
separator("STEP 8 – MC VaR Intraday Replay (10k paths)")
mc_results = run_mc_var_replay(
    shares         = shares,
    intraday_prices = intraday,
    cov_matrix     = cov_rolling,
    n_paths        = 10_000,
    save           = True,
)

# ===========================================================================
# Step 9 – Backtesting
# ===========================================================================
separator("STEP 9 – Backtesting + Kupiec Test")
backtest_summary = summarize_backtest_results(
    daily_prices = daily,
    shares       = shares,
    log_returns  = log_returns,
    cov_matrix   = cov_rolling,
    save         = True,
)

# ===========================================================================
# Step 10 – Latency summary
# ===========================================================================
separator("STEP 10 – Latency Summary")
lat_summary = summarize_latency(latency_records, save=True)
print(lat_summary.to_string(index=False))

# ===========================================================================
# Step 11 – Generate figures
# ===========================================================================
separator("STEP 11 – Generating Figures")
os.makedirs(config.RESULTS_FIGURES, exist_ok=True)

# --- Figure 1: Portfolio value ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(var_results["portfolio_value"], color="#0d6efd", linewidth=1.5)
ax.set_title("Intraday Portfolio Value")
ax.set_ylabel("Value ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_FIGURES, "portfolio_value.png"), dpi=150)
plt.close()
print("Saved portfolio_value.png")

# --- Figure 2: VaR comparison (parametric vs MC) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(var_results["var_99_rolling"], label="99% VaR Rolling",  color="#dc3545")
axes[0].plot(var_results["var_99_ewma"],    label="99% VaR EWMA",     color="#fd7e14", linestyle="--")
if not mc_results.empty:
    axes[0].plot(mc_results["var_99_mc"],   label="99% VaR MC (10k)", color="#198754", linestyle=":")
axes[0].legend(); axes[0].set_ylabel("VaR ($)")
axes[0].set_title("99% VaR: Rolling vs EWMA vs Monte Carlo")

axes[1].plot(var_results["var_95_rolling"], label="95% VaR Rolling",  color="#6f42c1")
axes[1].plot(var_results["var_95_ewma"],    label="95% VaR EWMA",     color="#20c997", linestyle="--")
if not mc_results.empty:
    axes[1].plot(mc_results["var_95_mc"],   label="95% VaR MC (10k)", color="#adb5bd", linestyle=":")
axes[1].legend(); axes[1].set_ylabel("VaR ($)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_FIGURES, "var_comparison.png"), dpi=150)
plt.close()
print("Saved var_comparison.png")

# --- Figure 3: Latency comparison ---
if not lat_summary.empty:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#0d6efd", "#dc3545", "#fd7e14", "#198754", "#6f42c1"]
    bars = ax.bar(lat_summary["method"], lat_summary["mean_ms"],
                  color=colors[:len(lat_summary)], edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f ms", padding=3, fontsize=8)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Computation Latency per VaR Method")
    ax.set_yscale("log")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES, "latency_comparison.png"), dpi=150)
    plt.close()
    print("Saved latency_comparison.png")

# --- Figure 4: Backtesting exceedances ---
bt_series_path = os.path.join(config.DATA_PROCESSED, "backtest_series.csv")
if os.path.exists(bt_series_path):
    bt_series = pd.read_csv(bt_series_path, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bt_series["actual_loss"], label="Actual Loss",     color="#0d6efd", linewidth=1)
    ax.plot(bt_series["var_99"],      label="99% VaR",         color="#dc3545", linestyle="--", linewidth=1.2)
    if "breach_99" in bt_series.columns:
        breaches = bt_series[bt_series["breach_99"]]
        ax.scatter(breaches.index, breaches["actual_loss"],
                   color="red", zorder=5, s=40, label=f"Breach ({len(breaches)} days)")
    ax.legend()
    ax.set_ylabel("Dollar Loss ($)")
    ax.set_title("Backtesting: Actual Loss vs 99% VaR Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES, "backtesting_exceedances.png"), dpi=150)
    plt.close()
    print("Saved backtesting_exceedances.png")

# ===========================================================================
# Step 12 – Report assets
# ===========================================================================
separator("STEP 12 – Generating Report Assets")
generate_report_assets()

# ===========================================================================
# Done
# ===========================================================================
separator("PIPELINE COMPLETE")
print(f"\n  Daily data    : data/raw/daily_prices.csv")
print(f"  Intraday data : data/raw/intraday_1min_prices.csv")
print(f"  VaR results   : data/processed/var_results.csv")
print(f"  MC results    : data/processed/mc_var_results.csv")
print(f"  Backtest      : data/processed/backtest_series.csv")
print(f"  Figures       : results/figures/")
print(f"  Tables        : results/tables/")
print(f"  Report assets : report/figures/  report/tables/")
print(f"\n  Launch dashboard : python src/dashboard.py")
print(f"  Compile PDF      : cd report && latexmk -pdf final_report.tex")
