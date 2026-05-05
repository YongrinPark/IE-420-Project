# IE 420 Final Project
## Real-Time Portfolio VaR Monitoring System: Balancing Computational Speed and Risk Accuracy

---

## Overview

This project implements a near real-time Value-at-Risk (VaR) monitoring
system for a five-asset equity portfolio (AAPL, MSFT, NVDA, SPY, QQQ).
It compares three VaR computation methods across the dimensions of
computational speed and risk accuracy:

| Method | Estimator | Speed |
|---|---|---|
| Parametric VaR | Rolling sample covariance | Very fast (< 0.1 ms) |
| Parametric VaR | EWMA covariance (λ = 0.94) | Very fast (< 0.1 ms) |
| Monte Carlo VaR | GBM with correlated shocks | Moderate (1–100 ms) |

An interactive Plotly/Dash dashboard displays portfolio value, VaR
estimates, P&L vs. VaR threshold, and exceedance markers in real time.

---

## Project Structure

```
IE 420 Project/
├── config.py                  # All global settings
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                   # Downloaded CSVs (daily + intraday)
│   └── processed/             # Log returns, aligned prices, VaR results
│
├── notebooks/                 # Exploration and demo notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_parametric_var_baseline.ipynb
│   ├── 03_monte_carlo_var.ipynb
│   ├── 04_backtesting_analysis.ipynb
│   └── 05_dashboard_demo.ipynb
│
├── src/
│   ├── data_loader.py         # yfinance download / CSV load
│   ├── preprocess.py          # Log returns, alignment, cleaning
│   ├── portfolio.py           # Shares, weights, portfolio value
│   ├── covariance.py          # Rolling & EWMA covariance
│   ├── parametric_var.py      # Parametric VaR + intraday replay
│   ├── monte_carlo_var.py     # GBM simulation + MC VaR
│   ├── backtesting.py         # Exceedances + Kupiec test
│   ├── latency.py             # Timing utilities
│   ├── dashboard.py           # Plotly/Dash dashboard
│   ├── report_generator.py    # LaTeX table generation + figure copy
│   └── utils.py               # I/O helpers, validation
│
├── results/
│   ├── figures/               # PNG plots (portfolio, VaR, latency)
│   └── tables/                # Summary CSVs (latency, backtest, model)
│
└── report/
    ├── final_report.tex       # Main LaTeX report
    ├── references.bib         # BibTeX bibliography
    ├── figures/               # Figures copied here for LaTeX
    └── tables/                # LaTeX table .tex snippets
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the project

Open `config.py` and verify:
- `TICKERS` — asset symbols
- `DAILY_START` / `DAILY_END` — daily history window
- `INTRADAY_DATE` — the day used for 1-minute replay
- `INITIAL_CAPITAL` — starting portfolio size

> **Note on intraday data**: yfinance provides 1-minute data only for
> the last ~30 days. Set `INTRADAY_DATE` to a recent date before
> downloading, or provide a pre-downloaded `intraday_1min_prices.csv`
> in `data/raw/`.

---

## Running the Pipeline

Run scripts from the **project root** directory.

### Step 1 — Download data

```python
from src.data_loader import download_daily_prices, download_intraday_prices

daily     = download_daily_prices()
intraday  = download_intraday_prices()
```

### Step 2 — Preprocess

```python
from src.preprocess import compute_log_returns, save_processed_data

log_returns = compute_log_returns(daily)
save_processed_data(log_returns, "daily_log_returns.csv")
```

### Step 3 — Initialise portfolio

```python
from src.portfolio import initialize_portfolio

portfolio = initialize_portfolio(intraday.iloc[0])
shares    = portfolio["shares"]
```

### Step 4 — Estimate covariance

```python
from src.covariance import rolling_covariance, ewma_covariance

cov_roll = rolling_covariance(log_returns)
cov_ewma = ewma_covariance(log_returns)
```

### Step 5 — Run parametric VaR replay

```python
from src.parametric_var import run_parametric_var_replay

results = run_parametric_var_replay(shares, intraday, cov_roll, cov_ewma)
```

### Step 6 — Run Monte Carlo benchmark

```python
from src.monte_carlo_var import run_monte_carlo_experiment

mc_results = run_monte_carlo_experiment(shares, intraday.iloc[0], cov_roll)
```

### Step 7 — Backtesting

```python
from src.backtesting import summarize_backtest_results

backtest = summarize_backtest_results(daily, shares, log_returns, cov_roll)
```

### Step 8 — Launch dashboard

```bash
python src/dashboard.py
```

Then open `http://127.0.0.1:8050` in your browser.

---

## Generating the Report

### Step 1 — Generate figures and tables

After running the pipeline, generate all report assets:

```bash
python src/report_generator.py
```

This copies figures to `report/figures/` and writes LaTeX table snippets
to `report/tables/`.

### Step 2 — Compile the PDF

Navigate to the `report/` directory:

```bash
cd report
```

#### Option A — latexmk (recommended)

```bash
latexmk -pdf final_report.tex
```

#### Option B — manual pdflatex

```bash
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

The PDF appears as `report/final_report.pdf`.

---

## LaTeX Installation

### Windows

| Option | Notes |
|---|---|
| **MiKTeX** | https://miktex.org — recommended for Windows, installs packages on demand |
| **TeX Live** | https://tug.org/texlive — full distribution, larger download |
| **Overleaf** | https://overleaf.com — online, no local install required; upload the entire `report/` folder |

### macOS

```bash
brew install --cask mactex   # Full TeX Live for macOS
```

Or install [MacTeX](https://tug.org/mactex/).

### Linux

```bash
sudo apt-get install texlive-full latexmk   # Debian/Ubuntu
sudo dnf install texlive-scheme-full        # Fedora
```

---

## Key Configuration Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | `["AAPL","MSFT","NVDA","SPY","QQQ"]` | Portfolio assets |
| `INITIAL_CAPITAL` | `100_000` | Starting capital (USD) |
| `COV_WINDOW` | `252` | Rolling covariance window (trading days) |
| `EWMA_LAMBDA` | `0.94` | RiskMetrics decay factor |
| `MC_PATHS` | `[1000, 5000, 10000]` | Monte Carlo path counts |
| `MC_RANDOM_SEED` | `42` | Reproducibility seed |
| `PRIMARY_CONFIDENCE` | `0.99` | Main VaR confidence level |
| `INTRADAY_DATE` | `"2024-12-10"` | Day used for minute-by-minute replay |

---

## Mathematical Summary

| Symbol | Definition |
|---|---|
| $n_i$ | Fixed share count for asset $i$ |
| $P_{i,t}$ | Price of asset $i$ at time $t$ |
| $V_{p,t} = \sum_i n_i P_{i,t}$ | Portfolio value |
| $w_{i,t} = n_i P_{i,t} / V_{p,t}$ | Dynamic weight |
| $\sigma_{p,t} = \sqrt{w_t^\top \Sigma w_t}$ | Portfolio volatility |
| $\text{VaR}_{\alpha,t} = z_\alpha \cdot \sigma_{p,t} \cdot V_{p,t}$ | Parametric VaR |
| $\Sigma_t = \lambda \Sigma_{t-1} + (1-\lambda) r_{t-1} r_{t-1}^\top$ | EWMA covariance |

---

## License

For academic use only — IE 420 course project.
