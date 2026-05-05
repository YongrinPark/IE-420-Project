"""
config.py
=========
Global project configuration for the IE 420 VaR Monitoring System.
Edit this file to change tickers, dates, model parameters, or paths.
"""

import os

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
INITIAL_CAPITAL = 100_000          # USD

# ---------------------------------------------------------------------------
# VaR model parameters
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS = [0.95, 0.99]   # Both levels shown on dashboard
PRIMARY_CONFIDENCE = 0.99          # Used in primary reporting

RETURN_TYPE = "log"                # log returns throughout
HOLDING_PERIOD = 1                 # 1-day VaR

# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
COV_WINDOW = 252                   # rolling window (~1 trading year)
EWMA_LAMBDA = 0.94                 # RiskMetrics daily decay factor

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------
MC_PATHS = [1_000, 5_000, 10_000]  # path counts to benchmark
MC_RANDOM_SEED = 42                # reproducibility

# ---------------------------------------------------------------------------
# Data date ranges
# ---------------------------------------------------------------------------
DAILY_START = "2022-01-01"         # start of daily historical window
DAILY_END   = "2024-12-31"         # end   of daily historical window
INTRADAY_DATE = "2026-05-01"       # single day used for intraday replay

# ---------------------------------------------------------------------------
# Trading hours (US Eastern, 24-h format)
# ---------------------------------------------------------------------------
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"

# ---------------------------------------------------------------------------
# Filesystem paths  (derived from this file's location = project root)
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_RAW         = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED   = os.path.join(BASE_DIR, "data", "processed")
RESULTS_FIGURES  = os.path.join(BASE_DIR, "results", "figures")
RESULTS_TABLES   = os.path.join(BASE_DIR, "results", "tables")
REPORT_DIR       = os.path.join(BASE_DIR, "report")
REPORT_FIGURES   = os.path.join(BASE_DIR, "report", "figures")
REPORT_TABLES    = os.path.join(BASE_DIR, "report", "tables")
