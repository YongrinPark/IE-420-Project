"""
run_live.py
===========
Entry point for the IE 420 Real-Time VaR Dashboard.

Steps
-----
1. Load environment variables from .env (FINNHUB_API_KEY, etc.)
2. Download / load historical daily prices and build covariance matrices
3. Compute fixed share allocation from initial capital
4. Start the LiveVaREngine background thread
5. Launch the Dash web dashboard on http://localhost:8050

Usage
-----
    python run_live.py

Set your Finnhub API key in a .env file (see .env.example) or as an
environment variable before running:

    export FINNHUB_API_KEY="your_key_here"
    python run_live.py
"""

import os
import sys

# ── Load .env before importing anything that reads config ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed; rely on shell environment

# Add project root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np

import config
from src.data_loader    import download_daily_prices, load_saved_data
from src.preprocess     import compute_log_returns
from src.covariance     import rolling_covariance, ewma_covariance
from src.portfolio      import compute_shares
from src.live_data      import get_live_prices
from src.live_engine    import LiveVaREngine
from src.live_dashboard import create_live_app


# ---------------------------------------------------------------------------
# 1.  Historical data & covariance
# ---------------------------------------------------------------------------

def _load_prices() -> pd.DataFrame:
    """Load historical daily prices (cached on disk, download if missing)."""
    try:
        prices = load_saved_data("daily_prices")
        if prices is not None and not prices.empty:
            print(f"[startup] Loaded cached prices — {len(prices)} rows, "
                  f"{prices.index[0].date()} → {prices.index[-1].date()}")
            return prices
    except Exception:
        pass

    print("[startup] Downloading historical prices …")
    prices = download_daily_prices()
    return prices


def _build_covariances(prices: pd.DataFrame):
    """Return (cov_rolling, cov_ewma) from historical price data."""
    returns = compute_log_returns(prices)

    # Drop any rows with all-NaN (first row after log-return)
    returns = returns.dropna(how="all")

    cov_rolling = rolling_covariance(returns, window=config.COV_WINDOW)
    cov_ewma    = ewma_covariance(returns, lam=config.EWMA_LAMBDA)

    print(f"[startup] Covariance matrices built — "
          f"rolling window={config.COV_WINDOW}, EWMA λ={config.EWMA_LAMBDA}")
    return cov_rolling, cov_ewma


# ---------------------------------------------------------------------------
# 2.  Portfolio initialisation
# ---------------------------------------------------------------------------

def _init_portfolio(prices: pd.DataFrame) -> pd.Series:
    """
    Compute fixed share counts from the last available historical close.
    Returns a pd.Series indexed by ticker.
    """
    latest_prices = prices.iloc[-1][config.TICKERS]
    shares = compute_shares(latest_prices, config.INITIAL_CAPITAL)
    port_val = (shares * latest_prices).sum()
    print(f"[startup] Portfolio initialised — "
          f"${port_val:,.2f} across {len(config.TICKERS)} tickers")
    for t in config.TICKERS:
        print(f"          {t:>5s}: {shares[t]:.4f} shares @ ${latest_prices[t]:.2f}")
    return shares


# ---------------------------------------------------------------------------
# 3.  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IE 420 — Real-Time Portfolio VaR Dashboard")
    print("=" * 60)

    api_key = config.FINNHUB_API_KEY or os.environ.get("FINNHUB_API_KEY", "")
    if api_key:
        print(f"[startup] Finnhub API key detected — real-time quotes enabled.")
    else:
        print("[startup] No Finnhub key found — using yfinance fallback.")
        print("          Set FINNHUB_API_KEY in .env for real-time data.")

    # Historical data
    prices     = _load_prices()
    cov_r, cov_e = _build_covariances(prices)
    shares     = _init_portfolio(prices)

    # Engine
    engine = LiveVaREngine(
        shares      = shares,
        cov_rolling = cov_r,
        cov_ewma    = cov_e,
        interval_sec = config.LIVE_INTERVAL_SEC,
        mc_every_n   = config.LIVE_MC_EVERY_N,
        mc_paths     = 10_000,
    )
    engine.start()

    # Dashboard
    app = create_live_app(engine)

    print("\n" + "=" * 60)
    print("  Dashboard running at  http://localhost:8050")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
