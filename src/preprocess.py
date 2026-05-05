"""
preprocess.py
=============
Data cleaning and alignment utilities for the IE 420 VaR project.

Public API
----------
compute_log_returns()    – log(P_t / P_{t-1}) for a price DataFrame
align_price_data()       – common-index alignment with optional gap-fill
handle_missing_values()  – NaN fill / drop strategy
filter_trading_hours()   – restrict intraday data to market hours
save_processed_data()    – helper to persist DataFrames to data/processed/
"""

import os
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    Formula
    -------
    r_t = ln(P_t / P_{t-1})

    The first row (NaN) is dropped automatically.

    Parameters
    ----------
    prices : pd.DataFrame
        Price series; index = Date or Datetime, columns = tickers.

    Returns
    -------
    pd.DataFrame
        Log returns with one fewer row than the input.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def align_price_data(prices: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Align a multi-ticker price DataFrame to a common time index.

    When tickers have slightly different trading calendars (e.g. ETFs vs
    individual stocks) some rows may contain NaN for one ticker.  This
    function handles that by either dropping or forward-filling those rows.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price data (may contain NaN).
    dropna : bool
        True  → drop any row where at least one ticker is NaN.
        False → forward-fill up to 5 consecutive missing bars.

    Returns
    -------
    pd.DataFrame
        Aligned, sorted price data.
    """
    prices = prices.copy().sort_index()

    if dropna:
        prices = prices.dropna()
    else:
        prices = prices.ffill(limit=5)

    return prices


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "ffill",
    max_gap: int = 5,
) -> pd.DataFrame:
    """
    Handle missing values in price or return data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str
        'ffill' – forward-fill gaps up to max_gap bars.
        'bfill' – backward-fill.
        'drop'  – drop all rows with any NaN.
    max_gap : int
        Maximum consecutive NaN bars to fill (ignored for 'drop').

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs treated per the chosen strategy.
    """
    df = df.copy()

    nan_summary = df.isna().sum()
    if nan_summary.sum() > 0:
        print(f"[preprocess] Missing value counts:\n{nan_summary[nan_summary > 0]}")

    if method == "ffill":
        df = df.ffill(limit=max_gap)
    elif method == "bfill":
        df = df.bfill(limit=max_gap)
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'ffill', 'bfill', or 'drop'.")

    # Any remaining NaN after fill means the gap exceeded max_gap; drop those rows
    remaining = df.isna().sum().sum()
    if remaining > 0:
        print(f"[preprocess] Dropping {remaining} residual NaN entries (gap > {max_gap}).")
        df = df.dropna()

    return df


def filter_trading_hours(
    df: pd.DataFrame,
    open_time: str = None,
    close_time: str = None,
) -> pd.DataFrame:
    """
    Restrict an intraday DataFrame to regular US equity trading hours.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with a DatetimeIndex.
    open_time : str, optional
        e.g. '09:30'. Defaults to config.MARKET_OPEN.
    close_time : str, optional
        e.g. '16:00'. Defaults to config.MARKET_CLOSE.

    Returns
    -------
    pd.DataFrame
        Filtered to [open_time, close_time].
    """
    open_time  = open_time  or config.MARKET_OPEN
    close_time = close_time or config.MARKET_CLOSE

    h_open,  m_open  = map(int, open_time.split(":"))
    h_close, m_close = map(int, close_time.split(":"))

    t_open  = dtime(h_open,  m_open)
    t_close = dtime(h_close, m_close)

    mask = (df.index.time >= t_open) & (df.index.time <= t_close)
    return df.loc[mask]


def save_processed_data(df: pd.DataFrame, filename: str) -> str:
    """
    Save a DataFrame to data/processed/<filename>.

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
        e.g. 'daily_log_returns.csv'

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    path = os.path.join(config.DATA_PROCESSED, filename)
    df.to_csv(path)
    print(f"[preprocess] Saved processed data → {path}")
    return path
