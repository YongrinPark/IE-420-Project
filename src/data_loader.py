"""
data_loader.py
==============
Download and load market data for the IE 420 VaR project.

Public API
----------
download_daily_prices()     – fetch daily adjusted-close prices via yfinance
download_intraday_prices()  – fetch 1-minute intraday prices via yfinance
load_saved_data()           – load a previously saved CSV from disk
"""

import os
import sys
from datetime import time as dtime

import pandas as pd
import yfinance as yf

# Allow running this file directly or importing from project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def download_daily_prices(
    tickers: list = None,
    start: str = None,
    end: str = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list, optional
        Ticker symbols. Defaults to config.TICKERS.
    start : str, optional
        Start date 'YYYY-MM-DD'. Defaults to config.DAILY_START.
    end : str, optional
        End date 'YYYY-MM-DD'. Defaults to config.DAILY_END.
    save : bool
        Persist the result to data/raw/daily_prices.csv.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices; index = Date, columns = tickers.
    """
    tickers = tickers or config.TICKERS
    start   = start   or config.DAILY_START
    end     = end     or config.DAILY_END

    print(f"[data_loader] Downloading daily prices for {tickers}  {start} → {end} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Guarantee column order matches config
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    if save:
        _save(prices, "daily_prices.csv", config.DATA_RAW)

    return prices


def download_intraday_prices(
    tickers: list = None,
    date: str = None,
    interval: str = "1m",
    save: bool = True,
) -> pd.DataFrame:
    """
    Download 1-minute intraday prices for a single trading day.

    Note
    ----
    yfinance restricts 1-minute history to roughly the last 30 days.
    For historical replay, pre-download and use load_saved_data().

    Parameters
    ----------
    tickers : list, optional
        Ticker symbols.
    date : str, optional
        Trading date 'YYYY-MM-DD'. Defaults to config.INTRADAY_DATE.
    interval : str
        yfinance interval string. Default '1m'.
    save : bool
        Persist to data/raw/intraday_1min_prices.csv.

    Returns
    -------
    pd.DataFrame
        1-minute prices filtered to regular trading hours;
        index = Datetime, columns = tickers.
    """
    tickers = tickers or config.TICKERS
    date    = date    or config.INTRADAY_DATE

    start_dt = pd.Timestamp(date)
    end_dt   = start_dt + pd.Timedelta(days=1)

    print(f"[data_loader] Downloading {interval} intraday prices for {tickers} on {date} …")

    frames = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print(f"  WARNING: No intraday data returned for {ticker}.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            frames[ticker] = df["Close"].squeeze()
        else:
            frames[ticker] = df["Close"]

    if not frames:
        raise RuntimeError("No intraday data was downloaded. Check the date and tickers.")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Datetime"

    # Remove timezone info for consistent handling
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices = _filter_trading_hours(prices)

    if save:
        _save(prices, "intraday_1min_prices.csv", config.DATA_RAW)

    return prices


def load_saved_data(
    filename: str,
    data_dir: str = None,
    parse_dates: bool = True,
    index_col: int = 0,
) -> pd.DataFrame:
    """
    Load a previously saved CSV from the data directory.

    Parameters
    ----------
    filename : str
        e.g. 'daily_prices.csv'
    data_dir : str, optional
        Defaults to config.DATA_RAW.
    parse_dates : bool
        Parse the index as dates.
    index_col : int
        Column to use as index.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or config.DATA_RAW
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run the download functions first or place the CSV in the correct directory."
        )

    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    print(f"[data_loader] Loaded '{filename}'  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only rows within regular US equity trading hours."""
    h_open,  m_open  = map(int, config.MARKET_OPEN.split(":"))
    h_close, m_close = map(int, config.MARKET_CLOSE.split(":"))
    t_open  = dtime(h_open,  m_open)
    t_close = dtime(h_close, m_close)
    mask = (df.index.time >= t_open) & (df.index.time <= t_close)
    return df.loc[mask]


def _save(df: pd.DataFrame, filename: str, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    df.to_csv(path)
    print(f"[data_loader] Saved → {path}")
