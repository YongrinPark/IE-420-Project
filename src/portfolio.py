"""
portfolio.py
============
Portfolio initialisation and dynamic calculations.

Mathematical framework
----------------------
Portfolio value  :  V_{p,t} = Σ_i  n_i · P_{i,t}
Dynamic weight   :  w_{i,t} = (n_i · P_{i,t}) / V_{p,t}
P&L              :  PnL_t   = V_{p,t} − V_{p,t-1}

The portfolio uses *fixed shares* (not fixed weights).  Equal capital
is allocated to each asset at inception; thereafter weights drift as
prices move, which is the source of dynamic risk exposure.
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def initialize_portfolio(
    prices_at_open: pd.Series,
    initial_capital: float = None,
    tickers: list = None,
) -> dict:
    """
    Create an equal-weight portfolio and compute fixed share allocations.

    Each asset receives an equal *dollar* allocation at market open:
        allocation_i = initial_capital / n_assets
        n_i          = allocation_i / P_{i,0}

    Parameters
    ----------
    prices_at_open : pd.Series
        Opening prices; index = ticker symbols.
    initial_capital : float, optional
        Total investment in USD. Defaults to config.INITIAL_CAPITAL.
    tickers : list, optional
        Asset universe. Defaults to config.TICKERS.

    Returns
    -------
    dict with keys:
        shares          – pd.Series of fixed share counts
        initial_capital – float
        initial_value   – float (may differ slightly from capital due to rounding)
        initial_weights – pd.Series
        tickers         – list
    """
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    tickers         = tickers         or config.TICKERS

    shares = compute_shares(prices_at_open, initial_capital, tickers)

    initial_value   = compute_portfolio_value(shares, prices_at_open)
    initial_weights = compute_dynamic_weights(shares, prices_at_open)

    print(f"[portfolio] Portfolio initialised")
    print(f"  Capital : ${initial_capital:>12,.2f}")
    print(f"  Value   : ${initial_value:>12,.2f}")
    print(f"  Shares  :\n{shares.to_string()}")
    print(f"  Weights :\n{initial_weights.round(4).to_string()}")

    return {
        "shares":          shares,
        "initial_capital": initial_capital,
        "initial_value":   initial_value,
        "initial_weights": initial_weights,
        "tickers":         tickers,
    }


def compute_shares(
    prices_at_open: pd.Series,
    initial_capital: float = None,
    tickers: list = None,
) -> pd.Series:
    """
    Compute fixed share counts for an equal-dollar allocation.

    Parameters
    ----------
    prices_at_open : pd.Series
        Opening prices indexed by ticker.
    initial_capital : float
        Total capital.
    tickers : list
        Asset subset to allocate to.

    Returns
    -------
    pd.Series
        Share counts; index = tickers, name = 'shares'.
    """
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    tickers         = tickers         or config.TICKERS

    n_assets   = len(tickers)
    allocation = initial_capital / n_assets       # equal dollar split
    shares     = allocation / prices_at_open[tickers]
    shares.name = "shares"
    return shares


def compute_portfolio_value(shares: pd.Series, prices: pd.Series) -> float:
    """
    Compute total portfolio value.

    V_{p,t} = Σ_i n_i · P_{i,t}

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts (index = tickers).
    prices : pd.Series
        Current prices (index must include all tickers in shares).

    Returns
    -------
    float
        Dollar portfolio value.
    """
    return float((shares * prices[shares.index]).sum())


def compute_dynamic_weights(shares: pd.Series, prices: pd.Series) -> pd.Series:
    """
    Compute current portfolio weights from fixed shares and live prices.

    w_{i,t} = (n_i · P_{i,t}) / V_{p,t}

    Even with fixed shares, weights drift intraday as prices move.

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts.
    prices : pd.Series
        Current market prices.

    Returns
    -------
    pd.Series
        Weights summing to 1; index = tickers.
    """
    position_values = shares * prices[shares.index]
    total_value     = position_values.sum()
    return position_values / total_value


def compute_portfolio_pnl(
    shares: pd.Series,
    prices_current: pd.Series,
    prices_previous: pd.Series,
) -> float:
    """
    Dollar P&L between two consecutive time steps.

    PnL = V_{p,t} − V_{p,t-1}

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts.
    prices_current : pd.Series
        Prices at time t.
    prices_previous : pd.Series
        Prices at time t-1.

    Returns
    -------
    float
        Signed dollar P&L (positive = gain, negative = loss).
    """
    v_current  = compute_portfolio_value(shares, prices_current)
    v_previous = compute_portfolio_value(shares, prices_previous)
    return v_current - v_previous


def compute_portfolio_series(
    shares: pd.Series,
    intraday_prices: pd.DataFrame,
) -> pd.Series:
    """
    Vectorised portfolio value time series for a full intraday session.

    Equivalent to calling compute_portfolio_value() at every minute but
    uses a dot product for efficiency.

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts; index = tickers.
    intraday_prices : pd.DataFrame
        Intraday prices; DatetimeIndex, columns = tickers.

    Returns
    -------
    pd.Series
        Portfolio value at every minute; name = 'portfolio_value'.
    """
    values = intraday_prices[shares.index].dot(shares)
    values.name = "portfolio_value"
    return values
