"""
covariance.py
=============
Covariance matrix estimation for portfolio VaR.

Two estimators
--------------
1. Rolling Sample Covariance
   Use the most recent `window` daily log returns (pandas .cov()).

2. EWMA Covariance  (RiskMetrics, JP Morgan 1996)
   Recursive formula:
       Σ_t = λ · Σ_{t-1} + (1 − λ) · r_{t-1} · r_{t-1}ᵀ
   Default λ = 0.94 for daily data.
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

def rolling_covariance(
    log_returns: pd.DataFrame,
    window: int = None,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Estimate a covariance matrix from the most recent `window` daily log returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns; rows = dates, columns = tickers.
    window : int, optional
        Look-back period in trading days. Defaults to config.COV_WINDOW (252).
    annualize : bool
        Multiply by 252 to convert daily cov → annual cov.

    Returns
    -------
    pd.DataFrame
        (n_assets × n_assets) covariance matrix.
    """
    window = window or config.COV_WINDOW

    if len(log_returns) < window:
        print(f"[covariance] Only {len(log_returns)} obs available; using all data.")
        window = len(log_returns)

    recent   = log_returns.iloc[-window:]
    cov      = recent.cov()

    if annualize:
        cov = cov * 252

    return cov


def ewma_covariance(
    log_returns: pd.DataFrame,
    lam: float = None,
    initial_window: int = 30,
) -> pd.DataFrame:
    """
    Estimate covariance using the EWMA (RiskMetrics) method.

    The seed is the sample covariance of the first `initial_window` observations.
    The recursive update then runs forward through all remaining observations.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    lam : float, optional
        Decay factor (λ). Defaults to config.EWMA_LAMBDA (0.94).
    initial_window : int
        Number of rows used to seed the recursion.

    Returns
    -------
    pd.DataFrame
        EWMA covariance matrix at the last available date.
    """
    lam = lam if lam is not None else config.EWMA_LAMBDA

    R   = log_returns.values          # shape (T, n)
    T, n = R.shape
    tickers = log_returns.columns

    # Seed from sample covariance of first `initial_window` rows
    cov = np.cov(R[:initial_window].T, ddof=1)

    # Walk forward; each step updates Σ using the *previous* return vector
    for t in range(initial_window, T):
        r   = R[t - 1].reshape(-1, 1)          # column vector (n×1)
        cov = lam * cov + (1 - lam) * (r @ r.T)

    return pd.DataFrame(cov, index=tickers, columns=tickers)


def ewma_covariance_series(
    log_returns: pd.DataFrame,
    lam: float = None,
    initial_window: int = 30,
) -> dict:
    """
    Compute the full time series of EWMA covariance matrices.

    Returns a dict mapping each date (after the seed) to its EWMA covariance
    matrix.  Used in backtesting where the covariance is updated daily.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    lam : float, optional
        Decay factor.
    initial_window : int
        Seed window.

    Returns
    -------
    dict
        {date: pd.DataFrame(n×n)}  for dates[initial_window:]
    """
    lam = lam if lam is not None else config.EWMA_LAMBDA

    R       = log_returns.values
    T, n    = R.shape
    dates   = log_returns.index
    tickers = log_returns.columns

    cov = np.cov(R[:initial_window].T, ddof=1)
    cov_series = {}

    for t in range(initial_window, T):
        r   = R[t - 1].reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
        cov_series[dates[t]] = pd.DataFrame(cov, index=tickers, columns=tickers)

    return cov_series


def compare_covariance_matrices(
    cov_rolling: pd.DataFrame,
    cov_ewma: pd.DataFrame,
) -> pd.DataFrame:
    """
    Print and return a side-by-side comparison of rolling vs EWMA covariance.

    Shows per-asset variance and annualised volatility for both estimators,
    plus the full correlation matrices for visual inspection.

    Parameters
    ----------
    cov_rolling : pd.DataFrame
        Rolling sample covariance.
    cov_ewma : pd.DataFrame
        EWMA covariance.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by ticker.
    """
    tickers = cov_rolling.columns.tolist()

    rows = []
    for t in tickers:
        rows.append({
            "Ticker":              t,
            "Rolling Var (daily)": cov_rolling.loc[t, t],
            "EWMA Var (daily)":    cov_ewma.loc[t, t],
            "Rolling Vol (daily)": np.sqrt(cov_rolling.loc[t, t]),
            "EWMA Vol (daily)":    np.sqrt(cov_ewma.loc[t, t]),
            "Rolling Vol (ann.)":  np.sqrt(cov_rolling.loc[t, t] * 252),
            "EWMA Vol (ann.)":     np.sqrt(cov_ewma.loc[t, t] * 252),
        })

    comparison = pd.DataFrame(rows).set_index("Ticker")

    corr_rolling = _cov_to_corr(cov_rolling)
    corr_ewma    = _cov_to_corr(cov_ewma)

    print("\n[covariance] Rolling Correlation Matrix:")
    print(corr_rolling.round(4).to_string())
    print("\n[covariance] EWMA Correlation Matrix:")
    print(corr_ewma.round(4).to_string())
    print("\n[covariance] Variance / Volatility Comparison:")
    print(comparison.round(6).to_string())

    return comparison


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    std  = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
