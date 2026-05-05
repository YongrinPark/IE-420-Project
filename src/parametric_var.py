"""
parametric_var.py
=================
Parametric (variance-covariance) Value-at-Risk computation.

Mathematical framework
----------------------
Portfolio volatility:
    σ_{p,t} = √(w_t ᵀ · Σ · w_t)

Parametric VaR (1-day, confidence α):
    VaR_{α,t} = z_α · σ_{p,t} · V_{p,t}

where
    w_t          = dynamic portfolio weights at time t
    Σ            = daily covariance matrix from log returns
    z_α          = Φ⁻¹(α)  standard normal quantile
    V_{p,t}      = total portfolio value at time t

Assumption: portfolio log returns are normally distributed (Gaussian).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# Pre-compute quantiles once at import time (minor speed optimisation)
_Z = {cl: stats.norm.ppf(cl) for cl in config.CONFIDENCE_LEVELS}


# ---------------------------------------------------------------------------
# Core VaR functions
# ---------------------------------------------------------------------------

def compute_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Daily portfolio standard deviation.

    σ_p = √(wᵀ Σ w)

    Parameters
    ----------
    weights : np.ndarray
        1-D weight vector summing to 1.
    cov_matrix : np.ndarray
        (n × n) daily log-return covariance matrix.

    Returns
    -------
    float
        Daily portfolio volatility (dimensionless, in return space).
    """
    variance = float(weights @ cov_matrix @ weights)
    return np.sqrt(max(variance, 0.0))      # guard against tiny negative floats


def compute_parametric_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
    confidence: float = None,
) -> float:
    """
    Parametric VaR for a single snapshot.

    VaR_α = z_α · σ_p · V_p

    Parameters
    ----------
    weights : np.ndarray
        Current portfolio weights.
    cov_matrix : np.ndarray
        Daily covariance matrix.
    portfolio_value : float
        Current portfolio value in USD.
    confidence : float, optional
        Confidence level. Defaults to config.PRIMARY_CONFIDENCE.

    Returns
    -------
    float
        Dollar VaR (positive = maximum expected loss at confidence α).
    """
    confidence = confidence or config.PRIMARY_CONFIDENCE
    z_alpha    = _Z.get(confidence) or stats.norm.ppf(confidence)
    sigma_p    = compute_portfolio_volatility(weights, cov_matrix)
    return z_alpha * sigma_p * portfolio_value


def compute_var_95_99(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
) -> dict:
    """
    Compute parametric VaR at 95 % and 99 % in one call.

    Parameters
    ----------
    weights : np.ndarray
        Current portfolio weights.
    cov_matrix : np.ndarray
        Daily covariance matrix.
    portfolio_value : float
        Current portfolio value in USD.

    Returns
    -------
    dict
        {var_95, var_99, sigma_p, portfolio_value}
    """
    sigma_p = compute_portfolio_volatility(weights, cov_matrix)

    z_95 = _Z.get(0.95, stats.norm.ppf(0.95))
    z_99 = _Z.get(0.99, stats.norm.ppf(0.99))

    return {
        "var_95":          z_95 * sigma_p * portfolio_value,
        "var_99":          z_99 * sigma_p * portfolio_value,
        "sigma_p":         sigma_p,
        "portfolio_value": portfolio_value,
    }


# ---------------------------------------------------------------------------
# Intraday replay
# ---------------------------------------------------------------------------

def run_parametric_var_replay(
    shares: pd.Series,
    intraday_prices: pd.DataFrame,
    cov_matrix_rolling: pd.DataFrame,
    cov_matrix_ewma: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Walk through 1-minute intraday price data and compute parametric VaR
    at every bar for both the rolling-covariance and EWMA estimators.

    At each minute t:
        1. Update portfolio value   V_{p,t}
        2. Compute dynamic weights  w_{i,t}
        3. Compute VaR_95, VaR_99   (rolling covariance)
        4. Compute VaR_95, VaR_99   (EWMA covariance)
        5. Record wall-clock latency for each estimator

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts; index = tickers.
    intraday_prices : pd.DataFrame
        1-minute prices; DatetimeIndex, columns = tickers.
    cov_matrix_rolling : pd.DataFrame
        Rolling sample covariance (updated once at market open).
    cov_matrix_ewma : pd.DataFrame
        EWMA covariance (updated once at market open).
    save : bool
        Persist results to data/processed/var_results.csv.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp.  Columns include portfolio_value,
        var_95/99 for both methods, per-ticker weights, and latencies.
    """
    tickers = shares.index.tolist()
    prices  = intraday_prices[tickers].dropna()

    # Pre-convert to numpy for speed inside the loop
    arr_rolling = cov_matrix_rolling.loc[tickers, tickers].values.astype(float)
    arr_ewma    = cov_matrix_ewma.loc[tickers, tickers].values.astype(float)
    shares_arr  = shares[tickers].values.astype(float)

    records = []
    print(f"[parametric_var] Replay over {len(prices)} bars …")

    for timestamp, row in prices.iterrows():
        price_arr = row[tickers].values.astype(float)

        # Portfolio value and weights (inline for minimal overhead)
        position_vals = shares_arr * price_arr
        port_value    = position_vals.sum()
        weights       = position_vals / port_value

        # ---- Rolling covariance VaR ----------------------------------------
        t0 = time.perf_counter()
        res_r = compute_var_95_99(weights, arr_rolling, port_value)
        lat_rolling = (time.perf_counter() - t0) * 1_000   # → ms

        # ---- EWMA covariance VaR -------------------------------------------
        t1 = time.perf_counter()
        res_e = compute_var_95_99(weights, arr_ewma, port_value)
        lat_ewma = (time.perf_counter() - t1) * 1_000      # → ms

        rec = {
            "timestamp":          timestamp,
            "portfolio_value":    port_value,
            "var_95_rolling":     res_r["var_95"],
            "var_99_rolling":     res_r["var_99"],
            "var_95_ewma":        res_e["var_95"],
            "var_99_ewma":        res_e["var_99"],
            "sigma_p_rolling":    res_r["sigma_p"],
            "sigma_p_ewma":       res_e["sigma_p"],
            "latency_rolling_ms": lat_rolling,
            "latency_ewma_ms":    lat_ewma,
        }
        for i, ticker in enumerate(tickers):
            rec[f"weight_{ticker}"] = weights[i]

        records.append(rec)

    results = pd.DataFrame(records).set_index("timestamp")

    if save:
        os.makedirs(config.DATA_PROCESSED, exist_ok=True)
        path = os.path.join(config.DATA_PROCESSED, "var_results.csv")
        results.to_csv(path)
        print(f"[parametric_var] Saved → {path}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n[parametric_var] Replay complete  ({len(results)} bars)")
    print(f"  Avg 99% VaR  Rolling : ${results['var_99_rolling'].mean():>10,.2f}")
    print(f"  Avg 99% VaR  EWMA    : ${results['var_99_ewma'].mean():>10,.2f}")
    print(f"  Avg latency  Rolling : {results['latency_rolling_ms'].mean():.4f} ms")
    print(f"  Avg latency  EWMA    : {results['latency_ewma_ms'].mean():.4f} ms")

    return results
