"""
monte_carlo_var.py
==================
Monte Carlo VaR via Geometric Brownian Motion (GBM).

Mathematical framework
----------------------
For each simulated path m = 1 … M:

    ε^(m) = L z^(m),   z^(m) ~ N(0, I)       (correlated shocks via Cholesky)

    P_i^(m) = P_{i,t} · exp(ε_i^(m))         (1-day GBM step, μ=0 assumption)

    L^(m)   = V_{p,t} − Σ_i n_i P_i^(m)     (simulated portfolio loss)

    VaR_α^MC = quantile({L^(m)}, 1−α)        (empirical quantile)

All path generation uses vectorised NumPy operations for speed.
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


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def generate_correlated_shocks(
    cov_matrix: np.ndarray,
    n_paths: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate correlated standard normal shocks via the Cholesky decomposition.

    Σ = L Lᵀ  →  ε = L z,  z ~ N(0, I)

    Parameters
    ----------
    cov_matrix : np.ndarray
        (n × n) daily covariance matrix.
    n_paths : int
        Number of simulation paths M.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Correlated shocks, shape (n_paths, n_assets).
    """
    if rng is None:
        rng = np.random.default_rng(config.MC_RANDOM_SEED)

    n = cov_matrix.shape[0]
    L = np.linalg.cholesky(cov_matrix)          # lower-triangular factor

    z = rng.standard_normal((n_paths, n))        # iid standard normals
    return z @ L.T                               # correlated shocks (M × n)


def simulate_gbm_paths(
    current_prices: np.ndarray,
    cov_matrix: np.ndarray,
    n_paths: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate 1-day ahead prices for each asset under GBM.

    P_i^(m) = P_{i,t} · exp(ε_i^(m))   (zero-drift, 1-day horizon)

    Parameters
    ----------
    current_prices : np.ndarray
        Current prices for each asset, shape (n_assets,).
    cov_matrix : np.ndarray
        Daily covariance matrix (n × n).
    n_paths : int
        Number of paths to simulate.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        Simulated prices, shape (n_paths, n_assets).
    """
    shocks = generate_correlated_shocks(cov_matrix, n_paths, rng)
    # Broadcast: each row of shocks is added to log(current_prices)
    log_prices = np.log(current_prices) + shocks    # (M × n)
    return np.exp(log_prices)


def compute_mc_var(
    shares: np.ndarray,
    current_prices: np.ndarray,
    cov_matrix: np.ndarray,
    n_paths: int,
    portfolio_value: float,
    rng: np.random.Generator = None,
) -> dict:
    """
    Compute Monte Carlo VaR at 95% and 99% for one snapshot.

    Parameters
    ----------
    shares : np.ndarray
        Fixed share counts, shape (n_assets,).
    current_prices : np.ndarray
        Current prices, shape (n_assets,).
    cov_matrix : np.ndarray
        Daily covariance matrix.
    n_paths : int
        Number of simulation paths.
    portfolio_value : float
        Current portfolio value V_{p,t}.
    rng : np.random.Generator, optional

    Returns
    -------
    dict
        {var_95, var_99, n_paths, portfolio_value}
    """
    sim_prices  = simulate_gbm_paths(current_prices, cov_matrix, n_paths, rng)
    sim_values  = sim_prices @ shares                  # simulated portfolio values (M,)
    losses      = portfolio_value - sim_values         # positive = loss

    var_95 = float(np.percentile(losses, 95))
    var_99 = float(np.percentile(losses, 99))

    return {
        "var_95":          var_95,
        "var_99":          var_99,
        "n_paths":         n_paths,
        "portfolio_value": portfolio_value,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_monte_carlo_experiment(
    shares: pd.Series,
    current_prices: pd.Series,
    cov_matrix: pd.DataFrame,
    path_counts: list = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Benchmark Monte Carlo VaR across multiple path counts.

    For each M in path_counts, simulate M GBM paths and measure:
        - VaR at 95% and 99%
        - Wall-clock latency (ms)

    Also reports the equivalent parametric VaR for comparison.

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts; index = tickers.
    current_prices : pd.Series
        Prices at the snapshot time; index = tickers.
    cov_matrix : pd.DataFrame
        Daily covariance matrix.
    path_counts : list, optional
        List of path counts to test. Defaults to config.MC_PATHS.
    save : bool
        Persist results to results/tables/model_comparison.csv.

    Returns
    -------
    pd.DataFrame
        One row per path count with VaR estimates and latency.
    """
    path_counts = path_counts or config.MC_PATHS
    tickers     = shares.index.tolist()

    shares_arr  = shares[tickers].values.astype(float)
    prices_arr  = current_prices[tickers].values.astype(float)
    cov_arr     = cov_matrix.loc[tickers, tickers].values.astype(float)
    port_value  = float(np.dot(shares_arr, prices_arr))

    # Parametric 99% VaR for reference
    weights     = shares_arr * prices_arr / port_value
    sigma_p     = np.sqrt(weights @ cov_arr @ weights)
    z_95, z_99  = stats.norm.ppf(0.95), stats.norm.ppf(0.99)
    var_99_param = z_99 * sigma_p * port_value
    var_95_param = z_95 * sigma_p * port_value

    rng = np.random.default_rng(config.MC_RANDOM_SEED)

    records = []
    print(f"[monte_carlo] Running MC VaR benchmark  (portfolio value = ${port_value:,.2f})")

    for M in path_counts:
        t0 = time.perf_counter()
        res = compute_mc_var(shares_arr, prices_arr, cov_arr, M, port_value, rng)
        latency_ms = (time.perf_counter() - t0) * 1_000

        record = {
            "n_paths":           M,
            "var_95_mc":         res["var_95"],
            "var_99_mc":         res["var_99"],
            "var_95_parametric": var_95_param,
            "var_99_parametric": var_99_param,
            "latency_ms":        latency_ms,
        }
        records.append(record)
        print(f"  M={M:>6,}  VaR99=${res['var_99']:>10,.2f}  latency={latency_ms:>8.2f} ms")

    results = pd.DataFrame(records)

    if save:
        os.makedirs(config.RESULTS_TABLES, exist_ok=True)
        path = os.path.join(config.RESULTS_TABLES, "model_comparison.csv")
        results.to_csv(path, index=False)
        print(f"[monte_carlo] Saved → {path}")

    return results


def run_mc_var_replay(
    shares: pd.Series,
    intraday_prices: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    n_paths: int = 10_000,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run Monte Carlo VaR at every intraday bar (for a single path count).

    Used to produce a time series of MC VaR estimates comparable to
    the parametric replay output.

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts.
    intraday_prices : pd.DataFrame
        1-minute prices with DatetimeIndex.
    cov_matrix : pd.DataFrame
        Daily covariance matrix.
    n_paths : int
        Number of simulation paths per bar.
    save : bool
        Persist results to data/processed/mc_var_results.csv.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp; columns: var_95_mc, var_99_mc, latency_mc_ms.
    """
    tickers    = shares.index.tolist()
    prices_df  = intraday_prices[tickers].dropna()
    shares_arr = shares[tickers].values.astype(float)
    cov_arr    = cov_matrix.loc[tickers, tickers].values.astype(float)
    rng        = np.random.default_rng(config.MC_RANDOM_SEED)

    records = []
    print(f"[monte_carlo] Intraday MC replay  M={n_paths:,}  over {len(prices_df)} bars …")

    for timestamp, row in prices_df.iterrows():
        prices_arr = row[tickers].values.astype(float)
        port_value = float(np.dot(shares_arr, prices_arr))

        t0  = time.perf_counter()
        res = compute_mc_var(shares_arr, prices_arr, cov_arr, n_paths, port_value, rng)
        lat = (time.perf_counter() - t0) * 1_000

        records.append({
            "timestamp":    timestamp,
            "var_95_mc":    res["var_95"],
            "var_99_mc":    res["var_99"],
            "latency_mc_ms": lat,
        })

    results = pd.DataFrame(records).set_index("timestamp")

    if save:
        os.makedirs(config.DATA_PROCESSED, exist_ok=True)
        path = os.path.join(config.DATA_PROCESSED, "mc_var_results.csv")
        results.to_csv(path)
        print(f"[monte_carlo] Saved → {path}")

    print(f"[monte_carlo] Done.  Avg latency per bar: {results['latency_mc_ms'].mean():.2f} ms")
    return results
