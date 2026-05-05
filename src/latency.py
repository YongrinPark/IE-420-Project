"""
latency.py
==========
Utilities for measuring and summarising computation latency.

Latency is the primary engineering metric in this project:
it quantifies the computational cost of each VaR method.
"""

import os
import sys
import time
from typing import Callable, Any

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def time_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Execute `func(*args, **kwargs)` and return (result, latency_ms).

    Parameters
    ----------
    func : callable
        Any callable to benchmark.
    *args, **kwargs
        Passed directly to func.

    Returns
    -------
    tuple
        (result, latency_ms)  where latency_ms is a float.
    """
    t0     = time.perf_counter()
    result = func(*args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1_000
    return result, latency_ms


def record_latency(
    latency_records: list,
    method: str,
    latency_ms: float,
    n_paths: int = None,
) -> None:
    """
    Append one latency observation to a mutable list.

    Parameters
    ----------
    latency_records : list
        Mutable list of dicts (accumulates across calls).
    method : str
        Human-readable method label, e.g. 'Parametric_Rolling', 'MC_5000'.
    latency_ms : float
        Wall-clock latency in milliseconds.
    n_paths : int, optional
        Monte Carlo path count (None for parametric methods).
    """
    latency_records.append({
        "method":     method,
        "latency_ms": latency_ms,
        "n_paths":    n_paths,
    })


def summarize_latency(
    latency_records: list,
    save: bool = True,
) -> pd.DataFrame:
    """
    Aggregate latency observations into a summary table.

    Computes mean, median, std, min, max, and 95th-percentile latency
    for each unique method label.

    Parameters
    ----------
    latency_records : list
        List of dicts with keys 'method' and 'latency_ms'.
    save : bool
        Persist to results/tables/latency_summary.csv.

    Returns
    -------
    pd.DataFrame
        One row per method with descriptive latency statistics.
    """
    df = pd.DataFrame(latency_records)

    def p95(x):
        return float(np.percentile(x, 95))

    summary = (
        df.groupby("method")["latency_ms"]
        .agg(
            n_obs   ="count",
            mean_ms ="mean",
            median_ms="median",
            std_ms  ="std",
            min_ms  ="min",
            max_ms  ="max",
        )
        .reset_index()
    )

    # Append 95th-percentile column separately (lambda in agg causes issues in
    # some pandas versions when combined with named agg)
    p95_vals = df.groupby("method")["latency_ms"].apply(p95).reset_index()
    p95_vals.columns = ["method", "p95_ms"]
    summary = summary.merge(p95_vals, on="method")

    if save:
        os.makedirs(config.RESULTS_TABLES, exist_ok=True)
        path = os.path.join(config.RESULTS_TABLES, "latency_summary.csv")
        summary.to_csv(path, index=False)
        print(f"[latency] Saved summary → {path}")

    return summary
