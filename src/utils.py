"""
utils.py
========
General-purpose helper functions for the IE 420 VaR project.
"""

import os
import sys
import shutil

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    directory: str = None,
) -> str:
    """
    Save a DataFrame to CSV, creating the directory if it does not exist.

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
    directory : str, optional
        Defaults to config.DATA_PROCESSED.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    directory = directory or config.DATA_PROCESSED
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    df.to_csv(path)
    print(f"[utils] Saved → {path}")
    return path


def load_dataframe(
    filename: str,
    directory: str = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV from the specified directory (default: data/processed/).

    Parameters
    ----------
    filename : str
    directory : str, optional
    **read_csv_kwargs
        Forwarded to pd.read_csv (index_col and parse_dates default to 0 / True).

    Returns
    -------
    pd.DataFrame
    """
    directory = directory or config.DATA_PROCESSED
    path      = os.path.join(directory, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] File not found: {path}")

    kwargs = {"index_col": 0, "parse_dates": True}
    kwargs.update(read_csv_kwargs)
    df = pd.read_csv(path, **kwargs)
    print(f"[utils] Loaded '{filename}'  shape={df.shape}")
    return df


def copy_file(src: str, dst_dir: str, overwrite: bool = True) -> str:
    """
    Copy a file into a target directory, creating it if necessary.

    Parameters
    ----------
    src : str
        Source file path.
    dst_dir : str
        Destination directory.
    overwrite : bool
        If False and the destination exists, skip the copy.

    Returns
    -------
    str
        Full destination path.
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not overwrite and os.path.exists(dst):
        print(f"[utils] Skipping (exists): {dst}")
        return dst
    shutil.copy2(src, dst)
    print(f"[utils] Copied {src} → {dst}")
    return dst


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_currency(value: float, symbol: str = "$") -> str:
    """Return a dollar-formatted string, e.g. '$12,345.67'."""
    return f"{symbol}{value:,.2f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Return a percentage string, e.g. '1.23%'."""
    return f"{value * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_inputs(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
) -> None:
    """
    Validate inputs for VaR computation.

    Raises
    ------
    ValueError
        If weights do not sum to 1, the covariance matrix is mis-shaped,
        portfolio value is non-positive, or the matrix is not PSD.
    """
    n = len(weights)

    if not np.isclose(weights.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"Weights must sum to 1.  Got: {weights.sum():.6f}"
        )

    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} does not match "
            f"weight vector length {n}."
        )

    if portfolio_value <= 0:
        raise ValueError(
            f"Portfolio value must be positive.  Got: {portfolio_value}"
        )

    # Check positive semi-definiteness (smallest eigenvalue ≥ -ε)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigenvalues < -1e-8):
        raise ValueError(
            "Covariance matrix is not positive semi-definite.  "
            f"Smallest eigenvalue: {eigenvalues.min():.2e}"
        )
