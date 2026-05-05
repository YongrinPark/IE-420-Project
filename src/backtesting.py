"""
backtesting.py
==============
VaR model validation via exceedance analysis and the Kupiec PoF test.

A VaR breach (exceedance) occurs when the realised daily portfolio loss
exceeds the previous day's VaR forecast.

Kupiec Proportion of Failures (PoF) test
-----------------------------------------
H0: the observed breach rate equals the model-implied rate p = 1 − α.

Likelihood-ratio statistic:
    LR_POF = −2 ln [ p^x (1−p)^(T−x) / (x/T)^x (1−x/T)^(T−x) ]
           ~ χ²(1) under H0.

Reject H0 when LR_POF > χ²_1(1−β), i.e. p-value < β.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_actual_losses(
    shares: pd.Series,
    daily_prices: pd.DataFrame,
) -> pd.Series:
    """
    Compute the daily dollar loss series from close-to-close price changes.

    Loss_t = V_{p,t-1} − V_{p,t}   (positive when portfolio declines)

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts; index = tickers.
    daily_prices : pd.DataFrame
        Daily adjusted closing prices; index = Date, columns = tickers.

    Returns
    -------
    pd.Series
        Daily losses (positive = loss); name = 'actual_loss'.
    """
    tickers    = shares.index.tolist()
    prices     = daily_prices[tickers].dropna()
    port_vals  = prices.dot(shares)               # portfolio value each day
    losses     = -port_vals.diff().dropna()       # negative of change = loss
    losses.name = "actual_loss"
    return losses


def identify_var_breaches(
    actual_losses: pd.Series,
    var_forecasts: pd.Series,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Mark days where the actual loss exceeded the VaR forecast.

    Parameters
    ----------
    actual_losses : pd.Series
        Realised daily losses (positive = loss); DatetimeIndex.
    var_forecasts : pd.Series
        VaR dollar threshold from the previous day's model; same index.
    confidence : float
        Confidence level used to label the result column.

    Returns
    -------
    pd.DataFrame
        Columns: actual_loss, var_forecast, breach.
    """
    df = pd.DataFrame({
        "actual_loss":  actual_losses,
        "var_forecast": var_forecasts,
    }).dropna()

    label = f"breach_{int(confidence * 100)}"
    df[label] = df["actual_loss"] > df["var_forecast"]
    return df


def kupiec_test(
    n_obs: int,
    n_breaches: int,
    confidence: float = 0.99,
) -> dict:
    """
    Kupiec Proportion of Failures (PoF) likelihood-ratio test.

    Parameters
    ----------
    n_obs : int
        Total number of observations T.
    n_breaches : int
        Number of VaR breaches x.
    confidence : float
        Model confidence level α (expected exceedance rate = 1 − α).

    Returns
    -------
    dict
        {n_obs, n_breaches, expected_breaches, breach_rate,
         expected_rate, lr_stat, p_value, reject_h0}
    """
    p  = 1.0 - confidence          # expected exceedance probability
    x  = n_breaches
    T  = n_obs

    if x == 0:
        # No breaches: LR = -2 ln(1-p)^T
        lr_stat = -2 * T * np.log(1 - p)
    elif x == T:
        lr_stat = np.inf
    else:
        hat_p = x / T
        lr_stat = -2 * (
            x * np.log(p / hat_p) + (T - x) * np.log((1 - p) / (1 - hat_p))
        )

    p_value  = 1 - stats.chi2.cdf(lr_stat, df=1)
    reject   = p_value < 0.05

    return {
        "n_obs":              T,
        "n_breaches":         x,
        "expected_breaches":  round(T * p, 2),
        "breach_rate":        round(x / T, 4) if T > 0 else np.nan,
        "expected_rate":      p,
        "lr_stat":            round(lr_stat, 4),
        "p_value":            round(p_value, 4),
        "reject_h0":          reject,
    }


def summarize_backtest_results(
    daily_prices: pd.DataFrame,
    shares: pd.Series,
    log_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    confidence: float = 0.99,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the full backtesting pipeline and return a summary table.

    Steps
    -----
    1. Compute daily VaR forecasts using rolling covariance (updated each day).
    2. Compute actual daily losses.
    3. Align forecasts with realised losses (forecast is 1-day ahead).
    4. Count breaches and run Kupiec test.
    5. Save detailed breach series to data/processed/backtest_series.csv.

    Parameters
    ----------
    daily_prices : pd.DataFrame
        Daily adjusted closing prices.
    shares : pd.Series
        Fixed share counts.
    log_returns : pd.DataFrame
        Daily log returns (pre-computed).
    cov_matrix : pd.DataFrame
        Covariance matrix (used as a fixed estimate here for simplicity;
        a rolling update version is used inside the loop).
    confidence : float
        VaR confidence level.
    save : bool
        Persist results to results/tables/backtesting_summary.csv.

    Returns
    -------
    pd.DataFrame
        Summary row(s) with Kupiec test results.
    """
    from src.parametric_var import compute_var_95_99

    tickers    = shares.index.tolist()
    prices     = daily_prices[tickers].dropna()
    returns    = log_returns[tickers].dropna()

    # Align dates: use returns index as the valid date range
    common_dates = prices.index.intersection(returns.index)
    prices  = prices.loc[common_dates]
    returns = returns.loc[common_dates]

    window  = config.COV_WINDOW
    z_alpha = stats.norm.ppf(confidence)

    var_forecasts = []
    dates         = []

    for i in range(window, len(prices) - 1):
        # Rolling covariance over preceding `window` days
        ret_window = returns.iloc[i - window : i]
        cov_w      = ret_window.cov().values

        prices_row  = prices.iloc[i][tickers].values.astype(float)
        shares_arr  = shares[tickers].values.astype(float)
        port_value  = float(np.dot(shares_arr, prices_row))
        weights     = shares_arr * prices_row / port_value

        res = compute_var_95_99(weights, cov_w, port_value)

        # Forecast applies to the *next* trading day
        dates.append(prices.index[i + 1])
        var_forecasts.append(res[f"var_{int(confidence * 100)}"])

    forecast_series = pd.Series(var_forecasts, index=dates, name="var_forecast")

    # Actual losses
    actual_losses = compute_actual_losses(shares, daily_prices)
    actual_losses = actual_losses.loc[actual_losses.index.intersection(forecast_series.index)]

    breach_df = identify_var_breaches(actual_losses, forecast_series, confidence)
    breach_col = f"breach_{int(confidence * 100)}"

    # Kupiec test
    kupiec = kupiec_test(len(breach_df), int(breach_df[breach_col].sum()), confidence)

    # Persist the detailed series
    if save:
        os.makedirs(config.DATA_PROCESSED, exist_ok=True)
        breach_path = os.path.join(config.DATA_PROCESSED, "backtest_series.csv")
        breach_df.rename(columns={"var_forecast": "var_99"}).to_csv(breach_path)
        print(f"[backtesting] Saved breach series → {breach_path}")

    summary = pd.DataFrame([{
        "Model":             "Parametric (Rolling)",
        "Confidence":        confidence,
        "T":                 kupiec["n_obs"],
        "Breaches":          kupiec["n_breaches"],
        "Expected Breaches": kupiec["expected_breaches"],
        "Breach Rate":       kupiec["breach_rate"],
        "LR Stat":           kupiec["lr_stat"],
        "p-value":           kupiec["p_value"],
        "Reject H0 (5%)":    kupiec["reject_h0"],
    }])

    if save:
        os.makedirs(config.RESULTS_TABLES, exist_ok=True)
        path = os.path.join(config.RESULTS_TABLES, "backtesting_summary.csv")
        summary.to_csv(path, index=False)
        print(f"[backtesting] Saved summary → {path}")

    print(f"\n[backtesting] Results:")
    print(f"  T = {kupiec['n_obs']}  |  Breaches: {kupiec['n_breaches']}"
          f"  (expected {kupiec['expected_breaches']})")
    print(f"  LR stat = {kupiec['lr_stat']:.4f}  |  p-value = {kupiec['p_value']:.4f}")
    print(f"  Reject H0: {kupiec['reject_h0']}")

    return summary
