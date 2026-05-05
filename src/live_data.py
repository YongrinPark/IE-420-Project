"""
live_data.py
============
Real-time price fetching for the IE 420 VaR Live Monitor.

Primary source : Finnhub REST API  (real-time, free, 60 calls/min)
Fallback source: yfinance          (last available bar, no API key needed)

Sign up for a free Finnhub key at https://finnhub.io  (no credit card).
Then set it:
    export FINNHUB_API_KEY="your_key_here"
  or add to a .env file in the project root.
"""

import os
import sys
import requests
import time as _time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo          # Python 3.9+ built-in

import pandas as pd
import yfinance as yf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

EASTERN = ZoneInfo("America/New_York")
FINNHUB_BASE = "https://finnhub.io/api/v1"


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """Return True if the US equity market is currently open (9:30–16:00 ET)."""
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:          # Saturday or Sunday
        return False
    return dtime(9, 30) <= now.time() <= dtime(16, 0)


def market_status_label() -> tuple[str, str]:
    """
    Return (status_text, badge_color) for display in the dashboard.
    Colors are Bootstrap contextual colours.
    """
    if is_market_open():
        return "🟢  Market Open", "success"
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return "⚫  Weekend", "secondary"
    if now.time() < dtime(9, 30):
        return "🟡  Pre-Market", "warning"
    return "🔴  Market Closed", "danger"


# ---------------------------------------------------------------------------
# Finnhub REST client
# ---------------------------------------------------------------------------

def _finnhub_key() -> str:
    return (
        os.environ.get("FINNHUB_API_KEY", "")
        or getattr(config, "FINNHUB_API_KEY", "")
    )


def fetch_finnhub_quote(symbol: str, api_key: str) -> dict:
    """
    Call Finnhub /quote for one symbol.

    Returns dict with keys: price, change, change_pct, high, low, open,
    prev_close, timestamp.  Raises on HTTP or data error.
    """
    resp = requests.get(
        f"{FINNHUB_BASE}/quote",
        params={"symbol": symbol, "token": api_key},
        timeout=6,
    )
    resp.raise_for_status()
    d = resp.json()

    if not d.get("c"):
        raise ValueError(f"Empty quote for {symbol}")

    return {
        "price":      float(d["c"]),
        "change":     float(d.get("d",  0)),
        "change_pct": float(d.get("dp", 0)),
        "high":       float(d.get("h",  0)),
        "low":        float(d.get("l",  0)),
        "open":       float(d.get("o",  0)),
        "prev_close": float(d.get("pc", 0)),
        "timestamp":  pd.Timestamp.now(),
    }


def fetch_all_finnhub(tickers: list = None, api_key: str = None) -> tuple[dict, dict]:
    """
    Fetch quotes for all tickers from Finnhub.

    Returns
    -------
    prices : dict  {ticker: float}
    details: dict  {ticker: full quote dict}
    """
    tickers = tickers or config.TICKERS
    api_key = api_key or _finnhub_key()

    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is not set.")

    prices, details = {}, {}
    for ticker in tickers:
        try:
            q = fetch_finnhub_quote(ticker, api_key)
            prices[ticker]  = q["price"]
            details[ticker] = q
            _time.sleep(0.1)        # gentle rate-limit spacing
        except Exception as e:
            print(f"[live_data] Finnhub {ticker}: {e}")

    return prices, details


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def fetch_all_yfinance(tickers: list = None) -> tuple[dict, dict]:
    """
    Fetch latest available prices via yfinance (no API key required).
    During market hours this is roughly the last completed 1-minute bar.
    After hours this is the most recent close.

    Returns
    -------
    prices : dict  {ticker: float}
    details: dict  {ticker: {price, change, change_pct, ...}}
    """
    tickers = tickers or config.TICKERS

    try:
        raw = yf.download(
            tickers,
            period="5d",          # grab 5 days to ensure we have prev close
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[live_data] yfinance download error: {e}")
        return {}, {}

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    closes = closes.ffill()
    latest = closes.iloc[-1]

    # Previous close = last price from the day before today
    today = pd.Timestamp.now().normalize()
    prev_day = closes[closes.index.normalize() < today]
    prev_close = prev_day.iloc[-1] if not prev_day.empty else latest

    prices, details = {}, {}
    for t in tickers:
        price = float(latest.get(t, 0))
        pc    = float(prev_close.get(t, price))
        chg   = price - pc
        prices[t] = price
        details[t] = {
            "price":      price,
            "change":     chg,
            "change_pct": (chg / pc * 100) if pc else 0,
            "high":       float(closes[t].resample("1D").max().iloc[-1]),
            "low":        float(closes[t].resample("1D").min().iloc[-1]),
            "open":       float(closes[t].resample("1D").first().iloc[-1]),
            "prev_close": pc,
            "timestamp":  closes.index[-1],
        }

    return prices, details


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_live_prices(tickers: list = None) -> tuple[dict, dict, str]:
    """
    Get the latest prices, automatically choosing the best available source.

    Priority:
        1. Finnhub REST API  (if FINNHUB_API_KEY is set)
        2. yfinance          (always available, ~1 min latency)

    Returns
    -------
    prices  : {ticker: float}
    details : {ticker: quote_dict}
    source  : 'finnhub' | 'yfinance'
    """
    key = _finnhub_key()

    if key:
        try:
            prices, details = fetch_all_finnhub(tickers, key)
            if len(prices) == len(tickers or config.TICKERS):
                return prices, details, "finnhub"
            print("[live_data] Finnhub incomplete — falling back to yfinance.")
        except Exception as e:
            print(f"[live_data] Finnhub failed ({e}) — falling back to yfinance.")

    prices, details = fetch_all_yfinance(tickers)
    return prices, details, "yfinance"
