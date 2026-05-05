"""
live_engine.py
==============
Background thread that fetches live prices every minute and recomputes VaR.

The engine stores today's full history in memory so the dashboard can plot
the entire intraday session without touching disk.

Usage
-----
    engine = LiveVaREngine(shares, cov_rolling, cov_ewma)
    engine.start()                  # non-blocking — starts a daemon thread
    snapshot = engine.get_latest()  # latest VaR snapshot dict
    df       = engine.get_history() # full today's history as DataFrame
    engine.stop()
"""

import os
import sys
import time
import threading
from collections import deque

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from src.live_data       import get_live_prices, is_market_open
from src.parametric_var  import compute_var_95_99
from src.monte_carlo_var import compute_mc_var


_EMPTY_SNAPSHOT = {
    "timestamp":        None,
    "portfolio_value":  None,
    "var_99_rolling":   None,
    "var_95_rolling":   None,
    "var_99_ewma":      None,
    "var_95_ewma":      None,
    "var_99_mc":        None,
    "var_95_mc":        None,
    "latency_rolling_ms": None,
    "latency_ewma_ms":    None,
    "latency_mc_ms":      None,
    "prices":           {},
    "details":          {},
    "source":           None,
    "error":            None,
}


class LiveVaREngine:
    """
    Continuously fetches live prices and recomputes portfolio VaR.

    Parameters
    ----------
    shares : pd.Series
        Fixed share counts; index = tickers.
    cov_rolling : pd.DataFrame
        Rolling sample covariance matrix (updated at market open).
    cov_ewma : pd.DataFrame
        EWMA covariance matrix (updated at market open).
    interval_sec : int
        Seconds between each price fetch.  Defaults to config.LIVE_INTERVAL_SEC.
    mc_every_n : int
        Run Monte Carlo VaR every N updates (expensive).
        Defaults to config.LIVE_MC_EVERY_N.
    mc_paths : int
        Number of MC simulation paths.
    """

    def __init__(
        self,
        shares: pd.Series,
        cov_rolling: pd.DataFrame,
        cov_ewma: pd.DataFrame,
        interval_sec: int = None,
        mc_every_n: int = None,
        mc_paths: int = 10_000,
    ):
        self.shares      = shares
        self.cov_rolling = cov_rolling.loc[shares.index, shares.index]
        self.cov_ewma    = cov_ewma.loc[shares.index, shares.index]

        self.interval_sec = interval_sec or config.LIVE_INTERVAL_SEC
        self.mc_every_n   = mc_every_n   or config.LIVE_MC_EVERY_N
        self.mc_paths     = mc_paths

        # Pre-convert to numpy for speed inside the loop
        self._arr_rolling = self.cov_rolling.values.astype(float)
        self._arr_ewma    = self.cov_ewma.values.astype(float)
        self._shares_arr  = shares.values.astype(float)
        self._tickers     = shares.index.tolist()

        # Thread-safe state
        self._lock      = threading.Lock()
        self._history   = deque(maxlen=500)   # ~8 h of 1-min bars
        self._latest    = dict(_EMPTY_SNAPSHOT)
        self._is_running = False
        self._thread    = None
        self._update_count = 0
        self._rng = np.random.default_rng(config.MC_RANDOM_SEED)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background update thread (non-blocking)."""
        if self._is_running:
            print("[engine] Already running.")
            return

        self._is_running = True

        # Do one immediate fetch so the dashboard has data right away
        self._fetch_and_compute()

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="LiveVaREngine"
        )
        self._thread.start()
        print(f"[engine] Started — updating every {self.interval_sec}s.")

    def stop(self) -> None:
        """Stop the background thread."""
        self._is_running = False
        print("[engine] Stopped.")

    def get_latest(self) -> dict:
        """Return a copy of the most recent VaR snapshot."""
        with self._lock:
            return dict(self._latest)

    def get_history(self) -> pd.DataFrame:
        """Return today's full history as a DataFrame indexed by timestamp."""
        with self._lock:
            if not self._history:
                return pd.DataFrame()
            df = pd.DataFrame(list(self._history))
            df = df.set_index("timestamp").sort_index()
            return df

    def n_updates(self) -> int:
        """Total number of successful updates since start."""
        with self._lock:
            return self._update_count

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._is_running:
            time.sleep(self.interval_sec)
            try:
                self._fetch_and_compute()
            except Exception as e:
                print(f"[engine] Update error: {e}")
                with self._lock:
                    self._latest["error"] = str(e)

    def _fetch_and_compute(self) -> None:
        """Single update: fetch prices → compute VaR → store snapshot."""
        ts = pd.Timestamp.now()

        # ── 1. Fetch prices ───────────────────────────────────────────
        prices, details, source = get_live_prices(self._tickers)

        if len(prices) < len(self._tickers):
            print(f"[engine] Incomplete prices ({len(prices)}/{len(self._tickers)}). Skipping.")
            with self._lock:
                self._latest["error"] = "Incomplete price data"
            return

        prices_arr = np.array([prices[t] for t in self._tickers], dtype=float)

        # ── 2. Portfolio value & weights ──────────────────────────────
        position_vals = self._shares_arr * prices_arr
        port_value    = position_vals.sum()
        weights       = position_vals / port_value

        # ── 3. Parametric VaR — rolling covariance ────────────────────
        t0 = time.perf_counter()
        var_r = compute_var_95_99(weights, self._arr_rolling, port_value)
        lat_r = (time.perf_counter() - t0) * 1_000

        # ── 4. Parametric VaR — EWMA covariance ──────────────────────
        t1 = time.perf_counter()
        var_e = compute_var_95_99(weights, self._arr_ewma, port_value)
        lat_e = (time.perf_counter() - t1) * 1_000

        # ── 5. Monte Carlo VaR (every mc_every_n updates) ────────────
        var_mc_99 = var_mc_95 = lat_mc = None
        with self._lock:
            n = self._update_count

        if n % self.mc_every_n == 0:
            t2 = time.perf_counter()
            mc = compute_mc_var(
                self._shares_arr, prices_arr,
                self._arr_rolling, self.mc_paths,
                port_value, self._rng,
            )
            lat_mc    = (time.perf_counter() - t2) * 1_000
            var_mc_99 = mc["var_99"]
            var_mc_95 = mc["var_95"]

        # ── 6. Build snapshot and store ───────────────────────────────
        snapshot = {
            "timestamp":           ts,
            "portfolio_value":     port_value,
            "var_99_rolling":      var_r["var_99"],
            "var_95_rolling":      var_r["var_95"],
            "var_99_ewma":         var_e["var_99"],
            "var_95_ewma":         var_e["var_95"],
            "var_99_mc":           var_mc_99,
            "var_95_mc":           var_mc_95,
            "latency_rolling_ms":  lat_r,
            "latency_ewma_ms":     lat_e,
            "latency_mc_ms":       lat_mc,
            "prices":              prices,
            "details":             details,
            "source":              source,
            "error":               None,
        }

        with self._lock:
            self._history.append(snapshot)
            self._latest = snapshot
            self._update_count += 1

        print(
            f"[engine] #{n+1:>4}  {ts.strftime('%H:%M:%S')}  "
            f"V=${port_value:>10,.2f}  "
            f"VaR99(roll)=${var_r['var_99']:>8,.2f}  "
            f"src={source}  "
            f"lat={lat_r:.2f}ms"
        )
