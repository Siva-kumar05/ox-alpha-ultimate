"""Online-dataset validation: real history through the SSRF guard.

Fetches public daily CSVs from stooq.com (no key, no auth) for a set of
symbols (NIFTY index, BTC-USD) and runs the purged walk-forward evaluation
of a small set of *reference* strategies from the classic quant literature
(Donchian breakout, EMA cross, RSI(2) pullback — the same families as the
quant-trading repo's backtests).  Purpose: sanity-check the platform's
validation machinery against real data and give honest out-of-sample
numbers, including full cost drag.

Offline-safe: any fetch failure prints a clear message and exits 0 with a
SKIP, so CI/paper environments without network don't fail.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import indicators as I
from .agents.tearsheet import compute_metrics
from .purged_cv import leakage_report, purged_walk_forward
from .ssrf import SafeURLViolation, safe_fetch

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5y&interval=1d"
BINANCE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=1000"

# symbol -> (source, url symbol)
SYMBOLS = {
    "^NSEI": ("yahoo", "%5ENSEI"),
    "BTCUSDT": ("binance", "BTCUSDT"),
    "ETHUSDT": ("binance", "ETHUSDT"),
}

COSTS_PER_TRADE = 0.0012  # 12 bps round-trip: fees + slippage, conservative for daily bars


def fetch_daily(symbol: str) -> np.ndarray:
    """Fetch adjusted-close daily series (Yahoo chart JSON or Binance klines)."""
    source, url_sym = SYMBOLS[symbol]
    if source == "yahoo":
        raw = safe_fetch(YAHOO_URL.format(symbol=url_sym), timeout=15, max_bytes=8_000_000)
        import json as _json

        payload = _json.loads(raw)
        quote = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        closes = [float(c) for c in quote if c is not None]
    else:
        raw = safe_fetch(BINANCE_URL.format(symbol=url_sym), timeout=15, max_bytes=8_000_000)
        import json as _json

        klines = _json.loads(raw)
        closes = [float(k[4]) for k in klines]
    if len(closes) < 400:
        raise SafeURLViolation(f"insufficient history for {symbol}: {len(closes)} rows")
    return np.asarray(closes, dtype=float)


# ── reference strategies (each: closes -> position array of 0/1) ────────────

def strat_donchian(closes: np.ndarray) -> np.ndarray:
    upper, mid, lower = I.donchian(closes, closes, 20)
    pos = np.zeros(len(closes))
    in_pos = False
    for i in range(1, len(closes)):
        if not np.isnan(upper[i - 1]):
            if not in_pos and closes[i] > upper[i - 1]:
                in_pos = True
            elif in_pos and closes[i] < mid[i - 1]:
                in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pos


def strat_ema_cross(closes: np.ndarray) -> np.ndarray:
    fast, slow = I.ema(closes, 20), I.ema(closes, 50)
    with np.errstate(invalid="ignore"):
        pos = (fast > slow).astype(float)
    pos[np.isnan(slow)] = 0.0
    return pos


def strat_rsi2(closes: np.ndarray) -> np.ndarray:
    rsi = I.rsi(closes, 2)
    pos = np.zeros(len(closes))
    in_pos = False
    for i in range(1, len(closes)):
        if not np.isnan(rsi[i]):
            if not in_pos and rsi[i] < 10:
                in_pos = True
            elif in_pos and rsi[i] > 60:
                in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pos


STRATEGIES = {
    "donchian_20": strat_donchian,
    "ema_20_50": strat_ema_cross,
    "rsi2_pullback": strat_rsi2,
}


def evaluate(symbol: str, closes: np.ndarray) -> List[Dict]:
    """Purged walk-forward evaluation; one honest OOS metric set per strategy."""
    n = len(closes)
    results = []
    rets = np.diff(np.log(closes))
    for name, fn in STRATEGIES.items():
        pos = fn(closes)
        pnl_all, pnl_oos = [], []
        for train_idx, test_idx in purged_walk_forward(
                n, train_bars=750, test_bars=125, embargo_bars=10, horizon_bars=2):
            # Strategy has no fitted parameters here, so training is unused;
            # the folds still define honest disjoint OOS windows.
            for i in test_idx:
                if i + 1 < n:
                    pnl = pos[i] * rets[i] - abs(pos[i] - pos[i - 1]) * COSTS_PER_TRADE
                    pnl_oos.append(float(pnl))
                    pnl_all.append(float(pnl))
        m_oos = compute_metrics(pnl_oos, budget=1.0)
        results.append({
            "strategy": name,
            "total_return_oos": float(np.sum(pnl_oos)),
            "sharpe_oos": m_oos.sharpe,
            "max_dd_oos": m_oos.max_drawdown,
            "win_rate": m_oos.win_rate,
            "trades": m_oos.trades,
        })
    return results


def run() -> int:
    print("ONLINE VALIDATION — public daily data (Yahoo chart API + Binance klines), "
          f"purged walk-forward, costs {COSTS_PER_TRADE:.2%}/turn")
    print(f"CV structure: {leakage_report(2000, 750, 125, 10, 2)}")
    any_data = False
    for symbol, (source, _url_sym) in SYMBOLS.items():
        try:
            closes = fetch_daily(symbol)
        except Exception as exc:
            print(f"  {symbol}: SKIP ({exc.__class__.__name__}: {exc})")
            continue
        any_data = True
        buy_hold = float(np.sum(np.diff(np.log(closes))))
        print(f"\n{symbol} [{source}]: {len(closes)} bars, buy&hold {buy_hold:+.1%} (log)")
        for row in evaluate(symbol, closes):
            print(f"  {row['strategy']:<14} OOS ret {row['total_return_oos']:+7.1%}  "
                  f"sharpe {row['sharpe_oos']:+5.2f}  maxDD {row['max_dd_oos']:5.1%}  "
                  f"win {row['win_rate']:4.0%}  n={row['trades']}")
    if not any_data:
        print("\nNo network access to public datasets — validation skipped (not failed).")
    else:
        print("\nRead honestly: these are reference strategies on daily bars, not the live "
              "intraday agents. They bound expectations and exercise the validation path.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
