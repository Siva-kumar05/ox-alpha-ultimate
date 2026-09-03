"""Month-odds simulator: distribution of 1-month returns on a fixed capital.

Bootstraps *out-of-sample* daily P&L series (from `ox.validate_online`'s
real-data folds, or synthetic edge scenarios) over ~22 trading days and
reports the honest distribution of total return: mean, median, mode (of a
2%-wide histogram), percentile band, and probability of profit / ruin.

This is a description of variance, not a forecast: bootstrapping assumes
daily returns are i.i.d. and the past repeats — neither is true.  The value
is in seeing how leverage and edge size move the WHOLE distribution, not
just the average.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .agents.tearsheet import compute_metrics
from .purged_cv import purged_walk_forward

TRADING_DAYS_PER_MONTH = 22


def simulate(
    daily_pnl: np.ndarray,
    capital: float = 5000.0,
    risk_multiple: float = 1.0,
    days: int = TRADING_DAYS_PER_MONTH,
    paths: int = 100_000,
    seed: int = 11,
) -> Dict[str, float]:
    """Bootstrap ``paths`` months from i.i.d. draws of ``daily_pnl``.

    ``risk_multiple`` is the fraction of equity deployed per day (leverage).
    A day whose loss would take equity to/below zero liquidates the month
    at -100% (ruin), matching how leveraged positions actually die.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(daily_pnl, dtype=float), size=(paths, days), replace=True)
    factors = 1.0 + risk_multiple * draws
    ruined = (factors <= 0).any(axis=1)
    multiples = np.where(ruined, 0.0, np.prod(np.clip(factors, 1e-12, None), axis=1))
    returns = multiples - 1.0

    # Mode from a 2%-wide histogram over the finite return range.
    lo, hi = -100.0, max(100.0, float(np.percentile(returns, 99.5) * 100) + 10)
    counts, edges = np.histogram(returns * 100.0, bins=np.arange(lo, hi + 2.0, 2.0))
    mode_bin = float(edges[int(np.argmax(counts))])
    mode_val = mode_bin + 1.0  # bin centre

    return {
        "mean_pct": float(returns.mean() * 100),
        "median_pct": float(np.median(returns) * 100),
        "mode_pct": mode_val,
        "p5_pct": float(np.percentile(returns, 5) * 100),
        "p95_pct": float(np.percentile(returns, 95) * 100),
        "p_profit": float((returns > 0).mean()),
        "p_ruin": float((returns <= -0.9999).mean()),
        "mean_inr": float(returns.mean() * capital),
        "median_inr": float(np.median(returns) * capital),
        "mode_inr": mode_val / 100.0 * capital,
        "p5_inr": float(np.percentile(returns, 5) * capital),
        "p95_inr": float(np.percentile(returns, 95) * capital),
    }


def oos_daily_pnl(closes: np.ndarray, pos: np.ndarray, cost: float = 0.0012) -> np.ndarray:
    """Daily P&L series restricted to out-of-sample purged-CV fold days."""
    rets = np.diff(np.log(closes))
    pnl = np.zeros(len(rets))
    for i in range(1, len(rets)):
        pnl[i] = pos[i] * rets[i] - abs(pos[i] - pos[i - 1]) * cost
    oos = np.zeros(len(rets), dtype=bool)
    for _train, test in purged_walk_forward(len(closes), 750, 125, 10, 2):
        for i in test:
            if i < len(oos):
                oos[i] = True
    return pnl[oos]


def synthetic_edge_pnl(win_rate: float, avg_win: float, avg_loss: float,
                       n: int = 2000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wins = rng.random(n) < win_rate
    return np.where(wins, avg_win, -avg_loss)


def format_row(label: str, s: Dict[str, float]) -> str:
    return (f"  {label:<34} mean {s['mean_pct']:+6.1f}%  median {s['median_pct']:+6.1f}%  "
            f"mode {s['mode_pct']:+5.1f}%  P5..P95 [{s['p5_pct']:+6.1f}%, {s['p95_pct']:+6.1f}%]  "
            f"P(profit) {s['p_profit']:4.0%}  P(ruin) {s['p_ruin']:5.2%}\n"
            f"  {'':<34} in INR: mean {s['mean_inr']:+7.0f}  median {s['median_inr']:+7.0f}  "
            f"mode {s['mode_inr']:+7.0f}")


def run(capital: float = 5000.0) -> int:
    from .validate_online import STRATEGIES, SYMBOLS, fetch_daily

    print(f"MONTH-ODDS on Rs.{capital:,.0f} — 100,000 simulated months (22 trading days), "
          "bootstrapped from REAL out-of-sample daily returns (purged CV, 12 bps costs)")
    scenarios: List[tuple[str, np.ndarray, float]] = []

    for symbol in ("^NSEI", "BTCUSDT"):
        try:
            closes = fetch_daily(symbol)
        except Exception as exc:
            print(f"  {symbol}: SKIP ({exc.__class__.__name__})")
            continue
        strat = STRATEGIES["ema_20_50"] if symbol == "^NSEI" else STRATEGIES["donchian_20"]
        label = "NIFTY EMA-cross" if symbol == "^NSEI" else "BTC Donchian"
        scenarios.append((f"{label} 1x (full notional)",
                          oos_daily_pnl(closes, strat(closes)), 1.0))
        if symbol == "BTCUSDT":
            scenarios.append(("BTC Donchian 2.5x (ladder start)",
                              oos_daily_pnl(closes, strat(closes)), 2.5))
            scenarios.append(("BTC Donchian 10x (max cap)",
                              oos_daily_pnl(closes, strat(closes)), 10.0))

    scenarios.append(("Skilled intraday 55%/1.2:1.0 1x",
                      synthetic_edge_pnl(0.55, 0.012, 0.010), 1.0))
    scenarios.append(("Skilled intraday at 5x MIS",
                      synthetic_edge_pnl(0.55, 0.012, 0.010), 5.0))
    scenarios.append(("Unskilled 48%/1:1 at 5x",
                      synthetic_edge_pnl(0.48, 0.01, 0.01), 5.0))

    print(f"  {'scenario':<34} {'distribution of total return':^70}")
    for label, series, mult in scenarios:
        m = compute_metrics(list(series * mult), budget=1.0)
        print(f"\n{label}  (OOS sharpe {m.sharpe:+.2f})")
        print(format_row("₹5,000 after 1 month", simulate(series, capital, mult)))

    print("\nRead this way: the MEAN is what compounding averages to, the MEDIAN is the "
          "typical month,\nthe MODE is the single most likely outcome band (2%-wide). "
          "Leverage widens everything —\nmean AND the ruin column. i.i.d. bootstrap; "
          "regimes change; past returns do not promise future ones.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
