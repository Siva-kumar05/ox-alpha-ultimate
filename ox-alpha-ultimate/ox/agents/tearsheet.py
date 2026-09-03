"""
quantstats-lite tear sheet for the PRIME trade ledger.
=====================================================

Computes the metrics that matter from the ``promax_trades`` table (and the
optional equity curve): return profile, risk-adjusted ratios, drawdown, tail
risk, streaks.  Formulas follow the conventions quantstats uses (annualised
Sharpe/Sortino on per-period returns, Calmar on max drawdown, omega on a
threshold of zero).  No plotting dependencies — plain text report.

Honesty note: with few trades every metric here is noise.  The report says
so explicitly below ``min_trades``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

MIN_TRADES_FOR_SIGNIFICANCE = 30
TRADING_DAYS = 252


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) else default


@dataclass
class Metrics:
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float            # average pnl per trade
    total_pnl: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_trades: int     # peak-to-trough length in trades
    var95: float                 # 5th percentile of per-trade returns (fraction of budget)
    cvar95: float
    best: float
    worst: float
    max_win_streak: int
    max_loss_streak: int
    significant: bool

    def as_dict(self) -> Dict[str, float]:
        return {k: (v if not isinstance(v, bool) else v) for k, v in self.__dict__.items()}


def compute_metrics(pnls: List[float], budget: float) -> Metrics:
    """Per-trade P&L list -> full metric set. ``budget`` scales return fractions."""
    n = len(pnls)
    if n == 0:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)

    arr = np.asarray(pnls, dtype=float)
    rets = arr / budget if budget > 0 else arr
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    downside = float(losses.std(ddof=1)) if len(losses) > 1 else 0.0

    # Equity curve over trades, normalised to budget units for drawdown.
    equity = budget + np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max()) if len(dd) else 0.0
    if max_dd > 0:
        trough_idx = int(np.argmax(dd))
        peak_idx = int(np.argmax(equity[:trough_idx + 1])) if trough_idx > 0 else 0
        dd_len = trough_idx - peak_idx
    else:
        dd_len = 0

    var95 = float(np.percentile(rets, 5)) if n >= 5 else 0.0
    tail = rets[rets <= var95] if n >= 5 else np.array([0.0])
    cvar95 = float(tail.mean()) if len(tail) else var95

    # Streaks
    win_streak = loss_streak = cur_w = cur_l = 0
    for pnl in arr:
        if pnl > 0:
            cur_w += 1
            cur_l = 0
        elif pnl < 0:
            cur_l += 1
            cur_w = 0
        win_streak = max(win_streak, cur_w)
        loss_streak = max(loss_streak, cur_l)

    total = float(arr.sum())
    # Annualisation: assume ~1 trade/day per agent as a conservative proxy
    # for per-trade Sharpe -> per-year scale.
    per_period_sharpe = _safe_div(mean, std)
    sharpe = per_period_sharpe * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    sortino = _safe_div(mean, downside) * math.sqrt(TRADING_DAYS) if downside > 0 else 0.0
    annual_return = _safe_div(total, budget)  # un-annualised; see note in report()
    calmar = _safe_div(annual_return, max_dd)

    return Metrics(
        trades=n,
        win_rate=float(len(wins) / n),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        expectancy=mean,
        total_pnl=total,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        calmar=round(calmar, 3),
        max_drawdown=round(max_dd, 4),
        max_drawdown_trades=dd_len,
        var95=round(var95, 5),
        cvar95=round(cvar95, 5),
        best=float(arr.max()),
        worst=float(arr.min()),
        max_win_streak=win_streak,
        max_loss_streak=loss_streak,
        significant=n >= MIN_TRADES_FOR_SIGNIFICANCE,
    )


def report(ledger_rows: List[Dict], budget: float, agent: str = "ALL") -> str:
    """Render a text tear sheet from ``CapitalAllocator.closed_trades`` rows."""
    lines = [
        f"TEAR SHEET — agent: {agent} — generated {datetime.now().isoformat(timespec='seconds')}",
        "=" * 68,
    ]
    if not ledger_rows:
        lines.append("No closed trades yet. Nothing to report (honestly).")
        return "\n".join(lines)

    pnls = [float(r["pnl"]) for r in ledger_rows]
    m = compute_metrics(pnls, budget)
    lines += [
        f"trades            {m.trades}",
        f"total pnl         {m.total_pnl:+.2f}  (budget {budget:.0f} -> {budget + m.total_pnl:.2f})",
        f"win rate          {m.win_rate:.1%}",
        f"profit factor     {m.profit_factor:.2f}" + ("" if math.isfinite(m.profit_factor) else " (no losses yet)"),
        f"expectancy/trade {m.expectancy:+.2f}",
        f"best / worst      {m.best:+.2f} / {m.worst:+.2f}",
        f"streaks  W/L      {m.max_win_streak} / {m.max_loss_streak}",
        "-" * 68,
        f"sharpe*           {m.sharpe:+.2f}",
        f"sortino*          {m.sortino:+.2f}",
        f"calmar            {m.calmar:+.2f}",
        f"max drawdown      {m.max_drawdown:.1%} over {m.max_drawdown_trades} trades",
        f"VaR95 / CVaR95    {m.var95:+.2%} / {m.cvar95:+.2%}  (per-trade, of budget)",
        "-" * 68,
    ]
    if not m.significant:
        lines.append(
            f"NOTE: {m.trades} trades < {MIN_TRADES_FOR_SIGNIFICANCE} — every number above is"
            " statistically meaningless. Keep paper-trading."
        )
    else:
        lines.append(
            "* Sharpe/Sortino annualised assuming ~1 trade/day. Verify against your"
            " actual trade cadence before believing them."
        )
    return "\n".join(lines)


def agent_report(allocator, agent_id: Optional[str] = None) -> str:
    """Convenience: build the report straight from the capital allocator."""
    rows = allocator.closed_trades(agent_id, limit=1000)
    budget = allocator.budget(agent_id) if agent_id else allocator.total_capital
    return report(rows, float(budget or allocator.total_capital), agent_id or "ALL")
