"""Pure decision math for the legacy agent's entry path.

Extracted from ``ox/agent.py`` so the bracket / regime-vote / quorum / clamp
arithmetic is directly unit-testable and the agent's hot path stays thin.
Every function here is pure: no I/O, no side effects, no wall-clock
dependence.  Behaviour is byte-identical to the code it replaces (the full
agent suite - decisions, resilience, boot drills - is the regression gate).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import REG


def bracket_from_supporters(
    frame: pd.DataFrame,
    entry_price: float,
    supporters: list[tuple[str, dict, float]],
) -> tuple[float, float, dict]:
    """Build a bracket from the parameters of strategies that fired.

    Mirrors the backtester's ATR/minimum-distance model.  When more than one
    approved strategy supports the entry, its ``sl_atr`` and ``tp_atr``
    values are score-weighted rather than overwritten by a hard-coded
    ensemble default.
    """
    if not supporters:
        raise ValueError("A positive ensemble vote has no approved entry strategy")
    high = frame["h"].to_numpy(dtype=float)
    low = frame["l"].to_numpy(dtype=float)
    close = frame["c"].to_numpy(dtype=float)
    atr_series = np.asarray(REG["atr"](high, low, close), dtype=float)
    current_atr = float(atr_series[-1]) if len(atr_series) else float("nan")
    if not math.isfinite(current_atr) or current_atr <= 0:
        current_atr = max(float(high[-1] - low[-1]), entry_price * 0.005)
    weights = np.asarray([weight for _, _, weight in supporters], dtype=float)
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Approved entry-strategy weights are invalid")
    stop_atr = float(sum(params["sl_atr"] * weight for _, params, weight in supporters) / total_weight)
    target_atr = float(sum(params["tp_atr"] * weight for _, params, weight in supporters) / total_weight)
    atr_distance = max(current_atr, entry_price * 0.005)
    stop_distance = max(atr_distance * stop_atr, entry_price * 0.002)
    target_distance = max(atr_distance * target_atr, entry_price * 0.004)
    supporters_detail = {
        "strategy_ids": [strategy_id for strategy_id, _, _ in supporters],
        "sl_atr": round(stop_atr, 4),
        "tp_atr": round(target_atr, 4),
        "atr": round(current_atr, 4),
        "stop_distance": round(stop_distance, 4),
        "target_distance": round(target_distance, 4),
    }
    return entry_price - stop_distance, entry_price + target_distance, supporters_detail


def adjust_votes_by_regime(
    votes: float,
    supporters: list[tuple[str, dict, float]],
    regime_weights: dict[str, float],
) -> tuple[float, list[tuple[str, dict, float]]]:
    """Regime-conditioned vote adjustment (tick_once behaviour, made pure).

    Each supporter's weight is scaled by its template's regime multiplier and
    the vote is scaled by the mean multiplier across supporters.
    """
    if not supporters:
        return votes, supporters
    adjusted: list[tuple[str, dict, float]] = []
    for sid, params, weight in supporters:
        template = sid.split("_")[0] if "_" in sid else sid
        adjusted.append((sid, params, weight * regime_weights.get(template, 1.0)))
    mean_multiplier = float(np.mean([
        regime_weights.get(s[0].split("_")[0] if "_" in s[0] else s[0], 1.0)
        for s in adjusted
    ]))
    return votes * mean_multiplier, adjusted


def quorum_verdict(
    votes: float,
    total_weight: float,
    min_fraction: float,
    supporters: list[tuple[str, dict, float]] | None,
    min_support: int,
) -> tuple[bool, str, float]:
    """Ensemble quorum + support gates.  Returns (allowed, reason, required)."""
    required = min_fraction * total_weight
    if total_weight > 0 and votes < required:
        return False, "ENSEMBLE_QUORUM", required
    if len(supporters or []) < min_support:
        return False, "SUPPORT_QUORUM", required
    return True, "", required


def clamp_quantity(
    quantity: int,
    ltp: float,
    requested_leverage: float,
    risk_limits: dict,
    current_gross: float,
) -> tuple[int, int, int]:
    """Clamp a sized quantity to the hard risk caps.

    Returns ``(clamped, notional_cap_qty, exposure_headroom_qty)``; a
    request beyond the caps shrinks the position rather than pushing quantity
    past a limit the risk gate re-checks below.
    """
    notional_cap_qty = int(float(risk_limits["max_notional_per_trade"]) / ltp)
    if requested_leverage > 0:
        exposure_headroom_qty = int(
            (float(risk_limits["max_gross_exposure"]) - current_gross) / (ltp * requested_leverage)
        )
    else:
        exposure_headroom_qty = notional_cap_qty
    return min(quantity, notional_cap_qty, exposure_headroom_qty), notional_cap_qty, exposure_headroom_qty