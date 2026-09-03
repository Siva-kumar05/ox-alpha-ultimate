"""Market microstructure analytics: order flow, footprint, absorption,
liquidity sweeps, volume profile, auction theory, DOM/tape, institutional
metrics, smart-money concepts, and derived hybrid signals.

Inputs are either bar arrays (o/h/l/c/v) for estimated analytics or explicit
 aggressor/trade series for exact ones. Estimated flow fields are marked
``estimated=True`` downstream consumers can distinguish evidence tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ── 1. Core order flow ────────────────────────────────────────────────────

def bar_delta(o, h, l, c, v) -> np.ndarray:
    """Aggressor delta per bar, estimated from close position in range."""
    o, h, l, c, v = _f(o), _f(h), _f(l), _f(c), _f(v)
    rng = np.where(h - l == 0, np.nan, h - l)
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_frac = (c - l) / rng
    return np.nan_to_num(2 * buy_frac - 1) * v


def cumulative_volume_delta(o, h, l, c, v) -> np.ndarray:
    return np.cumsum(bar_delta(o, h, l, c, v))


def volume_delta_divergence(h, l, c, v, look=5) -> np.ndarray:
    """+1: price HH but delta falling (weak buying). -1: price LL, delta rising."""
    h, c = _f(h), _f(c)
    d = bar_delta(c, h, l, c, v)
    out = np.zeros_like(c)
    for i in range(look, len(c)):
        price_hh = c[i] > np.max(c[i - look:i])
        price_ll = c[i] < np.min(c[i - look:i])
        delta_falling = d[i] < np.mean(d[i - look:i])
        delta_rising = d[i] > np.mean(d[i - look:i])
        if price_hh and delta_falling:
            out[i] = 1
        elif price_ll and delta_rising:
            out[i] = -1
    return out


@dataclass
class Footprint:
    """Per-bar bid×ask volume grid (diagnostic when intrabar ticks absent)."""
    levels: np.ndarray
    bid_volume: np.ndarray
    ask_volume: np.ndarray

    def imbalance_by_level(self, threshold: float = 3.0) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = self.ask_volume / np.where(self.bid_volume == 0, np.nan, self.bid_volume)
        return np.where(ratio >= threshold, 1, np.where(ratio <= 1 / threshold, -1, 0))


def footprint(h, l, v, bins: int = 10, bar: int = -1) -> Footprint:
    h, l, v = _f(h), _f(l), _f(v)
    i = bar if bar >= 0 else len(h) - 1
    edges = np.linspace(l[i], h[i], bins + 1)
    levels = (edges[:-1] + edges[1:]) / 2
    # Without tick data, split uniform volume with a buy-skew above mid.
    mid = (h[i] + l[i]) / 2
    skew = np.clip((levels - mid) / max(h[i] - l[i], 1e-9) + 0.5, 0.1, 0.9)
    total = v[i] / bins
    return Footprint(levels, total * (1 - skew), total * skew)


def bid_ask_imbalance(bid_volume, ask_volume) -> float:
    bid, ask = float(bid_volume), float(ask_volume)
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def stacked_imbalances(fp: Footprint, threshold: float = 3.0, run: int = 3) -> list[tuple[str, float]]:
    """3+ consecutive imbalanced levels -> support/resistance zone."""
    flags = fp.imbalance_by_level(threshold)
    zones = []
    count, side, start = 0, 0, 0.0
    for level, flag in zip(fp.levels, flags):
        if flag == side and flag != 0:
            count += 1
            if count >= run:
                zones.append(("support" if side > 0 else "resistance", float(level)))
        else:
            side, count, start = int(flag), 1 if flag != 0 else 0, float(level)
    return zones


def unfinished_auction(fp: Footprint, threshold: float = 3.0) -> list[float]:
    """Imbalance at the extreme levels — price often revisits."""
    flags = fp.imbalance_by_level(threshold)
    edges = []
    if flags[0] != 0:
        edges.append(float(fp.levels[0]))
    if flags[-1] != 0:
        edges.append(float(fp.levels[-1]))
    return edges


def effort_vs_result(h, l, c, v, n: int = 20) -> np.ndarray:
    """Volume effort vs price result; mismatch flags absorption/exhaustion."""
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    move = np.abs(c - np.roll(c, 1))
    vol_norm = v / (np.convolve(v, np.ones(n) / n, mode="same") + 1e-9)
    result_norm = move / (np.convolve(move, np.ones(n) / n, mode="same") + 1e-9)
    return vol_norm / (result_norm + 1e-9)


# ── 2. Absorption & liquidity ─────────────────────────────────────────────

@dataclass
class AbsorptionSignal:
    kind: str          # "highs" | "lows" | ""
    strength: float    # 0..1
    bar: int


def absorption(h, l, c, v, n: int = 20, climax_mult: float = 2.0) -> list[AbsorptionSignal]:
    """Heavy volume, stalled price: passive size soaking aggression."""
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    evr = effort_vs_result(h, l, c, v, n)
    avg_v = np.convolve(v, np.ones(n) / n, mode="same") + 1e-9
    signals = []
    for i in range(n, len(c)):
        if v[i] > climax_mult * avg_v[i] and evr[i] > 1.5:
            wick_up = (h[i] - max(c[i], (h[i] + l[i]) / 2)) / max(h[i] - l[i], 1e-9)
            wick_dn = (min(c[i], (h[i] + l[i]) / 2) - l[i]) / max(h[i] - l[i], 1e-9)
            if wick_up > 0.4:
                signals.append(AbsorptionSignal("highs", min(1.0, evr[i] / 3), i))
            elif wick_dn > 0.4:
                signals.append(AbsorptionSignal("lows", min(1.0, evr[i] / 3), i))
    return signals


def high_volume_nodes(h, l, v, bins: int = 30, window: int = 200) -> dict:
    """HVN/LVN structure: volume traded per price over trailing window."""
    h, l, v = _f(h), _f(l), _f(v)
    lo, hi = float(np.nanmin(l[-window:])), float(np.nanmax(h[-window:]))
    edges = np.linspace(lo, hi, bins + 1)
    hist = np.zeros(bins)
    for i in range(max(0, len(h) - window), len(h)):
        if np.isnan(h[i]) or h[i] == l[i]:
            continue
        lo_i = np.searchsorted(edges, l[i]) - 1
        hi_i = np.searchsorted(edges, h[i])
        for b in range(max(0, lo_i), min(bins, hi_i)):
            hist[b] += v[i] / max(1, hi_i - lo_i)
    threshold = np.percentile(hist, 70)
    hvn = [float((edges[b] + edges[b + 1]) / 2) for b in range(bins) if hist[b] >= threshold]
    lvn = [float((edges[b] + edges[b + 1]) / 2) for b in range(bins)
           if hist[b] <= np.percentile(hist, 15)]
    return {"hvn": hvn, "lvn": lvn, "poc": float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2)}


def volume_dry_up(v, n: int = 20, ratio: float = 0.5) -> np.ndarray:
    v = _f(v)
    avg = np.convolve(v, np.ones(n) / n, mode="same") + 1e-9
    return v < ratio * avg


def climactic_volume(v, n: int = 20, mult: float = 2.5) -> np.ndarray:
    v = _f(v)
    avg = np.convolve(v, np.ones(n) / n, mode="same") + 1e-9
    return v > mult * avg


def iceberg_detection(trade_sizes: np.ndarray, refills: np.ndarray) -> np.ndarray:
    """Repeated small refills at a level while size depletes: hidden orders."""
    sizes, refills = _f(trade_sizes), _f(refills)
    return (refills > np.percentile(refills, 80)) & (sizes < np.percentile(sizes, 40))


def spoofing_detector(depth_updates: list[dict]) -> list[int]:
    """Large orders placed then cancelled before price arrives (heuristic)."""
    flagged = []
    for i, upd in enumerate(depth_updates):
        if upd.get("placed_notional", 0) > 5 * upd.get("median_notional", 1) and \
           upd.get("cancelled_before_fill", False):
            flagged.append(i)
    return flagged


# ── 3. Liquidity & sweeps ─────────────────────────────────────────────────

@dataclass
class SweepSignal:
    bar: int
    direction: str   # "buy-side" | "sell-side"
    reclaimed: bool
    strength: float


def liquidity_sweeps(h, l, c, k: int = 5, reclaim_tol: float = 0.3) -> list[SweepSignal]:
    """Wick through prior swing (stop run) followed by reclaim = sweep."""
    h, l, c = _f(h), _f(l), _f(c)
    signals = []
    for i in range(k * 2, len(c)):
        prior_high = np.max(h[i - k:i])
        prior_low = np.min(l[i - k:i])
        rng = max(prior_high - prior_low, 1e-9)
        if h[i] > prior_high and c[i] < prior_high - reclaim_tol * rng * 0.1:
            reclaimed = c[i] > prior_high - 0.3 * rng
            signals.append(SweepSignal(i, "buy-side", reclaimed, (h[i] - prior_high) / rng))
        if l[i] < prior_low and c[i] > prior_low + reclaim_tol * rng * 0.1:
            reclaimed = c[i] < prior_low + 0.3 * rng
            signals.append(SweepSignal(i, "sell-side", reclaimed, (prior_low - l[i]) / rng))
    return signals


def equal_highs_lows(h, l, tol_pct: float = 0.05, k: int = 20) -> dict:
    """EQH/EQL clusters — pools of resting stops."""
    h, l = _f(h), _f(l)
    highs = [i for i in range(k, len(h))
             if np.sum(np.abs(h[i - k:i] - h[i]) / h[i] < tol_pct / 100) >= 2]
    lows = [i for i in range(k, len(l))
            if np.sum(np.abs(l[i - k:i] - l[i]) / l[i] < tol_pct / 100) >= 2]
    return {"equal_highs": highs, "equal_lows": lows}


def fair_value_gaps(o, h, l, c) -> np.ndarray:
    """3-candle imbalance: bullish +1 (gap up unfilled), bearish -1."""
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    out = np.zeros_like(c)
    for i in range(2, len(c)):
        if l[i] > h[i - 2]:
            out[i] = 1
        elif h[i] < l[i - 2]:
            out[i] = -1
    return out


def liquidity_voids(h, l, avg_range: float | None = None, mult: float = 2.5) -> np.ndarray:
    """Bars whose range is a multiple of the local average — fast one-way moves."""
    h, l = _f(h), _f(l)
    rng = h - l
    avg = avg_range if avg_range is not None else float(np.nanmean(rng))
    return rng > mult * avg


def stop_run_reversal(h, l, c, k: int = 5) -> list[SweepSignal]:
    """Sweep → immediate reclaim: the classic reversal entry pattern."""
    return [s for s in liquidity_sweeps(h, l, c, k) if s.reclaimed]


def sweep_vs_breakout(h, l, c, k: int = 5, confirm_bars: int = 3) -> list[dict]:
    """Classify each level breach: sweep (reclaimed) vs true breakout (accepted)."""
    h, l, c = _f(h), _f(l), _f(c)
    calls = []
    for s in liquidity_sweeps(h, l, c, k):
        end = min(len(c), s.bar + confirm_bars)
        if s.direction == "buy-side":
            accepted = all(c[j] > np.max(h[s.bar - k:s.bar]) for j in range(s.bar + 1, end))
            calls.append({"bar": s.bar, "level": float(np.max(h[s.bar - k:s.bar])),
                          "type": "breakout" if accepted else "sweep"})
        else:
            accepted = all(c[j] < np.min(l[s.bar - k:s.bar]) for j in range(s.bar + 1, end))
            calls.append({"bar": s.bar, "level": float(np.min(l[s.bar - k:s.bar])),
                          "type": "breakout" if accepted else "sweep"})
    return calls


# ── 4. Volume profile family ──────────────────────────────────────────────

@dataclass
class Profile:
    poc: float
    vah: float
    val: float
    histogram: np.ndarray
    edges: np.ndarray
    shape: str

    def value_area(self) -> tuple[float, float]:
        return self.vah, self.val


def volume_profile(h, l, v, bins: int = 50, window: int = None, value_pct: float = 0.70) -> Profile:
    h, l, v = _f(h), _f(l), _f(v)
    if window:
        h, l, v = h[-window:], l[-window:], v[-window:]
    lo, hi = float(np.nanmin(l)), float(np.nanmax(h))
    edges = np.linspace(lo, hi, bins + 1)
    hist = np.zeros(bins)
    for i in range(len(h)):
        if np.isnan(h[i]) or h[i] == l[i]:
            continue
        lo_i = np.searchsorted(edges, l[i]) - 1
        hi_i = np.searchsorted(edges, h[i])
        for b in range(max(0, lo_i), min(bins, hi_i)):
            hist[b] += v[i] / max(1, hi_i - lo_i)
    poc_idx = int(np.argmax(hist))
    total = hist.sum()
    va, va_indices = hist[poc_idx], {poc_idx}
    lo_idx, hi_idx = poc_idx, poc_idx
    while va < value_pct * total and (lo_idx > 0 or hi_idx < bins - 1):
        below = hist[lo_idx - 1] if lo_idx > 0 else -1
        above = hist[hi_idx + 1] if hi_idx < bins - 1 else -1
        if above >= below:
            hi_idx += 1
            va += hist[hi_idx]
            va_indices.add(hi_idx)
        else:
            lo_idx -= 1
            va += hist[lo_idx]
            va_indices.add(lo_idx)
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    vah = float(edges[max(va_indices) + 1])
    val = float(edges[min(va_indices)])
    # Shape classification
    peak = poc_idx / max(bins - 1, 1)
    if 0.4 <= peak <= 0.6:
        shape = "normal"
    elif peak > 0.65:
        shape = "p-shape"  # buying tail
    elif peak < 0.35:
        shape = "b-shape"  # selling tail
    else:
        shape = "trend"
    return Profile(poc, vah, val, hist, edges, shape)


def naked_poc(profiles: list[Profile], current_price: float) -> list[float]:
    """Untested POCs from prior sessions — magnet zones."""
    out = []
    for p in profiles:
        if not (p.val <= current_price <= p.vah):
            out.append(p.poc)
    return out


def initial_balance(h, l, ib_bars: int = 60) -> tuple[float, float]:
    return float(np.max(h[:ib_bars])), float(np.min(l[:ib_bars]))


def value_migration(profiles: list[Profile]) -> str:
    """POC direction day over day."""
    if len(profiles) < 2:
        return "unknown"
    deltas = [b.poc - a.poc for a, b in zip(profiles, profiles[1:])]
    if all(d > 0 for d in deltas):
        return "up"
    if all(d < 0 for d in deltas):
        return "down"
    return "rotating"


def poor_highs_lows(h, l, single_print_tol: float = 0.001) -> list[int]:
    """Single-print extremes — magnets for revisits."""
    out = []
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and (h[i] - h[i - 1]) / h[i] < single_print_tol:
            out.append(i)
    return out


# ── 5. Auction market theory ──────────────────────────────────────────────

def day_type(o, h, l, c) -> str:
    """open-drive / open-test / open-rejection / open-auction classification."""
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    if len(c) < 10:
        return "unknown"
    opening = o[0]
    lo, hi = float(np.min(l)), float(np.max(h))
    rng = max(hi - lo, 1e-9)
    pos_open = (opening - lo) / rng
    pos_close = (c[-1] - lo) / rng
    if pos_open < 0.2 and pos_close > 0.6:
        return "open-drive-up"
    if pos_open > 0.8 and pos_close < 0.4:
        return "open-drive-down"
    if pos_open < 0.25 and pos_close < 0.4:
        return "open-rejection-low"
    if pos_open > 0.75 and pos_close > 0.6:
        return "open-rejection-high"
    return "open-auction"


def one_timeframing(h, l, k: int = 3) -> int:
    """Consecutive higher lows (+n) or lower highs (-n): directional auction."""
    h, l = _f(h), _f(l)
    up = dn = 0
    for i in range(len(l) - k, len(l)):
        if l[i] > l[i - 1]:
            up += 1
            dn = 0
        elif h[i] < h[i - 1]:
            dn += 1
            up = 0
    return up if up >= k else (-dn if dn >= k else 0)


def excess_tails(profile: Profile, tail_pct: float = 0.1) -> dict:
    n = len(profile.histogram)
    tail = max(1, int(n * tail_pct))
    return {"upper_tail": float(profile.histogram[-tail:].sum()),
            "lower_tail": float(profile.histogram[:tail].sum())}


# ── 6. DOM / tape ─────────────────────────────────────────────────────────

@dataclass
class TapePrint:
    ts: float
    price: float
    size: float
    aggressor: int  # +1 buy, -1 sell, 0 unknown


def lee_ready_classification(trades: list[TapePrint], quotes: list[tuple[float, float]]) -> list[int]:
    """Tick rule blended with quote midpoint: classic trade signing."""
    signs = []
    prev_price = trades[0].price if trades else 0.0
    for i, t in enumerate(trades):
        bid, ask = quotes[i] if i < len(quotes) else (t.price - 1e-9, t.price + 1e-9)
        mid = (bid + ask) / 2
        if t.price > mid:
            signs.append(1)
        elif t.price < mid:
            signs.append(-1)
        else:
            signs.append(1 if t.price > prev_price else (-1 if t.price < prev_price else 0))
        prev_price = t.price
    return signs


def large_lot_ratio(sizes: np.ndarray, quantile: float = 0.9) -> float:
    sizes = _f(sizes)
    threshold = np.percentile(sizes, quantile * 100)
    return float(np.sum(sizes >= threshold) / max(len(sizes), 1))


def sweep_prints(trades: list[TapePrint], depth_notional: float) -> list[TapePrint]:
    return [t for t in trades if t.size * t.price >= depth_notional]


def held_bids(depth_updates: list[dict], persist_bars: int = 5) -> list[float]:
    """Persistent passive size defending a level (ladder absorption)."""
    levels: dict[float, int] = {}
    held = []
    for upd in depth_updates:
        level, size = upd.get("best_bid"), upd.get("bid_size", 0)
        if level is None:
            continue
        if size >= np.median([u.get("bid_size", 1) for u in depth_updates]):
            levels[level] = levels.get(level, 0) + 1
            if levels[level] >= persist_bars:
                held.append(level)
    return sorted(set(held))


# ── 7. Institutional / microstructure metrics ─────────────────────────────

def order_flow_imbalance(bid_depth_series: np.ndarray, ask_depth_series: np.ndarray) -> np.ndarray:
    """OFI: normalized change in top-of-book liquidity differential."""
    bid, ask = _f(bid_depth_series), _f(ask_depth_series)
    dbid, dask = np.diff(bid, prepend=bid[0]), np.diff(ask, prepend=ask[0])
    denom = np.abs(dbid) + np.abs(dask) + 1e-9
    return (dbid - dask) / denom


def vpin(buy_volume: np.ndarray, sell_volume: np.ndarray, bucket_size: float | None = None) -> float:
    """Volume-synchronized probability of informed trading."""
    buy, sell = _f(buy_volume), _f(sell_volume)
    total = buy + sell
    if bucket_size is None:
        bucket_size = float(np.mean(total)) * 8 if len(total) else 1.0
    buckets, b_acc, s_acc = [], 0.0, 0.0
    for b, s, t in zip(buy, sell, total):
        b_acc, s_acc = b_acc + b, s_acc + s
        if b_acc + s_acc >= bucket_size:
            buckets.append(abs(b_acc - s_acc) / (b_acc + s_acc))
            b_acc = s_acc = 0.0
    return float(np.mean(buckets)) if buckets else 0.0


def kyles_lambda(returns: np.ndarray, signed_volume: np.ndarray) -> float:
    """Price impact per unit signed flow (regression slope)."""
    r, v = _f(returns), _f(signed_volume)
    if len(r) < 2 or np.allclose(v, 0):
        return 0.0
    var = np.var(v)
    return float(np.cov(r, v)[0, 1] / var) if var > 0 else 0.0


def amihud_illiquidity(returns: np.ndarray, volume: np.ndarray) -> float:
    r, v = _f(returns), _f(volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.abs(r) / np.where(v == 0, np.nan, v)
    return float(np.nanmean(daily)) if np.any(~np.isnan(daily)) else 0.0


def roll_spread(prices: np.ndarray) -> float:
    """Effective spread estimator from serial covariance of price changes."""
    p = _f(prices)
    dp = np.diff(p)
    if len(dp) < 2:
        return 0.0
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    return float(2 * np.sqrt(-cov)) if cov < 0 else 0.0


def quote_stuffing(order_count: np.ndarray, trade_count: np.ndarray, mult: float = 3.0) -> np.ndarray:
    """Order-to-trade ratio spikes."""
    oc, tc = _f(order_count), _f(trade_count)
    ratio = oc / np.maximum(tc, 1)
    return ratio > mult * np.nanmedian(ratio)


def order_book_resilience(depth_series: np.ndarray, depletion_events: list[int], horizon: int = 10) -> float:
    """How fast liquidity refills after depletion."""
    d = _f(depth_series)
    refills = []
    baseline = float(np.nanmedian(d))
    for i in depletion_events:
        window = d[i + 1:i + 1 + horizon]
        if len(window):
            refills.append(float(np.nanmax(window)) / max(baseline, 1e-9))
    return float(np.mean(refills)) if refills else 1.0


def adverse_selection(spread_series: np.ndarray, mid_returns: np.ndarray) -> float:
    """Fraction of spread explained by subsequent price move."""
    s, r = _f(spread_series), _f(mid_returns)
    if len(s) < 3 or np.std(s) == 0:
        return 0.0
    return abs(float(np.corrcoef(s, r)[0, 1]))


def square_root_impact(size: float, adv: float, price: float, k: float = 1.0) -> float:
    """Impact ~ k * sigma * sqrt(Q/ADV) — the empirical square-root law."""
    return float(k * price * np.sqrt(max(size, 0.0) / max(adv, 1e-9)))


# ── 8. Smart money concepts ───────────────────────────────────────────────

def order_blocks(o, h, l, c, look: int = 5) -> list[dict]:
    """Last opposing candle before an impulsive move."""
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    blocks = []
    rng = np.roll(h - l, 1)
    avg_rng = np.nanmean(rng) * 1.5
    for i in range(look + 1, len(c)):
        impulsive = (h[i] - l[i]) > avg_rng and abs(c[i] - o[i]) > 0.6 * (h[i] - l[i])
        if not impulsive:
            continue
        prior_bear = c[i - 1] < o[i - 1]
        prior_bull = c[i - 1] > o[i - 1]
        if prior_bear and c[i] > o[i]:
            blocks.append({"bar": i, "zone": (float(l[i - 1]), float(h[i - 1])), "kind": "bullish"})
        elif prior_bull and c[i] < o[i]:
            blocks.append({"bar": i, "zone": (float(l[i - 1]), float(h[i - 1])), "kind": "bearish"})
    return blocks


def breaker_blocks(o, h, l, c) -> list[dict]:
    """Order blocks that failed and flipped polarity."""
    blocks = order_blocks(o, h, l, c)
    out = []
    for b in blocks:
        lo, hi = b["zone"]
        if b["kind"] == "bullish" and float(np.nanmin(l[b["bar"]:])) < lo:
            out.append({**b, "kind": "bearish-breaker"})
        elif b["kind"] == "bearish" and float(np.nanmax(h[b["bar"]:])) > hi:
            out.append({**b, "kind": "bullish-breaker"})
    return out


def bos_choch(h, l, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Break of structure (trend continuation) and change of character (reversal)."""
    h, l = _f(h), _f(l)
    n = len(h)
    bos = np.zeros(n)
    choch = np.zeros(n)
    last_high = last_low = np.nan
    direction = 0
    for i in range(k, n):
        ph = np.max(h[i - k:i])
        pl = np.min(l[i - k:i])
        if h[i] > ph:
            if direction < 0:
                choch[i] = 1
            else:
                bos[i] = 1
            direction = 1
            last_high = h[i]
        if l[i] < pl:
            if direction > 0:
                choch[i] = -1
            else:
                bos[i] = -1
            direction = -1
            last_low = l[i]
    return bos, choch


def premium_discount(c, swing_high: float, swing_low: float) -> float:
    """Position in range: <0.5 discount (buy zone), >0.5 premium."""
    rng = max(swing_high - swing_low, 1e-9)
    return float((_f(c)[-1] - swing_low) / rng)


def optimal_trade_entry(c, swing_high, swing_low) -> tuple[float, float]:
    """OTE band: the 61.8–79% retracement zone."""
    rng = swing_high - swing_low
    if _f(c)[-1] >= _f(c)[-2]:
        return swing_high - 0.705 * rng, swing_high - 0.79 * rng
    return swing_low + 0.705 * rng, swing_low + 0.79 * rng


def judas_swing(o, h, l, c, first_bars: int = 15) -> dict:
    """Opening fake move against the day's true direction."""
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    if len(c) < first_bars * 3:
        return {"detected": False}
    open_drive = c[first_bars] - o[0]
    day_move = c[-1] - o[0]
    fake = np.sign(open_drive) != np.sign(day_move) and abs(open_drive) > 0.2 * abs(day_move)
    return {"detected": bool(fake), "direction": "bearish-fake" if open_drive > 0 else "bullish-fake"}


# ── 9. Derived hybrid signals ─────────────────────────────────────────────

def delta_flip(o, h, l, c, v, smooth: int = 5) -> np.ndarray:
    """Cumulative delta changing sign mid-move."""
    cvd = cumulative_volume_delta(o, h, l, c, v)
    smooth_cvd = np.convolve(cvd, np.ones(smooth) / smooth, mode="same")
    sign_change = np.sign(np.diff(np.sign(smooth_cvd), prepend=0))
    return np.where(sign_change != 0, sign_change, 0)


def volume_climax_reversal(o, h, l, c, v, n: int = 20, mult: float = 2.5) -> np.ndarray:
    """Climax volume + reversal wick: exhaustion entry."""
    o, h, l, c, v = _f(o), _f(h), _f(l), _f(c), _f(v)
    climax = climactic_volume(v, n, mult)
    upper_wick = (h - np.maximum(o, c)) / np.maximum(h - l, 1e-9)
    lower_wick = (np.minimum(o, c) - l) / np.maximum(h - l, 1e-9)
    out = np.zeros_like(c)
    out[climax & (upper_wick > 0.5)] = -1
    out[climax & (lower_wick > 0.5)] = 1
    return out


def sweep_absorption_combo(h, l, c, v) -> list[dict]:
    """Stop-run into a defended level: high-probability reversal stack."""
    sweeps = stop_run_reversal(h, l, c)
    absorptions = absorption(h, l, c, v)
    combo = []
    for s in sweeps:
        for a in absorptions:
            if abs(a.bar - s.bar) <= 2:
                combo.append({"bar": s.bar, "sweep": s.direction,
                              "absorption": a.kind, "strength": s.strength * a.strength})
    return combo


def imbalance_ob_confluence(o, h, l, c) -> list[dict]:
    """FVG overlapping an order block: stacked flow evidence."""
    gaps = fair_value_gaps(o, h, l, c)
    blocks = order_blocks(o, h, l, c)
    out = []
    for b in blocks[-5:]:
        zone_lo, zone_hi = b["zone"]
        for i in range(b["bar"], min(len(gaps), b["bar"] + 5)):
            if gaps[i] == 1 and b["kind"] == "bullish":
                out.append({"bar": i, "type": "bullish", "zone": b["zone"]})
                break
            if gaps[i] == -1 and b["kind"] == "bearish":
                out.append({"bar": i, "type": "bearish", "zone": b["zone"]})
                break
    return out


def session_delta_trend(o, h, l, c, v, smooth: int = 10) -> float:
    cvd = cumulative_volume_delta(o, h, l, c, v)
    if len(cvd) < smooth:
        return 0.0
    return float(np.sign(np.nanmean(np.diff(cvd[-smooth:]))))


def time_weighted_aggression(o, h, l, c, v, window: int = 20) -> np.ndarray:
    """Delta per unit time — aggression acceleration."""
    d = bar_delta(o, h, l, c, v)
    per_bar = d / np.arange(1, len(d) + 1)
    return np.convolve(per_bar, np.ones(window) / window, mode="same")
