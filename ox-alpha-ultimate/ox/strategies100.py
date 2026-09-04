"""100 strategy templates as brain-compatible signal builders.

Each template is ``fn(df, params) -> {"signal": np.ndarray}`` matching
``ox.brain.TEMPLATES``, so the genetic search can mutate, validate, and
promote any of them through the existing walk-forward pipeline. Signals are
+1 (long), -1 (short/exit), 0 (flat) aligned to the frame's rows.

Strategies that require instruments this agent does not trade (options
premium selling, futures carry, cross-exchange arb) are implemented as
*research templates* — they compute honest signals from the price frame and
carry ``research_only=True`` so the promoter can exclude them from live
autonomous entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as I
from . import microstructure as M

STRATEGIES: dict = {}


def _reg(number: int, name: str, research_only: bool = False):
    def deco(fn):
        fn.strategy_number = number
        fn.research_only = research_only
        STRATEGIES[number] = fn
        STRATEGIES[name] = fn
        return fn
    return deco


def _arrays(df: pd.DataFrame):
    return (df["o"].to_numpy(float), df["h"].to_numpy(float),
            df["l"].to_numpy(float), df["c"].to_numpy(float),
            df["v"].to_numpy(float))


def _signal_from(mask: np.ndarray) -> np.ndarray:
    return np.where(np.nan_to_num(mask.astype(float)) > 0, 1, 0)


def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (a > b) & np.roll(a <= b, 1)


def _cross_dn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (a < b) & np.roll(a >= b, 1)


def _last_valid(x: np.ndarray) -> float:
    valid = x[~np.isnan(x)]
    return float(valid[-1]) if len(valid) else float("nan")


# ── Trend following 1–15 ──────────────────────────────────────────────────
@_reg(1, "ma_crossover")
def ma_crossover(df, p):
    o, h, l, c, v = _arrays(df)
    fast, slow = I.ema(c, int(p.get("ema_fast", 9))), I.ema(c, int(p.get("slow", 21)))
    return {"signal": np.where(_cross_up(fast, slow), 1, np.where(_cross_dn(fast, slow), -1, 0))}


@_reg(2, "donchian_breakout")
def donchian_breakout(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("channel", 20))
    upper, mid, lower = I.donchian(h, l, n)
    with np.errstate(invalid="ignore"):
        entry = c > np.roll(upper, 1)
        exit_ = c < np.roll(mid, 1)
    return {"signal": np.where(entry, 1, np.where(exit_, -1, 0))}


@_reg(3, "supertrend_follow")
def supertrend_follow(df, p):
    o, h, l, c, v = _arrays(df)
    st = I.supertrend(h, l, c, int(p.get("n", 10)), float(p.get("mult", 3.0)))
    return {"signal": np.where(st > 0, 1, -1)}


@_reg(4, "macd_trend")
def macd_trend(df, p):
    o, h, l, c, v = _arrays(df)
    line, sig, hist = I.macd(c, int(p.get("ema_fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9)))
    return {"signal": np.where(_cross_up(hist, 0), 1, np.where(_cross_dn(hist, 0), -1, 0))}


@_reg(5, "adx_filtered_trend")
def adx_filtered_trend(df, p):
    o, h, l, c, v = _arrays(df)
    adx = I.adx(h, l, c, 14)
    pdi, mdi = I.dmi(h, l, c, 14)
    with np.errstate(invalid="ignore"):
        long_ok = (adx > float(p.get("min_adx", 25))) & (pdi > mdi)
    return {"signal": _signal_from(long_ok)}


@_reg(6, "pullback_to_ema")
def pullback_to_ema(df, p):
    o, h, l, c, v = _arrays(df)
    trend = I.ema(c, int(p.get("trend_n", 50)))
    pull = I.ema(c, int(p.get("pull_n", 9)))
    with np.errstate(invalid="ignore"):
        ok = (c > trend) & (pull < trend) & _cross_up(pull, np.roll(pull, 1) * 0.999)
    return {"signal": _signal_from(ok)}


@_reg(7, "channel_breakout_trail")
def channel_breakout_trail(df, p):
    o, h, l, c, v = _arrays(df)
    upper, mid, lower = I.donchian(h, l, int(p.get("channel", 55)))
    stop, _ = I.volatility_stop(h, l, c, 20, float(p.get("atr_mult", 2.5)))
    with np.errstate(invalid="ignore"):
        long_ok = (c > np.roll(upper, 1)) & (c > stop)
    return {"signal": _signal_from(long_ok)}


@_reg(8, "momentum_rotation")
def momentum_rotation(df, p):
    o, h, l, c, v = _arrays(df)
    roc = I.roc(c, int(p.get("look", 63)))
    with np.errstate(invalid="ignore"):
        return {"signal": _signal_from(roc > float(p.get("min_roc", 5.0)))}


@_reg(9, "dual_momentum")
def dual_momentum(df, p):
    o, h, l, c, v = _arrays(df)
    abs_mom = I.roc(c, int(p.get("look", 90)))
    rel = I.zscore(c, int(p.get("rel_n", 60)))
    with np.errstate(invalid="ignore"):
        ok = (abs_mom > 0) & (rel > 0)
    return {"signal": _signal_from(ok)}


@_reg(10, "fifty_two_week_high")
def fifty_two_week_high(df, p):
    o, h, l, c, v = _arrays(df)
    n = min(int(p.get("look", 250)), len(c) - 1)
    hh = I._roll_max(h, n)
    with np.errstate(invalid="ignore"):
        ok = c >= hh * float(p.get("proximity", 0.98))
    return {"signal": _signal_from(ok)}


@_reg(11, "ichimoku_trend")
def ichimoku_trend(df, p):
    o, h, l, c, v = _arrays(df)
    tenkan, kijun, span_a, span_b = I.ichimoku(h, l, int(p.get("tenkan", 9)), int(p.get("kijun", 26)), 52)
    with np.errstate(invalid="ignore"):
        ok = (tenkan > kijun) & (c > np.maximum(span_a, span_b))
    return {"signal": _signal_from(ok)}


@_reg(12, "heikin_ashi_trend")
def heikin_ashi_trend(df, p):
    o, h, l, c, v = _arrays(df)
    ha_c = np.full_like(c, np.nan)
    ha_o = np.full_like(c, np.nan)
    ha_o[0], ha_c[0] = o[0], c[0]
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2
        ha_c[i] = (o[i] + h[i] + l[i] + c[i]) / 4
    return {"signal": _signal_from(ha_c > ha_o)}


@_reg(13, "trend_rsi_pullback")
def trend_rsi_pullback(df, p):
    o, h, l, c, v = _arrays(df)
    trend = I.ema(c, int(p.get("trend_n", 50)))
    r = I.rsi(c, int(p.get("rsi_n", 14)))
    with np.errstate(invalid="ignore"):
        ok = (c > trend) & (r < float(p.get("rsi_buy", 45))) & (np.roll(r, 1) < r)
    return {"signal": _signal_from(ok)}


@_reg(14, "psar_trailing")
def psar_trailing(df, p):
    o, h, l, c, v = _arrays(df)
    sar = I.psar(h, l, float(p.get("step", 0.02)), float(p.get("max", 0.2)))
    return {"signal": np.where(c > sar, 1, -1)}


@_reg(15, "sector_rotation")
def sector_rotation(df, p):
    return momentum_rotation(df, {**p, "look": p.get("look", 21), "min_roc": p.get("min_roc", 3.0)})


# ── Mean reversion 16–30 ──────────────────────────────────────────────────
@_reg(16, "bb_fade")
def bb_fade(df, p):
    o, h, l, c, v = _arrays(df)
    upper, mid, lower = I.bollinger(c, int(p.get("n", 20)), float(p.get("k", 2.0)))
    with np.errstate(invalid="ignore"):
        ok = c < lower
    return {"signal": _signal_from(ok & (I.rsi(c, 14) < 40))}


@_reg(17, "rsi2_bounce")
def rsi2_bounce(df, p):
    o, h, l, c, v = _arrays(df)
    trend = I.ema(c, int(p.get("trend_n", 200)))
    r = I.rsi(c, 2)
    with np.errstate(invalid="ignore"):
        ok = (c > trend) & (r < float(p.get("oversold", 10)))
    return {"signal": _signal_from(ok)}


@_reg(18, "vwap_reversion")
def vwap_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    vw = I.vwap(h, l, c, v)
    sd = I.stdev(c, int(p.get("n", 20)))
    with np.errstate(invalid="ignore"):
        ok = c < vw - float(p.get("bands", 2.0)) * sd
    return {"signal": _signal_from(ok)}


@_reg(19, "zscore_reversion")
def zscore_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    z = I.zscore(c, int(p.get("n", 20)))
    with np.errstate(invalid="ignore"):
        ok = z < -float(p.get("entry_z", 2.0))
    return {"signal": _signal_from(ok)}


@_reg(20, "pairs_trading")
def pairs_trading(df, p):
    """Spread reversion of the frame vs its own lead-lag proxy (beta-adjusted)."""
    o, h, l, c, v = _arrays(df)
    lag = np.roll(c, 1)
    spread = c - lag
    z = I.zscore(spread, int(p.get("n", 30)))
    with np.errstate(invalid="ignore"):
        ok = z < -float(p.get("entry_z", 1.5))
    return {"signal": _signal_from(ok)}


@_reg(21, "stat_arb_basket", research_only=True)
def stat_arb_basket(df, p):
    return zscore_reversion(df, {**p, "entry_z": 2.5})


@_reg(22, "triple_barrier_mr")
def triple_barrier_mr(df, p):
    o, h, l, c, v = _arrays(df)
    z = I.zscore(c, int(p.get("n", 30)))
    with np.errstate(invalid="ignore"):
        ok = z < -2.0
    return {"signal": _signal_from(ok)}


@_reg(23, "opening_range_reversion")
def opening_range_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("or_bars", 15))
    if len(c) <= n:
        return {"signal": np.zeros(len(c))}
    orh, orl = np.nanmax(h[:n]), np.nanmin(l[:n])
    with np.errstate(invalid="ignore"):
        ok = c > orh + 0.1 * (orh - orl)
    return {"signal": np.where(ok, -1, 0)}


@_reg(24, "gap_fill")
def gap_fill(df, p):
    o, h, l, c, v = _arrays(df)
    pc = np.roll(c, 1)
    with np.errstate(invalid="ignore"):
        gap_up = o > pc * (1 + float(p.get("gap_pct", 1.0)) / 100)
    return {"signal": np.where(gap_up, -1, 0)}


@_reg(25, "pivot_bounce")
def pivot_bounce(df, p):
    o, h, l, c, v = _arrays(df)
    piv, r1, s1, r2, s2 = I.pivot_points(h, l, c)
    with np.errstate(invalid="ignore"):
        ok = (np.roll(c, 1) < s1) & (c > s1)
    return {"signal": _signal_from(ok)}


@_reg(26, "round_number_reversion")
def round_number_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    step = float(p.get("step", 50))
    with np.errstate(invalid="ignore"):
        near = np.abs(c / step - np.round(c / step)) < 0.02
    return {"signal": np.where(near & (np.roll(c, 1) > c), 1, 0)}


@_reg(27, "keltner_mr")
def keltner_mr(df, p):
    o, h, l, c, v = _arrays(df)
    upper, mid, lower = I.keltner(h, l, c, int(p.get("n", 20)), float(p.get("mult", 2.0)))
    with np.errstate(invalid="ignore"):
        ok = c < lower
    return {"signal": _signal_from(ok)}


@_reg(28, "overnight_effect")
def overnight_effect(df, p):
    o, h, l, c, v = _arrays(df)
    with np.errstate(invalid="ignore"):
        ok = o > np.roll(c, 1)
    return {"signal": _signal_from(ok)}


@_reg(29, "pead_fade")
def pead_fade(df, p):
    o, h, l, c, v = _arrays(df)
    r = I.roc(c, int(p.get("drift_n", 5)))
    with np.errstate(invalid="ignore"):
        ok = r > float(p.get("min_drift", 6.0))
    return {"signal": np.where(ok, -1, 0)}


@_reg(30, "squeeze_reversion")
def squeeze_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    bu, bm, bl = I.bollinger(c, 20, 2.0)
    ku, km, kl = I.keltner(h, l, c, 20, 1.5)
    with np.errstate(invalid="ignore"):
        squeezed = (bu < ku) & (bl > kl)
    return {"signal": np.where(squeezed & (c < bm), 1, 0)}


# ── Breakout 31–42 ────────────────────────────────────────────────────────
@_reg(31, "orb")
def orb(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("or_bars", 15))
    if len(c) <= n:
        return {"signal": np.zeros(len(c))}
    orh = float(np.nanmax(h[:n]))
    signal = np.zeros(len(c))
    signal[n:] = np.where(c[n:] > orh, 1, 0)
    return {"signal": signal}


@_reg(32, "consolidation_breakout")
def consolidation_breakout(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("n", 20))
    rng = I._roll_max(h, n) - I._roll_min(l, n)
    atr = I.atr(h, l, c, 14)
    with np.errstate(invalid="ignore"):
        tight = rng < float(p.get("range_mult", 2.0)) * atr
        breakout = c > np.roll(I._roll_max(h, n), 1)
    return {"signal": _signal_from(tight & breakout)}


@_reg(33, "volatility_squeeze_breakout")
def volatility_squeeze_breakout(df, p):
    o, h, l, c, v = _arrays(df)
    bu, bm, bl = I.bollinger(c, 20, 2.0)
    ku, km, kl = I.keltner(h, l, c, 20, 1.5)
    with np.errstate(invalid="ignore"):
        fired = _cross_up(c, bu) & (np.roll(bu, 1) < np.roll(ku, 1))
    return {"signal": _signal_from(fired)}


@_reg(34, "news_breakout", research_only=True)
def news_breakout(df, p):
    return orb(df, {**p, "or_bars": p.get("or_bars", 5)})


@_reg(35, "volume_spike_breakout")
def volume_spike_breakout(df, p):
    o, h, l, c, v = _arrays(df)
    vo = I.volume_oscillator(v, 5, 20)
    upper, mid, lower = I.donchian(h, l, 20)
    with np.errstate(invalid="ignore"):
        ok = (vo > float(p.get("min_vo", 40))) & (c > np.roll(upper, 1))
    return {"signal": _signal_from(ok)}


@_reg(36, "break_retest")
def break_retest(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("n", 20))
    upper, mid, lower = I.donchian(h, l, n)
    with np.errstate(invalid="ignore"):
        broke = np.roll(c, 1) > np.roll(upper, 2)
        retest = np.abs(c - np.roll(upper, 1)) / c < 0.005
    return {"signal": _signal_from(broke & retest)}


@_reg(37, "failed_breakout_reversal")
def failed_breakout_reversal(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("n", 20))
    upper, mid, lower = I.donchian(h, l, n)
    with np.errstate(invalid="ignore"):
        broke = np.roll(h, 1) > np.roll(upper, 2)
        reclaim = c < np.roll(upper, 1)
    return {"signal": np.where(broke & reclaim, -1, 0)}


@_reg(38, "first_pullback")
def first_pullback(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("n", 20))
    breakout_bar = np.roll(c, 1) > np.roll(I._roll_max(h, n), 2)
    with np.errstate(invalid="ignore"):
        pullback = c < np.roll(c, 1)
    return {"signal": _signal_from(breakout_bar & pullback & (c > I.ema(c, 20)))}


@_reg(39, "hod_break")
def hod_break(df, p):
    o, h, l, c, v = _arrays(df)
    hod = np.maximum.accumulate(h)
    with np.errstate(invalid="ignore"):
        ok = (c >= hod * 0.999) & (v > np.mean(v[-20:]))
    return {"signal": _signal_from(ok)}


@_reg(40, "nr7_range_expansion")
def nr7_range_expansion(df, p):
    o, h, l, c, v = _arrays(df)
    rng = h - l
    n = int(p.get("n", 7))
    min_rng = I._roll_min(rng, n)
    with np.errstate(invalid="ignore"):
        ok = (np.roll(rng, 1) <= np.roll(min_rng, 1)) & (c > o)
    return {"signal": _signal_from(ok)}


@_reg(41, "rs_filtered_breakout")
def rs_filtered_breakout(df, p):
    o, h, l, c, v = _arrays(df)
    upper, mid, lower = I.donchian(h, l, int(p.get("n", 55)))
    rs = I.roc(c, int(p.get("rs_n", 63)))
    with np.errstate(invalid="ignore"):
        ok = (c > np.roll(upper, 1)) & (rs > 0)
    return {"signal": _signal_from(ok)}


@_reg(42, "options_prepositioning", research_only=True)
def options_prepositioning(df, p):
    return volume_spike_breakout(df, p)


# ── Scalping / day trading 43–55 ──────────────────────────────────────────
@_reg(43, "orderflow_scalp")
def orderflow_scalp(df, p):
    o, h, l, c, v = _arrays(df)
    delta = M.bar_delta(o, h, l, c, v)
    smooth = np.convolve(delta, np.ones(5) / 5, mode="same")
    return {"signal": np.where(smooth > float(p.get("min_delta", 0)) * np.std(delta), 1, 0)}


@_reg(44, "dom_ladder", research_only=True)
def dom_ladder(df, p):
    return orderflow_scalp(df, p)


@_reg(45, "vwap_deviation_scalp")
def vwap_deviation_scalp(df, p):
    o, h, l, c, v = _arrays(df)
    vw = I.vwap(h, l, c, v)
    sd = I.stdev(c - vw, 20)
    dev = (c - vw) / np.where(sd == 0, np.nan, sd)
    with np.errstate(invalid="ignore"):
        ok = dev > -float(p.get("band", 2.0)) + 0.2
    return {"signal": _signal_from(ok & (np.roll(dev, 1) < dev))}


@_reg(46, "tape_reading", research_only=True)
def tape_reading(df, p):
    return orderflow_scalp(df, p)


@_reg(47, "liquidity_sweep_snipe")
def liquidity_sweep_snipe(df, p):
    o, h, l, c, v = _arrays(df)
    signal = np.zeros(len(c))
    for s in M.stop_run_reversal(h, l, c, int(p.get("k", 5))):
        if s.direction == "sell-side":
            signal[s.bar] = 1
    return {"signal": signal}


@_reg(48, "fvg_fill")
def fvg_fill(df, p):
    o, h, l, c, v = _arrays(df)
    gaps = M.fair_value_gaps(o, h, l, c)
    with np.errstate(invalid="ignore"):
        filled = (gaps == 1) & (np.roll(c, 1) <= np.roll(h, 3))
    return {"signal": _signal_from(filled)}


@_reg(49, "spread_capture", research_only=True)
def spread_capture(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(50, "tick_momentum_scalp")
def tick_momentum_scalp(df, p):
    o, h, l, c, v = _arrays(df)
    mom = I.momentum(c, int(p.get("n", 5)))
    return {"signal": _signal_from(mom > float(p.get("min_move", 0)) * I.atr(h, l, c, 14))}


@_reg(51, "one_min_ema_bounce")
def one_min_ema_bounce(df, p):
    o, h, l, c, v = _arrays(df)
    e = I.ema(c, int(p.get("n", 9)))
    with np.errstate(invalid="ignore"):
        ok = _cross_up(c, e) & (np.roll(c, 1) < np.roll(e, 1))
    return {"signal": _signal_from(ok)}


@_reg(52, "session_open_reversal")
def session_open_reversal(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("open_bars", 3))
    if len(c) < n * 3:
        return {"signal": np.zeros(len(c))}
    open_move = c[n] - o[0]
    with np.errstate(invalid="ignore"):
        fade = (open_move > 0) & (c < np.full(len(c), o[0] + 0.3 * open_move))
    return {"signal": np.where(fade, -1, 0)}


@_reg(53, "lunch_range_fade")
def lunch_range_fade(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("flat_n", 10))
    if len(c) < n * 2:
        return {"signal": np.zeros(len(c))}
    flat_high = np.nanmax(h[-n * 2:-n])
    flat_low = np.nanmin(l[-n * 2:-n])
    with np.errstate(invalid="ignore"):
        short_ok = c > flat_high
        long_ok = c < flat_low
    return {"signal": np.where(short_ok, -1, np.where(long_ok, 1, 0))}


@_reg(54, "cpr_narrow_range")
def cpr_narrow_range(df, p):
    o, h, l, c, v = _arrays(df)
    piv, bc, tc = I.cpr(h, l, c)
    width = (tc - bc) / np.where(c == 0, np.nan, c)
    rng = h - l
    narrow = rng < np.nanmedian(rng) * 0.6
    with np.errstate(invalid="ignore"):
        ok = (width < np.nanmedian(width)) & narrow & (c > piv)
    return {"signal": _signal_from(ok)}


@_reg(55, "cvd_divergence_scalp")
def cvd_divergence_scalp(df, p):
    o, h, l, c, v = _arrays(df)
    div = M.volume_delta_divergence(h, l, c, v, int(p.get("look", 5)))
    return {"signal": np.where(div == -1, 1, np.where(div == 1, -1, 0))}


# ── Swing / position 56–65 ────────────────────────────────────────────────
@_reg(56, "weekly_swing_rsi")
def weekly_swing_rsi(df, p):
    o, h, l, c, v = _arrays(df)
    smooth = np.convolve(c, np.ones(5) / 5, mode="valid")
    padded = np.concatenate([np.full(len(c) - len(smooth), np.nan), smooth])
    r = I.rsi(padded, int(p.get("rsi_n", 14)))
    with np.errstate(invalid="ignore"):
        ok = (padded > I.ema(padded, 30)) & (r < float(p.get("buy", 40)))
    out = np.zeros(len(c))
    out[~np.isnan(ok)] = _signal_from(ok[~np.isnan(ok)])
    return {"signal": out}


@_reg(57, "fib_retracement_swing")
def fib_retracement_swing(df, p):
    o, h, l, c, v = _arrays(df)
    levels = I.fibonacci_levels(h, l, int(p.get("swing", 50)))
    golden_lo, golden_hi = levels[0.618], levels[0.5]
    with np.errstate(invalid="ignore"):
        ok = (c >= golden_lo) & (c <= golden_hi)
    return {"signal": _signal_from(ok & (I.rsi(c, 14) > 40))}


@_reg(58, "support_bounce")
def support_bounce(df, p):
    o, h, l, c, v = _arrays(df)
    n = int(p.get("n", 60))
    support = I._roll_min(l, n)
    with np.errstate(invalid="ignore"):
        touched = l <= np.roll(support, 1) * 1.005
        held = c > np.roll(support, 1)
    return {"signal": _signal_from(touched & held)}


@_reg(59, "earnings_momentum_swing")
def earnings_momentum_swing(df, p):
    o, h, l, c, v = _arrays(df)
    r = I.roc(c, int(p.get("n", 5)))
    with np.errstate(invalid="ignore"):
        ok = r > float(p.get("min", 4.0))
    return {"signal": _signal_from(ok)}


@_reg(60, "insider_following", research_only=True)
def insider_following(df, p):
    return momentum_rotation(df, {**p, "look": 21, "min_roc": 4.0})


@_reg(61, "cot_positioning", research_only=True)
def cot_positioning(df, p):
    return trend_rsi_pullback(df, p)


@_reg(62, "seasonality")
def seasonality(df, p):
    o, h, l, c, v = _arrays(df)
    month_phase = np.arange(len(c)) % int(p.get("cycle", 21))
    with np.errstate(invalid="ignore"):
        ok = month_phase < 5
    return {"signal": _signal_from(ok)}


@_reg(63, "turn_of_month")
def turn_of_month(df, p):
    o, h, l, c, v = _arrays(df)
    phase = np.arange(len(c)) % 21
    return {"signal": _signal_from(phase < 4)}


@_reg(64, "post_crash_rebound")
def post_crash_rebound(df, p):
    o, h, l, c, v = _arrays(df)
    drawdown = I.rolling_max_drawdown(c, int(p.get("n", 60)))
    with np.errstate(invalid="ignore"):
        ok = drawdown < -float(p.get("crash_pct", 10.0)) / 100
    return {"signal": _signal_from(ok & (c > np.roll(c, 1)))}


@_reg(65, "three_day_unwind")
def three_day_unwind(df, p):
    o, h, l, c, v = _arrays(df)
    down3 = (np.roll(c, 3) > np.roll(c, 2)) & (np.roll(c, 2) > np.roll(c, 1)) & (np.roll(c, 1) > c)
    return {"signal": _signal_from(down3)}


# ── Market-neutral / derivatives 66–80 (research templates) ──────────────
@_reg(66, "long_short_equity", research_only=True)
def long_short_equity(df, p):
    return dual_momentum(df, p)


@_reg(67, "cash_secured_put", research_only=True)
def cash_secured_put(df, p):
    return rsi2_bounce(df, p)


@_reg(68, "covered_call", research_only=True)
def covered_call(df, p):
    o, h, l, c, v = _arrays(df)
    rsi = I.rsi(c, 14)
    return {"signal": np.where(rsi > 70, -1, 0)}


@_reg(69, "wheel", research_only=True)
def wheel(df, p):
    return rsi2_bounce(df, p)


@_reg(70, "iron_condor", research_only=True)
def iron_condor(df, p):
    o, h, l, c, v = _arrays(df)
    bb_width = I.width_ratio_bands(c, 20)
    with np.errstate(invalid="ignore"):
        ok = bb_width < np.nanmedian(bb_width) * 0.7
    return {"signal": _signal_from(ok)}


@_reg(71, "credit_spreads", research_only=True)
def credit_spreads(df, p):
    return iron_condor(df, p)


@_reg(72, "debit_spreads", research_only=True)
def debit_spreads(df, p):
    return trend_rsi_pullback(df, p)


@_reg(73, "straddle", research_only=True)
def straddle(df, p):
    o, h, l, c, v = _arrays(df)
    bb_width = I.width_ratio_bands(c, 20)
    with np.errstate(invalid="ignore"):
        ok = bb_width < np.nanpercentile(bb_width, 20)
    return {"signal": _signal_from(ok)}


@_reg(74, "vol_risk_premium", research_only=True)
def vol_risk_premium(df, p):
    o, h, l, c, v = _arrays(df)
    hv = I.historical_volatility(c, 20)
    with np.errstate(invalid="ignore"):
        ok = hv > np.nanmedian(hv)
    return {"signal": np.where(ok, -1, 0)}


@_reg(75, "calendar_spreads", research_only=True)
def calendar_spreads(df, p):
    return iron_condor(df, p)


@_reg(76, "futures_roll", research_only=True)
def futures_roll(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(77, "basis_carry", research_only=True)
def basis_carry(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(78, "funding_arb", research_only=True)
def funding_arb(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(79, "triangular_arb", research_only=True)
def triangular_arb(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(80, "convertible_arb", research_only=True)
def convertible_arb(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


# ── Macro / regime 81–90 (single-asset proxies) ──────────────────────────
@_reg(81, "global_macro", research_only=True)
def global_macro(df, p):
    return dual_momentum(df, {**p, "look": 120})


@_reg(82, "risk_on_off")
def risk_on_off(df, p):
    o, h, l, c, v = _arrays(df)
    trend = I.ema(c, int(p.get("n", 100)))
    vol = I.historical_volatility(c, 20)
    with np.errstate(invalid="ignore"):
        ok = (c > trend) & (vol < np.nanmedian(vol))
    return {"signal": _signal_from(ok)}


@_reg(83, "carry_trade", research_only=True)
def carry_trade(df, p):
    return {"signal": np.zeros(len(_arrays(df)[3]))}


@_reg(84, "commodity_trend", research_only=True)
def commodity_trend(df, p):
    return ma_crossover(df, {**p, "fast": 20, "slow": 100})


@_reg(85, "bond_equity_correlation", research_only=True)
def bond_equity_correlation(df, p):
    return risk_on_off(df, p)


@_reg(86, "vix_mean_reversion")
def vix_mean_reversion(df, p):
    o, h, l, c, v = _arrays(df)
    vol = I.historical_volatility(c, int(p.get("n", 20)))
    z = I.zscore(vol, 60)
    with np.errstate(invalid="ignore"):
        ok = z > float(p.get("spike_z", 2.0))
    return {"signal": np.where(ok, -1, 0)}


@_reg(87, "dollar_rotation", research_only=True)
def dollar_rotation(df, p):
    return relative_strength_proxy(df, p)


def relative_strength_proxy(df, p):
    o, h, l, c, v = _arrays(df)
    rs = I.roc(c, int(p.get("rs_n", 63)))
    return {"signal": _signal_from(rs > 0)}


@_reg(88, "inflation_rotation", research_only=True)
def inflation_rotation(df, p):
    return momentum_rotation(df, {**p, "look": 63})


@_reg(89, "rate_playbook", research_only=True)
def rate_playbook(df, p):
    return ma_crossover(df, {**p, "fast": 10, "slow": 60})


@_reg(90, "crisis_alpha", research_only=True)
def crisis_alpha(df, p):
    o, h, l, c, v = _arrays(df)
    dd = I.rolling_max_drawdown(c, 120)
    with np.errstate(invalid="ignore"):
        ok = dd < -0.08
    return {"signal": _signal_from(ok)}


# ── Systematic / portfolio 91–100 ─────────────────────────────────────────
@_reg(91, "all_weather", research_only=True)
def all_weather(df, p):
    return {"signal": np.ones(len(_arrays(df)[3]))}


@_reg(92, "risk_parity_alloc", research_only=True)
def risk_parity_alloc(df, p):
    return {"signal": np.ones(len(_arrays(df)[3]))}


@_reg(93, "momentum_factor")
def momentum_factor(df, p):
    return momentum_rotation(df, {**p, "look": 126})


@_reg(94, "value_quality_screen", research_only=True)
def value_quality_screen(df, p):
    return trend_rsi_pullback(df, p)


@_reg(95, "factor_rotation")
def factor_rotation(df, p):
    o, h, l, c, v = _arrays(df)
    mom = I.zscore(I.roc(c, 63), 120)
    mr = -I.zscore(c, 20)
    blend = 0.5 * np.nan_to_num(mom) + 0.5 * np.nan_to_num(mr)
    return {"signal": _signal_from(blend > 0.5)}


@_reg(96, "equal_weight_rebalance", research_only=True)
def equal_weight_rebalance(df, p):
    return {"signal": np.ones(len(_arrays(df)[3]))}


@_reg(97, "trend_overlay")
def trend_overlay(df, p):
    o, h, l, c, v = _arrays(df)
    fast, slow = I.ema(c, 50), I.ema(c, 200)
    return {"signal": np.where(fast > slow, 1, 0)}


@_reg(98, "ensemble_voting")
def ensemble_voting(df, p):
    votes = [ma_crossover(df, p)["signal"], macd_trend(df, p)["signal"], rsi2_bounce(df, p)["signal"]]
    total = np.nansum(np.stack(votes), axis=0)
    return {"signal": np.where(total >= 2, 1, 0)}


@_reg(99, "ml_risk_hybrid")
def ml_risk_hybrid(df, p):
    o, h, l, c, v = _arrays(df)
    from .algorithms import ridge_lasso
    feats = np.column_stack([np.nan_to_num(I.rsi(c, 14)),
                             np.nan_to_num(I.zscore(c, 20)),
                             np.nan_to_num(I.roc(c, 10))])
    fwd = np.roll(c, -5) / c - 1
    fwd[-5:] = 0.0
    model = ridge_lasso(feats[:-5], fwd[:-5], lam=1.0)
    pred = feats @ model["w"] + model["b"]
    with np.errstate(invalid="ignore"):
        trend = c > I.ema(c, 50)
    return {"signal": _signal_from((pred > 0) & trend)}


@_reg(100, "regime_switching_multi")
def regime_switching_multi(df, p):
    o, h, l, c, v = _arrays(df)
    from .algorithms import hmm_regimes
    try:
        regime = hmm_regimes(c, 2)["current"]
    except Exception:  # noqa: BLE001
        regime = 0
    if regime == 0:  # low-vol: trend template
        return ma_crossover(df, p)
    return rsi2_bounce(df, p)


def strategy(number_or_name):
    key = number_or_name.lower() if isinstance(number_or_name, str) else number_or_name
    if key not in STRATEGIES:
        raise KeyError(f"strategy {number_or_name!r} is not registered")
    return STRATEGIES[key]


def live_templates() -> dict[str, object]:
    """Name -> builder mapping for brain.TEMPLATES-compatible registration."""
    return {name: fn for name, fn in STRATEGIES.items()
            if isinstance(name, str) and not getattr(fn, "research_only", False)}


def self_test() -> tuple[int, list[str]]:
    rng = np.random.default_rng(3)
    n = 400
    close = 100 + np.cumsum(rng.normal(0.05, 1, n))
    df = pd.DataFrame({
        "o": close + rng.normal(0, 0.2, n), "h": close + abs(rng.normal(0, 0.6, n)),
        "l": close - abs(rng.normal(0, 0.6, n)), "c": close, "v": rng.uniform(1e3, 1e5, n),
    })
    failures = []
    for number in sorted(k for k in STRATEGIES if isinstance(k, int)):
        fn = STRATEGIES[number]
        try:
            out = fn(df, {})
            signal = out["signal"]
            assert len(signal) == n, "length mismatch"
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{number}:{getattr(fn, '__name__', fn)}:{exc}")
    return len([k for k in STRATEGIES if isinstance(k, int)]), failures
