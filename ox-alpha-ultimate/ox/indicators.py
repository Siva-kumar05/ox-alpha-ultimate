"""100 market indicators as pure numpy functions with a unified registry.

Convention: inputs are 1-D float arrays (open o, high h, low l, close c,
volume v); outputs are arrays aligned to input length (leading warm-up values
are NaN). Market-data indicators (VIX, funding, OI, breadth) accept their
external series as an argument. Every function is registered in ``IND`` under
its number and name, so ``ind(21)`` or ``ind('rsi')`` returns the callable.
"""

from __future__ import annotations

import numpy as np

IND: dict = {}


def _reg(number: int, name: str):
    def deco(fn):
        IND[number] = fn
        IND[name] = fn
        fn.indicator_number = number
        return fn
    return deco


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _nan_like(*xs) -> np.ndarray:
    n = len(xs[0])
    return np.full(n, np.nan)


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (cumsum[n:] - cumsum[:-n]) / n
    return out


def _roll_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    mean = _roll_mean(x, n)
    sq = _roll_mean(x * x, n)
    var = np.maximum(sq - mean * mean, 0.0)
    out[n - 1:] = np.sqrt(var[n - 1:])
    return out


def _roll_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].max()
    return out


def _roll_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].min()
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = np.full_like(x, np.nan)
    if not len(x):
        return out
    start = int(np.argmax(~np.isnan(x))) if np.isnan(x).any() else 0
    out[start] = x[start]
    for i in range(start + 1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _wma(x: np.ndarray, n: int) -> np.ndarray:
    weights = np.arange(1, n + 1, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = float(np.dot(x[i - n + 1:i + 1], weights) / weights.sum())
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def _true_range(h, l, c) -> np.ndarray:
    pc = _shift(c, 1)
    return np.nanmax(np.stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)


# ── Trend 1–20 ────────────────────────────────────────────────────────────
@_reg(1, "sma")
def sma(c, n=20): return _roll_mean(_f(c), int(n))


@_reg(2, "ema")
def ema(c, n=20): return _ema(_f(c), int(n))


@_reg(3, "wma")
def wma(c, n=20): return _wma(_f(c), int(n))


@_reg(4, "hma")
def hma(c, n=20):
    c = _f(c)
    half = _wma(c, max(1, int(n) // 2))
    full = _wma(c, int(n))
    raw = 2 * half - full
    return _wma(raw, max(1, int(np.sqrt(n))))


@_reg(5, "tema")
def tema(c, n=20):
    c = _f(c)
    e1, e2, e3 = _ema(c, n), _ema(_ema(c, n), n), _ema(_ema(_ema(c, n), n), n)
    return 3 * e1 - 3 * e2 + e3


@_reg(6, "dema")
def dema(c, n=20):
    c = _f(c)
    e1, e2 = _ema(c, n), _ema(_ema(c, n), n)
    return 2 * e1 - e2


@_reg(7, "vwma")
def vwma(c, v, n=20):
    c, v = _f(c), _f(v)
    pv = _roll_mean(c * v, n)
    vv = _roll_mean(v, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pv / vv


@_reg(8, "kama")
def kama(c, n=10, fast=2, slow=30):
    c = _f(c)
    out = np.full_like(c, np.nan)
    if len(c) < n + 1:
        return out
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    out[n] = c[n]
    for i in range(n + 1, len(c)):
        change = abs(c[i] - c[i - n])
        volatility = np.sum(np.abs(c[i - n + 1:i + 1] - c[i - n:i]))
        er = change / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


@_reg(9, "supertrend")
def supertrend(h, l, c, n=10, mult=3.0):
    h, l, c = _f(h), _f(l), _f(c)
    atr = _ema(_true_range(h, l, c), n)
    mid = (h + l) / 2
    upper, lower = mid + mult * atr, mid - mult * atr
    trend = np.ones(len(c))
    line = lower.copy()
    for i in range(1, len(c)):
        if np.isnan(atr[i]):
            continue
        if c[i] > upper[i - 1]:
            trend[i] = 1
        elif c[i] < lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
        line[i] = lower[i] if trend[i] > 0 else upper[i]
    return line * trend


@_reg(10, "psar")
def psar(h, l, step=0.02, max_step=0.2):
    h, l = _f(h), _f(l)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    bull, af, ep = True, step, h[0]
    sar = l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, l[i - 1], l[max(0, i - 2)])
            if h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
            if l[i] < sar:
                bull, sar, ep, af = False, ep, l[i], step
        else:
            sar = max(sar, h[i - 1], h[max(0, i - 2)])
            if l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
            if h[i] > sar:
                bull, sar, ep, af = True, ep, h[i], step
        out[i] = sar
    return out


@_reg(11, "adx")
def adx(h, l, c, n=14):
    h, l, c = _f(h), _f(l), _f(c)
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(h, l, c)
    atr = _ema(tr, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * _ema(plus_dm, n) / atr
        mdi = 100 * _ema(minus_dm, n) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return _ema(dx, n)


@_reg(12, "aroon")
def aroon(h, l, n=25):
    h, l = _f(h), _f(l)
    up = np.full_like(h, np.nan)
    dn = np.full_like(h, np.nan)
    for i in range(n, len(h)):
        window = h[i - n:i + 1]
        up[i] = 100 * (n - (n - int(np.argmax(window)))) / n
        window_l = l[i - n:i + 1]
        dn[i] = 100 * (n - (n - int(np.argmin(window_l)))) / n
    return up, dn


@_reg(13, "vortex")
def vortex(h, l, c, n=14):
    h, l, c = _f(h), _f(l), _f(c)
    pc = _shift(c, 1)
    vm_p = np.abs(h - pc)
    vm_m = np.abs(l - pc)
    tr = _true_range(h, l, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        vip = _roll_sum(vm_p, n) / _roll_sum(tr, n)
        vim = _roll_sum(vm_m, n) / _roll_sum(tr, n)
    return vip, vim


def _roll_sum(x, n):
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = cumsum[n:] - cumsum[:-n]
    return out


@_reg(14, "macd")
def macd(c, fast=12, slow=26, signal=9):
    c = _f(c)
    line = _ema(c, fast) - _ema(c, slow)
    sig = _ema(line, signal)
    return line, sig, line - sig


@_reg(15, "ppo")
def ppo(c, fast=12, slow=26):
    c = _f(c)
    ef, es = _ema(c, fast), _ema(c, slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (ef - es) / es


@_reg(16, "trix")
def trix(c, n=15):
    c = _f(c)
    e3 = _ema(_ema(_ema(c, n), n), n)
    return 100 * np.diff(e3, prepend=e3[0]) / np.where(e3 == 0, np.nan, e3)


@_reg(17, "dmi")
def dmi(h, l, c, n=14):
    h, l, c = _f(h), _f(l), _f(c)
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _ema(_true_range(h, l, c), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * _ema(plus_dm, n) / atr, 100 * _ema(minus_dm, n) / atr


@_reg(18, "gmma")
def gmma(c, short=(3, 5, 8, 10, 12, 15), long=(30, 35, 40, 45, 50, 60)):
    c = _f(c)
    return ([_ema(c, n) for n in short], [_ema(c, n) for n in long])


@_reg(19, "zigzag")
def zigzag(h, l, pct=5.0):
    h, l = _f(h), _f(l)
    pivots: list[tuple[int, float, int]] = []
    direction = 1
    extreme_i, extreme = 0, h[0]
    for i in range(1, len(h)):
        if direction > 0:
            if h[i] > extreme:
                extreme_i, extreme = i, h[i]
            elif l[i] < extreme * (1 - pct / 100):
                pivots.append((extreme_i, extreme, 1))
                direction, extreme_i, extreme = -1, i, l[i]
        else:
            if l[i] < extreme:
                extreme_i, extreme = i, l[i]
            elif h[i] > extreme * (1 + pct / 100):
                pivots.append((extreme_i, extreme, -1))
                direction, extreme_i, extreme = 1, i, h[i]
    pivots.append((extreme_i, extreme, direction))
    out = np.full(len(h), np.nan)
    for i, value, _ in pivots:
        out[i] = value
    return out


@_reg(20, "linreg_slope")
def linreg_slope(c, n=14):
    c = _f(c)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = np.sum((x - xm) ** 2)
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        y = c[i - n + 1:i + 1]
        if not np.isnan(y).any():
            out[i] = np.sum((x - xm) * (y - y.mean())) / denom
    return out


# ── Momentum 21–40 ────────────────────────────────────────────────────────
@_reg(21, "rsi")
def rsi(c, n=14):
    c = _f(c)
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag, al = _ema(gain, n), _ema(loss, n)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al != 0)
    return 100 - 100 / (1 + rs)


@_reg(22, "stochastic")
def stochastic(h, l, c, n=14, d=3):
    h, l, c = _f(h), _f(l), _f(c)
    hh, ll = _roll_max(h, n), _roll_min(l, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 100 * (c - ll) / (hh - ll)
    return k, _roll_mean(k, d)


@_reg(23, "stochrsi")
def stochrsi(c, n=14, smooth=3):
    r = rsi(c, n)
    hh, ll = _roll_max(r, n), _roll_min(r, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (r - ll) / (hh - ll)
    return _roll_mean(k, smooth)


@_reg(24, "williams_r")
def williams_r(h, l, c, n=14):
    h, l, c = _f(h), _f(l), _f(c)
    hh, ll = _roll_max(h, n), _roll_min(l, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -100 * (hh - c) / (hh - ll)


@_reg(25, "cci")
def cci(h, l, c, n=20):
    tp = (_f(h) + _f(l) + _f(c)) / 3
    ma = _roll_mean(tp, n)
    dev = _roll_mean(np.abs(tp - ma), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tp - ma) / (0.015 * dev)


@_reg(26, "roc")
def roc(c, n=12):
    c = _f(c)
    prev = _shift(c, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (c - prev) / prev


@_reg(27, "momentum")
def momentum(c, n=10):
    return _f(c) - _shift(_f(c), n)


@_reg(28, "ultimate_osc")
def ultimate_osc(h, l, c, p1=7, p2=14, p3=28):
    h, l, c = _f(h), _f(l), _f(c)
    pc = _shift(c, 1)
    true_low = np.minimum(l, pc)
    bp = c - true_low
    tr = np.maximum(h, pc) - true_low
    with np.errstate(invalid="ignore", divide="ignore"):
        avg1 = _roll_sum(bp, p1) / _roll_sum(tr, p1)
        avg2 = _roll_sum(bp, p2) / _roll_sum(tr, p2)
        avg3 = _roll_sum(bp, p3) / _roll_sum(tr, p3)
        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


@_reg(29, "awesome_osc")
def awesome_osc(h, l, n1=5, n2=34):
    mid = (_f(h) + _f(l)) / 2
    return _ema(mid, n1) - _ema(mid, n2)


@_reg(30, "kst")
def kst(c, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15):
    c = _f(c)
    rcas = [roc(c, r) for r in (r1, r2, r3, r4)]
    smoothed = [_roll_sum(rc, s) for rc, s in zip(rcas, (s1, s2, s3, s4))]
    return sum(smoothed)


@_reg(31, "tsi")
def tsi(c, r=25, s=13):
    c = _f(c)
    m = np.diff(c, prepend=c[0])
    return _ema(_ema(m, r), s) / np.where(_ema(_ema(np.abs(m), r), s) == 0, np.nan,
                                          _ema(_ema(np.abs(m), r), s))


@_reg(32, "chande_momentum")
def chande_momentum(c, n=14):
    c = _f(c)
    delta = np.diff(c, prepend=c[0])
    up = _roll_sum(np.where(delta > 0, delta, 0.0), n)
    dn = _roll_sum(np.where(delta < 0, -delta, 0.0), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (up - dn) / (up + dn)


@_reg(33, "fisher_transform")
def fisher_transform(h, l, n=9):
    h, l = _f(h), _f(l)
    mid = (h + l) / 2
    hh, ll = _roll_max(mid, n), _roll_min(mid, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = 2 * ((mid - ll) / (hh - ll) - 0.5)
    fisher = np.full_like(raw, np.nan)
    value = 0.0
    for i in range(len(raw)):
        if np.isnan(raw[i]):
            continue
        v = 0.66 * raw[i] + 0.67 * value
        v = max(min(v, 0.999), -0.999)
        fisher[i] = 0.5 * np.log((1 + v) / (1 - v))
        value = v
    return fisher


@_reg(34, "relative_vigor")
def relative_vigor(o, h, l, c, n=10):
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    num = c - o
    den = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll_sum(num, n) / _roll_sum(den, n)


@_reg(35, "connors_rsi")
def connors_rsi(h, l, c, rsi_n=3, streak_n=2, roc_n=100):
    c = _f(c)
    r = rsi(c, rsi_n)
    streak = np.zeros_like(c)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            streak[i] = max(1.0, streak[i - 1] + 1) if streak[i - 1] > 0 else 1.0
        elif c[i] < c[i - 1]:
            streak[i] = min(-1.0, streak[i - 1] - 1) if streak[i - 1] < 0 else -1.0
        else:
            streak[i] = 0.0
    s_rsi = _streak_rsi(streak, streak_n)
    pct_rank = _percent_rank(roc(c, roc_n), roc_n)
    with np.errstate(invalid="ignore"):
        return (r + s_rsi + pct_rank) / 3


def _streak_rsi(streak, n):
    delta = np.diff(streak, prepend=streak[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag, al = _ema(gain, n), _ema(loss, n)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al != 0)
    return 100 - 100 / (1 + rs)


def _percent_rank(x, n):
    out = np.full_like(x, np.nan)
    for i in range(n, len(x)):
        window = x[i - n:i]
        valid = window[~np.isnan(window)]
        if len(valid) and not np.isnan(x[i]):
            out[i] = np.mean(valid < x[i])
    return out


@_reg(36, "smi")
def smi(h, l, c, n=10, smooth=3):
    h, l, c = _f(h), _f(l), _f(c)
    hh, ll = _roll_max(h, n), _roll_min(l, n)
    m = ll + (hh - ll) / 2
    d = c - m
    num = _ema(_ema(d, smooth), smooth)
    den = _ema(_ema(hh - ll, smooth), smooth)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * num / np.where(den / 2 == 0, np.nan, den / 2)


@_reg(37, "inertia")
def inertia(c, n=14, rsi_n=14):
    r = rsi(c, rsi_n)
    out = np.full_like(_f(c), np.nan)
    x = np.arange(n, dtype=float)
    for i in range(n, len(r)):
        y = r[i - n:i]
        if not np.isnan(y).any():
            out[i] = np.sqrt(np.mean((y - (np.polyval(np.polyfit(x, y, 1), x))) ** 2))
    return out


@_reg(38, "qstick")
def qstick(o, c, n=14):
    return _roll_mean(_f(c) - _f(o), n)


@_reg(39, "balance_of_power")
def balance_of_power(o, h, l, c):
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c - o) / np.where(h - l == 0, np.nan, h - l)


@_reg(40, "coppock")
def coppock(c, w1=14, w2=11, w3=10):
    c = _f(c)
    rc = roc(c, w1) + roc(c, w2)
    return _wma(np.nan_to_num(rc, nan=0.0), w3)


# ── Volatility 41–55 ──────────────────────────────────────────────────────
@_reg(41, "atr")
def atr(h, l, c, n=14):
    return _ema(_true_range(_f(h), _f(l), _f(c)), n)


@_reg(42, "bollinger")
def bollinger(c, n=20, k=2.0):
    c = _f(c)
    mid = _roll_mean(c, n)
    sd = _roll_std(c, n)
    return mid + k * sd, mid, mid - k * sd


@_reg(43, "keltner")
def keltner(h, l, c, n=20, mult=2.0):
    c = _f(c)
    mid = _ema(c, n)
    a = atr(h, l, c, n)
    return mid + mult * a, mid, mid - mult * a


@_reg(44, "donchian")
def donchian(h, l, n=20):
    upper, lower = _roll_max(_f(h), n), _roll_min(_f(l), n)
    return upper, (upper + lower) / 2, lower


@_reg(45, "stdev")
def stdev(c, n=20): return _roll_std(_f(c), n)


@_reg(46, "chaikin_volatility")
def chaikin_volatility(h, l, c, n=10, roc_n=10):
    hl = (_f(h) - _f(l))
    ema_hl = _ema(hl, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (ema_hl - _shift(ema_hl, roc_n)) / np.where(_shift(ema_hl, roc_n) == 0, np.nan, _shift(ema_hl, roc_n))


@_reg(47, "historical_volatility")
def historical_volatility(c, n=20, periods_per_year=252):
    c = _f(c)
    logret = np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=np.nan)
    return _roll_std(logret, n) * np.sqrt(periods_per_year) * 100


@_reg(48, "ulcer_index")
def ulcer_index(c, n=14):
    c = _f(c)
    out = np.full_like(c, np.nan)
    for i in range(n, len(c)):
        window = c[i - n:i + 1]
        dd = 100 * (window - np.maximum.accumulate(window)) / np.maximum.accumulate(window)
        out[i] = np.sqrt(np.mean(dd ** 2))
    return out


@_reg(49, "volatility_stop")
def volatility_stop(h, l, c, n=20, mult=2.0):
    a = atr(h, l, c, n)
    mid = _f(c)
    return mid - mult * a, mid + mult * a


@_reg(50, "mass_index")
def mass_index(h, l, c, n=25):
    h, l = _f(h), _f(l)
    rng = h - l
    e1, e2 = _ema(rng, 9), _ema(_ema(rng, 9), 9)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = e1 / np.where(e2 == 0, np.nan, e2)
    return _roll_sum(ratio, n)


@_reg(51, "aroon_volatility")
def aroon_volatility(h, l, n=25):
    up, dn = aroon(h, l, n)
    with np.errstate(invalid="ignore"):
        return np.abs(up - dn) / 2


@_reg(52, "relative_volatility")
def relative_volatility(c, o, n=10):
    c, o = _f(c), _f(o)
    sd = _roll_std(c, n)
    direction = np.sign(c - o)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * _ema(direction * sd, n) / np.where(_roll_std(sd, n) == 0, np.nan, sd)


@_reg(53, "width_ratio_bands")
def width_ratio_bands(c, n=20, k=2.0):
    upper, mid, lower = bollinger(c, n, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (upper - lower) / np.where(mid == 0, np.nan, mid)


@_reg(54, "choppiness_index")
def choppiness_index(h, l, c, n=14):
    h, l, c = _f(h), _f(l), _f(c)
    tr = _true_range(h, l, c)
    sum_tr = _roll_sum(tr, n)
    hh, ll = _roll_max(h, n), _roll_min(l, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.log10(sum_tr / (hh - ll)) / np.log10(n)


@_reg(55, "implied_vol_proxy")
def implied_vol_proxy(vix_series, c, n=20):
    """IV proxy: external implied-vol series (e.g. India VIX) blended with realized vol."""
    vix, c = _f(vix_series), _f(c)
    return 0.5 * _roll_mean(vix, n) + 0.5 * historical_volatility(c, n)


# ── Volume 56–75 ──────────────────────────────────────────────────────────
@_reg(56, "obv")
def obv(c, v):
    c, v = _f(c), _f(v)
    direction = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(direction * v)


@_reg(57, "vwap")
def vwap(h, l, c, v):
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    tp = (h + l + c) / 3
    return np.cumsum(tp * v) / np.cumsum(v)


@_reg(58, "mfi")
def mfi(h, l, c, v, n=14):
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    tp = (h + l + c) / 3
    raw = tp * v
    pos = np.where(tp > _shift(tp, 1), raw, 0.0)
    neg = np.where(tp < _shift(tp, 1), raw, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + _roll_sum(pos, n) / _roll_sum(neg, n))


@_reg(59, "chaikin_money_flow")
def chaikin_money_flow(h, l, c, v, n=20):
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = ((c - l) - (h - c)) / np.where(h - l == 0, np.nan, h - l)
    mf = np.nan_to_num(mfm) * v
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll_sum(mf, n) / _roll_sum(v, n)


@_reg(60, "ad_line")
def ad_line(h, l, c, v):
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = ((c - l) - (h - c)) / np.where(h - l == 0, np.nan, h - l)
    return np.cumsum(np.nan_to_num(mfm) * v)


@_reg(61, "volume_oscillator")
def volume_oscillator(v, fast=5, slow=20):
    v = _f(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (_roll_mean(v, fast) - _roll_mean(v, slow)) / _roll_mean(v, slow)


@_reg(62, "pvt")
def pvt(c, v):
    c, v = _f(c), _f(v)
    pct = np.diff(c, prepend=c[0]) / np.where(_shift(c, 1) == 0, np.nan, _shift(c, 1))
    return np.cumsum(np.nan_to_num(pct) * v)


@_reg(63, "nvi")
def nvi(c, v):
    c, v = _f(c), _f(v)
    out = np.full_like(c, 1000.0)
    for i in range(1, len(c)):
        if v[i] < v[i - 1]:
            ret = (c[i] - c[i - 1]) / c[i - 1] if c[i - 1] else 0
            out[i] = out[i - 1] * (1 + ret)
        else:
            out[i] = out[i - 1]
    return out


@_reg(64, "pvi")
def pvi(c, v):
    c, v = _f(c), _f(v)
    out = np.full_like(c, 1000.0)
    for i in range(1, len(c)):
        if v[i] > v[i - 1]:
            ret = (c[i] - c[i - 1]) / c[i - 1] if c[i - 1] else 0
            out[i] = out[i - 1] * (1 + ret)
        else:
            out[i] = out[i - 1]
    return out


@_reg(65, "volume_profile_poc")
def volume_profile_poc(h, l, v, bins=50):
    """POC / value-area per bar from a trailing window of bar volumes."""
    h, l, v = _f(h), _f(l), _f(v)
    poc = np.full_like(h, np.nan)
    for i in range(len(h)):
        if np.isnan(h[i]) or np.isnan(l[i]) or h[i] == l[i]:
            continue
        edges = np.linspace(l[i], h[i], bins + 1)
        hist = np.full(bins, v[i] / bins)  # uniform proxy without intrabar data
        poc[i] = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2
    return poc


@_reg(66, "cvd")
def cvd(o, h, l, c, v):
    """Cumulative volume delta estimated from intrabar geometry (close position)."""
    o, h, l, c, v = _f(o), _f(h), _f(l), _f(c), _f(v)
    rng = np.where(h - l == 0, np.nan, h - l)
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_frac = (c - l) / rng
    delta = np.nan_to_num(2 * buy_frac - 1) * v
    return np.cumsum(delta)


@_reg(67, "ease_of_movement")
def ease_of_movement(h, l, v, n=14, scale=1e6):
    h, l, v = _f(h), _f(l), _f(v)
    mid_move = (h + l) / 2 - (_shift(h, 1) + _shift(l, 1)) / 2
    box_ratio = (v / scale) / np.where(h - l == 0, np.nan, h - l)
    return _roll_mean(mid_move / box_ratio, n)


@_reg(68, "klinger")
def klinger(h, l, c, v, fast=34, slow=55):
    h, l, c, v = _f(h), _f(l), _f(c), _f(v)
    trend = np.sign(h + l + c - _shift(h + l + c, 1))
    vf = v * trend
    sig = _ema(_ema(vf, fast) - _ema(vf, slow), 13)
    return _ema(vf, fast) - _ema(vf, slow), sig


@_reg(69, "force_index")
def force_index(c, v, n=13):
    c, v = _f(c), _f(v)
    fi = np.diff(c, prepend=c[0]) * v
    return _ema(fi, n)


@_reg(70, "vwrsi")
def volume_weighted_rsi(c, v, n=14):
    c, v = _f(c), _f(v)
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta * v, 0.0)
    loss = np.where(delta < 0, -delta * v, 0.0)
    ag, al = _roll_mean(gain, n), _roll_mean(loss, n)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al != 0)
    return 100 - 100 / (1 + rs)


@_reg(71, "mfi_mtf")
def mfi_mtf(h, l, c, v, n=14, resample=5):
    base = mfi(h, l, c, v, n)
    return _roll_mean(base, resample)


@_reg(72, "up_down_volume_ratio")
def up_down_volume_ratio(c, v, n=14):
    c, v = _f(c), _f(v)
    up = np.where(np.diff(c, prepend=c[0]) > 0, v, 0.0)
    dn = np.where(np.diff(c, prepend=c[0]) < 0, v, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll_sum(up, n) / _roll_sum(dn, n)


@_reg(73, "tick_volume")
def tick_volume(v, n=1):
    return _roll_sum(_f(v), n)


@_reg(74, "demand_index")
def demand_index(c, v, n=14):
    """Smoothed buy-pressure vs sell-pressure proxy."""
    c, v = _f(c), _f(v)
    delta = np.diff(c, prepend=c[0])
    buy = np.where(delta > 0, v, 0.0)
    sell = np.where(delta < 0, v, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _ema(100 * (buy - sell) / (buy + sell), n)


@_reg(75, "volume_delta")
def volume_delta(o, h, l, c, v):
    o, h, l, c, v = _f(o), _f(h), _f(l), _f(c), _f(v)
    rng = np.where(h - l == 0, np.nan, h - l)
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_frac = (c - l) / rng
    return np.nan_to_num(2 * buy_frac - 1) * v


# ── Breadth / sentiment / derived 76–100 ─────────────────────────────────
@_reg(76, "advance_decline")
def advance_decline(advancing, declining):
    return np.cumsum(_f(advancing) - _f(declining))


@_reg(77, "mcclellan")
def mcclellan(advancing, declining, fast=19, slow=39):
    ad = _f(advancing) - _f(declining)
    return _ema(ad, fast) - _ema(ad, slow)


@_reg(78, "put_call_ratio")
def put_call_ratio(put_volume, call_volume):
    with np.errstate(invalid="ignore", divide="ignore"):
        return _f(put_volume) / _f(call_volume)


@_reg(79, "fear_greed")
def fear_greed(c, v, n=20):
    """Composite proxy when no external sentiment feed: momentum + volume tilt."""
    c, v = _f(c), _f(v)
    mom = np.clip(roc(c, n) / 5.0, -1, 1)
    vol_tilt = np.clip(volume_delta(c, v, c, c, v) / np.maximum(_roll_sum(v, n), 1) * 2, -1, 1)
    return 50 + 25 * (mom + vol_tilt)


@_reg(80, "funding_rate")
def funding_rate(funding_series, n=8):
    return _roll_mean(_f(funding_series), n)


@_reg(81, "open_interest")
def open_interest(oi_series, n=5):
    return _roll_mean(_f(oi_series), n)


@_reg(82, "short_interest")
def short_interest(si_series, n=5):
    return _roll_mean(_f(si_series), n)


@_reg(83, "cot_positioning")
def cot_positioning(commercial_long, commercial_short, n=4):
    cl, cs = _f(commercial_long), _f(commercial_short)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll_mean(100 * (cl - cs) / (cl + cs), n)


@_reg(84, "hurst_exponent")
def hurst_exponent(c, max_lag=20):
    """R/S-analysis Hurst exponent; >0.5 trending, <0.5 mean-reverting."""
    c = _f(c)
    out = np.full_like(c, np.nan)
    lags = np.arange(2, max_lag + 1)
    for i in range(max_lag * 2, len(c)):
        window = np.log(np.where(c[i - max_lag * 2:i] > 0, c[i - max_lag * 2:i], np.nan))
        window = window[~np.isnan(window)]
        if len(window) < max_lag + 2:
            continue
        tau = [np.std(window[lag:] - window[:-lag]) for lag in lags]
        if all(t > 0 for t in tau):
            out[i] = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return out


@_reg(85, "zscore")
def zscore(c, n=20):
    c = _f(c)
    mean, sd = _roll_mean(c, n), _roll_std(c, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c - mean) / sd


@_reg(86, "rolling_sharpe")
def rolling_sharpe(c, n=60, periods_per_year=252):
    c = _f(c)
    ret = np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=np.nan)
    mean, sd = _roll_mean(ret, n), _roll_std(ret, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mean / sd * np.sqrt(periods_per_year)


@_reg(87, "rolling_max_drawdown")
def rolling_max_drawdown(c, n=60):
    c = _f(c)
    out = np.full_like(c, np.nan)
    for i in range(n, len(c)):
        window = c[i - n:i + 1]
        peak = np.maximum.accumulate(window)
        out[i] = float(np.min((window - peak) / peak))
    return out


@_reg(88, "beta")
def beta(c, benchmark, n=60):
    c, b = _f(c), _f(benchmark)
    rc = np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=np.nan)
    rb = np.diff(np.log(np.where(b > 0, b, np.nan)), prepend=np.nan)
    cov = _roll_mean((rc - _roll_mean(rc, n)) * (rb - _roll_mean(rb, n)), n)
    var = _roll_mean((rb - _roll_mean(rb, n)) ** 2, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / var


@_reg(89, "correlation")
def correlation(c, other, n=60):
    c, o = _f(c), _f(other)
    mc, mo = _roll_mean(c, n), _roll_mean(o, n)
    cov = _roll_mean((c - mc) * (o - mo), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / (_roll_std(c, n) * _roll_std(o, n))


@_reg(90, "rsi_divergence")
def rsi_divergence(c, n=14, look=5):
    """+1 bullish divergence (price LL, RSI HL), -1 bearish (price HH, RSI LL)."""
    c = _f(c)
    r = rsi(c, n)
    out = np.zeros_like(c)
    for i in range(look * 2, len(c)):
        pl = c[i - look]
        rl, rh = r[i - look], r[i]
        if np.isnan(rl) or np.isnan(rh):
            continue
        if pl < c[i - 2 * look] and rl > r[i - 2 * look]:
            out[i] = 1
        elif pl > c[i - 2 * look] and rl < r[i - 2 * look]:
            out[i] = -1
    return out


@_reg(91, "fractal_dimension")
def fractal_dimension(c, n=30):
    """Higuchi fractal dimension: ~1.5 random, >1.5 choppy, <1.5 directional."""
    c = _f(c)
    out = np.full_like(c, np.nan)
    k_max = 8
    for i in range(n, len(c)):
        x = c[i - n:i]
        lengths = []
        for k in range(1, k_max + 1):
            l_k = 0.0
            for m in range(k):
                idx = np.arange(m, len(x), k)
                seq = x[idx]
                norm = (len(x) - 1) / (len(idx) * k)
                l_k += np.sum(np.abs(np.diff(seq))) * norm / k
            lengths.append((k, l_k / k))
        ln_k = np.log([p[0] for p in lengths])
        ln_l = np.log(np.maximum([p[1] for p in lengths], 1e-12))
        out[i] = -np.polyfit(ln_k, ln_l, 1)[0]
    return out


@_reg(92, "schaff_trend_cycle")
def schaff_trend_cycle(c, fast=23, slow=50, cycle=10):
    c = _f(c)
    macd_line = _ema(c, fast) - _ema(c, slow)
    k1 = _stoch_macd(macd_line, cycle)
    d1 = _ema(k1, 3)
    k2 = _stoch_macd(d1, cycle)
    return _ema(k2, 3)


def _stoch_macd(x, n):
    hh, ll = _roll_max(x, n), _roll_min(x, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (x - ll) / (hh - ll)


@_reg(93, "elder_ray")
def elder_ray(h, l, c, n=13):
    c = _f(c)
    e = _ema(c, n)
    return _f(h) - e, _f(l) - e  # bull power, bear power


@_reg(94, "ichimoku")
def ichimoku(h, l, tenkan=9, kijun=26, senkou=52):
    h, l = _f(h), _f(l)
    mid = lambda a, b: (_roll_max(h, a) + _roll_min(l, b)) / 2
    t = mid(tenkan, tenkan)
    k = mid(kijun, kijun)
    span_a = (t + k) / 2
    span_b = mid(senkou, senkou)
    return t, k, span_a, span_b


@_reg(95, "pivot_points")
def pivot_points(h, l, c, n=1):
    h, l, c = _f(h), _f(l), _f(c)
    p = np.full_like(c, np.nan)
    r1 = np.full_like(c, np.nan)
    s1 = np.full_like(c, np.nan)
    r2 = np.full_like(c, np.nan)
    s2 = np.full_like(c, np.nan)
    for i in range(1, len(c)):
        pp = (h[i - 1] + l[i - 1] + c[i - 1]) / 3
        p[i], r1[i], s1[i] = pp, 2 * pp - l[i - 1], 2 * pp - h[i - 1]
        r2[i], s2[i] = pp + (h[i - 1] - l[i - 1]), pp - (h[i - 1] - l[i - 1])
    return p, r1, s1, r2, s2


@_reg(96, "fibonacci_levels")
def fibonacci_levels(h, l, swing_n=50):
    h, l = _f(h), _f(l)
    hh, ll = _roll_max(h, swing_n), _roll_min(l, swing_n)
    levels = {}
    for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
        levels[ratio] = ll + ratio * (hh - ll)
    return levels


@_reg(97, "cpr")
def cpr(h, l, c, n=1):
    h, l, c = _f(h), _f(l), _f(c)
    pivot = np.full_like(c, np.nan)
    bc = np.full_like(c, np.nan)
    tc = np.full_like(c, np.nan)
    for i in range(1, len(c)):
        p = (h[i - 1] + l[i - 1] + c[i - 1]) / 3
        pivot[i] = p
        bcl = (h[i - 1] + l[i - 1]) / 2
        bc[i] = min(bcl, p + (h[i - 1] - l[i - 1]) / 2)
        tc[i] = max(bcl, p + (h[i - 1] - l[i - 1]) / 2)
    return pivot, bc, tc


@_reg(98, "relative_strength")
def relative_strength(c, benchmark, n=60):
    c, b = _f(c), _f(benchmark)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = c / b
    return rs, roc(rs, n)


@_reg(99, "iv_skew")
def iv_skew(call_iv, put_iv, n=5):
    return _roll_mean(_f(call_iv) - _f(put_iv), n)


@_reg(100, "liquidity_depth")
def liquidity_depth(bid_depth, ask_depth, n=10):
    bid, ask = _f(bid_depth), _f(ask_depth)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _roll_mean(100 * (bid - ask) / (bid + ask), n)


def ind(number_or_name):
    """Fetch a registered indicator by number (21 -> RSI) or name ('rsi')."""
    key = number_or_name.lower() if isinstance(number_or_name, str) else number_or_name
    if key not in IND:
        raise KeyError(f"indicator {number_or_name!r} is not registered")
    return IND[key]


def available() -> list[int]:
    return sorted(k for k in IND if isinstance(k, int))


def self_test() -> tuple[int, list[str]]:
    rng = np.random.default_rng(7)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = (high + low) / 2
    volume = rng.uniform(1e3, 1e5, n)
    failures = []
    series_args = {"o": open_, "h": high, "l": low, "c": close, "v": volume,
                   "vix_series": np.full(n, 15.0), "benchmark": close * 1.01,
                   "other": close * 0.99, "advancing": rng.uniform(50, 150, n),
                   "declining": rng.uniform(50, 150, n), "put_volume": rng.uniform(1e3, 5e3, n),
                   "call_volume": rng.uniform(1e3, 5e3, n), "funding_series": rng.normal(0, 0.01, n),
                   "oi_series": rng.uniform(1e5, 1e6, n), "si_series": rng.uniform(1e4, 1e5, n),
                   "commercial_long": rng.uniform(1e3, 5e3, n), "commercial_short": rng.uniform(1e3, 5e3, n),
                   "call_iv": rng.uniform(0.1, 0.3, n), "put_iv": rng.uniform(0.1, 0.3, n),
                   "bid_depth": rng.uniform(1e4, 1e6, n), "ask_depth": rng.uniform(1e4, 1e6, n)}
    for number in available():
        fn = IND[number]
        try:
            params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
            fn(**{name: series_args[name] for name in params if name in series_args})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{number}:{fn.__name__}:{exc}")
    return len(available()), failures
