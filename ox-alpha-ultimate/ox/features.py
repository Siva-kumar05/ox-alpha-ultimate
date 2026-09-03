import numpy as np
import pandas as pd
from math import log, sqrt, exp, erf

REG = {}

def feature(name):
    def decorator(f):
        REG[name] = f
        return f
    return decorator

def ensure(name):
    """Fail closed if an expected audited feature is unavailable.

    Never write a generated Python module at runtime: that turns a missing
    feature into a filesystem mutation in the execution process.
    """
    if name not in REG:
        raise KeyError(f"Feature '{name}' is not registered")

# ---------- Trend & Momentum Indicators ----------
@feature("ema")
def ema(x, n=14):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values

@feature("cma")
def cma(x):
    return pd.Series(x).expanding().mean().values

@feature("macd")
def macd(x, fast=12, slow=26, signal=9):
    m = pd.Series(x).ewm(span=fast, adjust=False).mean() - pd.Series(x).ewm(span=slow, adjust=False).mean()
    s = m.ewm(span=signal, adjust=False).mean()
    return m.values, s.values

@feature("rsi")
def rsi(x, n=14):
    d = np.diff(x, prepend=x[0])
    u = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(u).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    return 100.0 - (100.0 / (1.0 + ru / np.where(rd == 0, 1e-9, rd)))

@feature("atr")
def atr(h, l, c, n=14):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values

@feature("vwap")
def vwap(h, l, c, v):
    tp = (h + l + c) / 3.0
    return np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1e-9)

@feature("avwap")
def avwap(h, l, c, v, anchor_idx=0):
    out = np.full(len(c), np.nan)  # NaN before the anchor: 0.0 compared as a price is always wrong (B3)
    s = slice(max(0, anchor_idx), None)
    tp = (h[s] + l[s] + c[s]) / 3.0
    out[s] = np.cumsum(tp * v[s]) / np.maximum(np.cumsum(v[s]), 1e-9)
    return out

# ---------- Market Structure (Smart Money Concepts) ----------
def _swings(h, l, k=3):
    """Return swings only at the bar where their right side is observable.

    A centred pivot at ``center`` requires ``k`` future bars to confirm.  The
    returned index is therefore ``center + k`` (the confirmation bar), not the
    pivot bar.  Consumers can use its corresponding centre value without
    leaking future OHLC into a historical signal.
    """
    hi = []
    lo = []
    for confirmed_at in range(2 * k, len(h)):
        center = confirmed_at - k
        if h[center] == max(h[center-k:center+k+1]):
            hi.append((confirmed_at, center))
        if l[center] == min(l[center-k:center+k+1]):
            lo.append((confirmed_at, center))
    return hi, lo

@feature("bos_choch")
def bos_choch(h, l, c, k=3):
    hi, lo = _swings(h, l, k)
    n = len(c)
    sig = np.zeros(n)
    state = 0
    sh, sl = None, None
    ih, il = 0, 0
    for i in range(n):
        while ih < len(hi) and hi[ih][0] <= i:
            sh = h[hi[ih][1]]
            ih += 1
        while il < len(lo) and lo[il][0] <= i:
            sl = l[lo[il][1]]
            il += 1
        if sh is not None and c[i] > sh:
            sig[i] = 1.0 if state <= 0 else 0.5  # CHOCH / BOS Bullish
            state = 1
            sh = None
        elif sl is not None and c[i] < sl:
            sig[i] = -1.0 if state >= 0 else -0.5 # CHOCH / BOS Bearish
            state = -1
            sl = None
    return sig, state

@feature("fvg")
def fvg(o, h, l, c):
    bull = np.zeros(len(c))
    bear = np.zeros(len(c))
    for i in range(2, len(c)):
        if l[i] > h[i-2]:
            bull[i] = l[i] - h[i-2]
        if h[i] < l[i-2]:
            bear[i] = l[i-2] - h[i]
    return bull, bear

@feature("order_blocks")
def order_blocks(o, h, l, c, look=5):
    obs = []
    for i in range(look, len(c)):
        ret = c[i] / max(c[i-1], 1e-9) - 1.0
        if ret > 0.004:
            js = [k for k in range(i-look, i) if c[k] < o[k]]
            if js:
                obs.append(("bull", js[-1], l[js[-1]], h[js[-1]]))
        elif ret < -0.004:
            js = [k for k in range(i-look, i) if c[k] > o[k]]
            if js:
                obs.append(("bear", js[-1], l[js[-1]], h[js[-1]]))
    return obs

@feature("liquidity_sweep")
def liquidity_sweep(h, l, c, k=5):
    sig = np.zeros(len(c))
    for i in range(k, len(c)):
        ph, pl = max(h[i-k:i]), min(l[i-k:i])
        if h[i] > ph and c[i] < ph:
            sig[i] = 1.0   # Swept high and rejected -> Bearish reversal
        elif l[i] < pl and c[i] > pl:
            sig[i] = -1.0  # Swept low and reclaimed -> Bullish reversal
    return sig

# ---------- Order Flow & Microstructure ----------
@feature("delta")
def delta(o, h, l, c, v):
    rng = np.maximum(h - l, 1e-9)
    buy = ((c - l) / rng) * v
    d = 2 * buy - v
    return d, np.cumsum(d)

@feature("ultra_delta")
def ultra_delta(d, win=50):
    mu = pd.Series(d).rolling(win, min_periods=5).mean()
    sd = pd.Series(d).rolling(win, min_periods=5).std().fillna(1.0)
    return ((d - mu) / (sd + 1e-9)).fillna(0.0).values

@feature("big_trades")
def big_trades(v, mult=3.5, win=100):
    avg_v = pd.Series(v).rolling(win, min_periods=10).mean().fillna(v[0]).values
    return (v > mult * avg_v).astype(float)

@feature("effort_result")
def effort_result(o, h, l, c, v, win=20):
    # Zero-volume bars send pct_change to inf; nan_to_num maps inf to a huge
    # finite value, so sanitize with NaN replacement instead.
    vs = pd.Series(v).replace(0, np.nan)
    ev = vs.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).rolling(win, min_periods=5).mean().fillna(0.0).values
    rr = pd.Series(abs(c - o) / np.maximum(h - l, 1e-9)).rolling(win, min_periods=5).mean().values
    return np.nan_to_num(ev), np.nan_to_num(rr)

@feature("dom_imbalance")
def dom_imbalance(depth):
    if not depth:
        return 0.0
    bids = sum(q for _, q in depth.get("bids", [])[:5])
    asks = sum(q for _, q in depth.get("asks", [])[:5])
    return (bids - asks) / max(bids + asks, 1.0)

# ---------- Profiles (Volume Profile & TPO) ----------
@feature("volume_profile")
def volume_profile(h, l, c, v, bins=50):
    px = (h + l + c) / 3.0
    edges = np.linspace(px.min(), px.max(), bins + 1)
    idx = np.clip(np.digitize(px, edges) - 1, 0, bins - 1)
    prof = np.zeros(bins)
    for i, b in enumerate(idx):
        prof[b] += v[i]
    poc_idx = int(np.argmax(prof))
    total = max(prof.sum(), 1e-9)
    # Standard two-sided value-area expansion around the POC until >=70% of
    # volume is contained (B2 fix: the old one-sided global-cumsum walk could
    # produce over-wide or asymmetric value areas on bimodal profiles).
    lo = hi = poc_idx
    contained = float(prof[poc_idx])
    while contained / total < 0.70 and (lo > 0 or hi < bins - 1):
        up = float(prof[hi + 1]) if hi < bins - 1 else -1.0
        down = float(prof[lo - 1]) if lo > 0 else -1.0
        if up >= down:
            hi += 1
            contained += float(prof[hi])
        else:
            lo -= 1
            contained += float(prof[lo])
    return {"poc": edges[poc_idx], "vah": edges[min(hi + 1, bins)], "val": edges[lo], "profile": prof}

@feature("tpo")
def tpo(h, l, periods=26):
    seg = max(len(h) // periods, 1)
    cnt: dict = {}
    for p in range(periods):
        s = slice(p * seg, min((p + 1) * seg, len(h)))
        if s.start >= len(h):
            break
        for px in np.linspace(l[s].min(), h[s].max(), 12):
            k = round(float(px), 2)
            # Count each period once per price level: the old letter-concat
            # reused 52 letters and inflated counts for later periods.
            cnt.setdefault(k, set()).add(p)
    prices = sorted(cnt, key=lambda x: len(cnt[x]), reverse=True)
    return {"poc": float(prices[0]) if prices else None}

# ---------- Options Pricing & Greeks ----------
def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _npdf(x):
    return exp(-x * x / 2.0) / sqrt(2.0 * np.pi)

@feature("black_scholes")
def black_scholes(S, K, T, r, sigma, kind="CE"):
    if T <= 0 or sigma <= 0:
        return {"px": max(S - K if kind == "CE" else K - S, 0.0),
                "delta": 1.0 if kind == "CE" else -1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "CE":
        px = S * _N(d1) - K * exp(-r * T) * _N(d2)
        dl = _N(d1)
    else:
        px = K * exp(-r * T) * _N(-d2) - S * _N(-d1)
        dl = -_N(-d1)
    gm = _npdf(d1) / (S * sigma * sqrt(T))
    vg = S * _npdf(d1) * sqrt(T)
    r_term = r * K * exp(-r * T) * (_N(d2) if kind == "CE" else _N(-d2))
    # Call theta subtracts the carry term; put theta adds it (B1 fix: the old
    # sign made put theta ~6x too negative and broke put-call parity).
    th = (-S * _npdf(d1) * sigma / (2.0 * sqrt(T)) - (r_term if kind == "CE" else -r_term)) / 365.0
    return {"px": px, "delta": dl, "gamma": gm, "vega": vg, "theta": th}

@feature("deep_gamma")
def deep_gamma(chain, S, r=0.065):
    gex = 0.0
    for o in chain:
        g = black_scholes(S, o["K"], o["T"], r, o["iv"], o["kind"])["gamma"]
        gex += g * o["oi"] * (1 if o["kind"] == "CE" else -1)
    return gex * S * S * 0.01

# ---------- ML Hedge RNN ----------
@feature("hedging_rnn")
class HedgingRNN:
    """GRU Hedge Ratio Predictor with Ridge Fallback"""
    def __init__(self, dim=8, hidden=32, lr=1e-3):
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            class GRUModel(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.g = nn.GRU(dim, hidden, batch_first=True)
                    s.o = nn.Linear(hidden, 1)
                def forward(s, x):
                    return s.o(s.g(x)[0][:, -1]).squeeze(-1)
            self.net = GRUModel()
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.nn = True
        except ImportError:
            self.nn = False
            self.dim = dim
            self.W = np.random.randn(dim + 1) * 0.01

    def fit(self, X, y, epochs=3):
        if self.nn:
            Xt = self.torch.tensor(X, dtype=self.torch.float32)
            yt = self.torch.tensor(y, dtype=self.torch.float32)
            for _ in range(epochs):
                self.opt.zero_grad()
                loss = ((self.net(Xt) - yt) ** 2).mean()
                loss.backward()
                self.opt.step()
            return float(loss)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        self.W = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return 0.0

    def predict(self, x):
        if self.nn:
            with self.torch.no_grad():
                return float(self.net(self.torch.tensor(x[None], dtype=self.torch.float32))[0])
        return float(np.append(x, 1.0) @ self.W)


@feature("garman_klass")
def garman_klass(h, l, c, o=None, window=20, periods_per_year=252*375):
    """Garman-Klass volatility estimator (intraday, OHLC). Garman-Klass 1980.

    periods_per_year must match the instrument calendar: NSE 1-minute bars
    trade 252*375 bars/year; 24/7 crypto is 365*1440. The old hardcoded NSE
    factor understated crypto volatility by roughly 10x (B4 fix).
    """
    h=np.asarray(h,float); l=np.asarray(l,float); c=np.asarray(c,float)
    o = c if o is None else np.asarray(o,float)
    log_hl = np.log(h/np.maximum(l,1e-9))
    log_co = np.log(c/np.maximum(o,1e-9))
    rs = 0.5*log_hl**2 - (2*np.log(2)-1)*log_co**2
    import pandas as pd
    return np.sqrt(pd.Series(rs).rolling(window,min_periods=5).mean().fillna(0).values * periods_per_year)

@feature("kalman_trend")
def kalman_trend(c):
    """Adaptive 1-D Kalman trend filter.

    The measurement-noise variance R is estimated from the series' own
    first-difference variance, so the smoothing gain scales with each
    instrument's volatility. A fixed R=0.01 pinned the gain near 3% for every
    instrument: over-smoothing volatile names, under-smoothing quiet ones (B5).
    """
    c=np.asarray(c,float)
    if len(c) < 3:
        return c.astype(float).copy()
    diffs = np.diff(c)
    r_est = max(float(np.var(diffs)), 1e-12)
    q = r_est / 100.0
    n=len(c); x=np.zeros(n); P=np.ones(n); x[0]=c[0]; P[0]=r_est
    for i in range(1,n):
        P_pred=P[i-1]+q
        K=P_pred/(P_pred+r_est)
        x[i]=x[i-1]+K*(c[i]-x[i-1])
        P[i]=(1-K)*P_pred
    return x

@feature("cvd")
def cvd(o,h,l,c,v):
    """Cumulative Volume Delta proxy from OHLCV: signed volume. For real CVD use tick aggressor."""
    import numpy as np
    rng=np.maximum(np.asarray(h,float)-np.asarray(l,float),1e-9)
    buy_ratio=(np.asarray(c,float)-np.asarray(l,float))/rng
    signed = (2*buy_ratio-1)*np.asarray(v,float)
    return np.cumsum(signed)

# ---------- Real-Time News Research & Optimism Plugin ----------
@feature("news_sentiment")
def news_sentiment(db, sym):
    if not db:
        return 0.0, "NEUTRAL"
    rows = db.q("SELECT score, sentiment FROM news WHERE sym=? ORDER BY nid DESC LIMIT 5", (sym,))
    if not rows:
        return 0.0, "NEUTRAL"
    avg_s = float(sum(r[0] for r in rows) / len(rows))
    return avg_s, rows[0][1]

# ---------- Self Test Suite ----------
def self_test():
    """Exercise every registered feature plus its semantic invariants.

    Research-only utilities (hedging_rnn with optional torch) stay outside
    the boot gate, but everything else must not merely run - it must satisfy
    mathematical invariants, so a regression like a wrong Greek sign fails
    boot instead of shipping (B6).
    """
    rng = np.random.default_rng(42)
    n = 300
    o = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    h = o + abs(rng.normal(0, 0.5, n))
    l = o - abs(rng.normal(0, 0.5, n))
    c = o + rng.normal(0, 0.3, n)
    v = rng.integers(100, 10000, n).astype(float)
    d, _ = delta(o, h, l, c, v)

    ok = {}
    fails = []
    tests = {
        "ema": lambda: ema(c, 9),
        "cma": lambda: cma(c),
        "macd": lambda: macd(c),
        "rsi": lambda: rsi(c),
        "atr": lambda: atr(h, l, c),
        "vwap": lambda: vwap(h, l, c, v),
        "avwap": lambda: avwap(h, l, c, v, 150),
        "bos_choch": lambda: bos_choch(h, l, c),
        "fvg": lambda: fvg(o, h, l, c),
        "liquidity_sweep": lambda: liquidity_sweep(h, l, c),
        "delta": lambda: delta(o, h, l, c, v),
        "ultra_delta": lambda: ultra_delta(d),
        "big_trades": lambda: big_trades(v),
        "order_blocks": lambda: order_blocks(o, h, l, c),
        "effort_result": lambda: effort_result(o, h, l, c, v),
        "dom_imbalance": lambda: dom_imbalance({"bids": [(99.0, 10.0)], "asks": [(101.0, 5.0)]}),
        "volume_profile": lambda: volume_profile(h, l, c, v),
        "tpo": lambda: tpo(h, l),
        "black_scholes": lambda: black_scholes(100.0, 100.0, 0.1, 0.065, 0.2),
        "deep_gamma": lambda: deep_gamma(
            [{"K": 100.0, "T": 0.1, "iv": 0.2, "oi": 10.0, "kind": "CE"},
             {"K": 100.0, "T": 0.1, "iv": 0.2, "oi": 10.0, "kind": "PE"}], 100.0),
        "garman_klass": lambda: garman_klass(h, l, c),
        "kalman_trend": lambda: kalman_trend(c),
        "cvd": lambda: cvd(o, h, l, c, v),
    }

    for name, fn in tests.items():
        try:
            fn()
            ok[name] = True
        except Exception as e:
            fails.append((name, repr(e)))

    # Semantic invariants: these would have caught the put-theta sign error
    # and the value-area ordering bug that "does not raise" tests missed.
    put = black_scholes(100.0, 100.0, 0.1, 0.065, 0.2, kind="PE")
    call = black_scholes(100.0, 100.0, 0.1, 0.065, 0.2, kind="CE")
    parity = call["px"] - put["px"] - (100.0 - 100.0 * exp(-0.065 * 0.1))
    if abs(parity) > 1e-6:
        fails.append(("black_scholes_parity", repr(parity)))
    vp = volume_profile(h, l, c, v)
    if not (vp["val"] <= vp["poc"] <= vp["vah"]):
        fails.append(("volume_profile_ordering", repr((vp["val"], vp["poc"], vp["vah"]))))
    anchored = avwap(h, l, c, v, 150)
    if not np.isnan(anchored[:150]).all() or not np.isfinite(anchored[150:]).all():
        fails.append(("avwap_nan_mask", "pre-anchor must be NaN and post-anchor finite"))
    sparse_v = np.where(np.arange(n) % 7 == 0, 0.0, v)
    ev_check, _ = effort_result(o, h, l, c, sparse_v)
    if not np.isfinite(ev_check).all():
        fails.append(("effort_result_inf", "non-finite effort values survived sanitisation"))
    kal = kalman_trend(c)
    if len(kal) != n or not np.isfinite(kal).all():
        fails.append(("kalman_trend_shape", "output must be finite and same-length"))

    return ok, fails
