"""10x Cross-Asset Signals: Index trend, VIX overlay, relative strength, breadth."""
from __future__ import annotations
import numpy as np
import pandas as pd


class CrossAssetAnalyzer:
    """Derive market-wide context from index and VIX signals."""

    def __init__(self, cfg=None):
        xcfg = (cfg or {}).get("cross_asset", {})
        self.enabled = xcfg.get("enabled", True)
        self.nifty_filter = xcfg.get("nifty_trend_filter", True)
        self.vix_overlay = xcfg.get("vix_overlay", True)
        self.vix_high = float(xcfg.get("vix_high_threshold", 20))
        self.rs_enabled = xcfg.get("relative_strength", True)
        self.min_rs_rank = int(xcfg.get("min_rs_rank", 3))

    def nifty_trend(self, nifty_frame):
        if nifty_frame is None or len(nifty_frame) < 50:
            return {"direction": 0, "strength": 0.0, "available": False}
        c = nifty_frame["c"].to_numpy(dtype=float)
        ema20 = pd.Series(c).ewm(span=20).mean().values
        ema50 = pd.Series(c).ewm(span=50).mean().values
        if c[-1] > ema20[-1] > ema50[-1]:
            return {"direction": 1,
                    "strength": (c[-1] / ema50[-1] - 1) * 100,
                    "available": True}
        elif c[-1] < ema20[-1] < ema50[-1]:
            return {"direction": -1,
                    "strength": (ema50[-1] / c[-1] - 1) * 100,
                    "available": True}
        return {"direction": 0, "strength": 0.0, "available": True}

    def relative_strength_rank(self, symbol_frames, lookback=20):
        returns = {}
        for sym, frame in symbol_frames.items():
            if frame is None or len(frame) < lookback:
                continue
            c = frame["c"].to_numpy(dtype=float)
            ret = ((c[-1] / c[-lookback] - 1) * 100
                   if c[-lookback] > 0 else 0)
            returns[sym] = ret
        if not returns:
            return {}
        sorted_syms = sorted(returns, key=returns.get, reverse=True)
        return {sym: rank + 1
                for rank, sym in enumerate(sorted_syms)}

    def breadth(self, symbol_frames, sma_period=20):
        above = 0
        total = 0
        for sym, frame in symbol_frames.items():
            if frame is None or len(frame) < sma_period:
                continue
            c = frame["c"].to_numpy(dtype=float)
            sma = np.mean(c[-sma_period:])
            total += 1
            if c[-1] > sma:
                above += 1
        return above / max(total, 1)
