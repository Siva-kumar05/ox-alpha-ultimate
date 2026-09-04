"""10x Multi-Timeframe Analysis: Derive higher-timeframe context from 1m candles."""
from __future__ import annotations
import pandas as pd


class MultiTimeframeAnalyzer:
    """Derive 5m, 15m candles from 1m history for trend alignment."""

    def __init__(self, cfg=None):
        mtf_cfg = (cfg or {}).get("multi_timeframe", {})
        self.enabled = mtf_cfg.get("enabled", True)
        self.timeframes = mtf_cfg.get("timeframes", [5, 15])
        self.alignment_threshold = mtf_cfg.get("alignment_threshold", 0.6)
        self.ema_fast = mtf_cfg.get("ema_fast", 9)
        self.ema_slow = mtf_cfg.get("ema_slow", 21)

    def aggregate(self, df_1m, timeframe):
        if df_1m is None or df_1m.empty or timeframe <= 1:
            return df_1m if df_1m is not None else pd.DataFrame()
        df = df_1m.copy()
        df["Timestamp"] = pd.to_datetime(df["ts"], unit="s")
        df = df.set_index("Timestamp")
        agg = df.resample(str(timeframe) + "min").agg({
            "ts": "last", "o": "first", "h": "max",
            "l": "min", "c": "last", "v": "sum"
        }).dropna(subset=["o", "h", "l", "c"])
        return agg.reset_index(drop=True)

    def trend_direction(self, df):
        if df is None or len(df) < self.ema_slow + 5:
            return 0
        c = df["c"].to_numpy(dtype=float)
        fast = pd.Series(c).ewm(span=self.ema_fast).mean().values
        slow = pd.Series(c).ewm(span=self.ema_slow).mean().values
        if fast[-1] > slow[-1] and c[-1] > slow[-1]:
            return 1
        elif fast[-1] < slow[-1] and c[-1] < slow[-1]:
            return -1
        return 0

    def alignment_score(self, df_1m):
        if not self.enabled or df_1m is None:
            return {"score": 0.5, "details": {}, "aligned": True}
        directions = {}
        for tf in self.timeframes:
            htf = self.aggregate(df_1m, tf)
            directions[str(tf) + "m"] = self.trend_direction(htf)
        base = self.trend_direction(df_1m)
        if base == 0:
            score = 0.5
        else:
            agreeing = sum(1 for d in directions.values() if d == base)
            score = (agreeing + 1) / (len(directions) + 1)
        return {
            "score": round(score, 4),
            "details": directions,
            "aligned": score >= self.alignment_threshold,
            "base_direction": base,
        }
