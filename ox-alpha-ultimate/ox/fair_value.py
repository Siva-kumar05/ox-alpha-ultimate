"""Fair value calculation engine.
Combines VWAP, anchored VWAP, volume profile POC, Kalman trend, and order-flow
microprice to estimate a symbol's fair value. Used for mispricing flags.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import REG

class FairValueEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate(self, df: pd.DataFrame, orderflow=None) -> dict:
        if df is None or len(df) < 20:
            return {"fair_value": 0.0, "confidence": 0.0, "components": {}}
        h = df["h"].to_numpy(float); l = df["l"].to_numpy(float)
        c = df["c"].to_numpy(float); v = df["v"].to_numpy(float)
        # Session VWAP
        try:
            from .brain import _session_vwap
            svwap = float(_session_vwap(df, h, l, c, v)[-1])
        except Exception:
            svwap = float(np.mean(c[-20:]))
        kalman = float(REG["kalman_trend"](c)[-1])
        poc = 0.0
        try:
            vp = REG["volume_profile"](h, l, c, v)
            poc = float(vp.get("poc", c[-1]))
        except Exception:
            poc = float(c[-1])
        # Order-flow microprice if available
        microprice = 0.0
        if orderflow is not None and getattr(orderflow, "microprice", 0):
            microprice = float(orderflow.microprice)
        last = float(c[-1])
        # Weighted blend
        fair = svwap * 0.35 + kalman * 0.30 + poc * 0.20 + (microprice if microprice else last) * 0.15
        components = {"svwap": svwap, "kalman": kalman, "poc": poc, "microprice": microprice, "last": last}
        # Confidence based on component agreement
        vals = [svwap, kalman, poc]
        spread = max(vals) - min(vals) if vals else 0
        confidence = max(0.0, 1.0 - spread / max(abs(fair), 1.0) * 5)
        return {"fair_value": round(fair, 2), "confidence": round(confidence, 4), "components": components}
