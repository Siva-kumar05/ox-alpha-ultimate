"""MARAL — Strategy Execution Intelligence cockpit.

Sits ABOVE the strategy. The strategy gives the setup; MARAL checks whether
execution deserves permission. Directly addresses the image's failure modes:
early entry, late entry, trap entry, FOMO, weak confirmation, poor location,
no risk space, no discipline after entry.

Each check returns a permission verdict + human-readable reason.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import REG

class MaralCockpit:
    def __init__(self, cfg):
        self.cfg = cfg

    def check(self, df: pd.DataFrame, orderflow=None, atr: np.ndarray | None = None) -> dict:
        if df is None or len(df) < 30:
            return {"permission": "WAIT", "reason": "INSUFFICIENT_DATA", "details": {}}
        h = df["h"].to_numpy(float); l = df["l"].to_numpy(float)
        c = df["c"].to_numpy(float); o = df["o"].to_numpy(float); v = df["v"].to_numpy(float)
        last = c[-1]
        # Trap risk: recent sweep without confirmation
        sweep = REG["liquidity_sweep"](h, l, c)[-1]
        trap_risk = abs(float(sweep)) > 0.5
        # Candle quality: body vs range
        body = abs(c[-1] - o[-1]); rng = max(h[-1] - l[-1], 1e-9)
        weak_candle = body / rng < 0.25
        # Momentum expanding vs fading
        delta, _ = REG["delta"](o, h, l, c, v)
        ultra = float(REG["ultra_delta"](delta)[-1])
        fading = abs(ultra) < 0.3
        # Risk space: ATR distance to recent swing
        if atr is not None and len(atr) and np.isfinite(atr[-1]):
            risk_atr = float(atr[-1])
        else:
            risk_atr = float(REG["atr"](h, l, c)[-1])
        recent_low = float(np.min(l[-10:])); risk_space = (last - recent_low) / max(risk_atr, 1e-9)
        poor_risk = risk_space < 0.8 or risk_space > 4.0
        # Market context: order-flow support
        context_ok = True
        if orderflow is not None:
            context_ok = bool(getattr(orderflow, "long_entry", False) or getattr(orderflow, "ready", False))
        # Early entry: signal at extreme wick without confirmation
        early = bool(trap_risk and weak_candle)
        # Late entry: price already extended >1.5 ATR from last swing
        late = False
        try:
            from .brain import _session_vwap  # noqa
            # Approximate: last close far from 20-bar EMA
            ema20 = float(REG["ema"](c, 20)[-1])
            late = abs(last - ema20) > risk_atr * 1.5
        except Exception:
            pass
        # Decision
        details = {"trap_risk": trap_risk, "weak_candle": weak_candle, "fading": fading,
                   "poor_risk": poor_risk, "context_ok": context_ok, "early": early, "late": late,
                   "risk_atr": round(risk_atr, 2), "ultra_delta": round(ultra, 3)}
        if early:
            return {"permission": "AVOID", "reason": "EARLY_ENTRY_TRAP", "details": details}
        if late:
            return {"permission": "WAIT", "reason": "LATE_ENTRY_EXTENDED", "details": details}
        if trap_risk and not context_ok:
            return {"permission": "AVOID", "reason": "TRAP_RISK", "details": details}
        if weak_candle and fading:
            return {"permission": "WAIT", "reason": "WEAK_CONFIRMATION", "details": details}
        if poor_risk:
            return {"permission": "MANAGE_CAREFULLY", "reason": "POOR_RISK_SPACE", "details": details}
        if not context_ok:
            return {"permission": "WAIT", "reason": "CONTEXT_NOT_SUPPORTIVE", "details": details}
        return {"permission": "EXECUTE", "reason": "ALL_CLEAR", "details": details}
