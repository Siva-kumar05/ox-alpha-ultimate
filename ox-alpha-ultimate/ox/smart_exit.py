"""Smart exit engine: forecast-based hold vs. immediate loss cut.

- Holds winners beyond 10-20% if Kalman-trend + VWAP forecast stays bullish.
- Cuts losers immediately when price falls below stop or forecast turns bearish.
- Late entry: if price already ran > 1.5*ATR from signal bar, skip entry.
- Early entry: if signal fires but order-flow streak < 2, defer one bar.
"""
from __future__ import annotations
from .features import REG

class SmartExit:
    def __init__(self, cfg):
        self.cfg = cfg

    def forecast_bullish(self, df, horizon: int = 5) -> bool:
        try:
            c = df["c"].to_numpy(float)
            trend = REG["kalman_trend"](c)
            # Forecast: linear extrapolate last 5 kalman slope
            slope = float(trend[-1] - trend[-5]) / 5 if len(trend) >= 5 else 0.0
            forecast = float(trend[-1] + slope * horizon)
            # Session VWAP as anchor
            h = df["h"].to_numpy(float); l = df["l"].to_numpy(float); v = df["v"].to_numpy(float)
            svwap = float(REG["vwap"](h, l, c, v)[-1]) if len(c) > 5 else float(c[-1])
            return forecast > svwap and slope > 0
        except Exception:
            return False

    def should_hold(self, df, entry_price: float, current_price: float, unrealized_pct: float) -> bool:
        # Don't sell at +10-20% if forecast still bullish and not at target
        if unrealized_pct >= 0.08:
            if self.forecast_bullish(df):
                return True
        return False

    def should_cut(self, df, current_price: float, stop_price: float, entry_price: float) -> bool:
        # Immediate loss cut when below stop
        if current_price <= stop_price:
            return True
        # Bearish forecast while underwater cuts early
        try:
            if current_price < entry_price and not self.forecast_bullish(df):
                return True
        except Exception:
            pass
        return False

    def is_late_entry(self, df, signal_idx: int, current_price: float) -> bool:
        try:
            h = df["h"].to_numpy(float); l = df["l"].to_numpy(float); c = df["c"].to_numpy(float)
            atr = float(REG["atr"](h, l, c)[-1])
            signal_close = float(df["c"].iloc[signal_idx])
            return abs(current_price - signal_close) > atr * 1.5
        except Exception:
            return False
