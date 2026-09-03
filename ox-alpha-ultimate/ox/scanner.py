"""Market scanner for 500-1000 NSE stocks.
Ranks symbols by volatility, volume, momentum, and order-flow score.
Focuses on small/mid-cap stocks for low-capital efficiency.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

SMALLCAP_PRICE_RANGE = (20.0, 500.0)
MIN_AVG_VOLUME = 500
MIN_VOLATILITY_PCT = 0.5

class MarketScanner:
    def __init__(self, cfg, db, broker):
        self.cfg = cfg
        self.db = db
        self.broker = broker

    def scan(self, universe: list[str], top_k: int = 20) -> list[dict]:
        """Score each symbol and return top_k ranked for trading."""
        scored = []
        for sym in universe[:1000]:
            frame = self._get_frame(sym)
            if frame is None or len(frame) < 30:
                continue
            c = frame["c"].to_numpy(float)
            v = frame["v"].to_numpy(float)
            h = frame["h"].to_numpy(float)
            low = frame["l"].to_numpy(float)
            last_price = float(c[-1])
            # Small-cap filter: focus on low-price, high-opportunity stocks
            if not (SMALLCAP_PRICE_RANGE[0] <= last_price <= SMALLCAP_PRICE_RANGE[1]):
                # Don't reject outright, just penalize large caps for low capital
                cap_penalty = 0.5 if last_price > 1500 else 0.0
            else:
                cap_penalty = 0.0
            avg_vol = float(np.mean(v[-20:]))
            if avg_vol < MIN_AVG_VOLUME:
                continue
            volatility = float(np.std(np.diff(c[-20:]) / np.maximum(c[-20:-1], 1.0)) * 100)
            if volatility < MIN_VOLATILITY_PCT * 0.5:
                continue
            momentum = (c[-1] / max(c[-20], 1.0) - 1.0) * 100
            range_pct = (max(h[-20:]) - min(low[-20:])) / max(float(c[-1]), 1.0) * 100
            score = volatility * 0.35 + range_pct * 0.25 + abs(momentum) * 0.2 + (avg_vol / 10000) * 0.2 - cap_penalty
            scored.append({"symbol": sym, "score": round(score, 4), "last_price": last_price,
                           "avg_volume": int(avg_vol), "volatility": round(volatility, 2),
                           "momentum": round(momentum, 2), "range_pct": round(range_pct, 2)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _get_frame(self, sym: str):
        try:
            rows = self.db.q("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts DESC LIMIT 50", (sym,))
            if not rows:
                # No local data; try broker history for 5 days
                try:
                    hist = self.broker.hist(sym, 5, 5)
                    if hist:
                        return pd.DataFrame(hist, columns=["ts","o","h","l","c","v"])
                except Exception:
                    pass
                return None
            return pd.DataFrame(rows[::-1], columns=["ts","o","h","l","c","v"])
        except Exception:
            return None

    def small_cap_universe(self, all_symbols: list[str]) -> list[str]:
        """Return symbols identified as small/mid-cap candidates."""
        ranked = self.scan(all_symbols, top_k=len(all_symbols))
        small = [r["symbol"] for r in ranked if SMALLCAP_PRICE_RANGE[0] <= r["last_price"] <= SMALLCAP_PRICE_RANGE[1]]
        return small if small else [r["symbol"] for r in ranked[:20]]
