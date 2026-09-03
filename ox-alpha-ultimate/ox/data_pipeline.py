"""Multi-source training data pipeline.
Aggregates Dhan OHLCV, TradingView signals, news sentiment, and order-flow
snapshots into unified frames for the Brain. Also handles small-cap universe
expansion and financial dataset merging.
"""
from __future__ import annotations
import pandas as pd
from .scanner import MarketScanner

class DataPipeline:
    def __init__(self, cfg, db, broker):
        self.cfg = cfg
        self.db = db
        self.broker = broker
        self.scanner = MarketScanner(cfg, db, broker)

    def build_training_frames(self, symbols: list[str], lookback_days: int | None = None):
        """Pull candles for all symbols, return dict symbol->DataFrame."""
        frames = {}
        for sym in symbols:
            try:
                from .core import DB  # noqa
                rows = self.db.q("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts", (sym,))
                if rows:
                    frames[sym] = pd.DataFrame(rows, columns=["ts","o","h","l","c","v"])
                else:
                    # Try broker history as fallback (for new symbols from scanner)
                    tf = int(self.cfg.get("timeframe_sec", 60) // 60)
                    days = lookback_days or int(self.cfg.get("history_days", 95))
                    hist = self.broker.hist(sym, tf, days)
                    if hist:
                        frames[sym] = pd.DataFrame(hist, columns=["ts","o","h","l","c","v"])
            except Exception:
                continue
        return frames

    def expanded_universe(self, base_symbols: list[str], scan_top: int = 20) -> list[str]:
        """Blend configured symbols with scanner-discovered small-caps."""
        # Try to discover from DB-known symbols
        known = [r[0] for r in self.db.q("SELECT DISTINCT sym FROM candles LIMIT 1000")]
        candidates = list(set(base_symbols + known))
        try:
            ranked = self.scanner.scan(candidates, top_k=scan_top)
            discovered = [r["symbol"] for r in ranked if r["symbol"] not in base_symbols]
            return base_symbols + discovered[:max(0, 1000 - len(base_symbols))][:10]
        except Exception:
            return base_symbols
