
"""RESEARCH-ONLY LIBRARY - NOT WIRED TO THE AGENT LOOP.

No code path constructs ScalpingEngine yet; config.yaml has no supported
key to enable it. Wiring it requires a tick-cadence decision loop fed by
the depth websocket, not the 3-second REST poll.

Tick-level scalping engine (NSE + crypto).

Uses L2 book imbalance + CVD + microprice edge on 1s cadence.
Fail-closed: any stale book, wide spread, or thin notional blocks entry.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ScalpSignal:
    action: str  # BUY/SELL/HOLD
    confidence: float
    stop: float
    target: float
    reason: str

class ScalpingEngine:
    def __init__(self, cfg):
        self.cfg=cfg
        s=cfg.get("scalping",{})
        self.stop_pct=float(s.get("stop_pct_fast",0.003))
        self.target_pct=float(s.get("target_pct_fast",0.005))
        self.trailing_pct=float(s.get("trailing_pct",0.0015))
        self.max_hold=int(s.get("max_hold_seconds",180))
    def evaluate(self, ltp: float, book_imbalance: float, cvd_slope: float, flow_ready: bool, trend_vote: float):
        if not flow_ready:
            return ScalpSignal("HOLD",0,ltp,ltp,"STALE_BOOK")
        # buy scalps require: strong bid imbalance + rising CVD + trend not bearish
        if book_imbalance>0.12 and cvd_slope>0 and trend_vote>=0:
            stop=ltp*(1-self.stop_pct); tgt=ltp*(1+self.target_pct)
            conf=min(1.0, book_imbalance*2 + cvd_slope)
            return ScalpSignal("BUY",conf,stop,tgt,"SCALP_LONG")
        if book_imbalance<-0.12 and cvd_slope<0 and trend_vote<=0:
            stop=ltp*(1+self.stop_pct); tgt=ltp*(1-self.target_pct)
            # long-only NSE: SELL here means exit only, not naked short
            return ScalpSignal("SELL",abs(book_imbalance),stop,tgt,"SCALP_EXIT")
        return ScalpSignal("HOLD",0,ltp,ltp,"NO_EDGE")
