
"""RESEARCH-ONLY SCAFFOLD - NOT WIRED TO THE LIVE PATH.

This module is unreachable from config.yaml on purpose: ox/core.py validates
platform to paper|dhan only. It does not implement the BrokerBase order
surface (no place_super_order/positions/hist), so it cannot back the OMS.
Do not trade live capital through anything built on this file until it
passes the same fail-closed review as the Dhan adapter.

Crypto micro-scalping adapter (Binance spot) + tick scalping engine.

This module is intentionally separate from NSE equity flow. It supports
fractional balances starting from $0.9 USDT in paper mode by simulating
sub-min-notional until the venue minimum is reached, then enforces
venue minima. No leverage, no futures, no withdrawal surface.

Design mirrors NSE adapter interface (BrokerBase) so Agent can select
via config platforms.supported and runtime switching.
"""
from __future__ import annotations
import math, time, threading, random
from dataclasses import dataclass
import requests

class CryptoMicroBroker:
    """Paper + live Binance spot (ccxt optional). Defaults to paper."""
    name = "crypto_paper"
    def __init__(self, cfg, db):
        self.cfg = cfg; self.db=db
        self.usdt = float(cfg.get("crypto",{}).get("paper_start_usdt",0.9))
        self.min_notional = float(cfg.get("crypto",{}).get("min_notional_usdt",5.0))
        self.fee_bps = float(cfg.get("crypto",{}).get("fees_bps",10.0))/10000
        self.slip_bps = float(cfg.get("crypto",{}).get("slippage_bps",5.0))/10000
        self.prices = {"BTCUSDT": 68000, "ETHUSDT": 3200, "SOLUSDT": 150}
        self.pos = {}  # sym -> {qty, avg}
        self._lock=threading.RLock()
        self._rnd=random.Random(42)
    def login(self): return True
    def ltp(self,sym): 
        # in paper, random walk; in live would call ccxt fetch_ticker
        with self._lock:
            p=self.prices.get(sym,100)
            p=max(0.01, p + self._rnd.gauss(0,p*0.0008))
            self.prices[sym]=p
            return p
    def place_market(self,sym,side,qty):
        price=self.ltp(sym)
        notional=qty*price
        # The order surface enforces the venue minimum unconditionally; the
        # below-minimum fractional start is simulated only through
        # simulate_compound(), never through an order that a live venue
        # would reject.
        if notional < self.min_notional:
            raise ValueError(
                f"order notional {notional:.2f} USDT is below venue minimum {self.min_notional:.2f}"
            )
        fee=qty*price*self.fee_bps
        slip=price*self.slip_bps
        fill=price*(1+ (slip/price) if side=="BUY" else -(slip/price))
        cost=qty*fill + fee if side=="BUY" else -qty*fill + fee
        return {"order_id": f"CR{int(time.time()*1000)}", "price":fill, "qty":qty, "fee":fee, "side":side}
    def simulate_compound(self, trades=1000, win_rate=0.55, rr=1.0):
        """Illustrative compounding from paper_start; NOT a profit promise."""
        bal=self.usdt; notional=self.min_notional
        # below-minimum fractional mode
        for i in range(trades):
            if bal < self.min_notional:
                risk=bal*0.02
                qty=risk / self.prices["BTCUSDT"]
            else:
                qty=min(bal*0.02, notional)/self.prices["BTCUSDT"]
            win=self._rnd.random() < win_rate
            pnl = qty*self.prices["BTCUSDT"]*0.005*rr if win else -qty*self.prices["BTCUSDT"]*0.005
            pnl -= qty*self.prices["BTCUSDT"]*self.fee_bps
            bal=max(0, bal+pnl)
            if bal> notional*2:
                notional=min(notional*1.05, 50)
        return bal
