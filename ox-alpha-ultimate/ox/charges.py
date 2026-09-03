from __future__ import annotations
from dataclasses import dataclass

import numpy as np

class ChargesCalculator:
    """
    Indian Market Transaction Fee & Tax Calculator for Intraday Equities (MIS).
    Calculates exact Brokerage, STT, Exchange Txn Fees, GST, SEBI Turnover Fee, and Stamp Duty.
    Enforces breakeven sell price threshold so selling price covers investment + all charges.
    """
    def __init__(self, costs_cfg=None):
        c = costs_cfg or {}
        self.flat_brokerage = c.get("brokerage_per_order", 20.0)
        self.brokerage_pct = 0.0003  # 0.03% max cap
        self.stt_pct = c.get("stt_pct", 0.025) / 100.0          # 0.025% on sell side
        self.txn_charge_pct = c.get("txn_charge_pct", 0.00297) / 100.0  # 0.00297% NSE
        self.gst_pct = c.get("gst_pct", 18.0) / 100.0           # 18% on brokerage + txn fee
        self.sebi_fee_pct = c.get("sebi_fee_pct", 0.0001) / 100.0 # 0.0001%
        self.stamp_duty_pct = c.get("stamp_duty_pct", 0.003) / 100.0 # 0.003% on buy side

    def compute_charges(self, buy_price: float, sell_price: float, qty: int) -> dict:
        if qty <= 0 or buy_price <= 0 or sell_price <= 0:
            return {"total_charges": 0.0, "net_pnl": 0.0}

        buy_turnover = buy_price * qty
        sell_turnover = sell_price * qty
        total_turnover = buy_turnover + sell_turnover

        # Brokerage per leg
        buy_brokerage = min(self.flat_brokerage, buy_turnover * self.brokerage_pct)
        sell_brokerage = min(self.flat_brokerage, sell_turnover * self.brokerage_pct)
        total_brokerage = buy_brokerage + sell_brokerage

        # STT (only on sell side for intraday equity)
        stt = sell_turnover * self.stt_pct

        # Exchange Transaction Charge
        txn_charge = total_turnover * self.txn_charge_pct

        # GST (18% on Brokerage + Txn Charge)
        gst = (total_brokerage + txn_charge) * self.gst_pct

        # SEBI Fee
        sebi_fee = total_turnover * self.sebi_fee_pct

        # Stamp Duty (only on buy side)
        stamp_duty = buy_turnover * self.stamp_duty_pct

        total_charges = total_brokerage + stt + txn_charge + gst + sebi_fee + stamp_duty
        gross_pnl = (sell_price - buy_price) * qty
        net_pnl = gross_pnl - total_charges

        return {
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "brokerage": total_brokerage,
            "stt": stt,
            "txn_charge": txn_charge,
            "gst": gst,
            "sebi_fee": sebi_fee,
            "stamp_duty": stamp_duty,
            "total_charges": round(total_charges, 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2)
        }

    def min_breakeven_sell_price(self, buy_price: float, qty: int, buffer_pct: float = 0.001) -> float:
        """
        Calculates the exact minimum target sell price to cover buy cost + all broker charges & taxes
        plus an optional small profit buffer.
        Guarantees that selling at or above this price generates zero net loss.
        """
        if qty <= 0 or buy_price <= 0:
            return buy_price

        # Iterative search / solver for exact target price
        target_price = buy_price * (1.0 + buffer_pct)
        for _ in range(20):
            chg = self.compute_charges(buy_price, target_price, qty)
            if chg["net_pnl"] >= 0.0:
                break
            # Add price delta required to cover deficit
            deficit = -chg["net_pnl"]
            target_price += (deficit / qty) + 0.05

        return round(target_price, 2)


@dataclass
class SlippageEstimate:
    """Dynamic slippage estimate based on market microstructure."""
    base_slippage_bps: float
    spread_cost_bps: float
    market_impact_bps: float
    total_slippage_bps: float
    total_slippage_pct: float


class DynamicSlippageModel:
    """Dynamic slippage model based on spread, depth, order size, and volatility."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        slippage_cfg = cfg.get("slippage_model", {})
        self.base_slippage_bps = slippage_cfg.get("base_slippage_bps", 3.0)
        self.spread_weight = slippage_cfg.get("spread_weight", 0.5)
        self.depth_weight = slippage_cfg.get("depth_weight", 0.3)
        self.volume_weight = slippage_cfg.get("volume_weight", 0.2)
        self.volatility_weight = slippage_cfg.get("volatility_weight", 0.1)
        self.max_slippage_bps = slippage_cfg.get("max_slippage_bps", 50.0)
        self.min_slippage_bps = slippage_cfg.get("min_slippage_bps", 0.5)
        
    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        bid: float,
        ask: float,
        bid_depth: float,
        ask_depth: float,
        adv: float,  # Average daily volume
        volatility: float
    ) -> SlippageEstimate:
        """Estimate slippage based on market microstructure."""
        
        # Spread cost (half spread for market orders)
        spread_bps = (ask - bid) / max((ask + bid) / 2, 1e-9) * 10000
        spread_cost = spread_bps * self.spread_weight * 0.5
        
        # Market impact using square-root law
        participation = min(quantity * price / max(adv, 1), 0.25)
        market_impact = volatility * 100 * np.sqrt(participation) * self.volume_weight
        
        # Depth impact - larger orders relative to depth cause more slippage
        relevant_depth = bid_depth if side == "SELL" else ask_depth
        depth_ratio = quantity * price / max(relevant_depth, 1)
        depth_impact = min(depth_ratio * 100, 20) * self.depth_weight
        
        # Volatility impact
        vol_impact = volatility * 100 * self.volatility_weight
        
        # Total slippage
        total_bps = self.base_slippage_bps + spread_cost + market_impact + depth_impact + vol_impact
        total_bps = float(np.clip(total_bps, self.min_slippage_bps, self.max_slippage_bps))
        
        return SlippageEstimate(
            base_slippage_bps=self.base_slippage_bps,
            spread_cost_bps=spread_cost,
            market_impact_bps=market_impact,
            total_slippage_bps=total_bps,
            total_slippage_pct=total_bps / 10000.0
        )
    
    def get_slippage_price(self, side: str, price: float, slippage: SlippageEstimate) -> float:
        """Calculate execution price with slippage."""
        if side == "BUY":
            return price * (1 + slippage.total_slippage_pct)
        else:
            return price * (1 - slippage.total_slippage_pct)
