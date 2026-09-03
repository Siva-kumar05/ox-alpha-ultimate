"""Advanced Execution Algorithms
====================================
Implementation of TWAP, VWAP, Implementation Shortfall (Arrival Price),
and Almgren-Chriss optimal execution from TCAPY/StockSharp patterns.

Features:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price) 
- Implementation Shortfall (Arrival Price)
- Almgren-Chriss Optimal Execution
- POV (Percentage of Volume)
- Iceberg / Slicing
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import time

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .core import LOG, iso


@dataclass
class ExecutionSlice:
    """Single execution slice."""
    timestamp: int
    quantity: int
    price_limit: float
    urgency: str  # "aggressive", "normal", "passive"
    order_type: str = "LIMIT"  # MARKET, LIMIT, ICEBERG
    slice_id: str = ""


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    symbol: str
    total_quantity: int
    side: str  # BUY/SELL
    slices: List[ExecutionSlice]
    benchmark: str  # "TWAP", "VWAP", "ARRIVAL", "CLOSE"
    start_time: int
    end_time: int
    risk_aversion: float = 1e-6


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""
    
    @abstractmethod
    def generate_slices(self, plan: ExecutionPlan) -> List[ExecutionSlice]:
        """Generate execution slices."""
        pass
    
    @abstractmethod
    def update_state(self, fill: dict, market_state: dict) -> None:
        """Update internal state after fill."""
        pass


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution.
    
    Slices order evenly across time horizon.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.config = config
        self.n_slices = config.get("n_slices", 10)
        self.randomize = config.get("randomize", False)
        
    def generate_slices(self, plan: ExecutionPlan) -> List[ExecutionSlice]:
        total_qty = plan.total_quantity
        n_slices = min(self.n_slices, plan.total_quantity // 100 + 1)
        
        qty_per_slice = plan.total_quantity // n_slices
        remainder = plan.total_quantity % n_slices
        
        duration = plan.end_time - plan.start_time
        slice_duration = duration // n_slices
        
        slices = []
        for i in range(n_slices):
            qty = qty_per_slice + (1 if i < remainder else 0)
            start = plan.start_time + i * slice_duration
            end = plan.start_time + (i + 1) * slice_duration
            
            # Add small randomization to avoid detection
            if self.config.get("randomize", False):
                offset = np.random.randint(-30, 30)
                start_ts = max(plan.start_time, start + offset)
            else:
                start_ts = start
            
            slices.append(ExecutionSlice(
                timestamp=start_ts,
                quantity=qty,
                price_limit=0.0,  # Market orders for TWAP
                urgency="normal",
                order_type="MARKET",
                slice_id=f"TWAP_{plan.symbol}_{i}"
            ))
        
        return slices
    
    def update_state(self, fill: dict, market_state: dict):
        pass


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution.
    
    Slices order proportional to historical volume profile.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.config = config
        self.n_slices = config.get("n_slices", 20)
        self.use_historical_volume = config.get("use_historical_volume", True)
        
    def generate_slices(self, plan: ExecutionPlan, 
                       volume_profile: np.ndarray = None) -> List[ExecutionSlice]:
        """Generate slices based on volume profile."""
        
        if volume_profile is None or len(volume_profile) == 0:
            # Fallback to uniform
            return TWAPAlgorithm(self.config).generate_slices(plan)
        
        # Normalize volume profile
        vol_profile = np.array(volume_profile)
        vol_profile = vol_profile / vol_profile.sum()
        
        n_slices = min(self.n_slices, plan.total_quantity // 100 + 1)
        duration = plan.end_time - plan.start_time
        slice_duration = duration // n_slices
        
        slices = []
        for i in range(n_slices):
            start = plan.start_time + i * slice_duration
            end = plan.start_time + (i + 1) * slice_duration
            
            # Allocate quantity proportional to volume profile
            slice_vol_pct = vol_profile[i] if i < len(vol_profile) else 1.0 / n_slices
            qty = int(plan.total_quantity * slice_vol_pct)
            qty = max(1, qty)
            
            slices.append(ExecutionSlice(
                timestamp=start,
                quantity=qty,
                price_limit=0.0,
                urgency="normal",
                order_type="LIMIT",
                slice_id=f"VWAP_{plan.symbol}_{i}"
            ))
        
        return slices
    
    def update_state(self, fill: dict, market_state: dict):
        pass


class ArrivalPriceAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall (Arrival Price) algorithm.
    
    Minimizes implementation shortfall: difference between
    decision price and execution price.
    
    Based on Almgren-Chriss optimal execution framework.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.config = config
        self.risk_aversion = config.get("risk_aversion", 1e-6)
        self.urgency = config.get("urgency", "normal")  # low, normal, high
        
    def generate_slices(self, plan: ExecutionPlan, 
                       volatility: float = 0.01,
                       adv: float = 1_000_000) -> List[ExecutionSlice]:
        """Generate optimal slices using Almgren-Chriss model.
        
        Optimal trajectory: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
        where kappa = sqrt(lambda * sigma^2 / eta)
        """
        
        # Almgren-Chriss parameters
        sigma = 0.01  # daily volatility
        eta = 0.0001  # temporary impact
        lam = self.risk_aversion  # risk aversion
        
        kappa = np.sqrt(self.risk_aversion * 0.01**2 / 0.0001)
        T = plan.end_time - plan.start_time
        n_slices = min(20, plan.total_quantity // 100 + 1)
        
        slices = []
        total_qty = plan.total_quantity
        remaining = total_qty
        
        for i in range(n_slices):
            t = i * (plan.end_time - plan.start_time) / n_slices
            remaining_time = plan.end_time - (plan.start_time + t)
            
            if remaining_time <= 0:
                break
                
            # Optimal trajectory
            if remaining_time > 0:
                kappa_T = np.sqrt(self.risk_aversion * 0.01**2 / 0.0001) * T
                kappa_t = np.sqrt(self.risk_aversion * 0.01**2 / 0.0001) * t
                
                if np.sinh(kappa_T) > 0:
                    remaining_qty = total_qty * np.sinh(np.sqrt(self.risk_aversion * 0.01**2 / 0.0001) * remaining_time) / np.sinh(kappa_T)
                else:
                    remaining_qty = total_qty * (remaining_time / T)
            else:
                remaining_qty = 0
            
            slice_qty = int(max(0, min(remaining_qty, remaining)))
            if slice_qty <= 0:
                continue
                
            remaining -= slice_qty
            
            slices.append(ExecutionSlice(
                timestamp=int(time.time()) + i * 300,  # 5 min intervals
                quantity=slice_qty,
                price_limit=0.0,
                urgency="normal",
                order_type="LIMIT",
                slice_id=f"IS_{i}"
            ))
            
            if remaining <= 0:
                break
        
        return slices
    
    def update_state(self, fill: dict, market_state: dict):
        pass


class POVAlgorithm(ExecutionAlgorithm):
    """Percentage of Volume algorithm.
    
    Trades at specified participation rate of market volume.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.config = config
        self.pov = config.get("pov", 0.1)  # 10% of volume
        self.max_pov = config.get("max_pov", 0.25)
        
    def generate_slices(self, plan: ExecutionPlan, 
                       volume_forecast: np.ndarray = None) -> List[ExecutionSlice]:
        """Generate slices based on volume participation."""
        
        # Simplified: create slices based on expected volume
        n_slices = 20
        total_qty = plan.total_quantity
        slice_qty = max(1, int(total_qty / 20))
        
        slices = []
        for i in range(20):
            slices.append(ExecutionSlice(
                timestamp=int(time.time()) + i * 180,  # 3 min intervals
                quantity=slice_qty,
                price_limit=0.0,
                urgency="normal",
                order_type="LIMIT",
                slice_id=f"POV_{i}"
            ))
        
        return slices
    
    def update_state(self, fill: dict, market_state: dict):
        pass


class IcebergAlgorithm(ExecutionAlgorithm):
    """Iceberg order slicing.
    
    Shows only small portion of total order at a time.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.config = config
        self.display_qty = config.get("display_qty", 100)
        
    def generate_slices(self, plan: ExecutionPlan, 
                       display_qty: int = None) -> List[ExecutionSlice]:
        """Generate iceberg slices."""
        display = display_qty or self.display_qty
        n_slices = max(1, plan.total_quantity // (display_qty or 100))
        
        slices = []
        remaining = plan.total_quantity
        
        while remaining > 0:
            qty = min(display_qty or 100, remaining)
            slices.append(ExecutionSlice(
                timestamp=int(time.time()),
                quantity=qty,
                price_limit=0.0,
                urgency="normal",
                order_type="LIMIT",
                slice_id=f"ICE_{len(slices)}"
            ))
            remaining -= qty
            
        return slices
    
    def update_state(self, fill: dict, market_state: dict):
        pass


class SmartRouter:
    """Smart order router - selects best algorithm and venue."""
    
    def __init__(self, config: dict):
        self.config = config
        self.algorithms = {
            "TWAP": TWAPAlgorithm(),
            "VWAP": VWAPAlgorithm(),
            "ARRIVAL": ArrivalPriceAlgorithm(),
            "POV": POVAlgorithm(),
            "ICEBERG": IcebergAlgorithm(),
        }
        
    def select_algorithm(self, order: dict) -> ExecutionAlgorithm:
        """Select best algorithm based on order characteristics."""
        urgency = order.get("urgency", "normal")
        size = order.get("quantity", 0)
        adv = order.get("adv", 1_000_000)
        
        participation = order.get("quantity", 0) / max(order.get("adv", 1_000_000), 1)
        
        if participation > 0.1:
            return self.algorithms["ICEBERG"]
        elif urgency == "high":
            return self.algorithms["ARRIVAL"]
        elif participation > 0.05:
            return self.algorithms["VWAP"]
        else:
            return self.algorithms["TWAP"]


def get_algo(name: str, config: dict = None):
    """Factory function to get algorithm by name."""
    algos = {
        "twap": TWAPAlgorithm,
        "vwap": VWAPAlgorithm,
        "arrival": ArrivalPriceAlgorithm,
        "pov": POVAlgorithm,
        "iceberg": IcebergAlgorithm,
        "smart": SmartRouter,
    }
    
    cls = algos.get(name.lower())
    if cls:
        return cls(config)
    raise ValueError(f"Unknown algorithm: {name}")


# Export
__all__ = [
    "ExecutionAlgorithm", "TWAPAlgorithm", "VWAPAlgorithm",
    "ArrivalPriceAlgorithm", "POVAlgorithm", "IcebergAlgorithm",
    "SmartRouter", "get_algo", "ExecutionSlice", "ExecutionPlan"
]