"""Dynamic Rebalancing and Hedging Capability."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum
import numpy as np
from .core import iso, LOG
from .risk import RiskManager, CovarianceMatrix


class HedgeType(Enum):
    INDEX_FUTURE = "index_future"
    SECTOR_ETF = "sector_etf"
    OPTIONS_PUT = "options_put"
    INVERSE_ETF = "inverse_etf"
    CASH = "cash"


class RebalanceTrigger(Enum):
    THRESHOLD = "threshold"
    TIME_BASED = "time_based"
    REGIME_CHANGE = "regime_change"
    RISK_LIMIT = "risk_limit"
    MANUAL = "manual"


@dataclass
class TargetAllocation:
    """Target portfolio allocation."""
    symbol: str
    target_weight: float
    min_weight: float
    max_weight: float
    hedge_ratio: float = 0.0
    hedge_instrument: str = ""


@dataclass
class RebalanceOrder:
    """Single rebalance order."""
    symbol: str
    current_weight: float
    target_weight: float
    current_qty: int
    target_qty: int
    order_qty: int
    side: str  # BUY/SELL
    priority: int
    reason: str


@dataclass
class HedgePosition:
    """Hedge position details."""
    hedge_type: HedgeType
    symbol: str
    quantity: int
    notional: float
    hedge_ratio: float
    underlying_symbols: List[str]
    expiry: Optional[str] = None
    strike: Optional[float] = None


class PortfolioRebalancer:
    """Handles dynamic portfolio rebalancing."""
    
    def __init__(self, cfg, risk_manager: RiskManager, db):
        self.cfg = cfg
        self.risk = risk_manager
        self.db = db
        self.rebalance_cfg = cfg.get("rebalancing", {})
        self.enabled = self.rebalance_cfg.get("enabled", True)
        self.threshold = self.rebalance_cfg.get("threshold", 0.05)
        self.min_trade_notional = self.rebalance_cfg.get("min_trade_notional", 5000)
        self.max_turnover_pct = self.rebalance_cfg.get("max_turnover_pct", 0.3)
        self.rebalance_interval = self.rebalance_cfg.get("interval_hours", 4)
        
        # Target allocations
        self.targets: Dict[str, TargetAllocation] = {}
        self._last_rebalance = 0.0
    
    def set_targets(self, targets: Dict[str, TargetAllocation]):
        """Set target allocations."""
        self.targets = targets
    
    def compute_current_weights(self, positions: Dict[str, dict], prices: Dict[str, float]) -> Dict[str, float]:
        """Compute current portfolio weights."""
        total_value = sum(
            pos.get("qty", 0) * prices.get(sym, 0)
            for sym, pos in positions.items()
        )
        
        if total_value <= 0:
            return {}
        
        weights = {}
        for sym, pos in positions.items():
            price = prices.get(sym, 0)
            qty = pos.get("qty", 0)
            weights[sym] = (qty * price) / total_value
        
        return weights
    
    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD
    ) -> Tuple[bool, List[RebalanceOrder]]:
        """Check if rebalancing is needed and generate orders."""
        if not self.enabled:
            return False, []
        
        if not self.targets:
            return False, []
        
        orders = []
        max_deviation = 0.0
        
        for symbol, target in self.targets.items():
            current_w = current_weights.get(symbol, 0.0)
            deviation = abs(current_w - target.target_weight)
            max_deviation = max(max_deviation, deviation)
            
            if deviation > self.threshold:
                # Generate rebalance order
                order = self._generate_rebalance_order(symbol, current_w, target, prices=None)
                if order:
                    orders.append(order)
        
        # Also check for symbols in portfolio but not in targets (should be closed)
        for symbol, current_w in current_weights.items():
            if symbol not in self.targets and current_w > 0.01:
                orders.append(RebalanceOrder(
                    symbol=symbol,
                    current_weight=current_w,
                    target_weight=0.0,
                    current_qty=0,  # Will be filled
                    target_qty=0,
                    order_qty=0,
                    side="SELL",
                    priority=1,
                    reason="symbol_removed_from_targets"
                ))
        
        # Check turnover limit
        total_turnover = sum(abs(o.current_weight - o.target_weight) for o in orders)
        if total_turnover > self.max_turnover_pct:
            LOG.warning(f"Rebalance turnover {total_turnover:.2%} exceeds limit {self.max_turnover_pct:.2%}")
            return False, []
        
        needed = max_deviation > self.threshold or len(orders) > 0
        return needed, orders
    
    def _generate_rebalance_order(
        self,
        symbol: str,
        current_weight: float,
        target: TargetAllocation,
        prices: Optional[Dict[str, float]] = None
    ) -> Optional[RebalanceOrder]:
        """Generate a rebalance order for a symbol."""
        # This would need current position and price
        # Simplified for now
        return RebalanceOrder(
            symbol=symbol,
            current_weight=current_weight,
            target_weight=target.target_weight,
            current_qty=0,
            target_qty=0,
            order_qty=0,
            side="BUY" if target.target_weight > current_weight else "SELL",
            priority=1,
            reason="threshold_breach"
        )
    
    def execute_rebalance(self, orders: List[RebalanceOrder], broker) -> List[dict]:
        """Execute rebalance orders."""
        results = []
        
        for order in sorted(orders, key=lambda o: o.priority):
            try:
                # Would implement actual order execution
                # For now, return mock result
                results.append({
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.order_qty,
                    "status": "SUBMITTED",
                    "reason": order.reason
                })
            except Exception as e:
                LOG.error(f"Rebalance order failed for {order.symbol}: {e}")
                results.append({
                    "symbol": order.symbol,
                    "status": "FAILED",
                    "error": str(e)
                })
        
        self._last_rebalance = iso()
        return results


class PortfolioHedger:
    """Manages portfolio hedging."""
    
    def __init__(self, cfg, risk_manager: RiskManager, db):
        self.cfg = cfg
        self.risk = risk_manager
        self.db = db
        self.hedge_cfg = cfg.get("hedging", {})
        self.enabled = self.hedge_cfg.get("enabled", False)
        self.max_hedge_ratio = self.hedge_cfg.get("max_hedge_ratio", 1.0)
        self.min_hedge_ratio = self.hedge_cfg.get("min_hedge_ratio", 0.0)
        self.hedge_instruments = self.hedge_cfg.get("instruments", {
            "index_future": "NIFTY_FUT",
            "sector_etf": "NIFTYBEES",
            "inverse_etf": "NIFTYBEES_INV"
        })
        
        self.active_hedges: Dict[str, HedgePosition] = {}
    
    def calculate_portfolio_beta(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        index_returns: np.ndarray
    ) -> float:
        """Calculate portfolio beta to index."""
        if not positions:
            return 0.0
        
        weights = self.risk.compute_current_weights(positions, prices)
        
        # Get historical returns for each position
        for sym, weight in weights.items():
            if weight <= 0:
                continue
            # Would need historical data
            pass
        
        # Simplified: return weighted average of individual betas
        return 1.0  # Placeholder
    
    def calculate_hedge_ratio(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        target_beta: float = 0.0
    ) -> float:
        """Calculate required hedge ratio."""
        if portfolio_beta == 0:
            return 0.0
        
        hedge_ratio = (target_beta - portfolio_beta) / portfolio_beta
        return float(np.clip(hedge_ratio, -self.max_hedge_ratio, self.max_hedge_ratio))
    
    def create_index_hedge(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        index_price: float,
        index_lot_size: int = 75
    ) -> HedgePosition:
        """Create index future hedge."""
        hedge_notional = portfolio_value * abs(hedge_ratio)
        contracts = int(hedge_notional / (index_price * index_lot_size))
        
        return HedgePosition(
            hedge_type=HedgeType.INDEX_FUTURE,
            symbol=self.hedge_instruments.get("index_future", "NIFTY_FUT"),
            quantity=contracts if hedge_ratio < 0 else -contracts,
            notional=abs(contracts) * index_price * index_lot_size,
            hedge_ratio=hedge_ratio,
            underlying_symbols=list(positions.keys()) if (positions := {}) else []
        )
    
    def create_sector_hedges(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> List[HedgePosition]:
        """Create sector-level hedges."""
        sector_exposure = defaultdict(float)
        
        for sym, pos in positions.items():
            sector = sector_map.get(sym)
            if sector:
                sector_exposure[sector] += pos.get("qty", 0) * prices.get(sym, 0)
        
        hedges = []
        for sector, exposure in sector_exposure.items():
            if abs(exposure) > 10000:  # Minimum notional
                etf_symbol = self.hedge_instruments.get(f"sector_{sector}", "NIFTYBEES")
                etf_price = prices.get(etf_symbol, 1000)
                qty = int(exposure / etf_price)
                
                hedges.append(HedgePosition(
                    hedge_type=HedgeType.SECTOR_ETF,
                    symbol=etf_symbol,
                    quantity=-qty,  # Opposite direction
                    notional=abs(qty) * etf_price,
                    hedge_ratio=1.0,
                    underlying_symbols=[s for s, sec in sector_map.items() if sec == sector]
                ))
        
        return hedges
    
    def update_hedges(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        index_price: float
    ) -> List[HedgePosition]:
        """Update hedge positions based on current portfolio."""
        if not self.enabled:
            return []
        
        portfolio_value = sum(
            pos.get("qty", 0) * prices.get(sym, 0)
            for sym, pos in positions.items()
        )
        
        if portfolio_value <= 0:
            return []
        
        # Calculate portfolio beta
        # For now, use simplified approach
        beta = 1.0  # Would compute from historical data
        
        # Target beta (0 for market neutral, or configurable)
        target_beta = self.hedge_cfg.get("target_beta", 0.0)
        
        hedge_ratio = self.calculate_hedge_ratio(portfolio_value, beta, target_beta)
        
        if abs(hedge_ratio) < 0.05:
            # Close existing hedges
            self.close_all_hedges()
            return []
        
        # Create/update index hedge
        hedge = self.create_index_hedge(portfolio_value, hedge_ratio, index_price)
        self.active_hedges["index"] = hedge
        
        return [hedge]
    
    def close_all_hedges(self) -> List[dict]:
        """Close all active hedge positions."""
        results = []
        for name, hedge in self.active_hedges.items():
            results.append({
                "hedge": name,
                "symbol": hedge.symbol,
                "quantity": -hedge.quantity,  # Opposite to close
                "action": "CLOSE"
            })
        self.active_hedges.clear()
        return results
    
    def get_hedge_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate unrealized PnL on hedges."""
        total_pnl = 0.0
        for hedge in self.active_hedges.values():
            current_price = prices.get(hedge.symbol, 0)
            if current_price > 0 and hedge.quantity != 0:
                # Simplified PnL calculation
                pnl = hedge.quantity * (current_price - hedge.notional / abs(hedge.quantity))
                total_pnl += pnl
        return total_pnl


class RiskBudgetAllocator:
    """Allocates risk budget across strategies/assets."""
    
    def __init__(self, cfg, risk_manager: RiskManager):
        self.cfg = cfg
        self.risk = risk_manager
        self.budget_cfg = cfg.get("risk_budget", {})
        self.total_budget = self.budget_cfg.get("total_var_budget", 0.05)  # 5% portfolio VaR
        self.strategy_budgets: Dict[str, float] = {}
    
    def allocate_budget(
        self,
        strategies: List[dict],
        covariance: Optional[CovarianceMatrix] = None
    ) -> Dict[str, float]:
        """Allocate risk budget across strategies."""
        if not strategies:
            return {}
        
        # Equal risk contribution (risk parity) across strategies
        n = len(strategies)
        equal_budget = self.total_budget / n
        
        # Adjust by strategy quality (Sharpe, consistency)
        allocations = {}
        for strat in strategies:
            sid = strat.get("id", "")
            sharpe = strat.get("sharpe", 0.0)
            consistency = strat.get("consistency", 0.5)
            
            # Quality multiplier
            quality = max(0.5, min(2.0, 0.5 + sharpe * 0.5 + consistency * 0.5))
            allocations[sid] = equal_budget * quality
        
        # Normalize to total budget
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total * self.total_budget for k, v in allocations.items()}
        
        self.strategy_budgets = allocations
        return allocations
    
    def check_budget_usage(
        self,
        strategy_id: str,
        current_var: float
    ) -> Tuple[bool, float]:
        """Check if strategy is within its risk budget."""
        budget = self.strategy_budgets.get(strategy_id, self.total_budget)
        usage = current_var / budget if budget > 0 else 0.0
        return usage <= 1.0, usage
    
    def get_remaining_budget(self, strategy_id: str) -> float:
        """Get remaining risk budget for strategy."""
        budget = self.strategy_budgets.get(strategy_id, self.total_budget)
        # Would need current usage tracking
        return budget