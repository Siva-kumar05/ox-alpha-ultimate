"""
Crypto Funding Rate Arbitrage Agent
====================================
Delta-neutral funding rate arbitrage across exchanges and symbols.
Pure arbitrage: long spot + short perp when funding > threshold.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent, AgentConfig, RiskParams, Signal, Position, AgentType
from .risk_coordinator import RiskCoordinator
from .capital_allocator import CapitalAllocator

LOG = logging.getLogger("promax.crypto_funding")


@dataclass
class FundingOpportunity:
    """Funding arbitrage opportunity."""
    symbol: str
    exchange: str
    funding_rate: float
    funding_apr: float
    basis_pct: float
    spot_price: float
    perp_price: float
    perp_exchange: str
    spot_exchange: str
    estimated_yield: float
    risk_score: float
    expires_at: datetime


class CryptoFundingArbAgent(BaseAgent):
    """
    Crypto Funding Rate Arbitrage Agent.
    
    Pure delta-neutral arbitrage:
    - Long spot + Short perp when funding > threshold
    - Short spot + Long perp when funding < -threshold
    - Cross-exchange arbitrage (different funding rates)
    - Multi-venue optimization
    
    Risk: Exchange risk, execution slippage, funding rate changes, liquidation
    """
    
    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator: RiskCoordinator,
        capital_allocator: CapitalAllocator
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)
        
        # Arbitrage parameters
        self.min_funding_rate = config.custom_params.get('min_funding_rate', 0.0001)  # 0.01% per 8h
        self.max_basis_pct = config.custom_params.get('max_basis_pct', 0.01)  # 1%
        self.min_yield_apr = config.custom_params.get('min_yield_apr', 0.10)  # 10% APR
        
        # Execution
        self.max_slippage_bps = config.custom_params.get('max_slippage_bps', 5)
        self.use_post_only = config.custom_params.get('use_post_only', True)
        self.hedge_ratio = config.custom_params.get('hedge_ratio', 1.0)
        
        # Risk
        self.max_exchange_exposure = config.custom_params.get('max_exchange_exposure', 0.30)
        self.max_symbol_exposure = config.custom_params.get('max_symbol_exposure', 0.20)
        
        # State
        self.opportunities: List[FundingOpportunity] = []
        self.active_arbs: Dict[str, Dict] = {}  # symbol -> arb details
        self.exchange_funding: Dict[str, Dict[str, float]] = {}  # exchange -> symbol -> funding
        self.exchange_basis: Dict[str, Dict[str, float]] = {}
        
        # Exchange clients
        self.exchanges: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        try:
            self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
            self.capital_allocator.register_agent(self.agent_id)
            
            # Subscribe to multi-exchange data
            self.data_bus.subscribe("exchange:funding", self._on_exchange_funding)
            self.data_bus.subscribe("exchange:basis", self._on_exchange_basis)
            self.data_bus.subscribe("exchange:orderbook", self._on_orderbook)
            
            LOG.info(f"CryptoFundingArbAgent initialized for {len(self.config.symbols)} symbols")
            return True
        except Exception as e:
            LOG.error(f"Failed to initialize CryptoFundingArbAgent: {e}")
            return False
    
    def _get_loop_interval(self) -> float:
        return 30.0  # 30-second scan for arb opportunities
    
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process funding rate data for arbitrage detection."""
        if symbol not in self.config.symbols:
            return []
        
        signals = []
        
        try:
            # Scan for opportunities
            opps = self._scan_opportunities()
            
            for opp in opps:
                signal = self._create_arb_signal(opp)
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            LOG.error(f"Error scanning funding arb for {symbol}: {e}")
        
        return signals
    
    def _scan_opportunities(self) -> List[FundingOpportunity]:
        """Scan all exchanges/symbols for funding arb opportunities."""
        opportunities = []
        
        # Would iterate over all exchanges and symbols
        # Simplified for now
        for symbol in self.config.symbols:
            # Get funding rates across exchanges
            funding_data = self._get_funding_across_exchanges(symbol)
            
            if len(funding_data) < 2:
                continue
            
            # Find best long funding (to collect) and best short funding
            for exch_long, data_long in funding_data.items():
                for exch_short, data_short in funding_data.items():
                    if exch_long == exch_short:
                        continue
                    
                    # Arb: long on exchange with lower funding, short on higher
                    funding_diff = data_short['rate'] - data_long['rate']
                    
                    if funding_diff > self.min_funding_rate:
                        # Check basis alignment
                        basis = abs(data_long['basis_pct'])
                        if basis < self.max_basis_pct:
                            opp = FundingOpportunity(
                                symbol=symbol,
                                exchange=exch_long,
                                funding_rate=data_short['rate'] - data_long['rate'],
                                funding_apr=funding_diff * 365 * 3,
                                basis_pct=basis,
                                spot_price=data_long['spot_price'],
                                perp_price=data_short['perp_price'],
                                perp_exchange=exch_short,
                                spot_exchange=exch_long,
                                estimated_yield=funding_diff * 365 * 3,
                                risk_score=self._calc_risk_score(data_long, data_short),
                                expires_at=datetime.now() + timedelta(hours=8)
                            )
                            opportunities.append(opp)
        
        # Sort by yield * risk_score
        opportunities.sort(key=lambda x: x.estimated_yield * x.risk_score, reverse=True)
        return opportunities[:10]  # Top 10
    
    def _get_funding_across_exchanges(self, symbol: str) -> Dict[str, Dict]:
        """Get funding rates for symbol across all exchanges."""
        # Would query multiple exchanges
        # Simplified mock data
        return {
            'binance': {'rate': 0.0001, 'basis_pct': 0.002, 'spot_price': 50000, 'perp_price': 50100},
            'bybit': {'rate': 0.00015, 'basis_pct': 0.003, 'spot_price': 50000, 'perp_price': 50150},
            'okx': {'rate': 0.00008, 'basis_pct': 0.001, 'spot_price': 50000, 'perp_price': 50080},
        }
    
    def _calc_risk_score(self, long_data: Dict, short_data: Dict) -> float:
        """Calculate risk score for opportunity."""
        score = 1.0
        
        # Basis risk
        basis_risk = 1 - min(1, abs(long_data['basis_pct']) / 0.02)
        score *= basis_risk
        
        # Exchange risk (simplified)
        score *= 0.9  # Each exchange adds risk
        
        # Liquidity risk
        score *= 0.95
        
        return max(0.1, min(1.0, score))
    
    def _create_arb_signal(self, opp: FundingOpportunity) -> Optional[Signal]:
        """Create signal for funding arbitrage."""
        # Check if already have this arb
        key = f"{opp.symbol}_{opp.spot_exchange}_{opp.perp_exchange}"
        if key in self.active_arbs:
            return None
        
        # Check risk limits
        if not self._can_open_arb(opp):
            return None
        
        # Create signal - this is a complex multi-leg order
        # Signal represents the short perp leg (long spot handled separately)
        return Signal(
            agent_id=self.agent_id,
            symbol=opp.symbol,
            action="sell",  # Short perp
            strength=min(1.0, opp.risk_score),
            price=opp.perp_price,
            quantity=0,  # Calculated at execution
            leverage=1.0,  # Delta neutral
            metadata={
                "strategy": "funding_arb",
                "arb_type": "delta_neutral",
                "spot_exchange": opp.spot_exchange,
                "perp_exchange": opp.perp_exchange,
                "spot_price": opp.spot_price,
                "perp_price": opp.perp_price,
                "funding_rate": opp.funding_rate,
                "estimated_yield_apr": opp.estimated_yield,
                "basis_pct": opp.basis_pct,
                "arb_key": f"{opp.symbol}_{opp.spot_exchange}_{opp.perp_exchange}",
                "legs": [
                    {"exchange": opp.spot_exchange, "symbol": opp.symbol, "side": "buy", "type": "spot"},
                    {"exchange": opp.perp_exchange, "symbol": opp.symbol, "side": "sell", "type": "perp"}
                ]
            }
        )
    
    def _can_open_arb(self, opp: FundingOpportunity) -> bool:
        """Check if we can open this arbitrage."""
        # Check exchange exposure
        spot_exposure = self._get_exchange_exposure(opp.spot_exchange)
        perp_exposure = self._get_exchange_exposure(opp.perp_exchange)
        
        max_exposure = self.capital_allocator.get_allocation(self.agent_id) * self.max_exchange_exposure
        
        if spot_exposure > max_exposure or perp_exposure > max_exposure:
            return False
        
        # Check symbol exposure
        symbol_exposure = self._get_symbol_exposure(opp.symbol)
        max_symbol = self.capital_allocator.get_allocation(self.agent_id) * self.max_symbol_exposure
        if symbol_exposure > max_symbol:
            return False
        
        return True
    
    def _get_exchange_exposure(self, exchange: str) -> float:
        """Get current exposure on an exchange."""
        total = 0
        for arb in self.active_arbs.values():
            if arb.get('spot_exchange') == exchange or arb.get('perp_exchange') == exchange:
                total += arb.get('notional', 0)
        return total
    
    def _get_symbol_exposure(self, symbol: str) -> float:
        """Get current exposure to a symbol."""
        total = 0
        for arb in self.active_arbs.values():
            if arb.get('symbol') == symbol:
                total += arb.get('notional', 0)
        return total
    
    async def manage_positions(self) -> List[Signal]:
        """Manage active arbitrage positions."""
        signals = []
        
        for key, arb in list(self.active_arbs.items()):
            # Check if funding rate still favorable
            # Check if basis widened
            # Check if near funding payment
            # Check for liquidation risk
            
            # Simplified: check if funding flipped
            current_funding = self._get_current_funding(arb['symbol'], arb['perp_exchange'])
            if current_funding * arb['initial_funding'] < 0:
                # Funding flipped, close arb
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=arb['symbol'],
                    action="close_arb",
                    strength=1.0,
                    price=0,
                    quantity=0,
                    metadata={
                        "arb_key": key,
                        "reason": "funding_flipped",
                        "current_funding": current_funding
                    }
                ))
            
            # Check basis widening
            current_basis = self._get_current_basis(arb['symbol'], arb['perp_exchange'])
            if abs(current_basis) > arb['max_basis'] * 2:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=arb['symbol'],
                    action="close_arb",
                    strength=1.0,
                    price=0,
                    quantity=0,
                    metadata={
                        "arb_key": key,
                        "reason": "basis_widened",
                        "current_basis": current_basis
                    }
                ))
        
        return signals
    
    def _get_current_funding(self, symbol: str, exchange: str) -> float:
        """Get current funding rate."""
        return self.exchange_funding.get(exchange, {}).get(symbol, 0)
    
    def _get_current_basis(self, symbol: str, exchange: str) -> float:
        """Get current basis."""
        return self.exchange_basis.get(exchange, {}).get(symbol, 0)
    
    def _on_exchange_funding(self, data: Dict) -> None:
        """Handle funding rate updates from exchanges."""
        exchange = data.get('exchange')
        symbol = data.get('symbol')
        rate = float(data.get('funding_rate', 0))
        
        if exchange not in self.exchange_funding:
            self.exchange_funding[exchange] = {}
        self.exchange_funding[exchange][symbol] = rate
    
    def _on_exchange_basis(self, data: Dict) -> None:
        exchange = data.get('exchange')
        symbol = data.get('symbol')
        basis = float(data.get('basis_pct', 0))
        
        if exchange not in self.exchange_basis:
            self.exchange_basis[exchange] = {}
        self.exchange_basis[exchange][symbol] = basis
    
    def _on_orderbook(self, data: Dict) -> None:
        # For slippage estimation
        pass
    
    def _can_open_position(self, symbol: str) -> bool:
        if len(self.positions) >= self.config.risk_params.max_concurrent_positions:
            return False
        return True