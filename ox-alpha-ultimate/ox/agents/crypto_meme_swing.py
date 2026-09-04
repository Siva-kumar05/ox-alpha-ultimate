"""
Crypto Meme/Low-Cap Swing Agent
================================
Swing trading on low-cap, high-volatility crypto assets.
Uses social sentiment, volume spikes, and momentum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


from .base import BaseAgent, AgentConfig, Signal
from .risk_coordinator import RiskCoordinator
from .capital_allocator import CapitalAllocator

LOG = logging.getLogger("promax.crypto_meme")


@dataclass
class MemeState:
    """State for meme/low-cap token."""
    symbol: str
    price: float = 0.0
    volume_24h: float = 0.0
    volume_change_24h: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    market_cap: float = 0.0
    liquidity: float = 0.0
    holders: int = 0
    holder_change_24h: int = 0
    social_mentions: int = 0
    sentiment_score: float = 0.0
    smart_money_flow: float = 0.0
    whale_activity: float = 0.0
    is_rug_risk: bool = False
    rug_score: float = 0.0
    price_history: List[float] = None
    volume_history: List[float] = None


class CryptoMemeSwingAgent(BaseAgent):
    """
    Meme/Low-Cap Crypto Swing Trading Agent.
    
    Strategies:
    - Volume breakout detection (volume spike > 5x average)
    - Social sentiment momentum (Twitter/Telegram/Discord)
    - Smart money / whale tracking
    - New listing momentum
    - Narrative/theme rotation (AI, RWA, DePIN, etc.)
    - Liquidity/rug pull detection
    - Holder growth momentum
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
        
        # Strategy parameters
        self.volume_spike_threshold = config.custom_params.get('volume_spike_threshold', 5.0)
        self.min_market_cap = config.custom_params.get('min_market_cap', 100000)
        self.max_market_cap = config.custom_params.get('max_market_cap', 50000000)
        self.min_liquidity = config.custom_params.get('min_liquidity', 50000)
        
        # Social parameters
        self.min_social_mentions = config.custom_params.get('min_social_mentions', 50)
        self.sentiment_threshold = config.custom_params.get('sentiment_threshold', 0.6)
        
        # Risk parameters (higher for meme coins)
        self.max_position_pct = config.risk_params.max_position_pct
        self.max_daily_loss = config.risk_params.max_daily_loss_pct
        self.stop_loss_pct = config.custom_params.get('stop_loss_pct', 0.15)  # 15%
        self.take_profit_pct = config.custom_params.get('take_profit_pct', 0.50)  # 50%
        self.max_hold_days = config.custom_params.get('max_hold_days', 7)
        
        # Rug pull protection
        self.max_rug_score = config.custom_params.get('max_rug_score', 0.3)
        self.min_liquidity_ratio = config.custom_params.get('min_liquidity_ratio', 0.1)
        
        # State
        self.states: Dict[str, MemeState] = {}
        self.social_data: Dict[str, Dict] = {}
        self.whale_wallets: Dict[str, List[str]] = {}
        
        # Social monitor reference
        self.social_monitor = None
    
    async def initialize(self) -> bool:
        try:
            for symbol in self.config.symbols:
                self.states[symbol] = MemeState(symbol=symbol, price_history=[], volume_history=[])
            
            self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
            self.capital_allocator.register_agent(self.agent_id)
            
            # Subscribe to social data
            self.data_bus.subscribe("social:mentions", self._on_social_mentions)
            self.data_bus.subscribe("social:sentiment", self._on_sentiment)
            self.data_bus.subscribe("onchain:whale", self._on_whale_activity)
            self.data_bus.subscribe("onchain:new_token", self._on_new_token)
            
            LOG.info(f"CryptoMemeSwingAgent initialized for {len(self.config.symbols)} symbols")
            return True
        except Exception as e:
            LOG.error(f"Failed to initialize CryptoMemeSwingAgent: {e}")
            return False
    
    def _get_loop_interval(self) -> float:
        return 60.0  # 1-minute loop for swing
    
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process market data for meme coins."""
        if symbol not in self.config.symbols:
            return []
        
        signals = []
        
        try:
            state = self.states.get(symbol)
            if not state:
                return []
            
            # Update state
            self._update_state(symbol, data)
            
            # Check for entry signals
            signals.extend(self._check_volume_breakout(symbol, state))
            signals.extend(self._check_social_momentum(symbol, state))
            signals.extend(self._check_whale_accumulation(symbol, state))
            signals.extend(self._check_new_narrative(symbol, state))
            
        except Exception as e:
            LOG.error(f"Error processing {symbol}: {e}")
        
        return signals
    
    def _update_state(self, symbol: str, data: Dict) -> None:
        """Update token state from market data."""
        state = self.states[symbol]
        
        state.price = float(data.get('price', data.get('last_price', 0)))
        state.volume_24h = float(data.get('volume_24h', 0))
        state.volume_change_24h = float(data.get('volume_change_24h', 0))
        state.price_change_1h = float(data.get('price_change_1h', 0))
        state.price_change_24h = float(data.get('price_change_24h', 0))
        state.price_change_7d = float(data.get('price_change_7d', 0))
        state.market_cap = float(data.get('market_cap', 0))
        state.liquidity = float(data.get('liquidity', 0))
        state.holders = int(data.get('holders', 0))
        state.holder_change_24h = int(data.get('holder_change_24h', 0))
        
        # Track history
        state.price_history.append(state.price)
        state.volume_history.append(state.volume_24h)
        
        max_hist = 500
        if len(state.price_history) > max_hist:
            state.price_history = state.price_history[-max_hist:]
            state.volume_history = state.volume_history[-max_hist:]
        
        # Social data
        social = self.social_data.get(symbol, {})
        state.social_mentions = social.get('mentions_24h', 0)
        state.sentiment_score = social.get('sentiment', 0)
        
        # Rug check
        state.is_rug_risk, state.rug_score = self._check_rug_risk(symbol)
        
        # Whale activity
        state.whale_activity = self._get_whale_activity(symbol)
    
    def _check_volume_breakout(self, symbol: str, state: MemeState) -> List[Signal]:
        """Detect volume breakouts."""
        signals = []
        
        if symbol in self.positions:
            return signals
        
        if state.volume_change_24h >= self.volume_spike_threshold:
            # Volume spike with price confirmation
            if state.price_change_24h > 0:
                # Verify liquidity
                if state.liquidity >= self.min_liquidity:
                    # Check rug risk
                    if state.rug_score < self.max_rug_score:
                        signals.append(Signal(
                            agent_id=self.agent_id,
                            symbol=symbol,
                            action="buy",
                            strength=min(1.0, state.volume_change_24h / 10),
                            price=state.price,
                            quantity=0,
                            leverage=1.0,
                            metadata={
                                "strategy": "volume_breakout",
                                "volume_spike": state.volume_change_24h,
                                "price_change_24h": state.price_change_24h,
                                "market_cap": state.market_cap,
                                "liquidity": state.liquidity
                            }
                        ))
        
        return signals
    
    def _check_social_momentum(self, symbol: str, state: MemeState) -> List[Signal]:
        """Check social media momentum."""
        signals = []
        
        if symbol in self.positions:
            return signals
        
        # High social mentions + positive sentiment + price up
        if (state.social_mentions >= self.min_social_mentions and
            state.sentiment_score > self.sentiment_threshold and
            state.price_change_24h > 10):
            
            # Check holder growth
            holder_growth = state.holder_change_24h / max(state.holders, 1)
            if holder_growth > 0.05:  # 5% holder growth in 24h
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="buy",
                    strength=0.8,
                    price=state.price,
                    quantity=0,
                    leverage=1.0,
                    metadata={
                        "strategy": "social_momentum",
                        "mentions": state.social_mentions,
                        "sentiment": state.sentiment_score,
                        "holder_growth": holder_growth
                    }
                ))
        
        return signals
    
    def _check_whale_accumulation(self, symbol: str, state: MemeState) -> List[Signal]:
        """Check for whale/smart money accumulation."""
        signals = []
        
        if symbol in self.positions:
            return signals
        
        if state.whale_activity > 0.7:  # High whale activity
            if state.smart_money_flow > 0:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="buy",
                    strength=0.9,
                    price=state.price,
                    quantity=0,
                    leverage=1.0,
                    metadata={
                        "strategy": "whale_accumulation",
                        "whale_activity": state.whale_activity,
                        "smart_money_flow": state.smart_money_flow
                    }
                ))
        
        return signals
    
    def _check_new_narrative(self, symbol: str, state: MemeState) -> List[Signal]:
        """Check for new narrative/theme momentum."""
        signals = []
        
        # Would check for trending narratives (AI, RWA, DePIN, etc.)
        # Simplified: check 7d momentum
        if state.price_change_7d > 50 and state.price_change_24h > 0:
            if state.market_cap < self.max_market_cap and state.market_cap > self.min_market_cap:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="buy",
                    strength=0.7,
                    price=state.price,
                    quantity=0,
                    leverage=1.0,
                    metadata={
                        "strategy": "narrative_momentum",
                        "price_change_7d": state.price_change_7d,
                        "market_cap": state.market_cap
                    }
                ))
        
        return signals
    
    def _check_rug_risk(self, symbol: str) -> tuple:
        """Check rug pull risk."""
        # Simplified rug detection
        state = self.states.get(symbol)
        if not state:
            return True, 1.0
        
        rug_score = 0.0
        
        # Low liquidity
        if state.liquidity < state.market_cap * 0.05:
            rug_score += 0.3
        
        # No lock/vesting info (would check on-chain)
        rug_score += 0.2
        
        # Holder concentration
        if state.holders > 0:
            # Would check top 10 holder %
            pass
        
        # Contract verified
        # Would check contract verification
        
        is_risk = rug_score > 0.5
        return is_risk, min(1.0, rug_score)
    
    def _get_whale_activity(self, symbol: str) -> float:
        """Get whale activity score."""
        # Would query on-chain data
        return 0.0
    
    def _can_open_position(self, symbol: str) -> bool:
        if len(self.positions) >= self.config.risk_params.max_concurrent_positions:
            return False
        
        state = self.states.get(symbol)
        if not state:
            return False
        
        # Rug check
        if state.is_rug_risk:
            return False
        
        # Market cap filter
        if state.market_cap < self.min_market_cap or state.market_cap > self.max_market_cap:
            return False
        
        # Liquidity filter
        if state.liquidity < self.min_liquidity:
            return False
        
        capital = self.capital_allocator.get_allocation(self.agent_id)
        used = sum(p.quantity * p.current_price for p in self.positions.values())
        if used > capital * 0.8:
            return False
        
        return True
    
    async def manage_positions(self) -> List[Signal]:
        """Manage meme coin positions."""
        signals = []
        
        for symbol, position in list(self.positions.items()):
            state = self.states.get(symbol)
            if not state:
                continue
            
            current_price = state.price
            entry_price = position.entry_price
            hold_days = (datetime.now() - position.entry_time).days
            
            # Stop loss
            if position.side == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -self.stop_loss_pct:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=state.price, quantity=position.quantity,
                        metadata={"reason": "stop_loss", "pnl_pct": pnl_pct}
                    ))
                elif pnl_pct >= self.take_profit_pct:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=state.price, quantity=position.quantity,
                        metadata={"reason": "take_profit", "pnl_pct": pnl_pct}
                    ))
            
            # Time stop
            if hold_days >= self.max_hold_days:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=state.price, quantity=position.quantity,
                    metadata={"reason": "time_stop", "hold_days": hold_days}
                ))
            
            # Rug pull protection
            if state.is_rug_risk:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=state.price, quantity=position.quantity,
                    metadata={"reason": "rug_risk_detected", "rug_score": state.rug_score}
                ))
            
            # Momentum loss
            if state.price_change_24h < -20:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=0.8, price=state.price, quantity=position.quantity,
                    metadata={"reason": "momentum_loss", "change_24h": state.price_change_24h}
                ))
        
        return signals
    
    def _on_social_mentions(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol in self.states:
            self.social_data[symbol] = data
    
    def _on_sentiment(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol in self.states:
            self.social_data[symbol]['sentiment'] = data.get('sentiment', 0)
    
    def _on_whale_activity(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol in self.states:
            self.states[symbol].whale_activity = data.get('activity_score', 0)
            self.states[symbol].smart_money_flow = data.get('smart_money_flow', 0)
    
    def _on_new_token(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol not in self.states and symbol not in self.config.symbols:
            # Could dynamically add to watchlist
            pass