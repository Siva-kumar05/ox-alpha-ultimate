"""
Pro Max Multi-Agent Trading System
===================================
Modular, independent agents for specialized trading strategies.
Each agent is a self-contained unit with its own risk, data, and execution.

Entry point: ``python -m ox.agents.orchestrator --config config_promax.yaml``
(or ``python run.py promax``).

Operator rule enforced everywhere: capital-deploying orders (buys) wait for
human approval; risk-reducing orders (sells/closes) never wait.
"""

from __future__ import annotations

# Core infrastructure
from .base import BaseAgent, AgentState, AgentConfig, RiskParams, ResourcePool, SharedDataBus
from .orchestrator import AgentOrchestrator, DataPump, ExecutionRouter

# Equity agents
from .equity_momentum import EquityMomentumAgent
from .equity_growth import EquityGrowthAgent
from .intraday_scalper import IntradayScalperAgent

# Crypto agents
from .crypto_perp import CryptoPerpAgent
from .crypto_funding import CryptoFundingArbAgent
from .crypto_meme_swing import CryptoMemeSwingAgent

# Derivatives agents
from .options_0dte import Options0DTEAgent

# Market making (simulated for retail)
from .market_maker import MarketMakerAgent

# Intelligence
from .news_intel import NewsIntelligenceAgent
from .social_monitor import SocialMonitorAgent

# Risk & coordination
from .risk_coordinator import RiskCoordinator, LeverageLadder, monte_carlo_survival
from .capital_allocator import CapitalAllocator
from .approvals import ApprovalGateway
from .debate import DebatePanel, TradeMemory

__all__ = [
    # Core
    "BaseAgent", "AgentState", "AgentConfig", "RiskParams",
    "ResourcePool", "SharedDataBus",
    "AgentOrchestrator", "DataPump", "ExecutionRouter",
    # Equity
    "EquityMomentumAgent", "EquityGrowthAgent", "IntradayScalperAgent",
    # Crypto
    "CryptoPerpAgent", "CryptoFundingArbAgent", "CryptoMemeSwingAgent",
    # Derivatives
    "Options0DTEAgent",
    # Market Making
    "MarketMakerAgent",
    # Intelligence
    "NewsIntelligenceAgent", "SocialMonitorAgent",
    # Coordination
    "RiskCoordinator", "LeverageLadder", "monte_carlo_survival",
    "CapitalAllocator", "ApprovalGateway", "DebatePanel", "TradeMemory",
]
