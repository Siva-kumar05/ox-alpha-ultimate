"""
OxAlpha Pro Max - Multi-Agent Trading System
=============================================
A polyglot trading system with specialized agents for different strategies.
"""

__version__ = "0.1.0"
__author__ = "OxAlpha Team"

from .config import Settings, get_settings
from .orchestrator import AgentOrchestrator
from .agents.base import BaseAgent, AgentConfig, AgentState

__all__ = [
    "Settings",
    "get_settings",
    "AgentOrchestrator",
    "BaseAgent",
    "AgentConfig",
    "AgentState",
]