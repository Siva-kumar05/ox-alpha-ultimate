"""
Configuration management using Pydantic Settings
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices


class RiskLimits(BaseModel):
    """Risk limits for agents"""
    max_position_pct: float = 0.15
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_leverage: float = 1.0
    max_positions: int = 5
    max_notional_per_trade: float = 50000
    var_limit_pct: float = 0.05
    correlation_limit: float = 0.7


class AgentConfig(BaseModel):
    """Configuration for a single agent"""
    agent_id: str
    name: str
    type: str  # equity_momentum, intraday_scalper, crypto_perp, etc.
    symbols: List[str] = Field(default_factory=list)
    enabled: bool = True
    priority: int = 5
    risk: RiskLimits = Field(default_factory=RiskLimits)
    params: Dict[str, Any] = Field(default_factory=dict)


class ExchangeConfig(BaseModel):
    """Exchange connection configuration"""
    name: str
    enabled: bool = False
    ws_url: str = ""
    rest_url: str = ""
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    subscriptions: List[str] = Field(default_factory=list)


class KafkaConfig(BaseModel):
    brokers: List[str] = Field(default_factory=lambda: ["localhost:9092"])
    topic: str = "market_ticks"
    consumer_group: str = "oxalpha-marketdata"
    batch_size: int = 100
    batch_timeout_ms: int = 10


class DatabaseConfig(BaseModel):
    url: str = "postgresql://localhost/oxalpha"
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    max_connections: int = 50


class MonitoringConfig(BaseModel):
    prometheus_port: int = 9090
    health_check_port: int = 8080
    metrics_interval: int = 10
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Main application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "oxalpha-promax"
    environment: str = "development"
    debug: bool = False
    
    # Capital
    total_capital: float = 1000000.0
    base_currency: str = "INR"
    
    # Trading hours (IST)
    market_open: str = "09:15"
    market_close: str = "15:30"
    entry_cutoff: str = "14:45"
    squareoff: str = "15:15"
    
    # Symbols
    symbols: List[str] = Field(default_factory=lambda: [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    ])
    
    # Agents
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # Exchanges
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    
    # Infrastructure
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Risk
    global_risk: RiskLimits = Field(default_factory=RiskLimits)
    
    # ML
    ml_enabled: bool = True
    ml_model_path: str = "./models"
    
    # Paths
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    config_dir: Path = Path("./config")
    
    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        return self.agents.get(agent_id)
    
    def get_enabled_agents(self) -> List[AgentConfig]:
        return [a for a in self.agents.values() if a.enabled]
    
    def get_exchange_config(self, name: str) -> Optional[ExchangeConfig]:
        return self.exchanges.get(name)
    
    @property
    def market_open_time(self) -> time:
        from datetime import time
        h, m = map(int, self.market_open.split(":"))
        return time(h, m)
    
    @property
    def market_close_time(self) -> time:
        from datetime import time
        h, m = map(int, self.market_close.split(":"))
        return time(h, m)
    
    @property
    def is_market_open(self) -> bool:
        from datetime import datetime, time
        now = datetime.now().time()
        return self.market_open_time <= now <= self.market_close_time
    
    def model_post_init(self, __context: Any) -> None:
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default agents if none configured
        if not self.agents:
            self.agents = self._default_agents()
        
        # Set default exchanges if none configured
        if not self.exchanges:
            self.exchanges = self._default_exchanges()
    
    def _default_agents(self) -> Dict[str, AgentConfig]:
        return {
            "equity_momentum": AgentConfig(
                agent_id="equity_momentum",
                name="Equity Momentum",
                type="equity_momentum",
                symbols=["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS"],
                priority=1,
                risk=RiskLimits(max_leverage=1.0, max_position_pct=0.15),
            ),
            "intraday_scalper": AgentConfig(
                agent_id="intraday_scalper",
                name="Intraday Scalper",
                type="intraday_scalper",
                symbols=["RELIANCE", "HDFCBANK", "ICICIBANK"],
                priority=1,
                risk=RiskLimits(max_leverage=2.0, max_position_pct=0.10, max_daily_loss_pct=0.015),
            ),
            "crypto_perp": AgentConfig(
                agent_id="crypto_perp",
                name="Crypto Perpetuals",
                type="crypto_perp",
                symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                priority=2,
                risk=RiskLimits(max_leverage=10.0, max_position_pct=0.20, max_daily_loss_pct=0.05),
            ),
            "crypto_funding": AgentConfig(
                agent_id="crypto_funding",
                name="Funding Rate Arb",
                type="crypto_funding",
                symbols=["BTCUSDT", "ETHUSDT"],
                priority=2,
                risk=RiskLimits(max_leverage=5.0, max_position_pct=0.25),
            ),
            "crypto_meme": AgentConfig(
                agent_id="crypto_meme",
                name="Meme/Low-Cap Swing",
                type="crypto_meme_swing",
                symbols=["PEPEUSDT", "WIFUSDT", "BONKUSDT"],
                priority=3,
                risk=RiskLimits(max_leverage=5.0, max_position_pct=0.10, max_daily_loss_pct=0.10),
            ),
            "options_0dte": AgentConfig(
                agent_id="options_0dte",
                name="0DTE Options",
                type="options_0dte",
                symbols=["BANKNIFTY", "NIFTY"],
                priority=1,
                risk=RiskLimits(max_leverage=10.0, max_position_pct=0.10),
            ),
        }
    
    def _default_exchanges(self) -> Dict[str, ExchangeConfig]:
        return {
            "binance": ExchangeConfig(
                name="binance",
                enabled=False,
                ws_url="wss://stream.binance.com:9443/ws",
                rest_url="https://api.binance.com",
            ),
            "bybit": ExchangeConfig(
                name="bybit",
                enabled=False,
                ws_url="wss://stream.bybit.com/v5/public/linear",
                rest_url="https://api.bybit.com",
            ),
            "coinbase": ExchangeConfig(
                name="coinbase",
                enabled=False,
                ws_url="wss://ws-feed.pro.coinbase.com",
                rest_url="https://api.pro.coinbase.com",
            ),
            "dhan": ExchangeConfig(
                name="dhan",
                enabled=False,
                ws_url="wss://depth-api-feed.dhan.co/twentydepth",
                rest_url="https://api.dhan.co/v2",
            ),
        }


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from YAML file if exists, otherwise from env"""
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return Settings(**data)
    return get_settings()