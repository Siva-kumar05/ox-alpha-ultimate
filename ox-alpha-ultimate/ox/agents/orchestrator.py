"""
Agent Orchestrator — runs every specialist agent with full independence.
========================================================================

Responsibilities:

* Load ``config_promax.yaml`` and construct every enabled agent with its own
  AgentConfig, risk params, capital budget and schedule.
* Feed market data to the shared bus from the paper/live brokers (DataPump).
* Execute approved signals on the right broker and keep the capital ledger
  in sync (ExecutionRouter).
* Pause agents outside their schedule or when unused, resume them when their
  window opens, restart agents that crash (bounded retries), and stop
  everything when ``promax_kill.flag`` exists.
* Evaluate the leverage ladder periodically and expire stale approvals.

Run with:  python -m ox.agents.orchestrator --config config_promax.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..brokers import make_broker
from ..core import DB
from .approvals import ApprovalGateway
from .base import (
    AgentConfig,
    AgentState,
    AgentType,
    BaseAgent,
    ResourcePool,
    RiskParams,
    SharedDataBus,
    Signal,
    Position,
)
from .capital_allocator import CapitalAllocator
from .crypto_funding import CryptoFundingArbAgent
from .crypto_meme_swing import CryptoMemeSwingAgent
from .crypto_perp import CryptoPerpAgent
from .debate import DebatePanel
from .equity_growth import EquityGrowthAgent
from .equity_momentum import EquityMomentumAgent
from .intraday_scalper import IntradayScalperAgent
from .market_maker import MarketMakerAgent
from .news_intel import NewsIntelligenceAgent
from .options_0dte import Options0DTEAgent
from .risk_coordinator import RiskCoordinator
from .social_monitor import SocialMonitorAgent

LOG = logging.getLogger("promax.orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EQUITY_TYPES = {
    AgentType.EQUITY_MOMENTUM, AgentType.EQUITY_GROWTH,
    AgentType.INTRADAY_SCALPER, AgentType.MARKET_MAKER,
    AgentType.OPTIONS_0DTE,
}
CRYPTO_TYPES = {
    AgentType.CRYPTO_PERP, AgentType.CRYPTO_FUNDING, AgentType.CRYPTO_MEME_SWING,
}
SERVICE_TYPES = {AgentType.NEWS_INTEL, AgentType.SOCIAL_MONITOR,
                 AgentType.RISK_COORDINATOR, AgentType.CAPITAL_ALLOCATOR}

AGENT_CLASS_MAP: Dict[AgentType, str] = {
    AgentType.EQUITY_MOMENTUM: "EquityMomentumAgent",
    AgentType.EQUITY_GROWTH: "EquityGrowthAgent",
    AgentType.INTRADAY_SCALPER: "IntradayScalperAgent",
    AgentType.CRYPTO_PERP: "CryptoPerpAgent",
    AgentType.CRYPTO_FUNDING: "CryptoFundingArbAgent",
    AgentType.CRYPTO_MEME_SWING: "CryptoMemeSwingAgent",
    AgentType.OPTIONS_0DTE: "Options0DTEAgent",
    AgentType.MARKET_MAKER: "MarketMakerAgent",
    AgentType.NEWS_INTEL: "NewsIntelligenceAgent",
    AgentType.SOCIAL_MONITOR: "SocialMonitorAgent",
}


# ────────────────────────────────────────────────────────────────────────────
# Market data pump
# ────────────────────────────────────────────────────────────────────────────

class DataPump:
    """Publishes broker quotes onto ``market:<symbol>`` topics.

    Paper mode: PaperBroker random-walks prices on every poll, so agents see
    a live-ish feed offline.  Live mode: the same code path with real broker
    quotes.  Crypto symbols are served by the crypto micro broker.

    Crypto ticks are enriched with slowly-varying synthetic fundamentals
    (funding rate, index price, open interest, 24h stats) in paper mode so
    the perp / funding / meme agents have something honest-shaped to chew
    on; ``exchange:funding`` / ``exchange:basis`` events are emitted on the
    same cadence for the funding-arb agent.  In live mode these fields come
    from the venue feed instead.
    """

    FUNDING_PUBLISH_EVERY = 6  # ticks between exchange:funding events

    def __init__(self, data_bus: SharedDataBus, equity_broker, crypto_broker,
                 symbols: List[str], crypto_symbols: List[str], interval_seconds: float = 5.0):
        self.data_bus = data_bus
        self.equity_broker = equity_broker
        self.crypto_broker = crypto_broker
        self.symbols = list(dict.fromkeys(symbols))
        self.crypto_symbols = list(dict.fromkeys(crypto_symbols))
        self.interval_seconds = float(interval_seconds)
        self.ticks = 0
        # Synthetic per-symbol fundamentals (paper only).
        self._fundamentals: Dict[str, Dict[str, float]] = {
            sym: {
                "funding_rate": 0.0001, "open_interest": 250_000_000.0,
                "long_short_ratio": 1.05, "volume_24h": 800_000_000.0,
                "volume_change_24h": 0.05, "price_change_24h": 0.0,
                "price_change_7d": 0.0, "market_cap": 4_000_000_000.0,
                "liquidity": 2_500_000.0, "holders": 120_000.0,
                "basis_pct": 0.0002,
            } for sym in self.crypto_symbols
        }
        self._ref_prices: Dict[str, float] = {}

    def _tick_fundamentals(self, sym: str, price: float) -> Dict[str, float]:
        import random as _random

        f = self._fundamentals[sym]
        ref = self._ref_prices.setdefault(sym, price)
        # Slow random walk keeps funding/OI plausible without exploding.
        f["funding_rate"] = max(-0.00075, min(0.00075,
            f["funding_rate"] + _random.uniform(-3e-5, 3e-5)))
        f["open_interest"] = max(1e6, f["open_interest"] * _random.uniform(0.995, 1.005))
        f["long_short_ratio"] = max(0.5, min(2.0,
            f["long_short_ratio"] + _random.uniform(-0.02, 0.02)))
        f["volume_24h"] = max(1e5, f["volume_24h"] * _random.uniform(0.98, 1.02))
        f["volume_change_24h"] = _random.uniform(-0.2, 0.25)
        f["market_cap"] = max(1e5, f["market_cap"] * _random.uniform(0.999, 1.001))
        f["liquidity"] = max(1e4, f["liquidity"] * _random.uniform(0.99, 1.01))
        f["holders"] = max(1000.0, f["holders"] * _random.uniform(0.9995, 1.0005))
        f["basis_pct"] = f["funding_rate"] * 2.0
        if ref > 0:
            f["price_change_24h"] = price / ref - 1.0
            f["price_change_7d"] = f["price_change_24h"] * 1.6
        return f

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                if self.symbols and self.equity_broker is not None:
                    quotes = self.equity_broker.ltps(self.symbols)
                    for sym, price in quotes.items():
                        self.data_bus.publish(f"market:{sym}", {
                            "symbol": sym, "price": float(price),
                            "ts": datetime.now().isoformat(), "source": "equity_broker",
                        })
                if self.crypto_symbols and self.crypto_broker is not None:
                    for sym in self.crypto_symbols:
                        price = float(self.crypto_broker.ltp(sym))
                        f = self._tick_fundamentals(sym, price)
                        self.data_bus.publish(f"market:{sym}", {
                            "symbol": sym, "price": price, "mark_price": price,
                            "index_price": price * (1.0 - f["basis_pct"]),
                            "ts": datetime.now().isoformat(), "source": "crypto_broker",
                            **f,
                        })
                        if self.ticks % self.FUNDING_PUBLISH_EVERY == 0:
                            self.data_bus.publish("exchange:funding", {
                                "exchange": "paper_perp", "symbol": sym,
                                "rate": f["funding_rate"], "ts": datetime.now().isoformat(),
                            })
                            self.data_bus.publish("exchange:funding", {
                                "exchange": "paper_spot_margin", "symbol": sym,
                                "rate": f["funding_rate"] * 0.6,
                                "ts": datetime.now().isoformat(),
                            })
                            self.data_bus.publish("exchange:basis", {
                                "exchange": "paper_perp", "symbol": sym,
                                "basis_pct": f["basis_pct"],
                                "ts": datetime.now().isoformat(),
                            })
                self.ticks += 1
            except Exception as exc:
                LOG.warning(f"Data pump tick failed: {exc.__class__.__name__}: {exc}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass


# ────────────────────────────────────────────────────────────────────────────
# Execution router
# ────────────────────────────────────────────────────────────────────────────

class ExecutionRouter:
    """Turns approved signals into broker fills and ledger records.

    Buys arrive here only after the human approval gateway released them;
    sells/closes arrive immediately (risk-reducing actions never wait).
    """

    def __init__(self, data_bus: SharedDataBus, equity_broker, crypto_broker,
                 capital_allocator: CapitalAllocator, risk_coordinator: RiskCoordinator,
                 debate_panel: Optional["DebatePanel"] = None):
        self.data_bus = data_bus
        self.equity_broker = equity_broker
        self.crypto_broker = crypto_broker
        self.allocator = capital_allocator
        self.risk_coordinator = risk_coordinator
        self.debate_panel = debate_panel
        self.agents: Dict[str, BaseAgent] = {}
        self.order_ids: Dict[tuple, str] = {}
        self.fills = 0
        self.closes = 0
        self.rejected: List[str] = []

    def watch_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.agent_id] = agent
        self.data_bus.subscribe(f"signals:{agent.agent_id}", self._on_signal)
        # Trade-memory outcome feedback: every close teaches the debate panel.
        panel = self.debate_panel

        def _on_fill(event: Dict, agent_id: str = agent.agent_id) -> None:
            if event.get("side") == "closed" and panel is not None:
                panel.memory(agent_id).update_outcome(str(event.get("symbol", "")),
                                                      float(event.get("pnl", 0.0)))

        self.data_bus.subscribe(f"fills:{agent.agent_id}", _on_fill)

    def _on_signal(self, signal: Signal) -> None:
        asyncio.get_event_loop().create_task(self._execute(signal))

    def _broker_for(self, agent_id: str):
        agent = self.agents.get(agent_id)
        if agent is not None and agent.config.agent_type in CRYPTO_TYPES and self.crypto_broker is not None:
            return self.crypto_broker
        return self.equity_broker

    async def _execute(self, signal: Signal) -> None:
        action = str(signal.action).lower()
        try:
            if action in ("buy", "open", "add"):
                await self._open(signal)
            elif action in ("sell", "close", "flatten"):
                await self._close(signal)
            elif action == "modify":
                await self._modify(signal)
        except Exception as exc:
            LOG.error(f"Execution failed for {signal.agent_id} {action} "
                      f"{signal.symbol}: {exc.__class__.__name__}: {exc}")
            self.rejected.append(f"{signal.agent_id}:{signal.symbol}:{action}")

    # ── entries ───────────────────────────────────────────────────────
    async def _open(self, signal: Signal) -> None:
        agent = self.agents.get(signal.agent_id)
        if agent is None:
            return
        broker = self._broker_for(signal.agent_id)
        price = float(signal.price) if signal.price > 0 else float(broker.ltp(signal.symbol))
        if price <= 0:
            return

        budget = self.allocator.available(signal.agent_id)
        leverage = max(1.0, float(signal.leverage))
        max_notional = budget * leverage
        desired_notional = float(signal.quantity) * price
        notional = min(desired_notional, max_notional) if desired_notional > 0 else max_notional * 0.95
        if notional <= 0:
            self.rejected.append(f"{signal.agent_id}:{signal.symbol}:no-budget")
            return

        is_crypto = broker is self.crypto_broker
        min_notional = 5.5 if is_crypto else price  # crypto venue minimum; equity >= 1 share
        if notional < min_notional:
            self.rejected.append(f"{signal.agent_id}:{signal.symbol}:below-min-notional")
            LOG.info(f"Skip {signal.agent_id} {signal.symbol}: notional {notional:.2f} below minimum")
            return

        qty = int(notional / price) if not is_crypto else round(notional / price, 6)
        if qty <= 0:
            self.rejected.append(f"{signal.agent_id}:{signal.symbol}:qty-zero")
            return

        margin = qty * price / leverage
        if not self.allocator.reserve(signal.agent_id, margin):
            self.rejected.append(f"{signal.agent_id}:{signal.symbol}:reserve-denied")
            return

        stop = float(signal.stop_loss) if signal.stop_loss else price * 0.98
        target = float(signal.take_profit) if signal.take_profit else price * 1.04
        tag = f"PROMAX_{signal.agent_id[:12]}_{signal.signal_id}"[:40]

        if is_crypto:
            receipt = self.crypto_broker.place_market(signal.symbol, "BUY", qty)
            fill_price = float(receipt["price"])
            order_id = receipt["order_id"]
        else:
            receipt = self.equity_broker.place_super_order(signal.symbol, "BUY", qty, target, stop, tag)
            confirmation = self.equity_broker.wait_super_order(receipt.order_id, timeout_seconds=10)
            if confirmation.filled_qty <= 0:
                self.allocator.release(signal.agent_id, margin)
                self.rejected.append(f"{signal.agent_id}:{signal.symbol}:unfilled")
                return
            fill_price = float(confirmation.average_price) or price
            order_id = receipt.order_id
            qty = int(confirmation.filled_qty)

        position = Position(
            agent_id=signal.agent_id, symbol=signal.symbol, side="long",
            quantity=float(qty), entry_price=fill_price, current_price=fill_price,
            entry_time=datetime.now(), stop_loss=stop, take_profit=target,
            leverage=leverage, metadata={"order_id": order_id, "signal_id": signal.signal_id},
        )
        agent.positions[signal.symbol] = position
        self.order_ids[(signal.agent_id, signal.symbol)] = order_id
        self.fills += 1
        fill_event = {
            "agent_id": signal.agent_id, "symbol": signal.symbol, "side": "long",
            "qty": float(qty), "fill_price": fill_price, "order_id": order_id,
            "leverage": leverage, "ts": datetime.now().isoformat(),
        }
        self.data_bus.publish(f"fills:{signal.agent_id}", fill_event)
        LOG.info(f"FILLED {signal.agent_id} BUY {signal.symbol} qty={qty} @ {fill_price:.2f}")

    # ── exits ─────────────────────────────────────────────────────────
    async def _close(self, signal: Signal) -> None:
        agent = self.agents.get(signal.agent_id)
        if agent is None:
            return
        position = agent.positions.pop(signal.symbol, None)
        if position is None:
            return
        broker = self._broker_for(signal.agent_id)

        if broker is self.crypto_broker:
            receipt = self.crypto_broker.place_market(signal.symbol, "SELL", position.quantity)
            exit_price = float(receipt["price"])
        else:
            receipt = self.equity_broker.exit_position(
                signal.symbol, "SELL", int(position.quantity),
                f"PROMAX_EXIT_{str(signal.metadata.get('reason', ''))[:16]}"[:40],
            )
            exit_price = float(receipt.average_price) or float(signal.price) or position.current_price

        margin = position.quantity * position.entry_price / max(1.0, position.leverage)
        self.allocator.release(signal.agent_id, margin)
        trade = self.allocator.record_trade(
            agent_id=signal.agent_id, symbol=signal.symbol, side=position.side,
            qty=position.quantity, entry_price=position.entry_price,
            exit_price=exit_price, leverage=position.leverage,
            reason=str(signal.metadata.get("reason", "signal")) if signal.metadata else "signal",
            opened=position.entry_time.isoformat(),
        )
        position.realized_pnl = trade["pnl"]
        self.order_ids.pop((signal.agent_id, signal.symbol), None)
        self.closes += 1
        self.data_bus.publish(f"fills:{signal.agent_id}", {
            "agent_id": signal.agent_id, "symbol": signal.symbol, "side": "closed",
            "qty": position.quantity, "fill_price": exit_price,
            "pnl": trade["pnl"], "ts": datetime.now().isoformat(),
        })
        LOG.info(f"CLOSED {signal.agent_id} {signal.symbol} @ {exit_price:.2f} pnl={trade['pnl']:.2f}")

    # ── modifications ─────────────────────────────────────────────────
    async def _modify(self, signal: Signal) -> None:
        order_id = self.order_ids.get((signal.agent_id, signal.symbol))
        if order_id and signal.take_profit and self.equity_broker is not None:
            try:
                self.equity_broker.modify_super_target(order_id, float(signal.take_profit))
            except Exception as exc:
                LOG.warning(f"Modify failed for {order_id}: {exc.__class__.__name__}")

    def open_notional(self) -> float:
        total = 0.0
        for agent in self.agents.values():
            for position in agent.positions.values():
                total += position.quantity * position.current_price
        return total


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """Constructs, feeds, executes, pauses, resumes and monitors all agents."""

    def __init__(self, config_path: str | Path = "config_promax.yaml",
                 db_path: str | Path | None = None, paper_prices: Optional[Dict[str, float]] = None):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle) or {}

        self.db = DB(db_path or self.config.get("db_path", "promax.db"))
        self.resource_pool = ResourcePool()
        self.data_bus = SharedDataBus()

        self.capital_allocator = CapitalAllocator(self.data_bus, self.config.get("capital", {}), db=self.db)
        self.risk_coordinator = RiskCoordinator(self.data_bus, self.config.get("risk", {}))
        if self.config.get("ladder", {}).get("enabled", True):
            self.risk_coordinator.attach_ladder(self.capital_allocator, self.config.get("ladder", {}))

        self.approval_gateway = ApprovalGateway(
            self.db, ttl_seconds=int(self.config.get("approval", {}).get("ttl_seconds", 300))
        )
        self.resource_pool.acquire("approval_gateway", lambda: self.approval_gateway)
        # Shared DB handle for intelligence agents (news persistence) and
        # anything else that must not open its own connection per process.
        self.resource_pool.acquire("db", lambda: self.db)
        # Deterministic bull/bear debate + per-agent mistake memory.
        self.debate_panel = DebatePanel(
            threshold=float(self.config.get("debate", {}).get("threshold", 0.15)),
            state_dir=PROJECT_ROOT / "state" / "promax",
        )
        self.resource_pool.acquire("debate_panel", lambda: self.debate_panel)

        # Broker shim configs (paper default; live requires the legacy
        # runtime's env gate — see ox.core.Cfg._validate).
        shim = {
            "mode": self.config.get("mode", "paper"),
            "platform": self.config.get("platform", "paper"),
            "costs": {
                "slippage_pct": 0.03, "brokerage_per_order": 20, "stt_pct": 0.025,
                "txn_charge_pct": 0.00297, "gst_pct": 18.0, "sebi_fee_pct": 0.0001,
                "stamp_duty_pct": 0.003,
            },
            "order_flow": {
                "enabled": True, "primary": False, "depth_levels": 20,
                "max_staleness_seconds": 2.0, "min_observations": 300,
                "min_side_notional": 50000, "max_spread_bps": 12.0,
                "min_book_imbalance": 0.12, "min_flow_imbalance": 0.04,
                "min_microprice_edge_bps": 0.5, "pressure_ema_alpha": 0.2,
                "min_pressure_ema": 0.08, "min_positive_streak": 3,
                "min_liquidity_score": 0.0, "require_replay_validation": False,
                "replay_min_signals": 30, "replay_horizon_candles": 5,
                "replay_min_hit_rate": 0.5, "replay_min_mean_return_bps": 0.0,
                "replay_max_records": 10000,
            },
            "paper_seed": int(self.config.get("paper_seed", 7)),
            "paper_prices": dict(self.config.get("paper_prices", {})),
        }
        if paper_prices:
            shim["paper_prices"].update(paper_prices)
        self.equity_broker = make_broker(shim, self.db)
        self.equity_broker.login()
        from ..crypto import CryptoMicroBroker
        self.crypto_broker = CryptoMicroBroker(shim, self.db)
        self.crypto_broker.login()

        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._load_agent_configs()

        self.router = ExecutionRouter(
            self.data_bus, self.equity_broker, self.crypto_broker,
            self.capital_allocator, self.risk_coordinator, debate_panel=self.debate_panel,
        )
        symbols: List[str] = []
        crypto_symbols: List[str] = []
        for agent_id, cfg in self.agent_configs.items():
            for sym in cfg.symbols:
                if cfg.agent_type in CRYPTO_TYPES:
                    crypto_symbols.append(sym)
                else:
                    symbols.append(sym)
        pump_cfg = self.config.get("data_pump", {})
        self.data_pump = DataPump(
            self.data_bus, self.equity_broker, self.crypto_broker,
            symbols, crypto_symbols,
            interval_seconds=float(pump_cfg.get("interval_seconds", 5.0)),
        )

        self.stop_event = asyncio.Event()
        self._restart_counts: Dict[str, int] = {}
        self._pause_reasons: Dict[str, str] = {}
        self._last_ladder_eval = datetime.min
        self._tasks: List[asyncio.Task] = []

    # ── config ────────────────────────────────────────────────────────
    def _load_agent_configs(self) -> None:
        agents_cfg = self.config.get("agents", {})
        for agent_id, raw in agents_cfg.items():
            if not raw.get("enabled", True):
                continue
            try:
                agent_type = AgentType(raw.get("type", agent_id))
            except ValueError:
                LOG.warning(f"Unknown agent type for {agent_id}: {raw.get('type')}")
                continue
            risk = raw.get("risk", {})
            self.agent_configs[agent_id] = AgentConfig(
                agent_id=agent_id,
                agent_type=agent_type,
                name=raw.get("name", agent_id.replace("_", " ").title()),
                symbols=list(raw.get("symbols", [])),
                risk_params=RiskParams(**risk),
                enabled=True,
                priority=int(raw.get("priority", 5)),
                custom_params=dict(raw.get("params", {})),
            )

    def create_agent(self, agent_id: str) -> Optional[BaseAgent]:
        config = self.agent_configs.get(agent_id)
        if not config:
            return None
        class_name = AGENT_CLASS_MAP.get(config.agent_type)
        if class_name is None:
            LOG.warning(f"No agent class for type {config.agent_type}")
            return None
        agent_class = globals()[class_name]
        return agent_class(
            config=config,
            resource_pool=self.resource_pool,
            data_bus=self.data_bus,
            risk_coordinator=self.risk_coordinator,
            capital_allocator=self.capital_allocator,
        )

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start_all(self) -> None:
        self.stop_event.clear()
        for agent_id, cfg in sorted(self.agent_configs.items(), key=lambda kv: kv[1].priority):
            agent = self.create_agent(agent_id)
            if agent is None:
                continue
            if await agent.start():
                self.agents[agent_id] = agent
                self.router.watch_agent(agent)
            else:
                LOG.error(f"Agent failed to start: {agent_id}")
        self._tasks = [
            asyncio.create_task(self.data_pump.run(self.stop_event), name="data_pump"),
            asyncio.create_task(self._monitor_loop(), name="monitor"),
        ]
        LOG.info(f"Orchestrator running with {len(self.agents)} agents")

    async def stop_all(self) -> None:
        self.stop_event.set()
        for task in self._tasks:
            task.cancel()
        for agent_id in list(self.agents):
            await self.stop_agent(agent_id)
        LOG.info("Orchestrator stopped")

    async def stop_agent(self, agent_id: str) -> None:
        agent = self.agents.pop(agent_id, None)
        if agent:
            await agent.stop()

    async def pause_agent(self, agent_id: str, reason: str = "manual") -> bool:
        agent = self.agents.get(agent_id)
        if agent and agent.state == AgentState.RUNNING:
            await agent.pause()
            self._pause_reasons[agent_id] = reason
            self.data_bus.publish("system:agent_paused", {"agent_id": agent_id, "reason": reason})
            LOG.info(f"Paused {agent_id} ({reason})")
            return True
        return False

    async def resume_agent(self, agent_id: str) -> bool:
        agent = self.agents.get(agent_id)
        if agent and agent.state == AgentState.PAUSED:
            await agent.resume()
            self._pause_reasons.pop(agent_id, None)
            self.data_bus.publish("system:agent_resumed", {"agent_id": agent_id})
            LOG.info(f"Resumed {agent_id}")
            return True
        return False

    # ── schedules ─────────────────────────────────────────────────────
    def _in_schedule(self, agent_id: str) -> bool:
        raw = self.config.get("agents", {}).get(agent_id, {})
        window = raw.get("active_hours", {})
        if not window or window.get("always"):
            return True
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        days = window.get("days", ["Mon", "Tue", "Wed", "Thu", "Fri"])
        if now.strftime("%a") not in days:
            return False
        start = str(window.get("start", "00:00"))
        end = str(window.get("end", "23:59"))
        start_h, start_m = (int(x) for x in start.split(":"))
        end_h, end_m = (int(x) for x in end.split(":"))
        minute = now.hour * 60 + now.minute
        if (start_h, start_m) <= (end_h, end_m):
            return start_h * 60 + start_m <= minute < end_h * 60 + end_m
        return minute >= start_h * 60 + start_m or minute < end_h * 60 + end_m

    # ── monitor loop ──────────────────────────────────────────────────
    async def _monitor_loop(self) -> None:
        interval = float(self.config.get("monitor_interval_seconds", 10))
        ladder_minutes = float(self.config.get("ladder", {}).get("eval_minutes", 30))
        while not self.stop_event.is_set():
            try:
                # Kill switch
                if (PROJECT_ROOT / "promax_kill.flag").exists():
                    LOG.critical("promax_kill.flag detected — stopping everything")
                    await self.stop_all()
                    return

                self.approval_gateway.expire_stale()
                self._resubmit_human_approved_intents()


                # Schedule-based auto pause/resume — unused agents park.
                for agent_id in list(self.agents):
                    if self._in_schedule(agent_id):
                        if (self.agents[agent_id].state == AgentState.PAUSED
                                and self._pause_reasons.get(agent_id) == "off_schedule"):
                            await self.resume_agent(agent_id)
                    else:
                        await self.pause_agent(agent_id, reason="off_schedule")

                # Health: restart crashed agents (bounded).
                for agent_id in list(self.agents):
                    agent = self.agents[agent_id]
                    if agent.state == AgentState.ERROR and self._restart_counts.get(agent_id, 0) < 3:
                        self._restart_counts[agent_id] = self._restart_counts.get(agent_id, 0) + 1
                        await agent.stop()
                        fresh = self.create_agent(agent_id)
                        if fresh and await fresh.start():
                            self.agents[agent_id] = fresh
                            self.router.watch_agent(fresh)
                            LOG.warning(f"Restarted agent {agent_id}")

                # Equity + risk sync
                self.risk_coordinator.portfolio_positions = {
                    f"{aid}:{sym}": pos
                    for aid, agent in self.agents.items()
                    for sym, pos in agent.positions.items()
                }
                await self.risk_coordinator.update_portfolio_equity(self.capital_allocator.equity())

                # Leverage ladder evaluation
                since_eval = (datetime.now() - self._last_ladder_eval).total_seconds()
                if since_eval >= ladder_minutes * 60:
                    self._last_ladder_eval = datetime.now()
                    for agent_id in self.agents:
                        level, action, reason = self.risk_coordinator.ladder.evaluate(agent_id)
                        self.data_bus.publish("ladder:level", {
                            "agent_id": agent_id, "level": level, "action": action, "reason": reason,
                        })

                self.data_bus.publish("system:status", self.get_system_status())
            except Exception as exc:
                LOG.error(f"Monitor loop error: {exc.__class__.__name__}: {exc}")
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def _resubmit_human_approved_intents(self) -> None:
        """Publish signals for intents a human approved after submission.

        The CLI (or Telegram bot) writes the decision to sqlite; this loop
        picks it up within one monitor interval and re-publishes the signal
        to the router.  Executed iids are marked in the DB kv table so a
        restart never double-fills an old approval.
        """
        done = {str(k) for (k,) in self.db.q(
            "SELECT k FROM kv WHERE k LIKE 'intent_done:%'")}
        pending = self.approval_gateway.human_approved_unexecuted(done)
        for intent in pending:
            agent = self.agents.get(intent["agent"])
            if agent is None:
                continue
            if agent.positions.get(intent["symbol"]) is not None:
                self.db.kv_set(f"intent_done:{intent['iid']}", "skipped_open_position")
                continue
            signal = Signal(
                agent_id=intent["agent"], symbol=intent["symbol"],
                action=intent["action"], strength=0.8,
                price=float(intent["price"] or 0.0), quantity=float(intent["qty"] or 0),
                stop_loss=intent["stop_loss"], take_profit=intent["take_profit"],
                leverage=float(intent["leverage"] or 1.0),
                metadata={"reason": intent["reason"], "approved_iid": intent["iid"]},
            )
            self.db.kv_set(f"intent_done:{intent['iid']}", "published")
            self.data_bus.publish(f"signals:{signal.agent_id}", signal)
            LOG.info(f"Published human-approved intent {intent['iid']} "
                     f"({intent['agent']} {intent['action']} {intent['symbol']})")

    # ── status ────────────────────────────────────────────────────────
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._tasks),
            "capital": self.capital_allocator.get_status(),
            "data_pump_ticks": self.data_pump.ticks,
            "execution": {"fills": self.router.fills, "closes": self.router.closes,
                          "rejected": len(self.router.rejected)},
            "open_notional": self.router.open_notional(),
            "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
            "ladder": self.risk_coordinator.ladder.report() if self.risk_coordinator.ladder else {},
            "pending_approvals": len(self.approval_gateway.list_intents("PENDING")),
            "timestamp": datetime.now().isoformat(),
        }

    async def run_forever(self, max_seconds: Optional[float] = None) -> None:
        await self.start_all()
        try:
            if max_seconds:
                await asyncio.wait_for(self.stop_event.wait(), timeout=max_seconds)
            else:
                await self.stop_event.wait()
        except asyncio.TimeoutError:
            pass
        finally:
            await self.stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="OX-ALPHA multi-agent orchestrator")
    parser.add_argument("--config", default="config_promax.yaml")
    parser.add_argument("--seconds", type=float, default=None,
                        help="run for N seconds then stop (smoke/CI)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s")
    orchestrator = AgentOrchestrator(config_path=args.config)
    try:
        asyncio.run(orchestrator.run_forever(max_seconds=args.seconds))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
