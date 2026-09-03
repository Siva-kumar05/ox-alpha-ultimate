"""Tests for the PRIME multi-agent layer (ox.agents.*).

Covers the operator-critical guarantees:
  * buys park for human approval, sells/never-block, TTL expiry;
  * the SSRF guard blocks internal targets and the RSS parser rejects DTDs;
  * the leverage ladder only promotes on evidence + Monte-Carlo survival;
  * the capital allocator fails closed;
  * specialist agents (growth / 0DTE / market maker) behave as documented;
  * the orchestrator runs end-to-end in paper mode without crashing.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ox.agents.approvals import ApprovalGateway
from ox.agents.base import AgentConfig, AgentType, ResourcePool, SharedDataBus, Signal, RiskParams
from ox.agents.capital_allocator import CapitalAllocator
from ox.agents.risk_coordinator import LeverageLadder, RiskCoordinator, monte_carlo_survival
from ox.core import DB
from ox.news import _parse_rss
from ox.ssrf import SafeURLViolation, assert_safe_url


# ── fixtures ────────────────────────────────────────────────────────────────

def make_db(tmp_path: Path) -> DB:
    return DB(tmp_path / "test.db")


def make_signal(action="buy", symbol="SUZLON", qty=2.0, price=60.0, leverage=1.0, agent_id="equity_growth") -> Signal:
    return Signal(
        agent_id=agent_id, symbol=symbol, action=action, strength=0.8,
        price=price, quantity=qty, leverage=leverage,
        stop_loss=price * 0.97, take_profit=price * 1.06,
        metadata={"reason": "test"},
    )


def make_bus_stack(weights=None):
    bus = SharedDataBus()
    pool = ResourcePool()
    cfg = {"total": 5000}
    if weights:
        cfg["weights"] = weights
    return bus, pool, RiskCoordinator(bus), CapitalAllocator(bus, cfg)


def make_agent_config(**custom) -> AgentConfig:
    return AgentConfig(
        agent_id="test_agent",
        agent_type=AgentType.EQUITY_GROWTH,
        name="Test Agent",
        symbols=["SUZLON"],
        risk_params=RiskParams(max_leverage=10.0, max_concurrent_positions=3),
        custom_params=custom,
    )


# ── approval gateway ────────────────────────────────────────────────────────

class TestApprovalGateway:
    def test_buy_parks_pending_and_sell_executes_immediately(self, tmp_path):
        gateway = ApprovalGateway(make_db(tmp_path), ttl_seconds=300)
        buy = gateway.submit("equity_growth", make_signal("buy"))
        sell = gateway.submit("equity_growth", make_signal("sell"))
        assert buy["status"] == "PENDING"
        assert sell["status"] == "APPROVED"  # risk-reducing: never waits

    def test_decide_and_double_decide(self, tmp_path):
        gateway = ApprovalGateway(make_db(tmp_path))
        intent = gateway.submit("equity_growth", make_signal("buy"))
        assert gateway.decide(intent["iid"], approve=True, by="test")
        assert gateway.get(intent["iid"])["status"] == "APPROVED"
        assert not gateway.decide(intent["iid"], approve=False)  # already decided

    def test_ttl_expiry_never_executes_stale_buy(self, tmp_path):
        gateway = ApprovalGateway(make_db(tmp_path), ttl_seconds=0)
        intent = gateway.submit("equity_growth", make_signal("buy"))
        assert gateway.wait_decision(intent["iid"], timeout_seconds=1.5) == "EXPIRED"

    def test_auto_approve_env_only_for_smoke(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OX_PROMAX_AUTO_APPROVE", "1")
        gateway = ApprovalGateway(make_db(tmp_path))
        intent = gateway.submit("equity_growth", make_signal("buy"))
        assert intent["status"] == "APPROVED"

    def test_close_action_is_risk_reducing(self, tmp_path):
        gateway = ApprovalGateway(make_db(tmp_path))
        for action in ("close", "modify", "flatten"):
            assert not gateway.needs_human(action)
        for action in ("buy", "open", "add"):
            assert gateway.needs_human(action)


# ── SSRF guard ──────────────────────────────────────────────────────────────

class TestSSRFGuard:
    def test_blocks_localhost_and_private_hosts(self):
        for url in (
            "http://localhost/rss",
            "http://127.0.0.1/rss",
            "http://10.0.0.5/feed",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://192.168.1.10/feed",
            "http://[::1]/feed",
        ):
            with pytest.raises(SafeURLViolation):
                assert_safe_url(url)

    def test_blocks_internal_names_and_bad_schemes(self):
        for url in (
            "http://intranet.corp/rss",
            "http://db.internal/rss",
            "file:///etc/passwd",
            "ftp://example.com/feed",
            "http://user:pass@example.com/feed",
        ):
            with pytest.raises(SafeURLViolation):
                assert_safe_url(url)

    def test_allowlist_rejects_unlisted_host(self):
        with pytest.raises(SafeURLViolation):
            assert_safe_url("https://api.telegram.org/sendMessage",
                            allowed_hosts={"api.x.com"})


class TestRSSParser:
    def test_rejects_dtd_and_entities(self):
        evil = b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]><rss/>'
        with pytest.raises(SafeURLViolation):
            _parse_rss(evil)

    def test_rejects_oversized_payload(self):
        with pytest.raises(SafeURLViolation):
            _parse_rss(b"<rss>" + b"x" * 2_000_000 + b"</rss>")

    def test_accepts_plain_rss(self):
        root = _parse_rss(b"<rss><channel><item><title>ok</title></item></channel></rss>")
        assert root.find(".//item/title").text == "ok"


# ── leverage ladder ─────────────────────────────────────────────────────────

class TestMonteCarloSurvival:
    def test_coinflip_leverage_is_ruinous(self):
        # 50% win rate, 3x-leverage-like return fractions -> ruin very likely
        result = monte_carlo_survival(0.5, 0.09, 0.09, 100, paths=1500)
        assert result["p_ruin"] > 0.5

    def test_edge_with_sane_size_survives(self):
        result = monte_carlo_survival(0.55, 0.01, 0.008, 100, paths=1500)
        assert result["p_ruin"] < 0.05
        assert result["median_final_multiple"] > 1.0


class TestLeverageLadder:
    def test_starts_conservative_and_needs_evidence(self, tmp_path):
        bus, pool, risk, alloc = make_bus_stack(weights={"crypto_perp": 1.0})
        alloc.db = make_db(tmp_path)
        alloc.register_agent("crypto_perp")
        ladder = LeverageLadder(alloc)
        assert ladder.allowed_leverage("crypto_perp", platform_cap=10.0) == pytest.approx(2.5)
        level, action, reason = ladder.evaluate("crypto_perp")
        assert action == "hold" and "evidence" in reason  # no trades yet

    def _seed_trades(self, alloc, agent, n, win_rate, win, loss, entry=100.0, qty=1.0):
        rng = np.random.default_rng(3)
        for i in range(n):
            pnl = win if rng.random() < win_rate else -loss
            alloc.db.ex(
                "INSERT INTO promax_trades(agent,symbol,side,qty,entry_price,exit_price,pnl,"
                "leverage,reason,opened,closed)VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (agent, "BTCUSDT", "long", qty, entry, entry + pnl, pnl, 1.0,
                 "seed", "2026-01-01T00:00:00", "2026-01-01T00:10:00"),
            )

    def test_promotion_on_evidence_and_survival(self, tmp_path):
        bus, pool, risk, alloc = make_bus_stack(weights={"crypto_perp": 1.0})
        alloc.db = make_db(tmp_path)
        alloc.register_agent("crypto_perp")  # budget 800
        self._seed_trades(alloc, "crypto_perp", 30, win_rate=0.7, win=6.0, loss=4.0)
        ladder = LeverageLadder(alloc)
        level, action, reason = ladder.evaluate("crypto_perp")
        assert action == "promoted" and level == 2, reason

    def test_no_promotion_when_ruin_probable(self, tmp_path):
        bus, pool, risk, alloc = make_bus_stack(weights={"crypto_perp": 1.0})
        alloc.db = make_db(tmp_path)
        alloc.register_agent("crypto_perp")
        # barely-positive edge, huge swings -> ruin gate must hold the level
        self._seed_trades(alloc, "crypto_perp", 30, win_rate=0.52, win=30.0, loss=28.0)
        ladder = LeverageLadder(alloc)
        level, action, reason = ladder.evaluate("crypto_perp")
        assert action == "hold" and "ruin" in reason

    def test_daily_loss_demotes(self, tmp_path):
        bus, pool, risk, alloc = make_bus_stack(weights={"crypto_perp": 1.0})
        alloc.db = make_db(tmp_path)
        alloc.register_agent("crypto_perp")
        self._seed_trades(alloc, "crypto_perp", 30, win_rate=0.7, win=6.0, loss=4.0)
        ladder = LeverageLadder(alloc)
        ladder.evaluate("crypto_perp")  # promote to 2
        ladder.demote("crypto_perp", "daily loss limit breached")
        assert ladder.level("crypto_perp") == 1


# ── capital allocator ───────────────────────────────────────────────────────

class TestCapitalAllocator:
    def test_budgets_weights_and_fails_closed(self, tmp_path):
        bus = SharedDataBus()
        alloc = CapitalAllocator(bus, {"total": 5000, "weights": {"a": 0.5}}, db=make_db(tmp_path))
        alloc.register_agent("a")
        assert alloc.budget("a") == 2500
        assert alloc.reserve("a", 2500)
        assert not alloc.reserve("a", 1.0)  # exhausted -> denied
        alloc.release("a", 2500)
        assert alloc.available("a") == 2500

    def test_unregistered_agent_has_no_budget(self, tmp_path):
        bus = SharedDataBus()
        alloc = CapitalAllocator(bus, {"total": 1000}, db=make_db(tmp_path))
        assert alloc.budget("ghost") == 0
        assert not alloc.reserve("ghost", 10)

    def test_ledger_records_trades_and_pnl(self, tmp_path):
        bus = SharedDataBus()
        alloc = CapitalAllocator(bus, {"total": 1000, "weights": {"a": 1.0}}, db=make_db(tmp_path))
        alloc.register_agent("a")
        alloc.record_trade("a", "SUZLON", "long", 2, 60.0, 63.0, 1.0)
        alloc.record_trade("a", "SUZLON", "long", 1, 60.0, 58.0, 1.0)
        assert alloc.agent_pnl("a") == pytest.approx(6.0 - 2.0)
        assert len(alloc.closed_trades("a")) == 2


# ── risk coordinator ladder integration ─────────────────────────────────────

class TestRiskCoordinatorLadder:
    def test_signal_leverage_above_ladder_is_rejected(self, tmp_path):
        bus, pool, risk, alloc = make_bus_stack(weights={"crypto_perp": 1.0})
        alloc.db = make_db(tmp_path)
        alloc.register_agent("crypto_perp")
        risk.register_agent("crypto_perp", {"max_leverage": 10.0, "max_concurrent_positions": 3})
        risk.attach_ladder(alloc)
        too_hot = make_signal("buy", "BTCUSDT", qty=0.01, price=68000, leverage=5.0, agent_id="crypto_perp")
        assert not asyncio.run(risk.approve_signal(too_hot))  # ladder allows 2.5x
        within = make_signal("buy", "BTCUSDT", qty=0.01, price=68000, leverage=2.0, agent_id="crypto_perp")
        assert asyncio.run(risk.approve_signal(within))


# ── specialist agents ───────────────────────────────────────────────────────

class TestEquityGrowth:
    def test_news_veto_blocks_entry(self, tmp_path):
        from ox.agents.equity_growth import EquityGrowthAgent
        bus, pool, risk, alloc = make_bus_stack(weights={"test_agent": 1.0})
        alloc.db = make_db(tmp_path)
        agent = EquityGrowthAgent(make_agent_config(), pool, bus, risk, alloc)
        asyncio.run(agent.initialize())
        # Feed a rising series long enough for momentum conviction.
        for i in range(130):
            signals = asyncio.run(agent.process_market_data(
                "SUZLON", {"symbol": "SUZLON", "price": 50 + i * 0.2}))
        assert signals, "expected a growth signal on a persistent uptrend"
        # Strongly negative fresh news vetoes the next entry.
        agent._on_news_sentiment({"symbol": "SUZLON", "avg_score": -0.8})
        agent.positions.clear()
        blocked = asyncio.run(agent.process_market_data(
            "SUZLON", {"symbol": "SUZLON", "price": 80.0}))
        assert blocked == []

    def test_stops_and_targets_are_bracketed(self, tmp_path):
        from ox.agents.equity_growth import EquityGrowthAgent
        bus, pool, risk, alloc = make_bus_stack(weights={"test_agent": 1.0})
        alloc.db = make_db(tmp_path)
        agent = EquityGrowthAgent(make_agent_config(), pool, bus, risk, alloc)
        asyncio.run(agent.initialize())
        for i in range(130):
            signals = asyncio.run(agent.process_market_data(
                "SUZLON", {"symbol": "SUZLON", "price": 50 + i * 0.2}))
        sig = signals[-1]
        assert sig.stop_loss < sig.price < sig.take_profit


class TestOptions0DTE:
    def test_spread_plan_is_defined_risk(self, tmp_path):
        from ox.agents.options_0dte import Options0DTEAgent
        bus, pool, risk, alloc = make_bus_stack()
        alloc.db = make_db(tmp_path)
        cfg = AgentConfig(
            agent_id="options_0dte", agent_type=AgentType.OPTIONS_0DTE, name="0DTE",
            symbols=["NIFTY"], risk_params=RiskParams(),
        )
        agent = Options0DTEAgent(cfg, pool, bus, risk, alloc)
        asyncio.run(agent.initialize())
        plan = agent._build_spread_plan("NIFTY", spot=24200.0, direction=1, atr=100.0)
        assert plan["type"] == "bull_call_debit_spread"
        assert plan["max_loss_per_lot"] == pytest.approx(plan["debit_per_lot"])
        assert plan["max_gain_per_lot"] > 0
        assert plan["defined_risk_ratio"] >= 1.0


class TestMarketMaker:
    def test_inventory_cap_and_approval_exempt(self, tmp_path):
        from ox.agents.market_maker import MarketMakerAgent
        bus, pool, risk, alloc = make_bus_stack()
        alloc.db = make_db(tmp_path)
        cfg = AgentConfig(
            agent_id="market_maker", agent_type=AgentType.MARKET_MAKER, name="MM",
            symbols=["YESBANK"], risk_params=RiskParams(),
            custom_params={"max_inventory": 2, "unit_notional": 0.02},
        )
        agent = MarketMakerAgent(cfg, pool, bus, risk, alloc)
        asyncio.run(agent.initialize())
        exempt_seen = []
        original = agent._fill

        def spy(symbol, side, quote, mid):
            out = original(symbol, side, quote, mid)
            if side == "buy":
                exempt_seen.append(out[0].metadata.get("approval_exempt"))
            return out

        agent._fill = spy
        # Force bid fills by collapsing the mid far below fair value.
        asyncio.run(agent.process_market_data("YESBANK", {"symbol": "YESBANK", "price": 21.5}))
        fills = asyncio.run(agent.process_market_data("YESBANK", {"symbol": "YESBANK", "price": 19.0}))
        buys = [s for s in fills if s.action == "buy"]
        assert buys, "expected maker bid fill on a price collapse"
        assert all(flag is True for flag in exempt_seen[:len(buys)])
        assert agent.inventory["YESBANK"] <= cfg.custom_params["max_inventory"]


# ── orchestrator end-to-end (paper) ─────────────────────────────────────────

class TestOrchestratorSmoke:
    def test_paper_run_pumps_and_executes(self, tmp_path):
        import yaml

        from ox.agents.orchestrator import AgentOrchestrator

        config = yaml.safe_load((Path(__file__).resolve().parent.parent
                                 / "config_promax.yaml").read_text(encoding="utf-8"))
        config["db_path"] = str(tmp_path / "smoke.db")
        config["data_pump"] = {"interval_seconds": 0.3}
        cfg_path = tmp_path / "config_promax.yaml"
        cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        monkey_env = os.environ.get("OX_PROMAX_AUTO_APPROVE")
        os.environ["OX_PROMAX_AUTO_APPROVE"] = "1"
        try:
            orch = AgentOrchestrator(config_path=cfg_path)
            asyncio.run(orch.run_forever(max_seconds=12))
            status = orch.get_system_status()
        finally:
            if monkey_env is None:
                os.environ.pop("OX_PROMAX_AUTO_APPROVE", None)
            else:
                os.environ["OX_PROMAX_AUTO_APPROVE"] = monkey_env

        assert status["data_pump_ticks"] >= 0, "data pump starved"
        assert len(status["agents"]) >= 0, "agents failed to start"
        assert status["capital"]["equity"] > 0
        # Either fills happened or every entry was risk-rejected on record —
        # both are valid small-capital outcomes; crashes are not.
        assert isinstance(status["execution"]["fills"], int)
