"""ExecutionRouter NSE/equity live tests: real router + real DhanBroker, scripted transport.

The promax router's NSE side was never test-driven: only the paper smoke ever
ran it, and only the crypto branch had been exercised behind a fake venue.
This file drives the REAL ``ExecutionRouter`` and the REAL ``DhanBroker``
against the scripted ``_FakeSession`` HTTP transport reused from
``test_live_broker_contract`` — same fakes, no new scaffolding.

It also proves the equity branch carries (and now fixes) the same two failure
modes the crypto pass found, with the same guards:

  * a venue failure after margin was reserved releases the reservation
    (placement dies on the wire, or confirmation dies mid-poll);
  * a failed exit restores the local position so state matches broker truth;
  * no phantom ledger entries, no fabricated fills, rejections recorded;
  * the unfilled-confirmation path (pre-existing guard) still releases;
  * equity signals never touch the crypto broker (MustNotTouch sentinel).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import requests
import yaml

import pytest

from ox.agents.base import AgentType, SharedDataBus
from ox.agents.capital_allocator import CapitalAllocator
from ox.agents.orchestrator import ExecutionRouter
from ox.brokers import BrokerError, DhanBroker
from ox.core import DB
from support import _AttrDict, _FakeSession, _LTP_PAYLOAD, MustNotTouch, buy, close, make_agent


def make_equity_harness(monkeypatch, tmp_path):
    """Real bus + allocator + DhanBroker (scripted session) + real router."""
    monkeypatch.setenv("OX_AUDIT_KEY", "live-contract-audit-key-at-least-thirty-two-chars")
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
    cfg = _AttrDict(raw)
    cfg.update({
        "root": str(tmp_path),
        "db_path": str(tmp_path / "test.db"),
        "security_map": {"TCS": "1333"},
        "execution": dict(raw["execution"], order_confirm_timeout_seconds=2),
    })
    db = DB(tmp_path / "test.db")
    session = _FakeSession()
    broker = DhanBroker(cfg, db)
    broker.client_id = "TEST-CLIENT-ID"
    broker.session = session
    broker._set_token("live-test-access-token-not-dummy-32chars")

    bus = SharedDataBus()
    allocator = CapitalAllocator(bus, config={"total": 500_000.0, "weights": {"mom1": 1.0}}, db=None)
    agent = make_agent("mom1", AgentType.EQUITY_MOMENTUM)
    router = ExecutionRouter(bus, broker, MustNotTouch(), allocator, risk_coordinator=object())
    router.watch_agent(agent)
    allocator.register_agent("mom1")  # budget 500_000
    return router, broker, allocator, bus, session, db, agent


def _script_open(session, order_id="SO1", status="PENDING"):
    session.script("POST", "/marketfeed/ltp", payload=_LTP_PAYLOAD)
    session.script("POST", "/super/orders", payload={
        "orderId": order_id, "orderStatus": status, "filledQty": 0, "averageTradedPrice": 0.0,
    })


def _script_confirm(session, status="TRADED", filled_qty=75, avg=1100.05, order_id="SO1"):
    session.script("GET", "/super/orders", payload=[{
        "orderId": order_id, "orderStatus": status, "filledQty": filled_qty,
        "averageTradedPrice": avg,
    }])


# ── happy paths: real Dhan payloads through the router ───────────────────────

def test_equity_full_entry_through_real_dhan_payload(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    _script_open(session)
    _script_confirm(session)

    asyncio.run(router._execute(buy("mom1", "TCS", 1100.0, 75)))

    pos = agent.positions["TCS"]
    assert pos.quantity == 75
    assert pos.entry_price == 1100.05  # confirmed fill, not the signal price
    assert pos.leverage == 1.0
    assert router.fills == 1
    assert router.order_ids[("mom1", "TCS")] == "SO1"
    assert allocator.reserved["mom1"] == 82_500.0  # 75 * 1100 / 1x
    # the order that went out is the real Dhan Super Order payload
    order_body = dict(session.calls[1][2])
    assert order_body["securityId"] == "1333"
    assert order_body["quantity"] == 75
    assert order_body["productType"] == "INTRADAY"
    assert order_body["targetPrice"] == 1144.0   # price * 1.04
    assert order_body["stopLossPrice"] == 1078.0  # price * 0.98
    assert session.call_paths() == [
        ("POST", "/marketfeed/ltp"), ("POST", "/super/orders"), ("GET", "/super/orders"),
    ]
    assert bus.get("fills:mom1")["side"] == "long"


def test_equity_close_goes_out_through_real_dhan(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    _script_open(session)
    _script_confirm(session)
    asyncio.run(router._execute(buy("mom1", "TCS", 1100.0, 75)))
    session.script("POST", "/orders", payload={
        "orderId": "EO1", "orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 1103.0,
    })

    asyncio.run(router._execute(close("mom1", "TCS", 1103.0, 75)))

    assert "TCS" not in agent.positions
    assert allocator.reserved["mom1"] == 0.0
    assert router.closes == 1
    assert router.order_ids == {}
    fill = bus.get("fills:mom1")
    assert fill["side"] == "closed"
    assert fill["pnl"] == pytest.approx((1103.0 - 1100.05) * 75)
    trade = bus.get("capital:trade")
    assert trade["qty"] == 75
    assert trade["pnl"] == pytest.approx((1103.0 - 1100.05) * 75)
    # the exit that went out is a real Dhan market SELL
    exit_body = [c[2] for c in session.calls if c[0] == "POST" and c[1] == "/orders"][0]
    assert exit_body["transactionType"] == "SELL"
    assert exit_body["securityId"] == "1333"
    assert exit_body["quantity"] == 75


# ── fail-closed: same guards as the crypto branch ────────────────────────────

def test_equity_open_placement_failure_releases_reservation(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    session.script("POST", "/marketfeed/ltp", payload=_LTP_PAYLOAD)
    session.script("POST", "/super/orders", exc=requests.exceptions.ConnectionError("reset"))

    signal = buy("mom1", "TCS", 1100.0, 75)
    with pytest.raises(BrokerError):
        asyncio.run(router._open(signal))
    # the reserved margin was released: a failed open must not leak capital
    assert allocator.reserved["mom1"] == 0.0
    assert "TCS" not in agent.positions
    assert router.fills == 0
    assert bus.get("fills:mom1") is None

    asyncio.run(router._execute(signal))
    assert router.rejected == ["mom1:TCS:buy"]


def test_equity_open_confirmation_failure_releases_reservation(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    _script_open(session)
    session.script("GET", "/super/orders", exc=requests.exceptions.ConnectionError("reset"))

    signal = buy("mom1", "TCS", 1100.0, 75)
    with pytest.raises(BrokerError):
        asyncio.run(router._open(signal))
    assert allocator.reserved["mom1"] == 0.0
    assert "TCS" not in agent.positions
    assert router.fills == 0

    asyncio.run(router._execute(signal))
    assert router.rejected == ["mom1:TCS:buy"]


def test_equity_open_unfilled_confirmation_still_releases_and_rejects(monkeypatch, tmp_path):
    # Pre-existing guard: a confirmation that reports zero fills must release
    # the reservation and record the rejection — never book a phantom position.
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    _script_open(session)
    _script_confirm(session, status="TRADED", filled_qty=0, avg=0.0)

    asyncio.run(router._execute(buy("mom1", "TCS", 1100.0, 75)))

    assert router.rejected == ["mom1:TCS:unfilled"]
    assert allocator.reserved["mom1"] == 0.0
    assert "TCS" not in agent.positions
    assert router.fills == 0
    assert bus.get("fills:mom1") is None


def test_equity_close_failure_restores_position_matching_broker_truth(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    _script_open(session)
    _script_confirm(session)
    asyncio.run(router._execute(buy("mom1", "TCS", 1100.0, 75)))
    session.script("POST", "/orders", exc=requests.exceptions.ConnectionError("reset"))

    signal = close("mom1", "TCS", 1103.0, 75)
    with pytest.raises(BrokerError):
        asyncio.run(router._close(signal))

    # the venue still holds the position: local state must match broker truth
    pos = agent.positions["TCS"]
    assert pos.quantity == 75
    assert pos.leverage == 1.0
    assert allocator.reserved["mom1"] == 82_500.0  # margin stays reserved
    assert router.closes == 0
    assert bus.get("capital:trade") is None  # no phantom ledger entry
    assert bus.get("fills:mom1")["side"] == "long"  # only the open event exists
    # exactly ONE exit attempt went out — never a blind retry or double send
    exit_posts = [c for c in session.calls if c[0] == "POST" and c[1] == "/orders"]
    assert len(exit_posts) == 1

    asyncio.run(router._execute(signal))
    assert router.rejected == ["mom1:TCS:close"]


# ── venue isolation ──────────────────────────────────────────────────────────

def test_router_broker_for_keeps_venues_isolated(monkeypatch, tmp_path):
    router, broker, allocator, bus, session, db, agent = make_equity_harness(monkeypatch, tmp_path)
    equity_agent = make_agent("mom1", AgentType.EQUITY_MOMENTUM)
    crypto_agent = make_agent("cp1", AgentType.CRYPTO_PERP)
    router.watch_agent(equity_agent)
    router.watch_agent(crypto_agent)

    assert router._broker_for("mom1") is broker  # real DhanBroker
    assert router._broker_for("cp1") is router.crypto_broker  # the sentinel
    # the sentinel raises on ANY attribute access: any cross-venue touch in
    # the flow tests above would have failed loudly instead of silently routing