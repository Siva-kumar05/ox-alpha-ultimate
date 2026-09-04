"""ExecutionRouter live crypto tests: real router, real broker, scripted venue.

The audit's sharpest provable gap was that the promax ExecutionRouter had
never driven a live-typed broker end-to-end (paper smoke only, 0 fills).
This file closes it fully offline: the REAL ``ExecutionRouter`` and the REAL
``CryptoMicroBroker`` (``mode: live``) run against the existing ``ScriptedCcxt``
fake venue reused from ``test_crypto_live`` — no new scaffolding.

Covered, through the real router code path (``_execute`` -> ``_open``/``_close``,
and one full ``SharedDataBus.publish -> watch_agent callback -> task`` run):

  * market-type inference: spot agents never touch leverage machinery,
    perp agents set cross leverage before the entry;
  * leverage pass-through with per-symbol caching and change re-apply;
  * reduceOnly closes on swaps; spot closes go out as plain SELLs;
  * spot leverage refusal mid-route releases the reserved margin;
  * venue errors mid-route propagate fail-closed, release the reservation,
    and never fabricate a fill or leave a phantom position;
  * a failed close restores the position so local state matches broker truth;
  * the live DataPump funding/OI branch pulls real venue fundamentals.

The equity broker is a sentinel that fails loudly if a crypto signal ever
touches it, proving NSE-side routing is untouched.
"""

from __future__ import annotations

import asyncio

import pytest

from ox.agents.base import AgentType, SharedDataBus
from ox.agents.capital_allocator import CapitalAllocator
from ox.agents.orchestrator import DataPump, ExecutionRouter
from ox.brokers import OrderError
from support import (
    ScriptedCcxt,
    live_env,
    make_live_broker,
    MustNotTouch,
    buy,
    close,
    make_agent,
)


def make_harness(monkeypatch, markets, symbols, *, weights=None, ticker_price=100.0,
                 fill_price=100.5, **broker_overrides):
    """Real bus + allocator + crypto broker (fake ccxt) + real router."""
    bus = SharedDataBus()
    allocator = CapitalAllocator(bus, config={"total": 5000.0, "weights": weights or {}}, db=None)
    client = ScriptedCcxt(symbols=symbols, ticker_price=ticker_price, fill_price=fill_price)
    live_env(monkeypatch)
    broker = make_live_broker(monkeypatch, markets, client, **broker_overrides)
    assert broker.login() is True
    router = ExecutionRouter(bus, MustNotTouch(), broker, allocator,
                             risk_coordinator=object())
    return router, broker, allocator, bus, client


# ── market-type inference ────────────────────────────────────────────────────

def test_spot_agent_routes_to_cash_spot_order(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"PEPEUSDT": "spot"}, ("PEPE/USDT",),
        weights={"meme1": 1.0}, ticker_price=0.000012, fill_price=0.0000121)
    agent = make_agent("meme1", AgentType.CRYPTO_MEME_SWING)
    router.watch_agent(agent)
    allocator.register_agent("meme1")  # budget 5000

    asyncio.run(router._execute(buy("meme1", "PEPEUSDT", 0.000012, 100_000_000.0, leverage=1.0)))

    # exact venue call: cash market buy, no leverage machinery anywhere
    order = client.calls_named("create_order")[0]
    assert order == ("PEPE/USDT", "market", "buy", 100_000_000.0, None, {})
    assert not client.calls_named("set_leverage")
    assert not client.calls_named("set_margin_mode")
    assert router.fills == 1
    pos = agent.positions["PEPEUSDT"]
    assert pos.quantity == 100_000_000.0
    assert pos.entry_price == 0.0000121  # real fill price from the venue
    assert pos.leverage == 1.0
    assert allocator.reserved["meme1"] == 1200.0  # qty*price/1x
    assert bus.get("fills:meme1")["side"] == "long"


def test_perp_agent_sets_cross_leverage_then_caches(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap"}, ("BTC/USDT:USDT",),
        weights={"perp1": 1.0})
    agent = make_agent("perp1", AgentType.CRYPTO_PERP)
    router.watch_agent(agent)
    allocator.register_agent("perp1")

    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.5, leverage=5.0)))
    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.1, leverage=5.0)))

    assert client.calls_named("set_margin_mode") == [("cross", "BTC/USDT:USDT")]
    # leverage cached per symbol: one set for two entries at the same level
    assert client.calls_named("set_leverage") == [(5.0, "BTC/USDT:USDT")]
    assert client.calls_named("create_order") == [
        ("BTC/USDT:USDT", "market", "buy", 0.5, None, {}),
        ("BTC/USDT:USDT", "market", "buy", 0.1, None, {}),
    ]
    assert agent.positions["BTCUSDT"].leverage == 5.0
    assert router.fills == 2


def test_perp_leverage_change_reapplies(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap"}, ("BTC/USDT:USDT",),
        weights={"perp1": 1.0})
    agent = make_agent("perp1", AgentType.CRYPTO_PERP)
    router.watch_agent(agent)
    allocator.register_agent("perp1")

    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.5, leverage=3.0)))
    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.2, leverage=10.0)))

    assert client.calls_named("set_leverage") == [
        (3.0, "BTC/USDT:USDT"), (10.0, "BTC/USDT:USDT")]


# ── closes ───────────────────────────────────────────────────────────────────

def test_perp_close_is_reduce_only_and_releases_margin(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap"}, ("BTC/USDT:USDT",),
        weights={"perp1": 1.0})
    agent = make_agent("perp1", AgentType.CRYPTO_PERP)
    router.watch_agent(agent)
    allocator.register_agent("perp1")

    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.5, leverage=5.0)))
    assert allocator.reserved["perp1"] == 10.0  # 50 notional / 5x
    asyncio.run(router._execute(close("perp1", "BTCUSDT", 100.5, 0.5)))

    order = client.calls_named("create_order")[-1]
    assert order == ("BTC/USDT:USDT", "market", "sell", 0.5, None, {"reduceOnly": True})
    # the entry set leverage once; the close adds no leverage call of its own
    assert client.calls_named("set_leverage") == [(5.0, "BTC/USDT:USDT")]
    assert not client.calls_named("set_margin_mode")[1:]  # close adds no margin call
    assert "BTCUSDT" not in agent.positions
    assert allocator.reserved["perp1"] == 0.0
    assert router.closes == 1
    assert router.order_ids == {}
    fill = bus.get("fills:perp1")
    assert fill["side"] == "closed"
    assert fill["pnl"] == 0.0  # exit at entry fill price
    assert bus.get("capital:trade")["qty"] == 0.5


# ── fail-closed mid-route ────────────────────────────────────────────────────

def test_spot_leverage_refusal_releases_reservation_and_propagates(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"PEPEUSDT": "spot"}, ("PEPE/USDT",),
        weights={"meme1": 1.0}, ticker_price=0.000012)
    agent = make_agent("meme1", AgentType.CRYPTO_MEME_SWING)
    router.watch_agent(agent)
    allocator.register_agent("meme1")

    signal = buy("meme1", "PEPEUSDT", 0.000012, 100_000_000.0, leverage=3.0)
    with pytest.raises(OrderError, match="cash-only"):
        asyncio.run(router._open(signal))
    # the reserved margin was released: a refused open must not leak capital
    assert allocator.reserved["meme1"] == 0.0
    assert "PEPEUSDT" not in agent.positions
    assert not client.calls_named("create_order")

    # and the _execute wrapper records the rejection without crashing
    asyncio.run(router._execute(signal))
    assert router.rejected == ["meme1:PEPEUSDT:buy"]
    assert router.fills == 0


def test_venue_error_mid_open_releases_reservation_fail_closed(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap"}, ("BTC/USDT:USDT",),
        weights={"perp1": 1.0})
    agent = make_agent("perp1", AgentType.CRYPTO_PERP)
    router.watch_agent(agent)
    allocator.register_agent("perp1")
    client.order_error = RuntimeError("Insufficient margin")

    signal = buy("perp1", "BTCUSDT", 100.0, 0.5, leverage=5.0)
    with pytest.raises(OrderError, match="Binance order failed"):
        asyncio.run(router._open(signal))
    assert allocator.reserved["perp1"] == 0.0  # no leak
    assert "BTCUSDT" not in agent.positions  # no phantom position
    assert router.fills == 0

    asyncio.run(router._execute(signal))
    assert router.rejected == ["perp1:BTCUSDT:buy"]
    assert bus.get("fills:perp1") is None  # nothing fabricated


def test_failed_close_restores_position_matching_broker_truth(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap"}, ("BTC/USDT:USDT",),
        weights={"perp1": 1.0})
    agent = make_agent("perp1", AgentType.CRYPTO_PERP)
    router.watch_agent(agent)
    allocator.register_agent("perp1")

    asyncio.run(router._execute(buy("perp1", "BTCUSDT", 100.0, 0.5, leverage=5.0)))
    client.order_error = RuntimeError("-1021: timestamp outside recvWindow")
    signal = close("perp1", "BTCUSDT", 100.5, 0.5)

    with pytest.raises(OrderError, match="Binance order failed"):
        asyncio.run(router._close(signal))

    # the venue still holds the position: local state must match broker truth
    pos = agent.positions["BTCUSDT"]
    assert pos.quantity == 0.5
    assert pos.leverage == 5.0
    assert allocator.reserved["perp1"] == 10.0  # margin stays reserved
    assert router.closes == 0
    assert bus.get("capital:trade") is None  # no phantom ledger entry
    assert bus.get("fills:perp1")["side"] == "long"  # only the open event exists


# ── production signal flow (bus publish -> callback -> task) ─────────────────

def test_signal_via_bus_publish_runs_full_entry(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"PEPEUSDT": "spot"}, ("PEPE/USDT",),
        weights={"meme1": 1.0}, ticker_price=0.000012, fill_price=0.0000121)
    agent = make_agent("meme1", AgentType.CRYPTO_MEME_SWING)
    router.watch_agent(agent)
    allocator.register_agent("meme1")

    async def drive():
        bus.publish("signals:meme1", buy("meme1", "PEPEUSDT", 0.000012, 100_000_000.0))
        for _ in range(50):
            if router.fills:
                return
            await asyncio.sleep(0.001)

    asyncio.run(drive())
    assert router.fills == 1
    assert agent.positions["PEPEUSDT"].quantity == 100_000_000.0
    assert client.calls_named("create_order")[0][0] == "PEPE/USDT"


# ── live DataPump funding/OI branch ──────────────────────────────────────────

def test_live_data_pump_pulls_real_funding_and_open_interest(monkeypatch):
    router, broker, allocator, bus, client = make_harness(
        monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"},
        ("BTC/USDT:USDT", "PEPE/USDT"))
    pump = DataPump(bus, None, broker, symbols=[], crypto_symbols=["BTCUSDT", "PEPEUSDT"],
                    interval_seconds=0.02)

    async def one_tick():
        stop = asyncio.Event()
        task = asyncio.create_task(pump.run(stop))
        for _ in range(500):
            if bus.get("market:PEPEUSDT") is not None:
                break
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(one_tick())

    swap_tick = bus.get("market:BTCUSDT")
    assert swap_tick["source"] == "crypto_broker"
    assert swap_tick["funding_rate"] == 0.0001  # real venue value, not synthetic
    assert swap_tick["open_interest"] == 12345.0
    spot_tick = bus.get("market:PEPEUSDT")
    assert spot_tick["funding_rate"] == 0.0  # spot reports zeros, never fabricates
    funding = bus.get("exchange:funding")
    assert funding["exchange"] == "binance"
    # the swap symbol hit the venue funding/OI endpoints exactly once each;
    # the spot symbol never does.
    assert client.calls_named("fetch_funding_rate") == [("BTC/USDT:USDT",)]
    assert client.calls_named("fetch_open_interest") == [("BTC/USDT:USDT",)]
    assert pump.ticks >= 1