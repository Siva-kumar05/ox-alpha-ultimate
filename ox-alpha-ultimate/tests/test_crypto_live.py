"""Tests for the live Binance crypto path (ox/crypto.py + promax live gate).

Covers the operator-critical guarantees of the live wiring:
  * paper mode never imports ccxt and keeps its simulated fill surface;
  * live login fails closed without API keys and resolves every configured
    symbol against the venue market table;
  * spot symbols refuse leveraged orders before reaching a venue;
  * the FULL live order cycle is driven through a scripted fake ccxt client
    for both spot and perp: exact venue calls and parameters (market-type
    inference, cross leverage set on swaps only and cached, amount-precision
    rounding, reduceOnly on closes, fill-price validation, ticker caching,
    positions reconciliation, funding/open-interest gating);
  * live-venue calls made before login() fail closed with clear errors;
  * the orchestrator refuses live mode without the operator approval env var.

No network is used: the ccxt module is replaced in ``sys.modules`` by a fake
whose ``binance`` factory hands back the scripted client, so ``login()``
runs its real control flow against canned venue state.
"""

from __future__ import annotations

import builtins

import pytest

from ox.agents.orchestrator import AgentOrchestrator
from ox.brokers import AuthenticationError, BrokerError, MarketDataError, OrderError
from ox.crypto import CryptoMicroBroker
from support import ScriptedCcxt, live_env, make_live_broker


# ── paper / separation ───────────────────────────────────────────────────────

def make_paper_broker():
    return CryptoMicroBroker({"crypto": {"paper_start_usdt": 0.9, "min_notional_usdt": 5.0}}, None)


def test_paper_mode_never_imports_ccxt(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ccxt":
            raise ImportError("ccxt must not be required in paper mode")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    broker = make_paper_broker()
    assert broker.login() is True
    assert broker.live is False
    assert broker.name == "crypto_paper"
    receipt = broker.place_market("BTCUSDT", "BUY", 1.0)  # ~68k USDT notional, above min
    assert receipt["order_id"].startswith("CR")
    assert receipt["price"] > 0
    # Paper ignores live-only params and never fabricates a venue error for them.
    receipt = broker.place_market("BTCUSDT", "SELL", 1.0, leverage=3.0, reduce_only=True)
    assert receipt["side"] == "SELL"
    assert broker._ccxt is None


def test_live_login_fails_closed_without_keys(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"})
    assert broker.live is True
    assert broker.name == "binance"
    with pytest.raises(AuthenticationError):
        broker.login()


def test_live_login_without_configured_markets_needs_no_keys(monkeypatch):
    # Live promax may trade NSE only; crypto with zero configured symbols
    # must not demand Binance credentials at boot.
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    broker = make_live_broker(monkeypatch, {})
    assert broker.login() is True  # nothing configured -> no venue session needed


def test_live_spot_refuses_leverage():
    broker = CryptoMicroBroker(
        {"mode": "live", "crypto": {"markets": {"PEPEUSDT": "spot", "BTCUSDT": "swap"}}}, None
    )
    with pytest.raises(OrderError, match="cash-only"):
        broker.place_market("PEPEUSDT", "BUY", 100.0, leverage=2.0)
    # 1x spot is allowed past the leverage gate (venue call not reached: no ccxt).
    with pytest.raises(OrderError):
        broker.place_market("PEPEUSDT", "BUY", 100.0, leverage=1.0)


def test_swap_and_spot_symbol_mapping():
    broker = CryptoMicroBroker(
        {"mode": "live", "crypto": {"markets": {"BTCUSDT": "swap", "PEPEUSDT": "spot"}}}, None
    )
    assert broker._ccxt_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert broker._ccxt_symbol("btcusdt") == "BTC/USDT:USDT"
    assert broker._ccxt_symbol("PEPEUSDT") == "PEPE/USDT"
    assert broker._market_type("BTCUSDT") == "swap"
    assert broker._market_type("PEPEUSDT") == "spot"
    # Unknown symbols default to spot (cash-only).
    assert broker._ccxt_symbol("DOGEUSDT") == "DOGE/USDT"
    assert broker._market_type("DOGEUSDT") == "spot"


def test_orchestrator_live_mode_requires_approval_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OX_LIVE_EXECUTION_APPROVED", raising=False)
    config = tmp_path / "live.yaml"
    config.write_text("mode: live\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OX_LIVE_EXECUTION_APPROVED"):
        AgentOrchestrator(config_path=config)


# ── live login resolution ────────────────────────────────────────────────────

def test_live_login_resolves_symbols_and_authenticates(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT", "PEPE/USDT"))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"}, client)
    assert broker.login() is True
    assert client.auth_cfg is not None
    assert client.auth_cfg["enableRateLimit"] is True
    assert "load_markets" in [name for name, _, _ in client.calls]
    assert broker.authenticated() is True


def test_live_login_fails_closed_on_unresolvable_symbol(monkeypatch):
    live_env(monkeypatch)
    # Venue only lists the spot pair; the configured swap symbol must fail boot.
    client = ScriptedCcxt(symbols=("PEPE/USDT",))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"}, client)
    with pytest.raises(BrokerError, match="BTCUSDT"):
        broker.login()
    assert broker.authenticated() is False


# ── live order cycle: spot ───────────────────────────────────────────────────

def test_live_spot_buy_sends_cash_market_order(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("PEPE/USDT",), decimals=0, ticker_price=0.000012,
                          fill_price=0.0000121)
    broker = make_live_broker(monkeypatch, {"PEPEUSDT": "spot"}, client)
    broker.login()
    receipt = broker.place_market("PEPEUSDT", "BUY", 123_456_789.5, leverage=1.0)
    # amount precision rounded to whole tokens
    order_call = client.calls_named("create_order")[0]
    assert order_call == ("PEPE/USDT", "market", "buy", 123456790.0, None, {})
    # spot never touches leverage machinery
    assert not client.calls_named("set_leverage")
    assert not client.calls_named("set_margin_mode")
    assert receipt["order_id"] == "BINFILL001"
    assert receipt["price"] == 0.0000121
    assert receipt["qty"] == 123456790.0
    assert receipt["fee"] == 0.05
    assert receipt["side"] == "BUY"


def test_live_spot_close_sends_no_reduce_only_flag(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("PEPE/USDT",))
    broker = make_live_broker(monkeypatch, {"PEPEUSDT": "spot"}, client)
    broker.login()
    broker.place_market("PEPEUSDT", "SELL", 100.0, reduce_only=True)
    order_call = client.calls_named("create_order")[0]
    # reduceOnly is a swap concept; a spot SELL must go out as a plain order.
    assert order_call == ("PEPE/USDT", "market", "sell", 100.0, None, {})


# ── live order cycle: perp ───────────────────────────────────────────────────

def test_live_swap_buy_sets_cross_leverage_once_then_caches(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    receipt = broker.place_market("BTCUSDT", "BUY", 0.5, leverage=5.0)
    assert client.calls_named("set_margin_mode") == [("cross", "BTC/USDT:USDT")]
    assert client.calls_named("set_leverage") == [(5.0, "BTC/USDT:USDT")]
    order_call = client.calls_named("create_order")[0]
    assert order_call == ("BTC/USDT:USDT", "market", "buy", 0.5, None, {})
    assert receipt["order_id"] == "BINFILL001"
    # Same leverage again: no repeat venue calls (per-symbol cache).
    broker.place_market("BTCUSDT", "BUY", 0.1, leverage=5.0)
    assert len(client.calls_named("set_margin_mode")) == 1
    assert len(client.calls_named("set_leverage")) == 1


def test_live_swap_leverage_change_reapplies(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    broker.place_market("BTCUSDT", "BUY", 0.5, leverage=3.0)
    broker.place_market("BTCUSDT", "BUY", 0.2, leverage=10.0)
    assert client.calls_named("set_leverage") == [(3.0, "BTC/USDT:USDT"), (10.0, "BTC/USDT:USDT")]


def test_live_swap_close_is_reduce_only_and_skips_leverage(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    broker.place_market("BTCUSDT", "SELL", 0.5, reduce_only=True, leverage=5.0)
    order_call = client.calls_named("create_order")[0]
    # reduceOnly must protect the exit from flipping into a short; a close
    # must not re-request leverage.
    assert order_call == ("BTC/USDT:USDT", "market", "sell", 0.5, None, {"reduceOnly": True})
    assert not client.calls_named("set_leverage")
    assert not client.calls_named("set_margin_mode")


def test_live_unknown_symbol_defaults_to_cash_spot(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("DOGE/USDT",))
    broker = make_live_broker(monkeypatch, {"DOGEUSDT": "spot"}, client)
    broker.login()
    broker.place_market("DOGEUSDT", "BUY", 10.0, leverage=1.0)
    assert client.calls_named("create_order")[0][0] == "DOGE/USDT"
    assert not client.calls_named("set_leverage")


# ── live fail-closed behavior ────────────────────────────────────────────────

def test_live_fill_without_confirmed_price_raises(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    client.fill_price = None  # venue fills but reports no average/price
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    with pytest.raises(OrderError, match="confirmed price"):
        broker.place_market("BTCUSDT", "BUY", 0.5, leverage=5.0)


def test_live_order_venue_error_is_wrapped_not_fabricated(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    client.order_error = RuntimeError("Invalid API-key, IP, or permissions for action")
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    with pytest.raises(OrderError, match="Binance order failed"):
        broker.place_market("BTCUSDT", "BUY", 0.5, leverage=5.0)
    assert broker.authenticated() is True  # auth survived; the ORDER failed


def test_live_ticker_failure_is_wrapped(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()

    def boom(symbol):
        raise RuntimeError("rate limit")

    client.fetch_ticker = boom
    with pytest.raises(MarketDataError, match="Binance ticker failed"):
        broker.ltp("BTCUSDT")


def test_live_ticker_invalid_price_raises(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",))
    client.ticker_price = 0.0
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    with pytest.raises(MarketDataError, match="invalid price"):
        broker.ltp("BTCUSDT")


def test_live_venue_calls_before_login_fail_closed(monkeypatch):
    live_env(monkeypatch)
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"})
    # No login() was performed: every live surface must refuse cleanly,
    # never crash with an AttributeError on a missing client.
    with pytest.raises(OrderError, match="login"):
        broker.place_market("PEPEUSDT", "BUY", 10.0, leverage=1.0)
    with pytest.raises(OrderError, match="login"):
        broker.place_market("BTCUSDT", "BUY", 0.5, leverage=5.0)
    with pytest.raises(MarketDataError, match="login"):
        broker.ltp("BTCUSDT")
    with pytest.raises(MarketDataError, match="login"):
        broker.hist("BTCUSDT")
    with pytest.raises(MarketDataError, match="login"):
        broker.positions()


# ── live market data: ticker cache, funding, positions ──────────────────────

def test_live_ltp_caches_then_refreshes(monkeypatch):
    live_env(monkeypatch)
    clock = {"now": 0.0}
    monkeypatch.setattr("ox.crypto.time.monotonic", lambda: clock["now"])
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT",), ticker_price=100.0)
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, client)
    broker.login()
    clock["now"] = 1.0
    assert broker.ltp("BTCUSDT") == 100.0  # fetch + cache at t=1
    clock["now"] = 1.5
    assert broker.ltp("BTCUSDT") == 100.0  # cache hit (< 2s)
    assert len(client.calls_named("fetch_ticker")) == 1
    clock["now"] = 3.2
    assert broker.ltp("BTCUSDT") == 100.0  # cache expired -> refetch
    assert len(client.calls_named("fetch_ticker")) == 2


def test_live_positions_merge_swaps_and_spot(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT", "PEPE/USDT"))
    client.positions_rows = [{
        "symbol": "BTC/USDT:USDT", "contracts": 0.5, "entryPrice": 66_000.0,
        "leverage": 5.0, "side": "long",
    }]
    client.balances = {"PEPE": {"free": 1_000_000.0, "used": 0.0, "total": 1_000_000.0}}
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"}, client)
    broker.login()
    rows = broker.positions()
    assert rows["BTC/USDT:USDT"]["qty"] == 0.5
    assert rows["BTC/USDT:USDT"]["leverage"] == 5.0
    assert rows["PEPE/USDT"]["qty"] == 1_000_000.0
    assert rows["PEPE/USDT"]["leverage"] == 1.0  # spot is always cash
    assert "PEPE/USDT" in rows


def test_live_funding_and_open_interest_are_swap_only(monkeypatch):
    live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT", "PEPE/USDT"))
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap", "PEPEUSDT": "spot"}, client)
    broker.login()
    assert broker.funding_rate("BTCUSDT") == 0.0001
    assert broker.open_interest("BTCUSDT") == 12345.0
    # Spot symbols report None and never hit the venue funding endpoints.
    assert broker.funding_rate("PEPEUSDT") is None
    assert broker.open_interest("PEPEUSDT") is None
    assert len(client.calls_named("fetch_funding_rate")) == 1
    assert len(client.calls_named("fetch_open_interest")) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])