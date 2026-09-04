"""Shared offline test doubles.

Single source of truth for the scripted transports and broker fakes that
several test modules reuse.  test_live_broker_contract and
test_run_loop_resilience re-export the names they historically defined so
existing import chains (e.g. test_eod_squareoff -> test_run_loop_resilience)
keep working; new imports should come straight from here.

Nothing in this module touches the network or a wall clock.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from types import SimpleNamespace

from ox.crypto import CryptoMicroBroker

_LTP_PAYLOAD = {"data": {"NSE_EQ": {"1333": {"last_price": 1100.0}}}}


class _FakeResponse:
    def __init__(self, status=200, payload=None, reason="OK", headers=None, text=None):
        self.status_code = status
        self._payload = payload
        self.reason = reason
        self.headers = headers or {}
        self._text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._text is not None:
            raise ValueError("no json body")
        return self._payload


class _FakeSession:
    """Scripted HTTP transport: responses keyed by (method, path), call log kept.

    ``request`` serves the Dhan-style JSON transport; ``post`` serves the
    Noren/Choice form transport (jData=<json>&jKey=<token>).
    """

    def __init__(self):
        self.responses: dict[tuple[str, str], deque] = {}
        self.calls: list[tuple[str, str, dict]] = []

    def script(self, method, path, *, status=200, payload=None, exc=None, headers=None, text=None):
        self.responses.setdefault((method, path), deque()).append(
            dict(status=status, payload=payload, exc=exc, headers=headers, text=text)
        )

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.replace("https://api.dhan.co/v2", "")
        self.calls.append((method, path, json or {}))
        queue = self.responses.get((method, path))
        if not queue:
            raise AssertionError(f"unscripted HTTP call: {method} {path}")
        entry = queue.popleft()
        if entry["exc"] is not None:
            raise entry["exc"]
        return _FakeResponse(entry["status"], entry["payload"], headers=entry["headers"], text=entry["text"])

    def post(self, url, data=None, headers=None, timeout=None):
        """Noren-style form POST used by ChoiceBroker: jData=<json>&jKey=<token>."""
        path = url.replace("https://api.shoonya.com/NorenWClientTP/", "/")
        body: dict = {}
        jkey = None
        if data:
            for part in str(data).split("&"):
                if part.startswith("jData="):
                    body = json.loads(part[len("jData="):])
                elif part.startswith("jKey="):
                    jkey = part[len("jKey="):]
        self.calls.append(("POST", path, {"jData": body, "jKey": jkey}))
        queue = self.responses.get(("POST", path))
        if not queue:
            raise AssertionError(f"unscripted HTTP call: POST {path}")
        entry = queue.popleft()
        if entry["exc"] is not None:
            raise entry["exc"]
        return _FakeResponse(entry["status"], entry["payload"], headers=entry["headers"], text=entry["text"])

    def call_paths(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _ in self.calls]


class _AttrDict(dict):
    """Real config objects expose keys as attributes (Cfg.root); mirror that."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _StubRisk:
    def __init__(self):
        self.closed_pnls: list[float] = []

    def on_trade_close(self, pnl: float) -> None:
        self.closed_pnls.append(float(pnl))


class ScriptedCcxt:
    """Deterministic stand-in for a ccxt exchange, recording every call."""

    def __init__(self, symbols=("BTC/USDT:USDT", "PEPE/USDT"), decimals=5,
                 ticker_price=100.0, fill_price=100.5, order_error=None,
                 funding_rate=0.0001, open_interest=12345.0):
        self.markets = {sym: {"symbol": sym} for sym in symbols}
        self.decimals = decimals
        self.ticker_price = ticker_price
        self.fill_price = fill_price
        self.order_error = order_error
        self.funding_rate = funding_rate
        self.open_interest = open_interest
        self.positions_rows = []
        self.balances = {}
        self.calls = []  # (name, args, kwargs)
        self.auth_cfg = None

    def _log(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def calls_named(self, name):
        return [args for call, args, _ in self.calls if call == name]

    def load_markets(self):
        self._log("load_markets")
        return self.markets

    def fetch_ticker(self, symbol):
        self._log("fetch_ticker", symbol)
        return {"last": self.ticker_price}

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        self._log("fetch_ohlcv", symbol, timeframe, limit)
        return [[1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0]]

    def amount_to_precision(self, symbol, amount):
        self._log("amount_to_precision", symbol, amount)
        return format(float(amount), f".{self.decimals}f")

    def set_margin_mode(self, mode, symbol):
        self._log("set_margin_mode", mode, symbol)

    def set_leverage(self, leverage, symbol):
        self._log("set_leverage", leverage, symbol)

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self._log("create_order", symbol, order_type, side, amount, price, params)
        if self.order_error is not None:
            raise self.order_error
        return {
            "id": "BINFILL001", "average": self.fill_price, "price": self.fill_price,
            "filled": amount, "fee": {"cost": 0.05},
        }

    def fetch_positions(self):
        self._log("fetch_positions")
        return self.positions_rows

    def fetch_balance(self):
        self._log("fetch_balance")
        return self.balances

    def fetch_funding_rate(self, symbol):
        self._log("fetch_funding_rate", symbol)
        return {"fundingRate": self.funding_rate}

    def fetch_open_interest(self, symbol):
        self._log("fetch_open_interest", symbol)
        return {"openInterestAmount": self.open_interest}


def install_fake_ccxt(monkeypatch, client):
    """Route crypto.py's ``import ccxt`` to a fake module returning ``client``."""
    def factory(cfg):
        client.auth_cfg = cfg
        return client

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=factory))
    return client


def live_env(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-api-key-0123456789abcdef")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-api-secret-0123456789abcdef")


def make_live_broker(monkeypatch, markets, client=None, **overrides):
    # The leverage tests exercise the venue path, so the fixture explicitly
    # opts into perp leverage; the no-debt default (gate closed) is covered
    # by its own dedicated test.
    cfg = {"mode": "live", "crypto": {"markets": markets, "allow_perp_leverage": True}}
    cfg["crypto"].update(overrides)
    broker = CryptoMicroBroker(cfg, None)
    if client is not None:
        install_fake_ccxt(monkeypatch, client)
    return broker


class MustNotTouch:
    """Sentinel equity broker: any attribute access is a routing bug."""

    def __getattr__(self, name):
        raise AssertionError(f"equity broker was touched for a crypto signal: {name}")


def make_agent(agent_id: str, agent_type, positions=None):
    return SimpleNamespace(
        agent_id=agent_id,
        config=SimpleNamespace(agent_type=agent_type),
        positions=positions if positions is not None else {},
    )


def buy(agent_id, symbol, price, qty, leverage=1.0, stop=None, target=None):
    from ox.agents.base import Signal

    return Signal(agent_id=agent_id, symbol=symbol, action="buy", strength=1.0,
                  price=price, quantity=qty, leverage=leverage,
                  stop_loss=stop, take_profit=target)


def close(agent_id, symbol, price, qty):
    from ox.agents.base import Signal

    return Signal(agent_id=agent_id, symbol=symbol, action="close", strength=1.0,
                  price=price, quantity=qty, metadata={"reason": "test"})


def _candle_payload(rows: int = 400, base: float = 1000.0) -> dict:
    epoch = int(time.time()) - rows * 60
    timestamp = [epoch + i * 60 for i in range(rows)]
    close = [base + i * 0.05 for i in range(rows)]
    return {
        "timestamp": timestamp,
        "open": close,
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1000] * rows,
    }