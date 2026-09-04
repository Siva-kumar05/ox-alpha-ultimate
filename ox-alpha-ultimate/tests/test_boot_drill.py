"""Deployment-readiness boot drills for the exact configuration run on AWS.

Deterministic and offline, reusing the existing fakes only (scripted Dhan
transport from ``test_run_loop_resilience``, fake ccxt venue from
``test_crypto_live``) — no new scaffolding.

Covers both real live entry paths with the EXACT settings an operator seds
into ``config.yaml`` / ``config_promax.yaml``:

* legacy ``python run.py run`` with ``mode: live`` / ``platform: dhan``:
  Cfg validation on the real live settings (public-IP-shaped whitelist
  merged from ``DHAN_STATIC_IP``, ``tick_seconds >= 2``, complete
  ``security_map``), the ``OX_LIVE_EXECUTION_APPROVED`` gate, the Dhan
  credential gates, and the full boot -> "Autonomous tick loop started" ->
  ACTIVE steady state;
* the promax orchestrator with crypto live: the live gate, real
  ``CryptoMicroBroker`` login against the fake venue, real DhanBroker equity
  side through the scripted transport, and DataPump ticks flowing for BOTH
  venues;
* paper boot with zero env vars still works;
* the exact-venue failure that previously stalled live promax (equity side
  without a resolvable securityId) now degrades per-branch instead of
  starving the crypto agents;
* the ``live-test`` and ``track-record`` operator markers.

Every drill that reaches steady state boots the REAL Agent/AgentOrchestrator
and lets the REAL run loops tick; only wall-clock sleep and the egress-IP
lookup are pinned.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

import pytest

from ox.agents.orchestrator import AgentOrchestrator, DataPump
from ox.brokers import AuthenticationError, DhanBroker, MarketDataError, make_broker
from ox.core import Cfg, ConfigError, DB
from test_crypto_live import ScriptedCcxt, install_fake_ccxt, live_env, make_live_broker
from test_run_loop_resilience import _candle_payload, _FakeSession

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_AUDIT_KEY = "boot-drill-audit-key-at-least-thirty-two-chars"
_LIVE_IP = "13.207.244.242"  # the public-IP-shaped value already in config.yaml


# ── config builders: the exact files the operator runs, plus drill pins ──────

def _legacy_config(tmp: Path, *, live: bool) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw.update({"mode": "live" if live else "paper",
                "platform": "dhan" if live else "paper",
                "db_path": "smoke.db"})
    # Boot determinism only (the tick path is untouched): the universe scan,
    # the order-flow websocket and boot-time genetic training are not part of
    # the steady-state contract under test, and the schedule is pinned away
    # from wall clock so EOD / nightly training can never fire mid-drill.
    raw["universe"]["auto_scan"] = False
    raw["order_flow"].update({"enabled": False, "primary": False})
    raw["auto_train_on_boot"] = False
    raw["history_days"] = 2  # one chart chunk per symbol at boot
    raw.update({"market_open": "23:00", "entry_cutoff": "23:40",
                "squareoff": "23:55", "market_close": "23:59"})
    raw["execution"].update({"order_confirm_timeout_seconds": 2,
                             "rate_limit_backoff_seconds": 0.01,
                             "max_rate_limit_backoff_seconds": 0.05})
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _promax_config(tmp: Path, *, live: bool) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config_promax.yaml").read_text(encoding="utf-8"))
    raw.update({"mode": "live" if live else "paper",
                "platform": "dhan" if live else "paper",
                "db_path": str(tmp / "promax.db")})
    if live:
        # The operator's NSE universe inside promax, resolved to securityIds
        # exactly like the legacy security_map.
        raw["security_map"] = {sym: str(101 + i) for i, sym in enumerate(
            ("YESBANK", "IDFCFIRSTB", "SUZLON", "IRFC", "IOB", "CANBK", "NIFTY"))}
    raw["data_pump"]["interval_seconds"] = 0.05  # drill-time cadence only
    path = tmp / "config_promax.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("OX_AUDIT_KEY", _AUDIT_KEY)
    monkeypatch.setenv("OX_LIVE_EXECUTION_APPROVED", "YES_I_UNDERSTAND_LIVE_TRADING")
    monkeypatch.setenv("DHAN_CLIENT_ID", "TEST-CLIENT-ID")
    monkeypatch.setenv("DHAN_TOKEN", "live-test-access-token-not-dummy-32chars")
    monkeypatch.setenv("DHAN_STATIC_IP", _LIVE_IP)
    live_env(monkeypatch)  # BINANCE_API_KEY / BINANCE_API_SECRET


def _clear_app_env(monkeypatch) -> None:
    for key in ("OX_AUDIT_KEY", "OX_LIVE_EXECUTION_APPROVED", "DHAN_CLIENT_ID",
                "DHAN_TOKEN", "DHAN_ACCESS_TOKEN", "DHAN_STATIC_IP",
                "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def log_capture(monkeypatch):
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger = logging.getLogger("ox")
    # pytest's capture root handler defaults to WARNING; the drill asserts on
    # INFO markers, so the ox logger must emit them.
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


def _agent_health(agent) -> str:
    """Read agent_health from the sqlite file after run_forever closed the DB."""
    conn = sqlite3.connect(str(Path(agent.cfg.root) / "smoke.db"))
    try:
        row = conn.execute("SELECT v FROM kv WHERE k='agent_health'").fetchone()
        return str(row[0]) if row else ""
    finally:
        conn.close()


def _audit_actions(agent) -> list[str]:
    conn = sqlite3.connect(str(Path(agent.cfg.root) / "smoke.db"))
    try:
        return [row[0] for row in conn.execute("SELECT action FROM audit")]
    finally:
        conn.close()


def _run_and_reach_steady_state(agent, session, *, min_ltp_ticks=2):
    """Run the REAL run_forever in a thread; stop after N healthy LTP ticks."""
    thread = threading.Thread(target=agent.run_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            ltp_calls = [c for c in session.calls if c[:2] == ("POST", "/marketfeed/ltp")]
            if len(ltp_calls) >= min_ltp_ticks:
                break
            time.sleep(0.02)
        agent.stop = True
        thread.join(timeout=20.0)
    finally:
        agent.stop = True
        thread.join(timeout=2.0)
    return thread


def _ltp_payload_for(security_map: dict[str, str], price: float = 1000.0) -> dict:
    return {"data": {"NSE_EQ": {sid: {"last_price": price} for sid in security_map.values()}}}


def _quote_payload_for(security_map: dict[str, str], price: float = 1000.0) -> dict:
    # day-cumulative volume 0 keeps _apply_volumes a no-op
    return {"data": {"NSE_EQ": {sid: {"last_price": price, "volume": 0} for sid in security_map.values()}}}


# ── 1. legacy: exact live settings validate ──────────────────────────────────

def test_legacy_live_config_exact_settings_validate(monkeypatch, tmp_path):
    monkeypatch.setenv("OX_AUDIT_KEY", _AUDIT_KEY)
    monkeypatch.setenv("OX_LIVE_EXECUTION_APPROVED", "YES_I_UNDERSTAND_LIVE_TRADING")
    monkeypatch.setenv("DHAN_STATIC_IP", _LIVE_IP)  # ip_whitelist_env resolution
    path = _legacy_config(tmp_path, live=True)
    cfg = Cfg(path)
    # the exact live settings survive validation
    assert cfg["mode"] == "live" and cfg["platform"] == "dhan"
    assert cfg["tick_seconds"] >= 2
    assert _LIVE_IP in cfg["ip_whitelist"]  # literal + DHAN_STATIC_IP merged
    assert cfg["security_map"]  # complete for all 5 configured symbols


def test_legacy_live_gate_fails_closed_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OX_LIVE_EXECUTION_APPROVED", raising=False)
    path = _legacy_config(tmp_path, live=True)
    with pytest.raises(ConfigError, match="OX_LIVE_EXECUTION_APPROVED"):
        Cfg(path)


def test_legacy_live_setting_rejections(monkeypatch, tmp_path):
    monkeypatch.setenv("OX_AUDIT_KEY", _AUDIT_KEY)
    monkeypatch.setenv("OX_LIVE_EXECUTION_APPROVED", "YES_I_UNDERSTAND_LIVE_TRADING")
    base = _legacy_config(tmp_path, live=True)

    def with_raw(**changes):
        raw = yaml.safe_load(base.read_text(encoding="utf-8"))
        raw.update(changes)
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    with pytest.raises(ConfigError, match="tick_seconds of at least 2"):
        Cfg(with_raw(tick_seconds=1))
    with pytest.raises(ConfigError, match="missing entries for configured symbols: INFY"):
        Cfg(with_raw(security_map={"RELIANCE": "2885", "HDFCBANK": "1333",
                                   "TCS": "11536", "ICICIBANK": "4963"}))
    with pytest.raises(ConfigError, match="not in symbols: GHOST"):
        Cfg(with_raw(security_map={**yaml.safe_load(base.read_text(encoding="utf-8"))["security_map"],
                                   "GHOST": "9999"}))
    with pytest.raises(ConfigError, match="Invalid IP address"):
        Cfg(with_raw(ip_whitelist=["not-an-ip"]))
    with pytest.raises(ConfigError, match="at least one whitelisted static IP"):
        Cfg(with_raw(ip_whitelist=[], ip_whitelist_env=""))


def test_legacy_credential_gates_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("OX_AUDIT_KEY", _AUDIT_KEY)
    monkeypatch.setenv("OX_LIVE_EXECUTION_APPROVED", "YES_I_UNDERSTAND_LIVE_TRADING")
    path = _legacy_config(tmp_path, live=True)
    cfg = Cfg(path)
    db = DB(tmp_path / "gate.db")

    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.delenv("DHAN_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    broker = make_broker(cfg, db)
    assert isinstance(broker, DhanBroker)
    with pytest.raises(AuthenticationError, match="DHAN_CLIENT_ID is not set"):
        broker.login()

    monkeypatch.setenv("DHAN_CLIENT_ID", "TEST-CLIENT-ID")
    # env is read at construction; a freshly built broker with a client id
    # but no token must refuse with the token-specific message
    broker = make_broker(cfg, db)
    with pytest.raises(AuthenticationError, match="real Dhan access token"):
        broker.login()


# ── 2. legacy: live boot to steady state ─────────────────────────────────────

def test_legacy_live_boot_reaches_steady_state(monkeypatch, tmp_path, log_capture):
    from ox.agent import Agent

    _live_env(monkeypatch)
    # The real egress-IP check runs, but the lookup itself is pinned to the
    # whitelisted address (the check would hit api.ipify.org on the instance).
    monkeypatch.setattr("ox.compliance.requests.get",
                        lambda url, timeout=None: SimpleNamespace(text=_LIVE_IP))
    # The loop cadence is the only other wall-clock dependency; the loop
    # structure, boot sequence and tick path are untouched.
    monkeypatch.setattr("ox.agent.time.sleep", lambda seconds: None)
    monkeypatch.setattr("ox.agent.hhmm", lambda: "08:00")  # never hit EOD/nightly

    path = _legacy_config(tmp_path, live=True)
    agent = Agent(str(path))
    agent.cognition = None
    for attr in ("health_checker", "backup_manager", "config_watcher",
                 "secret_manager", "shutdown_manager"):
        setattr(agent, attr, None)

    session = _FakeSession()
    agent.broker.session = session
    security_map = dict(agent.cfg["security_map"])
    # boot-time calls: login confirmation, static-IP confirmation, reconcile
    session.script("GET", "/fundlimit", payload={"data": {"availabelBalance": 500000.0}})
    session.script("GET", "/ip/getIP", payload={"ip": _LIVE_IP})
    session.script("GET", "/positions", payload=[])
    for _ in security_map:
        session.script("POST", "/charts/intraday", payload=_candle_payload())
    # steady-state ticks: batched LTP + quote snapshot per tick
    for _ in range(3):
        session.script("POST", "/marketfeed/ltp", payload=_ltp_payload_for(security_map))
        session.script("POST", "/marketfeed/quote", payload=_quote_payload_for(security_map))

    thread = _run_and_reach_steady_state(agent, session)

    assert not thread.is_alive(), "run_forever did not stop"
    assert not agent.comp.halted, f"live boot halted: {agent.comp.halt_reason}"
    assert not (Path(agent.cfg.root) / "KILL.flag").exists()
    assert any("Autonomous tick loop started" in r for r in log_capture)
    health = _agent_health(agent)
    assert '"state": "ACTIVE"' in health, f"never reached ACTIVE: {health}"
    actions = _audit_actions(agent)
    assert "BROKER_SESSION_VERIFIED" in actions
    assert "COMPLIANCE_HALT" not in actions


def test_legacy_paper_boot_zero_env_reaches_steady_state(monkeypatch, tmp_path, log_capture):
    from ox.agent import Agent

    _clear_app_env(monkeypatch)
    monkeypatch.setattr("ox.agent.time.sleep", lambda seconds: None)
    monkeypatch.setattr("ox.agent.hhmm", lambda: "08:00")

    path = _legacy_config(tmp_path, live=False)
    agent = Agent(str(path))
    agent.cognition = None
    for attr in ("health_checker", "backup_manager", "config_watcher",
                 "secret_manager", "shutdown_manager"):
        setattr(agent, attr, None)
    session = _FakeSession()  # paper broker never touches it; proves no HTTP
    agent.broker.session = session

    thread = threading.Thread(target=agent.run_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if '"state": "ACTIVE"' in _agent_health(agent):
                break
            time.sleep(0.02)
        agent.stop = True
        thread.join(timeout=20.0)
    finally:
        agent.stop = True
        thread.join(timeout=2.0)

    assert not thread.is_alive(), "paper run_forever did not stop"
    assert not agent.comp.halted
    assert not session.calls, "paper boot with zero env made a network call"
    assert any("Autonomous tick loop started" in r for r in log_capture)
    assert '"state": "ACTIVE"' in _agent_health(agent)


# ── 3. promax: live gate + full live boot ────────────────────────────────────

def test_promax_live_gate_fails_closed_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OX_LIVE_EXECUTION_APPROVED", raising=False)
    config = _promax_config(tmp_path, live=True)
    with pytest.raises(RuntimeError, match="OX_LIVE_EXECUTION_APPROVED"):
        AgentOrchestrator(config_path=config)


def test_promax_live_boot_reaches_steady_state(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    client = ScriptedCcxt(symbols=("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                                   "PEPE/USDT", "WIF/USDT", "BONK/USDT"))
    install_fake_ccxt(monkeypatch, client)
    config = _promax_config(tmp_path, live=True)
    session = _FakeSession()

    def fake_make_broker(shim, db):
        # the fix under test: the shim must carry the operator's security_map
        assert shim.get("security_map"), "live promax Dhan side lost security_map"
        broker = DhanBroker(shim, db)
        broker.session = session
        return broker

    monkeypatch.setattr("ox.agents.orchestrator.make_broker", fake_make_broker)
    session.script("GET", "/fundlimit", payload={"data": {"availabelBalance": 5000.0}})
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    security_map = raw["security_map"]
    for _ in range(4):
        session.script("POST", "/marketfeed/ltp", payload=_ltp_payload_for(security_map))

    orch = AgentOrchestrator(config_path=config)
    assert orch.equity_broker is not None and orch.crypto_broker is not None

    async def _run():
        await orch.start_all()
        deadline = time.monotonic() + 90.0
        while orch.data_pump.ticks < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        # snapshot BEFORE stop_all: stop_agent pops from orch.agents
        snapshot = {
            "ticks": orch.data_pump.ticks,
            "agents": len(orch.agents),
            "crypto_tick": orch.data_bus.get("market:BTCUSDT"),
            "equity_tick": orch.data_bus.get("market:YESBANK"),
        }
        await orch.stop_all()
        return snapshot

    state = asyncio.run(_run())
    assert state["ticks"] >= 2, "DataPump never reached steady state"
    assert state["agents"] >= 8, "agents did not all start"
    assert client.auth_cfg is not None, "live crypto broker did not log in"
    # both venues flowed: crypto fundamentals came from the venue, equity
    # quotes came through the scripted Dhan transport
    crypto_tick = state["crypto_tick"]
    assert crypto_tick is not None and crypto_tick["source"] == "crypto_broker"
    assert crypto_tick["funding_rate"] == 0.0001
    equity_tick = state["equity_tick"]
    assert equity_tick is not None and equity_tick["source"] == "equity_broker"
    assert equity_tick["price"] == 1000.0


def test_data_pump_equity_failure_does_not_starve_crypto(monkeypatch):
    """The exact live-promax failure mode, fixed: an equity side that cannot
    resolve a securityId must degrade, not stall the crypto agents."""
    live_env(monkeypatch)
    broker = make_live_broker(monkeypatch, {"BTCUSDT": "swap"}, ScriptedCcxt())
    broker.login()
    from ox.agents.base import SharedDataBus
    bus = SharedDataBus()

    class BrokenEquity:
        name = "dhan"
        security_map = {}  # no resolvable securityId -> OrderError on ltps

        def ltps(self, syms):
            raise MarketDataError("No Dhan securityId is configured for YESBANK")

    pump = DataPump(bus, BrokenEquity(), broker,
                    symbols=["YESBANK"], crypto_symbols=["BTCUSDT"],
                    interval_seconds=0.02)

    async def _two_ticks():
        stop = asyncio.Event()
        task = asyncio.create_task(pump.run(stop))
        for _ in range(500):
            if pump.ticks >= 2:
                break
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(_two_ticks())
    assert pump.ticks >= 2, "broken equity side starved the whole pump"
    crypto_tick = bus.get("market:BTCUSDT")
    assert crypto_tick is not None and crypto_tick["source"] == "crypto_broker"
    assert bus.get("market:YESBANK") is None  # the failing venue publishes nothing


# ── 4. operator markers: live-test / track-record ────────────────────────────

def test_live_test_fails_closed_without_credentials(monkeypatch, capsys):
    from ox.live_test import run as live_test_run
    monkeypatch.delenv("DHAN_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    code = live_test_run(prime_seconds=0.0)
    out = capsys.readouterr().out
    assert code == 2
    assert "LIVE-TEST: set DHAN_TOKEN and DHAN_CLIENT_ID in the environment" in out


def test_track_record_empty_marker(monkeypatch, tmp_path, capsys):
    from ox.track_record import main as track_main
    journal = tmp_path / "sessions.jsonl"
    assert track_main([str(journal)]) == 0
    out = capsys.readouterr().out
    assert f"track-record: no sessions journaled yet at {journal}" in out
    assert "python run.py live-test" in out