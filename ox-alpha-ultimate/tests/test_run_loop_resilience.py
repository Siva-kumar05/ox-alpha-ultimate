"""run_forever resilience contract: slow, flaky and silently-hanging brokers.

The Agent's 3-second tick loop calls the Dhan adapter synchronously.  The
designed protections live in run_forever's exception ladder (RateLimitError
-> exponential backoff, BrokerError -> consecutive-error counter, OrderError
-> immediate halt, other -> immediate halt) and in tick_once's circuit
breaker.  None of it was ever exercised against a broker that fails, because
every prior harness used the paper broker or a healthy scripted session.

This module boots a REAL paper-mode Agent whose broker is the REAL DhanBroker
adapter pointed at a scripted HTTP transport, then drives the REAL
run_forever loop and asserts the protections hold:

- consecutive read/network failures escalate 1..5 and halt + kill-switch
- a recovered session resets the counters (no stale escalation, no half-state)
- rate limits back off and never halt; recovery resumes cleanly
- an OrderError (broker/local divergence) halts immediately
- a silently-hanging call costs one bounded read-timeout stall per tick and
  then escalates - it can never stall the loop forever
- circuit-breaker bands map L1 -> L2 -> HALT and a HALT state trips the kill
  switch from inside tick_once

No network, no real broker account, no wall-clock waits beyond the small
configured timeouts.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path

import requests
import yaml

from ox.metrics import CircuitBreaker

_AUDIT_KEY = "run-loop-audit-key-at-least-thirty-two-chars"


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
    """Scripted HTTP transport with an optional socket-latency simulation.

    When latency_sim is enabled, a scripted ReadTimeout behaves like a real
    silent socket: it first waits out the read-timeout the adapter passed,
    then raises - so the test measures the actual bounded stall.
    """

    def __init__(self, latency_sim=False):
        self.responses: dict[tuple[str, str], deque] = {}
        self.calls: list[tuple[str, str, dict]] = []
        self.timeouts: list = []
        self.latency_sim = latency_sim

    def script(self, method, path, *, status=200, payload=None, exc=None):
        self.responses.setdefault((method, path), deque()).append(
            dict(status=status, payload=payload, exc=exc)
        )

    @property
    def remaining(self) -> int:
        return sum(len(q) for q in self.responses.values())

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.replace("https://api.dhan.co/v2", "")
        self.calls.append((method, path, json or {}))
        self.timeouts.append(tuple(timeout) if timeout else None)
        queue = self.responses.get((method, path))
        if not queue:
            raise AssertionError(f"unscripted HTTP call: {method} {path}")
        entry = queue.popleft()
        if entry["exc"] is not None:
            if self.latency_sim and isinstance(entry["exc"], requests.exceptions.ReadTimeout):
                # the socket stays silent until the read budget is spent
                read_budget = (timeout or (0, 10))[1]
                time.sleep(read_budget + 0.02)
            raise entry["exc"]
        return _FakeResponse(entry["status"], entry["payload"])


class _Cfg(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


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


def _ltp_payload(price: float = 1100.0) -> dict:
    return {"data": {"NSE_EQ": {"1333": {"last_price": price}}}}


def _quote_payload() -> dict:
    # day-cumulative volume: 0 keeps _apply_volumes a no-op
    return {"data": {"NSE_EQ": {"1333": {"last_price": 1100.0, "volume": 0}}}}


def _script_successful_tick(session: _FakeSession) -> None:
    session.script("POST", "/marketfeed/ltp", payload=_ltp_payload())
    session.script("POST", "/marketfeed/quote", payload=_quote_payload())


def _position_payload(sym="TCS", qty=75, avg=1100.05, sid="1333") -> list[dict]:
    return [{"securityId": sid, "tradingSymbol": sym, "netQty": qty,
             "averagePrice": avg, "productType": "INTRADAY"}]


def _script_full_entry(session: _FakeSession, order_id="SO1", qty=75) -> None:
    """Healthy broker-managed entry: quote -> bracket -> full fill -> bump."""
    session.script("POST", "/marketfeed/ltp", payload=_ltp_payload())
    session.script("POST", "/super/orders", payload={
        "orderId": order_id, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
    })
    session.script("GET", "/super/orders", payload=[{
        "orderId": order_id, "orderStatus": "TRADED", "filledQty": qty,
        "averageTradedPrice": 1100.05,
    }])
    # breakeven (fees + buffer) exceeds the requested 1100.6 target -> bump
    session.script("PUT", f"/super/orders/{order_id}", payload={})


def _script_full_exit(session: _FakeSession, order_id="XO1", order_id_super="SO1", qty=75, price=1103.0) -> None:
    """Healthy flatten: cancel bracket -> market exit -> confirmed fill."""
    session.script("DELETE", f"/super/orders/{order_id_super}/ENTRY_LEG", payload={})
    session.script("POST", "/orders", payload={
        "orderId": order_id, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
    })
    session.script("GET", f"/orders/{order_id}", payload={
        "orderId": order_id, "orderStatus": "TRADED", "filledQty": qty, "averageTradedPrice": price,
    })


def _script_failed_tick(session: _FakeSession, exc: Exception) -> None:
    # the LTP call fails first; the quote snapshot is never reached
    session.script("POST", "/marketfeed/ltp", exc=exc)


def _make_config(directory: Path, **overrides) -> Path:
    import run

    path = run._smoke_config(directory)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update({
        "platform": "dhan",
        "mode": "paper",
        "symbols": ["TCS"],
        "security_map": {"TCS": "1333"},
        "market_open": "00:00", "entry_cutoff": "23:50", "squareoff": "23:55", "market_close": "23:59",
        "tick_seconds": 0.001,
        "order_flow": {"enabled": False, "primary": False},
        "auto_train_on_boot": False,
    })
    # Universe scanning runs inside boot() post-login and would consume the
    # scripted /charts/intraday responses meant for refresh_history; the
    # resilience contract does not cover scanning.
    raw["universe"]["auto_scan"] = False
    raw["training"]["min_symbols"] = 1  # harness trades a single symbol
    raw["execution"].update({
        "rate_limit_backoff_seconds": 0.01,
        "max_rate_limit_backoff_seconds": 0.05,
        "max_consecutive_broker_errors": 5,
    })
    raw["execution"].update(overrides.get("execution", {}))
    raw.update(overrides.get("top", {}))
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


class RunLoopResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in ("OX_AUDIT_KEY", "DHAN_CLIENT_ID", "DHAN_TOKEN")}
        os.environ.update({
            "OX_AUDIT_KEY": _AUDIT_KEY,
            "DHAN_CLIENT_ID": "TEST-CLIENT-ID",
            "DHAN_TOKEN": "run-loop-access-token-not-dummy-32chars",
        })
        self._directory = tempfile.mkdtemp(prefix="ox-runloop-")
        self._agents = []

    def tearDown(self) -> None:
        for agent in self._agents:
            try:
                agent.close()
            except Exception:
                pass
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._directory, ignore_errors=True)

    def _agent_with_scripted_session(self, session: _FakeSession, *, reconcile_every_tick=False, **cfg_overrides):
        """Real Agent over the real DhanBroker with a scripted transport.

        run_forever() runs boot() itself, so boot-time HTTP calls are scripted
        here and the caller then hands the loop straight to run_forever.
        """
        from ox.agent import Agent

        cfg_path = _make_config(Path(self._directory), **cfg_overrides)
        agent = Agent(str(cfg_path))
        agent.cognition = None  # distillation is not part of the resilience path
        # Background infra managers (backup/rotation/config-watch/shutdown) and
        # the health sweep add minutes of thread churn on this repo's slow
        # sync layer and are unrelated to the sync broker-resilience contract
        # under test (the run_forever error ladder + tick classification).
        for attr in ("health_checker", "backup_manager", "config_watcher",
                     "secret_manager", "shutdown_manager"):
            setattr(agent, attr, None)
        # run_forever schedules nightly_training at the hardcoded 18:00 IST
        # boundary and fires it on the first tick past that hour.  With the
        # smoke-relaxed promotion gates (min_trades 1 / promote_score -9.0)
        # that retrain PROMOTES strategies mid-test after 18:00 IST, so these
        # broker-resilience contracts only ever passed before 18:00 - the
        # agent would start trading against an unscripted session.  Training
        # is not part of the resilience contract under test (auto_train_on_boot
        # is False here), so disable the scheduled retrain for this agent.
        agent.nightly_training = lambda: None  # type: ignore[method-assign]
        agent.broker.session = session  # scripted transport, everything else stays real
        session.script("GET", "/fundlimit", payload={})
        session.script("GET", "/positions", payload=[])  # oms.restore() reconcile
        for _symbol in agent.cfg["symbols"]:
            session.script("POST", "/charts/intraday", payload=_candle_payload())
        if reconcile_every_tick:
            agent.cfg["execution"]["reconcile_interval_seconds"] = 0.0
        self._agents.append(agent)
        return agent

    @staticmethod
    def _run_forever(agent) -> None:
        agent.run_forever()

    def _audit_actions(self) -> list[str]:
        """run_forever ends with agent.close(), which closes agent.db; read the
        audit trail from the sqlite file directly for post-run assertions."""
        conn = sqlite3.connect(os.path.join(self._directory, "smoke.db"))
        try:
            return [row[0] for row in conn.execute("SELECT action FROM audit")]
        finally:
            conn.close()

    # ------------------------------------------------------------------ A #

    def test_consecutive_failures_escalate_to_halt_and_kill_switch(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session)
        fail = requests.exceptions.ConnectionError("reset")
        # ok, fail, ok, fail, ok then five consecutive failures: counters must
        # reset on each success so only the final run trips the threshold.
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=[])  # first-tick reconcile
        _script_failed_tick(session, fail)
        _script_successful_tick(session)
        _script_failed_tick(session, fail)
        _script_successful_tick(session)
        for _ in range(5):
            _script_failed_tick(session, fail)

        self._run_forever(agent)  # returns only when run_forever stops itself

        self.assertTrue(agent.comp.halted)
        self.assertIn("Repeated broker/data error", agent.comp.halt_reason)
        self.assertEqual(agent.broker_error_count, 5)  # reset by the ok ticks
        self.assertTrue(agent.stop)
        self.assertTrue((Path(self._directory) / "KILL.flag").exists())
        actions = self._audit_actions()
        self.assertIn("KILL_SWITCH", actions)
        self.assertEqual(agent.oms.positions, {})          # no half-state left
        self.assertEqual(agent.oms.inflight_orders, {})

    # ------------------------------------------------------------------ B #

    def test_rate_limits_back_off_never_halt_and_recovery_resumes(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session)
        for _ in range(4):
            session.script("POST", "/marketfeed/ltp", status=429,
                           payload={"errorMessage": "DH-904 rate limit"})
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=[])  # first-tick reconcile
        for _ in range(5):
            _script_successful_tick(session)

        thread = threading.Thread(target=self._run_forever, args=(agent,), daemon=True)
        thread.start()
        saw_rate_limited = False
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if agent.db.kv_get("agent_health", {}).get("state") == "RATE_LIMITED":
                    saw_rate_limited = True
                if saw_rate_limited and agent.rate_limit_count == 0 and session.remaining == 0:
                    break
                time.sleep(0.01)
            agent.stop = True
            thread.join(timeout=10.0)
        finally:
            agent.stop = True
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive(), "run_forever did not stop")
        self.assertTrue(saw_rate_limited, "health never entered RATE_LIMITED")
        self.assertFalse(agent.comp.halted, "rate limits must not halt the agent")
        self.assertEqual(agent.rate_limit_count, 0)  # reset by the recovery ticks
        self.assertFalse((Path(self._directory) / "KILL.flag").exists())
        self.assertNotIn("HALT", self._audit_actions())

    # ------------------------------------------------------------------ C #

    def test_order_error_halts_immediately_on_broker_divergence(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session, reconcile_every_tick=True)
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=[])  # first reconcile: clean
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=[{"tradingSymbol": "ZZZ", "netQty": 3}])

        self._run_forever(agent)

        self.assertTrue(agent.comp.halted)
        self.assertTrue(agent.comp.halt_reason.startswith("Order execution uncertainty"))
        self.assertFalse(agent.oms.live)  # reconcile failed closed before the raise
        self.assertTrue((Path(self._directory) / "KILL.flag").exists())

    # ------------------------------------------------------------------ D #

    def test_hung_call_is_bounded_by_read_timeout_then_escalates(self):
        read_budget = 0.2
        session = _FakeSession(latency_sim=True)
        agent = self._agent_with_scripted_session(
            session,
            execution={"broker_connect_timeout_seconds": 0.05, "broker_read_timeout_seconds": read_budget},
        )
        for _ in range(5):
            _script_failed_tick(session, requests.exceptions.ReadTimeout("silent socket"))

        started = time.monotonic()
        self._run_forever(agent)
        elapsed = time.monotonic() - started

        # every call carried the configured budget, and each hung tick cost
        # about one read timeout before escalating -> bounded, not forever
        self.assertTrue(all(t == (0.05, read_budget) for t in session.timeouts),
                        f"timeout wiring wrong: {session.timeouts}")
        self.assertTrue(agent.comp.halted)
        self.assertIn("Repeated broker/data error", agent.comp.halt_reason)
        self.assertGreaterEqual(elapsed, 0.9, "hung calls must stall for their read budget")
        self.assertLess(elapsed, 8.0, "a hung call stalled the loop far beyond one budget")

    # ------------------------------------------------------------------ E #

    def test_circuit_breaker_bands_map_l1_l2_to_halt(self):
        breaker = CircuitBreaker({"self_healing": {
            "l1_sharpe_threshold": 0.3, "l2_sharpe_threshold": 0.0,
            "l3_sharpe_threshold": -0.5, "degraded_size_multiplier": 0.5,
        }})
        self.assertEqual(breaker.evaluate(0.2), "L1_REDUCE_SIZE")
        self.assertEqual(breaker.get_size_multiplier(), 0.5)
        self.assertEqual(breaker.evaluate(-0.2), "L2_STOP_ENTRIES")
        self.assertEqual(breaker.get_size_multiplier(), 0.0)
        self.assertTrue(breaker.should_block_entries())
        self.assertFalse(breaker.should_halt())
        self.assertEqual(breaker.evaluate(-1.0), "HALT")
        self.assertTrue(breaker.should_halt())
        self.assertEqual(breaker.evaluate(1.0), "NORMAL")

    def test_persistent_losses_trip_breaker_halt_and_kill_switch(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session)
        # monotonic drawdown: 25 equity points from 500000 -> 460000
        base = 500000.0
        for index in range(25):
            agent.db.ex("INSERT OR REPLACE INTO equity VALUES(?,?)",
                        (f"2026-01-0{1 + index // 10}T00:{index:02d}:00+05:30",
                         round(base - index * 1666.0, 2)))
        session.script("POST", "/marketfeed/ltp", payload=_ltp_payload())
        session.script("POST", "/marketfeed/quote", payload=_quote_payload())
        session.script("GET", "/positions", payload=[])  # first-tick reconcile

        self._run_forever(agent)

        self.assertTrue(agent.comp.halted)
        self.assertIn("Circuit breaker", agent.comp.halt_reason)
        self.assertEqual(agent.circuit_breaker.state, "HALT")
        self.assertTrue((Path(self._directory) / "KILL.flag").exists())
        actions = self._audit_actions()
        self.assertIn("KILL_SWITCH", actions)

    # ------------------------------------------------------------ chaos C #
    # Chaos scenarios the clean-path harness never exercised: a quote feed
    # that dies and never recovers while a position is open, a KILL.flag
    # dropped mid-session, and a feed that disconnects and reconnects.

    def test_feed_death_with_open_position_halts_and_flattens_exactly_once(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session)
        # Boot restore must reconcile against the position we open next
        # (replace the helper's empty-book entry with the position book).
        session.responses[("GET", "/positions")] = deque([
            dict(status=200, payload=_position_payload(), exc=None),
        ])
        _script_full_entry(session)
        position = agent.oms.open_position("TCS", "BUY", 75, "chaos", 1000.0, 1100.6, "feed_death")
        self.assertIsNotNone(position)
        self.assertEqual(agent.oms.positions["TCS"]["qty"], 75)

        # One healthy tick (its trailing reconcile sees the position), then
        # the feed dies and NEVER recovers.
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=_position_payload())
        for _ in range(5):
            _script_failed_tick(session, requests.exceptions.ConnectionError("feed down"))
        # The halt kill-switch flattens through the broker; script that exit.
        _script_full_exit(session)

        self._run_forever(agent)

        self.assertTrue(agent.comp.halted)
        self.assertIn("Repeated broker/data error", agent.comp.halt_reason)
        self.assertTrue(agent.stop)
        self.assertTrue((Path(self._directory) / "KILL.flag").exists())
        # Flattened exactly once through the venue: one market SELL, position
        # gone locally, P&L journaled - nothing fabricated, nothing doubled.
        order_posts = [body for method, path, body in session.calls
                       if (method, path) == ("POST", "/orders")]
        self.assertEqual(len(order_posts), 1, "kill switch must flatten exactly once")
        self.assertEqual(order_posts[0]["transactionType"], "SELL")
        self.assertEqual(agent.oms.positions, {})
        actions = self._audit_actions()
        self.assertIn("KILL_SWITCH", actions)
        self.assertIn("POSITION_CLOSED", actions)
        self.assertIn("POSITION_OPENED", actions)

    def test_kill_flag_dropped_mid_session_stops_and_flattens_once(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session)
        _script_full_entry(session)
        position = agent.oms.open_position("TCS", "BUY", 75, "chaos", 1000.0, 1100.6, "kill_flag")
        self.assertIsNotNone(position)
        self.assertTrue(agent.oms.live)

        # Operator drops the flag; the next tick_once must stop deliberately
        # and flatten through the broker, not fabricate or double-execute.
        kill_path = Path(agent.cfg.root) / "KILL.flag"
        kill_path.write_text("OPERATOR\n", encoding="utf-8")
        _script_full_exit(session)

        agent.tick_once()

        self.assertTrue(agent.stop, "KILL.flag must stop the loop deliberately")
        order_posts = [body for method, path, body in session.calls
                       if (method, path) == ("POST", "/orders")]
        self.assertEqual(len(order_posts), 1, "kill switch must flatten exactly once")
        self.assertEqual(agent.oms.positions, {})
        self.assertFalse(agent.oms.live)
        self.assertTrue(kill_path.exists())
        actions = [row[0] for row in agent.db.q("SELECT action FROM audit")]
        self.assertIn("KILL_SWITCH", actions)
        self.assertIn("POSITION_CLOSED", actions)
        kill_events = agent.db.q("SELECT msg FROM events WHERE kind='KILL'")
        self.assertTrue(kill_events and "KILL.flag detected" in kill_events[0][0])

    def test_feed_disconnect_mid_run_reconnects_without_half_state(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session, reconcile_every_tick=True)
        # healthy, dead, then healthy again (with slack so the loop never runs
        # out of scripted responses before the test stops it); every healthy
        # tick reconciles against an empty broker book and must stay consistent.
        _script_successful_tick(session)
        session.script("GET", "/positions", payload=[])
        _script_failed_tick(session, requests.exceptions.ConnectionError("blip"))
        for _ in range(12):
            _script_successful_tick(session)
            session.script("GET", "/positions", payload=[])

        def _ltp_calls():
            return [c for c in session.calls if c[:2] == ("POST", "/marketfeed/ltp")]

        thread = threading.Thread(target=self._run_forever, args=(agent,), daemon=True)
        thread.start()
        try:
            # Stop once the loop has demonstrably recovered: several healthy
            # ticks after the single failed one (counter reset to 0).  Plain
            # attribute/call signals only - no DB reads that could race close.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if (len(_ltp_calls()) >= 6 and agent.broker_error_count == 0
                        and agent.rate_limit_count == 0):
                    break
                time.sleep(0.01)
            agent.stop = True
            thread.join(timeout=15.0)
        finally:
            agent.stop = True
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive(), "run_forever did not stop")
        self.assertFalse(agent.comp.halted, "a transient feed blip must not halt")
        self.assertEqual(agent.broker_error_count, 0, "recovery must reset the counter")
        self.assertTrue(agent.oms.live, "reconcile stayed consistent: no half-state")
        self.assertEqual(agent.oms.positions, {})
        self.assertFalse((Path(self._directory) / "KILL.flag").exists())
        self.assertNotIn("HALT", self._audit_actions())


if __name__ == "__main__":
    unittest.main()
