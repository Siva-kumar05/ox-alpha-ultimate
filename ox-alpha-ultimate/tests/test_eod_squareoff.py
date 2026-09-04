"""Run-loop-level EOD square-off drill across the 15:15 boundary.

The user's no-overnight / no-debt rule lives in ``tick_once``: while
``squareoff <= hhmm() < market_close`` the loop squares off every open
position before any other work, and after ``market_close`` the EOD stats
journal once.  The OMS-level mechanics (bounded attempts, partial journal,
stop re-arm) are contract-tested; what was never driven is the RUN LOOP
crossing the boundary with real positions held.

This drill inherits the resilience harness (real Agent + real DhanBroker over
the scripted ``_FakeSession`` transport, real ``run_forever``) and pins
``hhmm``/``now`` to a deterministic Monday so the session-end boundary fires
exactly when the test says so:

* scenario 1: two positions open before 15:15, zero exit attempts until the
  boundary, then square-off flattens each exactly once, trades + audits are
  journaled, EOD stats fire past 15:30, the loop survives into the
  post-session state, and a reconcile against an empty broker book passes;
* scenario 2: a partial square-off (30/75 fills) journals the partial,
  re-arms a broker stop, raises -> run_forever halts LOUDLY (KILL.flag +
  audit), and the kill switch flattens the residual 45 through the venue
  exactly once - the code path cannot leave a position open overnight.

The harness's per-tick reconcile cadence (``reconcile_every_tick``) is used
so the drill scripts the broker position book alongside every healthy tick;
the square-off tick itself early-returns before the reconcile block, so the
boundary needs no extra position scripts.  No network, no wall-clock
dependence: the boundary is clock-pinned, every broker call is scripted.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from unittest import mock

from ox.core import IST
from test_run_loop_resilience import (
    RunLoopResilienceTests,
    _FakeSession,
    _position_payload,
)

_EOD_TOP = {
    "symbols": ["TCS", "INFY"],
    "security_map": {"TCS": "1333", "INFY": "1594"},
    "market_open": "09:15",
    "entry_cutoff": "14:45",
    "squareoff": "15:15",
    "market_close": "15:30",
}

# _position_payload returns a LIST of position rows; concatenate the two
# books.  Each row must carry its own securityId or the broker-side reverse
# map resolves both rows to TCS and boot reconcile drops INFY locally.
_BOOK = _position_payload(sym="TCS", qty=75) + _position_payload(sym="INFY", qty=75, avg=1400.05, sid="1594")


class EODSquareoffDrill(RunLoopResilienceTests):
    """Real run_forever across the session-end boundary, clock pinned."""

    def _pinned_clock(self, clock):
        monday = datetime(2026, 9, 7, 9, 20, tzinfo=IST)  # a trading day, fixed
        return mock.patch("ox.agent.hhmm", side_effect=lambda: clock["hhmm"]), \
            mock.patch("ox.agent.now", return_value=monday)

    def _rows(self, sql: str):
        conn = sqlite3.connect(os.path.join(self._directory, "smoke.db"))
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def _wait_until(self, predicate, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("condition not met within timeout")

    @staticmethod
    def _quote_count(session: _FakeSession) -> int:
        return len([c for c in session.calls if c[:2] == ("POST", "/marketfeed/quote")])

    @staticmethod
    def _exit_posts(session: _FakeSession) -> list[dict]:
        return [body for method, path, body in session.calls
                if (method, path) == ("POST", "/orders")]

    def _script_entry(self, agent, session, sym, sid, price, order_id, *, avg, target):
        session.script("POST", "/marketfeed/ltp",
                       payload={"data": {"NSE_EQ": {sid: {"last_price": price}}}})
        session.script("POST", "/super/orders", payload={
            "orderId": order_id, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        session.script("GET", "/super/orders", payload=[{
            "orderId": order_id, "orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": avg,
        }])
        breakeven = agent.oms.calculator.min_breakeven_sell_price(avg, 75, buffer_pct=0.001)
        if breakeven > target:
            session.script("PUT", f"/super/orders/{order_id}", payload={})
        return agent.oms.open_position(sym, "BUY", 75, "eod_drill", price * 0.9, target, "signal")

    def _script_full_exit(self, session, super_id, order_id, *, qty=75, price=1103.0):
        session.script("DELETE", f"/super/orders/{super_id}/ENTRY_LEG", payload={})
        session.script("POST", "/orders", payload={
            "orderId": order_id, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        session.script("GET", f"/orders/{order_id}", payload={
            "orderId": order_id, "orderStatus": "TRADED", "filledQty": qty, "averageTradedPrice": price,
        })

    def _script_tick(self, session, n: int = 3) -> None:
        # Prices must sit BETWEEN each position's stop and target, or the
        # paper-mode OMS.mark would close them early (TCS sl 990 / tp 1100.6,
        # INFY sl 1260 / tp 1410).  ltp response carries per-sid dicts (the
        # shape DhanBroker.ltps parses); the quote keeps day-cumulative volume
        # at 0 so _apply_volumes stays a no-op; the per-tick reconcile sees
        # the two open positions on the broker book.
        ltp_payload = {"data": {"NSE_EQ": {
            "1333": {"last_price": 1100.0}, "1594": {"last_price": 1400.0}}}}
        quote_payload = {"data": {"NSE_EQ": {
            "1333": {"last_price": 1100.0, "volume": 0},
            "1594": {"last_price": 1400.0, "volume": 0}}}}
        for _ in range(n):
            session.script("POST", "/marketfeed/ltp", payload=ltp_payload)
            session.script("POST", "/marketfeed/quote", payload=quote_payload)
            session.script("GET", "/positions", payload=_BOOK)

    # ── scenario 1: full square-off, exactly once per position ────────────────

    def test_full_squareoff_flattens_each_position_exactly_once(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session, top=_EOD_TOP, reconcile_every_tick=True)
        # Boot restore must reconcile against the positions we open next.
        session.responses[("GET", "/positions")] = deque([
            dict(status=200, payload=_BOOK, exc=None),
        ])
        self.assertIsNotNone(
            self._script_entry(agent, session, "TCS", "1333", 1100.0, "SO1", avg=1100.05, target=1100.6))
        self.assertIsNotNone(
            self._script_entry(agent, session, "INFY", "1594", 1400.0, "SO2", avg=1400.05, target=1410.0))
        self.assertEqual(len(agent.oms.positions), 2)

        # Generous pre-boundary tick budget: the clock flip is driven from
        # this thread, so the loop must never run out of scripts first.  Each
        # healthy tick reconciles against the two open positions.
        self._script_tick(session, n=100)
        self._script_full_exit(session, "SO1", "EO1", price=1103.0)
        self._script_full_exit(session, "SO2", "EO2", qty=75, price=1403.0)

        clock = {"hhmm": "09:20"}
        hhmm_patch, now_patch = self._pinned_clock(clock)
        with hhmm_patch, now_patch:
            thread = threading.Thread(target=self._run_forever, args=(agent,), daemon=True)
            thread.start()
            try:
                # phase 1: healthy ticks BEFORE the boundary, positions open
                self._wait_until(lambda: self._quote_count(session) >= 2, 60)
                self.assertEqual(self._exit_posts(session), [],
                                 "no exit may go out before the square-off window")
                # phase 2: cross into [15:15, 15:30) -> square-off fires
                clock["hhmm"] = "15:16"
                self._wait_until(lambda: len(self._exit_posts(session)) == 2, 60)
                self.assertTrue(thread.is_alive(), "loop died at square-off")
                # phase 3: past market_close -> EOD stats journal, loop survives
                clock["hhmm"] = "15:31"
                self._wait_until(
                    lambda: agent.db.kv_get("portfolio_stats", None) is not None, 60)
                self.assertTrue(thread.is_alive(), "loop died at EOD stats")
                # ledger vs broker truth: an empty broker book reconciles clean.
                # Swap the scripted book for an empty one (it sits ahead of the
                # leftover phase-1 BOOK scripts in the queue).
                session.responses[("GET", "/positions")] = deque([
                    dict(status=200, payload=[], exc=None)])
                agent.oms.reconcile()
                self.assertTrue(agent.oms.live)
            finally:
                agent.stop = True
                thread.join(timeout=15)

        self.assertFalse(thread.is_alive(), "run_forever did not stop")
        self.assertFalse(agent.comp.halted, agent.comp.halt_reason)
        self.assertFalse(os.path.exists(os.path.join(self._directory, "KILL.flag")))
        self.assertEqual(agent.oms.positions, {}, "positions left open after square-off")
        trades = sorted((str(r[0]), int(r[1]), str(r[2]))
                        for r in self._rows("SELECT sym,qty,exit_reason FROM trades"))
        self.assertEqual(trades, [("INFY", 75, "EOD_SQUAREOFF"), ("TCS", 75, "EOD_SQUAREOFF")])
        actions = self._audit_actions()
        self.assertEqual(actions.count("POSITION_CLOSED"), 2)
        # exactly one market SELL per open position, all at the boundary
        sells = self._exit_posts(session)
        self.assertEqual(len(sells), 2)
        self.assertTrue(all(body["transactionType"] == "SELL" for body in sells))
        ltp_indexes = [i for i, c in enumerate(session.calls)
                       if c[:2] == ("POST", "/marketfeed/ltp")]
        order_indexes = [i for i, c in enumerate(session.calls)
                         if c[:2] == ("POST", "/orders")]
        self.assertGreater(min(order_indexes), max(ltp_indexes),
                           "exits went out before the boundary ticks")
        # EOD stats journaled past market_close
        self.assertTrue(self._rows("SELECT COUNT(*) FROM equity")[0][0] >= 1)

    # ── scenario 2: partial square-off halts loud, kill flattens residual ─────

    def test_partial_squareoff_halts_loudly_and_kill_flattens_residual(self):
        session = _FakeSession()
        agent = self._agent_with_scripted_session(session, top=_EOD_TOP, reconcile_every_tick=True)
        session.responses[("GET", "/positions")] = deque([
            dict(status=200, payload=_position_payload(sym="TCS", qty=75), exc=None),
        ])
        self.assertIsNotNone(
            self._script_entry(agent, session, "TCS", "1333", 1100.0, "SO1", avg=1100.05, target=1100.6))
        self.assertEqual(len(agent.oms.positions), 1)

        # square-off: bracket cancel, then legs fill 30, 0, 0 -> journal the
        # partial, re-arm a broker stop for the 45, raise OrderError.  The
        # halt then kill-switches: bracket cancel again, market exit 45.
        session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        for attempt, filled in (("PE1", 30), ("PE2", 0), ("PE3", 0)):
            session.script("POST", "/orders", payload={
                "orderId": attempt, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0})
            session.script("GET", f"/orders/{attempt}", payload={
                "orderId": attempt, "orderStatus": "TRADED", "filledQty": filled,
                "averageTradedPrice": 1103.0})
        session.script("POST", "/orders", payload={
            "orderId": "PS1", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0})
        # kill switch flattens the residual 45 through the venue, exactly once
        session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        session.script("POST", "/orders", payload={
            "orderId": "KO1", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0})
        session.script("GET", "/orders/KO1", payload={
            "orderId": "KO1", "orderStatus": "TRADED", "filledQty": 45, "averageTradedPrice": 1103.0})
        session.script("GET", "/positions", payload=[])  # post-halt reconcile

        clock = {"hhmm": "15:16"}  # straight into the square-off window
        hhmm_patch, now_patch = self._pinned_clock(clock)
        with hhmm_patch, now_patch:
            thread = threading.Thread(target=self._run_forever, args=(agent,), daemon=True)
            thread.start()
            try:
                self._wait_until(lambda: agent.comp.halted, 90)
            finally:
                agent.stop = True
                thread.join(timeout=15)

        self.assertFalse(thread.is_alive(), "run_forever did not stop after halt")
        self.assertTrue(agent.comp.halted)
        self.assertIn("partially filled", agent.comp.halt_reason)
        # the halt was LOUD, never silent
        self.assertTrue(os.path.exists(os.path.join(self._directory, "KILL.flag")))
        self.assertEqual(agent.oms.positions, {}, "residual was not flattened")
        # 30 journaled partial + 45 kill-switch flatten: nothing open overnight
        trades = sorted((str(r[0]), int(r[1]), str(r[2]))
                        for r in self._rows("SELECT sym,qty,exit_reason FROM trades"))
        self.assertEqual(trades, [("TCS", 30, "EOD_SQUAREOFF_PARTIAL"), ("TCS", 45, "KILL_SWITCH")])
        actions = self._audit_actions()
        for expected in ("KILL_SWITCH", "PARTIAL_EXIT_JOURNALED",
                         "PROTECTIVE_STOP_REARMED", "POSITION_CLOSED"):
            self.assertIn(expected, actions)
        posts = self._exit_posts(session)
        self.assertEqual(len(posts), 5)  # 3 exit legs + 1 stop re-arm + 1 kill flatten
        self.assertEqual(posts[3]["orderType"], "STOP_LOSS")
        self.assertEqual(posts[3]["quantity"], 45)
        self.assertEqual(posts[4]["transactionType"], "SELL")
        self.assertEqual(posts[4]["quantity"], 45)
        # broker truth: an empty book reconciles clean after the halt; the
        # fail-closed state persists (the kill switch is not silently undone)
        agent.oms.reconcile()
        self.assertFalse(agent.oms.live)


# The base class is imported at module level, so pytest's unittest collector
# would re-run its scenarios from THIS file too; drop it from the namespace
# after subclassing (the subclass keeps the MRO) and null the inherited
# methods on the subclass so only the two drills run here.
del RunLoopResilienceTests
for _inherited in [name for name in dir(EODSquareoffDrill)
                   if name.startswith("test_") and name not in EODSquareoffDrill.__dict__]:
    setattr(EODSquareoffDrill, _inherited, None)


if __name__ == "__main__":
    import unittest

    unittest.main()