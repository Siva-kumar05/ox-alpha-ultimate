"""Live Dhan broker boundary contract, exercised offline.

The agent's whole purpose is autonomous buy/sell on a live broker, yet the
Dhan adapter - super-order bracket placement, order-flow confirmation, HTTP
error classification, and the fail-closed kill switch - is never exercised by
the paper-broker smoke suite.  This module drives the REAL DhanBroker and OMS
code against a scripted HTTP session so ordering and failure behaviour are
proven deterministically:

- entry -> LTP quote -> bracket Super Order -> confirmation -> breakeven target
- timeout / disconnect handling while awaiting a fill
- half-state handling when the uncertainty-cancel itself fails (kill switch)
- PART_TRADED residual cancellation and re-confirmation
- 429 classification: mutations fail closed (OrderError), reads are retryable
  (RateLimitError), network faults wrap as BrokerError

No network, no broker account, and no wall-clock sleeps are involved.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from collections import deque
from pathlib import Path

import requests
import yaml

from ox.brokers import BrokerError, DhanBroker, OrderError, RateLimitError
from ox.compliance import Compliance
from ox.core import DB, SecurityError
from ox.oms import OMS

_AUDIT_KEY = "live-contract-audit-key-at-least-thirty-two-chars"
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
    """Scripted HTTP transport: responses keyed by (method, path), call log kept."""

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


class LiveBrokerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prior_key = os.environ.get("OX_AUDIT_KEY")
        os.environ["OX_AUDIT_KEY"] = _AUDIT_KEY
        self._directory = tempfile.mkdtemp(prefix="ox-live-contract-")
        raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
        self.cfg = _AttrDict(raw)
        self.cfg.update({
            "root": self._directory,
            "db_path": str(Path(self._directory) / "test.db"),
            "security_map": {"TCS": "1333"},
            "execution": dict(raw["execution"], order_confirm_timeout_seconds=2),
        })
        self.db = DB(Path(self._directory) / "test.db")
        self.session = _FakeSession()
        self.broker = DhanBroker(self.cfg, self.db)
        self.broker.client_id = "TEST-CLIENT-ID"
        self.broker.session = self.session
        self.broker._set_token("live-test-access-token-not-dummy-32chars")
        self.risk = _StubRisk()
        self.oms = OMS(self.cfg, self.db, self.broker, self.risk)

    def tearDown(self) -> None:
        self.db.close()
        if self._prior_key is None:
            os.environ.pop("OX_AUDIT_KEY", None)
        else:
            os.environ["OX_AUDIT_KEY"] = self._prior_key
        shutil.rmtree(self._directory, ignore_errors=True)

    def _script_ltp(self) -> None:
        self.session.script("POST", "/marketfeed/ltp", payload=_LTP_PAYLOAD)

    def _script_open(self, order_id="SO1", *, pending_status="PENDING") -> None:
        self._script_ltp()
        self.session.script("POST", "/super/orders", payload={
            "orderId": order_id, "orderStatus": pending_status, "filledQty": 0, "averageTradedPrice": 0.0,
        })

    def _script_confirm(self, *, status="TRADED", filled_qty=75, avg=1100.05, order_id="SO1") -> None:
        self.session.script("GET", "/super/orders", payload=[{
            "orderId": order_id, "orderStatus": status, "filledQty": filled_qty,
            "averageTradedPrice": avg,
        }])

    # ------------------------------------------------------------- boundary #

    def test_network_fault_wraps_as_broker_error_and_paths_are_locked(self):
        self.session.script("GET", "/positions", exc=requests.exceptions.ConnectionError("reset"))
        with self.assertRaises(BrokerError) as ctx:
            self.broker.positions()
        self.assertIn("network request failed", str(ctx.exception))
        # fixed relative paths only
        with self.assertRaises(SecurityError):
            self.broker._request("GET", "https://evil.example/steal")

    def test_mutation_rate_limit_fails_closed_as_order_error(self):
        # A throttled order POST has uncertain broker state: fail closed, never
        # silently retry a mutation.
        self._script_ltp()
        self.session.script("POST", "/super/orders", status=429,
                            headers={"Retry-After": "7"},
                            payload={"errorMessage": "DH-904: rate limit exceeded"})
        with self.assertRaises(OrderError) as ctx:
            self.broker.place_super_order("TCS", "BUY", 75, 1100.6, 1000.0, "tag")
        self.assertIn("rate-limited", str(ctx.exception))

    def test_read_rate_limit_is_retryable_with_backoff_hint(self):
        self.session.script("GET", "/positions", status=429, headers={"Retry-After": "7.5"},
                            payload={"errorMessage": "too many requests"})
        with self.assertRaises(RateLimitError) as ctx:
            self.broker.positions()
        self.assertEqual(ctx.exception.retry_after_seconds, 7.5)

    def test_rejected_super_order_raises_and_is_persisted(self):
        self._script_ltp()
        self.session.script("POST", "/super/orders", payload={
            "orderId": "SOX", "orderStatus": "REJECTED", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        with self.assertRaises(OrderError) as ctx:
            self.broker.place_super_order("TCS", "BUY", 75, 1100.6, 1000.0, "tag")
        self.assertIn("REJECTED", str(ctx.exception))
        status = self.db.q("SELECT status FROM orders WHERE oid='SOX'")[0][0]
        self.assertEqual(status, "REJECTED")

    def test_bracket_must_contain_market_price_no_order_is_sent(self):
        # LTP is 1100.0 but the requested bracket (1050, 1090) sits below it:
        # the broker adapter refuses before any order POST happens.
        self.session.script("POST", "/marketfeed/ltp", payload=_LTP_PAYLOAD)
        with self.assertRaises(OrderError) as ctx:
            self.broker.place_super_order("TCS", "BUY", 75, 1090.0, 1050.0, "tag")
        self.assertIn("bracket", str(ctx.exception))
        self.assertEqual(self.session.call_paths(), [("POST", "/marketfeed/ltp")])

    def test_non_json_response_wraps_as_broker_error(self):
        self.session.script("POST", "/marketfeed/ltp", text="not-json-at-all")
        with self.assertRaises(BrokerError) as ctx:
            self.broker.ltps(["TCS"])
        self.assertIn("non-JSON", str(ctx.exception))

    # ------------------------------------------------------------ lifecycle #

    def test_entry_to_bracket_to_target_exit(self):
        self._script_open()
        self._script_confirm(filled_qty=75, avg=1100.05)
        breakeven = self.oms.calculator.min_breakeven_sell_price(1100.05, 75, buffer_pct=0.001)
        target = 1100.6
        self.assertGreater(breakeven, target, "fixture must exercise the breakeven bump")
        self.session.script("PUT", "/super/orders/SO1", payload={})

        position = self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, target, "signal")
        self.assertIsNotNone(position)
        self.assertEqual(position["qty"], 75)
        self.assertEqual(position["super_order_id"], "SO1")
        self.assertAlmostEqual(position["tp"], breakeven, places=4)  # bumped past fees

        paths = self.session.call_paths()
        order_body = dict(self.session.calls[1][2])
        self.assertEqual(order_body["securityId"], "1333")
        self.assertEqual(order_body["quantity"], 75)
        self.assertEqual(order_body["stopLossPrice"], 1000.0)
        self.assertEqual(order_body["targetPrice"], target)
        self.assertEqual(order_body["productType"], "INTRADAY")
        # ordering: quote -> bracket placement -> confirmation -> target modify
        self.assertEqual(paths, [
            ("POST", "/marketfeed/ltp"), ("POST", "/super/orders"),
            ("GET", "/super/orders"), ("PUT", "/super/orders/SO1"),
        ])

        # target exit: cancel bracket, market exit, journal the trade
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        self.session.script("POST", "/orders", payload={
            "orderId": "EO1", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        self.session.script("GET", "/orders/EO1", payload={
            "orderId": "EO1", "orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 1103.0,
        })
        self.assertTrue(self.oms.close("TCS", "TAKE_PROFIT"))
        self.assertNotIn("TCS", self.oms.positions)
        trade = self.db.q("SELECT sym,qty,pnl,exit_reason FROM trades")[0]
        self.assertEqual((trade[0], trade[1], trade[3]), ("TCS", 75, "TAKE_PROFIT"))
        self.assertGreater(float(trade[2]), 0.0)
        self.assertEqual(self.risk.closed_pnls, [float(trade[2])])
        actions = [row[1] for row in self.db.q("SELECT aid,action FROM audit")]
        self.assertIn("POSITION_OPENED", actions)
        self.assertIn("POSITION_CLOSED", actions)

    def test_confirmed_fill_outside_bracket_fails_closed(self):
        # Broker reports a fill at 1200, far above the requested target: the
        # fill is not inside the protective bracket, so the entry must not be
        # accepted locally and the tracked order must be cancelled.
        self._script_open()
        self._script_confirm(filled_qty=75, avg=1200.0)
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        with self.assertRaises(OrderError) as ctx:
            self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertIn("outside the requested protective bracket", str(ctx.exception))
        self.assertNotIn("TCS", self.oms.positions)
        self.assertEqual(self.oms.inflight_orders, {})
        self.assertEqual(self.db.q("SELECT COUNT(*) FROM trades")[0][0], 0)
        self.assertIn(("DELETE", "/super/orders/SO1/ENTRY_LEG"), self.session.call_paths())

    def test_disconnect_while_awaiting_confirmation_cancels_tracked_entry(self):
        # Network drops between placement and confirmation: the tracked Super
        # Order is cancelled (clean half-state) and the failure propagates so
        # the run loop can count it toward its broker-error threshold.
        self._script_open()
        self.session.script("GET", "/super/orders", exc=requests.exceptions.ConnectionError("reset"))
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        with self.assertRaises(BrokerError):
            self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertNotIn("TCS", self.oms.positions)
        self.assertEqual(self.oms.inflight_orders, {})

    def test_cancel_failure_keeps_half_state_then_kill_switch_marks_it(self):
        # Worst case: the confirm call dies AND the uncertainty-cancel dies.
        # OMS must keep the order id (never pretend the entry vanished) and
        # raise an OrderError so the run loop halts; a kill switch then
        # records the unconfirmed order instead of flattening blind.
        self._script_open()
        self.session.script("GET", "/super/orders", exc=requests.exceptions.ConnectionError("reset"))
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", exc=requests.exceptions.ConnectionError("reset"))
        with self.assertRaises(OrderError) as ctx:
            self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertIn("Could not cancel uncertain Super Order SO1", str(ctx.exception))
        self.assertEqual(self.oms.inflight_orders, {"TCS": "SO1"})  # half-state retained

        # A second cancel attempt during the kill switch also fails -> the
        # unconfirmed order is journaled, never silently dropped.
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", exc=requests.exceptions.ConnectionError("reset"))
        compliance = Compliance(self.cfg, self.db)
        compliance.wire_kill_switch(self.oms)
        compliance.halt("broker unreachable mid-entry")
        self.assertTrue(compliance.halted)
        self.assertFalse(self.oms.live)
        self.assertEqual(self.oms.inflight_orders, {"TCS": "SO1"})
        actions = [row[1] for row in self.db.q("SELECT aid,action FROM audit")]
        self.assertIn("KILL_SWITCH", actions)
        kill_events = self.db.q("SELECT msg FROM events WHERE kind='KILL'")
        self.assertTrue(kill_events and "uncertain entry TCS/SO1" in kill_events[0][0])
        flag = Path(self.cfg["root"]) / "KILL.flag"
        self.assertTrue(flag.exists())
        self.assertIn("HALTED", flag.read_text(encoding="utf-8"))

    def test_fill_above_target_but_below_breakeven_is_rejected(self):
        # Narrow regression for the vacuous-upper-bound defect: breakeven is
        # always above the fill, so a guard of stop < fill < max(target,
        # breakeven) could never reject a fill above the requested target.  A
        # fill at 1100.7 (target 1100.6, breakeven ~1101.15) must fail closed
        # instead of being accepted via a target bump the broker never armed.
        self._script_open()
        self._script_confirm(filled_qty=75, avg=1100.7)
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        with self.assertRaises(OrderError) as ctx:
            self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertIn("outside the requested protective bracket", str(ctx.exception))
        self.assertNotIn("TCS", self.oms.positions)
        self.assertEqual(self.oms.inflight_orders, {})

    def test_part_traded_residual_is_cancelled_and_reconfirmed(self):
        self._script_open()
        self.session.script("GET", "/super/orders", payload=[{
            "orderId": "SO1", "orderStatus": "PART_TRADED", "filledQty": 45, "averageTradedPrice": 1100.05,
        }])
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        self.session.script("GET", "/super/orders", payload=[{
            "orderId": "SO1", "orderStatus": "TRADED", "filledQty": 45, "averageTradedPrice": 1100.05,
        }])
        breakeven_45 = self.oms.calculator.min_breakeven_sell_price(1100.05, 45, buffer_pct=0.001)
        if breakeven_45 > 1100.6:  # fee-coverage bump applies to the smaller fill too
            self.session.script("PUT", "/super/orders/SO1", payload={})
        position = self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertEqual(position["qty"], 45)  # never assumes the full 75 filled
        self.assertEqual(self.oms.positions["TCS"]["qty"], 45)
        rows = self.db.q("SELECT qty FROM positions WHERE sym='TCS'")
        self.assertEqual(rows[0][0], 45)
        paths = self.session.call_paths()
        # quote -> bracket -> PART_TRADED read -> residual cancel -> final read
        # -> (optional) breakeven target bump
        self.assertEqual(paths[:5], [
            ("POST", "/marketfeed/ltp"), ("POST", "/super/orders"),
            ("GET", "/super/orders"), ("DELETE", "/super/orders/SO1/ENTRY_LEG"),
            ("GET", "/super/orders"),
        ])
        self.assertEqual(len(paths), 6 if breakeven_45 > 1100.6 else 5)

    def test_part_traded_residual_unresolved_fails_closed(self):
        self._script_open()
        self.session.script("GET", "/super/orders", payload=[{
            "orderId": "SO1", "orderStatus": "PART_TRADED", "filledQty": 45, "averageTradedPrice": 1100.05,
        }])
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        self.session.script("GET", "/super/orders", payload=[{
            "orderId": "SO1", "orderStatus": "PART_TRADED", "filledQty": 45, "averageTradedPrice": 1100.05,
        }])
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        with self.assertRaises(OrderError) as ctx:
            self.oms.open_position("TCS", "BUY", 75, "test_live", 1000.0, 1100.6, "signal")
        self.assertIn("residual unresolved", str(ctx.exception))
        self.assertNotIn("TCS", self.oms.positions)
        self.assertEqual(self.oms.inflight_orders, {})

    # ------------------------------------------------------------ chaos #
    # Order-sequence chaos on the live execution path: an exit placement
    # that raises mid-sequence and partial fills on the EOD square-off.

    def _open_75(self):
        self._script_open()
        self._script_confirm(filled_qty=75, avg=1100.05)
        # breakeven (fees + buffer) exceeds the 1100.6 target -> target bump
        self.session.script("PUT", "/super/orders/SO1", payload={})
        position = self.oms.open_position("TCS", "BUY", 75, "chaos", 1000.0, 1100.6, "signal")
        self.assertIsNotNone(position)
        return position

    def test_exit_order_raise_mid_sequence_preserves_position_no_fabrication(self):
        self._open_75()
        # Bracket cancel succeeds, then the exit market order dies on the wire.
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        self.session.script("POST", "/orders", exc=requests.exceptions.ConnectionError("reset"))
        with self.assertRaises(BrokerError):
            self.oms.close("TCS", "OPPOSITE_SIGNAL")
        # Local state still matches broker truth: 75 held, no trade fabricated,
        # and exactly ONE exit attempt - never a blind retry or double send.
        self.assertEqual(self.oms.positions["TCS"]["qty"], 75)
        self.assertEqual(self.db.q("SELECT COUNT(*) FROM trades")[0][0], 0)
        self.assertEqual(
            [(m, p) for m, p in self.session.call_paths() if p == "/orders"],
            [("POST", "/orders")],
        )
        # Venue recovers: a fresh close goes out exactly once more and lands.
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        self.session.script("POST", "/orders", payload={
            "orderId": "EO9", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        self.session.script("GET", "/orders/EO9", payload={
            "orderId": "EO9", "orderStatus": "TRADED", "filledQty": 75, "averageTradedPrice": 1103.0,
        })
        self.assertTrue(self.oms.close("TCS", "OPPOSITE_SIGNAL"))
        self.assertNotIn("TCS", self.oms.positions)
        trades = self.db.q("SELECT qty,exit_reason FROM trades")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0][1], "OPPOSITE_SIGNAL")
        order_posts = [m for m, p in self.session.call_paths() if p == "/orders"]
        self.assertEqual(order_posts.count("POST"), 2)  # one failed + one clean

    def test_partial_exit_on_eod_squareoff_journals_and_rearms_stop(self):
        self._open_75()
        # EOD square-off: exit legs keep filling short of the 75 held.
        self.session.script("DELETE", "/super/orders/SO1/ENTRY_LEG", payload={})
        # attempt 1: 30 of 75 fill
        self.session.script("POST", "/orders", payload={
            "orderId": "PE1", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })
        self.session.script("GET", "/orders/PE1", payload={
            "orderId": "PE1", "orderStatus": "TRADED", "filledQty": 30, "averageTradedPrice": 1103.0,
        })
        # attempts 2 and 3: nothing fills (thin book at the bell)
        for attempt in ("PE2", "PE3"):
            self.session.script("POST", "/orders", payload={
                "orderId": attempt, "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
            })
            self.session.script("GET", f"/orders/{attempt}", payload={
                "orderId": attempt, "orderStatus": "TRADED", "filledQty": 0, "averageTradedPrice": 0.0,
            })
        # protective stop re-arm for the residual 45 (A2)
        self.session.script("POST", "/orders", payload={
            "orderId": "PS1", "orderStatus": "PENDING", "filledQty": 0, "averageTradedPrice": 0.0,
        })

        with self.assertRaises(OrderError) as ctx:
            self.oms.squareoff_eod()
        self.assertIn("partially filled", str(ctx.exception))
        # The filled 30 were journaled (P&L never vanishes), the residual 45
        # stays protected by a broker stop, local state shrinks to broker truth.
        trades = self.db.q("SELECT qty,exit_reason FROM trades")
        self.assertEqual(len(trades), 1)
        self.assertEqual((int(trades[0][0]), str(trades[0][1])), (30, "EOD_SQUAREOFF_PARTIAL"))
        self.assertEqual(self.oms.positions["TCS"]["qty"], 45)
        audits = [row[1] for row in self.db.q("SELECT aid,action FROM audit")]
        self.assertIn("PARTIAL_EXIT_JOURNALED", audits)
        self.assertIn("PROTECTIVE_STOP_REARMED", audits)
        # bounded: exactly 3 exit attempts + 1 stop re-arm, never more
        stop_posts = [body for method, path, body in self.session.calls
                      if (method, path) == ("POST", "/orders")]
        self.assertEqual(len(stop_posts), 4)
        self.assertEqual(stop_posts[-1]["orderType"], "STOP_LOSS")
        self.assertEqual(stop_posts[-1]["quantity"], 45)
        # ledger stays consistent with the broker: reconcile of 45 passes
        self.session.script("GET", "/positions", payload=[{
            "securityId": "1333", "tradingSymbol": "TCS", "netQty": 45,
            "averagePrice": 1100.05, "productType": "INTRADAY",
        }])
        self.oms.reconcile()
        self.assertTrue(self.oms.live)


if __name__ == "__main__":
    unittest.main()
