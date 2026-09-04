"""Choice India (Shoonya/Noren) broker contract, exercised offline.

ChoiceBroker is a live-typed adapter whose transport contract is mirrored
from the official Shoonya wrapper (Shoonya-Dev/ShoonyaApi-py) and its
NorenRestApiPy base: form POSTs of 'jData=<json>&jKey=<token>' against
https://api.shoonya.com/NorenWClientTP/<Route>.  This module drives the REAL
ChoiceBroker against the scripted HTTP session reused from
test_live_broker_contract (extended with the Noren form-POST transport) so the
auth handshake, exact order payloads, LTP / history parsing, positions, exits,
and fail-closed behaviour are proven deterministically.

No network, no broker account, and no wall-clock sleeps are involved.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
import yaml

from ox.brokers import (
    AuthenticationError,
    BrokerError,
    ChoiceBroker,
    MarketDataError,
    OrderError,
    PaperBroker,
    make_broker,
)
from ox.core import DB
from support import _AttrDict, _FakeSession

_AUDIT_KEY = "choice-contract-audit-key-at-least-thirty-two-chars"

_ENV_VARS = (
    "CHOICE_USER_ID",
    "CHOICE_PASSWORD",
    "CHOICE_TOTP",
    "CHOICE_VENDOR_CODE",
    "CHOICE_API_KEY",
    "CHOICE_IMEI",
)

_QUOTE_TCS = {"stat": "Ok", "lp": "1100.25", "tok": "2885", "exch": "NSE"}
_SECURITY_MAP = {
    "TCS": "NSE|2885|TCS-EQ",
    "INFY": "NSE|26000|INFY-EQ",
}


class ChoiceBrokerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prior = {name: os.environ.get(name) for name in _ENV_VARS}
        for name in _ENV_VARS:
            os.environ.pop(name, None)
        self._prior_key = os.environ.get("OX_AUDIT_KEY")
        os.environ["OX_AUDIT_KEY"] = _AUDIT_KEY
        os.environ["CHOICE_USER_ID"] = "TEST-UID"
        os.environ["CHOICE_PASSWORD"] = "secret-pass"
        os.environ["CHOICE_TOTP"] = "123456"
        os.environ["CHOICE_VENDOR_CODE"] = "TEST-VC"
        os.environ["CHOICE_API_KEY"] = "test-api-key"
        os.environ["CHOICE_IMEI"] = "test-imei"
        self._directory = tempfile.mkdtemp(prefix="ox-choice-contract-")
        raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
        self.cfg = _AttrDict(raw)
        self.cfg.update({
            "root": self._directory,
            "db_path": str(Path(self._directory) / "test.db"),
            "security_map": dict(_SECURITY_MAP),
            "execution": dict(raw["execution"], order_confirm_timeout_seconds=2),
        })
        self.db = DB(Path(self._directory) / "test.db")
        self.session = _FakeSession()
        self.broker = ChoiceBroker(self.cfg, self.db)
        self.broker.session = self.session

    def tearDown(self) -> None:
        self.db.close()
        for name, value in self._prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if self._prior_key is None:
            os.environ.pop("OX_AUDIT_KEY", None)
        else:
            os.environ["OX_AUDIT_KEY"] = self._prior_key

    # -- helpers ---------------------------------------------------------

    def script_login(self, payload=None):
        self.session.script(
            "POST", "/QuickAuth",
            payload=payload if payload is not None else {"stat": "Ok", "susertoken": "live-test-susertoken-not-dummy", "actid": "TEST-UID"},
        )

    def login(self) -> None:
        self.script_login()
        self.assertTrue(self.broker.login())

    def last_call(self):
        return self.session.calls[-1]

    # -- auth handshake ---------------------------------------------------

    def test_login_handshake_exact_payload(self):
        self.script_login()
        self.assertTrue(self.broker.login())
        method, path, body = self.last_call()
        self.assertEqual((method, path), ("POST", "/QuickAuth"))
        self.assertIsNone(body["jKey"])
        expected_pwd = hashlib.sha256(b"secret-pass").hexdigest()
        expected_appkey = hashlib.sha256(b"TEST-UID|test-api-key").hexdigest()
        self.assertEqual(body["jData"], {
            "source": "API", "apkversion": "1.0.0",
            "uid": "TEST-UID", "pwd": expected_pwd, "factor2": "123456",
            "vc": "TEST-VC", "appkey": expected_appkey, "imei": "test-imei",
        })
        self.assertTrue(self.broker.authenticated())
        self.assertEqual(self.broker.token, "live-test-susertoken-not-dummy")

    def test_login_fails_closed_on_missing_credentials(self):
        for name in _ENV_VARS:
            os.environ.pop(name, None)
        broker = ChoiceBroker(self.cfg, self.db)
        broker.session = self.session
        with self.assertRaisesRegex(
            AuthenticationError,
            "CHOICE_USER_ID.*CHOICE_PASSWORD.*CHOICE_TOTP.*CHOICE_VENDOR_CODE.*CHOICE_API_KEY",
        ):
            broker.login()
        self.assertFalse(broker.authenticated())
        self.assertEqual(self.session.calls, [])  # no HTTP call without credentials

    def test_login_rejected_by_gateway(self):
        self.script_login({"stat": "Not_Ok", "emsg": "Invalid credentials"})
        with self.assertRaisesRegex(AuthenticationError, "Invalid credentials"):
            self.broker.login()
        self.assertFalse(self.broker.authenticated())

    def test_login_without_token_fails_closed(self):
        self.script_login({"stat": "Ok"})
        with self.assertRaisesRegex(AuthenticationError, "no session token"):
            self.broker.login()
        self.assertFalse(self.broker.authenticated())

    def test_login_dummy_token_fails_closed(self):
        self.script_login({"stat": "Ok", "susertoken": "DUMMY-something"})
        with self.assertRaisesRegex(AuthenticationError, "no session token"):
            self.broker.login()

    # -- market data ------------------------------------------------------

    def test_ltps_exact_payload_and_parsing(self):
        self.login()
        self.session.script("POST", "/GetQuotes", payload=_QUOTE_TCS)
        self.assertEqual(self.broker.ltps(["TCS"]), {"TCS": 1100.25})
        method, path, body = self.last_call()
        self.assertEqual((method, path), ("POST", "/GetQuotes"))
        self.assertEqual(body["jData"], {"uid": "TEST-UID", "exch": "NSE", "token": "2885"})
        self.assertEqual(body["jKey"], "live-test-susertoken-not-dummy")

    def test_ltps_malformed_quote_fails_closed(self):
        self.login()
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok"})
        with self.assertRaisesRegex(MarketDataError, "TCS"):
            self.broker.ltps(["TCS"])
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok", "lp": "0"})
        with self.assertRaisesRegex(MarketDataError, "invalid price"):
            self.broker.ltps(["TCS"])

    def test_ltps_without_session_fails_closed(self):
        with self.assertRaisesRegex(AuthenticationError, "not authenticated"):
            self.broker.ltps(["TCS"])
        self.assertEqual(self.session.calls, [])

    def test_history_payload_and_parsing(self):
        self.login()
        self.session.script("POST", "/TPSeries", payload=[
            {"time": "1642438794", "into": "100", "inth": "101", "intl": "99", "intc": "100.5", "intv": "1234"},
            {"time": "1642438854", "into": "100.5", "inth": "102", "intl": "100", "intc": "101.25", "intv": "900"},
        ])
        rows = self.broker.hist("TCS", 1, 5)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], (1642438794, 100.0, 101.0, 99.0, 100.5, 1234))
        self.assertEqual(rows[1][-1], 900)
        method, path, body = self.last_call()
        self.assertEqual((method, path), ("POST", "/TPSeries"))
        self.assertEqual(body["jData"]["exch"], "NSE")
        self.assertEqual(body["jData"]["token"], "2885")
        self.assertEqual(body["jData"]["intrv"], "1")
        self.assertTrue(str(body["jData"]["st"]).isdigit())  # epoch seconds
        self.assertTrue(str(body["jData"]["et"]).isdigit())

    def test_history_malformed_bar_fails_closed(self):
        self.login()
        self.session.script("POST", "/TPSeries", payload=[{"time": "1642438794", "into": "nope"}])
        with self.assertRaisesRegex(MarketDataError, "Malformed Choice history bar"):
            self.broker.hist("TCS", 1, 5)
        self.session.script("POST", "/TPSeries", payload={"stat": "Ok"})
        with self.assertRaisesRegex(MarketDataError, "incomplete or invalid"):
            self.broker.hist("TCS", 1, 5)

    def test_history_rejects_unsupported_interval(self):
        self.login()
        with self.assertRaisesRegex(MarketDataError, "Choice supports"):
            self.broker.hist("TCS", 2, 5)
        self.assertNotIn("/TPSeries", [path for _method, path, _body in self.session.calls])

    # -- orders -----------------------------------------------------------

    def test_place_super_order_exact_payload(self):
        self.login()
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok", "lp": "1100.0"})
        self.session.script("POST", "/PlaceOrder", payload={"stat": "Ok", "norenordno": "N0001"})
        receipt = self.broker.place_super_order("TCS", "BUY", 75, target=1150.0, stop=1050.0, tag="mom1")
        self.assertEqual((receipt.order_id, receipt.status, receipt.filled_qty, receipt.average_price),
                         ("N0001", "PENDING", 0, 0.0))
        method, path, body = self.last_call()
        self.assertEqual((method, path), ("POST", "/PlaceOrder"))
        self.assertEqual(body["jData"], {
            "ordersource": "API", "uid": "TEST-UID", "actid": "TEST-UID",
            "trantype": "BUY", "prd": "I", "exch": "NSE", "tsym": "TCS-EQ",
            "qty": "75", "dscqty": "0", "prctyp": "MKT", "prc": "0", "trgprc": "0",
            "ret": "DAY", "remarks": "mom1", "amo": "NO",
        })
        self.assertEqual(body["jKey"], "live-test-susertoken-not-dummy")
        rows = self.db.q("SELECT sym,side,qty,type FROM orders WHERE oid=?", ("N0001",))
        self.assertEqual(rows, [("TCS", "BUY", 75, "SUPER")])

    def test_super_order_bracket_violation_never_reaches_the_venue(self):
        self.login()
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok", "lp": "1100.0"})
        with self.assertRaisesRegex(OrderError, "bracket the current market price"):
            self.broker.place_super_order("TCS", "BUY", 75, target=1090.0, stop=1050.0, tag="x")
        paths = [path for method, path, _ in self.session.calls]
        self.assertEqual(paths, ["/QuickAuth", "/GetQuotes"])  # no PlaceOrder

    def test_wait_super_order_confirmation(self):
        self.login()
        self.session.script("POST", "/OrderBook", payload=[
            {"norenordno": "N0001", "status": "COMPLETE", "fillshares": "75", "avgprc": "1100.05", "tsym": "TCS-EQ"},
        ])
        receipt = self.broker.wait_super_order("N0001", 10)
        self.assertEqual((receipt.order_id, receipt.status, receipt.filled_qty, receipt.average_price),
                         ("N0001", "TRADED", 75, 1100.05))
        method, path, body = self.last_call()
        self.assertEqual(body["jData"], {"ordersource": "API", "uid": "TEST-UID"})

    def test_wait_super_order_partial_then_complete(self):
        self.login()
        self.session.script("POST", "/OrderBook", payload=[
            {"norenordno": "N0001", "status": "OPEN", "fillshares": "30", "avgprc": "1100.02"},
        ])
        self.session.script("POST", "/OrderBook", payload=[
            {"norenordno": "N0001", "status": "COMPLETE", "fillshares": "75", "avgprc": "1100.05"},
        ])
        with mock.patch("ox.brokers.time.sleep"):
            first = self.broker.wait_super_order("N0001", 2)
        self.assertEqual((first.status, first.filled_qty), ("PART_TRADED", 30))
        with mock.patch("ox.brokers.time.sleep"):
            second = self.broker.wait_super_order("N0001", 2)
        self.assertEqual((second.status, second.filled_qty), ("TRADED", 75))

    def test_wait_super_order_rejected_surfaces_rejection(self):
        self.login()
        self.session.script("POST", "/OrderBook", payload=[
            {"norenordno": "N0001", "status": "REJECTED", "fillshares": "0", "rejreason": "bad"},
        ])
        receipt = self.broker.wait_super_order("N0001", 10)
        self.assertEqual(receipt.status, "REJECTED")
        self.assertEqual(receipt.filled_qty, 0)

    def test_wait_super_order_absent_fails_closed(self):
        self.login()
        self.session.script("POST", "/OrderBook", payload=[])
        with mock.patch("ox.brokers.time.sleep"):
            with self.assertRaisesRegex(OrderError, "absent from the order book"):
                self.broker.wait_super_order("N0001", 0.1)

    def test_wait_super_order_timeout_is_uncertain(self):
        self.login()
        self.session.script("POST", "/OrderBook", payload=[
            {"norenordno": "N0001", "status": "OPEN", "fillshares": "0"},
        ])
        # Pin the clock: the first poll happens at t=1.0 (deadline 1.1), the
        # second while-check runs at t=10.0 and must exit without a second poll.
        with mock.patch("ox.brokers.time.sleep"), mock.patch(
            "ox.brokers.time.monotonic", side_effect=[1.0, 1.0, 10.0, 10.0]
        ):
            with self.assertRaisesRegex(OrderError, "broker state is uncertain"):
                self.broker.wait_super_order("N0001", 0.1)

    def test_cancel_exact_payload(self):
        self.login()
        self.session.script("POST", "/CancelOrder", payload={"stat": "Ok"})
        self.broker.cancel_super_order("N0001")
        method, path, body = self.last_call()
        self.assertEqual((method, path), ("POST", "/CancelOrder"))
        self.assertEqual(body["jData"], {"ordersource": "API", "uid": "TEST-UID", "norenordno": "N0001"})

    def test_modify_target_fails_closed_without_http(self):
        self.login()
        with self.assertRaisesRegex(OrderError, "does not support Dhan-style"):
            self.broker.modify_super_target("N0001", 1200.0)
        self.assertEqual([p for m, p, _ in self.session.calls], ["/QuickAuth"])

    # -- exits / square-off ------------------------------------------------

    def test_exit_position_exact_payload(self):
        self.login()
        self.session.script("POST", "/PlaceOrder", payload={"stat": "Ok", "norenordno": "N0002"})
        receipt = self.broker.exit_position("TCS", "SELL", 75, "eod")
        self.assertEqual((receipt.order_id, receipt.status), ("N0002", "PENDING"))
        method, path, body = self.last_call()
        self.assertEqual(body["jData"]["trantype"], "SELL")
        self.assertEqual(body["jData"]["prctyp"], "MKT")
        self.assertEqual(body["jData"]["qty"], "75")
        rows = self.db.q("SELECT sym,side,qty,type FROM orders WHERE oid=?", ("N0002",))
        self.assertEqual(rows, [("TCS", "SELL", 75, "MARKET")])

    def test_protective_stop_payload(self):
        self.login()
        self.session.script("POST", "/PlaceOrder", payload={"stat": "Ok", "norenordno": "N0003"})
        receipt = self.broker.place_protective_stop("TCS", 45, 1075.5, "residual")
        self.assertEqual(receipt.order_id, "N0003")
        method, path, body = self.last_call()
        self.assertEqual(body["jData"]["prctyp"], "SL-MKT")
        self.assertEqual(body["jData"]["trgprc"], "1075.5")
        self.assertEqual(body["jData"]["qty"], "45")
        rows = self.db.q("SELECT sym,side,qty,type FROM orders WHERE oid=?", ("N0003",))
        self.assertEqual(rows, [("TCS", "SELL", 45, "SL")])

    # -- positions ---------------------------------------------------------

    def test_positions_parsing(self):
        self.login()
        self.session.script("POST", "/PositionBook", payload=[
            {"netqty": "75", "avgprc": "1100.05", "tsym": "TCS-EQ", "token": "2885", "exch": "NSE", "prd": "I"},
            {"netqty": "0", "avgprc": "0", "tsym": "INFY-EQ", "token": "26000", "exch": "NSE", "prd": "I"},
        ])
        rows = self.broker.positions()
        self.assertEqual(rows, [{
            "sym": "TCS", "tradingSymbol": "TCS-EQ", "netQty": 75,
            "averagePrice": 1100.05, "productType": "I",
        }])
        method, path, body = self.last_call()
        self.assertEqual(body["jData"], {"uid": "TEST-UID", "actid": "TEST-UID"})

    def test_positions_unknown_token_falls_back_to_tsym(self):
        self.login()
        self.session.script("POST", "/PositionBook", payload=[
            {"netqty": "-10", "avgprc": "100", "tsym": "SOMENEW-EQ", "token": "99999", "exch": "NSE", "prd": "I"},
        ])
        rows = self.broker.positions()
        self.assertEqual(rows[0]["sym"], "SOMENEW-EQ")

    # -- error wrapping / fail-closed ---------------------------------------

    def test_network_faults_wrap_as_broker_error(self):
        self.login()
        self.session.script("POST", "/GetQuotes", exc=requests.ConnectionError("boom"))
        with self.assertRaisesRegex(BrokerError, "network request failed"):
            self.broker.ltps(["TCS"])

    def test_http_and_json_failures_wrap_as_broker_error(self):
        self.login()
        self.session.script("POST", "/GetQuotes", status=500, payload={"errorMessage": "up"})
        with self.assertRaisesRegex(BrokerError, "500"):
            self.broker.ltps(["TCS"])
        self.session.script("POST", "/GetQuotes", text="<html>not json</html>")
        with self.assertRaisesRegex(BrokerError, "non-JSON"):
            self.broker.ltps(["TCS"])

    def test_gateway_rejection_on_order_is_order_error(self):
        self.login()
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok", "lp": "1100.0"})
        self.session.script("POST", "/PlaceOrder", payload={"stat": "Not_Ok", "emsg": "Insufficient margin"})
        with self.assertRaisesRegex(OrderError, "Insufficient margin"):
            self.broker.place_super_order("TCS", "BUY", 75, target=1150.0, stop=1050.0, tag="x")

    def test_no_fill_is_ever_fabricated(self):
        # Fills exist only in venue fields: an order placed but never
        # confirmed stays PENDING with zero fill - the OMS must not see a
        # phantom position.
        self.login()
        self.session.script("POST", "/GetQuotes", payload={"stat": "Ok", "lp": "1100.0"})
        self.session.script("POST", "/PlaceOrder", payload={"stat": "Ok", "norenordno": "N0009"})
        receipt = self.broker.place_super_order("TCS", "BUY", 75, target=1150.0, stop=1050.0, tag="x")
        self.assertEqual((receipt.filled_qty, receipt.average_price), (0, 0.0))
        self.assertEqual(receipt.status, "PENDING")

    # -- isolation / config ------------------------------------------------

    def test_platform_selection_never_falls_through_to_paper(self):
        os.environ["CHOICE_USER_ID"] = "TEST-UID"
        os.environ["CHOICE_API_KEY"] = "key"
        from_paper = make_broker(dict(self.cfg, platform="paper"), self.db)
        self.assertIsInstance(from_paper, PaperBroker)
        from_choice = make_broker(dict(self.cfg, platform="choice"), self.db)
        self.assertIsInstance(from_choice, ChoiceBroker)
        # Even with mode: paper, platform selects the adapter - and the
        # adapter itself refuses to act without a real session.
        self.assertFalse(from_choice.authenticated())

    def test_security_map_fail_closed(self):
        self.login()
        self.broker.security_map = {"TCS": "1333"}  # Dhan-style numeric id
        with self.assertRaisesRegex(OrderError, "Malformed Choice security_map entry"):
            self.broker.ltps(["TCS"])
        self.broker.security_map = {}
        with self.assertRaisesRegex(OrderError, "No Choice security is configured"):
            self.broker.ltps(["INFY"])

    def test_whitelisted_ips_is_none(self):
        # Shoonya authenticates by token; compliance must skip the Dhan-style
        # static-IP confirmation (covered in the boot drill).
        self.assertIsNone(self.broker.whitelisted_ips())


if __name__ == "__main__":
    unittest.main()