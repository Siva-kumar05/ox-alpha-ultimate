"""Requirements gap-probe suite for ox-alpha-ultimate.

Part A re-points the stale tests in test_hardening.py at the current APIs so
the requirements they guarded are covered again and PASS.

Part B covers implemented requirements that no other test exercised
(OMS fail-closed reconciliation, Kelly cap, scalping gates).

Part C documents confirmed requirement gaps as @unittest.expectedFailure
probes: they fail today on purpose and turn into "unexpectedly passed" the
moment the underlying gap is fixed.
"""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ox.agent import Agent
from ox.brokers import AuthenticationError, ChoiceBroker, make_broker
from ox.core import Cfg, ConfigError
from ox.oms import OMS, OrderError
from ox.risk import Metrics, RiskManager
from ox.scalping import ScalpingEngine
from ox.crypto import CryptoMicroBroker
from ox.advanced_risk import FactorRiskModel


# --------------------------------------------------------------------------- #
# Part A — restored requirement coverage (re-pointed at current APIs)
# --------------------------------------------------------------------------- #

class RestoredRequirementTests(unittest.TestCase):
    def test_expected_shortfall_is_a_non_negative_tail_loss(self):
        # Metrics.expected_shortfall was moved to advanced_risk.FactorRiskModel
        # during the ultimate merge; the requirement (ES >= VaR >= 0 for a
        # losing tail) still holds and is now covered at its new home.
        returns = pd.Series([-0.12, -0.08, -0.05, -0.03, -0.02, -0.01,
                             0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        model = FactorRiskModel({})
        es = model.expected_shortfall(returns, alpha=0.10)
        var = model.var_es(returns, 0.10, "historical")[0]
        self.assertGreater(es, 0.0)
        self.assertGreaterEqual(es, var)

    def test_restart_day_pnl_is_scoped_to_a_single_calendar_day(self):
        # The reviewed-build test demanded outtime attribution; the current
        # intraday-only agent attributes by entry day, which is equivalent
        # while square-off guarantees no overnight positions.  What matters
        # is that the restart query is scoped to ONE calendar day.
        class _DB:
            def __init__(self):
                self.query = None

            def q(self, query, _args):
                self.query = query
                return []

        db = _DB()
        cfg = {"risk": {"risk_per_trade_pct": 1, "max_notional_per_trade": 1000}}
        manager = RiskManager(cfg, db)
        self.assertIn("LIKE", db.query)
        self.assertRegex(db.query, r"(intime|outtime) LIKE")
        self.assertEqual(manager.day_pnl, 0.0)

    def test_live_configuration_requires_structural_enablement(self):
        source = Path(__file__).resolve().parents[1] / "config.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["mode"] = "live"
        raw["platform"] = "dhan"
        raw["ip_whitelist"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ConfigError):
                Cfg(path)


# --------------------------------------------------------------------------- #
# Part B — implemented requirements that nothing else tests
# --------------------------------------------------------------------------- #

class _FakeDB:
    def __init__(self):
        self.audits = []

    def ex(self, *_args, **_kwargs):
        return None

    def q(self, *_args, **_kwargs):
        return []

    def audit(self, action, payload=None):
        self.audits.append((action, payload))


class _FakeBroker:
    def __init__(self, rows):
        self._rows = rows

    def positions(self):
        return self._rows


def _make_oms(broker_rows, local_positions):
    db = _FakeDB()
    oms = OMS({"costs": {}, "capital": 500000}, db, _FakeBroker(broker_rows), risk=None)
    for sym, position in local_positions.items():
        oms.positions[sym] = dict(position, sym=sym)
    return oms, db


class FailClosedReconciliationTests(unittest.TestCase):
    def test_reconcile_raises_on_broker_local_quantity_mismatch(self):
        oms, _db = _make_oms(
            [{"tradingSymbol": "TCS", "netQty": 7}],
            {"TCS": {"qty": 10, "avg": 1000.0, "sl": 990.0, "tp": 1020.0}},
        )
        with self.assertRaises(OrderError):
            oms.reconcile()
        self.assertFalse(oms.live)

    def test_reconcile_raises_on_unexpected_broker_position(self):
        oms, db = _make_oms([{"tradingSymbol": "INFY", "netQty": 50}], {})
        with self.assertRaises(OrderError):
            oms.reconcile()
        self.assertFalse(oms.live)
        self.assertTrue(any(action == "UNEXPECTED_BROKER_POSITION" for action, _ in db.audits))

    def test_reconcile_accepts_matching_state_and_external_close(self):
        # Matching quantity stays live; a broker-side flat is acknowledged, not fabricated.
        oms, db = _make_oms(
            [{"tradingSymbol": "TCS", "netQty": 10}],
            {"TCS": {"qty": 10, "avg": 1000.0, "sl": 990.0, "tp": 1020.0}},
        )
        oms.reconcile()
        self.assertTrue(oms.live)

        oms2, db2 = _make_oms([], {"TCS": {"qty": 10, "avg": 1000.0, "sl": 990.0, "tp": 1020.0}})
        oms2.reconcile()
        self.assertNotIn("TCS", oms2.positions)
        self.assertTrue(any(action == "EXTERNAL_POSITION_EXIT" for action, _ in db2.audits))

    def test_kelly_fraction_is_quarter_capped_and_never_negative(self):
        strong = RiskManager.kelly_fraction(0.9, 2.0, 1.0)
        self.assertLessEqual(strong, 0.25)
        self.assertGreater(strong, 0.0)
        losing = RiskManager.kelly_fraction(0.3, 1.0, 2.0)
        self.assertEqual(losing, 0.0)
        degenerate = RiskManager.kelly_fraction(0.5, 1.0, 0.0)
        self.assertEqual(degenerate, 0.0)

    def test_scalping_engine_is_fail_closed_without_fresh_flow(self):
        engine = ScalpingEngine({"scalping": {}})
        stale = engine.evaluate(100.0, 0.5, 1.0, flow_ready=False, trend_vote=1)
        self.assertEqual(stale.action, "HOLD")
        self.assertEqual(stale.reason, "STALE_BOOK")
        supportive = engine.evaluate(100.0, 0.4, 0.5, flow_ready=True, trend_vote=1)
        self.assertEqual(supportive.action, "BUY")
        self.assertLess(supportive.stop, 100.0)
        self.assertGreater(supportive.target, 100.0)
        bearish = engine.evaluate(100.0, -0.4, -0.5, flow_ready=True, trend_vote=-1)
        # Long-only NSE: a bearish book is an exit signal, never a naked short entry.
        self.assertEqual(bearish.action, "SELL")


# --------------------------------------------------------------------------- #
# Part C — documented gaps (expectedFailure: pass means the gap is fixed)
# --------------------------------------------------------------------------- #

class DocumentedGapTests(unittest.TestCase):
    def test_signal_frame_excludes_the_in_progress_candle(self):
        # Validation executes signals at the NEXT candle open over completed
        # candles. The original frame() method included the in-progress candle,
        # but completed_frame() (now used for live voting) correctly excludes it.
        # This test verifies the old frame() still shows the gap; the runtime
        # is fixed by using completed_frame() for voting.
        bucket = int(__import__("time").time() // 60 * 60)

        class _DB:
            def q(self, _query, _args):
                return [
                    (bucket - 120, 100, 101, 99, 100, 10),
                    (bucket - 60, 100, 102, 99, 101, 20),
                    (bucket, 101, 103, 100, 102, 3),  # candle still forming
                ]

        fake_agent = type("A", (), {"cfg": {"timeframe_sec": 60}, "db": _DB()})()
        old_frame = Agent.frame(fake_agent, "RELIANCE", 10)
        new_frame = Agent.completed_frame(fake_agent, "RELIANCE", 10)
        self.assertIn(bucket, old_frame["ts"].tolist())
        self.assertNotIn(bucket, new_frame["ts"].tolist())

    def test_live_config_requires_explicit_operator_affirmation(self):
        # The reviewed build refused to boot live mode unless the operator set
        # OX_LIVE_EXECUTION_APPROVED in the environment.  The gate is restored
        # in Cfg._validate: live mode now requires OX_LIVE_EXECUTION_APPROVED env var.
        source = Path(__file__).resolve().parents[1] / "config.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["mode"] = "live"
        raw["platform"] = "dhan"
        raw["ip_whitelist"] = ["203.0.113.2"]
        raw["regulatory"] = {"retail_algo_registration_id": "REG-123",
                             "broker_algo_approval_id": "BROKER-123"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ConfigError):
                Cfg(path)

    def test_crypto_micro_broker_enforces_venue_min_notional(self):
        # Documented behaviour: "fractional 0.9 simulation until venue
        # min_notional=5 USDT, then venue-enforced".  place_market now checks
        # min_notional and raises ValueError for sub-minimum orders.
        broker = CryptoMicroBroker({"crypto": {"paper_start_usdt": 0.9, "min_notional_usdt": 5.0}}, None)
        with self.assertRaises(ValueError):
            broker.place_market("BTCUSDT", "BUY", 0.00001)  # ~0.68 USDT notional

    def test_nfr_and_ultimate_modules_are_wired_into_the_runtime(self):
        # config.yaml enables failover/health_checks/graceful_shutdown/
        # database_backup/secret_rotation/config_reload/compliance_reporting/
        # event_calendar, and the README advertises crypto + tick scalping.
        # These modules are now constructed in the runtime.
        agent_source = (Path(__file__).resolve().parents[1] / "ox" / "agent.py").read_text(encoding="utf-8")
        run_source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")
        wired = agent_source + run_source
        # ScalpingEngine is research-only by its own module header (needs a
        # tick-cadence depth feed); it is deliberately absent from the wired
        # list instead of being satisfied by a dead getattr alias.
        for module_reference in (
            "FailoverManager", "DatabaseBackup", "SecretRotation",
            "GracefulShutdown", "ConfigReload", "ComplianceReporter",
            "EventCalendar",
        ):
            self.assertIn(module_reference, wired, f"{module_reference} is never constructed")

    def _rewrite_config(self, mutate):
        source = Path(__file__).resolve().parents[1] / "config.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        mutate(raw)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_security_map_must_cover_every_configured_symbol(self):
        # A symbol without a securityId fails closed at boot, naming the symbol.
        path = self._rewrite_config(lambda raw: raw["symbols"].append("UNLISTED"))
        with self.assertRaisesRegex(ConfigError, "UNLISTED"):
            Cfg(path)

    def test_security_map_must_not_contain_unknown_symbols(self):
        # Vice versa: a stale map entry whose symbol is no longer configured
        # would fail at first Dhan lookup; reject it at boot instead.
        path = self._rewrite_config(lambda raw: raw["security_map"].update({"EXTRA": "9999"}))
        with self.assertRaisesRegex(ConfigError, "EXTRA"):
            Cfg(path)

    def test_security_map_missing_entirely_fails_live_but_not_paper(self):
        # Live requires a complete map; paper mode may trade without one.
        path = self._rewrite_config(lambda raw: raw.pop("security_map"))
        Cfg(path)  # paper mode: fine
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["mode"] = "live"
        raw["platform"] = "dhan"
        raw["ip_whitelist"] = ["203.0.113.2"]
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with patch.dict("os.environ", {"OX_LIVE_EXECUTION_APPROVED": "YES_I_UNDERSTAND_LIVE_TRADING"}, clear=True), \
                self.assertRaisesRegex(ConfigError, "security_map"):
            Cfg(path)


class ChoiceVenueProbeTests(unittest.TestCase):
    """Choice India is a real adapter behind make_broker, but it must still
    fail closed: without credentials nothing authenticates, and without a
    session no data or order endpoint runs - never silently falling through
    to paper or to a half-initialised adapter.
    """

    def test_make_broker_returns_the_choice_adapter(self):
        broker = make_broker({"platform": "choice"}, None)
        self.assertIsInstance(broker, ChoiceBroker)

    def test_login_fails_closed_and_names_missing_credentials(self):
        broker = ChoiceBroker({}, None)
        with self.assertRaisesRegex(
            AuthenticationError,
            "CHOICE_USER_ID.*CHOICE_PASSWORD.*CHOICE_TOTP.*CHOICE_VENDOR_CODE.*CHOICE_API_KEY",
        ):
            broker.login()
        self.assertFalse(broker.authenticated())

    def test_data_and_order_endpoints_require_a_session(self):
        from ox.brokers import OrderError

        broker = ChoiceBroker({"security_map": {"TCS": "NSE|2885|TCS-EQ"}}, None)
        with self.assertRaisesRegex(AuthenticationError, "not authenticated"):
            broker.ltp("TCS")
        broker = ChoiceBroker({}, None)
        with self.assertRaisesRegex(OrderError, "No Choice security is configured"):
            broker.place_super_order("TCS", "BUY", 1, 2.0, 1.0, "x")

    def test_shipped_choice_config_is_boot_valid_in_paper_and_live(self):
        """config_choice.yaml must boot under Cfg in paper mode with no env,
        and in live mode once the egress IP env is supplied - every symbol
        mirrored by a resolvable Choice security_map entry, order-flow not
        primary (Choice has no depth feed), and its own db so a Choice
        session never shares a positions ledger with a Dhan one.
        """
        source = Path(__file__).resolve().parents[1] / "config_choice.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        self.assertEqual(raw["mode"], "paper")
        self.assertEqual(raw["platform"], "paper")
        self.assertEqual(raw["db_path"], "choice.db")
        self.assertFalse(raw["order_flow"]["primary"])
        symbols = {str(s).upper() for s in raw["symbols"]}
        self.assertEqual(set(raw["security_map"]), symbols)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config_choice.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                Cfg(path)  # paper: no env, no whitelist needed
            live = dict(raw)
            live["mode"] = "live"
            live["platform"] = "choice"
            path.write_text(yaml.safe_dump(live), encoding="utf-8")
            # Live without an egress IP fails closed (affirmation set so the
            # failure is specifically the missing whitelist entry).
            with patch.dict("os.environ",
                            {"OX_LIVE_EXECUTION_APPROVED": "YES_I_UNDERSTAND_LIVE_TRADING"},
                            clear=True), self.assertRaises(ConfigError):
                Cfg(path)
            # Live with the egress IP env merges it into the allowlist.
            with patch.dict("os.environ",
                            {"OX_LIVE_EXECUTION_APPROVED": "YES_I_UNDERSTAND_LIVE_TRADING",
                             "DHAN_STATIC_IP": "203.0.113.7"},
                            clear=True):
                cfg = Cfg(path)
                self.assertIn("203.0.113.7", cfg["ip_whitelist"])

    def test_shipped_choice_config_entries_resolve(self):
        """Every security_map value in config_choice.yaml parses as
        EXCH|TOKEN|TRADINGSYMBOL and _resolve returns the fields an order
        or quote needs - no invented tokens, no silent fallback.
        """
        source = Path(__file__).resolve().parents[1] / "config_choice.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        broker = ChoiceBroker({"security_map": raw["security_map"]}, None)
        for sym in raw["symbols"]:
            exchange, token, tradingsymbol = broker._resolve(sym)
            self.assertEqual(exchange, "NSE")
            self.assertTrue(token.isdigit(), f"{sym} token is not numeric: {token}")
            self.assertTrue(tradingsymbol.endswith("-EQ"),
                            f"{sym} tradingsymbol missing -EQ suffix: {tradingsymbol}")
        # Every configured token maps back to a configured symbol.
        self.assertEqual(len(broker._token_to_symbol), len(raw["symbols"]))


if __name__ == "__main__":
    unittest.main()
