"""Regression tests locking the audit fixes from the last review pass.

Part 1 locks the live-sizing contract: a leverage-engine request beyond the
hard risk caps must CLAMP the position down to the admissible size instead of
pushing quantity past a cap the risk gate then rejects.  Before the fix every
entry with a leverage request above the 3x baseline was blocked outright
(reproduced: base qty 181 within the 200k notional cap, 5x request scaled it
to 301, and risk.approve() rejected the trade).

Part 2 locks every formerly-crashing path (undefined names, duplicated dict
key, silent failure) so each executes on minimal offline inputs.  No network
and no real broker is involved anywhere in this module.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ox.advanced_risk import FactorRiskModel
from ox.agent import Agent
from ox.event_calendar import CalendarEvent, EventCalendar, EventImpact, EventType
from ox.execution_algos import (
    ArrivalPriceAlgorithm,
    ExecutionPlan,
    ExecutionSlice,
    IcebergAlgorithm,
    POVAlgorithm,
    SmartRouter,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from ox.rebalancing import PortfolioHedger

_AUDIT_KEY = "smoke-audit-key-is-at-least-thirty-two-characters"


class LeverageCapRegressionTests(unittest.TestCase):
    """An over-cap leverage request shrinks the entry; it never blocks it."""

    def test_leverage_request_is_clamped_to_caps_not_rejected(self):
        # Offline harness mirrors run.run_smoketest(): a temp config points the
        # agent at the paper broker and a seeded random-walk candle history.
        import run  # local import keeps module top level free of run.py effects

        prior_key = os.environ.get("OX_AUDIT_KEY")
        os.environ["OX_AUDIT_KEY"] = _AUDIT_KEY
        directory = tempfile.mkdtemp(prefix="ox-regression-")
        agent = None
        try:
            agent = Agent(str(run._smoke_config(Path(directory))))
            # The contract under test is the leverage clamp in sizing, not the
            # paper depth gate: the simulated depth ages out during
            # nightly_training below (boot -> _act spans longer than
            # max_staleness_seconds), which intermittently blocks the entry as
            # STALE_DEPTH and makes the clamp unreachable.  Disable the flow
            # gate so sizing is exercised deterministically.
            agent.cfg["order_flow"]["primary"] = False
            agent.cfg["order_flow"]["enabled"] = False
            agent.broker.set_px("TCS", 1100.0)
            rng = np.random.default_rng(7)
            base_time = int(time.time()) - 400 * 60
            for symbol in agent.cfg["symbols"]:
                price = 1000.0
                for index in range(400):
                    price += 0.3 + rng.normal(0, 1.5)
                    agent.db.ex(
                        "INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",
                        (symbol, base_time + index * 60, price - 1, price + 2,
                         price - 2, price, 5000),
                    )
            agent.nightly_training()
            agent.load_strategies()
            self.assertTrue(agent.strategies, "no validated strategy was promoted")

            supporter = [(agent.strategies[0][0], agent.strategies[0][2], 1.0)]
            frame = agent.frame("TCS", 400)
            agent._act("TCS", 1100.0, frame, votes=1, supporters=supporter)

            # The core regression: the entry OPENS.  Pre-fix the >3x leverage
            # request produced a quantity past the notional cap and the risk
            # gate rejected the trade, so no position ever existed.
            position = agent.oms.positions.get("TCS")
            self.assertIsNotNone(position, "entry must open; a >3x leverage request used to block it outright")
            self.assertGreater(int(position["qty"]), 0)

            # The clamped quantity respects the per-trade notional cap even
            # after the leverage request is applied.
            notional_cap = float(agent.cfg["risk"]["max_notional_per_trade"])
            self.assertLessEqual(int(position["qty"]) * float(position["avg"]), notional_cap)

            # Self-check that the scenario is non-vacuous: the engine must have
            # asked for more than the 3x baseline and the clamp must have run.
            rows = agent.db.q(
                "SELECT detail FROM decisions WHERE sym='TCS' AND action='ENTRY_REQUEST' "
                "ORDER BY ts DESC LIMIT 1"
            )
            self.assertEqual(len(rows), 1)
            detail = json.loads(rows[0][0])
            self.assertGreater(float(detail.get("leverage", 1.0)), 3.0,
                               "test is vacuous: leverage engine must request >3x")
            self.assertTrue(detail.get("leverage_clamped", False),
                            "clamp path must have executed")
            self.assertEqual(int(detail["quantity"]), int(position["qty"]))
        finally:
            if agent is not None:
                agent.close()
            # Windows keeps the log file handle open until handlers close;
            # without this the temp directory cannot be removed.
            for handler in list(logging.getLogger("ox").handlers):
                logging.getLogger("ox").removeHandler(handler)
                handler.close()
            if prior_key is None:
                os.environ.pop("OX_AUDIT_KEY", None)
            else:
                os.environ["OX_AUDIT_KEY"] = prior_key
            shutil.rmtree(directory, ignore_errors=True)


class FormerlyCrashingPathTests(unittest.TestCase):
    """Each audit fix is executable proof on minimal inputs."""

    @staticmethod
    def _plan() -> ExecutionPlan:
        return ExecutionPlan(
            symbol="TCS", total_quantity=10_000, side="BUY", slices=[],
            benchmark="TWAP", start_time=1_000, end_time=3_000,
        )

    def test_execution_algorithms_generate_real_slices(self):
        # Pre-fix these crashed: `Slice` was never defined (the class is
        # ExecutionSlice) and `time` was missing.
        plan = self._plan()
        cases = {
            "twap": (TWAPAlgorithm, 5),
            "vwap": (VWAPAlgorithm, 5),
            "arrival": (ArrivalPriceAlgorithm, 1),
            "pov": (POVAlgorithm, 20),
            "iceberg": (IcebergAlgorithm, 100),
        }
        for name, (cls, expected_slices) in cases.items():
            with self.subTest(algo=name):
                slices = cls({"n_slices": 5}).generate_slices(plan)
                self.assertEqual(len(slices), expected_slices)
                self.assertTrue(all(isinstance(s, ExecutionSlice) for s in slices))

    def test_twap_and_vwap_fully_allocate_the_order(self):
        plan = self._plan()
        for name, cls in (("twap", TWAPAlgorithm), ("vwap", VWAPAlgorithm)):
            with self.subTest(algo=name):
                allocated = sum(s.quantity for s in cls({"n_slices": 7}).generate_slices(plan))
                self.assertEqual(allocated, plan.total_quantity)

    def test_smart_router_selects_by_urgency_and_participation(self):
        # Pre-fix SmartRouter crashed in __init__: it built child algorithms
        # with config=None while their constructors read the raw argument.
        router = SmartRouter({})
        self.assertIsInstance(
            router.select_algorithm({"urgency": "high", "quantity": 100, "adv": 100_000}),
            ArrivalPriceAlgorithm,
        )
        self.assertIsInstance(
            router.select_algorithm({"urgency": "normal", "quantity": 50_000, "adv": 100_000}),
            IcebergAlgorithm,
        )
        self.assertIsInstance(
            router.select_algorithm({"urgency": "normal", "quantity": 100, "adv": 100_000}),
            TWAPAlgorithm,
        )

    def test_factor_portfolio_optimization_fallbacks_run(self):
        # Pre-fix every non-CVXPY objective crashed: the function read a
        # `cov_matrix` variable that never existed (the parameter is
        # `covariance`).
        model = FactorRiskModel({})
        covariance = np.array([[0.04, 0.01], [0.01, 0.02]])
        expected = np.array([0.10, 0.08])
        for objective in ("min_variance", "risk_parity", "max_sharpe"):
            with self.subTest(objective=objective):
                weights = model.optimize_portfolio(expected, covariance, objective=objective)
                self.assertEqual(weights.shape, (2,))
                self.assertTrue(np.all(np.isfinite(weights)), f"{objective} weights not finite")
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_kupiec_var_backtest_executes(self):
        # Pre-fix this raised NameError: scipy `chi2` was used but never
        # imported.  500 normal returns against a 2% VaR breach ~2.3% of the
        # time, so 0 < violations < n and the Kupiec branch is exercised.
        model = FactorRiskModel({})
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0, 0.01, 500))
        var_series = pd.Series(np.full(500, 0.02))
        result = model.backtest_var(returns, var_series, alpha=0.05)
        self.assertGreaterEqual(int(result["violations"]), 1)
        self.assertLess(int(result["violations"]), 500)
        self.assertTrue(np.isfinite(float(result["kupiec_stat"])))
        self.assertTrue(np.isfinite(float(result["kupiec_pvalue"])))

    def test_sector_hedges_build_from_positions(self):
        # Pre-fix this raised NameError: `defaultdict` was used but never
        # imported in rebalancing.py.
        hedger = PortfolioHedger({}, None, None)
        hedges = hedger.create_sector_hedges(
            {"TCS": {"qty": 10}}, {"TCS": 3_000.0}, {"TCS": "IT"}
        )
        self.assertEqual(len(hedges), 1)
        hedge = hedges[0]
        self.assertEqual(hedge.underlying_symbols, ["TCS"])
        self.assertEqual(hedge.quantity, -30)          # short the offsetting hedge
        self.assertAlmostEqual(hedge.notional, 30_000.0)  # matches 10 * 3000 exposure

    def test_event_calendar_roundtrip_and_avoidance(self):
        # Pre-fix this raised NameError: `defaultdict` and `Tuple` were used
        # but never imported in event_calendar.py.
        class _FakeDB:
            def ex(self, *_args, **_kwargs):
                return None

            def q(self, *_args, **_kwargs):
                return []

            def kv_set(self, *_args, **_kwargs):
                return None

        calendar = EventCalendar({"event_calendar": {"enabled": True}}, _FakeDB())
        calendar.add_event(CalendarEvent(
            "e1", "TCS", EventType.EARNINGS, EventImpact.HIGH, date(2099, 1, 4),
        ))
        # 2099-01-01 sits inside the default 5-day pre-event window of a HIGH
        # earnings print on 2099-01-04 -> the rule action is "close".
        avoid, reason, rule = calendar.check_avoidance("TCS", date(2099, 1, 1), 0.5)
        self.assertTrue(avoid)
        self.assertEqual(reason, "close_earnings_high")
        self.assertEqual(rule.action, "close")
        # Far outside any window the same symbol is clear to trade.
        avoid_later, reason_later, _ = calendar.check_avoidance("TCS", date(2099, 3, 1), 0.5)
        self.assertFalse(avoid_later)
        self.assertEqual(reason_later, "ok")

    def test_funding_mock_returns_one_row_per_exchange(self):
        # Pre-fix the 'bybit' key appeared twice, so the second rate silently
        # overwrote the first and the dict had only two exchanges.
        from ox.agents.crypto_funding import CryptoFundingArbAgent

        stub = object.__new__(CryptoFundingArbAgent)
        funding = CryptoFundingArbAgent._get_funding_across_exchanges(stub, "BTCUSDT")
        self.assertEqual(list(funding.keys()), ["binance", "bybit", "okx"])
        self.assertEqual(funding["bybit"]["rate"], 0.00015)


class UniverseScanOrderingTests(unittest.TestCase):
    """The universe auto-scan must run only after broker auth inside boot().

    Previously it ran in Agent.__init__ before any login, so in live mode
    every scan failed unauthenticated and silently kept the static symbols.
    The class-level spy below is installed before construction, so the
    pre-fix code (scan in __init__) appends 'scan' immediately and both
    tests fail.
    """

    _AUDIT_KEY = "scan-audit-key-is-at-least-thirty-two-characters"

    def setUp(self) -> None:
        self._saved = os.environ.get("OX_AUDIT_KEY")
        os.environ["OX_AUDIT_KEY"] = self._AUDIT_KEY
        self._directory = tempfile.mkdtemp(prefix="ox-uniscan-")

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("OX_AUDIT_KEY", None)
        else:
            os.environ["OX_AUDIT_KEY"] = self._saved
        shutil.rmtree(self._directory, ignore_errors=True)

    def _make_cfg(self) -> Path:
        import run  # local import keeps module top level free of run.py effects

        path = run._smoke_config(Path(self._directory))
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["symbols"] = ["TCS"]
        # security_map must mirror symbols in both directions (Cfg validation).
        raw["security_map"] = {k: v for k, v in raw.get("security_map", {}).items() if k in raw["symbols"]}
        raw["training"]["min_symbols"] = 1  # pre-scan universe is a single symbol
        raw["universe"]["auto_scan"] = True
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    def test_scan_runs_only_after_broker_auth_and_applies_result(self):
        events: list[str] = []
        original_scan = Agent._apply_universe_scan

        def spy_scan(agent):
            events.append("scan")
            return original_scan(agent)

        Agent._apply_universe_scan = spy_scan
        agent = None
        try:
            agent = Agent(str(self._make_cfg()))
            # Post-fix the scan lives in boot() after daily_auth; construction
            # must not attempt one (pre-fix it ran unauthenticated here).
            self.assertEqual(events, [], "universe scan must not run at construction")

            # Seed smallcap candles (20-500 range, volume >= 500) for every
            # candidate so the post-auth scan deterministically ranks >= 2 of
            # them instead of fail-opening back to the static symbol.
            rng = np.random.default_rng(11)
            base_time = int(time.time()) - 90 * 60
            for candidate in agent.cfg["universe"]["candidates"]:
                price = 250.0
                for index in range(90):
                    price = max(price + 0.4 + rng.normal(0, 2.0), 30.0)
                    agent.db.ex(
                        "INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",
                        (candidate, base_time + index * 60, price - 1, price + 2,
                         price - 2, price, 5000),
                    )

            real_auth = agent.comp.daily_auth

            def spy_auth(broker):
                events.append("auth")
                return real_auth(broker)

            agent.comp.daily_auth = spy_auth
            self.assertTrue(agent.boot())
            self.assertEqual(events, ["auth", "scan"],
                             "scan must run exactly once, strictly after broker auth")

            # The scan result is applied: symbols replaced by the affordable
            # candidate list (>= 2 names), journaled as a UNIVERSE_SCAN event.
            self.assertGreaterEqual(len(agent.cfg["symbols"]), 2)
            count = agent.db.q("SELECT COUNT(*) FROM events WHERE kind='UNIVERSE_SCAN'")
            self.assertEqual(int(count[0][0]), 1, "UNIVERSE_SCAN event must be journaled")
        finally:
            Agent._apply_universe_scan = original_scan
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass

    def test_scan_not_attempted_when_broker_auth_fails(self):
        events: list[str] = []
        original_scan = Agent._apply_universe_scan

        def spy_scan(agent):
            events.append("scan")
            return original_scan(agent)

        Agent._apply_universe_scan = spy_scan
        agent = None
        try:
            agent = Agent(str(self._make_cfg()))
            self.assertEqual(events, [], "no scan during construction")
            agent.comp.daily_auth = lambda broker: False  # broker auth refused
            self.assertFalse(agent.boot())
            self.assertEqual(events, [], "scan must not run when broker auth failed")
            self.assertEqual(agent.cfg["symbols"], ["TCS"],
                             "static symbols must be preserved when auth fails")
        finally:
            Agent._apply_universe_scan = original_scan
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass


class SizingMultiplierRegressionTests(unittest.TestCase):
    """A zero or sub-unit circuit-breaker size multiplier must block the entry.

    Previously `max(1, int(qty * multiplier))` made the quantity<=0 guard dead
    code, so a 0.0 multiplier (L2/HALT state) silently produced a 1-share
    entry.  The guard must record CIRCUIT_BREAKER_SIZE and open nothing.
    """

    def test_zero_and_subunit_multipliers_block_entry(self):
        import run  # local import keeps module top level free of run.py effects

        prior_key = os.environ.get("OX_AUDIT_KEY")
        os.environ["OX_AUDIT_KEY"] = "size-audit-key-is-at-least-thirty-two-chars"
        directory = tempfile.mkdtemp(prefix="ox-sizemult-")
        agent = None
        try:
            agent = Agent(str(run._smoke_config(Path(directory))))
            # The contract under test is the breaker-multiplier guard in
            # sizing; the paper depth gate is orthogonal (see the leverage
            # regression above for why it would intermittently block).
            agent.cfg["order_flow"]["primary"] = False
            agent.cfg["order_flow"]["enabled"] = False
            agent.broker.set_px("TCS", 1100.0)
            rng = np.random.default_rng(7)
            base_time = int(time.time()) - 400 * 60
            for symbol in agent.cfg["symbols"]:
                price = 1000.0
                for index in range(400):
                    price += 0.3 + rng.normal(0, 1.5)
                    agent.db.ex(
                        "INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",
                        (symbol, base_time + index * 60, price - 1, price + 2,
                         price - 2, price, 5000),
                    )
            agent.nightly_training()
            agent.load_strategies()
            self.assertTrue(agent.strategies, "no validated strategy was promoted")
            supporter = [(agent.strategies[0][0], agent.strategies[0][2], 1.0)]
            frame = agent.frame("TCS", 400)

            def call_act() -> None:
                agent._act("TCS", 1100.0, frame, votes=1, supporters=supporter)

            # 1. L2/HALT multiplier 0.0 -> block; pre-fix this opened 1 share.
            agent._position_size_multiplier = 0.0
            call_act()
            self.assertNotIn("TCS", agent.oms.positions,
                             "a 0.0 multiplier (L2/HALT) must never open a position")

            # 2. Sub-unit scaled size: base qty 1 at this cap * 0.5 floors to
            #    zero -> block, not a 1-share entry.
            agent.risk.rules["max_notional_per_trade"] = 1100.0
            agent._position_size_multiplier = 0.5
            call_act()
            self.assertNotIn("TCS", agent.oms.positions,
                             "a sub-unit scaled size must block")

            # 3. Same 1-share cap with the full multiplier still opens - the
            #    guard is not over-broad.
            agent._position_size_multiplier = 1.0
            call_act()
            position = agent.oms.positions.get("TCS")
            self.assertIsNotNone(position, "full multiplier must still open at cap qty 1")
            self.assertEqual(int(position["qty"]), 1)

            # _record_decision throttles duplicate (sym, action, reason) rows,
            # so both blocks may share one journal row; the behavioural
            # contract above (nothing opened in cases 1-2, qty 1 in case 3) is
            # the assertion.  At least one row proves the guard fired.
            blocked = agent.db.q(
                "SELECT reason FROM decisions WHERE action='BLOCK' AND sym='TCS' AND reason='CIRCUIT_BREAKER_SIZE'")
            self.assertGreaterEqual(len(blocked), 1,
                                    "the CIRCUIT_BREAKER_SIZE guard must be journaled")
        finally:
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass
            if prior_key is None:
                os.environ.pop("OX_AUDIT_KEY", None)
            else:
                os.environ["OX_AUDIT_KEY"] = prior_key
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
