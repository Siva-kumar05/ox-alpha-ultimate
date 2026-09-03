from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

from ox.agent import Agent
from ox.core import Cfg, ConfigError, allowed_dhan_route
from ox.risk import Metrics, RiskManager


class HardeningTests(unittest.TestCase):
    def test_dhan_route_allowlist_blocks_unreviewed_actions(self):
        self.assertTrue(allowed_dhan_route("POST", "/super/orders"))
        self.assertTrue(allowed_dhan_route("DELETE", "/super/orders/abc-123/ENTRY_LEG"))
        self.assertFalse(allowed_dhan_route("POST", "/fundtransfer"))
        self.assertFalse(allowed_dhan_route("DELETE", "/positions"))

    def test_expected_shortfall_is_a_non_negative_tail_loss(self):
        returns = [-0.12, -0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        self.assertGreater(Metrics.expected_shortfall(returns, alpha=0.90), 0.0)
        self.assertGreaterEqual(Metrics.expected_shortfall(returns, alpha=0.90), Metrics.var(returns, alpha=0.90))

    def test_live_configuration_requires_real_enablement(self):
        source = Path(__file__).resolve().parents[1] / "config.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["mode"] = "live"
        raw["platform"] = "dhan"
        raw["ip_whitelist"] = ["203.0.113.2"]
        raw["regulatory"] = {"retail_algo_registration_id": "REG-123", "broker_algo_approval_id": "BROKER-123"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ConfigError):
                Cfg(path)
            with patch.dict("os.environ", {"OX_LIVE_EXECUTION_APPROVED": "YES_I_UNDERSTAND_LIVE_TRADING"}, clear=True):
                self.assertEqual(Cfg(path)["mode"], "live")

    def test_signal_frame_excludes_the_in_progress_candle(self):
        bucket = int(__import__("time").time() // 60 * 60)

        class _DB:
            def q(self, _query, _args):
                return [(bucket - 60, 100, 101, 99, 100, 10)]

        fake_agent = type("AgentForFrameTest", (), {"cfg": {"timeframe_sec": 60}, "db": _DB()})()
        frame = Agent.completed_frame(fake_agent, "RELIANCE", 10)
        self.assertEqual(frame["ts"].tolist(), [bucket - 60])

    def test_day_pnl_uses_trade_exit_date(self):
        class _DB:
            def __init__(self):
                self.query = None

            def q(self, query, _args):
                self.query = query
                return []

        db = _DB()
        cfg = {"risk": {"risk_per_trade_pct": 1, "max_notional_per_trade": 1000}}
        manager = RiskManager(cfg, db)
        self.assertIn("outtime LIKE", db.query)
        self.assertEqual(manager.day_pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
