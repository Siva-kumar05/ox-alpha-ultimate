"""Offline tests for scripts/analyze_patterns.py (read-only DB analysis).

Builds a synthetic agent DB (same strategies/backtests DDL the runtime
creates in ox/core.py) and asserts the structured analyze() output and that
the rendered dump covers every candidate - structure asserts only, no
brittle string matching on formatted numbers.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_patterns import analyze, render

_STRATEGIES_DDL = ("sid TEXT PRIMARY KEY, json TEXT, score REAL, status TEXT, "
                   "gen INT, parent TEXT, hash TEXT, created TEXT, "
                   "approved_by TEXT, approved_at TEXT")
_BACKTESTS_DDL = ("bid INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, "
                  "is_oos TEXT, score REAL, stats TEXT, ts TEXT")

_GATES = {"min_trades": 25.0, "promote_score": 0.8, "min_signal_stability": 0.95}


def _stats_json(**overrides) -> str:
    import json

    stats = {"sharpe": 0.1, "sortino": 0.1, "maxdd": -0.1, "trades": 0,
             "pf": 0.0, "ret": 0.0, "icir": 0.2, "frame_ic": 0.05,
             "win_rate": 0.5, "cost_drag": 0.0, "symbols": 3,
             "execution": "NEXT_CANDLE_OPEN_LONG_ONLY",
             "signal_stability": 0.99, "promotion_eligible": True}
    stats.update(overrides)
    return json.dumps(stats)


def _make_db(path: Path, empty: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE strategies({_STRATEGIES_DDL})")
    conn.execute(f"CREATE TABLE backtests({_BACKTESTS_DDL})")
    if not empty:
        conn.executemany(
            "INSERT INTO strategies(sid,json,score,status,gen,parent,hash,created)"
            " VALUES(?,?,?,?,?,?,?,?)",
            [
                ("s_a", "{}", -9.0, "CANDIDATE", 0, None, "h", "2026-09-04T00:00:00+05:30"),
                ("s_b", "{}", -9.0, "CANDIDATE", 0, None, "h", "2026-09-04T00:00:00+05:30"),
                ("s_c", "{}", -9.0, "ELITE", 1, "s_a", "h", "2026-09-04T00:01:00+05:30"),
                ("s_d", "{}", 1.5, "PENDING_APPROVAL", 1, "s_b", "h", "2026-09-04T00:02:00+05:30"),
                ("s_e", "{}", 2.0, "LIVE_APPROVED", 1, "s_c", "h", "2026-09-04T00:03:00+05:30"),
                ("s_f", "{}", -99.0, "CANDIDATE", 2, "s_d", "h", "2026-09-04T00:04:00+05:30"),
            ])
        # s_a has TWO OOS rows: the newest (higher bid) must win.
        conn.executemany(
            "INSERT INTO backtests(sid,is_oos,score,stats,ts) VALUES(?,?,?,?,?)",
            [
                ("s_a", "IS", -9.0, _stats_json(trades=9999, ret=-1.0, pf=0.1), "t0"),
                ("s_a", "OOS", -9.0, _stats_json(trades=9999, ret=-1.0, pf=0.1), "t0"),
                ("s_a", "OOS", -9.0,
                 _stats_json(trades=5961, ret=-0.368, pf=0.581, signal_stability=0.99), "t1"),
                ("s_b", "OOS", -9.0,
                 _stats_json(trades=10, ret=0.05, pf=1.2, signal_stability=0.99), "t1"),
                ("s_c", "OOS", -9.0,
                 _stats_json(trades=200, ret=0.05, pf=1.3, signal_stability=0.90), "t1"),
                ("s_d", "OOS", 1.5,
                 _stats_json(trades=400, ret=0.04, pf=1.4, signal_stability=0.99), "t1"),
                ("s_e", "OOS", 2.0,
                 _stats_json(trades=500, ret=0.05, pf=1.5, signal_stability=0.99), "t1"),
            ])
    conn.commit()
    conn.close()


class AnalyzePatternsTests(unittest.TestCase):
    def test_populated_db_renders_per_candidate_gate_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "ox.db"
            _make_db(db_path)
            result = analyze(db_path, _GATES)
            self.assertFalse(result["empty"])
            self.assertEqual(result["never_evaluated"], 1)  # s_f seeded only

            summary = result["summary"]
            self.assertEqual(summary["evaluated"], 5)
            self.assertEqual(summary["cleared_min_trades"], 4)   # a, c, d, e
            self.assertEqual(summary["rejected_ret_pf"], 1)      # s_a
            self.assertEqual(summary["rejected_stability"], 1)   # s_c
            self.assertEqual(summary["rejected_count"], 1)       # s_b
            self.assertEqual(summary["scored_ge_promote"], 2)    # s_d, s_e
            self.assertEqual(summary["score_neg9"], 3)

            rows = {row["sid"]: row for row in result["rows"]}
            # Latest OOS row won for s_a (trades 5961, not the stale 9999).
            self.assertEqual(rows["s_a"]["trades"], 5961)
            self.assertEqual(rows["s_a"]["cause"], "rejected: ret<=0 or pf<=1.0")
            self.assertEqual(rows["s_b"]["cause"], "rejected: min_trades count")
            self.assertEqual(rows["s_c"]["cause"], "rejected: signal stability")
            self.assertIn("scored 1.500", rows["s_d"]["cause"])
            self.assertIn("scored 2.000", rows["s_e"]["cause"])
            self.assertEqual(rows["s_e"]["status"], "LIVE_APPROVED")

            rendered = render(result)
            for sid in ("s_a", "s_b", "s_c", "s_d", "s_e"):
                self.assertIn(sid, rendered)
            for status in ("CANDIDATE", "ELITE", "PENDING_APPROVAL", "LIVE_APPROVED"):
                self.assertIn(f"--- {status} (", rendered)
            self.assertIn("Gate breakdown", rendered)
            self.assertNotIn("no training results", rendered)

    def test_empty_or_missing_db_reports_no_results_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            empty_db = directory_path / "empty.db"
            _make_db(empty_db, empty=True)
            result = analyze(empty_db, _GATES)
            self.assertTrue(result["empty"])
            self.assertIn("no training results", render(result))

            missing = analyze(directory_path / "absent.db", _GATES)
            self.assertTrue(missing["empty"])
            self.assertIn("no database at", render(missing))


if __name__ == "__main__":
    unittest.main()
