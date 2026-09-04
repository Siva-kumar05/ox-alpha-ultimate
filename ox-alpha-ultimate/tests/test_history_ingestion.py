"""History ingestion: one malformed broker candle must not halt the agent.

Regression for the RELIANCE boot halt (the KILL.flag left in the repo): a
single non-numeric row from the venue froze the whole agent at startup.
Malformed rows are now skipped and logged; only systemic corruption (more
than ``MAX_BAD_CANDLE_FRACTION`` of the batch unusable) fails closed, and an
all-bad or empty batch still raises ``MarketDataError``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ox.agent import Agent, MAX_BAD_CANDLE_FRACTION, MIN_REJECTED_FOR_SYSTEMIC
from ox.brokers import MarketDataError
from ox.core import DB


def _run(broker, db_path, sym) -> DB:
    """Drive Agent.refresh_history against a stub agent and return the DB.

    The DB handle is closed on failure so the temp directory can be removed
    (Windows keeps SQLite files locked until the connection is released).
    """
    db = DB(db_path)
    stub = type("HistoryAgent", (), {
        "cfg": {"timeframe_sec": 60, "history_days": 5},
        "db": db,
        "broker": broker,
    })()
    try:
        Agent.refresh_history(stub, sym)
        return db
    except BaseException:
        db.close()
        raise


class _Broker:
    def __init__(self, rows):
        self.rows = rows

    def hist(self, sym, tf, days):
        self.called = (sym, tf, days)
        return self.rows


class HistoryIngestionTests(unittest.TestCase):
    def test_single_non_numeric_candle_is_skipped_not_fatal(self):
        # The RELIANCE case: one row carries a non-numeric timestamp while the
        # rest of the batch is healthy. The agent must keep the good candles.
        rows = [
            (1700000000, 100.0, 101.0, 99.0, 100.0, 5000),
            ("2026-09-01T02:40:14+05:30", 100.0, 101.0, 99.0, 100.0, 5000),
            (1700000100, 100.0, 101.0, 99.0, 100.0, 5000),
        ]
        with tempfile.TemporaryDirectory() as directory:
            db = _run(_Broker(rows), str(Path(directory) / "t.db"), "RELIANCE")
            try:
                stored = db.q("SELECT ts FROM candles WHERE sym='RELIANCE' ORDER BY ts")
                self.assertEqual([r[0] for r in stored], [1700000000, 1700000100])
            finally:
                db.close()

    def test_nan_and_inconsistent_ohlc_rows_are_skipped(self):
        rows = [
            (1700000000, 100.0, 101.0, 99.0, 100.0, 5000),
            (1700000100, float("nan"), 101.0, 99.0, 100.0, 5000),
            (1700000200, 100.0, 99.0, 101.0, 100.0, 5000),  # high < low
            (1700000300, 100.0, 101.0, 99.0, 100.0, 5000),
        ]
        with tempfile.TemporaryDirectory() as directory:
            db = _run(_Broker(rows), str(Path(directory) / "t.db"), "TCS")
            try:
                stored = db.q("SELECT ts FROM candles WHERE sym='TCS' ORDER BY ts")
                self.assertEqual([r[0] for r in stored], [1700000000, 1700000300])
            finally:
                db.close()

    def test_minority_corruption_is_tolerated_up_to_the_fraction(self):
        # Exactly 1 of 5 rejected == MAX_BAD_CANDLE_FRACTION: still tolerated
        # (the gate is strictly greater-than), healthy rows survive.
        rows = [(1700000000 + i * 60, 100.0, 101.0, 99.0, 100.0, 5000) for i in range(5)]
        rows[2] = ("bad", 100.0, 101.0, 99.0, 100.0, 5000)
        with tempfile.TemporaryDirectory() as directory:
            db = _run(_Broker(rows), str(Path(directory) / "t.db"), "TCS")
            try:
                stored = db.q("SELECT ts FROM candles WHERE sym='TCS' ORDER BY ts")
                self.assertEqual(len(stored), 4)
            finally:
                db.close()

    def test_systemic_corruption_fails_closed(self):
        # 6 of 10 rejected clears both gates (>= 5 rows, > 25%): the series
        # itself is suspect, so boot must still halt.
        rows = [(1700000000 + i * 60, 100.0, 101.0, 99.0, 100.0, 5000) for i in range(10)]
        for i in range(6):
            rows[i] = ("bad", 100.0, 101.0, 99.0, 100.0, 5000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MarketDataError):
                _run(_Broker(rows), str(Path(directory) / "t.db"), "TCS")

    def test_all_bad_rows_still_fails_closed(self):
        rows = [("bad", 1.0, 2.0, 0.5, 1.5, 100), ("worse", 1.0, 2.0, 0.5, 1.5, 100)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MarketDataError):
                _run(_Broker(rows), str(Path(directory) / "t.db"), "TCS")

    def test_empty_batch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MarketDataError):
                _run(_Broker([]), str(Path(directory) / "t.db"), "TCS")

    def test_gate_constants_are_sane(self):
        self.assertGreater(MAX_BAD_CANDLE_FRACTION, 0.0)
        self.assertLess(MAX_BAD_CANDLE_FRACTION, 1.0)
        self.assertGreater(MIN_REJECTED_FOR_SYSTEMIC, 1)


if __name__ == "__main__":
    unittest.main()