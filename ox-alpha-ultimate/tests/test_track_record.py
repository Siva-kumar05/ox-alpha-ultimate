"""Offline contract for the supervised live-test track-record journal.

``ox.live_test`` appends one JSON line per supervised session to
track_record/sessions.jsonl; the pure reader/aggregator must tolerate a
missing or partially corrupted journal without hiding the rest of the record.
No network, broker, or credentials are involved.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ox.track_record import load_records, summarize, write_record


class TrackRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.mkdtemp(prefix="ox-trackrec-")
        self.path = Path(self._directory) / "sessions.jsonl"

    def tearDown(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)

    def test_write_then_load_and_summarize(self) -> None:
        self.assertTrue(write_record({
            "ts": "2026-09-04T00:00:00+00:00", "tool": "live-test",
            "read_only": True, "ok": True, "failures": 0,
            "symbol_count": 5, "prime": None}, self.path))
        self.assertTrue(write_record({
            "ts": "2026-09-05T00:00:00+00:00", "tool": "live-test",
            "read_only": True, "ok": False, "failures": 2,
            "symbol_count": 5,
            "prime": {"seconds": 300.0, "fills": 4}}, self.path))

        records = load_records(self.path)
        self.assertEqual(len(records), 2)
        summary = summarize(records)
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["prime_seconds_total"], 300.0)
        self.assertEqual(summary["prime_fills_total"], 4)
        self.assertEqual(summary["first_ts"], "2026-09-04T00:00:00+00:00")
        self.assertEqual(summary["last_ts"], "2026-09-05T00:00:00+00:00")

    def test_load_skips_malformed_lines_and_missing_file(self) -> None:
        # Missing journal: an empty record, never an exception.
        self.assertEqual(load_records(Path(self._directory) / "absent.jsonl"), [])

        self.path.write_text(
            "not json at all\n"
            + json.dumps({"ok": True, "ts": "2026-09-04T00:00:00+00:00"}) + "\n"
            + "[1,2,3]\n"                      # valid JSON but not a dict
            + "{broken\n"
            + json.dumps({"ok": False, "ts": "2026-09-05T00:00:00+00:00"}) + "\n",
            encoding="utf-8")
        records = load_records(self.path)
        self.assertEqual(len(records), 2, "malformed lines must be skipped")
        self.assertEqual(summarize(records)["passed"], 1)
        self.assertEqual(summarize(records)["failed"], 1)

    def test_empty_summary_is_stable(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["sessions"], 0)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertIsNone(summary["first_ts"])
        self.assertEqual(summary["prime_seconds_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
