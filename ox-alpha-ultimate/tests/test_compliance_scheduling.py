"""Compliance cadence: reports are generated at boot, once per period.

Regression for the audit finding that ``ComplianceReporter.generate_report``
existed but was never triggered by any schedule.  ``run_due_reports()`` (called
from ``Agent.boot``) evaluates the configured daily/weekly/monthly cadence and
records a kv completion marker so each report is generated at most once per
period even across restarts.  ``generate_report`` itself is stubbed here so the
test covers the cadence driver deterministically without depending on live DB
contents or the wall-clock weekday.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from ox.compliance_reporting import (
    ComplianceReport,
    ComplianceReporter,
    ReportFormat,
)
from ox.core import DB


def _reporter(directory: str) -> tuple[ComplianceReporter, DB]:
    db = DB(str(Path(directory) / "oxalpha.db"))
    cfg = {
        "db_path": str(Path(directory) / "oxalpha.db"),
        "compliance_reporting": {
            "enabled": True,
            "reports_dir": "compliance_reports",
            "default_format": "json",
            "schedule": {"daily": "06:00", "weekly": "mon 06:00", "monthly": "1 06:00"},
        },
    }
    return ComplianceReporter(cfg, db), db


class ComplianceSchedulingTests(unittest.TestCase):
    @staticmethod
    def _stub_generate(reporter: ComplianceReporter, directory: str) -> list[str]:
        """Replace DB-driven generation with a deterministic file write."""
        produced: list[str] = []

        def fake(report_type, period_start=None, period_end=None, format=None):  # noqa: A002 - mirrors real signature
            path = Path(directory) / f"{report_type.value}_{len(produced)}.json"
            path.write_text("{}", encoding="utf-8")
            produced.append(str(path))
            return ComplianceReport(
                report_id=f"x_{len(produced)}",
                report_type=report_type,
                period_start=period_start or "",
                period_end=period_end or "",
                generated_at=datetime.now().isoformat(),
                format=format or ReportFormat.JSON,
                file_path=str(path),
                sections={},
            )

        reporter.generate_report = fake  # type: ignore[method-assign]
        return produced

    def test_daily_report_generated_once_per_period(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter, db = _reporter(directory)
            try:
                produced = self._stub_generate(reporter, directory)
                out = reporter.run_due_reports()
                self.assertEqual(len(out), 1, f"expected exactly the daily report, got {out}")
                self.assertTrue(out[0].endswith("daily_0.json"))
                marker = db.kv_get("compliance_done_daily")
                # Marker records the period the report covers (the previous day).
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                self.assertEqual(str(marker), yesterday)
                # Same period again (e.g. a restart within the day): marker present.
                self.assertEqual(reporter.run_due_reports(), [])
                self.assertEqual(len(produced), 1)
            finally:
                db.close()

    def test_weekly_and_monthly_cadence_respect_config_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter, db = _reporter(directory)
            try:
                self._stub_generate(reporter, directory)
                today = datetime.now()
                # Weekly fires only on the configured weekday (default mon).
                expected = 1  # daily always due on a fresh DB
                if today.weekday() == 0:
                    expected += 1  # weekly (mon)
                if today.day == 1:
                    expected += 1  # monthly (1st)
                out = reporter.run_due_reports()
                self.assertEqual(len(out), expected, f"daily/weekly/monthly cadence wrong: {out}")
            finally:
                db.close()

    def test_disabled_reporter_generates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            db = DB(str(Path(directory) / "oxalpha.db"))
            try:
                cfg = {"db_path": str(Path(directory) / "oxalpha.db"),
                       "compliance_reporting": {"enabled": False}}
                reporter = ComplianceReporter(cfg, db)
                self.assertEqual(reporter.run_due_reports(), [])
            finally:
                db.close()

    def test_reports_anchor_under_the_db_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter, db = _reporter(directory)
            try:
                # Windows may hand back a short (8.3) temp path while the
                # resolved DB path is long-form: compare resolved parents.
                self.assertIn(Path(directory).resolve(), reporter.reports_dir.parents,
                              f"reports_dir must be anchored under the db dir: {reporter.reports_dir}")
                self.assertTrue(reporter.reports_dir.is_dir())
            finally:
                db.close()

    def test_daily_stays_due_when_generation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter, db = _reporter(directory)
            try:

                def broken(report_type, period_start=None, period_end=None, format=None):  # noqa: A002
                    raise RuntimeError("collector failed")

                reporter.generate_report = broken  # type: ignore[method-assign]
                self.assertEqual(reporter.run_due_reports(), [])
                # No marker written on failure: the report stays due next boot.
                self.assertIsNone(db.kv_get("compliance_done_daily"))
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
