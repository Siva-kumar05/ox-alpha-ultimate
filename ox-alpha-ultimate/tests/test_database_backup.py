"""Database backup: WAL checkpoint + verification that actually verifies.

The backup manager snapshots through sqlite's online backup API (already
crash-consistent) after a passive WAL checkpoint, then verifies the copy
with a real PRAGMA integrity_check result (previously the result row was
never fetched, so verification could never fail).
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ox.database_backup import DatabaseBackupManager


class DatabaseBackupTests(unittest.TestCase):
    def _manager(self, directory: Path, compress: bool = False):
        db_path = directory / "ox.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE trades(sym TEXT, qty INT)")
        conn.execute("INSERT INTO trades VALUES('TCS', 10),('RELIANCE', 5)")
        conn.commit()
        conn.close()
        cfg = {"database_backup": {
            "enabled": True,
            "backup_dir": str(directory / "backups"),
            "compress": compress,
            "interval_hours": 6,
            "retention_days": 30,
            "max_backups": 100,
        }}
        manager = DatabaseBackupManager(cfg, str(db_path))
        return manager, db_path

    def test_backup_preserves_rows_and_passes_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self._manager(Path(directory), compress=False)
            info = manager.create_backup(force=True)
            self.assertEqual(info.status, "success", info.error)
            self.assertTrue(Path(info.backup_path).exists())
            conn = sqlite3.connect(info.backup_path)
            rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            conn.close()
            self.assertEqual(rows, 2)
            self.assertEqual(integrity, "ok")

    def test_backup_survives_wal_mode(self):
        """Committed WAL frames must land in the backup (checkpoint + backup API)."""
        with tempfile.TemporaryDirectory() as directory:
            manager, db_path = self._manager(Path(directory), compress=False)
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=wal")
            conn.execute("INSERT INTO trades VALUES('HDFCBANK', 3)")
            conn.commit()
            conn.close()
            info = manager.create_backup(force=True)
            self.assertEqual(info.status, "success", info.error)
            conn = sqlite3.connect(info.backup_path)
            rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            conn.close()
            self.assertEqual(rows, 3)

    def test_verify_rejects_corrupt_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self._manager(Path(directory))
            bogus = Path(directory) / "bogus.db"
            bogus.write_bytes(b"this is not a sqlite database at all........")
            self.assertFalse(manager._verify_backup(bogus))


if __name__ == "__main__":
    unittest.main()
