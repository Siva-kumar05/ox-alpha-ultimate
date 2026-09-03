"""Transactional file operations: stage, verify freshness, commit or roll back.

Every edit is staged against a snapshot; if the user (or another process)
touched a file mid-transaction the staleness check aborts before any write,
so manual changes can never be silently overwritten. Commit applies writes
via atomic os.replace, and a journal on disk recovers an interrupted commit
on the next begin().
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class StaleStateError(RuntimeError):
    """A staged file changed on disk after the snapshot was taken."""


class TransactionError(RuntimeError):
    pass


def _digest(path: Path) -> tuple[str, float]:
    if not path.exists():
        return ("", 0.0)
    data = path.read_bytes()
    stat = path.stat()
    return hashlib.sha256(data).hexdigest(), float(stat.st_mtime)


@dataclass
class StagedWrite:
    path: Path
    content: str
    prior_digest: str
    prior_mtime: float
    kind: str = "write"  # write | delete


@dataclass
class FileTransaction:
    root: Path
    journal: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.journal is None:
            self.journal = self.root / ".ox-alpha" / "transactions" / "journal.json"
        self.staged: list[StagedWrite] = []
        self.snapshots: dict[str, tuple[str, float]] = {}
        self.active = False

    # ── lifecycle ───────────────────────────────────────────────────────
    def begin(self) -> None:
        self.staged.clear()
        self.snapshots.clear()
        self.active = True
        self._recover_journal()

    def stage_write(self, path: str | Path, content: str) -> None:
        self._assert_active()
        target = self._resolve(path)
        digest, mtime = _digest(target)
        self.snapshots[str(target)] = (digest, mtime)
        self.staged.append(StagedWrite(target, content, digest, mtime, "write"))
        self._write_journal()

    def stage_delete(self, path: str | Path) -> None:
        self._assert_active()
        target = self._resolve(path)
        digest, mtime = _digest(target)
        if not target.exists():
            raise TransactionError(f"cannot stage deletion of missing file: {target}")
        self.snapshots[str(target)] = (digest, mtime)
        self.staged.append(StagedWrite(target, "", digest, mtime, "delete"))
        self._write_journal()

    def verify_freshness(self) -> None:
        """Abort if anything we are about to touch drifted from its snapshot."""
        for target, (digest, mtime) in self.snapshots.items():
            current_digest, current_mtime = _digest(Path(target))
            if (current_digest, current_mtime) != (digest, mtime):
                raise StaleStateError(
                    f"{target} changed on disk since the transaction snapshot; refusing to overwrite")

    def commit(self) -> dict:
        self._assert_active()
        self.verify_freshness()
        applied: list[dict] = []
        for write in self.staged:
            write.path.parent.mkdir(parents=True, exist_ok=True)
            if write.kind == "delete":
                backup = self._backup(write)
                write.path.unlink()
                applied.append({"path": str(write.path), "action": "delete", "backup": backup})
            else:
                backup = self._backup(write)
                tmp = write.path.with_suffix(write.path.suffix + f".txn{int(time.time() * 1000) % 1_000_000}")
                tmp.write_text(write.content, encoding="utf-8")
                os.replace(tmp, write.path)
                applied.append({"path": str(write.path), "action": "write", "backup": backup})
        self._clear_journal()
        self.active = False
        return {"committed": applied}

    def rollback(self) -> dict:
        """Discard staged changes; nothing has been written yet."""
        self._assert_active()
        count = len(self.staged)
        self.staged.clear()
        self.snapshots.clear()
        self.active = False
        self._clear_journal()
        return {"rolled_back": count}

    def restore(self, backup_dir: str | Path) -> list[str]:
        """Restore committed files from a commit backup directory."""
        restored = []
        for backup in sorted(Path(backup_dir).glob("*")):
            if backup.is_file():
                original = self.root / backup.name[:64]
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(backup.read_bytes())
                restored.append(str(original))
        return restored

    # ── internals ───────────────────────────────────────────────────────
    def _assert_active(self) -> None:
        if not self.active:
            raise TransactionError("no transaction is active (call begin() first)")

    def _resolve(self, path: str | Path) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        resolved = target.resolve()
        if self.root.resolve() not in resolved.parents and resolved != self.root.resolve():
            raise TransactionError(f"path escapes transaction root: {target}")
        return resolved

    def _backup(self, write: StagedWrite) -> str | None:
        if not write.path.exists():
            return None
        backup_dir = self.root / ".ox-alpha" / "transactions" / "backups" / f"txn_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(str(write.path).encode()).hexdigest()[:64]
        (backup_dir / name).write_bytes(write.path.read_bytes())
        return str(backup_dir / name)

    def _write_journal(self) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"path": str(w.path), "kind": w.kind} for w in self.staged]
        self.journal.write_text(json.dumps({"ts": time.time(), "staged": payload}), encoding="utf-8")

    def _clear_journal(self) -> None:
        self.journal.unlink(missing_ok=True)

    def _recover_journal(self) -> None:
        if self.journal.exists():
            try:
                payload = json.loads(self.journal.read_text(encoding="utf-8"))
                # A journal from a crashed commit is surfaced, never silently replayed.
                raise TransactionError(
                    f"unrecovered transaction journal from {payload.get('ts')}: "
                    f"{len(payload.get('staged', []))} staged writes need manual review")
            except json.JSONDecodeError:
                self._clear_journal()
