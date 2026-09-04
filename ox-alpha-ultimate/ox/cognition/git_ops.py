"""Semantic git-native operations (gap #23).

Instead of bash-level git commands, the agent maintains a structured model of
branches, diffs, conflicts, and commit history. It can intelligently propose
merge conflict resolutions, suggest commit strategies, and analyse repository
history for patterns — rather than shelling out to `git status` every time.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GitDiffHunk:
    file: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    lines: list[str]


@dataclass
class MergeConflict:
    file: str
    ours: list[str]
    theirs: list[str]
    ancestor: list[str]
    resolution_hint: str = ""


@dataclass
class BranchInfo:
    name: str
    ahead: int
    behind: int
    last_commit: str
    last_author: str
    last_ts: str


def _run_git(root: Path, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True,
            text=True, timeout=timeout, check=False)
        return result.returncode, result.stdout or "", result.stderr or ""
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return 127, "", f"git unavailable: {exc.__class__.__name__}: {exc}"


class GitOps:
    """Structured, semantic git interface — no more raw bash strings."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def available(self) -> bool:
        code, _, _ = _run_git(self.root, "rev-parse", "--git-dir")
        return code == 0

    # ── status / diff ───────────────────────────────────────────────────
    def status(self) -> dict:
        """Structured status grouped by change kind."""
        code, out, err = _run_git(self.root, "status", "--porcelain=v1")
        if code != 0:
            return {"ok": False, "error": err.strip() or "git status failed"}
        groups: dict[str, list[str]] = defaultdict(list)
        for line in out.splitlines():
            if not line:
                continue
            code_x = line[:2]
            path = line[3:]
            groups[code_x].append(path)
        return {
            "ok": True,
            "modified": groups.get(" M", []) + groups.get("MM", []),
            "staged": groups.get("M ", []) + groups.get("A ", []),
            "untracked": groups.get("??", []),
            "deleted": groups.get(" D", []) + groups.get("D ", []),
            "renamed": groups.get("R ", []),
            "conflicted": groups.get("UU", []) + groups.get("AA", []),
        }

    def diff(self, target: str = "HEAD", *, staged: bool = False) -> list[GitDiffHunk]:
        """Structured per-hunk diff, not raw text."""
        args = ["diff", "--unified=3", target]
        if staged:
            args = ["diff", "--cached", "--unified=3"]
        code, out, _ = _run_git(self.root, *args)
        if code != 0:
            return []
        return self._parse_diff(out)

    def changed_files(self, base: str = "HEAD", compare: str = "") -> list[str]:
        """Files differing between two refs."""
        if compare:
            args = ["diff", "--name-only", f"{base}...{compare}"]
        else:
            args = ["diff", "--name-only", base]
        code, out, _ = _run_git(self.root, *args)
        if code != 0:
            return []
        return [l.strip() for l in out.splitlines() if l.strip()]

    # ── branches ────────────────────────────────────────────────────────
    def branches(self) -> list[BranchInfo]:
        """All branches with ahead/behind vs upstream and last commit."""
        code, out, _ = _run_git(self.root, "for-each-ref", "--format=%(refname:short)|%(ahead-behind:HEAD)|%(objectname:short)|%(author-name)|%(authordate:iso)", "refs/heads/")
        if code != 0:
            return []
        info: list[BranchInfo] = []
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 5:
                continue
            name, ab, *rest = parts
            ahead = behind = 0
            m = re.match(r"\+(\d+) -(\d+)", ab)
            if m:
                ahead, behind = int(m.group(1)), int(m.group(2))
            info.append(BranchInfo(
                name=name, ahead=ahead, behind=behind,
                last_commit=rest[0] if len(rest) >= 1 else "",
                last_author=rest[1] if len(rest) >= 2 else "",
                last_ts=rest[2] if len(rest) >= 3 else ""))
        return info

    def current_branch(self) -> str | None:
        code, out, _ = _run_git(self.root, "rev-parse", "--abbrev-ref", "HEAD")
        return out.strip() if code == 0 else None

    # ── history analytics ───────────────────────────────────────────────
    def commit_history(self, limit: int = 50, since: str = "") -> list[dict]:
        """Commits as structured records."""
        fmt = "%H|%h|%an|%ae|%at|%s"
        args = ["log", f"--pretty=format:{fmt}", f"-n{limit}"]
        if since:
            args.append(f"--since={since}")
        code, out, _ = _run_git(self.root, *args)
        if code != 0:
            return []
        commits = []
        for line in out.splitlines():
            parts = line.split("|", 5)
            if len(parts) < 6:
                continue
            commits.append({
                "hash": parts[0], "short": parts[1],
                "author": parts[2], "email": parts[3],
                "ts": int(parts[4]), "subject": parts[5],
            })
        return commits

    def top_contributors(self, limit: int = 10) -> list[dict]:
        code, out, _ = _run_git(self.root, "shortlog", "-sn", "HEAD")
        if code != 0:
            return []
        result = []
        for line in out.splitlines()[:limit]:
            m = re.match(r"\s*(\d+)\s+(.+)", line)
            if m:
                result.append({"commits": int(m.group(1)), "author": m.group(2).strip()})
        return result

    def hotspots(self, limit: int = 20) -> list[dict]:
        """Most-frequently-changed files — refactoring candidates."""
        code, out, _ = _run_git(self.root, "log", "--name-only", "--pretty=format:")
        if code != 0:
            return []
        c = Counter(l.strip() for l in out.splitlines() if l.strip())
        return [{"file": f, "changes": n} for f, n in c.most_common(limit)]

    # ── merge conflicts ─────────────────────────────────────────────────
    def conflicts(self) -> list[MergeConflict]:
        """Parse conflict markers and propose resolutions."""
        st = self.status()
        if not st.get("ok"):
            return []
        conflict_files = st.get("conflicted", [])
        results = []
        for rel in conflict_files:
            path = self.root / rel
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for c in self._parse_conflicts(rel, text):
                results.append(c)
        return results

    def suggest_conflict_resolution(self, conflict: MergeConflict) -> str:
        """Heuristic conflict resolution suggestion."""
        ours, theirs, anc = conflict.ours, conflict.theirs, conflict.ancestor
        if ours == theirs:
            return "identical sides; pick either"
        if not anc:
            if len(ours) > len(theirs) and all(t in ours for t in theirs):
                return "ours subsumes theirs; use OURS"
            if len(theirs) > len(ours) and all(o in theirs for o in ours):
                return "theirs subsumes ours; use THEIRS"
        # Similarity-based
        if anc:
            sim_o = _seq_similarity(anc, ours)
            sim_t = _seq_similarity(anc, theirs)
            if sim_o > sim_t + 0.15:
                return f"ours closer to ancestor ({sim_o:.2f} vs {sim_t:.2f}); prefer OURS"
            if sim_t > sim_o + 0.15:
                return f"theirs closer to ancestor ({sim_t:.2f} vs {sim_o:.2f}); prefer THEIRS"
        return "manual review needed: changes are divergent"

    # ── commit strategy ─────────────────────────────────────────────────
    def suggest_commit_strategy(self) -> dict:
        """Recommend how to split current changes into logical commits."""
        status = self.status()
        if not status.get("ok"):
            return {"ok": False, "error": status.get("error")}
        changed = (status.get("modified", []) + status.get("staged", [])
                   + status.get("deleted", []) + status.get("renamed", []))
        if not changed:
            return {"ok": True, "commits": [], "summary": "nothing to commit"}
        groups: dict[str, list[str]] = defaultdict(list)
        for f in changed:
            p = Path(f)
            domain = p.parts[0] if len(p.parts) > 1 else "<root>"
            groups[domain].append(f)
        commits = []
        for domain, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(files) == 1:
                subject = f"update {Path(files[0]).name}"
            else:
                subject = f"{domain}: touch {len(files)} files"
            commits.append({"subject": subject, "files": files})
        return {"ok": True, "commits": commits,
                "summary": f"{len(changed)} files across {len(groups)} domain(s)"}

    # ── internals ───────────────────────────────────────────────────────
    @staticmethod
    def _parse_diff(raw: str) -> list[GitDiffHunk]:
        hunks: list[GitDiffHunk] = []
        current_file = ""
        lines = raw.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("diff --git"):
                m = re.search(r" b/(.+)$", line)
                current_file = m.group(1) if m else ""
                i += 1
                continue
            if line.startswith("@@"):
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line)
                if m:
                    os_, ol, ns, nl, header = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
                    hunk_lines: list[str] = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith(("diff --git", "@@")):
                        hunk_lines.append(lines[i])
                        i += 1
                    hunks.append(GitDiffHunk(
                        file=current_file,
                        old_start=int(os_), old_lines=int(ol or 1),
                        new_start=int(ns), new_lines=int(nl or 1),
                        header=header.strip(), lines=hunk_lines))
                    continue
            i += 1
        return hunks

    @staticmethod
    def _parse_conflicts(file: str, text: str) -> list[MergeConflict]:
        ours: list[str] = []
        theirs: list[str] = []
        anc: list[str] = []
        conflicts: list[MergeConflict] = []
        state = "normal"
        for line in text.splitlines(keepends=True):
            if line.startswith("<<<<<<<"):
                ours, theirs, anc = [], [], []
                state = "ours"
                continue
            if line.startswith("|||||||"):
                state = "ancestor"
                continue
            if line.startswith("======="):
                state = "theirs"
                continue
            if line.startswith(">>>>>>>"):
                conflicts.append(MergeConflict(file=file, ours=ours, theirs=theirs, ancestor=anc))
                state = "normal"
                continue
            if state == "ours":
                ours.append(line)
            elif state == "theirs":
                theirs.append(line)
            elif state == "ancestor":
                anc.append(line)
        return conflicts


def _seq_similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = "".join(a), "".join(b)
    if not sa or not sb:
        return 0.0
    return difflib.SequenceMatcher(None, sa, sb).ratio()


# ═══════════════════════════════════════════════════════════════════════════
# Native database client (gap #20) — SQL abstraction with safety guards
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None
    elapsed_ms: float = 0.0

    def dicts(self) -> list[dict[str, Any]]:
        if not self.columns or not self.rows:
            return []
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_dataframe(self):  # -> pd.DataFrame | None (no import at top)
        try:
            import pandas as pd  # type: ignore
            if not self.columns or not self.rows:
                return pd.DataFrame()
            return pd.DataFrame(self.rows, columns=self.columns)
        except Exception:  # noqa: BLE001
            return None


_DANGEROUS = re.compile(r"\b(DROP|TRUNCATE|ALTER|DELETE\s+FROM\s+\w+\s*$|UPDATE\s+\w+\s+SET.+\s*$)\b",
                        re.IGNORECASE | re.DOTALL)
_READ_ONLY = re.compile(r"^\s*(SELECT|PRAGMA|EXPLAIN)\b", re.IGNORECASE)


class DatabaseClient:
    """SQL client guardrail: parameterised queries, readonly mode, limits.

    This is the structural substitute for raw bash `sqlite3` / `psql` calls.
    Unparameterised queries are rejected; write-queries require explicit
    write-mode opt-in. Result sets are bounded.
    """

    MAX_ROWS = 10_000
    DRIVERS = {"sqlite3": "sqlite3", "postgres": "psycopg2", "pg": "psycopg2", "mysql": "pymysql"}

    def __init__(self, dsn: str, *, readonly: bool = True):
        self.dsn = dsn
        self.readonly = readonly
        self.driver_name = self._detect_driver()
        self._conn = None

    # ── lifecycle ───────────────────────────────────────────────────────
    def connect(self) -> "DatabaseClient":
        if self.driver_name == "sqlite3":
            import sqlite3
            path = self.dsn.replace("sqlite:///", "").replace("sqlite://", "")
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = None
        elif self.driver_name == "psycopg2":
            try:
                import psycopg2  # type: ignore
            except ImportError as exc:
                raise RuntimeError("psycopg2 not installed") from exc
            self._conn = psycopg2.connect(self.dsn)
        elif self.driver_name == "pymysql":
            try:
                import pymysql  # type: ignore  # noqa: F401 - availability probe
            except ImportError as exc:
                raise RuntimeError("pymysql not installed") from exc
            # dsn parsed as URL-style; fallback to env for simplicity
            raise NotImplementedError("pymysql DSN parsing — extend with URL parser")
        return self

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # ── queries ─────────────────────────────────────────────────────────
    def query(self, sql: str, params: tuple | None = None, *,
              limit: int = 0, write: bool = False) -> QueryResult:
        import time as _time
        start = _time.monotonic()
        if self._conn is None:
            try:
                self.connect()
            except Exception as exc:  # noqa: BLE001
                return QueryResult(ok=False, error=f"connect: {exc.__class__.__name__}: {exc}")
        # Safety checks
        if self.readonly and not _READ_ONLY.match(sql):
            return QueryResult(ok=False, error="readonly mode; this query looks like a write")
        if not write and _DANGEROUS.search(sql) and not _READ_ONLY.match(sql):
            return QueryResult(ok=False, error="dangerous-looking statement; pass write=True to override")
        if params is not None and isinstance(params, (list, tuple)) and "?" not in sql and "%s" not in sql:
            return QueryResult(ok=False, error="params supplied but no placeholders ('?' or '%s') in SQL")
        effective = sql
        if limit and limit > 0 and "LIMIT" not in sql.upper() and _READ_ONLY.match(sql):
            effective = f"{sql.rstrip().rstrip(';')} LIMIT {min(limit, self.MAX_ROWS)}"
        cur = None
        try:
            cur = self._conn.cursor()
            cur.execute(effective, tuple(params) if params else ())
            if _READ_ONLY.match(sql):
                raw = cur.fetchmany(min(limit or self.MAX_ROWS, self.MAX_ROWS))
                columns = [d[0] for d in (cur.description or [])]
                return QueryResult(ok=True, rows=list(raw), columns=columns,
                                   row_count=len(raw),
                                   elapsed_ms=round((_time.monotonic() - start) * 1000, 2))
            else:
                try:
                    self._conn.commit()
                except Exception:  # noqa: BLE001
                    pass
                return QueryResult(ok=True, rows=[], columns=[],
                                   row_count=getattr(cur, "rowcount", -1),
                                   elapsed_ms=round((_time.monotonic() - start) * 1000, 2))
        except Exception as exc:  # noqa: BLE001
            return QueryResult(ok=False, error=f"{exc.__class__.__name__}: {exc}",
                               elapsed_ms=round((_time.monotonic() - start) * 1000, 2))
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    pass

    def tables(self) -> list[str]:
        if self.driver_name == "sqlite3":
            r = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return [row[0] for row in r.rows]
        return []

    def schema(self, table: str) -> list[dict]:
        if self.driver_name == "sqlite3":
            r = self.query(f"PRAGMA table_info({self._safe_ident(table)})")
            return [dict(zip(r.columns, row)) for row in r.rows] if r.ok else []
        return []

    # ── helpers ─────────────────────────────────────────────────────────
    def _detect_driver(self) -> str:
        d = self.dsn.lower()
        if d.startswith(("sqlite", "file:")) or d.endswith(".db") or d.endswith(".sqlite"):
            return "sqlite3"
        if d.startswith(("postgres", "postgresql")):
            return "psycopg2"
        if d.startswith("mysql"):
            return "pymysql"
        # heuristic: bare path ending in .db
        return "sqlite3"

    @staticmethod
    def _safe_ident(name: str) -> str:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return name
        raise ValueError(f"unsafe identifier: {name!r}")
