"""Hierarchical long-term memory: working, episodic, semantic, procedural, failure.

SQLite-backed with vector recall so retrieval is semantic, not keyword-exact.
Every layer survives across sessions; consolidation decays stale episodic
entries and merges near-duplicate semantic facts so the store stays useful as
it grows instead of degrading into an unranked log.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


from .vectors import cosine, embed, pack, unpack

SCHEMA = {
    "episodic": (
        "eid INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL, "
        "content TEXT NOT NULL, tags TEXT NOT NULL, importance REAL NOT NULL DEFAULT 0.5, "
        "access_count INTEGER NOT NULL DEFAULT 0, embedding BLOB"
    ),
    "semantic": (
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1.0, "
        "updated_at TEXT NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, embedding BLOB"
    ),
    "procedural": (
        "name TEXT PRIMARY KEY, pattern TEXT NOT NULL, success_count INTEGER NOT NULL DEFAULT 0, "
        "failure_count INTEGER NOT NULL DEFAULT 0, last_used TEXT, embedding BLOB"
    ),
    "failures": (
        "fid INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, tool TEXT NOT NULL, "
        "error TEXT NOT NULL, context TEXT, fix TEXT, resolved INTEGER NOT NULL DEFAULT 0, embedding BLOB"
    ),
    "working": "key TEXT PRIMARY KEY, value TEXT NOT NULL, set_at TEXT NOT NULL",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, path: str | Path = ".ox-alpha/memory/cognition.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.c = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        with self.lock:
            self.c.execute("PRAGMA journal_mode=WAL")
            for table, definition in SCHEMA.items():
                self.c.execute(f"CREATE TABLE IF NOT EXISTS {table}({definition})")
            self.c.commit()

    # ── episodic ────────────────────────────────────────────────────────
    def record_episodic(self, kind: str, content: Mapping, tags: list[str] | None = None,
                        importance: float = 0.5) -> int:
        text = f"{kind} {json.dumps(content, default=str)} {' '.join(tags or [])}"
        with self.lock:
            cur = self.c.execute(
                "INSERT INTO episodic(ts,kind,content,tags,importance,embedding)VALUES(?,?,?,?,?,?)",
                (_utcnow(), kind, json.dumps(content, default=str), json.dumps(tags or []),
                 float(importance), pack(embed(text))),
            )
            self.c.commit()
            return int(cur.lastrowid)

    def episodic_recent(self, limit: int = 20, kind: str | None = None) -> list[dict]:
        query = "SELECT eid,ts,kind,content,tags,importance FROM episodic"
        args: tuple = ()
        if kind:
            query += " WHERE kind=?"
            args = (kind,)
        query += " ORDER BY eid DESC LIMIT ?"
        args += (int(limit),)
        with self.lock:
            rows = self.c.execute(query, args).fetchall()
        return [
            {"id": r[0], "ts": r[1], "kind": r[2], "content": json.loads(r[3]),
             "tags": json.loads(r[4]), "importance": r[5]}
            for r in rows
        ]

    # ── semantic ────────────────────────────────────────────────────────
    def set_fact(self, key: str, value: Any, confidence: float = 1.0) -> None:
        with self.lock:
            self.c.execute(
                "INSERT OR REPLACE INTO semantic(key,value,confidence,updated_at,embedding)VALUES(?,?,?,?,?)",
                (key, json.dumps(value, default=str), float(confidence), _utcnow(),
                 pack(embed(f"{key} {value}"))),
            )
            self.c.commit()

    def get_fact(self, key: str, default=None):
        with self.lock:
            row = self.c.execute("SELECT value FROM semantic WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ── procedural ──────────────────────────────────────────────────────
    def learn_pattern(self, name: str, pattern: Mapping, *, success: bool) -> None:
        with self.lock:
            row = self.c.execute("SELECT success_count,failure_count FROM procedural WHERE name=?", (name,)).fetchone()
            wins, losses = (row[0], row[1]) if row else (0, 0)
            wins, losses = (wins + 1, losses) if success else (wins, losses + 1)
            self.c.execute(
                "INSERT OR REPLACE INTO procedural(name,pattern,success_count,failure_count,last_used,embedding)"
                "VALUES(?,?,?,?,?,?)",
                (name, json.dumps(pattern, default=str), wins, losses, _utcnow(),
                 pack(embed(f"{name} {pattern}"))),
            )
            self.c.commit()

    def pattern(self, name: str) -> dict | None:
        with self.lock:
            row = self.c.execute(
                "SELECT pattern,success_count,failure_count,last_used FROM procedural WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return None
        total = row[1] + row[2]
        return {"pattern": json.loads(row[0]), "success_count": row[1], "failure_count": row[2],
                "success_rate": (row[1] / total) if total else 0.0, "last_used": row[3]}

    def best_patterns(self, min_uses: int = 2) -> list[dict]:
        with self.lock:
            rows = self.c.execute(
                "SELECT name,pattern,success_count,failure_count FROM procedural"
            ).fetchall()
        out = []
        for name, pattern, wins, losses in rows:
            if wins + losses >= min_uses and wins / (wins + losses) >= 0.5:
                out.append({"name": name, "pattern": json.loads(pattern),
                            "success_rate": wins / (wins + losses)})
        return sorted(out, key=lambda p: p["success_rate"], reverse=True)

    # ── failure ─────────────────────────────────────────────────────────
    def record_failure(self, tool: str, error: str, context: Mapping | None = None, fix: str | None = None) -> int:
        resolved = fix is not None
        with self.lock:
            cur = self.c.execute(
                "INSERT INTO failures(ts,tool,error,context,fix,resolved,embedding)VALUES(?,?,?,?,?,?,?)",
                (_utcnow(), tool, str(error), json.dumps(context or {}, default=str), fix,
                 int(resolved), pack(embed(f"{tool} {error}"))),
            )
            self.c.commit()
            return int(cur.lastrowid)

    def similar_failures(self, tool: str, error: str, limit: int = 5) -> list[dict]:
        query_vector = embed(f"{tool} {error}")
        with self.lock:
            rows = self.c.execute("SELECT fid,tool,error,fix,resolved,embedding FROM failures").fetchall()
        scored = []
        for fid, ftool, ferr, fix, resolved, blob in rows:
            similarity = cosine(query_vector, unpack(blob)) if blob else 0.0
            if ftool == tool:
                similarity += 0.25
            scored.append((similarity, {"id": fid, "tool": ftool, "error": ferr,
                                        "fix": fix, "resolved": bool(resolved), "similarity": round(similarity, 3)}))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def resolve_failure(self, fid: int, fix: str) -> None:
        with self.lock:
            self.c.execute("UPDATE failures SET fix=?, resolved=1 WHERE fid=?", (fix, fid))
            self.c.commit()

    # ── working ─────────────────────────────────────────────────────────
    def set_working(self, key: str, value: Any) -> None:
        with self.lock:
            self.c.execute("INSERT OR REPLACE INTO working VALUES(?,?,?)", (key, json.dumps(value, default=str), _utcnow()))
            self.c.commit()

    def get_working(self, key: str, default=None):
        with self.lock:
            row = self.c.execute("SELECT value FROM working WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def clear_working(self) -> None:
        with self.lock:
            self.c.execute("DELETE FROM working")
            self.c.commit()

    # ── unified semantic recall across layers ───────────────────────────
    def recall(self, query: str, limit: int = 5, *, min_similarity: float = 0.05) -> dict:
        """Rank episodic events, semantic facts, procedural patterns, and known
        failures by vector similarity to the query — the memory "comes back"
        unprompted by exact keywords."""
        query_vector = embed(query)
        results: dict[str, list] = {"episodic": [], "semantic": [], "procedural": [], "failure": []}
        with self.lock:
            episodes = self.c.execute(
                "SELECT eid,ts,kind,content,tags,importance,embedding FROM episodic").fetchall()
            facts = self.c.execute("SELECT key,value,confidence,embedding FROM semantic").fetchall()
            patterns = self.c.execute("SELECT name,pattern,embedding FROM procedural").fetchall()
        for eid, ts, kind, content, tags, importance, blob in episodes:
            similarity = cosine(query_vector, unpack(blob)) if blob else 0.0
            if similarity >= min_similarity:
                results["episodic"].append({"id": eid, "ts": ts, "kind": kind,
                                            "content": json.loads(content), "similarity": round(similarity, 3)})
        for key, value, confidence, blob in facts:
            similarity = cosine(query_vector, unpack(blob)) if blob else 0.0
            if similarity >= min_similarity:
                results["semantic"].append({"key": key, "value": json.loads(value),
                                            "confidence": confidence, "similarity": round(similarity, 3)})
        for name, pattern, blob in patterns:
            similarity = cosine(query_vector, unpack(blob)) if blob else 0.0
            if similarity >= min_similarity:
                results["procedural"].append({"name": name, "pattern": json.loads(pattern),
                                              "similarity": round(similarity, 3)})
        for layer in results.values():
            layer.sort(key=lambda entry: entry["similarity"], reverse=True)
            del layer[limit:]
        return results

    # ── consolidation ───────────────────────────────────────────────────
    def consolidate(self, episodic_ttl_days: int = 30, duplicate_threshold: float = 0.92) -> dict:
        """Archive decayed episodes and merge near-duplicate semantic facts.

        Memory is curated, not just accumulated: low-importance episodes older
        than the TTL are dropped, and facts whose vectors are near-identical
        keep only the higher-confidence copy."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=episodic_ttl_days)).isoformat()
        with self.lock:
            self.c.execute("DELETE FROM episodic WHERE importance < ? AND ts < ?", (0.4, cutoff))
            facts = self.c.execute("SELECT key,confidence,embedding FROM semantic").fetchall()
            dropped = set()
            for i, (key_a, conf_a, blob_a) in enumerate(facts):
                if key_a in dropped or blob_a is None:
                    continue
                for key_b, conf_b, blob_b in facts[i + 1:]:
                    if key_b in dropped or blob_b is None:
                        continue
                    if cosine(unpack(blob_a), unpack(blob_b)) >= duplicate_threshold:
                        dropped.add(key_b if conf_b <= conf_a else key_a)
            for key in dropped:
                self.c.execute("DELETE FROM semantic WHERE key=?", (key,))
            self.c.commit()
        return {"merged_facts": len(dropped)}

    def status(self) -> dict:
        with self.lock:
            counts = {
                table: self.c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("episodic", "semantic", "procedural", "failures", "working")
            }
        return counts

    def close(self) -> None:
        with self.lock:
            self.c.close()
