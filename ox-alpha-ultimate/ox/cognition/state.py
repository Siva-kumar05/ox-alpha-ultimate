"""Environmental state: a persistent mental model of the workspace.

The agent indexes files (hash + mtime), derives a lightweight import graph,
and diffs the world against its last observation — so it *knows* when the
user changed something between turns instead of trusting a stale in-memory
copy. This is state about the environment, independent of chat history.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".venv", "venv",
                 ".ox-alpha", ".streamlit", "build", "dist", ".freebuff"}
_IMPORT = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)


class WorkspaceState:
    def __init__(self, root: str | Path, store: str | Path | None = None):
        self.root = Path(root).resolve()
        self.store = Path(store) if store else self.root / ".ox-alpha" / "state" / "workspace.json"

    # ── indexing ────────────────────────────────────────────────────────
    def scan(self) -> dict:
        index: dict[str, dict] = {}
        for path in self._iter_files():
            rel = path.relative_to(self.root).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            index[rel] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mtime": path.stat().st_mtime,
            }
        return index

    def persist(self, index: dict | None = None) -> dict:
        index = index if index is not None else self.scan()
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({"ts": time.time(), "index": index}), encoding="utf-8")
        return index

    def load(self) -> dict | None:
        if not self.store.exists():
            return None
        payload = json.loads(self.store.read_text(encoding="utf-8"))
        return payload.get("index")

    # ── change detection ────────────────────────────────────────────────
    def diff(self) -> dict:
        """Compare disk to the last persisted observation."""
        previous = self.load() or {}
        current = self.scan()
        return {
            "added": sorted(set(current) - set(previous)),
            "removed": sorted(set(previous) - set(current)),
            "modified": sorted(
                rel for rel in set(current) & set(previous)
                if current[rel]["sha256"] != previous[rel]["sha256"]
            ),
        }

    def external_changes(self) -> dict:
        diff = self.diff()
        return {kind: files for kind, files in diff.items() if files}

    def assert_fresh(self, paths: list[str | Path]) -> list[str]:
        """Return paths whose on-disk hash no longer matches the index."""
        index = self.load() or {}
        stale = []
        for raw in paths:
            path = Path(raw)
            rel = path.relative_to(self.root).as_posix() if not path.is_absolute() else str(path)
            if rel in index:
                data = (self.root / rel).read_bytes() if (self.root / rel).exists() else b""
                if hashlib.sha256(data).hexdigest() != index[rel]["sha256"]:
                    stale.append(rel)
        return stale

    # ── structural model ────────────────────────────────────────────────
    def import_graph(self) -> dict[str, list[str]]:
        """Local-module import edges derived from Python sources."""
        edges: dict[str, list[str]] = {}
        local_roots = {p.stem for p in self._iter_files() if p.suffix == ".py"}
        for path in self._iter_files():
            if path.suffix != ".py":
                continue
            rel = path.relative_to(self.root).as_posix()
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            targets = set()
            for match in _IMPORT.findall(source):
                root_module = match.split(".")[0]
                if root_module in local_roots or match.startswith(("ox", "app_pages")):
                    targets.add(root_module if root_module in local_roots else match)
            edges[rel] = sorted(targets - {path.stem})
        return edges

    def importers_of(self, module: str) -> list[str]:
        graph = self.import_graph()
        name = module.replace("\\", "/")
        return sorted(src for src, targets in graph.items()
                      if module in targets or name in targets or any(t.startswith(module) for t in targets))

    def blast_radius(self, module: str) -> list[str]:
        """Transitive dependents: what breaks if this module changes."""
        graph = self.import_graph()
        seen: set[str] = set()
        frontier = [module]
        while frontier:
            current = frontier.pop()
            for src, targets in graph.items():
                if src not in seen and (current in targets or any(t.startswith(current) for t in targets)):
                    seen.add(src)
                    frontier.append(src)
        return sorted(seen - {module})

    def summary(self) -> dict:
        index = self.scan()
        by_ext: dict[str, int] = {}
        for rel in index:
            ext = Path(rel).suffix or "<none>"
            by_ext[ext] = by_ext.get(ext, 0) + 1
        return {"files": len(index), "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
                "total_bytes": sum(entry["size"] for entry in index.values())}

    def _iter_files(self):
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            yield path
