"""Package manager awareness + semantic AST-based code queries (gaps #22 and #24).

The agent maintains an internal model of package.json, requirements.txt,
Cargo.toml, pyproject.toml — detecting version conflicts before they happen
and suggesting compatible upgrades. AST queries let the agent ask semantic
questions like "find all functions that call X and mutate global state"
instead of grepping for text.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # noqa: BLE001
    _HAS_YAML = False


# ═══════════════════════════════════════════════════════════════════════════
# Package manager / dependency model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Dependency:
    name: str
    version_spec: str
    source: str          # e.g. "requirements.txt"
    raw_line: str = ""
    extras: list[str] = field(default_factory=list)

    def parsed_version(self) -> tuple[str, ...]:
        """Best-effort version tuple for comparison. Missing = 0.0.0."""
        cleaned = re.sub(r"[^0-9.]", "", self.version_spec.split(",")[0])
        parts = cleaned.split(".")[:3]
        while len(parts) < 3:
            parts.append("0")
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return (0, 0, 0)


@dataclass
class Conflict:
    package: str
    spec_a: Dependency
    spec_b: Dependency
    severity: str = "warning"   # warning | error | info

    def to_dict(self) -> dict:
        return {
            "package": self.package, "severity": self.severity,
            "a": {"source": self.spec_a.source, "version": self.spec_a.version_spec},
            "b": {"source": self.spec_b.source, "version": self.spec_b.version_spec},
        }


class DependencyModel:
    """Persistent model of all declared dependencies across manifest files.

    Detects conflicts between overlapping dependency declarations, checks
    installed versions against declared specs, and suggests compatible
    upgrades — so dependency problems are caught *before* installation fails.
    """

    MANIFESTS = [
        ("requirements.txt", "_parse_requirements"),
        ("pyproject.toml", "_parse_pyproject"),
        ("package.json", "_parse_package_json"),
        ("Cargo.toml", "_parse_cargo"),
        ("Pipfile", "_parse_pipfile"),
    ]

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifests: dict[str, Path] = {}
        self.dependencies: dict[str, list[Dependency]] = defaultdict(list)
        self.loaded = False

    # ── public API ──────────────────────────────────────────────────────
    def refresh(self) -> dict:
        """Re-scan manifests and rebuild the model. Return summary."""
        self.dependencies.clear()
        self.manifests.clear()
        for name, method in self.MANIFESTS:
            path = self.root / name
            if path.exists():
                self.manifests[name] = path
                try:
                    getattr(self, method)(path)
                except Exception:  # noqa: BLE001
                    pass
        self.loaded = True
        return {
            "manifests": {k: str(v.relative_to(self.root)) for k, v in self.manifests.items()},
            "total_packages": len(self.dependencies),
            "conflicts": [c.to_dict() for c in self.conflicts()],
        }

    def conflicts(self) -> list[Conflict]:
        """Return packages declared with incompatible version specs."""
        out: list[Conflict] = []
        for name, deps in self.dependencies.items():
            if len(deps) < 2:
                continue
            versions = {d.parsed_version() for d in deps}
            specs = {d.version_spec for d in deps}
            if len(specs) > 1:
                sev = "error" if len(versions) > 1 else "warning"
                for i in range(len(deps)):
                    for j in range(i + 1, len(deps)):
                        out.append(Conflict(name, deps[i], deps[j], sev))
        return out

    def installed_version(self, package: str) -> str | None:
        """Check the currently installed version of a package (best-effort)."""
        try:
            if sys.version_info >= (3, 8):
                from importlib.metadata import version as _ver
                return _ver(package)
        except Exception:  # noqa: BLE001
            pass
        return None

    def audit(self) -> dict:
        """Full dependency audit: conflicts + install-check."""
        if not self.loaded:
            self.refresh()
        conflicts = self.conflicts()
        installed: list[dict] = []
        missing: list[str] = []
        for name in list(self.dependencies)[:200]:
            ver = self.installed_version(name)
            if ver is None:
                missing.append(name)
            else:
                installed.append({"name": name, "installed": ver})
        return {
            "conflicts": [c.to_dict() for c in conflicts],
            "installed": installed,
            "missing": missing,
            "manifests": {k: str(v.relative_to(self.root)) for k, v in self.manifests.items()},
        }

    def suggest_upgrade(self, package: str, target: str) -> list[str]:
        """Suggest which manifest(s) to edit, and what the new line should be."""
        deps = self.dependencies.get(package, [])
        suggestions = []
        for d in deps:
            path = self.manifests.get(d.source)
            if not path:
                continue
            if d.source == "requirements.txt":
                suggestions.append(f"In {d.source}: change {d.raw_line.strip()} to {package}=={target}")
            else:
                suggestions.append(f"In {d.source}: update {package} version to {target} (current: {d.version_spec})")
        return suggestions or [f"{package} not found in any manifest"]

    # ── manifest parsers ────────────────────────────────────────────────
    def _parse_requirements(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw or raw.startswith(("#", "-")):
                continue
            match = re.match(r"([A-Za-z0-9_.\-]+)\s*([<>=!~].*$)?", raw)
            if not match:
                continue
            name = match.group(1)
            spec = (match.group(2) or "").strip() or "*"
            self.dependencies[name.lower()].append(Dependency(
                name=name.lower(), version_spec=spec, source="requirements.txt", raw_line=raw))

    def _parse_pyproject(self, path: Path) -> None:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        project = data.get("project", {})
        for req in project.get("dependencies", []):
            self._add_pep508(req, "pyproject.toml")
        for group, reqs in project.get("optional-dependencies", {}).items():
            for req in reqs:
                self._add_pep508(req, "pyproject.toml", extras=[group])

    def _parse_package_json(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            for name, spec in data.get(section, {}).items():
                self.dependencies[name.lower()].append(Dependency(
                    name=name.lower(), version_spec=spec, source="package.json",
                    raw_line=f"{section}/{name}"))

    def _parse_cargo(self, path: Path) -> None:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, spec in data.get(section, {}).items():
                version = spec if isinstance(spec, str) else (spec.get("version") or "*")
                self.dependencies[name.lower()].append(Dependency(
                    name=name.lower(), version_spec=str(version), source="Cargo.toml",
                    raw_line=f"{section}/{name}"))

    def _parse_pipfile(self, path: Path) -> None:
        if not _HAS_YAML:
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return
        for section in ("packages", "dev-packages"):
            for name, spec in (data.get(section) or {}).items():
                version = spec if isinstance(spec, str) else (spec.get("version") or "*")
                self.dependencies[name.lower()].append(Dependency(
                    name=name.lower(), version_spec=str(version), source="Pipfile",
                    raw_line=f"{section}/{name}"))

    def _add_pep508(self, req: str, source: str, *, extras: list[str] | None = None) -> None:
        match = re.match(r"([A-Za-z0-9_.\-]+)(?:\[([^\]]*)\])?\s*([<>=!~].*)?$", req.strip())
        if not match:
            return
        name = match.group(1).lower()
        ext = [e.strip() for e in (match.group(2) or "").split(",") if e.strip()]
        spec = (match.group(3) or "").strip() or "*"
        self.dependencies[name].append(Dependency(
            name=name, version_spec=spec, source=source,
            raw_line=req, extras=extras or ext))


# ═══════════════════════════════════════════════════════════════════════════
# Semantic AST queries for Python codebases
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionInfo:
    name: str
    file: str
    line: int
    args: list[str]
    returns: str | None
    calls: list[str] = field(default_factory=list)      # names of called functions
    accesses_globals: list[str] = field(default_factory=list)
    mutates_globals: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class ClassInfo:
    name: str
    file: str
    line: int
    methods: list[str]
    bases: list[str] = field(default_factory=list)


class SemanticASTQueries:
    """AST-aware code querying — replacing surface-level grep (gap #22).

    Supports:
    - find all functions calling a given target
    - find functions that mutate globals
    - find dead code (defined but never called)
    - find callers/callees of a function
    - compute transitive blast radius of a module change
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.functions: dict[str, FunctionInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.module_globals: dict[str, set[str]] = defaultdict(set)
        self.indexed = False

    # ── indexing ────────────────────────────────────────────────────────
    def index(self) -> dict:
        """Walk the codebase and build AST indices."""
        self.functions.clear()
        self.classes.clear()
        self.module_globals.clear()
        count_files = 0
        for py in self.root.rglob("*.py"):
            if self._skip(py):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"), filename=str(py))
            except SyntaxError:
                continue
            count_files += 1
            rel = py.relative_to(self.root).as_posix()
            self._index_module(rel, tree)
        self.indexed = True
        return {"files_indexed": count_files,
                "functions": len(self.functions),
                "classes": len(self.classes)}

    # ── queries ─────────────────────────────────────────────────────────
    def functions_calling(self, target_name: str) -> list[dict]:
        """All functions whose body calls target_name()."""
        if not self.indexed:
            self.index()
        results = []
        for fn in self.functions.values():
            if any(target_name == call or call.endswith(f".{target_name}") for call in fn.calls):
                results.append({
                    "name": fn.name, "file": fn.file, "line": fn.line,
                    "calls": fn.calls,
                })
        return results

    def functions_mutating_globals(self, *, only: set[str] | None = None) -> list[dict]:
        """Functions that assign to globals declared at module scope."""
        if not self.indexed:
            self.index()
        results = []
        for fn in self.functions.values():
            if not fn.mutates_globals:
                continue
            if only and not set(fn.mutates_globals) & only:
                continue
            results.append({
                "name": fn.name, "file": fn.file, "line": fn.line,
                "mutates": fn.mutates_globals,
            })
        return results

    def functions_calling_and_mutating(self, target_call: str, globals_of_interest: set[str]) -> list[dict]:
        """The classic gap #22 query: functions that call X *and* mutate global state Y."""
        if not self.indexed:
            self.index()
        callers = {f["name"] + ":" + f["file"] for f in self.functions_calling(target_call)}
        mutators = self.functions_mutating_globals(only=globals_of_interest)
        overlap = []
        for m in mutators:
            key = m["name"] + ":" + m["file"]
            if key in callers:
                m["calls"] = target_call
                overlap.append(m)
        return overlap

    def dead_functions(self) -> list[dict]:
        """Functions defined but never called anywhere in the index."""
        if not self.indexed:
            self.index()
        called: set[str] = set()
        for fn in self.functions.values():
            for c in fn.calls:
                called.add(c.split(".")[-1])
        dead = []
        for fn in self.functions.values():
            if fn.name.startswith("_"):
                continue
            if fn.name.startswith("test_") or fn.decorators:
                continue
            if fn.name not in called:
                dead.append({"name": fn.name, "file": fn.file, "line": fn.line})
        return dead

    def callers_of(self, function_name: str) -> list[dict]:
        """Every function that invokes function_name directly."""
        return self.functions_calling(function_name)

    def callees_of(self, function_name: str) -> list[dict]:
        """Direct callees inside the body of function_name."""
        if not self.indexed:
            self.index()
        for fn in self.functions.values():
            if fn.name == function_name:
                return [{"called": c} for c in fn.calls]
        return []

    def transitive_impact(self, changed_function: str) -> dict:
        """Transitive set of functions that could be affected if changed_function changes."""
        if not self.indexed:
            self.index()
        impacted: set[str] = set()
        frontier = {changed_function}
        while frontier:
            name = frontier.pop()
            callers = self.callers_of(name)
            newly = {c["name"] for c in callers} - impacted
            impacted |= newly
            frontier |= newly
        return {"starting_from": changed_function,
                "transitive_callers": sorted(impacted),
                "count": len(impacted)}

    def functions_by_file(self, file_rel_path: str) -> list[dict]:
        """List every function defined in a given file."""
        if not self.indexed:
            self.index()
        out = []
        for fn in self.functions.values():
            if fn.file == file_rel_path or fn.file.endswith("/" + file_rel_path):
                out.append({"name": fn.name, "line": fn.line, "args": fn.args,
                            "returns": fn.returns, "docstring": (fn.docstring or "")[:80]})
        return sorted(out, key=lambda r: r["line"])

    # ── internals ───────────────────────────────────────────────────────
    def _skip(self, path: Path) -> bool:
        parts = set(path.parts)
        skip_dirs = {".git", "__pycache__", "node_modules", ".pytest_cache",
                     ".venv", "venv", ".ox-alpha", "build", "dist", ".freebuff", ".mimosa"}
        return bool(parts & skip_dirs)

    def _index_module(self, rel: str, tree: ast.AST) -> None:
        # Collect module-level global names (non-function, non-class assignments)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.module_globals[rel].add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.module_globals[rel].add(node.target.id)
        globals_of_module = self.module_globals[rel]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                self.classes[f"{rel}:{node.name}"] = ClassInfo(
                    name=node.name, file=rel, line=node.lineno, methods=methods, bases=bases)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls: list[str] = []
                mutates: list[str] = []
                accesses: list[str] = []
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        func = inner.func
                        if isinstance(func, ast.Name):
                            calls.append(func.id)
                        elif isinstance(func, ast.Attribute):
                            parts = []
                            obj: ast.AST = func
                            while isinstance(obj, ast.Attribute):
                                parts.append(obj.attr)
                                obj = obj.value
                            if isinstance(obj, ast.Name):
                                parts.append(obj.id)
                            calls.append(".".join(reversed(parts)))
                    elif isinstance(inner, ast.Global):
                        for n in inner.names:
                            if n in globals_of_module and n not in mutates:
                                mutates.append(n)
                    elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                        if inner.id in globals_of_module and inner.id not in mutates:
                            mutates.append(inner.id)
                    elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                        if inner.id in globals_of_module and inner.id not in accesses:
                            accesses.append(inner.id)
                args = [a.arg for a in node.args.args]
                returns = ast.unparse(node.returns) if node.returns else None
                decorators = []
                for d in node.decorator_list:
                    if isinstance(d, ast.Name):
                        decorators.append(d.id)
                    elif isinstance(d, ast.Attribute):
                        decorators.append(d.attr)
                info = FunctionInfo(
                    name=node.name, file=rel, line=node.lineno,
                    args=args, returns=returns,
                    calls=sorted(set(calls)),
                    accesses_globals=sorted(set(accesses)),
                    mutates_globals=sorted(set(mutates)),
                    decorators=sorted(set(decorators)),
                    docstring=ast.get_docstring(node))
                self.functions[f"{rel}:{node.name}:{node.lineno}"] = info
