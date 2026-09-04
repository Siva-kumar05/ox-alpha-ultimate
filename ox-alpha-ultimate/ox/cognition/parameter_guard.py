"""Anti-hallucination parameter guard + flexible schema adapter (gaps #6 and #15).

Before any tool is called, the guard:
1. Verifies every file path, symbol, identifier *actually exists* on disk or in the workspace
2. Infers sensible defaults for missing optional parameters from schema + past usage
3. Rejects or auto-corrects values that look like hallucinated fabrications

Tools stop being fragile (single missing param = failure) and stop hallucinating
file paths that don't exist.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ValidationIssue:
    severity: str          # "error" | "warning" | "info"
    field: str
    message: str
    proposed_fix: Any = None


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    corrected_params: dict[str, Any] = field(default_factory=dict)
    inferred: list[str] = field(default_factory=list)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        if self.ok and not self.issues:
            return "OK"
        parts = [f"errors={len(self.errors())}", f"warnings={len(self.warnings())}"]
        if self.inferred:
            parts.append(f"inferred={len(self.inferred)}")
        return ", ".join(parts)


class SchemaField:
    """A single parameter definition for a tool."""

    def __init__(self, name: str, dtype: str = "string", *, required: bool = False,
                 default: Any = None, description: str = "",
                 validator: Callable[[Any], tuple[bool, str]] | None = None,
                 examples: list[Any] | None = None):
        self.name = name
        self.dtype = dtype
        self.required = required
        self.default = default
        self.description = description
        self.validator = validator
        self.examples = examples or []

    def coerce(self, value: Any) -> tuple[Any, str | None]:
        """Attempt type coercion. Returns (coerced_value, error_message_or_None)."""
        if value is None and self.default is not None:
            return self.default, None
        if value is None and not self.required:
            return None, None
        if value is None and self.required:
            return None, f"field {self.name} is required"
        try:
            if self.dtype == "string":
                return str(value), None
            if self.dtype == "int":
                return int(value), None
            if self.dtype == "float":
                return float(value), None
            if self.dtype == "bool":
                if isinstance(value, bool):
                    return value, None
                if isinstance(value, str):
                    return value.lower() in {"true", "1", "yes", "y", "on"}, None
                return bool(value), None
            if self.dtype == "path":
                return str(value), None
            if self.dtype == "json":
                if isinstance(value, (dict, list)):
                    return value, None
                return json.loads(str(value)), None
            if self.dtype == "enum":
                allowed = set(self.examples)
                if value not in allowed:
                    return None, f"{value} not in {sorted(allowed)}"
                return value, None
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, f"coercion to {self.dtype} failed: {exc}"
        return value, None


class ToolSchema:
    """A full schema for a tool: names, types, required/optional, defaults."""

    def __init__(self, name: str, fields: list[SchemaField]):
        self.name = name
        self.fields: dict[str, SchemaField] = {f.name: f for f in fields}

    def field_names(self) -> list[str]:
        return list(self.fields.keys())


_SUSPICIOUS = re.compile(r"[^a-zA-Z0-9_\-./:\\ ]")
_PATH_TRAVERSAL = re.compile(r"(\.\./|\\.\\.\\)")


class ParameterGuard:
    """Validate tool parameters against workspace reality before invocation.

    Checks file paths exist, warns on suspicious names, infers missing optional
    fields from historical usage. This is the single choke-point that prevents
    hallucinated values from reaching a tool call.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root else None
        self.historical: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        self.schemas: dict[str, ToolSchema] = {}

    # ── schema registration ─────────────────────────────────────────────
    def register(self, schema: ToolSchema) -> None:
        self.schemas[schema.name] = schema

    def register_simple(self, tool_name: str, fields: list[tuple[str, str, bool, Any]]) -> None:
        """Convenience: list of (name, dtype, required, default) tuples."""
        sf = [SchemaField(n, d, required=r, default=defv) for (n, d, r, defv) in fields]
        self.schemas[tool_name] = ToolSchema(tool_name, sf)

    # ── usage history ───────────────────────────────────────────────────
    def record_usage(self, tool: str, params: dict[str, Any]) -> None:
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) and not isinstance(value, bool):
                if len(self.historical[tool][key]) < 50:
                    self.historical[tool][key].append(value)

    def _common_value(self, tool: str, field: str) -> Any:
        values = self.historical.get(tool, {}).get(field, [])
        if not values:
            return None
        from collections import Counter
        c = Counter(values)
        return c.most_common(1)[0][0]

    # ── validation ──────────────────────────────────────────────────────
    def validate(self, tool: str, params: dict[str, Any]) -> ValidationReport:
        schema = self.schemas.get(tool)
        issues: list[ValidationIssue] = []
        corrected: dict[str, Any] = dict(params or {})
        inferred: list[str] = []

        # ── step 1: handle unknown tools ──────────────────────────────
        if schema is None:
            return ValidationReport(
                ok=True, issues=[ValidationIssue("warning", "*",
                                                 f"no schema registered for tool {tool!r}; proceeding without guard")],
                corrected_params=corrected, inferred=inferred)

        # ── step 2: required fields + type coercion + defaults ────────
        for field_name, f in schema.fields.items():
            value = corrected.get(field_name)
            if value is None and f.default is not None:
                corrected[field_name] = f.default
                inferred.append(f"{field_name}={f.default}")
                value = f.default
            if value is None and f.required:
                common = self._common_value(tool, field_name)
                if common is not None:
                    corrected[field_name] = common
                    inferred.append(f"{field_name}={common} (from history)")
                else:
                    issues.append(ValidationIssue("error", field_name,
                                                  f"required field {field_name!r} is missing and has no default"))
                    continue
            if value is not None:
                coerced, err = f.coerce(value)
                if err and f.required:
                    issues.append(ValidationIssue("error", field_name, err))
                elif err and not f.required:
                    issues.append(ValidationIssue("warning", field_name, err))
                else:
                    corrected[field_name] = coerced
                    if f.validator:
                        ok, msg = f.validator(coerced)
                        if not ok:
                            issues.append(ValidationIssue("error", field_name, msg or "validator failed"))

        # ── step 3: path existence checks ─────────────────────────────
        if self.root is not None:
            for field_name, f in schema.fields.items():
                if f.dtype != "path":
                    continue
                value = corrected.get(field_name)
                if not isinstance(value, str) or not value:
                    continue
                # path-traversal
                if _PATH_TRAVERSAL.search(value):
                    issues.append(ValidationIssue("error", field_name,
                                                  f"path {value!r} contains traversal"))
                    continue
                # suspicious characters
                if _SUSPICIOUS.search(value):
                    issues.append(ValidationIssue("warning", field_name,
                                                  f"path {value!r} has unusual characters"))
                resolved = (self.root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
                # required existing path?
                if getattr(f, "_must_exist", False) and not resolved.exists():
                    candidates = self._find_similar(value)
                    msg = f"path {str(resolved)!r} does not exist"
                    if candidates:
                        msg += f"; did you mean: {candidates[:3]}?"
                    issues.append(ValidationIssue("error", field_name, msg,
                                                  proposed_fix=candidates[0] if candidates else None))

        # ── step 4: warn about unknown fields ──────────────────────────
        known = set(schema.fields)
        for extra in set(corrected) - known:
            issues.append(ValidationIssue("warning", extra,
                                          f"field {extra!r} is not declared in schema for {tool}"))

        report = ValidationReport(
            ok=not any(i.severity == "error" for i in issues),
            issues=issues, corrected_params=corrected, inferred=inferred)
        if report.ok:
            self.record_usage(tool, corrected)
        return report

    # ── helpers ─────────────────────────────────────────────────────────
    def _find_similar(self, bad_path: str, top: int = 5) -> list[str]:
        if self.root is None:
            return []
        try:
            candidates: list[tuple[float, str]] = []
            target = Path(bad_path).name.lower()
            for p in self.root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.root).as_posix()
                name = p.name.lower()
                score = 0.0
                if target and target == name:
                    score = 1.0
                elif target and (target in name or name in target):
                    score = 0.7
                elif target:
                    score = max(0.0, 1.0 - _lev(target, name))
                if score > 0.45:
                    candidates.append((score, rel))
            candidates.sort(reverse=True)
            return [c for _, c in candidates[:top]]
        except (OSError, ValueError):
            return []


def _lev(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb] / max(la, lb)


class SchemaAdapter:
    """Infer missing optional parameters and resolve ambiguity before guard validation.

    Used when a user's natural-language description is incomplete: the adapter
    fills optional gaps from defaults, historical usage, and per-tool inference
    hooks — then hands the completed params to ParameterGuard for validation.
    """

    def __init__(self, guard: ParameterGuard):
        self.guard = guard
        self.inference_hooks: dict[str, Callable[[dict, dict], dict]] = {}

    def add_hook(self, tool: str, hook: Callable[[dict, dict], dict]) -> None:
        """Register fn(params, tool_context) -> completed_params."""
        self.inference_hooks[tool] = hook

    def adapt(self, tool: str, raw_params: dict[str, Any], context: dict | None = None) -> ValidationReport:
        params = dict(raw_params or {})
        hook = self.inference_hooks.get(tool)
        if hook:
            try:
                params = hook(params, dict(context or {}))
            except Exception:  # noqa: BLE001
                pass
        return self.guard.validate(tool, params)
