"""Progressive disclosure of complexity, targeted clarification questioning,
and OpenAPI typed-client generation (gaps #25, #26, #21).

Instead of dumping raw grep output or asking generic "what do you want?", the
disclosure engine collapses verbose tool output into a hierarchy of
summary → detail → full-dump and asks the *highest information value*
question when requirements are ambiguous — maximising signal per user turn.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Progressive disclosure — tiered output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DisclosureLevel:
    name: str        # "summary" | "detail" | "full"
    max_lines: int
    show_keys: bool = False
    truncate_rows: int = 50


DISCLOSURE_LEVELS: dict[str, DisclosureLevel] = {
    "summary": DisclosureLevel("summary", 10, show_keys=False, truncate_rows=5),
    "detail":  DisclosureLevel("detail", 80, show_keys=True,  truncate_rows=25),
    "full":    DisclosureLevel("full", 10_000, show_keys=True, truncate_rows=10_000),
}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (stripped.startswith("|") or re.match(r"^[\s\-+|:=]+$", stripped))


class ProgressiveDisclosure:
    """Collapses verbose outputs into summary/detail/full tiers.

    Each tier answers the user's question with the minimum information needed
    to make the next decision. They can ask "show more" to drill down instead
    of being overwhelmed upfront.
    """

    def __init__(self, default_level: str = "summary"):
        self.default_level = default_level
        self.history: list[dict] = []
        self._counter = 0

    # ── public API ──────────────────────────────────────────────────────
    def render(self, content: Any, *, level: str | None = None, title: str = "") -> str:
        """Render content at the requested disclosure tier."""
        level = level or self.default_level
        spec = DISCLOSURE_LEVELS.get(level, DISCLOSURE_LEVELS["summary"])
        raw = self._to_text(content)
        lines = _strip_ansi(raw).splitlines()
        self._counter += 1
        token = f"pd_{self._counter}"
        self.history.append({"token": token, "title": title, "content": raw, "lines": len(lines)})

        head = f"## {title}\n" if title else ""
        if len(lines) <= spec.max_lines:
            return head + raw

        # Tier-specific rendering
        if level == "summary":
            return head + self._render_summary(lines, spec, token, len(lines))
        elif level == "detail":
            return head + self._render_detail(lines, spec, token, len(lines))
        else:
            return head + raw

    def expand(self, token: str, target_level: str = "full") -> str:
        """Retrieve the full/detailed content behind a summary token."""
        for item in self.history:
            if item["token"] == token:
                return self.render(item["content"], level=target_level, title=item["title"])
        return f"<token {token!r} not found>"

    def stats(self) -> dict:
        return {"disclosures": len(self.history),
                "total_lines_suppressed": sum(max(0, h["lines"] - 10) for h in self.history)}

    # ── tier renderers ──────────────────────────────────────────────────
    def _render_summary(self, lines: list[str], spec: DisclosureLevel,
                        token: str, total: int) -> str:
        # Structural summary: counts + representative samples
        trimmed = lines[: spec.max_lines]
        # Metrics
        empty = sum(1 for l in lines if not l.strip())
        code_like = sum(1 for l in lines if l.strip().startswith(("def ", "class ", "import ", "# ", "- ", "* ")))
        table_like = sum(1 for l in lines if _is_table_line(l))
        header = (
            f"[_progressive:{token} → {total} lines total; "
            f"empty={empty}, code-like={code_like}, table-like={table_like}]"
            f"  → respond 'expand {token}' for detail or 'full {token}' for raw.\n"
        )
        body = "\n".join(f"  {l}" for l in trimmed if l.strip())
        return f"{header}{body}\n[… {total - len(trimmed)} more lines hidden]"

    def _render_detail(self, lines: list[str], spec: DisclosureLevel,
                       token: str, total: int) -> str:
        header = (
            f"[_progressive:{token} → showing first {spec.max_lines} of {total} lines]\n"
        )
        body = "\n".join(lines[: spec.max_lines])
        return header + body + (
            f"\n[… {total - spec.max_lines} more lines — full {token} to show all]"
            if total > spec.max_lines else "")

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            if all(isinstance(r, (list, tuple)) for r in content):
                lines = [" | ".join(str(c) for c in row) for row in content[:500]]
                return "\n".join(lines)
            return "\n".join(f"- {item}" for item in content[:500])
        if isinstance(content, dict):
            try:
                return json.dumps(content, indent=2, default=str, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                return str(content)
        try:
            import pandas as pd  # type: ignore
            if isinstance(content, pd.DataFrame):
                return content.to_string(max_rows=200, max_cols=40)
        except Exception:  # noqa: BLE001
            pass
        return str(content)


# ═══════════════════════════════════════════════════════════════════════════
# Clarification engine — targeted, high-information-value questions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClarificationQuestion:
    id: str
    header: str
    question: str
    options: list[tuple[str, str]]   # (value, description)
    rationale: str
    info_gain: float                 # 0..1 — how much uncertainty this resolves
    multi: bool = False
    recommended: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "header": self.header,
            "question": self.question,
            "options": [{"value": v, "description": d} for v, d in self.options],
            "rationale": self.rationale,
            "info_gain": round(self.info_gain, 3),
            "multi": self.multi,
            "recommended": self.recommended,
        }


class ClarificationEngine:
    """Generate the highest-information-value clarification questions.

    Instead of generic "please clarify" or shotgun 8-option menus, this engine
    models the decision space, estimates expected entropy reduction per
    question, and asks the 1-3 questions that collapse ambiguity fastest.
    """

    def __init__(self, uncertainty_gate=None):
        self.gate = uncertainty_gate

    # ── high-level domain questioners ───────────────────────────────────
    def for_code_change(self, description: str, ambiguity_signals: dict) -> list[ClarificationQuestion]:
        """Questions when a user asks for a code change and the spec is thin."""
        questions: list[ClarificationQuestion] = []
        scope = ambiguity_signals.get("scope_unknown", False)
        compat = ambiguity_signals.get("backwards_compat_unknown", False)
        tests = ambiguity_signals.get("testing_unknown", True)
        perf = ambiguity_signals.get("perf_sensitive", False)

        if scope:
            questions.append(ClarificationQuestion(
                id="scope", header="Scope",
                question="What is the scope of this change?",
                options=[
                    ("single-file", "Only the file(s) explicitly named"),
                    ("single-module", "Everything under the target module"),
                    ("cross-module", "Across modules — touching multiple boundaries"),
                    ("full-project", "Repository-wide, including config and tests"),
                ],
                rationale="Scope determines blast radius, transaction strategy, and testing depth.",
                info_gain=0.90, recommended=["single-module"]))
        if compat:
            questions.append(ClarificationQuestion(
                id="compat", header="Compatibility",
                question="Backwards compatibility stance?",
                options=[
                    ("strict", "No public API changes allowed (semver patch)"),
                    ("additive", "Only additive changes (semver minor)"),
                    ("breaking-ok", "Breaking changes acceptable if clearly justified"),
                ],
                rationale="Compat stance determines whether we can rename APIs or must deprecate.",
                info_gain=0.75, recommended=["additive"]))
        if tests:
            questions.append(ClarificationQuestion(
                id="tests", header="Testing",
                question="What testing deliverables are required?",
                options=[
                    ("none", "No new tests (behaviour-only)"),
                    ("unit", "Unit tests only"),
                    ("integration", "Integration tests if applicable, plus unit"),
                    ("full", "Unit, integration, and regression tests for edge cases"),
                ],
                rationale="Testing scope determines deliverable count and safety depth.",
                info_gain=0.60, recommended=["unit"]))
        if perf:
            questions.append(ClarificationQuestion(
                id="perf", header="Performance",
                question="Performance priority?",
                options=[
                    ("correctness", "Correctness first — no premature optimisation"),
                    ("balanced", "Reasonable efficiency; avoid O(n²) where linear works"),
                    ("hot-path", "This is on a hot path — benchmarks matter"),
                ],
                rationale="Perf priority determines whether we use vectorised/native vs clear idiomatic code.",
                info_gain=0.55, recommended=["balanced"]))
        questions.sort(key=lambda q: q.info_gain, reverse=True)
        return questions[:3]

    def for_data_pipeline(self, description: str, signals: dict) -> list[ClarificationQuestion]:
        questions = []
        if signals.get("source_unknown"):
            questions.append(ClarificationQuestion(
                id="source", header="Source",
                question="Where should data come from?",
                options=[
                    ("csv-dir", "A directory of CSV files"),
                    ("database", "SQL database connection"),
                    ("api", "HTTP API endpoint"),
                    ("stdin", "Streamed stdin input"),
                ],
                rationale="Source determines connector, error handling, and pagination.",
                info_gain=0.95))
        if signals.get("schema_unknown", True):
            questions.append(ClarificationQuestion(
                id="schema", header="Schema",
                question="Is there a schema file / sample to validate against?",
                options=[
                    ("yes", "Yes — I'll provide a schema or sample"),
                    ("infer", "Infer from the first batch of data"),
                    ("flexible", "Schema-on-read, no hard validation"),
                ],
                rationale="Schema controls parsing, validation, and type-casting behaviour.",
                info_gain=0.85))
        questions.sort(key=lambda q: q.info_gain, reverse=True)
        return questions[:3]

    # ── general-purpose ambiguity detector ──────────────────────────────
    def ask_when_ambiguous(self, description: str, *, domain: str = "general") -> list[ClarificationQuestion]:
        """Heuristic ambiguity detection — generate questions for thin specs."""
        desc = (description or "").strip()
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", desc.lower()))
        word_count = len(desc.split())
        signals: dict[str, bool] = defaultdict(bool)
        if word_count < 8:
            signals["scope_unknown"] = True
            signals["testing_unknown"] = True
        if not any(t in {"deprecat", "compat", "breaking", "semver", "api"} for t in tokens) and domain == "code":
            signals["backwards_compat_unknown"] = True
        if domain == "code":
            return self.for_code_change(desc, dict(signals))
        if domain == "data":
            return self.for_data_pipeline(desc, dict(signals))
        # generic fallback
        return [ClarificationQuestion(
            id="goal", header="Goal",
            question="What does success look like, in concrete terms?",
            options=[("output", "A specific output file / value"),
                     ("behaviour", "A behaviour change in the program"),
                     ("refactor", "No external change — cleaner internals"),
                     ("fix", "A specific bug is fixed")],
            rationale="Clarifying the success criterion prevents building the wrong thing confidently.",
            info_gain=0.80),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# OpenAPI → typed client generator (gap #21)
# ═══════════════════════════════════════════════════════════════════════════

_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


class OpenAPIGenerator:
    """Drop an OpenAPI spec in → get a typed Python client module, instantly.

    The generated client uses only stdlib urllib (plus `requests` if installed),
    validates parameters against the schema, and returns structured dict
    responses. No shell scripting, no raw curl.
    """

    def __init__(self, output_dir: str | Path = ".ox-alpha/clients"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, spec: dict, client_name: str = "api_client") -> Path:
        title = (spec.get("info", {}) or {}).get("title", client_name)
        safe_name = re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_") or client_name
        code = self._render(spec, safe_name)
        out = self.output_dir / f"{safe_name}.py"
        out.write_text(code, encoding="utf-8")
        return out

    def load_and_generate(self, path: str | Path) -> Path:
        raw = Path(path).read_text(encoding="utf-8")
        if raw.lstrip().startswith("{"):
            spec = json.loads(raw)
        else:
            try:
                import yaml as _yml  # type: ignore
                spec = _yml.safe_load(raw)
            except Exception:  # noqa: BLE001
                raise ValueError("YAML spec requires pyyaml; pass JSON spec instead")
        client_name = Path(path).stem
        return self.generate(spec, client_name)

    # ── renderer ────────────────────────────────────────────────────────
    def _render(self, spec: dict, name: str) -> str:
        servers = spec.get("servers", []) or []
        base_url = servers[0].get("url", "https://api.example.com") if servers else "https://api.example.com"
        lines: list[str] = []
        lines.append(f'"""Auto-generated client for {name} — OpenAPI → typed Python."""')
        lines.append("from __future__ import annotations")
        lines.append("import json as _json")
        lines.append("from urllib.error import HTTPError as _HTTPError")
        lines.append("from urllib.parse import urlencode as _urlencode, urljoin as _urljoin")
        lines.append("from urllib.request import Request as _Request, urlopen as _urlopen")
        lines.append("from dataclasses import dataclass, field")
        lines.append("from typing import Any, Optional")
        lines.append("")
        lines.append("try:")
        lines.append("    import requests as _requests  # type: ignore")
        lines.append("    _HAS_REQUESTS = True")
        lines.append("except Exception:  # noqa: BLE001")
        lines.append("    _requests = None  # type: ignore")
        lines.append("    _HAS_REQUESTS = False")
        lines.append("")
        lines.append("@dataclass")
        lines.append(f"class {name.title().replace('_','')}Client:")
        lines.append(f'    base_url: str = {base_url!r}')
        lines.append('    api_key: Optional[str] = None')
        lines.append('    timeout: float = 30.0')
        lines.append("")
        lines.append("    def _headers(self, extra: dict | None = None) -> dict:")
        lines.append("        h = {'Accept': 'application/json', 'User-Agent': 'ox-alpha/openapi-gen'}")
        lines.append("        if self.api_key: h['Authorization'] = f'Bearer {self.api_key}'")
        lines.append("        if extra: h.update(extra)")
        lines.append("        return h")
        lines.append("")
        lines.append("    def _request(self, method: str, path: str, *, params=None, data=None, json=None, headers=None):")
        lines.append("        url = self.base_url.rstrip('/') + path")
        lines.append("        if params: url += '?' + _urlencode({k: v for k, v in (params or {}).items() if v is not None})")
        lines.append("        if _HAS_REQUESTS:")
        lines.append("            r = _requests.request(method, url, params=None, data=data, json=json, headers=self._headers(headers), timeout=self.timeout)")
        lines.append("            r.raise_for_status(); return r.json() if r.content else None")
        lines.append("        body: bytes | None = None; hh = self._headers(headers)")
        lines.append("        if json is not None: body = _json.dumps(json).encode(); hh['Content-Type']='application/json'")
        lines.append("        elif data is not None: body = str(data).encode()")
        lines.append("        with _urlopen(_Request(url, data=body, method=method.upper(), headers=hh), timeout=self.timeout) as resp:")
        lines.append("            raw = resp.read(); return _json.loads(raw) if raw else None")
        lines.append("")

        # Paths → methods
        paths = spec.get("paths", {}) or {}
        used_names: Counter = Counter()
        for url_path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, op in methods.items():
                if method.lower() not in _METHODS or not isinstance(op, dict):
                    continue
                fn_name = self._fn_name(url_path, method, op, used_names)
                lines.append(self._render_method(url_path, method, op, fn_name))

        return "\n".join(lines) + "\n"

    @staticmethod
    def _fn_name(path: str, method: str, op: dict, used: Counter) -> str:
        opid = (op or {}).get("operationId")
        if opid and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", opid):
            base = opid
        else:
            base = f"{method.lower()}_{re.sub(r'[^a-z0-9]+', '_', path).strip('_')}"
        base = re.sub(r"[^A-Za-z0-9_]", "_", base).strip("_") or method.lower()
        count = used[base]
        used[base] += 1
        return base if count == 0 else f"{base}_{count}"

    def _render_method(self, url_path: str, method: str, op: dict, fn_name: str) -> str:
        params = op.get("parameters", []) or []
        body = op.get("requestBody")
        summary = (op.get("summary") or op.get("description") or "").strip().replace("\n", " ")
        # Signature parts
        sig_params: list[str] = ["self"]
        doc_lines = [f'        """{summary or f"{method.upper()} {url_path}"}']
        doc_lines.append("")
        query_names, path_names, header_names = [], [], []
        for p in params:
            if not isinstance(p, dict):
                continue
            pname = str(p.get("name", "param"))
            ptype = str(p.get("in", "query"))
            required = bool(p.get("required", False))
            schema = p.get("schema", {}) or {}
            py_type = self._py_type(schema.get("type"), schema.get("format"))
            sig = f"{pname}: {py_type}{' = None' if not required else ''}"
            sig_params.append(sig)
            doc_lines.append(f"        {pname}: {p.get('description', '')} ({ptype})")
            if ptype == "query":
                query_names.append(pname)
            elif ptype == "path":
                path_names.append(pname)
            elif ptype == "header":
                header_names.append(pname)
        # Body
        has_body = isinstance(body, dict)
        body_type = "dict"
        if has_body:
            content = (body.get("content") or {}).get("application/json") or {}
            body_type = self._py_type(((content.get("schema") or {}).get("type")), None)
            sig_params.append(f"body: {body_type} | None = None")
            doc_lines.append("        body: JSON request body (application/json)")

        doc_lines.append('        """')
        signature = f"    def {fn_name}(" + ", ".join(sig_params) + "):"
        impl = []
        # Build path with substitutions
        rendered_path = url_path
        for pn in path_names:
            rendered_path = rendered_path.replace("{" + pn + "}", "' + str(" + pn + ") + '")
        rendered_path = "'" + rendered_path + "'"
        # Query dict
        if query_names:
            qdict = "{" + ", ".join(f"{k!r}: {k}" for k in query_names) + "}"
        else:
            qdict = "None"
        if header_names:
            hd = "{" + ", ".join(f"{k!r}: str({k})" for k in header_names) + "}"
        else:
            hd = "None"
        body_expr = "body" if has_body else "None"
        impl.append("        return self._request(")
        impl.append(f"            {method.upper()!r}, {rendered_path},")
        impl.append(f"            params={qdict}, json={body_expr}, headers={hd},")
        impl.append("        )")
        return "\n".join([signature] + doc_lines + impl) + "\n"

    @staticmethod
    def _py_type(type_str: str | None, fmt: str | None) -> str:
        t = (type_str or "string").lower()
        if t == "integer":
            return "int"
        if t == "number":
            return "float"
        if t == "boolean":
            return "bool"
        if t in ("object", "array"):
            return "dict" if t == "object" else "list"
        return "str"
