"""Security hygiene: secret redaction and tool-output injection defence.

Redaction finds and masks credentials before anything enters logs or memory.
The sanitizer treats every tool output and file read as untrusted input:
prompt-injection directives are demoted to quoted data, and shell
metacharacters in filenames are rejected before they reach a shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered (pattern, kind) pairs. Kind feeds the exposure report; the mask
# keeps a short prefix so leaks remain identifiable without being usable.
SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:sk|rk)-[A-Za-z0-9_-]{16,}\b", "openai-style-key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "aws-access-key"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", "github-token"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "slack-token"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", "jwt"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "private-key"),
    (r"\b(?:password|passwd|pwd|secret|api[_-]?key|token)\s*[:=]\s*\S+",
     "credential-assignment"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email"),
    (r"\b(?:\d[ -]?){13,19}\b", "possible-card-number"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "possible-aadhaar"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "pan-number"),
]

INJECTION_MARKERS = [
    r"ignore (?:all )?(?:previous|prior|above) instructions",
    r"disregard (?:all )?(?:previous|prior|above)",
    r"you are now",
    r"new instructions?:",
    r"system prompt\s*:",
    r"</?\s*(?:system|assistant|tool)\s*>",
    r"exfiltrate|send this file to|curl\s+http[^\s]*\s*[-|].*\.env",
    r"rm\s+-rf\s+/",
]

_SHELL_META = re.compile(r"[;&|`$><\n\r]|(\.\./)+")


@dataclass
class ExposureReport:
    kinds_found: list[str] = field(default_factory=list)
    redactions: int = 0

    @property
    def clean(self) -> bool:
        return self.redactions == 0


class Redactor:
    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None):
        compiled = [(re.compile(pattern, re.IGNORECASE), kind)
                    for pattern, kind in (SECRET_PATTERNS + (extra_patterns or []))]
        self.patterns = compiled

    def scan(self, text: str) -> ExposureReport:
        report = ExposureReport()
        for pattern, kind in self.patterns:
            matches = pattern.findall(text)
            if matches:
                report.kinds_found.append(kind)
                report.redactions += len(matches)
        return report

    def redact(self, text: str) -> tuple[str, ExposureReport]:
        report = ExposureReport()
        for pattern, kind in self.patterns:

            def _mask(match: re.Match) -> str:
                report.redactions += 1
                if kind not in report.kinds_found:
                    report.kinds_found.append(kind)
                matched = match.group(0)
                keep = matched[:4] if len(matched) > 8 else ""
                return f"[REDACTED:{kind}:{keep}…]"

            text = pattern.sub(_mask, text)
        return text, report


class InjectionSanitizer:
    """Neutralise prompt-injection payloads inside tool output or file content."""

    def __init__(self):
        self.markers = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_MARKERS]

    def risk(self, text: str) -> int:
        return sum(1 for marker in self.markers if marker.search(text))

    def sanitize(self, text: str) -> str:
        """Wrap the content as inert data; demote any embedded directive lines."""
        if not self.risk(text):
            return text
        flagged = []
        for line in text.splitlines():
            if any(marker.search(line) for marker in self.markers):
                flagged.append(f"[UNTRUSTED-DATA, possible injection] {line.strip()}")
            else:
                flagged.append(line)
        return "\n".join(flagged)

    @staticmethod
    def safe_filename(name: str) -> str:
        candidate = str(name).strip().strip('"\'')
        if not candidate or candidate in {".", ".."}:
            raise ValueError(f"unsafe filename: {name!r}")
        if _SHELL_META.search(candidate):
            raise ValueError(f"filename contains shell metacharacters: {name!r}")
        return candidate


def safe_command_args(args: list[str]) -> list[str]:
    """Reject arguments that would break out of quoting or chain commands."""
    sanitized = []
    for arg in args:
        if _SHELL_META.search(arg) and not arg.startswith("-"):
            raise ValueError(f"argument contains shell metacharacters: {arg!r}")
        sanitized.append(arg)
    return sanitized
