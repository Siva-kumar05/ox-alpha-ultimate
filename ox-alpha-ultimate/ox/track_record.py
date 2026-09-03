"""Supervised real-venue track record: journal + aggregate live-test sessions.

The agent itself is proven offline (tests, contract suites) but a real-venue
track record can only accumulate from supervised sessions on a real Dhan
connection.  ``ox.live_test`` appends one JSON line per session to
``track_record/sessions.jsonl`` (read-only endpoints + optional paper-ledger
session fed by live quotes).  This module reads and aggregates that journal so
an operator can see, at a glance, how many supervised sessions passed, over
what window, and with how many failures - the honest pre-live track record.

Everything here is pure and offline-testable; nothing touches a broker.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_JOURNAL = Path("track_record") / "sessions.jsonl"


def default_path() -> Path:
    return DEFAULT_JOURNAL


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read journal lines, skipping blank and malformed entries.

    A corrupted line must never hide the rest of the record, so each line is
    parsed independently.
    """
    records: list[dict[str, Any]] = []
    try:
        handle = open(path, encoding="utf-8")
    except OSError:
        return records
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_record(record: dict[str, Any], path: str | Path | None = None) -> bool:
    """Append one session record as a JSON line.  Returns False on failure."""
    try:
        target = Path(path) if path else default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return True
    except OSError:
        return False


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate session records into a compact summary dict."""
    if not records:
        return {
            "sessions": 0, "passed": 0, "failed": 0,
            "first_ts": None, "last_ts": None,
            "prime_seconds_total": 0.0, "prime_fills_total": 0,
        }
    passed = [r for r in records if r.get("ok")]
    prime_seconds = 0.0
    prime_fills = 0
    for record in records:
        prime = record.get("prime")
        if isinstance(prime, dict):
            prime_seconds += float(prime.get("seconds", 0.0) or 0.0)
            prime_fills += int(prime.get("fills", 0) or 0)
    timestamps = sorted(ts for ts in (r.get("ts") for r in records) if isinstance(ts, str))
    return {
        "sessions": len(records),
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "first_ts": timestamps[0] if timestamps else None,
        "last_ts": timestamps[-1] if timestamps else None,
        "prime_seconds_total": round(prime_seconds, 1),
        "prime_fills_total": prime_fills,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else default_path()
    summary = summarize(load_records(path))
    if not summary["sessions"]:
        print(f"track-record: no sessions journaled yet at {path}")
        print("Run supervised sessions first:  python run.py live-test <seconds>")
        return 0
    print(f"track record: {path}")
    print(f"  sessions:            {summary['sessions']}")
    print(f"  clean passes:        {summary['passed']}  ({summary['passed'] / summary['sessions']:.0%})")
    print(f"  sessions with issues:{summary['failed']}")
    print(f"  window:              {summary['first_ts']} .. {summary['last_ts']}")
    print(f"  live-quote paper time: {summary['prime_seconds_total']}s, "
          f"fills={summary['prime_fills_total']}")
    return 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
