#!/usr/bin/env python
"""Offline pattern analysis of the latest training run (read-only).

Dumps, from the agent DB's real training tables (``strategies`` + the latest
OOS row of ``backtests`` for each strategy), a per-candidate table of pooled
OOS stats - sid, status, generation, OOS score, pooled OOS trades / ret / pf,
signal stability and why the candidate was rejected - grouped by status,
plus a score histogram and a gate-breakdown summary (the same numbers the
Scorer uses: count gate, ret/pf gate, promote-score bar).

Constraints honored by design:

- Pure offline DB analysis: opens the SQLite file read-only (``mode=ro``).
- Never touches a broker and never uses the network.
- Never requires OX_LIVE_EXECUTION_APPROVED: the config file is read with a
  plain YAML load (no ``Cfg`` validation), so a ``mode: live`` config is fine.
- No writes, no training, no semantics change of any kind.

Usage (run from the project root; on a live config this is safe):

    python scripts/analyze_patterns.py [config.yaml]

    # e.g. after a closed-market live session on the launch copy:
    python scripts/analyze_patterns.py config.yaml

If no training results exist yet it prints a clear no-results message and
exits 0.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - core dep per requirements.txt
    yaml = None  # type: ignore[assignment]

# Display order; any other statuses appear after these, alphabetically.
_STATUS_ORDER = ("LIVE_APPROVED", "PENDING_APPROVAL", "ELITE", "CANDIDATE",
                 "QUARANTINED", "APPROVED")
_GATE_SCORE = -9.0   # Scorer.score rejection sentinel (brain.py)
_SEED_SCORE = -99.0  # inserted-but-never-evaluated sentinel (brain.py)

_DEFAULT_MIN_TRADES = 25.0
_DEFAULT_PROMOTE_SCORE = 0.8
_DEFAULT_MIN_STABILITY = 0.95


def _load_config(config_path: Path) -> tuple[Optional[Path], dict[str, float]]:
    """Return (db_path, {min_trades, promote_score, min_signal_stability}).

    Reads the file with a plain YAML load - deliberately NOT ``Cfg`` so a
    ``mode: live`` config never trips the affirmation gate in an analysis
    tool.  Relative db_path is anchored to the config's directory exactly
    like ``Cfg`` does.
    """
    raw: dict[str, Any] = {}
    if yaml is not None and config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    db_name = str(raw.get("db_path", "oxalpha.db"))
    db_path = Path(db_name)
    if not db_path.is_absolute():
        db_path = config_path.parent / db_path
    training = raw.get("training", {}) or {}
    try:
        min_trades = float(training.get("min_trades", _DEFAULT_MIN_TRADES))
    except (TypeError, ValueError):
        min_trades = _DEFAULT_MIN_TRADES
    try:
        promote_score = float(training.get("promote_score", _DEFAULT_PROMOTE_SCORE))
    except (TypeError, ValueError):
        promote_score = _DEFAULT_PROMOTE_SCORE
    try:
        stability = float(training.get("min_signal_stability", _DEFAULT_MIN_STABILITY))
    except (TypeError, ValueError):
        stability = _DEFAULT_MIN_STABILITY
    return db_path.resolve(), {
        "min_trades": min_trades,
        "promote_score": promote_score,
        "min_signal_stability": stability,
    }


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _has_tables(conn: sqlite3.Connection) -> bool:
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    return {"strategies", "backtests"}.issubset(names)


def _latest_oos_rows(conn: sqlite3.Connection) -> list[tuple]:
    """(sid, status, gen, score, stats_json) for the newest OOS eval per sid."""
    return conn.execute(
        """
        SELECT s.sid, s.status, s.gen, s.score, b.stats
        FROM strategies s
        JOIN backtests b ON b.sid = s.sid AND b.is_oos = 'OOS'
        WHERE b.bid = (
            SELECT MAX(b2.bid) FROM backtests b2
            WHERE b2.sid = b.sid AND b2.is_oos = 'OOS')
        ORDER BY s.gen, s.sid
        """
    ).fetchall()


def _classify(score: float, stats: dict, gates: dict[str, float]) -> str:
    """Why this candidate ended where it did, mirroring Scorer.score + evaluate."""
    if score > _GATE_SCORE:
        if score >= gates["promote_score"]:
            return f"scored {score:.3f} >= promote_score"
        return f"scored {score:.3f} < promote_score"
    # score == -9.0: reject.  Determine which limb rejected it (brain.py:
    # stability override, then Scorer's count gate, then ret/pf gate).
    stability = stats.get("signal_stability")
    if stability is not None and float(stability) < gates["min_signal_stability"]:
        return "rejected: signal stability"
    if int(stats.get("trades", 0)) < gates["min_trades"]:
        return "rejected: min_trades count"
    if float(stats.get("ret", 0.0)) <= 0.0 or float(stats.get("pf", 0.0)) <= 1.0:
        return "rejected: ret<=0 or pf<=1.0"
    return "rejected: other gate"


def analyze(db_path: Path, gates: dict[str, float]) -> dict[str, Any]:
    """Read the DB and return a structured result (testable without parsing)."""
    result: dict[str, Any] = {
        "db_path": str(db_path),
        "gates": gates,
        "empty": True,
        "rows": [],
        "by_status": {},
        "summary": {},
        "histogram": [],
        "never_evaluated": 0,
        "message": "",
    }
    if not db_path.exists():
        result["message"] = f"no database at {db_path}"
        return result
    conn = _connect_ro(db_path)
    try:
        if not _has_tables(conn):
            result["message"] = f"{db_path} has no strategies/backtests tables"
            return result
        rows_raw = _latest_oos_rows(conn)
        never = conn.execute(
            "SELECT COUNT(*) FROM strategies s WHERE NOT EXISTS ("
            "SELECT 1 FROM backtests b WHERE b.sid = s.sid AND b.is_oos = 'OOS')"
        ).fetchone()[0]
    finally:
        conn.close()
    if not rows_raw:
        result["message"] = (
            "no training results - run a training/live session first "
            "(e.g. a closed-market 'bash scripts/live.sh dhan', which "
            "fetches history and auto-trains at boot), then re-run this tool")
        result["never_evaluated"] = int(never)
        return result

    rows: list[dict[str, Any]] = []
    for sid, status, gen, score, stats_json in rows_raw:
        try:
            stats = json.loads(stats_json) if stats_json else {}
        except (TypeError, ValueError):
            stats = {}
        try:
            parsed_score = float(score)
        except (TypeError, ValueError):
            parsed_score = _SEED_SCORE
        rows.append({
            "sid": sid,
            "status": str(status or "CANDIDATE"),
            "gen": int(gen) if gen is not None else -1,
            "score": parsed_score,
            "trades": int(stats.get("trades", 0)),
            "ret": float(stats.get("ret", 0.0)),
            "pf": float(stats.get("pf", 0.0)),
            "stability": stats.get("signal_stability"),
            "icir": stats.get("icir"),
            "cause": _classify(parsed_score, stats, gates),
        })

    by_status: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row)
    status_order = list(_STATUS_ORDER) + sorted(set(by_status) - set(_STATUS_ORDER))

    min_trades = gates["min_trades"]
    cleared = [r for r in rows if r["trades"] >= min_trades]
    retpf_rejected = [r for r in cleared
                      if r["score"] <= _GATE_SCORE
                      and r["cause"] == "rejected: ret<=0 or pf<=1.0"]
    stability_rejected = [r for r in cleared
                          if r["score"] <= _GATE_SCORE
                          and r["cause"] == "rejected: signal stability"]
    count_rejected = [r for r in rows
                      if r["cause"] == "rejected: min_trades count"]
    promoted_bar = [r for r in rows if r["score"] >= gates["promote_score"]]
    gate_scores = sum(1 for r in rows if r["score"] == _GATE_SCORE)
    seed_scores = sum(1 for r in rows if r["score"] == _SEED_SCORE)

    scored = [r["score"] for r in rows
              if r["score"] not in (_GATE_SCORE, _SEED_SCORE)]
    histogram: list[dict[str, Any]] = []
    if scored:
        lo, hi = min(scored), max(scored)
        span = (hi - lo) or 1.0
        bucket_count = min(8, max(4, int(round(span * 2))))
        edges = [lo + span * i / bucket_count for i in range(bucket_count)]
        buckets = [0] * bucket_count
        for value in scored:
            index = min(bucket_count - 1,
                        max(0, int((value - lo) / span * bucket_count)))
            buckets[index] += 1
        histogram = [
            {"bucket": f"{edges[i]:.2f}..{edges[i+1]:.2f}" if i + 1 < bucket_count
                       else f"{edges[i]:.2f}..{hi:.2f}",
             "count": buckets[i]}
            for i in range(bucket_count)
        ]

    result.update({
        "empty": False,
        "rows": rows,
        "by_status": by_status,
        "status_order": status_order,
        "never_evaluated": int(never),
        "histogram": histogram,
        "histogram_range": [round(min(scored), 3), round(max(scored), 3)] if scored else None,
        "summary": {
            "evaluated": len(rows),
            "cleared_min_trades": len(cleared),
            "rejected_ret_pf": len(retpf_rejected),
            "rejected_stability": len(stability_rejected),
            "rejected_count": len(count_rejected),
            "scored_ge_promote": len(promoted_bar),
            "score_neg9": gate_scores,
            "score_seed": seed_scores,
        },
        "message": "",
    })
    return result


def render(result: dict[str, Any]) -> str:
    """Human-readable dump of the structured result."""
    if result.get("empty"):
        return ("== OX-ALPHA pattern analysis ==\n"
                f"DB: {result.get('db_path')}\n\n"
                f"{result.get('message', 'no training results yet')}\n"
                f"(strategies present but never OOS-evaluated: "
                f"{result.get('never_evaluated', 0)})\n")
    lines: list[str] = []
    g = result["gates"]
    lines.append("== OX-ALPHA pattern analysis (read-only, offline) ==")
    lines.append(f"DB: {result['db_path']}")
    lines.append(f"gates: min_trades={g['min_trades']:g}  "
                 f"promote_score={g['promote_score']:g}  "
                 f"min_signal_stability={g['min_signal_stability']:g}")
    lines.append("")
    for status in result["status_order"]:
        group = result["by_status"].get(status, [])
        lines.append(f"--- {status} ({len(group)}) ---")
        if not group:
            continue
        lines.append(f"  {'sid':<12} {'gen':>3} {'score':>8} {'trades':>7} "
                     f"{'ret':>8} {'pf':>6} {'stability':>9}  cause")
        for row in sorted(group, key=lambda r: (-r["score"], r["sid"])):
            stab = (f"{row['stability']:.3f}" if isinstance(row["stability"], (int, float))
                    else "-")
            lines.append(f"  {row['sid']:<12} {row['gen']:>3} {row['score']:>8.3f} "
                         f"{row['trades']:>7} {row['ret']:>8.3f} {row['pf']:>6.3f} "
                         f"{stab:>9}  {row['cause']}")
        lines.append("")
    s = result["summary"]
    lines.append(f"Gate breakdown ({s['evaluated']} OOS-evaluated candidates, "
                 f"{result['never_evaluated']} never evaluated):")
    lines.append(f"  cleared min_trades ({g['min_trades']:g})      : {s['cleared_min_trades']}")
    lines.append(f"  rejected ret<=0 / pf<=1.0   : {s['rejected_ret_pf']}")
    lines.append(f"  rejected signal stability   : {s['rejected_stability']}")
    lines.append(f"  rejected min_trades count   : {s['rejected_count']}")
    lines.append(f"  scored >= promote_score {g['promote_score']:g}   : {s['scored_ge_promote']}")
    lines.append(f"  score == -9.0 (gate)        : {s['score_neg9']}")
    lines.append("")
    lines.append("Score histogram (non-gate scores):")
    if result["histogram"]:
        for bucket in result["histogram"]:
            width = int(bucket["count"])
            lines.append(f"  [{bucket['bucket']:<24}] {'#' * width} {bucket['count']}")
    else:
        lines.append("  (no candidate scored above the -9.0 gate)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only offline pattern analysis of the latest training run.")
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="agent config file (default: config.yaml)")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    db_path, gates = _load_config(config_path)
    result = analyze(db_path, gates)
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
