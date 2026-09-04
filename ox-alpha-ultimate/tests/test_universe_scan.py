"""Tests for the universe-scan resolvability gate in Agent._apply_universe_scan.

The scan previously handed every configured candidate to the scanner; on a
Dhan session a candidate outside ``security_map`` self-defeated inside the
scanner and a broad except silently kept the static symbols.  The redesigned
flow is a designed gate:

  * the broker decides resolvability — a broker exposing ``security_map``
    (Dhan) restricts candidates to map members; a broker without one (paper)
    scans everything;
  * unresolvable candidates are excluded and logged BEFORE the scanner runs;
  * if nothing is resolvable, or fewer than two names survive, or the
    scanner raises, the static symbols are kept and a UNIVERSE_SCAN_SKIP
    event records why — no path is silent;
  * a final guard drops any result symbol the venue cannot resolve.

These tests drive the method on a fake Agent surface (no broker, no network,
no DB file); the real paper-broker boot path is covered by the regression
suite's universe tests.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from ox.agent import Agent


# ── fake surface ─────────────────────────────────────────────────────────────

class FakeEventsDB:
    """Records (kind, msg) pairs from the event INSERTs the scan makes."""

    def __init__(self):
        self.events = []

    def ex(self, sql, params):
        # The kind is a SQL literal; params carry (msg, ts).
        kind = re.search(r"VALUES\('([A-Z_]+)'", sql).group(1)
        self.events.append((kind, params[0]))


class ScriptedScanner:
    """MarketScanner stand-in: deterministic ranking from a price table."""

    prices = {"TCS": 250.0, "INFY": 300.0, "YESBANK": 60.0, "SUZLON": 45.0}
    last_universe = None
    last_top_k = None

    def __init__(self, cfg, db, broker):
        self.cfg, self.db, self.broker = cfg, db, broker

    def scan(self, universe, top_k=20):
        ScriptedScanner.last_universe = list(universe)
        ScriptedScanner.last_top_k = top_k
        return [{"symbol": sym, "last_price": ScriptedScanner.prices[sym]}
                for sym in universe if sym in ScriptedScanner.prices][:top_k]


class RogueScanner(ScriptedScanner):
    """Returns a symbol the venue could never resolve, to test the final guard."""

    def scan(self, universe, top_k=20):
        ScriptedScanner.last_universe = list(universe)
        ScriptedScanner.last_top_k = top_k
        return [{"symbol": "PENNY", "last_price": 1.0}] + super().scan(universe, top_k)


def make_agent(broker, candidates=("TCS", "INFY", "YESBANK", "SUZLON"),
               symbols=("RELIANCE",), auto_scan=True):
    agent = object.__new__(Agent)
    agent.cfg = {
        "symbols": list(symbols),
        "universe": {
            "auto_scan": auto_scan,
            "candidates": list(candidates),
            "price_ceiling": 500.0,
            "top_k": 12,
        },
    }
    agent.broker = broker
    agent.db = FakeEventsDB()
    return agent


@pytest.fixture(autouse=True)
def _reset_scanner():
    ScriptedScanner.last_universe = None
    ScriptedScanner.last_top_k = None
    yield


# ── paper mode: unrestricted ─────────────────────────────────────────────────

def test_paper_scan_sees_all_candidates_and_applies_selection(monkeypatch):
    broker = SimpleNamespace(name="paper")  # no security_map attribute
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", ScriptedScanner)
    agent._apply_universe_scan()
    # Paper resolves any symbol synthetically: nothing filtered before scan
    # and every affordable candidate (all four, each <= price ceiling) lands.
    assert set(ScriptedScanner.last_universe) == {"TCS", "INFY", "YESBANK", "SUZLON"}
    assert agent.cfg["symbols"] == ["TCS", "INFY", "YESBANK", "SUZLON"]
    assert agent.db.events == [("UNIVERSE_SCAN", "TCS,INFY,YESBANK,SUZLON")]


def test_paper_scan_skips_when_fewer_than_two_affordable(monkeypatch):
    broker = SimpleNamespace(name="paper")
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", ScriptedScanner)
    ScriptedScanner.prices = {"TCS": 250.0}  # only one candidate has data
    agent._apply_universe_scan()
    assert agent.cfg["symbols"] == ["RELIANCE"]  # static kept
    assert agent.db.events[0][0] == "UNIVERSE_SCAN_SKIP"
    assert "fewer than two" in agent.db.events[0][1]
    ScriptedScanner.prices = {"TCS": 250.0, "INFY": 300.0, "YESBANK": 60.0, "SUZLON": 45.0}


# ── Dhan mode: resolvability gate ────────────────────────────────────────────

def test_dhan_scan_offers_only_resolvable_candidates(monkeypatch):
    broker = SimpleNamespace(name="dhan", security_map={"TCS": "111", "INFY": "222"})
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", ScriptedScanner)
    agent._apply_universe_scan()
    # YESBANK/SUZLON are outside security_map: never offered to the scanner.
    assert set(ScriptedScanner.last_universe) == {"TCS", "INFY"}
    assert agent.cfg["symbols"] == ["TCS", "INFY"]
    assert all(sym in broker.security_map for sym in agent.cfg["symbols"])
    assert agent.db.events == [("UNIVERSE_SCAN", "TCS,INFY")]


def test_dhan_scan_skipped_when_no_candidate_resolvable(monkeypatch):
    # Every configured candidate falls outside security_map: the scanner must
    # never run, static symbols stay, and a recorded reason names the drop.
    broker = SimpleNamespace(name="dhan", security_map={"RELIANCE": "999"})
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", ScriptedScanner)
    agent._apply_universe_scan()
    assert ScriptedScanner.last_universe is None  # scanner never invoked
    assert agent.cfg["symbols"] == ["RELIANCE"]
    assert len(agent.db.events) == 1
    kind, msg = agent.db.events[0]
    assert kind == "UNIVERSE_SCAN_SKIP"
    assert "YESBANK" in msg and "TCS" in msg
    assert "static symbols" in msg


def test_dhan_scan_result_never_exceeds_resolvable(monkeypatch):
    # Even a misbehaving scanner returning an unresolvable symbol cannot get
    # it into the universe: the final guard filters it out.
    broker = SimpleNamespace(name="dhan", security_map={"TCS": "111", "INFY": "222"})
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", RogueScanner)
    agent._apply_universe_scan()
    assert agent.cfg["symbols"] == ["TCS", "INFY"]
    assert "PENNY" not in agent.cfg["symbols"]
    assert all(sym in broker.security_map for sym in agent.cfg["symbols"])


# ── failure paths are non-silent ─────────────────────────────────────────────

def test_scan_exception_keeps_static_symbols_with_recorded_reason(monkeypatch):
    class BoomScanner(ScriptedScanner):
        def scan(self, universe, top_k=20):
            ScriptedScanner.last_universe = list(universe)
            raise RuntimeError("broker transport down")

    broker = SimpleNamespace(name="paper")
    agent = make_agent(broker)
    monkeypatch.setattr("ox.scanner.MarketScanner", BoomScanner)
    agent._apply_universe_scan()
    assert agent.cfg["symbols"] == ["RELIANCE"]
    kind, msg = agent.db.events[0]
    assert kind == "UNIVERSE_SCAN_SKIP"
    assert "RuntimeError" in msg and "static symbols" in msg


def test_scan_disabled_is_a_noop(monkeypatch):
    broker = SimpleNamespace(name="dhan", security_map={"TCS": "111"})
    agent = make_agent(broker, auto_scan=False)
    monkeypatch.setattr("ox.scanner.MarketScanner", ScriptedScanner)
    agent._apply_universe_scan()
    assert ScriptedScanner.last_universe is None
    assert agent.db.events == []
    assert agent.cfg["symbols"] == ["RELIANCE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])