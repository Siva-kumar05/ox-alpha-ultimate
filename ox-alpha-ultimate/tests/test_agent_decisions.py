"""Unit tests for Agent._act() core decision logic (entry/exit/no-op paths).

The reviewed P0 list claimed undefined-variable and NaN crashes in the
decision path.  This suite locks the contract with a FULLY FAKE surface:
no broker, no network, no ccxt, no live clock, no database file.  The Agent
is built with ``object.__new__`` so ``__init__`` never runs; every attribute
the decision path touches is replaced by a deterministic stub.

Covered paths:
  * _act(): exits (opposite signal, order-flow reversal), held-position
    no-op, every BLOCK gate (order-flow unavailable / not-long, trend
    confirmation, ensemble quorum, empty-supporter support quorum, negative
    news, bracket unavailable, risk-cap clamped, risk gate + daily halt,
    circuit-breaker size), and the full ENTRY_REQUEST path with real
    ATR bracket sizing, optional leverage overlay, and the defensive
    bracket/mtf defaults.
  * tick_once(): vote aggregation with an EMPTY supporter list must leave
    votes untouched (no np.mean([]) NaN), the MTF-disabled defensive default
    must reach the decision record, the circuit-breaker BLOCK path must run
    with cb_state defined, and a missing frame must skip the symbol without
    a crash.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ox.agent import Agent
from ox.leverage_engine import LeverageEngine


# ── fake surface ────────────────────────────────────────────────────────────

class FakeFlow:
    def __init__(self, ready=True, long_entry=True, long_exit=False,
                 reason="ORDER_FLOW_OK", pressure_ema=0.3):
        self.ready = ready
        self.long_entry = long_entry
        self.long_exit = long_exit
        self.reason = reason
        self.pressure_ema = pressure_ema

    def details(self):
        return {"flow_reason": self.reason, "pressure_ema": self.pressure_ema}


class FakeBroker:
    def __init__(self, flow=None, quotes=None):
        self.flow = flow
        self._quotes = quotes or {}

    def order_flow(self, sym):
        return self.flow

    def ltps(self, syms):
        return {sym: self._quotes.get(sym, 100.0) for sym in syms}

    def quote_snapshot(self, syms):
        # Empty snapshot: _apply_volumes() skips silently, matching brokers
        # without a volume feed, and never touches a live clock.
        return {}


class FakeOMS:
    def __init__(self):
        self.positions = {}
        self.closed = []
        self.opened = []

    def close(self, sym, reason):
        self.closed.append((sym, reason))

    def mark(self, sym, price):
        pass

    def reconcile(self):
        pass

    def open_position(self, sym, side, qty, label, stop, target, votes):
        self.opened.append({
            "sym": sym, "side": side, "qty": qty, "label": label,
            "stop": stop, "target": target, "votes": votes,
        })


class FakeRisk:
    def __init__(self, size=300, approve=True, approve_reason=""):
        self._size = size
        self._approve = approve
        self._approve_reason = approve_reason

    def size_with_kelly(self, price, stop_distance, confidence=0.5):
        return self._size

    def gross_exposure(self, positions):
        return 0.0

    def approve(self, sym, side, qty, px, positions, var_pct):
        return self._approve, self._approve_reason


class FakeNews:
    def __init__(self, optimism=0.5):
        self.optimism = optimism

    def get_optimism_score(self, db, sym):
        return self.optimism, 0.2


class FakeComp:
    def __init__(self):
        self.halted = False
        self.halt_calls = []

    def halt(self, reason):
        self.halted = True
        self.halt_calls.append(reason)


class FakeMetrics:
    def __init__(self):
        self.counters = {}
        self.gauges = {}

    def counter(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def gauge(self, name, value):
        self.gauges[name] = value


class FakeDB:
    def __init__(self):
        self.decisions = []
        self.kv = {}
        self.queries = []

    def record_decision(self, symbol, action, reason, detail=None):
        self.decisions.append((symbol, action, reason, detail))

    def kv_set(self, key, value):
        self.kv[key] = value

    def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    def q(self, sql, *args):
        self.queries.append(sql)
        return []


class FakeCircuitBreaker:
    def __init__(self, block=False, halt=False, multiplier=1.0, state="NORMAL"):
        self._block = block
        self._halt = halt
        self._multiplier = multiplier
        self.state = state

    def should_halt(self):
        return self._halt

    def should_block_entries(self):
        return self._block

    def get_size_multiplier(self):
        return self._multiplier

    def evaluate(self, sharpe):
        return self.state


class FakeAttribution:
    def detect_degradation(self):
        return {"rolling_sharpe": 0.5}


class FakeRegimeDetector:
    def __init__(self, weights=None):
        self._weights = weights or {"core": 1.0, "scalp": 1.0, "breakout": 1.0}

    def detect(self, frame):
        return SimpleNamespace(
            regime=SimpleNamespace(value="RANGING"),
            confidence=0.6, volatility_percentile=30.0,
            trend_strength=15.0, ts="2026-01-01T09:30:00",
        )

    def regime_weights(self):
        return self._weights


class FakeMtf:
    def __init__(self, enabled=False, score=None):
        self.enabled = enabled
        self.score = score

    def alignment_score(self, frame):
        return {"score": self.score or 0.8, "details": {}, "aligned": True}


# ── builders ────────────────────────────────────────────────────────────────

DEFAULT_MTF_RESULT = {"score": 0.5, "details": {}, "aligned": True}

UNSET = object()  # distinguish "not provided" from an explicit flow=None


class FakeCfg(dict):
    """Dict-style cfg that also exposes ``.root`` (Agent.kill_path needs it)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = "."


def make_cfg(**overrides):
    cfg = FakeCfg({
        "symbols": ["TCS"],
        "entry_cutoff": "14:45",
        "squareoff": "15:15",
        "market_close": "15:30",
        "capital": 500_000,
        "order_flow": {"enabled": True, "primary": True},
        "execution": {
            "min_vote_fraction": 0.0,
            "min_support_strategies": 1,
            "signal_history_candles": 300,
            "autonomous": True,
        },
        "risk": {"max_notional_per_trade": 200_000, "max_gross_exposure": 500_000},
        "multi_timeframe": {"enabled": True},
    })
    cfg.update(overrides)
    return cfg


def make_frame(rows=320, base=100.0):
    x = np.linspace(base, base + 2.0, rows)
    return pd.DataFrame({"o": x, "h": x + 1.0, "l": x - 1.0, "c": x + 0.25, "v": 1000.0})


def build_agent(tmp_path, *, flow=UNSET, strategies=None, positions=None,
                risk=None, news=None, cb=None, mtf=None, leverage_engine=None,
                size_multiplier=1.0, cfg=None, votes=None, supporters=None,
                act=None, completed_frame=None, frame_fn=None):
    """Create an Agent whose __init__ never ran, with a fully fake surface.

    The default flow is READY and long-entry so tests reach the gate they
    target; pass ``flow=None`` to simulate a broker with no order-flow feed.
    """
    agent = object.__new__(Agent)
    agent.cfg = cfg or make_cfg()
    flow_obj = FakeFlow() if flow is UNSET else flow
    agent.broker = FakeBroker(flow=flow_obj, quotes={"TCS": 100.0})
    agent.oms = FakeOMS()
    if positions:
        agent.oms.positions.update(positions)
    agent.risk = risk or FakeRisk()
    agent.news = news or FakeNews()
    agent.comp = FakeComp()
    agent.metrics = FakeMetrics()
    agent.db = FakeDB()
    agent.leverage_engine = leverage_engine
    agent._current_regime = SimpleNamespace(value="RANGING")
    agent.strategies = strategies if strategies is not None else [
        ("core", None, {"sl_atr": 2.0, "tp_atr": 3.0}, 1.0),
    ]
    agent._last_decision = {}
    agent._last_reconcile = 0.0
    agent.cognition = None
    agent._position_size_multiplier = size_multiplier
    agent._set_health = lambda state, message, force=False: None
    agent.cfg.root = str(tmp_path)  # kill_path property resolves here
    agent.stop = False
    agent.circuit_breaker = cb or FakeCircuitBreaker()
    agent.attribution = FakeAttribution()
    agent.regime_detector = FakeRegimeDetector()
    agent.mtf_analyzer = mtf if mtf is not None else FakeMtf()
    agent.frame = frame_fn if frame_fn is not None else (lambda sym, n: None)
    agent.completed_frame = completed_frame if completed_frame is not None else (lambda sym, n=240: make_frame())
    # Offline no-ops for the real methods tick_once() would otherwise call.
    agent._trading_day = lambda: True
    agent.in_session = lambda: True
    agent._refresh_news = lambda force=False: None
    agent.ingest_quote = lambda sym, price: None
    if votes is not None:
        agent._vote_details = lambda frame: (votes, supporters or [])
    if act is not None:
        agent._act = act
    return agent


@pytest.fixture
def fake_time(monkeypatch):
    """Deterministic monotonic clock (starts past the 30s decision throttle)
    and a fixed in-session wall-clock time; no live time source."""
    state = {"now": 100.0}

    def monotonic():
        state["now"] += 1.0
        return state["now"]

    monkeypatch.setattr("ox.agent.time.monotonic", monotonic)
    monkeypatch.setattr("ox.agent.hhmm", lambda: "10:00")
    return state


def decisions(db):
    return [(a, r) for _, a, r, _ in db.decisions]


def act_decisions(agent):
    return [(d[0], d[1], d[2]) for d in agent.db.decisions]


def entry_supporter(sid="core"):
    return [(sid, {"sl_atr": 2.0, "tp_atr": 3.0}, 1.0)]


# ── _act(): exits and no-ops ────────────────────────────────────────────────

def test_exit_on_opposite_signal(tmp_path, fake_time):
    agent = build_agent(tmp_path, positions={"TCS": {"qty": 5}})
    agent._act("TCS", 100.0, make_frame(), votes=-1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("EXIT", "OPPOSITE_SIGNAL")]
    assert agent.oms.closed == [("TCS", "OPPOSITE_SIGNAL")]
    assert agent.oms.opened == []


def test_exit_on_order_flow_reversal(tmp_path, fake_time):
    flow = FakeFlow(long_exit=True)
    agent = build_agent(tmp_path, flow=flow, positions={"TCS": {"qty": 5}})
    # votes positive: only the flow reversal can justify the exit
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("EXIT", "ORDER_FLOW_REVERSAL")]
    assert agent.oms.closed == [("TCS", "ORDER_FLOW_REVERSAL")]
    assert agent.oms.opened == []


def test_held_position_positive_votes_is_noop(tmp_path, fake_time):
    agent = build_agent(tmp_path, flow=FakeFlow(long_exit=False),
                        positions={"TCS": {"qty": 5}})
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert agent.db.decisions == []
    assert agent.oms.closed == []
    assert agent.oms.opened == []


# ── _act(): entry gates ─────────────────────────────────────────────────────

def test_blocks_when_primary_order_flow_missing(tmp_path, fake_time):
    agent = build_agent(tmp_path, flow=None)  # broker has no order-flow feed
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "ORDER_FLOW_UNAVAILABLE")]
    assert agent.oms.opened == []


def test_blocks_when_order_flow_not_long_entry(tmp_path, fake_time):
    flow = FakeFlow(long_entry=False, reason="SPREAD_TOO_WIDE")
    agent = build_agent(tmp_path, flow=flow)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "SPREAD_TOO_WIDE")]
    assert agent.oms.opened == []


def test_blocks_on_non_positive_votes(tmp_path, fake_time):
    agent = build_agent(tmp_path)
    agent._act("TCS", 100.0, make_frame(), votes=0.0, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "TREND_CONFIRMATION_MISSING")]


def test_blocks_when_no_strategies_loaded(tmp_path, fake_time):
    agent = build_agent(tmp_path, strategies=[])
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "TREND_CONFIRMATION_MISSING")]


def test_blocks_on_ensemble_quorum(tmp_path, fake_time):
    cfg = make_cfg()
    cfg["execution"]["min_vote_fraction"] = 0.5  # total_weight 1.0 -> need 0.5
    agent = build_agent(tmp_path, cfg=cfg)
    agent._act("TCS", 100.0, make_frame(), votes=0.2, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "ENSEMBLE_QUORUM")]


def test_blocks_on_empty_supporters(tmp_path, fake_time):
    # Empty supporter list with min_support_strategies=1 must BLOCK cleanly
    # (the _act-level companion of the tick-level empty-aggregation guard).
    agent = build_agent(tmp_path)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=[])
    assert decisions(agent.db) == [("BLOCK", "SUPPORT_QUORUM")]
    assert agent.oms.opened == []


def test_blocks_on_negative_news(tmp_path, fake_time):
    agent = build_agent(tmp_path, news=FakeNews(optimism=-0.5))
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "NEGATIVE_NEWS")]


def test_blocks_when_bracket_unavailable(tmp_path, fake_time):
    agent = build_agent(tmp_path)
    # _bracket_from_supporters raising must be caught, never propagate.
    def broken_bracket(frame, entry_price, supporters):
        raise ValueError("A positive ensemble vote has no approved entry strategy")

    agent._bracket_from_supporters = broken_bracket
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "ENTRY_BRACKET_UNAVAILABLE")]
    assert agent.oms.opened == []


# ── _act(): sizing, risk gate, entry ────────────────────────────────────────

def test_entry_requests_full_path_with_defensive_defaults(tmp_path, fake_time):
    # Real ATR bracket sizing; mtf_result=None must not crash and the risk
    # gate approves; the default bracket detail rides into the ENTRY_REQUEST.
    agent = build_agent(tmp_path)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter(),
               mtf_result=None)
    assert act_decisions(agent) == [("TCS", "ENTRY_REQUEST", "ORDER_FLOW_PRIMARY")]
    symbol, action, reason, detail = agent.db.decisions[0]
    assert action == "ENTRY_REQUEST"
    assert symbol == "TCS"
    assert detail["quantity"] == 300
    assert detail["strategy_ids"] == ["core"]
    assert detail["atr"] > 0
    assert detail["stop_distance"] > 0
    assert "mtf_score" not in detail  # no fabricated MTF claim without a result
    assert len(agent.oms.opened) == 1
    opened = agent.oms.opened[0]
    assert opened["qty"] == 300
    assert opened["side"] == "BUY"
    assert opened["label"].startswith("blend:core")
    assert opened["stop"] < 100.0 < opened["target"]


def test_leverage_overlay_scales_quantity_within_caps(tmp_path, fake_time):
    # Real LeverageEngine (pure numpy, offline): the request scales the base
    # quantity by lev/3 and the hard caps still clamp the result.
    engine = LeverageEngine({"leverage_engine": {"enabled": True}})
    agent = build_agent(tmp_path, strategies=[("scalp", None, {"sl_atr": 2.0, "tp_atr": 3.0}, 1.0)],
                        supporters=entry_supporter("scalp"), leverage_engine=engine)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter("scalp"))
    detail = agent.db.decisions[0][3]
    leverage = float(detail["leverage"])
    assert leverage >= 1.0
    assert detail["leverage_tier"] in ("equity_scalp", "equity_intraday")
    notional_cap = int(200_000 / 100.0)
    headroom = int(500_000 / (100.0 * leverage))
    expected = min(max(1, int(300 * (leverage / 3.0))), notional_cap, headroom)
    assert agent.oms.opened[0]["qty"] == expected
    assert expected > 0


def test_blocks_when_risk_cap_clamped_to_zero(tmp_path, fake_time):
    cfg = make_cfg()
    cfg["risk"]["max_gross_exposure"] = 0.0  # no exposure headroom at all
    agent = build_agent(tmp_path, cfg=cfg)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "RISK_CAP_CLAMPED")]
    assert agent.oms.opened == []


def test_risk_gate_reject_halts_on_daily_loss(tmp_path, fake_time):
    risk = FakeRisk(approve=False, approve_reason="daily loss cap hit: 2.0%")
    agent = build_agent(tmp_path, risk=risk)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "RISK_GATE")]
    assert agent.comp.halt_calls == ["daily loss cap hit: 2.0%"]
    assert agent.comp.halted is True
    assert agent.oms.opened == []


def test_risk_gate_reject_without_daily_loss_does_not_halt(tmp_path, fake_time):
    risk = FakeRisk(approve=False, approve_reason="correlation limit")
    agent = build_agent(tmp_path, risk=risk)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "RISK_GATE")]
    assert agent.comp.halt_calls == []
    assert agent.oms.opened == []


def test_circuit_breaker_zero_size_blocks_entry(tmp_path, fake_time):
    agent = build_agent(tmp_path, size_multiplier=0.0)
    agent._act("TCS", 100.0, make_frame(), votes=1.5, supporters=entry_supporter())
    assert decisions(agent.db) == [("BLOCK", "CIRCUIT_BREAKER_SIZE")]
    assert agent.oms.opened == []


# ── tick_once(): wiring paths from the reviewed P0 list ────────────────────

def test_tick_empty_supporters_keeps_votes_intact(tmp_path, fake_time):
    # np.mean over an empty supporter list would be NaN; the guard must keep
    # votes untouched and hand them to _act unchanged.
    captured = {}

    def record_act(sym, ltp, frame, votes, supporters, mtf_result=None):
        captured["votes"] = votes
        captured["supporters"] = supporters
        captured["mtf_result"] = mtf_result

    agent = build_agent(tmp_path, votes=2.0, supporters=[], act=record_act)
    agent.tick_once()
    assert math.isfinite(captured["votes"])
    assert captured["votes"] == 2.0
    assert captured["supporters"] == []
    assert agent.oms.opened == []


def test_tick_mtf_disabled_uses_defensive_default(tmp_path, fake_time):
    captured = {}
    agent = build_agent(tmp_path, mtf=FakeMtf(enabled=False), votes=1.5,
                        supporters=entry_supporter(),
                        act=lambda sym, ltp, frame, votes, supporters, mtf_result=None: captured.update(mtf_result=mtf_result))
    agent.tick_once()
    assert agent.db.kv["mtf_TCS"] == DEFAULT_MTF_RESULT
    assert captured["mtf_result"] == DEFAULT_MTF_RESULT


def test_tick_mtf_enabled_uses_analyzer_score(tmp_path, fake_time):
    captured = {}
    mtf = FakeMtf(enabled=True, score=0.8)
    agent = build_agent(tmp_path, mtf=mtf, votes=1.5, supporters=entry_supporter(),
                        act=lambda sym, ltp, frame, votes, supporters, mtf_result=None: captured.update(mtf_result=mtf_result))
    agent.tick_once()
    assert agent.db.kv["mtf_TCS"]["score"] == 0.8
    assert captured["mtf_result"]["score"] == 0.8


def test_tick_circuit_breaker_blocks_entries(tmp_path, fake_time):
    called = []
    agent = build_agent(tmp_path, cb=FakeCircuitBreaker(block=True), votes=1.5,
                        supporters=entry_supporter(),
                        act=lambda *a, **k: called.append(a))
    agent.tick_once()
    # cb_state came from circuit_breaker.evaluate() before the loop: defined.
    assert decisions(agent.db) == [("BLOCK", "CIRCUIT_BREAKER")]
    assert called == []
    assert agent.oms.opened == []


def test_tick_symbol_without_frame_is_skipped_without_crash(tmp_path, fake_time):
    called = []
    agent = build_agent(tmp_path, votes=1.5, supporters=entry_supporter(),
                        completed_frame=lambda sym, n=240: None,
                        act=lambda *a, **k: called.append(a))
    agent.tick_once()
    assert called == []
    assert agent.db.decisions == []


def test_tick_full_entry_via_real_act(tmp_path, fake_time):
    # End-to-end through the fake surface: tick_once() -> real _act() ->
    # real ATR bracket -> ENTRY_REQUEST with the MTF default in the record.
    agent = build_agent(tmp_path, mtf=FakeMtf(enabled=False), votes=1.5,
                        supporters=entry_supporter())
    agent.tick_once()
    symbol, action, reason, detail = agent.db.decisions[-1]
    assert (action, reason) == ("ENTRY_REQUEST", "ORDER_FLOW_PRIMARY")
    assert symbol == "TCS"
    assert detail["mtf_score"] == 0.5
    assert detail["mtf_aligned"] is True
    assert detail["quantity"] > 0
    assert agent.oms.opened[0]["sym"] == "TCS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])