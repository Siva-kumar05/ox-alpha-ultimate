from __future__ import annotations

import sys
import tempfile
import time
import math
import os
import struct
import sqlite3
from datetime import datetime
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def _restore_audit_key(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("OX_AUDIT_KEY", None)
    else:
        os.environ["OX_AUDIT_KEY"] = previous


def _close_log_handlers(logger) -> None:
    # Windows otherwise holds the temporary smoke-test log open after a failed assertion.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _smoke_config(root: Path) -> Path:
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["db_path"] = "smoke.db"
    config["symbols"] = ["TCS", "RELIANCE", "INFY"]
    # security_map must mirror symbols in both directions (Cfg validation).
    config["security_map"] = {
        symbol: value for symbol, value in (config.get("security_map") or {}).items()
        if symbol in config["symbols"]
    }
    config["training"] = {
        "iterations": 1, "population": 6, "elite_k": 2, "min_trades": 1,
        "promote_score": -9.0, "min_signal_stability": 0.01,
        "walk_forward_folds": 3, "embargo_candles": 5,
        "require_human_approval": False,
        "random_seed": 4,  # pinned: a smoke test must never fail on candidate-entropy luck
    }
    config["execution"]["signal_history_candles"] = 100
    config["execution"]["min_vote_fraction"] = 0.0
    config["history_days"] = 2
    config["order_flow"] = {
        "min_observations": 1,
        "min_flow_imbalance": 0.0,
        "min_microprice_edge_bps": 0.0,
        "min_pressure_ema": 0.0,
        "min_positive_streak": 1,
        "min_liquidity_score": 0.0,
    }
    config["auto_train_on_boot"] = False
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_smoketest() -> None:
    """Runs in a temporary directory; it never deletes project or live-account data."""
    from ox.agent import Agent
    from ox.brokers import DhanDepthFeed
    from ox.brain import Backtester
    from ox.core import DB, LOG
    from ox.decision import bracket_from_supporters
    from ox.core import IST, iso
    from ox.features import _swings
    from ox.orderflow import DhanDepthParser, DepthParseError, OrderFlowEngine, OrderFlowReplayValidator

    with tempfile.TemporaryDirectory(prefix="ox-alpha-smoke-") as directory, ExitStack() as cleanup:
        prior_audit_key = os.environ.get("OX_AUDIT_KEY")
        cleanup.callback(_restore_audit_key, prior_audit_key)
        cleanup.callback(_close_log_handlers, LOG)
        os.environ["OX_AUDIT_KEY"] = "smoke-audit-key-is-at-least-thirty-two-characters"
        config_path = _smoke_config(Path(directory))
        agent = Agent(str(config_path))
        cleanup.callback(agent.close)
        assert agent.comp.check_ip(), "paper IP prerequisite failed"
        assert agent.comp.daily_auth(agent.broker), "paper broker authentication failed"
        agent.broker.set_px("TCS", 1_000.0)
        flow = agent.broker.order_flow("TCS")
        assert flow is not None and flow.ready and flow.long_entry and flow.source == "SIMULATED_DEPTH", "paper order-flow gate did not produce an explicit simulated snapshot"
        assert 0.0 <= flow.liquidity_score <= 1.0 and flow.positive_streak >= 1, "paper order-flow insight metrics were not bounded"

        # The Dhan full-depth wire decoder and bid/ask pairing are tested with
        # locally constructed binary packets.  No network or broker account is
        # involved in this smoke test.
        def depth_packet(code: int, security_id: int, sequence: int, start_price: float, quantity: int) -> bytes:
            body = b"".join(
                struct.pack("<dII", start_price + (index * (-0.05 if code == DhanDepthParser.BID_CODE else 0.05)), quantity - index, 1)
                for index in range(DhanDepthParser.LEVELS)
            )
            return struct.pack("<HBBII", DhanDepthParser.PACKET_SIZE, code, 1, security_id, sequence) + body

        bid_wire = depth_packet(DhanDepthParser.BID_CODE, 12345, 7, 999.95, 1_500)
        ask_wire = depth_packet(DhanDepthParser.ASK_CODE, 12345, 8, 1_000.05, 1_100)
        parsed = DhanDepthParser.parse(bid_wire + ask_wire)
        assert len(parsed) == 2 and parsed[0].sequence == 7 and parsed[1].sequence == 8
        try:
            DhanDepthParser.parse(b"\x00")
        except DepthParseError:
            pass
        else:
            raise AssertionError("malformed Dhan depth data was accepted")
        wrong_length_wire = bytearray(bid_wire)
        struct.pack_into("<H", wrong_length_wire, 0, 320)
        try:
            DhanDepthParser.parse(wrong_length_wire)
        except DepthParseError:
            pass
        else:
            raise AssertionError("unexpected Dhan depth packet length was accepted")
        validation_flow = OrderFlowEngine(agent.cfg, agent.db)
        depth_feed = DhanDepthFeed("test-token", "test-client", {"12345": "TCS"}, validation_flow)
        depth_feed._on_message(None, bid_wire)
        depth_feed._on_message(None, ask_wire)
        paired_flow = validation_flow.assessment("TCS")
        assert paired_flow is not None and paired_flow.source == "DHAN_DEPTH20", "Dhan bid/ask depth pairing did not produce an L2 snapshot"
        observations_before_stale_pair = paired_flow.observations
        depth_feed._partial["TCS"]["BID"] = (parsed[0].levels, time.monotonic() - 3.0)
        depth_feed._on_message(None, ask_wire)
        assert validation_flow.assessment("TCS").observations == observations_before_stale_pair, "stale bid/ask pairing was admitted as current L2 state"

        # Existing local databases receive the additional insight columns
        # without a destructive rebuild or loss of historic audit data.
        legacy_path = Path(directory) / "legacy.db"
        legacy = sqlite3.connect(legacy_path)
        legacy.execute("CREATE TABLE orderflow(ofid INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT, source TEXT, bid REAL, ask REAL, mid REAL, microprice REAL, spread_bps REAL, book_imbalance REAL, flow_imbalance REAL, microprice_edge_bps REAL, bid_notional REAL, ask_notional REAL, observations INTEGER, ready INTEGER, entry_signal INTEGER, exit_signal INTEGER, ts TEXT)")
        legacy.commit()
        legacy.close()
        migrated_db = DB(legacy_path)
        cleanup.callback(migrated_db.close)
        migrated_columns = {row[1] for row in migrated_db.q("PRAGMA table_info(orderflow)")}
        assert {"pressure_ema", "positive_streak", "liquidity_score", "book_state", "reason"}.issubset(migrated_columns), "order-flow database migration did not preserve schema compatibility"

        # Primary L2 admission is not promoted from OHLCV alone.  The replay
        # validator accepts only retained real-depth entry snapshots followed
        # by later recorded candles, and labels this a gate study rather than
        # an execution backtest.
        replay_db = DB(Path(directory) / "replay.db")
        cleanup.callback(replay_db.close)
        base_epoch = int(time.time()) - 10_000
        for index in range(30):
            timestamp = base_epoch + index * 120
            replay_db.ex(
                "INSERT INTO orderflow(sym,source,bid,ask,mid,microprice,spread_bps,book_imbalance,flow_imbalance,pressure_ema,positive_streak,liquidity_score,book_state,microprice_edge_bps,bid_notional,ask_notional,observations,ready,entry_signal,exit_signal,reason,ts)VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("TCS", "DHAN_DEPTH20", 100.0, 100.1, 100.05, 100.06, 10.0, 0.2, 0.1, 0.1, 3, 0.9, "BUY_SUPPORT", 1.0, 100000.0, 90000.0, 500, 1, 1, 0, "FLOW_CONFIRMATION", iso(datetime.fromtimestamp(timestamp, IST))),
            )
            for step in range(1, 6):
                replay_db.ex(
                    "INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",
                    ("TCS", timestamp + step * 60, 101.0, 101.2, 100.8, 101.0, 1000),
                )
        replay = OrderFlowReplayValidator(agent.cfg, replay_db).evaluate()
        assert replay["passed"] and replay["samples"] >= 30 and replay["kind"] == "L2_GATE_REPLAY_NOT_EXECUTION_BACKTEST", "retained-depth gate replay did not enforce its evidence threshold"

        # A centred swing can only be used after its right-hand confirmation
        # bar has closed.  The pivot at index 1 is visible at index 2, never
        # at index 1 where it would leak future price action.
        confirmed_highs, _ = _swings(np.array([1.0, 3.0, 1.0]), np.array([0.0, -1.0, 0.0]), k=1)
        assert confirmed_highs == [(2, 1)], "swing confirmation leaked future-candle information"

        assert len(agent.broker.hist("TCS", 1, 5)) >= 4 * 300, "paper broker did not generate session-aware bootstrap candles"
        agent.refresh_history("TCS")
        assert agent.db.q("SELECT COUNT(*) FROM candles WHERE sym='TCS'")[0][0] >= 300, "historical ingestion did not persist a candle batch"
        # Keep the model-validation fixture deterministic and independent from
        # the separate ingestion check above.
        agent.db.ex("DELETE FROM candles WHERE sym='TCS'")

        audit_db = DB(Path(directory) / "audit.db")
        cleanup.callback(audit_db.close)
        audit_db.audit("SMOKE", {"safe": True})
        assert audit_db.verify_audit(), "audit chain did not verify"
        audit_db.ex("UPDATE audit SET action='TAMPERED' WHERE aid=1")
        assert not audit_db.verify_audit(), "audit tampering was not detected"

        # Validation must match the deployable execution path: an entry signal
        # uses the following candle open, while a negative signal cannot create
        # a short position in a long-only agent.
        validation_frame = pd.DataFrame({
            "ts": [1, 2, 3, 4, 5, 6],
            "o": [100, 100, 80, 110, 110, 110],
            "h": [101, 101, 85, 112, 112, 112],
            "l": [99, 99, 79, 108, 108, 108],
            "c": [100, 100, 82, 110, 110, 110],
            "v": [1000, 1000, 1000, 1000, 1000, 1000],
        })
        params = {"sl_atr": 1.0, "tp_atr": 2.0}
        def entry_then_exit(_, __):
            return {"signal": np.array([0, 1, -1, 0, 0, 0]), "atr": np.ones(6)}
        def negative_only(_, __):
            return {"signal": np.array([0, -1, 0, 0, 0, 0]), "atr": np.ones(6)}
        backtester = Backtester(agent.cfg)
        aligned_stats, _ = backtester.run(validation_frame, entry_then_exit, params)
        no_short_stats, _ = backtester.run(validation_frame, negative_only, params)
        assert aligned_stats["execution"] == "NEXT_CANDLE_OPEN_LONG_ONLY" and aligned_stats["ret"] > 0.30
        assert no_short_stats["trades"] == 0, "historical validation opened a prohibited short"

        # Build deterministic local OHLCV history, then autonomously validate a template strategy.
        rng = np.random.default_rng(7)
        base_time = int(time.time()) - 400 * 60
        for symbol in agent.cfg["symbols"]:
            price = 1000.0
            for index in range(400):
                price += 0.3 + rng.normal(0, 1.5)
                agent.db.ex(
                    "INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",
                    (symbol, base_time + index * 60, price - 1, price + 2, price - 2, price, 5000),
                )
        agent.nightly_training()
        agent.load_strategies()
        assert agent.strategies, "validated template was not autonomously promoted"
        folds = agent._walk_forward_slices(agent.frame("TCS", 400))
        assert len(folds) == 3 and all(len(test) == 60 for _, test in folds), "walk-forward training folds were not constructed"

        # Stored source is quarantined, never executed.
        agent.db.ex(
            "INSERT OR REPLACE INTO strategies(sid,json,score,status,gen,parent,hash,created)VALUES(?,?,?,?,?,?,?,?)",
            ("legacy_code", '{"code":"raise RuntimeError(\\"must not run\\")"}', 99, "LIVE_APPROVED", 0, None, "x", "test"),
        )
        agent.load_strategies()
        assert agent.db.q("SELECT status FROM strategies WHERE sid='legacy_code'")[0][0] == "QUARANTINED"
        agent.db.ex(
            "INSERT OR REPLACE INTO strategies(sid,json,score,status,gen,parent,hash,created)VALUES(?,?,?,?,?,?,?,?)",
            ("legacy_schema", '{"template":"core","params":{"ema_fast":9}}', 99, "LIVE_APPROVED", 0, None, "x", "test"),
        )
        agent.load_strategies()
        assert agent.db.q("SELECT status FROM strategies WHERE sid='legacy_schema'")[0][0] == "QUARANTINED"

        # Autonomous BUY -> broker-managed bracket -> autonomous SELL at target.
        agent.broker.set_px("TCS", 1100.0)
        supporter = [(agent.strategies[0][0], agent.strategies[0][2], 1.0)]
        stop, target, bracket = bracket_from_supporters(agent.frame("TCS"), 1100.0, supporter)
        assert bracket["sl_atr"] == agent.strategies[0][2]["sl_atr"] and bracket["tp_atr"] == agent.strategies[0][2]["tp_atr"] and stop < 1100.0 < target, "live bracket diverged from approved strategy parameters"
        agent._act("TCS", 1100.0, agent.frame("TCS"), votes=1, supporters=supporter)
        first = agent.oms.positions["TCS"]
        assert first["qty"] > 0 and first["sl"] < first["avg"] < first["tp"]
        agent.broker.set_px("TCS", first["tp"] + 5.0)
        agent.oms.mark("TCS", first["tp"] + 5.0)
        assert not agent.oms.positions.get("TCS"), "target exit did not flatten the paper position"

        # Opposite signal automatically sells an agent-owned long; it does not open a naked short.
        agent.broker.set_px("RELIANCE", 1500.0)
        agent._act("RELIANCE", 1500.0, agent.frame("RELIANCE"), votes=1, supporters=supporter)
        assert agent.oms.positions.get("RELIANCE")
        agent.broker.set_px("RELIANCE", 1510.0)
        agent._act("RELIANCE", 1510.0, agent.frame("RELIANCE"), votes=-1)
        assert not agent.oms.positions.get("RELIANCE"), "opposite signal did not sell the long"

        # --- v3 hardening & accuracy suite ---
        from ox.features import black_scholes as _bs, volume_profile as _vp_fn, avwap as _avwap
        _put = _bs(100.0, 100.0, 0.1, 0.065, 0.2, kind="PE")
        _call = _bs(100.0, 100.0, 0.1, 0.065, 0.2, kind="CE")
        _parity = _call["px"] - _put["px"] - (100.0 - 100.0 * math.exp(-0.065 * 0.1))
        assert abs(_parity) < 1e-6, f"Black-Scholes broke put-call parity: {_parity}"
        _vp_res = _vp_fn(np.full(40, 100.5), np.full(40, 99.5), np.full(40, 100.0), np.full(40, 10.0))
        assert _vp_res["val"] <= _vp_res["poc"] <= _vp_res["vah"], "volume-profile value area mis-ordered"
        assert np.isnan(_avwap(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]), anchor_idx=1)[0]), "avwap pre-anchor must be NaN"

        def _refresh_until_flow_ready(sym: str) -> None:
            # Paper depth pulses its bid size, so the displayed-flow delta
            # oscillates with the snapshot phase. Republish until the book is
            # entry-supportive so these fixtures test THEIR gate, not luck.
            for _ in range(8):
                agent.broker.set_px(sym, agent.broker.ltp(sym))
                flow = agent.broker.order_flow(sym)
                if flow is not None and flow.ready and flow.long_entry:
                    return
            raise AssertionError(f"paper depth never became entry-supportive for {sym}")

        # Ensemble quorum blocks a lone weak vote when the fraction gate is on.
        _saved_fraction = agent.cfg["execution"]["min_vote_fraction"]
        agent.cfg["execution"]["min_vote_fraction"] = 0.5
        agent.broker.set_px("TCS", 1100.0)
        _refresh_until_flow_ready("TCS")
        agent._act("TCS", 1100.0, agent.frame("TCS"), votes=0.1, supporters=supporter)
        assert not agent.oms.positions.get("TCS"), "ensemble quorum did not block a weak vote"
        agent.cfg["execution"]["min_vote_fraction"] = _saved_fraction

        # Partial entry fill: residual leg cancelled, local qty == broker qty.
        agent.broker.partial_entry_once = True
        _refresh_until_flow_ready("TCS")
        agent._act("TCS", 1100.0, agent.frame("TCS"), votes=1, supporters=supporter)
        _partial = agent.oms.positions.get("TCS")
        assert _partial and _partial["qty"] >= 1, "partial entry did not open a reduced position"
        assert int(agent.broker.pos["TCS"]["qty"]) == int(_partial["qty"]), "partial entry left broker/local qty mismatch"
        agent.oms.close("TCS", "SMOKE_PARTIAL_ENTRY", force=True)
        agent.broker.partial_entry_once = False

        # Partial exit fill: retry completes the exit; nothing left unprotected.
        agent.broker.partial_exit_once = True
        agent.broker.set_px("INFY", 1400.0)
        _refresh_until_flow_ready("INFY")
        agent._act("INFY", 1400.0, agent.frame("INFY"), votes=1, supporters=supporter)
        assert agent.oms.positions.get("INFY"), "partial-exit fixture did not open INFY"
        agent.oms.close("INFY", "SMOKE_PARTIAL_EXIT", force=True)
        assert not agent.oms.positions.get("INFY"), "partial exit retry did not fully close"
        assert int(agent.broker.pos.get("INFY", {}).get("qty", 0)) == 0, "partial exit left broker quantity behind"
        agent.broker.partial_exit_once = False

        # Quote snapshot supplies day-cumulative volume for true candle volumes.
        _snap = agent.broker.quote_snapshot(["TCS"])
        assert _snap.get("TCS", {}).get("volume", 0) > 0, "paper quote snapshot produced no volume"
        agent._apply_volumes()

        # Kill switch confirms flattening of agent-owned positions and writes only inside the temporary test root.
        agent.broker.set_px("INFY", 1400.0)
        _refresh_until_flow_ready("INFY")
        agent._act("INFY", 1400.0, agent.frame("INFY"), votes=1, supporters=supporter)
        agent.oms.kill_switch("smoke test")
        assert not agent.oms.positions and (Path(directory) / "KILL.flag").exists()
        assert agent.db.q("SELECT COUNT(*) FROM trades")[0][0] >= 3
    print("PASS: secure autonomous buy/sell, broker-side brackets, strategy quarantine, and kill switch")


def run_validate_online() -> None:
    from ox.validate_online import run as _run
    _run()


def handle_promax(command: str, args: list) -> None:
    """Multi-agent (PRIME) CLI: run orchestrator, list/decide order intents."""
    import asyncio

    def _db():
        from ox.core import DB
        return DB(PROJECT_ROOT / "promax.db")

    if command == "promax":
        from ox.agents.orchestrator import AgentOrchestrator
        seconds = float(args[0]) if args else None
        orch = AgentOrchestrator(config_path=PROJECT_ROOT / "config_promax.yaml")
        try:
            asyncio.run(orch.run_forever(max_seconds=seconds))
        except KeyboardInterrupt:
            pass
        return

    if command == "report":
        from ox.agents.capital_allocator import CapitalAllocator
        from ox.agents.base import SharedDataBus
        from ox.agents.tearsheet import agent_report
        from ox.core import DB
        import yaml as _yaml
        cfg = _yaml.safe_load((PROJECT_ROOT / "config_promax.yaml").read_text(encoding="utf-8"))
        db = DB(PROJECT_ROOT / "promax.db")
        alloc = CapitalAllocator(SharedDataBus(), cfg.get("capital", {}), db=db)
        target = args[0] if args else None
        if target:
            print(agent_report(alloc, target))
        else:
            print(agent_report(alloc, None))
            for agent_id in (cfg.get("capital", {}).get("weights") or {}):
                rows = alloc.closed_trades(agent_id, limit=1000)
                if rows:
                    print("\n" + agent_report(alloc, agent_id))
        return

    if command == "validate-online":
        run_validate_online()
        return

    if command == "live-test":
        from ox.live_test import run as _run
        seconds = float(args[0]) if args else 0.0
        raise SystemExit(_run(prime_seconds=seconds))

    if command == "odds-month":
        from ox.month_odds import run as _run
        capital = float(args[0]) if args else 5000.0
        _run(capital=capital)
        return

    if command == "promax-smoke":
        import os
        import shutil
        import tempfile
        os.environ["OX_PROMAX_AUTO_APPROVE"] = "1"
        from ox.agents.orchestrator import AgentOrchestrator
        tmp = tempfile.mkdtemp(prefix="promax_smoke_")
        try:
            db_path = str(Path(tmp) / "smoke.db")
            orch = AgentOrchestrator(config_path=PROJECT_ROOT / "config_promax.yaml", db_path=db_path)

            async def _run():
                await orch.start_all()
                try:
                    await asyncio.wait_for(orch.stop_event.wait(), timeout=25)
                except asyncio.TimeoutError:
                    pass
                return orch.get_system_status()  # snapshot while still running

            status = asyncio.run(_run())
            asyncio.run(orch.stop_all())
            print(f"agents={len(status['agents'])} ticks={status['data_pump_ticks']} "
                  f"fills={status['execution']['fills']} closes={status['execution']['closes']} "
                  f"rejected={status['execution']['rejected']} "
                  f"equity={status['capital']['equity']:.2f}")
            assert status["data_pump_ticks"] > 0, "data pump never ticked"
            assert status["agents"], "no agents started"
            orch.db.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # Windows may hold WAL briefly
        print("PROMAX SMOKE OK")
        return

    if command == "promax-kill":
        (PROJECT_ROOT / "promax_kill.flag").write_text("manual CLI promax-kill\n", encoding="utf-8")
        print("promax_kill.flag written — orchestrator will halt on next monitor pass")
        return

    if command == "intents":
        from ox.agents.approvals import ApprovalGateway
        gateway = ApprovalGateway(_db())
        gateway.expire_stale()
        rows = gateway.list_intents(args[0] if args else "PENDING", limit=20)
        if not rows:
            print(f"No {args[0] if args else 'PENDING'} intents")
            return
        for r in rows:
            print(f"{r['iid']} [{r['status']}] {r['agent']} {r['action'].upper()} "
                  f"{r['symbol']} qty={r['quantity'] if 'quantity' in r else r['qty']} @ {r['price']} "
                  f"lev={r['leverage']} reason={r['reason']}")
            print(f"    decide: python run.py ok {r['iid']}   |   python run.py deny {r['iid']}")
        return

    if command in {"ok", "deny"}:
        if not args:
            raise SystemExit(f"Usage: python run.py {command} <iid>")
        from ox.agents.approvals import ApprovalGateway
        gateway = ApprovalGateway(_db())
        decided = gateway.decide(args[0], approve=(command == "ok"), by="cli")
        print(f"intent {args[0]} {'approved' if command == 'ok' else 'denied'}"
              if decided else f"intent {args[0]} not decided (missing or not PENDING)")
        return

    if command == "promax-status":
        from ox.agents.approvals import ApprovalGateway
        db = _db()
        gateway = ApprovalGateway(db)
        gateway.expire_stale()
        print("--- PENDING INTENTS ---")
        for r in gateway.list_intents("PENDING", limit=10):
            print(f"{r['iid']} {r['agent']} {r['action'].upper()} {r['symbol']} "
                  f"qty={r['qty']} @ {r['price']} lev={r['leverage']}")
        print("\n--- CLOSED TRADES (last 10) ---")
        for row in db.q("SELECT agent,symbol,side,round(qty,4),round(entry_price,2),"
                        "round(exit_price,2),round(pnl,2),round(leverage,2),closed "
                        "FROM promax_trades ORDER BY ptid DESC LIMIT 10"):
            print(row)
        print("\n--- LADDER LEVELS ---")
        for row in db.q("SELECT agent, level, updated FROM ladder_levels"):
            print(row)
        return


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "smoketest":
        run_smoketest()
        return

    if command == "preflight":
        # Zero-credential readiness: no keys, no config flips, no network
        # requirement.  Exit 1 when any check FAILs (WARN/SKIP never fail).
        from ox.preflight import main as _preflight_main
        raise SystemExit(_preflight_main(sys.argv[2:]))

    if command == "venue-check":
        from ox.venue_check import check_venue
        import yaml
        from ox.core import DB
        from ox.brokers import make_broker

        state_dir = PROJECT_ROOT / "state"
        state_dir.mkdir(exist_ok=True)
        db = DB(state_dir / "venue_check.db")
        builders: dict[str, object] = {}

        if os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_TOKEN"):
            raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))

            def _dhan(raw=raw):
                cfg = dict(raw)
                cfg["mode"] = "live"
                cfg["platform"] = "dhan"
                return make_broker(cfg, db)

            builders["dhan"] = _dhan

        if os.getenv("CHOICE_USER_ID") and os.getenv("CHOICE_PASSWORD"):
            raw = yaml.safe_load((PROJECT_ROOT / "config_choice.yaml").read_text(encoding="utf-8"))

            def _choice(raw=raw):
                cfg = dict(raw)
                cfg["mode"] = "live"
                cfg["platform"] = "choice"
                return make_broker(cfg, db)

            builders["choice"] = _choice

        if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
            raw = yaml.safe_load((PROJECT_ROOT / "config_promax.yaml").read_text(encoding="utf-8"))

            def _binance(raw=raw):
                cfg = dict(raw)
                cfg["mode"] = "live"
                cfg["platform"] = "crypto"
                return make_broker(cfg, db)

            builders["binance"] = _binance

        if not builders:
            print("No venue credentials found in the environment; nothing to verify.")
            print("Run 'bash scripts/setup-live.sh' first, or export the venue keys.")
            raise SystemExit(2)

        failed = False
        for name, builder in builders.items():
            status, detail = check_venue(name, builder)  # type: ignore[arg-type]
            print(f"VENUE {name:8s} {status:4s} {detail}")
            failed = failed or status == "FAIL"
        raise SystemExit(2 if failed else 0)

    if command == "track-record":
        from ox.track_record import main as _track_main
        raise SystemExit(_track_main(sys.argv[2:]))

    # ── multi-agent (PRIME) commands ─────────────────────────────────
    if command in {"promax", "intents", "ok", "deny", "promax-status", "promax-kill",
                   "promax-smoke", "report", "validate-online", "odds-month", "live-test"}:
        handle_promax(command, sys.argv[2:])
        return

    from ox.agent import Agent

    # run/status accept an optional config path (e.g. config_choice.yaml for a
    # Choice India session, which lives on its own db); subcommands such as
    # approve keep argv[2] as their own argument, so only commands with a free
    # argv[2] honor a trailing config.
    config_arg = str(PROJECT_ROOT / "config.yaml")
    if command in ("run", "status") and len(sys.argv) > 2:
        config_arg = sys.argv[2]
    agent = Agent(config_arg)
    if command == "train":
        agent.nightly_training()
    elif command == "status":
        print("\n--- VALIDATED STRATEGIES ---")
        for row in agent.db.q("SELECT sid, round(score,3), status, approved_at FROM strategies ORDER BY score DESC LIMIT 10"):
            print(row)
        pending = agent.db.q("SELECT sid, round(score,3) FROM strategies WHERE status='PENDING_APPROVAL' ORDER BY score DESC")
        print("\n--- PENDING APPROVAL ---")
        for row in pending:
            print(row)
        print("\n--- AGENT-OWNED POSITIONS ---")
        print(agent.db.q("SELECT * FROM positions"))
        print("\n--- RECENT TRADES ---")
        for row in agent.db.q("SELECT tid, sym, side, qty, inpx, outpx, pnl, charges, exit_reason FROM trades ORDER BY tid DESC LIMIT 5"):
            print(row)
    elif command == "kill":
        # Authenticate and restore state first, so the kill action targets only recorded agent positions.
        if agent.comp.daily_auth(agent.broker):
            try:
                agent.oms.restore()
            except Exception:
                pass
            agent.oms.kill_switch("manual CLI kill command")
    elif command == "approve":
        sid = sys.argv[2] if len(sys.argv) > 2 else ""
        if not sid:
            pending = agent.db.q("SELECT sid, round(score,3) FROM strategies WHERE status='PENDING_APPROVAL' ORDER BY score DESC")
            print(pending if pending else "No strategies pending approval")
            print("Usage: python run.py approve <sid>")
        else:
            print("approved" if agent.brain.approve(sid) else f"{sid} not found in PENDING_APPROVAL")
    elif command == "run":
        agent.run_forever()
    else:
        raise SystemExit(
            "Usage: python run.py [run [config.yaml]|train|status [config.yaml]|approve|kill|smoketest|track-record]\n"
            "       python run.py [preflight|live-test <seconds>|validate-online]\n"
            "       python run.py [promax [seconds]|promax-smoke|promax-status|"
            "promax-kill|intents [STATUS]|ok <iid>|deny <iid>]"
        )


if __name__ == "__main__":
    main()
