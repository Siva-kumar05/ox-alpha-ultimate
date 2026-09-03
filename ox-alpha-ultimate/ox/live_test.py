"""Live-data test runner for the Dhan adapter (read-only endpoints only).

Never places/amends/cancels orders.  Verifies, in order:
  0.  public egress IP vs the token's IP whitelist expectation
  1.  funds/limit read            (GET  /fundlimit)
  2.  LTP batch for the universe  (POST /marketfeed/ltp)
  3.  full quote + depth fields   (POST /marketfeed/quote)
  4.  intraday chart candles      (POST /charts/intraday)
  5.  the debate panel + indicators computed on REAL candles
  6.  (optional) a short PRIME paper session fed by live quotes

Credentials come exclusively from the environment (DHAN_TOKEN,
DHAN_CLIENT_ID); nothing is written to source.  Every failure prints an
actionable message and the run exits non-zero, so it can gate deploys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def run(prime_seconds: float = 0.0) -> int:
    token = os.environ.get("DHAN_TOKEN", os.environ.get("DHAN_ACCESS_TOKEN", "")).strip()
    client_id = os.environ.get("DHAN_CLIENT_ID", "").strip()
    if not token or not client_id:
        print("LIVE-TEST: set DHAN_TOKEN and DHAN_CLIENT_ID in the environment "
              "(e.g. source ~/.ox_secrets.env) — nothing run.")
        return 2

    import yaml
    from ox.agents.debate import DebatePanel
    from ox.brokers import DhanBroker
    from ox.core import DB
    from ox import indicators as I
    import numpy as np

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    cfg["mode"] = "live"
    cfg["platform"] = "dhan"
    db = DB(Path("state") / "live_test.db")
    broker = DhanBroker(cfg, db)
    broker._set_token(token)

    failures = 0

    # 0. egress IP -----------------------------------------------------------
    print("[0] egress IP check")
    try:
        from ox.ssrf import safe_fetch
        ip = safe_fetch("https://api.ipify.org", timeout=8).decode()
        expected = (cfg.get("ip_whitelist") or [None])[0]
        status = "OK" if ip == expected else f"MISMATCH (expect {expected})"
        print(f"  public IP {ip} -> {status}")
        if ip != expected:
            failures += 1
            _fail("Dhan will reject this host; run from the whitelisted machine "
                  "or update the whitelist in the Dhan portal.")
    except Exception as exc:
        failures += 1
        _fail(f"ipify unreachable: {exc}")

    # 1. login-shaped read: funds -------------------------------------------
    print("[1] funds (GET /fundlimit)")
    try:
        funds = broker._request("GET", "/fundlimit")
        avail = funds.get("data", {}).get("availabelBalance") or funds.get("data", {})
        print(f"  OK: {str(avail)[:90]}")
    except Exception as exc:
        failures += 1
        _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    syms = list(cfg.get("symbols", []))[:6]

    # 2. LTP -----------------------------------------------------------------
    print("[2] LTP batch")
    quotes = {}
    try:
        quotes = broker.ltps(syms)
        for sym, px in list(quotes.items())[:6]:
            print(f"  {sym}: {px}")
        if not quotes:
            failures += 1
            _fail("no quotes returned (check security_map)")
    except Exception as exc:
        failures += 1
        _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    # 3. full quote ----------------------------------------------------------
    print("[3] full quote (OHLC + volume + depth)")
    try:
        snapshot = broker.quote_snapshot(syms[:2])
        for sym, row in list(snapshot.items())[:2]:
            keys = sorted(row.keys())[:8] if isinstance(row, dict) else []
            print(f"  {sym}: fields={keys}")
        if not snapshot:
            failures += 1
            _fail("empty quote snapshot")
    except Exception as exc:
        failures += 1
        _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    # 4. intraday candles ------------------------------------------------------
    print("[4] intraday candles")
    closes_by_sym = {}
    try:
        for sym in syms[:3]:
            hist = broker.hist(sym, 5, 5)
            closes = [float(r[4]) for r in (hist or []) if r and r[4]]
            closes_by_sym[sym] = closes
            print(f"  {sym}: {len(closes)} candles, last={closes[-1] if closes else 'n/a'}")
    except Exception as exc:
        failures += 1
        _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    # 5. pipeline on real data -------------------------------------------------
    print("[5] indicators + debate panel on REAL candles")
    try:
        panel = DebatePanel(state_dir=Path("state") / "promax")
        for sym, closes in closes_by_sym.items():
            if len(closes) < 60:
                print(f"  {sym}: only {len(closes)} closes — skipped")
                continue
            arr = np.asarray(closes, dtype=float)
            ema = I.ema(arr, 20)
            rsi = I.rsi(arr, 14)
            verdict = panel.debate("live_probe", sym, arr)
            print(f"  {sym}: ema20={ema[-1]:.2f} rsi14={rsi[-1]:.1f} "
                  f"debate={verdict['verdict']:+.2f} pass={verdict['pass']}")
    except Exception as exc:
        failures += 1
        _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    # 6. optional PRIME session on live quotes ---------------------------------
    if prime_seconds > 0:
        print(f"[6] PRIME session {prime_seconds:.0f}s on live quotes (paper ledger)")
        try:
            import asyncio
            import tempfile
            os.environ.setdefault("OX_PROMAX_AUTO_APPROVE", "1")
            from ox.agents.orchestrator import AgentOrchestrator
            with tempfile.TemporaryDirectory() as tmp:
                orch = AgentOrchestrator(config_path="config_promax.yaml",
                                         db_path=Path(tmp) / "live.db")
                live_syms = list(quotes) or syms
                async def _feed():
                    await orch.start_all()
                    end = asyncio.get_event_loop().time() + prime_seconds
                    while asyncio.get_event_loop().time() < end:
                        try:
                            q = broker.ltps(live_syms)
                            for s, px in q.items():
                                orch.data_bus.publish(f"market:{s}",
                                                      {"symbol": s, "price": float(px)})
                        except Exception:
                            pass
                        await asyncio.sleep(3)
                    return orch.get_system_status()
                status = asyncio.run(_feed())
                asyncio.run(orch.stop_all())
                print(f"  agents={len(status['agents'])} ticks={status['data_pump_ticks']} "
                      f"fills={status['execution']['fills']}")
        except Exception as exc:
            failures += 1
            _fail(f"{exc.__class__.__name__}: {str(exc)[:160]}")

    db.close()
    print(f"LIVE-TEST: {'PASS' if failures == 0 else f'{failures} failure(s)'} "
          "(read-only endpoints; no orders were placed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run(prime_seconds=float(sys.argv[1]) if len(sys.argv) > 1 else 0.0))
