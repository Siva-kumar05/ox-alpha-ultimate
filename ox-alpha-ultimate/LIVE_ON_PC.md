# OX-ALPHA LIVE ON YOUR PC — COMPLETE RUNBOOK

Verified against this checkout (2026-09-04): `config.yaml`, `ox/orderflow.py`
(`OrderFlowReplayValidator`), `ox/agent.py` boot gates, `ox/core.py` defaults,
`run.py`, `START_DAILY.ps1`, `SET_STATIC_IP.ps1`, `verify_ip.py`,
`RUNBOOKS.md`, `README.md`.

> Real money, NSE equities, long-only intraday. SEBI's retail-algo framework
> applies to this broker model from 1 Apr 2026 (see README). Nothing here is
> investment advice or a guarantee of returns. Use a **dedicated Dhan intraday
> account with no manual positions or working orders** — the agent halts on any
> broker/database mismatch rather than touching a position it does not own.

---

## Table of contents

1. [Your machine's three hard facts](#0-your-machine-s-three-hard-facts)
2. [Get the code onto this PC and set up](#1-get-the-code-onto-this-pc-and-set-up)
3. [Dhan account, IP, credentials (one-time)](#2-dhan-account-ip-credentials-one-time)
4. [The depth-replay evidence gate](#3-the-depth-replay-evidence-gate)
5. [First supervised session — collection only](#4-first-supervised-session--collection-only)
6. [Daily routine](#5-daily-routine)
7. [Going live with money](#6-going-live-with-money)
8. [Incidents — stop/halt checklist](#7-incidents--stophalt-checklist)
9. [Exact day-one command sequence](#8-exact-day-one-command-sequence)

---

## 0. Your machine's three hard facts

1. **Your PC's current egress IP is `49.43.232.235`; the config whitelist is
   `13.207.244.242`** (your AWS VPS). `config.yaml` ships with
   `ip_whitelist: [13.207.244.242]` and `ip_whitelist_env: DHAN_STATIC_IP`.
   Dhan enforces a server-side static-IP allowlist and the agent's live boot
   halts on mismatch (`Egress IP … is not in configured allowlist`). Home ISP
   IPs are dynamic, so a Dhan-live session from this PC means re-registering
   the new address in the Dhan portal **and** re-exporting `DHAN_STATIC_IP`
   after every lease change. The alternative that needs no re-registration is
   running on the AWS VPS (`13.207.244.242`, already in the config) and
   driving it remotely. The steps below assume **this PC**, Dhan venue.

2. **The project currently lives inside OneDrive**
   (`C:\Users\siva kumar\OneDrive\cursor_projects\Default Project\ox-alpha-ultimate`).
   Never run live from a cloud-synced folder — file locks and conflict copies
   can corrupt `oxalpha.db` or `KILL.flag` mid-session. Move to a plain path
   such as `C:\ox-alpha-src\ox-alpha-ultimate`.

3. **Git reality:** all fixes this tree needed — the quality/correctness
   pass (including the `run.py` smoketest fix), the `KILL.flag`
   history-ingestion tolerance in `ox/agent.py` with its regression tests,
   and this runbook — are **committed and pushed to `master`/`main`**. A
   fresh clone is therefore safe and contains the working smoketest and the
   ingestion fix. Clone and copy are both valid ways to get the code; the
   steps below use a copy of this local checkout.

---

## 1. Get the code onto this PC and set up

Two equivalent options — clone from GitHub (now that the fixes are pushed):

```powershell
cd C:\
git clone https://github.com/Siva-kumar05/ox-alpha-ultimate.git ox-alpha-src
cd C:\ox-alpha-src\ox-alpha-ultimate
```

…or copy this local working tree (same result, works offline):

```powershell
robocopy "C:\Users\siva kumar\OneDrive\cursor_projects\Default Project\ox-alpha-ultimate" `
         "C:\ox-alpha-src\ox-alpha-ultimate" /E /NFL /NDL /NJH /NJS
cd C:\ox-alpha-src\ox-alpha-ultimate
```

Then set up the environment (either way):

```powershell
# Virtual env + dependencies (system Python 3.14 already passes the suite)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Baseline sanity — everything must pass before anything live
python run.py smoketest          # prints: PASS: secure autonomous buy/sell, ...
python run.py preflight          # zero-credential check; egress FAILs until step 2 of §2
```

- The OneDrive copy stays behind as a cold backup only — never run it.
- Only ever run **one** instance against the same `oxalpha.db`.
- A fresh `git clone` of the repo is safe — the pushed tree includes the
  smoketest fix and the ingestion tolerance, so `python run.py smoketest`
  passes on a clean checkout.

---

## 2. Dhan account, IP, credentials (one-time)

### 2.1 Broker account

Open a **dedicated Dhan intraday account** for this agent. No manual
positions, no manual working orders. Dhan's own docs govern the static-IP and
Super Order / Full Market Depth contracts this agent uses (links in README).

### 2.2 Static IP

1. Register this PC's current public IP in the Dhan portal (Profile →
   DhanHQ Trading APIs → static IP). Today that address is `49.43.232.235`.
2. Set the same address in the host environment — **never** in `config.yaml`:

```powershell
[Environment]::SetEnvironmentVariable("DHAN_STATIC_IP", "49.43.232.235", "User")
```

> Do **not** use `verify_ip.py` on this PC for the final say — it hardcodes
> the VPS IP `13.207.244.242`. `START_DAILY.ps1` happens to call it; ignore
> its MISMATCH lines. The authoritative check is the agent boot's own IP check
> against the `DHAN_STATIC_IP` in the launching session.

### 2.3 Audit key (required in live mode)

```powershell
[Environment]::SetEnvironmentVariable("OX_AUDIT_KEY", "a-long-random-secret-of-at-least-32-chars", "User")
```

`OX_AUDIT_KEY` feeds the HMAC audit chain. Live boot refuses to start without
it and verifies the existing chain. Keep it forever — changing it invalidates
the historic chain and blocks live boot by design.

### 2.4 Daily token (recommended; no PIN/TOTP on this PC)

Each morning, generate a 24-hour access token in the Dhan web console
(Profile → DhanHQ Trading APIs → Generate Access Token; the PIN/TOTP prompt
happens in the **browser**, so the 2FA seed never touches this machine), then
launch with `start-daily.cmd` (or `START_DAILY.ps1`). The script masks the
paste, validates the token against the read-only `/fundlimit` endpoint before
booting, keeps it process-scope only, and scrubs it when the agent exits.

Unattended auto-renewal is optional: set `DHAN_PIN` and `DHAN_TOTP_SECRET` in
host env and `secret_rotation` auto-rotates the token. The daily paste is the
safer posture.

### 2.5 Re-verify

```powershell
python run.py preflight   # all green except anything needing a live token (expected pre-launch)
```

---

## 3. The depth-replay evidence gate

### 3.1 Why the boot halts on a fresh install

`config.yaml` ships with:

```yaml
order_flow:
  enabled: true
  primary: true            # L2 depth is the primary entry gate
  require_replay_validation: true   # live boot gate (see below)
  replay_min_signals: 30
  replay_horizon_candles: 5
  replay_min_hit_rate: 0.5
  replay_min_mean_return_bps: 0.0
  replay_max_records: 10000
```

At every **live** boot, `Agent.boot()` (when `mode=live`, `primary=true`,
`require_replay_validation=true`) runs `OrderFlowReplayValidator.evaluate()`
(`ox/orderflow.py`), stores the result in kv key `orderflow_replay_validation`,
and **halts** unless it passed:

> `Primary order-flow gate lacks sufficient positive retained Dhan depth
> replay evidence`

This is a boot-time *gate study*, not an execution backtest: it only checks
whether real recorded `DHAN_DEPTH20` entry snapshots were followed by a
favourable recorded candle move.

### 3.2 Exactly what the validator measures

1. Reads `orderflow` table rows with `source='DHAN_DEPTH20' AND
   entry_signal=1` — real live depth snapshots where the book was
   admission-ready (supportive: spread ≤ `max_spread_bps` 12.0, side notional
   ≥ `min_side_notional` 50 000, imbalance/pressure floors met, streak ≥
   `min_positive_streak` 3, liquidity ≥ `min_liquidity_score` 0.6).
2. Takes the newest `replay_max_records` (10 000) such rows, dedupes to one
   observation per symbol per minute bucket, and looks forward
   `replay_horizon_candles` (5) one-minute candles.
3. Computes the move in bps from the entry midpoint to the 5th candle close.
4. **Discards** rows whose forward candles are missing or gapped
   (> 2× `timeframe_sec` between candle times — overnight, holidays, feed
   gaps never become favourable observations).
5. Passes only when **all three** hold:

| Threshold | Config key | Shipped value |
|---|---|---|
| Usable forward-looking samples | `order_flow.replay_min_signals` | `30` |
| Share of samples with a positive forward move | `order_flow.replay_min_hit_rate` | `0.5` |
| Mean forward move | `order_flow.replay_min_mean_return_bps` | `0.0` |

### 3.3 How evidence is collected

The qualifying rows only accrue while the agent runs **live** through market
hours with the Dhan depth feed connected (`DhanBroker.start_orderflow()` feeds
`"DHAN_DEPTH20"` into the engine, which persists ready observations to the
`orderflow` table). Paper depth is synthetic and explicitly never accepted
(`source` is not `DHAN_DEPTH20`). A brand-new install has **zero** rows, so
the first live window must run with the gate relaxed (below). Keep collecting
until the study passes; weakening the thresholds to force it through would
defeat the point of the gate.

### 3.4 Week-1 collection-window config

```yaml
# config.yaml — WEEK-1 COLLECTION WINDOW (revert in §6)
mode: live
platform: dhan
capital: 100000
risk:
  risk_per_trade_pct: 0.25
  daily_loss_cap_abs: 2000
  max_notional_per_trade: 50000
order_flow:
  require_replay_validation: false     # TEMPORARY — flip back to true in §6
```

### 3.5 How you confirm the gate is satisfied

Run weekly after market close from `C:\ox-alpha-src\ox-alpha-ultimate`:

```powershell
.venv\Scripts\Activate.ps1
python -c "from ox.core import DB; print(DB('oxalpha.db').kv_get('orderflow_replay_validation'))"
# Expect, when ready:
# {'kind': 'L2_GATE_REPLAY_NOT_EXECUTION_BACKTEST', 'source': 'DHAN_DEPTH20',
#  'samples': >=30, 'hit_rate': >=0.5, 'mean_return_bps': >=0.0, 'passed': True}
```

Raw accrual check:

```powershell
python -c "from ox.core import DB; db=DB('oxalpha.db'); print(db.q(\"SELECT COUNT(*), COUNT(DISTINCT sym) FROM orderflow WHERE source='DHAN_DEPTH20' AND entry_signal=1\"))"
```

If after 1–2 weeks `samples < 30` or `hit_rate < 0.5`, keep collecting — that
is the honest signal that the L2 gate has not yet shown predictive value on
your names.

---

## 4. First supervised session — collection only

Two independent ways to keep the first window supervised while evidence
accrues; use both.

1. **Do not approve any strategy.** `training.require_human_approval: true`
   means newly trained strategies land in `PENDING_APPROVAL`, and
   `load_strategies()` only loads `LIVE_APPROVED` rows. With nothing approved
   the agent boots into **observation mode**: it fetches history, streams the
   depth feed, records `DHAN_DEPTH20` rows, and blocks every entry
   (`TREND_CONFIRMATION_MISSING`). Exactly what week 1 should be.
2. **Keep the tiny risk budget** from §3.4 even if you do approve something —
   per-trade risk `0.25%` of `capital: 100000` ≈ ₹250/trade, daily cap ₹2 000.
   Entries additionally need each symbol's warm-up
   (`order_flow.min_observations: 300`) and a genuinely supportive live book
   before the OMS may act.

Symbols: keep at least **three** (training needs `training.min_symbols: 3` ×
3 walk-forward folds). The shipped set RELIANCE / HDFCBANK / INFY / TCS /
ICICIBANK is fine; trim to three liquid names if you want less surface.

Day-one launch (fresh terminal, secrets live only in that session):

```powershell
cd C:\ox-alpha-src\ox-alpha-ultimate
.venv\Scripts\Activate.ps1
start-daily.cmd     # client ID prompt + masked token paste -> /fundlimit check -> python run.py run
```

Watch the boot-gate order in the log: audit chain (`OX_AUDIT_KEY`) → static
IP → market holidays (the 2026 list is populated; **2026-09-14 is Ganesh
Chaturthi — markets shut**) → depth-feed connect → broker reconcile + history
fetch → auto-training → observation loop.

```powershell
python run.py status     # VALIDATED STRATEGIES / PENDING APPROVAL / positions / trades
start-dashboard.cmd      # browser dashboard http://127.0.0.1:8501
```

Keep the machine awake and online 09:15–15:30 IST — disable sleep/hibernate,
set the lid to do nothing. The square-off only fires while the loop runs; a
laptop that sleeps at 14:00 leaves a position open until the next boot.

---

## 5. Daily routine

| When | Action |
|---|---|
| Before 09:15 IST | `start-daily.cmd` — paste a fresh 24 h token from the Dhan web console. Keep the console open (token is process-scope, scrubbed on exit). |
| 09:15–15:30 | Machine awake, lid never closed, network up. Monitor with the dashboard and `python run.py status`. |
| 14:45 | `entry_cutoff` — no new entries after this; protective exits, opposite-signal exits, and square-off still run. |
| 15:15 | Automatic square-off (`squareoff`) of agent-owned longs. |
| 15:30 | EOD equity/PnL and stats written (`market_close`). |
| Any time | `python run.py status` · emergency `python run.py kill` or create `KILL.flag`. |
| Weekly | Replay check (§3.5), `python run.py preflight`, review strategy P&L via `python run.py status`. |

The defaults that protect you and should stay put: `daily_loss_cap_pct: 2.0`,
`daily_loss_cap_abs` (₹10 000 full-size), `max_positions: 5`,
`risk_per_trade_pct: 0.5` full-size, `allow_short: false` (long-only),
`tick_seconds: 3` (rate-limit headroom).

---

## 6. Going live with money (after the replay study passes)

```powershell
# 1. Confirm kv orderflow_replay_validation shows 'passed': True (§3.5)
# 2. Revert config.yaml to the shipped operating values:
#      order_flow.require_replay_validation: true
#      capital: 500000
#      risk.risk_per_trade_pct: 0.5
#      risk.daily_loss_cap_abs: 10000
#      risk.max_notional_per_trade: 200000
# 3. Approve the strategies you want — autonomous capital follows approval:
python run.py approve <sid>          # list candidates first with: python run.py status
# 4. Restart so load_strategies() picks up the LIVE_APPROVED rows:
start-daily.cmd
```

Scale exposure only with a real, retained track record — the agent's own
post-trade analysis (`post_trade_analysis`, `parameter_drift`,
`cost_aware_selection`) exists precisely to tell you when a strategy stops
paying for its costs.

---

## 7. Incidents — stop/halt checklist

### KILL.flag semantics

- Every tick, the agent checks for `KILL.flag`; when present it triggers the
  OMS kill switch (flattens only **recorded agent-owned** positions) and
  halts the loop.
- **Boot refuses to start while `KILL.flag` exists** — that halt is a
  question to answer, never noise to restart around.
- Procedure:
  1. Read the reason: `Get-Content KILL.flag` (this repo previously halted on
     a non-numeric RELIANCE history candle; that class of failure is now
     skipped-with-logging in `ox/agent.py` and only systemic corruption halts).
  2. `python run.py status` — confirm positions are flat.
  3. Investigate the cause (broker state, data error, IP change).
  4. Only then `Remove-Item KILL.flag` and restart via `start-daily.cmd`.

### Egress-IP rotation (home ISP)

Symptom: boot halts with an egress/allowlist message. Fix:

1. Get the current address: `https://api.ipify.org`.
2. Register it in the Dhan portal.
3. `[Environment]::SetEnvironmentVariable("DHAN_STATIC_IP", "<new-ip>", "User")`
4. Restart with `start-daily.cmd`.

**Dhan portal limit (seen in the live UI): "You can re-set your IP Address in
the interval of 7 days."** If the home ISP rotates the IP and the new address
is not already registered, Dhan rejects every API call (403) for up to 7
days — the agent stays halted and cannot trade from home during that window.
Mitigations: confirm the ISP provides a stable/static IP before running real
money from home, or keep the live session on the registered VPS
(`13.207.244.242`, IP Address 1) and use this PC for paper/dev. The portal
supports both addresses simultaneously (IP Address 2 = the PC's
`49.43.232.235` as of 2026-09-04).

### Dhan token rotation

Symptom: authentication halt mid-session (24 h token expired or revoked).
Fix: regenerate the token in the Dhan web console, restart with
`start-daily.cmd`. Never paste tokens into config files or chat.

### Circuit breaker / risk halt

The agent self-halts on persistent negative performance (circuit breaker) or
daily-loss-cap breach. `python run.py status` and the dashboard show the
reason; do not raise the caps to restart — investigate the strategy first.

### General rule

Every halt is fail-closed by design (unconfirmed order, stale/thin book,
audit failure, depth feed down with `primary: true`, reconcile mismatch).
Read the halt, fix the cause, then restart. Never run two instances against
the same database.

---

## 8. Exact day-one command sequence

```powershell
# ── ONE-TIME (do once) ─────────────────────────────────────────────────────
# Get the code per §1 (clone from GitHub or copy this checkout), then:
cd C:\ox-alpha-src\ox-alpha-ultimate
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
[Environment]::SetEnvironmentVariable("DHAN_STATIC_IP", "49.43.232.235", "User")
[Environment]::SetEnvironmentVariable("OX_AUDIT_KEY", "<32+ char random secret>", "User")

# Edit config.yaml by hand:
#   mode: live
#   platform: dhan
#   capital: 100000
#   risk.risk_per_trade_pct: 0.25
#   risk.daily_loss_cap_abs: 2000
#   risk.max_notional_per_trade: 50000
#   order_flow.require_replay_validation: false

# ── VERIFY, THEN LAUNCH (Monday, ~09:00 IST) ───────────────────────────────
python run.py smoketest
python run.py preflight
start-daily.cmd                 # client ID + today's token; agent boots live in observation mode

# ── SUPERVISE ──────────────────────────────────────────────────────────────
python run.py status            # expect PENDING APPROVAL rows, zero positions
start-dashboard.cmd             # watch orderflow / admission on http://127.0.0.1:8501

# ── WEEKLY ─────────────────────────────────────────────────────────────────
python -c "from ox.core import DB; print(DB('oxalpha.db').kv_get('orderflow_replay_validation'))"
# once 'passed': True -> revert config per §6, approve strategies, restart, scale slowly
```

### Stop/halt checklist (end of any session or incident)

- [ ] Positions flat — confirmed via `python run.py status`
- [ ] Agent stopped cleanly (Ctrl+C runs the graceful-shutdown hooks in order)
- [ ] Session secrets scrubbed (`START_DAILY.ps1` does this on exit)
- [ ] `KILL.flag` never deleted before its cause is understood
- [ ] No second instance running against the same `oxalpha.db`
- [ ] Nothing run from the OneDrive copy
