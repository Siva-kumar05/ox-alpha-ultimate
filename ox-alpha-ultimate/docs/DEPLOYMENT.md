# Deployment, CI/ops, and the real-venue track record

This document covers how the code that would go live is packaged, verified,
supervised, and given an honest *track record* before any live capital is at
risk.  The go/no-go rule is simple: **the agent trades live only after
supervised, real-quote sessions pass cleanly and accumulate in
`track_record/sessions.jsonl`** — read-only Dhan endpoints and a paper ledger,
no orders.

## 1. CI (`.github/workflows/ci.yml`)

Three jobs run on every push/PR to `master`:

- **lint** — `ruff check ox/ tests/ --select F821,F601,F811` (undefined names,
  duplicate keys, redefinitions: the bug classes that actually broke live
  paths in the past).
- **test** — the full offline suite (`python -m pytest tests/ -q`): paper
  broker, scripted HTTP transport, temp databases.  No network, no venue.
- **smoketest** — `python run.py smoketest` (autonomous paper buy/sell,
  broker-side brackets, strategy quarantine, kill switch).

The suite must stay green before a container is built or a session is run.

## 2. Container (Dockerfile)

```bash
docker build -t ox-alpha-ultimate .
# autonomous paper run (default CMD)
docker run --rm -v "$PWD/state:/srv/oxalpha/state" \
  -v "$PWD/backups:/srv/oxalpha/backups" ox-alpha-ultimate
# run any run.py command instead
docker run --rm ox-alpha-ultimate smoketest
docker run --rm ox-alpha-ultimate track-record
# dashboard
docker run --rm -p 8501:8501 ox-alpha-ultimate streamlit run dashboard.py
```

Credentials are **never baked into the image**; they come from `-e` at run
time (`OX_AUDIT_KEY`, `DHAN_CLIENT_ID`, `DHAN_TOKEN`).  The runtime user is
non-root (uid 1000).

## 3. Service supervision (deploy/)

- `deploy/ox-alpha-ultimate.service` — systemd unit running `run.py run` with
  an env file at `/etc/ox-alpha/ox.env`.  `Restart=on-failure` is safe because
  boot refuses to trade while `KILL.flag` exists: a halted agent restarts into
  a halted state, it does not re-enter the market.
- `deploy/logrotate.conf` — daily rotation of `oxalpha.log` (14 days,
  compressed).

## 4. Secrets discipline

- All secrets are environment variables (`config.yaml` references only the
  variable *names*).  Rotate the Dhan access token before every live session
  — tokens expire (~24 h); a token pasted into chat or logs must be treated as
  compromised and rotated immediately.
- The HMAC audit key (`OX_AUDIT_KEY`, >= 32 chars) chains the audit table; if
  it is rotated the chain cannot verify and live mode refuses to boot.  Keep
  it stable and back it up with the database.
- Run live sessions only from the static IP whitelisted in the Dhan portal
  (the live-test step 0 verifies the egress IP and fails otherwise).

## 5. The real-venue track record protocol

The engine is proven *offline* by 74+ tests; a *real-venue* track record can
only come from supervised sessions on an actual Dhan connection.  The harness
(`ox/live_test.py`, invoked as `run.py live-test`) uses **read-only**
endpoints only — funds, LTP, quote+depth, intraday candles, indicators on real
data — and never places, amends, or cancels an order.  Optionally
(`live-test <seconds>`) it feeds live quotes into a paper-ledger PRIME session
and reports agent ticks and simulated fills.

Every session appends one JSON line to `track_record/sessions.jsonl`
(journaled even on failure, so problem sessions are part of the record):

```json
{"ts": "...", "tool": "live-test", "read_only": true,
 "ok": true, "failures": 0, "symbol_count": 5, "prime": null}
```

Protocol to accumulate a credible track record:

1. Rotate the Dhan token; source it + `DHAN_CLIENT_ID` into the environment.
2. From the whitelisted machine: `python run.py live-test` — all read-only
   checks must PASS.
3. Then supervised prime sessions across several trading days:
   `python run.py live-test 1800` (30 min of live quotes on the paper ledger;
   watch decisions/halts, then review).
4. After each session run `python run.py track-record` and read the summary:
   sessions, clean-pass rate, window, live-quote paper time, fills.
5. Only after a sustained clean streak across full sessions does a *tiny*
   live-capital step (well under the config caps, kill switch armed) become
   defensible — and the same journal then documents that step too.

## 6. Runtime hygiene

- `state/`, `backups/`, `compliance_reports/`, `*.db`, `*.log`, and
  `track_record/` are gitignored: the repository pins code, never account
  state or secrets.
- Backup automation (`ox/database_backup.py`) copies the SQLite database; keep
  the backups volume mounted and test a restore before relying on it.
