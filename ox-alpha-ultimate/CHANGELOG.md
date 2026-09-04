# Review changelog — this pass

## Root-fix sweep: affirmation-gate UX + capital-safety/hygiene defects (2026-09-04)

Message/UX and contained root fixes from the FLAWS_AND_GAPS adjudication and a
bare-`python run.py` operator stumble on the week-1 live config:

- **Affirmation-gate UX (ox/core.py, ox/preflight.py):** the live-mode
  ConfigError now names the supported ceremony (setup-live.sh once, then
  launch only via `bash scripts/live.sh <venue>`; the flag is launcher-scoped
  so a bare `python run.py` never starts live). `run.py preflight` on a
  `mode: live` config emits a `launch gate` WARN with the same guidance.
  Message-only - gate semantics, exit codes, and launch behavior unchanged.
- **Kill-switch orphan guidance (ox/oms.py):** when an entry was still in
  flight at kill time, a partial fill may have orphaned a broker position
  that never reached local state. The kill now logs that case loudly and the
  KILL.flag body records the reconcile-before-restart step durably.
- **Backup correctness (ox/database_backup.py):** passive WAL checkpoint
  before the snapshot; `_verify_backup` now actually reads the
  `PRAGMA integrity_check` result instead of discarding it (verification
  could never fail before); fixed a TypeError in `_cleanup_old_backups`
  (aware vs naive datetime) that crashed every real `create_backup()`, and
  connection leaks on the corrupt-backup path. New regression tests.
- **Per-frame ICIR honesty (ox/brain.py):** a single-frame `Backtester.run()`
  reports `icir: NaN` (ICIR is only meaningful after `aggregate()` pools
  frames' ICs) instead of a misleading `0.0`.
- **Dead slippage code removed (ox/charges.py, ox/__init__.py, configs):**
  `DynamicSlippageModel`/`SlippageEstimate` were exported and configured in
  config.yaml but never instantiated anywhere (backtests lack the depth feed
  they need). Removed the classes, exports, and the inert config blocks.
- **Resilience-suite time-of-day bug (tests/test_run_loop_resilience.py):**
  the run loop schedules `nightly_training()` at the hardcoded 18:00 IST
  boundary; under the smoke-relaxed promotion gates that retrain promoted
  strategies mid-test, so the broker-resilience contracts only passed before
  18:00 IST. The harness now disables the scheduled retrain per agent
  (training is not part of the resilience contract under test).

## Audit sweep: incremental history fetch + compliance cadence wiring (2026-09-04)

Root-fix pass for the two genuine (b)-tagged defects from the FLAWS_AND_GAPS
adjudication (43/43 claims classified; 6 PLAUSIBLE left for live data):

- **Incremental history (#13 remainder):** `Agent.refresh_history` re-fetched
the full configured history every boot. It now reads the persisted per-symbol
watermark (newest `ts` in the candles table) and fetches only the gap since
it — fresh symbols still get the full fetch. The `(sym, ts)` PK +
`INSERT OR REPLACE` idempotency, per-row skip-not-halt logging, and the
systemic-corruption gate (`MIN_REJECTED_FOR_SYSTEMIC`/`MAX_BAD_CANDLE_FRACTION`)
are unchanged, so the RELIANCE incident behaviour is preserved exactly.
- **Compliance cadence (#27):** `ComplianceReporter.generate_report` was fully
implemented but never triggered. The agent now calls `run_due_reports()` at
boot (the tick loop is not running at 06:00); daily/weekly/monthly cadence is
evaluated from the configured schedule with kv completion markers recording
the covered period, so each report generates at most once per period even
across restarts. Fail-closed: a failed report stays due and is logged, never
blocks boot. Reports are anchored under the DB directory so they cannot leak
into the process CWD.

Order-execution, sizing, risk-gate, and bracket semantics are untouched.

**Verified:** Windows/3.14 228 passed + 10 subtests (was 222 + 10);
Linux/3.12 Docker 228 passed; ruff F 0; smoketest PASS.

## CI 3.12 verdict: scipy optional-dependency crash in Kupiec backtest (2026-09-04)

First-ever GitHub Actions run (root-level workflow) failed the pytest job on
Linux/Python 3.12: `FactorRiskModel.backtest_var` did a lazy
`from scipy.stats import chi2` with no fallback, but scipy is an optional
dependency (probed at module import; absent from `requirements.txt`) — so
the Kupiec branch crashed in CI's minimal env while passing locally where
scipy happened to be installed. The other 25 Docker-reported failures were
artifacts of an uncommitted `config.yaml` edit (duplicate top-level
`risk:`/`mode:` keys) and do not reproduce on the committed tree.

**Fix:** chi² with 1 degree of freedom is the square of a standard normal,
so its survival function is `math.erfc(sqrt(x/2))` — bit-identical to
scipy across the tested range, stdlib only. `backtest_var` no longer
requires scipy.

**Verified:** Linux/3.12 Docker full suite 222 passed (was 26 failed);
Windows/3.14 222 passed + 10 subtests; ruff F 0; smoketest PASS.

## KILL.flag resolution: tolerant history ingestion (2026-09-04)

**Root cause of the RELIANCE halt:** `Agent.refresh_history` raised
`MarketDataError` on the first malformed row in the broker's history batch
(a non-numeric timestamp), which `boot()` treats as a fatal reconcile
failure — so one bad candle out of thousands halted the agent and wrote
`KILL.flag`. The halt was fail-closed by design but the trip threshold was
wrong: it fired on a single noisy row instead of on genuinely corrupt data.

**Fix:** malformed rows (non-numeric fields, non-positive/inf prices,
inconsistent OHLC) are now skipped with a warning-level log line naming the
symbol and failure class. The gate still fails closed when corruption is
systemic — at least `MIN_REJECTED_FOR_SYSTEMIC` (5) rows AND more than
`MAX_BAD_CANDLE_FRACTION` (25%) of the batch rejected — and an empty or
all-bad batch still raises. The absolute floor keeps a stray bad row in a
short batch from counting as systemic. `KILL.flag` cleared after the fix.

**Tests:** new `tests/test_history_ingestion.py` (7 cases): single
non-numeric candle skipped while good candles persist, NaN/inconsistent
OHLC rows skipped, minority corruption tolerated up to the gate, systemic
corruption (6/10) fails closed, all-bad and empty batches fail closed, gate
constants sanity. Suite: 222 passed, 10 subtests. Smoketest: PASS.

---

## Quality/correctness pass (ruff + full suite, 2026-09-04)

**Lint: 249 pyflakes findings fixed to zero.** `ruff check --select F`
(now the CI gate) found 188 unused imports, 56 unused locals, 5 pointless
f-strings across `ox/`, `tests/`, `run.py`, `dashboard.py`,
`dashboard_data.py`, `app_pages/`. 185 auto-fixed; the rest reviewed
individually. Optional-dependency probes (sklearn/scipy/cvxpy/pymysql
imports inside `try/except` that set `*_AVAILABLE` flags) were kept with
`# noqa: F401 - availability probe` comments. Dynamically-resolved agent
classes restored in `ox/agents/orchestrator.py` (`AGENT_CLASS_MAP` +
`globals()`, invisible to static analysis) and `tests/test_boot_drill.py`
now imports `install_fake_ccxt` from its canonical home `tests/support.py`.

**Correctness bugs found and fixed:**
- `run.py` smoketest called the removed `Agent._bracket_from_supporters`
  after brackets moved to `ox/decision.py` — the CI smoketest job was red.
  Now calls the module function; smoketest passes end-to-end.
- `RiskMonitor.check_limits` populated a dead local `alerts` and returned
  `self.alerts`, so repeated calls returned ever-growing stale alert lists.
  Now builds and returns the fresh local list.
- `FactorRiskModel.estimate_covariance` computed `returns_clean` but fit
  LedoitWolf/OAS on the raw frame (NaN would crash). Fits now use the
  cleaned frame.
- `RequestTracer.trace_request` computed `trace_id` (request_id or
  contextvar) and dropped it; `TraceContext` now threads `trace_id` into
  `start_span`, so an explicit request id actually lands on the span.
- Dead computed values removed where they masked intent: `fit_factor_model`
  redundant walrus if/else (both branches identical), duplicated `loadings`
  and `factor_names`, ArrivalPrice's unused `sigma/eta/lam` (the code now
  uses the `kappa` it computed instead of repeating the inline expression
  three times), Iceberg's ignored `display` variable, `bos_choch`'s dead
  `last_high/last_low`, `bayesian_optimization`'s dead `best_y`, and ~25
  other unused locals across agents/indicators/risk modules.

**CI widened:** `.github/workflows/ci.yml` lint job now enforces the full
pyflakes set (`--select F`) instead of only F821/F601/F811, so dead code
cannot silently re-accumulate.

**Verification:** `ruff check ox/ tests/ run.py dashboard.py dashboard_data.py
app_pages/ --select F` → 0 findings; `py_compile` clean on every touched
file; `python -m pytest tests/ -q` → 215 passed, 10 subtests (matches
baseline); `python run.py smoketest` → PASS; all four config YAMLs load
(promax via its orchestrator schema, which legitimately has no top-level
`symbols` list).

---

Verification method for everything below: full independent read of all 27
files (not a diff-only pass), a custom AST-based lint sweep (ruff/pyflakes
were unavailable offline), `py_compile` on every file, and the real
`python run.py smoketest` suite executed to completion multiple times —
not just inspected.

## Confirmed already fixed (verified against the uploaded bundle, not just asserted)

The 9 issues from the original review were re-checked line-by-line against
the actual code, not taken on trust:

1. Live bracket sizing uses each approved strategy's own `sl_atr`/`tp_atr`,
   score-weighted across whichever strategies actually voted long
   (`Agent._bracket_from_supporters`), not a generic hardcoded multiplier.
2. `tick_seconds` defaults to 3 with a live-mode floor of 2; Dhan rate-limit
   (429 / DH-904) responses are split into `RateLimitError` (read/quote —
   exponential backoff) vs `OrderError` (order mutation — fails closed,
   since a throttled mutation leaves broker state genuinely uncertain).
3. The order-flow admission gate has a dedicated `OrderFlowReplayValidator`
   that scores it against retained real `DHAN_DEPTH20` snapshots vs. later
   candles, explicitly labelled as a gate study and never as an execution
   backtest, and gates live boot.
4. `random.Random(seed)` is a local instance on `Brain`, not a mutation of
   the global `random` module.
5. Swing confirmation (`_swings`) returns the confirmation bar as
   `center + k`; `bos_choch` only uses a swing from that bar onward, so a
   pivot can't leak its own right-hand future bars into the signal.
6. News is refreshed on a configurable interval during the session (not
   once at boot) and has a separate `max_age_minutes` staleness cutoff.
7. Walk-forward validation is a real expanding-window, embargoed multi-fold
   split (`Agent._walk_forward_slices`), not a single static hold-out.
8. Dead `tick_loop()` is gone (grepped for confirmation, zero references),
   `market_close` is a validated config key used everywhere session timing
   is checked, and `ip_whitelist` is read from an environment variable.

## Fixed in this pass

**Encoding corruption (new find, not previously flagged).** 16 instances
across 5 files (`README.md`, `app_pages/chart.py`, `app_pages/overview.py`,
`ox/agent.py`, `docs/QUANT_RESEARCH_NOTES.md`) where em dashes, curly
apostrophes/quotes, and an accented letter (in "López de Prado") had been
mangled into mojibake (e.g. `backtesterâ€™s` instead of `backtester's`).
Some of these were in live dashboard captions and the top-level README, so
they'd have rendered as garbled text for any real user. Fixed by exact
codepoint-level replacement and verified with a full character-frequency
sweep of the repository afterward (zero remaining non-ASCII anomalies).

**`PaperBroker.hist()` time-of-day flakiness (new find, caught by actually
running the smoke test, not by reading the code).** The bootstrap candle
generator counted "today" as one of the requested N trading sessions. Run
before market open, today contributes zero candles, so a request for N
days could silently return well under N sessions of data — this is exactly
what happened when the smoke test was run early in the morning IST: the
`refresh_history` assertion failed because a 2-day request returned only
375 candles (one completed session) instead of ≥600. Fixed by always
generating N *fully completed* prior sessions and appending today as a
bonus partial session on top, so the candle count returned no longer
depends on what time of day the call happens to run. Verified by re-running
the smoke test (now passes with margin instead of at an exact boundary)
three times in a row.

**Lint (6 items, re-verified from scratch with a custom AST checker since
ruff/pyflakes aren't available in this offline environment):** unused
imports in `app_pages/audit.py` (`money`), `app_pages/orderflow.py` and
`app_pages/scanner.py` (`pandas`), `ox/charges.py` (`math`),
`ox/features.py` (`os`); one dead local (`security_id` in
`DhanBroker.ltp()` — `ltps()` already resolves it internally).

**`market_holidays` populated.** Was shipped empty, which blocks live-mode
boot entirely. Populated with the 16 real 2026 NSE/BSE equity trading
holidays, cross-checked between two independent broker-published calendars
(Zerodha and Groww, both sourced from NSE/BSE circulars) that agreed
exactly on all 16 dates. Settlement-only closures and festivals that fall
on an already-closed weekend are intentionally excluded, since equity
trading itself is open on those days.

**New `oos_frame_consistency` metric.** `Backtester.evaluate()` pools every
walk-forward fold/symbol OOS frame into one score, which means a strategy
that's genuinely inconsistent — say, profitable in one fold and a loser in
the other two — can still clear promotion if the pooled numbers look fine.
Added `Backtester.frame_consistency()`, which reports what share of the
individual OOS frames that actually traded were themselves net profitable,
plus the traded/total frame counts. This is stored alongside the existing
pooled stats (`oos_frame_consistency`, `oos_frames_traded`,
`oos_frames_total`) and surfaced in the audit dashboard's validation table
as "Fold consistency" and "OOS frames", with a caption explaining what it
means. It does not gate promotion by itself — it's evidence displayed next
to the pooled score, not a new automatic cutoff — since turning it into a
hard gate is a threshold-tuning decision this pass didn't have grounds to
make unilaterally.

**Two small robustness fixes:**
- `Agent.run_forever()`'s EOD trigger was a hardcoded `"15:35"`, independent
  of the configurable `market_close`. Changed it to key off
  `self.cfg["market_close"]` directly, so a changed close time can't leave
  EOD stats computed too early or too late. (`nightly_training`'s `"18:00"`
  trigger is a plain offline batch-job time unrelated to market hours, and
  is left as-is.)
- `OrderFlowReplayValidator.evaluate()` recomputed `timeframe` from config
  on every loop iteration instead of once outside the loop. Harmless
  (the value can't change mid-loop) but removed for clarity.

## What's unchanged and why

`start_orderflow()` still fails closed if the Dhan Full Market Depth
subscription can't connect while `order_flow.primary` is true. That's
correct fail-safe behaviour, not a bug — it's a one-time operational
prerequisite documented in the README, not something to route around in
code.

## Verification performed

- Every file read in full, independently, against the actual reconstructed
  source — not inferred from a prior summary.
- Custom AST-based unused-import/unused-local checker across all 20 Python
  files: 0 hits after fixes.
- `py_compile` on all 20 Python files: clean.
- Grep sweep for bare `except:`, mutable default args, `== None`/`!= None`,
  and the specific previously-flagged patterns (`tick_loop`, global
  `random.seed(`, hardcoded IPs): all clean.
- `python run.py smoketest`: passes, run 3 times consecutively for
  stability. Covers Dhan depth wire parsing (valid, malformed, wrong
  length), stale bid/ask pairing rejection, legacy DB schema migration,
  order-flow replay validation, swing-confirmation no-lookahead, walk-
  forward fold construction (3 folds × 60 candles), autonomous strategy
  promotion, raw-code and stale-schema strategy quarantine, live bracket
  parameters matching the approved strategy, autonomous buy → target exit,
  opposite-signal exit, and kill-switch flattening.
- Live-mode config validation (structural: mode/platform/IP/holidays/
  security_map, no real broker credentials involved) passes cleanly with
  the populated holiday list.


## Hardening pass (v3)

Money-path fixes (A1-A3): partial-entry residual legs are cancelled and
re-queried before local state is written; partial exits retry to completion,
aggregate a VWAP, shrink local state and re-arm a broker-side STOP_LOSS for
any remainder before failing closed; a failed target-modify no longer routes
a filled position into the uncertainty/cancel path.

Availability (A4-A6): the depth feed's reconnect backoff resets after any
successful session; Super Order polling slowed to 1s; broker position
reconciliation runs on `execution.reconcile_interval_seconds` (30s default)
instead of every tick; Dhan intraday history is fetched in
`execution.history_chunk_days` chunks so long windows cannot fail boot.

Accuracy (B/C): Black-Scholes put-theta sign fixed (put-call parity now
asserted); two-sided volume-profile value area; NaN-masked anchored VWAP;
calendar-parameterised Garman-Klass; volatility-adaptive Kalman gain; TPO
period counting; effort-result inf guard; true per-frame Information
Coefficient and ICIR across frames; bar-marked max drawdown; trade-level
loser-regime forensics feeding parameter refinement; 20-bar signal-stability
check; ensemble quorum gate (`execution.min_vote_fraction`);
Kelly-blended position sizing after >=30 closed trades; live candles now
carry true share volume via day-volume deltas from the Market Quote
endpoint instead of quote-tick counts; zero-drift paper bootstrap candles;
news deduplication; OrderFlowEngine internal key casing unified.

Governance: `training.require_human_approval: true` stops validated
strategies at PENDING_APPROVAL until an operator runs
`python run.py approve <sid>` - restoring approval-before-live-capital.

Self-test now covers all registered features plus parity/ordering/NaN/
finite invariants, so math regressions fail boot instead of shipping.
Smoke fixtures made deterministic: paper partial fills are single-shot and
entries republish depth until the book is entry-supportive, so the suite
tests its own gates rather than depth-pulse phase luck.
