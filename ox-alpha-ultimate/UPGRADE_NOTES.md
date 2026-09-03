# Ultimate Upgrade - 100x vs reviewed/corrected

- Reviewed is secure build (base). Corrected has pre-market fix but missing portfolio VaR, audit improvements. Ultimate merges best.
- ox/brokers.py uses corrected pre-market logic + reviewed depth feed.
- ox/brain.py keeps reviewed frame_consistency.
- ox/agent.py keeps reviewed market_close scheduling.
- config.yaml preserves reviewed 2026 holidays + adds crypto/scalping.

Verified sources: Cont 2014, Berkowitz 1988, Wilder 1978, Black-Scholes 1973, Jorion VaR, Sortino 1991, Kelly 1956, Lopez de Prado purged CV.


## v3 hardening audit trail

Every flaw from the deep audit has a tagged fix:
A1 partial-entry residual cancel (oms.py), A2 exit retry + stop re-arm
(oms.py), A3 isolated target modify (oms.py), A4 depth backoff reset
(brokers.py), A5 chunked history (brokers.py), A6 poll cadence +
reconcile interval (brokers.py/agent.py/config), B1 put theta,
B2 value area, B3 avwap NaN, B4 Garman-Klass calendar, B5 adaptive
Kalman, B6 full self-test, B7 flow-key casing; C1 true volumes via
quote_snapshot deltas, C2 zero-drift paper history, C3 real IC/ICIR,
C4 loser-aware failure analysis, C5 bar-marked maxdd, C6 ensemble
quorum, C7 documented OHLCV-delta caveat remains (templates are
secondary while order_flow.primary=true), C9 news dedupe; E1-E5 dead
research scaffolds labelled RESEARCH-ONLY and the static-IP script no
longer stores the address in-repo.

New operator surface: `python run.py approve <sid>` promotes
PENDING_APPROVAL strategies; `python run.py status` lists them.
