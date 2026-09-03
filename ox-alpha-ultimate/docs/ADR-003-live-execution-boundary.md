# ADR-003: Verified live-execution boundary

**Status:** Accepted  
**Date:** 2026-08-29

## Decision

Only the reviewed Dhan order surface may send live orders. Live boot requires
the configured broker and retail-algo registration references, explicit
environment-level enablement, an intact audit chain, static-IP confirmation,
and human approval for newly trained strategies. Groww remains a future
adapter until broker-native linked protection and the identical controls can
be verified; TradingView remains a signal source, not an execution broker.

## Consequences

- Paper training remains autonomous.
- Completed-candle signals, VaR, expected shortfall, outward tick rounding,
  and a fixed API allowlist are mandatory before a Dhan entry.
- The system must fail closed when approvals, evidence, or data are absent.
