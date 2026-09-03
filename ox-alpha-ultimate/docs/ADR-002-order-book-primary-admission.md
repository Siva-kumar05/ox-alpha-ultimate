# ADR-002: Use L2 order-book admission, not claimed cross-market arbitrage

**Status:** Accepted  
**Date:** 2026-08-23  
**Decider:** ox-alpha owner

## Context

The supplied social-media examples describe alleged crypto or prediction-market opportunities based on venue-to-venue price delays. ox-alpha is a long-only NSE-equity agent with one Dhan execution venue. It has access to Dhan Full Market Depth and to its own order execution state, but it does not have simultaneous executable quotes, fill guarantees, or hedge legs for Solana, Polymarket, or any other venue.

Dhan's 20-level feed provides separate real-time bid and ask market-by-price ladders for NSE equity and derivatives. It does not provide enough information to call displayed liquidity executed trade delta, queue position, or individual-order intent.

## Decision

Use the transferable part of the examples: act only on fresh, observable market state. A new long must pass this sequence:

1. Pair contemporaneous Dhan bid and ask ladders.
2. Reject stale, thin, or wide-spread books.
3. Require displayed buy-side imbalance, supportive microprice, smoothed persistent pressure, and multiple supportive snapshots.
4. Require a separately validated candle-regime confirmation.
5. Pass the existing portfolio-risk gate before submitting a Dhan Super Order.
6. Before live primary use, require a retained real `DHAN_DEPTH20` gate replay to meet configured sample, hit-rate, and mean-forward-return criteria. The replay is explicitly not an execution backtest because it cannot infer fills, queue priority, fees, or aggressor-side trades from the retained inputs.

The dashboard records the underlying book metrics, decision reason, and current bid/ask ladder. Live orders still use broker-managed stop and target legs.

## Options considered

### A. Add crypto/prediction-market arbitrage

| Dimension | Assessment |
|---|---|
| Executable venues | Missing |
| Atomic/hedged execution | Missing |
| Data and latency proof | Missing |
| Operational and compliance scope | High |

Rejected. Adding a label or a simulated price-gap signal would be misleading and unsafe.

### B. Use Dhan L2 book as a primary admission signal

| Dimension | Assessment |
|---|---|
| Executable venue | Present |
| Data granularity | 20 displayed price levels |
| Validation boundary | Requires recorded L2 replay before live capital |
| Operational scope | Bounded |

Accepted. This improves entry selectivity while remaining inside the Dhan/NSE boundary.

### C. Use candle volume as order flow

Rejected. OHLCV cannot show current book pressure, executed aggressor side, or queue changes.

## Consequences

- The agent can explain every order-flow block as a concrete feed, liquidity, persistence, or risk condition.
- A stale or unavailable depth feed cannot create new exposure.
- The design does not claim millisecond arbitrage, market-making priority, spoof detection, or profit certainty.
- The L2 logic is rejected for primary live admission until retained depth snapshots meet its configured replay threshold; bar-only backtests are insufficient.

## References

- [Dhan Full Market Depth](https://dhanhq.co/docs/v2/full-market-depth/)
- [NautilusTrader backtest execution flow](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/execution-flow.md)
