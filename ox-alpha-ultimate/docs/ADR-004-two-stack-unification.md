# ADR-004: Unify the two execution stacks behind one decision core

**Status:** Accepted
**Date:** 2026-09-04

## Context

Two execution stacks implement the same entry/exit/risk concepts with
different shapes:

- **Legacy runtime** (`config.yaml`, `ox/agent.py` + `ox/oms.py`): bracket
  super-orders on Dhan/Choice/paper, `oms.positions` ledger, compliance gates,
  Dhan depth feed.
- **Promax orchestrator** (`config_promax.yaml`, `ox/agents/*`): per-agent
  positions, `ExecutionRouter` with `place_market`/`reduceOnly`,
  DataPump/approvals/ladder, equity + crypto venues.

The duplication is a real cost: the reservation-leak defect found in the
router's crypto branch had to be re-derived and fixed in the parallel equity
branch, and the legacy path still carries its own copy of the same sizing
logic.  `ox/agent.py` is the god class (~1,000 lines, 30+ responsibilities).

## Decision

Consolidate incrementally behind a pure decision core rather than a big-bang
rewrite of either stack:

1. **`ox/decision.py`** (done) owns bracket construction, Kelly/cap sizing,
   leverage overlay, and clamps as pure functions.  The legacy agent's four
   call sites now delegate to it; the existing `test_agent_decisions` suite
   proves zero behavior change.
2. The promax `ExecutionRouter` entry/exit sizing is next to consume
   `ox/decision.py` so one fix lands once for both stacks.
3. The two stacks keep their broker contracts (`place_super_order` vs
   `place_market`) and ledgers, but both reconcile against broker truth and
   both guarantee: no phantom ledger entries, no fabricated fills, no
   position left overnight.
4. No single "merge" flag or dual-mode agent; the decision core is the seam,
   and tests (`test_agent_decisions`, router live/equity, OMS contract,
   chaos, EOD square-off) are the safety net that makes each extraction
   provably behavior-preserving.

## Consequences

- `agent.py` shrinks toward a thin tick-loop over `ox/decision.py`.
- A sizing or exit-defect fix lands in one file and is covered by both
  stacks' tests.
- Legacy `Agent`/`OMS` vs promax duplication is explicitly staged for
  removal, not accepted as permanent.
