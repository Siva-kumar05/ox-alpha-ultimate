# PRIME (v2) — actual architecture

> The older sections of this file describe the legacy single-agent runtime
> (`ox/agent.py`, still functional). The multi-agent layer below is the
> current top of the stack and is fully wired and tested
> (`tests/test_promax.py`, 25 tests; full suite 42 green).

## What actually runs

```
python run.py promax
        │
        ▼
AgentOrchestrator (ox/agents/orchestrator.py)
 ├─ loads config_promax.yaml
 ├─ DataPump ── broker quotes + paper fundamentals ──► SharedDataBus topics
 │     market:<sym>, exchange:funding, exchange:basis
 ├─ 10 independent agents (ox/agents/*.py)  ── consume bus, emit Signal
 │     risk gate (RiskCoordinator + LeverageLadder)
 │     approval gate (ApprovalGateway: buys PENDING, sells instant)
 ├─ ExecutionRouter ── approved signals ──► PaperBroker / CryptoMicroBroker
 │     fills ──► agent.positions + fills:<agent> + CapitalAllocator ledger
 └─ monitor loop: schedules (auto-pause/resume), health/restart,
       ladder evaluation, approval TTL expiry, kill flag, equity sync
```

## Components

| Piece | File | Notes |
|---|---|---|
| Orchestrator, DataPump, ExecutionRouter | `ox/agents/orchestrator.py` | paper/live same code path |
| Agent base (states, bus, signals) | `ox/agents/base.py` | `_emit_signal` = risk gate + approval gate |
| Risk coordination + ladder + Monte Carlo | `ox/agents/risk_coordinator.py` | `LeverageLadder`, `monte_carlo_survival` |
| Approval gateway (buys gated, sells free) | `ox/agents/approvals.py` | sqlite `order_intents`, TTL, Telegram via SSRF-guarded POST |
| Capital budgets + trade ledger | `ox/agents/capital_allocator.py` | `promax_trades` table |
| SSRF-safe HTTP | `ox/ssrf.py` | scheme/host/IP validation, guarded redirects, allowlists |
| News/Twitter/Telegram monitor | `ox/news.py` + `ox/agents/news_intel.py`, `social_monitor.py` | RSS (DTD-rejecting parser), X API v2 (optional bearer), t.me previews, Nitter |
| Ledger/risk/intent tables | `ox/core.py` `DB.SCHEMA` | additive: `order_intents`, `promax_trades` |

## Polyglot scaffolds — honest status

`cpp/` (lock-free execution engine), `go/` (market-data service),
`java/` (Spring compliance), `rust/` (VaR/ES engine) contain real code but
are **not wired into the runtime** and are not compiled on this machine (no
Go/C++/Maven toolchains installed; JDK 25 is present). The Python path is
the verified system. Wiring any of them requires an IPC contract (the bus
topics above are the natural seam) plus a toolchain — treat them as
research assets, not running infrastructure.
