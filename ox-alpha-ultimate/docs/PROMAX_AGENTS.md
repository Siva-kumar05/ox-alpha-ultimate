# PRIME Multi-Agent System — Agent Roster & Mandates

Entry point: `python run.py promax` (foreground, Ctrl-C to stop) ·
`python run.py promax 60` (60-second run) · `python run.py promax-smoke` ·
`python run.py promax-status` · `python run.py report [agent]` (tear sheet) ·
`python run.py validate-online` (real-data purged-CV validation) ·
config: `config_promax.yaml`.

## Buy-path gate order (enforced in `base.py::_emit_signal`)

1. **Risk gate** — per-agent limits + ladder-allowed leverage.
2. **Bull/bear debate** (`ox/agents/debate.py`, ported from TradingAgents
   arXiv:2412.20138, deterministic indicator debaters — no LLM, no
   hallucination): a weak case (< +0.15 verdict) is vetoed with the reason
   published on `debate:veto`; a pass scales signal strength by |verdict|.
   The panel carries a **per-agent memory of past mistakes** (JSON, survives
   restarts): losing streaks on the symbol and loss rates in similar
   indicator regimes raise the bear's score — every closed trade feeds it
   back through the execution router.
3. **Human approval** — capital-deploying buys park as PENDING intents
   (`python run.py intents` / `ok <iid>`); monitor loop re-publishes
   human-approved intents to the executor exactly once (kv-marked, restart
   safe). Sells/closes skip 2 and 3 entirely.

Every agent is an independent unit: own universe, own capital budget, own
risk limits, own schedule, own signals. Shared infrastructure (data bus,
news engine, DB, approval gateway, debate panel) is pooled so nothing is
duplicated.

**Operator rule (enforced in code, not policy):** capital-deploying orders
(buy/open/add) park in the ApprovalGateway as PENDING intents until a human
approves (`python run.py ok <iid>`, optional Telegram). Risk-reducing orders
(sell/close/modify) execute immediately — you never wait to exit.
`OX_PROMAX_AUTO_APPROVE=1` bypasses approvals **for paper smoke tests only**.

## Roster

| Agent | Mandate | Universe | Venue/leverage cap | Schedule |
|---|---|---|---|---|
| `intraday_scalper` | Intense intraday scalps (VWAP, opening range, CVD, flow imbalance) | Low-cost liquid NSE stocks (₹20–₹140 band) — not NIFTY50 heavyweights | MIS-style, cap 1x risked (broker offers 5x; ladder can raise risked exposure) | NSE hours |
| `equity_momentum` | Trend-riding with RS ranking, multi-window momentum, ATR trails | Low-cost NSE | 1x | NSE hours |
| `equity_growth` | Short-term (days) + long-term (weeks) high-growth picks; news-vetoed entries | Growth screen | 1x | NSE hours |
| `options_0dte` | Same-day expiry **debit spreads only** (defined risk), momentum-burst triggered, forced square-off 15:05 | NIFTY/BANKNIFTY | Defined-risk ratio (max gain/max loss), loss capped at debit | 09:35–14:30 entries |
| `market_maker` | Two-sided quoting with inventory skew; earns the spread; flat by EOD | Liquid low-cost names | 1x, inventory-capped, approval-exempt by design (set `approval_required` to gate) | NSE hours |
| `crypto_perp` | Perp momentum + mean-reversion + basis + liquidation-hunt filters | BTC/ETH/SOL perps | Platform cap 10x — **ladder starts at 2.5x** | 24×7 |
| `crypto_funding` | Funding-rate arbitrage across venues, market-neutral bias | BTC/ETH | Cap 5x — ladder starts 1.25x | 24×7 |
| `crypto_meme_swing` | Meme/low-cap swings, gated by social-spike + momentum + liquidity, tiny size | PEPE/WIF/BONK | Cap 5x, min size | 24×7 |
| `news_intel` | Single shared poller: RSS feeds, X search, Telegram previews → scored sentiment on the bus | — | no positions | 24×7 |
| `social_monitor` | Mention-velocity spikes per keyword group → `social:<sym>` events | meme watchlist | no positions | 24×7 |

## Leverage ladder (read this before dreaming about 50x)

Defined in `ox/agents/risk_coordinator.py` (`LeverageLadder`). Semantics:

- Level 1 = `start_fraction` (25%) of the agent's platform leverage cap.
- Each level doubles the allowed fraction; level 3 = 100% of cap.
- Promotion requires, over the last 60 closed trades: ≥20 trades,
  profit factor ≥ 1.3, win rate ≥ 50%, recent drawdown ≤ 15% of budget,
  **and** a Monte-Carlo estimate that the probability of a 50% drawdown at
  the *next* level stays ≤ 5% (`monte_carlo_survival`).
- Daily-loss breach ⇒ immediate demotion.

There is no configuration that guarantees "70% win rate at 50x". At 20x
leverage a 5% adverse move liquidates the position; that is exchange math,
not opinion. What the system *does* maximize is compounding survival — the
only statistically real path for small capital (₹5,000) to grow: small
proven edges, leveraged only as far as the evidence supports. Realistic
good-month outcome on ₹5,000 with discipline: +15–40%. Anything promising
more, monthly, is a coin flip with extra steps.

## Scheduler & pause behavior

`active_hours` per agent in `config_promax.yaml`. The monitor loop pauses
agents outside their window (equity agents overnight/weekends; crypto 24×7)
and resumes them when the window opens. Paused agents consume no signal
loop. Crashed agents restart up to 3 times, then stay stopped for review.
`promax_kill.flag` in the repo root halts everything on the next pass
(`python run.py promax-kill`).

## Manual positions (bought by you, not the agents)

Monitored via the legacy runtime (`python run.py status`) or the news gate:
the news-intel agent scores headlines for any symbol you add to its
`symbols` list — add your holdings there and their sentiment flows onto the
bus. Agent exits never require approval; if an agent holds nothing of yours
it will not sell your holdings.

## Validation & analytics

- **Tear sheet** (`ox/agents/tearsheet.py`, quantstats conventions):
  `python run.py report` — win rate, profit factor, expectancy, Sharpe/
  Sortino (annualisation caveats printed), Calmar, max drawdown + length,
  VaR95/CVaR95, streaks; flags samples < 30 trades as statistically
  meaningless instead of flattering them.
- **Purged walk-forward CV** (`ox/purged_cv.py`): purge (drop train samples
  whose label window touches the test fold) + embargo (gap after each fold);
  folds start only after a full training window. Zero label-overlap
  violations verified by test.
- **Online real-data validation**: `python run.py validate-online` — pulls
  NIFTY 50 (Yahoo chart API) and BTC/ETH (Binance public klines) daily
  history through the SSRF guard, runs Donchian/EMA-cross/RSI2 reference
  strategies through the purged CV with 12 bps costs, and prints honest
  OOS results (e.g. NIFTY: EMA cross +23% OOS, Sharpe 0.53; RSI2 −12% —
  which is exactly why the meme/RSI-style agents are size-capped).

## Polyglot & upstream repos cross-check

Cross-checked against the local repos: **TradingAgents** (debate + memory +
risk-discussion patterns — debate/memory ported above; its LLM calls were
deliberately replaced by deterministic indicator debaters), **quantstats**
(tear-sheet metric conventions), **purged-cross-validation** (purge/embargo
discipline), **quant-trading** (reference strategy families used in the
online validator), **ox-alpha-next** (event-driven/spec aspirations — the
wired PRIME bus+router already implements its core pattern), Lean/backtrader/
qlib (framework practices: separate alpha/risk/execution modules — mirrored
by agent/risk-coordinator/router separation).

## Paper vs live

Everything defaults to paper (PaperBroker random walk + crypto micro
broker). Live NSE execution goes through the same `DhanBroker` fail-closed
adapter as the legacy runtime (env-gated in `ox/core.py`). Live crypto is
wired: with `mode: live`, `CryptoMicroBroker` trades real Binance through
ccxt — swap symbols (BTCUSDT/ETHUSDT/SOLUSDT) on USDT-M perpetuals with
ladder-gated leverage and real funding rates, spot symbols
(PEPEUSDT/WIFUSDT/BONKUSDT) cash-only with leveraged orders refused. Live
mode requires `BINANCE_API_KEY`/`BINANCE_API_SECRET` and the same
operator gate as NSE (`OX_LIVE_EXECUTION_APPROVED`). Do not point this at
real money before a month of paper logs you have actually read.
