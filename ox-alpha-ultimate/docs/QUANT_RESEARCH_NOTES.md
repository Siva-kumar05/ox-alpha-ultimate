# Quant research guardrails

This note records the research ideas that inform ox-alpha's design.  They are
*testable hypotheses and risk controls*, not a prediction of profit.  The
evidence cited below comes largely from other venues, instruments, and time
periods; it must be reproduced using genuine Dhan/NSE data before it can
support a production decision.

## 1. Order flow must be measured from the book, and interpreted with depth

Cont, Kukanov, and Stoikov study short-horizon price changes and find a more
robust relationship with best-quote order-flow imbalance than with trade
volume, with impact that varies inversely with market depth.  That motivates
the agent's use of contemporaneous bid and ask ladders, displayed-book
imbalance, microprice edge, spread and available notional together.  It does
not justify treating OHLCV volume as buyer/seller delta or using an imbalance
threshold unchanged across symbols.

Implementation boundary:

- `OrderFlowEngine` records only observable L2 fields and labels them
  `DHAN_DEPTH20`.
- The entry gate needs both book support and adequate two-sided depth; it
  rejects stale, thin, or wide markets rather than assuming an imbalance is
  tradeable.
- The live admission replay retains one observation per candle, drops feed
  gaps, and evaluates the actual recorded depth gate.  It is explicitly not
  an execution backtest: no queue position, fill probability, fees, or market
  impact is inferred from it.

Before changing an order-flow threshold, segment the replay by symbol,
time-of-day, spread bucket, and depth bucket.  A coefficient which only works
in one liquid name or one session period is not a portfolio-wide parameter.

Primary source: [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402)
(Cont, Kukanov & Stoikov, 2014).

## 2. A promising backtest is not enough to allocate capital

Bailey, Borwein, López de Prado, and Zhu describe how trying many variations
can select a lucky historical pattern.  Ordinary single hold-out splits can be
misleading for time-dependent financial data.  ox-alpha therefore has
causally confirmed structure features, next-candle execution, costs and
slippage, embargoed expanding walk-forward folds, score caps, and minimum
out-of-sample trade thresholds.

Implementation boundary:

- A strategy is a constrained parameter record for an audited template, not
  database-supplied code.
- A new template or materially widened parameter grid requires a fresh
  evaluation cycle; old schema versions are quarantined.
- The current walk-forward test is a strong admission screen, not a formal
  estimate of probability of backtest overfitting (PBO).  Do not label its
  result PBO or a confidence level without implementing the paper's complete
  combinatorially symmetric cross-validation procedure and recording every
  trial.

Primary source: [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
(Bailey, Borwein, López de Prado & Zhu, 2015).

## 3. Execution quality is part of the signal

Cont and Kukanov's limit-order-placement work shows that placement depends on
order flow, queue sizes, and the fee/rebate structure.  The practical lesson
for this project is modest: a signal cannot be promoted as a live edge when
the evaluation omits the cost of getting in and out.

Implementation boundary:

- Candle-template tests use next-candle opens, configured slippage and Indian
  transaction-charge calculations.
- A confirmed Dhan Super Order supplies the broker-managed stop and target;
  an uncertain mutation is reconciled or halted, never retried as though no
  order could exist.
- The retained L2 replay is deliberately excluded from strategy return and
  position-sizing claims until executable Dhan fills, charges, and latency are
  captured for it.

Primary source: [Optimal Order Placement in Limit Order Markets](https://arxiv.org/abs/1210.1625)
(Cont & Kukanov, 2014).

## Research admission checklist

Use this checklist for a paper-trading experiment before any live change:

1. State the hypothesis, instrument universe, observation interval, and
   intended holding horizon before reading the result.
2. Keep the trigger, costs, slippage, data timestamp convention, and every
   rejected configuration in the experiment record.
3. Test causally separated, embargoed periods and report results by symbol
   and market regime, not only the pooled average.
4. Compare the forward move against all-in breakeven cost and a conservative
   error allowance.  A positive gross return is not sufficient.
5. Keep the agent in paper mode unless the retained Dhan evidence meets the
   configured gate and an operator independently accepts the live-risk
   configuration.

These safeguards are intentionally conservative.  They make it harder for an
attractive chart or a social-media profit claim to become an unsupported
capital-deployment rule.
