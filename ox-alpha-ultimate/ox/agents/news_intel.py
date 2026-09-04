"""
News Intelligence Agent — live multi-source monitor and sentiment bus.
======================================================================

Mandate: keep every trading agent's news view fresh without duplicating
fetch work.  This agent is the *single* poller for all sources (RSS feeds,
X/Twitter search, Telegram channel previews, Nitter), scores every headline
with the shared NewsEngine, persists to the ``news`` table, and publishes
per-symbol aggregate sentiment on the bus (``news:<sym>`` and
``news:sentiment``) so trading agents react in seconds, not on their own
poll schedules.

HTTP endpoints for the monitor list: GET-only and fetched with the SSRF
guard; no credentials in URLs.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from ..core import iso
from ..news import NewsEngine
from .base import AgentConfig, BaseAgent, Signal
from .capital_allocator import CapitalAllocator
from .risk_coordinator import RiskCoordinator

LOG = logging.getLogger("promax.news")

X_BEARER_ENV = "OX_X_BEARER"


class NewsIntelligenceAgent(BaseAgent):
    """Single shared poller + sentiment publisher for all agents."""

    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator: RiskCoordinator,
        capital_allocator: CapitalAllocator,
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)
        self.engine = NewsEngine({"news": {}})
        params = config.custom_params
        self.rss_sources: List[Dict[str, Any]] = list(params.get("rss_sources", []))
        self.x_queries: List[str] = list(params.get("x_queries", []))
        self.telegram_channels: List[str] = list(params.get("telegram_channels", []))
        self.nitter: Dict[str, Any] = dict(params.get("nitter", {}))
        self.symbols: List[str] = list(config.symbols)
        self.lookback_minutes = int(params.get("lookback_minutes", 240))
        self.min_poll_interval = float(params.get("poll_interval_seconds", 120))

        # Resource-pool sharing: trading agents fetch nothing themselves.
        key = "news_engine"
        pooled = resource_pool.get(key)
        self.engine = pooled if pooled is not None else self.engine
        if pooled is None:
            resource_pool.acquire(key, lambda: self.engine)
        self._last_poll: datetime | None = None
        self._symbol_scores: Dict[str, List[float]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
        self.capital_allocator.register_agent(self.agent_id)
        LOG.info(f"NewsIntelligenceAgent initialized: {len(self.rss_sources)} RSS, "
                 f"{len(self.x_queries)} X queries, {len(self.telegram_channels)} TG channels")
        return True

    def _get_loop_interval(self) -> float:
        return self.min_poll_interval

    # ── polling ───────────────────────────────────────────────────────
    def _poll_all(self) -> None:
        db = self._db()
        if db is not None and self.rss_sources:
            self.engine.poll_sources(db, self.rss_sources)
        if db is not None and self.symbols:
            for sym in self.symbols:
                for article in self.engine.fetch_news_rss(sym):
                    exists = db.q("SELECT 1 FROM news WHERE sym=? AND headline=? LIMIT 1",
                                  (article["sym"], article["headline"]))
                    if not exists:
                        db.ex("INSERT INTO news(sym,headline,source,sentiment,score,ts)"
                              "VALUES(?,?,?,?,?,?)",
                              (article["sym"], article["headline"], article["source"],
                               article["sentiment"], article["score"], article["ts"]))

        bearer = os.environ.get(X_BEARER_ENV, "").strip()
        posts: List[Dict[str, Any]] = []
        if bearer and self.x_queries:
            for query in self.x_queries:
                posts.extend(self.engine.fetch_x_search(bearer, query))
        for channel in self.telegram_channels:
            posts.extend(self.engine.fetch_telegram_channel(channel))
        nitter_instance = self.nitter.get("instance", "")
        if nitter_instance:
            for handle in self.nitter.get("handles", []):
                posts.extend(self.engine.fetch_nitter_posts(nitter_instance, handle))

        if posts and db is not None:
            for post in posts:
                headline = f"[{post['source']}] {post['text']}"
                exists = db.q("SELECT 1 FROM news WHERE sym=? AND headline=? LIMIT 1",
                              ("social", headline))
                if not exists:
                    db.ex("INSERT INTO news(sym,headline,source,sentiment,score,ts)"
                          "VALUES(?,?,?,?,?,?)",
                          ("social", headline, post["source"], post["sentiment"],
                           post["score"], post["ts"]))

        self._last_poll = datetime.now()

    def _db(self):
        pooled = self.resource_pool.get("db")
        return pooled

    def _publish_sentiment(self) -> None:
        db = self._db()
        if db is None:
            return
        for sym in self.symbols:
            score, sentiment = self.engine.get_optimism_score(db, sym)
            self._symbol_scores.setdefault(sym, []).append(score)
            self._symbol_scores[sym] = self._symbol_scores[sym][-20:]
            payload = {"symbol": sym, "avg_score": score, "sentiment": sentiment,
                       "ts": iso()}
            self.data_bus.publish(f"news:{sym}", payload)
            self.data_bus.publish("news:sentiment", payload)

    # ── main loop ─────────────────────────────────────────────────────
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        return []  # data-driven work happens in manage_positions

    async def manage_positions(self) -> List[Signal]:
        try:
            self._poll_all()
        except Exception as exc:
            LOG.warning(f"News poll failed: {exc.__class__.__name__}: {exc}")
        try:
            self._publish_sentiment()
        except Exception as exc:
            LOG.warning(f"Sentiment publish failed: {exc.__class__.__name__}: {exc}")
        return []

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["last_poll"] = self._last_poll.isoformat() if self._last_poll else None
        status["symbol_scores"] = {k: v[-1] if v else 0.0
                                   for k, v in self._symbol_scores.items()}
        return status
