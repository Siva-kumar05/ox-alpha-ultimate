"""
Social Monitor Agent — crypto social momentum tracker.
======================================================

Mandate: watch social channels (Telegram preview pages, Nitter instances,
optional X API) for mention-volume spikes and sentiment shifts on the meme
/ low-cap watchlist, and publish ``social:<sym>`` momentum events.  It never
trades; the meme-swing agent consumes its events as *one input among many*
(social spikes are manipulation-prone and never sufficient on their own).

Mention-volume model: exponential moving average of per-poll mention counts
per keyword; an event fires when the latest count exceeds
``spike_multiple`` x the EMA baseline.  Sentiment comes from the shared
NewsEngine keyword scorer (crypto vocabulary included).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List

from ..core import iso
from ..news import NewsEngine
from .base import AgentConfig, BaseAgent, Signal
from .capital_allocator import CapitalAllocator
from .risk_coordinator import RiskCoordinator

LOG = logging.getLogger("promax.social")

X_BEARER_ENV = "OX_X_BEARER"


class SocialMonitorAgent(BaseAgent):
    """Mention-volume spike detector for the meme/low-cap watchlist."""

    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator: RiskCoordinator,
        capital_allocator: CapitalAllocator,
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)
        pooled = resource_pool.get("news_engine")
        self.engine: NewsEngine = pooled if pooled is not None else NewsEngine({"news": {}})
        params = config.custom_params
        self.watch_keywords: Dict[str, List[str]] = dict(params.get("watch", {}))
        self.telegram_channels: List[str] = list(params.get("telegram_channels", []))
        self.x_queries: List[str] = list(params.get("x_queries", []))
        self.nitter: Dict[str, Any] = dict(params.get("nitter", {}))
        self.spike_multiple = float(params.get("spike_multiple", 3.0))
        self.ema_alpha = float(params.get("ema_alpha", 0.25))
        self.min_mentions = int(params.get("min_mentions", 2))
        self.poll_interval = float(params.get("poll_interval_seconds", 90))

        self.mention_ema: Dict[str, float] = defaultdict(float)
        self.mention_counts: Dict[str, int] = defaultdict(int)
        self.recent_events: deque = deque(maxlen=50)
        self._last_poll: datetime | None = None

    # ── lifecycle ─────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
        self.capital_allocator.register_agent(self.agent_id)
        LOG.info(f"SocialMonitorAgent watching {len(self.watch_keywords)} keyword groups")
        return True

    def _get_loop_interval(self) -> float:
        return self.poll_interval

    # ── collection ────────────────────────────────────────────────────
    def _collect_texts(self) -> List[Dict[str, Any]]:
        posts: List[Dict[str, Any]] = []
        bearer = os.environ.get(X_BEARER_ENV, "").strip()
        if bearer and self.x_queries:
            for query in self.x_queries:
                posts.extend(self.engine.fetch_x_search(bearer, query))
        for channel in self.telegram_channels:
            posts.extend(self.engine.fetch_telegram_channel(channel))
        instance = self.nitter.get("instance", "")
        if instance:
            for handle in self.nitter.get("handles", []):
                posts.extend(self.engine.fetch_nitter_posts(instance, handle))
        return posts

    def _process_poll(self) -> None:
        posts = self._collect_texts()
        counts: Dict[str, int] = defaultdict(int)
        sentiment_buckets: Dict[str, List[float]] = defaultdict(list)
        for post in posts:
            text = str(post.get("text", "")).lower()
            for group, keywords in self.watch_keywords.items():
                if any(kw.lower() in text for kw in keywords):
                    counts[group] += 1
                    sentiment_buckets[group].append(float(post.get("score", 0.0)))

        for group, count in counts.items():
            baseline = self.mention_ema.get(group, 0.0)
            updated = baseline + self.ema_alpha * (count - baseline)
            self.mention_ema[group] = updated
            self.mention_counts[group] = count
            if baseline > 0 and count >= self.min_mentions and count >= baseline * self.spike_multiple:
                scores = sentiment_buckets.get(group, [0.0])
                avg = sum(scores) / len(scores)
                event = {
                    "symbol": group,
                    "symbol_group": group,
                    "mentions": count,
                    "baseline": round(baseline, 2),
                    "sentiment": round(avg, 3),
                    "ts": iso(),
                }
                self.recent_events.append(event)
                self.data_bus.publish(f"social:{group}", event)
                self.data_bus.publish("social:mentions", event)
                self.data_bus.publish("social:sentiment", {
                    "symbol": group, "avg_score": avg, "sentiment": round(avg, 3),
                    "mentions": count, "ts": iso(),
                })
                LOG.info(f"Social spike {group}: {count} mentions vs baseline {baseline:.1f} "
                         f"(sentiment {avg:+.2f})")
        self._last_poll = datetime.now()

    # ── main loop ─────────────────────────────────────────────────────
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        return []

    async def manage_positions(self) -> List[Signal]:
        try:
            self._process_poll()
        except Exception as exc:
            LOG.warning(f"Social poll failed: {exc.__class__.__name__}: {exc}")
        return []

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["last_poll"] = self._last_poll.isoformat() if self._last_poll else None
        status["mention_counts"] = dict(self.mention_counts)
        status["recent_events"] = list(self.recent_events)[-5:]
        return status
