import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .core import LOG, iso, now
from .ssrf import SafeURLViolation, safe_fetch

_MAX_RSS_BYTES = 1_000_000


def _parse_rss(raw: bytes):
    """Parse an RSS/Atom feed defensively.

    Untrusted XML must never reach ElementTree unguarded: DTD entity
    expansion (billion-laughs) is a resource-exhaustion vector, so any
    document declaring DOCTYPE/ENTITY is rejected outright and the payload
    is size-capped before parsing.
    """
    if raw and len(raw) > _MAX_RSS_BYTES:
        raise SafeURLViolation(f"rss payload exceeds {_MAX_RSS_BYTES} bytes")
    head = raw[:4096].decode("utf-8", errors="replace").lower()
    if "<!doctype" in head or "<!entity" in head:
        raise SafeURLViolation("rss payload declares DTD/ENTITY — rejected")
    return ET.fromstring(raw.decode("utf-8", errors="replace"))

BULLISH_KEYWORDS = {
    "growth": 1.5, "profit": 2.0, "surge": 2.0, "rally": 2.0, "outperform": 2.0,
    "breakout": 1.8, "record": 1.5, "dividend": 1.2, "expansion": 1.5, "upgrade": 2.0,
    "bullish": 2.0, "order win": 2.2, "contract": 1.5, "revenue jump": 2.2, "partnership": 1.5,
    "buyback": 1.8, "acquisition": 1.2, "strong results": 2.0, "beaten estimates": 2.0,
    "target raised": 2.2, "robust": 1.5, "optimistic": 1.8, "all-time high": 2.0,
    # crypto vocabulary
    "etf approval": 2.5, "institutional inflow": 2.0, "short squeeze": 2.0,
    "halving": 1.2, "mainnet launch": 1.5, "listing": 1.2, "whale buying": 1.8,
    "accumulation": 1.2, "burn": 1.0, "integration": 1.2,
}

BEARISH_KEYWORDS = {
    "loss": 2.0, "slump": 2.0, "drop": 1.5, "crash": 2.5, "downgrade": 2.0,
    "fraud": 3.0, "default": 3.0, "penalty": 2.0, "investigation": 2.2, "decline": 1.5,
    "bearish": 2.0, "debt": 1.8, "resignation": 1.8, "warning": 2.0, "misses estimates": 2.0,
    "cut target": 2.2, "layoffs": 1.8, "sanction": 2.0, "lawsuit": 2.0,
    # crypto vocabulary
    "liquidation": 2.2, "hack": 3.0, "exploit": 2.8, "rug pull": 3.0, "delisting": 2.0,
    "sec lawsuit": 2.5, "ban": 1.8, "outflow": 1.5, "whale selling": 1.8,
    "stablecoin depeg": 2.5, "insolvency": 3.0, "ponzi": 3.0,
}

class NewsEngine:
    """
    Best-effort research filter. It may suppress a long on strongly negative
    headlines, but unavailable or untrusted news never creates a bullish signal.
    """
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def score_text(self, text: str) -> tuple:
        t = text.lower()
        pos_score = sum(weight for kw, weight in BULLISH_KEYWORDS.items() if kw in t)
        neg_score = sum(weight for kw, weight in BEARISH_KEYWORDS.items() if kw in t)

        tot = pos_score + neg_score
        if tot == 0.0:
            return "NEUTRAL", 0.0

        score = (pos_score - neg_score) / max(tot, 1.0)
        sentiment = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
        return sentiment, round(score, 3)

    def fetch_news_rss(self, sym: str) -> list:
        articles = []
        url = f"https://news.google.com/rss/search?q={sym}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            raw = safe_fetch(url, timeout=8)
            root = _parse_rss(raw)
            for item in root.findall(".//item")[:5]:
                title = item.findtext("title", "")
                clean_title = re.sub("<[^<]+?>", "", title).strip()[:500]
                if not clean_title:
                    continue
                sentiment, score = self.score_text(clean_title)
                articles.append({
                    "sym": sym,
                    "headline": clean_title,
                    "source": "GoogleNews",
                    "sentiment": sentiment,
                    "score": score,
                    "ts": iso()
                })
        except Exception as e:
            LOG.warning(f"News fetch note for {sym}: {e}")

        return articles

    def poll_and_save(self, db, syms: list):
        saved = 0
        for s in syms:
            arts = self.fetch_news_rss(s)
            for a in arts:
                # Re-inserting the same headlines every refresh skewed the
                # optimism average toward whatever persisted longest (C9).
                exists = db.q("SELECT 1 FROM news WHERE sym=? AND headline=? LIMIT 1", (a["sym"], a["headline"]))
                if exists:
                    continue
                db.ex(
                    "INSERT INTO news(sym,headline,source,sentiment,score,ts)VALUES(?,?,?,?,?,?)",
                    (a["sym"], a["headline"], a["source"], a["sentiment"], a["score"], a["ts"])
                )
                saved += 1
        return saved

    def get_optimism_score(self, db, sym: str) -> tuple:
        max_age = int(self.cfg.get("news", {}).get("max_age_minutes", 180))
        cutoff = now() - timedelta(minutes=max_age)
        rows = db.q("SELECT score, sentiment, ts FROM news WHERE sym=? ORDER BY nid DESC LIMIT 50", (sym,))
        fresh = []
        for score, sentiment, timestamp in rows:
            try:
                if datetime.fromisoformat(str(timestamp)) >= cutoff:
                    fresh.append((float(score), str(sentiment)))
            except (TypeError, ValueError):
                continue
            if len(fresh) >= 10:
                break
        if not fresh:
            return 0.0, "NEUTRAL"
        scores = [row[0] for row in fresh]
        avg_score = float(sum(scores) / len(scores))
        overall_sentiment = "BULLISH" if avg_score > 0.1 else ("BEARISH" if avg_score < -0.1 else "NEUTRAL")
        return round(avg_score, 3), overall_sentiment

    # ── multi-source monitor (news-intel / social-monitor agents) ─────

    def fetch_feed(self, url: str, source_name: str, topic: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch any RSS/Atom feed through the SSRF guard and score it."""
        out: List[Dict[str, Any]] = []
        try:
            root = _parse_rss(safe_fetch(url, timeout=8))
            entries = root.findall(".//item")[:limit] or root.findall(".//entry")[:limit]
            for item in entries:
                title = item.findtext("title", "") or ""
                clean = re.sub("<[^<]+?>", "", title).strip()[:500]
                if not clean:
                    continue
                sentiment, score = self.score_text(clean)
                out.append({
                    "sym": topic, "headline": clean, "source": source_name,
                    "sentiment": sentiment, "score": score, "ts": iso(),
                })
        except SafeURLViolation as exc:
            LOG.warning(f"Feed {source_name} blocked by SSRF guard: {exc}")
        except Exception as exc:
            LOG.warning(f"Feed {source_name} unavailable: {exc.__class__.__name__}")
        return out

    def poll_sources(self, db, sources: List[Dict[str, Any]]) -> int:
        """Poll configured feeds and persist scored, deduped headlines."""
        saved = 0
        for source in sources:
            for article in self.fetch_feed(
                str(source.get("url", "")), str(source.get("name", "feed")),
                str(source.get("topic", "general")),
            ):
                exists = db.q(
                    "SELECT 1 FROM news WHERE sym=? AND headline=? LIMIT 1",
                    (article["sym"], article["headline"]),
                )
                if exists:
                    continue
                db.ex(
                    "INSERT INTO news(sym,headline,source,sentiment,score,ts)VALUES(?,?,?,?,?,?)",
                    (article["sym"], article["headline"], article["source"],
                     article["sentiment"], article["score"], article["ts"]),
                )
                saved += 1
        return saved

    def fetch_x_search(self, bearer: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """X/Twitter API v2 recent search (requires OX_X_BEARER token)."""
        try:
            from urllib.parse import quote

            url = (f"https://api.x.com/2/tweets/search/recent?query={quote(query)}"
                   f"&max_results={min(100, max(10, limit))}")
            raw = safe_fetch(
                url, timeout=8,
                headers={"Authorization": f"Bearer {bearer}"},
                allowed_hosts={"api.x.com", "api.twitter.com"},
            )
            import json as _json

            payload = _json.loads(raw.decode("utf-8", errors="replace"))
            out = []
            for tweet in payload.get("data", [])[:limit]:
                text = str(tweet.get("text", ""))[:400]
                sentiment, score = self.score_text(text)
                out.append({"text": text, "sentiment": sentiment, "score": score,
                            "source": "x_search", "query": query, "ts": iso()})
            return out
        except SafeURLViolation as exc:
            LOG.warning(f"X search blocked by SSRF guard: {exc}")
        except Exception as exc:
            LOG.warning(f"X search unavailable: {exc.__class__.__name__}")
        return []

    def fetch_telegram_channel(self, channel: str) -> List[Dict[str, Any]]:
        """Public Telegram channel preview (t.me/s/<channel>) — no API key."""
        out: List[Dict[str, Any]] = []
        channel = re.sub(r"[^A-Za-z0-9_]", "", channel)
        if not channel:
            return out
        try:
            html = safe_fetch(f"https://t.me/s/{channel}", timeout=8).decode("utf-8", errors="replace")
            blocks = re.findall(
                r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.S,
            )[-15:]
            for block in blocks:
                text = re.sub("<[^<]+?>", " ", block)
                text = re.sub(r"\s+", " ", text).strip()[:400]
                if not text:
                    continue
                sentiment, score = self.score_text(text)
                out.append({"text": text, "sentiment": sentiment, "score": score,
                            "source": f"tg:{channel}", "ts": iso()})
        except SafeURLViolation as exc:
            LOG.warning(f"Telegram preview blocked by SSRF guard: {exc}")
        except Exception as exc:
            LOG.warning(f"Telegram preview unavailable for {channel}: {exc.__class__.__name__}")
        return out

    def fetch_nitter_posts(self, instance: str, handle: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Best-effort Nitter scrape (instances come and go; purely optional)."""
        out: List[Dict[str, Any]] = []
        instance = re.sub(r"[^a-zA-Z0-9.\-]", "", instance)
        handle = re.sub(r"[^A-Za-z0-9_]", "", handle)
        if not instance or not handle:
            return out
        try:
            html = safe_fetch(f"https://{instance}/{handle}", timeout=8).decode("utf-8", errors="replace")
            blocks = re.findall(r'class="tweet-content[^"]*"[^>]*>(.*?)</div>', html, re.S)[:limit]
            for block in blocks:
                text = re.sub("<[^<]+?>", " ", block)
                text = re.sub(r"\s+", " ", text).strip()[:400]
                if not text:
                    continue
                sentiment, score = self.score_text(text)
                out.append({"text": text, "sentiment": sentiment, "score": score,
                            "source": f"nitter:{handle}", "ts": iso()})
        except SafeURLViolation as exc:
            LOG.warning(f"Nitter fetch blocked by SSRF guard: {exc}")
        except Exception as exc:
            LOG.warning(f"Nitter unavailable ({instance}/{handle}): {exc.__class__.__name__}")
        return out

