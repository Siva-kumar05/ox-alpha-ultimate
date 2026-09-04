"""TradingView webhook integration.
Receives TradingView alert POSTs (with HMAC), stores signals, and blends them
into the ensemble. Also provides training datasets from webhook history.

Enable via: config tradingview: enabled: true, webhook_secret_env: TV_WEBHOOK_SECRET
The agent polls TradingViewBroker._signals via get_latest_signals().
"""
from __future__ import annotations
import hmac
import hashlib
import json

class TradingViewHub:
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.secret_env = cfg.get("tradingview", {}).get("webhook_secret_env", "TV_WEBHOOK_SECRET")

    def ingest(self, payload: dict, signature: str | None = None) -> bool:
        import os
        secret = os.getenv(self.secret_env, "").strip()
        if secret and signature:
            expected = hmac.new(secret.encode(), json.dumps(payload, sort_keys=True).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return False
        # Persist for training
        try:
            self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('TV_SIGNAL',?,?)",
                       (json.dumps(payload)[:1000], __import__("ox.core", fromlist=["iso"]).iso()))
            # Also kv for quick lookup
            sym = str(payload.get("symbol", payload.get("ticker", ""))).upper()
            if sym:
                self.db.kv_set(f"tv_last:{sym}", payload)
            return True
        except Exception:
            return False

    def get_signal(self, symbol: str) -> dict | None:
        try:
            return self.db.kv_get(f"tv_last:{symbol.upper()}")
        except Exception:
            return None

    def training_dataset(self, symbol: str, limit: int = 500) -> list[dict]:
        rows = self.db.q("SELECT msg,ts FROM events WHERE kind='TV_SIGNAL' ORDER BY eid DESC LIMIT ?", (limit,))
        out = []
        for msg, ts in rows:
            try:
                data = json.loads(msg)
                if str(data.get("symbol", "")).upper() == symbol.upper():
                    out.append(data)
            except Exception:
                continue
        return out
