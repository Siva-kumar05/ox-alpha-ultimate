"""
Human Approval Gateway — capital-deploying orders wait for a human.
===================================================================

Rule enforced system-wide (operator requirement):

* BUY / OPEN / any capital-deploying signal  ->  PENDING intent.  The signal
  is NOT published to the execution router until a human approves it via
  ``run.py approve <iid>``, Telegram, or the OX_PROMAX_AUTO_APPROVE=1
  smoke-test escape hatch.  Unapproved intents expire (default 5 minutes)
  so a stale approval can never execute at a stale price.
* SELL / CLOSE / modify  ->  risk-reducing, executed immediately.  A human
  is never required to get *out* of a position.

Intents persist in the ``order_intents`` table so decisions survive
restarts and are auditable.  Telegram notification is best-effort and goes
through the SSRF guard (``ox.ssrf``) with a hard host allowlist; a
notification failure must never block or delay the trading path.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core import LOG, iso
from ..ssrf import safe_post_json

AUTO_APPROVE_ENV = "OX_PROMAX_AUTO_APPROVE"
TG_TOKEN_ENV = "OX_TG_BOT_TOKEN"
TG_CHAT_ENV = "OX_TG_CHAT_ID"

# Actions that deploy capital and therefore need a human decision.
CAPITAL_DEPLOYING = {"buy", "open", "add"}
# Actions that reduce risk and therefore never wait for a human.
IMMEDIATE_ACTIONS = {"sell", "close", "modify", "flatten"}


class ApprovalGateway:
    """Queue + decision store for order intents. Fail-closed for buys."""

    def __init__(self, db, ttl_seconds: int = 300):
        self.db = db
        self.ttl_seconds = int(ttl_seconds)

    # ── classification ────────────────────────────────────────────────
    def needs_human(self, action: str) -> bool:
        return str(action).lower() in CAPITAL_DEPLOYING

    # ── submit ────────────────────────────────────────────────────────
    def submit(self, agent_id: str, signal) -> Dict[str, Any]:
        """Register a signal as an order intent.

        Returns the intent record with its final submit-time status:
        ``"PENDING"`` (waits for a human) or ``"APPROVED"`` (risk-reducing
        action, or OX_PROMAX_AUTO_APPROVE=1 in paper smoke tests).
        """
        action = str(signal.action).lower()
        now = datetime.now()
        iid = uuid.uuid4().hex[:12]
        expires = (now + timedelta(seconds=self.ttl_seconds)).isoformat()

        if not self.needs_human(action) or os.environ.get(AUTO_APPROVE_ENV) == "1":
            status = "APPROVED"
            decided_by = "auto (risk-reducing)" if not self.needs_human(action) else "auto (smoke env)"
        else:
            status = "PENDING"
            decided_by = ""

        self.db.ex(
            "INSERT INTO order_intents(iid,agent,symbol,action,qty,price,leverage,stop_loss,"
            "take_profit,reason,status,created,decided_at,decided_by,expires_at,signal_id)"
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                iid, agent_id, signal.symbol, action,
                float(signal.quantity), float(signal.price), float(signal.leverage),
                float(signal.stop_loss) if signal.stop_loss else None,
                float(signal.take_profit) if signal.take_profit else None,
                str(signal.metadata.get("reason", ""))[:200] if signal.metadata else "",
                status, iso(), iso() if status == "APPROVED" else None,
                decided_by, expires, signal.signal_id,
            ),
        )
        intent = self.get(iid)
        if status == "PENDING":
            self._notify_telegram(intent)
            LOG.info(f"Approval required: intent {iid} [{agent_id} {action.upper()} "
                     f"{signal.symbol} x{signal.quantity} @ {signal.price}]")
        return intent

    # ── decisions ─────────────────────────────────────────────────────
    def decide(self, iid: str, approve: bool, by: str = "cli") -> bool:
        row = self.get(iid)
        if not row:
            LOG.warning(f"decide: unknown intent {iid}")
            return False
        if row["status"] != "PENDING":
            LOG.warning(f"decide: intent {iid} already {row['status']}")
            return False
        self.db.ex(
            "UPDATE order_intents SET status=?, decided_at=?, decided_by=? WHERE iid=?",
            ("APPROVED" if approve else "DENIED", iso(), by, iid),
        )
        LOG.info(f"Intent {iid} {'APPROVED' if approve else 'DENIED'} by {by}")
        return True

    def wait_decision(self, iid: str, timeout_seconds: float, poll_seconds: float = 0.25) -> str:
        """Poll until the intent leaves PENDING (or TTL expiry wins)."""
        deadline = datetime.now() + timedelta(seconds=float(timeout_seconds))
        while datetime.now() < deadline:
            self.expire_stale()
            row = self.get(iid)
            if row and row["status"] != "PENDING":
                return row["status"]
            time.sleep(poll_seconds)
        return "EXPIRED"

    def expire_stale(self) -> int:
        """Expire PENDING intents past their TTL. Returns count expired."""
        now_iso = datetime.now().isoformat()
        stale = self.db.q(
            "SELECT iid FROM order_intents WHERE status='PENDING' AND expires_at < ?",
            (now_iso,),
        )
        for (iid,) in stale:
            self.db.ex(
                "UPDATE order_intents SET status='EXPIRED', decided_at=?, decided_by='ttl' WHERE iid=?",
                (iso(), iid),
            )
            LOG.info(f"Intent {iid} expired (stale approval window)")
        return len(stale)

    # ── queries ───────────────────────────────────────────────────────
    def get(self, iid: str) -> Optional[Dict[str, Any]]:
        rows = self.db.q("SELECT * FROM order_intents WHERE iid=?", (iid,))
        return self._as_dict(rows[0]) if rows else None

    def list_intents(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if status:
            rows = self.db.q(
                "SELECT * FROM order_intents WHERE status=? ORDER BY created DESC LIMIT ?",
                (status.upper(), int(limit)),
            )
        else:
            rows = self.db.q("SELECT * FROM order_intents ORDER BY created DESC LIMIT ?", (int(limit),))
        return [self._as_dict(row) for row in rows]

    def human_approved_unexecuted(self, done_keys: set[str]) -> List[Dict[str, Any]]:
        """APPROVED intents decided by a human that the router has not filled.

        ``done_keys`` carries the orchestrator's already-executed markers
        (persisted in the DB kv table as ``intent_done:<iid>``) so a restart
        cannot double-execute an old approval.
        """
        rows = self.db.q(
            "SELECT * FROM order_intents WHERE status='APPROVED' "
            "AND decided_by NOT LIKE 'auto%' ORDER BY created ASC LIMIT 50"
        )
        return [self._as_dict(row) for row in rows
                if f"intent_done:{row[0]}" not in done_keys]

    _COLUMNS = (
        "iid", "agent", "symbol", "action", "qty", "price", "leverage", "stop_loss",
        "take_profit", "reason", "status", "created", "decided_at", "decided_by",
        "expires_at", "signal_id",
    )

    def _as_dict(self, row: tuple) -> Dict[str, Any]:
        return dict(zip(self._COLUMNS, row))

    # ── notification (best effort, never blocks) ──────────────────────
    def _notify_telegram(self, intent: Dict[str, Any]) -> None:
        token = os.environ.get(TG_TOKEN_ENV, "").strip()
        chat = os.environ.get(TG_CHAT_ENV, "").strip()
        if not token or not chat:
            return
        text = (
            "OX-ALPHA approval needed\n"
            f"id: {intent['iid']}\n"
            f"{intent['agent']} {intent['action'].upper()} {intent['symbol']} "
            f"qty={intent['qty']} @ {intent['price']} lev={intent['leverage']}\n"
            f"approve:  run.py ok {intent['iid']}"
        )
        # Host is pinned by allowlist; the token only fills the fixed path.
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            safe_post_json(
                url,
                {"chat_id": chat, "text": text},
                timeout=3.0,
                allowed_hosts={"api.telegram.org"},
            )
        except Exception as exc:  # notification is strictly best-effort
            LOG.warning(f"Telegram notify failed (ignored): {exc.__class__.__name__}")


def open_gateway_from_db(db, cfg: Dict[str, Any] | None = None) -> ApprovalGateway:
    cfg = cfg or {}
    return ApprovalGateway(db, ttl_seconds=int(cfg.get("ttl_seconds", 300)))
