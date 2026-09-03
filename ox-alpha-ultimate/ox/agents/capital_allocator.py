"""
Capital Allocator — per-agent capital budgets and the closed-trade ledger.
==========================================================================

Splits system capital across agents, reserves/releases margin as positions
open and close, and persists every closed trade to ``promax_trades`` so the
LeverageLadder and the dashboard can compute honest, per-agent performance.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from ..core import LOG, iso
from .base import SharedDataBus


class CapitalAllocator:
    """Budgets capital per agent and keeps the closed-trade ledger."""

    def __init__(self, data_bus: SharedDataBus, config: Optional[Dict] = None, db=None):
        self.data_bus = data_bus
        self.config = config or {}
        self.db = db

        self.total_capital: float = float(self.config.get("total", 5000.0))
        weights: Dict[str, float] = dict(self.config.get("weights", {}))

        self._lock = threading.RLock()
        self.budgets: Dict[str, float] = {}
        self.reserved: Dict[str, float] = {k: 0.0 for k in weights}
        self._registered: set[str] = set()
        self._weights = weights

    # ── registration ──────────────────────────────────────────────────
    def register_agent(self, agent_id: str) -> None:
        with self._lock:
            self._registered.add(agent_id)
            weight = float(self._weights.get(agent_id, 0.0))
            self.budgets[agent_id] = self.total_capital * weight
            self.reserved.setdefault(agent_id, 0.0)
        LOG.info(f"Capital budget for {agent_id}: {self.budgets[agent_id]:.2f} "
                 f"(weight {weight:.0%} of {self.total_capital:.0f})")
        self.data_bus.publish("capital:allocation", {
            "agent_id": agent_id,
            "budget": self.budgets[agent_id],
            "reserved": self.reserved[agent_id],
        })

    def unregister_agent(self, agent_id: str) -> None:
        with self._lock:
            self._registered.discard(agent_id)

    # ── budgeting ─────────────────────────────────────────────────────
    def budget(self, agent_id: str) -> float:
        with self._lock:
            return self.budgets.get(agent_id, 0.0)

    def get_allocation(self, agent_id: str) -> float:
        """Legacy API (used by the pre-existing agents) — same as budget()."""
        return self.budget(agent_id)

    def get_agent_usage(self, agent_id: str) -> float:
        with self._lock:
            return self.reserved.get(agent_id, 0.0)

    def available(self, agent_id: str) -> float:
        with self._lock:
            return self.budgets.get(agent_id, 0.0) - self.reserved.get(agent_id, 0.0)

    def reserve(self, agent_id: str, amount: float) -> bool:
        """Reserve margin for a new position. Fails closed beyond budget."""
        with self._lock:
            free = self.available(agent_id)
            if amount <= 0 or amount > free + 1e-9:
                LOG.warning(f"Capital reserve denied for {agent_id}: "
                            f"need {amount:.2f}, available {free:.2f}")
                return False
            self.reserved[agent_id] = self.reserved.get(agent_id, 0.0) + float(amount)
        self.data_bus.publish("capital:allocation", {
            "agent_id": agent_id, "reserved": self.reserved[agent_id],
        })
        return True

    def release(self, agent_id: str, amount: float) -> None:
        with self._lock:
            self.reserved[agent_id] = max(0.0, self.reserved.get(agent_id, 0.0) - float(amount))

    def reallocate(self, weights: Dict[str, float]) -> None:
        """Redistribute budgets (e.g. shift capital away from paused agents)."""
        with self._lock:
            self._weights.update(weights)
            for agent_id in self._registered:
                self.budgets[agent_id] = self.total_capital * float(self._weights.get(agent_id, 0.0))
        LOG.info(f"Capital reallocated: {self._weights}")

    # ── ledger ────────────────────────────────────────────────────────
    def record_trade(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        leverage: float,
        reason: str = "",
        opened: str = "",
    ) -> Dict[str, Any]:
        side_multiplier = 1.0 if str(side).lower() == "long" else -1.0
        pnl = (float(exit_price) - float(entry_price)) * float(qty) * side_multiplier
        opened = opened or iso()
        if self.db is not None:
            self.db.ex(
                "INSERT INTO promax_trades(agent,symbol,side,qty,entry_price,exit_price,pnl,"
                "leverage,reason,opened,closed)VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (agent_id, symbol, side, float(qty), float(entry_price), float(exit_price),
                 float(pnl), float(leverage), reason[:120], opened, iso()),
            )
        trade = {
            "agent_id": agent_id, "symbol": symbol, "side": side, "qty": qty,
            "entry_price": entry_price, "exit_price": exit_price, "pnl": pnl,
            "leverage": leverage, "reason": reason, "closed": iso(),
        }
        self.data_bus.publish("capital:trade", trade)
        return trade

    def closed_trades(self, agent_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        if agent_id:
            rows = self.db.q(
                "SELECT agent,symbol,side,qty,entry_price,exit_price,pnl,leverage,reason,opened,closed "
                "FROM promax_trades WHERE agent=? ORDER BY ptid DESC LIMIT ?", (agent_id, int(limit))
            )
        else:
            rows = self.db.q(
                "SELECT agent,symbol,side,qty,entry_price,exit_price,pnl,leverage,reason,opened,closed "
                "FROM promax_trades ORDER BY ptid DESC LIMIT ?", (int(limit),)
            )
        keys = ("agent_id", "symbol", "side", "qty", "entry_price", "exit_price",
                "pnl", "leverage", "reason", "opened", "closed")
        return [dict(zip(keys, row)) for row in rows]

    def agent_pnl(self, agent_id: str) -> float:
        return sum(t["pnl"] for t in self.closed_trades(agent_id, limit=1000))

    def total_pnl(self) -> float:
        return sum(t["pnl"] for t in self.closed_trades(limit=1000))

    def equity(self) -> float:
        """Current system equity = initial capital + realized P&L."""
        return self.total_capital + self.total_pnl()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_capital": self.total_capital,
                "equity": self.equity(),
                "realized_pnl": self.total_pnl(),
                "budgets": dict(self.budgets),
                "reserved": dict(self.reserved),
            }
