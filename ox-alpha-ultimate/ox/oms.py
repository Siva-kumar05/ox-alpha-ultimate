"""Order management with confirmed broker state and broker-side protective exits."""

from __future__ import annotations

import math
import threading
from pathlib import Path

from .brokers import BrokerError, OrderError
from .charges import ChargesCalculator
from .core import LOG, iso


class OMS:
    def __init__(self, cfg, db, broker, risk):
        self.cfg = cfg
        self.db = db
        self.broker = broker
        self.risk = risk
        self.calculator = ChargesCalculator(cfg.get("costs", {}))
        self.positions: dict[str, dict] = {}
        self.inflight: set[str] = set()
        self.inflight_orders: dict[str, str] = {}
        self.lock = threading.RLock()
        self.live = True

    @staticmethod
    def _finite_price(value: float, name: str) -> float:
        price = float(value)
        if not math.isfinite(price) or price <= 0:
            raise OrderError(f"{name} must be a positive finite price")
        return price

    def _position_key(self, sym: str) -> str:
        return f"position_super_order:{sym}"

    def restore(self) -> None:
        """Restore only positions previously opened by this process, then reconcile them."""
        with self.lock:
            self.positions.clear()
            for sym, quantity, average, stop, target, opened, strategy in self.db.q("SELECT sym,qty,avg,sl,tp,opened,strat FROM positions"):
                if quantity <= 0:
                    self.db.ex("DELETE FROM positions WHERE sym=?", (sym,))
                    continue
                order_id = self.db.kv_get(self._position_key(sym))
                if not order_id:
                    raise OrderError(f"Persisted position {sym} has no linked Super Order")
                self.positions[sym] = {
                    "sym": sym, "qty": int(quantity), "avg": float(average), "sl": float(stop), "tp": float(target),
                    "opened": opened, "strat": strategy, "super_order_id": order_id,
                }
        self.reconcile()

    def open_position(self, sym: str, side: str, qty: int, strategy: str, stop: float, target: float, reason: str = "signal") -> dict | None:
        """Open a long with its stop and target atomically at the broker.

        Long-only is deliberate: automatic SELL actions close agent-owned longs;
        shorting is not enabled until its borrowing, tax, and broker constraints
        are modelled and tested separately.
        """
        if side != "BUY":
            raise OrderError("This agent is long-only; SELL is reserved for autonomous exits")
        if qty <= 0:
            raise OrderError("Position quantity must be positive")
        stop, target = self._finite_price(stop, "stop"), self._finite_price(target, "target")
        with self.lock:
            if not self.live:
                raise OrderError("OMS is halted")
            if sym in self.inflight or self.positions.get(sym, {}).get("qty", 0):
                return None
            self.inflight.add(sym)
        try:
            receipt = self.broker.place_super_order(sym, "BUY", qty, target, stop, f"OX_ENTRY_{strategy}_{reason}"[:80])
            with self.lock:
                self.inflight_orders[sym] = receipt.order_id
            confirmation = self.broker.wait_super_order(receipt.order_id, self.cfg["execution"]["order_confirm_timeout_seconds"])
            if confirmation.status not in {"TRADED", "PART_TRADED"} or confirmation.filled_qty <= 0:
                raise OrderError(f"Super Order {receipt.order_id} was not filled: {confirmation.status}")
            if confirmation.status == "PART_TRADED":
                # A partial entry leaves a working residual leg at the broker.
                # Cancel it and re-read the final filled quantity so local state
                # can never diverge from a later surprise fill (A1).
                try:
                    self.broker.cancel_super_order(receipt.order_id)
                    final = self.broker.wait_super_order(receipt.order_id, self.cfg["execution"]["order_confirm_timeout_seconds"])
                except BrokerError as cancel_exc:
                    raise OrderError(
                        f"Partial Super Order {receipt.order_id} residual could not be cancelled: {cancel_exc}"
                    ) from cancel_exc
                if int(final.filled_qty) <= 0 or final.status == "PART_TRADED":
                    raise OrderError(f"Partial Super Order {receipt.order_id} residual unresolved: {final.status}")
                confirmation = final
                self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('PARTIAL_ENTRY_CANCELLED',?,?)", (f"{sym} {receipt.order_id} filled={confirmation.filled_qty}", iso()))
            fill = self._finite_price(confirmation.average_price, "confirmed fill")
            filled_quantity = min(int(confirmation.filled_qty), qty)
            breakeven_target = self.calculator.min_breakeven_sell_price(fill, filled_quantity, buffer_pct=0.001)
            # The confirmation must sit inside the ORIGINALLY requested bracket.
            # Comparing against max(target, breakeven_target) would make the
            # upper bound vacuous: breakeven is always above the fill (fees +
            # buffer), so any fill above the requested target would be accepted
            # and then "covered" by bumping the target - leaving the broker's
            # armed bracket below the local one.  A fill beyond the requested
            # target is anomalous (gap through the take-profit); fail closed.
            if not stop < fill < target:
                # The broker already has its originally requested bracket.  Treat an invalid confirmation as uncertain state.
                raise OrderError("Confirmed fill is outside the requested protective bracket")
            if breakeven_target > target:
                # The fee-coverage bump is an optimisation, not protection: the
                # broker's original target and stop stay armed if this fails,
                # so a modify error must never orphan a filled position (A3).
                try:
                    self.broker.modify_super_target(receipt.order_id, breakeven_target)
                    enforced_target = breakeven_target
                except BrokerError as modify_exc:
                    self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('TARGET_MODIFY_FAILED',?,?)", (f"{sym} {receipt.order_id}: {modify_exc}", iso()))
                    self.db.audit("TARGET_MODIFY_FAILED", {"sym": sym, "order_id": receipt.order_id})
                    LOG.warning("Kept broker-side target for %s after modify failure: %s", sym, modify_exc.__class__.__name__)
                    enforced_target = target
            else:
                enforced_target = target
            position = {
                "sym": sym, "qty": filled_quantity, "avg": fill, "sl": stop, "tp": enforced_target,
                "strat": strategy, "opened": iso(), "super_order_id": receipt.order_id,
            }
            with self.lock:
                self.positions[sym] = position
                self.db.ex("INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?)", (sym, filled_quantity, fill, stop, enforced_target, position["opened"], strategy))
                self.db.kv_set(self._position_key(sym), receipt.order_id)
                self.inflight_orders.pop(sym, None)
                self.db.audit("POSITION_OPENED", {"sym": sym, "qty": filled_quantity, "entry": round(fill, 2), "stop": round(stop, 2), "target": round(enforced_target, 2), "super_order_id": receipt.order_id})
            LOG.info("Confirmed Super Order: %s BUY %s @ %.2f; SL %.2f / TP %.2f", sym, filled_quantity, fill, stop, enforced_target)
            return position
        except Exception as exc:
            # A timeout leaves broker state uncertain. Attempt to cancel the tracked entry,
            # but retain its ID for kill-switch auditing if cancellation itself fails.
            with self.lock:
                pending_order = self.inflight_orders.get(sym)
            if pending_order:
                try:
                    self.broker.cancel_super_order(pending_order)
                except Exception as cancel_error:
                    LOG.critical("Could not cancel uncertain Super Order %s: %s", pending_order, cancel_error)
                    raise OrderError(
                        f"Could not cancel uncertain Super Order {pending_order}; "
                        "execution state requires reconciliation"
                    ) from exc
                else:
                    with self.lock:
                        self.inflight_orders.pop(sym, None)
            raise
        finally:
            with self.lock:
                self.inflight.discard(sym)

    def _remove_position(self, sym: str) -> None:
        self.positions.pop(sym, None)
        self.db.ex("DELETE FROM positions WHERE sym=?", (sym,))
        self.db.ex("DELETE FROM kv WHERE k=?", (self._position_key(sym),))

    def mark(self, sym: str, ltp: float) -> None:
        """Paper simulation needs local trigger handling; live exits stay broker-side."""
        if self.cfg["mode"] != "paper":
            return
        position = self.positions.get(sym)
        if not position:
            return
        if ltp <= position["sl"]:
            self.close(sym, "STOP_LOSS")
        elif ltp >= position["tp"]:
            self.close(sym, "TAKE_PROFIT")

    def close(self, sym: str, reason: str, *, force: bool = False) -> bool:
        """Cancel the linked bracket then verify a market exit before changing local state."""
        with self.lock:
            position = self.positions.get(sym)
            if not position:
                return True
            if not self.live and not force:
                return False
            quantity = int(position["qty"])
            super_order_id = position["super_order_id"]
        try:
            self.broker.cancel_super_order(super_order_id)
        except BrokerError:
            # The bracket may already be gone (filled, cancelled, expired).
            # The exit path below still runs; protection is re-checked there.
            LOG.warning("Bracket cancel for %s %s failed; continuing to exit path", sym, super_order_id)
        exit_vwap = 0.0
        total_filled = 0
        remaining = quantity
        attempts = 3
        while remaining > 0 and attempts > 0:
            attempts -= 1
            receipt = self.broker.exit_position(sym, "SELL", remaining, f"OX_EXIT_{reason}"[:80])
            confirmation = self.broker.wait_order(receipt.order_id, self.cfg["execution"]["order_confirm_timeout_seconds"])
            filled = int(confirmation.filled_qty)
            if filled > 0:
                price = self._finite_price(confirmation.average_price, "confirmed exit fill")
                exit_vwap = (exit_vwap * total_filled + price * filled) / (total_filled + filled)
                total_filled += filled
                remaining -= filled
        if remaining > 0:
            # Journal the already-filled portion first: otherwise those units'
            # P&L vanishes from the trades/equity record while only the
            # remainder stays open (found in the v3 self-review).
            if total_filled > 0:
                partial_charges = self.calculator.compute_charges(position["avg"], exit_vwap, total_filled)
                partial_pnl = partial_charges["net_pnl"]
                with self.lock:
                    self.db.ex(
                        "INSERT INTO trades(sym,side,qty,inpx,outpx,pnl,charges,strat,intime,outtime,exit_reason)VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (sym, "LONG", total_filled, position["avg"], exit_vwap, round(partial_pnl, 2), partial_charges["total_charges"], position["strat"], position["opened"], iso(), f"{reason}_PARTIAL"),
                    )
                    self.risk.on_trade_close(partial_pnl)
                self.db.audit("PARTIAL_EXIT_JOURNALED", {"sym": sym, "qty": total_filled, "exit": round(exit_vwap, 2), "net_pnl": round(float(partial_pnl), 2)})
                LOG.warning("Journaled partial exit %s: %s units @ %.2f (net %.2f); %s remain", sym, total_filled, exit_vwap, partial_pnl, remaining)
            # Partial exit (A2): re-arm a broker-side stop for the remainder so
            # exposure is never left unprotected, shrink local state to match
            # the broker, then fail closed for operator reconciliation.
            try:
                self.broker.place_protective_stop(sym, remaining, position["sl"], f"OX_REARM_{reason}"[:80])
                self.db.audit("PROTECTIVE_STOP_REARMED", {"sym": sym, "qty": remaining, "trigger": round(float(position["sl"]), 2)})
                LOG.critical("Exit for %s incomplete; re-armed broker stop for %s units", sym, remaining)
            except BrokerError as rearm_exc:
                raise OrderError(f"Exit incomplete for {sym} ({remaining} left) and stop re-arm failed: {rearm_exc}")
            with self.lock:
                position["qty"] = remaining
                self.db.ex("UPDATE positions SET qty=? WHERE sym=?", (remaining, sym))
            raise OrderError(f"Exit for {sym} partially filled; {remaining} units protected by re-armed stop")
        exit_price = exit_vwap
        charges = self.calculator.compute_charges(position["avg"], exit_price, quantity)
        pnl = charges["net_pnl"]
        with self.lock:
            self.db.ex(
                "INSERT INTO trades(sym,side,qty,inpx,outpx,pnl,charges,strat,intime,outtime,exit_reason)VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (sym, "LONG", quantity, position["avg"], exit_price, round(pnl, 2), charges["total_charges"], position["strat"], position["opened"], iso(), reason),
            )
            self.risk.on_trade_close(pnl)
            self._remove_position(sym)
            self.db.audit("POSITION_CLOSED", {"sym": sym, "qty": quantity, "exit": round(exit_price, 2), "net_pnl": round(float(pnl), 2), "reason": reason})
        LOG.info("Confirmed exit: %s %s @ %.2f; net PnL %.2f", sym, reason, exit_price, pnl)
        return True

    def reconcile(self) -> None:
        """Refuse to trade through broker/local discrepancies; never invent fills or P&L."""
        broker_positions = {row.get("sym") or row.get("tradingSymbol"): row for row in self.broker.positions()}
        with self.lock:
            for sym, position in list(self.positions.items()):
                broker_quantity = int(broker_positions.pop(sym, {}).get("netQty", 0))
                if broker_quantity == position["qty"]:
                    continue
                if broker_quantity == 0:
                    self._remove_position(sym)
                    self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('BROKER_EXIT',?,?)", (f"{sym} no longer open; reconcile P&L from broker statement", iso()))
                    self.db.audit("EXTERNAL_POSITION_EXIT", {"sym": sym, "local_qty": int(position["qty"])})
                    LOG.warning("Broker closed %s outside process; local P&L intentionally not fabricated", sym)
                    continue
                self.live = False
                self.db.audit("POSITION_MISMATCH", {"sym": sym, "broker_qty": broker_quantity, "local_qty": int(position["qty"])})
                raise OrderError(f"Broker/local position mismatch for {sym}: broker={broker_quantity}, local={position['qty']}")
            unexpected = {sym: int(row.get("netQty", 0)) for sym, row in broker_positions.items() if int(row.get("netQty", 0))}
            if unexpected:
                self.live = False
                self.db.audit("UNEXPECTED_BROKER_POSITION", {"symbols": sorted(unexpected)})
                raise OrderError(f"Unexpected broker positions detected: {unexpected}")

    def kill_switch(self, reason: str) -> None:
        """Fail closed and flatten known agent positions without touching unrelated account activity.

        The KILL.flag is written even when there is nothing to flatten: a
        compliance halt is a durable stop, and the flag is what makes a
        restarted agent refuse to boot until an operator reconciles.  Without
        it, a halt that had no open position or pending order would leave no
        marker and the agent could silently restart.
        """
        with self.lock:
            self.live = False
            symbols = list(self.positions)
            pending_orders = dict(self.inflight_orders)
        LOG.critical("OMS KILL SWITCH: %s", reason)
        self.db.audit("KILL_SWITCH", {"reason": reason[:200], "positions": symbols, "pending_orders": sorted(pending_orders)})
        failures = []
        for sym, order_id in pending_orders.items():
            try:
                self.broker.cancel_super_order(order_id)
                with self.lock:
                    self.inflight_orders.pop(sym, None)
            except Exception as exc:
                failures.append(f"uncertain entry {sym}/{order_id}: {exc}")
                LOG.critical("Unable to cancel uncertain entry %s: %s", order_id, exc)
        for sym in symbols:
            try:
                self.close(sym, "KILL_SWITCH", force=True)
            except Exception as exc:
                failures.append(f"{sym}: {exc}")
                LOG.critical("Unable to confirm emergency exit for %s: %s", sym, exc)
        kill_path = Path(self.cfg.root) / "KILL.flag"
        kill_path.write_text(f"HALTED: {reason} at {iso()}\n", encoding="utf-8")
        message = reason if not failures else f"{reason}; unconfirmed exits: {' | '.join(failures)}"
        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('KILL',?,?)", (message, iso()))

    def squareoff_eod(self) -> None:
        for sym in list(self.positions):
            self.close(sym, "EOD_SQUAREOFF")
