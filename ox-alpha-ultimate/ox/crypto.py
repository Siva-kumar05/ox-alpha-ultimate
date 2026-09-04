"""Crypto micro broker: paper simulation (default) + live Binance (spot & USDT-M perps).

Paper mode (default) simulates fractional balances starting from $0.9 USDT
and enforces the venue minimum notionals exactly as before: the below-minimum
fractional start is simulated only through ``simulate_compound()``, never
through an order a live venue would reject.  Paper ``place_market`` returns a
fabricated fill the same way it always has.

Live mode (``mode: live`` in the config) wires the SAME order surface to a
real Binance account through ccxt:

* Spot symbols (``crypto.markets: {PEPEUSDT: spot}``) are cash-only: any
  order with leverage > 1.0 is refused before it reaches the venue.
* Swap symbols (``crypto.markets: {BTCUSDT: swap}``) trade USDT-M
  perpetuals: leverage is applied per symbol before the entry (cross
  margin), funding is real, and closes are sent ``reduceOnly`` so an exit
  can never accidentally open a new position.

Fail-closed rules: live login requires BINANCE_API_KEY/BINANCE_API_SECRET
and refuses to start when a configured symbol does not resolve on the
venue; every live order re-checks leverage vs market type; a live tick or
fill that cannot be confirmed raises instead of fabricating a result.
Paper mode never imports ccxt, so the offline suite runs without it.

Design mirrors the NSE adapter surface so the promax ExecutionRouter can
trade both venues through one code path.
"""
from __future__ import annotations

import time, threading, random

from .brokers import AuthenticationError, BrokerError, MarketDataError, OrderError


class CryptoMicroBroker:
    """Paper + live Binance (spot and USDT-M perp) adapter. Defaults to paper."""

    def __init__(self, cfg, db):
        self.cfg = cfg or {}
        self.db = db
        self.mode = str(self.cfg.get("mode", "paper")).lower()
        self.live = self.mode == "live"
        self.name = "binance" if self.live else "crypto_paper"

        crypto_cfg = dict(self.cfg.get("crypto", {}) or {})
        # sym -> "spot" | "swap"; symbols absent from the map default to spot.
        self.markets = {str(k).upper(): str(v).lower() for k, v in (crypto_cfg.get("markets", {}) or {}).items()}
        self.api_key_env = crypto_cfg.get("api_key_env", "BINANCE_API_KEY")
        self.api_secret_env = crypto_cfg.get("api_secret_env", "BINANCE_API_SECRET")

        self.usdt = float(crypto_cfg.get("paper_start_usdt", 0.9))
        self.min_notional = float(crypto_cfg.get("min_notional_usdt", 5.0))
        self.fee_bps = float(crypto_cfg.get("fees_bps", 10.0)) / 10000
        self.slip_bps = float(crypto_cfg.get("slippage_bps", 5.0)) / 10000
        self.prices = {"BTCUSDT": 68000, "ETHUSDT": 3200, "SOLUSDT": 150}
        self.pos = {}  # sym -> {qty, avg}
        self._lock = threading.RLock()
        self._rnd = random.Random(42)

        self._ccxt = None  # live only; created in login()
        self._ticker_cache = {}  # sym -> (ts, price)
        self._lev_cache = {}  # ccxt symbol -> leverage already set
        self._warned = set()  # symbols with a logged funding/OI failure

    # ── auth / venue wiring ────────────────────────────────────────────
    def login(self):
        if not self.live:
            return True
        if not self.markets:
            # Live promax may trade NSE only: with zero crypto symbols
            # configured there is nothing to authenticate against.
            return True
        import os

        api_key = os.getenv(self.api_key_env, "").strip()
        api_secret = os.getenv(self.api_secret_env, "").strip()
        if not api_key or not api_secret:
            raise AuthenticationError(
                f"live crypto requires {self.api_key_env} and {self.api_secret_env} environment variables"
            )
        import ccxt

        self._ccxt = ccxt.binance({
            "apiKey": api_key, "secret": api_secret, "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        try:
            self._ccxt.load_markets()
        except Exception as exc:  # network or venue auth failure: fail closed
            self._ccxt = None
            raise AuthenticationError(f"Binance market data unavailable: {exc.__class__.__name__}: {exc}") from exc
        missing = [sym for sym in self.markets if self._ccxt_symbol(sym) not in self._ccxt.markets]
        if missing:
            self._ccxt = None  # a client that cannot resolve every symbol is unusable
            raise BrokerError(
                f"configured crypto symbols do not resolve on Binance: {', '.join(sorted(missing))}"
            )
        return True

    def authenticated(self) -> bool:
        return self._ccxt is not None if self.live else True

    def _require_live_client(self, what: str, exc=OrderError) -> None:
        """Fail closed when a live venue call happens before login()."""
        if self.live and self._ccxt is None:
            raise exc(f"live crypto {what} requires login() first: broker is not authenticated")

    def _market_type(self, sym: str) -> str:
        return self.markets.get(str(sym).upper(), "spot")

    def _ccxt_symbol(self, sym: str) -> str:
        """Map exchange symbol (BTCUSDT) to a ccxt unified symbol."""
        sym = str(sym).upper()
        base = sym[:-4] if sym.endswith("USDT") else sym
        unified = f"{base}/USDT"
        if self._market_type(sym) == "swap":
            unified += ":USDT"
        return unified

    # ── market data ────────────────────────────────────────────────────
    def ltp(self, sym):
        sym = str(sym).upper()
        if not self.live:
            with self._lock:
                p = self.prices.get(sym, 100)
                p = max(0.01, p + self._rnd.gauss(0, p * 0.0008))
                self.prices[sym] = p
                return p
        self._require_live_client("ticker", MarketDataError)
        now = time.monotonic()
        cached = self._ticker_cache.get(sym)
        if cached and now - cached[0] < 2.0:
            return cached[1]
        try:
            ticker = self._ccxt.fetch_ticker(self._ccxt_symbol(sym))
            price = float(ticker.get("last") or 0.0)
        except Exception as exc:
            raise MarketDataError(f"Binance ticker failed for {sym}: {exc.__class__.__name__}: {exc}") from exc
        if price <= 0:
            raise MarketDataError(f"Binance returned an invalid price for {sym}")
        self._ticker_cache[sym] = (now, price)
        return price

    def hist(self, sym, tf_min=1, days=5):
        """OHLCV history for charting; paper returns [] (no fabricated candles)."""
        if not self.live:
            return []
        self._require_live_client("history", MarketDataError)
        sym = str(sym).upper()
        timeframe = {1: "1m", 5: "5m", 15: "15m", 60: "1h"}.get(int(tf_min), f"{int(tf_min)}m")
        limit = max(1, int(days) * 1440 // max(1, int(tf_min)))
        try:
            rows = self._ccxt.fetch_ohlcv(self._ccxt_symbol(sym), timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise MarketDataError(f"Binance history failed for {sym}: {exc.__class__.__name__}: {exc}") from exc
        return [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]

    # ── perp fundamentals (live only; None in paper -> DataPump synthesizes) ──
    def funding_rate(self, sym):
        if not self.live or self._market_type(sym) != "swap":
            return None
        self._require_live_client("funding rate", MarketDataError)
        try:
            row = self._ccxt.fetch_funding_rate(self._ccxt_symbol(sym))
            return float(row.get("fundingRate") or 0.0)
        except Exception as exc:
            if sym not in self._warned:
                self._warned.add(sym)
                print(f"[crypto] funding rate unavailable for {sym}: {exc.__class__.__name__}: {exc}")
            return None

    def open_interest(self, sym):
        if not self.live or self._market_type(sym) != "swap":
            return None
        self._require_live_client("open interest", MarketDataError)
        try:
            row = self._ccxt.fetch_open_interest(self._ccxt_symbol(sym))
            return float(row.get("openInterestAmount") or row.get("openInterestValue") or 0.0)
        except Exception as exc:
            if sym not in self._warned:
                self._warned.add(sym)
                print(f"[crypto] open interest unavailable for {sym}: {exc.__class__.__name__}: {exc}")
            return None

    # ── orders ─────────────────────────────────────────────────────────
    def place_market(self, sym, side, qty, leverage=1.0, reduce_only=False):
        sym = str(sym).upper()
        side = str(side).upper()
        if side not in {"BUY", "SELL"} or float(qty) <= 0:
            raise OrderError("Invalid crypto order side or quantity")
        leverage = max(1.0, float(leverage))

        if not self.live:
            price = self.ltp(sym)
            notional = float(qty) * price
            # The order surface enforces the venue minimum unconditionally; the
            # below-minimum fractional start is simulated only through
            # simulate_compound(), never through an order that a live venue
            # would reject.
            if notional < self.min_notional:
                raise ValueError(
                    f"order notional {notional:.2f} USDT is below venue minimum {self.min_notional:.2f}"
                )
            fee = float(qty) * price * self.fee_bps
            slip = price * self.slip_bps
            fill = price * (1 + (slip / price) if side == "BUY" else -(slip / price))
            cost = float(qty) * fill + fee if side == "BUY" else -float(qty) * fill + fee
            return {"order_id": f"CR{int(time.time()*1000)}", "price": fill, "qty": float(qty), "fee": fee, "side": side}

        market = self._market_type(sym)
        if market == "spot" and leverage > 1.0 + 1e-9:
            raise OrderError(
                f"Refusing leveraged spot order for {sym}: spot is cash-only "
                f"(leverage {leverage:.2f}x requested)"
            )
        self._require_live_client("order")
        ccxt_sym = self._ccxt_symbol(sym)
        amount = float(self._ccxt.amount_to_precision(ccxt_sym, float(qty)))
        if amount <= 0:
            raise OrderError(f"Quantity {qty} for {sym} rounds to zero at venue precision")
        params = {}
        if market == "swap":
            if not reduce_only:
                self._set_leverage(ccxt_sym, leverage)
            # reduceOnly is only meaningful on exits; entries go out without it.
            if reduce_only:
                params["reduceOnly"] = True
        try:
            order = self._ccxt.create_order(ccxt_sym, "market", side.lower(), amount, None, params)
        except Exception as exc:
            raise OrderError(f"Binance order failed for {sym}: {exc.__class__.__name__}: {exc}") from exc
        fill_price = float(order.get("average") or order.get("price") or 0.0)
        if fill_price <= 0:
            # A market fill without a price means the venue did not confirm.
            raise OrderError(f"Binance fill for {sym} lacks a confirmed price; state is uncertain")
        fee = 0.0
        order_fee = order.get("fee")
        if isinstance(order_fee, dict):
            fee = float(order_fee.get("cost") or 0.0)
        return {
            "order_id": str(order.get("id") or f"BIN{int(time.time()*1000)}"),
            "price": fill_price,
            "qty": float(order.get("filled") or amount),
            "fee": fee,
            "side": side,
        }

    def _set_leverage(self, ccxt_sym: str, leverage: float) -> None:
        if self._lev_cache.get(ccxt_sym) == leverage:
            return
        try:
            self._ccxt.set_margin_mode("cross", ccxt_sym)
            self._ccxt.set_leverage(leverage, ccxt_sym)
            self._lev_cache[ccxt_sym] = leverage
        except Exception as exc:
            raise OrderError(
                f"Binance leverage {leverage:.1f}x could not be set for {ccxt_sym}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

    # ── positions ──────────────────────────────────────────────────────
    def positions(self):
        """Normalized {sym: {qty, avg, leverage, side}}; paper returns simulated state."""
        if not self.live:
            return dict(self.pos)
        self._require_live_client("positions", MarketDataError)
        rows = {}
        # USDT-M swaps: real contracts held, with entry price and leverage.
        try:
            for p in self._ccxt.fetch_positions():
                qty = float(p.get("contracts") or 0.0)
                if qty != 0 and p.get("symbol"):
                    rows[p["symbol"]] = {
                        "qty": qty,
                        "avg": float(p.get("entryPrice") or 0.0),
                        "leverage": float(p.get("leverage") or 1.0),
                        "side": str(p.get("side") or "long"),
                    }
        except Exception as exc:
            if "swap" not in self._warned:
                self._warned.add("swap")
                print(f"[crypto] positions fetch failed: {exc.__class__.__name__}: {exc}")
        # Spot: holdings of the base asset for configured spot markets.
        try:
            balance = self._ccxt.fetch_balance()
        except Exception as exc:
            if "balance" not in self._warned:
                self._warned.add("balance")
                print(f"[crypto] balance fetch failed: {exc.__class__.__name__}: {exc}")
            balance = {}
        for sym, market in self.markets.items():
            if market != "spot":
                continue
            base = sym[:-4] if sym.endswith("USDT") else sym
            asset = balance.get(base)
            if isinstance(asset, dict):
                qty = float(asset.get("total") or 0.0)
            else:
                qty = float(asset or 0.0)
            if qty > 0:
                rows[self._ccxt_symbol(sym)] = {"qty": qty, "avg": 0.0, "leverage": 1.0, "side": "long"}
        return rows

    def simulate_compound(self, trades=1000, win_rate=0.55, rr=1.0):
        """Illustrative compounding from paper_start; NOT a profit promise."""
        bal = self.usdt
        notional = self.min_notional
        # below-minimum fractional mode
        for i in range(trades):
            if bal < self.min_notional:
                risk = bal * 0.02
                qty = risk / self.prices["BTCUSDT"]
            else:
                qty = min(bal * 0.02, notional) / self.prices["BTCUSDT"]
            win = self._rnd.random() < win_rate
            pnl = qty * self.prices["BTCUSDT"] * 0.005 * rr if win else -qty * self.prices["BTCUSDT"] * 0.005
            pnl -= qty * self.prices["BTCUSDT"] * self.fee_bps
            bal = max(0, bal + pnl)
            if bal > notional * 2:
                notional = min(notional * 1.05, 50)
        return bal