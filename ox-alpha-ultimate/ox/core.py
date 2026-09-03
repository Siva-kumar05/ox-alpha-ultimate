"""Shared configuration, persistence, and security primitives for ox-alpha."""

from __future__ import annotations

import ipaddress
import hashlib
import hmac
import json
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
import re

IST = timezone(timedelta(hours=5, minutes=30))
LOG = logging.getLogger("ox")



# The execution client is deliberately narrower than the broker API.  This is
# a defence-in-depth allowlist: adding a new remote mutation requires a code
# review, rather than merely knowing a URL.
_DHAN_FIXED_ROUTES = {
    ("GET", "/fundlimit"), ("GET", "/positions"),
    ("GET", "/super/orders"), ("GET", "/ip/getIP"),
    ("POST", "/marketfeed/ltp"), ("POST", "/marketfeed/quote"),
    ("POST", "/charts/intraday"), ("POST", "/orders"),
    ("POST", "/super/orders"),
}
_DHAN_DYNAMIC_ROUTES = (
    ("GET", re.compile(r"^/orders/[A-Za-z0-9_-]+$")),
    ("PUT", re.compile(r"^/super/orders/[A-Za-z0-9_-]+$")),
    ("DELETE", re.compile(r"^/super/orders/[A-Za-z0-9_-]+/ENTRY_LEG$")),
)


def allowed_dhan_route(method: str, path: str) -> bool:
    """Return whether method + path is inside the reviewed Dhan surface."""
    pair = (str(method).upper(), str(path))
    if pair in _DHAN_FIXED_ROUTES:
        return True
    return any(pair[0] == perm and pattern.fullmatch(pair[1])
               for perm, pattern in _DHAN_DYNAMIC_ROUTES)


class ConfigError(ValueError):
    """Raised when a configuration is malformed or unsafe."""


class SecurityError(RuntimeError):
    """Raised when code attempts an action outside the trading boundary."""


def now() -> datetime:
    return datetime.now(IST)


def hhmm() -> str:
    return now().strftime("%H:%M")


def iso(t: datetime | None = None) -> str:
    return (t or now()).isoformat()


def setup_log(log_path: str | Path = "oxalpha.log") -> None:
    """Configure the process logger once, without leaking secrets into output."""
    if LOG.handlers:
        return
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
    LOG.addHandler(stream_handler)
    LOG.propagate = False


BLOCK_PATTERNS = ("withdraw", "payout", "transfer_fund", "fund_transfer", "withdrawals", "payouts")


def guard_endpoint(url: str) -> None:
    """Enforce the intentionally narrow scope: market data and order management only."""
    normalized = str(url).lower()
    if any(pattern in normalized for pattern in BLOCK_PATTERNS):
        raise SecurityError(f"Blocked prohibited funds endpoint: {url}")


def _finite_number(value: object, key: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "a positive finite number" if positive else "finite"
        raise ConfigError(f"{key} must be {qualifier}")
    return number


class Cfg:
    """Loads a small, validated configuration and anchors relative paths to it."""

    def __init__(self, path: str | Path = "config.yaml"):
        self.path = Path(path).expanduser().resolve()
        self.root = self.path.parent
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration: {self.path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError("Configuration is not valid YAML") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError("Configuration must be a YAML mapping")
        self.d = dict(raw)
        self._validate()

    def _validate(self) -> None:
        mode = self.d.get("mode", "paper")
        platform = self.d.get("platform", "paper")
        if mode not in {"paper", "live"}:
            raise ConfigError("mode must be 'paper' or 'live'")
        if platform not in {"paper", "dhan"}:
            raise ConfigError(
                "platform must be 'paper' or 'dhan' (groww/tradingview/crypto adapters are "
                "research-only scaffolds and are deliberately not selectable: see ox/core.py)"
            )
        if mode == "live" and platform != "dhan":
            raise ConfigError("live mode currently supports platform: dhan only")
        if mode == "live" and os.getenv("OX_LIVE_EXECUTION_APPROVED", "") != "YES_I_UNDERSTAND_LIVE_TRADING":
            raise ConfigError(
                "live mode requires explicit operator affirmation in the host environment: "
                "OX_LIVE_EXECUTION_APPROVED=YES_I_UNDERSTAND_LIVE_TRADING"
            )

        symbols = self.d.get("symbols")
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) and s.isalnum() for s in symbols):
            raise ConfigError("symbols must be a non-empty list of alphanumeric trading symbols")
        self.d["symbols"] = [s.upper() for s in symbols]
        self.d["capital"] = _finite_number(self.d.get("capital"), "capital", positive=True)
        self.d["tick_seconds"] = max(1, int(_finite_number(self.d.get("tick_seconds", 1), "tick_seconds", positive=True)))
        if mode == "live" and self.d["tick_seconds"] < 2:
            raise ConfigError("live mode requires tick_seconds of at least 2 to leave rate-limit headroom")
        self.d["timeframe_sec"] = max(60, int(_finite_number(self.d.get("timeframe_sec", 60), "timeframe_sec", positive=True)))
        self.d["history_days"] = int(_finite_number(self.d.get("history_days", 95), "history_days", positive=True))
        if self.d["history_days"] > 365:
            raise ConfigError("history_days cannot exceed 365; retain more only after capacity testing")

        execution = dict(self.d.get("execution") or {})
        execution.setdefault("autonomous", True)
        execution.setdefault("allow_short", False)
        execution.setdefault("order_confirm_timeout_seconds", 8)
        execution.setdefault("max_data_staleness_seconds", 10)
        execution.setdefault("trailing_jump", 0.0)
        execution.setdefault("max_consecutive_broker_errors", 5)
        execution.setdefault("rate_limit_backoff_seconds", 3.0)
        execution.setdefault("max_rate_limit_backoff_seconds", 30.0)
        execution.setdefault("signal_history_candles", 300)
        execution.setdefault("reconcile_interval_seconds", 30)
        execution.setdefault("min_vote_fraction", 0.0)
        execution.setdefault("min_support_strategies", 1)
        execution.setdefault("history_chunk_days", 25)
        if not isinstance(execution["autonomous"], bool) or not isinstance(execution["allow_short"], bool):
            raise ConfigError("execution.autonomous and execution.allow_short must be booleans")
        if execution["allow_short"]:
            raise ConfigError("allow_short is not supported: SELL actions are autonomous exits of agent-owned longs")
        for key in ("order_confirm_timeout_seconds", "max_data_staleness_seconds", "max_consecutive_broker_errors", "signal_history_candles"):
            execution[key] = int(_finite_number(execution[key], f"execution.{key}", positive=True))
        if execution["signal_history_candles"] < 100:
            raise ConfigError("execution.signal_history_candles must be at least 100")
        execution["reconcile_interval_seconds"] = int(_finite_number(execution["reconcile_interval_seconds"], "execution.reconcile_interval_seconds", positive=True))
        if execution["reconcile_interval_seconds"] < 5:
            raise ConfigError("execution.reconcile_interval_seconds must be at least 5")
        execution["min_support_strategies"] = int(_finite_number(execution["min_support_strategies"], "execution.min_support_strategies", positive=True))
        execution["history_chunk_days"] = int(_finite_number(execution["history_chunk_days"], "execution.history_chunk_days", positive=True))
        if execution["history_chunk_days"] < 5:
            raise ConfigError("execution.history_chunk_days must be at least 5")
        execution["min_vote_fraction"] = _finite_number(execution["min_vote_fraction"], "execution.min_vote_fraction")
        if not 0.0 <= execution["min_vote_fraction"] <= 1.0:
            raise ConfigError("execution.min_vote_fraction must be between 0 and 1")
        execution["trailing_jump"] = _finite_number(execution["trailing_jump"], "execution.trailing_jump")
        for key in ("rate_limit_backoff_seconds", "max_rate_limit_backoff_seconds"):
            execution[key] = _finite_number(execution[key], f"execution.{key}", positive=True)
        if execution["max_rate_limit_backoff_seconds"] < execution["rate_limit_backoff_seconds"]:
            raise ConfigError("execution.max_rate_limit_backoff_seconds must be at least execution.rate_limit_backoff_seconds")
        self.d["execution"] = execution

        news = dict(self.d.get("news") or {})
        news.setdefault("refresh_seconds", 900)
        news.setdefault("max_age_minutes", 180)
        for key in ("refresh_seconds", "max_age_minutes"):
            news[key] = int(_finite_number(news[key], f"news.{key}", positive=True))
        self.d["news"] = news

        risk = dict(self.d.get("risk") or {})
        required_risk = (
            "risk_per_trade_pct", "max_positions", "daily_loss_cap_pct", "daily_loss_cap_abs",
            "max_notional_per_trade", "portfolio_var_limit_pct", "cooldown_after_losses",
        )
        missing = [key for key in required_risk if key not in risk]
        if missing:
            raise ConfigError(f"risk is missing: {', '.join(missing)}")
        for key in ("risk_per_trade_pct", "daily_loss_cap_pct", "daily_loss_cap_abs", "max_notional_per_trade", "portfolio_var_limit_pct"):
            risk[key] = _finite_number(risk[key], f"risk.{key}", positive=True)
        for key in ("max_positions", "cooldown_after_losses"):
            risk[key] = int(_finite_number(risk[key], f"risk.{key}", positive=True))
        risk.setdefault("max_gross_exposure", self.d["capital"])
        risk["max_gross_exposure"] = _finite_number(risk["max_gross_exposure"], "risk.max_gross_exposure", positive=True)
        self.d["risk"] = risk

        costs = dict(self.d.get("costs") or {})
        cost_defaults = {
            "slippage_pct": 0.03,
            "brokerage_per_order": 20.0,
            "stt_pct": 0.025,
            "txn_charge_pct": 0.00297,
            "gst_pct": 18.0,
            "sebi_fee_pct": 0.0001,
            "stamp_duty_pct": 0.003,
        }
        for key, default in cost_defaults.items():
            value = _finite_number(costs.get(key, default), f"costs.{key}")
            if value < 0:
                raise ConfigError(f"costs.{key} cannot be negative")
            costs[key] = value
        self.d["costs"] = costs

        order_flow = dict(self.d.get("order_flow") or {})
        flow_defaults = {
            "enabled": True,
            "primary": True,
            "depth_levels": 20,
            "max_staleness_seconds": 2.0,
            "min_observations": 300,
            "min_side_notional": 50_000.0,
            "max_spread_bps": 12.0,
            "min_book_imbalance": 0.12,
            "min_flow_imbalance": 0.04,
            "min_microprice_edge_bps": 0.5,
            "pressure_ema_alpha": 0.20,
            "min_pressure_ema": 0.08,
            "min_positive_streak": 3,
            "min_liquidity_score": 0.60,
            "require_replay_validation": True,
            "replay_min_signals": 30,
            "replay_horizon_candles": 5,
            "replay_min_hit_rate": 0.50,
            "replay_min_mean_return_bps": 0.0,
            "replay_max_records": 10_000,
        }
        for key, default in flow_defaults.items():
            order_flow.setdefault(key, default)
        if not all(isinstance(order_flow[key], bool) for key in ("enabled", "primary", "require_replay_validation")):
            raise ConfigError("order_flow.enabled, order_flow.primary, and order_flow.require_replay_validation must be booleans")
        if order_flow["primary"] and not order_flow["enabled"]:
            raise ConfigError("order_flow.primary requires order_flow.enabled")
        order_flow["depth_levels"] = int(_finite_number(order_flow["depth_levels"], "order_flow.depth_levels", positive=True))
        order_flow["min_observations"] = int(_finite_number(order_flow["min_observations"], "order_flow.min_observations", positive=True))
        order_flow["min_positive_streak"] = int(_finite_number(order_flow["min_positive_streak"], "order_flow.min_positive_streak", positive=True))
        for key in ("replay_min_signals", "replay_horizon_candles", "replay_max_records"):
            order_flow[key] = int(_finite_number(order_flow[key], f"order_flow.{key}", positive=True))
        if not 1 <= order_flow["depth_levels"] <= 20:
            raise ConfigError("order_flow.depth_levels must be between 1 and 20 for this Dhan adapter")
        for key in ("max_staleness_seconds", "min_side_notional", "max_spread_bps", "min_book_imbalance", "min_flow_imbalance", "min_microprice_edge_bps", "pressure_ema_alpha", "min_pressure_ema", "min_liquidity_score", "replay_min_hit_rate", "replay_min_mean_return_bps"):
            order_flow[key] = _finite_number(order_flow[key], f"order_flow.{key}")
            if order_flow[key] < 0:
                raise ConfigError(f"order_flow.{key} cannot be negative")
        for key in ("max_staleness_seconds", "min_side_notional", "max_spread_bps"):
            if order_flow[key] <= 0:
                raise ConfigError(f"order_flow.{key} must be positive")
        if not 0 < order_flow["pressure_ema_alpha"] <= 1:
            raise ConfigError("order_flow.pressure_ema_alpha must be greater than 0 and at most 1")
        if order_flow["min_book_imbalance"] > 1 or order_flow["min_flow_imbalance"] > 1 or order_flow["min_pressure_ema"] > 1:
            raise ConfigError("order-flow imbalance thresholds must be between 0 and 1")
        if not 0 <= order_flow["min_liquidity_score"] <= 1:
            raise ConfigError("order_flow.min_liquidity_score must be between 0 and 1")
        if not 0 <= order_flow["replay_min_hit_rate"] <= 1:
            raise ConfigError("order_flow.replay_min_hit_rate must be between 0 and 1")
        if order_flow["min_positive_streak"] > 50:
            raise ConfigError("order_flow.min_positive_streak cannot exceed 50 snapshots")
        self.d["order_flow"] = order_flow

        training = dict(self.d.get("training") or {})
        training_defaults = {
            "iterations": 5,
            "population": 12,
            "elite_k": 4,
            "min_trades": 25,
            "promote_score": 0.80,
            "min_symbols": 3,
            "training_history_candles": 10_000,
            "min_oos_candles": 60,
            "walk_forward_folds": 3,
            "embargo_candles": 5,
            "random_seed": None,
        }
        for key, default in training_defaults.items():
            training.setdefault(key, default)
        for key in ("iterations", "population", "elite_k", "min_trades", "min_symbols", "training_history_candles", "min_oos_candles", "walk_forward_folds"):
            training[key] = int(_finite_number(training[key], f"training.{key}", positive=True))
        training["embargo_candles"] = int(_finite_number(training["embargo_candles"], "training.embargo_candles"))
        if training["embargo_candles"] < 0:
            raise ConfigError("training.embargo_candles cannot be negative")
        if training["random_seed"] is not None:
            training["random_seed"] = int(_finite_number(training["random_seed"], "training.random_seed"))
        for key in ("promote_score",):
            training[key] = _finite_number(training[key], f"training.{key}")
        if training["population"] < 2 or not 1 <= training["elite_k"] < training["population"]:
            raise ConfigError("training.elite_k must be at least 1 and below training.population")
        if training["min_symbols"] > len(self.d["symbols"]):
            raise ConfigError("training.min_symbols cannot exceed the configured symbol count")
        if training["training_history_candles"] < execution["signal_history_candles"]:
            raise ConfigError("training.training_history_candles must cover the live signal history window")
        minimum_walk_forward = (training["min_oos_candles"] + training["embargo_candles"]) * training["walk_forward_folds"] + execution["signal_history_candles"]
        if training["training_history_candles"] < minimum_walk_forward:
            raise ConfigError("training.training_history_candles is too short for its walk-forward folds and embargo")
        training.setdefault("min_signal_stability", 0.95)
        training["min_signal_stability"] = _finite_number(training["min_signal_stability"], "training.min_signal_stability")
        if not 0.0 < training["min_signal_stability"] <= 1.0:
            raise ConfigError("training.min_signal_stability must be greater than 0 and at most 1")
        training.setdefault("require_human_approval", True)
        if not isinstance(training["require_human_approval"], bool):
            raise ConfigError("training.require_human_approval must be a boolean")
        self.d["training"] = training

        holidays = self.d.get("market_holidays", [])
        if not isinstance(holidays, list):
            raise ConfigError("market_holidays must be a list of YYYY-MM-DD dates")
        validated_holidays: list[str] = []
        for value in holidays:
            try:
                validated_holidays.append(datetime.strptime(str(value), "%Y-%m-%d").date().isoformat())
            except ValueError as exc:
                raise ConfigError("market_holidays must use YYYY-MM-DD dates") from exc
        self.d["market_holidays"] = sorted(set(validated_holidays))
        if mode == "live" and not any(datetime.strptime(day, "%Y-%m-%d").year == now().year for day in self.d["market_holidays"]):
            raise ConfigError("live mode requires the official NSE holiday calendar for the current year")

        self.d.setdefault("market_close", "15:30")
        for time_key in ("market_open", "entry_cutoff", "squareoff", "market_close"):
            value = self.d.get(time_key)
            try:
                datetime.strptime(value, "%H:%M")
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{time_key} must use HH:MM (24-hour IST) format") from exc
        minutes = {key: int(self.d[key][:2]) * 60 + int(self.d[key][3:]) for key in ("market_open", "entry_cutoff", "squareoff", "market_close")}
        if not minutes["market_open"] < minutes["entry_cutoff"] <= minutes["squareoff"] < minutes["market_close"]:
            raise ConfigError("market_open must be before entry_cutoff, squareoff, and market_close")

        whitelist = list(self.d.get("ip_whitelist", []))
        ip_env_name = self.d.get("ip_whitelist_env", "")
        if ip_env_name:
            if not isinstance(ip_env_name, str) or not ip_env_name.replace("_", "").isalnum():
                raise ConfigError("ip_whitelist_env must be an environment-variable name")
            whitelist.extend(value.strip() for value in os.getenv(ip_env_name, "").split(",") if value.strip())
        if mode == "live" and not whitelist:
            raise ConfigError("live mode requires at least one whitelisted static IP")
        for address in whitelist:
            try:
                ipaddress.ip_address(address)
            except ValueError as exc:
                raise ConfigError(f"Invalid IP address in ip_whitelist: {address}") from exc
        self.d["ip_whitelist"] = sorted(set(whitelist))

        db_name = Path(str(self.d.get("db_path", "oxalpha.db")))
        if db_name.is_absolute() or ".." in db_name.parts:
            raise ConfigError("db_path must be a file inside the project directory")
        self.d["db_path"] = str((self.root / db_name).resolve())

        if mode == "live":
            mapping = self.d.get("security_map")
            if not isinstance(mapping, Mapping) or any(symbol not in mapping for symbol in self.d["symbols"]):
                raise ConfigError("live mode requires a security_map entry for every configured symbol")

    def __getitem__(self, key: str):
        return self.d[key]

    def get(self, key: str, default=None):
        return self.d.get(key, default)


class DB:
    """A small serialized SQLite store with durable audit records."""

    SCHEMA = {
        "kv": "k TEXT PRIMARY KEY, v TEXT NOT NULL",
        "candles": "sym TEXT, ts INT, o REAL, h REAL, l REAL, c REAL, v INT, PRIMARY KEY(sym,ts)",
        "orders": "oid TEXT PRIMARY KEY, sym TEXT, side TEXT, qty INT, px REAL, type TEXT, status TEXT, tag TEXT, ts TEXT, broker TEXT",
        "positions": "sym TEXT PRIMARY KEY, qty INT, avg REAL, sl REAL, tp REAL, opened TEXT, strat TEXT",
        "trades": "tid INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT, side TEXT, qty INT, inpx REAL, outpx REAL, pnl REAL, charges REAL, strat TEXT, intime TEXT, outtime TEXT, exit_reason TEXT",
        "equity": "ts TEXT PRIMARY KEY, equity REAL",
        "strategies": "sid TEXT PRIMARY KEY, json TEXT, score REAL, status TEXT, gen INT, parent TEXT, hash TEXT, created TEXT, approved_by TEXT, approved_at TEXT",
        "backtests": "bid INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, is_oos TEXT, score REAL, stats TEXT, ts TEXT",
        "failures": "fid INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, report TEXT, ts TEXT",
        "news": "nid INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT, headline TEXT, source TEXT, sentiment TEXT, score REAL, ts TEXT",
        "events": "eid INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, msg TEXT, ts TEXT",
        "decisions": "did INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT, action TEXT, reason TEXT, detail TEXT, ts TEXT",
        "audit": "aid INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, digest TEXT NOT NULL, ts TEXT NOT NULL",
        "orderflow": "ofid INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT NOT NULL, source TEXT NOT NULL, bid REAL NOT NULL, ask REAL NOT NULL, mid REAL NOT NULL, microprice REAL NOT NULL, spread_bps REAL NOT NULL, book_imbalance REAL NOT NULL, flow_imbalance REAL NOT NULL, pressure_ema REAL NOT NULL, positive_streak INTEGER NOT NULL, liquidity_score REAL NOT NULL, book_state TEXT NOT NULL, microprice_edge_bps REAL NOT NULL, bid_notional REAL NOT NULL, ask_notional REAL NOT NULL, observations INTEGER NOT NULL, ready INTEGER NOT NULL, entry_signal INTEGER NOT NULL, exit_signal INTEGER NOT NULL, reason TEXT NOT NULL, ts TEXT NOT NULL",
        "calendar_events": "eid INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, symbol TEXT, event_type TEXT, impact TEXT, event_date TEXT, event_time TEXT, details TEXT, source TEXT, created_at TEXT",
        "order_intents": "iid TEXT PRIMARY KEY, agent TEXT, symbol TEXT, action TEXT, qty REAL, price REAL, leverage REAL, stop_loss REAL, take_profit REAL, reason TEXT, status TEXT, created TEXT, decided_at TEXT, decided_by TEXT, expires_at TEXT, signal_id TEXT",
        "promax_trades": "ptid INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, symbol TEXT, side TEXT, qty REAL, entry_price REAL, exit_price REAL, pnl REAL, leverage REAL, reason TEXT, opened TEXT, closed TEXT",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self.l = threading.RLock()
        self.audit_key = os.getenv("OX_AUDIT_KEY", "").encode("utf-8")
        with self.l:
            self.c.execute("PRAGMA journal_mode=WAL")
            self.c.execute("PRAGMA foreign_keys=ON")
            self.c.execute("PRAGMA busy_timeout=10000")
            for table, definition in self.SCHEMA.items():
                self.c.execute(f"CREATE TABLE IF NOT EXISTS {table}({definition})")
            self._ensure_columns(
                "orderflow",
                {
                    "pressure_ema": "REAL NOT NULL DEFAULT 0",
                    "positive_streak": "INTEGER NOT NULL DEFAULT 0",
                    "liquidity_score": "REAL NOT NULL DEFAULT 0",
                    "book_state": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
                    "reason": "TEXT NOT NULL DEFAULT 'LEGACY_SNAPSHOT'",
                },
            )
            self.c.execute("CREATE INDEX IF NOT EXISTS idx_orderflow_symbol_time ON orderflow(sym, ofid DESC)")
            self.c.commit()

    def _ensure_columns(self, table: str, additions: Mapping[str, str]) -> None:
        """Apply narrowly-scoped, idempotent local schema additions."""
        existing = {str(row[1]) for row in self.c.execute(f"PRAGMA table_info({table})")}
        for column, definition in additions.items():
            if column not in existing:
                self.c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def ex(self, query: str, args: tuple = ()) -> None:
        with self.l:
            self.c.execute(query, args)
            self.c.commit()

    def many(self, query: str, rows) -> None:
        """Commit a bounded batch atomically; used for historical candle ingestion."""
        with self.l:
            self.c.executemany(query, rows)
            self.c.commit()

    def q(self, query: str, args: tuple = ()) -> list[tuple]:
        with self.l:
            return self.c.execute(query, args).fetchall()

    def kv_set(self, key: str, value) -> None:
        self.ex("INSERT OR REPLACE INTO kv VALUES(?,?)", (key, json.dumps(value)))

    def kv_get(self, key: str, default=None):
        rows = self.q("SELECT v FROM kv WHERE k=?", (key,))
        return json.loads(rows[0][0]) if rows else default

    @property
    def audit_enabled(self) -> bool:
        return len(self.audit_key) >= 32

    @staticmethod
    def _audit_payload(payload: Mapping | None) -> str:
        if payload is None:
            return "{}"
        if not isinstance(payload, Mapping):
            raise ValueError("audit payload must be a mapping")
        return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)

    def audit(self, action: str, payload: Mapping | None = None) -> None:
        """Append an HMAC-linked critical-action record when an audit key is present."""
        if not self.audit_enabled:
            return
        if not isinstance(action, str) or not action or len(action) > 80:
            raise ValueError("audit action must be a short non-empty string")
        encoded_payload = self._audit_payload(payload)
        timestamp = iso()
        with self.l:
            row = self.c.execute("SELECT digest FROM audit ORDER BY aid DESC LIMIT 1").fetchone()
            previous = str(row[0]) if row else ""
            body = "|".join((previous, action, encoded_payload, timestamp)).encode("utf-8")
            digest = hmac.new(self.audit_key, body, hashlib.sha256).hexdigest()
            self.c.execute(
                "INSERT INTO audit(action,payload,previous_hash,digest,ts)VALUES(?,?,?,?,?)",
                (action, encoded_payload, previous, digest, timestamp),
            )
            self.c.commit()

    def verify_audit(self) -> bool:
        """Verify the critical-action chain. Live mode refuses to start without it."""
        if not self.audit_enabled:
            return False
        previous = ""
        with self.l:
            rows = self.c.execute("SELECT action,payload,previous_hash,digest,ts FROM audit ORDER BY aid").fetchall()
        for action, payload, stored_previous, digest, timestamp in rows:
            if not hmac.compare_digest(str(stored_previous), previous):
                return False
            body = "|".join((previous, str(action), str(payload), str(timestamp))).encode("utf-8")
            expected = hmac.new(self.audit_key, body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(str(digest), expected):
                return False
            previous = str(digest)
        return True

    def record_decision(self, symbol: str, action: str, reason: str, detail: Mapping | None = None) -> None:
        """Persist bounded, structured decision evidence without credentials or raw prompts."""
        if not all(isinstance(value, str) and value for value in (symbol, action, reason)):
            raise ValueError("decision symbol, action, and reason are required")
        if len(symbol) > 40 or len(action) > 80 or len(reason) > 160:
            raise ValueError("decision fields exceed their safe size limits")
        detail_json = self._audit_payload(detail)
        self.ex(
            "INSERT INTO decisions(sym,action,reason,detail,ts)VALUES(?,?,?,?,?)",
            (symbol.upper(), action, reason, detail_json[:2_000], iso()),
        )

    def close(self) -> None:
        with self.l:
            self.c.close()
