"""Read-only data helpers for the ox-alpha local dashboard.

The dashboard only opens the agent database in SQLite read-only mode.  Broker
credentials, order endpoints, and trading controls are intentionally absent
from this module.
"""

from __future__ import annotations

import sqlite3
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parent
KILL_PATH = ROOT / "KILL.flag"
MAX_CANDLES_PER_SYMBOL = 1_500
MAX_CHART_CANDLES = 40_000
CHART_WINDOWS = ("1D", "5D", "1M", "3M", "All available")
ORDERFLOW_COLUMNS = [
    "ofid", "sym", "source", "bid", "ask", "mid", "microprice", "spread_bps",
    "book_imbalance", "flow_imbalance", "pressure_ema", "positive_streak",
    "liquidity_score", "book_state", "microprice_edge_bps", "bid_notional",
    "ask_notional", "observations", "ready", "entry_signal", "exit_signal", "reason", "ts",
]

ORDERFLOW_DEFAULTS: dict[str, Any] = {
    "ofid": 0,
    "sym": "",
    "source": "UNKNOWN",
    "bid": float("nan"),
    "ask": float("nan"),
    "mid": float("nan"),
    "microprice": float("nan"),
    "spread_bps": float("nan"),
    "book_imbalance": 0.0,
    "flow_imbalance": 0.0,
    "pressure_ema": 0.0,
    "positive_streak": 0,
    "liquidity_score": 0.0,
    "book_state": "UNKNOWN",
    "microprice_edge_bps": 0.0,
    "bid_notional": 0.0,
    "ask_notional": 0.0,
    "observations": 0,
    "ready": 0,
    "entry_signal": 0,
    "exit_signal": 0,
    "reason": "NO_SNAPSHOT",
    "ts": pd.NaT,
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _read_config(root: Path) -> dict[str, Any]:
    try:
        with (root / "config.yaml").open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_table(connection: sqlite3.Connection, statement: str, columns: list[str], params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(statement, connection, params=params)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return _empty(columns)


def _load_orderflow(connection: sqlite3.Connection) -> pd.DataFrame:
    """Read order-flow records compatibly while a local DB is being migrated."""
    existing = {str(row[1]) for row in connection.execute("PRAGMA table_info(orderflow)")}
    if not existing:
        return _empty(ORDERFLOW_COLUMNS)
    defaults = {
        "pressure_ema": "0", "positive_streak": "0", "liquidity_score": "0",
        "book_state": "'UNKNOWN'", "reason": "'LEGACY_SNAPSHOT'",
    }
    selected = [column if column in existing else f"{defaults[column]} AS {column}" for column in ORDERFLOW_COLUMNS]
    return _read_table(
        connection,
        f"SELECT {','.join(selected)} FROM orderflow ORDER BY ofid DESC LIMIT 1_000",
        ORDERFLOW_COLUMNS,
    )


def _load_orderflow_books(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = _read_table(connection, "SELECT k,v FROM kv WHERE k LIKE 'orderflow_book:%'", ["k", "v"])
    books: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        try:
            value = json.loads(str(row["v"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        symbol = str(row["k"]).partition(":")[2].upper()
        if symbol and isinstance(value, dict):
            books[symbol] = value
    return books


def _load_json_kv(connection: sqlite3.Connection, key: str) -> dict[str, Any]:
    rows = _read_table(connection, "SELECT v FROM kv WHERE k=?", ["v"], (key,))
    if rows.empty:
        return {}
    try:
        value = json.loads(str(rows.iloc[0]["v"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@st.cache_data(ttl=3, max_entries=4, show_spinner=False)
def load_snapshot(project: str) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Load bounded dashboard data without exposing a writable DB connection."""
    root = Path(project)
    config = _read_config(root)
    raw_db_path = Path(str(config.get("db_path", "oxalpha.db")))
    db_path = raw_db_path if raw_db_path.is_absolute() else root / raw_db_path
    tables: dict[str, Any] = {
        "positions": _empty(["sym", "qty", "avg", "sl", "tp", "opened", "strat"]),
        "trades": _empty(["tid", "sym", "side", "qty", "inpx", "outpx", "pnl", "charges", "strat", "outtime", "exit_reason"]),
        "strategies": _empty(["sid", "score", "status", "gen", "created", "approved_at", "validation"]),
        "backtests": _empty(["bid", "sid", "is_oos", "score", "stats", "ts"]),
        "events": _empty(["eid", "kind", "msg", "ts"]),
        "decisions": _empty(["did", "sym", "action", "reason", "detail", "ts"]),
        "orderflow": _empty(ORDERFLOW_COLUMNS),
        "orderflow_books": {},
        "orderflow_replay": {},
        "equity": _empty(["ts", "equity"]),
        "health": {},
        "candles": {},
    }
    if not db_path.exists():
        return config, tables, False

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
        tables["positions"] = _read_table(connection, "SELECT sym,qty,avg,sl,tp,opened,strat FROM positions ORDER BY sym", list(tables["positions"].columns))
        tables["trades"] = _read_table(connection, "SELECT tid,sym,side,qty,inpx,outpx,pnl,charges,strat,outtime,exit_reason FROM trades ORDER BY tid DESC LIMIT 100", list(tables["trades"].columns))
        strategies = _read_table(connection, "SELECT sid,json,score,status,gen,created,approved_at FROM strategies ORDER BY score DESC LIMIT 50", ["sid", "json", "score", "status", "gen", "created", "approved_at"])
        if not strategies.empty:
            def validation_state(raw: object) -> str:
                try:
                    value = json.loads(str(raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    return "Revalidation required"
                return "Current validation" if isinstance(value, dict) and value.get("schema_version") == 5 else "Revalidation required"

            strategies["validation"] = strategies["json"].map(validation_state)
            strategies.loc[strategies["validation"] != "Current validation", "status"] = "REVALIDATION_REQUIRED"
            tables["strategies"] = strategies.drop(columns=["json"])
        tables["backtests"] = _read_table(connection, "SELECT bid,sid,is_oos,score,stats,ts FROM backtests ORDER BY bid DESC LIMIT 100", list(tables["backtests"].columns))
        tables["events"] = _read_table(connection, "SELECT eid,kind,msg,ts FROM events ORDER BY eid DESC LIMIT 100", list(tables["events"].columns))
        tables["decisions"] = _read_table(connection, "SELECT did,sym,action,reason,detail,ts FROM decisions ORDER BY did DESC LIMIT 100", list(tables["decisions"].columns))
        tables["orderflow"] = _load_orderflow(connection)
        tables["orderflow_books"] = _load_orderflow_books(connection)
        tables["orderflow_replay"] = _load_json_kv(connection, "orderflow_replay_validation")
        tables["equity"] = _read_table(connection, "SELECT ts,equity FROM equity ORDER BY ts", list(tables["equity"].columns))
        tables["health"] = _load_json_kv(connection, "agent_health")
        for symbol in configured_symbols(config):
            rows = _read_table(
                connection,
                "SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts DESC LIMIT ?",
                ["ts", "o", "h", "l", "c", "v"],
                (symbol, MAX_CANDLES_PER_SYMBOL),
            )
            tables["candles"][symbol] = rows.iloc[::-1].reset_index(drop=True)
        return config, tables, True
    except (OSError, sqlite3.DatabaseError):
        return config, tables, False
    finally:
        if connection is not None:
            connection.close()


def configured_symbols(config: dict[str, Any]) -> list[str]:
    return [str(symbol).upper() for symbol in config.get("symbols", []) if str(symbol).isalnum()]


def money(value: float | int | None) -> str:
    return f"INR {float(value or 0):,.2f}"


def percent(value: float | int | None) -> str:
    return f"{float(value or 0):+.2f}%"


def candle_frame(data: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Return a display-ready OHLCV frame with transparent local indicators."""
    return candle_frame_from_raw(data.get("candles", {}).get(symbol, pd.DataFrame()))


def candle_frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw stored OHLCV rows without inventing a value when data is absent."""
    raw = raw.copy()
    if raw.empty:
        return pd.DataFrame()
    result = raw.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    for column in ("Open", "High", "Low", "Close", "Volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["Timestamp"] = pd.to_datetime(result["ts"], unit="s", utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    result = result.dropna(subset=["Timestamp", "Open", "High", "Low", "Close", "Volume"]).drop_duplicates(subset=["Timestamp"], keep="last")
    if result.empty:
        return result
    return add_chart_indicators(result.reset_index(drop=True))


def add_chart_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Recalculate indicators after every chart aggregation level."""
    result = frame.copy()
    if result.empty:
        return result
    result["SMA 20"] = result["Close"].rolling(20, min_periods=1).mean()
    result["SMA 50"] = result["Close"].rolling(50, min_periods=1).mean()
    session = result["Timestamp"].dt.normalize()
    cumulative_volume = result["Volume"].groupby(session, sort=False).cumsum().replace(0, pd.NA)
    cumulative_value = (result["Close"] * result["Volume"]).groupby(session, sort=False).cumsum()
    result["VWAP"] = cumulative_value / cumulative_volume
    result["Volume average"] = result["Volume"].rolling(20, min_periods=2).mean()
    result["Volume ratio"] = (result["Volume"] / result["Volume average"].replace(0, pd.NA)).fillna(1.0)
    return result.reset_index(drop=True)


@st.cache_data(ttl=3, max_entries=8, show_spinner=False)
def load_chart_frame(project: str, symbol: str) -> tuple[pd.DataFrame, bool]:
    """Load one bounded symbol history for the chart, read-only and cached."""
    root = Path(project)
    config = _read_config(root)
    if symbol not in configured_symbols(config):
        return pd.DataFrame(), False
    raw_db_path = Path(str(config.get("db_path", "oxalpha.db")))
    db_path = raw_db_path if raw_db_path.is_absolute() else root / raw_db_path
    if not db_path.exists():
        return pd.DataFrame(), False
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
        raw = _read_table(
            connection,
            "SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts DESC LIMIT ?",
            ["ts", "o", "h", "l", "c", "v"],
            (symbol, MAX_CHART_CANDLES),
        )
        return candle_frame_from_raw(raw.iloc[::-1].reset_index(drop=True)), True
    except (OSError, sqlite3.DatabaseError):
        return pd.DataFrame(), False
    finally:
        if connection is not None:
            connection.close()


def chart_window(frame: pd.DataFrame, window: str) -> tuple[pd.DataFrame, str]:
    """Select a truthful local-history range and aggregate it for readability."""
    if frame.empty or window not in CHART_WINDOWS:
        return pd.DataFrame(), "No recorded history is available for this range."
    anchor = frame["Timestamp"].max()
    if window == "1D":
        selected = frame.loc[frame["Timestamp"].dt.normalize() == anchor.normalize()].copy()
        frequency = None
    elif window == "5D":
        days = frame["Timestamp"].dt.normalize().drop_duplicates().tail(5)
        selected = frame.loc[frame["Timestamp"].dt.normalize().isin(days)].copy()
        frequency = "5min"
    elif window == "1M":
        selected = frame.loc[frame["Timestamp"] >= anchor - pd.Timedelta(days=31)].copy()
        frequency = "15min"
    elif window == "3M":
        selected = frame.loc[frame["Timestamp"] >= anchor - pd.Timedelta(days=93)].copy()
        frequency = "60min"
    else:
        selected = frame.copy()
        frequency = "1D"

    if frequency and not selected.empty:
        selected = (
            selected.set_index("Timestamp")
            .resample(frequency, origin="start_day")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "High", "Low", "Close"])
            .reset_index()
        )
    selected = add_chart_indicators(selected)
    available_days = int(frame["Timestamp"].dt.normalize().nunique())
    requested_days = {"1D": 1, "5D": 5, "1M": 20, "3M": 60}.get(window)
    if requested_days and available_days < requested_days:
        note = f"Only {available_days} recorded trading day(s) are available; showing all available data in this range."
    else:
        note = f"{available_days} recorded trading day(s) are retained locally."
    return selected, note


def market_overview(config: dict[str, Any], data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in configured_symbols(config):
        frame = candle_frame(data, symbol)
        if frame.empty:
            continue
        last = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) > 1 else last
        reference = frame.iloc[-21] if len(frame) > 20 else frame.iloc[0]
        change = (float(last["Close"]) / float(previous["Close"]) - 1.0) * 100 if float(previous["Close"]) else 0.0
        change_20 = (float(last["Close"]) / float(reference["Close"]) - 1.0) * 100 if float(reference["Close"]) else 0.0
        if last["Close"] > last["SMA 20"] > last["SMA 50"] and last["Volume ratio"] >= 1:
            state = "Momentum"
        elif last["Close"] < last["SMA 20"] < last["SMA 50"]:
            state = "Weak"
        else:
            state = "Neutral"
        rows.append(
            {
                "Symbol": symbol,
                "Last price": float(last["Close"]),
                "Move": change,
                "20-candle move": change_20,
                "Recorded volume": int(last["Volume"]),
                "Volume ratio": float(last["Volume ratio"]),
                "Range": (float(last["High"]) / float(last["Low"]) - 1.0) * 100 if float(last["Low"]) else 0.0,
                "State": state,
                "Updated": last["Timestamp"],
            }
        )
    return pd.DataFrame(rows)


def orderflow_overview(data: dict[str, Any], config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Latest real or explicitly simulated L2 state per configured symbol."""
    raw_flow = data.get("orderflow", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    if not isinstance(raw_flow, pd.DataFrame):
        return _empty(["Symbol", "Source", "Book state", "Bid", "Ask", "Spread (bps)", "Book imbalance", "Displayed book change", "Persistent pressure", "Support snapshots", "Liquidity quality", "Microprice edge (bps)", "Bid notional", "Ask notional", "Admission ready", "Entry signal", "Exit signal", "Decision", "Updated"])
    flow = raw_flow.copy()
    if flow.empty:
        return _empty(["Symbol", "Source", "Book state", "Bid", "Ask", "Spread (bps)", "Book imbalance", "Displayed book change", "Persistent pressure", "Support snapshots", "Liquidity quality", "Microprice edge (bps)", "Bid notional", "Ask notional", "Admission ready", "Entry signal", "Exit signal", "Decision", "Updated"])
    for column, default in ORDERFLOW_DEFAULTS.items():
        if column not in flow:
            flow[column] = default
    flow = flow.sort_values("ofid", ascending=False).drop_duplicates(subset=["sym"], keep="first")
    numeric = ("bid", "ask", "spread_bps", "book_imbalance", "flow_imbalance", "pressure_ema", "positive_streak", "liquidity_score", "microprice_edge_bps", "bid_notional", "ask_notional")
    for column in numeric:
        flow[column] = pd.to_numeric(flow[column], errors="coerce")
    timestamps = pd.to_datetime(flow["ts"], errors="coerce", utc=True)
    flow["Updated"] = timestamps.dt.tz_convert("Asia/Kolkata")
    order_flow = config.get("order_flow", {}) if isinstance(config, dict) else {}
    try:
        max_staleness = float(order_flow.get("max_staleness_seconds", 2.0)) if isinstance(order_flow, dict) else 2.0
    except (TypeError, ValueError):
        max_staleness = 2.0
    if not math.isfinite(max_staleness) or max_staleness <= 0:
        max_staleness = 2.0
    age_seconds = (pd.Timestamp.now(tz="UTC") - timestamps).dt.total_seconds()
    raw_ready = pd.to_numeric(flow["ready"], errors="coerce").fillna(0).astype(bool)
    admission_ready = raw_ready & age_seconds.between(0, max_staleness, inclusive="both")
    def bool_column(name: str) -> pd.Series:
        return pd.to_numeric(flow[name], errors="coerce").fillna(0).astype(bool)
    result = pd.DataFrame(
        {
            "Symbol": flow["sym"].astype(str),
            "Source": flow["source"].astype(str),
            "Book state": flow["book_state"].astype(str),
            "Bid": flow["bid"],
            "Ask": flow["ask"],
            "Spread (bps)": flow["spread_bps"],
            "Book imbalance": flow["book_imbalance"],
            "Displayed book change": flow["flow_imbalance"],
            "Persistent pressure": flow["pressure_ema"],
            "Support snapshots": flow["positive_streak"],
            "Liquidity quality": flow["liquidity_score"],
            "Microprice edge (bps)": flow["microprice_edge_bps"],
            "Bid notional": flow["bid_notional"],
            "Ask notional": flow["ask_notional"],
            "Admission ready": admission_ready,
            "Entry signal": bool_column("entry_signal") & admission_ready,
            "Exit signal": bool_column("exit_signal") & admission_ready,
            "Decision": flow["reason"].astype(str),
            "Updated": flow["Updated"],
        }
    )
    return result.sort_values("Symbol").reset_index(drop=True)


def orderflow_history(data: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Return bounded recorded L2 metrics for one symbol without fabricating gaps."""
    flow = data.get("orderflow", pd.DataFrame()).copy()
    if flow.empty:
        return _empty(["Timestamp", "Book imbalance", "Displayed book change", "Persistent pressure", "Liquidity quality", "Entry signal", "Decision"])
    flow = flow.loc[flow["sym"].astype(str).str.upper() == symbol.upper()].copy()
    if flow.empty:
        return _empty(["Timestamp", "Book imbalance", "Displayed book change", "Persistent pressure", "Liquidity quality", "Entry signal", "Decision"])
    for column in ("book_imbalance", "flow_imbalance", "pressure_ema", "liquidity_score"):
        flow[column] = pd.to_numeric(flow[column], errors="coerce")
    flow["Timestamp"] = pd.to_datetime(flow["ts"], errors="coerce")
    flow["Entry signal"] = pd.to_numeric(flow["entry_signal"], errors="coerce").fillna(0).astype(bool)
    return flow.rename(columns={
        "book_imbalance": "Book imbalance", "flow_imbalance": "Displayed book change",
        "pressure_ema": "Persistent pressure", "liquidity_score": "Liquidity quality", "reason": "Decision",
    })[["Timestamp", "Book imbalance", "Displayed book change", "Persistent pressure", "Liquidity quality", "Entry signal", "Decision"]].dropna(subset=["Timestamp"]).sort_values("Timestamp")


def orderflow_book(data: dict[str, Any], symbol: str) -> tuple[pd.DataFrame, str]:
    """Expose the last recorded market-by-price ladder, never account data."""
    raw = data.get("orderflow_books", {}).get(symbol.upper(), {})
    if not isinstance(raw, dict):
        return _empty(["Price", "Quantity", "Orders", "Side"]), ""
    rows: list[dict[str, Any]] = []
    for side, label in (("bids", "Bid"), ("asks", "Ask")):
        for item in raw.get(side, []):
            if not isinstance(item, list) or len(item) != 3:
                continue
            try:
                price, quantity, orders = float(item[0]), int(item[1]), int(item[2])
            except (TypeError, ValueError):
                continue
            if price > 0 and quantity > 0 and orders >= 0:
                rows.append({"Price": price, "Quantity": quantity, "Orders": orders, "Side": label})
    return pd.DataFrame(rows), str(raw.get("source", ""))


def marked_positions(config: dict[str, Any], data: dict[str, Any]) -> pd.DataFrame:
    positions = data["positions"].copy()
    if positions.empty:
        return positions
    prices = market_overview(config, data).set_index("Symbol")["Last price"].to_dict()
    positions["Latest mark"] = positions["sym"].map(prices)
    positions["Marked value"] = positions["qty"].astype(float) * positions["Latest mark"].astype(float)
    positions["Unrealised P&L"] = (positions["Latest mark"].astype(float) - positions["avg"].astype(float)) * positions["qty"].astype(float)
    return positions


def normalised_prices(config: dict[str, Any], data: dict[str, Any], limit: int = 240) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in configured_symbols(config):
        frame = candle_frame(data, symbol).tail(limit).copy()
        if frame.empty or not float(frame["Close"].iloc[0]):
            continue
        frame["Symbol"] = symbol
        frame["Normalised price"] = frame["Close"] / float(frame["Close"].iloc[0])
        rows.append(frame[["Timestamp", "Symbol", "Normalised price"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Timestamp", "Symbol", "Normalised price"])


@st.cache_data(ttl=30, max_entries=5, show_spinner=False)
def load_relative_prices(project: str, symbols: tuple[str, ...], window: str) -> pd.DataFrame:
    """Build a bounded, comparable chart from each symbol's local history."""
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        frame, available = load_chart_frame(project, symbol)
        if not available or frame.empty:
            continue
        display, _ = chart_window(frame, window)
        if display.empty or not float(display["Close"].iloc[0]):
            continue
        display = display.copy()
        display["Symbol"] = symbol
        display["Normalised price"] = display["Close"] / float(display["Close"].iloc[0])
        rows.append(display[["Timestamp", "Symbol", "Normalised price"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Timestamp", "Symbol", "Normalised price"])


def runtime_badge(config: dict[str, Any]) -> None:
    if KILL_PATH.exists():
        st.badge("Emergency stop active", icon=":material/emergency:", color="red")
    elif config.get("mode") == "live":
        st.badge("Live mode configured", icon=":material/power:", color="orange")
    else:
        st.badge("Paper mode", icon=":material/science:", color="blue")


def render_sidebar() -> None:
    config, _, _ = load_snapshot(str(ROOT))
    symbols = configured_symbols(config)
    with st.sidebar:
        st.title("ox-alpha")
        runtime_badge(config)
        st.caption("Local market and agent monitor")
        if symbols:
            if st.session_state.get("chart_symbol") not in symbols:
                st.session_state.chart_symbol = symbols[0]
            st.selectbox("Chart symbol", symbols, key="chart_symbol")
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            load_snapshot.clear()
            st.rerun()

        st.subheader("Safety controls")
        if KILL_PATH.exists():
            st.warning("Emergency stop is active. The agent will not restart while this flag exists.", icon=":material/warning:")
            acknowledge = st.checkbox("I reviewed the stop reason and want to allow a future restart.", key="clear_stop_ack")
            if st.button("Clear emergency stop", disabled=not acknowledge, icon=":material/restart_alt:", width="stretch"):
                KILL_PATH.unlink(missing_ok=True)
                st.toast("Emergency stop cleared. Starting the agent remains a separate action.")
                st.rerun()
        elif st.button("Activate emergency stop", type="primary", icon=":material/emergency:", width="stretch"):
            KILL_PATH.write_text("HALTED BY LOCAL DASHBOARD\n", encoding="utf-8")
            st.toast("Stop request saved. The running agent will respond on its next check.")
            st.rerun()

        st.caption("This dashboard is local-only. It cannot show or save Dhan credentials, move funds, or place manual orders.")
