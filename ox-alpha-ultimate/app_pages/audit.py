"""Audit and safety view for the ox-alpha dashboard."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard_data import KILL_PATH, ROOT, load_snapshot, marked_positions, orderflow_overview


st.title("Agent audit")
st.caption("Read-only execution history and safety state. No API tokens, account details, or fund-transfer controls are displayed here.")

with st.expander("Quant-research basis and limits"):
    st.markdown(
        "The order-flow gate combines observed book imbalance with available depth, "
        "spread, microprice and persistence. Research motivates testing those "
        "relationships; it does not establish a universal or profitable threshold."
    )
    st.markdown(
        "Strategy promotion uses causally separated, embargoed walk-forward results "
        "because trying many variants can overfit historical data. The depth replay "
        "below is an admission-gate study, not a fill, fee, or profitability backtest."
    )
    st.markdown(
        "Primary papers: [order-book events](https://arxiv.org/abs/1011.6402) and "
        "[backtest overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)."
    )
    st.caption("The detailed implementation boundary and experiment checklist are in docs/QUANT_RESEARCH_NOTES.md.")


@st.fragment(run_every=5)
def live_audit() -> None:
    config, data, available = load_snapshot(str(ROOT))
    if not available:
        st.info("The agent database has not been created yet.", icon=":material/database:")
        return
    risk = config.get("risk", {})
    execution = config.get("execution", {})
    health = data.get("health", {})
    replay = data.get("orderflow_replay", {})
    agent_state = str(health.get("state", "No heartbeat"))
    flow = orderflow_overview(data, config)
    with st.container(horizontal=True):
        st.metric("Agent state", agent_state, border=True)
        st.metric("Autonomous execution", "Enabled" if execution.get("autonomous") else "Disabled", border=True)
        st.metric("Naked shorting", "Blocked" if not execution.get("allow_short") else "Enabled", border=True)
        st.metric("Risk per trade", f"{float(risk.get('risk_per_trade_pct', 0)):.2f}%", border=True)
        st.metric("Emergency stop", "Active" if KILL_PATH.exists() else "Ready", border=True)
        st.metric("Order-flow books", f"{int(flow['Admission ready'].sum())}/{len(config.get('symbols', []))}" if not flow.empty else "Waiting", border=True)

    positions = marked_positions(config, data)
    with st.container(border=True):
        st.subheader("Open agent positions")
        if positions.empty:
            st.caption("No agent-owned positions are open.")
        else:
            st.dataframe(
                positions,
                hide_index=True,
                column_config={
                    "avg": st.column_config.NumberColumn("Entry (INR)", format="%.2f"),
                    "sl": st.column_config.NumberColumn("Stop (INR)", format="%.2f"),
                    "tp": st.column_config.NumberColumn("Target (INR)", format="%.2f"),
                    "Latest mark": st.column_config.NumberColumn("Latest recorded mark (INR)", format="%.2f"),
                    "Marked value": st.column_config.NumberColumn("Marked value (INR)", format="%.2f"),
                    "Unrealised P&L": st.column_config.NumberColumn("Unrealised P&L (INR)", format="%.2f"),
                },
            )
            st.caption("Marks use the latest recorded candle close and are not a broker-confirmed account valuation.")

    left, right = st.columns(2, vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Recent closed trades")
            trades = data["trades"].copy()
            if trades.empty:
                st.caption("No completed trades yet.")
            else:
                st.dataframe(
                    trades,
                    hide_index=True,
                    column_config={
                        "inpx": st.column_config.NumberColumn("Entry (INR)", format="%.2f"),
                        "outpx": st.column_config.NumberColumn("Exit (INR)", format="%.2f"),
                        "pnl": st.column_config.NumberColumn("Net P&L (INR)", format="%.2f"),
                        "charges": st.column_config.NumberColumn("Charges (INR)", format="%.2f"),
                    },
                )
    with right:
        with st.container(border=True):
            st.subheader("Strategy validation")
            strategies = data["strategies"].copy()
            if strategies.empty:
                st.caption("No strategies have been validated yet.")
            else:
                st.dataframe(strategies, hide_index=True, column_config={"score": st.column_config.NumberColumn("Score", format="%.3f")})

    with st.container(border=True):
        st.subheader("Latest out-of-sample validation")
        validations = data["backtests"].copy()
        validations = validations.loc[validations["is_oos"] == "OOS"].copy() if not validations.empty else validations
        if validations.empty:
            st.caption("No out-of-sample validation has been recorded yet.")
        else:
            def unpack_stats(raw: object) -> dict:
                try:
                    value = json.loads(str(raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
                return value if isinstance(value, dict) else {}

            expanded = validations["stats"].map(unpack_stats)
            def percentage(item: dict, key: str) -> float | None:
                value = item.get(key)
                return float(value) * 100.0 if isinstance(value, (int, float)) else None

            validations["Trades"] = expanded.map(lambda item: item.get("trades"))
            validations["Return"] = expanded.map(lambda item: percentage(item, "ret"))
            validations["Max drawdown"] = expanded.map(lambda item: percentage(item, "maxdd"))
            validations["Profit factor"] = expanded.map(lambda item: item.get("pf"))
            validations["Win rate"] = expanded.map(lambda item: percentage(item, "win_rate"))
            validations["Signal stability"] = expanded.map(lambda item: percentage(item, "signal_stability"))
            validations["Fold consistency"] = expanded.map(lambda item: percentage(item, "oos_frame_consistency"))
            validations["OOS frames"] = expanded.map(
                lambda item: f"{item.get('oos_frames_traded')}/{item.get('oos_frames_total')}"
                if item.get("oos_frames_traded") is not None and item.get("oos_frames_total") is not None
                else None
            )
            validations["Eligible"] = expanded.map(lambda item: bool(item.get("promotion_eligible", False)))
            validations["Method"] = expanded.map(lambda item: item.get("execution", "Recorded"))
            st.dataframe(
                validations[["sid", "score", "Trades", "Return", "Max drawdown", "Profit factor", "Win rate", "Signal stability", "Fold consistency", "OOS frames", "Eligible", "Method", "ts"]],
                hide_index=True,
                column_config={
                    "score": st.column_config.NumberColumn("Validation score", format="%.3f"),
                    "Trades": st.column_config.NumberColumn("Trades", format="%d"),
                    "Return": st.column_config.NumberColumn("Net return", format="%.2f%%"),
                    "Max drawdown": st.column_config.NumberColumn("Max drawdown", format="%.2f%%"),
                    "Profit factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                    "Win rate": st.column_config.NumberColumn("Win rate", format="%.2f%%"),
                    "Signal stability": st.column_config.NumberColumn("Signal stability", format="%.1f%%"),
                    "Fold consistency": st.column_config.NumberColumn("Fold consistency", format="%.1f%%"),
                    "OOS frames": st.column_config.TextColumn("Frames traded/total"),
                    "Eligible": st.column_config.CheckboxColumn("Promotion eligible"),
                    "Method": st.column_config.TextColumn("Validation method"),
                    "ts": st.column_config.TextColumn("Recorded"),
                },
            )
            st.caption("A strategy can be promoted only from embargoed expanding walk-forward results. Validation executes at the next candle open, uses causally confirmed swings and long-only exits, and includes the configured taxes, charges, and slippage model.")
            st.caption("Fold consistency is the share of individual walk-forward fold/symbol OOS frames that were themselves net profitable, out of the frames that traded at all (shown in 'Frames traded/total'). It exists so a pooled score cannot quietly hide one strong fold carrying several losing ones.")

    with st.container(border=True):
        st.subheader("Order-flow safety evidence")
        if replay:
            evidence = pd.DataFrame([replay]).rename(columns={
                "samples": "Real-depth samples", "hit_rate": "Positive move rate",
                "mean_return_bps": "Mean forward move (bps)", "horizon_candles": "Horizon candles",
                "passed": "Replay gate passed", "kind": "Method", "source": "Source",
            })
            if "Positive move rate" in evidence:
                evidence["Positive move rate"] = pd.to_numeric(evidence["Positive move rate"], errors="coerce") * 100.0
            shown = [column for column in ("Method", "Source", "Real-depth samples", "Positive move rate", "Mean forward move (bps)", "Horizon candles", "Replay gate passed") if column in evidence]
            st.dataframe(
                evidence[shown],
                hide_index=True,
                column_config={
                    "Positive move rate": st.column_config.NumberColumn("Positive move rate", format="%.1f%%"),
                    "Mean forward move (bps)": st.column_config.NumberColumn("Mean forward move", format="%.2f bps"),
                    "Replay gate passed": st.column_config.CheckboxColumn("Replay gate passed"),
                },
            )
            st.caption("This checks retained real depth-entry snapshots against later recorded candles. It is not an execution backtest and does not imply profit.")
        if flow.empty:
            st.caption("No depth snapshots have been recorded yet. In primary order-flow mode, new live entries remain blocked.")
        else:
            st.dataframe(
                flow[["Symbol", "Source", "Book state", "Persistent pressure", "Support snapshots", "Liquidity quality", "Admission ready", "Decision", "Updated"]],
                hide_index=True,
                column_config={
                    "Persistent pressure": st.column_config.NumberColumn("Persistent pressure", format="%.3f"),
                    "Support snapshots": st.column_config.NumberColumn("Support snapshots", format="%d"),
                    "Liquidity quality": st.column_config.ProgressColumn("Liquidity quality", min_value=0.0, max_value=1.0, format="percent"),
                    "Admission ready": st.column_config.CheckboxColumn("Admission ready"),
                    "Updated": st.column_config.DatetimeColumn("Latest snapshot", format="HH:mm:ss"),
                },
            )
            st.caption("This evidence shows the live admission checks. It is not a prediction, performance claim, or a record of trade-aggressor delta.")

    with st.container(border=True):
        st.subheader("Recent decision journal")
        decisions = data["decisions"].copy()
        if decisions.empty:
            st.caption("Decisions will appear once the agent evaluates live or paper market data.")
        else:
            def concise_detail(raw: object) -> str:
                try:
                    parsed = json.loads(str(raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    return ""
                return json.dumps(parsed, sort_keys=True)[:300] if isinstance(parsed, dict) else ""

            decisions["Evidence"] = decisions["detail"].map(concise_detail)
            st.dataframe(
                decisions[["ts", "sym", "action", "reason", "Evidence"]],
                hide_index=True,
                column_config={
                    "ts": st.column_config.TextColumn("Time"),
                    "sym": st.column_config.TextColumn("Symbol"),
                    "action": st.column_config.TextColumn("Action"),
                    "reason": st.column_config.TextColumn("Reason"),
                    "Evidence": st.column_config.TextColumn("Bounded evidence", width="large"),
                },
            )
            st.caption("The journal stores only bounded numeric and operational evidence; it never includes broker credentials, account numbers, or raw external prompts.")

    with st.container(border=True):
        st.subheader("Recent agent events")
        events = data["events"].copy()
        if events.empty:
            st.caption("No events yet.")
        else:
            events["msg"] = events["msg"].astype(str).str.slice(0, 300)
            st.dataframe(events, hide_index=True, column_config={"ts": st.column_config.TextColumn("Time"), "msg": st.column_config.TextColumn("Message", width="large")})


live_audit()
