"""Overview page for the local ox-alpha dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard_data import CHART_WINDOWS, ROOT, load_relative_prices, load_snapshot, market_overview, money, orderflow_overview, runtime_badge


st.title("Market overview")
st.caption("Recorded OHLCV data from the agent database. This page is a monitor, not a trading recommendation.")

relative_window = st.segmented_control(
    "Relative performance window",
    options=CHART_WINDOWS,
    default="5D",
    required=True,
    key="overview_relative_window",
    width="stretch",
)


@st.fragment(run_every=5)
def live_overview() -> None:
    config, data, available = load_snapshot(str(ROOT))
    runtime_badge(config)
    if not available:
        st.info("The agent database has not been created yet. Run the paper agent once, then return here.", icon=":material/database:")
        return
    market = market_overview(config, data)
    trades = data["trades"]
    positions = data["positions"]
    realised = float(pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    last_equity = data["equity"]
    equity_value = float(pd.to_numeric(last_equity["equity"], errors="coerce").dropna().iloc[-1]) if not last_equity.empty and pd.to_numeric(last_equity["equity"], errors="coerce").notna().any() else float(config.get("capital", 0))
    momentum = int((market["State"] == "Momentum").sum()) if not market.empty else 0
    flow = orderflow_overview(data, config)
    ready_books = int(flow["Admission ready"].sum()) if not flow.empty else 0

    with st.container(horizontal=True):
        st.metric("Configured capital", money(config.get("capital", 0)), border=True)
        st.metric("Last recorded equity", money(equity_value), delta=money(realised), border=True)
        st.metric("Open agent positions", len(positions), border=True)
        st.metric("Momentum states", momentum, border=True)
        st.metric("Fresh order-flow books", f"{ready_books}/{len(config.get('symbols', []))}", border=True)
    st.caption("Refreshes every 5 seconds while this page is open.")

    left, right = st.columns((3, 2), vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Relative price performance")
            chart_data = load_relative_prices(str(ROOT), tuple(config.get("symbols", [])), relative_window)
            if chart_data.empty:
                st.caption("Price history will appear after the agent has recorded candles.")
            else:
                chart = (
                    alt.Chart(chart_data)
                    .mark_line()
                    .encode(
                        x=alt.X("Timestamp:T", title=None),
                        y=alt.Y("Normalised price:Q", title="Start = 1.00", scale=alt.Scale(zero=False)),
                        color=alt.Color("Symbol:N", title="Symbol"),
                        tooltip=[alt.Tooltip("Timestamp:T", title="Time"), "Symbol:N", alt.Tooltip("Normalised price:Q", format=".3f")],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(chart, width="stretch", key="overview_relative_chart")
    with right:
        with st.container(border=True):
            st.subheader("Portfolio equity")
            equity = data["equity"].copy()
            if equity.empty:
                st.caption("Equity snapshots are written at the end of a completed session.")
            else:
                equity["Timestamp"] = pd.to_datetime(equity["ts"], errors="coerce")
                equity["Equity (INR)"] = pd.to_numeric(equity["equity"], errors="coerce")
                st.line_chart(equity.dropna(subset=["Timestamp", "Equity (INR)"]), x="Timestamp", y="Equity (INR)")

    with st.container(border=True):
        st.subheader("Watchlist")
        if market.empty:
            st.caption("No recorded candles are available yet.")
        else:
            st.dataframe(
                market,
                hide_index=True,
                column_config={
                    "Last price": st.column_config.NumberColumn("Last price (INR)", format="%.2f"),
                    "Move": st.column_config.NumberColumn("Latest move", format="%.2f%%"),
                    "20-candle move": st.column_config.NumberColumn("20-candle move", format="%.2f%%"),
                    "Recorded volume": st.column_config.NumberColumn("Recorded volume", format="%,d"),
                    "Volume ratio": st.column_config.NumberColumn("Volume ratio", format="%.2fx"),
                    "Range": st.column_config.NumberColumn("Candle range", format="%.2f%%"),
                    "Updated": st.column_config.DatetimeColumn("Last candle", format="DD MMM, HH:mm"),
                },
            )
            st.caption("‘Recorded volume’ is the candle volume supplied by historical data. The live, still-forming bar counts received price ticks until a full-volume feed is connected.")


live_overview()
