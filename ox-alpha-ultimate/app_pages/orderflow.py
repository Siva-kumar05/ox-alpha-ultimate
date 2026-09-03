"""Transparent market-by-price depth and admission evidence."""

from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard_data import ROOT, configured_symbols, load_snapshot, orderflow_book, orderflow_history, orderflow_overview


st.title("Order-flow intelligence")
st.caption("Real Dhan L2 book analysis when connected. This page never presents displayed liquidity as executed trade delta, queue position, or a cross-market arbitrage signal.")

config, data, available = load_snapshot(str(ROOT))
symbols = configured_symbols(config)
if not available:
    st.info("The agent database has not been created yet.", icon=":material/database:")
    st.stop()
if not symbols:
    st.info("No configured symbols are available.", icon=":material/manage_search:")
    st.stop()

selected_symbol = st.selectbox("Symbol", symbols, key="orderflow_symbol")


@st.fragment(run_every=5)
def live_orderflow() -> None:
    config, data, _ = load_snapshot(str(ROOT))
    overview = orderflow_overview(data, config)
    latest = overview.loc[overview["Symbol"] == selected_symbol]
    history = orderflow_history(data, selected_symbol)
    book, source = orderflow_book(data, selected_symbol)

    if latest.empty:
        st.info("Waiting for a depth snapshot for this symbol. Primary live entries stay blocked until a fresh Dhan depth book has warmed up.", icon=":material/database:")
        return

    current = latest.iloc[0]
    with st.container(horizontal=True):
        st.metric("Admission decision", "Ready" if bool(current["Admission ready"]) else "Blocked", delta=str(current["Decision"]), delta_color="off", border=True)
        st.metric("Book state", str(current["Book state"]), border=True)
        st.metric("Persistent pressure", f"{float(current['Persistent pressure']):+.3f}", border=True)
        st.metric("Liquidity quality", f"{float(current['Liquidity quality']):.0%}", border=True)
        st.metric("Support snapshots", int(current["Support snapshots"]), border=True)
        st.metric("Feed source", str(current["Source"]), border=True)

    left, right = st.columns(2, vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Recorded L2 pressure")
            if history.empty:
                st.caption("No stored order-flow history is available yet.")
            else:
                pressure = history.melt(
                    id_vars=["Timestamp"],
                    value_vars=["Book imbalance", "Persistent pressure", "Displayed book change"],
                    var_name="Measure",
                    value_name="Value",
                )
                chart = (
                    alt.Chart(pressure)
                    .mark_line()
                    .encode(
                        x=alt.X("Timestamp:T", title=None),
                        y=alt.Y("Value:Q", title="Displayed-book pressure", scale=alt.Scale(domain=(-1, 1))),
                        color=alt.Color("Measure:N", title=None),
                        tooltip=[alt.Tooltip("Timestamp:T", format="HH:mm:ss"), "Measure:N", alt.Tooltip("Value:Q", format="+.3f")],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(chart, width="stretch", key="orderflow_pressure_history")
    with right:
        with st.container(border=True):
            st.subheader("Latest market-by-price ladder")
            if book.empty:
                st.caption("The latest L2 ladder is not recorded yet.")
            else:
                book_chart = (
                    alt.Chart(book)
                    .mark_bar()
                    .encode(
                        x=alt.X("Quantity:Q", title="Displayed quantity"),
                        y=alt.Y("Price:Q", title="Price (INR)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Side:N", scale=alt.Scale(domain=["Bid", "Ask"], range=["#34D399", "#F87171"]), title=None),
                        tooltip=["Side:N", alt.Tooltip("Price:Q", format=".2f"), alt.Tooltip("Quantity:Q", format=",.0f"), alt.Tooltip("Orders:Q", format=",d")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(book_chart, width="stretch", key="orderflow_book_ladder")
                st.caption(f"Last recorded source: {source}. The ladder is public market data, not account data.")

    with st.container(border=True):
        st.subheader("What this decision means")
        st.dataframe(
            latest[["Symbol", "Bid", "Ask", "Spread (bps)", "Book imbalance", "Displayed book change", "Persistent pressure", "Microprice edge (bps)", "Bid notional", "Ask notional", "Entry signal", "Exit signal", "Decision", "Updated"]],
            hide_index=True,
            column_config={
                "Bid": st.column_config.NumberColumn("Best bid (INR)", format="%.2f"),
                "Ask": st.column_config.NumberColumn("Best ask (INR)", format="%.2f"),
                "Spread (bps)": st.column_config.NumberColumn("Spread", format="%.2f bps"),
                "Book imbalance": st.column_config.NumberColumn("Book imbalance", format="%.3f"),
                "Displayed book change": st.column_config.NumberColumn("Displayed book change", format="%.3f"),
                "Persistent pressure": st.column_config.NumberColumn("Persistent pressure", format="%.3f"),
                "Microprice edge (bps)": st.column_config.NumberColumn("Microprice edge", format="%.2f bps"),
                "Bid notional": st.column_config.NumberColumn("Bid depth (INR)", format="%.0f"),
                "Ask notional": st.column_config.NumberColumn("Ask depth (INR)", format="%.0f"),
                "Entry signal": st.column_config.CheckboxColumn("Long gate"),
                "Exit signal": st.column_config.CheckboxColumn("Exit gate"),
                "Updated": st.column_config.DatetimeColumn("Snapshot", format="HH:mm:ss"),
            },
        )
        st.caption("The agent requires a fresh, liquid, persistent buy-side book, then a separate candle-regime confirmation, then its risk gate. A negative book state can close an agent-owned long but never opens a naked short.")

    with st.expander("Scope and limits", icon=":material/info:"):
        st.write("The social-media examples describe cross-venue crypto or prediction-market latency trades. Those require simultaneous executable prices, atomic or hedged execution, fee and latency measurement, and venue-specific compliance. This Dhan/NSE agent does not claim that capability. It adopts the transferable part: react to fresh current market state, prove liquidity first, and log why an action was blocked or allowed.")


live_orderflow()
