"""Recorded-data scanner for the ox-alpha dashboard."""

from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard_data import ROOT, configured_symbols, load_snapshot, market_overview, orderflow_overview


st.title("Market scanner")
st.caption("A local order-flow and market-state scan. The order-flow section uses recorded L2 depth snapshots; it never invents trade delta or queue position.")


@st.fragment(run_every=5)
def live_scanner() -> None:
    config, data, available = load_snapshot(str(ROOT))
    if not available:
        st.info("The agent database has not been created yet.", icon=":material/database:")
        return
    market = market_overview(config, data)
    flow = orderflow_overview(data, config)
    if market.empty:
        st.caption("No recorded candles are available yet.")
        return

    positive = int((market["20-candle move"] > 0).sum())
    active_volume = int((market["Volume ratio"] >= 1.0).sum())
    range_average = float(market["Range"].mean())
    flow_ready = int(flow["Admission ready"].sum()) if not flow.empty else 0
    with st.container(horizontal=True):
        st.metric("Fresh order-flow books", f"{flow_ready}/{len(configured_symbols(config))}", border=True)
        st.metric("Positive over 20 candles", f"{positive}/{len(market)}", border=True)
        st.metric("Above average volume", f"{active_volume}/{len(market)}", border=True)
        st.metric("Average candle range", f"{range_average:.2f}%", border=True)
        st.metric("Configured symbols", len(configured_symbols(config)), border=True)

    left, right = st.columns(2, vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Relative strength")
            strength = market.sort_values("20-candle move", ascending=False)
            strength_chart = (
                alt.Chart(strength)
                .mark_bar()
                .encode(
                    x=alt.X("20-candle move:Q", title="Return (%)"),
                    y=alt.Y("Symbol:N", sort="-x", title=None),
                    color=alt.condition(alt.datum["20-candle move"] >= 0, alt.value("#34D399"), alt.value("#F87171")),
                    tooltip=["Symbol:N", alt.Tooltip("20-candle move:Q", format=".2f"), alt.Tooltip("Volume ratio:Q", format=".2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(strength_chart, width="stretch", key="scanner_strength")
    with right:
        with st.container(border=True):
            st.subheader("Volume activity")
            volume_chart = (
                alt.Chart(market.sort_values("Volume ratio", ascending=False))
                .mark_bar(color="#60A5FA")
                .encode(
                    x=alt.X("Volume ratio:Q", title="Current / 20-candle average"),
                    y=alt.Y("Symbol:N", sort="-x", title=None),
                    tooltip=["Symbol:N", alt.Tooltip("Recorded volume:Q", format=",.0f"), alt.Tooltip("Volume ratio:Q", format=".2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(volume_chart, width="stretch", key="scanner_volume")

    with st.container(border=True):
        st.subheader("Order-flow admission monitor")
        if flow.empty:
            st.info("Waiting for order-book snapshots. Live entries remain blocked until Dhan 20-level depth is connected, fresh, and warmed up.", icon=":material/database:")
        else:
            pressure = flow.melt(
                id_vars=["Symbol"],
                value_vars=["Book imbalance", "Persistent pressure", "Displayed book change"],
                var_name="Metric",
                value_name="Imbalance",
            )
            pressure_chart = (
                alt.Chart(pressure)
                .mark_bar()
                .encode(
                    x=alt.X("Imbalance:Q", title="Buy pressure (+) / sell pressure (-)", scale=alt.Scale(domain=(-1, 1))),
                    y=alt.Y("Symbol:N", title=None),
                    color=alt.Color("Metric:N", title=None),
                    tooltip=["Symbol:N", "Metric:N", alt.Tooltip("Imbalance:Q", format=".3f")],
                )
                .properties(height=240)
            )
            st.altair_chart(pressure_chart, width="stretch", key="scanner_orderflow_pressure")
            st.dataframe(
                flow,
                hide_index=True,
                column_config={
                    "Bid": st.column_config.NumberColumn("Best bid (INR)", format="%.2f"),
                    "Ask": st.column_config.NumberColumn("Best ask (INR)", format="%.2f"),
                    "Spread (bps)": st.column_config.NumberColumn("Spread", format="%.2f bps"),
                    "Book imbalance": st.column_config.NumberColumn("Book imbalance", format="%.3f"),
                    "Displayed book change": st.column_config.NumberColumn("Displayed book change", format="%.3f"),
                    "Persistent pressure": st.column_config.NumberColumn("Persistent pressure", format="%.3f"),
                    "Support snapshots": st.column_config.NumberColumn("Support snapshots", format="%d"),
                    "Liquidity quality": st.column_config.ProgressColumn("Liquidity quality", min_value=0.0, max_value=1.0, format="percent"),
                    "Microprice edge (bps)": st.column_config.NumberColumn("Microprice edge", format="%.2f bps"),
                    "Bid notional": st.column_config.NumberColumn("Bid depth (INR)", format="%.0f"),
                    "Ask notional": st.column_config.NumberColumn("Ask depth (INR)", format="%.0f"),
                    "Admission ready": st.column_config.CheckboxColumn("Fresh / ready"),
                    "Entry signal": st.column_config.CheckboxColumn("Long gate"),
                    "Exit signal": st.column_config.CheckboxColumn("Exit gate"),
                    "Decision": st.column_config.TextColumn("Decision", width="large"),
                    "Updated": st.column_config.DatetimeColumn("Snapshot", format="HH:mm:ss"),
                },
            )
            st.caption("Persistent pressure is a smoothed displayed-book signal. Displayed-book change measures updates in resting size; neither is labelled as executed buyer/seller volume, queue position, or arbitrage proof.")

    with st.container(border=True):
        st.subheader("Scanner table")
        st.dataframe(
            market.sort_values(["State", "20-candle move"], ascending=[True, False]),
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

    with st.expander("How the scanner labels a stock", icon=":material/info:"):
        st.write("Momentum means price is above its 20- and 50-candle moving averages and recorded volume is at least its 20-candle average. Weak means price is below both moving averages. All other states are neutral.")


live_scanner()
