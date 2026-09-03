"""TradingView-style chart detail for recorded agent candles."""

from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard_data import CHART_WINDOWS, ROOT, chart_window, configured_symbols, load_chart_frame, load_snapshot, money, percent


st.title("Price chart")
st.caption("Recorded candles, volume, moving averages, and session VWAP. Ranges use only locally stored history; no prices are invented when data is missing.")

config, _, database_available = load_snapshot(str(ROOT))
symbols = configured_symbols(config)
if not database_available or not symbols:
    st.info("No market history is available yet. Run the paper agent once, then come back to this chart.", icon=":material/database:")
    st.stop()

symbol = st.session_state.get("chart_symbol", symbols[0])
if symbol not in symbols:
    symbol = symbols[0]

controls, indicator_control = st.columns((2, 3), vertical_alignment="bottom")
with controls:
    window = st.segmented_control(
        "Chart window",
        options=CHART_WINDOWS,
        default="5D",
        required=True,
        key="chart_window",
        width="stretch",
    )
with indicator_control:
    indicators = st.pills(
        "Indicators",
        options=["SMA 20", "SMA 50", "VWAP"],
        default=["SMA 20", "SMA 50", "VWAP"],
        selection_mode="multi",
        key="chart_indicators",
    )


def chart_for(frame, selected_indicators, chart_range: str):
    axis_format = "%d %b\n%H:%M" if chart_range in {"1D", "5D"} else "%d %b"
    base = alt.Chart(frame).encode(
        x=alt.X("Timestamp:T", title=None, axis=alt.Axis(format=axis_format, labelAngle=0)),
        tooltip=[
            alt.Tooltip("Timestamp:T", title="Time", format="DD MMM YYYY, HH:mm"),
            alt.Tooltip("Open:Q", format=".2f"),
            alt.Tooltip("High:Q", format=".2f"),
            alt.Tooltip("Low:Q", format=".2f"),
            alt.Tooltip("Close:Q", format=".2f"),
            alt.Tooltip("Volume:Q", title="Recorded volume", format=",.0f"),
        ],
    )
    colour = alt.condition(alt.datum.Close >= alt.datum.Open, alt.value("#34D399"), alt.value("#F87171"))
    wicks = base.mark_rule().encode(y=alt.Y("Low:Q", title="Price (INR)", scale=alt.Scale(zero=False)), y2="High:Q", color=colour)
    bodies = base.mark_bar().encode(y=alt.Y("Open:Q", scale=alt.Scale(zero=False)), y2="Close:Q", color=colour)
    price_layers = [wicks, bodies]
    indicator_colours = {"SMA 20": "#60A5FA", "SMA 50": "#A78BFA", "VWAP": "#FBBF24"}
    for indicator in selected_indicators:
        price_layers.append(base.mark_line(color=indicator_colours[indicator], strokeWidth=1.6).encode(y=alt.Y(f"{indicator}:Q", scale=alt.Scale(zero=False))))
    price = alt.layer(*price_layers).properties(height=390, title=f"{symbol} — price")
    volume = base.mark_bar(opacity=0.72).encode(y=alt.Y("Volume:Q", title="Recorded volume"), color=colour).properties(height=145, title="Volume")
    return alt.vconcat(price, volume).resolve_scale(x="shared")


@st.fragment(run_every=5)
def live_chart() -> None:
    frame, available = load_chart_frame(str(ROOT), symbol)
    if not available:
        st.warning("The local database is temporarily unavailable.", icon=":material/database:")
        return
    if frame.empty:
        st.caption(f"No candles are recorded for {symbol} yet.")
        return
    display, availability_note = chart_window(frame, window)
    if display.empty:
        st.caption(f"No candles are recorded for {symbol} in the {window} range.")
        return
    last = display.iloc[-1]
    previous = display.iloc[-2] if len(display) > 1 else last
    move = (float(last["Close"]) / float(previous["Close"]) - 1.0) * 100 if float(previous["Close"]) else 0.0
    with st.container(horizontal=True):
        st.metric("Last recorded price", money(last["Close"]), delta=percent(move), border=True)
        st.metric("Candle high", money(last["High"]), border=True)
        st.metric("Candle low", money(last["Low"]), border=True)
        st.metric("Volume ratio", f"{float(last['Volume ratio']):.2f}x", border=True)
    with st.container(border=True):
        shown_indicators = [item for item in indicators if not (item == "VWAP" and window == "All available")]
        st.altair_chart(chart_for(display, shown_indicators, window), width="stretch", key="detail_candle_chart")
    st.caption(f"{availability_note} Drag to pan and use the mouse wheel to zoom. Indicator values are visual context only; the agent’s risk gates remain authoritative.")
    if "VWAP" in indicators and window == "All available":
        st.caption("VWAP is hidden for the daily ‘All available’ view because it has no meaningful intraday session context there.")


live_chart()
