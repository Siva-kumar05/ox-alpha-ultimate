"""Local-only ox-alpha market dashboard entry point."""

from __future__ import annotations

import streamlit as st

from dashboard_data import render_sidebar


st.set_page_config(page_title="ox-alpha dashboard", page_icon=":material/query_stats:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/chart.py", title="Chart", icon=":material/candlestick_chart:"),
        st.Page("app_pages/scanner.py", title="Scanner", icon=":material/manage_search:"),
        st.Page("app_pages/orderflow.py", title="Flow intelligence", icon=":material/waterfall_chart:"),
        st.Page("app_pages/audit.py", title="Agent audit", icon=":material/fact_check:"),
    ],
    position="top",
)
render_sidebar()
page.run()
