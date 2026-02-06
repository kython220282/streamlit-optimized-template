"""
Dashboard Page
==============
Example dashboard with metrics, charts, and data display.
Demonstrates performance patterns: caching, fragments, and efficient layouts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.cache_utils import load_sample_data, compute_metrics

# ============================================================================
# FILTERS
# ============================================================================
# Put filters at the top for easy access
col1, col2, col3 = st.columns(3)

with col1:
    date_range = st.date_input(
        "Date range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        help="Filter data by date range"
    )

with col2:
    category = st.selectbox(
        "Category",
        ["All", "Sales", "Marketing", "Support"],
        help="Filter by category"
    )

with col3:
    refresh = st.button("Refresh data", icon=":material/refresh:")
    if refresh:
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# LOAD DATA
# ============================================================================
# Using cached function from utils - data persists across reruns
# TTL ensures fresh data every 5 minutes
df = load_sample_data()

# Apply filters (don't cache this - filter values change frequently)
if category != "All":
    df = df[df["category"] == category]

# ============================================================================
# KPI METRICS
# ============================================================================
st.subheader("Key metrics")

# Compute metrics using cached function
metrics = compute_metrics(df)

# Horizontal container for responsive metric row
with st.container(horizontal=True):
    st.metric(
        "Total revenue",
        f"${metrics['revenue']:,.0f}",
        delta=f"{metrics['revenue_change']:+.1f}%",
        border=True
    )
    st.metric(
        "Orders",
        f"{metrics['orders']:,}",
        delta=f"{metrics['orders_change']:+.1f}%",
        border=True
    )
    st.metric(
        "Avg order value",
        f"${metrics['avg_order']:,.2f}",
        delta=f"{metrics['avg_order_change']:+.1f}%",
        border=True
    )
    st.metric(
        "Conversion rate",
        f"{metrics['conversion']:.1f}%",
        delta=f"{metrics['conversion_change']:+.1f}%",
        border=True
    )

# ============================================================================
# CHARTS
# ============================================================================
st.subheader("Trends")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    with st.container(border=True):
        st.markdown("**Revenue over time**")
        # Native Streamlit chart - fast and simple
        st.line_chart(
            df.set_index("date")["revenue"],
            height=250
        )

with chart_col2:
    with st.container(border=True):
        st.markdown("**Orders by category**")
        category_data = df.groupby("category")["orders"].sum()
        st.bar_chart(category_data, height=250)

# ============================================================================
# AUTO-REFRESHING FRAGMENT
# ============================================================================
# This fragment auto-refreshes every 30 seconds without rerunning the whole page
st.subheader("Live updates")

@st.fragment(run_every="30s")
def live_status():
    """Auto-refreshing status indicator - only this fragment reruns"""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", "Active", help="System status")
            st.badge("Online", icon=":material/check_circle:", color="green")
        
        with col2:
            st.metric("Last update", current_time)
            
        with col3:
            # Simulate real-time metric
            active_users = np.random.randint(150, 200)
            st.metric("Active users", active_users)

live_status()

# ============================================================================
# DATA TABLE
# ============================================================================
st.subheader("Recent transactions")

with st.container(border=True):
    # Configured dataframe with proper column types
    st.dataframe(
        df.head(10),
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "revenue": st.column_config.NumberColumn(
                "Revenue",
                format="$%.2f",
            ),
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "category": st.column_config.TextColumn("Category"),
            "status": st.column_config.TextColumn("Status"),
        },
        hide_index=True,
        use_container_width=True,
    )

# ============================================================================
# CONDITIONAL RENDERING EXAMPLE
# ============================================================================
# Only render advanced analytics when toggled on
# This prevents expensive computations when not needed
if st.toggle("Show advanced analytics"):
    st.subheader("Advanced analytics")
    
    # This expensive computation only runs when toggled on
    with st.container(border=True):
        st.caption("This section demonstrates conditional rendering - it only loads when visible")
        
        # Simulate expensive computation
        advanced_data = df.copy()
        advanced_data["rolling_avg"] = advanced_data["revenue"].rolling(window=7).mean()
        
        st.line_chart(
            advanced_data.set_index("date")[["revenue", "rolling_avg"]],
            height=300
        )
