"""
Streamlit App Template - Main Entry Point
==========================================
This is the main entry point for your Streamlit app. It handles:
- Page configuration (wide layout, title, icon)
- Global state initialization
- Navigation setup
- Shared UI elements (logo, title)

The app uses multi-page architecture for better organization.
Individual pages are in the app_pages/ directory.
"""

import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Must be the first Streamlit command - sets browser tab and layout
st.set_page_config(
    page_title="My App",
    page_icon=":material/analytics:",
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL STATE INITIALIZATION
# ============================================================================
# Initialize state that's shared across all pages
# Only run once per session using setdefault
st.session_state.setdefault("initialized", True)
st.session_state.setdefault("user_preferences", {})

# Example: Initialize API clients, database connections, or other resources
# that should persist across page navigation
# @st.cache_resource
# def init_connection():
#     return create_connection()
# st.session_state.setdefault("db_conn", init_connection())

# ============================================================================
# NAVIGATION SETUP
# ============================================================================
# Define all pages and their organization
# Using position="sidebar" for multi-page apps
page = st.navigation(
    {
        "": [  # Ungrouped pages (appear first)
            st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
        ],
        "Analytics": [
            st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
            # Add more analytics pages here
        ],
        "Settings": [
            st.Page("app_pages/settings.py", title="Configuration", icon=":material/settings:"),
            # Add more settings pages here
        ],
    },
    position="sidebar",
)

# ============================================================================
# SHARED UI ELEMENTS
# ============================================================================
# Elements here appear on all pages

# Optional: Add logo to sidebar
# st.logo("assets/logo.png")

# Page title (uses current page's icon and title)
st.title(f"{page.icon} {page.title}")

# Optional: Shared sidebar widgets
with st.sidebar:
    st.caption("v1.0.0")
    # Add global filters, settings, or info here

# ============================================================================
# RUN CURRENT PAGE
# ============================================================================
# This renders the selected page
page.run()
