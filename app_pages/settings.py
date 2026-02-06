"""
Settings Page
=============
Configuration and preferences management.
Demonstrates forms, state management, and user preferences.
"""

import streamlit as st
from utils.state_utils import get_user_pref, set_user_pref, clear_all_prefs

# ============================================================================
# USER PREFERENCES
# ============================================================================
st.subheader("User preferences")

# Using a form to batch inputs - only reruns on submit
with st.form("preferences", border=False):
    
    display_name = st.text_input(
        "Display name",
        value=get_user_pref("display_name", ""),
        help="Your name as it appears in the app"
    )
    
    theme_preference = st.segmented_control(
        "Theme preference",
        ["Auto", "Light", "Dark"],
        default=get_user_pref("theme", "Auto"),
        help="Choose your preferred theme"
    )
    
    notifications = st.toggle(
        "Enable notifications",
        value=get_user_pref("notifications", True),
        help="Show status updates and alerts"
    )
    
    auto_refresh = st.toggle(
        "Auto-refresh data",
        value=get_user_pref("auto_refresh", False),
        help="Automatically refresh dashboard data"
    )
    
    submitted = st.form_submit_button("Save preferences", type="primary")
    
    if submitted:
        set_user_pref("display_name", display_name)
        set_user_pref("theme", theme_preference)
        set_user_pref("notifications", notifications)
        set_user_pref("auto_refresh", auto_refresh)
        st.success("Preferences saved!", icon=":material/check_circle:")

# ============================================================================
# DATA & CACHE MANAGEMENT
# ============================================================================
st.subheader("Data management")

with st.container(border=True):
    st.markdown("**Cache control**")
    st.caption("Clear cached data to force a refresh of all data sources")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("Current cache status: Active")
    
    with col2:
        if st.button("Clear cache", icon=":material/delete:"):
            st.cache_data.clear()
            st.toast("Cache cleared successfully", icon=":material/check_circle:")

st.space("small")

with st.container(border=True):
    st.markdown("**Reset preferences**")
    st.caption("Clear all saved preferences and return to defaults")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prefs_count = len(st.session_state.get("user_preferences", {}))
        st.caption(f"Stored preferences: {prefs_count}")
    
    with col2:
        if st.button("Reset all", icon=":material/restart_alt:"):
            clear_all_prefs()
            st.toast("Preferences reset", icon=":material/check_circle:")
            st.rerun()

# ============================================================================
# APP INFORMATION
# ============================================================================
st.subheader("About")

with st.expander("Template information", icon=":material/info:"):
    st.markdown("""
    **Streamlit App Template v1.0.0**
    
    This template includes:
    - Multi-page architecture with st.navigation
    - Performance optimizations (caching, fragments)
    - State management utilities
    - Responsive layouts and clean design
    - Material icons and modern UI components
    
    Built with Streamlit and best practices from the Streamlit skills library.
    """)

with st.expander("Performance features", icon=":material/speed:"):
    st.markdown("""
    **Caching**
    - `@st.cache_data` for data processing with TTL
    - `@st.cache_resource` for connections and models
    
    **Fragments**
    - `@st.fragment` for isolated reruns
    - `run_every` for auto-refreshing components
    
    **Optimization**
    - Conditional rendering (toggle-based loading)
    - Forms for batching inputs
    - Efficient data structures
    """)

with st.expander("Customization guide", icon=":material/palette:"):
    st.markdown("""
    **Theme**
    - Edit `.streamlit/config.toml` to change colors, fonts, and styling
    - See `creating-streamlit-themes` skill for full options
    
    **Pages**
    - Add new pages in `app_pages/` directory
    - Register them in `streamlit_app.py` navigation
    
    **Utils**
    - Add helper functions in `utils/` directory
    - Keep business logic separate from UI code
    """)

# ============================================================================
# DANGER ZONE
# ============================================================================
st.subheader("Danger zone")

with st.container(border=True):
    st.markdown("**:material/warning: Reset application**")
    st.caption("This will clear all session state and cached data")
    
    if st.button("Reset application", type="primary", icon=":material/warning:"):
        # Clear everything
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("Application reset complete", icon=":material/check_circle:")
        st.rerun()
