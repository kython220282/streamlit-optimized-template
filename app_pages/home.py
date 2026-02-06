"""
Home Page
=========
Welcome page with overview and quick links.
Demonstrates basic Streamlit components and patterns.
"""

import streamlit as st
from utils.state_utils import get_user_pref, set_user_pref

# ============================================================================
# PAGE CONTENT
# ============================================================================
# Note: Title is handled in streamlit_app.py, don't repeat it here

st.markdown("""
Welcome to your Streamlit app! This template includes:
- :material/dashboard: Multi-page navigation
- :material/speed: Performance optimizations (caching, fragments)
- :material/palette: Custom theming
- :material/code: Clean code organization
- :material/check_circle: Best practices built-in
""")

# ============================================================================
# QUICK START SECTION
# ============================================================================
st.subheader("Quick start")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("**:material/edit: Building your app**")
        st.caption("Replace this home page with your content")
        st.page_link("app_pages/dashboard.py", label="See dashboard example", icon=":material/arrow_forward:")

with col2:
    with st.container(border=True):
        st.markdown("**:material/tune: Customization**")
        st.caption("Modify theme in .streamlit/config.toml")
        st.page_link("app_pages/settings.py", label="Check settings", icon=":material/arrow_forward:")

# ============================================================================
# INTERACTIVE DEMO
# ============================================================================
st.subheader("Interactive demo")

# Example of state management with user preferences
name = st.text_input(
    "Your name",
    value=get_user_pref("name", ""),
    help="This value is saved in session state"
)

if name:
    set_user_pref("name", name)
    st.success(f"Hello, {name}! :wave:")

# Example of selection widgets
view_mode = st.segmented_control(
    "View mode",
    ["Grid", "List", "Compact"],
    default=get_user_pref("view_mode", "Grid")
)
set_user_pref("view_mode", view_mode)

# ============================================================================
# FEATURES OVERVIEW
# ============================================================================
st.subheader("Template features")

with st.expander("Performance optimizations", icon=":material/speed:"):
    st.markdown("""
    - **Caching**: Examples in `utils/cache_utils.py`
        - `@st.cache_data` for data processing
        - `@st.cache_resource` for connections
    - **Fragments**: Isolated reruns for interactive components
    - **Forms**: Batch multiple inputs to reduce reruns
    - **Conditional rendering**: Only load what's visible
    """)

with st.expander("Code organization", icon=":material/folder:"):
    st.markdown("""
    ```
    streamlit_app.py        # Main entry & navigation
    app_pages/              # Page modules
    utils/                  # Helper functions
    .streamlit/config.toml  # Theme & settings
    ```
    """)

with st.expander("Best practices", icon=":material/check_circle:"):
    st.markdown("""
    - Wide layout for dashboards
    - Material icons (not emojis)
    - Sentence casing for labels
    - Clean spacing (no excessive dividers)
    - Proper state management
    - TTL on cached data
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.space("large")
st.caption("Built with Streamlit • Template v1.0.0")
