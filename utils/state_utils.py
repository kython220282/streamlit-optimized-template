"""
State Management Utilities
===========================
Helper functions for managing session state and user preferences.

Session state persists for the duration of a user's session.
Use these utilities to keep state management consistent across pages.
"""

import streamlit as st
from typing import Any, Optional


def get_user_pref(key: str, default: Any = None) -> Any:
    """
    Get a user preference from session state.
    
    Preferences are stored in st.session_state.user_preferences
    to keep them organized and separate from page-specific state.
    
    Args:
        key: Preference key
        default: Default value if preference doesn't exist
        
    Returns:
        The preference value or default
        
    Example:
        theme = get_user_pref("theme", "Auto")
    """
    prefs = st.session_state.get("user_preferences", {})
    return prefs.get(key, default)


def set_user_pref(key: str, value: Any) -> None:
    """
    Set a user preference in session state.
    
    Args:
        key: Preference key
        value: Value to store
        
    Example:
        set_user_pref("theme", "Dark")
    """
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}
    st.session_state.user_preferences[key] = value


def clear_all_prefs() -> None:
    """
    Clear all user preferences.
    
    Useful for reset functionality or testing.
    """
    st.session_state.user_preferences = {}


def init_page_state(page_name: str, defaults: dict) -> None:
    """
    Initialize page-specific state with defaults.
    
    Use prefixed keys (page_name_key) to avoid conflicts between pages.
    Only initializes values that don't already exist.
    
    Args:
        page_name: Name of the page (used as prefix)
        defaults: Dictionary of key-value pairs to initialize
        
    Example:
        init_page_state("dashboard", {
            "date_range": (start_date, end_date),
            "category": "All",
        })
    """
    for key, value in defaults.items():
        state_key = f"{page_name}_{key}"
        st.session_state.setdefault(state_key, value)


def get_page_state(page_name: str, key: str, default: Any = None) -> Any:
    """
    Get page-specific state value.
    
    Args:
        page_name: Name of the page (used as prefix)
        key: State key
        default: Default value if state doesn't exist
        
    Returns:
        The state value or default
        
    Example:
        category = get_page_state("dashboard", "category", "All")
    """
    state_key = f"{page_name}_{key}"
    return st.session_state.get(state_key, default)


def set_page_state(page_name: str, key: str, value: Any) -> None:
    """
    Set page-specific state value.
    
    Args:
        page_name: Name of the page (used as prefix)
        key: State key
        value: Value to store
        
    Example:
        set_page_state("dashboard", "category", "Sales")
    """
    state_key = f"{page_name}_{key}"
    st.session_state[state_key] = value


def clear_page_state(page_name: str) -> None:
    """
    Clear all state for a specific page.
    
    Args:
        page_name: Name of the page whose state to clear
        
    Example:
        clear_page_state("dashboard")
    """
    prefix = f"{page_name}_"
    keys_to_delete = [key for key in st.session_state.keys() if key.startswith(prefix)]
    for key in keys_to_delete:
        del st.session_state[key]
