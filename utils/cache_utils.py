"""
Caching Utilities
=================
Helper functions demonstrating caching patterns for performance.

Key principles:
- Use @st.cache_data for data processing and transformations
- Use @st.cache_resource for connections, models, and non-serializable objects
- Set TTL to keep data fresh (e.g., ttl="5m" for 5 minutes)
- Set max_entries to prevent unbounded cache growth
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@st.cache_data(ttl="5m")  # Cache for 5 minutes
def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for the dashboard.
    
    This is cached with a 5-minute TTL, so the data automatically
    refreshes every 5 minutes without manual intervention.
    
    In a real app, replace this with your data loading logic:
    - Database queries
    - API calls
    - File reads (CSV, Parquet, etc.)
    
    Returns:
        pd.DataFrame: Sample transaction data
    """
    # Generate sample data (replace with your actual data loading)
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    
    data = {
        "date": dates,
        "revenue": np.random.uniform(1000, 5000, 100),
        "orders": np.random.randint(20, 100, 100),
        "category": np.random.choice(["Sales", "Marketing", "Support"], 100),
        "status": np.random.choice(["Completed", "Pending", "Processing"], 100),
    }
    
    return pd.DataFrame(data)


@st.cache_data(ttl="5m")
def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute dashboard metrics from data.
    
    Cached separately from data loading so filtering doesn't
    break the cache. The dataframe hash is used as cache key.
    
    Args:
        df: Transaction data
        
    Returns:
        dict: Computed metrics
    """
    # Current period metrics
    revenue = df["revenue"].sum()
    orders = df["orders"].sum()
    avg_order = revenue / orders if orders > 0 else 0
    conversion = np.random.uniform(2, 5)  # Simulated conversion rate
    
    # Previous period comparison (simulated)
    revenue_change = np.random.uniform(-10, 20)
    orders_change = np.random.uniform(-5, 15)
    avg_order_change = np.random.uniform(-8, 12)
    conversion_change = np.random.uniform(-2, 3)
    
    return {
        "revenue": revenue,
        "orders": orders,
        "avg_order": avg_order,
        "conversion": conversion,
        "revenue_change": revenue_change,
        "orders_change": orders_change,
        "avg_order_change": avg_order_change,
        "conversion_change": conversion_change,
    }


@st.cache_data(ttl="1h", max_entries=50)
def expensive_computation(param: str) -> dict:
    """
    Example of caching expensive operations.
    
    Use max_entries to prevent unbounded cache growth when
    the function is called with many different parameters.
    
    Args:
        param: Parameter that affects the computation
        
    Returns:
        dict: Computation results
    """
    # Simulate expensive operation
    result = {
        "param": param,
        "computed_at": datetime.now().isoformat(),
        "result": np.random.random(),
    }
    return result


# Example: Resource caching for connections
# Uncomment and adapt for your use case
#
# @st.cache_resource
# def get_database_connection():
#     """
#     Create and cache a database connection.
#     
#     Use @st.cache_resource for:
#     - Database connections
#     - API clients
#     - ML models
#     - Any non-serializable objects
#     
#     WARNING: Never mutate objects returned by @st.cache_resource
#     as changes will affect all users!
#     """
#     import sqlalchemy
#     return sqlalchemy.create_engine("sqlite:///app.db")


# @st.cache_resource
# def load_ml_model():
#     """Load and cache a machine learning model."""
#     import joblib
#     return joblib.load("model.pkl")
