"""
ML Utilities
=============
Helper functions for machine learning workflows.

Demonstrates:
- Model loading and caching with @st.cache_resource
- Prediction caching with @st.cache_data
- File upload handling
- Feature preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List
import pickle
import io


@st.cache_resource
def load_sklearn_model(model_path: str):
    """
    Load a scikit-learn model from disk.
    
    Cached with @st.cache_resource so the model stays in memory
    and is shared across all users/sessions.
    
    Args:
        model_path: Path to the pickled model file
        
    Returns:
        Loaded model object
        
    Example:
        model = load_sklearn_model("models/my_classifier.pkl")
        predictions = model.predict(X_test)
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_pytorch_model(model_class, model_path: str, **model_kwargs):
    """
    Load a PyTorch model.
    
    Args:
        model_class: The model class (e.g., MyNeuralNet)
        model_path: Path to the saved state dict
        **model_kwargs: Arguments to pass to model constructor
        
    Returns:
        Loaded PyTorch model in eval mode
        
    Example:
        import torch
        model = load_pytorch_model(MyModel, "models/weights.pth", num_classes=10)
        with torch.no_grad():
            output = model(input_tensor)
    """
    try:
        import torch
        
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except ImportError:
        st.error("PyTorch not installed. Run: pip install torch")
        return None
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None


@st.cache_resource
def load_tensorflow_model(model_path: str):
    """
    Load a TensorFlow/Keras model.
    
    Args:
        model_path: Path to saved model directory or .h5 file
        
    Returns:
        Loaded TensorFlow model
        
    Example:
        model = load_tensorflow_model("models/my_model.h5")
        predictions = model.predict(X_test)
    """
    try:
        import tensorflow as tf
        
        model = tf.keras.models.load_model(model_path)
        return model
    except ImportError:
        st.error("TensorFlow not installed. Run: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"Error loading TensorFlow model: {e}")
        return None


@st.cache_data(ttl="1h")
def predict_batch(_model, X: pd.DataFrame) -> np.ndarray:
    """
    Run batch predictions with caching.
    
    Note: Model parameter has underscore prefix (_model) to exclude it from
    caching hash since it's already cached with @st.cache_resource.
    
    Args:
        _model: Trained model with .predict() method
        X: Feature DataFrame
        
    Returns:
        Predictions array
    """
    try:
        predictions = _model.predict(X)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def upload_and_parse_csv() -> pd.DataFrame:
    """
    Handle CSV file upload and parsing.
    
    Returns:
        Parsed DataFrame or None if no file uploaded
        
    Example:
        df = upload_and_parse_csv()
        if df is not None:
            st.dataframe(df)
    """
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file with your data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    return None


def upload_and_parse_image():
    """
    Handle image file upload.
    
    Returns:
        PIL Image object or None
        
    Example:
        image = upload_and_parse_image()
        if image is not None:
            st.image(image)
    """
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for prediction"
    )
    
    if uploaded_file is not None:
        try:
            from PIL import Image
            image = Image.open(uploaded_file)
            return image
        except ImportError:
            st.error("PIL not installed. Run: pip install pillow")
            return None
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None
    return None


@st.cache_data
def preprocess_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Preprocess features for model input.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        Processed DataFrame with selected features
    """
    # Select only specified features
    X = df[feature_cols].copy()
    
    # Handle missing values (example strategy)
    X = X.fillna(X.mean())
    
    return X


def create_sample_ml_data() -> pd.DataFrame:
    """
    Create sample dataset for ML demonstrations.
    
    Returns:
        DataFrame with features and target
    """
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.uniform(0, 100, n_samples),
        'feature_4': np.random.randint(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create a simple target based on features
    df['target'] = (
        (df['feature_1'] * 0.5 + 
         df['feature_2'] * 0.3 + 
         df['feature_3'] * 0.01 +
         np.random.randn(n_samples) * 0.5) > 0
    ).astype(int)
    
    return df


def display_feature_importance(feature_names: List[str], importances: np.ndarray):
    """
    Display feature importance as a bar chart.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance scores
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    st.bar_chart(importance_df.set_index('Feature'))


def display_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Display classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with col2:
            st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
        with col3:
            st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
        with col4:
            st.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
            
    except ImportError:
        st.error("scikit-learn not installed. Run: pip install scikit-learn")
