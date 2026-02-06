"""
ML Demo Page
============
Example machine learning interface demonstrating:
- Model training and caching
- Interactive predictions
- Feature importance visualization
- Model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.ml_utils import (
    create_sample_ml_data,
    display_feature_importance,
    display_classification_metrics,
    upload_and_parse_csv,
    preprocess_features
)

st.markdown("Interactive machine learning demonstration with a simple classifier (Random Forest, limited to 20 features)")

# ============================================================================
# TABS FOR DIFFERENT ML WORKFLOWS
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    ":material/model_training: Train Model",
    ":material/analytics: Make Predictions", 
    ":material/assessment: Model Evaluation"
])

# ============================================================================
# TAB 1: TRAIN MODEL
# ============================================================================
with tab1:
    st.header("Train a Simple Classifier - Random Forest (max 20 features)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        # Model hyperparameters
        with st.expander("⚙️ Hyperparameters", expanded=True):
            max_depth = st.slider("Max Depth", 1, 20, 5, 
                                help="Maximum depth of decision tree")
            n_estimators = st.number_input("Number of Trees", 10, 200, 100,
                                          help="Number of trees in the forest")
            random_state = st.number_input("Random Seed", 0, 999, 42)
        
        # Data split
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 
                            help="Proportion of data for testing")
        
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Training Data")
        
        # Option to use sample data or upload
        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload CSV"],
            help="Choose between built-in sample or your own data"
        )
        
        if data_source == "Sample Data":
            df = create_sample_ml_data()
            st.info(f"📊 Using sample dataset: {len(df)} rows, {len(df.columns)} features")
            
        else:
            uploaded_df = upload_and_parse_csv()
            if uploaded_df is not None:
                df = uploaded_df
            else:
                st.warning("⚠️ Please upload a CSV file to continue")
                df = None
        
        # Show data preview
        if df is not None:
            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
    
    # Train model when button clicked
    if train_button and df is not None:
        with st.spinner("Training model..."):
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                
                # Prepare data
                feature_cols = [col for col in df.columns if col != 'target']
                X = df[feature_cols]
                y = df['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train model
                model = RandomForestClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store in session state
                st.session_state['trained_model'] = model
                st.session_state['feature_cols'] = feature_cols
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.subheader("Training Results")
                display_classification_metrics(y_test, y_pred)
                
                # Feature importance
                st.subheader("Feature Importance")
                display_feature_importance(feature_cols, model.feature_importances_)
                
            except ImportError:
                st.error("❌ scikit-learn not installed. Run: `pip install scikit-learn`")
            except Exception as e:
                st.error(f"❌ Training error: {e}")

# ============================================================================
# TAB 2: MAKE PREDICTIONS
# ============================================================================
with tab2:
    st.header("Make Predictions")
    
    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model first (see Train Model tab)")
    else:
        model = st.session_state['trained_model']
        feature_cols = st.session_state['feature_cols']
        
        st.subheader("Input Features")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        input_data = {}
        for i, feature in enumerate(feature_cols):
            col = col1 if i % 2 == 0 else col2
            with col:
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    help=f"Enter value for {feature}"
                )
        
        # Predict button
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            # Create DataFrame from inputs
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display result
            st.divider()
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(
                    "Predicted Class",
                    f"Class {prediction}",
                    help="The predicted class label"
                )
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{max(prediction_proba):.1%}",
                    help="Model confidence in this prediction"
                )
            
            with col3:
                st.metric(
                    "Class Probabilities",
                    f"{prediction_proba[1]:.3f}",
                    help="Probability of positive class"
                )
            
            # Show probability distribution
            st.subheader("Class Probabilities")
            proba_df = pd.DataFrame({
                'Class': [f"Class {i}" for i in range(len(prediction_proba))],
                'Probability': prediction_proba
            })
            st.bar_chart(proba_df.set_index('Class'))

# ============================================================================
# TAB 3: MODEL EVALUATION
# ============================================================================
with tab3:
    st.header("Model Evaluation")
    
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model first (see Train Model tab)")
    else:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        model = st.session_state['trained_model']
        feature_cols = st.session_state['feature_cols']
        
        # Performance metrics
        st.subheader("Performance Metrics")
        display_classification_metrics(y_test, y_pred)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        try:
            from sklearn.metrics import confusion_matrix
            import plotly.express as px
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=[f"Class {i}" for i in range(len(cm))],
                y=[f"Class {i}" for i in range(len(cm))],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.info("Install plotly for visualizations: `pip install plotly`")
            # Fallback to simple display
            st.write(pd.DataFrame(
                confusion_matrix(y_test, y_pred),
                columns=[f"Pred {i}" for i in range(2)],
                index=[f"Actual {i}" for i in range(2)]
            ))
        
        # Feature Importance
        st.subheader("Feature Importance")
        display_feature_importance(feature_cols, model.feature_importances_)
        
        # Model info
        with st.expander("📋 Model Details"):
            st.write("**Model Type:**", type(model).__name__)
            st.write("**Number of Features:**", len(feature_cols))
            st.write("**Feature Names:**", ", ".join(feature_cols))
            if hasattr(model, 'n_estimators'):
                st.write("**Number of Trees:**", model.n_estimators)
            if hasattr(model, 'max_depth'):
                st.write("**Max Depth:**", model.max_depth)

# ============================================================================
# SIDEBAR INFO
# ============================================================================
with st.sidebar:
    st.header("💡 Tips")
    
    with st.expander("About This Demo", expanded=True):
        st.markdown("""
        This demo shows common ML patterns:
        
        **Training**
        - Hyperparameter tuning
        - Train/test split
        - Model caching
        
        **Prediction**
        - Interactive inputs
        - Probability estimates
        - Real-time inference
        
        **Evaluation**
        - Performance metrics
        - Confusion matrix
        - Feature importance
        """)
    
    with st.expander("Performance Tips"):
        st.markdown("""
        **Cache models:**
        ```python
        @st.cache_resource
        def load_model():
            return joblib.load('model.pkl')
        ```
        
        **Cache predictions:**
        ```python
        @st.cache_data
        def predict(_model, X):
            return _model.predict(X)
        ```
        
        **Use fragments for isolated updates:**
        ```python
        @st.fragment
        def prediction_form():
            # Only this reruns on input change
            pass
        ```
        """)
