# Streamlit App Template

A production-ready Streamlit template with performance optimizations, clean architecture, and best practices built in.

📦 **[GitHub Repository](https://github.com/kython220282/streamlit-optimized-template)**

## Features

✨ **Performance First**
- Comprehensive caching with `@st.cache_data` and `@st.cache_resource`
- Fragment-based reruns for isolated updates
- Auto-refreshing components
- Conditional rendering to avoid unnecessary computations

🎨 **Modern UI**
- Wide layout for dashboards
- Material icons throughout
- Clean, professional design
- Custom theming via `config.toml`
- Responsive layouts with cards and containers

🏗️ **Clean Architecture**
- Multi-page structure with `st.navigation`
- Separated business logic (utils/) and UI (app_pages/)
- Centralized state management
- Well-documented code

📊 **Dashboard Ready**
- Metrics with deltas and sparklines
- Interactive charts
- Configurable dataframes
- Real-time updates with fragments

🤖 **ML-Ready**
- Model loading patterns (scikit-learn, PyTorch, TensorFlow)
- Interactive prediction interfaces
- File upload for datasets and images
- Model evaluation dashboards
- Training interface examples

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kython220282/streamlit-optimized-template.git
cd streamlit-optimized-template
```

### 2. Install dependencies

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install streamlit pandas numpy

# Or using pip
pip install streamlit pandas numpy
```

### 3. Run the app

```bash
streamlit run streamlit_app.py

# Or with uv
uv run streamlit run streamlit_app.py
```

### 4. Open your browser

Navigate to `http://localhost:8501`

## Project Structure

```
streamlit_app.py          # Main entry point with navigation
app_pages/                # Page modules
    ├── home.py           # Welcome page
    ├── dashboard.py      # Example dashboard with metrics
    ├── ml_demo.py        # ML demo with training & prediction
    └── settings.py       # Settings and preferences
utils/                    # Helper functions
    ├── cache_utils.py    # Caching patterns and data loading
    ├── ml_utils.py       # ML model loading and utilities
    └── state_utils.py    # State management utilities
.streamlit/
    └── config.toml       # Theme and configuration
README.md                 # This file
```

## Building Your App

### Adding Pages

1. Create a new file in `app_pages/`:
```python
# app_pages/my_page.py
import streamlit as st

st.write("My new page content")
```

2. Register it in `streamlit_app.py`:
```python
page = st.navigation({
    "": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
        st.Page("app_pages/my_page.py", title="My Page", icon=":material/star:"),
    ],
})
```

### Loading Your Data

Replace the sample data in `utils/cache_utils.py`:

```python
@st.cache_data(ttl="5m")
def load_sample_data() -> pd.DataFrame:
    # Replace with your data loading logic
    df = pd.read_csv("your_data.csv")
    # or
    # df = fetch_from_api()
    # or
    # conn = st.connection("snowflake")
    # df = conn.query("SELECT * FROM table")
    return df
```

### Customizing the Theme

Edit `.streamlit/config.toml` to match your brand:

```toml
[theme]
primaryColor = "#FF4B4B"      # Your brand color
backgroundColor = "#FFFFFF"
textColor = "#262730"
font = "sans-serif"
```

See the [creating-streamlit-themes skill](https://docs.streamlit.io/develop/concepts/configuration/theming) for comprehensive theming options.

## Performance Best Practices

### Caching

```python
# Cache data transformations
@st.cache_data(ttl="5m")
def process_data(df):
    return df.groupby("category").sum()

# Cache connections and resources
@st.cache_resource
def get_database_connection():
    return create_connection()
```

### Fragments

```python
# Isolate reruns to specific components
@st.fragment
def interactive_chart():
    chart_type = st.selectbox("Type", ["bar", "line"])
    st.line_chart(data) if chart_type == "line" else st.bar_chart(data)

# Auto-refresh without rerunning the whole page
@st.fragment(run_every="30s")
def live_metrics():
    st.metric("Users", get_current_users())
```

### Conditional Rendering

```python
# Only compute expensive operations when needed
if st.toggle("Show advanced analytics"):
    expensive_computation()  # Only runs when toggled on
```

### Forms

```python
# Batch multiple inputs to reduce reruns
with st.form("filters"):
    start_date = st.date_input("Start")
    end_date = st.date_input("End")
    category = st.selectbox("Category", options)
    submitted = st.form_submit_button("Apply")

if submitted:
    filtered_data = load_data(start_date, end_date, category)
```

## State Management

```python
# User preferences (persists across pages)
from utils.state_utils import get_user_pref, set_user_pref

theme = get_user_pref("theme", "Auto")
set_user_pref("theme", "Dark")

# Page-specific state (organized with prefixes)
from utils.state_utils import get_page_state, set_page_state

category = get_page_state("dashboard", "category", "All")
set_page_state("dashboard", "category", "Sales")
```

## Common Patterns

### KPI Metrics Row

```python
with st.container(horizontal=True):
    st.metric("Revenue", "$1.2M", "+12%", border=True)
    st.metric("Orders", "1.4k", "+5%", border=True)
    st.metric("Users", "324", "+8%", border=True)
```

### Dashboard Cards

```python
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("**Revenue Trend**")
        st.line_chart(revenue_data)

with col2:
    with st.container(border=True):
        st.markdown("**Top Products**")
        st.dataframe(top_products)
```

### Selection Widgets

```python
# Visible options (2-5 choices)
view = st.segmented_control("View", ["Grid", "List", "Compact"])
tags = st.pills("Tags", ["Sales", "Marketing", "Support"], selection_mode="multi")

# Many options (dropdown)
country = st.selectbox("Country", long_list_of_countries)
regions = st.multiselect("Regions", long_list_of_regions)
```

## Machine Learning Integration

This template is ML-ready with patterns for model loading, predictions, and evaluation.

### Loading Models

```python
# scikit-learn models
from utils.ml_utils import load_sklearn_model

@st.cache_resource
def get_model():
    return load_sklearn_model("models/classifier.pkl")

model = get_model()
predictions = model.predict(X_test)
```

```python
# PyTorch models
from utils.ml_utils import load_pytorch_model
import torch

@st.cache_resource
def get_pytorch_model():
    return load_pytorch_model(MyNeuralNet, "models/weights.pth", num_classes=10)

model = get_pytorch_model()
with torch.no_grad():
    output = model(input_tensor)
```

```python
# TensorFlow/Keras models
from utils.ml_utils import load_tensorflow_model

@st.cache_resource
def get_tf_model():
    return load_tensorflow_model("models/my_model.h5")

model = get_tf_model()
predictions = model.predict(X_test)
```

### File Upload for ML

```python
from utils.ml_utils import upload_and_parse_csv, upload_and_parse_image

# CSV data upload
df = upload_and_parse_csv()
if df is not None:
    predictions = model.predict(df)
    st.write("Predictions:", predictions)

# Image upload for computer vision
image = upload_and_parse_image()
if image is not None:
    st.image(image, caption="Uploaded Image")
    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
```

### Prediction Interface

```python
# Interactive prediction form
st.subheader("Make Predictions")

col1, col2 = st.columns(2)
with col1:
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature_3 = st.slider("Feature 3", 0, 100, 50)
    feature_4 = st.selectbox("Feature 4", options=["A", "B", "C"])

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame([{
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'feature_4': feature_4
    }])
    
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0].max()
    
    st.metric("Prediction", prediction)
    st.metric("Confidence", f"{confidence:.1%}")
```

### Model Evaluation Dashboard

```python
from utils.ml_utils import display_classification_metrics, display_feature_importance

# Show performance metrics
y_pred = model.predict(X_test)
display_classification_metrics(y_test, y_pred)

# Show feature importance
if hasattr(model, 'feature_importances_'):
    display_feature_importance(feature_names, model.feature_importances_)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import plotly.express as px

cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
st.plotly_chart(fig)
```

### Caching for ML

```python
# Cache model loading (resource - not serialized)
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

# Cache predictions (data - serialized and hashed)
@st.cache_data(ttl="1h")
def predict_batch(_model, X):
    # Underscore prefix excludes _model from hash
    return _model.predict(X)

# Cache preprocessing
@st.cache_data
def preprocess_data(df):
    # Expensive transformations cached
    df_processed = df.copy()
    df_processed = apply_transformations(df_processed)
    return df_processed
```

### Model Training Interface

```python
st.header("Train Model")

# Hyperparameters
with st.expander("Hyperparameters"):
    max_depth = st.slider("Max Depth", 1, 20, 5)
    n_estimators = st.number_input("Estimators", 10, 200, 100)
    learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.1)

# Train button
if st.button("Train Model", type="primary"):
    with st.spinner("Training..."):
        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state['model'] = model
        
        # Show results
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Training complete! Accuracy: {accuracy:.2%}")
```

### Real-time Model Monitoring

```python
# Auto-refresh metrics every 30 seconds
@st.fragment(run_every="30s")
def live_model_metrics():
    current_predictions = get_recent_predictions()
    current_accuracy = calculate_accuracy(current_predictions)
    
    st.metric(
        "Live Accuracy", 
        f"{current_accuracy:.2%}",
        delta=f"{current_accuracy - baseline_accuracy:.2%}"
    )
    st.line_chart(get_accuracy_over_time())
```

## Deployment

### Streamlit Community Cloud

1. Fork this repository: [streamlit-optimized-template](https://github.com/kython220282/streamlit-optimized-template)
2. Connect your fork at [share.streamlit.io](https://share.streamlit.io)
3. Deploy!

Or use this template directly:
- Click "Use this template" on [GitHub](https://github.com/kython220282/streamlit-optimized-template)
- Create your own repository
- Deploy to Streamlit Community Cloud

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install streamlit pandas numpy

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Secrets Management

Store sensitive data in `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
[database]
host = "your-db-host"
username = "your-username"
password = "your-password"

[api]
api_key = "your-api-key"
```

Access in your app:
```python
import streamlit as st

db_host = st.secrets["database"]["host"]
api_key = st.secrets["api"]["api_key"]
```

⚠️ **Important:** Add `.streamlit/secrets.toml` to your `.gitignore`!

## Production Checklist

Before deploying to production:

- [ ] Set reasonable TTL on all caches (avoid stale data)
- [ ] Test with production-scale data volumes
- [ ] Configure secrets properly in deployment platform
- [ ] Add error handling for data loading failures
- [ ] Remove debug/development code
- [ ] Test all pages and navigation flows
- [ ] Verify responsive layout on different screen sizes
- [ ] Set up monitoring/logging if needed

## Resources

- [GitHub Repository](https://github.com/kython220282/streamlit-optimized-template)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Forum](https://discuss.streamlit.io)
- [Performance Tips](https://docs.streamlit.io/develop/concepts/architecture/caching)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/kython220282/streamlit-optimized-template).

## License

MIT License - feel free to use this template for any project!
