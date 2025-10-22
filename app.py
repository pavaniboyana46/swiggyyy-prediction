import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Swiggy Prediction AI/ML Platform",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API configuration
BACKEND_URL = "http://localhost:5000"

class SwiggyMLFrontend:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.current_data = None
        self.model_metrics = None
        
    def check_backend_connection(self):
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_data(self, file):
        """Upload data to backend"""
        try:
            files = {'file': file}
            response = requests.post(f"{self.backend_url}/upload_data", files=files)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def process_data(self, params):
        """Process data through backend"""
        try:
            response = requests.post(f"{self.backend_url}/process_data", json=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def engineer_features(self, params):
        """Engineer features through backend"""
        try:
            response = requests.post(f"{self.backend_url}/engineer_features", json=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def train_model(self, params):
        """Train model through backend"""
        try:
            response = requests.post(f"{self.backend_url}/train_model", json=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def make_predictions(self, data=None):
        """Make predictions through backend"""
        try:
            if data is None:
                response = requests.post(f"{self.backend_url}/predict")
            else:
                response = requests.post(f"{self.backend_url}/predict", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_metrics(self):
        """Get model metrics from backend"""
        try:
            response = requests.get(f"{self.backend_url}/get_metrics")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_data_info(self):
        """Get data information from backend"""
        try:
            response = requests.get(f"{self.backend_url}/get_data_info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    st.title("🍽️ Swiggy Prediction AI/ML Platform")
    st.markdown("---")
    
    # Initialize frontend
    frontend = SwiggyMLFrontend()
    
    # Check backend connection
    if not frontend.check_backend_connection():
        st.error("❌ Backend server is not running! Please start the backend server first.")
        st.code("python backend/app.py")
        return
    
    st.success("✅ Backend server is connected!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Data Upload & Processing", "🔧 Feature Engineering", "🤖 Model Training", "🔮 Predictions", "📈 Analytics"]
    )
    
    if page == "📊 Data Upload & Processing":
        data_upload_page(frontend)
    elif page == "🔧 Feature Engineering":
        feature_engineering_page(frontend)
    elif page == "🤖 Model Training":
        model_training_page(frontend)
    elif page == "🔮 Predictions":
        predictions_page(frontend)
    elif page == "📈 Analytics":
        analytics_page(frontend)

def data_upload_page(frontend):
    st.header("📊 Data Upload & Processing")
    
    # File upload section
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your Swiggy delivery dataset"
    )
    
    if uploaded_file is not None:
        # Upload to backend
        result = frontend.upload_data(uploaded_file)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.success("✅ Data uploaded successfully!")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", result['shape'][0])
            with col2:
                st.metric("Columns", result['shape'][1])
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show data preview
            st.subheader("Data Preview")
            preview_df = pd.DataFrame(result['preview'])
            st.dataframe(preview_df)
            
            # Data processing section
            st.subheader("Data Processing")
            
            with st.expander("Processing Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_strategy = st.selectbox(
                        "Missing Value Strategy",
                        ["auto", "drop", "mean", "median"]
                    )
                    outlier_method = st.selectbox(
                        "Outlier Handling",
                        ["iqr", "zscore"]
                    )
                
                with col2:
                    encoding_method = st.selectbox(
                        "Categorical Encoding",
                        ["label", "onehot"]
                    )
                    scaling_method = st.selectbox(
                        "Feature Scaling",
                        ["standard", "minmax"]
                    )
            
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    params = {
                        "missing_strategy": missing_strategy,
                        "outlier_method": outlier_method,
                        "encoding_method": encoding_method,
                        "scaling_method": scaling_method
                    }
                    
                    result = frontend.process_data(params)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success("✅ Data processed successfully!")
                        
                        # Show processing results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processed Rows", result['shape'][0])
                        with col2:
                            st.metric("Processed Columns", result['shape'][1])
                        
                        # Show missing values info
                        if result['missing_values']:
                            st.subheader("Missing Values After Processing")
                            missing_df = pd.DataFrame(list(result['missing_values'].items()), 
                                                    columns=['Column', 'Missing Count'])
                            st.dataframe(missing_df)

def feature_engineering_page(frontend):
    st.header("🔧 Feature Engineering")
    
    # Get current data info
    data_info = frontend.get_data_info()
    
    if "error" in data_info:
        st.error("❌ No data available. Please upload data first.")
        return
    
    st.success("✅ Data is available for feature engineering!")
    
    # Display current data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data_info['shape'][0])
    with col2:
        st.metric("Columns", data_info['shape'][1])
    with col3:
        st.metric("Missing Values", sum(data_info['missing_values'].values()))
    
    # Feature engineering options
    st.subheader("Feature Engineering Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Time Features**")
        time_features = st.checkbox("Create time-based features", value=True)
        
        st.write("**Distance Features**")
        distance_features = st.checkbox("Create distance-based features", value=True)
        
        st.write("**Statistical Features**")
        statistical_features = st.checkbox("Create statistical features", value=True)
    
    with col2:
        st.write("**Interaction Features**")
        interaction_features = st.checkbox("Create interaction features", value=True)
        
        st.write("**Polynomial Features**")
        polynomial_features = st.checkbox("Create polynomial features", value=False)
        if polynomial_features:
            polynomial_degree = st.slider("Polynomial Degree", 2, 4, 2)
    
    if st.button("Engineer Features"):
        with st.spinner("Creating features..."):
            params = {
                "time_features": time_features,
                "distance_features": distance_features,
                "statistical_features": statistical_features,
                "interaction_features": interaction_features,
                "polynomial_features": polynomial_features
            }
            
            if polynomial_features:
                params["polynomial_degree"] = polynomial_degree
            
            result = frontend.engineer_features(params)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Features engineered successfully!")
                
                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Features", result['shape'][1])
                with col2:
                    st.metric("New Features", len(result['new_features']))
                
                # Show new features
                if result['new_features']:
                    st.subheader("New Features Created")
                    st.write(", ".join(result['new_features']))

def model_training_page(frontend):
    st.header("🤖 Model Training")
    
    # Get current data info
    data_info = frontend.get_data_info()
    
    if "error" in data_info:
        st.error("❌ No data available. Please upload and process data first.")
        return
    
    st.success("✅ Data is available for model training!")
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "xgboost", "lightgbm"]
        )
        
        target_column = st.selectbox(
            "Select Target Column",
            data_info['columns']
        )
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        # Model-specific parameters
        if model_type == "random_forest":
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            max_depth = st.slider("Max Depth", 5, 20, 10)
        elif model_type == "xgboost":
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        elif model_type == "lightgbm":
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            params = {
                "model_type": model_type,
                "target_column": target_column,
                "test_size": test_size
            }
            
            result = frontend.train_model(params)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                metrics = result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.3f}")
                with col4:
                    st.metric("MSE", f"{metrics['mse']:.3f}")
                
                # Show detailed metrics
                st.subheader("Detailed Metrics")
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df)

def predictions_page(frontend):
    st.header("🔮 Predictions")
    
    # Get model metrics to check if model is trained
    metrics_result = frontend.get_metrics()
    
    if "error" in metrics_result:
        st.error("❌ No trained model available. Please train a model first.")
        return
    
    st.success("✅ Trained model is available!")
    
    # Prediction options
    st.subheader("Prediction Options")
    
    prediction_type = st.radio(
        "Choose prediction type",
        ["Use current dataset", "Enter custom data"]
    )
    
    if prediction_type == "Use current dataset":
        if st.button("Make Predictions"):
            with st.spinner("Making predictions..."):
                result = frontend.make_predictions()
                
                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display predictions
                    predictions = result['predictions']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Mean Prediction", f"{np.mean(predictions):.2f}")
                    with col3:
                        st.metric("Std Prediction", f"{np.std(predictions):.2f}")
                    
                    # Show predictions chart
                    fig = px.histogram(
                        x=predictions,
                        title="Prediction Distribution",
                        labels={'x': 'Predicted Value', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.subheader("Custom Data Input")
        
        # Create input form
        with st.form("custom_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0)
                order_value = st.number_input("Order Value (₹)", min_value=50.0, max_value=5000.0, value=500.0)
                preparation_time = st.number_input("Preparation Time (min)", min_value=5.0, max_value=120.0, value=30.0)
            
            with col2:
                restaurant_rating = st.number_input("Restaurant Rating", min_value=1.0, max_value=5.0, value=4.0)
                weather = st.selectbox("Weather", ["sunny", "rainy", "cloudy", "windy"])
                traffic_condition = st.selectbox("Traffic Condition", ["low", "medium", "high"])
            
            submitted = st.form_submit_button("Make Prediction")
            
            if submitted:
                custom_data = {
                    "distance": distance,
                    "order_value": order_value,
                    "preparation_time": preparation_time,
                    "restaurant_rating": restaurant_rating,
                    "weather": weather,
                    "traffic_condition": traffic_condition
                }
                
                with st.spinner("Making prediction..."):
                    result = frontend.make_predictions([custom_data])
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        prediction = result['predictions'][0]
                        st.success(f"✅ Predicted Delivery Time: {prediction:.2f} minutes")

def analytics_page(frontend):
    st.header("📈 Analytics & Insights")
    
    # Get data info
    data_info = frontend.get_data_info()
    
    if "error" in data_info:
        st.error("❌ No data available. Please upload data first.")
        return
    
    # Get model metrics
    metrics_result = frontend.get_metrics()
    
    st.subheader("Data Overview")
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", data_info['shape'][0])
    with col2:
        st.metric("Features", data_info['shape'][1])
    with col3:
        st.metric("Missing Values", sum(data_info['missing_values'].values()))
    with col4:
        if "error" not in metrics_result:
            st.metric("Model R² Score", f"{metrics_result['metrics']['r2']:.3f}")
        else:
            st.metric("Model Status", "Not Trained")
    
    # Data types distribution
    st.subheader("Data Types Distribution")
    dtypes_df = pd.DataFrame(list(data_info['dtypes'].items()), columns=['Column', 'Data Type'])
    dtype_counts = dtypes_df['Data Type'].value_counts()
    
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Data Types Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values visualization
    if any(data_info['missing_values'].values()):
        st.subheader("Missing Values Analysis")
        missing_df = pd.DataFrame(list(data_info['missing_values'].items()), 
                                columns=['Column', 'Missing Count'])
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing Count',
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model performance (if available)
    if "error" not in metrics_result:
        st.subheader("Model Performance")
        metrics = metrics_result['metrics']
        
        # Performance metrics visualization
        metrics_data = {
            'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE'],
            'Value': [metrics['r2'], metrics['rmse'], metrics['mae'], metrics['mse']]
        }
        
        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Value',
            title="Model Performance Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
