"""
Dashboard Application

Main entry point for the Credit Risk Analytics Dashboard.
This can be run with Streamlit or adapted for Dash.

Usage:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_streamlit_app():
    """
    Run the Streamlit dashboard application.
    
    This function sets up and runs the main dashboard interface.
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Install with: pip install streamlit")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Credit Risk Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("📊 Credit Risk Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Model Performance", "Predictions", "Monitoring", "Model Comparison"]
    )
    
    # Main content
    if page == "Overview":
        render_overview_page()
    elif page == "Model Performance":
        render_performance_page()
    elif page == "Predictions":
        render_predictions_page()
    elif page == "Monitoring":
        render_monitoring_page()
    elif page == "Model Comparison":
        render_comparison_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Credit Risk Analytics v1.0")


def render_overview_page():
    """Render the overview dashboard page."""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("📊 Credit Risk Analytics Overview")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model AUC-ROC",
            value="0.85",
            delta="0.02",
            help="Area Under ROC Curve"
        )
    
    with col2:
        st.metric(
            label="KS Statistic",
            value="0.52",
            delta="0.01",
            help="Kolmogorov-Smirnov Statistic"
        )
    
    with col3:
        st.metric(
            label="Default Rate",
            value="15.3%",
            delta="-0.5%",
            help="Current portfolio default rate"
        )
    
    with col4:
        st.metric(
            label="Predictions Today",
            value="1,234",
            delta="123",
            help="Number of predictions made today"
        )
    
    st.markdown("---")
    
    # Charts placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Trend")
        st.info("📈 Performance chart will be displayed here")
        # TODO: Add actual chart
        # fig = create_performance_chart(...)
        # st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Distribution")
        st.info("📊 Prediction distribution will be displayed here")
        # TODO: Add actual chart
    
    # Recent activity
    st.markdown("---")
    st.subheader("Recent Activity")
    st.info("📋 Recent predictions and alerts will be displayed here")


def render_performance_page():
    """Render the model performance page."""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("📈 Model Performance")
    st.markdown("---")
    
    # Model selector
    model_name = st.selectbox(
        "Select Model",
        ["Loan Default Classifier v1", "Credit Risk Scorer v1", "Future Risk Predictor v1"]
    )
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", "0.87")
        st.metric("Precision", "0.82")
    
    with col2:
        st.metric("Recall", "0.79")
        st.metric("F1 Score", "0.80")
    
    with col3:
        st.metric("AUC-ROC", "0.85")
        st.metric("Gini", "0.70")
    
    # Charts
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Confusion Matrix", "Lift Chart", "Calibration"])
    
    with tab1:
        st.info("🔄 ROC curve will be displayed here")
    
    with tab2:
        st.info("📊 Confusion matrix will be displayed here")
    
    with tab3:
        st.info("📈 Lift chart will be displayed here")
    
    with tab4:
        st.info("📉 Calibration curve will be displayed here")


def render_predictions_page():
    """Render the predictions page."""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("🎯 Predictions")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Predictions"])
    
    with tab1:
        st.subheader("Make a Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
            annual_income = st.number_input("Annual Income", min_value=0, value=50000)
            employment_length = st.slider("Employment Length (years)", 0, 30, 5)
        
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 700)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
        
        if st.button("Get Prediction", type="primary"):
            st.success("✅ Prediction: **Low Risk** (Default Probability: 12.3%)")
            
            st.subheader("Explanation")
            st.info("📊 SHAP/LIME explanation will be displayed here")
    
    with tab2:
        st.subheader("Batch Predictions")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            st.info("📁 File uploaded. Processing...")
            st.info("📊 Results will be displayed here")


def render_monitoring_page():
    """Render the monitoring page."""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("🔍 Model Monitoring")
    st.markdown("---")
    
    # Alerts
    st.subheader("Active Alerts")
    
    alert_data = [
        {"Severity": "⚠️ Warning", "Type": "Data Drift", "Message": "Feature 'income' distribution shifted", "Time": "2h ago"},
        {"Severity": "ℹ️ Info", "Type": "Performance", "Message": "AUC decreased by 1.5%", "Time": "5h ago"},
    ]
    
    st.dataframe(alert_data, use_container_width=True)
    
    st.markdown("---")
    
    # Drift monitoring
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Drift")
        st.metric("PSI Score", "0.15", delta="0.03")
        st.info("📊 Feature drift chart will be displayed here")
    
    with col2:
        st.subheader("Performance Drift")
        st.metric("AUC Change", "-1.5%", delta="-1.5%")
        st.info("📈 Performance trend will be displayed here")


def render_comparison_page():
    """Render the model comparison page."""
    try:
        import streamlit as st
    except ImportError:
        return
    
    st.title("⚖️ Model Comparison")
    st.markdown("---")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_a = st.selectbox("Model A", ["Random Forest v1", "XGBoost v1", "Logistic Regression v1"])
    
    with col2:
        model_b = st.selectbox("Model B", ["XGBoost v1", "Random Forest v1", "Logistic Regression v1"])
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("Performance Comparison")
    
    comparison_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "KS Statistic"],
        model_a: [0.87, 0.82, 0.79, 0.80, 0.85, 0.52],
        model_b: [0.86, 0.84, 0.77, 0.80, 0.84, 0.50]
    }
    
    st.dataframe(comparison_data, use_container_width=True)
    
    # Visual comparison
    st.info("📊 ROC curve comparison will be displayed here")


if __name__ == "__main__":
    run_streamlit_app()
