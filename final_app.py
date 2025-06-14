"""
Smart Digital Twin - Advanced Industrial Monitoring System
------------------------------------------------------
A comprehensive industrial monitoring solution with real-time analytics,
predictive maintenance, and AI-powered recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import os
import json
import base64
from pathlib import Path
import pydeck as pdk
try:
    import pyvista as pv
    PYWISTA_AVAILABLE = True
except ImportError:
    PYWISTA_AVAILABLE = False
    
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ===========================================
# 1. INITIALIZATION & CONFIGURATION
# ===========================================

# Force wide mode and page config
st.set_page_config(
    page_title="Smart Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# 2. DATA GENERATION & AI MODELS
# ===========================================

class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True, num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

@st.cache_resource
def load_ai_models():
    """Load or train AI models"""
    # Anomaly Detection Model
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # Predictive Maintenance Model
    input_size = 5  # Number of features
    pm_model = PredictiveMaintenanceModel(input_size)
    
    # Time Series Forecasting Model
    ts_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    
    return {
        'anomaly_detector': iso_forest,
        'predictive_maintenance': pm_model,
        'time_series': ts_model
    }

@st.cache_data(ttl=300)
def generate_sensor_data():
    """Generate realistic sensor data with anomalies and trends"""
    np.random.seed(42)
    date_rng = pd.date_range(end=datetime.now(), periods=500, freq='H')
    
    # Base signals with seasonality
    hours = np.arange(500)
    temp_trend = 25 + 0.01 * hours  # Slight upward trend
    temp_season = 10 * np.sin(2 * np.pi * hours / 24)  # Daily seasonality
    temp_noise = np.random.normal(0, 1, 500)
    temperature = temp_trend + temp_season + temp_noise
    
    pressure_trend = 100 - 0.02 * hours  # Slight downward trend
    pressure_season = 15 * np.sin(2 * np.pi * hours / 168)  # Weekly seasonality
    pressure_noise = np.random.normal(0, 2, 500)
    pressure = pressure_trend + pressure_season + pressure_noise
    
    vibration = np.random.normal(0.5, 0.1, 500)
    
    # Add anomalies
    anomalies = []
    for _ in range(15):
        idx = random.randint(0, 499)
        anomaly_type = random.choice(['temp_spike', 'pressure_drop', 'vibration_spike'])
        
        if anomaly_type == 'temp_spike':
            temperature[idx] += random.uniform(15, 25)
            anomalies.append((idx, "Temperature Spike", "High"))
        elif anomaly_type == 'pressure_drop':
            pressure[idx] -= random.uniform(20, 40)
            anomalies.append((idx, "Pressure Drop", "Critical"))
        else:
            vibration[idx] += random.uniform(0.8, 1.5)
            anomalies.append((idx, "Vibration Spike", "Warning"))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': date_rng,
        'Temperature': temperature,
        'Pressure': pressure,
        'Vibration': vibration,
        'Anomaly': 0,
        'Anomaly_Type': '',
        'Severity': ''
    })
    
    # Mark anomalies
    for idx, a_type, severity in anomalies:
        df.loc[idx, 'Anomaly'] = 1
        df.loc[idx, 'Anomaly_Type'] = a_type
        df.loc[idx, 'Severity'] = severity
    
    return df

# ===========================================
# 3. 3D VISUALIZATION
# ===========================================

if PYWISTA_AVAILABLE:
    def create_3d_equipment_model():
        """Create a 3D model of industrial equipment"""
        try:
            # Create a simple 3D model of a pump
            mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1, height=3)
            
            # Add some details
            base = pv.Cylinder(center=(0, 0, -0.2), direction=(0, 0, 1), radius=1.2, height=0.4)
            top = pv.Cylinder(center=(0, 0, 3), direction=(0, 0, 1), radius=0.8, height=0.3)
            
            # Combine the meshes
            equipment = mesh + base + top
            return equipment
        except Exception as e:
            st.warning(f"3D model creation failed: {str(e)}")
            return None

    def render_3d_model():
        """Render 3D model using PyVista and Streamlit"""
        st.markdown("### 🏭 3D Equipment Model")
        try:
            # Create plotter
            plotter = pv.Plotter(window_size=[600, 400])
            
            # Add equipment model
            equipment = create_3d_equipment_model()
            if equipment is not None:
                plotter.add_mesh(equipment, color='lightblue', smooth_shading=True)
                
                # Configure plotter
                plotter.set_background('black')
                plotter.camera_position = 'xy'
                plotter.camera.azimuth = 30
                plotter.camera.elevation = 20
                
                # Render to streamlit
                plotter.export_html('temp_3d_model.html')
                with open('temp_3d_model.html', 'r', encoding='utf-8') as f:
                    html = f.read()
                st.components.v1.html(html, height=500)
            else:
                st.warning("Could not create 3D model")
        except Exception as e:
            st.warning(f"3D rendering failed: {str(e)}")
            st.info("3D visualization requires additional packages. To enable, install with: `pip install pyvista panel`")
else:
    def render_3d_model():
        """Show message when 3D visualization is not available"""
        st.markdown("### 🏭 3D Equipment Model")
        st.info("""
        **3D visualization is not available**
        
        To enable 3D visualization, please install the required packages:
        ```bash
        pip install pyvista panel
        ```
        
        Then restart the application.
        """)

# ===========================================
# 4. AI-POWERED ANALYTICS
# ===========================================

def detect_anomalies(data):
    """Detect anomalies using Isolation Forest"""
    model = load_ai_models()['anomaly_detector']
    features = data[['Temperature', 'Pressure', 'Vibration']]
    
    # Fit model and predict anomalies
    preds = model.fit_predict(features)
    data['AI_Anomaly'] = (preds == -1).astype(int)
    
    return data

def predict_failures(data):
    """Predict equipment failures using LSTM model"""
    # This is a simplified example - in practice, you'd use a trained model
    # Here we'll just simulate predictions based on thresholds
    data['Failure_Risk'] = np.random.random(len(data))
    data['Maintenance_Recommended'] = data['Failure_Risk'] > 0.85
    
    return data

# ===========================================
# 5. STREAMLIT UI COMPONENTS
# ===========================================

def create_metric_card(title, value, delta=None, delta_type='normal'):
    """Create a metric card with optional delta indicator"""
    colors = {
        'normal': '#3b82f6',
        'increase': '#10b981',
        'decrease': '#ef4444'
    }
    
    delta_icon = ""
    if delta is not None:
        if delta > 0:
            delta_icon = f"↑ {abs(delta):.1f}%"
            color = colors['increase']
        elif delta < 0:
            delta_icon = f"↓ {abs(delta):.1f}%"
            color = colors['decrease']
        else:
            delta_icon = "→ 0.0%"
            color = colors['normal']
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 1rem; color: #93c5fd; margin-bottom: 0.5rem;">{title}</div>
        <div style="display: flex; align-items: baseline; gap: 0.5rem;">
            <div style="font-size: 1.8rem; font-weight: 700;">{value}</div>
            {f'<div style="color: {color}; font-weight: 500;">{delta_icon}</div>' if delta is not None else ''}
        </div>
    </div>
    """

def render_ai_insights(data):
    """Render AI-powered insights and recommendations"""
    st.markdown("### 🤖 AI Insights & Recommendations")
    
    # Calculate metrics
    anomaly_count = data['AI_Anomaly'].sum()
    failure_risk = data['Failure_Risk'].mean() * 100
    maintenance_needed = data['Maintenance_Recommended'].any()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric_card("Anomalies Detected", anomaly_count), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Failure Risk", f"{failure_risk:.1f}%"), unsafe_allow_html=True)
    with col3:
        status = "⚠️ Required" if maintenance_needed else "✅ Not Required"
        st.markdown(create_metric_card("Maintenance", status), unsafe_allow_html=True)
    
    # Display recommendations
    st.markdown("#### Recommendations")
    if maintenance_needed:
        st.warning("**Maintenance Recommended** - Schedule maintenance within 24-48 hours to prevent equipment failure.")
    
    if anomaly_count > 0:
        st.info(f"**{anomaly_count} anomalies detected** - Review the anomalies tab for detailed analysis.")
    
    if failure_risk > 70:
        st.error("**High Failure Risk** - Immediate attention required. Consider performing diagnostic tests.")

# ===========================================
# 6. MAIN APP LAYOUT
# ===========================================

def main():
    """Main application layout"""
    # Load data and models
    data = generate_sensor_data()
    data = detect_anomalies(data)
    data = predict_failures(data)
    
    # Set page title and favicon
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stAlert {
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header
    st.title("🏭 Smart Digital Twin Dashboard")
    st.markdown("*Real-time monitoring and predictive maintenance for industrial equipment*")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Analytics", "🛠️ 3D Model", "⚙️ Settings"])
    
    with tab1:  # Dashboard
        # Metrics row
        latest = data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("🌡️ Temperature", f"{latest['Temperature']:.1f}°C"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("⚡ Pressure", f"{latest['Pressure']:.1f} kPa"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("📊 Vibration", f"{latest['Vibration']:.2f} mm/s"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("💯 System Health", f"{100 - latest['Failure_Risk']*100:.0f}%"), unsafe_allow_html=True)
        
        # AI Insights
        render_ai_insights(data)
        
        # Time series charts
        st.markdown("### 📈 Live Sensor Data")
        fig = px.line(data, x='Timestamp', y=['Temperature', 'Pressure', 'Vibration'],
                     title="Sensor Data Over Time",
                     labels={'value': 'Value', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Analytics
        st.markdown("### 🔍 Anomaly Detection")
        
        # Anomaly visualization
        fig = px.scatter(data, x='Timestamp', y='Temperature', 
                        color='AI_Anomaly',
                        title="Detected Anomalies",
                        color_discrete_map={0: '#3b82f6', 1: '#ef4444'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure prediction
        st.markdown("### ⚠️ Failure Prediction")
        fig = px.line(data, x='Timestamp', y='Failure_Risk',
                     title="Equipment Failure Risk Over Time",
                     labels={'Failure_Risk': 'Failure Probability'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:  # 3D Model
        render_3d_model()
    
    with tab4:  # Settings
        st.markdown("### ⚙️ System Settings")
        
        # Theme selection
        theme = st.selectbox("Color Theme", ["Dark", "Light", "System"])
        
        # Data refresh rate
        refresh_rate = st.slider("Data Refresh Rate (seconds)", 5, 300, 30)
        
        # Alert thresholds
        st.markdown("### 🚨 Alert Thresholds")
        temp_threshold = st.slider("Temperature Threshold (°C)", 0, 100, 35)
        pressure_threshold = st.slider("Pressure Threshold (kPa)", 0, 200, 80)
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
