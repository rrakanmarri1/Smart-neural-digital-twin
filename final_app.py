"""
NEURAL DIGITAL TWIN ELITE - Advanced Industrial Monitoring
---------------------------------------------------------
A comprehensive digital twin solution with real-time analytics,
AI-powered anomaly detection, and predictive maintenance.
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
import sys
from pathlib import Path

# Optional imports with graceful fallbacks
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Anomaly detection requires scikit-learn. Install with: pip install scikit-learn")

try:
    import pyvista as pv
    PYWISTA_AVAILABLE = True
except ImportError:
    PYWISTA_AVAILABLE = False
    st.warning("3D visualization requires pyvista. Install with: pip install pyvista")

# ===========================================
# 1. PAGE CONFIGURATION
# ===========================================

def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Neural Digital Twin Elite",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #6e48aa, #9d50bb) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .status-normal { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FFC107; font-weight: bold; }
    .status-critical { 
        color: #F44336; 
        font-weight: bold; 
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================================
# 2. DATA GENERATION
# ===========================================

class EquipmentSimulator:
    """Simulates industrial equipment sensor data"""
    
    def __init__(self, equipment_id: str = "Pump-001"):
        self.equipment_id = equipment_id
        self.data = None
    
    def generate_data(self, hours: int = 500) -> pd.DataFrame:
        """Generate synthetic sensor data"""
        date_rng = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        time = np.arange(len(date_rng))
        
        # Base trends
        temp_trend = 25 + 0.01 * time
        pressure_trend = 100 - 0.02 * time
        vibration_trend = 5 + 0.005 * time
        
        # Add seasonality and noise
        daily = 5 * np.sin(2 * np.pi * time / 24)
        noise = np.random.normal(0, 2, len(date_rng))
        
        # Combine components
        temperature = np.clip(temp_trend + daily + noise, 0, 120)
        pressure = np.clip(pressure_trend + daily * 0.5 + noise * 0.7, 0, 200)
        vibration = np.clip(vibration_trend + daily * 0.3 + noise * 0.5, 0, 30)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': date_rng,
            'Temperature': temperature,
            'Pressure': pressure,
            'Vibration': vibration,
            'Equipment_ID': self.equipment_id,
            'Status': 'Normal'
        })
        
        # Add some anomalies
        self._add_anomalies(df)
        self.data = df
        return df
    
    def _add_anomalies(self, df: pd.DataFrame) -> None:
        """Inject realistic anomalies"""
        for _ in range(20):
            idx = random.randint(0, len(df)-1)
            anomaly_type = random.choice(['temp', 'pressure', 'vibration'])
            
            if anomaly_type == 'temp':
                df.at[idx, 'Temperature'] += random.uniform(15, 30)
            elif anomaly_type == 'pressure':
                df.at[idx, 'Pressure'] -= random.uniform(20, 40)
            else:
                df.at[idx, 'Vibration'] += random.uniform(5, 15)

# ===========================================
# 3. ANOMALY DETECTION
# ===========================================

class AnomalyDetector:
    """AI-powered anomaly detection"""
    
    def __init__(self, contamination: float = 0.05):
        self.model = None
        self.scaler = StandardScaler()
        self.contamination = contamination
    
    def fit(self, X: np.ndarray) -> None:
        """Train the anomaly detection model"""
        if not SKLEARN_AVAILABLE:
            return None
            
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies in the data"""
        if self.model is None:
            return np.zeros(len(X), dtype=bool)
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) == -1

# ===========================================
# 4. VISUALIZATION
# ===========================================

def plot_sensor_data(df: pd.DataFrame) -> None:
    """Create interactive sensor data visualization"""
    fig = go.Figure()
    
    # Add traces for each sensor
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Temperature'],
        name='Temperature (°C)',
        line=dict(color='#FF6B6B')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Pressure'],
        name='Pressure (kPa)',
        yaxis='y2',
        line=dict(color='#4ECDC4')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Vibration'] * 20,  # Scale for visibility
        name='Vibration (x20)',
        yaxis='y3',
        line=dict(color='#45B7D1')
    ))
    
    # Update layout
    fig.update_layout(
        title='Equipment Sensor Data',
        xaxis=dict(domain=[0.1, 0.9]),
        yaxis=dict(title='Temperature (°C)', titlefont=dict(color='#FF6B6B')),
        yaxis2=dict(
            title='Pressure (kPa)',
            titlefont=dict(color='#4ECDC4'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Vibration (mm/s)',
            titlefont=dict(color='#45B7D1'),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.95
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_equipment_status(data: pd.DataFrame) -> None:
    """Display current equipment status"""
    if data is None or len(data) == 0:
        return
    
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    
    # Determine status
    temp_status = latest['Temperature'] > 80
    pressure_status = latest['Pressure'] > 150 or latest['Pressure'] < 50
    vib_status = latest['Vibration'] > 0.8
    
    if temp_status or pressure_status or vib_status:
        status = "<span class='status-critical'>Critical</span>"
    elif latest['Temperature'] > 70 or latest['Pressure'] > 130 or latest['Vibration'] > 0.6:
        status = "<span class='status-warning'>Warning</span>"
    else:
        status = "<span class='status-normal'>Normal</span>"
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_temp = latest['Temperature'] - prev['Temperature']
        st.metric("Temperature", f"{latest['Temperature']:.1f}°C", f"{delta_temp:+.1f}°C")
    
    with col2:
        delta_pressure = latest['Pressure'] - prev['Pressure']
        st.metric("Pressure", f"{latest['Pressure']:.1f} kPa", f"{delta_pressure:+.1f} kPa")
    
    with col3:
        delta_vib = latest['Vibration'] - prev['Vibration']
        st.metric("Vibration", f"{latest['Vibration']:.2f} mm/s", f"{delta_vib:+.2f} mm/s")
    
    with col4:
        st.markdown(f"### Status: {status}", unsafe_allow_html=True)

# ===========================================
# 5. MAIN APPLICATION
# ===========================================

def main():
    # Setup page
    setup_page()
    
    # Title and description
    st.title("🤖 Neural Digital Twin Elite")
    st.markdown("*Advanced industrial equipment monitoring and analytics*")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = EquipmentSimulator()
        st.session_state.data = st.session_state.simulator.generate_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Refresh Data"):
            st.session_state.data = st.session_state.simulator.generate_data()
        
        st.markdown("---")
        st.markdown("### Equipment Settings")
        
        # Equipment selection
        equipment_type = st.selectbox(
            "Equipment Type",
            ["Centrifugal Pump", "Compressor", "Turbine"],
            index=0
        )
        
        # Alert thresholds
        temp_threshold = st.slider("Temperature Alert (°C)", 50, 100, 80)
        pressure_threshold = st.slider("Pressure Alert (kPa)", 50, 200, 150)
        vib_threshold = st.slider("Vibration Alert (mm/s)", 0.1, 2.0, 0.8, 0.1)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Real-time Monitoring")
        display_equipment_status(st.session_state.data)
        
        st.markdown("### Sensor Data")
        plot_sensor_data(st.session_state.data)
    
    with tab2:
        st.header("Advanced Analytics")
        
        if SKLEARN_AVAILABLE:
            # Prepare data for anomaly detection
            X = st.session_state.data[['Temperature', 'Pressure', 'Vibration']].values
            
            # Train anomaly detector
            detector = AnomalyDetector()
            detector.fit(X)
            anomalies = detector.predict(X)
            
            # Add anomalies to data
            data_with_anomalies = st.session_state.data.copy()
            data_with_anomalies['Anomaly'] = anomalies
            
            # Plot anomalies
            fig = px.scatter(
                data_with_anomalies,
                x='Timestamp',
                y='Temperature',
                color='Anomaly',
                title='Anomaly Detection',
                color_discrete_map={True: '#FF5252', False: '#4CAF50'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Anomaly detection requires scikit-learn. Install with: pip install scikit-learn")
    
    with tab3:
        st.header("System Settings")
        
        st.markdown("### Data Management")
        if st.button("Export Data"):
            # Create a download link for the data
            csv = st.session_state.data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="equipment_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("### About")
        st.markdown("""
        **Neural Digital Twin Elite**  
        Version 1.0.0  
        
        A comprehensive industrial monitoring solution with AI-powered analytics.
        """)

if __name__ == "__main__":
    main()
