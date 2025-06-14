import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import base64
import sys
from pathlib import Path

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Neural Digital Twin Elite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Enhanced Dark Theme & Custom CSS ==========
st.markdown("""
<style>
:root {
    --primary-color: #6e48aa;
    --secondary-color: #9d50bb;
    --dark-bg: #0f0c29;
    --darker-bg: #06051a;
    --metric-bg: rgba(255, 255, 255, 0.1);
}

body {
    background: linear-gradient(135deg, var(--dark-bg), #302b63, #24243e) !important;
    color: white !important;
}

.stApp {
    background: transparent !important;
}

.stSidebar {
    background: var(--darker-bg) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stButton>button {
    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

.stMetric {
    background: var(--metric-bg) !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
}

.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
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
    100% { opacity: 1;
}

/* Enhanced tooltips */
.tooltip-icon {
    display: inline-block;
    width: 16px;
    height: 16px;
    background-color: var(--primary-color);
    color: white;
    border-radius: 50%;
    text-align: center;
    font-size: 12px;
    line-height: 16px;
    cursor: help;
    margin-left: 5px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .stMetric {
        padding: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ========== Project Header with Logo ==========
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://via.placeholder.com/100x100.png?text=NDTE", width=100)
with col2:
    st.title("Neural Digital Twin Elite 🤖")
    st.markdown("""
    *Advanced industrial equipment monitoring with AI-powered analytics and predictive maintenance*
    """)

# ========== Enhanced Equipment Simulator ==========
class AdvancedEquipmentSimulator:
    """Enhanced simulator with more realistic degradation patterns"""
    
    def __init__(self, equipment_id: str = "Pump-001"):
        self.equipment_id = equipment_id
        self.data = None
        self.degradation_rate = {
            'temp': random.uniform(0.01, 0.03),
            'pressure': random.uniform(0.02, 0.04),
            'vibration': random.uniform(0.005, 0.015)
        }
    
    def generate_data(self, hours: int = 500) -> pd.DataFrame:
        """Generate more realistic sensor data with progressive degradation"""
        date_rng = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        time = np.arange(len(date_rng))
        
        # Progressive degradation trends
        temp_trend = 25 + self.degradation_rate['temp'] * time
        pressure_trend = 100 - self.degradation_rate['pressure'] * time
        vibration_trend = 5 + self.degradation_rate['vibration'] * time
        
        # Enhanced seasonality and noise patterns
        daily = 5 * np.sin(2 * np.pi * time / 24)
        weekly = 2 * np.sin(2 * np.pi * time / (24*7))
        noise = np.random.normal(0, 1.5, len(date_rng))
        
        # Combine components
        temperature = np.clip(temp_trend + daily + weekly + noise, 0, 120)
        pressure = np.clip(pressure_trend + daily*0.7 + weekly*0.5 + noise*0.8, 0, 200)
        vibration = np.clip(vibration_trend + daily*0.4 + weekly*0.3 + noise*0.6, 0, 30)
        
        df = pd.DataFrame({
            'Timestamp': date_rng,
            'Temperature': temperature,
            'Pressure': pressure,
            'Vibration': vibration,
            'Equipment_ID': self.equipment_id,
            'Status': 'Normal'
        })
        
        self._add_enhanced_anomalies(df)
        self.data = df
        return df
    
    def _add_enhanced_anomalies(self, df: pd.DataFrame) -> None:
        """Add more realistic anomalies with different severity levels"""
        # Major anomalies
        for _ in range(5):
            idx = random.randint(0, len(df)-24)  # Avoid last 24 hours
            df.at[idx, 'Temperature'] += random.uniform(30, 50)
            df.at[idx+1:idx+6, 'Temperature'] += random.uniform(10, 20)  # Ripple effect
        
        # Medium anomalies
        for _ in range(15):
            idx = random.randint(0, len(df)-1)
            anomaly_type = random.choice(['temp', 'pressure', 'vibration'])
            
            if anomaly_type == 'temp':
                df.at[idx, 'Temperature'] += random.uniform(15, 30)
            elif anomaly_type == 'pressure':
                df.at[idx, 'Pressure'] -= random.uniform(25, 45)
            else:
                df.at[idx, 'Vibration'] += random.uniform(5, 15)
        
        # Minor anomalies
        for _ in range(30):
            idx = random.randint(0, len(df)-1)
            anomaly_type = random.choice(['temp', 'pressure', 'vibration'])
            
            if anomaly_type == 'temp':
                df.at[idx, 'Temperature'] += random.uniform(5, 10)
            elif anomaly_type == 'pressure':
                df.at[idx, 'Pressure'] -= random.uniform(10, 20)
            else:
                df.at[idx, 'Vibration'] += random.uniform(2, 5)

# ========== Initialize Session State ==========
if 'data' not in st.session_state:
    st.session_state.simulator = AdvancedEquipmentSimulator()
    st.session_state.data = st.session_state.simulator.generate_data()
    st.session_state.last_updated = datetime.now()

# ========== Enhanced Sidebar Controls ==========
with st.sidebar:
    st.header("⚙️ System Controls")
    
    if st.button("🔄 Generate New Dataset", help="Generate a new set of simulated sensor data"):
        st.session_state.data = st.session_state.simulator.generate_data()
        st.session_state.last_updated = datetime.now()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏭 Equipment Configuration")
    
    equipment_type = st.selectbox(
        "Equipment Type",
        ["Centrifugal Pump", "Compressor", "Turbine", "Heat Exchanger"],
        index=0,
        help="Select the type of equipment being monitored"
    )
    
    st.markdown("### ⚠️ Alert Thresholds")
    temp_threshold = st.slider(
        "Max Temperature (°C)", 
        50, 120, 80,
        help="Temperature threshold for critical alerts"
    )
    pressure_threshold = st.slider(
        "Max Pressure (kPa)", 
        50, 250, 150,
        help="Pressure threshold for critical alerts"
    )
    vib_threshold = st.slider(
        "Max Vibration (mm/s)", 
        0.1, 3.0, 0.8, 0.1,
        help="Vibration threshold for critical alerts"
    )
    
    st.markdown("---")
    st.markdown("### 🔮 Predictive Maintenance")
    prediction_horizon = st.slider(
        "Forecast Horizon (days)", 
        1, 30, 7,
        help="Number of days to predict future equipment health"
    )
    
    if st.button("📊 Run Predictive Analysis", type="primary"):
        st.session_state.show_predictive = True
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Last Updated:**\n{st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

# ========== Main Dashboard Tabs ==========
tab1, tab2, tab3 = st.tabs(["📊 Real-time Dashboard", "📈 Predictive Analytics", "ℹ️ Project Information"])

with tab1:
    st.header("Real-time Equipment Monitoring")
    
    # Enhanced Equipment Status Metrics
    latest_data = st.session_state.data.iloc[-1]
    prev_data = st.session_state.data.iloc[-2] if len(st.session_state.data) > 1 else latest_data
    
    temp_status = latest_data['Temperature'] > temp_threshold
    pressure_status = latest_data['Pressure'] > pressure_threshold or latest_data['Pressure'] < (pressure_threshold * 0.5)
    vib_status = latest_data['Vibration'] > vib_threshold
    
    if temp_status or pressure_status or vib_status:
        status = "<span class='status-critical'>CRITICAL</span>"
        status_icon = "🔴"
    elif latest_data['Temperature'] > (temp_threshold * 0.9) or latest_data['Pressure'] > (pressure_threshold * 0.9) or latest_data['Vibration'] > (vib_threshold * 0.9):
        status = "<span class='status-warning'>WARNING</span>"
        status_icon = "🟡"
    else:
        status = "<span class='status-normal'>NORMAL</span>"
        status_icon = "🟢"
    
    # Enhanced metrics layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_temp = latest_data['Temperature'] - prev_data['Temperature']
        st.metric(
            "Temperature", 
            f"{latest_data['Temperature']:.1f}°C", 
            f"{delta_temp:+.1f}°C", 
            delta_color="inverse",
            help="Current temperature with change from previous reading"
        )
    
    with col2:
        delta_pressure = latest_data['Pressure'] - prev_data['Pressure']
        st.metric(
            "Pressure", 
            f"{latest_data['Pressure']:.1f} kPa", 
            f"{delta_pressure:+.1f} kPa", 
            delta_color="inverse",
            help="Current pressure with change from previous reading"
        )
    
    with col3:
        delta_vib = latest_data['Vibration'] - prev_data['Vibration']
        st.metric(
            "Vibration", 
            f"{latest_data['Vibration']:.2f} mm/s", 
            f"{delta_vib:+.2f} mm/s", 
            delta_color="inverse",
            help="Current vibration levels with change from previous reading"
        )
    
    with col4:
        st.markdown(f"### System Status: {status_icon} {status}", unsafe_allow_html=True)
        health_score = 100 - (latest_data['Temperature'] / temp_threshold * 30 + 
                            (1 - latest_data['Pressure'] / pressure_threshold) * 30 + 
                            latest_data['Vibration'] / vib_threshold * 40)
        health_score = max(0, min(100, health_score))
        st.progress(health_score / 100)
        st.caption(f"Equipment Health Score: {health_score:.0f}/100")
    
    # Enhanced Sensor Data Visualization
    st.markdown("### 📈 Sensor Data Trends")
    fig = go.Figure()
    
    # Add traces with enhanced hover information
    fig.add_trace(go.Scatter(
        x=st.session_state.data['Timestamp'],
        y=st.session_state.data['Temperature'],
        name='Temperature (°C)',
        line=dict(color='#FF6B6B', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>%{y:.1f}°C<extra></extra>',
        mode='lines+markers',
        marker=dict(size=4, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=st.session_state.data['Timestamp'],
        y=st.session_state.data['Pressure'],
        name='Pressure (kPa)',
        yaxis='y2',
        line=dict(color='#4ECDC4', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>%{y:.1f} kPa<extra></extra>',
        mode='lines+markers',
        marker=dict(size=4, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=st.session_state.data['Timestamp'],
        y=st.session_state.data['Vibration'] * 20,
        name='Vibration (x20)',
        yaxis='y3',
        line=dict(color='#45B7D1', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>%{y:.2f} mm/s<extra></extra>',
        mode='lines+markers',
        marker=dict(size=4, opacity=0.5)
    ))
    
    # Enhanced threshold lines with annotations
    fig.add_hline(
        y=temp_threshold, 
        line_dash="dot", 
        line_color="red", 
        opacity=0.7,
        annotation_text=f"Temp Threshold ({temp_threshold}°C)", 
        annotation_position="top right",
        annotation_font_color="red"
    )
    
    fig.add_hline(
        y=pressure_threshold, 
        line_dash="dot", 
        line_color="red", 
        opacity=0.7,
        annotation_text=f"Pressure Threshold ({pressure_threshold}kPa)", 
        annotation_position="top right",
        row=1, col=1, yref="y2",
        annotation_font_color="red"
    )
    
    # Enhanced layout
    fig.update_layout(
        title='<b>Real-time Sensor Data with Threshold Alerts</b>',
        xaxis=dict(
            domain=[0.1, 0.9],
            title='Timeline',
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='<b>Temperature (°C)</b>', 
            titlefont=dict(color='#FF6B6B'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis2=dict(
            title='<b>Pressure (kPa)</b>',
            titlefont=dict(color='#4ECDC4'),
            anchor='x',
            overlaying='y',
            side='right',
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis3=dict(
            title='<b>Vibration (mm/s)</b>',
            titlefont=dict(color='#45B7D1'),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.95,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=550,
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Smart Recommendations
    st.markdown("### 🧠 AI-Powered Recommendations")
    
    if temp_status:
        with st.expander("🚨 Critical Temperature Alert", expanded=True):
            st.markdown("""
            **Recommended Actions:**
            1. 🔧 **Immediate Inspection**: Check cooling system and heat exchangers
            2. ⚡ **Reduce Load**: Decrease operational load by 20-30%
            3. 🧯 **Emergency Protocols**: Activate backup cooling systems
            4. 📞 **Contact Maintenance**: Notify equipment specialist immediately
            
            **Potential Causes:**
            - Cooling system failure
            - Excessive operational load
            - Heat exchanger fouling
            - Sensor malfunction
            """)
    
    if pressure_status:
        with st.expander("🚨 Critical Pressure Alert", expanded=True):
            st.markdown("""
            **Recommended Actions:**
            1. 🔍 **Leak Detection**: Perform full system leak check
            2. 🛑 **Safety Shutdown**: Initiate controlled shutdown procedure
            3. 🛠️ **Valve Inspection**: Check all relief and control valves
            4. 📞 **Contact Maintenance**: Notify pressure systems engineer
            
            **Potential Causes:**
            - Pipe or vessel rupture
            - Pump failure
            - Control system malfunction
            - Blocked flow path
            """)
    
    if vib_status:
        with st.expander("🚨 Critical Vibration Alert", expanded=True):
            st.markdown("""
            **Recommended Actions:**
            1. 🛑 **Immediate Shutdown**: Prevent catastrophic failure
            2. 🔩 **Mechanical Inspection**: Check all bearings and couplings
            3. ⚖️ **Re-balance Rotating Elements**: Schedule dynamic balancing
            4. 📞 **Contact Vibration Specialist**: Request detailed analysis
            
            **Potential Causes:**
            - Bearing failure
            - Misalignment
            - Imbalance
            - Mechanical looseness
            - Resonance issues
            """)
    
    if not (temp_status or pressure_status or vib_status):
        with st.expander("✅ System Normal - Maintenance Plan", expanded=False):
            st.markdown("""
            **Preventive Maintenance Schedule:**
            - Next routine inspection: **2 weeks**
            - Predictive maintenance window: **3-4 weeks**
            - Full system overhaul: **6 months**
            
            **Recommended Activities:**
            1. Monthly bearing lubrication
            2. Quarterly alignment checks
            3. Annual full system calibration
            """)

with tab2:
    st.header("Predictive Analytics & Anomaly Detection")
    
    # Enhanced Anomaly Detection Section
    st.markdown("### 🔍 Advanced Anomaly Detection")
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        X = st.session_state.data[['Temperature', 'Pressure', 'Vibration']].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = IsolationForest(
            contamination=0.07, 
            random_state=42,
            n_estimators=150
        )
        anomalies = model.fit_predict(X_scaled)
        
        data_with_anomalies = st.session_state.data.copy()
        data_with_anomalies['Anomaly'] = anomalies == -1
        data_with_anomalies['Anomaly_Score'] = model.decision_function(X_scaled)
        
        # Enhanced anomaly visualization
        fig = px.scatter(
            data_with_anomalies,
            x='Timestamp',
            y='Temperature',
            color='Anomaly',
            size=np.abs(data_with_anomalies['Anomaly_Score']),
            title='<b>Detected Anomalies with Severity Scores</b>',
            color_discrete_map={True: '#FF5252', False: '#4CAF50'},
            hover_data=['Pressure', 'Vibration', 'Anomaly_Score'],
            size_max=15
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            legend_title_text='Anomaly Detected'
        )
        
        fig.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly statistics
        anomaly_count = sum(anomalies == -1)
        avg_severity = data_with_anomalies.loc[data_with_anomalies['Anomaly'], 'Anomaly_Score'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Anomalies Detected", anomaly_count)
        with col2:
            st.metric("Average Severity Score", f"{avg_severity:.2f}" if not np.isnan(avg_severity) else "N/A")
        
        # Anomaly timeline
        st.markdown("#### 🕰️ Anomaly Timeline")
        anomaly_dates = data_with_anomalies[data_with_anomalies['Anomaly']]['Timestamp']
        if not anomaly_dates.empty:
            st.write("Anomalies detected at:")
            st.write(anomaly_dates.dt.strftime('%Y-%m-%d %H:%M').tolist())
        else:
            st.info("No significant anomalies detected in historical data")
        
    except ImportError:
        st.error("""
        Anomaly detection requires scikit-learn. Install with:
        ```bash
        pip install scikit-learn
        ```
        """)
    
    # Enhanced Predictive Maintenance Forecast
    st.markdown("### 🔮 Predictive Maintenance Forecast")
    
    if st.session_state.get('show_predictive', False):
        try:
            # Simulate more realistic predictive model results
            future_dates = pd.date_range(
                start=datetime.now(),
                periods=prediction_horizon,
                freq='D'
            )
            
            # Enhanced degradation modeling
            temp_coef = st.session_state.simulator.degradation_rate['temp'] * 24 * 1.5
            pressure_coef = st.session_state.simulator.degradation_rate['pressure'] * 24 * 1.2
            vib_coef = st.session_state.simulator.degradation_rate['vibration'] * 24 * 2
            
            temp_trend = latest_data['Temperature'] * (1 + temp_coef * np.arange(prediction_horizon))
            pressure_trend = latest_data['Pressure'] * (1 - pressure_coef * np.arange(prediction_horizon))
            vibration_trend = latest_data['Vibration'] * (1 + vib_coef * np.arange(prediction_horizon))
            
            # Enhanced noise and variability
            temp_trend += np.random.normal(0, 2 * (1 + np.arange(prediction_horizon)/10), prediction_horizon)
            pressure_trend += np.random.normal(0, 3 * (1 + np.arange(prediction_horizon)/8), prediction_horizon)
            vibration_trend += np.random.normal(0, 0.1 * (1 + np.arange(prediction_horizon)/5), prediction_horizon)
            
            # Enhanced failure risk calculation
            time_factor = np.arange(prediction_horizon) ** 1.8
            temp_factor = np.clip((temp_trend - temp_threshold) / temp_threshold, 0, 2)
            pressure_factor = np.clip((pressure_threshold - pressure_trend) / pressure_threshold, 0, 2)
            vib_factor = np.clip((vibration_trend - vib_threshold) / vib_threshold, 0, 2)
            
            failure_risk = np.clip(0.05 * time_factor * (0.4*temp_factor + 0.3*pressure_factor + 0.3*vib_factor), 0, 1)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Temperature': temp_trend,
                'Pressure': pressure_trend,
                'Vibration': vibration_trend,
                'Failure_Risk': failure_risk,
                'Maintenance_Urgency': np.where(failure_risk > 0.7, "Critical",
                                          np.where(failure_risk > 0.4, "High",
                                                 np.where(failure_risk > 0.2, "Medium", "Low")))
            })
            
            # Enhanced forecast visualization
            fig = go.Figure()
            
            # Temperature forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Temperature'],
                name='Temperature Forecast',
                line=dict(color='#FF6B6B', width=3),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.1f}°C<extra></extra>'
            ))
            
            # Pressure forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Pressure'],
                name='Pressure Forecast',
                yaxis='y2',
                line=dict(color='#4ECDC4', width=3),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.1f} kPa<extra></extra>'
            ))
            
            # Failure risk
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Failure_Risk'] * 100,
                name='Failure Risk %',
                yaxis='y3',
                line=dict(color='#FF5252', width=3, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(255, 82, 82, 0.2)',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.1f}% risk<extra></extra>'
            ))
            
            # Threshold lines
            fig.add_hline(
                y=temp_threshold, 
                line_dash="dash", 
                line_color="red", 
                opacity=0.7,
                annotation_text=f"Temp Threshold ({temp_threshold}°C)", 
                annotation_position="top right",
                annotation_font_color="red"
            )
            
            fig.add_hline(
                y=pressure_threshold, 
                line_dash="dash", 
                line_color="red", 
                opacity=0.7,
                annotation_text=f"Pressure Threshold ({pressure_threshold}kPa)", 
                annotation_position="top right",
                row=1, col=1, yref="y2",
                annotation_font_color="red"
            )
            
            # Enhanced layout
            fig.update_layout(
                title='<b>Equipment Health Forecast with Failure Risk Prediction</b>',
                xaxis=dict(title='<b>Date</b>', gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(
                    title='<b>Temperature (°C)</b>', 
                    titlefont=dict(color='#FF6B6B'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis2=dict(
                    title='<b>Pressure (kPa)</b>',
                    titlefont=dict(color='#4ECDC4'),
                    anchor='x',
                    overlaying='y',
                    side='right',
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis3=dict(
                    title='<b>Failure Risk %</b>',
                    titlefont=dict(color='#FF5252'),
                    anchor='free',
                    overlaying='y',
                    side='right',
                    position=0.95,
                    range=[0, 100],
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                plot_bgcolor='rgba(0,0,0,0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Maintenance recommendation
            max_risk_day = forecast_df.loc[forecast_df['Failure_Risk'].idxmax()]
            risk_category = max_risk_day['Maintenance_Urgency']
            
            if risk_category == "Critical":
                alert_icon = "🚨"
                alert_color = "red"
            elif risk_category == "High":
                alert_icon = "⚠️"
                alert_color = "orange"
            else:
                alert_icon = "ℹ️"
                alert_color = "blue"
            
            with st.expander(f"{alert_icon} Maintenance Recommendation - {risk_category} Risk", expanded=True):
                st.markdown(f"""
                **Optimal Maintenance Window:**  
                📅 **{max_risk_day['Date'].strftime('%Y-%m-%d')} ± 2 days**  
                🔴 **Peak Risk:** {max_risk_day['Failure_Risk']*100:.1f}% probability of failure
                
                **Predicted Issues:**
                - {"🌡️ Temperature exceeding threshold" if max_risk_day['Temperature'] > temp_threshold else ""}  
                - {"💨 Pressure drop" if max_risk_day['Pressure'] < pressure_threshold*0.6 else ""}  
                - {"📳 Vibration increase" if max_risk_day['Vibration'] > vib_threshold*1.2 else ""}
                
                **Recommended Actions:**
                1. Schedule maintenance during the identified window
                2. Prepare replacement parts based on predicted failures
                3. Notify maintenance team in advance
                4. Plan for possible production downtime
                """, unsafe_allow_html=True)
                
                # Gantt chart for maintenance planning
                st.markdown("#### 🗓️ Maintenance Planning Timeline")
                maintenance_df = pd.DataFrame({
                    "Task": ["Preparation", "Inspection", "Parts Replacement", "Testing", "Restart"],
                    "Start": [
                        (max_risk_day['Date'] - timedelta(days=2)),
                        (max_risk_day['Date'] - timedelta(days=1)),
                        max_risk_day['Date'],
                        (max_risk_day['Date'] + timedelta(days=1)),
                        (max_risk_day['Date'] + timedelta(days=2))
                    ],
                    "Finish": [
                        (max_risk_day['Date'] - timedelta(days=1)),
                        max_risk_day['Date'],
                        (max_risk_day['Date'] + timedelta(days=1)),
                        (max_risk_day['Date'] + timedelta(days=2)),
                        (max_risk_day['Date'] + timedelta(days=3))
                    ]
                })
                
                fig = px.timeline(
                    maintenance_df, 
                    x_start="Start", 
                    x_end="Finish", 
                    y="Task",
                    title="Recommended Maintenance Schedule"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.3)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in predictive analysis: {str(e)}")
    else:
        st.info("Click 'Run Predictive Analysis' in the sidebar to generate forecast")

with tab3:
    st.header("About Neural Digital Twin Elite")
    
    # Project Overview with Cards
    st.markdown("## Project Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("""
            ### 🎯 Mission
            Transform industrial asset management through AI-powered digital twins that:
            - Predict failures before they occur
            - Optimize maintenance schedules
            - Enhance operational efficiency
            """)
    
    with col2:
        with st.container(border=True):
            st.markdown("""
            ### 🌟 Vision
            Become the industry standard for predictive maintenance in:
            - Oil & Gas
            - Power Generation
            - Manufacturing
            - Water Treatment
            """)
    
    # Key Features with Expanders
    st.markdown("## Key Features")
    
    with st.expander("📊 Real-time Monitoring Dashboard", expanded=False):
        st.markdown("""
        - Live visualization of critical parameters
        - Customizable alert thresholds
        - Equipment health scoring
        - Historical trend analysis
        """)
    
    with st.expander("🔍 Advanced Anomaly Detection", expanded=False):
        st.markdown("""
        - Machine learning-powered anomaly identification
        - Multi-variable correlation analysis
        - Automated root cause suggestions
        - Historical anomaly timeline
        """)
    
    with st.expander("🔮 Predictive Maintenance", expanded=False):
        st.markdown("""
        - Equipment degradation forecasting
        - Failure probability estimation
        - Optimal maintenance window calculation
        - Resource planning tools
        """)
    
    with st.expander("🧠 Smart Recommendations", expanded=False):
        st.markdown("""
        - Actionable maintenance suggestions
        - Emergency protocols
        - Spare parts recommendations
        - Operational adjustments
        """)
    
    # Technical Specifications
    st.markdown("## Technical Specifications")
    
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown("""
        ### 🛠️ System Requirements
        - Python 3.10+
        - Streamlit 1.30+
        - Plotly 5.15+
        - Scikit-learn 1.3+
        - Pandas 2.0+
        """)
    
    with spec_col2:
        st.markdown("""
        ### 📊 Data Integration
        - Real-time API connectivity
        - Historical data import
        - Custom data source configuration
        - Automated data validation
        """)
    
    st.markdown
