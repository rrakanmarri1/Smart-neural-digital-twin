import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Initialize session state for theme and recommendation history if not already done
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# تعيين نمط الصفحة #
st.set_page_config(
    page_title="Smart Neural Digital Twin - Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Configuration ---
def get_theme_colors(theme_mode):
    if theme_mode == "dark":
        return {
            "primary-color": "#0E1117",  # Dark background
            "secondary-color": "#3498DB", # Blue accent
            "text-color": "#FFFFFF",
            "chart-background": "#1E212B",
            "card-background": "#2E313B"
        }
    else:
        return {
            "primary-color": "#FFFFFF",  # Light background
            "secondary-color": "#2980B9", # Darker blue accent
            "text-color": "#000000",
            "chart-background": "#F0F2F6",
            "card-background": "#E0E2E6"
        }

theme_colors = get_theme_colors(st.session_state.theme)

# Apply CSS based on theme
st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: {theme_colors['primary-color']};
        color: {theme_colors['text-color']};
    }}
    .sidebar .sidebar-content {{
        background-color: {theme_colors['primary-color']};
    }}
    .stButton>button {{
        background-color: {theme_colors['secondary-color']};
        color: {theme_colors['text-color']};
    }}
    .stTextInput>div>div>input {{
        color: {theme_colors['text-color']};
    }}
    .stSelectbox>div>div {{
        color: {theme_colors['text-color']};
    }}
    .stMarkdown {{
        color: {theme_colors['text-color']};
    }}
    .stAlert {{
        background-color: {theme_colors['card-background']};
        color: {theme_colors['text-color']};
    }}
    .stProgress>div>div>div>div {{
        background-color: {theme_colors['secondary-color']};
    }}
    .css-1d3f8as {{
        background-color: {theme_colors['card-background']};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Smart Neural Digital Twin")

# Theme toggle
if st.sidebar.button(f"Switch to {'Dark' if st.session_state.theme == 'light' else 'Light'} Mode"):
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

menu_options = [
    "Dashboard",
    "Predictive Analytics",
    "Smart Recommendations",
    "About Project",
    "Settings"
]

chosen_menu = st.sidebar.radio("Navigation", menu_options)

# --- Data Simulation (for demonstration) ---
def generate_sensor_data(num_points=100):
    time_series = pd.date_range(end=datetime.now(), periods=num_points, freq='H')
    temperature = np.random.normal(30, 5, num_points)
    pressure = np.random.normal(100, 10, num_points)
    vibration = np.random.normal(0.5, 0.1, num_points)
    
    # Introduce some anomalies
    for _ in range(int(num_points * 0.05)):
        idx = random.randint(0, num_points - 1)
        temperature[idx] *= random.uniform(1.5, 2.0)
        pressure[idx] *= random.uniform(0.5, 1.5)
        vibration[idx] *= random.uniform(2.0, 3.0)
        
    df = pd.DataFrame({
        'Timestamp': time_series,
        'Temperature': temperature,
        'Pressure': pressure,
        'Vibration': vibration
    })
    return df

sensor_data = generate_sensor_data(200)

# --- Anomaly Detection (Placeholder) ---
def detect_anomalies(df):
    # In a real scenario, this would use a trained ML model (e.g., Isolation Forest)
    # For demonstration, we'll simulate anomalies based on simple thresholds
    anomalies = df[(df['Temperature'] > 40) | (df['Pressure'] < 80) | (df['Vibration'] > 0.8)]
    return anomalies

anomalies = detect_anomalies(sensor_data)

# --- Predictive Model (Placeholder) ---
def predict_future_data(df, hours_ahead=72):
    last_timestamp = df['Timestamp'].max()
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=hours_ahead, freq='H')
    
    # Simulate future data based on last known values with some noise
    last_temp = df['Temperature'].iloc[-1]
    last_press = df['Pressure'].iloc[-1]
    last_vib = df['Vibration'].iloc[-1]
    
    future_temp = [last_temp + random.uniform(-2, 2) for _ in range(hours_ahead)]
    future_press = [last_press + random.uniform(-5, 5) for _ in range(hours_ahead)]
    future_vib = [last_vib + random.uniform(-0.1, 0.1) for _ in range(hours_ahead)]
    
    # Simulate confidence intervals (simple range for demo)
    temp_upper = [t + random.uniform(1, 3) for t in future_temp]
    temp_lower = [t - random.uniform(1, 3) for t in future_temp]
    
    future_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Temperature': future_temp,
        'Pressure': future_press,
        'Vibration': future_vib,
        'Temperature_Upper': temp_upper,
        'Temperature_Lower': temp_lower
    })
    return future_df

future_predictions = predict_future_data(sensor_data, hours_ahead=72)

# --- Smart Recommendations Logic ---
def generate_recommendation(risk_level):
    recommendations = {
        "Low": [
            {"title": "Monitor System Performance", "details": "Continue routine monitoring of all sensor data.", "priority": "Low", "estimated_time": "1 hour", "effectiveness": "High"},
            {"title": "Review Log Files", "details": "Check system logs for any unusual entries.", "priority": "Low", "estimated_time": "30 mins", "effectiveness": "Medium"}
        ],
        "Medium": [
            {"title": "Perform Diagnostic Check", "details": "Run a full diagnostic scan on the affected components.", "priority": "Medium", "estimated_time": "2 hours", "effectiveness": "High"},
            {"title": "Adjust Sensor Thresholds", "details": "Slightly adjust anomaly detection thresholds to reduce false positives.", "priority": "Medium", "estimated_time": "1 hour", "effectiveness": "Medium"}
        ],
        "High": [
            {"title": "Immediate System Shutdown", "details": "Initiate emergency shutdown procedures to prevent further damage.", "priority": "Critical", "estimated_time": "10 mins", "effectiveness": "Very High"},
            {"title": "Dispatch Maintenance Team", "details": "Send a specialized maintenance team to inspect and repair the equipment.", "priority": "High", "estimated_time": "4 hours", "effectiveness": "Very High"},
            {"title": "Isolate Affected Area", "details": "Implement safety protocols to isolate the area with high risk.", "priority": "High", "estimated_time": "30 mins", "effectiveness": "High"}
        ]
    }
    return random.choice(recommendations.get(risk_level, recommendations["Low"]))

# --- Main App Logic ---
if chosen_menu == "Dashboard":
    st.header("Real-time Sensor Data Dashboard")

    # Current Status Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="css-1d3f8as">
            <h3>Temperature</h3>
            <p style="font-size: 24px;">{sensor_data['Temperature'].iloc[-1]:.2f} °C</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="css-1d3f8as">
            <h3>Pressure</h3>
            <p style="font-size: 24px;">{sensor_data['Pressure'].iloc[-1]:.2f} kPa</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="css-1d3f8as">
            <h3>Vibration</h3>
            <p style="font-size: 24px;">{sensor_data['Vibration'].iloc[-1]:.2f} mm/s</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Recent Sensor Readings")
    fig = px.line(sensor_data, x='Timestamp', y=['Temperature', 'Pressure', 'Vibration'],
                  title='Sensor Data Over Time')
    fig.update_layout(plot_bgcolor=theme_colors['chart-background'], paper_bgcolor=theme_colors['primary-color'], font_color=theme_colors['text-color'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Detection")
    if not anomalies.empty:
        st.warning(f"Detected {len(anomalies)} anomalies!")
        st.dataframe(anomalies)
        # Simulate risk level for recommendation
        risk_level = "High" if len(anomalies) > 5 else "Medium"
        st.session_state.current_risk_level = risk_level
        st.session_state.last_recommendation_time = datetime.now()
        st.toast(f"Risk level is {risk_level}! Check Smart Recommendations.")
    else:
        st.success("No anomalies detected. System is stable.")
        st.session_state.current_risk_level = "Low"

elif chosen_menu == "Predictive Analytics":
    st.header("Future Data Predictions (Next 72 Hours)")

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=sensor_data['Timestamp'], y=sensor_data['Temperature'], mode='lines', name='Actual Temperature', line=dict(color='blue')))
    fig_temp.add_trace(go.Scatter(x=future_predictions['Timestamp'], y=future_predictions['Temperature'], mode='lines', name='Predicted Temperature', line=dict(color='red', dash='dash')))
    fig_temp.add_trace(go.Scatter(x=future_predictions['Timestamp'], y=future_predictions['Temperature_Upper'], mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig_temp.add_trace(go.Scatter(x=future_predictions['Timestamp'], y=future_predictions['Temperature_Lower'], mode='lines', name='Lower Bound', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', showlegend=False))
    
    fig_temp.update_layout(
        title='Temperature Prediction',
        xaxis_title='Timestamp',
        yaxis_title='Temperature (°C)',
        plot_bgcolor=theme_colors['chart-background'],
        paper_bgcolor=theme_colors['primary-color'],
        font_color=theme_colors['text-color']
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader("Predicted Data Table")
    st.dataframe(future_predictions[['Timestamp', 'Temperature', 'Pressure', 'Vibration']])

elif chosen_menu == "Smart Recommendations":
    st.header("Smart Recommendations for Anomaly Management")

    current_risk = st.session_state.get('current_risk_level', 'Low')
    st.info(f"Current System Risk Level: **{current_risk}**")

    if st.button("Generate New Recommendation"):
        new_rec = generate_recommendation(current_risk)
        st.session_state.recommendation_history.append({"timestamp": datetime.now(), "recommendation": new_rec})
        st.success("New recommendation generated!")

    if st.session_state.recommendation_history:
        st.subheader("Latest Recommendation")
        latest_rec = st.session_state.recommendation_history[-1]['recommendation']
        st.markdown(f"""
        <div class="css-1d3f8as">
            <h4>{latest_rec['title']}</h4>
            <p><b>Details:</b> {latest_rec['details']}</p>
            <p><b>Priority:</b> <span style="color: {'red' if latest_rec['priority'] == 'Critical' else 'orange' if latest_rec['priority'] == 'High' else 'green'}">{latest_rec['priority']}</span></p>
            <p><b>Estimated Time:</b> {latest_rec['estimated_time']}</p>
            <p><b>Effectiveness:</b> {latest_rec['effectiveness']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Recommendation History")
        for i, rec_entry in enumerate(reversed(st.session_state.recommendation_history[:-1])):
            st.markdown(f"""
            <div class="css-1d3f8as" style="opacity: {1 - (i*0.1)}; margin-top: 10px;">
                <p><b>Timestamp:</b> {rec_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h4>{rec_entry['recommendation']['title']}</h4>
                <p><b>Priority:</b> {rec_entry['recommendation']['priority']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recommendations generated yet.")

elif chosen_menu == "About Project":
    st.header("About Smart Neural Digital Twin")
    st.write("""
    This project develops a Smart Neural Digital Twin for oilfield safety, leveraging real-time sensor data, advanced machine learning, and predictive analytics to prevent disasters.
    
    **Key Features:**
    - Real-time sensor data monitoring
    - Anomaly detection using Isolation Forest
    - Predictive modeling for future conditions
    - Smart recommendations for proactive intervention
    - Interactive and user-friendly Streamlit interface
    """)
    st.subheader("Our Vision")
    st.write("To revolutionize industrial safety by transforming raw data into actionable insights, ensuring a safer and more efficient operational environment.")

elif chosen_menu == "Settings":
    st.header("Application Settings")
    st.subheader("Theme Settings")
    current_theme = st.session_state.theme
    st.write(f"Current Theme: **{current_theme.capitalize()}**")
    if st.button("Toggle Theme"):
        st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
        st.rerun()

    st.subheader("Data Simulation Settings")
    num_data_points = st.slider("Number of historical data points", 50, 500, 200)
    if st.button("Regenerate Data"):
        sensor_data = generate_sensor_data(num_data_points)
        st.success("Data regenerated!")
        st.rerun()

    st.subheader("Reset Recommendations History")
    if st.button("Clear History"):
        st.session_state.recommendation_history = []
        st.success("Recommendation history cleared!")
        st.rerun()

