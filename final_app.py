import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# --- Load and prepare data ---
@st.cache_data(ttl=600)
def load_history():
    df = pd.read_csv(
        "sensor_data_simulated.csv",
        parse_dates=["Timestamp"],
        dayfirst=False
    )
    return df.rename(columns={
        "Timestamp": "timestamp",
        "Temperature (°C)": "temp",
        "Pressure (psi)": "pressure",
        "Vibration (g)": "vibration",
        "Methane (CH₄,, ppm)": "gas",
        "H₂S (ppm)": "h2s"
    })

history = load_history()

# --- Page config ---
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

# --- Sidebar menu ---
st.markdown("<h1 style='text-align:center;'>🧠 Smart Neural Digital Twin</h1>", unsafe_allow_html=True)
page = st.radio(
    "Select Page",
    ["Dashboard", "Simulation", "Predictive Analysis", "Smart Solutions", "About"],
    horizontal=True
)

# --- Dashboard ---
if page == "Dashboard":
    st.header("Dashboard")
    latest = history.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature (°C)", f"{latest.temp:.2f}")
    col2.metric("Pressure (psi)", f"{latest.pressure:.2f}")
    col3.metric("Vibration (g)", f"{latest.vibration:.2f}")
    col4.metric("Gas (ppm)", f"{latest.gas:.2f}")
    st.markdown("### Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.timestamp, y=history.temp, name="Temp"))
    fig.add_trace(go.Scatter(x=history.timestamp, y=history.pressure, name="Pressure"))
    fig.add_trace(go.Scatter(x=history.timestamp, y=history.vibration, name="Vibration"))
    fig.add_trace(go.Scatter(x=history.timestamp, y=history.gas, name="Gas"))
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Heatmap (temp vs time of day)")
    history["hour"] = history.timestamp.dt.hour
    heat = history.pivot_table(index="hour", columns=history.timestamp.dt.day, values="temp", aggfunc="mean")
    fig2 = px.imshow(heat, labels=dict(x="Day", y="Hour", color="Temp"))
    st.plotly_chart(fig2, use_container_width=True)

# --- Simulation ---
elif page == "Simulation":
    st.header("Simulation")
    sd = latest.to_dict()
    sd['temp'] = st.slider("Temperature (°C)", 20.0, 50.0, float(sd['temp']), 0.1)
    sd['pressure'] = st.slider("Pressure (psi)", 60.0, 120.0, float(sd['pressure']), 0.1)
    sd['vibration'] = st.slider("Vibration (g)", 0.0, 1.5, float(sd['vibration']), 0.01)
    sd['gas'] = st.slider("Gas (ppm)", 0.0, 10.0, float(sd['gas']), 0.01)
    st.write(pd.DataFrame([sd]).T.rename(columns={0:"Value"}))

# --- Predictive Analysis ---
elif page == "Predictive Analysis":
    st.header("Predictive Analysis (Next 72h)")
    last_time = history.timestamp.max()
    future = pd.DataFrame({
        "timestamp": [last_time + pd.Timedelta(hours=i) for i in range(1,73)],
        "temp": np.linspace(latest.temp, latest.temp + random.uniform(-2,2), 72),
        "pressure": np.linspace(latest.pressure, latest.pressure + random.uniform(-5,5), 72),
        "vibration": np.linspace(latest.vibration, latest.vibration + random.uniform(-0.2,0.2), 72),
        "gas": np.linspace(latest.gas, latest.gas + random.uniform(-1,1), 72)
    })
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=history.timestamp.tail(24), y=history.temp.tail(24), name="Actual Temp"))
    fig3.add_trace(go.Scatter(x=future.timestamp, y=future.temp, name="Predicted Temp", line=dict(dash="dash")))
    fig3.update_layout(title="Temperature Forecast", height=300)
    st.plotly_chart(fig3, use_container_width=True)

# --- Smart Solutions ---
elif page == "Smart Solutions":
    st.header("Smart Solutions")
    def recommend(sd):
        if sd['temp']>40: return {"Solution":"Activate cooling","Duration":"10m","Priority":"Critical"}
        if sd['pressure']<80: return {"Solution":"Inspect valves","Duration":"30m","Priority":"High"}
        return {"Solution":"Routine check","Duration":"1h","Priority":"Low"}
    sol = recommend(latest)
    st.table(pd.DataFrame([sol]).T.rename(columns={0:"Details"}))

# --- About ---
else:
    st.header("About")
    st.markdown("""
**Disasters don't wait... and neither do we. Predict. Prevent. Protect.**

**Vision:** Revolutionize industrial safety by turning raw data into actionable insights.

**Features:**
- Real-time Monitoring  
- Anomaly Detection  
- Predictive Analytics  
- Smart Recommendations  

**Team:**  
Rakan Almarri | rakan.almarri.2@aramco.com | 0532559664  
Abdulrahman Alzhrani | abdulrahman.alzhrani.1@aramco.com | 0549202574  
""")
