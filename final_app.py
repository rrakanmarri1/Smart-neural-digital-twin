import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import random

# --- Load and prepare data ---
@st.cache_data(ttl=600)
def load_history():
    if os.path.exists("sensor_data_simulated.csv"):
        df = pd.read_csv(
            "sensor_data_simulated.csv",
            parse_dates=["Time"],
            dayfirst=False
        )
        df = df.rename(columns={
            "Time": "timestamp",
            "Temperature (°C)": "temp",
            "Pressure (psi)":     "pressure",
            "Vibration (g)":      "vibration",
            "Methane (CH₄ ppm)":  "gas",
            "H₂S (ppm)":          "h2s"
        })
        return df.dropna(subset=["timestamp"])
    else:
        return pd.DataFrame(columns=["timestamp","temp","pressure","vibration","gas","h2s"])

history = load_history()

# --- Page config ---
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

# --- Header & menu ---
st.markdown("<h1 style='text-align:center;'>🧠 Smart Neural Digital Twin</h1>", unsafe_allow_html=True)
page = st.radio(
    "Select Page",
    ["Dashboard", "Simulation", "Predictive Analysis", "Smart Solutions", "About"],
    horizontal=True
)

# --- Dashboard ---
if page == "Dashboard":
    st.header("Dashboard")
    if history.empty:
        st.info("No data available. Please add sensor_data_simulated.csv")
    else:
        latest = history.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature (°C)", f"{latest.temp:.2f}")
        c2.metric("Pressure (psi)", f"{latest.pressure:.2f}")
        c3.metric("Vibration (g)", f"{latest.vibration:.2f}")
        c4.metric("Gas (ppm)", f"{latest.gas:.2f}")
        st.markdown("### Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.temp, name="Temp"))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.pressure, name="Pressure"))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.vibration, name="Vibration"))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.gas, name="Gas"))
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Heatmap (hour vs day)")
        history["hour"] = history.timestamp.dt.hour
        history["day"]  = history.timestamp.dt.day
        heat = history.pivot_table(index="hour", columns="day", values="temp", aggfunc="mean")
        fig2 = px.imshow(heat, labels=dict(x="Day", y="Hour", color="Temp"))
        st.plotly_chart(fig2, use_container_width=True)

# --- Simulation ---
elif page == "Simulation":
    st.header("Simulation")
    if history.empty:
        st.info("No baseline data available.")
    else:
        current = history.iloc[-1].to_dict()
        current["temp"]      = st.slider("Temperature (°C)",     0.0, 100.0, float(current["temp"]),      0.1)
        current["pressure"]  = st.slider("Pressure (psi)",        0.0, 200.0, float(current["pressure"]),  0.1)
        current["vibration"] = st.slider("Vibration (g)",         0.0,   5.0, float(current["vibration"]), 0.01)
        current["gas"]       = st.slider("Methane (CH₄ ppm)",     0.0,  20.0, float(current["gas"]),       0.1)
        current["h2s"]       = st.slider("H₂S (ppm)",             0.0,  10.0, float(current["h2s"]),       0.1)
        df_sim = pd.DataFrame([current]).T.rename(columns={0:"Value"})
        st.table(df_sim)

# --- Predictive Analysis ---
elif page == "Predictive Analysis":
    st.header("Predictive Analysis (Next 72h)")
    if history.empty:
        st.info("No data available.")
    else:
        last = history.timestamp.max()
        future = pd.DataFrame({
            "timestamp": [last + timedelta(hours=i) for i in range(1, 73)],
            "temp":      np.linspace(history.temp.iloc[-1],      history.temp.iloc[-1] + random.uniform(-2,2),    72),
            "pressure":  np.linspace(history.pressure.iloc[-1],  history.pressure.iloc[-1] + random.uniform(-10,10),72),
            "vibration": np.linspace(history.vibration.iloc[-1], history.vibration.iloc[-1] + random.uniform(-0.5,0.5),72),
            "gas":       np.linspace(history.gas.iloc[-1],       history.gas.iloc[-1] + random.uniform(-2,2),       72)
        })
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=history.timestamp.tail(24), y=history.temp.tail(24), name="Actual Temp"))
        fig3.add_trace(go.Scatter(x=future.timestamp,             y=future.temp,             name="Predicted Temp", line=dict(dash="dash")))
        fig3.update_layout(title="Temperature Forecast", height=300)
        st.plotly_chart(fig3, use_container_width=True)

# --- Smart Solutions ---
elif page == "Smart Solutions":
    st.header("Smart Solutions")
    if history.empty:
        st.info("No data to analyze.")
    else:
        def recommend(row):
            if row.temp > 40:           return {"Solution":"Activate cooling","Duration":"10m","Priority":"Critical"}
            if row.pressure < 80:       return {"Solution":"Inspect valves","Duration":"30m","Priority":"High"}
            if row.vibration > 1.0:     return {"Solution":"Check bearings","Duration":"45m","Priority":"Medium"}
            return {"Solution":"Routine check","Duration":"1h","Priority":"Low"}

        sol = recommend(history.iloc[-1])
        st.table(pd.DataFrame([sol]).T.rename(columns={0:"Value"}))

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
