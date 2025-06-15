import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")

PALETTES = {
    "Ocean": ["#1976D2", "#0288D1", "#26C6DA"],
    "Forest": ["#2E7D32", "#388E3C", "#66BB6A"],
    "Sunset": ["#EF5350", "#FFA726", "#FF7043"],
    "Purple": ["#7E57C2", "#8E24AA", "#BA68C8"],
    "Slate": ["#455A64", "#546E7A", "#78909C"]
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "lang" not in st.session_state:
    st.session_state.lang = "العربية"
if "palette" not in st.session_state:
    st.session_state.palette = "Ocean"

st.markdown(f"""
<style>
body {{ background: #121212; color: #EEE; }}
.header {{ display:flex; align-items:center; gap:0.5rem;
           background:{PALETTES[st.session_state.palette][0]}; padding:1rem; border-radius:0.5rem; }}
.header h1 {{ margin:0; font-size:2rem; }}
.menu {{ margin:1rem 0; }}
.menu label {{ margin-right:1rem; }}
.chart-box {{ background:#1E1E1E; padding:1rem; border-radius:0.5rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='header'><span style='font-size:2.5rem;'>🧠</span><h1>Smart Neural Digital Twin</h1></div>", unsafe_allow_html=True)

st.session_state.page = st.radio(
    ("اختر الصفحة" if st.session_state.lang=="العربية" else "Select page"),
    ["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"],
    index=["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"].index(st.session_state.page),
    horizontal=True
)

def load_history():
    if os.path.exists("sensor_data_simulated.csv"):
        return pd.read_csv("sensor_data_simulated.csv", parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp","temp","pressure","vibration","gas"])

history = load_history()

if st.session_state.page == "Dashboard":
    st.markdown(f"## {'لوحة البيانات' if st.session_state.lang=='العربية' else 'Dashboard'}")
    if history.empty:
        st.info("No data available")
    else:
        df = history.copy()
        colors = PALETTES[st.session_state.palette]
        fig = go.Figure()
        for col,c in zip(["temp","pressure","vibration","gas"],colors):
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col, line=dict(color=c)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        heat = df.pivot(index="day", columns="hour", values="temp").fillna(method="ffill")
        heat_fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Inferno"))
        st.plotly_chart(heat_fig, use_container_width=True)

elif st.session_state.page == "Simulation":
    st.markdown(f"## {'المحاكاة' if st.session_state.lang=='العربية' else 'Simulation'}")
    temp = st.slider("Temperature (°C)", 0.0,100.0,36.0)
    pressure = st.slider("Pressure (kPa)", 0.0,200.0,95.0)
    vibration = st.slider("Vibration (mm/s)", 0.0,5.0,0.5)
    gas = st.slider("Gas (ppm)", 0.0,10.0,5.0)
    st.map(pd.DataFrame({"lat":[24.7],"lon":[46.7]}))
    st.table(pd.DataFrame({"Sensor":["Temperature","Pressure","Vibration","Gas"],
                           "Value":[temp,pressure,vibration,gas]}).set_index("Sensor"))

elif st.session_state.page == "Predictive Analysis":
    st.markdown(f"## {'التحليل التنبؤي' if st.session_state.lang=='العربية' else 'Predictive Analysis'}")
    if history.empty:
        st.info("No data available")
    else:
        df = history.copy()
        colors = PALETTES[st.session_state.palette]
        fig = px.line(df, x="timestamp", y="temp", title="Temperature", line_color=colors[0])
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "Smart Solutions":
    st.markdown(f"## {'الحلول الذكية' if st.session_state.lang=='العربية' else 'Smart Solutions'}")
    if st.button("Generate Solution"):
        sol = {"Name":"Cooling System Check","Details":"Inspect cooling and airflow",
               "Duration":"30m","Priority":"High","Effectiveness":"Very High"}
        st.table(pd.DataFrame([sol]).T.rename(columns={0:"Value"}))

elif st.session_state.page == "Settings":
    st.markdown(f"## {'الإعدادات' if st.session_state.lang=='العربية' else 'Settings'}")
    st.session_state.lang = st.radio("Language", ["العربية","English"], index=0 if st.session_state.lang=="العربية" else 1)
    st.session_state.palette = st.radio("Palette", list(PALETTES.keys()),
                                        index=list(PALETTES.keys()).index(st.session_state.palette))

else:
    st.markdown(f"## {'حول' if st.session_state.lang=='العربية' else 'About'}")
    st.markdown("**Disasters don’t wait… and neither do we. Predict. Prevent. Protect.**")
    st.markdown("**Vision:** Revolutionize industrial safety by turning raw data into actionable insights.")
    st.markdown("- Real-time Monitoring • Anomaly Detection • Predictive Analytics • Smart Recommendations")
    st.markdown("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com | 0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com | 0549202574")
