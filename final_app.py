import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")

# --- Color palettes ---
PALETTES = {
    "Ocean": ["#1976D2", "#0288D1", "#26C6DA"],
    "Forest": ["#2E7D32", "#388E3C", "#66BB6A"],
    "Sunset": ["#EF5350", "#FFA726", "#FF7043"],
    "Purple": ["#7E57C2", "#8E24AA", "#BA68C8"],
    "Slate": ["#455A64", "#546E7A", "#78909C"]
}

# --- Session defaults ---
if "menu" not in st.session_state:
    st.session_state.menu = "Dashboard"
if "lang" not in st.session_state:
    st.session_state.lang = "العربية"
if "palette" not in st.session_state:
    st.session_state.palette = "Ocean"

# --- CSS for dark theme & layout ---
st.markdown(f"""
<style>
body {{ background: #121212; color: #EEE; }}
.main-header {{ display: flex; align-items: center; gap: 0.5rem;
               background: {PALETTES[st.session_state.palette][0]}; padding: 1rem; border-radius: 0.5rem; }}
.main-header h1 {{ margin: 0; font-size: 2rem; }}
.menu-radio {{ flex-direction: row; gap: 1rem; margin-bottom: 1rem; }}
.menu-radio label {{ margin-right: 0.5rem; }}
.section {{ padding: 1rem 0; }}
.chart-container {{ background: #1E1E1E; padding: 1rem; border-radius: 0.5rem; }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="main-header">
  <span style="font-size:2.5rem;">🧠</span>
  <h1>Smart Neural Digital Twin</h1>
</div>
""", unsafe_allow_html=True)

# Main menu
st.session_state.menu = st.radio(
    ("اختر الصفحة" if st.session_state.lang=="العربية" else "Select page"),
    ["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"],
    key="menu", horizontal=True
)

# Load or simulate data
def load_history():
    if os.path.exists("sensor_data_simulated.csv"):
        df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["timestamp","temp","pressure","vibration","gas"])
    return df

history = load_history()

# Pages
if st.session_state.menu == "Dashboard":
    st.markdown(f"## {'لوحة البيانات' if st.session_state.lang=='العربية' else 'Dashboard'}", unsafe_allow_html=True)
    if history.empty:
        st.info("No data available")
    else:
        df = history.copy()
        palette = PALETTES[st.session_state.palette]
        fig = go.Figure()
        for col, color in zip(["temp","pressure","vibration","gas"], palette):
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col, line=dict(color=color)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        heat = df.pivot(index="day", columns="hour", values="temp").fillna(method="ffill")
        heat_fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Inferno"))
        st.plotly_chart(heat_fig, use_container_width=True)

elif st.session_state.menu == "Simulation":
    st.markdown(f"## {'المحاكاة' if st.session_state.lang=='العربية' else 'Simulation'}")
    sim = {
        "temp": st.slider("Temperature (°C)", 0.0, 100.0, 36.0),
        "pressure": st.slider("Pressure (kPa)", 0.0, 200.0, 95.0),
        "vibration": st.slider("Vibration (mm/s)", 0.0, 5.0, 0.5),
        "gas": st.slider("Gas (ppm)", 0.0, 10.0, 5.0)
    }
    st.map(pd.DataFrame({"lat":[24.7],"lon":[46.7]}))
    st.table(pd.DataFrame([sim]).T.rename(columns={0:"Value"}))

elif st.session_state.menu == "Predictive Analysis":
    st.markdown(f"## {'التحليل التنبؤي' if st.session_state.lang=='العربية' else 'Predictive Analysis'}")
    df = history.copy()
    if df.empty:
        st.info("No data available")
    else:
        palette = PALETTES[st.session_state.palette]
        fig = px.line(df, x="timestamp", y="temp", title="Temperature", line_color=palette[0])
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.menu == "Smart Solutions":
    st.markdown(f"## {'الحلول الذكية' if st.session_state.lang=='العربية' else 'Smart Solutions'}")
    if st.button("Generate Solution"):
        sol = {
            "Name": "Cooling System Check",
            "Details": "Inspect cooling and airflow",
            "Duration": "30m",
            "Priority": "High",
            "Effectiveness": "Very High"
        }
        st.table(pd.DataFrame([sol]).T.rename(columns={0:"Value"}))

elif st.session_state.menu == "Settings":
    st.markdown(f"## {'الإعدادات' if st.session_state.lang=='العربية' else 'Settings'}")
    st.session_state.lang = st.radio("Language", ["العربية","English"], index=0 if st.session_state.lang=="العربية" else 1)
    st.session_state.palette = st.radio("Palette", list(PALETTES.keys()), index=list(PALETTES.keys()).index(st.session_state.palette))

else:
    st.markdown(f"## {'حول' if st.session_state.lang=='العربية' else 'About'}")
    st.markdown("**Disasters don’t wait… and neither do we. Predict. Prevent. Protect.**")
    st.markdown("**Vision:** Revolutionize industrial safety by turning raw data into actionable insights.")
    st.markdown("- Real-time Monitoring • Anomaly Detection • Predictive Analytics • Smart Recommendations")
    st.markdown("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com | 0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com | 0549202574")
