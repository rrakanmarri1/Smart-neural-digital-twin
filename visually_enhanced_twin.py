
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 🎨 Page Config
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🛡️",
    layout="wide"
)

# 🌈 Custom Theme Styling
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: linear-gradient(to bottom right, #f0f8ff, #e6f2ff);
        padding: 2rem;
        border-radius: 12px;
    }
    .metric {
        background-color: #ffffff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title & Description
st.markdown("<div class='header'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
st.markdown("🚨 **Disasters don't wait.. and neither do we.**  
🔮 Predict • 🛡️ Prevent • 🧯 Protect", unsafe_allow_html=True)
st.markdown("---")

# 📊 Simulated Sensor Data
st.subheader("📊 Live Sensor Data")
data = {
    "Temperature (°C)": round(np.random.normal(36, 2), 2),
    "Pressure (kPa)": round(np.random.normal(95, 5), 2),
    "Vibration (mm/s)": round(np.random.normal(0.5, 0.1), 2)
}

for label, value in data.items():
    st.markdown(f"<div class='metric'><b>{label}</b>: {value}</div>", unsafe_allow_html=True)

# 💡 Recommendation Section
st.subheader("💡 Smart Recommendation")
if data["Temperature (°C)"] > 38:
    st.error("🔥 High temperature detected! Consider activating the cooling system.")
elif data["Pressure (kPa)"] < 85:
    st.warning("⚠️ Low pressure detected! Check the valves and pipeline.")
else:
    st.success("✅ All systems are operating within safe parameters.")

# 📅 Timestamp
st.markdown(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
