import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")

# Load data
df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["Timestamp"])
df = df.rename(columns={
    "Timestamp": "timestamp",
    "Temperature (°C)": "temperature",
    "Pressure (psi)": "pressure",
    "Vibration (g)": "vibration",
    "Methane (CH₄, ppm)": "gas"
})

# UI controls
lang = st.radio("", ["العربية", "English"], horizontal=True, key="lang")
pal = st.radio("", ["Ocean", "Forest", "Sunset", "Purple", "Slate"], horizontal=True, key="pal")

# Translations
t = {
    "العربية": {
        "menu": ["لوحة البيانات", "المحاكاة", "التحليل التنبؤي", "الحلول الذكية", "الإعدادات", "حول"],
        "dashboard": "لوحة البيانات",
        "simulation": "المحاكاة",
        "predictive": "التحليل التنبؤي",
        "solutions": "الحلول الذكية",
        "settings": "الإعدادات",
        "about": "حول",
        "generate": "توليد الحل",
        "no_data": "لا توجد بيانات",
        "select_page": "اختر الصفحة",
        "select_lang": "اختر اللغة",
        "select_pal": "لوحة الألوان",
        "sensitivity": "حساسية كشف الشذوذ",
        "temp": "درجة الحرارة (°C)",
        "pres": "الضغط (psi)",
        "vib": "الاهتزاز (g)",
        "gas": "غاز الميثان (ppm)",
        "dashboard_title": "لوحة البيانات",
        "sim_title": "المحاكاة",
        "pred_title": "التحليل التنبؤي خلال 72 ساعة",
        "sol_title": "الحلول الذكية",
        "set_title": "الإعدادات",
        "about_title": "حول المشروع"
    },
    "English": {
        "menu": ["Dashboard", "Simulation", "Predictive Analysis", "Smart Solutions", "Settings", "About"],
        "dashboard": "Dashboard",
        "simulation": "Simulation",
        "predictive": "Predictive Analysis",
        "solutions": "Smart Solutions",
        "settings": "Settings",
        "about": "About",
        "generate": "Generate Solution",
        "no_data": "No data available",
        "select_page": "Select Page",
        "select_lang": "Select Language",
        "select_pal": "Palette",
        "sensitivity": "Anomaly Sensitivity",
        "temp": "Temperature (°C)",
        "pres": "Pressure (psi)",
        "vib": "Vibration (g)",
        "gas": "Methane (ppm)",
        "dashboard_title": "Dashboard",
        "sim_title": "Simulation",
        "pred_title": "72h Predictive Analysis",
        "sol_title": "Smart Solutions",
        "set_title": "Settings",
        "about_title": "About"
    }
}[lang]

# Backgrounds
bg = {
    "Ocean": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Forest": "https://images.unsplash.com/photo-1501785888041-af3ef285b470",
    "Sunset": "https://images.unsplash.com/photo-1518837695005-2083093ee35b",
    "Purple": "https://images.unsplash.com/photo-1519985176271-adb1088fa94c",
    "Slate": "https://images.unsplash.com/photo-1503602642458-232111445657"
}[pal]

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{bg}?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260');
        background-size: cover;
        color: white;
    }}
    .menu-container {{
        display: flex; gap: 2rem; margin-top: 1rem;
    }}
    .menu-container label {{
        cursor: pointer; font-size: 1.1rem;
    }}
    .main-header {{
        background: rgba(0,0,0,0.5); padding: 1rem; border-radius: 8px;
        display: flex; align-items: center; gap: 0.5rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Page selector
page = st.radio("", t["menu"], index=0, format_func=lambda x: x, horizontal=True)

# Header
st.markdown(f"<div class='main-header'><span>🧠</span><h1>{t[page.lower().replace(' ', '_')]}</h1></div>", unsafe_allow_html=True)

if page == t["dashboard"]:
    st.subheader(t["dashboard_title"])
    if df.empty:
        st.info(t["no_data"])
    else:
        cols = st.columns(4)
        latest = df.iloc[-1]
        cols[0].metric(t["temp"], f"{latest.temperature:.2f}")
        cols[1].metric(t["pres"], f"{latest.pressure:.2f}")
        cols[2].metric(t["vib"], f"{latest.vibration:.2f}")
        cols[3].metric(t["gas"], f"{latest.gas:.2f}")

elif page == t["simulation"]:
    st.subheader(t["sim_title"])
    sens = st.slider(t["sensitivity"], 0.01, 0.30, 0.05)
    sd = {
        "temperature": st.slider(t["temp"], 20.0, 50.0, float(latest.temperature)),
        "pressure": st.slider(t["pres"], 60.0, 120.0, float(latest.pressure)),
        "vibration": st.slider(t["vib"], 0.0, 1.5, float(latest.vibration)),
        "gas": st.slider(t["gas"], 0.0, 10.0, float(latest.gas))
    }
    sim_df = pd.DataFrame([sd])
    fig = px.heatmap(sim_df.T, labels={"value": "Value"}, y=sim_df.T.index, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

elif page == t["predictive"]:
    st.subheader(t["pred_title"])
    df72 = df.tail(72)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df72.timestamp, y=df72.temperature, mode="lines", name=t["temp"]))
    fig.add_trace(go.Scatter(x=df72.timestamp, y=df72.pressure, mode="lines", name=t["pres"], yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)

elif page == t["solutions"]:
    st.subheader(t["sol_title"])
    if st.button(t["generate"]):
        sol = {
            "Name": [ "Diagnose Cooling System" ],
            "Details": [ "Check fans, fluid levels, clear vents." ],
            "Duration": [ "30 min" ],
            "Priority": [ "High" ],
            "Effectiveness": [ "Very High" ]
        }
        sol_df = pd.DataFrame(sol)
        st.table(sol_df)

elif page == t["settings"]:
    st.subheader(t["set_title"])
    st.write(f"{t['select_lang']}: {lang}")
    st.write(f"{t['select_pal']}: {pal}")

elif page == t["about"]:
    st.subheader(t["about_title"])
    if lang == "العربية":
        st.write("نظام توأم رقمي عصبي ذكي يعرض البيانات الحية ويكتشف الشذوذ ويتنبأ بالمستقبل ويوفر حلولاً ذكية.")
    else:
        st.write("A Smart Neural Digital Twin showcasing live data, anomaly detection, predictive analytics and smart solutions.")
