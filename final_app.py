import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

# Load the simulated data
df = pd.read_csv(
    "sensor_data_simulated.csv",
    parse_dates=["Time"],
    sep=","
)
df.rename(columns={
    "Time": "timestamp",
    "Temperature (°C)": "temp",
    "Pressure (psi)": "pressure",
    "Vibration (g)": "vibration",
    "Methane (CH₄ ppm)": "gas",
    "H₂S (ppm)": "h2s"
}, inplace=True)

# Sidebar: language & theme
st.sidebar.title("MENU")
lang = st.sidebar.radio("Select Language", ("العربية", "English"))
palette = st.sidebar.radio("لوحة الألوان" if lang=="العربية" else "Palette",
    ("Ocean", "Forest", "Sunset", "Purple", "Slate"))

# Main header
bg_colors = {
    "Ocean": "#1E90FF",
    "Forest": "#228B22",
    "Sunset": "#FF4500",
    "Purple": "#800080",
    "Slate": "#2F4F4F",
}
st.markdown(f"""
    <div style="background-color:{bg_colors[palette]};
                color:white;
                padding:1rem;
                border-radius:5px;
                display:flex;
                align-items:center;
                gap:0.5rem;">
        <span style="font-size:1.5rem;">🧠</span>
        <h1 style="margin:0;">{"التوأم الرقمي العصبي الذكي" if lang=="العربية" else "Smart Neural Digital Twin"}</h1>
    </div>
""", unsafe_allow_html=True)

# Menu selection
pages = ["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"]
labels_ar = ["لوحة البيانات","المحاكاة","التحليل التنبؤي","الحلول الذكية","الإعدادات","حول"]
items = []
for i, p in enumerate(pages):
    items.append(st.radio(
        "", 
        (f"{'🔴 ' if i==0 else '⚪️ '}{(labels_ar[i] if lang=='العربية' else p)}"), 
        key=p
    ))
current = pages[items.index(next(filter(lambda x: x, items)))]

if current == "Dashboard":
    st.subheader("لوحة البيانات" if lang=="العربية" else "Dashboard")
    cols = st.columns(4)
    latest = df.iloc[-1]
    cols[0].metric("درجة الحرارة (°C)" if lang=="العربية" else "Temperature (°C)", f"{latest.temp:.2f}")
    cols[1].metric("الضغط (psi)" if lang=="العربية" else "Pressure (psi)", f"{latest.pressure:.2f}")
    cols[2].metric("الاهتزاز (g)" if lang=="العربية" else "Vibration (g)", f"{latest.vibration:.2f}")
    cols[3].metric("غاز الميثان (ppm)" if lang=="العربية" else "Methane (ppm)", f"{latest.gas:.2f}")

    st.markdown("---")
    fig = px.line(df, x="timestamp", y=["temp","pressure","vibration","gas"],
                  labels={"timestamp": "الوقت" if lang=="العربية" else "Time",
                          "value": "القيمة" if lang=="العربية" else "Value",
                          "variable": ""}, title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("خريطة حرارية" if lang=="العربية" else "Heatmap")
    heat = df.pivot_table(index=df.timestamp.dt.hour, columns=df.timestamp.dt.day, values="temp", aggfunc="mean")
    fig2 = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Viridis"))
    st.plotly_chart(fig2, use_container_width=True)

elif current == "Predictive Analysis":
    st.subheader("التحليل التنبؤي" if lang=="العربية" else "Predictive Analysis")
    future = df.set_index("timestamp").resample("H").mean().ffill().last("72H")
    fig3 = px.line(future, x=future.index, y="temp",
                   labels={"timestamp": "الوقت" if lang=="العربية" else "Time",
                           "temp": "درجة الحرارة (°C)" if lang=="العربية" else "Temperature (°C)"},
                   title="")
    st.plotly_chart(fig3, use_container_width=True)

elif current == "About":
    st.subheader("حول المشروع" if lang=="العربية" else "About")
    st.write("نظام للتوأم الرقمي العصبي يتتبع ويحلل بيانات المستشعرات..." 
             if lang=="العربية" else
             "A smart neural digital twin for real-time sensor monitoring, anomaly detection, predictive analytics, and smart recommendations.")

elif current == "Settings":
    st.subheader("الإعدادات" if lang=="العربية" else "Settings")
    st.write("اختر اللغة ولوحة الألوان أعلى" if lang=="العربية" else "Use the sidebar to change language and palette.")

else:
    pass
