import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session state defaults ---
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "lang" not in st.session_state:
    st.session_state.lang = "العربية"
if "palette" not in st.session_state:
    st.session_state.palette = "Ocean"

# --- Color palettes ---
PALETTES = {
    "Ocean": ["#1976D2", "#0288D1", "#26C6DA"],
    "Forest": ["#2E7D32", "#388E3C", "#66BB6A"],
    "Sunset": ["#EF5350", "#FFA726", "#FF7043"],
    "Purple": ["#7E57C2", "#8E24AA", "#BA68C8"],
    "Slate": ["#455A64", "#546E7A", "#78909C"]
}

def darken(color, amount=0.1):
    import colorsys
    c = color.lstrip('#')
    r, g, b = [int(c[i:i+2],16)/255 for i in (0,2,4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

primary = PALETTES[st.session_state.palette][0]
secondary = PALETTES[st.session_state.palette][1]
accent = PALETTES[st.session_state.palette][2]
bg2 = darken(primary, 0.1)

# --- Global CSS (dark theme) ---
st.markdown(f"""
<style>
body {{ background: #121212; color: #EEE; }}
.main-header {{ display:flex; align-items:center; gap:0.5rem;
               background:{primary}; padding:1rem; border-radius:0.5rem; }}
.main-header h1 {{ margin:0; font-size:2rem; color:#FFF; }}
.menu {{ margin:1rem 0; }}
.menu .stRadio {{ flex-direction: row; gap:1rem; }}
.chart-box {{ background:#1E1E1E; padding:1rem; border-radius:0.5rem; }}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div class="main-header">
  <span style="font-size:2.5rem;">🧠</span>
  <h1>Smart Neural Digital Twin</h1>
</div>
""", unsafe_allow_html=True)

# --- Page menu ---
st.session_state.page = st.radio(
    "",
    ["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"],
    index=["Dashboard","Simulation","Predictive Analysis","Smart Solutions","Settings","About"]
          .index(st.session_state.page),
    horizontal=True,
    label_visibility="collapsed"
)

# --- Load CSV data with correct column names ---
@st.cache_data
def load_data():
    if os.path.exists("sensor_data_simulated.csv"):
        df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["Time"])
        df = df.rename(columns={
            "Time": "Timestamp",
            "Temperature (°C)": "Temperature",
            "Pressure (psi)":     "Pressure",
            "Vibration (g)":      "Vibration",
            "Methane (CH₄ ppm)":  "Gas"
        })
    else:
        df = pd.DataFrame(columns=["Timestamp","Temperature","Pressure","Vibration","Gas"])
    return df.sort_values("Timestamp")

data = load_data()

# --- Pages ---
if st.session_state.page == "Dashboard":
    title = "لوحة البيانات" if st.session_state.lang=="العربية" else "Dashboard"
    st.markdown(f"## 📊 {title}")
    if data.empty:
        msg = "لا توجد بيانات للعرض. تأكد من وجود ملف sensor_data_simulated.csv" if st.session_state.lang=="العربية" else "No data available. Please add sensor_data_simulated.csv"
        st.info(msg)
    else:
        latest = data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ " + ("درجة الحرارة" if st.session_state.lang=="العربية" else "Temperature"), f"{latest['Temperature']:.2f}°C")
        col2.metric("⚡ " + ("الضغط" if st.session_state.lang=="العربية" else "Pressure"), f"{latest['Pressure']:.2f} kPa")
        col3.metric("📳 " + ("الاهتزاز" if st.session_state.lang=="العربية" else "Vibration"), f"{latest['Vibration']:.2f} g")
        col4.metric("🛢️ " + ("الميثان" if st.session_state.lang=="العربية" else "Methane"), f"{latest['Gas']:.2f} ppm")
        st.markdown("---")
        st.markdown("### 📈 " + ("مخطط البيانات الزمنية" if st.session_state.lang=="العربية" else "Sensor Readings Over Time"))
        fig = px.line(
            data,
            x="Timestamp",
            y=["Temperature","Pressure","Vibration","Gas"],
            labels={
                "value": "القيمة" if st.session_state.lang=="العربية" else "Value",
                "variable": "المستشعر" if st.session_state.lang=="العربية" else "Sensor",
                "Timestamp": "الوقت" if st.session_state.lang=="العربية" else "Time"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🌡️ " + ("خريطة الحرارة" if st.session_state.lang=="العربية" else "Temperature Heatmap"))
        df = data.copy()
        df["day"] = df["Timestamp"].dt.date
        df["hour"] = df["Timestamp"].dt.hour
        heat = df.pivot_table(index="day", columns="hour", values="Temperature", aggfunc="mean").fillna(method="ffill", axis=1)
        heat_fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Inferno"))
        st.plotly_chart(heat_fig, use_container_width=True)

elif st.session_state.page == "Simulation":
    title = "المحاكاة" if st.session_state.lang=="العربية" else "Simulation"
    st.markdown(f"## 🎮 {title}")
    temp = st.slider("Temperature (°C)", 0.0, 100.0, 36.0)
    pressure = st.slider("Pressure (psi)", 0.0, 200.0, 95.0)
    vibration = st.slider("Vibration (g)", 0.0, 5.0, 0.5)
    gas = st.slider("Methane (CH₄ ppm)", 0.0, 10.0, 5.0)
    st.map(pd.DataFrame({"lat":[24.7],"lon":[46.7]}))
    df_sim = pd.DataFrame({
        "Sensor": ["Temperature","Pressure","Vibration","Methane"],
        "Value": [temp,pressure,vibration,gas]
    }).set_index("Sensor")
    st.table(df_sim)

elif st.session_state.page == "Predictive Analysis":
    title = "التحليل التنبؤي" if st.session_state.lang=="العربية" else "Predictive Analysis"
    st.markdown(f"## 🔮 {title} (72h)")
    if data.empty:
        msg = "لا توجد بيانات للعرض." if st.session_state.lang=="العربية" else "No data available"
        st.info(msg)
    else:
        df = data.dropna(subset=["Timestamp","Temperature"]).copy()
        df["ts_num"] = df["Timestamp"].map(datetime.timestamp)
        x = df["ts_num"].values
        y = df["Temperature"].values
        coef, intercept = np.polyfit(x, y, 1)
        last_time = df["Timestamp"].iloc[-1]
        future_times = [last_time + timedelta(hours=i) for i in range(1, 73)]
        future_ts = np.array([t.timestamp() for t in future_times])
        future_temp = coef * future_ts + intercept
        hist = px.line(
            df, x="Timestamp", y="Temperature",
            labels={"Temperature":"Temp (°C)","Timestamp":"Time"},
            title=("التاريخي + تنبؤ 72 ساعة" if st.session_state.lang=="العربية" else "Historical + 72h Forecast")
        )
        hist.add_scatter(x=future_times, y=future_temp, mode="lines", name=("تنبؤ" if st.session_state.lang=="العربية" else "Prediction"), line=dict(dash="dash"))
        st.plotly_chart(hist, use_container_width=True)
        st.markdown("### 📋 " + ("ملخص التنبؤ" if st.session_state.lang=="العربية" else "Forecast Summary") + " (72h)")
        st.write(f"- {('متوسط' if st.session_state.lang=='العربية' else 'Mean')}: **{future_temp.mean():.2f}°C**")
        st.write(f"- {('أعلى' if st.session_state.lang=='العربية' else 'Max')}: **{future_temp.max():.2f}°C**")
        st.write(f"- {('أدنى' if st.session_state.lang=='العربية' else 'Min')}: **{future_temp.min():.2f}°C**")

elif st.session_state.page == "Smart Solutions":
    title = "الحلول الذكية" if st.session_state.lang=="العربية" else "Smart Solutions"
    st.markdown(f"## 🛠️ {title}")
    if st.button(("توليد حل" if st.session_state.lang=="العربية" else "Generate Solution")):
        sol = {
            "Name": "فحص التبريد" if st.session_state.lang=="العربية" else "Cooling System Check",
            "Details": "افحص نظام التبريد وتدفق الهواء" if st.session_state.lang=="العربية" else "Inspect cooling & airflow",
            "Duration": "30 دقيقة" if st.session_state.lang=="العربية" else "30m",
            "Priority": "عالي" if st.session_state.lang=="العربية" else "High",
            "Effectiveness": "عالي جداً" if st.session_state.lang=="العربية" else "Very High"
        }
        st.table(pd.DataFrame([sol]).T.rename(columns={0: ("القيمة" if st.session_state.lang=="العربية" else "Value")}))

elif st.session_state.page == "Settings":
    title = "الإعدادات" if st.session_state.lang=="العربية" else "Settings"
    st.markdown(f"## ⚙️ {title}")
    st.session_state.lang = st.radio(
        ("اختر اللغة" if st.session_state.lang=="العربية" else "Language"),
        ["العربية","English"],
        index=0 if st.session_state.lang=="العربية" else 1
    )
    st.session_state.palette = st.radio(
        ("لوحة الألوان" if st.session_state.lang=="العربية" else "Color Palette"),
        list(PALETTES.keys()),
        index=list(PALETTES.keys()).index(st.session_state.palette)
    )

else:  # About
    title = "حول" if st.session_state.lang=="العربية" else "About"
    st.markdown(f"## ℹ️ {title}")
    st.markdown("**Disasters don’t wait... and neither do we. Predict. Prevent. Protect.**")
    st.markdown("**Vision:** " + ("إحداث ثورة في السلامة الصناعية من خلال تحويل البيانات الخام إلى رؤى قابلة للتنفيذ." if st.session_state.lang=="العربية" else "Revolutionize industrial safety by turning raw data into actionable insights."))
    st.markdown("- " + ("مراقبة في الوقت الفعلي • اكتشاف الشذوذ • التحليلات التنبؤية • التوصيات الذكية" if st.session_state.lang=="العربية" else "Real-time Monitoring • Anomaly Detection • Predictive Analytics • Smart Recommendations"))
    st.markdown("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com | 0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com | 0549202574")

# Footer
st.markdown("<div style='text-align:center;color:#555;'>© Rakan & Abdulrahman</div>", unsafe_allow_html=True)
