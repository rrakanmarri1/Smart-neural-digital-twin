import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# ===== Page Config =====
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Responsive & Mobile CSS =====
st.markdown("""
    <style>
    /* Ensure mobile responsiveness */
    @media only screen and (max-width: 600px) {
        .main-title { font-size: 1.5em !important; }
        .section-box { padding: 1rem !important; }
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1976d2,#2196f3 90%);
        color: white;
    }
    .main-title { font-size:2.2em; text-align:center; margin-bottom:1rem; color:#0d47a1; }
    .section-box { background:#fff; border-left:6px solid #1976d2; padding:1.3rem; margin-bottom:1.3rem; border-radius:10px; box-shadow:1px 1px 6px rgba(33,150,243,0.1); }
    </style>
""", unsafe_allow_html=True)

# ===== Database Setup =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
                timestamp TEXT,
                temp REAL,
                pressure REAL,
                vibration REAL,
                gas REAL)''')
    conn.commit()
    return conn

conn = init_db()

# ===== Fetch Sensor Data from API (Dummy) =====
def fetch_api_data():
    # Dummy API call - replace URL with real endpoint
    # resp = requests.get('https://api.example.com/sensors')
    # data = resp.json()
    # Simulate
    return {
        'temp': np.random.normal(36,2),
        'pressure': np.random.normal(95,5),
        'vibration': np.random.normal(0.5,0.1),
        'gas': np.random.normal(5,1)
    }

# ===== Logging =====
def log_data(data):
    c = conn.cursor()
    c.execute('INSERT INTO logs VALUES (?,?,?,?,?)', (
        datetime.now().isoformat(), data['temp'], data['pressure'], data['vibration'], data['gas']
    ))
    conn.commit()

# ===== Load Historical Data =====
@st.cache_data(ttl=300)
def load_history():
    df = pd.read_sql('SELECT * FROM logs', conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Train RandomForest Model =====
@st.cache_data(ttl=600)
def train_model(df, target):
    # Use previous 5 readings to predict next
    df = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df)):
        X.append(df[[ 'temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten())
        y.append(df[target].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

# ===== Main =====
language = st.sidebar.selectbox("🌐 Choose Language | اختر اللغة", ["English","العربية"])
menu = st.sidebar.radio("🚀 Navigate | تنقل إلى", ["Dashboard","Simulation","Predictive","Recommendations","About"])

# Fetch & log current data
data = fetch_api_data()
log_data(data)
history = load_history()

# Dashboard Page
if menu == "Dashboard":
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    st.subheader("📊 Live Sensor Dashboard" if language=='English' else "📊 لوحة البيانات الحية")
    cols = st.columns(4)
    keys = ['temp','pressure','vibration','gas']
    labels_en = ['Temperature (°C)','Pressure (kPa)','Vibration (mm/s)','Gas (ppm)']
    labels_ar = ['درجة الحرارة (°C)','الضغط (kPa)','الاهتزاز (mm/s)','الغاز (ppm)']
    for i,key in enumerate(keys):
        with cols[i]:
            st.metric(labels_en[i] if language=='English' else labels_ar[i], f"{data[key]:.2f}")
    # Time-series plot
    df = history
    fig = px.line(df, x='timestamp', y=keys, labels={'value':'Reading','timestamp':'Time'})
    st.plotly_chart(fig, use_container_width=True)

# Simulation Page
elif menu == "Simulation":
    st.markdown("<div class='main-title'>🎛️ Simulation</div>" if language=='English' else "<div class='main-title'>🎛️ المحاكاة</div>", unsafe_allow_html=True)
    temp = st.slider(labels_en[0] if language=='English' else labels_ar[0],20,50,float(data['temp']))
    pressure = st.slider(labels_en[1] if language=='English' else labels_ar[1],60,120,float(data['pressure']))
    vibration = st.slider(labels_en[2] if language=='English' else labels_ar[2],0.0,1.5,float(data['vibration']),0.01)
    gas = st.slider(labels_en[3] if language=='English' else labels_ar[3],0,10,float(data['gas']),0.1)
    data_sim = {'temp':temp,'pressure':pressure,'vibration':vibration,'gas':gas}
    st.markdown("---")
    st.subheader("Simulation Data")
    st.table(pd.DataFrame([data_sim]).T)

# Predictive Page
elif menu == "Predictive":
    st.markdown("<div class='main-title'>📈 Predictive Analysis</div>" if language=='English' else "<div class='main-title'>📈 التحليل التنبؤي</div>", unsafe_allow_html=True)
    model_temp = train_model(history,'temp')
    latest = history[['temp','pressure','vibration','gas']].iloc[-5:].values.flatten().reshape(1,-1)
    pred = model_temp.predict(latest)[0]
    future_times = [datetime.now()+timedelta(minutes=15*i) for i in range(1,11)]
    preds = [model_temp.predict(latest)[0] + np.random.normal(0,0.2) for _ in future_times]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=future_times,y=preds,mode='lines',name='Prediction'))
    st.plotly_chart(fig2,use_container_width=True)

# Recommendations Page
elif menu == "Recommendations":
    st.markdown("<div class='main-title'>💡 Smart Recommendations</div>" if language=='English' else "<div class='main-title'>💡 التوصيات الذكية</div>", unsafe_allow_html=True)
    rec_en = lambda d: ("High temp! cool down." if d['temp']>38 else "Low pressure! check valves." if d['pressure']<85 else "All safe.","error" if d['temp']>38 else "warning" if d['pressure']<85 else "success")
    rec_ar = lambda d: ("🔥 درجة حرارة عالية! فعّل التبريد." if d['temp']>38 else "⚠️ الضغط منخفض." if d['pressure']<85 else "✅ الوضع آمن.","error" if d['temp']>38 else "warning" if d['pressure']<85 else "success")
    rec,rt = rec_en(data) if language=='English' else rec_ar(data)
    getattr(st, rt)(rec)

# About Page
elif menu == "About":
    st.markdown("<div class='main-title'>ℹ️ About / حول المشروع</div>", unsafe_allow_html=True)
    if language=='English':
        st.write("Developed by Rakan Almarri & Abdulrahman Alzhrani")
        st.write("Contact: rakan.almarri.2@aramco.com | +966532559664")
        st.write("Contact: abdulrahman.alzhrani.1@aramco.com | +966549202574")
        st.write("Vision: To revolutionize industrial safety by transforming raw data into actionable insights.")
    else:
        st.write("تم التطوير بواسطة راكان المري وعبدالرحمن الزهراني")
        st.write("تواصل: rakan.almarri.2@aramco.com | 0532559664")
        st.write("تواصل: abdulrahman.alzhrani.1@aramco.com | 0549202574")
        st.write("رؤيتنا: إحداث ثورة في السلامة الصناعية بتحويل البيانات لخدمات تنفيذية.")
