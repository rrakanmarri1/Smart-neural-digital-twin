import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ===== Page Configuration =====
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Styles & Themes =====
theme = st.sidebar.radio("🎨 Theme / الثيم", ["Light","Dark"])
if theme == "Dark":
    bg_color = "#1e1e1e"
    text_color = "#f5f5f5"
else:
    bg_color = "#f7fafd"
    text_color = "#000000"
st.markdown(f"""
<style>
body {{ background-color: {bg_color}; color: {text_color}; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#002f6c,#00509e); color: white; }}
.main-title {{ font-size:2.5rem; font-weight:700; text-align:center; color:#002f6c; margin-bottom:1rem; }}
.section-box {{ background:#ffffff; border-left:6px solid #00509e; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; box-shadow:0 4px 8px rgba(0,0,0,0.05); }}
@media(max-width:600px) {{ .main-title{{ font-size:1.8rem!important; }} .section-box{{ padding:1rem!important; }} }}
</style>
""", unsafe_allow_html=True)

# ===== Database Initialization =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS logs (
           timestamp TEXT,
           temp REAL,
           pressure REAL,
           vibration REAL,
           gas REAL)'''
    )
    conn.commit()
    return conn

conn = init_db()

# ===== Sensor Data Fetch & Log =====
def fetch_api_data():
    # Dummy API data; replace with real endpoint if available
    return {
        'temp': float(np.random.normal(36, 2)),
        'pressure': float(np.random.normal(95, 5)),
        'vibration': float(np.random.normal(0.5, 0.1)),
        'gas': float(np.random.normal(5, 1))
    }


def log_data(data):
    c = conn.cursor()
    c.execute(
        'INSERT INTO logs VALUES (?,?,?,?,?)',
        (datetime.now().isoformat(), data['temp'], data['pressure'], data['vibration'], data['gas'])
    )
    conn.commit()

@st.cache_data(ttl=300)
def load_history():
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Predictive Model Training =====
@st.cache_data(ttl=600)
def train_model(df, target):
    if len(df) < 6:
        return None
    df = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df)):
        window = df[['temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten()
        X.append(window)
        y.append(df[target].iloc[i])
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# ===== Smart Recommendation =====
def get_rec(data, lang='en'):
    if lang == 'en':
        if data['temp'] > 38:
            return "🔥 High temperature detected! Activate cooling.", "error"
        if data['pressure'] < 85:
            return "⚠️ Low pressure! Check valves.", "warning"
        return "✅ All systems stable.", "success"
    else:
        if data['temp'] > 38:
            return "🔥 درجة حرارة عالية! فعّل التبريد.", "error"
        if data['pressure'] < 85:
            return "⚠️ الضغط منخفض! افحص الصمامات.", "warning"
        return "✅ النظام مستقر.", "success"

# ===== UI Setup =====
language = st.sidebar.selectbox("🌐 Language / اللغة", ["English","العربية"])
menu = st.sidebar.radio("📋 Menu / القائمة", ["Dashboard","Simulation","Predictive","Solutions","About"])

# Fetch, log, and load data
data = fetch_api_data()
log_data(data)
history = load_history()

# ===== Anomaly Detection =====
if len(history) >= 20:
    iso_model = IsolationForest(contamination=0.05)
    history['anomaly'] = iso_model.fit_predict(history[['temp','pressure','vibration','gas']])
else:
    history['anomaly'] = 1

# ===== Pages =====
# Dashboard
if menu == 'Dashboard':
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin Dashboard</div>", unsafe_allow_html=True)
    st.subheader("📊 Live Sensor Data" if language=='en' else "📊 بيانات المستشعرات الحية")
    cols = st.columns(4)
    labels_en = ['Temperature', 'Pressure', 'Vibration', 'Gas']
    labels_ar = ['درجة الحرارة', 'الضغط', 'الاهتزاز', 'الغاز']
    keys = ['temp','pressure','vibration','gas']
    for i, key in enumerate(keys):
        cols[i].metric(labels_en[i] if language=='en' else labels_ar[i], f"{data[key]:.2f}")
    st.markdown("---")
    if not history.empty:
        fig = px.line(history, x='timestamp', y=keys, color='anomaly', labels={'value':'Reading','timestamp':'Time','anomaly':'Anomaly'})
        st.plotly_chart(fig, use_container_width=True)
        # Map View
        st.subheader("📍 Sensor Locations")
        locs = pd.DataFrame([
            {'lat':26.369, 'lon':50.133, 'sensor':'S1'},
            {'lat':26.370, 'lon':50.134, 'sensor':'S2'}
        ])
        mfig = px.scatter_mapbox(locs, lat='lat', lon='lon', hover_name='sensor', zoom=12, height=300)
        mfig.update_layout(mapbox_style='open-street-map', margin={'r':0,'t':0,'l':0,'b':0})
        st.plotly_chart(mfig, use_container_width=True)
        # Calendar Heatmap
        history['day'] = history['timestamp'].dt.day
        history['hour'] = history['timestamp'].dt.hour
        heat = history.pivot_table(index='day', columns='hour', values='temp')
        hfig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Viridis'))
        st.plotly_chart(hfig, use_container_width=True)
    else:
        st.info("Waiting for data..." if language=='en' else "بانتظار البيانات...")
    # Data Export
    csv = history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", csv, "history.csv", "text/csv")
    xlsx = history.to_excel(index=False).encode('utf-8')
    st.sidebar.download_button("Download Excel", xlsx, "history.xlsx")

# Simulation
elif menu == 'Simulation':
    st.markdown("<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
    st.subheader("Adjust Values" if language=='en' else "ضبط القيم")
    sim = {}
    sim['temp'] = st.slider('Temperature (°C)' if language=='en' else 'درجة الحرارة (°C)', 20, 50, int(data['temp']))
    sim['pressure'] = st.slider('Pressure (kPa)' if language=='en' else 'الضغط (kPa)', 60, 120, int(data['pressure']))
    sim['vibration'] = st.slider('Vibration (mm/s)' if language=='en' else 'الاهتزاز (mm/s)', 0.0, 1.5, float(data['vibration']), 0.01)
    sim['gas'] = st.slider('Gas (ppm)' if language=='en' else 'الغاز (ppm)', 0.0, 10.0, float(data['gas']), 0.1)
    st.table(pd.DataFrame([sim]).T)

# Predictive
elif menu == 'Predictive':
    st.markdown("<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
    model = train_model(history, 'temp')
    if model is None:
        st.warning("Not enough data." if language=='en' else "لا توجد بيانات كافية.")
    else:
        last_vals = history[['temp','pressure','vibration','gas']].iloc[-5:].values.flatten().reshape(1,-1)
        future = [datetime.now() + timedelta(minutes=10*i) for i in range(1,11)]
        preds = model.predict(np.repeat(last_vals, 10, axis=0))
        pfig = go.Figure([go.Scatter(x=future, y=preds, mode='lines+markers')])
        pfig.update_layout(
            title="Temperature Forecast" if language=='en' else "توقع درجة الحرارة",
            xaxis_title="Time" if language=='en' else "الوقت",
            yaxis_title='Temperature (°C)' if language=='en' else 'درجة الحرارة (°C)'
        )
        st.plotly_chart(pfig, use_container_width=True)

# Smart Solutions
elif menu == 'Solutions':
    st.markdown("<div class='main-title'>🛠️ Smart Solutions</div>", unsafe_allow_html=True)
    if st.button("Generate Solution" if language=='en' else "توليد الحل"):
        sol = {
            'Name': 'Cooling System Check' if language=='en' else 'فحص نظام التبريد',
            'Details': 'Perform full diagnostic of cooling fans and coolant levels.' if language=='en' else 'إجراء فحص شامل للمراوح ومستويات سائل التبريد.',
            'Duration': '30 mins' if language=='en' else '30 دقيقة',
            'Priority': 'High' if language=='en' else 'عالية',
            'Effectiveness': 'Very High' if language=='en' else 'عالية جداً'
        }
        st.table(pd.DataFrame(sol, index=[0]).T)

# About
elif menu == 'About':
    st.markdown("<div class='main-title'>ℹ️ About / حول المشروع</div>", unsafe_allow_html=True)
    if language=='en':
        st.write("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
        st.write("**Contact:** rakan.almarri.2@aramco.com | +966532559664")
        st.write("**Contact:** abdulrahman.alzhrani.1@aramco.com | +966549202574")
        st.write("---")
        st.write("**Vision:** Revolutionize Aramco's industrial safety with actionable insights and predictive prevention.")
        st.write("**Features:** Live dashboard, anomaly detection, predictive modeling, smart solutions, bilingual UI.")
    else:
        st.write("**الفريق:** راكان المري & عبد الرحمن الزهراني")
        st.write("**التواصل:** rakan.almarri.2@aramco.com | 0532559664")
        st.write("**التواصل:** abdulrahman.alzhrani.1@aramco.com | 0549202574")
        st.write("---")
        st.write("**رؤيتنا:** إحداث ثورة في سلامة أرامكو الصناعية برؤى تنفيذية وتنبؤ استباقي.")
        st.write("**المزايا:** لوحة بيانات حية، اكتشاف شذوذ، نماذج تنبؤية، حلول ذكية، واجهة ثنائية اللغة.")
