import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# ===== Page Configuration =====
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS for Professional Look =====
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002f6c, #00509e);
    color: white;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] .css-1d391kg {
    color: white;
}
/* Main title */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #002f6c;
    text-align: center;
    margin-bottom: 1rem;
}
/* Section box */
.section-box {
    background: #ffffff;
    border-left: 6px solid #00509e;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
/* Responsive */
@media(max-width: 600px) {
    .main-title { font-size: 1.8rem !important; }
    .section-box { padding: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ===== Initialize Database =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
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

# ===== Fetch Sensor Data (Dummy API) =====
def fetch_api_data():
    # Replace with real API call if available
    # resp = requests.get('https://api.example.com/sensors')
    # data = resp.json()
    return {
        'temp': float(np.random.normal(36, 2)),
        'pressure': float(np.random.normal(95, 5)),
        'vibration': float(np.random.normal(0.5, 0.1)),
        'gas': float(np.random.normal(5, 1))
    }

# ===== Log Data =====
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
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Train Predictive Model =====
@st.cache_data(ttl=600)
def train_model(df, target):
    # Need at least 6 points (5 for features + 1 target)
    if len(df) < 6:
        return None
    df = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df)):
        X.append(df[['temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten())
        y.append(df[target].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

# ===== Smart Recommendations =====
def get_rec(data, lang="en"):
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
menu = st.sidebar.radio("📋 Menu / القائمة", ["Dashboard","Simulation","Predictive","Recommendations","About"])

# Fetch and log current sensor data
current = fetch_api_data()
log_data(current)
history = load_history()

# Common labels
labels = {
    'en': ['Temperature (°C)','Pressure (kPa)','Vibration (mm/s)','Gas (ppm)'],
    'ar': ['درجة الحرارة (°C)','الضغط (kPa)','الاهتزاز (mm/s)','الغاز (ppm)']
}
keys = ['temp','pressure','vibration','gas']

# ===== Pages =====
# Dashboard
if menu == 'Dashboard':
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin Dashboard</div>", unsafe_allow_html=True)
    st.subheader("📊 Live Data" if language=='en' else "📊 البيانات الحية")
    cols = st.columns(4)
    for i, key in enumerate(keys):
        cols[i].metric(labels['en' if language=='en' else 'ar'][i], f"{current[key]:.2f}")
    st.markdown("---")
    if not history.empty:
        fig = px.line(history, x='timestamp', y=keys, labels={'value':'Reading','timestamp':'Time'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data logs..." if language=='en' else "بانتظار تسجيل البيانات...")

# Simulation
elif menu == 'Simulation':
    st.markdown("<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
    st.subheader("Adjust Values" if language=='en' else "ضبط القيم")
    sim = {}
    sim['temp'] = st.slider(labels['en'][0] if language=='en' else labels['ar'][0], 20, 50, int(current['temp']))
    sim['pressure'] = st.slider(labels['en'][1] if language=='en' else labels['ar'][1], 60, 120, int(current['pressure']))
    sim['vibration'] = st.slider(labels['en'][2] if language=='en' else labels['ar'][2], 0.0, 1.5, float(current['vibration']), 0.01)
    sim['gas'] = st.slider(labels['en'][3] if language=='en' else labels['ar'][3], 0.0, 10.0, float(current['gas']), 0.1)
    st.markdown("---")
    rec, rtype = get_rec(sim, 'en' if language=='en' else 'ar')
    getattr(st, rtype)(rec)
    df_sim = pd.DataFrame([sim]).T
    df_sim.columns = ['Value']
    st.table(df_sim)

# Predictive
elif menu == 'Predictive':
    st.markdown("<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
    model = train_model(history, 'temp')
    if model is None:
        st.warning("Not enough data for prediction." if language=='en' else "لا توجد بيانات كافية للتنبؤ.")
    else:
        last_vals = history[['temp','pressure','vibration','gas']].iloc[-5:].values.flatten().reshape(1,-1)
        future = [datetime.now() + timedelta(minutes=10*i) for i in range(1,11)]
        preds = model.predict(np.repeat(last_vals,10,axis=0))
        fig2 = go.Figure([go.Scatter(x=future, y=preds, mode='lines+markers')])
        fig2.update_layout(title="Temperature Forecast" if language=='en' else "توقع درجة الحرارة",
                           xaxis_title="Time" if language=='en' else "الوقت",
                           yaxis_title=labels['en'][0] if language=='en' else labels['ar'][0])
        st.plotly_chart(fig2, use_container_width=True)

# Recommendations
elif menu == 'Recommendations':
    st.markdown("<div class='main-title'>💡 Recommendations</div>", unsafe_allow_html=True)
    rec, rtype = get_rec(current, 'en' if language=='en' else 'ar')
    getattr(st, rtype)(rec)

# About
elif menu == 'About':
    st.markdown("<div class='main-title'>ℹ️ About / حول المشروع</div>", unsafe_allow_html=True)
    if language == 'en':
        st.write("**Developed by:** Rakan Almarri & Abdulrahman Alzhrani")
        st.write("**Contact:** rakan.almarri.2@aramco.com | +966532559664")
        st.write("**Contact:** abdulrahman.alzhrani.1@aramco.com | +966549202574")
        st.write("---")
        st.write("**Vision:** To revolutionize industrial safety by transforming raw data into actionable insights.")
        st.write("**Features:** Real-time dashboard, predictive modeling, smart recommendations, bilingual UI.")
    else:
        st.write("**المطورون:** راكان المري & عبد الرحمن الزهراني")
        st.write("**التواصل:** rakan.almarri.2@aramco.com | 0532559664")
        st.write("**التواصل:** abdulrahman.alzhrani.1@aramco.com | 0549202574")
        st.write("---")
        st.write("**رؤيتنا:** إحداث ثورة في السلامة الصناعية عبر تحويل البيانات لرؤى تنفيذية.")
        st.write("**المميزات:** لوحة بيانات حية، نماذج تنبؤية، توصيات ذكية، واجهة ثنائية اللغة.")

# ===== End =====
