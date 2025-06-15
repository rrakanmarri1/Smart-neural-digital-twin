import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ===== Page Config =====
st.set_page_config(page_title="🧠 Smart Neural Digital Twin", page_icon="🌐", layout="wide")

# ===== Session Defaults =====
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'

# ===== Color Palettes =====
PALETTES = {
    'Ocean': ['#1976D2', '#0288D1', '#26C6DA'],
    'Forest': ['#2E7D32', '#388E3C', '#66BB6A'],
    'Sunset': ['#EF5350', '#FFA726', '#FF7043']
}
colors = PALETTES[st.session_state.palette]
primary, secondary, accent = colors

# ===== Styling =====
def darken(hex_color, amount=0.1):
    import colorsys
    c = hex_color.lstrip('#')
    r, g, b = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

sidebar_bg = f"linear-gradient(180deg, {primary}, {secondary})"
main_bg = darken(primary, 0.1)
box_bg = darken(accent, 0.1)

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background: {sidebar_bg} !important; color: white !important; }}
body {{ background: #121212 !important; color: #E0E0E0 !important; }}
.main-title {{ text-align: center; font-size:2.5rem; color: {primary}; margin-bottom:1rem; }}
.section-box {{ background: {main_bg}; border-left: 6px solid {secondary}; padding:1rem; margin:1rem 0; border-radius:8px; }}
.menu-item {{ background: {box_bg}; padding:0.5rem 1rem; border-radius:6px; margin:0.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# ===== Database =====
def init_db():
    conn = sqlite3.connect('logs.db', check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)'
    )
    conn.commit()
    return conn

conn = init_db()

# ===== Data Functions =====
def fetch_data():
    return {
        'temp': float(np.random.normal(36, 2)),
        'pressure': float(np.random.normal(95, 5)),
        'vibration': float(np.random.normal(0.5, 0.1)),
        'gas': float(np.random.normal(5, 1))
    }

def log_data(d):
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO logs VALUES (?,?,?,?,?)',
        (datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas'])
    )
    conn.commit()

@st.cache_data(ttl=300)
def load_history():
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Prediction Model =====n@st.cache_data(ttl=300)
def train_model(df, target):
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

# ===== Smart Solutions =====
def generate_solution(lang):
    if lang == 'en':
        return {
            'Name':'Cooling Diagnostic',
            'Details':'Run diagnostic on cooling system.',
            'Duration':'30 mins',
            'Priority':'High',
            'Effectiveness':'Very High'
        }
    return {
        'الاسم':'تشخيص التبريد',
        'التفاصيل':'فحص نظام التبريد.',
        'المدة':'30 دقيقة',
        'الأولوية':'عالية',
        'الفعالية':'عالية جداً'
    }

# ===== Menu =====
en = ['📊 Dashboard','🎛️ Simulation','📈 Predictive Analysis','🛠️ Smart Solutions','⚙️ Settings','ℹ️ About']
ar = ['📊 لوحة البيانات','🎛️ المحاكاة','📈 التحليل التنبؤي','🛠️ الحلول الذكية','⚙️ الإعدادات','ℹ️ حول']
menu = st.sidebar.radio(
    'Menu', en if st.session_state.lang=='en' else ar,
    format_func=lambda x: f"<div class='menu-item'>{x}</div>"
)

# ===== Settings =====
if menu == (en[4] if st.session_state.lang=='en' else ar[4]):
    st.markdown(
        f"<div class='main-title'>{'⚙️ Settings' if st.session_state.lang=='en' else '⚙️ الإعدادات'}</div>",
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button('English'):
            st.session_state.lang = 'en'
    with col2:
        if st.button('العربية'):
            st.session_state.lang = 'ar'

    st.markdown(
        '### Palettes' if st.session_state.lang=='en' else '### لوحات الألوان'
    )
    for name in PALETTES:
        if st.button(name):
            st.session_state.palette = name

    cont = st.slider(
        'Anomaly Sensitivity' if st.session_state.lang=='en' else 'حساسية كشف الشذوذ',
        0.01, 0.3, st.session_state.contamination, 0.01
    )
    st.session_state.contamination = cont

# ===== Pages =====
else:
    data = fetch_data()
    log_data(data)
    history = load_history()
    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
    
    # Dashboard
    if menu == en[0] or menu == ar[0]:
        st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        labels = ['Temperature','Pressure','Vibration','Gas'] if st.session_state.lang=='en' else ['درجة الحرارة','الضغط','الاهتزاز','الغاز']
        for i,key in enumerate(['temp','pressure','vibration','gas']):
            cols[i].metric(labels[i], f"{data[key]:.2f}")
        if not history.empty:
            fig = px.line(history, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
            fig.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0')
            st.plotly_chart(fig, use_container_width=True)

    # Simulation
    if menu == en[1] or menu == ar[1]:
        st.markdown("<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
        st.slider('Temperature (°C)' if st.session_state.lang=='en' else 'درجة الحرارة (°C)', 0, 100, int(data['temp']))
        st.map(pd.DataFrame({'lat':[24.7],'lon':[46.7]}))

    # Predictive Analysis
    if menu == en[2] or menu == ar[2]:
        st.markdown("<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
        if len(history) > 5:
            model = train_model(history, 'temp')
            if model:
                pred = model.predict([history[['temp','pressure','vibration','gas']].iloc[-5:].values.flatten()])
                st.write('Next Temp:', f"{pred[0]:.2f}")

    # Smart Solutions
    if menu == en[3] or menu == ar[3]:
        sol = generate_solution(st.session_state.lang)
        st.markdown("<div class='main-title'>🛠️ Smart Solutions</div>", unsafe_allow_html=True)
        st.table(sol)

    # About
    if menu == en[5] or menu == ar[5]:
        st.markdown("<div class='main-title'>ℹ️ About</div>", unsafe_allow_html=True)
        st.markdown("**Disasters don't wait... and neither do we. Predict. Prevent. Protect.**")
        st.markdown("Vision: Revolutionize industrial safety by transforming raw data into actionable insights.")
        st.markdown("Contact: rakan.almarri.2@aramco.com | 0532559664")
        st.markdown("Contact: abdulrahman.alzhrani.1@aramco.com | 0549202574")

# ===== Footer =====
st.markdown("<div style='text-align:center;color:#888;'>© Rakan Almarri & Abdulrahman Alzhrani</div>", unsafe_allow_html=True)
