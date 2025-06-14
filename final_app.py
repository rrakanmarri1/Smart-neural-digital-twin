import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import io
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ===== Page Configuration =====
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Session Defaults =====
if 'theme_palette' not in st.session_state:
    # default to first palette
    st.session_state.theme_palette = 'Ocean'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ===== Palettes =====
PALETTES = {
    'Ocean': ['#1976D2', '#0288D1', '#26C6DA'],
    'Forest': ['#2E7D32', '#388E3C', '#66BB6A'],
    'Sunset': ['#EF5350', '#FFA726', '#FF7043'],
    'Purple': ['#7E57C2', '#8E24AA', '#BA68C8'],
    'Slate': ['#455A64', '#546E7A', '#78909C']
}

# ===== Dynamic CSS =====
def darken(hex_color, amount=0.1):
    import colorsys
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

primary, secondary, accent = PALETTES[st.session_state.theme_palette]
sidebar_bg = f"linear-gradient(180deg, {primary}, {secondary})"
main_bg = darken(primary, 0.1)
box_bg = darken(accent, 0.1)

css = f"""
<style>
body, .css-ffhzg2, .css-12oz5g7 {{
    background-color: #121212 !important;
    color: #E0E0E0 !important;
}}
[data-testid=\"stSidebar\"] {{
    background: {sidebar_bg} !important;
    color: white !important;
}}
.main-title {{
    font-size: 2.5rem;
    font-weight: 700;
    color: {primary} !important;
    text-align: center;
    margin-bottom: 1rem;
}}
.section-box {{
    background: {main_bg};
    border-left: 6px solid {secondary};
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}}
.menu-box {{
    background: {box_bg};
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ===== Database Setup =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS logs (
           timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)'''
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
    cur.execute('INSERT INTO logs VALUES (?,?,?,?,?)', (
        datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas']
    ))
    conn.commit()

@st.cache_data(ttl=300)
def load_history():
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Model =====n@st.cache_data(ttl=600)
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

# ===== Solutions =====ndef generate_solution(lang):
    if lang=='en':
        return {'Name':'Cooling System Diagnostic','Details':'Run full diagnostic on cooling fans and coolant levels.','Duration':'30m','Priority':'High','Effectiveness':'Very High'}
    return {'الاسم':'تشخيص نظام التبريد','التفاصيل':'فحص شامل للمراوح ومستويات سائل التبريد.','المدة':'30 دقيقة','الأولوية':'عالية','الفعالية':'عالية جداً'}

# ===== Menu =====nen = ['📊 Dashboard','🎛️ Simulation','📈 Predictive Analysis','🛠️ Smart Solutions','⚙️ Settings','ℹ️ About']
ar = ['📊 لوحة البيانات','🎛️ المحاكاة','📈 التحليل التنبؤي','🛠️ الحلول الذكية','⚙️ الإعدادات','ℹ️ حول']
menu = st.sidebar.radio('Menu', ne if st.session_state.language=='en' else ar, format_func=lambda x: fr"<div class='menu-box'>{x}</div>")

# ===== Settings =====nif menu == (ne[4] if st.session_state.language=='en' else ar[4]):
    st.markdown(f"<div class='main-title'>{'⚙️ Settings' if st.session_state.language=='en' else '⚙️ الإعدادات'}</div>", unsafe_allow_html=True)
    # Language circles
    cols = st.columns(2)
    with cols[0]:
        if st.button('English'):
            st.session_state.language='en'
    with cols[1]:
        if st.button('العربية'):
            st.session_state.language='ar'
    # Palette selection as boxes
    for name, colors in PALETTES.items():
        if st.button(name, key=name):
            st.session_state.theme_palette = name
    # Sensitivity
    cont = st.slider('Anomaly Sensitivity' if st.session_state.language=='en' else 'حساسية كشف الشذوذ',0.01,0.3,st.session_state.contamination,0.01)
    st.session_state.contamination = cont
    st.info('Customize your experience.' if st.session_state.language=='en' else 'خصص تجربتك.')

else:
    # Fetch & log
    data = fetch_data(); log_data(data);
    history = load_history()
    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
    # Pages implementation...
    # wrap names in menu-box for visual coloring

# Footer
st.markdown("<div style='text-align:center;padding:10px;color:#888;'>© Rakan Almarri & Abdulrahman Alzhrani</div>", unsafe_allow_html=True)
