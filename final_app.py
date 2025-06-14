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
if 'theme_color' not in st.session_state:
    st.session_state.theme_color = '#1976D2'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ===== Dynamic CSS =====
primary = st.session_state.theme_color
css = f"""
<style>
body, .css-ffhzg2, .css-12oz5g7 {{
    background-color: #121212 !important;
    color: #E0E0E0 !important;
}}
[data-testid=\"stSidebar\"] {{
    background: linear-gradient(180deg, {primary} 0%, #003f7f 100%) !important;
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
    background: #1E1E1E;
    border-left: 6px solid {primary};
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ===== Database Setup =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
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


def log_data(data):
    cursor = conn.cursor()
    cursor.execute(
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

# ===== Model Training =====
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
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

# ===== Smart Solutions =====
def generate_solution(lang):
    if lang == 'en':
        return {
            'Name': 'Cooling System Diagnostic',
            'Details': 'Run full diagnostic on cooling fans and coolant levels.',
            'Duration': '30 minutes',
            'Priority': 'High',
            'Effectiveness': 'Very High'
        }
    else:
        return {
            'الاسم': 'تشخيص نظام التبريد',
            'التفاصيل': 'فحص شامل للمراوح ومستويات سائل التبريد.',
            'المدة': '30 دقيقة',
            'الأولوية': 'عالية',
            'الفعالية': 'عالية جداً'
        }

# ===== UI Setup =====
# Menu labels per language
en_menu = ['📊 Dashboard','🎛️ Simulation','📈 Predictive Analysis','🛠️ Smart Solutions','⚙️ Settings','ℹ️ About']
ar_menu = ['📊 لوحة البيانات','🎛️ المحاكاة','📈 التحليل التنبؤي','🛠️ الحلول الذكية','⚙️ الإعدادات','ℹ️ حول']
# Language & menu selection in Settings only

# Page Rendering
selected = st.session_state.language  # shorthand
menu = st.sidebar.radio(
    'Menu', en_menu if selected=='en' else ar_menu
)

if menu == (ar_menu[4] if selected=='ar' else en_menu[4]):  # Settings
    st.markdown(f"<div class='main-title'>{'⚙️ Settings' if selected=='en' else '⚙️ الإعدادات'}</div>", unsafe_allow_html=True)
    # Language selection
    lang = st.selectbox('Select Language' if selected=='en' else 'اختر اللغة', ['English','العربية'], index=0 if selected=='en' else 1)
    st.session_state.language = 'en' if lang=='English' else 'ar'
    # Palette selection
    palettes = {
        'Blue':'#1976D2','Teal':'#00897B','Purple':'#7E57C2','Orange':'#FFA726','Red':'#EF5350'
    }
    p_choice = st.selectbox('Accent Color' if selected=='en' else 'لون التمييز', list(palettes.keys()), index=list(palettes.values()).index(st.session_state.theme_color))
    st.session_state.theme_color = palettes[p_choice]
    # Anomaly sensitivity
    cont = st.slider('Anomaly Sensitivity' if selected=='en' else 'حساسية كشف الشذوذ', 0.01, 0.3, st.session_state.contamination, 0.01)
    st.session_state.contamination = cont
    st.info('Use these settings to customize the app.' if selected=='en' else 'استخدم هذه الإعدادات لتخصيص التطبيق.')
else:
    # Fetch, log, and analyze
    data = fetch_data()
    log_data(data)
    history = load_history()
    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
    else:
        history = pd.DataFrame(columns=['timestamp','temp','pressure','vibration','gas','anomaly'])

    if menu == (en_menu[0] if selected=='en' else ar_menu[0]):
        st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        keys = ['temp','pressure','vibration','gas']
        names = ['Temperature','Pressure','Vibration','Gas'] if selected=='en' else ['درجة الحرارة','الضغط','الاهتزاز','الغاز']
        for i,k in enumerate(keys): cols[i].metric(names[i], f"{data[k]:.2f}")
        st.markdown("---")
        if not history.empty:
            fig = px.line(history, x='timestamp', y=keys, color='anomaly')
            fig.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0')
            st.plotly_chart(fig, use_container_width=True)
    # ... other pages omitted for brevity
