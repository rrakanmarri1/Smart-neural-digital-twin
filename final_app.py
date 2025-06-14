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

# ===== Session Defaults =====n# Default theme color
if 'theme_color' not in st.session_state:
    st.session_state.theme_color = '#1976D2'
# Default anomaly contamination
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
# Default language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ===== Dynamic CSS =====nprimary = st.session_state.theme_color
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

# ===== Database Setup =====ndef init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS logs (
           timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)'''
    )
    conn.commit()
    return conn

conn = init_db()

# ===== Data Functions =====ndef fetch_data():
    return {
        'temp': float(np.random.normal(36, 2)),
        'pressure': float(np.random.normal(95, 5)),
        'vibration': float(np.random.normal(0.5, 0.1)),
        'gas': float(np.random.normal(5, 1))
    }
n
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

# ===== Model Training =====ndef train_model(df, target):
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

# ===== Smart Solutions =====ndef generate_solution(lang):
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

# ===== Sidebar Menu =====ndef get_menu_labels(lang):
    en = ['📊 Dashboard','🎛️ Simulation','📈 Predictive Analysis','🛠️ Smart Solutions','⚙️ Settings','ℹ️ About']
    ar = ['📊 لوحة البيانات','🎛️ المحاكاة','📈 التحليل التنبؤي','🛠️ الحلول الذكية','⚙️ الإعدادات','ℹ️ حول']
    return en if lang=='en' else ar

menu = st.sidebar.radio("Menu", get_menu_labels(st.session_state.language))

# ===== Settings Page =====nif menu == (get_menu_labels(st.session_state.language)[4]):
    st.markdown(f"<div class='main-title'>{'⚙️ Settings' if st.session_state.language=='en' else '⚙️ الإعدادات'}</div>", unsafe_allow_html=True)
    # Language
    lang = st.selectbox('Language / اللغة', ['English','العربية'], index=0 if st.session_state.language=='en' else 1)
    st.session_state.language = 'en' if lang=='English' else 'ar'
    # Theme Color Picker
    color = st.color_picker('Accent Color', st.session_state.theme_color)
    st.session_state.theme_color = color
    # Anomaly sensitivity
    cont = st.slider('Anomaly Sensitivity' if st.session_state.language=='en' else 'حساسية كشف الشذوذ', 0.01, 0.3, st.session_state.contamination, 0.01)
    st.session_state.contamination = cont
    st.info('Adjust to fine-tune anomaly detection.' if st.session_state.language=='en' else 'اضبط هذه القيمة لضبط كشف الشذوذ.')
else:
    # Fetch & Log
    data = fetch_data()
    log_data(data)
    history = load_history()
    # Anomaly detection
    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
    else:
        history['anomaly'] = 1
    # Pages logic follows...
    # ... (rest of dashboard, simulation, predictive, solutions, about as before)
