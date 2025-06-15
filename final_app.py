import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# ===== Session Defaults =====
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'theme_palette' not in st.session_state:
    st.session_state.theme_palette = 'Ocean'

# ===== Color Palettes =====
PALETTES = {
    'Ocean':   ['#1976D2', '#0288D1', '#26C6DA'],
    'Forest':  ['#2E7D32', '#388E3C', '#66BB6A'],
    'Sunset':  ['#EF5350', '#FFA726', '#FF7043'],
    'Purple':  ['#7E57C2', '#8E24AA', '#BA68C8'],
    'Slate':   ['#455A64', '#546E7A', '#78909C']
}

# ===== Utility: darken color =====
def darken(hex_color, amount=0.1):
    import colorsys
    c = hex_color.lstrip('#')
    r, g, b = tuple(int(c[i:i+2], 16) for i in (0,2,4))
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

# ===== Apply Dynamic CSS =====
primary, secondary, accent = PALETTES[st.session_state.theme_palette]
sidebar_bg = f"linear-gradient(180deg, {primary}, {secondary})"
main_bg    = darken(primary, 0.1)
box_bg     = darken(accent,  0.1)

st.markdown(f"""
<style>
body, .css-ffhzg2, .css-12oz5g7 {{
    background-color: #121212 !important;
    color: #E0E0E0 !important;
}}
[data-testid="stSidebar"] {{
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
}}
.menu-item {{
    background: {box_bg};
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
}}
</style>
""", unsafe_allow_html=True)

# ===== Database Setup =====
def init_db():
    conn = sqlite3.connect('sensor_logs.db', check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS logs '
        '(timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)'
    )
    conn.commit()
    return conn

conn = init_db()

# ===== Data Simulation & Logging =====
def fetch_data():
    return {
        'temp':      float(np.random.normal(36, 2)),
        'pressure':  float(np.random.normal(95, 5)),
        'vibration': float(np.random.normal(0.5, 0.1)),
        'gas':       float(np.random.normal(5, 1))
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

# ===== Model Training =====
@st.cache_data(ttl=300)
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
            'Name': 'Cooling Diagnostic',
            'Details': 'Run diagnostic on cooling system components.',
            'Duration': '30 mins',
            'Priority': 'High',
            'Effectiveness': 'Very High'
        }
    return {
        'الاسم': 'تشخيص نظام التبريد',
        'التفاصيل': 'فحص شامل لمكونات نظام التبريد.',
        'المدة': '30 دقيقة',
        'الأولوية': 'عالية',
        'الفعالية': 'عالية جداً'
    }

# ===== Menu Labels & Selection =====
en = [
    '📊 Dashboard', '🎛️ Simulation', '📈 Predictive Analysis',
    '🛠️ Smart Solutions', '⚙️ Settings', 'ℹ️ About'
]
ar = [
    '📊 لوحة البيانات', '🎛️ المحاكاة', '📈 التحليل التنبؤي',
    '🛠️ الحلول الذكية', '⚙️ الإعدادات', 'ℹ️ حول'
]
menu = st.sidebar.radio(
    'Menu',
    en if st.session_state.language == 'en' else ar,
    format_func=lambda x: f"<div class='menu-item'>{x}</div>"
)

# ===== Settings Page =====
if menu == (en[4] if st.session_state.language=='en' else ar[4]):
    st.markdown(
        f"<div class='main-title'>"
        f"{'⚙️ Settings' if st.session_state.language=='en' else '⚙️ الإعدادات'}"
        f"</div>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button('English'):
            st.session_state.language = 'en'
    with c2:
        if st.button('العربية'):
            st.session_state.language = 'ar'

    st.markdown(
        '### Palettes' if st.session_state.language=='en' else '### لوحات الألوان'
    )
    for name in PALETTES.keys():
        if st.button(name):
            st.session_state.theme_palette = name

    cont = st.slider(
        'Anomaly Sensitivity' if st.session_state.language=='en' else 'حساسية كشف الشذوذ',
        0.01, 0.3, st.session_state.contamination, 0.01
    )
    st.session_state.contamination = cont
    st.info(
        'Customize your app experience.'
        if st.session_state.language=='en'
        else 'خصص تجربتك.'
    )

# ===== Content Pages =====
else:
    data = fetch_data()
    log_data(data)
    history = load_history()

    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(
            history[['temp','pressure','vibration','gas']]
        )
    else:
        history = pd.DataFrame(
            columns=['timestamp','temp','pressure','vibration','gas','anomaly']
        )

    # Dashboard
    if menu == (en[0] if st.session_state.language=='en' else ar[0]):
        st.markdown(
            "<div class='main-title'>🧠 Smart Neural Digital Twin</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        keys = ['temp','pressure','vibration','gas']
        labels = (
            ['Temperature','Pressure','Vibration','Gas']
            if st.session_state.language == 'en'
            else ['درجة الحرارة','الضغط','الاهتزاز','الغاز']
        )
        for i, k in enumerate(keys):
            cols[i].metric(labels[i], f"{data[k]:.2f}")
        st.markdown('---')
        if not history.empty:
            fig = px.line(history, x='timestamp', y=keys, color='anomaly')
            fig.update_layout(
                paper_bgcolor='#121212',
                plot_bgcolor='#121212',
                font_color='#E0E0E0'
            )
            st.plotly_chart(fig, use_container_width=True)

    # (Simulation, Predictive Analysis, Smart Solutions, About pages...)
    # يمكنك تعبئتها بنفس النمط أعلاه حال الحاجة

# ===== Footer =====
st.markdown(
    "<div style='text-align:center;padding:10px;color:#888;'>"
    "© Rakan Almarri & Abdulrahman Alzhrani"
    "</div>",
    unsafe_allow_html=True
)
