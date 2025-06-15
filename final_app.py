import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ====== Page Config ======
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Session State Defaults ======
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {}

# ====== Color Palettes ======
PALETTES = {
    'Ocean':  ['#1976D2', '#0288D1', '#26C6DA'],
    'Forest': ['#2E7D32', '#388E3C', '#66BB6A'],
    'Sunset': ['#EF5350', '#FFA726', '#FF7043'],
    'Purple': ['#7E57C2', '#8E24AA', '#BA68C8'],
    'Slate':  ['#455A64', '#546E7A', '#78909C']
}
primary, secondary, accent = PALETTES[st.session_state.palette]

# ====== Utility to darken color ======
def darken(color: str, amount: float = 0.1) -> str:
    import colorsys
    c = color.lstrip('#')
    r, g, b = [int(c[i:i+2], 16) for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

sidebar_bg = f"linear-gradient(180deg, {primary}, {secondary})"
main_bg    = darken(primary, 0.1)
box_bg     = darken(accent, 0.1)

# ====== Dynamic CSS ======
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background: {sidebar_bg} !important; color: white !important; }}
body, .css-ffhzg2 {{ background: #121212 !important; color: #E0E0E0 !important; }}
.main-title {{ text-align: center; font-size: 2.5rem; color: {primary}; margin-bottom: 1rem; }}
.section-box {{ background: {main_bg}; border-left: 6px solid {secondary}; padding: 1rem; margin: 1rem 0; border-radius: 8px; }}
.menu-item {{ background: {box_bg}; padding: 0.5rem 1rem; border-radius: 6px; margin: 0.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# ====== Database Setup ======
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect('logs.db', check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)'
    )
    conn.commit()
    return conn

conn = init_db()

@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ====== Data Functions ======
def fetch_data() -> dict:
    return {
        'temp': round(np.random.normal(36, 2), 2),
        'pressure': round(np.random.normal(95, 5), 2),
        'vibration': round(np.random.normal(0.5, 0.1), 2),
        'gas': round(np.random.normal(5, 1), 2)
    }

def log_data(d: dict) -> None:
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO logs VALUES (?,?,?,?,?)',
        (datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas'])
    )
    conn.commit()

# ====== Prediction Model ======
@st.cache_data(ttl=300)
def train_model(df: pd.DataFrame, target: str = 'temp') -> RandomForestRegressor | None:
    if len(df) < 10:
        return None
    df = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df)):
        X.append(df[['temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten())
        y.append(df[target].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

# ====== Smart Solutions ======
def generate_solution(lang: str) -> dict:
    if lang == 'en':
        return {
            'Name': 'Cooling Diagnostic',
            'Details': 'Run comprehensive diagnostic on cooling system.',
            'Duration': '30 mins',
            'Priority': 'High',
            'Effectiveness': 'Very High'
        }
    return {
        'الاسم': 'تشخيص التبريد',
        'التفاصيل': 'فحص شامل لنظام التبريد.',
        'المدة': '30 دقيقة',
        'الأولوية': 'عالية',
        'الفعالية': 'عالية جداً'
    }

# ====== Sidebar Controls ======
st.sidebar.markdown('## 🌐 Language')
lang = st.sidebar.radio('', ['English','العربية'])
st.session_state.lang = 'en' if lang == 'English' else 'ar'

MENU_EN = ['Dashboard','Simulation','Predictive Analysis','Smart Solutions','Settings','About']
MENU_AR = ['الرئيسية','المحاكاة','التحليل التنبؤي','الحلول الذكية','الإعدادات','حول']
menu = st.sidebar.radio(
    '📋 Menu',
    MENU_EN if st.session_state.lang == 'en' else MENU_AR,
    format_func=lambda x: f"<div class='menu-item'>{x}</div>"
)

hist = load_history()

# ====== Pages ======
if menu == (MENU_EN[0] if st.session_state.lang=='en' else MENU_AR[0]):
    # Dashboard
    data = fetch_data(); log_data(data)
    st.markdown(f"<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    labels = (['Temperature','Pressure','Vibration','Gas']
              if st.session_state.lang=='en'
              else ['درجة الحرارة','الضغط','الاهتزاز','الغاز'])
    for i, k in enumerate(['temp','pressure','vibration','gas']):
        cols[i].metric(labels[i], data[k])
    if not hist.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        hist['anomaly'] = iso.fit_predict(hist[['temp','pressure','vibration','gas']])
        fig = px.line(hist, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
        fig.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0')
        st.plotly_chart(fig, use_container_width=True)

elif menu == (MENU_EN[1] if st.session_state.lang=='en' else MENU_AR[1]):
    # Simulation
    st.markdown(f"<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
    sd = st.session_state.sim_data or fetch_data()
    sd['temp'] = st.slider('Temperature (°C)' if st.session_state.lang=='en' else 'درجة الحرارة (°C)', 0, 100, int(sd['temp']))
    sd['pressure'] = st.slider('Pressure (kPa)' if st.session_state.lang=='en' else 'الضغط (kPa)', 0, 200, int(sd['pressure']))
    sd['vibration'] = st.slider('Vibration (mm/s)' if st.session_state.lang=='en' else 'الاهتزاز (mm/s)', 0.0, 5.0, float(sd['vibration']))
    sd['gas'] = st.slider('Gas (ppm)' if st.session_state.lang=='en' else 'الغاز (ppm)', 0.0, 10.0, float(sd['gas']))
    st.session_state.sim_data = sd
    st.map(pd.DataFrame({'lat': [24.7], 'lon': [46.7]}))
    st.write(sd)

elif menu == (MENU_EN[2] if st.session_state.lang=='en' else MENU_AR[2]):
    # Predictive Analysis
    st.markdown(f"<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
    if len(hist) > 10:
        model = train_model(hist, 'temp')
        if model:
            last = hist[['temp','pressure','vibration','gas']].tail(5).values.flatten()
            pred = model.predict([last])[0]
            st.write(('Next predicted temp:' if st.session_state.lang=='en' else 'التنبؤ التالي:'), round(pred, 2))
    else:
        st.info(('Need at least 10 data points.' if st.session_state.lang=='en' else 'تحتاج 10 نقاط بيانات على الأقل.'))

elif menu == (MENU_EN[3] if st.session_state.lang=='en' else MENU_AR[3]):
    # Smart Solutions
    sol = generate_solution(st.session_state.lang)
    st.markdown(f"<div class='main-title'>🛠️ Smart Solutions</div>", unsafe_allow_html=True)
    if st.button(('Generate Solution' if st.session_state.lang=='en' else 'توليد الحل')):
        st.table(sol)

elif menu == (MENU_EN[4] if st.session_state.lang=='en' else MENU_AR[4]):
    # Settings
    st.markdown(f"<div class='main-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    pal = st.selectbox(('Palette' if st.session_state.lang=='en' else 'لوحة الألوان'), list(PALETTES.keys()))
    st.session_state.palette = pal
    cont = st.slider(('Anomaly Sensitivity' if st.session_state.lang=='en' else 'حساسية كشف الشذوذ'), 
                     0.01, 0.3, st.session_state.contamination, 0.01)
    st.session_state.contamination = cont

else:
    # About
    st.markdown(f"<div class='main-title'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("**Disasters don't wait... and neither do we. Predict. Prevent. Protect.**")
    st.markdown("Vision: Revolutionize industrial safety by turning raw data into actionable insights.")
    features_text = ("- Real-time Monitoring  - Anomaly Detection  - Predictive Analytics  - Smart Recommendations"
                     if st.session_state.lang=='en' else
                     "- المراقبة الفورية  - اكتشاف الشذوذ  - التحليل التنبؤي  - التوصيات الذكية")
    st.markdown(features_text)
    st.markdown("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com  |  0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com  |  0549202574")

# ====== Footer ======
st.markdown("<div style='text-align:center;color:#888;'>© Rakan Almarri & Abdulrahman Alzhrani</div>", unsafe_allow_html=True)
