import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest

st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")

# Session state defaults
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'
if 'contam' not in st.session_state:
    st.session_state.contam = 0.05
if 'sim' not in st.session_state:
    st.session_state.sim = {}

# Color palettes
def get_palettes():
    return {
        'Ocean': ['#1976D2', '#0288D1', '#26C6DA'],
        'Forest': ['#2E7D32', '#388E3C', '#66BB6A'],
        'Sunset': ['#EF5350', '#FFA726', '#FF7043'],
        'Purple': ['#7E57C2', '#8E24AA', '#BA68C8'],
        'Slate': ['#455A64', '#546E7A', '#78909C']
    }
PALETTES = get_palettes()

# Darken color utility
def darken(color, amount=0.1):
    import colorsys
    c = color.lstrip('#')
    r, g, b = [int(c[i:i+2], 16)/255 for i in (0,2,4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

# Get theme colors
def get_colors():
    p, s, a = PALETTES[st.session_state.palette]
    return p, s, a, darken(p, 0.1)
primary, secondary, accent, bg_sec = get_colors()

# Styles
st.markdown(f"""
<style>
[data-testid="stSidebar"] > div:first-child {{ background: {primary}; }}
.main-header {{ background: linear-gradient(90deg, {primary}, {secondary}); padding:1rem; border-radius:0 0 10px 10px; color:white; display:flex; align-items:center; gap:1rem; }}
.metric-card {{ background:{bg_sec}; padding:1rem; border-radius:10px; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.1); }}
.section-box {{ background:{bg_sec}; padding:1rem; border-radius:8px; margin:1rem 0; }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="main-header">
    <span style="font-size:2rem;">🧠</span>
    <div>
        <h1 style="margin:0;">Smart Neural Digital Twin</h1>
        <p style="margin:0;">Transforming data into actionable insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Database setup
conn = sqlite3.connect('logs.db', check_same_thread=False)
conn.execute('CREATE TABLE IF NOT EXISTS logs(ts TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)')

@st.cache_data
def load_history():
    for path in ['sensor_data_simulated.csv', '/mnt/data/sensor_data_simulated.csv']:
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=['timestamp'])
    try:
        df = pd.read_sql_query('SELECT ts AS timestamp, temp, pressure, vibration, gas FROM logs', conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except:
        return pd.DataFrame(columns=['timestamp','temp','pressure','vibration','gas'])

history = load_history()

# Fetch and log functions

def fetch():
    return {
        'temp': round(np.random.normal(36, 2), 2),
        'pressure': round(np.random.normal(95, 5), 2),
        'vibration': round(np.random.normal(0.5, 0.1), 3),
        'gas': round(np.random.normal(5, 1), 2)
    }

def log_data(data):
    conn.execute('INSERT INTO logs VALUES(?,?,?,?,?)',
                 (datetime.now().isoformat(), data['temp'], data['pressure'], data['vibration'], data['gas']))
    conn.commit()

@st.cache_data
def train_model(df):
    if len(df) < 15:
        return None
    df_sorted = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df_sorted)):
        window = df_sorted.iloc[i-5:i][['temp','pressure','vibration','gas']].values.flatten()
        X.append(window)
        y.append(df_sorted['temp'].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

@st.cache_data
def detect_anomalies(df):
    iso = IsolationForest(contamination=st.session_state.contam)
    df['anomaly'] = iso.fit_predict(df[['temp','pressure','vibration','gas']])
    return df

# Sidebar
with st.sidebar:
    st.header('القائمة')
    menu = st.radio('', ['🏠 Dashboard', '🎮 Simulation', '📈 Predictive Analysis', '🛠️ Smart Solutions', '⚙️ Settings', 'ℹ️ About'])
    st.markdown('---')
    st.subheader('الإعدادات')
    st.session_state.lang = st.selectbox('اللغة', ['English', 'العربية'], index=0 if st.session_state.lang == 'en' else 1)
    st.session_state.palette = st.selectbox('لوحة الألوان', list(PALETTES.keys()), index=list(PALETTES.keys()).index(st.session_state.palette))
    st.session_state.contam = st.slider('حساسية التنبيه', 0.01, 0.3, st.session_state.contam, 0.01)
    st.markdown('© Rakan & Abdulrahman', unsafe_allow_html=True)

# Pages
if 'Dashboard' in menu:
    st.subheader('Dashboard')
    if history.empty:
        st.info('No data available')
    else:
        df = detect_anomalies(history.copy())
        cols = st.columns(4)
        for i, col in enumerate(['temp', 'pressure', 'vibration', 'gas']):
            vals = df[col].tail(20)
            spark = go.Figure(go.Scatter(y=vals, mode='lines', line=dict(color=accent, width=2)))
            spark.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=60, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            cols[i].markdown(f"<div class='metric-card'><h3>{col.capitalize()}</h3><h2>{vals.iloc[-1]}</h2></div>", unsafe_allow_html=True)
            cols[i].plotly_chart(spark, use_container_width=True)
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        heat = df.pivot(index='day', columns='hour', values='temp').fillna(method='ffill')
        fig_h = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Inferno'))
        st.plotly_chart(fig_h, use_container_width=True)
        fig_ts = px.line(df, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
        st.plotly_chart(fig_ts, use_container_width=True)

elif 'Simulation' in menu:
    st.subheader('Simulation')
    sim = st.session_state.sim or fetch()
    sim['temp'] = st.slider('Temperature (°C)', 0.0, 100.0, sim['temp'], 0.1)
    sim['pressure'] = st.slider('Pressure (kPa)', 0.0, 200.0, sim['pressure'], 0.1)
    sim['vibration'] = st.slider('Vibration (mm/s)', 0.0, 5.0, sim['vibration'], 0.01)
    sim['gas'] = st.slider('Gas (ppm)', 0.0, 10.0, sim['gas'], 0.01)
    st.session_state.sim = sim
    st.map(pd.DataFrame({'lat': [24.7], 'lon': [46.7]}))
    st.table(pd.DataFrame([sim]))

elif 'Predictive' in menu:
    st.subheader('Predictive Analysis')
    df = history.copy()
    if df.empty:
        st.info('No data available')
    else:
        fig = go.Figure()
        for col, clr in zip(['temp', 'pressure', 'vibration', 'gas'], PALETTES[st.session_state.palette]):
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=col, line=dict(color=clr)))
        st.plotly_chart(fig, use_container_width=True)
        model = train_model(df)
        if model:
            last_window = df[['temp','pressure','vibration','gas']].tail(5).values.flatten()
            pred = model.predict([last_window])[0]
            st.metric('Next Temperature', f"{pred:.2f}°C")

elif 'Smart' in menu:
    st.subheader('Smart Solutions')
    if st.button('Generate Solution'):
        sol = {'Name': 'Cooling System Check', 'Details': 'Inspect cooling system and airflow', 'Duration': '30 min', 'Priority': 'High', 'Effectiveness': 'Very High'}
        st.table(pd.DataFrame([sol]))

elif 'Settings' in menu:
    st.subheader('Settings')
    st.markdown('Language and appearance settings are available in the sidebar.')

else:
    st.subheader('About')
    st.markdown('**Disasters don’t wait... and neither do we. Predict. Prevent. Protect.**')
    st.markdown('Vision: Revolutionize industrial safety by turning raw data into actionable insights.')
    st.markdown('- Real-time Monitoring • Anomaly Detection • Predictive Analytics • Smart Recommendations')
    st.markdown('**Team:** Rakan Almarri & Abdulrahman Alzhrani')
    st.markdown('📧 rakan.almarri.2@aramco.com | 0532559664')
    st.markdown('📧 abdulrahman.alzhrani.1@aramco.com | 0549202574')

st.markdown("<div style='text-align:center;color:#888;'>© Rakan & Abdulrahman</div>", unsafe_allow_html=True)
