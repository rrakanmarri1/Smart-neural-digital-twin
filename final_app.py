import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
from sklearn.ensemble import RandomForestRegressor, IsolationForest

st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- session defaults ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'ar'
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'
if 'contam' not in st.session_state:
    st.session_state.contam = 0.05
if 'sim' not in st.session_state:
    st.session_state.sim = {}

# --- color palettes ---
PALETTES = {
    'Ocean': ['#1976D2', '#0288D1', '#26C6DA'],
    'Forest': ['#2E7D32', '#388E3C', '#66BB6A'],
    'Sunset': ['#EF5350', '#FFA726', '#FF7043'],
    'Purple': ['#7E57C2', '#8E24AA', '#BA68C8'],
    'Slate': ['#455A64', '#546E7A', '#78909C']
}

def darken(color, amount=0.1):
    import colorsys
    c = color.lstrip('#')
    r, g, b = [int(c[i:i+2], 16) / 255 for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

p, s, a = PALETTES[st.session_state.palette]
bg2 = darken(p, 0.1)

# --- global CSS ---
st.markdown(f"""
<style>
body {{ background: #121212; color: #DDD; }}
[data-testid="stSidebar"] > div:first-child {{ background: {p}!important; }}
.main-header {{ background: linear-gradient(90deg,{p},{s}); padding: 1rem; border-radius: 0 0 10px 10px; color: white; display: flex; align-items: center; gap: 1rem; }}
.metric-card {{ background: {bg2}; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }}
.section-box {{ background: {bg2}; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.menu-item {{ margin: 0.5rem 0; padding: 0.5rem; border-radius: 6px; cursor: pointer; }}
.menu-item:hover {{ background: {s}; color: #000; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
  <span style="font-size:2rem;">🧠</span>
  <div>
    <h1 style="margin:0;color:#FFF;">Smart Neural Digital Twin</h1>
    <p style="margin:0;color:#EEE;font-size:0.9rem;">Transforming data into actionable insights</p>
  </div>
</div>
""", unsafe_allow_html=True)

# --- data storage ---
conn = sqlite3.connect('logs.db', check_same_thread=False)
conn.execute('CREATE TABLE IF NOT EXISTS logs(ts TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)')

def load_history():
    path = 'sensor_data_simulated.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['timestamp'])
    else:
        df = pd.DataFrame(columns=['timestamp','temp','pressure','vibration','gas'])
    return df

history = load_history()

def fetch_sensor():
    return {
        'temp': round(np.random.normal(36,2),2),
        'pressure': round(np.random.normal(95,5),2),
        'vibration': round(np.random.normal(0.5,0.1),3),
        'gas': round(np.random.normal(5,1),2)
    }

def train_model(df):
    if len(df) < 10:
        return None
    df2 = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df2)):
        window = df2[['temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten()
        X.append(window)
        y.append(df2['temp'].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def detect_anomalies(df):
    iso = IsolationForest(contamination=st.session_state.contam)
    df['anomaly'] = iso.fit_predict(df[['temp','pressure','vibration','gas']])
    return df

# --- sidebar ---
with st.sidebar:
    st.markdown("## MENU")
    pages = [
        '🏠 Dashboard',
        '🎮 Simulation',
        '📈 Predictive Analysis',
        '🛠️ Smart Solutions',
        '⚙️ Settings',
        'ℹ️ About'
    ]
    choice = st.radio("", pages, index=0)
    st.markdown("---")
    st.markdown("## Settings")
    st.session_state.lang = st.selectbox("Language", ['العربية','English'], index=0)
    st.session_state.palette = st.selectbox("Palette", list(PALETTES.keys()), index=list(PALETTES.keys()).index(st.session_state.palette))
    st.session_state.contam = st.slider("Sensitivity", 0.01, 0.3, st.session_state.contam, 0.01)
    st.markdown("---")
    st.markdown("© Rakan & Abdulrahman")

# --- pages ---
if choice == '🏠 Dashboard':
    st.subheader("Dashboard")
    if history.empty:
        st.info("No data available")
    else:
        df = detect_anomalies(history.copy())
        cols = st.columns(4)
        for i, col_name in enumerate(['temp','pressure','vibration','gas']):
            vals = df[col_name].tail(20)
            fig = go.Figure(go.Scatter(y=vals, mode='lines', line=dict(color=a, width=2)))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis={'visible':False}, yaxis={'visible':False}, height=60, paper_bgcolor='rgba(0,0,0,0)')
            cols[i].markdown(f"<div class='metric-card'><h3>{col_name.capitalize()}</h3><h2>{vals.iloc[-1]}</h2></div>", unsafe_allow_html=True)
            cols[i].plotly_chart(fig, use_container_width=True)
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        heat = df.pivot(index='day', columns='hour', values='temp').fillna(method='ffill')
        map_fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Inferno'))
        st.plotly_chart(map_fig, use_container_width=True)
        line_fig = px.line(df, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
        st.plotly_chart(line_fig, use_container_width=True)

elif choice == '🎮 Simulation':
    st.subheader("Simulation")
    sim = st.session_state.sim or fetch_sensor()
    sim['temp'] = st.slider("Temperature (°C)", 0.0, 100.0, sim['temp'], 0.1)
    sim['pressure'] = st.slider("Pressure (kPa)", 0.0, 200.0, sim['pressure'], 0.1)
    sim['vibration'] = st.slider("Vibration (mm/s)", 0.0, 5.0, sim['vibration'], 0.01)
    sim['gas'] = st.slider("Gas (ppm)", 0.0, 10.0, sim['gas'], 0.01)
    st.session_state.sim = sim
    st.map(pd.DataFrame({'lat':[24.7],'lon':[46.7]}))
    st.table(pd.DataFrame([sim]))

elif choice == '📈 Predictive Analysis':
    st.subheader("Predictive Analysis")
    df = history.copy()
    if df.empty:
        st.info("No data available")
    else:
        fig = go.Figure()
        for cname, clr in zip(['temp','pressure','vibration','gas'], PALETTES[st.session_state.palette]):
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[cname], mode='lines', name=cname, line=dict(color=clr)))
        st.plotly_chart(fig, use_container_width=True)
        model = train_model(df)
        if model:
            window = df[['temp','pressure','vibration','gas']].tail(5).values.flatten()
            pred = model.predict([window])[0]
            st.metric("Next Temperature", f"{pred:.2f}°C")

elif choice == '🛠️ Smart Solutions':
    st.subheader("Smart Solutions")
    if st.button("Generate Solution"):
        sol = {
            'Name': 'Cooling System Check',
            'Details': 'Inspect cooling & airflow',
            'Duration': '30m',
            'Priority': 'High',
            'Effectiveness': 'Very High'
        }
        st.table(pd.DataFrame([sol]))

elif choice == '⚙️ Settings':
    st.subheader("Settings")
    st.write("Use the sidebar to change language, palette, and sensitivity.")

else:
    st.subheader("About")
    st.markdown("**Disasters don’t wait... and neither do we. Predict. Prevent. Protect.**")
    st.markdown("**Vision:** Revolutionize industrial safety by turning raw data into actionable insights.")
    st.markdown("- Real-time Monitoring • Anomaly Detection • Predictive Analytics • Smart Recommendations")
    st.markdown("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com | 0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com | 0549202574")

st.markdown(
    "<div style='text-align:center;color:#555;'>© Rakan & Abdulrahman</div>",
    unsafe_allow_html=True
)
