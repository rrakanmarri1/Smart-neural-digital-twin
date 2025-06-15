import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest

def darken(hex_color: str, amount: float = 0.1) -> str:
    import colorsys
    c = hex_color.lstrip('#')
    r, g, b = [int(c[i:i+2], 16) for i in (0,2,4)]
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

st.set_page_config(page_title="🧠 Smart Neural Digital Twin", page_icon="🌐", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'contamination' not in st.session_state:
    st.session_state.contamination = 0.05
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {}

PALETTES = {
    'Ocean':  ['#1976D2','#0288D1','#26C6DA'],
    'Forest': ['#2E7D32','#388E3C','#66BB6A'],
    'Sunset': ['#EF5350','#FFA726','#FF7043'],
    'Purple':['#7E57C2','#8E24AA','#BA68C8'],
    'Slate': ['#455A64','#546E7A','#78909C']
}
primary, secondary, accent = PALETTES[st.session_state.palette]
sidebar_bg = f"linear-gradient(180deg,{primary},{secondary})"
main_bg    = darken(primary, 0.1)
box_bg     = darken(accent, 0.1)

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:{sidebar_bg}!important;color:white!important;}}
body{{background:#121212!important;color:#E0E0E0!important;}}
.main-title{{text-align:center;font-size:2.5rem;color:{primary};margin-bottom:1rem;}}
.section-box{{background:{main_bg};padding:1rem;margin:1rem 0;border-left:6px solid {secondary};border-radius:8px;}}
.button-box{{background:{box_bg};padding:0.5rem 1rem;border-radius:6px;margin:0.5rem 0;}}
</style>
""", unsafe_allow_html=True)

conn = sqlite3.connect('logs.db', check_same_thread=False)
conn.execute('CREATE TABLE IF NOT EXISTS logs(timestamp TEXT, temp REAL, pressure REAL, vibration REAL, gas REAL)')

@st.cache_data(ttl=300)
def load_history():
    csv_path = 'sensor_data_simulated.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        return df
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

history = load_history()

def fetch_data():
    return {
        'temp': round(np.random.normal(36, 2), 2),
        'pressure': round(np.random.normal(95, 5), 2),
        'vibration': round(np.random.normal(0.5, 0.1), 2),
        'gas': round(np.random.normal(5, 1), 2)
    }

def log_data(d):
    conn.execute(
        'INSERT INTO logs VALUES(?,?,?,?,?)',
        (datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas'])
    )
    conn.commit()

@st.cache_data(ttl=300)
def train_model(df, target='temp'):
    if len(df) < 10:
        return None
    df = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df)):
        row = df[['temp','pressure','vibration','gas']].iloc[i-5:i].values.flatten()
        X.append(row)
        y.append(df[target].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def generate_solution(lang):
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

st.sidebar.title('🌐 MENU')
page = st.sidebar.radio('Select Page', [
    'Dashboard', 'Simulation', 'Predictive Analysis', 'Smart Solutions', 'Settings', 'About'
])

if page == 'Dashboard':
    data = fetch_data()
    log_data(data)
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    labels = ['Temperature','Pressure','Vibration','Gas']
    for i, k in enumerate(['temp','pressure','vibration','gas']):
        cols[i].metric(labels[i], data[k])
    if not history.empty:
        iso = IsolationForest(contamination=st.session_state.contamination)
        history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
        fig = px.line(history, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
        st.plotly_chart(fig, use_container_width=True)

elif page == 'Simulation':
    st.markdown("<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
    sd = st.session_state.sim_data or fetch_data()
    sd['temp'] = st.slider('Temperature (°C)', 0, 100, float(sd['temp']))
    sd['pressure'] = st.slider('Pressure (kPa)', 0, 200, float(sd['pressure']))
    sd['vibration'] = st.slider('Vibration (mm/s)', 0.0, 5.0, float(sd['vibration']), step=0.01)
    sd['gas'] = st.slider('Gas (ppm)', 0.0, 10.0, float(sd['gas']), step=0.01)
    st.session_state.sim_data = sd
    st.map(pd.DataFrame({'lat': [24.7], 'lon': [46.7]}))
    st.write(sd)

elif page == 'Predictive Analysis':
    st.markdown("<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
    if os.path.exists('sensor_data_simulated.csv'):
        df = pd.read_csv('sensor_data_simulated.csv', parse_dates=['timestamp'])
        fig = go.Figure()
        for col, color in zip(['temp','pressure','vibration','gas'], PALETTES[st.session_state.palette]):
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=col.capitalize(), line=dict(color=color)))
        fig.update_layout(title='Historical Data (72h)', xaxis_title='Time', yaxis_title='Value', plot_bgcolor=main_bg, paper_bgcolor='#121212', font_color='#E0E0E0')
        st.plotly_chart(fig, use_container_width=True)
        heat = df.copy()
        heat['hour'] = heat['timestamp'].dt.hour
        heat['day'] = heat['timestamp'].dt.day
        pivot = heat.pivot(index='day', columns='hour', values='temp').fillna(method='ffill')
        fig2 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Inferno'))
        fig2.update_layout(title='Temperature Heatmap', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#E0E0E0')
        st.plotly_chart(fig2, use_container_width=True)
        if len(df) >= 10:
            model = train_model(df)
            if model:
                last = df[['temp','pressure','vibration','gas']].tail(5).values.flatten()
                pred = model.predict([last])[0]
                st.metric('Next Predicted Temperature', f"{pred:.2f} °C")
        else:
            st.info('Need at least 10 data points for prediction.')
    else:
        st.info('Place sensor_data_simulated.csv in the app folder for analysis.')

elif page == 'Smart Solutions':
    st.markdown("<div class='main-title'>🛠️ Smart Solutions</div>", unsafe_allow_html=True)
    sol = generate_solution(st.session_state.lang)
    if st.button('Generate Solution'):
        st.table(sol)

elif page == 'Settings':
    st.markdown("<div class='main-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    langs = ['en','ar']
    st.session_state.lang = st.selectbox('Language', langs, index=langs.index(st.session_state.lang))
    palettes = list(PALETTES.keys())
    st.session_state.palette = st.selectbox('Color Palette', palettes, index=palettes.index(st.session_state.palette))
    st.session_state.contamination = st.slider('Anomaly Sensitivity', 0.01, 0.3, st.session_state.contamination, 0.01)

else:
    st.markdown("<div class='main-title'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("**Disasters don't wait... and neither do we. Predict. Prevent. Protect.**")
    st.markdown("**Vision**: Revolutionize industrial safety by turning raw data into actionable insights.")
    st.markdown("- Real-time Monitoring  - Anomaly Detection  - Predictive Analytics  - Smart Recommendations")
    st.markdown("**Team**: Rakan Almarri & Abdulrahman Alzhrani")
    st.markdown("📧 rakan.almarri.2@aramco.com | 0532559664")
    st.markdown("📧 abdulrahman.alzhrani.1@aramco.com | 0549202574")

st.markdown("<div style='text-align:center;color:#888;'>© Rakan Almarri & Abdulrahman Alzhrani</div>", unsafe_allow_html=True)
