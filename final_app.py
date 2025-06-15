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

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'
if 'contam' not in st.session_state:
    st.session_state.contam = 0.05
if 'sim' not in st.session_state:
    st.session_state.sim = {}

PALETTES = {
    'Ocean': ['#1976D2','#0288D1','#26C6DA'],
    'Forest': ['#2E7D32','#388E3C','#66BB6A'],
    'Sunset':['#EF5350','#FFA726','#FF7043'],
    'Purple':['#7E57C2','#8E24AA','#BA68C8'],
    'Slate': ['#455A64','#546E7A','#78909C']
}

def darken(hex_color: str, amount: float = 0.1) -> str:
    import colorsys
    c = hex_color.lstrip('#')
    r, g, b = [int(c[i:i+2], 16) for i in (0,2,4)]
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, l - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}"

def get_colors():
    pal = PALETTES[st.session_state.palette]
    return pal[0], pal[1], pal[2]

primary, secondary, accent = get_colors()
main_bg = darken(primary, 0.1)

st.markdown(f"""
<style>
[data-testid="stSidebar"] > div:first-child {{ background: linear-gradient(180deg, {primary}, {secondary}); }}
.main-title {{ font-size:2.5rem; color:{secondary}; text-align:center; margin-bottom:1rem; }}
.section-box {{ background:{main_bg}; padding:1rem; border-radius:8px; margin:1rem 0; }}
</style>
""", unsafe_allow_html=True)

db = sqlite3.connect('logs.db', check_same_thread=False)
db.execute('CREATE TABLE IF NOT EXISTS logs(ts TEXT,temp REAL,pressure REAL,vibration REAL,gas REAL)')

@st.cache_data
def load_history():
    # Try CSV data first
    csv_paths = ['sensor_data_simulated.csv', os.path.join(os.getcwd(), 'sensor_data_simulated.csv'), '/mnt/data/sensor_data_simulated.csv']
    for path in csv_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['timestamp'])
                return df
        except Exception:
            continue
    # Fallback to SQLite logs
    try:
        df = pd.read_sql_query(
            'SELECT ts AS timestamp, temp, pressure, vibration, gas FROM logs',
            con=db
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception:
        # no data
        return pd.DataFrame(columns=['timestamp','temp','pressure','vibration','gas'])

history = load_history()

def fetch():
    return {
        'temp': round(np.random.normal(36,2),2),
        'pressure': round(np.random.normal(95,5),2),
        'vibration': round(np.random.normal(0.5,0.1),3),
        'gas': round(np.random.normal(5,1),2)
    }

def log_data(d):
    db.execute('INSERT INTO logs VALUES(?,?,?,?,?)', (
        datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas']
    ))
    db.commit()

@st.cache_data
def train_model(df, target='temp'):
    if len(df) < 10:
        return None
    df_sorted = df.sort_values('timestamp')
    X, y = [], []
    for i in range(5, len(df_sorted)):
        window = df_sorted.iloc[i-5:i][['temp','pressure','vibration','gas']].values.flatten()
        X.append(window)
        y.append(df_sorted[target].iloc[i])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

@st.cache_data
def detect_anomalies(df):
    iso = IsolationForest(contamination=st.session_state.contam)
    df['anomaly'] = iso.fit_predict(df[['temp','pressure','vibration','gas']])
    return df

st.sidebar.header('Language')
st.session_state.lang = st.sidebar.radio('', ('English','العربية'))
st.sidebar.header('Palette')
st.session_state.palette = st.sidebar.radio('', list(PALETTES.keys()))
st.sidebar.header('Menu')
page = st.sidebar.radio('', ['Dashboard','Simulation','Predictive Analysis','Smart Solutions','Settings','About'])

if page == 'Dashboard':
    data = fetch()
    log_data(data)
    st.markdown('<div class="main-title">Smart Neural Digital Twin</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Temperature', f"{data['temp']}")
    c2.metric('Pressure', f"{data['pressure']}")
    c3.metric('Vibration', f"{data['vibration']}")
    c4.metric('Gas', f"{data['gas']}")
    if not history.empty:
        df_anom = detect_anomalies(history.copy())
        fig_line = px.line(df_anom, x='timestamp', y=['temp','pressure','vibration','gas'], color='anomaly')
        st.plotly_chart(fig_line, use_container_width=True)

elif page == 'Simulation':
    st.markdown('<div class="main-title">Simulation</div>', unsafe_allow_html=True)
    sim = st.session_state.sim or fetch()
    sim['temp'] = st.slider('Temperature (°C)', min_value=0.0, max_value=100.0, value=sim['temp'], step=0.1)
    sim['pressure'] = st.slider('Pressure (kPa)', 0.0, 200.0, sim['pressure'], 0.1)
    sim['vibration'] = st.slider('Vibration (mm/s)', 0.0, 5.0, sim['vibration'], 0.01)
    sim['gas'] = st.slider('Gas (ppm)', 0.0, 10.0, sim['gas'], 0.01)
    st.session_state.sim = sim
    st.map(pd.DataFrame({'lat':[24.7],'lon':[46.7]}))
    st.table(pd.DataFrame([sim]))

elif page == 'Predictive Analysis':
    st.markdown('<div class="main-title">Predictive Analysis</div>', unsafe_allow_html=True)
    df = history.copy()
    if df.empty:
        st.info('No data available')
    else:
        fig = go.Figure()
        for col, clr in zip(['temp','pressure','vibration','gas'], PALETTES[st.session_state.palette]):
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', line=dict(color=clr), name=col))
        st.plotly_chart(fig, use_container_width=True)
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        heat = df.pivot(index='day', columns='hour', values='temp').fillna(method='ffill')
        fig_heat = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Inferno'))
        st.plotly_chart(fig_heat, use_container_width=True)
        model = train_model(df)
        if model:
            last_window = df[['temp','pressure','vibration','gas']].tail(5).values.flatten()
            pred = model.predict([last_window])[0]
            st.metric('Next Temperature', f"{pred:.2f} °C")
        else:
            st.info('Need at least 10 data points for prediction')

elif page == 'Smart Solutions':
    st.markdown('<div class="main-title">Smart Solutions</div>', unsafe_allow_html=True)
    if st.button('Generate Solution'):
        sol = {'Name':'Cooling Check', 'Details':'Inspect cooling system and airflow','Duration':'30 min','Priority':'High','Effectiveness':'Very High'}
        st.table(pd.DataFrame([sol]))

elif page == 'Settings':
    st.markdown('<div class="main-title">Settings</div>', unsafe_allow_html=True)
    st.session_state.contam = st.slider('Anomaly Sensitivity', 0.01, 0.3, st.session_state.contam, 0.01)

else:
    st.markdown('<div class="main-title">About</div>', unsafe_allow_html=True)
    st.markdown("**Disasters don't wait... and neither do we. Predict. Prevent. Protect.**")
    st.markdown(f"Vision: Revolutionize industrial safety by turning raw data into actionable insights.")
    st.markdown('- Real-time Monitoring  •  Anomaly Detection  •  Predictive Analytics  •  Smart Recommendations')
    st.markdown('**Team:** Rakan Almarri & Abdulrahman Alzhrani')
    st.markdown('📧 rakan.almarri.2@aramco.com | 0532559664')
    st.markdown('📧 abdulrahman.alzhrani.1@aramco.com | 0549202574')

st.markdown("<div style='text-align:center;color:#888;'>© Rakan & Abdulrahman</div>", unsafe_allow_html=True)
