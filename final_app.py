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

# ===== Dark Theme CSS =====
st.markdown("""
<style>
body, .css-ffhzg2, .css-12oz5g7 {
    background-color: #121212 !important;
    color: #E0E0E0 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1, #1976D2) !important;
    color: white !important;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #BBDEFB !important;
    text-align: center;
    margin-bottom: 1rem;
}
.section-box {
    background: #1E1E1E;
    border-left: 6px solid #1976D2;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}
.download-btn {
    background-color: #1976D2; color: white; border:none; padding:0.5rem 1rem; border-radius:5px;
}
</style>
""", unsafe_allow_html=True)

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

# ===== Sensor Data Fetch =====
def fetch_data():
    return {
        'temp': float(np.random.normal(36,2)),
        'pressure': float(np.random.normal(95,5)),
        'vibration': float(np.random.normal(0.5,0.1)),
        'gas': float(np.random.normal(5,1))
    }

# ===== Logging =====
def log_data(d):
    c = conn.cursor()
    c.execute('INSERT INTO logs VALUES (?,?,?,?,?)', (
        datetime.now().isoformat(), d['temp'], d['pressure'], d['vibration'], d['gas']
    ))
    conn.commit()

@st.cache_data(ttl=300)
def load_history():
    df = pd.read_sql('SELECT * FROM logs', conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ===== Predictive Model =====
@st.cache_data(ttl=600)
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

# ===== Smart Solution =====
def generate_solution(lang):
    if lang=='en':
        return {
            'Name': 'Cooling System Check',
            'Details': 'Perform full diagnostic of cooling fans and coolant levels.',
            'Duration': '30 mins',
            'Priority': 'High',
            'Effectiveness': 'Very High'
        }
    return {
        'الاسم': 'فحص نظام التبريد',
        'التفاصيل': 'إجراء فحص شامل للمراوح ومستويات سائل التبريد.',
        'المدة': '30 دقيقة',
        'الأهمية': 'عالية',
        'الفعالية': 'عالية جداً'
    }

# ===== UI Elements =====
langs = ['English', 'العربية']
language = st.sidebar.selectbox("🌐 Language", langs)
labels = {
    'menu': ['📊 Dashboard','🎛️ Simulation','📈 Predictive','🛠️ Solutions','ℹ️ About'],
    'menu_ar': ['📊 لوحة البيانات','🎛️ المحاكاة','📈 التحليل التنبؤي','🛠️ الحلول الذكية','ℹ️ حول']
}
menu = st.sidebar.radio("Menu", labels['menu'] if language=='en' else labels['menu_ar'])

# ===== Data Flow =====
current = fetch_data()
log_data(current)
history = load_history()

# ===== Anomaly Detection =====
if len(history) >= 20:
    iso = IsolationForest(contamination=0.05)
    history['anomaly'] = iso.fit_predict(history[['temp','pressure','vibration','gas']])
else:
    history['anomaly'] = 1

# ===== Pages =====
# Dashboard
if menu.startswith('📊'):
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    keys = ['temp','pressure','vibration','gas']
    names = ['Temperature','Pressure','Vibration','Gas'] if language=='en' else ['درجة الحرارة','الضغط','الاهتزاز','الغاز']
    for i,k in enumerate(keys): cols[i].metric(names[i], f"{current[k]:.2f}")
    st.markdown("---")
    if not history.empty:
        fig = px.line(history, x='timestamp', y=keys, color='anomaly', labels={'timestamp':'Time','value':'Reading','anomaly':'Anomaly'})
        fig.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0')
        st.plotly_chart(fig, use_container_width=True)
        # Map
        st.subheader('📍 Sensor Locations' if language=='en' else '📍 مواقع المستشعرات')
        locs = pd.DataFrame([{'lat':26.369,'lon':50.133,'sensor':'S1'},{'lat':26.370,'lon':50.134,'sensor':'S2'}])
        m = px.scatter_mapbox(locs, lat='lat', lon='lon', hover_name='sensor', zoom=12)
        m.update_layout(mapbox_style='open-street-map', paper_bgcolor='#121212', font_color='#E0E0E0', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(m, use_container_width=True)
        # Heatmap
        st.subheader('📅 Temperature Heatmap' if language=='en' else '📅 خريطة حرارة')
        hist = history.copy(); hist['day']=hist['timestamp'].dt.day; hist['hour']=hist['timestamp'].dt.hour
        heat=hist.pivot('day','hour','temp').fillna(method='ffill')
        hfig=go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Viridis', colorbar=dict(title='Temp'))) 
        hfig.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0', xaxis_title='Hour', yaxis_title='Day')
        st.plotly_chart(hfig, use_container_width=True)
    else:
        st.info('Waiting for data...' if language=='en' else 'بانتظار البيانات...')
    # Export buttons
    if not history.empty:
        csv = history.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button('Download CSV' if language=='en' else 'تحميل CSV', csv, 'history.csv', 'text/csv', key='csv', help='Download full log')
        buf=io.BytesIO(); history.to_excel(buf,index=False); buf.seek(0)
        st.sidebar.download_button('Download Excel' if language=='en' else 'تحميل Excel', buf, 'history.xlsx', 'application/vnd.ms-excel', key='excel')

# Simulation
elif menu.startswith('🎛️'):
    st.markdown("<div class='main-title'>🎛️ Simulation</div>", unsafe_allow_html=True)
    sim = {}
    sim['temp']=st.slider('Temperature (°C)' if language=='en' else 'درجة الحرارة (°C)',20,50,int(current['temp']))
    sim['pressure']=st.slider('Pressure (kPa)' if language=='en' else 'الضغط (kPa)',60,120,int(current['pressure']))
    sim['vibration']=st.slider('Vibration (mm/s)' if language=='en' else 'الاهتزاز (mm/s)',0.0,1.5,float(current['vibration']),0.01)
    sim['gas']=st.slider('Gas (ppm)' if language=='en' else 'الغاز (ppm)',0.0,10.0,float(current['gas']),0.1)
    st.table(pd.DataFrame([sim]).T)

# Predictive Analysis
elif menu.startswith('📈'):
    st.markdown("<div class='main-title'>📈 Predictive Analysis</div>", unsafe_allow_html=True)
    model=train_model(history,'temp')
    if not model:
        st.warning('Not enough data.' if language=='en' else 'لا توجد بيانات كافية.')
    else:
        last=history[['temp','pressure','vibration','gas']].iloc[-5:].values.flatten().reshape(1,-1)
        future=[datetime.now()+timedelta(minutes=10*i) for i in range(1,11)]
        preds=model.predict(np.repeat(last,10,axis=0))
        pf=go.Figure(go.Scatter(x=future,y=preds,mode='lines+markers'))
        pf.update_layout(paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#E0E0E0', title='Forecast' if language=='en' else 'التوقع')
        st.plotly_chart(pf, use_container_width=True)

# Smart solutions
elif menu.startswith('🛠️'):
    st.markdown("<div class='main-title'>🛠️ Smart Solutions</div>", unsafe_allow_html=True)
    if st.button('Generate Solution' if language=='en' else 'توليد الحل'):
        sol=generate_solution(language[:2])
        df_sol=pd.DataFrame(sol, index=[0]).T
        st.table(df_sol)

# About our project
elif menu.startswith('ℹ️'):
    st.markdown("<div class='main-title'>ℹ️ About / حول</div>", unsafe_allow_html=True)
    if language=='en':
        st.write("**Team:** Rakan Almarri & Abdulrahman Alzhrani")
        st.write("**Contact:** rakan.almarri.2@aramco.com | +966532559664")
        st.write("**Contact:** abdulrahman.alzhrani.1@aramco.com | +966549202574")
        st.write("**Vision:** Revolutionize Aramco's safety with real-time insights.")
        st.write("**Features:** Live dashboard, anomaly detection, predictive modeling, solutions, bilingual UI.")
    else:
        st.write("**الفريق:** راكان المري & عبد الرحمن الزهراني")
        st.write("**التواصل:** rakan.almarri.2@aramco.com | 0532559664")
        st.write("**التواصل:** abdulrahman.alzhrani.1@aramco.com | 0549202574")
        st.write("**رؤيتنا:** إحداث ثورة في سلامة أرامكو من خلال رؤى آنية وتنبيهات استباقية.")
