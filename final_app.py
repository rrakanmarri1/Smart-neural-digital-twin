import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ===== Configuration =====
st.set_page_config(page_title="🧠 Smart Neural Digital Twin", page_icon="🌐", layout="wide")
if 'lang' not in st.session_state: st.session_state.lang='en'
if 'contamination' not in st.session_state: st.session_state.contamination=0.05
if 'palette' not in st.session_state: st.session_state.palette='Ocean'

# ===== Palettes =====
PALETTES={
    'Ocean': ['#1976D2','#0288D1','#26C6DA'],
    'Forest': ['#2E7D32','#388E3C','#66BB6A'],
    'Sunset': ['#EF5350','#FFA726','#FF7043'],
}
colors=PALETTES[st.session_state.palette]

# ===== Sidebar =====
st.sidebar.markdown("## 🌐 Language")
if st.sidebar.button('English'): st.session_state.lang='en'
elif st.sidebar.button('العربية'): st.session_state.lang='ar'

menu_labels_en=['Dashboard','Simulation','Predictive Analysis','Smart Solutions','Settings','About']
menu_labels_ar=['الرئيسية','المحاكاة','التحليل التنبؤي','الحلول الذكية','الإعدادات','حول']
menu=st.sidebar.radio("📋 Menu", menu_labels_en if st.session_state.lang=='en' else menu_labels_ar)

# ===== DB =====ndef init_db():
    conn=sqlite3.connect('logs.db',check_same_thread=False)
    cur=conn.cursor(); cur.execute('CREATE TABLE IF NOT EXISTS logs(timestamp TEXT,temp REAL,pressure REAL,vibration REAL,gas REAL)'); conn.commit(); return conn
conn=init_db()

@st.cache_data(ttl=300)
def load_history():
    df=pd.read_sql('SELECT * FROM logs',conn)
    if not df.empty: df['timestamp']=pd.to_datetime(df['timestamp'])
    return df

def fetch_data():
    return {k:float(np.random.normal(mu,sigma)) for k,mu,sigma in [('temp',36,2),('pressure',95,5),('vibration',0.5,0.1),('gas',5,1)]}

def log_data(d):
    cur=conn.cursor(); cur.execute('INSERT INTO logs VALUES(?,?,?,?,?)', (datetime.now().isoformat(),d['temp'],d['pressure'],d['vibration'],d['gas'])); conn.commit()

# ===== Pages =====
if menu==menu_labels_en[0] or menu==menu_labels_ar[0]:
    # Dashboard
    data=fetch_data(); log_data(data)
    cols=st.columns(4)
    keys=['temp','pressure','vibration','gas']
    labels_en=['Temperature','Pressure','Vibration','Gas']
    labels_ar=['درجة الحرارة','الضغط','الاهتزاز','الغاز']
    labels=labels_en if st.session_state.lang=='en' else labels_ar
    for i,k in enumerate(keys): cols[i].metric(labels[i],f"{data[k]:.2f}")
    hist=load_history()
    if not hist.empty:
        iso=IsolationForest(contamination=st.session_state.contamination)
        hist['anomaly']=iso.fit_predict(hist[keys])
        fig=px.line(hist,x='timestamp',y=keys,color='anomaly', title=labels[0]+' & '+labels[1]+' & ...')
        st.plotly_chart(fig,use_container_width=True)
elif menu==menu_labels_en[1] or menu==menu_labels_ar[1]:
    # Simulation
    st.header('🎛️ '+menu if st.session_state.lang=='en' else '🎛️ '+menu)
    d=fetch_data(); log_data(d)
    st.slider('Temperature (°C)' if st.session_state.lang=='en' else 'درجة الحرارة',0,100,int(d['temp']))
    st.map(pd.DataFrame({'lat':[24.7],'lon':[46.7]}))
elif menu==menu_labels_en[2] or menu==menu_labels_ar[2]:
    # Predictive
    st.header('📈 '+menu)
    hist=load_history()
    if len(hist)>10:
        mdl=RandomForestRegressor().fit(hist[['temp','pressure','vibration','gas']].iloc[:-1], hist['temp'].iloc[1:])
        pred=mdl.predict([hist[['temp','pressure','vibration','gas']].iloc[-1]])
        st.write('Next temp:',pred[0])
elif menu==menu_labels_en[3] or menu==menu_labels_ar[3]:
    # Solutions
    sol = {'Name':'Cooling Diagnostic','Details':'...','Duration':'30m','Priority':'High','Effectiveness':'High'}
    st.header('🛠️ '+menu)
    st.table(sol)
elif menu==menu_labels_en[4] or menu==menu_labels_ar[4]:
    # Settings
    st.header('⚙️ '+menu)
    pal=st.selectbox('Palette', list(PALETTES.keys()))
    st.session_state.palette=pal
    cont=st.slider('Sensitivity',0.01,0.3,st.session_state.contamination)
    st.session_state.contamination=cont
elif menu==menu_labels_en[5] or menu==menu_labels_ar[5]:
    # About
    st.header('ℹ️ About')
    st.markdown("**Smart Neural Digital Twin** prototype by Rakan & Abdulrahman.")
    st.markdown("Disasters don't wait... and neither do we. Predict. Prevent. Protect.")
    st.markdown("Vision: Revolutionize industrial safety by transforming raw data into actionable insights.")
    st.write('Contact: rakan.almarri.2@aramco.com | 0532559664')
    st.write('Contact: abdulrahman.alzhrani.1@aramco.com | 0549202574')
