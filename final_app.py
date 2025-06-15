import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import random

st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session defaults
if 'lang' not in st.session_state:
    st.session_state.lang = 'العربية'
if 'palette' not in st.session_state:
    st.session_state.palette = 'Ocean'

# Palette colors
PALETTE_COLORS = {
    'Ocean':  '#1B3B6F',
    'Forest': '#2E7D32',
    'Sunset': '#EF6C00',
    'Purple': '#6A1B9A',
    'Slate':  '#37474F'
}

# Apply background color
bg = PALETTE_COLORS.get(st.session_state.palette, '#FFFFFF')
st.markdown(f"""
<style>
    .stApp {{ background-color: {bg}; }}
</style>
""", unsafe_allow_html=True)

LANG = st.session_state.lang
# Translation helper
def T(ar, en): return ar if LANG=='العربية' else en

# Header
title = T('التوأم الرقمي الذكي','Smart Neural Digital Twin')
st.markdown(f"<h1 style='text-align:center;'>🧠 {title}</h1>", unsafe_allow_html=True)

# Page selection
pages_ar = ['الرئيسية','المحاكاة','التحليل التنبؤي','الحلول الذكية','الإعدادات','حول']
pages_en = ['Dashboard','Simulation','Predictive Analysis','Smart Solutions','Settings','About']
options = pages_ar if LANG=='العربية' else pages_en
page = st.radio('', options, index=0, horizontal=True, label_visibility='collapsed')
# Map to internal keys
key = pages_en[pages_ar.index(page)] if LANG=='العربية' else page

# Load data
def load_data():
    if os.path.exists('sensor_data_simulated.csv'):
        df = pd.read_csv('sensor_data_simulated.csv', parse_dates=['Time'], dayfirst=False)
        df = df.rename(columns={
            'Time':'timestamp',
            'Temperature (°C)':'temp',
            'Pressure (psi)':'pressure',
            'Vibration (g)':'vibration',
            'Methane (CH₄ ppm)':'gas',
            'H₂S (ppm)':'h2s'
        }).dropna(subset=['timestamp'])
    else:
        df = pd.DataFrame(columns=['timestamp','temp','pressure','vibration','gas','h2s'])
    return df
history = load_data()
latest = history.iloc[-1] if not history.empty else None

# Pages
if key=='Dashboard':
    st.header(T('الرئيسية','Dashboard'))
    if history.empty:
        st.info(T('لا توجد بيانات.','No data available.'))
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"🌡️ {T('الحرارة','Temperature')}", f"{latest.temp:.2f}°C")
        c2.metric(f"⚡ {T('الضغط','Pressure')}", f"{latest.pressure:.2f} psi")
        c3.metric(f"📳 {T('الاهتزاز','Vibration')}", f"{latest.vibration:.2f} g")
        c4.metric(f"🛢️ {T('الميثان','Methane')}", f"{latest.gas:.2f} ppm")
        st.markdown('---')
        st.subheader(T('📈 الاتجاهات','📈 Trends'))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.temp, name='Temp'))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.pressure, name='Pressure'))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.vibration, name='Vibration'))
        fig.add_trace(go.Scatter(x=history.timestamp, y=history.gas, name='Gas'))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(T('🌡️ خريطة الحرارة','🌡️ Heatmap'))
        history['hour']=history.timestamp.dt.hour
        history['day']=history.timestamp.dt.day
        heat=history.pivot_table(index='hour',columns='day',values='temp',aggfunc='mean')
        fig2=px.imshow(heat, labels={'x':T('اليوم','Day'),'y':T('الساعة','Hour'),'color':T('حرارة','Temp')})
        st.plotly_chart(fig2, use_container_width=True)

elif key=='Simulation':
    st.header(T('المحاكاة','Simulation'))
    if latest is None:
        st.info(T('لا توجد بيانات أساسية.','No baseline data.'))
    else:
        sd = latest.to_dict()
        sd['temp']=st.slider('Temperature (°C)',0.0,100.0,sd['temp'])
        sd['pressure']=st.slider('Pressure (psi)',0.0,200.0,sd['pressure'])
        sd['vibration']=st.slider('Vibration (g)',0.0,5.0,sd['vibration'])
        sd['gas']=st.slider('Methane (CH₄ ppm)',0.0,20.0,sd['gas'])
        sd['h2s']=st.slider('H₂S (ppm)',0.0,10.0,sd['h2s'])
        st.table(pd.DataFrame([sd]).T.rename(columns={0:T('القيمة','Value')}))

elif key=='Predictive Analysis':
    st.header(T('التحليل التنبؤي (72س)','Predictive Analysis (72h)'))
    if history.empty:
        st.info(T('لا توجد بيانات.','No data.'))
    else:
        last=history.timestamp.max()
        fut=pd.DataFrame({
            'timestamp':[last+timedelta(hours=i) for i in range(1,73)],
            'temp':np.linspace(latest.temp,latest.temp+random.uniform(-2,2),72)
        })
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=history.timestamp.tail(24),y=history.temp.tail(24),name='Actual'))
        fig3.add_trace(go.Scatter(x=fut.timestamp,y=fut.temp,name='Predicted',line=dict(dash='dash')))
        st.plotly_chart(fig3,use_container_width=True)

elif key=='Smart Solutions':
    st.header(T('الحلول الذكية','Smart Solutions'))
    if latest is None:
        st.info(T('لا توجد بيانات.','No data.'))
    else:
        if st.button(T('توليد حل','Generate Solution')):
            sol={
                T('الحل','Solution'):T('تفعيل التبريد','Activate cooling'),
                T('المدة','Duration'):'10m',
                T('الأولوية','Priority'):T('عالية','High')
            }
            st.table(pd.DataFrame([sol]).T.rename(columns={0:T('التفاصيل','Details')}))

elif key=='Settings':
    st.header(T('الإعدادات','Settings'))
    new_lang=st.radio(T('اختر اللغة','Choose Language'),['العربية','English'],index=0 if LANG=='العربية' else 1,horizontal=True)
    if new_lang!=LANG:
        st.session_state.lang=new_lang
        st.experimental_rerun()
    new_pal=st.radio(T('لوحة الألوان','Color Palette'),list(PALETTE_COLORS.keys()),index=list(PALETTE_COLORS.keys()).index(st.session_state.palette),horizontal=True)
    if new_pal!=st.session_state.palette:
        st.session_state.palette=new_pal
        st.experimental_rerun()

else:
    st.header(T('حول','About'))
    st.markdown(f"""
**{T("الكوارث لا تنتظر... ونحن أيضًا","Disasters don't wait... and neither do we.")}**

**{T('رؤيتنا: إحداث ثورة في السلامة الصناعية من خلال تحويل البيانات الخام إلى رؤى قابلة للتنفيذ.','Vision: Revolutionize industrial safety by turning raw data into actionable insights.')}**

**{T('الفريق','Team')}:**
- Rakan Almarri | rakan.almarri.2@aramco.com | 0532559664
- Abdulrahman Alzhrani | abdulrahman.alzhrani.1@aramco.com | 0549202574
""", unsafe_allow_html=True)
