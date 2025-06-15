import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# صفحة الإعدادات
st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")
if 'language' not in st.session_state:
    st.session_state.language = 'ar'
if 'palette' not in st.session_state:
    st.session_state.palette = 'Slate'

def _rerun():
    st.experimental_rerun()

def get_background_url(palette):
    urls = {
        'Ocean': 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e',
        'Forest': 'https://images.unsplash.com/photo-1501785888041-af3ef285b470',
        'Sunset': 'https://images.unsplash.com/photo-1518837695005-2083093ee35b',
        'Purple': 'https://images.unsplash.com/photo-1522661067900-22e2879de181',
        'Slate': 'https://images.unsplash.com/photo-1531959871807-30c29ab64749'
    }
    return urls.get(palette, '')

# الستايل الديناميكي
bg_url = get_background_url(st.session_state.palette)
st.markdown(f"""
<style>
body {{
    background: #111;
    background-image: url('{bg_url}');
    background-size: cover;
    background-attachment: fixed;
}}
.main-header {{
    background-color: rgba(0,0,0,0.5);
    padding: 1rem;
    border-radius: 8px;
}}
.menu-item {{
    margin-right: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# العنوان وتحديد الصفحة
st.markdown("<div class='main-header'><h1>🧠 Smart Neural Digital Twin</h1></div>", unsafe_allow_html=True)
page = st.radio(
    label="",
    options=['Dashboard', 'Simulation', 'Predictive Analysis', 'Smart Solutions', 'Settings', 'About'],
    format_func=lambda x: x if st.session_state.language=='en' else {'Dashboard':'لوحة البيانات','Simulation':'المحاكاة','Predictive Analysis':'التحليل التنبؤي','Smart Solutions':'الحلول الذكية','Settings':'الإعدادات','About':'حول'}[x],
    on_change=_rerun,
    key='page'
)

# إعدادات في صفحة Settings
if page=='Settings':
    lang = st.radio("اختر اللغة" if st.session_state.language=='ar' else "Select Language",
                     ('ar','en'),
                     format_func=lambda x: 'العربية' if x=='ar' else 'English',
                     index=0 if st.session_state.language=='ar' else 1,
                     on_change=_rerun,
                     key='language')
    st.session_state.language = lang
    pal = st.radio("لوحة الألوان" if lang=='ar' else "Palette",
                   ('Ocean','Forest','Sunset','Purple','Slate'),
                   index=list(('Ocean','Forest','Sunset','Purple','Slate')).index(st.session_state.palette),
                   on_change=_rerun,
                   key='palette')
    st.session_state.palette = pal

# صفحات أخرى
if page=='Dashboard':
    st.write("No data available")
elif page=='Simulation':
    sd = {'temp':34,'pressure':94,'vibration':0.34,'gas':3.45}
    sd['temp']=st.slider('Temperature (°C)',0,50,sd['temp'])
    sd['pressure']=st.slider('Pressure (kPa)',60,120,sd['pressure'])
    sd['vibration']=st.slider('Vibration (mm/s)',0.0,1.5,sd['vibration'])
    sd['gas']=st.slider('Gas (ppm)',0.0,10.0,sd['gas'])
    df = pd.DataFrame([sd])
    st.dataframe(df)
elif page=='Predictive Analysis':
    dates = pd.date_range(end=datetime.now(), periods=72, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': 37 + 0.1 * (pd.Series(range(72))%5),
        'pressure': 98 + 0.2 * (pd.Series(range(72))%7),
        'vibration': 0.5 + 0.01 * (pd.Series(range(72))%3)
    })
    fig = px.line(df, x='timestamp', y=['temperature','pressure','vibration'], title='Trends')
    st.plotly_chart(fig, use_container_width=True)
elif page=='Smart Solutions':
    if st.button('Generate Solution' if st.session_state.language=='en' else 'توليد الحل'):
        sol = {'Name':'تشخيص تبريد','Details':'فحص شامل','Duration':'30 دقيقة','Priority':'عالية','Effectiveness':'عالية جداً'}
        st.table(sol)
elif page=='About':
    txt = "Disasters don't wait..." if st.session_state.language=='en' else "الكوارث لا تنتظر..."
    st.write(txt)

# النهاية
st.markdown("© Rakan & Abdulrahman")
