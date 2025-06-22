import streamlit as st import pandas as pd

Page configuration must be first Streamlit call

st.set_page_config( page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide" )

@st.cache_data def load_data(): df = pd.read_csv('sensor_data_simulated.csv') df.rename(columns={ 'Timestamp': 'timestamp', 'Temperature (°C)': 'temperature', 'Pressure (psi)': 'pressure', 'Vibration (g)': 'vibration', 'Methane (CH₄, ppm)': 'methane', 'H₂S (ppm)': 'h2s' }, inplace=True) df['timestamp'] = pd.to_datetime(df['timestamp']) return df

df = load_data()

Initialize session state

if 'lang' not in st.session_state: st.session_state.lang = 'ar' if 'theme' not in st.session_state: st.session_state.theme = 'Ocean'

lang = st.session_state.lang theme = st.session_state.theme

Menus

en_menu = ['Dashboard', 'Predictive Analysis', 'Sensor Map', 'Smart Solutions', 'Settings', 'About'] ar_menu = ['لوحة البيانات', 'التحليل التنبؤي', 'خريطة الحساسات', 'الحلول الذكية', 'الإعدادات', 'حول'] menu = ar_menu if lang == 'ar' else en_menu

Theme palettes

themes = { 'Ocean':  {'bg': '#13375b', 'accent': '#00b4d8'}, 'Forest': {'bg': '#184d47', 'accent': '#81b214'}, 'Sunset': {'bg': '#ce6a85', 'accent': '#ffb86b'}, 'Purple': {'bg': '#3e206d', 'accent': '#b983ff'}, 'Slate':  {'bg': '#222c36', 'accent': '#e0e0e0'} } colors = themes.get(theme, themes['Ocean'])

CSS styling

st.markdown(f"""

<style>
    .stApp {{ background: {colors['bg']} !important; color: #fff !important; }}
    .main-header {{ font-size:2.2rem; font-weight:bold; color:{colors['accent']}; margin-bottom:0.5rem; }}
    .box {{ background:#202a34; border-radius:12px; padding:1rem; margin-bottom:1.5rem; }}
    .solution {{ background:#282828; border-left:6px solid {colors['accent']}; padding:0.8rem; margin-bottom:1rem; border-radius:8px; }}
    .settings {{ background:#222c36; padding:1rem; border-radius:8px; margin-bottom:1rem; }}
    .sidebar .sidebar-content {{ background:{colors['bg']} !important; color:#fff !important; }}
    .stButton>button {{ border-radius:15px !important; font-weight:bold; }}
</style>""", unsafe_allow_html=True)

Sidebar navigation

with st.sidebar: st.markdown(f"<div class='main-header'>{'🧠 التوأم الرقمي الذكي' if lang=='ar' else '🧠 Smart Neural Digital Twin'}</div>", unsafe_allow_html=True) choice = st.radio("", menu)

Dashboard

if choice == menu[0]: st.markdown(f"<div class='main-header'>{menu[0]}</div>", unsafe_allow_html=True) latest = df.iloc[-1] cols = st.columns(5) cols[0].metric(menu[0] if lang=='en' else 'درجة الحرارة', f"{latest.temperature:.2f} °C") cols[1].metric(menu[0] if lang=='en' else 'الضغط', f"{latest.pressure:.2f} psi") cols[2].metric(menu[0] if lang=='en' else 'الاهتزاز', f"{latest.vibration:.2f} g") cols[3].metric(menu[0] if lang=='en' else 'الميثان', f"{latest.methane:.2f} ppm") cols[4].metric('H₂S', f"{latest.h2s:.2f} ppm") st.line_chart(df[['temperature','pressure','vibration','methane','h2s']].tail(72))

Predictive Analysis

elif choice == menu[1]: st.markdown(f"<div class='main-header'>{menu[1]}</div>", unsafe_allow_html=True) st.line_chart(df[['temperature','pressure']].tail(72))

Sensor Map

elif choice == menu[2]: st.markdown(f"<div class='main-header'>{menu[2]}</div>", unsafe_allow_html=True) if 'lat' in df.columns and 'lon' in df.columns: st.map(df.rename(columns={'lat':'latitude','lon':'longitude'})) else: st.info('لا توجد إحداثيات' if lang=='ar' else 'No coordinates available')

Smart Solutions

elif choice == menu[3]: st.markdown(f"<div class='main-header'>{menu[3]}</div>", unsafe_allow_html=True) if st.button('🔍 توليد حل' if lang=='ar' else '🔍 Generate Solution'): st.markdown(f""" <div class='solution'> <b>{'الحل المقترح:' if lang=='ar' else 'Suggested Solution:'}</b><br> {'افحص الصمامات فورًا' if lang=='ar' else 'Inspect safety valves immediately.'}<br> <b>⏳ {'المدة المتوقعة' if lang=='ar' else 'Estimated Duration'}:</b> 15 min<br> <b>⭐ {'الأولوية' if lang=='ar' else 'Priority'}:</b> {'حرجة' if lang=='ar' else 'Critical'}<br> <b>📊 {'الفعالية' if lang=='ar' else 'Effectiveness'}:</b> 95% </div> """, unsafe_allow_html=True)

Settings

elif choice == menu[4]: st.markdown(f"<div class='main-header'>{menu[4]}</div>", unsafe_allow_html=True) st.markdown("<div class='settings'>", unsafe_allow_html=True)

# Theme selection labels
theme_keys = list(themes.keys())
theme_labels_ar = ['المحيط','الغابة','الغروب','أرجواني','رصاصي']
theme_labels_en = theme_keys
# Language labels
lang_keys = ['ar','en']
lang_labels_ar = ['العربية','الإنجليزية']
lang_labels_en = ['English','Arabic']

if lang == 'ar':
    sel_theme_label = st.selectbox('اختيار الثيم', theme_labels_ar, index=theme_keys.index(theme))
    sel_theme = theme_keys[theme_labels_ar.index(sel_theme_label)]
else:
    sel_theme_label = st.selectbox('Select Theme', theme_labels_en, index=theme_keys.index(theme))
    sel_theme = sel_theme_label

if sel_theme != theme:
    st.session_state.theme = sel_theme
    st.experimental_rerun()

if lang == 'ar':
    sel_lang_label = st.selectbox('اختيار اللغة', lang_labels_ar, index=lang_keys.index(lang))
    sel_lang = lang_keys[lang_labels_ar.index(sel_lang_label)]
else:
    sel_lang_label = st.selectbox('Select Language', lang_labels_en, index=lang_keys.index(lang))
    sel_lang = lang_keys[lang_labels_en.index(sel_lang_label)]

if sel_lang != lang:
    st.session_state.lang = sel_lang
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

About

elif choice == menu[5]: if lang == 'ar': st.header('حول المشروع') st.markdown(""" <div class='box'> <h3 class='main-header'>الكوارث لا تنتظر... ونحن أيضًا لا ننتظر.<br>Predict. Prevent. Protect.</h3> <hr> <b>مميزات المشروع وفوائده:</b> <ul> <li>رصد لحظي للحساسات الصناعية</li> <li>تنبيهات ذكية عند تجاوز الحدود</li> <li>تحليل تنبؤي لمدة ٧٢ ساعة</li> <li>حلول ذكية بنقرة واحدة</li> <li>خريطة تفاعلية لأماكن الحساسات</li> <li>تخصيص فوري للغة والثيم</li> <li>تصدير البيانات للتقارير</li> <li>دعم الجوال والكمبيوتر</li> </ul> <b>أهمية المشروع:</b> <p>رفع كفاءة وسلامة المنشآت عبر الاكتشاف المبكر وتقليل زمن الاستجابة.</p> <hr> <b>المطورون الرئيسيون:</b> <p>راكان المري | rakan.almarri.2@aramco.com | 0532559664<br> عبدالرحمن الزهراني | abdulrahman.alzhrani.1@aramco.com | 0549202574</p> </div> """, unsafe_allow_html=True) else: st.header('About the Project') st.markdown(""" <div class='box'>

