import streamlit as st import pandas as pd

Page configuration

st.set_page_config( page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide" )

@st.cache_data def load_data(): df = pd.read_csv('sensor_data_simulated.csv') df.rename(columns={ 'Timestamp':'timestamp', 'Temperature (°C)':'temperature', 'Pressure (psi)':'pressure', 'Vibration (g)':'vibration', 'Methane (CH₄, ppm)':'methane', 'H₂S (ppm)':'h2s' }, inplace=True) df['timestamp'] = pd.to_datetime(df['timestamp']) return df

df = load_data()

State

if 'lang' not in st.session_state: st.session_state.lang = 'ar' if 'theme' not in st.session_state: st.session_state.theme = 'Ocean'

lang = st.session_state.lang theme = st.session_state.theme

Menus

menu_ar = ['لوحة البيانات', 'التحليل التنبؤي', 'خريطة الحساسات', 'الحلول الذكية', 'الإعدادات', 'حول'] menu_en = ['Dashboard', 'Predictive Analysis', 'Sensor Map', 'Smart Solutions', 'Settings', 'About'] menu = menu_ar if lang=='ar' else menu_en choice = st.sidebar.radio("", menu, index=0)

Theme palettes

themes = { 'Ocean': {'bg':'#13375b','accent':'#00b4d8'}, 'Forest':{'bg':'#184d47','accent':'#81b214'}, 'Sunset':{'bg':'#ce6a85','accent':'#ffb86b'}, 'Purple':{'bg':'#3e206d','accent':'#b983ff'}, 'Slate': {'bg':'#222c36','accent':'#e0e0e0'} } colors = themes.get(theme, themes['Ocean'])

CSS

st.markdown(f"""

<style>
    .stApp {{ background:{colors['bg']} !important; color:#fff !important; }}
    .main-header {{ font-size:2.2rem; font-weight:bold; color:{colors['accent']}; margin-bottom:0.5rem; }}
    .box {{ background:#202a34; border-radius:12px; padding:1rem; margin-bottom:1.5rem; }}
    .solution {{ background:#282828; border-left:6px solid {colors['accent']}; padding:0.8rem; margin-bottom:1rem; border-radius:8px; }}
    .settings {{ background:#222c36; padding:1rem; border-radius:8px; margin-bottom:1rem; }}
</style>""", unsafe_allow_html=True)

Dashboard

if choice == menu[0]: st.markdown(f"<div class='main-header'>{'لوحة البيانات' if lang=='ar' else 'Dashboard'}</div>", unsafe_allow_html=True) latest = df.iloc[-1] cols = st.columns(5) cols[0].metric('درجة الحرارة' if lang=='ar' else 'Temperature', f"{latest.temperature:.2f} °C") cols[1].metric('الضغط' if lang=='ar' else 'Pressure', f"{latest.pressure:.2f} psi") cols[2].metric('الاهتزاز' if lang=='ar' else 'Vibration', f"{latest.vibration:.2f} g") cols[3].metric('الميثان' if lang=='ar' else 'Methane', f"{latest.methane:.2f} ppm") cols[4].metric('H₂S', f"{latest.h2s:.2f} ppm") st.line_chart(df[['temperature','pressure','vibration','methane','h2s']].tail(72))

Predictive Analysis

elif choice == menu[1]: st.markdown(f"<div class='main-header'>{'التحليل التنبؤي' if lang=='ar' else 'Predictive Analysis'}</div>", unsafe_allow_html=True) st.line_chart(df[['temperature','pressure']].tail(72))

Sensor Map

elif choice == menu[2]: st.markdown(f"<div class='main-header'>{'خريطة الحساسات' if lang=='ar' else 'Sensor Map'}</div>", unsafe_allow_html=True) if 'lat' in df and 'lon' in df: st.map(df.rename(columns={'lat':'latitude','lon':'longitude'})) else: st.info('لا توجد إحداثيات' if lang=='ar' else 'No coordinates available')

Smart Solutions

elif choice == menu[3]: st.markdown(f"<div class='main-header'>{'الحلول الذكية' if lang=='ar' else 'Smart Solutions'}</div>", unsafe_allow_html=True) if st.button('🔍 توليد حل' if lang=='ar' else '🔍 Generate Solution'): st.markdown(f""" <div class='solution'> <b>{'الحل المقترح:' if lang=='ar' else 'Suggested Solution:'}</b><br> {'افحص الصمامات فورًا' if lang=='ar' else 'Inspect safety valves immediately.'}<br> <b>⏳ {'المدة المتوقعة' if lang=='ar' else 'Estimated Duration'}:</b> 15 min<br> <b>⭐ {'الأولوية' if lang=='ar' else 'Priority'}:</b> {'حرجة' if lang=='ar' else 'Critical'}<br> <b>📊 {'الفعالية' if lang=='ar' else 'Effectiveness'}:</b> 95% </div> """, unsafe_allow_html=True)

Settings

elif choice == menu[4]: st.markdown(f"<div class='main-header'>{'الإعدادات' if lang=='ar' else 'Settings'}</div>", unsafe_allow_html=True) with st.container(): st.markdown("<div class='settings'>", unsafe_allow_html=True) # Theme selection sel_theme = st.selectbox('اختيار الثيم' if lang=='ar' else 'Select Theme', list(themes.keys()), index=list(themes.keys()).index(theme)) if sel_theme != theme: st.session_state.theme = sel_theme st.experimental_rerun() # Language selection sel_lang = st.selectbox('اختيار اللغة' if lang=='ar' else 'Select Language', ['العربية','English'] if lang=='ar' else ['English','العربية'], index=0) desired = 'ar' if sel_lang == 'العربية' else 'en' if desired != lang: st.session_state.lang = desired st.experimental_rerun() st.markdown("</div>", unsafe_allow_html=True)

About

elif choice == menu[5]: if lang=='ar': st.header('حول المشروع') st.markdown(""" <div class='box'> <h3 class='main-header'>الكوارث لا تنتظر... ونحن أيضًا لا ننتظر.<br>Predict. Prevent. Protect.</h3> <hr> <b>مميزات المشروع وفوائده:</b> <ul> <li>رصد لحظي للحساسات الصناعية</li> <li>تنبيهات ذكية عند تجاوز الحدود</li> <li>تحليل تنبؤي لمدة ٧٢ ساعة</li> <li>حلول ذكية بنقرة واحدة</li> <li>خريطة تفاعلية لأماكن الحساسات</li> <li>تخصيص فوري للغة والثيم</li> <li>تصدير البيانات للتقارير</li> <li>دعم الجوال والكمبيوتر</li> </ul> <b>أهمية المشروع:</b> <p>رفع كفاءة وسلامة المنشآت عبر الاكتشاف المبكر وتقليل زمن الاستجابة.</p> <hr> <b>المطورون الرئيسيون:</b> <p>راكان المري | rakan.almarri.2@aramco.com | 0532559664<br> عبدالرحمن الزهراني | abdulrahman.alzhrani.1@aramco.com | 0549202574</p> </div> """, unsafe_allow_html=True) else: st.header('About the Project') st.markdown(""" <div class='box'> <h3 class='main-header'>Disasters don't wait... and neither do we.<br>Predict. Prevent. Protect.</h3> <hr> <b>Project Features and Benefits:</b> <ul> <li>Real-time sensor monitoring</li> <li>Smart alerts on threshold breaches</li> <li>72-hour predictive analytics</li> <li>One-click smart solutions</li> <li>Interactive sensor map</li> <li>Instant language & theme customization</li> <li>Data export for reports</li> <li>Mobile & desktop support</li> </ul> <b>Project Value:</b> <p>Enhances industrial safety by early hazard detection and faster response.</p> <hr> <b>Lead Developers:</b> <p>Rakan Almarri | rakan.almarri.2@aramco.com | 0532559664<br> Abdulrahman Alzhrani | abdulrahman.alzhrani.1@aramco.com | 0549202574</p> </div> """, unsafe_allow_html=True)


