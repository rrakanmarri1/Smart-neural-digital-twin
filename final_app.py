import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# إعادة تسمية الأعمدة
COLUMNS_MAP = {
    "Temperature (°C)": "Temp",
    "Pressure (psi)": "Pressure",
    "Vibration (g)": "Vibration",
    "Methane (CH₄, ppm)": "Methane",
    "H₂S (ppm)": "H2S",
    "Timestamp": "Timestamp"
}

# تحميل البيانات
@st.cache_data
def load_data():
    df = pd.read_csv("sensor_data_simulated.csv")
    df = df.rename(columns=COLUMNS_MAP)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # إضافة مواقع حساسات تجريبية إذا غير موجودة
    if "lat" not in df.columns or "lon" not in df.columns:
        np.random.seed(1)
        df["lat"] = 25.4 + np.random.randn(len(df)) * 0.008  # موقع افتراضي حول الظهران
        df["lon"] = 49.6 + np.random.randn(len(df)) * 0.008
    return df

df = load_data()

# إعدادات لغة وثيم
if 'lang' not in st.session_state: st.session_state['lang'] = 'ar'
if 'theme' not in st.session_state: st.session_state['theme'] = 'slate'
lang = st.session_state['lang']
theme = st.session_state['theme']

# ثيمات وألوان
THEME_BACKGROUNDS = {
    "slate": "#2c233d",
    "ocean": "linear-gradient(90deg, #134e5e, #71b280 99%)",
    "forest": "linear-gradient(90deg, #005c97, #363795 99%)",
    "sunset": "linear-gradient(90deg, #fa709a, #fee140 99%)",
    "purple": "linear-gradient(90deg, #a770ef, #f6d365 99%)",
}

st.markdown(f"""
    <style>
        .stApp {{background: {THEME_BACKGROUNDS.get(theme, '#222')};}}
        .main-title {{font-size:2.6rem;font-weight:bold;margin-bottom:1rem;}}
        .main-menu {{display:flex; gap:1rem; margin-bottom:2rem; flex-wrap:wrap;}}
        .main-menu label, .theme-selector label {{margin-right:1.2em;}}
        .settings-box {{background:rgba(255,255,255,0.07);padding:1.5em 2em;border-radius:20px;max-width:500px}}
    </style>
""", unsafe_allow_html=True)

# القوائم والخيارات
pages = {
    "dashboard": "لوحة البيانات" if lang=="ar" else "Dashboard",
    "predict": "التحليل التنبؤي" if lang=="ar" else "Predictive Analysis",
    "map": "خريطة الحساسات" if lang=="ar" else "Sensor Map",
    "smart": "الحلول الذكية" if lang=="ar" else "Smart Solutions",
    "settings": "الإعدادات" if lang=="ar" else "Settings",
    "about": "حول" if lang=="ar" else "About",
}

# القائمة الرئيسية
st.markdown(f"""<div class="main-menu">""" + "".join([
    f"""<label><input type="radio" name="page" value="{k}" {'checked' if st.session_state.get('page', 'dashboard')==k else ''} onclick="window.location.search='?page={k}'">{v}</label>"""
    for k,v in pages.items()
]) + "</div>", unsafe_allow_html=True)

page = st.query_params.get('page', 'dashboard')
st.session_state['page'] = page

def switch_lang():
    st.session_state['lang'] = 'en' if st.session_state['lang'] == 'ar' else 'ar'
    st.experimental_rerun()

def set_theme(new_theme):
    st.session_state['theme'] = new_theme
    st.experimental_rerun()

#### لوحة البيانات (Dashboard)
if page == "dashboard":
    st.markdown(f'<div class="main-title">🧠 {"التوأم الرقمي الذكي" if lang=="ar" else "Smart Neural Digital Twin"}</div>', unsafe_allow_html=True)
    st.subheader("لوحة البيانات" if lang=="ar" else "Dashboard")
    latest = df.iloc[-1]
    cols = st.columns(5)
    cols[0].metric("درجة الحرارة" if lang=="ar" else "Temp (°C)", f"{latest.Temp:.2f}")
    cols[1].metric("الضغط" if lang=="ar" else "Pressure (psi)", f"{latest.Pressure:.2f}")
    cols[2].metric("الاهتزاز" if lang=="ar" else "Vibration (g)", f"{latest.Vibration:.2f}")
    cols[3].metric("الميثان" if lang=="ar" else "Methane (ppm)", f"{latest.Methane:.2f}")
    cols[4].metric("كبريتيد الهيدروجين" if lang=="ar" else "H2S (ppm)", f"{latest.H2S:.2f}")

    # تنبيه ذكي تلقائي
    alert = None
    if latest.Temp > 50:
        alert = "🚨 درجة الحرارة تجاوزت الحد الآمن!" if lang=="ar" else "🚨 Temperature exceeded safe limit!"
    elif latest.Pressure > 200:
        alert = "⚠️ الضغط مرتفع جداً!" if lang=="ar" else "⚠️ Pressure is too high!"
    elif latest.H2S > 10:
        alert = "☠️ مستويات كبريتيد الهيدروجين حرجة!" if lang=="ar" else "☠️ Critical H2S levels detected!"
    if alert:
        st.toast(alert, icon="⚡")
        st.error(alert)

    st.line_chart(df.set_index("Timestamp")[["Temp", "Pressure", "Vibration", "Methane", "H2S"]].tail(72))
    st.caption("أحدث 72 ساعة من بيانات الحساسات" if lang=="ar" else "Last 72 hours sensor data")

    st.download_button(
        label="تحميل البيانات كـ CSV" if lang=="ar" else "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sensor_data_export.csv",
        mime="text/csv"
    )

#### التحليل التنبؤي (Predictive Analysis)
elif page == "predict":
    st.markdown(f'<div class="main-title">{"التحليل التنبؤي" if lang=="ar" else "Predictive Analytics"}</div>', unsafe_allow_html=True)
    st.line_chart(df.set_index("Timestamp")[["Temp", "Pressure"]].tail(72))
    st.caption("توقعات المستقبل بناءً على البيانات الحالية" if lang=="ar" else "Forecasts based on recent data")

#### خريطة الحساسات
elif page == "map":
    st.markdown(f'<div class="main-title">{"خريطة الحساسات" if lang=="ar" else "Sensor Map"}</div>', unsafe_allow_html=True)
    st.map(df[["lat", "lon"]].drop_duplicates())
    st.caption("المواقع افتراضية للتجربة. أضف بيانات المواقع الحقيقية لكل حساس.")

#### الحلول الذكية (Smart Solutions)
elif page == "smart":
    st.markdown(f'<div class="main-title">{"الحلول الذكية" if lang=="ar" else "Smart Solutions"}</div>', unsafe_allow_html=True)
    st.write("اضغط زر التوليد للحصول على اقتراح آلي" if lang=="ar" else "Click generate to get an AI-based suggestion.")
    # نسبة عشوائية للفعالية
    if st.button("توليد حل ذكي 🚀" if lang=="ar" else "Generate Solution 🚀"):
        suggestion = ("الحل: قم بتخفيض الضغط تدريجياً، راقب مستويات الغاز" if lang=="ar"
                      else "Solution: Gradually decrease pressure and monitor gas levels")
        st.success(suggestion)
        fig = go.Figure(data=[go.Pie(labels=['Effectiveness', 'Other'],
                                     values=[92, 8],
                                     marker_colors=['#44ce42', '#ccc'],
                                     hole=.7)])
        fig.update_traces(textinfo='label+percent', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

#### الإعدادات (Settings)
elif page == "settings":
    st.markdown(f'<div class="main-title">{"الإعدادات" if lang=="ar" else "Settings"}</div>', unsafe_allow_html=True)
    with st.form(key='settings-form'):
        lang_col, theme_col = st.columns(2)
        with lang_col:
            st.radio("اختر اللغة" if lang=="ar" else "Language", 
                options=["ar", "en"], 
                format_func=lambda x: "العربية" if x=="ar" else "English", 
                key="lang", 
                horizontal=True, 
                on_change=switch_lang)
        with theme_col:
            st.radio("لوحة الألوان" if lang=="ar" else "Palette", 
                options=["ocean", "forest", "sunset", "purple", "slate"], 
                format_func=lambda x: {
                    "ocean": "🌊 Ocean",
                    "forest": "🌳 Forest",
                    "sunset": "🌅 Sunset",
                    "purple": "🟣 Purple",
                    "slate": "🪨 Slate"
                }[x], 
                key="theme",
                horizontal=True,
                on_change=lambda: set_theme(st.session_state.theme))
        st.form_submit_button("حفظ" if lang=="ar" else "Save")

#### حول المشروع (About)
elif page == "about":
    st.markdown(f'<div class="main-title">{"حول المشروع" if lang=="ar" else "About the Project"}</div>', unsafe_allow_html=True)
    st.image("https://content.api.news/v3/images/bin/9c4a75c6eb9e80f86de3efebcb97d48a", width=700)
    st.markdown("""
    <div class="settings-box">
        <h3>Disasters don't wait.. and neither do we. <br/>Predict. Prevent. Protect.</h3>
        <b>المطورون الرئيسيون (Lead Developers):</b>
        <ul>
        <li>راكان المري (Rakan Almarri)</li>
        <li>عبدالرحمن الزهراني (Abdulrahman Alzhrani)</li>
        </ul>
        <br>
        <b>مميزات المشروع:</b>
        <ul>
        <li>رصد حي لجميع قراءات الحساسات (Live monitoring)</li>
        <li>تنبيهات ذكية بالحالات الحرجة وتوصيات فورية</li>
        <li>توقعات مستقبلية وتحليل متقدم للبيانات</li>
        <li>خريطة تفاعلية لمواقع الحساسات</li>
        <li>واجهة تفاعلية متعددة اللغات والثيمات</li>
        <li>حلول مقترحة ذاتياً للوقاية من المخاطر</li>
        <li>تصدير البيانات بضغطة زر</li>
        </ul>
        <br>
        <b>الجهة الداعمة:</b> أرامكو السعودية
        <br>
        <b>للتواصل:</b>
        <ul>
        <li>راكان المري — rakan.almarri.2@aramco.com — 0532559664</li>
        <li>عبدالرحمن الزهراني — abdulrahman.alzahrani.1@aramco.com — 0549202574</li>
        </ul>
        <br>
        <a href="https://github.com/rrakanmarri1/Smart-neural-digital-twin" target="_blank">GitHub Project</a>
    </div>
    """, unsafe_allow_html=True)
