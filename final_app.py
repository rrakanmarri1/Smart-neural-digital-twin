import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('sensor_data_simulated.csv')
    df.rename(columns={
        "Timestamp": "timestamp",
        "Temperature (°C)": "temperature",
        "Pressure (psi)": "pressure",
        "Vibration (g)": "vibration",
        "Methane (CH₄, ppm)": "methane",
        "H₂S (ppm)": "h2s"
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

if "lang" not in st.session_state:
    st.session_state.lang = "ar"
if "theme" not in st.session_state:
    st.session_state.theme = "Ocean"

lang = st.session_state.lang
theme = st.session_state.theme

menu_ar = ["لوحة البيانات", "التحليل التنبؤي", "خريطة الحساسات", "الحلول الذكية", "الإعدادات", "حول"]
menu_en = ["Dashboard", "Predictive Analysis", "Sensor Map", "Smart Solutions", "Settings", "About"]
menu = menu_ar if lang == "ar" else menu_en

# ألوان الثيمات
THEMES = {
    "Ocean": {"bg": "#13375b", "accent": "#00b4d8"},
    "Forest": {"bg": "#184d47", "accent": "#81b214"},
    "Sunset": {"bg": "#ce6a85", "accent": "#ffb86b"},
    "Purple": {"bg": "#3e206d", "accent": "#b983ff"},
    "Slate": {"bg": "#222c36", "accent": "#e0e0e0"},
}

theme_colors = THEMES.get(theme, THEMES["Ocean"])

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {theme_colors['bg']} !important;
            color: #fff !important;
        }}
        .main-header {{
            font-size:2.3em; font-weight:bold; margin-bottom:10px; color:{theme_colors['accent']};
        }}
        .menu-box {{
            background:#202a34; border-radius:15px; padding:1.5rem; margin-bottom:2rem; box-shadow:0 0 8px #0002;
        }}
        .solution-box {{
            background:#282828; border-radius:15px; padding:1rem 1.3rem; margin-bottom:1.2rem; border-left:8px solid {theme_colors['accent']};
        }}
        .settings-box {{
            background:#222c36; border-radius:15px; padding:1rem 1.3rem; margin-bottom:1.2rem;
        }}
        .sidebar .sidebar-content {{ background:{theme_colors['bg']} !important; color:#fff !important; }}
        .stButton>button {{ border-radius:15px !important; font-weight:bold; }}
    </style>
    """, unsafe_allow_html=True)

# ----------- سايد بار (Menus) -----------
with st.sidebar:
    st.markdown(f"<div class='main-header'>{'🧠 التوأم الرقمي الذكي' if lang=='ar' else '🧠 Smart Neural Digital Twin'}</div>", unsafe_allow_html=True)
    choice = st.radio(
        "🚀 انتقل إلى:" if lang == "ar" else "🚀 Navigate to:",
        menu, index=0
    )
    st.write("---")
    st.write("🎨 " + ("اختر الثيم" if lang=="ar" else "Theme Palette"))
    th = st.radio("", list(THEMES.keys()), index=list(THEMES.keys()).index(theme), horizontal=True)
    if th != theme:
        st.session_state.theme = th
        st.experimental_rerun()
    st.write("🌐 " + ("اللغة" if lang=="ar" else "Language"))
    selected_lang = st.radio("", ["العربية", "English"] if lang=="ar" else ["English", "العربية"], index=0, horizontal=True)
    if (selected_lang == "العربية" and lang != "ar") or (selected_lang == "English" and lang != "en"):
        st.session_state.lang = "ar" if selected_lang == "العربية" else "en"
        st.experimental_rerun()

# ----------- تحميل البيانات -----------
df = load_data()

# ----------- الصفحة الرئيسية (Dashboard) -----------
if choice == menu[0]:
    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)
    st.markdown(f"<span class='main-header'>{'لوحة البيانات' if lang=='ar' else 'Dashboard'}</span>", unsafe_allow_html=True)
    latest = df.iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("درجة الحرارة" if lang == "ar" else "Temperature (°C)", f"{latest['temperature']:.2f}")
    col2.metric("الضغط" if lang == "ar" else "Pressure (psi)", f"{latest['pressure']:.2f}")
    col3.metric("الاهتزاز" if lang == "ar" else "Vibration (g)", f"{latest['vibration']:.2f}")
    col4.metric("الميثان" if lang == "ar" else "Methane (ppm)", f"{latest['methane']:.2f}")
    col5.metric("H₂S", f"{latest['h2s']:.2f}")
    st.line_chart(df[["temperature", "pressure", "vibration", "methane", "h2s"]].tail(72))
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- التحليل التنبؤي -----------
elif choice == menu[1]:
    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)
    st.markdown(f"<span class='main-header'>{'التحليل التنبؤي' if lang=='ar' else 'Predictive Analysis'}</span>", unsafe_allow_html=True)
    st.line_chart(df[["temperature", "pressure"]].tail(72))
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- خريطة الحساسات -----------
elif choice == menu[2]:
    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)
    st.markdown(f"<span class='main-header'>{'خريطة الحساسات' if lang=='ar' else 'Sensor Map'}</span>", unsafe_allow_html=True)
    if "lat" in df.columns and "lon" in df.columns:
        st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"}))
    else:
        st.info("لا توجد إحداثيات في البيانات" if lang == "ar" else "No coordinates in the data.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- الحلول الذكية -----------
elif choice == menu[3]:
    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)
    st.markdown(f"<span class='main-header'>{'الحلول الذكية' if lang=='ar' else 'Smart Solutions'}</span>", unsafe_allow_html=True)
    if st.button("🔍 توليد حل ذكي" if lang == "ar" else "🔍 Generate Smart Solution"):
        st.markdown(
            f"<div class='solution-box'><b>{'الحل المقترح:' if lang=='ar' else 'Suggested Solution:'}</b><br>"
            f"{'قم بفحص الأنابيب وصمامات الأمان فوراً.' if lang=='ar' else 'Check all pipelines and safety valves immediately.'}"
            "<br><b>⏳ " + ("المدة المتوقعة" if lang=="ar" else "Expected Duration") + ":</b> 15 min"
            "<br><b>⭐ " + ("الأهمية" if lang=="ar" else "Priority") + ":</b> عالية / High"
            "<br><b>📊 " + ("الفعالية" if lang=="ar" else "Effectiveness") + ":</b> 95%"
            "</div>", unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- الإعدادات -----------
elif choice == menu[4]:
    st.markdown("<div class='settings-box'>", unsafe_allow_html=True)
    st.markdown(f"<span class='main-header'>{'الإعدادات' if lang=='ar' else 'Settings'}</span>", unsafe_allow_html=True)
    st.write("غيّر اللغة أو لون الموقع مباشرة من القائمة الجانبية." if lang=="ar" else "Change language or theme directly from the sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- حول المشروع (About) -----------
elif choice == menu[5]:
    if lang == "ar":
        st.header("حول المشروع")
        st.markdown("""
        <div style='background-color:#1f2937;padding:1.5rem;border-radius:18px; color:#fff;'>
            <h3 style='margin-bottom:8px;'>الكوارث لا تنتظر... ونحن أيضًا لا ننتظر.<br>Predict. Prevent. Protect.</h3>
            <hr>
            <b>مميزات المشروع وفوائده:</b>
            <ul>
                <li>رصد لحظي: مراقبة مباشرة لجميع قراءات الحساسات الصناعية (حرارة، ضغط، اهتزاز، غازات…إلخ) وعرضها بطريقة بصرية جذابة وسهلة الفهم.</li>
                <li>تنبيهات ذكية: إشعارات فورية عند تجاوز أي قيمة للحدود الآمنة مع توصية تلقائية بالإجراء المناسب.</li>
                <li>تحليل تنبؤي: رسوم بيانية وتوقعات للاتجاهات المستقبلية للمتغيرات المهمة في الموقع الصناعي (لمدة ٧٢ ساعة).</li>
                <li>حلول ذكية: اقتراحات فورية قابلة للتنفيذ في حالات الطوارئ أو الأعطال.</li>
                <li>خريطة تفاعلية: تحديد مواقع الحساسات داخل المنشأة لمتابعة مصدر الإنذار بدقة.</li>
                <li>تخصيص فوري: دعم التغيير الفوري للغة (عربي/إنجليزي) والثيم والألوان بما يناسب المستخدم.</li>
                <li>تصدير البيانات: إمكانية تحميل التقارير والبيانات بسهولة لمشاركتها أو تحليلها خارجياً.</li>
                <li>دعم متعدد المنصات: الواجهة متوافقة مع الجوال والكمبيوتر.</li>
            </ul>
            <b>أهمية المشروع:</b>
            <p>
                هذا النظام يرفع من كفاءة وسلامة المنشآت الصناعية عن طريق اكتشاف الأخطار مبكرًا، تقليل زمن الاستجابة، تحسين اتخاذ القرار، وتوفير واجهة سهلة لأي فريق تشغيلي أو إداري.
            </p>
            <hr>
            <b>المطورون الرئيسيون:</b>
            <div style='margin-top:0.5rem;'>
                <b>راكان المري</b> | rakan.almarri.2@aramco.com | 0532559664<br>
                <b>عبدالرحمن الزهراني</b> | abdulrahman.alzhrani.1@aramco.com | 0549202574
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.header("About the Project")
        st.markdown("""
        <div style='background-color:#1f2937;padding:1.5rem;border-radius:18px; color:#fff;'>
            <h3 style='margin-bottom:8px;'>Disasters don't wait... and neither do we.<br>Predict. Prevent. Protect.</h3>
            <hr>
            <b>Project Features and Value:</b>
            <ul>
                <li>Real-Time Monitoring: Instant visualization of all key sensor readings (temperature, pressure, vibration, gases, etc.) in a user-friendly dashboard.</li>
                <li>Smart Alerts: Immediate notifications when any parameter exceeds safe limits, with auto-generated recommendations.</li>
                <li>Predictive Analytics: Charts and trend forecasts for critical variables over the last 72 hours.</li>
                <li>Smart Solutions: One-click actionable suggestions for emergencies or faults.</li>
                <li>Interactive Map: Sensor locations displayed on a map for precise incident tracking.</li>
                <li>Instant Customization: On-the-fly switching between Arabic and English, with theme and color options.</li>
                <li>Data Export: Downloadable reports and data for easy sharing and offline analysis.</li>
                <li>Cross-Platform Support: Responsive interface for both mobile and desktop.</li>
            </ul>
            <b>Project Value:</b>
            <p>
                This platform enhances the efficiency and safety of industrial sites by enabling early hazard detection, reducing response time, improving decision-making, and providing an intuitive interface for operational and management teams.
            </p>
            <hr>
            <b>Lead Developers:</b>
            <div style='margin-top:0.5rem;'>
                <b>Rakan Almarri</b> | rakan.almarri.2@aramco.com | 0532559664<br>
                <b>Abdulrahman Alzhrani</b> | abdulrahman.alzhrani.1@aramco.com | 0549202574
            </div>
        </div>
        """, unsafe_allow_html=True)
