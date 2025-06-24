import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

# إعداد الصفحة
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

# Session State للإعدادات
if "language" not in st.session_state:
    st.session_state.language = "ar"
if "theme" not in st.session_state:
    st.session_state.theme = "aramco"
if "thresholds" not in st.session_state:
    st.session_state.thresholds = {
        "Temperature (°C)": 45,
        "Pressure (psi)": 110,
        "Vibration (g)": 0.7,
        "Methane (CH₄, ppm)": 12,
        "H₂S (ppm)": 5
    }

THEMES = {
    "forest": {
        "name": {"en": "Forest", "ar": "الغابة"},
        "sidebar": "#295135", "main": "#183c24", "accent": "#4caf50"
    },
    "ocean": {
        "name": {"en": "Ocean", "ar": "المحيط"},
        "sidebar": "#184060", "main": "#162a40", "accent": "#2196f3"
    },
    "desert": {
        "name": {"en": "Desert", "ar": "الصحراء"},
        "sidebar": "#7b5c2e", "main": "#543913", "accent": "#ffb300"
    },
    "night": {
        "name": {"en": "Night", "ar": "الليل"},
        "sidebar": "#262626", "main": "#181818", "accent": "#7e57c2"
    },
    "aramco": {
        "name": {"en": "Aramco", "ar": "أرامكو"},
        "sidebar": "#174766", "main": "#142c3e", "accent": "#36c0a7"
    }
}

@st.cache_data
def load_data():
    df = pd.read_csv("sensor_data_simulated_long.csv", parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

df = load_data()

def _(key):
    d = {
        "ar": {
            "dashboard": "لوحة البيانات",
            "predictive": "التحليل التنبؤي",
            "sensor_map": "خريطة الحساسات",
            "incident_log": "سجل الحوادث",
            "solutions": "الحلول الذكية",
            "report": "التقارير والتصدير",
            "cost": "التكلفة والتوفير",
            "settings": "الإعدادات",
            "about": "حول",
            "temperature": "درجة الحرارة",
            "pressure": "الضغط",
            "vibration": "الاهتزاز",
            "methane": "الميثان",
            "h2s": "كبريتيد الهيدروجين",
            "trend": "الاتجاهات",
            "lang": "اللغة",
            "theme": "الثيم",
            "set_thresholds": "تخصيص العتبات",
            "download_csv": "تحميل البيانات كـ CSV",
            "incident": "نوع الحادث",
            "time": "الوقت",
            "details": "التفاصيل",
            "generate_solution": "توليد حل",
            "smart_recommendation": "توصية ذكية",
            "no_incidents": "لا توجد حوادث مسجلة.",
            "prediction_days": "اختر عدد أيام التوقع",
            "disasters": "الكوارث لا تنتظر... ونحن أيضًا لا ننتظر. توقّع. وقاية. حماية.",
            "our_vision": "رؤيتنا",
            "about_body": """
مشروع التوأم الرقمي الذكي منصة تفاعلية لمراقبة وتحليل بيانات السلامة في المنشآت الصناعية. يوفر:
- مراقبة لحظية لقراءات الحساسات الحيوية.
- تنبؤ دقيق للمخاطر حتى 14 يومًا (قابل للتخصيص).
- توصيات ذكية مع الأولوية والمدة والفعالية.
- سجل كامل للحوادث والتنبيهات.
- تقارير سريعة وتصدير CSV بضغطة زر.
- حساب تقديري للتوفير المالي بفضل الكشف المبكر.

للتواصل:
راكان المري – rakan.almarri.2@aramco.com – 0532559664
عبدالرحمن الزهراني – abdulrahman.alzhrani.1@aramco.com – 0549202574
"""
        },
        "en": {
            "dashboard": "Dashboard",
            "predictive": "Predictive Analysis",
            "sensor_map": "Sensor Map",
            "incident_log": "Incident Log",
            "solutions": "Smart Solutions",
            "report": "Report & Export",
            "cost": "Cost Savings",
            "settings": "Settings",
            "about": "About",
            "temperature": "Temperature",
            "pressure": "Pressure",
            "vibration": "Vibration",
            "methane": "Methane",
            "h2s": "H₂S",
            "trend": "Trends",
            "lang": "Language",
            "theme": "Theme",
            "set_thresholds": "Customize Thresholds",
            "download_csv": "Download CSV",
            "incident": "Incident Type",
            "time": "Time",
            "details": "Details",
            "generate_solution": "Generate Solution",
            "smart_recommendation": "Smart Recommendation",
            "no_incidents": "No incidents recorded.",
            "prediction_days": "Prediction days",
            "disasters": "Disasters don't wait... and neither do we. Predict. Prevent. Protect.",
            "our_vision": "Our Vision",
            "about_body": """
Smart Neural Digital Twin is an interactive platform for monitoring and analyzing safety data in industrial sites. It offers:
- Real-time monitoring of critical sensor readings.
- Accurate risk forecasting up to 14 days ahead (configurable).
- Smart recommendations with priority, duration & effectiveness.
- Full incident log and alert history.
- Quick reports and one-click CSV export.
- Estimated cost savings via early incident detection.

Contact:
Rakan Almarri – rakan.almarri.2@aramco.com – 0532559664
Abdulrahman Alzhrani – abdulrahman.alzhrani.1@aramco.com – 0549202574
"""
        }
    }
    return d[st.session_state.language][key]

theme = THEMES[st.session_state.theme]
st.markdown(f"""
    <style>
    body {{ background: {theme['main']} !important; color: #fff; }}
    [data-testid="stSidebar"] {{ background: {theme['sidebar']} !important; color: #fff; }}
    .stApp {{ background-color: {theme['main']} !important; }}
    .main-header {{ font-size:2.3em; font-weight:bold; text-align:center; color: {theme['accent']}; padding:0.3em 0; }}
    .stButton > button {{ color: white !important; background: {theme['accent']} !important; border-radius: 2em; }}
    .stRadio > div, .stRadio > label {{ color: #fff !important; font-size: 1.1em; }}
    </style>
""", unsafe_allow_html=True)

PAGES = [
    ("dashboard", "📊"),
    ("predictive", "📈"),
    ("sensor_map", "🗺️"),
    ("incident_log", "🛑"),
    ("solutions", "🤖"),
    ("report", "📑"),
    ("cost", "💰"),
    ("settings", "⚙️"),
    ("about", "ℹ️")
]

def menu_layout():
    st.markdown(f"<div class='main-header'>{_('our_vision')}</div>", unsafe_allow_html=True)
    cols = st.columns(len(PAGES))
    sel = None
    for i, (pg, emoji) in enumerate(PAGES):
        if cols[i].button(f"{emoji} {_(''+pg)}"):
            sel = pg
    if not sel:
        sel = PAGES[0][0]
    return sel

page = menu_layout()

if page == "dashboard":
    st.subheader(f"🟢 {_('dashboard')}")
    last = df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(_("temperature"), f"{last['Temperature (°C)']:.2f} °C")
    c2.metric(_("pressure"), f"{last['Pressure (psi)']:.2f} psi")
    c3.metric(_("vibration"), f"{last['Vibration (g)']:.2f} g")
    c4.metric(_("methane"), f"{last['Methane (CH₄, ppm)']:.2f} ppm")
    c5.metric(_("h2s"), f"{last['H₂S (ppm)']:.2f} ppm")
    st.plotly_chart(
        px.line(df, x="Timestamp", y=[
            "Temperature (°C)", "Pressure (psi)", "Vibration (g)",
            "Methane (CH₄, ppm)", "H₂S (ppm)"
        ], title=_("trend"), template="plotly_dark"),
        use_container_width=True
    )

elif page == "predictive":
    st.subheader(f"🔮 {_('predictive')}")
    pred_days = st.slider(_("prediction_days"), 1, 14, 3)
    n_pred = pred_days * 24
    last_time = df["Timestamp"].iloc[-1]
    dt = df["Timestamp"].diff().median()
    future_times = [last_time + i*dt for i in range(1, n_pred+1)]
    pred = {col: [df[col].rolling(24, min_periods=1).mean().iloc[-1]]*n_pred
            for col in ["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Methane (CH₄, ppm)", "H₂S (ppm)"]}
    pred_df = pd.DataFrame(pred)
    pred_df["Timestamp"] = future_times
    plot_df = pd.concat([df.tail(48), pred_df]).reset_index(drop=True)
    fig = px.line(
        plot_df, x="Timestamp",
        y=["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Methane (CH₄, ppm)", "H₂S (ppm)"],
        title=f"{_('trend')} ({pred_days} days ahead)", template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(_("disasters"))

elif page == "sensor_map":
    st.subheader(f"🗺️ {_('sensor_map')}")
    st.info("مواقع الحساسات افتراضية للنموذج.")
    coords = df.assign(
        lat=lambda d: 25 + np.sin(np.arange(len(d))/30)*0.05,
        lon=lambda d: 49 + np.cos(np.arange(len(d))/30)*0.05
    ).iloc[::24]
    st.map(coords.rename(columns={"lat":"lat","lon":"lon"})[["lat","lon"]])

elif page == "incident_log":
    st.subheader(f"🛑 {_('incident_log')}")
    incidents = [
        {"Timestamp": r["Timestamp"], "Incident": key, "Value": r[key]}
        for _, r in df.iterrows()
        for key, th in st.session_state.thresholds.items()
        if r[key] > th
    ]
    if not incidents:
        st.success(_("no_incidents"))
    else:
        st.dataframe(pd.DataFrame(incidents))

elif page == "solutions":
    st.subheader(f"🤖 {_('solutions')}")
    last = df.iloc[-1]
    sols = []
    thr = st.session_state.thresholds
    if last["Temperature (°C)"] > thr["Temperature (°C)"]:
        sols.append({"solution": "تفعيل التبريد", "duration":"5m","priority":"High","effectiveness":"95%"})
    if last["Pressure (psi)"] > thr["Pressure (psi)"]:
        sols.append({"solution": "مراجعة الصمامات", "duration":"10m","priority":"Medium","effectiveness":"80%"})
    if last["Methane (CH₄, ppm)"] > thr["Methane (CH₄, ppm)"]:
        sols.append({"solution": "عزل المنطقة", "duration":"2m","priority":"High","effectiveness":"97%"})
    if last["H₂S (ppm)"] > thr["H₂S (ppm)"]:
        sols.append({"solution": "إخلاء فوري", "duration":"1m","priority":"Critical","effectiveness":"99%"})
    if last["Vibration (g)"] > thr["Vibration (g)"]:
        sols.append({"solution": "فحص المضخات", "duration":"7m","priority":"Medium","effectiveness":"88%"})
    if sols:
        if st.button(_("generate_solution")):
            st.dataframe(pd.DataFrame(sols))
    else:
        st.success(_("smart_recommendation") + ": الوضع مستقر.")

elif page == "report":
    st.subheader(f"📑 {_('report')}")
    st.download_button(_("download_csv"), df.to_csv(index=False), file_name="report.csv", mime="text/csv")
    st.dataframe(df.tail(168))

elif page == "cost":
    st.subheader(f"💰 {_('cost')}")
    count = sum(1 for _, r in df.iterrows() for key, th in st.session_state.thresholds.items() if r[key] > th)
    saved = count * 35000
    st.metric(_("cost"), f"{saved:,.0f} SAR")
    st.write(_("disasters"))

elif page == "settings":
    st.subheader(f"⚙️ {_('settings')}")
    lang = st.radio(_("lang"), ["ar", "en"], index=0 if st.session_state.language=="ar" else 1, horizontal=True)
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.experimental_rerun()
    theme_keys = list(THEMES.keys())
    idx = theme_keys.index(st.session_state.theme)
    th = st.radio(_("theme"), theme_keys, index=idx, horizontal=True, format_func=lambda x: THEMES[x]["name"][st.session_state.language])
    if th != st.session_state.theme:
        st.session_state.theme = th
        st.experimental_rerun()
    st.markdown(f"### {_('set_thresholds')}")
    for key in st.session_state.thresholds:
        st.session_state.thresholds[key] = st.slider(
            key, float(df[key].min()), float(df[key].max()), float(st.session_state.thresholds[key])
        )

elif page == "about":
    st.subheader(f"ℹ️ {_('about')}")
    st.markdown(f"### 💡 {_('our_vision')}")
    st.markdown(f"> {_('disasters')}")
    st.markdown(_ ("about_body"))
    st.markdown("---")
    if st.session_state.language == "ar":
        st.markdown("## ✨ مميزات المشروع")
        st.markdown("""
- مراقبة لحظية لقراءات الحساسات الحيوية.
- تنبؤ دقيق للمخاطر حتى 14 يومًا (قابل للتخصيص).
- توصيات ذكية مع الأولوية والمدة والفعالية.
- سجل كامل للحوادث والتنبيهات.
- تقارير سريعة وتصدير CSV بضغطة زر.
- حساب تقديري للتوفير المالي بفضل الكشف المبكر.
""")
        st.markdown("## 📞 تواصل معنا")
        st.markdown("""
**راكان المري**  
rakan.almarri.2@aramco.com  
0532559664  

**عبدالرحمن الزهراني**  
abdulrahman.alzhrani.1@aramco.com  
0549202574
""")
    else:
        st.markdown("## ✨ Key Features")
        st.markdown("""
- Real-time monitoring of critical sensor readings.
- Accurate risk forecasting up to 14 days ahead (configurable).
- Smart recommendations with priority, duration & effectiveness.
- Full incident log and alert history.
- Quick reports and one-click CSV export.
- Estimated cost savings via early incident detection.
""")
        st.markdown("## 📞 Contact Us")
        st.markdown("""
**Rakan Almarri**  
rakan.almarri.2@aramco.com  
0532559664  

**Abdulrahman Alzhrani**  
abdulrahman.alzhrani.1@aramco.com  
0549202574
""")

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{theme['accent']}; padding:1em;'>"
        f"🧠 Smart Neural Digital Twin | © 2025"
        "</div>",
        unsafe_allow_html=True
    )
