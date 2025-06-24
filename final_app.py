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
    st.session_state.theme = "forest"
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
            "last_values": "آخر قيم الحساسات",
            "metrics": "المقاييس الرئيسية",
            "temperature": "درجة الحرارة",
            "pressure": "الضغط",
            "vibration": "الاهتزاز",
            "methane": "الميثان",
            "h2s": "كبريتيد الهيدروجين",
            "trend": "الاتجاهات",
            "select_page": "اختر الصفحة",
            "lang": "اللغة",
            "theme": "الثيم",
            "set_thresholds": "تخصيص العتبات",
            "export": "تصدير البيانات",
            "generate_report": "توليد تقرير",
            "incident": "الحادثة",
            "solution": "الحل",
            "generate_solution": "توليد حل",
            "duration": "المدة",
            "priority": "الأهمية",
            "effectiveness": "الفعالية",
            "cost_saved": "التوفير المالي",
            "incident_type": "نوع الحادث",
            "time": "الوقت",
            "details": "التفاصيل",
            "smart_twin": "التوأم الرقمي الذكي",
            "disasters": 'الكوارث لا تنتظر... ونحن أيضًا لا ننتظر. توقّع. وقاية. حماية.',
            "our_vision": "رؤيتنا",
            "about_body": """
مشروع التوأم الرقمي الذكي منصة تفاعلية لمراقبة وتحليل بيانات السلامة في المنشآت الصناعية. يوفر مراقبة لحظية، تنبؤ بالمخاطر، توصيات ذكية، تقارير وتوفير مالي.
يساعدك على:
- منع الحوادث والتسربات
- تحسين سرعة اتخاذ القرار
- تقليل الأعطال والتكاليف
- دعم التحول الرقمي والتحكم الذكي

للتواصل:  
راكان المري – rakan.almarri.2@aramco.com – 0532559664  
عبدالرحمن الزهراني – abdulrahman.alzhrani.1@aramco.com – 0549202574
""",
            "cost_body": "بناءً على التحذيرات والتنبيهات، قدر النظام أنك وفرت هذا المبلغ من خلال الاكتشاف المبكر للحوادث. كل حادث تم تجنبه يُحتسب بقيمة تقريبية حسب البيانات.",
            "download_csv": "تحميل البيانات كـ CSV",
            "download_pdf": "تحميل التقرير PDF",
            "smart_recommendation": "توصية ذكية",
            "no_incidents": "لا توجد حوادث مسجلة.",
            "prediction_days": "اختر عدد أيام التوقع"
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
            "last_values": "Latest Sensor Values",
            "metrics": "Main Metrics",
            "temperature": "Temperature",
            "pressure": "Pressure",
            "vibration": "Vibration",
            "methane": "Methane",
            "h2s": "H₂S",
            "trend": "Trends",
            "select_page": "Select Page",
            "lang": "Language",
            "theme": "Theme",
            "set_thresholds": "Customize Thresholds",
            "export": "Export Data",
            "generate_report": "Generate Report",
            "incident": "Incident",
            "solution": "Solution",
            "generate_solution": "Generate Solution",
            "duration": "Duration",
            "priority": "Priority",
            "effectiveness": "Effectiveness",
            "cost_saved": "Cost Saved",
            "incident_type": "Incident Type",
            "time": "Time",
            "details": "Details",
            "smart_twin": "Smart Neural Digital Twin",
            "disasters": 'Disasters don\'t wait... and neither do we. Predict. Prevent. Protect.',
            "our_vision": "Our Vision",
            "about_body": """
Smart Neural Digital Twin is an interactive platform for monitoring and analyzing safety data in industrial sites. It offers real-time monitoring, risk prediction, smart recommendations, reports, and cost savings.
It helps you:
- Prevent incidents and leaks
- Improve decision speed
- Reduce downtime and costs
- Support digital transformation and smart control

Contact:  
Rakan Almarri – rakan.almarri.2@aramco.com – 0532559664  
Abdulrahman Alzhrani – abdulrahman.alzhrani.1@aramco.com – 0549202574
""",
            "cost_body": "Based on warnings and alerts, the system estimates the saved amount by early detection of incidents. Each avoided incident is counted with an estimated value according to the data.",
            "download_csv": "Download CSV",
            "download_pdf": "Download PDF report",
            "smart_recommendation": "Smart Recommendation",
            "no_incidents": "No incidents recorded.",
            "prediction_days": "Prediction days"
        }
    }
    return d[st.session_state.language][key]

theme = THEMES[st.session_state.theme]
st.markdown(f"""
    <style>
    body {{ background: {theme['main']} !important; color: #fff; }}
    [data-testid="stSidebar"] {{
        background: {theme['sidebar']};
        color: #fff;
    }}
    .stApp {{
        background-color: {theme['main']} !important;
    }}
    .st-emotion-cache-1d391kg {{
        background: {theme['sidebar']} !important;
    }}
    .main-header {{
        font-size:2.3em; font-weight:bold; text-align:center;
        color: {theme['accent']};
        padding: 0.3em 0;
    }}
    .stButton > button {{
        color: white !important;
        background: {theme['accent']} !important;
        border-radius: 2em;
    }}
    .stRadio > div, .stRadio > label {{
        color: #fff !important;
        font-size: 1.1em;
    }}
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
    st.markdown(f"<div class='main-header'>{_('smart_twin')}</div>", unsafe_allow_html=True)
    col_menu = st.columns(len(PAGES))
    page_selected = None
    for i, (pg, emoji) in enumerate(PAGES):
        if col_menu[i].button(f"{emoji} {_(''+pg)}"):
            page_selected = pg
    if page_selected is None:
        page_selected = PAGES[0][0]
    return page_selected

page = menu_layout()

if page == "dashboard":
    st.subheader("🟢 " + _("dashboard"))
    last_row = df.iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(_("temperature"), f"{last_row['Temperature (°C)']:.2f} °C")
    col2.metric(_("pressure"), f"{last_row['Pressure (psi)']:.2f} psi")
    col3.metric(_("vibration"), f"{last_row['Vibration (g)']:.2f} g")
    col4.metric(_("methane"), f"{last_row['Methane (CH₄, ppm)']:.2f} ppm")
    col5.metric(_("h2s"), f"{last_row['H₂S (ppm)']:.2f} ppm")
    st.plotly_chart(px.line(df, x="Timestamp", y=["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Methane (CH₄, ppm)", "H₂S (ppm)"], title=_("trend"), template="plotly_dark"), use_container_width=True)
    st.markdown("#### Heatmap (Pressure vs. Temperature)")
    fig_hm = px.density_heatmap(df, x="Temperature (°C)", y="Pressure (psi)", nbinsx=30, nbinsy=30, color_continuous_scale="Viridis")
    st.plotly_chart(fig_hm, use_container_width=True)

elif page == "predictive":
    st.subheader("🔮 " + _("predictive"))
    # تخصيص مدة التوقع (1-14 يوم)
    max_days = 14
    default_days = 3
    pred_days = st.slider(_("prediction_days"), 1, max_days, default_days)
    n_pred = pred_days * 24  # ساعات

    last_time = df["Timestamp"].iloc[-1]
    dt = df["Timestamp"].diff().median()
    future_times = [last_time + i*dt for i in range(1, n_pred+1)]
    pred_dict = {}
    for col in ["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Methane (CH₄, ppm)", "H₂S (ppm)"]:
        ma = df[col].rolling(24, min_periods=1).mean()
        pred_vals = [ma.iloc[-1]]*n_pred
        pred_dict[col] = pred_vals
    pred_df = pd.DataFrame(pred_dict)
    pred_df["Timestamp"] = future_times
    pred_plot = pd.concat([df.tail(48), pred_df]).reset_index(drop=True)
    fig_pred = px.line(
        pred_plot, x="Timestamp",
        y=["Temperature (°C)", "Pressure (psi)", "Vibration (g)", "Methane (CH₄, ppm)", "H₂S (ppm)"],
        title=_("trend") + f" ({pred_days} days ahead)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    st.info(_("disasters"))

elif page == "sensor_map":
    st.subheader("🗺️ " + _("sensor_map"))
    st.info("موقع الحساسات افتراضي في هذا النموذج (للحقول الحقيقية يمكن إضافتها).")
    st.map(df.rename(columns={"Methane (CH₄, ppm)": "lat", "H₂S (ppm)": "lon"}).assign(lat=lambda x: 25.0 + np.sin(x.index/30)*0.05, lon=lambda x: 49.0 + np.cos(x.index/30)*0.05).iloc[::24, :][["lat", "lon"]])

elif page == "incident_log":
    st.subheader("🛑 " + _("incident_log"))
    incs = []
    for i, row in df.iterrows():
        for key, th in st.session_state.thresholds.items():
            if row[key] > th:
                incs.append({
                    "Timestamp": row["Timestamp"],
                    "Incident": key,
                    "Value": row[key],
                })
    if len(incs) == 0:
        st.success(_("no_incidents"))
    else:
        incdf = pd.DataFrame(incs)
        st.dataframe(incdf)

elif page == "solutions":
    st.subheader("🤖 " + _("solutions"))
    last = df.iloc[-1]
    st.markdown(f"#### {_('smart_recommendation')}")
    smart_solutions = []
    if last["Temperature (°C)"] > st.session_state.thresholds["Temperature (°C)"]:
        smart_solutions.append({"solution": _("temperature") + " مرتفعة: فعّل نظام التبريد", "duration": "5m", "priority": "High", "effectiveness": "95%"})
    if last["Pressure (psi)"] > st.session_state.thresholds["Pressure (psi)"]:
        smart_solutions.append({"solution": _("pressure") + " مرتفع: راقب الصمامات", "duration": "10m", "priority": "Medium", "effectiveness": "80%"})
    if last["Methane (CH₄, ppm)"] > st.session_state.thresholds["Methane (CH₄, ppm)"]:
        smart_solutions.append({"solution": _("methane") + " مرتفع: عزل المنطقة", "duration": "2m", "priority": "High", "effectiveness": "97%"})
    if last["H₂S (ppm)"] > st.session_state.thresholds["H₂S (ppm)"]:
        smart_solutions.append({"solution": _("h2s") + " مرتفع: أخرج الأفراد فوراً", "duration": "1m", "priority": "Critical", "effectiveness": "99%"})
    if last["Vibration (g)"] > st.session_state.thresholds["Vibration (g)"]:
        smart_solutions.append({"solution": _("vibration") + " مرتفع: افحص المضخات", "duration": "7m", "priority": "Medium", "effectiveness": "88%"})
    if smart_solutions:
        st.button(_("generate_solution"))
        st.dataframe(pd.DataFrame(smart_solutions))
    else:
        st.success(_("smart_recommendation") + ": الوضع مستقر.")

elif page == "report":
    st.subheader("📑 " + _("report"))
    st.markdown(_("generate_report"))
    st.dataframe(df.tail(168))
    st.download_button(_("download_csv"), df.to_csv(index=False), file_name="smart_twin_report.csv", mime="text/csv")

elif page == "cost":
    st.subheader("💰 " + _("cost"))
    incidents = 0
    for i, row in df.iterrows():
        for key, th in st.session_state.thresholds.items():
            if row[key] > th:
                incidents += 1
    cost_per_incident = 35000
    cost_saved = incidents * cost_per_incident
    st.metric(_("cost_saved"), f"{cost_saved:,.0f} ريال سعودي")
    st.markdown(_("cost_body"))
    st.plotly_chart(px.bar(x=[_("cost_saved")], y=[cost_saved], labels={"x": _("cost_saved"), "y": "SAR"}, template="plotly_dark"), use_container_width=True)

elif page == "settings":
    st.subheader("⚙️ " + _("settings"))
    lang = st.radio(_("lang"), ["ar", "en"], index=0 if st.session_state.language=="ar" else 1, horizontal=True)
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.experimental_rerun()
    theme_options = list(THEMES.keys())
    thidx = theme_options.index(st.session_state.theme)
    th = st.radio(_("theme"), theme_options, index=thidx, horizontal=True, format_func=lambda x: THEMES[x]["name"][st.session_state.language])
    if th != st.session_state.theme:
        st.session_state.theme = th
        st.experimental_rerun()
    st.markdown("### " + _("set_thresholds"))
    for key in st.session_state.thresholds:
        st.session_state.thresholds[key] = st.slider(f"{key}", float(df[key].min()), float(df[key].max()), float(st.session_state.thresholds[key]))

elif page == "about":
    st.subheader("ℹ️ " + _("about"))

    # عنوان ورؤيتنا
    st.markdown(f"### 💡 {_('our_vision')}")
    st.markdown(f"> {_('disasters')}")

    # نص التعريف بالمشروع
    st.markdown(_("'about_body'"))

    st.markdown("---")

    # مميزات المشروع
    if st.session_state.language == "ar":
        st.markdown("## ✨ مميزات المشروع")
        st.markdown("""
- مراقبة لحظية لقراءات الحساسات الحيوية.
- تنبؤ دقيق بالمخاطر حتى *14 يوم* (قابل للتخصيص).
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
- Accurate risk forecasting up to *14 days* ahead (configurable).
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

    # — نهاية التطبيق (Footer) —
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{theme['accent']}; padding:1em;'>"
        f"🧠 {_('smart_twin')} | © 2025"
        "</div>",
        unsafe_allow_html=True
    )
