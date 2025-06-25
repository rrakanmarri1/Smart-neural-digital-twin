"""
Smart Neural Digital Twin – Fancy Streamlit Dashboard
This version:
- Fixes the issue with HTML code blocks by avoiding HTML in KPI cards entirely.
- Uses Streamlit-native elements (st.metric in st.columns) for KPIs for 100% compatibility and a polished, modern look.
- All other pages remain fancy and use your CSS for the rest of the UI.
- "Estimated Time" in solutions always appears, with proper translation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from typing import Dict, Callable

# =========================
# 1. Theme System
# =========================

THEME_SETS: Dict[str, Dict[str, str]] = {
    "Ocean": {"primary": "#153243", "secondary": "#278ea5", "accent": "#21e6c1",
              "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#153243",
              "sidebar_bg": "#18465b", "card_bg": "#278ea5", "badge_bg": "#21e6c1",
              "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#153243"},
    "Sunset": {"primary": "#FF7043", "secondary": "#FFA726", "accent": "#FFD54F",
               "text_on_primary": "#232526", "text_on_secondary": "#232526", "text_on_accent": "#232526",
               "sidebar_bg": "#FFF8E1", "card_bg": "#FFE0B2", "badge_bg": "#FFA726",
               "alert": "#D7263D", "alert_text": "#fff", "plot_bg": "#FFF3E0"},
    "Emerald": {"primary": "#154734", "secondary": "#43e97b", "accent": "#38f9d7",
                "text_on_primary": "#fff", "text_on_secondary": "#153243", "text_on_accent": "#154734",
                "sidebar_bg": "#e0f2f1", "card_bg": "#e8fff3", "badge_bg": "#38f9d7",
                "alert": "#ff1744", "alert_text": "#fff", "plot_bg": "#e0f2f1"},
    "Night": {"primary": "#232526", "secondary": "#414345", "accent": "#e96443",
              "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#232526",
              "sidebar_bg": "#353749", "card_bg": "#414345", "badge_bg": "#e96443",
              "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#232526"},
    "Blossom": {"primary": "#fbd3e9", "secondary": "#bb377d", "accent": "#fa709a",
                "text_on_primary": "#232526", "text_on_secondary": "#fff", "text_on_accent": "#fff",
                "sidebar_bg": "#fce4ec", "card_bg": "#f8bbd0", "badge_bg": "#bb377d",
                "alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fce4ec"}
}
DEFAULT_THEME = "Ocean"

# =========================
# 2. Translations (all used keys)
# =========================

translations = {
    "en": {
        "Settings": "Settings", "Choose Language": "Choose Language",
        "English": "English", "Arabic": "Arabic",
        "Theme Set": "Theme Set", "Theme": "Theme", "Theme Preview": "Theme Preview",
        "Dashboard": "Dashboard", "Predictive Analysis": "Predictive Analysis",
        "Smart Solutions": "Smart Solutions", "Smart Alerts": "Smart Alerts",
        "Cost & Savings": "Cost & Savings", "Achievements": "Achievements",
        "Performance": "Performance", "Comparison": "Comparison", "Performance Comparison": "Performance Comparison",
        "Data Explorer": "Data Explorer",
        "About": "About", "Navigate to": "Navigate to",
        "Welcome to your Smart Digital Twin!": "Welcome to your Smart Neural Digital Twin!",
        "Temperature": "Temperature", "Pressure": "Pressure", "Vibration": "Vibration",
        "Methane": "Methane", "H2S": "H2S", "Live Data": "Live Data", "Trend": "Trend",
        "Forecast": "Forecast", "Simulate Disaster": "Simulate Disaster",
        "Details": "Details", "Reason": "Reason", "Priority": "Priority",
        "Effectiveness": "Effectiveness", "Estimated Time": "Estimated Time",
        "Generate Solution": "Generate Solution", "Generating solution...": "Generating solution...",
        "Press 'Generate Solution' for intelligent suggestions.": "Press 'Generate Solution' for intelligent suggestions.",
        "Emergency Vent Gas!": "Emergency Vent Gas!", "Immediate venting required in Tank 2 due to critical methane spike.": "Immediate venting required in Tank 2 due to critical methane spike.",
        "Critical disaster detected during simulation.": "Critical disaster detected during simulation.",
        "Reduce Pressure in Line 3": "Reduce Pressure in Line 3", "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.",
        "Abnormal vibration detected. This reduces risk.": "Abnormal vibration detected. This reduces risk.",
        "URGENT": "URGENT", "Now": "Now", "High": "High", "15 minutes": "15 minutes", "95%": "95%", "99%": "99%",
        "About Project Description": "Smart Neural Digital Twin is an AI-powered disaster prevention platform for industrial sites and oilfields. It connects live sensors to an intelligent digital twin for real-time monitoring, predictions, and emergency response.",
        "High Risk Area: Tank 3": "High Risk Area: Tank 3",
        "Monthly Savings": "Monthly Savings",
        "Yearly Savings": "Yearly Savings",
        "Reduction in Maintenance Costs": "Reduction in Maintenance Costs",
        "Savings": "Savings",
        "Source": "Source",
        "Amount (SAR)": "Amount (SAR)",
        "Savings Breakdown": "Savings Breakdown",
        "Current Alerts": "Current Alerts",
        "No alerts at the moment.": "No alerts at the moment.",
        "Congratulations!": "Congratulations!",
        "You have achieved": "You have achieved",
        "days without incidents": "days without incidents",
        "Compared to last period": "Compared to last period",
        "Milestones": "Milestones",
        "months zero downtime": "months zero downtime",
        "energy efficiency improvement": "energy efficiency improvement",
        "2025 Innovation Award, Best Digital Twin": "2025 Innovation Award, Best Digital Twin",
        "Data Filters": "Data Filters",
        "Select Metric": "Select Metric",
        "Summary Table": "Summary Table",
        "Current": "Current",
        "Previous": "Previous",
        "Change": "Change",
        "Metric": "Metric",
        "Month": "Month",
        "Energy Efficiency": "Energy Efficiency",
        "Maintenance Reduction": "Maintenance Reduction",
        "Downtime Prevention": "Downtime Prevention",
        "Smart Recommendations": "Smart Recommendations",
        "Severity": "Severity",
        "Time": "Time",
        "Location": "Location",
        "Message": "Message",
        "Medium": "Medium",
        "Low": "Low",
        "Main Developers": "Main Developers",
        "Our Vision": "Our Vision",
        "Disasters don't wait.. and neither do we.": "Disasters don't wait.. and neither do we.",
        "Features": "Features",
        "AI-powered predictive analytics": "AI-powered predictive analytics",
        "Instant smart solutions": "Instant smart solutions",
        "Live alerts and monitoring": "Live alerts and monitoring",
        "Multi-language support": "Multi-language support",
        "Stunning, responsive UI": "Stunning, responsive UI",
        "Dashboard loaded successfully!": "Dashboard loaded successfully!",
        "An error occurred loading the dashboard: ": "An error occurred loading the dashboard: ",
        "Prediction": "Prediction",
        "Live Monitoring": "Live Monitoring",
        "Ocean": "Ocean",
        "Sunset": "Sunset",
        "Emerald": "Emerald",
        "Night": "Night",
        "Blossom": "Blossom",
        "AI-powered recommendations for safety and efficiency": "AI-powered recommendations for safety and efficiency",
        "Methane Spike": "Methane Spike",
        "Pressure Drop": "Pressure Drop",
        "Vibration Anomaly": "Vibration Anomaly",
        "High Temperature": "High Temperature",
        "About the Project": "About the Project",
        "Contact us for partnership or demo!": "Contact us for partnership or demo!",
        "Performance": "Performance",
        "Comparison": "Comparison",
        "Lets Compare!": "Lets Compare!"
    },
    "ar": {
        "Settings": "الإعدادات", "Choose Language": "اختر اللغة",
        "English": "الإنجليزية", "Arabic": "العربية",
        "Theme Set": "مجموعة الألوان", "Theme": "السمة", "Theme Preview": "معاينة السمة",
        "Dashboard": "لوحة التحكم", "Predictive Analysis": "تحليل تنبؤي",
        "Smart Solutions": "حلول ذكية", "Smart Alerts": "تنبيهات ذكية",
        "Cost & Savings": "التكلفة والتوفير", "Achievements": "الإنجازات",
        "Performance": "الأداء", "Comparison": "مقارنة", "Performance Comparison": "مقارنة الأداء",
        "Data Explorer": "استكشاف البيانات",
        "About": "حول", "Navigate to": "انتقل إلى",
        "Welcome to your Smart Digital Twin!": "مرحبًا بك في التوأم الرقمي الذكي!",
        "Temperature": "درجة الحرارة", "Pressure": "الضغط", "Vibration": "الاهتزاز",
        "Methane": "الميثان", "H2S": "كبريتيد الهيدروجين", "Live Data": "بيانات مباشرة", "Trend": "الاتجاه",
        "Forecast": "التوقعات", "Simulate Disaster": "محاكاة كارثة",
        "Details": "التفاصيل", "Reason": "السبب", "Priority": "الأولوية",
        "Effectiveness": "الفعالية", "Estimated Time": "الوقت المتوقع",
        "Generate Solution": "توليد حل", "Generating solution...": "جاري توليد الحل…",
        "Press 'Generate Solution' for intelligent suggestions.": "اضغط 'توليد حل' للحصول على اقتراحات ذكية.",
        "Emergency Vent Gas!": "تنفيس الغاز فوراً!", "Immediate venting required in Tank 2 due to critical methane spike.": "مطلوب تنفيس فوري في الخزان 2 بسبب ارتفاع حاد في الميثان.",
        "Critical disaster detected during simulation.": "تم رصد كارثة حرجة أثناء المحاكاة.",
        "Reduce Pressure in Line 3": "قلل الضغط في الخط ٣", "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "قم بخفض الضغط بنسبة 15٪ في الخط ٣ ونبه فريق الصيانة للفحص.",
        "Abnormal vibration detected. This reduces risk.": "تم رصد اهتزاز غير طبيعي. هذا يقلل المخاطر.",
        "URGENT": "عاجل", "Now": "الآن", "High": "مرتفعة", "15 minutes": "١٥ دقيقة", "95%": "٩٥٪", "99%": "٩٩٪",
        "About Project Description": "التوأم الرقمي العصبي الذكي هو منصة مدعومة بالذكاء الاصطناعي للوقاية من الكوارث في المواقع الصناعية وحقول النفط. يربط أجهزة الاستشعار الحية بتوأم رقمي ذكي للمراقبة والتنبؤ والاستجابة للطوارئ في الوقت الفعلي.",
        "High Risk Area: Tank 3": "منطقة خطورة عالية: الخزان ٣",
        "Monthly Savings": "التوفير الشهري",
        "Yearly Savings": "التوفير السنوي",
        "Reduction in Maintenance Costs": "تقليل تكلفة الصيانة",
        "Savings": "التوفير",
        "Source": "المصدر",
        "Amount (SAR)": "المبلغ (ريال)",
        "Savings Breakdown": "تفصيل التوفير",
        "Current Alerts": "التنبيهات الحالية",
        "No alerts at the moment.": "لا توجد تنبيهات حاليًا.",
        "Congratulations!": "مبروك!",
        "You have achieved": "لقد حققت",
        "days without incidents": "يوم بدون حوادث",
        "Compared to last period": "مقارنة بالفترة السابقة",
        "Milestones": "إنجازات",
        "months zero downtime": "شهور بدون توقف",
        "energy efficiency improvement": "تحسن كفاءة الطاقة",
        "2025 Innovation Award, Best Digital Twin": "جائزة الابتكار 2025 - أفضل توأم رقمي",
        "Data Filters": "فلاتر البيانات",
        "Select Metric": "اختر المقياس",
        "Summary Table": "جدول ملخص",
        "Current": "الحالي",
        "Previous": "السابق",
        "Change": "التغير",
        "Metric": "المؤشر",
        "Month": "الشهر",
        "Energy Efficiency": "كفاءة الطاقة",
        "Maintenance Reduction": "خفض الصيانة",
        "Downtime Prevention": "منع التوقف",
        "Smart Recommendations": "توصيات ذكية",
        "Severity": "درجة الخطورة",
        "Time": "الوقت",
        "Location": "الموقع",
        "Message": "الرسالة",
        "Medium": "متوسطة",
        "Low": "منخفضة",
        "Main Developers": "المطورون الرئيسيون",
        "Our Vision": "رؤيتنا",
        "Disasters don't wait.. and neither do we.": "الكوارث لا تنتظر.. ونحن أيضًا لا ننتظر.",
        "Features": "المميزات",
        "AI-powered predictive analytics": "تحليلات تنبؤية مدعومة بالذكاء الاصطناعي",
        "Instant smart solutions": "حلول ذكية فورية",
        "Live alerts and monitoring": "تنبيهات ومراقبة حية",
        "Multi-language support": "دعم متعدد اللغات",
        "Stunning, responsive UI": "واجهة رائعة ومتجاوبة",
        "Dashboard loaded successfully!": "تم تحميل لوحة التحكم بنجاح!",
        "An error occurred loading the dashboard: ": "حدث خطأ أثناء تحميل لوحة التحكم: ",
        "Prediction": "تنبؤ",
        "Live Monitoring": "مراقبة حية",
        "Ocean": "أوشن",
        "Sunset": "غروب الشمس",
        "Emerald": "زمردي",
        "Night": "ليلي",
        "Blossom": "إزهار",
        "AI-powered recommendations for safety and efficiency": "توصيات ذكية مدعومة بالذكاء الاصطناعي للسلامة والكفاءة",
        "Methane Spike": "ارتفاع الميثان",
        "Pressure Drop": "انخفاض الضغط",
        "Vibration Anomaly": "خلل بالاهتزاز",
        "High Temperature": "درجة حرارة عالية",
        "About the Project": "عن المشروع",
        "Contact us for partnership or demo!": "تواصل معنا للشراكة أو العرض التوضيحي!",
        "Performance": "الأداء",
        "Comparison": "مقارنة",
        "Lets Compare!": "لنقارن!"
    }
}

# 3. Language & Theme State
def get_lang() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ar"
    return st.session_state["lang"]

def set_lang(lang: str):
    st.session_state["lang"] = lang

def _(key: str) -> str:
    lang = get_lang()
    return translations.get(lang, {}).get(key) or translations["en"].get(key) or key

def rtl_wrap(html: str) -> str:
    return f"<div class='rtl'>{html}</div>" if get_lang() == "ar" else html

def set_theme_in_session():
    if "theme_set" not in st.session_state:
        st.session_state["theme_set"] = DEFAULT_THEME

set_theme_in_session()
theme = THEME_SETS[st.session_state["theme_set"]]

# 4. CSS Injection (fancy for all except KPI row)
def inject_css():
    st.markdown(f"""
    <style>
    body, .stApp {{ background: linear-gradient(120deg, {theme['primary']} 60%, {theme['secondary']} 100%) !important; min-height:100vh; }}
    .stSidebar {{ background-color: {theme['sidebar_bg']} !important; }}
    .big-title {{ color: {theme['accent']}; font-size:2.8rem; font-weight:bold; margin-bottom:10px; letter-spacing:0.04em; text-shadow: 1px 2px 12px rgba(0,0,0,0.08); }}
    .sub-title {{ color: {theme['secondary']}; font-size:1.4rem; margin-bottom:14px; font-weight:600; letter-spacing: 0.03em; }}
    .card {{
        background: linear-gradient(120deg, {theme['card_bg']} 70%, {theme['accent']} 120%);
        border-radius:22px;
        padding:26px 24px 24px 24px;
        margin-bottom:22px;
        color:{theme['text_on_secondary']};
        box-shadow: 0 6px 30px rgba(0,0,0,0.10);
        min-width:120px;
    }}
    .badge {{ background:{theme['badge_bg']}; color:{theme['text_on_accent']}; padding:6px 18px; border-radius:22px; margin-right:8px; font-weight:600; font-size:1.07rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
    .rtl {{ direction:rtl; }}
    .status-badge {{
        display: inline-block;
        background: {theme['alert']};
        color: {theme['alert_text']};
        border-radius: 15px;
        padding: 5px 18px;
        font-weight: bold;
        font-size: 1.18rem;
        margin-bottom: 7px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.10);
    }}
    .sidebar-section {{
        margin-bottom: 21px;
        padding-bottom: 13px;
        border-bottom: 1.5px solid #e0e0e0;
    }}
    .about-card-gradient {{
        background: linear-gradient(120deg, {theme['secondary']} 40%, {theme['accent']} 100%);
        color: {theme['text_on_primary']};
        border-radius: 20px;
        padding: 20px 28px;
        box-shadow: 0 6px 28px rgba(0,0,0,0.10);
        margin-bottom: 18px;
    }}
    .about-dev {{
        display: flex; gap: 45px; align-items: center; justify-content: center; margin-top:18px;
    }}
    .about-dev .dev {{
        text-align:center;
        background:rgba(255,255,255,0.08);
        border-radius:18px;
        padding:12px 26px 11px 26px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .about-dev img {{
        border-radius:50%;
        border:3px solid {theme['badge_bg']};
        margin-bottom:7px;
    }}
    .about-features {{
        display: grid;
        grid-template-columns: repeat(2,1fr);
        gap: 13px;
        margin-top: 12px;
        font-size: 1.09rem;
        font-weight: 500;
    }}
    .about-features .fancy-icon {{
        font-size: 1.3em;
        margin-right: 6px;
    }}
    .about-milestones {{
        margin-top: 8px;
        margin-bottom: 8px;
        padding-left: 15px;
        font-size: 1.08rem;
    }}
    .about-contact {{
        margin-top: 22px;
        text-align:center;
        color:{theme['accent']};
        font-size:1.15rem;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# 5. Sidebar
def theme_selector(key="theme_selector_radio"):
    theme_names = list(THEME_SETS.keys())
    current_theme = st.session_state.get("theme_set", DEFAULT_THEME)
    selected = st.radio(
        _("Theme Set"),
        options=theme_names,
        format_func=lambda x: _(x),
        index=theme_names.index(current_theme),
        key=key
    )
    if selected != current_theme:
        st.session_state["theme_set"] = selected
        st.experimental_rerun()

def sidebar():
    with st.sidebar:
        with st.expander(_("Settings"), expanded=True):
            theme_selector(key="theme_selector_radio_1")
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            lang_choice = st.radio(_("Choose Language"),
                                   options=["ar", "en"],
                                   format_func=lambda x: _("Arabic") if x == "ar" else _("English"),
                                   index=0 if get_lang() == "ar" else 1,
                                   key="lang_radio")
            set_lang(lang_choice)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            theme_selector(key="theme_selector_radio_2")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        pages = [
            ("dashboard", _("Dashboard")), ("predictive", _("Predictive Analysis")),
            ("solutions", _("Smart Solutions")), ("alerts", _("Smart Alerts")),
            ("cost", _("Cost & Savings")), ("achievements", _("Achievements")),
            ("performance", _("Performance")), ("comparison", _("Comparison")),
            ("explorer", _("Data Explorer")), ("about", _("About"))
        ]
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.radio(_("Navigate to"), options=pages, format_func=lambda x: x[1], index=0, key="page_radio")
        st.markdown('</div>', unsafe_allow_html=True)

sidebar()

# 6. KPI Display - Streamlit-native metric, stays fancy with emojis & colors
def kpi_metrics_row(values, labels, units, icons):
    cols = st.columns(len(values))
    for i, col in enumerate(cols):
        col.metric(
            label=f"{icons[i]} {labels[i]}",
            value=f"{values[i]} {units[i]}"
        )

# 7. Pages
def show_dashboard():
    try:
        st.markdown(rtl_wrap(f"""
        <div class="big-title">
            <span>🧠 {_('Welcome to your Smart Digital Twin!')}</span>
        </div>
        <div style="margin-bottom:10px;">
            <span class="badge" style="font-size:1.1rem;background:{theme['accent']};color:{theme['text_on_accent']};">
                {_("Live Monitoring")} ⚡
            </span>
        </div>
        """), unsafe_allow_html=True)

        colA, colB = st.columns([4, 1])
        with colB:
            st.markdown(
                f"<div class='status-badge'>"
                f"🔴 {'CRITICAL' if st.session_state.get('simulate_disaster') else 'ONLINE'}</div>",
                unsafe_allow_html=True)
            if st.button("🚨 "+_("Simulate Disaster"), help=_("Trigger a simulated critical event.")):
                st.session_state["simulate_disaster"] = True
                st.session_state["simulate_time"] = time.time()
        if st.session_state.get("simulate_disaster") and time.time() - st.session_state.get("simulate_time", 0) > 30:
            st.session_state["simulate_disaster"] = False
        if st.session_state.get("simulate_disaster"):
            temp, pressure, vib, methane, h2s = 120, 340, 2.3, 9.5, 1.2
            st.markdown(f"<div class='card' style='background:{theme['alert']};color:{theme['alert_text']};font-size:1.25rem;'>🔥 {_('Critical disaster detected during simulation.')}</div>", unsafe_allow_html=True)
        else:
            temp, pressure, vib, methane, h2s = 82.7, 202.2, 0.61, 2.85, 0.30

        kpi_vals = [temp, pressure, vib, methane, h2s]
        kpi_labels = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
        kpi_units = ["°C", "psi", "g", "ppm", "ppm"]
        kpi_icons = ["🌡️", "💧", "🌀", "🟢", "⚗️"]
        kpi_metrics_row(kpi_vals, kpi_labels, kpi_units, kpi_icons)  # <-- Streamlit metric row

        st.markdown(rtl_wrap(
            f"""<div class="sub-title" style="margin-top:15px;margin-bottom:4px;">
                📈 {_("Live Data")}
            </div>"""
        ), unsafe_allow_html=True)

        dates = pd.date_range(end=pd.Timestamp.today(), periods=40)
        df = pd.DataFrame({
            _("Temperature"): 80 + 5 * np.random.rand(40),
            _("Pressure"): 200 + 10 * np.random.rand(40),
            _("Methane"): 2.5 + 0.5 * np.random.rand(40),
            _("Vibration"): 0.6 + 0.1 * np.random.rand(40),
            _("H2S"): 0.3 + 0.05 * np.random.rand(40)
        }, index=dates)
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(y=df[col], x=df.index, mode='lines', name=col, line=dict(width=3)))
        fig.update_layout(
            xaxis_title=_("Time"), yaxis_title=_("Trend"),
            plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['plot_bg'],
            font=dict(color=theme['text_on_primary']), legend=dict(orientation='h', y=1.02, x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(rtl_wrap(
            f"""<div class="sub-title" style="margin-top:20px;margin-bottom:4px;">
                🤖 {_("Prediction")}
            </div>"""
        ), unsafe_allow_html=True)
        x = np.arange(0, 7)
        temp_pred = 82 + 2 * np.sin(0.5 * x)
        pressure_pred = 200 + 3 * np.cos(0.5 * x)
        methane_pred = 2.8 + 0.2 * np.random.rand(7)
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=x, y=temp_pred, mode='lines+markers', name=_("Temperature")))
        pred_fig.add_trace(go.Scatter(x=x, y=pressure_pred, mode='lines+markers', name=_("Pressure")))
        pred_fig.add_trace(go.Scatter(x=x, y=methane_pred, mode='lines+markers', name=_("Methane")))
        pred_fig.update_layout(
            xaxis_title="Hours Ahead", yaxis_title=_("Forecast"),
            plot_bgcolor=theme["plot_bg"], paper_bgcolor=theme["plot_bg"],
            font=dict(color=theme["text_on_primary"])
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        st.toast(_("Dashboard loaded successfully!"))
    except Exception as e:
        st.error(_("An error occurred loading the dashboard: ") + str(e))

def show_predictive():
    st.markdown(rtl_wrap(f'<div class="big-title">🔮 {_("Predictive Analysis")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Forecast")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><b>{_("Temperature")}</b>: 84.2°C<br><b>{_("Pressure")}</b>: 205 psi<br><b>{_("Methane")}</b>: 3.1 ppm<br><span class="badge">{_("High Risk Area: Tank 3")}</span></div>'), unsafe_allow_html=True)
    kpi_vals = [84.2, 205, 0.66, 3.1, 0.32]
    kpi_labels = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    kpi_units = ["°C", "psi", "g", "ppm", "ppm"]
    kpi_icons = ["🌡️", "💧", "🌀", "🟢", "⚗️"]
    kpi_metrics_row(kpi_vals, kpi_labels, kpi_units, kpi_icons)
    x = np.arange(0, 7)
    temp_pred = 82 + 2 * np.sin(0.5 * x)
    pressure_pred = 200 + 3 * np.cos(0.5 * x)
    methane_pred = 2.8 + 0.2 * np.random.rand(7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=temp_pred, mode='lines+markers', name=_("Temperature")))
    fig.add_trace(go.Scatter(x=x, y=pressure_pred, mode='lines+markers', name=_("Pressure")))
    fig.add_trace(go.Scatter(x=x, y=methane_pred, mode='lines+markers', name=_("Methane")))
    fig.update_layout(
        xaxis_title="Hours Ahead", yaxis_title=_("Forecast"),
        plot_bgcolor=theme["plot_bg"], paper_bgcolor=theme["plot_bg"],
        font=dict(color=theme["text_on_primary"])
    )
    st.plotly_chart(fig, use_container_width=True)

def show_solutions():
    st.markdown(rtl_wrap(f'<div class="big-title">💡 {_("Smart Solutions")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("AI-powered recommendations for safety and efficiency")}</div>'), unsafe_allow_html=True)
    generate = st.button(_("Generate Solution"), help=_("Get an automatic recommendation for current conditions."))
    simulate = st.session_state.get("simulate_disaster", False)
    if generate or simulate:
        with st.spinner(_("Generating solution...")):
            if simulate:
                solutions = [{
                    "title": _("Emergency Vent Gas!"),
                    "details": _("Immediate venting required in Tank 2 due to critical methane spike."),
                    "reason": _("Critical disaster detected during simulation."),
                    "priority": _("URGENT"),
                    "effectiveness": _("99%"),
                    "estimated_time": _("Now")
                }]
            else:
                solutions = [{
                    "title": _("Reduce Pressure in Line 3"),
                    "details": _("Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection."),
                    "reason": _("Abnormal vibration detected. This reduces risk."),
                    "priority": _("High"),
                    "effectiveness": _("95%"),
                    "estimated_time": _("15 minutes")
                }]
        for sol in solutions:
            badge = f'<span class="badge" style="background:{theme["badge_bg"]};color:{theme["text_on_accent"]};font-size:1.08rem;">🔔 {_("Smart Recommendations")}</span>'
            st.markdown(rtl_wrap(
                f'<div class="card" style="box-shadow: 0 4px 24px rgba(0,0,0,0.14);background:linear-gradient(90deg,{theme["card_bg"]} 65%,{theme["accent"]} 100%);">'
                f"{badge}<br>"
                f"<b style='font-size:1.2rem'>{sol['title']}</b><br>"
                f"<b>{_('Details')}:</b> {sol['details']}<br>"
                f"<b>{_('Reason')}:</b> <span style='color:{theme['alert']};'>{sol['reason']}</span><br>"
                f"<b>{_('Priority')}:</b> <span style='color:{theme['alert']};font-weight:bold'>{sol['priority']}</span> "
                f"<b>{_('Effectiveness')}:</b> <span style='color:{theme['accent']}'>{sol['effectiveness']}</span> "
                f"<b>{_('Estimated Time')}:</b> <span style='color:{theme['secondary']}'>{sol['estimated_time']}</span>"
                f'</div>'
            ), unsafe_allow_html=True)
    else:
        st.info(_("Press 'Generate Solution' for intelligent suggestions."))

def show_alerts():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Smart Alerts")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Current Alerts")}</div>'), unsafe_allow_html=True)
    alerts = [
        {"timestamp": "2025-06-24 13:45", "location": "Tank 3", "msg": _("Methane Spike"), "severity": _("High")},
        {"timestamp": "2025-06-24 13:20", "location": "Pipeline 1", "msg": _("Pressure Drop"), "severity": _("Medium")},
        {"timestamp": "2025-06-24 12:55", "location": "Tank 1", "msg": _("Vibration Anomaly"), "severity": _("Low")},
        {"timestamp": "2025-06-24 12:45", "location": "Compressor B", "msg": _("High Temperature"), "severity": _("High")}
    ]
    if st.session_state.get("simulate_disaster", False):
        alerts.insert(0, {"timestamp": "NOW", "location": "Tank 2", "msg": _("Methane Spike"), "severity": _("High")})
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        df_alerts["severity_color"] = df_alerts["severity"].map({
            _("High"): "🔴", _("Medium"): "🟠", _("Low"): "🟢"
        })
        df_alerts = df_alerts[["severity_color", "severity", "timestamp", "location", "msg"]]
        df_alerts.columns = ["", _("Severity"), _("Time"), _("Location"), _("Message")]
        st.dataframe(df_alerts, hide_index=True, use_container_width=True)
    else:
        st.info(_("No alerts at the moment."))

def show_cost():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Cost & Savings")}</div>'), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.markdown(rtl_wrap(f'<div class="card"><div class="metric">5,215,000 SAR</div><div class="metric-label">{_("Yearly Savings")}</div></div>'), unsafe_allow_html=True)
    col2.markdown(rtl_wrap(f'<div class="card"><div class="metric">17%</div><div class="metric-label">{_("Reduction in Maintenance Costs")}</div></div>'), unsafe_allow_html=True)
    months = [f"{i+1}/2025" for i in range(6)]
    savings = [400000, 450000, 500000, 550000, 600000, 650000]
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Monthly Savings")}</div>'), unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=months, y=savings, marker_color=theme["accent"]))
    fig.update_layout(
        xaxis_title=_("Month"), yaxis_title=_("Savings"),
        plot_bgcolor=theme["plot_bg"], paper_bgcolor=theme["plot_bg"], font=dict(color=theme["secondary"]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Savings Breakdown")}</div>'), unsafe_allow_html=True)
    breakdown = pd.DataFrame({
        _("Source"): [_("Energy Efficiency"), _("Maintenance Reduction"), _("Downtime Prevention")],
        _("Amount (SAR)"): [2000000, 1500000, 1715000]
    })
    st.table(breakdown)

def show_achievements():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Achievements")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        '<div class="card"><span class="badge">🏆</span> ' +
        _("Congratulations!") + " " + _("You have achieved") +
        " <b>100</b> " + _("days without incidents") + "!</div>"), unsafe_allow_html=True)
    st.progress(0.85, text=_("Compared to last period"))
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Milestones")}</div>'), unsafe_allow_html=True)
    st.markdown("- 50 " + _("days without incidents"))
    st.markdown("- 3 " + _("months zero downtime"))
    st.markdown("- 10% " + _("energy efficiency improvement"))
    st.markdown("- " + _("2025 Innovation Award, Best Digital Twin"))

def show_performance():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Performance")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Savings"), _("Downtime (hrs)")]
    values_now = [82.7, 202.2, 650000, 2.1]
    values_prev = [85, 204, 500000, 8.4]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=values_now, name=_("Current"), marker_color=theme["accent"]))
    fig.add_trace(go.Bar(x=metrics, y=values_prev, name=_("Previous"), marker_color=theme["secondary"]))
    fig.update_layout(barmode='group', plot_bgcolor=theme["plot_bg"], paper_bgcolor=theme["plot_bg"], font=dict(color=theme["secondary"]))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Summary Table")}</div>'), unsafe_allow_html=True)
    summary = pd.DataFrame({
        _("Metric"): metrics, _("Current"): values_now, _("Previous"): values_prev,
        _("Change"): [now - prev for now, prev in zip(values_now, values_prev)]
    })
    st.table(summary)

def show_comparison():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Comparison")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Lets Compare!")}</div>'), unsafe_allow_html=True)
    months = [f"{i+1}/2025" for i in range(12)]
    metrics = [_("Temperature"), _("Pressure"), _("Savings")]
    data = pd.DataFrame({
        _("Temperature"): np.random.uniform(80, 90, 12),
        _("Pressure"): np.random.uniform(190, 210, 12),
        _("Savings"): np.random.uniform(400000, 700000, 12),
    }, index=months)
    st.line_chart(data)
    st.dataframe(data)
    st.markdown(rtl_wrap(f'<div class="card"><b>{_("Comparison")}</b> – ' + _("Performance Comparison") + '</div>'), unsafe_allow_html=True)

def show_explorer():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Data Explorer")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Data Filters")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    metric = st.selectbox(_("Select Metric"), options=metrics)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    data = pd.DataFrame({metric: 80 + 5 * np.random.rand(30)}, index=dates)
    st.line_chart(data)
    st.dataframe(data)

def show_about():
    st.markdown(rtl_wrap(f"""
    <div style="display: flex; align-items: center; justify-content:center;">
        <img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="72" style="margin-right:20px;" alt="AI logo"/>
        <div>
            <span class="big-title">{_("About the Project")}</span><br>
            <span class="sub-title" style="font-size:1.18rem;">{_("Smart Neural Digital Twin")}</span>
        </div>
    </div>
    """), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"<div class='about-card-gradient'><span style='font-size:1.2em;'>🧠</span> <b>{_('About Project Description')}</b></div>"
        ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"""<div class='card' style='font-style:italic;font-size:1.2rem;'><span class='badge'>{_('Our Vision')}</span>
        “{_("Disasters don't wait.. and neither do we.")}”</div>"""
    ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"""<div class='about-card-gradient'>
        <span class='badge'>✨ {_("Features")}</span>
        <div class='about-features'>
            <div><span class='fancy-icon'>🤖</span>{_('AI-powered predictive analytics')}</div>
            <div><span class='fancy-icon'>⚡</span>{_('Instant smart solutions')}</div>
            <div><span class='fancy-icon'>📡</span>{_('Live alerts and monitoring')}</div>
            <div><span class='fancy-icon'>🌐</span>{_('Multi-language support')}</div>
            <div><span class='fancy-icon'>🎨</span>{_('Stunning, responsive UI')}</div>
        </div>
        </div>"""
    ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"""<div class='about-card-gradient'>
        <span class="badge">🏆 {_("Milestones")}</span>
        <ul class='about-milestones'>
            <li>2024: MVP Launch 🚀</li>
            <li>2025: {_('2025 Innovation Award, Best Digital Twin')} 🥇</li>
            <li>100+ {_('days without incidents')} ⭐</li>
        </ul>
        </div>"""
    ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"""<div class='about-card-gradient'>
        <span class="badge">👨‍💻 {_("Main Developers")}</span>
        <div class='about-dev'>
            <div class='dev'>
                <img src="https://avatars.githubusercontent.com/u/rrakanmarri1?v=4" width="60"/><br>
                <b>Rakan Almarri</b><br>
                <span style="font-size:0.97em;">rakan.almarri.2@aramco.com</span>
            </div>
            <div class='dev'>
                <img src="https://ui-avatars.com/api/?name=Abdulrahman+Alzhrani&background=278ea5&color=fff" width="60"/><br>
                <b>Abdulrahman Alzhrani</b><br>
                <span style="font-size:0.97em;">abdulrahman.alzhrani.1@aramco.com</span>
            </div>
        </div>
        </div>"""
    ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        f"""<div class='about-contact'>📬 {_("Contact us for partnership or demo!")}<br>
            <a href="mailto:rakan.almarri.2@aramco.com" style="color:{theme['badge_bg']}; text-decoration:underline;">
                rakan.almarri.2@aramco.com
            </a>
        </div>"""
    ), unsafe_allow_html=True)

# 8. Routing & State
routes: Dict[str, Callable[[], None]] = {
    "dashboard": show_dashboard,
    "predictive": show_predictive,
    "solutions": show_solutions,
    "alerts": show_alerts,
    "cost": show_cost,
    "achievements": show_achievements,
    "performance": show_performance,
    "comparison": show_comparison,
    "explorer": show_explorer,
    "about": show_about
}

if "simulate_disaster" not in st.session_state:
    st.session_state["simulate_disaster"] = False
if "simulate_time" not in st.session_state:
    st.session_state["simulate_time"] = 0

selected_page = st.session_state.page_radio
routes[selected_page[0]]()
