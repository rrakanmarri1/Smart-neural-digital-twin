import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
import os
import time

# --- SESSION STATE INIT ---
if "simulate_disaster" not in st.session_state:
    st.session_state["simulate_disaster"] = False
if "simulate_time" not in st.session_state:
    st.session_state["simulate_time"] = 0

# --- COLOR THEMES ---
THEME_SETS = {
    "Ocean": {
        "primary": "#153243", "secondary": "#278ea5", "accent": "#21e6c1",
        "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#153243",
        "sidebar_bg": "#18465b", "card_bg": "#278ea5", "badge_bg": "#21e6c1",
        "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#153243"
    },
    "Sunset": {
        "primary": "#ff7043", "secondary": "#ffa726", "accent": "#ffd54f",
        "text_on_primary": "#fff", "text_on_secondary": "#232526", "text_on_accent": "#232526",
        "sidebar_bg": "#ffb28f", "card_bg": "#ffa726", "badge_bg": "#ff7043",
        "alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fff3e0"
    },
    "Emerald": {
        "primary": "#154734", "secondary": "#43e97b", "accent": "#38f9d7",
        "text_on_primary": "#fff", "text_on_secondary": "#153243", "text_on_accent": "#154734",
        "sidebar_bg": "#1d5c41", "card_bg": "#43e97b", "badge_bg": "#38f9d7",
        "alert": "#ff1744", "alert_text": "#fff", "plot_bg": "#e0f2f1"
    },
    "Night": {
        "primary": "#232526", "secondary": "#414345", "accent": "#e96443",
        "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#232526",
        "sidebar_bg": "#414345", "card_bg": "#232526", "badge_bg": "#e96443",
        "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#232526"
    },
    "Blossom": {
        "primary": "#fbd3e9", "secondary": "#bb377d", "accent": "#fa709a",
        "text_on_primary": "#232526", "text_on_secondary": "#fff", "text_on_accent": "#fff",
        "sidebar_bg": "#fcb7d4", "card_bg": "#fa709a", "badge_bg": "#bb377d",
        "alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fce4ec"
    }
}
DEFAULT_THEME = "Ocean"
if "theme_set" not in st.session_state:
    st.session_state["theme_set"] = DEFAULT_THEME
theme = THEME_SETS[st.session_state["theme_set"]]

# --- TRANSLATIONS ---
translations = {
    "en": {
        "Settings": "Settings", "Choose Language": "Choose Language",
        "Dashboard": "Dashboard", "Predictive Analysis": "Predictive Analysis",
        "Smart Solutions": "Smart Solutions", "Smart Alerts": "Smart Alerts",
        "Cost & Savings": "Cost & Savings", "Achievements": "Achievements",
        "Performance Comparison": "Performance Comparison", "Data Explorer": "Data Explorer",
        "About": "About", "Navigate to": "Navigate to",
        "Welcome to your Smart Digital Twin!": "Welcome to your Smart Neural Digital Twin!",
        "Temperature": "Temperature", "Pressure": "Pressure",
        "Vibration": "Vibration", "Methane": "Methane", "H2S": "H2S",
        "Live Data": "Live Data", "Trend": "Trend", "Forecast": "Forecast",
        "Risk Level": "Risk Level", "Best Solution": "Best Solution",
        "Smart Recommendations": "Smart Recommendations", "Reason": "Reason",
        "Apply": "Apply", "Export": "Export", "Feedback": "Feedback",
        "No alerts at the moment.": "No alerts at the moment.",
        "Generate Solution": "Generate Solution", "Generating solution...": "Generating solution...",
        "Press 'Generate Solution' for intelligent suggestions.": "Press 'Generate Solution' for intelligent suggestions.",
        "High Pressure Detected in Zone 2!": "High Pressure Detected in Zone 2!",
        "Methane levels rising in Tank 1.": "Methane levels rising in Tank 1.",
        "Yearly Savings": "Yearly Savings", "Monthly Savings": "Monthly Savings", "Savings": "Savings",
        "Milestone": "Milestone", "Congratulations!": "Congratulations!", "You have achieved": "You have achieved",
        "days without incidents": "days without incidents", "Compared to last period": "Compared to last period",
        "Current": "Current", "Previous": "Previous", "Select Metric": "Select Metric",
        "Data Filters": "Data Filters", "About the Project": "About the Project",
        "Our Vision": "Our Vision", "Project Summary": "Project Summary",
        "What does it do?": "What does it do?", "Features": "Features",
        "AI-powered predictive analytics": "AI-powered predictive analytics",
        "Instant smart solutions": "Instant smart solutions", "Live alerts and monitoring": "Live alerts and monitoring",
        "Multi-language support": "Multi-language support", "Stunning, responsive UI": "Stunning, responsive UI",
        "Main Developers": "Main Developers", "Contact information available upon request.": "Contact information available upon request.",
        "Disasters don't wait.. and neither do we.": "Disasters don't wait.. and neither do we.",
        "Smart Digital Twin is an advanced platform for oilfield safety that connects to real sensors, predicts anomalies, and offers actionable insights to prevent disasters before they happen.": "Smart Digital Twin is an advanced platform for oilfield safety that connects to real sensors, predicts anomalies, and offers actionable insights to prevent disasters before they happen.",
        "Details": "Details", "Priority": "Priority", "Effectiveness": "Effectiveness",
        "Estimated Time": "Estimated Time", "Severity": "Severity", "Time": "Time",
        "Location": "Location", "Message": "Message", "Methane Spike": "Methane Spike",
        "Pressure Drop": "Pressure Drop", "Vibration Anomaly": "Vibration Anomaly",
        "High Temperature": "High Temperature", "Reduction in Maintenance Costs": "Reduction in Maintenance Costs",
        "Month": "Month", "Savings Breakdown": "Savings Breakdown", "Source": "Source",
        "Energy Efficiency": "Energy Efficiency", "Maintenance Reduction": "Maintenance Reduction",
        "Downtime Prevention": "Downtime Prevention", "Amount (SAR)": "Amount (SAR)",
        "Milestones": "Milestones", "months zero downtime": "months zero downtime",
        "energy efficiency improvement": "energy efficiency improvement", "Innovation Award, Best Digital Twin": "Innovation Award, Best Digital Twin",
        "Downtime (hrs)": "Downtime (hrs)", "Summary Table": "Summary Table",
        "Metric": "Metric", "Change": "Change", "Reduce Pressure in Line 3": "Reduce Pressure in Line 3",
        "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.",
        "Abnormal vibration detected. This reduces risk.": "Abnormal vibration detected. This reduces risk.",
        "Schedule Pump Maintenance": "Schedule Pump Maintenance",
        "Temperature rising above normal.": "Temperature rising above normal.",
        "Emergency Vent Gas!": "Emergency Vent Gas!",
        "Immediate venting required in Tank 2 due to critical methane spike.": "Immediate venting required in Tank 2 due to critical methane spike.",
        "Critical disaster detected during simulation.": "Critical disaster detected during simulation.",
        "URGENT": "URGENT",
        "Now": "Now",
        "High": "High",
        "15 minutes": "15 minutes",
        "95%": "95%",
        "99%": "99%",
        "Medium": "Medium",
        "2 hours": "2 hours",
        "Low": "Low"
    },
    "ar": {
        "Settings": "الإعدادات", "Choose Language": "اختر اللغة",
        "Dashboard": "لوحة التحكم", "Predictive Analysis": "تحليل تنبؤي",
        "Smart Solutions": "حلول ذكية", "Smart Alerts": "تنبيهات ذكية",
        "Cost & Savings": "التكلفة والتوفير", "Achievements": "الإنجازات",
        "Performance Comparison": "مقارنة الأداء", "Data Explorer": "استكشاف البيانات",
        "About": "حول", "Navigate to": "انتقل إلى",
        "Welcome to your Smart Digital Twin!": "مرحبًا بك في التوأم الرقمي الذكي!",
        "Temperature": "درجة الحرارة", "Pressure": "الضغط", "Vibration": "الاهتزاز",
        "Methane": "الميثان", "H2S": "كبريتيد الهيدروجين",
        "Live Data": "بيانات مباشرة", "Trend": "الاتجاه", "Forecast": "التوقعات",
        "Risk Level": "مستوى الخطر", "Best Solution": "أفضل حل", "Smart Recommendations": "توصيات ذكية",
        "Reason": "السبب", "Apply": "تطبيق", "Export": "تصدير", "Feedback": "ملاحظات",
        "No alerts at the moment.": "لا توجد تنبيهات حالياً.", "Generate Solution": "توليد حل",
        "Generating solution...": "جاري توليد الحل...", "Press 'Generate Solution' for intelligent suggestions.": "اضغط 'توليد حل' للحصول على اقتراحات ذكية.",
        "High Pressure Detected in Zone 2!": "تم رصد ضغط مرتفع في المنطقة 2!", "Methane levels rising in Tank 1.": "ارتفاع مستويات الميثان في الخزان 1.",
        "Yearly Savings": "التوفير السنوي", "Monthly Savings": "التوفير الشهري", "Savings": "التوفير",
        "Milestone": "إنجاز", "Congratulations!": "تهانينا!", "You have achieved": "لقد حققت",
        "days without incidents": "يوماً بدون حوادث", "Compared to last period": "مقارنة بالفترة السابقة",
        "Current": "الحالي", "Previous": "السابق", "Select Metric": "اختر المقياس",
        "Data Filters": "تصفية البيانات", "About the Project": "عن المشروع",
        "Our Vision": "رؤيتنا", "Project Summary": "ملخص المشروع", "What does it do?": "ماذا يقدم؟",
        "Features": "الميزات", "AI-powered predictive analytics": "تحليلات تنبؤية مدعومة بالذكاء الاصطناعي",
        "Instant smart solutions": "حلول ذكية فورية", "Live alerts and monitoring": "تنبيهات ومراقبة مباشرة",
        "Multi-language support": "دعم تعدد اللغات", "Stunning, responsive UI": "واجهة مستخدم مذهلة وتفاعلية",
        "Main Developers": "المطورون الرئيسيون", "Contact information available upon request.": "معلومات التواصل متوفرة عند الطلب.",
        "Disasters don't wait.. and neither do we.": "الكوارث لا تنتظر.. ولا نحن أيضاً.",
        "Smart Digital Twin is an advanced platform for oilfield safety that connects to real sensors, predicts anomalies, and offers actionable insights to prevent disasters before they happen.": "التوأم الرقمي الذكي هو منصة متقدمة لسلامة الحقول النفطية تتصل بحساسات حقيقية وتتنبأ بالحالات الشاذة وتقدم حلولًا ذكية لمنع الكوارث قبل وقوعها.",
        "Details": "التفاصيل", "Priority": "الأولوية", "Effectiveness": "الفعالية", "Estimated Time": "الوقت المتوقع",
        "Severity": "الخطورة", "Time": "الوقت", "Location": "الموقع", "Message": "الرسالة",
        "Methane Spike": "ارتفاع الميثان", "Pressure Drop": "انخفاض الضغط", "Vibration Anomaly": "شذوذ الاهتزاز",
        "High Temperature": "درجة حرارة مرتفعة", "Reduction in Maintenance Costs": "خفض تكاليف الصيانة",
        "Month": "الشهر", "Savings Breakdown": "تفاصيل التوفير", "Source": "المصدر",
        "Energy Efficiency": "كفاءة الطاقة", "Maintenance Reduction": "تقليل الصيانة", "Downtime Prevention": "منع التوقف",
        "Amount (SAR)": "المبلغ (ريال)", "Milestones": "الإنجازات", "months zero downtime": "أشهر بدون توقف",
        "energy efficiency improvement": "تحسين كفاءة الطاقة", "Innovation Award, Best Digital Twin": "جائزة الابتكار - أفضل توأم رقمي",
        "Downtime (hrs)": "التوقف (ساعات)", "Summary Table": "جدول ملخص", "Metric": "المقياس", "Change": "التغير",
        "Reduce Pressure in Line 3": "قلل الضغط في الخط ٣",
        "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "قم بخفض الضغط بنسبة 15٪ في الخط 3 ونبّه فريق الصيانة للفحص.",
        "Abnormal vibration detected. This reduces risk.": "تم رصد اهتزاز غير طبيعي. هذا يقلل المخاطر.",
        "Schedule Pump Maintenance": "جدولة صيانة المضخة", "Temperature rising above normal.": "ارتفاع درجة الحرارة عن الحد الطبيعي.",
        "Emergency Vent Gas!": "تنفيس الغاز فوراً!",
        "Immediate venting required in Tank 2 due to critical methane spike.": "مطلوب تنفيس فوري في الخزان 2 بسبب ارتفاع خطير في الميثان.",
        "Critical disaster detected during simulation.": "تم رصد كارثة حرجة أثناء المحاكاة.",
        "URGENT": "عاجل",
        "Now": "الآن",
        "High": "مرتفع",
        "15 minutes": "15 دقيقة",
        "95%": "٩٥٪",
        "99%": "٩٩٪",
        "Medium": "متوسط",
        "2 hours": "ساعتان",
        "Low": "منخفض"
    }
}
def get_lang():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ar"
    return st.session_state["lang"]
def set_lang(lang):
    st.session_state["lang"] = lang
def _(key):
    lang = get_lang()
    return translations.get(lang, translations["en"]).get(key, key)

# --- CSS for THEME ---
st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {theme['primary']} !important; }}
    .stSidebar {{ background-color: {theme['sidebar_bg']} !important; }}
    .big-title {{ color: {theme['secondary']}; font-size:2.3rem; font-weight:bold; margin-bottom:10px; }}
    .sub-title {{ color: {theme['accent']}; font-size:1.4rem; margin-bottom:10px; }}
    .card {{ background: {theme['card_bg']}; border-radius: 16px; padding: 18px 24px; margin-bottom:16px; color: {theme['text_on_secondary']}; }}
    .metric {{font-size:2.1rem; font-weight:bold;}}
    .metric-label {{font-size:1.1rem; color:{theme['accent']};}}
    .alert-custom {{background:{theme['alert']}; color:{theme['alert_text']}; border-radius:12px; padding:12px; font-weight:bold;}}
    .badge {{ background: {theme['badge_bg']}; color:{theme['text_on_accent']}; padding: 2px 12px; border-radius: 20px; margin-right: 10px;}}
    .rtl {{ direction: rtl; }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    with st.expander(_("Settings"), expanded=True):
        lang_choice = st.radio(
            _("Choose Language"),
            options=["ar", "en"],
            format_func=lambda x: _("Arabic") if x == "ar" else _("English"),
            index=0 if get_lang() == "ar" else 1,
            key="lang_radio"
        )
        set_lang(lang_choice)
        theme_set = st.selectbox("Theme Set", options=list(THEME_SETS.keys()), index=list(THEME_SETS.keys()).index(st.session_state["theme_set"]))
        if theme_set != st.session_state["theme_set"]:
            st.session_state["theme_set"] = theme_set
            st.rerun()
    st.markdown("---")
    pages = [
        ("dashboard", _("Dashboard")),
        ("predictive", _("Predictive Analysis")),
        ("solutions", _("Smart Solutions")),
        ("alerts", _("Smart Alerts")),
        ("cost", _("Cost & Savings")),
        ("achievements", _("Achievements")),
        ("comparison", _("Performance Comparison")),
        ("explorer", _("Data Explorer")),
        ("about", _("About")),
    ]
    page = st.radio(_("Navigate to"), options=pages, format_func=lambda x: x[1], index=0, key="page_radio")

def rtl_wrap(html):
    return f'<div class="rtl">{html}</div>' if get_lang() == "ar" else html

# --- DASHBOARD ---
def show_dashboard():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Welcome to your Smart Digital Twin!")}'), unsafe_allow_html=True)
    colA, colB = st.columns([4,1])
    with colB:
        if st.button("🚨 Simulate Disaster"):
            st.session_state["simulate_disaster"] = True
            st.session_state["simulate_time"] = time.time()
    if st.session_state.get("simulate_disaster", False):
        if time.time() - st.session_state.get("simulate_time", 0) > 30:
            st.session_state["simulate_disaster"] = False
    # Display sensor data
    if st.session_state.get("simulate_disaster", False):
        temp = 120; pressure = 340; vib = 2.3; methane = 9.5; h2s = 1.2
    else:
        temp = 82.7; pressure = 202.2; vib = 0.61; methane = 2.85; h2s = 0.30
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(rtl_wrap(f'<div class="card"><div class="metric">{temp}°C</div><div class="metric-label">{_("Temperature")}</div></div>'), unsafe_allow_html=True)
    col2.markdown(rtl_wrap(f'<div class="card"><div class="metric">{pressure} psi</div><div class="metric-label">{_("Pressure")}</div></div>'), unsafe_allow_html=True)
    col3.markdown(rtl_wrap(f'<div class="card"><div class="metric">{vib} g</div><div class="metric-label">{_("Vibration")}</div></div>'), unsafe_allow_html=True)
    col4.markdown(rtl_wrap(f'<div class="card"><div class="metric">{methane} ppm</div><div class="metric-label">{_("Methane")}</div></div>'), unsafe_allow_html=True)
    col5.markdown(rtl_wrap(f'<div class="card"><div class="metric">{h2s} ppm</div><div class="metric-label">{_("H2S")}</div></div>'), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Live Data")}</div>'), unsafe_allow_html=True)

# --- PREDICTIVE ANALYSIS ---
def show_predictive():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Predictive Analysis")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Forecast")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><b>{_("Temperature")}</b>: 84.2°C<br><b>{_("Pressure")}</b>: 205 psi<br><b>{_("Methane")}</b>: 3.1 ppm<br><span class="badge">High Risk Area: Tank 3</span></div>'), unsafe_allow_html=True)
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

# --- SMART SOLUTIONS ---
def show_solutions():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Smart Solutions")}</div>'), unsafe_allow_html=True)
    generate = st.button(_("Generate Solution"))
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
            badge = f'<span class="badge">{_("Smart Recommendations")}</span>'
            st.markdown(rtl_wrap(
                f'<div class="card">'
                f"{badge}<br>"
                f"<b>{sol['title']}</b><br>"
                f"<b>{_('Details')}:</b> {sol['details']}<br>"
                f"<b>{_('Reason')}:</b> {sol['reason']}<br>"
                f"<b>{_('Priority')}:</b> {sol['priority']}<br>"
                f"<b>{_('Effectiveness')}:</b> {sol['effectiveness']}<br>"
                f"<b>{_('Estimated Time')}:</b> {sol['estimated_time']}<br>"
                f'</div>'
            ), unsafe_allow_html=True)
    else:
        st.info(_("Press 'Generate Solution' for intelligent suggestions."))

# --- SMART ALERTS ---
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

# --- COST & SAVINGS ---
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

# --- ACHIEVEMENTS ---
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

# --- PERFORMANCE COMPARISON ---
def show_comparison():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Performance Comparison")}</div>'), unsafe_allow_html=True)
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

# --- DATA EXPLORER ---
def show_explorer():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Data Explorer")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Data Filters")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    metric = st.selectbox(_("Select Metric"), options=metrics)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    data = pd.DataFrame({metric: 80 + 5 * np.random.rand(30)}, index=dates)
    st.line_chart(data)
    st.dataframe(data)

# --- ABOUT PAGE ---
def show_about():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("About the Project")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(
        "Smart Neural Digital Twin is an AI-powered disaster prevention platform for industrial sites and oilfields. It connects live sensors to an intelligent digital twin that predicts anomalies, generates instant smart solutions, and helps operators prevent accidents, downtime, and losses. The platform features multi-language support and interactive dashboards, making it accessible and actionable for everyone."
    ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><span class="badge">{_("Our Vision")}</span><br><i>{_("Disasters don\'t wait.. and neither do we.")}</i></div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><span class='badge'>{_('Features')}</span><ul>"
        f"<li>{_('AI-powered predictive analytics')}</li>"
        f"<li>{_('Instant smart solutions')}</li>"
        f"<li>{_('Live alerts and monitoring')}</li>"
        f"<li>{_('Multi-language support')}</li>"
        f"<li>{_('Stunning, responsive UI')}</li>"
        "</ul></div>"), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><span class='badge'>{_('Main Developers')}</span><br>"
        "<b>Rakan Almarri:</b> rakan.almarri.2@aramco.com &nbsp; <b>Phone:</b> 0532559664<br>"
        "<b>Abdulrahman Alzhrani:</b> abdulrahman.alzhrani.1@aramco.com &nbsp; <b>Phone:</b> 0549202574"
        "</div>"), unsafe_allow_html=True)

# --- ROUTING ---
routes = {
    "dashboard": show_dashboard,
    "predictive": show_predictive,
    "solutions": show_solutions,
    "alerts": show_alerts,
    "cost": show_cost,
    "achievements": show_achievements,
    "comparison": show_comparison,
    "explorer": show_explorer,
    "about": show_about
}
routes[st.session_state.page_radio[0][0]]()
