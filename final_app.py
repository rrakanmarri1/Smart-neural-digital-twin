"""
Smart Neural Digital Twin
Enhanced, modular, and polished Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from typing import Dict, Callable

# =========================
# 1. Theme System & Preview
# =========================

THEME_SETS: Dict[str, Dict[str, str]] = {
    "Ocean": {"primary": "#153243", "secondary": "#278ea5", "accent": "#21e6c1",
              "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#153243",
              "sidebar_bg": "#18465b", "card_bg": "#278ea5", "badge_bg": "#21e6c1",
              "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#153243"},
    "Sunset": {"primary": "#ff7043", "secondary": "#ffa726", "accent": "#ffd54f",
               "text_on_primary": "#fff", "text_on_secondary": "#232526", "text_on_accent": "#232526",
               "sidebar_bg": "#ffb28f", "card_bg": "#ffa726", "badge_bg": "#ff7043",
               "alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fff3e0"},
    "Emerald": {"primary": "#154734", "secondary": "#43e97b", "accent": "#38f9d7",
                "text_on_primary": "#fff", "text_on_secondary": "#153243", "text_on_accent": "#154734",
                "sidebar_bg": "#1d5c41", "card_bg": "#43e97b", "badge_bg": "#38f9d7",
                "alert": "#ff1744", "alert_text": "#fff", "plot_bg": "#e0f2f1"},
    "Night": {"primary": "#232526", "secondary": "#414345", "accent": "#e96443",
              "text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#232526",
              "sidebar_bg": "#414345", "card_bg": "#232526", "badge_bg": "#e96443",
              "alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#232526"},
    "Blossom": {"primary": "#fbd3e9", "secondary": "#bb377d", "accent": "#fa709a",
                "text_on_primary": "#232526", "text_on_secondary": "#fff", "text_on_accent": "#fff",
                "sidebar_bg": "#fcb7d4", "card_bg": "#fa709a", "badge_bg": "#bb377d",
                "alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fce4ec"}
}
DEFAULT_THEME = "Ocean"

# =========================
# 2. Translations & i18n
# =========================

translations = {
    "en": {
        # General
        "Settings": "Settings", "Choose Language": "Choose Language",
        "English": "English", "Arabic": "Arabic",
        "Theme Set": "Theme Set", "Theme": "Theme", "Theme Preview": "Theme Preview",
        "Dashboard": "Dashboard", "Predictive Analysis": "Predictive Analysis",
        "Smart Solutions": "Smart Solutions", "Smart Alerts": "Smart Alerts",
        "Cost & Savings": "Cost & Savings", "Achievements": "Achievements",
        "Performance Comparison": "Performance Comparison", "Data Explorer": "Data Explorer",
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
        "About Project Description": "Smart Neural Digital Twin is an AI-powered disaster prevention platform for industrial sites and oilfields. It connects live sensors to an intelligent digital twin, predicting risks and suggesting instant smart solutions in real time.",
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
        # Themes
        "Ocean": "Ocean",
        "Sunset": "Sunset",
        "Emerald": "Emerald",
        "Night": "Night",
        "Blossom": "Blossom",
    },
    "ar": {
        "Settings": "الإعدادات", "Choose Language": "اختر اللغة",
        "English": "الإنجليزية", "Arabic": "العربية",
        "Theme Set": "مجموعة الألوان", "Theme": "السمة", "Theme Preview": "معاينة السمة",
        "Dashboard": "لوحة التحكم", "Predictive Analysis": "تحليل تنبؤي",
        "Smart Solutions": "حلول ذكية", "Smart Alerts": "تنبيهات ذكية",
        "Cost & Savings": "التكلفة والتوفير", "Achievements": "الإنجازات",
        "Performance Comparison": "مقارنة الأداء", "Data Explorer": "استكشاف البيانات",
        "About": "حول", "Navigate to": "انتقل إلى",
        "Welcome to your Smart Digital Twin!": "مرحبًا بك في التوأم الرقمي الذكي!",
        "Temperature": "درجة الحرارة", "Pressure": "الضغط", "Vibration": "الاهتزاز",
        "Methane": "الميثان", "H2S": "كبريتيد الهيدروجين", "Live Data": "بيانات مباشرة", "Trend": "الاتجاه",
        "Forecast": "التوقعات", "Simulate Disaster": "محاكاة كارثة",
        "Details": "التفاصيل", "Reason": "السبب", "Priority": "الأولوية",
        "Effectiveness": "الفعالية", "Estimated Time": "الوقت المتوقع",
        "Generate Solution": "توليد حل", "Generating solution...": "جاري توليد الحل…",
        "Press 'Generate Solution' for intelligent suggestions.": "اضغط 'توليد حل' للحصول على اقتراحات ذكية.",
        "Emergency Vent Gas!": "تنفيس الغاز فوراً!", "Immediate venting required in Tank 2 due to critical methane spike.": "مطلوب تنفيس فوري في الخزان 2 بسبب ارتفاع خطير في الميثان.",
        "Critical disaster detected during simulation.": "تم رصد كارثة حرجة أثناء المحاكاة.",
        "Reduce Pressure in Line 3": "قلل الضغط في الخط ٣", "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "قم بخفض الضغط بنسبة 15٪ في الخط ٣ ونبه فريق الصيانة للفحص.",
        "Abnormal vibration detected. This reduces risk.": "تم رصد اهتزاز غير طبيعي. هذا يقلل المخاطر.",
        "URGENT": "عاجل", "Now": "الآن", "High": "مرتفع", "15 minutes": "١٥ دقيقة", "95%": "٩٥٪", "99%": "٩٩٪",
        "About Project Description": "التوأم الرقمي العصبي الذكي هو منصة مدعومة بالذكاء الاصطناعي للوقاية من الكوارث في المواقع الصناعية وحقول النفط. يربط أجهزة الاستشعار الحية بتوأم رقمي ذكي يتنبأ بالمخاطر ويقترح حلولاً ذكية فورية.",
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
        "Medium": "متوسط",
        "Low": "منخفض",
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
        # Themes
        "Ocean": "أوشن",
        "Sunset": "غروب الشمس",
        "Emerald": "زمردي",
        "Night": "ليلي",
        "Blossom": "إزهار",
    }
}

def get_lang() -> str:
    """Get current language from session_state, default to Arabic."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ar"
    return st.session_state["lang"]

def set_lang(lang: str):
    """Set current language in session_state."""
    st.session_state["lang"] = lang

def _(key: str) -> str:
    """
    Translation function with fallback.
    Tries current language, then English, then returns key.
    """
    lang = get_lang()
    return translations.get(lang, {}).get(key) or translations["en"].get(key) or key

def rtl_wrap(html: str) -> str:
    """Wrap HTML in RTL if Arabic."""
    return f"<div class='rtl'>{html}</div>" if get_lang() == "ar" else html

# =========================
# 3. Theme and CSS Injection
# =========================

def set_theme_in_session():
    if "theme_set" not in st.session_state:
        st.session_state["theme_set"] = DEFAULT_THEME

set_theme_in_session()
theme = THEME_SETS[st.session_state["theme_set"]]

def inject_css():
    """Inject custom CSS for theming and accessibility."""
    st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {theme['primary']} !important; }}
    .stSidebar {{ background-color: {theme['sidebar_bg']} !important; }}
    .big-title {{ color: {theme['secondary']}; font-size:2.3rem; font-weight:bold; margin-bottom:10px; }}
    .sub-title {{ color: {theme['accent']}; font-size:1.4rem; margin-bottom:10px; }}
    .card {{
        background: {theme['card_bg']};
        border-radius:16px;
        padding:18px 24px;
        margin-bottom:16px;
        color:{theme['text_on_secondary']};
        display: inline-block;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }}
    .metric {{ font-size:2.1rem; font-weight:bold; }}
    .metric-label {{ font-size:1.1rem; color:{theme['accent']}; }}
    .badge {{ background:{theme['badge_bg']}; color:{theme['text_on_accent']}; padding:2px 12px; border-radius:20px; margin-right:10px; }}
    .rtl {{ direction:rtl; }}
    .theme-swatch {{
        display:inline-block; width:24px; height:24px; border-radius:8px;
        margin-right:8px; border:2px solid #3333; vertical-align:middle;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# =========================
# 4. Sidebar & Navigation
# =========================

def theme_selector():
    """Show theme selector with swatch and translated names."""
    theme_names = list(THEME_SETS.keys())
    previews = [
        f"<span class='theme-swatch' style='background:{THEME_SETS[name]['primary']}'></span> {_(name)}"
        for name in theme_names
    ]
    theme_idx = theme_names.index(st.session_state["theme_set"])
    choice = st.radio(
        _( "Theme Set" ),
        options=theme_names,
        format_func=lambda x: _(x),
        index=theme_idx,
        key="theme_radio"
    )
    if choice != st.session_state["theme_set"]:
        st.session_state["theme_set"] = choice
        st.rerun()
    # Show preview
    st.markdown("<div><b>"+_("Theme Preview")+"</b></div>", unsafe_allow_html=True)
    for i, preview in enumerate(previews):
        sel = "✅" if theme_names[i]==choice else ""
        st.markdown(f"{preview} {sel}", unsafe_allow_html=True)

def sidebar():
    """Sidebar: Settings and Navigation."""
    with st.sidebar:
        with st.expander(_("Settings"), expanded=True):
            lang_choice = st.radio(_("Choose Language"),
                                   options=["ar", "en"],
                                   format_func=lambda x: _("Arabic") if x == "ar" else _("English"),
                                   index=0 if get_lang() == "ar" else 1,
                                   key="lang_radio")
            set_lang(lang_choice)
            theme_selector()
        st.markdown("---")
        pages = [
            ("dashboard", _("Dashboard")), ("predictive", _("Predictive Analysis")),
            ("solutions", _("Smart Solutions")), ("alerts", _("Smart Alerts")),
            ("cost", _("Cost & Savings")), ("achievements", _("Achievements")),
            ("comparison", _("Performance Comparison")), ("explorer", _("Data Explorer")),
            ("about", _("About"))
        ]
        st.radio(_("Navigate to"), options=pages, format_func=lambda x: x[1], index=0, key="page_radio")

sidebar()

# =========================
# 5. Page Functions
# =========================

def show_dashboard():
    """Dashboard page with error handling and accessibility."""
    try:
        st.markdown(rtl_wrap(f"<div class='big-title'>{_('Welcome to your Smart Digital Twin!')}</div>"), unsafe_allow_html=True)
        colA, colB = st.columns([4, 1])
        with colB:
            if st.button("🚨 "+_("Simulate Disaster"), help=_("Trigger a simulated critical event.")):
                st.session_state["simulate_disaster"] = True
                st.session_state["simulate_time"] = time.time()
        # Disaster simulation logic
        if st.session_state.get("simulate_disaster") and time.time() - st.session_state.get("simulate_time", 0) > 30:
            st.session_state["simulate_disaster"] = False
        if st.session_state.get("simulate_disaster"):
            temp, pressure, vib, methane, h2s = 120, 340, 2.3, 9.5, 1.2
        else:
            temp, pressure, vib, methane, h2s = 82.7, 202.2, 0.61, 2.85, 0.30
        cols = st.columns(5)
        metrics = [temp, pressure, vib, methane, h2s]
        labels = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
        units = ["°C", "psi", "g", "ppm", "ppm"]
        for c, m, l, u in zip(cols, metrics, labels, units):
            c.markdown(rtl_wrap(f"<div class='card'><div class='metric'>{m}{u}</div><div class='metric-label'>{l}</div></div>"), unsafe_allow_html=True)
        st.markdown(rtl_wrap(f"<div class='sub-title'>{_('Live Data')}</div>"), unsafe_allow_html=True)
        # Random time series
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
        st.toast(_("Dashboard loaded successfully!"))
    except Exception as e:
        st.error(_("An error occurred loading the dashboard: ") + str(e))

def show_predictive():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Predictive Analysis")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Forecast")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><b>{_("Temperature")}</b>: 84.2°C<br><b>{_("Pressure")}</b>: 205 psi<br><b>{_("Methane")}</b>: 3.1 ppm<br><span class="badge">{_("High Risk Area: Tank 3")}</span></div>'), unsafe_allow_html=True)
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
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Smart Solutions")}</div>'), unsafe_allow_html=True)
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
    st.markdown(rtl_wrap(f'<div class="big-title">{_("About the Project")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(_("About Project Description")), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><span class="badge">{_("Our Vision")}</span><br><i>{_("Disasters don\'t wait.. and neither do we.")}</i></div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><span class='badge'>{_('Features')}</span><ul>"
        f"<li>{_('AI-powered predictive analytics')}</li>"
        f"<li>{_('Instant smart solutions')}</li>"
        f"<li>{_('Live alerts and monitoring')}</li>"
        f"<li>{_('Multi-language support')}</li>"
        f"<li>{_('Stunning, responsive UI')}</li>"
        "</ul></div>"), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><span class='badge'>{_('Main Developers')}</span><br>"
        "<b>Rakan Almarri:</b> rakan.almarri.2@aramco.com<br>"
        "<b>Abdulrahman Alzhrani:</b> abdulrahman.alzhrani.1@aramco.com"
        "</div>"), unsafe_allow_html=True)

# =========================
# 6. Routing
# =========================

routes: Dict[str, Callable[[], None]] = {
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

# Ensure disaster sim state exists
if "simulate_disaster" not in st.session_state:
    st.session_state["simulate_disaster"] = False
if "simulate_time" not in st.session_state:
    st.session_state["simulate_time"] = 0

selected_page = st.session_state.page_radio
routes[selected_page[0]]()
