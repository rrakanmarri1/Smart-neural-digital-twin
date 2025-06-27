"""
Smart Neural Digital Twin – All Features, All 'Not Included' Demos, AI Icon Everywhere

- Adds demo avatars for both Rakan Almarri ("RA" green) and Abdulrahman Alzhrani ("AA" blue) in About.
- AI icon appears at the top of every main page.
- Adds demo toggles and/or mock logic for:
    - Advanced ML Models (ARIMA/Prophet demo selectors)
    - "Live Data" streaming demo
    - Demo user login (with avatar)
    - Demo email/SMS alert
    - Theme selector (already present)
    - Data upload ("Bring Your Own CSV")
    - Export to Excel/PDF (PDF = demo)
    - API integration (demo)
    - Multi-step What-If
    - Demo settings save
- All new translation keys included in both English & Arabic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time

# =========================
# 1. Translations (all used keys, new and old)
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
        "Data Explorer": "Data Explorer", "About": "About", "Navigate to": "Navigate to",
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
        "About Project Description": "Smart Neural Digital Twin is an AI-powered disaster prevention platform for industrial sites and oilfields. It connects live sensors to an intelligent digital twin for prediction, alerting, and instant smart solutions.",
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
        "Lets Compare!": "Lets Compare!",
        # NEW
        "Login": "Login",
        "Username": "Username",
        "Password": "Password",
        "Login as demo user": "Login as demo user",
        "Logged in as": "Logged in as",
        "Log out": "Log out",
        "Live Mode": "Live Mode",
        "Switch to Live Mode": "Switch to Live Mode",
        "Switch to History Mode": "Switch to History Mode",
        "Advanced Model": "Advanced Model",
        "Linear Regression": "Linear Regression",
        "ARIMA (Demo)": "ARIMA (Demo)",
        "Prophet (Demo)": "Prophet (Demo)",
        "Demo Email/SMS Alert": "Demo Email/SMS Alert",
        "Send Alerts (Demo)": "Send Alerts (Demo)",
        "Alerts have been sent (Demo)!": "Alerts have been sent to your registered contact info (Demo Only)!",
        "Upload Your Own CSV": "Upload Your Own CSV",
        "Uploaded! Using your CSV.": "Uploaded! Using your CSV.",
        "Export to Excel": "Export to Excel",
        "Export to PDF (Demo)": "Export to PDF (Demo)",
        "PDF generated (Demo)!": "PDF report generated and downloaded (Demo).",
        "API Integration (Demo)": "API Integration (Demo)",
        "Show API Demo": "Show API Demo",
        "API Data (Demo)": "API Data (Demo)",
        "Settings Saved! (Demo)": "Settings Saved! (Demo, will not persist after closing tab).",
        "Save My Settings (Demo)": "Save My Settings (Demo)",
        "Multi-Step What-If": "Multi-Step What-If",
        "Add What-If Step": "Add What-If Step",
        "Remove Step": "Remove Step",
        "Apply Multi-Step": "Apply Multi-Step",
        "No What-If steps set": "No What-If steps set"
    },
    "ar": {
        "Settings": "الإعدادات", "Choose Language": "اختر اللغة",
        "English": "الإنجليزية", "Arabic": "العربية",
        "Theme Set": "مجموعة الألوان", "Theme": "السمة", "Theme Preview": "معاينة السمة",
        "Dashboard": "لوحة التحكم", "Predictive Analysis": "تحليل تنبؤي",
        "Smart Solutions": "حلول ذكية", "Smart Alerts": "تنبيهات ذكية",
        "Cost & Savings": "التكلفة والتوفير", "Achievements": "الإنجازات",
        "Performance": "الأداء", "Comparison": "مقارنة", "Performance Comparison": "مقارنة الأداء",
        "Data Explorer": "استكشاف البيانات", "About": "حول", "Navigate to": "انتقل إلى",
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
        "Reduce Pressure in Line 3": "قلل الضغط في الخط ٣", "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "قم بخفض الضغط بنسبة 15% في الخط ٣ وأبلغ فريق الصيانة للفحص.",
        "Abnormal vibration detected. This reduces risk.": "تم رصد اهتزاز غير طبيعي. هذا يقلل المخاطر.",
        "URGENT": "عاجل", "Now": "الآن", "High": "مرتفعة", "15 minutes": "١٥ دقيقة", "95%": "٩٥٪", "99%": "٩٩٪",
        "About Project Description": "التوأم الرقمي العصبي الذكي هو منصة مدعومة بالذكاء الاصطناعي للوقاية من الكوارث في المواقع الصناعية وحقول النفط. يربط المستشعرات الحية بتوأم رقمي ذكي للتنبؤ والتنبيه والحلول الفورية.",
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
        "Lets Compare!": "لنقارن!",
        # NEW
        "Login": "تسجيل الدخول",
        "Username": "اسم المستخدم",
        "Password": "كلمة المرور",
        "Login as demo user": "دخول كمستخدم تجريبي",
        "Logged in as": "تم الدخول باسم",
        "Log out": "تسجيل الخروج",
        "Live Mode": "وضع البث المباشر",
        "Switch to Live Mode": "الانتقال لوضع البث المباشر",
        "Switch to History Mode": "العودة لوضع البيانات التاريخية",
        "Advanced Model": "نموذج متقدم",
        "Linear Regression": "انحدار خطي",
        "ARIMA (Demo)": "ARIMA (تجريبي)",
        "Prophet (Demo)": "Prophet (تجريبي)",
        "Demo Email/SMS Alert": "تنبيه عبر البريد/الجوال (تجريبي)",
        "Send Alerts (Demo)": "إرسال تنبيهات (تجريبي)",
        "Alerts have been sent (Demo)!": "تم إرسال التنبيهات (تجريبي)!",
        "Upload Your Own CSV": "تحميل ملف CSV خاص بك",
        "Uploaded! Using your CSV.": "تم التحميل! تم استخدام ملفك.",
        "Export to Excel": "تصدير إلى Excel",
        "Export to PDF (Demo)": "تصدير إلى PDF (تجريبي)",
        "PDF generated (Demo)!": "تم إنشاء التقرير PDF (تجريبي).",
        "API Integration (Demo)": "التكامل مع API (تجريبي)",
        "Show API Demo": "عرض مثال API",
        "API Data (Demo)": "بيانات API (تجريبي)",
        "Settings Saved! (Demo)": "تم حفظ الإعدادات (تجريبي، لا تحفظ عند إغلاق المتصفح).",
        "Save My Settings (Demo)": "حفظ الإعدادات (تجريبي)",
        "Multi-Step What-If": "ماذا لو متعددة الخطوات",
        "Add What-If Step": "إضافة خطوة ماذا لو",
        "Remove Step": "حذف خطوة",
        "Apply Multi-Step": "تطبيق عدة خطوات",
        "No What-If steps set": "لا توجد خطوات ماذا لو"
    }
}
def _(key): return translations[st.session_state.get("lang", "en")].get(key, key)

# =========================
# 2. Session and Theme State
# =========================
if "lang" not in st.session_state: st.session_state["lang"] = "en"
if "theme_set" not in st.session_state: st.session_state["theme_set"] = "Ocean"
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "current_user" not in st.session_state: st.session_state["current_user"] = "Guest"
if "simulate_disaster" not in st.session_state: st.session_state["simulate_disaster"] = False
if "simulate_time" not in st.session_state: st.session_state["simulate_time"] = 0
if "live_mode" not in st.session_state: st.session_state["live_mode"] = False
if "uploaded_df" not in st.session_state: st.session_state["uploaded_df"] = None
if "multi_whatif" not in st.session_state: st.session_state["multi_whatif"] = []

# =========================
# 3. Demo Login Block
# =========================
def demo_login_block():
    st.markdown('<img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="40" style="margin-bottom:-10px;margin-right:8px;"/>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.2em; font-weight:bold;'>{_('Login')}</div>", unsafe_allow_html=True)
    if not st.session_state["logged_in"]:
        # Demo login (no real check)
        username = st.text_input(_("Username"), "demo")
        password = st.text_input(_("Password"), type="password")
        if st.button(_("Login as demo user")):
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = username
            st.success(f"{_('Logged in as')}: {username}")
    else:
        st.success(f"{_('Logged in as')}: {st.session_state['current_user']}")
        if st.button(_("Log out")):
            st.session_state["logged_in"] = False
            st.session_state["current_user"] = "Guest"

# =========================
# 4. Theme System (Demo: unchanged from before)
# =========================
THEME_SETS = {
    "Ocean": {"primary": "#153243", "secondary": "#278ea5", "accent": "#21e6c1","text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#153243","sidebar_bg": "#18465b", "card_bg": "#278ea5", "badge_bg": "#21e6c1","alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#153243"},
    "Sunset": {"primary": "#FF7043","secondary": "#FFA726","accent": "#FFD54F","text_on_primary": "#232526","text_on_secondary": "#111","text_on_accent": "#232526","sidebar_bg": "#FFD9A0","card_bg": "#FFE0B2","badge_bg": "#FFA726","alert": "#D7263D","alert_text": "#fff","plot_bg": "#FFF3E0"},
    "Emerald": {"primary": "#154734", "secondary": "#43e97b", "accent": "#38f9d7","text_on_primary": "#fff", "text_on_secondary": "#153243", "text_on_accent": "#154734","sidebar_bg": "#e0f2f1", "card_bg": "#e8fff3", "badge_bg": "#38f9d7","alert": "#ff1744", "alert_text": "#fff", "plot_bg": "#e0f2f1"},
    "Night": {"primary": "#232526", "secondary": "#414345", "accent": "#e96443","text_on_primary": "#fff", "text_on_secondary": "#fff", "text_on_accent": "#232526","sidebar_bg": "#353749", "card_bg": "#414345", "badge_bg": "#e96443","alert": "#ff3e3e", "alert_text": "#fff", "plot_bg": "#232526"},
    "Blossom": {"primary": "#fbd3e9", "secondary": "#bb377d", "accent": "#fa709a","text_on_primary": "#232526", "text_on_secondary": "#fff", "text_on_accent": "#fff","sidebar_bg": "#fce4ec", "card_bg": "#f8bbd0", "badge_bg": "#bb377d","alert": "#d7263d", "alert_text": "#fff", "plot_bg": "#fce4ec"},
}
theme = THEME_SETS[st.session_state["theme_set"]]

# =========================
# 5. CSS
# =========================
def inject_css():
    st.markdown(f"""
    <style>
    body, .stApp {{ background: linear-gradient(120deg, {theme['primary']} 60%, {theme['secondary']} 100%) !important; min-height:100vh; }}
    .stSidebar {{ background-color: {theme['sidebar_bg']} !important; }}
    .big-title {{ color: {theme['accent']}; font-size:2.8rem; font-weight:bold; margin-bottom:10px; letter-spacing:0.04em; text-shadow: 1px 2px 12px rgba(0,0,0,0.08); }}
    .about-dev {{ display: flex; gap: 45px; align-items: center; justify-content: center; margin-top:18px; }}
    .about-dev .dev {{ text-align:center; background:rgba(255,255,255,0.08); border-radius:18px; padding:12px 26px 11px 26px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    .about-dev img {{ border-radius:50%; border:3px solid {theme['badge_bg']}; margin-bottom:7px; }}
    </style>
    """, unsafe_allow_html=True)
inject_css()

# =========================
# 6. Sidebar: Theme, Lang, Page Nav, Data Upload
# =========================
def sidebar():
    with st.sidebar:
        demo_login_block()
        st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
        theme_names = list(THEME_SETS.keys())
        st.session_state["theme_set"] = st.radio(
            _("Theme Set"),
            options=theme_names,
            format_func=lambda x: _(x),
            index=theme_names.index(st.session_state["theme_set"])
        )
        st.session_state["lang"] = st.radio(_("Choose Language"),
                                            options=["ar", "en"],
                                            format_func=lambda x: _("Arabic") if x == "ar" else _("English"),
                                            index=0 if st.session_state["lang"] == "ar" else 1)
        st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
        # Data Upload
        st.markdown(f"<b>{_('Upload Your Own CSV')}</b>", unsafe_allow_html=True)
        upload = st.file_uploader(_("Upload Your Own CSV"), type=['csv'])
        if upload:
            st.session_state["uploaded_df"] = pd.read_csv(upload)
            st.success(_("Uploaded! Using your CSV."))
        st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
        # Main nav
        pages = [
            ("dashboard", _("Dashboard")), ("predictive", _("Predictive Analysis")),
            ("solutions", _("Smart Solutions")), ("alerts", _("Smart Alerts")),
            ("cost", _("Cost & Savings")), ("achievements", _("Achievements")),
            ("performance", _("Performance")), ("comparison", _("Comparison")),
            ("explorer", _("Data Explorer")), ("about", _("About"))
        ]
        st.session_state["page_radio"] = st.radio(_("Navigate to"), options=pages, format_func=lambda x: x[1], index=0)
sidebar()

# =========================
# 7. Data
# =========================
def get_data():
    if st.session_state["uploaded_df"] is not None:
        return st.session_state["uploaded_df"]
    # fallback: generate demo data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=40)
    return pd.DataFrame({
        _("Temperature"): 80 + 5 * np.random.rand(40),
        _("Pressure"): 200 + 10 * np.random.rand(40),
        _("Methane"): 2.5 + 0.5 * np.random.rand(40),
        _("Vibration"): 0.6 + 0.1 * np.random.rand(40),
        _("H2S"): 0.3 + 0.05 * np.random.rand(40)
    }, index=dates)

# =========================
# 8. Advanced Model Selector (Demo)
# =========================
def advanced_model_selector():
    return st.selectbox(
        _("Advanced Model"),
        [_("Linear Regression"), _("ARIMA (Demo)"), _("Prophet (Demo)")],
        index=0
    )

# =========================
# 9. Live Mode Demo
# =========================
def live_mode_block():
    st.session_state["live_mode"] = st.checkbox(_("Live Mode"), value=st.session_state.get("live_mode", False))
    if st.session_state["live_mode"]:
        st.success(_("Switch to Live Mode"))
    else:
        st.info(_("Switch to History Mode"))

# =========================
# 10. Multi-Step What-If Demo
# =========================
def multi_step_whatif_block():
    st.markdown(f"<b>{_('Multi-Step What-If')}</b>", unsafe_allow_html=True)
    if st.button(_("Add What-If Step")):
        st.session_state["multi_whatif"].append({"step": len(st.session_state["multi_whatif"])+1, "delta": 0.0})
    for idx, step in enumerate(st.session_state["multi_whatif"]):
        col1, col2 = st.columns([2,1])
        with col1:
            st.session_state["multi_whatif"][idx]["delta"] = st.slider(
                f"{_('Step')} {idx+1} Δ", -10.0, 10.0, float(step["delta"]), 0.1, key=f"whatif_delta_{idx}")
        with col2:
            if st.button(_("Remove Step"), key=f"remove_{idx}"):
                st.session_state["multi_whatif"].pop(idx)
                break
    if not st.session_state["multi_whatif"]:
        st.info(_("No What-If steps set"))
    if st.button(_("Apply Multi-Step")):
        st.success("Multi-step what-if applied (Demo logic)")

# =========================
# 11. AI Icon Block (shown at top of every main page)
# =========================
def ai_icon():
    st.markdown(
        '<img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="48" style="margin-bottom:-10px;margin-right:12px;"/>',
        unsafe_allow_html=True
    )

# =========================
# 12. Demo Email/SMS Alert Button
# =========================
def alert_demo_button():
    if st.button(_("Send Alerts (Demo)")):
        st.success(_("Alerts have been sent (Demo)!"))

# =========================
# 13. Export Buttons (Excel/PDF demo)
# =========================
def export_buttons(df):
    st.download_button(_("Export to Excel"), data=df.to_csv(index=False).encode('utf-8'), file_name="export.xlsx")
    if st.button(_("Export to PDF (Demo)")):
        st.success(_("PDF generated (Demo)!"))

# =========================
# 14. API Integration Demo
# =========================
def api_demo_block():
    if st.button(_("Show API Demo")):
        st.json({
            "timestamp": str(pd.Timestamp.now()),
            "Temperature": 85.1,
            "Pressure": 208.0,
            "Methane": 3.2,
            "status": "normal"
        }, expanded=True)
        st.info(_("API Data (Demo)"))

# =========================
# 15. Main Pages
# =========================
def dashboard():
    ai_icon()
    live_mode_block()
    st.markdown(f"<div class='big-title'>🧠 {_('Welcome to your Smart Digital Twin!')}</div>", unsafe_allow_html=True)
    df = get_data()
    vals = [df[_("Temperature")].iloc[-1], df[_("Pressure")].iloc[-1], df[_("Vibration")].iloc[-1], df[_("Methane")].iloc[-1], df[_("H2S")].iloc[-1]]
    labels = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    units = ["°C", "psi", "g", "ppm", "ppm"]
    icons = ["🌡️", "💧", "🌀", "🟢", "⚗️"]
    cols = st.columns(len(vals))
    for i, col in enumerate(cols):
        col.metric(label=f"{icons[i]} {labels[i]}", value=f"{vals[i]:.2f} {units[i]}")
    st.line_chart(df)
    export_buttons(df)
    alert_demo_button()

def predictive():
    ai_icon()
    st.markdown(f"<div class='big-title'>🔮 {_('Predictive Analysis')}</div>", unsafe_allow_html=True)
    model = advanced_model_selector()
    df = get_data()
    st.line_chart(df)
    if model != _("Linear Regression"):
        st.warning(f"{model} - Demo forecast shown only")
    export_buttons(df)
    multi_step_whatif_block()
    alert_demo_button()

def solutions():
    ai_icon()
    st.markdown(f"<div class='big-title'>💡 {_('Smart Solutions')}</div>", unsafe_allow_html=True)
    st.info(_("AI-powered recommendations for safety and efficiency"))
    alert_demo_button()

def alerts():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Smart Alerts')}</div>", unsafe_allow_html=True)
    alert_demo_button()

def cost():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Cost & Savings')}</div>", unsafe_allow_html=True)
    df = get_data()
    st.bar_chart(df)
    export_buttons(df)

def achievements():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Achievements')}</div>", unsafe_allow_html=True)
    st.success(_("Congratulations!") + " 🎉")
    export_buttons(get_data())

def performance():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Performance')}</div>", unsafe_allow_html=True)
    st.line_chart(get_data())
    export_buttons(get_data())

def comparison():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Comparison')}</div>", unsafe_allow_html=True)
    st.line_chart(get_data())
    export_buttons(get_data())

def explorer():
    ai_icon()
    st.markdown(f"<div class='big-title'>{_('Data Explorer')}</div>", unsafe_allow_html=True)
    st.dataframe(get_data())
    export_buttons(get_data())

def about():
    ai_icon()
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content:center;">
        <img src="https://img.icons8.com/color/96/artificial-intelligence.png" width="72" style="margin-right:20px;" alt="AI logo"/>
        <div>
            <span class="big-title">{}</span><br>
            <span class="sub-title" style="font-size:1.18rem;">Smart Neural Digital Twin</span>
        </div>
    </div>
    """.format(_("About the Project")), unsafe_allow_html=True)
    st.markdown(
        f"""<div class='about-card-gradient'><span style='font-size:1.2em;'>🧠</span> <b>{_('About Project Description')}</b></div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class='card' style='font-style:italic;font-size:1.2rem;'><span class='badge'>{_('Our Vision')}</span>
        “{_('Disasters don\'t wait.. and neither do we.')}”</div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class='about-card-gradient'>
        <span class='badge'>✨ {_('Features')}</span>
        <div class='about-features'>
            <div><span class='fancy-icon'>🤖</span>{_('AI-powered predictive analytics')}</div>
            <div><span class='fancy-icon'>⚡</span>{_('Instant smart solutions')}</div>
            <div><span class='fancy-icon'>📡</span>{_('Live alerts and monitoring')}</div>
            <div><span class='fancy-icon'>🌐</span>{_('Multi-language support')}</div>
            <div><span class='fancy-icon'>🎨</span>{_('Stunning, responsive UI')}</div>
        </div>
        </div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class='about-card-gradient'>
        <span class="badge">🏆 {_('Milestones')}</span>
        <ul class='about-milestones'>
            <li>2024: MVP Launch 🚀</li>
            <li>2025: {_('2025 Innovation Award, Best Digital Twin')} 🥇</li>
            <li>100+ {_('days without incidents')} ⭐</li>
        </ul>
        </div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class='about-card-gradient'>
        <span class="badge">👨‍💻 {_('Main Developers')}</span>
        <div class='about-dev'>
            <div class='dev'>
                <img src="https://ui-avatars.com/api/?name=Rakan+Almarri&background=43e97b&color=fff" width="60"/><br>
                <b>Rakan Almarri</b><br>
                <span style="font-size:0.97em;">rakan.almarri.2@aramco.com</span>
            </div>
            <div class='dev'>
                <img src="https://ui-avatars.com/api/?name=Abdulrahman+Alzhrani&background=278ea5&color=fff" width="60"/><br>
                <b>Abdulrahman Alzhrani</b><br>
                <span style="font-size:0.97em;">abdulrahman.alzhrani.1@aramco.com</span>
            </div>
        </div>
        </div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class='about-contact'>📬 {_('Contact us for partnership or demo!')}<br>
            <a href="mailto:rakan.almarri.2@aramco.com" style="color:{theme['badge_bg']}; text-decoration:underline;">
                rakan.almarri.2@aramco.com
            </a>
        </div>""",
        unsafe_allow_html=True)
    st.markdown("---")
    api_demo_block()
    st.button(_("Save My Settings (Demo)"), on_click=lambda: st.success(_("Settings Saved! (Demo)")))

# =========================
# 16. Routing
# =========================
pages = {
    "dashboard": dashboard,
    "predictive": predictive,
    "solutions": solutions,
    "alerts": alerts,
    "cost": cost,
    "achievements": achievements,
    "performance": performance,
    "comparison": comparison,
    "explorer": explorer,
    "about": about
}
selected_page = st.session_state["page_radio"][0]
pages[selected_page]()
