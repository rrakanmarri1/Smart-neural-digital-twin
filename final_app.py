"""
Smart Neural Digital Twin
Fully Responsive, Visually Enhanced, and Consistent Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from typing import Dict, Callable

# =========================
# 1. Theme System & Preview (IMPROVED COLORS)
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
# 2. Translations & i18n (FULL)
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
        "About Project Description": "Smart Neural Digital Twin is an AI-powered disaster prevention platform for industrial sites and oilfields. It connects live sensors to an intelligent digital twin that predicts disasters, provides smart recommendations, and reduces maintenance costs.",
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
        # Themes
        "Ocean": "Ocean",
        "Sunset": "Sunset",
        "Emerald": "Emerald",
        "Night": "Night",
        "Blossom": "Blossom",
        "AI-powered recommendations for safety and efficiency": "AI-powered recommendations for safety and efficiency",
        "Methane Spike": "Methane Spike",
        "Pressure Drop": "Pressure Drop",
        "Vibration Anomaly": "Vibration Anomaly",
        "High Temperature": "High Temperature"
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
        "Emergency Vent Gas!": "تنفيس الغاز فوراً!", "Immediate venting required in Tank 2 due to critical methane spike.": "مطلوب تنفيس فوري في الخزان 2 بسبب ارتفاع حرج في الميثان.",
        "Critical disaster detected during simulation.": "تم رصد كارثة حرجة أثناء المحاكاة.",
        "Reduce Pressure in Line 3": "قلل الضغط في الخط ٣", "Reduce the pressure by 15% in Line 3 and alert the maintenance crew for inspection.": "قم بخفض الضغط بنسبة 15٪ في الخط ٣ وأبلغ طاقم الصيانة للفحص.",
        "Abnormal vibration detected. This reduces risk.": "تم رصد اهتزاز غير طبيعي. هذا يقلل المخاطر.",
        "URGENT": "عاجل", "Now": "الآن", "High": "مرتفعة", "15 minutes": "١٥ دقيقة", "95%": "٩٥٪", "99%": "٩٩٪",
        "About Project Description": "التوأم الرقمي العصبي الذكي هو منصة مدعومة بالذكاء الاصطناعي للوقاية من الكوارث في المواقع الصناعية وحقول النفط. يربط الحساسات الحية بتوأم رقمي ذكي يتنبأ بالكوارث، ويقدم توصيات ذكية، ويقلل من تكاليف الصيانة.",
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
        "High Temperature": "درجة حرارة عالية"
    }
}

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

# =========================
# 3. Theme and CSS Injection (IMPROVED)
# =========================

def set_theme_in_session():
    if "theme_set" not in st.session_state:
        st.session_state["theme_set"] = DEFAULT_THEME

set_theme_in_session()
theme = THEME_SETS[st.session_state["theme_set"]]

def inject_css():
    st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {theme['primary']} !important; }}
    .stSidebar {{ background-color: {theme['sidebar_bg']} !important; }}
    .big-title {{ color: {theme['secondary']}; font-size:2.5rem; font-weight:bold; margin-bottom:10px; letter-spacing:0.03em; }}
    .sub-title {{ color: {theme['accent']}; font-size:1.3rem; margin-bottom:12px; font-weight:500; }}
    .card {{
        background: {theme['card_bg']};
        border-radius:18px;
        padding:22px 20px 20px 20px;
        margin-bottom:20px;
        color:{theme['text_on_secondary']};
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        min-width:100px;
    }}
    .kpi-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .kpi-card {{
        flex: 1 1 160px;
        min-width: 140px;
        max-width: 180px;
        background: {theme['card_bg']};
        border-radius: 13px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.09);
        padding: 18px 8px 13px 8px;
        text-align: center;
        margin: 0;
        color: {theme['text_on_secondary']};
        margin-bottom: 0px;
        transition: transform 0.08s;
        position: relative;
    }}
    .kpi-card:hover {{ transform: translateY(-5px) scale(1.03); }}
    .metric {{ font-size:2.1rem; font-weight:700; line-height:1.2; }}
    .metric-label {{ font-size:1.09rem; color:{theme['accent']}; margin-top:4px; }}
    .badge {{ background:{theme['badge_bg']}; color:{theme['text_on_accent']}; padding:3px 14px; border-radius:20px; margin-right:8px; font-weight:500; font-size:1.03rem; }}
    .rtl {{ direction:rtl; }}
    .theme-swatch {{
        display:inline-block; width:22px; height:22px; border-radius:8px;
        margin-right:8px; border:2px solid #3333; vertical-align:middle;
    }}
    .status-badge {{
        display: inline-block;
        background: {theme['alert']};
        color: {theme['alert_text']};
        border-radius: 15px;
        padding: 3px 13px;
        font-weight: bold;
        font-size: 1.06rem;
        margin-bottom: 4px;
    }}
    .sidebar-section {{
        margin-bottom: 18px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
    }}
    .block-section {{
        margin-bottom: 32px;
    }}
    @media (max-width: 950px) {{
        .kpi-container {{ gap: 0.6rem; }}
        .kpi-card {{ min-width: 120px; max-width: 150px; padding: 10px 4px 8px 4px; font-size: 0.96rem; }}
    }}
    @media (max-width: 700px) {{
        .kpi-container {{ gap: 0.3rem; }}
        .kpi-card {{ min-width: 90px; max-width: 120px; padding: 6px 2px 6px 2px; font-size: 0.90rem; }}
    }}
    .data-table thead tr th, .data-table tbody tr td {{
        text-align: center !important;
        font-size: 1.09rem;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

