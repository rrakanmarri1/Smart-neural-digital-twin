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
theme = THEME_SETS.get(st.session_state["theme_set"], THEME_SETS[DEFAULT_THEME])

# --- TRANSLATIONS ---
translations = {
    "en": {
        "Settings": "Settings",
        "Choose Language": "Choose Language",
        "Dashboard": "Dashboard",
        "Predictive Analysis": "Predictive Analysis",
        "Smart Solutions": "Smart Solutions",
        "Smart Alerts": "Smart Alerts",
        "Cost & Savings": "Cost & Savings",
        "Achievements": "Achievements",
        "Performance Comparison": "Performance Comparison",
        "Data Explorer": "Data Explorer",
        "About": "About",
        "Navigate to": "Navigate to",
        "Welcome to your Smart Digital Twin!": "Welcome to your Smart Digital Twin!",
        "Simulate Disaster": "Simulate Disaster",
        "Temperature": "Temperature",
        "Pressure": "Pressure",
        "Vibration": "Vibration",
        "Methane": "Methane",
        "H2S": "H2S",
        "Live Data": "Live Data",
        "Time": "Time",
        "Trend": "Trend",
        "Theme Set": "Theme Set",
        "ar": "Arabic",
        "en": "English"
    },
    # Example for Arabic (add more keys as needed)
    "ar": {
        "Settings": "الإعدادات",
        "Choose Language": "اختر اللغة",
        "Dashboard": "لوحة القيادة",
        "Predictive Analysis": "التحليل التنبؤي",
        "Smart Solutions": "الحلول الذكية",
        "Smart Alerts": "التنبيهات الذكية",
        "Cost & Savings": "التكلفة والمدخرات",
        "Achievements": "الإنجازات",
        "Performance Comparison": "مقارنة الأداء",
        "Data Explorer": "مستكشف البيانات",
        "About": "حول",
        "Navigate to": "انتقل إلى",
        "Welcome to your Smart Digital Twin!": "مرحبًا بك في التوأم الرقمي الذكي!",
        "Simulate Disaster": "محاكاة كارثة",
        "Temperature": "درجة الحرارة",
        "Pressure": "الضغط",
        "Vibration": "الاهتزاز",
        "Methane": "الميثان",
        "H2S": "كبريتيد الهيدروجين",
        "Live Data": "بيانات حية",
        "Time": "الوقت",
        "Trend": "الاتجاه",
        "Theme Set": "سمة العرض",
        "ar": "العربية",
        "en": "الإنجليزية"
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
    translation_dict = translations.get(lang, translations.get("en", {}))
    return translation_dict.get(key, key)

# --- CSS for THEME + Responsive ---
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

    /* Responsive adjustments */
    @media (max-width: 900px) {{
      .big-title {{ font-size: 1.7rem; }}
      .sub-title {{ font-size: 1.1rem; }}
      .metric {{ font-size: 1.3rem; }}
      .card {{ padding: 12px 10px; }}
    }}
    @media (max-width: 600px) {{
      .big-title {{ font-size: 1.2rem; }}
      .sub-title {{ font-size: 1rem; }}
      .card {{ padding: 7px 5px; }}
      .metric-label {{ font-size: 0.9rem; }}
      .metric {{ font-size: 1.05rem; }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    with st.expander(_("Settings"), expanded=True):
        lang_choice = st.radio(
            _("Choose Language"),
            options=["ar", "en"],
            format_func=lambda x: _(x),
            index=0 if get_lang() == "ar" else 1,
            key="lang_radio"
        )
        set_lang(lang_choice)
        theme_set = st.selectbox(_("Theme Set"), options=list(THEME_SETS.keys()), index=list(THEME_SETS.keys()).index(st.session_state["theme_set"]))
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
    cols = st.columns(5)
    colA, colB = st.columns([4,1])
    with colB:
        if st.button("🚨 " + _("Simulate Disaster")):
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
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Live Data")}</div>'), unsafe_allow_html=True)
    # Dummy trend chart for demo
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
        font=dict(color=theme['text_on_primary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- ROUTING ---
def show_predictive(): pass
def show_solutions(): pass
def show_alerts(): pass
def show_cost(): pass
def show_achievements(): pass
def show_comparison(): pass
def show_explorer(): pass
def show_about(): pass

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
selected_page = st.session_state.page_radio  # This is a tuple like ('dashboard', 'Dashboard')
routes[selected_page[0]]()
