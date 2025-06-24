import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

# Import your real prediction engine
import prediction_engine
import joblib

# --- TRANSLATIONS ---
translations = {
    "en": {
        "Settings": "Settings",
        "Choose Language": "Choose Language",
        "Arabic": "Arabic",
        "English": "English",
        "Navigate to": "Navigate to",
        "Dashboard": "Dashboard",
        "Predictive Analysis": "Predictive Analysis",
        "Smart Solutions": "Smart Solutions",
        "Smart Alerts": "Smart Alerts",
        "Cost & Savings": "Cost & Savings",
        "Achievements": "Achievements",
        "Performance Comparison": "Performance Comparison",
        "Data Explorer": "Data Explorer",
        "About": "About",
        "Welcome to your Smart Digital Twin!": "Welcome to your Smart Digital Twin!",
        "System Status": "System Status",
        "Temperature": "Temperature",
        "Pressure": "Pressure",
        "Vibration": "Vibration",
        "Methane": "Methane",
        "H2S": "H2S",
        "Live Data": "Live Data",
        "View Details": "View Details",
        "Trend": "Trend",
        "Risk Level": "Risk Level",
        "Forecast": "Forecast",
        "Savings": "Savings",
        "Monthly Savings": "Monthly Savings",
        "Yearly Savings": "Yearly Savings",
        "Milestone": "Milestone",
        "Congratulations!": "Congratulations!",
        "You have achieved": "You have achieved",
        "Compared to last period": "Compared to last period",
        "Data Filters": "Data Filters",
        "Select Metric": "Select Metric",
        "About the Project": "About the Project",
        "Features": "Features",
        "Contact": "Contact",
        "No data available.": "No data available.",
        "Download Report": "Download Report",
        "Generate Solution": "Generate Solution",
        "Generating solution...": "Generating solution...",
        "Press 'Generate Solution' for intelligent suggestions.": "Press 'Generate Solution' for intelligent suggestions.",
        "Best Solution": "Best Solution",
        "Reason": "Reason",
        "Apply": "Apply",
        "Export": "Export",
        "Feedback": "Feedback",
        "Contact Us": "Contact Us",
        "Project Features": "Project Features",
        "Alerts": "Alerts",
        "Current Alerts": "Current Alerts",
        "No alerts at the moment.": "No alerts at the moment.",
        "Smart Recommendations": "Smart Recommendations"
    },
    "ar": {
        "Settings": "الإعدادات",
        "Choose Language": "اختر اللغة",
        "Arabic": "العربية",
        "English": "الإنجليزية",
        "Navigate to": "انتقل إلى",
        "Dashboard": "لوحة البيانات",
        "Predictive Analysis": "التحليل التنبؤي",
        "Smart Solutions": "الحلول الذكية",
        "Smart Alerts": "تنبيهات ذكية",
        "Cost & Savings": "التكلفة والتوفير",
        "Achievements": "لوحة الإنجازات",
        "Performance Comparison": "مقارنة الأداء",
        "Data Explorer": "استكشاف البيانات",
        "About": "حول",
        "Welcome to your Smart Digital Twin!": "مرحبًا بك في التوأم الرقمي الذكي!",
        "System Status": "حالة النظام",
        "Temperature": "درجة الحرارة",
        "Pressure": "الضغط",
        "Vibration": "الاهتزاز",
        "Methane": "الميثان",
        "H2S": "كبريتيد الهيدروجين",
        "Live Data": "بيانات حية",
        "View Details": "عرض التفاصيل",
        "Trend": "الاتجاه",
        "Risk Level": "مستوى الخطورة",
        "Forecast": "توقعات",
        "Savings": "التوفير",
        "Monthly Savings": "التوفير الشهري",
        "Yearly Savings": "التوفير السنوي",
        "Milestone": "الإنجاز",
        "Congratulations!": "مبروك!",
        "You have achieved": "لقد حققت",
        "Compared to last period": "مقارنة بالفترة السابقة",
        "Data Filters": "مرشحات البيانات",
        "Select Metric": "اختر المقياس",
        "About the Project": "حول المشروع",
        "Features": "المميزات",
        "Contact": "تواصل",
        "No data available.": "لا توجد بيانات متاحة.",
        "Download Report": "تنزيل التقرير",
        "Generate Solution": "توليد الحل",
        "Generating solution...": "جاري توليد الحل...",
        "Press 'Generate Solution' for intelligent suggestions.": "اضغط على 'توليد الحل' للحصول على اقتراحات ذكية.",
        "Best Solution": "أفضل حل",
        "Reason": "السبب",
        "Apply": "تطبيق",
        "Export": "تصدير",
        "Feedback": "ملاحظات",
        "Contact Us": "تواصل معنا",
        "Project Features": "مميزات المشروع",
        "Alerts": "تنبيهات",
        "Current Alerts": "التنبيهات الحالية",
        "No alerts at the moment.": "لا توجد تنبيهات حالياً.",
        "Smart Recommendations": "التوصيات الذكية"
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
    return translations[lang].get(key, key)

st.set_page_config(page_title="Smart Digital Twin", layout="wide", page_icon="🌐")

st.markdown("""
    <style>
    body, .stApp { background-color: #153243 !important; }
    .big-title { color: #21e6c1; font-size:2.3rem; font-weight:bold; margin-bottom:10px;}
    .sub-title { color: #21e6c1; font-size:1.4rem; margin-bottom:10px;}
    .card { background: #278ea5; border-radius: 16px; padding: 18px 24px; margin-bottom:16px; color: #fff; }
    .metric {font-size:2.1rem; font-weight:bold;}
    .metric-label {font-size:1.1rem; color:#21e6c1;}
    .alert {background:#ff3e3e; color:#fff; border-radius:12px; padding:12px;}
    .badge { background: #21e6c1; color:#153243; padding: 2px 12px; border-radius: 20px; margin-right: 10px;}
    .rtl { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title(_("Settings"))
    lang_choice = st.radio(
        _("Choose Language"),
        options=["ar", "en"],
        format_func=lambda x: _("Arabic") if x == "ar" else _("English"),
        index=0 if get_lang() == "ar" else 1
    )
    set_lang(lang_choice)
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
    page = st.selectbox(_("Navigate to"), options=pages, format_func=lambda x: x[1])

def rtl_wrap(html):
    return f'<div class="rtl">{html}</div>' if get_lang() == "ar" else html

# --- Load prediction models on startup (cache for performance) ---
@st.cache_resource
def load_models():
    model_path = "prediction_models.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

prediction_models = load_models()

def show_dashboard():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Welcome to your Smart Digital Twin!")}</div>'), unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(rtl_wrap(f'<div class="card"><div class="metric">82.7°C</div><div class="metric-label">{_("Temperature")}</div></div>'), unsafe_allow_html=True)
    col2.markdown(rtl_wrap(f'<div class="card"><div class="metric">202.2 psi</div><div class="metric-label">{_("Pressure")}</div></div>'), unsafe_allow_html=True)
    col3.markdown(rtl_wrap(f'<div class="card"><div class="metric">0.61 g</div><div class="metric-label">{_("Vibration")}</div></div>'), unsafe_allow_html=True)
    col4.markdown(rtl_wrap(f'<div class="card"><div class="metric">2.85 ppm</div><div class="metric-label">{_("Methane")}</div></div>'), unsafe_allow_html=True)
    col5.markdown(rtl_wrap(f'<div class="card"><div class="metric">0.30 ppm</div><div class="metric-label">{_("H2S")}</div></div>'), unsafe_allow_html=True)
    st.markdown("")

    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Live Data")}</div>'), unsafe_allow_html=True)
    df = pd.DataFrame({
        _("Temperature"): 82 + 2 * pd.np.sin(pd.np.linspace(0, 3.14, 40)),
        _("Pressure"): 200 + 4 * pd.np.cos(pd.np.linspace(0, 3.14, 40)),
        _("Vibration"): 0.6 + 0.05 * pd.np.sin(pd.np.linspace(0, 6.28, 40)),
        _("Methane"): 2.8 + 0.1 * pd.np.random.rand(40),
        _("H2S"): 0.3 + 0.05 * pd.np.random.rand(40),
    })
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(y=df[col], mode='lines', name=col))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=_("Trend"),
        plot_bgcolor="#153243",
        paper_bgcolor="#153243",
        font=dict(color="#21e6c1"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_predictive():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Predictive Analysis")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Forecast")}</div>'), unsafe_allow_html=True)

    if not prediction_models:
        st.error("Prediction model not found! Please train your model and place prediction_models.pkl in the app directory.")
        return

    try:
        predictions = prediction_engine.predict_future_values(prediction_models, hours_ahead=6)
    except Exception as e:
        st.error(f"Prediction engine error: {e}")
        return

    # Map repo sensor names to display names
    sensor_map = {
        'Temperature (°C)': _("Temperature"),
        'Pressure (psi)': _("Pressure"),
        'Vibration (g)': _("Vibration"),
        'Methane (CH₄ ppm)': _("Methane"),
        'H₂S (ppm)': _("H2S")
    }

    display_selected = [_("Temperature"), _("Pressure"), _("Methane")]

    # For each sensor, show predictions
    for repo_sensor, display_sensor in sensor_map.items():
        if display_sensor not in display_selected:
            continue
        future_list = predictions.get(repo_sensor, [])
        if not future_list:
            continue
        risk = "Low"
        if display_sensor == _("Methane"):
            risk = "High"
        elif display_sensor == _("Pressure"):
            risk = "Medium"
        risk_badge = f'<span class="badge">{_("Risk Level")}: {risk}</span>'
        # Show only the last prediction for card
        last_pred = future_list[-1]
        st.markdown(rtl_wrap(f'<div class="card">{risk_badge}<br><b>{display_sensor}:</b> {last_pred["value"]:.2f} {repo_sensor.split()[-1]}</div>'), unsafe_allow_html=True)

    # Plot all predictions
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Trend")}</div>'), unsafe_allow_html=True)
    fig = go.Figure()
    for repo_sensor, display_sensor in sensor_map.items():
        if display_sensor not in display_selected:
            continue
        y = [x["value"] for x in predictions.get(repo_sensor, [])]
        x = [x["hours_ahead"] for x in predictions.get(repo_sensor, [])]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=display_sensor))
    fig.update_layout(
        xaxis_title="Hours Ahead",
        yaxis_title=_("Forecast"),
        plot_bgcolor="#153243",
        paper_bgcolor="#153243",
        font=dict(color="#21e6c1"),
    )
    st.plotly_chart(fig, use_container_width=True)

def show_solutions():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Smart Solutions")}</div>'), unsafe_allow_html=True)
    if st.button(_("Generate Solution")):
        with st.spinner(_("Generating solution...")):
            solutions = [
                {"title": _("Reduce Pressure in Line 3"), "reason": _("Abnormal vibration detected. This reduces risk.")},
                {"title": _("Schedule Pump Maintenance"), "reason": _("Temperature rising above normal.")},
            ]
        for idx, sol in enumerate(solutions):
            badge = f'<span class="badge">{_("Best Solution") if idx==0 else _("Smart Recommendations")}</span>'
            st.markdown(rtl_wrap(f'<div class="card">{badge}<br><b>{sol["title"]}</b><br>{_("Reason")}: {sol["reason"]}<br>'
                        f'<button style="margin-top:8px;background:#21e6c1;color:#153243;border:none;border-radius:8px;padding:5px 12px;">{_("Apply")}</button> '
                        f'<button style="margin-top:8px;background:#278ea5;color:#fff;border:none;border-radius:8px;padding:5px 12px;">{_("Export")}</button> '
                        f'<button style="margin-top:8px;background:transparent;color:#21e6c1;border:1px solid #21e6c1;border-radius:8px;padding:5px 12px;">{_("Feedback")}</button>'
                        f'</div>'), unsafe_allow_html=True)
    else:
        st.info(_("Press 'Generate Solution' for intelligent suggestions."))

def show_alerts():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Smart Alerts")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Current Alerts")}</div>'), unsafe_allow_html=True)
    alerts = [
        {"msg": _("High Pressure Detected in Zone 2!"), "severity": "high"},
        {"msg": _("Methane levels rising in Tank 1."), "severity": "medium"},
    ]
    if alerts:
        for a in alerts:
            col = "#ff3e3e" if a["severity"]=="high" else "#ffc107"
            st.markdown(rtl_wrap(f'<div class="alert" style="background:{col}">{a["msg"]}</div>'), unsafe_allow_html=True)
    else:
        st.info(_("No alerts at the moment."))

def show_cost():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Cost & Savings")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><div class="metric">5,215,000 SAR</div><div class="metric-label">{_("Yearly Savings")}</div></div>'), unsafe_allow_html=True)
    months = [f"{i+1}/2025" for i in range(6)]
    savings = [400000, 450000, 500000, 550000, 600000, 650000]
    fig = go.Figure(go.Bar(x=months, y=savings, marker_color="#21e6c1"))
    fig.update_layout(
        xaxis_title=_("Monthly Savings"),
        yaxis_title=_("Savings"),
        plot_bgcolor="#153243",
        paper_bgcolor="#153243",
        font=dict(color="#21e6c1"),
    )
    st.plotly_chart(fig, use_container_width=True)

def show_achievements():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Achievements")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><span class="badge">{_("Milestone")}</span><br>{_("Congratulations!")}<br>{_("You have achieved")} <b>100</b> {"days without incidents"}!</div>'), unsafe_allow_html=True)
    st.progress(0.85, text=_("Compared to last period"))

def show_comparison():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Performance Comparison")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Savings")]
    values_now = [82.7, 202.2, 650000]
    values_prev = [85, 204, 500000]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=values_now, name=_("Current")))
    fig.add_trace(go.Bar(x=metrics, y=values_prev, name=_("Previous")))
    fig.update_layout(barmode='group', plot_bgcolor="#153243", paper_bgcolor="#153243", font=dict(color="#21e6c1"))
    st.plotly_chart(fig, use_container_width=True)

def show_explorer():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Data Explorer")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Data Filters")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    metric = st.selectbox(_("Select Metric"), options=metrics)
    data = pd.DataFrame({metric: 80 + 5 * pd.np.random.rand(30)})
    st.line_chart(data)

def show_about():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("About the Project")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><b>{_('Project Features')}</b><ul><li>{_('AI-powered predictive analytics')}</li><li>{_('Instant smart solutions')}</li><li>{_('Live alerts and monitoring')}</li><li>{_('Multi-language support')}</li><li>{_('Stunning, responsive UI')}</li></ul></div>"), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<div class='card'><b>{_('Contact Us')}</b><br>rrakanmarri1@gmail.com</div>"), unsafe_allow_html=True)

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
routes[page[0]]()
