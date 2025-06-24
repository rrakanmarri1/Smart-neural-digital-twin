import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import prediction_engine
import joblib

# --- COLOR THEMES ---
THEME_SETS = {
    "Ocean": {
        "primary": "#153243",
        "secondary": "#278ea5",
        "accent": "#21e6c1",
        "text_on_primary": "#fff",
        "text_on_secondary": "#fff",
        "text_on_accent": "#153243",
        "sidebar_bg": "#18465b",
        "card_bg": "#278ea5",
        "badge_bg": "#21e6c1",
        "alert": "#ff3e3e",
        "alert_text": "#fff",
        "plot_bg": "#153243"
    },
    "Sunset": {
        "primary": "#ff7043",
        "secondary": "#ffa726",
        "accent": "#ffd54f",
        "text_on_primary": "#fff",
        "text_on_secondary": "#232526",
        "text_on_accent": "#232526",
        "sidebar_bg": "#ffb28f",
        "card_bg": "#ffa726",
        "badge_bg": "#ff7043",
        "alert": "#d7263d",
        "alert_text": "#fff",
        "plot_bg": "#fff3e0"
    },
    "Emerald": {
        "primary": "#154734",
        "secondary": "#43e97b",
        "accent": "#38f9d7",
        "text_on_primary": "#fff",
        "text_on_secondary": "#153243",
        "text_on_accent": "#154734",
        "sidebar_bg": "#1d5c41",
        "card_bg": "#43e97b",
        "badge_bg": "#38f9d7",
        "alert": "#ff1744",
        "alert_text": "#fff",
        "plot_bg": "#e0f2f1"
    },
    "Night": {
        "primary": "#232526",
        "secondary": "#414345",
        "accent": "#e96443",
        "text_on_primary": "#fff",
        "text_on_secondary": "#fff",
        "text_on_accent": "#232526",
        "sidebar_bg": "#414345",
        "card_bg": "#232526",
        "badge_bg": "#e96443",
        "alert": "#ff3e3e",
        "alert_text": "#fff",
        "plot_bg": "#232526"
    },
    "Blossom": {
        "primary": "#fbd3e9",
        "secondary": "#bb377d",
        "accent": "#fa709a",
        "text_on_primary": "#232526",
        "text_on_secondary": "#fff",
        "text_on_accent": "#fff",
        "sidebar_bg": "#fcb7d4",
        "card_bg": "#fa709a",
        "badge_bg": "#bb377d",
        "alert": "#d7263d",
        "alert_text": "#fff",
        "plot_bg": "#fce4ec"
    }
}

DEFAULT_THEME = "Ocean"
if "theme_set" not in st.session_state:
    st.session_state["theme_set"] = DEFAULT_THEME

theme = THEME_SETS[st.session_state["theme_set"]]

# --- TRANSLATIONS & LANGUAGE LOGIC ---
# Use your previous translations and language logic here
translations = {
    # ... (same as before)
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
            st.experimental_rerun()  # Instantly update theme on select change!
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
        _("Temperature"): 82 + 2 * np.sin(np.linspace(0, 3.14, 40)),
        _("Pressure"): 200 + 4 * np.cos(np.linspace(0, 3.14, 40)),
        _("Vibration"): 0.6 + 0.05 * np.sin(np.linspace(0, 6.28, 40)),
        _("Methane"): 2.8 + 0.1 * np.random.rand(40),
        _("H2S"): 0.3 + 0.05 * np.random.rand(40),
    })
    fig = go.Figure()
    color_cycle = [theme['secondary'], theme['accent'], theme['badge_bg'], "#fa709a", "#ff7043"]
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(y=df[col], mode='lines', name=col, line=dict(color=color_cycle[i%len(color_cycle)], width=3)))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=_("Trend"),
        plot_bgcolor=theme['plot_bg'],
        paper_bgcolor=theme['plot_bg'],
        font=dict(color=theme['text_on_primary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_predictive():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Predictive Analysis")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Forecast")}</div>'), unsafe_allow_html=True)
    if not prediction_models:
        st.markdown(f'<div class="alert-custom">Prediction model not found! Please train your model and place prediction_models.pkl in the app directory.</div>', unsafe_allow_html=True)
        return
    try:
        predictions = prediction_engine.predict_future_values(prediction_models, hours_ahead=6)
    except Exception as e:
        st.markdown(f'<div class="alert-custom">Prediction engine error: {e}</div>', unsafe_allow_html=True)
        return
    sensor_map = {
        'Temperature (°C)': _("Temperature"),
        'Pressure (psi)': _("Pressure"),
        'Vibration (g)': _("Vibration"),
        'Methane (CH₄ ppm)': _("Methane"),
        'H₂S (ppm)': _("H2S")
    }
    display_selected = [_("Temperature"), _("Pressure"), _("Methane")]
    color_cycle = [theme['secondary'], theme['accent'], theme['badge_bg']]
    for i, (repo_sensor, display_sensor) in enumerate(sensor_map.items()):
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
        last_pred = future_list[-1]
        st.markdown(rtl_wrap(
            f'<div class="card" style="border-left: 8px solid {color_cycle[i%len(color_cycle)]};">'
            f'{risk_badge}<br><b>{display_sensor}:</b> {last_pred["value"]:.2f} {repo_sensor.split()[-1]}</div>'
        ), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Trend")}</div>'), unsafe_allow_html=True)
    fig = go.Figure()
    for i, (repo_sensor, display_sensor) in enumerate(sensor_map.items()):
        if display_sensor not in display_selected:
            continue
        y = [x["value"] for x in predictions.get(repo_sensor, [])]
        x = [x["hours_ahead"] for x in predictions.get(repo_sensor, [])]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=display_sensor, line=dict(color=color_cycle[i%len(color_cycle)], width=3)))
    fig.update_layout(
        xaxis_title="Hours Ahead",
        yaxis_title=_("Forecast"),
        plot_bgcolor=theme['plot_bg'],
        paper_bgcolor=theme['plot_bg'],
        font=dict(color=theme['text_on_primary']),
    )
    st.plotly_chart(fig, use_container_width=True)

# -- Other show_* functions: update all color uses to use theme['...'] as above. Add alert-custom for error messages everywhere.

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
            st.markdown(rtl_wrap(
                f'<div class="card" style="border-left: 8px solid {theme["badge_bg"]};">'
                f"{badge}<br><b>{sol['title']}</b><br>{_('Reason')}: {sol['reason']}<br>"
                f'<button style="margin-top:8px;background:{theme["accent"]};color:{theme["text_on_accent"]};border:none;border-radius:8px;padding:5px 12px;">{_("Apply")}</button> '
                f'<button style="margin-top:8px;background:{theme["secondary"]};color:{theme["text_on_secondary"]};border:none;border-radius:8px;padding:5px 12px;">{_("Export")}</button> '
                f'<button style="margin-top:8px;background:transparent;color:{theme["badge_bg"]};border:1px solid {theme["badge_bg"]};border-radius:8px;padding:5px 12px;">{_("Feedback")}</button>'
                f'</div>'
            ), unsafe_allow_html=True)
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
            col = theme["alert"] if a["severity"]=="high" else theme["accent"]
            st.markdown(rtl_wrap(f'<div class="alert-custom" style="background:{col};color:{theme["alert_text"]};">{a["msg"]}</div>'), unsafe_allow_html=True)
    else:
        st.info(_("No alerts at the moment."))

def show_cost():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Cost & Savings")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><div class="metric">5,215,000 SAR</div><div class="metric-label">{_("Yearly Savings")}</div></div>'), unsafe_allow_html=True)
    months = [f"{i+1}/2025" for i in range(6)]
    savings = [400000, 450000, 500000, 550000, 600000, 650000]
    fig = go.Figure(go.Bar(x=months, y=savings, marker_color=theme["accent"]))
    fig.update_layout(
        xaxis_title=_("Monthly Savings"),
        yaxis_title=_("Savings"),
        plot_bgcolor=theme["plot_bg"],
        paper_bgcolor=theme["plot_bg"],
        font=dict(color=theme["secondary"]),
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
    fig.add_trace(go.Bar(x=metrics, y=values_now, name=_("Current"), marker_color=theme["accent"]))
    fig.add_trace(go.Bar(x=metrics, y=values_prev, name=_("Previous"), marker_color=theme["secondary"]))
    fig.update_layout(barmode='group', plot_bgcolor=theme["plot_bg"], paper_bgcolor=theme["plot_bg"], font=dict(color=theme["secondary"]))
    st.plotly_chart(fig, use_container_width=True)

def show_explorer():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("Data Explorer")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="sub-title">{_("Data Filters")}</div>'), unsafe_allow_html=True)
    metrics = [_("Temperature"), _("Pressure"), _("Vibration"), _("Methane"), _("H2S")]
    metric = st.selectbox(_("Select Metric"), options=metrics)
    data = pd.DataFrame({metric: 80 + 5 * np.random.rand(30)})
    st.line_chart(data)

def show_about():
    st.markdown(rtl_wrap(f'<div class="big-title">{_("About the Project")}</div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><span class="badge">{_("Our Vision")}</span><br><i>{_("Disasters don\'t wait.. and neither do we.")}</i></div>'), unsafe_allow_html=True)
    st.markdown(rtl_wrap(f'<div class="card"><span class="badge">{_("What does it do?")}</span><br>{_("Smart Digital Twin is an advanced platform for oilfield safety that connects to real sensors, predicts anomalies, and offers actionable insights to prevent disasters before they happen.")}</div>'), unsafe_allow_html=True)
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
routes[st.session_state.page_radio[0]]()
