import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_extras.animated_number import st_animated_number  # NEW: animated KPIs
from streamlit_extras.animated_text import animated_text
import requests
import os
import time

# --- TRANSLATIONS BLOCK ---
translations = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "welcome": "Welcome to the Smart Neural Digital Twin!",
        "splash_msg": "Protection, prediction, and real intelligence for your plant.",
        "dashboard": "Dashboard",
        "predictive": "Predictive Analytics",
        "solutions": "Smart Solutions",
        "alerts": "Alerts",
        "cost": "Cost & Savings",
        "ai_vs_manual": "AI vs. Manual",
        "scenario": "Realistic Scenario",
        "roadmap": "Roadmap & Changelog",
        "about": "About",
        "select_lang": "Select Language",
        "generate": "✨ Generate Solutions",
        "no_solutions": "No solutions generated yet. Click the button to create AI-driven solutions.",
        "solution_title": "Solution",
        "solution_desc": "Description",
        "solution_eff": "Effectiveness",
        "solution_priority": "Priority",
        "solution_time": "Time Estimate",
        "priority_high": "High",
        "priority_med": "Medium",
        "priority_low": "Low",
        "apply": "Apply",
        "simulate": "Simulate",
        "live_dt": "Digital Twin Live",
        "plant_status": "Plant Status",
        "sensor": "Sensor",
        "status_ok": "OK",
        "status_warn": "Warning",
        "status_fault": "Fault",
        "ai_think": "Neural AI Thinking...",
        "ai_processing": "AI is analyzing your data...",
        "whatif": "What-If Simulator",
        "drag_label": "Adjust the value to simulate a scenario:",
        "ai_reaction": "AI Reaction",
        "manual_reaction": "Manual Reaction",
        "cost_savings": "Estimated Cost Savings",
        "milestones": "Milestones & Changelog",
        "story_title": "Our Story",
        "team_title": "Team",
        "contact": "Contact",
        "scenario_title": "Gas Leak Incident – Interactive Demo",
        "scenario_intro": "Experience a simulated gas leak and choose how to respond. See how AI vs. manual action impacts safety and cost.",
        "scenario_detected": "Gas leak detected near compressor room. What do you do?",
        "scenario_report_ai": "AI instantly flagged the leak and triggered emergency protocols. Incident contained in 50 seconds.",
        "scenario_wait": "Manual response: Leak spreads for 14 minutes before discovery! Major risk and high cost.",
        "scenario_check": "Manual check: Leak noticed after 7 minutes. Containment slow, moderate loss.",
        "scenario_stats": "Resulting cost: ",
        "scenario_safety": "Safety Impact: ",
        "scenario_fast": "Optimal! Risk minimized, cost saved.",
        "scenario_slow": "High risk, high cost. Faster action is critical!",
        "scenario_mod": "Reduced risk, some loss.",
        "scenario_restart": "Try again",
        "ai_powered": "AI-powered insight and protection",
        "ai_secures": "Intelligence that secures and saves",
        "ai_saves": "Save money and lives with AI",
        "section_divider": "---",
        "feature_splash": "Cutting-edge digital twin for rapid, smart safety.",
        "Navigation": "Navigation"
    },
    "ar": {
        "app_title": "التوأم الرقمي الذكي العصبي",
        "welcome": "مرحبًا في التوأم الرقمي الذكي!",
        "splash_msg": "حماية وتنبؤ وذكاء حقيقي لمنشأتك.",
        "dashboard": "لوحة التحكم",
        "predictive": "تحليلات تنبؤية",
        "solutions": "الحلول الذكية",
        "alerts": "التنبيهات",
        "cost": "التكلفة والتوفير",
        "ai_vs_manual": "الذكاء الاصطناعي مقابل اليدوي",
        "scenario": "سيناريو واقعي",
        "roadmap": "خريطة الطريق والتحديثات",
        "about": "عن المنصة",
        "select_lang": "اختر اللغة",
        "generate": "✨ توليد الحلول",
        "no_solutions": "لم يتم توليد حلول بعد. اضغط الزر لابتكار حلول بالذكاء الاصطناعي.",
        "solution_title": "الحل",
        "solution_desc": "الوصف",
        "solution_eff": "الفعالية",
        "solution_priority": "الأولوية",
        "solution_time": "الوقت المقدر",
        "priority_high": "عالية",
        "priority_med": "متوسطة",
        "priority_low": "منخفضة",
        "apply": "تطبيق",
        "simulate": "محاكاة",
        "live_dt": "التوأم الرقمي الحي",
        "plant_status": "حالة المصنع",
        "sensor": "المستشعر",
        "status_ok": "جيد",
        "status_warn": "تحذير",
        "status_fault": "خلل",
        "ai_think": "معالجة الذكاء العصبي...",
        "ai_processing": "الذكاء يعمل على تحليل بياناتك...",
        "whatif": "محاكاة ماذا لو",
        "drag_label": "اضبط القيمة لمحاكاة سيناريو:",
        "ai_reaction": "استجابة الذكاء الاصطناعي",
        "manual_reaction": "الاستجابة اليدوية",
        "cost_savings": "حساب التوفير",
        "milestones": "الإنجازات والتحديثات",
        "story_title": "قصتنا",
        "team_title": "الفريق",
        "contact": "تواصل",
        "scenario_title": "حادثة تسرب غاز – تجربة تفاعلية",
        "scenario_intro": "عش سيناريو تسرب غاز واختبر استجابتك: الذكاء الاصطناعي مقابل التدخل اليدوي، وشاهد أثر كل خيار على السلامة والتكلفة.",
        "scenario_detected": "تم رصد تسرب غاز قرب غرفة الضواغط. ماذا ستفعل؟",
        "scenario_report_ai": "النظام الذكي كشف التسرب فورًا وفعّل بروتوكولات الطوارئ. تم احتواء الحادث خلال ٥٠ ثانية.",
        "scenario_wait": "استجابة يدوية: انتشر التسرب ١٤ دقيقة قبل اكتشافه! خطر مرتفع وتكلفة عالية.",
        "scenario_check": "فحص يدوي: تم ملاحظة التسرب بعد ٧ دقائق. الاحتواء بطيء، خسارة متوسطة.",
        "scenario_stats": "التكلفة الناتجة: ",
        "scenario_safety": "أثر السلامة: ",
        "scenario_fast": "ممتاز! الخطر في أدنى حد والتكلفة وفرت.",
        "scenario_slow": "خطر مرتفع وتكلفة عالية. الاستجابة السريعة ضرورية!",
        "scenario_mod": "خطر أقل وخسارة متوسطة.",
        "scenario_restart": "جرب مرة أخرى",
        "ai_powered": "تحليل وحماية مدعومة بالذكاء",
        "ai_secures": "ذكاء يحميك ويوفر لك",
        "ai_saves": "وفر أموالك وأرواحك بالذكاء الاصطناعي",
        "section_divider": "---",
        "feature_splash": "توأم رقمي متطور للسلامة السريعة والذكية.",
        "Navigation": "التنقل"
    }
}
def _(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang].get(key, key)

# --- THEME AND CSS (unchanged) ---
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "rtl" not in st.session_state:
    st.session_state["rtl"] = False
def lang_switch():
    st.session_state["lang"] = "ar" if st.session_state["lang"] == "en" else "en"
    st.session_state["rtl"] = not st.session_state["rtl"]

def inject_css(theme="dark"):
    if theme == "dark":
        st.markdown("""
        <style>
        html, body, [class*="st-"] {
            font-family: 'Cairo', 'IBM Plex Sans', sans-serif !important;
        }
        body, .stApp {
            background: linear-gradient(120deg,#181a20 0%,#232526 65%,#43cea2 80%,#fee140 100%);
        }
        .particle-bg {position:fixed;z-index:0;top:0;left:0;width:100vw;height:100vh;pointer-events:none;}
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(135deg,#232526,#485563 70%,#43cea2);
        }
        .stButton>button {
            background: linear-gradient(90deg,#43cea2,#185a9d);
            color: white;
            border: none;
            font-size:1.13em;
            font-weight:bold;
            padding: 0.7em 2.3em;
            border-radius: 17px;
            box-shadow: 0 8px 22px #0003;
            transition: 0.15s;
            animation: pulse 1.7s infinite;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#fa709a,#fee140);
            color: #222;
            transform: scale(1.04);
            box-shadow: 0 12px 32px #fa709a55;
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 #43cea288; }
          70% { box-shadow: 0 0 0 15px #43cea200; }
          100% { box-shadow: 0 0 0 0 #43cea200; }
        }
        .fab {
          position: fixed;
          bottom: 2.6em;
          right: 2.6em;
          z-index: 15;
          background: linear-gradient(90deg,#43cea2,#fa709a);
          color: white;
          border-radius: 50%;
          width: 62px; height: 62px;
          display: flex;align-items:center;justify-content:center;
          box-shadow: 0 4px 24px #fa709a77;
          cursor: pointer;
          font-size:2em;
          transition: 0.19s;
        }
        .fab:hover {transform:scale(1.1);}
        .section-divider {margin: 2em 0 1.5em 0;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        html, body, [class*="st-"] {
            font-family: 'Cairo', 'IBM Plex Sans', sans-serif !important;
        }
        body, .stApp {
            background: linear-gradient(120deg,#fffbe6 0%,#f7f7f7 55%,#43cea2 90%,#fee140 100%);
            color: #222;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(135deg,#f7f7f7,#fee140 70%,#43cea2);
        }
        .stButton>button {
            background: linear-gradient(90deg,#fee140,#43cea2);
            color: #222;
            border: none;
            font-size:1.13em;
            font-weight:bold;
            padding: 0.7em 2.3em;
            border-radius: 17px;
            box-shadow: 0 8px 22px #fee14088;
            transition: 0.15s;
            animation: pulse 1.7s infinite;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#fa709a,#43cea2);
            color: #fff;
            transform: scale(1.04);
            box-shadow: 0 12px 32px #fa709a55;
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 #fee14088; }
          70% { box-shadow: 0 0 0 15px #fee14000; }
          100% { box-shadow: 0 0 0 0 #fee14000; }
        }
        .fab {
          position: fixed;
          bottom: 2.6em;
          right: 2.6em;
          z-index: 15;
          background: linear-gradient(90deg,#fee140,#43cea2);
          color: #222;
          border-radius: 50%;
          width: 62px; height: 62px;
          display: flex;align-items:center;justify-content:center;
          box-shadow: 0 4px 24px #fee14077;
          cursor: pointer;
          font-size:2em;
          transition: 0.19s;
        }
        .fab:hover {transform:scale(1.1);}
        .section-divider {margin: 2em 0 1.5em 0;}
        </style>
        """, unsafe_allow_html=True)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
inject_css(st.session_state["theme"])
def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
    inject_css(st.session_state["theme"])

def rtl_mirror():
    if st.session_state["rtl"]:
        st.markdown("""<style>body, .stApp, [data-testid="stSidebar"], .stButton>button {direction:rtl !important; text-align:right !important;}</style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>body, .stApp, [data-testid="stSidebar"], .stButton>button {direction:ltr !important; text-align:left !important;}</style>""", unsafe_allow_html=True)
rtl_mirror()

# --- LOTTIE UTILS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# --- ICONS/IMAGES ---
plant_img = "https://images.pexels.com/photos/236089/pexels-photo-236089.jpeg?auto=compress&w=800&q=80"
control_img = "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=800&q=80"
sensor_img = "https://images.pexels.com/photos/3850752/pexels-photo-3850752.jpeg?auto=compress&w=800&q=80"
plant_twin_img = "https://i.ibb.co/4R0vY4Q/plant-twin-demo.png"
twin_lottie = "https://assets1.lottiefiles.com/packages/lf20_wnqlfojb.json"
ai_lottie   = "https://assets10.lottiefiles.com/packages/lf20_5ngs2ksb.json"
plant_lottie = "https://assets3.lottiefiles.com/packages/lf20_5b2dh9jt.json"
alert_lottie = "https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"
sensor_lottie = "https://lottie.host/efb1a1d3-6a7c-4bfb-bb9e-8e1e4fae5c9e/gas_sensor.json"
iot_lottie = "https://lottie.host/0b74e5c4-9e2e-4d5b-9e6b-f315e5b6d82e/iot.json"
robot_lottie = "https://lottie.host/503c7d51-9b64-495a-8e0e-25e80c2e7aaa/robot.json"
fireworks_lottie = "https://assets7.lottiefiles.com/packages/lf20_8wREpI.json"

# --- ADVANCED PREDICTION ENGINE INTEGRATION ---
from advanced_prediction_engine import predict_future_values_72h, get_prediction_summary

@st.cache_resource
def load_prediction_models():
    return joblib.load("advanced_prediction_models.pkl")
models = load_prediction_models()

@st.cache_data
def load_sensor_data():
    fname = "sensor_data_simulated_long.csv" if os.path.exists("sensor_data_simulated_long.csv") else "sensor_data_simulated.csv"
    return pd.read_csv(fname, parse_dates=["Timestamp"] if fname.endswith("long.csv") else ["Time"])
sensor_df = load_sensor_data()

# --- SPLASH SCREEN ---
if "splash_shown" not in st.session_state:
    st_lottie(load_lottieurl(plant_lottie), height=240, loop=False)
    st.markdown(
        f"<h1 style='font-size:2.1em;color:#43cea2;font-weight:bold;text-align:center;margin-top:2em;'>{_('welcome')}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center;font-size:1.2em;color:#185a9d;margin-bottom:2em;'>{_('splash_msg')}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center;font-size:1.1em;color:#43cea2;'>{_('feature_splash')}</div>",
        unsafe_allow_html=True
    )
    st.session_state["splash_shown"] = True
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st_lottie(load_lottieurl(twin_lottie), height=70, key="icon-lottie", speed=1.1, loop=True)
    nav = option_menu(
        None,
        [
            _("dashboard"), _("predictive"), _("solutions"), _("scenario"),
            _("alerts"), _("cost"), _("ai_vs_manual"), _("live_dt"),
            _("roadmap"), _("about")
        ],
        icons=[
            "speedometer", "activity", "lightbulb", "flag", "bell",
            "cash", "cpu", "layers", "clock-history", "info-circle"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#fa709a", "font-size": "1.5em"},
            "nav-link-selected": {"background": "linear-gradient(90deg,#43cea2,#fa709a)", "color": "#fff"},
            "nav-link": {"color": "#fff" if st.session_state["theme"] == "dark" else "#222"}
        }
    )
    # Theme toggle
    theme_label = "🌑/🌕 Switch Mode" if st.session_state["lang"] == "en" else "الوضع الليلي/النهاري"
    st.button(theme_label, on_click=toggle_theme)
    # Language
    st.selectbox(
        f"🌐 {_('select_lang')}",
        options=[("English", "en"), ("العربية", "ar")],
        index=0 if st.session_state["lang"] == "en" else 1,
        key="lang_select",
        on_change=lang_switch,
        format_func=lambda x: x[0]
    )

# --- FLOATING ACTION BUTTON (Feedback) ---
st.markdown("""
<div class="fab" onclick="window.open('mailto:your@email.com?subject=SmartTwin Feedback','_blank')"
 title="Feedback/Help" aria-label="Feedback" tabindex="0" role="button">
    💬
</div>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown(f"<h1 style='font-weight:bold;color:#43cea2;text-shadow:0 2px 16px #185a9d44;'>{_('app_title')}</h1>", unsafe_allow_html=True)

# --- DASHBOARD PAGE ---
if nav == _("dashboard"):
    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader(_("plant_status"))
        st_lottie(load_lottieurl(plant_lottie), height=120, key="plant-lottie", loop=True)
        st.image(plant_img, use_container_width=True, caption="Industrial Plant")
        latest = sensor_df.iloc[-1]
        st.markdown(f"""
        <div style="margin-top:1.2em;font-size:1.07em;">
            <b>{_('sensor')} 1:</b> <span style="color:#43cea2">{latest['Temperature (°C)']:.1f} °C</span><br>
            <b>{_('sensor')} 2:</b> <span style="color:#fa709a">{latest['Pressure (psi)']:.1f} psi</span><br>
            <b>{_('sensor')} 3:</b> <span style="color:#fee140">{latest['Methane (CH₄ ppm)']:.2f} ppm</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st_lottie(load_lottieurl(ai_lottie), height=120, key="ai-lottie", loop=True)
        st_lottie(load_lottieurl(sensor_lottie), height=120, key="sensor-lottie", loop=True)
        st.info(_("about_text"))
    # Animated KPI Numbers
    st.markdown('<div class="dashboard-chart-container">', unsafe_allow_html=True)
    colk1, colk2, colk3 = st.columns(3)
    with colk1:
        st.markdown("**AI Savings**")
        st_animated_number(13500, format="$,.0f", animation_speed=2)
    with colk2:
        st.markdown("**Downtime Reduction**")
        st_animated_number(0.71, format=".2%", animation_speed=2)
    with colk3:
        st.markdown("**Predicted Loss (Manual)**")
        st_animated_number(22100, format="$,.0f", animation_speed=2)
    # Chart
    st.markdown("<b>Sensor 1 readings (last 24h):</b>", unsafe_allow_html=True)
    last_24h = sensor_df.tail(24)
    fig = px.line(last_24h, x=last_24h.columns[0], y="Temperature (°C)", title="Sensor 1 Temperature (Animated)", markers=True)
    fig.update_layout(
        transition=dict(duration=500),
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=30),
        height=320,
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        font=dict(size=15)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_powered')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- PREDICTIVE ANALYTICS PAGE ---
if nav == _("predictive"):
    st.subheader(_("predictive"))
    st_lottie(load_lottieurl(iot_lottie), height=140, key="iot-lottie", loop=True)
    st.image(sensor_img, use_container_width=True, caption="Gas Sensor Monitoring")
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_secures')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:1.08em;">
    <ul>
        <li>📈 Real-time anomaly detection on all sensors</li>
        <li>🔮 Early warning on predicted faults (e.g., days/hours before they occur)</li>
        <li>🌡️ Trend analysis for critical process parameters</li>
        <li>🧠 AI confidence heatmap for every prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Show last 72 hours
    st.markdown("#### Sensor Health Status (Last 72h Forecast)")
    with st.spinner(_("ai_processing")):
        predictions = predict_future_values_72h(models, 72)
        pred_temp = predictions['Temperature (°C)']
        pred_times = [p['time'] for p in pred_temp]
        pred_vals = [p['value'] for p in pred_temp]
        time.sleep(0.7)
    fig_pred = px.line(x=pred_times, y=pred_vals, title="Predicted Temperature (°C) - Next 72h")
    fig_pred.update_traces(line=dict(color="#fa709a", width=4))
    fig_pred.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_pred, use_container_width=True)

    # ML confidence
    summary = get_prediction_summary(predictions)
    conf = np.mean([p['confidence'] for p in pred_temp])
    st.progress(conf, text=f"AI Confidence: {int(conf*100)}%")

    # What-if Simulator
    st.markdown("---")
    st.subheader(_("whatif"))
    val = st.slider(_("drag_label"), min_value=0, max_value=100, value=30, step=5, key="simu_slider")
    ai_risk = min(val/100, 1.0)
    manual_risk = min((val+35)/100, 1.0)
    st.progress(ai_risk, text="AI Risk Level")
    st.progress(manual_risk, text="Manual Risk Level")
    st.markdown(f"<b>{_('cost_savings')}:</b> <span style='color:#43cea2;font-weight:bold;'>${(manual_risk-ai_risk)*8000:,.0f}</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_saves')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- SMART SOLUTIONS (Colored Cards) ---
if nav == _("solutions"):
    st.markdown(f"<h2>{_('solutions')}</h2>", unsafe_allow_html=True)
    st_lottie(load_lottieurl(robot_lottie), height=120, key="robot-lottie", loop=True)
    st.image(control_img, use_container_width=True, caption="Industrial Control Room")
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_powered')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)
    summary = get_prediction_summary(predict_future_values_72h(models, 24))
    solutions_data = []
    for sensor, stat in summary.items():
        eff = int(100 - stat["24h"]["volatility"])
        priority = _("priority_high") if stat["24h"]["trend"] == "increasing" else _("priority_med")
        icon = "🛡️" if priority == _("priority_high") else "💡"
        solutions_data.append({
            "icon": icon,
            "title": f"Optimize {sensor}",
            "desc": f"AI recommends action as {sensor} is showing a {stat['24h']['trend']} trend.",
            "eff": eff,
            "priority": priority,
            "time": "Immediate" if priority == _("priority_high") else "Within 2h"
        })
    if st.button(_( "generate" )):
        st.session_state["solutions"] = solutions_data
        st.success(_("ai_think"))
        time.sleep(1)
    if "solutions" not in st.session_state or not st.session_state["solutions"]:
        st.info(_( "no_solutions" ))
    else:
        for i, s in enumerate(st.session_state["solutions"]):
            eff_color = "#43cea2" if s["priority"] == _("priority_high") else "#fa709a"
            st.markdown(f"""
            <div style="border-radius:17px;background:linear-gradient(90deg,#232526,#43cea2 65%,#fee140 100%);box-shadow:0 4px 24px #43cea288;margin:1em 0;padding:1.5em 2em;color:white;">
                <span style="font-size:2em;margin-right:0.4em;">{s['icon']}</span>
                <span style="font-size:1.07em;font-weight:bold;">{s['title']}</span>
                <div style="margin:0.6em 0 0.5em 0;">{s['desc']}</div>
                <div>{_('solution_eff')}: <b style="color:{eff_color};">{s['eff']}%</b></div>
                <div>{_('solution_priority')}: <b>{s['priority']}</b></div>
                <div>{_('solution_time')}: <b>{s['time']}</b></div>
            </div>
            """, unsafe_allow_html=True)
        # Pie chart: Solution priorities
        st.markdown("<b>Solution Priority Distribution:</b>", unsafe_allow_html=True)
        priorities = [s["priority"] for s in st.session_state["solutions"]]
        labels = [_("priority_high"), _("priority_med"), _("priority_low")]
        counts = [priorities.count(lab) for lab in labels]
        figpie = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=0.4)])
        figpie.update_traces(marker=dict(colors=["#43cea2", "#fa709a", "#fee140"]))
        figpie.update_layout(showlegend=True)
        st.plotly_chart(figpie, use_container_width=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- INTERACTIVE DIGITAL TWIN LIVE ---
if nav == _("live_dt"):
    st.markdown(f"<h2>{_('live_dt')}</h2>", unsafe_allow_html=True)
    st_lottie(load_lottieurl(twin_lottie), height=180, key="livedt-lottie", loop=True)
    st.image(plant_twin_img, use_container_width=True, caption="Live Plant Digital Twin")
    # Overlay sensor values from latest row
    latest = sensor_df.iloc[-1]
    st.markdown(f"""
    <div style="font-size:1.03em;">
    <b>Plant Diagram (AI Overlay):</b>
    <br>
    <img src="{plant_twin_img}" style="width:80%;border-radius:14px;box-shadow:0 2px 12px #43cea277;"/>
    <br>
    <b>Sensors (Live):</b>
    <ul>
        <li>🌡️ Temp: <b style='color:#43cea2'>{latest['Temperature (°C)']:.1f} °C</b></li>
        <li>🧪 Gas: <b style='color:#fa709a'>{latest['Methane (CH₄ ppm)']:.2f} ppm</b></li>
        <li>💧 Pressure: <b style='color:#fee140'>{latest['Pressure (psi)']:.1f} psi</b></li>
        <li>🔋 Power: <b style='color:#43cea2'>Stable</b></li>
    </ul>
    <b>Fault Propagation:</b> <span style="color:#fa709a;font-weight:bold;">None</span>
    </div>
    """, unsafe_allow_html=True)
    # Animated surface plot (fake heatmap for demo)
    x, y = np.meshgrid(np.linspace(0, 10, 12), np.linspace(0, 10, 12))
    z = np.sin(x) * np.cos(y) * 10 + latest['Temperature (°C)']
    fig3d = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig3d.update_layout(title="Plant Temperature Surface", autosize=True, margin=dict(l=20, r=20, b=20, t=30))
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_secures')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- AI VS MANUAL (Animated Radar) ---
if nav == _("ai_vs_manual"):
    st.subheader(_("ai_vs_manual"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<b>{_('ai_reaction')}</b>", unsafe_allow_html=True)
        st_lottie(load_lottieurl(ai_lottie), height=90, key="ai-vs-lottie", loop=True)
        st.success("AI detects & responds in 1.2s")
    with col2:
        st.markdown(f"<b>{_('manual_reaction')}</b>", unsafe_allow_html=True)
        st_lottie(load_lottieurl(robot_lottie), height=90, key="manual-vs-lottie", loop=True)
        st.error("Manual detection: 18 min average")
    # Animated radar
    metrics = ["Speed", "Accuracy", "Cost", "Downtime", "Safety"] if st.session_state["lang"] == "en" else ["السرعة", "الدقة", "التكلفة", "التوقف", "السلامة"]
    ai_vals = [95, 92, 90, 85, 98]
    man_vals = [60, 70, 70, 60, 75]
    radar_df = pd.DataFrame({
        "Metric": metrics*2,
        "Value": ai_vals + man_vals,
        "Type": (["AI"]*5)+(["Manual"]*5)
    })
    fig_radar = px.line_polar(radar_df, r="Value", theta="Metric", color="Type", line_close=True, template="plotly_dark" if st.session_state["theme"]=="dark" else "plotly_white",
                              color_discrete_map={"AI": "#43cea2", "Manual": "#fa709a"})
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_powered')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- SCENARIO (Timeline Animation) ---
if nav == _("scenario"):
    st.subheader(_( "scenario_title" ))
    st.markdown(f"<div style='font-size:1.08em;color:#185a9d;font-weight:bold;'>{_('scenario_intro')}</div>", unsafe_allow_html=True)
    st_lottie(load_lottieurl(alert_lottie), height=130, key="scenario-lottie", loop=True)
    st.image(sensor_img, use_container_width=True, caption="Gas Sensor Scenario")
    if "scenario_state" not in st.session_state:
        st.session_state["scenario_state"] = 0
    def reset_scenario():
        st.session_state["scenario_state"] = 0
    # Timeline progress
    progress = [0.0, 0.33, 0.66, 1.0]
    st.progress(progress[st.session_state["scenario_state"]], text=f"Step {st.session_state['scenario_state']+1}/4")
    # Scenario logic
    if st.session_state["scenario_state"] == 0:
        st.markdown(f"<div class='scenario-box'>{_('scenario_detected')}</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📢 " + (_("apply") if st.session_state["lang"] == "en" else "أبلغ الذكاء الاصطناعي")):
                st.session_state["scenario_state"] = 1
        with col2:
            if st.button("⏳ " + ("Wait" if st.session_state["lang"] == "en" else "انتظر")):
                st.session_state["scenario_state"] = 2
        with col3:
            if st.button("🔍 " + ("Manual check" if st.session_state["lang"] == "en" else "فحص يدوي")):
                st.session_state["scenario_state"] = 3
    elif st.session_state["scenario_state"] == 1:
        st.success(_( "scenario_report_ai" ))
        st.markdown(f"<b>{_('scenario_stats')}</b> <span style='color:#43cea2;font-weight:bold;'>$700</span>", unsafe_allow_html=True)
        st.markdown(f"<b>{_('scenario_safety')}</b> <span style='color:#43cea2;font-weight:bold;'>{_('scenario_fast')}</span>", unsafe_allow_html=True)
        st_lottie(load_lottieurl(fireworks_lottie), height=100, key="scenario-fireworks", loop=False)
        if st.button(_( "scenario_restart" )):
            reset_scenario()
    elif st.session_state["scenario_state"] == 2:
        st.error(_( "scenario_wait" ))
        st.markdown(f"<b>{_('scenario_stats')}</b> <span style='color:#fa709a;font-weight:bold;'>$18,000</span>", unsafe_allow_html=True)
        st.markdown(f"<b>{_('scenario_safety')}</b> <span style='color:#fa709a;font-weight:bold;'>{_('scenario_slow')}</span>", unsafe_allow_html=True)
        if st.button(_( "scenario_restart" )):
            reset_scenario()
    elif st.session_state["scenario_state"] == 3:
        st.warning(_( "scenario_check" ))
        st.markdown(f"<b>{_('scenario_stats')}</b> <span style='color:#fee140;font-weight:bold;'>$8,000</span>", unsafe_allow_html=True)
        st.markdown(f"<b>{_('scenario_safety')}</b> <span style='color:#fee140;font-weight:bold;'>{_('scenario_mod')}</span>", unsafe_allow_html=True)
        if st.button(_( "scenario_restart" )):
            reset_scenario()
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_saves')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- ALERTS (Pie, Lottie) ---
if nav == _("alerts"):
    st.subheader(_("alerts"))
    st_lottie(load_lottieurl(alert_lottie), height=120, key="alert-lottie", loop=True)
    st.image(sensor_img, use_container_width=True, caption="Live Alerts")
    # Example: if methane forecast is risky
    predictions = predict_future_values_72h(models, 24)
    summary = get_prediction_summary(predictions)
    methane_max_24h = summary["Methane (CH₄ ppm)"]["24h"]["max"]
    if methane_max_24h > 4.0:
        st.error("⚠️ Predicted methane spike in next 24h! Immediate action suggested.")
    else:
        st.success("Methane levels predicted to remain safe in next 24h.")
    labels = ["Sensor Fault", "Leak", "Power", "Other"] if st.session_state["lang"] == "en" else ["خلل مستشعر", "تسرب", "كهرباء", "أخرى"]
    values = [2, 1, 0, 1]
    fig_alert = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig_alert.update_traces(marker=dict(colors=["#fa709a", "#fee140", "#43cea2", "#185a9d"]))
    fig_alert.update_layout(showlegend=True)
    st.plotly_chart(fig_alert, use_container_width=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_powered')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- COST & SAVINGS (Animated) ---
if nav == _("cost"):
    st.subheader(_("cost"))
    st.image(plant_img, use_container_width=True, caption="Cost Analysis")
    colk1, colk2, colk3 = st.columns(3)
    with colk1:
        st.markdown("**AI Savings This Month**")
        st_animated_number(13500, format="$,.0f", animation_speed=2)
    with colk2:
        st.markdown("**Predicted Loss (Manual)**")
        st_animated_number(22100, format="$,.0f", animation_speed=2)
    with colk3:
        st.markdown("**Downtime Reduction**")
        st_animated_number(0.71, format=".2%", animation_speed=2)
    st.progress(0.71, text="Downtime Reduced")
    cost_labels = ["Maintenance", "Downtime", "Energy", "Other"] if st.session_state["lang"] == "en" else ["صيانة", "توقف", "طاقة", "أخرى"]
    cost_vals = [5000, 6000, 3000, 1500]
    fig_cost = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_vals, hole=0.5)])
    fig_cost.update_traces(marker=dict(colors=["#43cea2", "#fa709a", "#fee140", "#185a9d"]))
    fig_cost.update_layout(showlegend=True)
    st.plotly_chart(fig_cost, use_container_width=True)
    # Cost trend (random for now, can be replaced with real values)
    months = pd.date_range(end=pd.Timestamp.now(), periods=12, freq="M")
    ai_cost = np.random.randint(8000, 12000, size=12)
    man_cost = ai_cost + np.random.randint(3000, 8000, size=12)
    df_cost = pd.DataFrame({
        "Month": months,
        "AI": ai_cost,
        "Manual": man_cost
    })
    fig_costline = px.line(df_cost, x="Month", y=["AI", "Manual"], title="Monthly Cost Comparison", markers=True)
    fig_costline.update_layout(transition=dict(duration=500))
    st.plotly_chart(fig_costline, use_container_width=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('ai_saves')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# --- ROADMAP & ABOUT (Timeline Animation) ---
if nav == _("roadmap"):
    st.subheader(_("milestones"))
    st.info(_( "roadmap_text" ))
    st.markdown("""
    <div class="timeline">
        <div class="timeline-event"><b>2025 Q1:</b> Platform launch 🚀</div>
        <div class="timeline-event"><b>2025 Q2:</b> Real-time AI alerts, scenario engine, and live dashboard</div>
        <div class="timeline-event"><b>2025 Q3:</b> External API integration, new ML models, custom analytics</div>
        <div class="timeline-event"><b>2025 Q4:</b> Full industrial deployment, mobile app, multi-language</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('feature_splash')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

if nav == _("about"):
    st.subheader(_("story_title"))
    animated_text(_("story"), speed=18)
    st.markdown(f"## { _('features_title') }")
    st.markdown(
        "\n".join([f"- {f}" for f in translations[st.session_state['lang']].get('features',[])])
    )
    st.markdown("---")
    st.subheader(_("team_title"))
    for member in translations[st.session_state["lang"]]["team"]:
        st.markdown(
            f"""
            <div class="team-card">
                <b>{member['name']}</b> — {member['role']}<br>
                <span style="font-size:.9em;color:#eee;">{member['email']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown(f"<b>{_('contact')}:</b> {translations[st.session_state['lang']]['team'][0]['email']}")
    st.markdown(f"<div style='font-size:1.1em;color:#43cea2;font-weight:bold;'>{_('feature_splash')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-divider'>{_('section_divider')}</div>", unsafe_allow_html=True)

# Accessibility/ARIA
st.markdown("""
<script>
document.querySelectorAll('.fab')[0]?.setAttribute('tabindex', 0);
document.querySelectorAll('.fab')[0]?.setAttribute('role', 'button');
document.querySelectorAll('.fab')[0]?.setAttribute('aria-label', 'Feedback');
</script>
""", unsafe_allow_html=True)
