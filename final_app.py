import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
import requests
import random
import pandas as pd
import numpy as np

# ----------------- FONTS & STYLE (NEW) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@700&family=IBM+Plex+Sans:wght@700&display=swap');
html, body, [class*="st-"] {
    font-family: 'Cairo', 'IBM Plex Sans', sans-serif !important;
}
body, .stApp {
    background: linear-gradient(120deg,#181a20,#232526 65%,#43cea2,#fee140 100%);
}
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
.solution-card {
    background: linear-gradient(90deg,#232526,#485563 70%,#56ab2f);
    color:white;
    border-radius: 19px;
    padding: 1.25em 1.7em;
    margin-bottom: 1.1em;
    box-shadow: 0 4px 24px #43cea244;
    position: relative;
    overflow: hidden;
    animation: slide-in 0.9s;
}
@keyframes slide-in {
    0% { opacity: 0; transform: translateY(50px);}
    100% { opacity: 1; transform: none;}
}
.solution-icon {
    font-size: 2.2em;
    position: absolute;
    top: 1.2em;
    right: 1.2em;
    opacity: 0.13;
    pointer-events: none;
}
.gradient-bar {
    height: 13px;
    border-radius: 8px;
    background: linear-gradient(90deg,#43cea2,#185a9d,#fa709a,#fee140);
    margin-bottom: 0.7em;
}
.team-card {
    background: linear-gradient(90deg,#232526, #43cea2 80%);
    color: white;
    border-radius: 15px;
    margin-bottom: 1em;
    padding: 1em 2em;
    box-shadow: 0 2px 16px #0003;
}
.timeline {
    border-left: 4px solid #43cea2;
    margin-left: 1em;
    padding-left: 1.6em;
}
.timeline-event {
    margin-bottom: 1.5em;
    position: relative;
}
.timeline-event:before {
    content: '';
    position: absolute;
    left: -1.6em;
    top: 0.3em;
    width: 1em;
    height: 1em;
    background: linear-gradient(90deg,#43cea2,#fa709a);
    border-radius: 50%;
    border: 2px solid white;
    box-shadow: 0 0 8px #43cea2;
}
.scenario-box {
    background: linear-gradient(90deg,#185a9d11,#fee14022);
    border-radius: 14px;
    padding: 1.2em 1em;
    margin: 1.1em 0 0.7em 0;
    box-shadow: 0 2px 12px #fa709a22;
}
</style>
""", unsafe_allow_html=True)

# ----------------- LOTTIE HELPER -----------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None

# Lottie URLs
twin_lottie = "https://assets1.lottiefiles.com/packages/lf20_wnqlfojb.json"
ai_lottie   = "https://assets10.lottiefiles.com/packages/lf20_5ngs2ksb.json"
plant_lottie = "https://assets3.lottiefiles.com/packages/lf20_5b2dh9jt.json"
confetti_lottie = "https://assets8.lottiefiles.com/packages/lf20_0os2dcp1.json"
alert_lottie = "https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"

# ----------------- TRANSLATIONS -----------------
translations = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
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
        "story": """Our journey began with a simple question: How can we detect gas leaks before disaster strikes? We tried everything, even innovated with drones—and it worked. But we asked ourselves: Why wait for the problem at all?
Our dream was a smart digital twin that predicts danger before it happens—not impossible, but difficult. We made the difficult easy connecting AI with plant data in a single platform that monitors, learns, and prevents disasters before they start.
Today, our platform is the first line of defense, changing the rules of industrial safety. This is the future.""",
        "team": [
            {"name": "Abdulrahman Alzahrani", "role": "Development & Design", "email": "abdulrahman.zahrani.1@aramco.com"},
            {"name": "Rakan Almarri", "role": "Development & Design", "email": "rakan.almarri.2@aramco.com"}
        ],
        "roadmap_text": "Our roadmap includes deeper integration, more process types, and advanced AI for prediction and prevention.",
        "about_text": "A new standard for AI-driven industrial safety, built by students passionate about smart tech.",
        "features_title": "Key Features",
        "features": [
            "Real AI-powered predictions (not just static rules)",
            "Interactive scenario: try a real gas leak simulation",
            "Visual charts and animated feedback",
            "Full Arabic and English support",
            "Simple, mobile-friendly UI",
            "Open source platform — experiment and improve it!"
        ]
    },
    "ar": {
        "app_title": "التوأم الرقمي الذكي العصبي",
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
        "story": """بدأت رحلتنا من سؤال بسيط: كيف نكشف تسرب الغاز قبل أن يتحول إلى كارثة؟ جربنا كل الحلول، وابتكرنا حتى استخدمنا الدرون بنجاح. لكن وقفنا وسألنا: ليه ننتظر أصلاً؟
حلمنا كان بناء توأم رقمي ذكي يتوقع الخطر قبل حدوثه. مو مستحيل، لكن كان صعب. إحنا أخذنا الصعب وخليناه سهل، وربطنا الذكاء الاصطناعي مع بيانات المصنع في منصة واحدة، تراقب وتتعلم وتمنع الكوارث قبل أن تبدأ.
اليوم، منصتنا هي خط الدفاع الأول، تغير قواعد الأمان الصناعي من أساسها. هذا هو المستقبل.""",
        "team": [
            {"name": "عبدالرحمن الزهراني", "role": "تطوير وتصميم", "email": "abdulrahman.zahrani.1@aramco.com"},
            {"name": "راكان المري", "role": "تطوير وتصميم", "email": "rakan.almarri.2@aramco.com"}
        ],
        "roadmap_text": "تشمل خطتنا التكامل الأعمق، وزيادة أنواع العمليات، وذكاء تنبؤي أقوى.",
        "about_text": "منصة ذكية للأمان الصناعي — عمل طلابي متحمس للتقنية الذكية.",
        "features_title": "المميزات الرئيسية",
        "features": [
            "توقعات ذكية فعلاً (مو مجرد قواعد ثابتة)",
            "سيناريو تفاعلي: جرب تسرب غاز واقعي!",
            "رسوم بيانية وواجهات متحركة",
            "دعم كامل للعربية والإنجليزية",
            "واجهة سهلة وسريعة (حتى من الجوال)",
            "كل شيء مفتوح المصدر وتقدر تطوره"
        ]
    }
}

solution_icons = [
    "🛑",  # Danger
    "⚡",  # Power
    "🔥",  # Fire/Heat
    "💧",  # Leak
    "🛠️",  # Maintenance
    "🌡️",  # Temperature
    "📈",  # Trend
]

solutions_data = {
    "en": [
        {
            "title": "Methane Spike",
            "desc": "AI detected rapid methane increase in Tank 3.",
            "eff": 98,
            "priority": "High",
            "time": "2h",
            "icon": solution_icons[0],
        },
        {
            "title": "Pressure Drop",
            "desc": "Sudden pressure drop in Pipeline B.",
            "eff": 92,
            "priority": "High",
            "time": "1h",
            "icon": solution_icons[3],
        },
        {
            "title": "Overheating Motor",
            "desc": "Compressor motor temperature abnormality.",
            "eff": 86,
            "priority": "Medium",
            "time": "3h",
            "icon": solution_icons[2],
        },
        {
            "title": "Power Fluctuation",
            "desc": "Unstable voltage detected at panel 5.",
            "eff": 80,
            "priority": "Medium",
            "time": "4h",
            "icon": solution_icons[1],
        },
        {
            "title": "Routine Maintenance",
            "desc": "Pump station scheduled for service.",
            "eff": 75,
            "priority": "Low",
            "time": "8h",
            "icon": solution_icons[4],
        },
    ],
    "ar": [
        {
            "title": "ارتفاع الميثان",
            "desc": "كشف الذكاء الاصطناعي زيادة سريعة في الميثان في الخزان 3.",
            "eff": 98,
            "priority": "عالية",
            "time": "ساعتان",
            "icon": solution_icons[0],
        },
        {
            "title": "انخفاض الضغط",
            "desc": "انخفاض مفاجئ في الضغط في الأنبوب ب.",
            "eff": 92,
            "priority": "عالية",
            "time": "ساعة",
            "icon": solution_icons[3],
        },
        {
            "title": "ارتفاع حرارة المحرك",
            "desc": "ارتفاع غير طبيعي في درجة حرارة ضاغط المحرك.",
            "eff": 86,
            "priority": "متوسطة",
            "time": "3 ساعات",
            "icon": solution_icons[2],
        },
        {
            "title": "تذبذب الكهرباء",
            "desc": "رصد تقلبات جهد غير مستقرة في لوحة 5.",
            "eff": 80,
            "priority": "متوسطة",
            "time": "4 ساعات",
            "icon": solution_icons[1],
        },
        {
            "title": "صيانة دورية",
            "desc": "جدولة صيانة لمحطة الضخ.",
            "eff": 75,
            "priority": "منخفضة",
            "time": "8 ساعات",
            "icon": solution_icons[4],
        },
    ]
}

# ----------------- LANGUAGE STATE -----------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

def _(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang].get(key, key)

def lang_switch():
    st.session_state["lang"] = "ar" if st.session_state["lang"] == "en" else "en"

# ----------------- SIDEBAR -----------------
st.set_page_config(page_title=_("app_title"), layout="wide", page_icon="🧠")

sidebar_lottie = load_lottieurl(twin_lottie)

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:row;align-items:center;gap:0.8em;">
            <div style="background:linear-gradient(135deg,#43cea2,#185a9d);border-radius:16px;padding:0.2em 0.6em;">
                <span style="font-size:2.3em;">🧠</span>
            </div>
            <span style="font-weight:bold;font-size:1.22em;letter-spacing:0.03em;background:linear-gradient(90deg,#43cea2,#fa709a,#fee140);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{_('app_title')}</span>
        </div>
        """, unsafe_allow_html=True
    )
    if sidebar_lottie:
        st_lottie(sidebar_lottie, height=70, key="icon-lottie", speed=1.1, loop=True)
    st.write("")

    nav = st.radio(
        _("Navigation") if st.session_state["lang"] == "en" else "التنقل",
        (
            _("dashboard"),
            _("predictive"),
            _("solutions"),
            _("scenario"),
            _("alerts"),
            _("cost"),
            _("ai_vs_manual"),
            _("live_dt"),
            _("roadmap"),
            _("about"),
        ),
        key="nav_radio"
    )

    st.markdown("---")
    st.selectbox(
        f"🌐 {_('select_lang')}",
        options=[("English", "en"), ("العربية", "ar")],
        index=0 if st.session_state["lang"] == "en" else 1,
        key="lang_select",
        on_change=lang_switch,
        format_func=lambda x: x[0]
    )
    st.markdown(
        """<style>
        .stRadio > div { flex-direction: column; }
        .stSelectbox > div { direction: ltr !important; }
        </style>""", unsafe_allow_html=True
    )

# ----------------- PAGE LOGIC -----------------
st.markdown(f"<h1 style='font-weight:bold;color:#43cea2;text-shadow:0 2px 16px #185a9d44;'>{_('app_title')}</h1>", unsafe_allow_html=True)

# ========== DASHBOARD ==========
if nav == _("dashboard"):
    col1, col2 = st.columns([2,3])

    with col1:
        st.subheader(_("plant_status"))
        lottie_data = load_lottieurl(plant_lottie)
        if lottie_data:
            st_lottie(lottie_data, height=170, key="plant-lottie", loop=True)
        st.markdown(f"""
        <div style="margin-top:1.2em;font-size:1.07em;">
            <b>{_('sensor')} 1:</b> <span style="color:#43cea2">{_('status_ok')}</span><br>
            <b>{_('sensor')} 2:</b> <span style="color:#fa709a">{_('status_warn')}</span><br>
            <b>{_('sensor')} 3:</b> <span style="color:#fee140">{_('status_fault')}</span>
        </div>""", unsafe_allow_html=True)

        # Animated line chart for sensor trend
        st.markdown("<b>Sensor 1 readings (last 24h):</b>", unsafe_allow_html=True)
        ts = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="h")
        y = np.cumsum(np.random.randn(24)) + 70
        df = pd.DataFrame({"Time": ts, "Temperature (°C)": y})
        fig = px.line(df, x="Time", y="Temperature (°C)", title="Sensor 1 Temperature (Animated)", markers=True)
        fig.update_layout(transition=dict(duration=500), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lottie_data = load_lottieurl(ai_lottie)
        if lottie_data:
            st_lottie(lottie_data, height=200, key="ai-lottie", loop=True)
        st.markdown(f"<div style='font-size:1.17em;color:#fa709a;font-weight:bold;'>{_('ai_think')}</div>", unsafe_allow_html=True)
        rain(emoji="💡", font_size=24, falling_speed=4, animation_length="medium")
        st.info(_("about_text"))

# ========== PREDICTIVE ANALYTICS ==========
if nav == _("predictive"):
    st.subheader(_("predictive"))
    lottie_data = load_lottieurl(ai_lottie)
    if lottie_data:
        st_lottie(lottie_data, height=150, key="ai-lottie2", loop=True)
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

    # Animated line chart for predicted sensor fault probability
    st.markdown("<b>Predicted Sensor Fault Probability (Animated):</b>", unsafe_allow_html=True)
    ts = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="h")
    pred = np.clip(np.sin(np.linspace(0, 3*np.pi, 30)) + np.random.rand(30)*0.3, 0, 1)
    df_pred = pd.DataFrame({"Time": ts, "Probability": pred})
    fig_pred = px.line(df_pred, x="Time", y="Probability", title="Fault Probability Over Time", markers=True, range_y=[0,1])
    fig_pred.update_traces(line=dict(color="#fa709a", width=4))
    fig_pred.update_layout(transition=dict(duration=500), showlegend=False)
    st.plotly_chart(fig_pred, use_container_width=True)

    # --- ML DEMO (new - realistic)
    st.markdown("<b>ML Model: Predictive Risk Classifier (Demo)</b>")
    from sklearn.linear_model import LogisticRegression
    X = np.array([[1,70],[2,80],[3,90],[4,110],[5,60],[2,100],[3,75]])
    y = [0,0,1,1,0,1,0]
    model = LogisticRegression()
    model.fit(X, y)
    time = st.slider("Hour", 1, 5, 2)
    value = st.slider("Sensor value", 60, 120, 80)
    pred = model.predict([[time, value]])
    st.success("AI Prediction: " + ("FAULT RISK" if pred[0] else "SAFE"))

    # What-if Simulator
    st.markdown("---")
    st.subheader(_("whatif"))
    val = st.slider(_("drag_label"), min_value=0, max_value=100, value=30, step=5, key="simu_slider")
    ai_risk = min(val/100, 1.0)
    manual_risk = min((val+35)/100, 1.0)
    st.progress(ai_risk, text="AI Risk Level")
    st.progress(manual_risk, text="Manual Risk Level")
    st.markdown(f"<b>{_('cost_savings')}:</b> <span style='color:#43cea2;font-weight:bold;'>${(manual_risk-ai_risk)*8000:,.0f}</span>", unsafe_allow_html=True)
    if ai_risk > 0.75:
        st.warning("Danger Zone! Immediate AI intervention." if st.session_state["lang"] == "en" else "منطقة خطرة! تدخل الذكاء الاصطناعي فوراً.")
        rain(emoji="🔥", font_size=24, falling_speed=6, animation_length="short")
    elif ai_risk > 0.4:
        st.info("Warning: Elevated risk detected." if st.session_state["lang"] == "en" else "تحذير: مستوى الخطر مرتفع.")
    else:
        st.success("Safe operation." if st.session_state["lang"] == "en" else "تشغيل آمن.")

# ========== SMART SOLUTIONS ==========
if nav == _("solutions"):
    st.markdown(f"<h2>{_('solutions')}</h2>", unsafe_allow_html=True)
    if "solutions" not in st.session_state or st.session_state["lang"] != st.session_state.get("solutions_lang", ""):
        st.session_state["solutions"] = []
        st.session_state["solutions_lang"] = st.session_state["lang"]

    if st.button(_( "generate" )):
        st.session_state["solutions"] = random.sample(solutions_data[st.session_state["lang"]], k=random.randint(3,5))
        rain(emoji="✨", font_size=18, falling_speed=6, animation_length="short")

    if not st.session_state["solutions"]:
        st.info(_( "no_solutions" ))
    else:
        for i, s in enumerate(st.session_state["solutions"]):
            eff_color = "#43cea2" if s["priority"] in ["High", "عالية"] else "#fa709a" if s["priority"] in ["Medium", "متوسطة"] else "#fee140"
            st.markdown(
                f"""
                <div class="solution-card">
                    <div class="solution-icon">{s['icon']}</div>
                    <div style="font-size:1.15em;font-weight:bold;margin-bottom:0.2em">{_('solution_title')} {i+1}: {s['title']}</div>
                    <div style="margin-bottom:0.4em">{_('solution_desc')}: {s['desc']}</div>
                    <div class="gradient-bar" style="width:{s['eff']}%"></div>
                    <span style="font-size:.98em"><b>{_('solution_eff')}:</b> {s['eff']}%</span> |
                    <span style="font-size:.98em"><b>{_('solution_priority')}:</b> <span style="color:{eff_color};font-weight:bold">{s['priority']}</span></span> |
                    <span style="font-size:.98em"><b>{_('solution_time')}:</b> {s['time']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Pie chart: Solution priorities
        st.markdown("<b>Solution Priority Distribution:</b>", unsafe_allow_html=True)
        sdata = st.session_state["solutions"]
        priorities = [s["priority"] for s in sdata]
        if st.session_state["lang"] == "en":
            labels = ["High", "Medium", "Low"]
        else:
            labels = ["عالية", "متوسطة", "منخفضة"]
        counts = [priorities.count(lab) for lab in labels]
        figpie = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=0.4)])
        figpie.update_traces(marker=dict(colors=["#43cea2", "#fa709a", "#fee140"]))
        figpie.update_layout(showlegend=True)
        st.plotly_chart(figpie, use_container_width=True)

# ========== REALISTIC SCENARIO (NEW) ==========
if nav == _("scenario"):
    st.subheader(_( "scenario_title" ))
    st.markdown(f"<div style='font-size:1.08em;color:#185a9d;font-weight:bold;'>{_('scenario_intro')}</div>", unsafe_allow_html=True)
    lottie_data = load_lottieurl(alert_lottie)
    if lottie_data:
        st_lottie(lottie_data, height=130, key="scenario-lottie", loop=True)
    if "scenario_state" not in st.session_state:
        st.session_state["scenario_state"] = 0

    def reset_scenario():
        st.session_state["scenario_state"] = 0

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

# ========== ALERTS ==========
if nav == _("alerts"):
    st.subheader(_("alerts"))
    lottie_data = load_lottieurl(alert_lottie)
    if lottie_data:
        st_lottie(lottie_data, height=120, key="alert-lottie", loop=True)
    st.warning("No active alerts. All systems stable. ✅" if st.session_state["lang"] == "en" else "لا توجد تنبيهات حالية. كل الأنظمة مستقرة. ✅")

    # Pie chart: Alert types (dummy, for illustration)
    st.markdown("<b>Alert Types Breakdown:</b>", unsafe_allow_html=True)
    labels = ["Sensor Fault", "Leak", "Power", "Other"] if st.session_state["lang"] == "en" else ["خلل مستشعر", "تسرب", "كهرباء", "أخرى"]
    values = [2, 1, 0, 1]
    fig_alert = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig_alert.update_traces(marker=dict(colors=["#fa709a", "#fee140", "#43cea2", "#185a9d"]))
    fig_alert.update_layout(showlegend=True)
    st.plotly_chart(fig_alert, use_container_width=True)

# ========== COST & SAVINGS ==========
if nav == _("cost"):
    st.subheader(_("cost"))
    st.markdown("""
    <div style="font-size:1.1em;">
    <b>AI Savings This Month:</b> <span style="color:#43cea2;font-weight:bold;">$13,500</span>  
    <br>
    <b>Predicted Loss (Manual):</b> <span style="color:#fa709a;font-weight:bold;">$22,100</span>  
    <br>
    <b>Downtime Reduction:</b> <span style="color:#fee140;font-weight:bold;">71%</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(0.71, text="Downtime Reduced")

    # Pie chart: Cost breakdown
    st.markdown("<b>Cost Breakdown:</b>", unsafe_allow_html=True)
    cost_labels = ["Maintenance", "Downtime", "Energy", "Other"] if st.session_state["lang"] == "en" else ["صيانة", "توقف", "طاقة", "أخرى"]
    cost_vals = [5000, 6000, 3000, 1500]
    fig_cost = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_vals, hole=0.5)])
    fig_cost.update_traces(marker=dict(colors=["#43cea2", "#fa709a", "#fee140", "#185a9d"]))
    fig_cost.update_layout(showlegend=True)
    st.plotly_chart(fig_cost, use_container_width=True)

    # Animated line chart: AI vs manual cost over time
    st.markdown("<b>Costs Over Time (AI vs Manual):</b>", unsafe_allow_html=True)
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

# ========== AI VS MANUAL ==========
if nav == _("ai_vs_manual"):
    st.subheader(_("ai_vs_manual"))
    st.markdown("### " + _("ai_vs_manual"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<b>{_('ai_reaction')}</b>", unsafe_allow_html=True)
        lottie_data = load_lottieurl(ai_lottie)
        if lottie_data:
            st_lottie(lottie_data, height=90, key="ai-vs-lottie", loop=True)
        st.success("AI detects & responds in 1.2s")
    with col2:
        st.markdown(f"<b>{_('manual_reaction')}</b>", unsafe_allow_html=True)
        st.error("Manual detection: 18 min average")

    # Radar chart: AI vs Manual on metrics
    st.markdown("<b>Performance Comparison:</b>", unsafe_allow_html=True)
    metrics = ["Speed", "Accuracy", "Cost", "Downtime", "Safety"] if st.session_state["lang"] == "en" else ["السرعة", "الدقة", "التكلفة", "التوقف", "السلامة"]
    ai_vals = [95, 92, 90, 85, 98]
    man_vals = [60, 70, 70, 60, 75]
    radar_df = pd.DataFrame({
        "Metric": metrics*2,
        "Value": ai_vals + man_vals,
        "Type": (["AI"]*5)+(["Manual"]*5)
    })
    fig_radar = px.line_polar(radar_df, r="Value", theta="Metric", color="Type", line_close=True, template="plotly_dark",
                              color_discrete_map={"AI": "#43cea2", "Manual": "#fa709a"})
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

# ========== DIGITAL TWIN LIVE ==========
if nav == _("live_dt"):
    st.markdown(f"<h2>{_('live_dt')}</h2>", unsafe_allow_html=True)
    lottie_data = load_lottieurl(twin_lottie)
    if lottie_data:
        st_lottie(lottie_data, height=180, key="livedt-lottie", loop=True)
    st.markdown("""
    <div style="font-size:1.03em;">
    <b>Plant Diagram (AI Overlay):</b>
    <br>
    <img src="https://i.ibb.co/4R0vY4Q/plant-twin-demo.png" style="width:80%;border-radius:14px;box-shadow:0 2px 12px #43cea277;"/>
    <br>
    <b>Sensors (Live):</b>
    <ul>
        <li>🌡️ Temp: <b style='color:#43cea2'>72°C</b></li>
        <li>🧪 Gas: <b style='color:#fa709a'>7.3%</b> <span style='color:#fa709a;font-weight:bold'>[Warning]</span></li>
        <li>💧 Pressure: <b style='color:#fee140'>3.1 bar</b></li>
        <li>🔋 Power: <b style='color:#43cea2'>Stable</b></li>
    </ul>
    <b>Fault Propagation:</b> <span style="color:#fa709a;font-weight:bold;">None</span>
    </div>
    """, unsafe_allow_html=True)
    lottie_data = load_lottieurl(ai_lottie)
    if lottie_data:
        st_lottie(lottie_data, height=140, key="neural-overlay-lottie", loop=True)

    # 3D plot: Plant sensor surface (demo)
    st.markdown("<b>Sensor Heat Map (3D):</b>", unsafe_allow_html=True)
    x, y = np.meshgrid(np.linspace(0, 10, 12), np.linspace(0, 10, 12))
    z = np.sin(x) * np.cos(y) * 10 + 72 + np.random.randn(*x.shape)
    fig3d = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig3d.update_layout(title="Plant Temperature Surface", autosize=True, margin=dict(l=20, r=20, b=20, t=30))
    st.plotly_chart(fig3d, use_container_width=True)

# ========== ROADMAP & CHANGELOG ==========
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

# ========== ABOUT ==========
if nav == _("about"):
    st.subheader(_("story_title"))
    st.markdown(f"<span style='font-size:1.1em;'>{_('story')}</span>", unsafe_allow_html=True)

    st.markdown(f"## { _('features_title') }")
    st.markdown(
        "\n".join([f"- {f}" for f in translations[st.session_state['lang']]['features']])
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
