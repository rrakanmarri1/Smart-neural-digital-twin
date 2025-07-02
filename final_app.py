import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import openai
from twilio.rest import Client
from datetime import datetime, timedelta
import threading
import paho.mqtt.client as mqtt
import os

# ----- LOGO SVG -----
logo_svg = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none">
  <circle cx="32" cy="32" r="32" fill="url(#grad1)"/>
  <defs>
    <linearGradient id="grad1" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
      <stop stop-color="#43cea2"/>
      <stop offset="1" stop-color="#185a9d"/>
    </linearGradient>
  </defs>
  <g>
    <ellipse cx="32" cy="32" rx="22" ry="10" fill="#fff" fill-opacity="0.18"/>
    <ellipse cx="32" cy="32" rx="12" ry="22" fill="#fff" fill-opacity="0.10"/>
    <path d="M20 32a12 12 0 1 0 24 0 12 12 0 1 0 -24 0" fill="#fff" fill-opacity="0.7"/>
    <path d="M32 16v32M16 32h32" stroke="#185a9d" stroke-width="2" stroke-linecap="round"/>
    <circle cx="32" cy="32" r="6" fill="#43cea2" stroke="#185a9d" stroke-width="2"/>
  </g>
</svg>
"""

# MQTT Config
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "digitaltwin/test/temperature"

# Secure config via environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
TWILIO_TO = os.environ.get("TWILIO_TO")

# App state initialization
for key, default in [
    ("lang", "en"), ("scenario_step", 0), ("solution_idx", 0), ("theme", "dark"),
    ("mqtt_temp", None), ("mqtt_last", None), ("mqtt_started", False), ("sms_sent", False),
    ("feedback_list", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# MQTT background thread
def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)
def on_message(client, userdata, msg):
    try:
        val = float(msg.payload.decode())
        st.session_state["mqtt_temp"] = val
        st.session_state["mqtt_last"] = datetime.now()
    except Exception:
        pass
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception:
        pass
if not st.session_state["mqtt_started"]:
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    st.session_state["mqtt_started"] = True

# OpenAI setup
openai.api_key = OPENAI_API_KEY
def ask_llm(prompt, lang):
    system_en = """You are an expert AI assistant for an industrial digital twin platform called 'Smart Neural Digital Twin'.
This platform monitors various plant parameters in real-time, including temperature, pressure, and methane levels.
It provides advanced dashboards, predictive analytics (forecasting methane and temperature for the next 7 days),
scenario playback, alerts and fault logs, smart solutions for issues like methane leaks or pump failures,
KPIs, plant heatmaps, root cause analysis, and incident timelines.

When answering questions, prioritize information related to the 'Smart Neural Digital Twin' project and its data.
If asked about current sensor readings (like temperature, pressure, methane), or recent alerts, or daily metrics for vibration or levels, or future predictions for the next few hours/days, provide answers based on the context of the project's capabilities.
For example, if asked 'What is the current temperature?', you can state that the platform monitors live temperature via MQTT.
If asked about predictions, mention the 7-day forecast capability.

If the question is general and not directly related to the project, answer it to the best of your general knowledge.
"""
    system_ar = """أنت مساعد ذكاء صناعي خبير لمنصة التوأم الرقمي الصناعي المسماة 'التوأم الرقمي العصبي الذكي'.
تراقب هذه المنصة معلمات المصنع المختلفة في الوقت الفعلي، بما في ذلك درجة الحرارة والضغط ومستويات الميثان.
توفر لوحات تحكم متقدمة، وتحليلات تنبؤية (تتنبأ بالميثان ودرجة الحرارة للأيام السبعة القادمة)،
وإعادة تشغيل السيناريوهات، وسجلات التنبيهات والأعطال، وحلول ذكية لمشكلات مثل تسرب الميثان أو أعطال المضخات،
ومؤشرات الأداء الرئيسية (KPIs)، وخرائط حرارة المصنع، وتحليل السبب الجذري، والجداول الزمنية للحوادث.

عند الإجابة على الأسئلة، أعط الأولوية للمعلومات المتعلقة بمشروع 'التوأم الرقمي العصبي الذكي' وبياناته.
إذا سُئلت عن قراءات المستشعرات الحالية (مثل درجة الحرارة، الضغط، الميثان)، أو التنبيهات الأخيرة، أو المقاييس اليومية للاهتزاز أو المستويات، أو التوقعات المستقبلية للساعات/الأيام القادمة، قدم إجابات بناءً على سياق قدرات المشروع.
على سبيل المثال، إذا سُئلت 'كم درجة الحرارة الحالية؟'، يمكنك الإشارة إلى أن المنصة تراقب درجة الحرارة الحية عبر MQTT.
إذا سُئلت عن التوقعات، اذكر قدرة التنبؤ لمدة 7 أيام.

إذا كان السؤال عامًا ولا يتعلق مباشرة بالمشروع، أجب عليه بناءً على معرفتك العامة.
"""
    system = system_en if lang == "en" else system_ar
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":prompt}
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return "LLM Error: "+str(e)

# Twilio SMS
def send_sms(to, message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        message = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=to
        )
        return True, "Sent."
    except Exception as e:
        return False, str(e)

# Helper functions
def to_arabic_numerals(num):
    return str(num).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
def rtl_wrap(txt):
    if st.session_state["lang"] == "ar":
        return f'<div style="direction:rtl;text-align:right">{txt}</div>'
    else:
        return f'<div style="direction:ltr;text-align:left">{txt}</div>'
def show_logo():
    st.markdown(f'<div style="text-align:center;padding-bottom:1.2em;">{logo_svg}</div>', unsafe_allow_html=True)

# Translations
texts = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "app_sub": "Intelligent Digital Plant Platform",
        "side_sections": [
            "Digital Twin", "Advanced Dashboard", "Predictive Analytics", "Scenario Playback",
            "Alerts & Fault Log", "Smart Solutions", "KPI Wall", "Plant Heatmap", "Root Cause Explorer", "AI Copilot Chat",
            "Live Plant 3D", "Incident Timeline", "Energy Optimization", "Future Insights", "Operator Feedback", "About"
        ],
        "lang_en": "English",
        "lang_ar": "Arabic",
        "solution_btn": "Next Solution",
        "logo_alt": "Smart Neural Digital Twin Logo",
        "about_header": "Our Story",
        "about_story": """Our journey began with a simple question: How can we detect gas leaks before they become disasters?\nWe tried every solution, even innovated with drones, and it worked. But we stopped and asked: Why wait for the problem at all?\nOur dream was to build a smart device that predicts danger before it happens. It wasn’t impossible, just difficult. But we made the difficult easy with the smart neural digital twin that connects AI and plant data.\nToday, our platform is the first line of defense, standing apart from any traditional system because it predicts problems hours before they happen. Even days!\nThis is the future of industrial safety… and this is our project.""",
        "about_colorful": [
            ("#43cea2", "AI at the Core"),
            ("#fa709a", "Real-time Sensing"),
            ("#ffb347", "Predictive Analytics"),
            ("#8fd3f4", "Instant Actions"),
            ("#185a9d", "Peace of Mind"),
        ],
        "features": [
            "Interactive plant schematic & overlays",
            "Advanced dashboards & KPIs",
            "AI-driven fault detection & smart solutions",
            "Root-cause explorer & scenario playback",
            "Live 3D plant visualization",
            "Bilingual support & peak design"
        ],
        "howto_extend": [
            "Connect to real plant historian data",
            "Add custom plant schematics & overlays",
            "Integrate with control and alerting systems",
            "Deploy on secure internal networks"
        ],
        "developers": [
            ("Rakan Almarri", "rakan.almarri.2@aramco.com", "0532559664"),
            ("Abdulrahman Alzahrani", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "Contact Info",
        "demo_note": "Demo use only: Not for live plant operation",
        "live3d_header": "Live Plant 3D",
        "live3d_intro": "Explore the interactive 3D model below. Use your mouse to zoom, rotate, and explore the plant!",
        "live3d_404": "The 3D model failed to load. View the static 3D plant image below.",
        "static_3d_caption": "Sample Plant 3D Visual",
        "solutions": [
            {
                "title": "Automated Methane Leak Response",
                "desc": "Integrate advanced sensors with automated shutdown logic to instantly contain future methane leaks.",
                "steps": ["Deploy new IoT sensors", "Implement AI detection", "Link to emergency shutdown", "Train operators"],
                "priority": "High", "effectiveness": "High", "time": "3 days", "cost": "$4,000", "savings": "$25,000/year",
                "icon": "🛡️"
            },
            {
                "title": "Pump Predictive Maintenance",
                "desc": "Monitor vibration and temperature to predict pump failures before they occur.",
                "steps": ["Install vibration sensors", "Run ML models", "Alert on anomaly", "Schedule just-in-time maintenance"],
                "priority": "Medium", "effectiveness": "High", "time": "1 week", "cost": "$5,000", "savings": "$18,000/year",
                "icon": "🔧"
            },
            {
                "title": "Energy Use Optimization",
                "desc": "AI analyzes compressor schedule to cut energy waste by 11%.",
                "steps": ["Analyze compressor cycles", "Optimize schedule", "Implement load shifting", "Track savings"],
                "priority": "High", "effectiveness": "Medium", "time": "2 weeks", "cost": "$6,000", "savings": "$32,000/year",
                "icon": "⚡"
            },
        ]
    },
    "ar": {
        "app_title": "التوأم الرقمي العصبي الذكي",
        "app_sub": "منصة المصنع الذكي الرقمي",
        "side_sections": [
            "التوأم الرقمي", "لوحة القيادة المتقدمة", "التحليلات التنبؤية", "تشغيل السيناريو",
            "التنبيهات وسجل الأعطال", "الحلول الذكية", "جدار المؤشرات", "خريطة حرارة المصنع", "مستكشف السبب الجذري",
            "محادثة الذكاء الصناعي", "مصنع ثلاثي الأبعاد", "جدول الحوادث", "تحسين الطاقة", "رؤى مستقبلية", "ملاحظات المشغل", "حول"
        ],
        "lang_en": "الإنجليزية",
        "lang_ar": "العربية",
        "solution_btn": "الحل التالي",
        "logo_alt": "شعار التوأم الرقمي العصبي الذكي",
        "about_header": "قصتنا",
        "about_story": """بدأنا رحلتنا من سؤال بسيط: كيف نكشف تسرب الغاز قبل أن يتحول إلى كارثة؟ جربنا كل الحلول، وابتكرنا حتى باستخدام الطائرات بدون طيار ونجحنا. لكن وقفنا وسألنا: لماذا ننتظر المشكلة أصلاً؟\nحلمنا كان بناء جهاز يتوقع الخطر قبل حدوثه. لم يكن مستحيلاً، لكنه كان صعبًا. جعلنا الصعب سهلاً مع التوأم الرقمي العصبي الذكي الذي يربط الذكاء الاصطناعي ببيانات المصنع.\nاليوم، منصتنا هي خط الدفاع الأول، وتختلف عن أي نظام تقليدي لأنها تتوقع المشكلة بساعات قبل وقوعها، وأحيانًا بأيام!\nهذا هو مستقبل الأمان الصناعي... وهذا هو مشروعنا.""",
        "about_colorful": [
            ("#43cea2", "الذكاء الاصطناعي في القلب"),
            ("#fa709a", "استشعار لحظي"),
            ("#ffb347", "تحليلات تنبؤية"),
            ("#8fd3f4", "إجراءات فورية"),
            ("#185a9d", "راحة البال"),
        ],
        "features": [
            "مخطط مصنع تفاعلي وتراكب مباشر",
            "لوحات ومؤشرات متقدمة",
            "كشف أعطال ذكي وحلول فورية",
            "مستكشف السبب الجذري وتشغيل السيناريوهات",
            "رؤية ثلاثية الأبعاد للمصنع",
            "دعم لغتين وتصميم عصري"
        ],
        "howto_extend": [
            "ربط مع بيانات المصنع الحقيقية",
            "إضافة مخططات وتراكب مخصص",
            "دمج مع أنظمة التحكم والتنبيه",
            "تشغيل داخلي آمن"
        ],
        "developers": [
            ("راكان المعاري", "rakan.almarri.2@aramco.com", "0532559664"),
            ("عبدالرحمن الزهراني", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "معلومات التواصل",
        "demo_note": "للعرض فقط: غير مخصص للتشغيل الفعلي",
        "live3d_header": "مصنع ثلاثي الأبعاد مباشر",
        "live3d_intro": "تفاعل مع النموذج الثلاثي الأبعاد أدناه. استخدم الماوس للتحريك والتكبير.",
        "live3d_404": "تعذر تحميل النموذج، شاهد صورة المصنع الثلاثي الأبعاد بالأسفل.",
        "static_3d_caption": "مشهد ثلاثي الأبعاد لمصنع صناعي",
        "solutions": [
            {
                "title": "استجابة آلية لتسرب الميثان",
                "desc": "دمج حساسات متطورة مع منطق إيقاف تلقائي لاحتواء التسربات فوراً.",
                "steps": ["تركيب حساسات إنترنت الأشياء", "تفعيل كشف الذكاء الاصطناعي", "ربط بالإيقاف الطارئ", "تدريب المشغلين"],
                "priority": "عالية", "effectiveness": "عالية", "time": "٣ أيام", "cost": "$٤٬٠٠٠", "savings": "$٢٥٬٠٠٠/سنة",
                "icon": "🛡️"
            },
            {
                "title": "صيانة استباقية للمضخات",
                "desc": "مراقبة الاهتزازات والحرارة للتنبؤ بالأعطال قبل وقوعها.",
                "steps": ["تركيب حساسات الاهتزاز", "تشغيل نماذج التعلم الآلي", "تنبيه عند وجود شذوذ", "جدولة صيانة فورية"],
                "priority": "متوسطة", "effectiveness": "عالية", "time": "أسبوع", "cost": "$٥٬٠٠٠", "savings": "$١٨٬٠٠٠/سنة",
                "icon": "🔧"
            },
            {
                "title": "تحسين استهلاك الطاقة",
                "desc": "تحلل الذكاء الاصطناعي جدول الضواغط لخفض الهدر بنسبة ١١٪.",
                "steps": ["تحليل دورات الضواغط", "تحسين الجدولة", "تطبيق نقل الأحمال", "متابعة التوفير"],
                "priority": "عالية", "effectiveness": "متوسطة", "time": "أسبوعان", "cost": "$٦٬٠٠٠", "savings": "$٣٢٬٠٠٠/سنة",
                "icon": "⚡"
            },
        ]
    }
}

# ----- THEME & CSS -----
if st.sidebar.button("🌗 Theme", key="themebtn"):
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
if st.session_state["theme"] == "dark":
    st.markdown("""
    <style>
    html, body, [class*="css"]  { background: #232526 !important; color:#fff !important;}
    </style>
    """,unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    html, body, [class*="css"]  { background: #f3f8fc !important; color:#232526 !important;}
    </style>
    """,unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url(\'https://fonts.googleapis.com/css2?family=Cairo:wght@700&family=Montserrat:wght@700&display=swap\');
    .peak-card {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,.15);
        margin-bottom: 1.5em;
        padding: 1.5em 2em;
        transition: box-shadow 0.2s;
    }
    .peak-card:hover {
        box-shadow: 0 12px 38px 0 rgba(31,38,135,.28);
    }
    .kpi-card {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        border-radius: 13px;
        color: #fff !important;
        font-size: 1.25em;
        font-weight: 700;
        box-shadow: 0 8px 24px 0 rgba(31,38,135,.10);
        padding: 1.3em 1.3em;
        text-align: center;
        margin-bottom: 1em;
    }
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: \'Cairo\', sans-serif !important;
    }
    .ltr {
        direction: ltr;
        text-align: left;
        font-family: \'Montserrat\', sans-serif !important;
    }
    .sidebar-title {
        font-size: 2em !important;
        font-weight: 900 !important;
        color: #43cea2 !important;
        letter-spacing: 0.5px;
        margin-bottom: 0.2em !important;
    }
    .sidebar-subtitle {
        font-size: 1.15em !important;
        color: #cfdef3 !important;
        margin-bottom: 1em;
        margin-top: -.7em !important;
    }
    .gradient-header, .gradient-ar {
        font-weight: 900;
        font-size: 2.1em;
        background: linear-gradient(90deg,#43cea2,#185a9d 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3em;
    }
    .timeline-step {
        border-left: 4px solid #43cea2;
        margin-left: 0.8em;
        padding-left: 1.2em;
        margin-bottom: 1em;
        position: relative;
    }
    .timeline-step:before {
        content: \'\';
        position: absolute;
        left: -14px;
        top: 0.18em;
        width: 18px;
        height: 18px;
        background: #43cea2;
        border-radius: 100%;
        border: 2px solid #fff;
    }
    .timeline-icon {
        font-size: 1.5em;
        margin-right: 0.5em;
        vertical-align: middle;
    }
    .about-bgcard {
        background: linear-gradient(120deg,#185a9d10,#43cea210 98%);
        border-radius: 22px;
        padding: 2.2em 2.1em 1.8em 2.1em;
        margin-top: 1.6em;
        margin-bottom: 2.2em;
        box-shadow: 0 7px 32px 0 rgba(31,38,135,.07);
        position: relative;
    }
    .about-story {
        font-size: 1.18em;
        font-weight: 600;
        margin-bottom: 2em;
        background: linear-gradient(90deg,#e0eafc,#8fd3f4 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.2em 0.1em;
    }
    .about-feature {
        font-weight: 700;
        font-size: 1.16em;
        margin: .45em 0 .14em 0;
    }
    .about-color {
        font-weight: 900;
        font-size: 1.20em;
        margin-bottom: .45em;
        display: inline-block;
        padding: .18em .9em;
        border-radius: 12px;
        margin-right: .9em;
    }
    .about-contact {
        font-size: 1.13em;
        margin-top: 1.9em;
        margin-bottom: .6em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(
        f"""<div class="sidebar-title">{texts[st.session_state["lang"]]["app_title"]}</div>
            <div class="sidebar-subtitle">{texts[st.session_state["lang"]]["app_sub"]}</div>""", unsafe_allow_html=True)
    lang_sel = st.radio(
        "", (texts["en"]["lang_en"], texts["en"]["lang_ar"]) if st.session_state["lang"] == "en" else (texts["ar"]["lang_en"], texts["ar"]["lang_ar"]),
        horizontal=True, index=0 if st.session_state["lang"] == "en" else 1
    )
    st.session_state["lang"] = "en" if lang_sel == texts["en"]["lang_en"] else "ar"
    section_list = texts[st.session_state["lang"]]["side_sections"]
    section = st.radio(" ", section_list, index=0)

lang = st.session_state["lang"]
T = texts[lang]
rtl = True if lang == "ar" else False

# Demo data
np.random.seed(1)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})

# ========== MAIN SECTIONS ==========

if section == T["side_sections"][0]:  # Digital Twin (Live MQTT)
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][0]}</div>', unsafe_allow_html=True)
    try:
        st.image("realtime_streaming.png", caption=rtl_wrap("MQTT Real-Time Streaming Example" if lang=="en" else "مثال مشاركة البيانات الحية"))
    except Exception:
        st.image("https://cdn.pixabay.com/photo/2016/11/29/10/07/architecture-1868667_1280.jpg", caption=rtl_wrap("Demo Image"))
    st.markdown(rtl_wrap("Live Temperature (MQTT, topic: digitaltwin/test/temperature)" if lang=="en" else "قراءة درجة الحرارة الحية (MQTT)"))
    temp = st.session_state["mqtt_temp"]
    if temp is not None:
        display_temp = to_arabic_numerals(round(temp,2)) if lang == "ar" else round(temp,2)
        st.metric(T["features"][0], f"{display_temp} °C", delta=None)
        # Trigger alert if temp > 60°C and send SMS
        if temp > 60 and not st.session_state["sms_sent"]:
            ok, msg = send_sms(TWILIO_TO, (f"ALERT: Plant temperature exceeded safe level! Temp={temp:.1f}°C" if lang=="en" else f"تنبيه: درجة حرارة المصنع تجاوزت الحد المسموح! درجة الحرارة={to_arabic_numerals(round(temp,1))}°م"))
            st.session_state["sms_sent"] = True
            st.warning("⚠️ SMS Alert sent to supervisor!" if lang=="en" else "⚠️ تم إرسال تنبيه SMS للمشرف!")
    else:
        st.info("Waiting for MQTT..." if lang=="en" else "في انتظار بيانات MQTT...")
    st.caption(f"{'Last update' if lang=='en' else 'آخر تحديث'}: {st.session_state['mqtt_last'] if st.session_state['mqtt_last'] else 'N/A'}")

elif section == T["side_sections"][1]:  # Advanced Dashboard
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][1]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("KPIs and live trends for the plant." if lang=="en" else "المؤشرات والاتجاهات الحية للمصنع."))
    fig = px.line(demo_df, x="time", y=["Temperature", "Pressure", "Methane"], labels={"value":"Reading", "variable":"Tag"})
    fig.update_layout(legend_title_text="Tag", height=350)
    st.plotly_chart(fig, use_container_width=True)

elif section == T["side_sections"][2]:  # Predictive Analytics
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][2]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("Forecast of methane and temperature for next 7 days." if lang=="en" else "توقع الميثان ودرجة الحرارة للأيام السبعة القادمة."))
    days = pd.date_range(datetime.now(), periods=7)
    forecast = pd.DataFrame({
        "Day": days,
        "Methane": np.linspace(1.2, 4.5, 7) + np.random.normal(0, 0.2, 7),
        "Temp": np.linspace(55, 63, 7) + np.random.normal(0, 1, 7)
    })
    st.line_chart(forecast.set_index("Day"))

elif section == T["side_sections"][3]:  # Scenario Playback
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][3]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("Replay plant incident scenarios hour by hour." if lang=="en" else "تشغيل سيناريوهات الحوادث ساعة بساعة."))
    step = st.slider(T["side_sections"][3], 0, 23, 0)
    st.markdown(rtl_wrap(f"Scenario at hour {step}" if lang=="en" else f"السيناريو عند الساعة {to_arabic_numerals(step)}"))
    chart_data = np.cumsum(np.random.randn(24)) + 50
    st.line_chart(chart_data[:step+1])

elif section == T["side_sections"][4]:  # Alerts & Fault Log
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][4]}</div>', unsafe_allow_html=True)
    alert_log = pd.DataFrame([
        {"Time":"2025-07-01 05:00","Type":("High Temp" if lang=="en" else "حرارة عالية"),"Status":("Open" if lang=="en" else "مفتوح")},
        {"Time":"2025-07-01 03:32","Type":("Methane Spike" if lang=="en" else "ارتفاع الميثان"),"Status":("Closed" if lang=="en" else "مغلق")},
        {"Time":"2025-06-30 22:10","Type":("Low Flow" if lang=="en" else "تدفق منخفض"),"Status":("Closed" if lang=="en" else "مغلق")},
    ])
    st.table(alert_log)

elif section == T["side_sections"][5]:  # Smart Solutions
    show_logo()
    idx = st.session_state["solution_idx"]
    solutions = T["solutions"]
    sol = solutions[idx]
    steps_html = "".join([f"<li>{s}</li>" for s in sol["steps"]])
    st.markdown(f"""
    <div class="peak-card">
        <div style="font-size:2em;">{sol["icon"]}</div>
        <b style="font-size:1.3em">{sol["title"]}</b>
        <div style="margin:0.8em 0 0.5em 0;">{sol["desc"]}</div>
        <ul style="margin-bottom:0.7em;">{steps_html}</ul>
        <div style="display:flex;gap:0.9em;flex-wrap:wrap;">
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{(\'Priority\' if lang==\'en\' else \'الأولوية\')}: {sol[\'priority\]}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{(\'Effectiveness\' if lang==\'en\' else \'الفعالية\')}: {sol[\'effectiveness\]}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{(\'Time\' if lang==\'en\' else \'المدة\')}: {sol[\'time\]}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{(\'Cost\' if lang==\'en\' else \'التكلفة\')}: {sol[\'cost\]}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{(\'Savings\' if lang==\'en\' else \'التوفير\')}: {sol[\'savings\]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(T["solution_btn"], key=f"solution-next-{idx}"):
        st.session_state["solution_idx"] = (idx + 1) % len(solutions)

elif section == T["side_sections"][6]:  # KPI Wall
    show_logo()
    kpis = [
        ("Overall Efficiency", "✅", "#8fd3f4"),
        ("Energy Used (MWh)", "⚡", "#fe8c00"),
        ("Water Saved (m³)", "💧", "#43cea2"),
        ("Incidents This Year", "🛑", "#fa709a")
    ] if lang == "en" else [
        ("الكفاءة العامة", "✅", "#8fd3f4"),
        ("الطاقة المستخدمة (ميغاواط)", "⚡", "#fe8c00"),
        ("الماء الموفر (م³)", "💧", "#43cea2"),
        ("الحوادث هذا العام", "🛑", "#fa709a")
    ]
    vals = [96, 272, 62, 1]
    goals = [98, 250, 70, 0]
    st.markdown("<div style=\'display:flex;gap:1.3em;flex-wrap:wrap;\'>", unsafe_allow_html=True)
    for i, (name, icon, color) in enumerate(kpis):
        display_val = to_arabic_numerals(vals[i]) if lang == "ar" else str(vals[i])
        display_goal = to_arabic_numerals(goals[i]) if lang == "ar" else str(goals[i])
        st.markdown(f"""<div class="kpi-card" style="background:{color}c0;">
            <span style="font-size:2.1em;">{icon}</span><br>
            <b>{name}</b><br>
            <span style="font-size:2.3em;font-weight:900">{display_val}</span>
            <div style="font-size:.95em;color:#222;">{(\'Goal\' if lang==\'en\' else \'الهدف\')}: {display_goal}}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][7]:  # Plant Heatmap
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][7]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("High temperature and pressure zones are highlighted below." if lang=="en" else "المناطق الحرجة للحرارة والضغط موضحة أدناه."))
    z = np.random.uniform(25, 70, (8, 10))
    fig = go.Figure(data=go.Heatmap(z=z, colorscale=\'YlOrRd\', colorbar=dict(title=(\'Temp °C\' if lang==\'en\' else \'حرارة\'))))
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

elif section == T["side_sections"][8]:  # Root Cause Explorer
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][8]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("Trace issues to their origin. Sample propagation path shown below." if lang=="en" else "تتبع المشكلات إلى أصلها. سلسلة السبب والنتيجة أدناه."))
    st.markdown("""
    <div style="margin-top:1em;display:flex;justify-content:center;">
    <svg width="340" height="180" viewBox="0 0 340 180">
      <rect x="20" y="70" width="80" height="38" rx="12" fill="#43cea2" opacity="0.89"/>
      <rect x="140" y="30" width="90" height="38" rx="12" fill="#ffb347" opacity="0.91"/>
      <rect x="140" y="110" width="90" height="38" rx="12" fill="#fa709a" opacity="0.91"/>
      <rect x="260" y="70" width="60" height="38" rx="12" fill="#8fd3f4" opacity="0.91"/>
      <text x="60" y="93" font-size="1.2em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="185" y="53" font-size="1.1em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="185" y="133" font-size="1.1em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="290" y="93" font-size="1.1em" fill="#185a9d" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <line x1="100" y1="89" x2="140" y2="49" stroke="#43cea2" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="100" y1="89" x2="140" y2="129" stroke="#43cea2" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="230" y1="49" x2="260" y2="89" stroke="#ffb347" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="230" y1="129" x2="260" y2="89" stroke="#fa709a" stroke-width="3" marker-end="url(#arrow)"/>
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#185a9d" />
        </marker>
      </defs>
    </svg>
    </div>
    """.format(
        "Root: Methane leak" if lang=="en" else "الجذر: تسرب ميثان",
        "Compressor 2 Fault" if lang=="en" else "عطل الضاغط ٢",
        "Pump Overload" if lang=="en" else "تحميل زائد على المضخة",
        "Incident: Shutdown" if lang=="en" else "حادث: إيقاف"
    ), unsafe_allow_html=True)

elif section == T["side_sections"][9]:  # AI Copilot Chat (LLM)
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][9]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("Ask the AI about plant issues, troubleshooting, or improvements." if lang=="en" else "اسأل الذكاء الصناعي عن الأعطال أو التحسينات أو المشاكل."))
    user_prompt = st.text_input(("Ask AI a question..." if lang=="en" else "اكتب سؤالاً للذكاء الصناعي..."), key="ai_input")
    if user_prompt:
        with st.spinner("Thinking..." if lang=="en" else "يفكر..."):
            answer = ask_llm(user_prompt, lang)
        st.markdown(f"**AI:** {answer}")

elif section == T["side_sections"][10]:  # Live Plant 3D
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["live3d_header"]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["live3d_intro"]), unsafe_allow_html=True)
    try:
        st.components.v1.iframe(
            "https://sketchfab.com/models/6ebc4e240be94b8caa1e1e6b0f2e3e7b/embed", height=480, scrolling=True
        )
        st.markdown(
            rtl_wrap(
                \'<sup>3D model courtesy of <a href="https://sketchfab.com" target="_blank">Sketchfab</a></sup>\' if lang=="en"
                else \'<sup>النموذج ثلاثي الأبعاد مقدم من <a href="https://sketchfab.com" target="_blank">Sketchfab</a></sup>\'
            ),
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown(f'<div class="peak-card" style="background:#fa709a22">{T["live3d_404"]}</div>', unsafe_allow_html=True)
    st.image(
        "https://cdn.pixabay.com/photo/2016/11/29/10/07/architecture-1868667_1280.jpg",
        caption=rtl_wrap(T["static_3d_caption"])
    )

elif section == T["side_sections"][11]:  # Incident Timeline
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][11]}</div>', unsafe_allow_html=True)
    timeline_steps = [
        ("2025-06-30 11:23", "🛑", "Methane leak detected at Compressor 2. Emergency shutdown triggered."),
        ("2025-06-30 10:58", "⚠️", "Flow rate anomaly at Pump 1. Operator notified."),
        ("2025-06-30 10:50", "ℹ️", "Temperature rising at Reactor. Trend within safe limits.")
    ] if lang == "en" else [
        ("2025-06-30 11:23", "🛑", "اكتشاف تسرب ميثان عند الضاغط ٢. إيقاف طارئ تلقائي."),
        ("2025-06-30 10:58", "⚠️", "شذوذ تدفق عند مضخة ١. تم إخطار المشغل."),
        ("2025-06-30 10:50", "ℹ️", "ارتفاع الحرارة بالمفاعل. الاتجاه ضمن الحدود الآمنة.")
    ]
    for t, icon, desc in timeline_steps:
        st.markdown(
            f"""<div class="timeline-step"><span class="timeline-icon">{icon}</span>
            <b>{t}</b><br>
            <span style="font-size:1.07em">{desc}</span></div>""",
            unsafe_allow_html=True
        )

elif section == T["side_sections"][12]:  # Energy Optimization
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][12]}</div>', unsafe_allow_html=True)
    st.markdown(rtl_wrap("Monitor and optimize plant energy use. AI recommendations below." if lang=="en" else "راقب وحسن استهلاك الطاقة. توصيات الذكاء الاصطناعي بالأسفل."))
    energy_sect = ["Compressor", "Pump", "Lighting", "Other"] if lang=="en" else ["ضاغط", "مضخة", "إضاءة", "أخرى"]
    vals = [51, 28, 9, 12]
    fig = px.bar(x=energy_sect, y=vals, color=energy_sect, color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    energy_recos = [
        ("Reduce compressor load during peak hours", "⚡"),
        ("Schedule maintenance for low demand windows", "🛠️")
    ] if lang == "en" else [
        ("تخفيض تشغيل الضواغط أوقات الذروة", "⚡"),
        ("جدولة الصيانة أوقات الطلب المنخفض", "🛠️")
    ]
    for txt, icon in energy_recos:
        st.markdown(f"""<div class="peak-card" style="background:#e0eafc;margin-bottom:0.6em;">
            <span class="timeline-icon">{icon}</span> {txt}
        </div>""", unsafe_allow_html=True)

elif section == T["side_sections"][13]:  # Future Insights
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][13]}</div>', unsafe_allow_html=True)
    st.markdown("<div style=\'display:flex;gap:1.3em;flex-wrap:wrap;\'>", unsafe_allow_html=True)
    future_cards = [
        ("Predictive Risk Alert", "AI models forecast a risk spike for methane at Compressor 2 next week.", "🚨"),
        ("Efficiency Opportunity", "Upgrade control logic to boost plant efficiency by 3%.", "🌱")
    ] if lang == "en" else [
        ("تنبيه مخاطر تنبؤي", "يتوقع الذكاء الاصطناعي ارتفاع الميثان عند الضاغط ٢ الأسبوع القادم.", "🚨"),
        ("فرصة كفاءة", "تحديث منطق التحكم لرفع الكفاءة ٣٪.", "🌱")
    ]
    for title, desc, icon in future_cards:
        st.markdown(f"""<div class="peak-card" style="min-width:220px;max-width:330px;">
            <span style="font-size:2.1em;">{icon}</span><br>
            <b>{title}</b><br>
            <span style="font-size:1.09em">{desc}</span>
        </div>""", unsafe_allow_html=True)
    x = [datetime.now() + timedelta(days=i) for i in range(7)]
    y = [1.2,1.5,2.0,2.8,3.6,3.9,4.8]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#fa709a", width=3), name="Methane Risk"))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), title=("Methane Risk Forecast" if lang=="en" else "توقع مخاطر الميثان"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][14]:  # Operator Feedback
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["side_sections"][14]}</div>', unsafe_allow_html=True)
    if "feedback_list" not in st.session_state:
        st.session_state["feedback_list"] = []
    feedback = st.text_area(rtl_wrap("Add operator feedback or incident note:" if lang=="en" else "أضف ملاحظة أو ملاحظة حادث للمشغل:"), key="feedbackbox")
    if st.button("Submit Feedback" if lang=="en" else "إرسال الملاحظة"):
        if feedback.strip():
            st.session_state["feedback_list"].append((datetime.now().strftime("%Y-%m-%d %H:%M"), feedback.strip()))
    for t, fb in reversed(st.session_state["feedback_list"]):
        st.info(rtl_wrap(f"**[{t}]** {fb}"))

elif section == T["side_sections"][15]:  # About
    show_logo()
    st.markdown(f'<div class="{"gradient-ar" if rtl else "gradient-header"}">{T["about_header"]}</div>', unsafe_allow_html=True)
    st.markdown(f"<div class=\'about-bgcard\'>", unsafe_allow_html=True)
    st.markdown(
        "".join([
            f"<span class=\'about-color\' style=\'background:{color}30;color:{color}\'>{value}</span> "
            for color, value in T["about_colorful"]
        ]), unsafe_allow_html=True
    )
    st.markdown(f"<div class=\'about-story\'>{rtl_wrap(T[\'about_story\'])}</div>", unsafe_allow_html=True)
    st.markdown(rtl_wrap("<div class=\'about-feature\'>Features</div>") if lang=="en" else rtl_wrap("<div class=\'about-feature\'>الميزات</div>"), unsafe_allow_html=True)
    st.markdown("<ul>"+"".join([f"<li>{f}</li>" for f in T["features"]])+"</ul>", unsafe_allow_html=True)
    st.markdown(rtl_wrap("<div class=\'about-feature\'>How to extend</div>") if lang=="en" else rtl_wrap("<div class=\'about-feature\'>كيفية التوسيع</div>"), unsafe_allow_html=True)
    st.markdown("<ul>"+"".join([f"<li>{f}</li>" for f in T["howto_extend"]])+"</ul>", unsafe_allow_html=True)
    st.markdown(rtl_wrap("<div class=\'about-contact\'><b>Contact</b></div>") if lang=="en" else rtl_wrap("<div class=\'about-contact\'><b>تواصل معنا</b></div>"), unsafe_allow_html=True)
    for name, mail, phone in T["developers"]:
        st.markdown(f"{T[\'contact\']}: {name}<br>Email: <a href=\'mailto:{mail}\'>{mail}</a><br>Phone: {phone}<br>", unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<i>{T[\'demo_note\']}</i>"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
