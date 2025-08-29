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
import random
import time
import json

# -------------------- LOGO SVG --------------------
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

# -------------------- MQTT Config --------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "digitaltwin/test/temperature"

# -------------------- Secure config via environment --------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "demo-key")
TWILIO_SID = os.environ.get("TWILIO_SID", "demo-sid")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH", "demo-auth")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "+15005550006")
TWILIO_TO = os.environ.get("TWILIO_TO", "+15005550006")

# -------------------- App state Initialization --------------------
for key, default in [
    ("lang", "en"), ("scenario_step", 0), ("solution_idx", 0), ("theme", "light"),
    ("mqtt_temp", None), ("mqtt_last", None), ("mqtt_started", False), ("sms_sent", False),
    ("feedback_list", []), ("generated_solutions", []), ("solution_generated", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- MQTT background thread --------------------
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

# -------------------- OpenAI setup --------------------
def ask_llm(prompt, lang):
    system_en = """You are an expert AI assistant for an industrial digital twin platform called 'Smart Neural Digital Twin'."""
    
    system = system_en
    
    try:
        # Simulate LLM response for demo
        responses_en = {
            "temperature": "The current temperature is monitored in real-time via MQTT sensors. Typical range is 50-65°C.",
            "pressure": "Pressure readings are stable across all units, averaging 7.2 bar with minimal fluctuations.",
            "methane": "Methane levels are within safe limits. The predictive model shows no significant increase expected in the next 48 hours."
        }
        
        prompt_lower = prompt.lower()
        if "temperature" in prompt_lower:
            return responses_en["temperature"]
        elif "pressure" in prompt_lower:
            return responses_en["pressure"]
        elif "methane" in prompt_lower:
            return responses_en["methane"]
        else:
            return "I can provide information about temperature, pressure, and methane levels. What would you like to know?"
            
    except Exception as e:
        return "AI service is currently unavailable. Please try again later."

# -------------------- Twilio SMS --------------------
def send_sms(to, message):
    try:
        # Simulate SMS sending for demo
        print(f"SMS would be sent to {to}: {message}")
        return True, "Sent (simulated)."
    except Exception as e:
        return False, str(e)

# -------------------- Helper functions --------------------
def to_arabic_numerals(num):
    return str(num).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def rtl_wrap(txt):
    if st.session_state["lang"] == "ar":
        return f'<div style="direction:rtl;text-align:right">{txt}</div>'
    else:
        return f'<div style="direction:ltr;text-align:left">{txt}</div>'

def show_logo():
    st.markdown(f'<div style="text-align:center;padding-bottom:1.2em;">{logo_svg}</div>', unsafe_allow_html=True)

# -------------------- Smart Solution Generator --------------------
def generate_smart_solution(lang):
    """Generate a smart solution based on current conditions"""
    
    # Sample solution templates
    solution_templates_en = [
        {
            "title": "Predictive Maintenance Alert",
            "description": "Initiate predictive maintenance for compressor unit C-203 based on vibration analysis showing early signs of bearing wear.",
            "priority": "High",
            "time_required": "2 hours",
            "impact": "Prevents unplanned downtime of 8+ hours and potential damage to adjacent equipment",
            "cost": "$1,200",
            "savings": "$15,000"
        },
        {
            "title": "Temperature Regulation Protocol",
            "description": "Adjust cooling system parameters to maintain optimal temperature range and prevent thermal stress on reactor vessels.",
            "priority": "Medium",
            "time_required": "45 minutes",
            "impact": "Improves product quality consistency and reduces energy consumption by 8%",
            "cost": "$350",
            "savings": "$8,500/year"
        },
        {
            "title": "Methane Leak Prevention Protocol",
            "description": "Implement enhanced monitoring and automated shutoff valves in high-risk areas to prevent methane leaks before they occur.",
            "priority": "Critical",
            "time_required": "4 hours",
            "impact": "Eliminates risk of safety incidents and potential regulatory fines",
            "cost": "$3,500",
            "savings": "$50,000+"
        }
    ]
    
    templates = solution_templates_en
    return random.choice(templates)

# -------------------- Translations --------------------
texts = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "app_sub": "Intelligent Digital Plant Platform",
        "side_sections": [
            "Dashboard", "Predictive Analytics", "Live Monitoring", 
            "Smart Solutions", "KPI Metrics", "3D Visualization",
            "About", "Smart Recommendations"
        ],
        "lang_en": "English",
        "lang_ar": "Arabic",
        "solution_btn": "Next Solution",
        "logo_alt": "Smart Neural Digital Twin Logo",
        "about_header": "Our Vision",
        "about_story": """Our Smart Neural Digital Twin platform represents the future of industrial safety and efficiency. 
By combining AI-powered predictive analytics with real-time monitoring, we can anticipate issues before they occur, 
saving time, resources, and most importantly - preventing accidents.""",
        "about_colorful": [
            ("#43cea2", "AI Predictive Analytics"),
            ("#fa709a", "Real-time Monitoring"),
            ("#ffb347", "Smart Automation"),
            ("#8fd3f4", "Safety Assurance"),
            ("#185a9d", "Cost Efficiency"),
        ],
        "features": [
            "Real-time sensor monitoring and visualization",
            "AI-powered predictive maintenance alerts",
            "Interactive 3D plant modeling",
            "Automated incident response systems",
            "Bilingual interface (English/Arabic)",
            "Comprehensive KPI tracking and reporting"
        ],
        "howto_extend": [
            "Integrate with existing plant control systems",
            "Add custom sensors and data sources",
            "Develop specialized predictive models",
            "Implement automated response protocols"
        ],
        "developers": [
            ("Rakan Almarri", "rakan.almarri.2@aramco.com", "0532559664"),
            ("Abdulrahman Alzahrani", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "Contact Our Team",
        "demo_note": "Demo version - Not for operational use",
        "live3d_header": "3D Plant Visualization",
        "live3d_intro": "Explore our interactive 3D plant model with real-time data overlay",
        "live3d_404": "3D model loading enhanced simulation",
        "static_3d_caption": "Advanced Plant Visualization",
        "solutions": [
            {
                "title": "Automated Methane Response",
                "desc": "Integrated sensor network with automated shutdown protocols",
                "steps": ["Deploy IoT sensors", "AI detection algorithms", "Emergency response linkage", "Operator training"],
                "priority": "High", "effectiveness": "94%", "time": "3 days", "cost": "$4,000", "savings": "$25,000/year",
                "icon": "🔄"
            }
        ],
        "smart_recommendations": "Smart Recommendations",
        "generate_solution": "GENERATE SOLUTION",
        "solution_title": "Solution Title",
        "solution_description": "Description",
        "solution_priority": "Priority",
        "solution_time": "Time Required",
        "solution_impact": "Impact",
        "solution_cost": "Cost",
        "solution_savings": "Estimated Savings",
        "dashboard_title": "Plant Overview Dashboard",
        "prediction_title": "Predictive Analytics",
        "monitoring_title": "Live Monitoring"
    },
    "ar": {
        "app_title": "التوأم الرقمي العصبي الذكي",
        "app_sub": "منصة المصنع الذكي الرقمي",
        "side_sections": [
            "لوحة التحكم", "التحليلات التنبؤية", "المراقبة الحية", 
            "الحلول الذكية", "مقاييس الأداء", "التصور ثلاثي الأبعاد",
            "حول المشروع", "التوصيات الذكية"
        ],
        "lang_en": "الإنجليزية",
        "lang_ar": "العربية",
        "solution_btn": "الحل التالي",
        "logo_alt": "شعار التوأم الرقمي العصبي الذكي",
        "about_header": "رؤيتنا",
        "about_story": """تمثل منصة التوأم الرقمي العصبي الذكي مستقبل السلامة والكفاءة الصناعية. 
بدمج التحليلات التنبؤية المدعومة بالذكاء الاصطناعي مع المراقبة في الوقت الحقيقي، 
يمكننا توقع المشكلات قبل حدوثها، مما يوفر الوقت والموارد والأهم من ذلك - منع الحوادث.""",
        "about_colorful": [
            ("#43cea2", "التحليلات التنبؤية بالذكاء الاصطناعي"),
            ("#fa709a", "المراقبة في الوقت الحقيقي"),
            ("#ffb347", "الأتمتة الذكية"),
            ("#8fd3f4", "ضمان السلامة"),
            ("#185a9d", "كفاءة التكاليف"),
        ],
        "features": [
            "مراقبة وتصور بيانات المستشعرات لحظياً",
            "تنبيهات الصيانة التنبؤية المدعومة بالذكاء الاصطناعي",
            "نمذجة المصنع ثلاثية الأبعاد التفاعلية",
            "أنظمة الاستجابة الآلية للحوادث",
            "واجهة ثنائية اللغة (الإنجليزية/العربية)",
            "تتبع وتقارير شاملة لمؤشرات الأداء"
        ],
        "howto_extend": [
            "التكامل مع أنظمة التحكم الحالية للمصنع",
            "إضافة مستشعرات ومصادر بيانات مخصصة",
            "تطوير نماذج تنبؤية متخصصة",
            "تنفيذ بروتوكولات استجابة آلية"
        ],
        "developers": [
            ("راكان المعاري", "rakan.almarri.2@aramco.com", "0532559664"),
            ("عبدالرحمن الزهراني", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "اتصل بفريقنا",
        "demo_note": "نسخة تجريبية - غير مخصصة للاستخدام التشغيلي",
        "live3d_header": "تصور المصنع ثلاثي الأبعاد",
        "live3d_intro": "استكشف نموذج المصنع التفاعلي ثلاثي الأبعاد مع تراكب البيانات في الوقت الحقيقي",
        "live3d_404": "تحميل النموذج ثلاثي الأبعاد مع محاكاة محسنة",
        "static_3d_caption": "تصور متقدم للمصنع",
        "solutions": [
            {
                "title": "استجابة آلية للميثان",
                "desc": "شبكة مستشعرات متكاملة مع بروتوكولات إيقاف تلقائية",
                "steps": ["نشر مستشعرات إنترنت الأشياء", "خوارزميات كشف بالذكاء الاصطناعي", "ربط الاستجابة الطارئة", "تدريب المشغلين"],
                "priority": "عالية", "effectiveness": "94%", "time": "٣ أيام", "cost": "$٤٬٠٠٠", "savings": "$٢٥٬٠٠٠/سنة",
                "icon": "🔄"
            }
        ],
        "smart_recommendations": "التوصيات الذكية",
        "generate_solution": "توليد حل",
        "solution_title": "عنوان الحل",
        "solution_description": "الوصف",
        "solution_priority": "الأولوية",
        "solution_time": "الوقت المطلوب",
        "solution_impact": "التأثير",
        "solution_cost": "التكلفة",
        "solution_savings": "التوفير المتوقع",
        "dashboard_title": "نظرة عامة على المصنع",
        "prediction_title": "التحليلات التنبؤية",
        "monitoring_title": "المراقبة الحية"
    }
}

# -------------------- Apply Theme & CSS --------------------
def apply_custom_css():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #1E88E5;
        --secondary: #FF6D00;
        --success: #43A047;
        --danger: #E53935;
        --dark: #263238;
        --light: #F5F5F5;
        --gradient-start: #43cea2;
        --gradient-end: #185a9d;
    }}
    
    .main {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: var(--dark);
    }}
    
    .stApp {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }}
    
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        color: white;
    }}
    
    .main-header {{
        font-size: 2.5rem;
        color: var(--gradient-end);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .sub-header {{
        font-size: 1.6rem;
        color: var(--dark);
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid var(--gradient-start);
        padding-left: 1rem;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--gradient-end);
    }}
    
    .kpi-label {{
        font-size: 0.9rem;
        color: var(--dark);
        font-weight: 500;
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(67, 206, 162, 0.3);
    }}
    
    .solution-card {{
        background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid var(--success);
    }}
    
    .solution-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--dark);
        margin-bottom: 0.8rem;
    }}
    
    .solution-detail {{
        margin-bottom: 0.8rem;
        font-size: 1rem;
    }}
    
    .solution-label {{
        font-weight: 600;
        color: var(--dark);
    }}
    
    .generate-btn {{
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(67, 206, 162, 0.3);
        width: 100%;
        margin-bottom: 1.5rem;
    }}
    
    .generate-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(67, 206, 162, 0.4);
    }}
    
    .about-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f3e5f5 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }}
    
    .feature-item {{
        background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--primary);
    }}
    
    .css-1d391kg {{
        background: linear-gradient(180deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown(
        f"""<div style="color:white;font-size:24px;font-weight:bold;text-align:center;margin-bottom:10px;">
        {texts[st.session_state["lang"]]["app_title"]}
        </div>
        <div style="color:rgba(255,255,255,0.8);text-align:center;margin-bottom:30px;">
        {texts[st.session_state["lang"]]["app_sub"]}
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Language selector
    lang_options = ["English", "Arabic"] if st.session_state["lang"] == "en" else ["الإنجليزية", "العربية"]
    lang_sel = st.radio("Language / اللغة", lang_options, index=0, key="lang_selector")
    st.session_state["lang"] = "en" if lang_sel == "English" or lang_sel == "الإنجليزية" else "ar"
    
    # Section navigation
    lang = st.session_state["lang"]
    t = texts[lang]
    section_list = t["side_sections"]
    section = st.radio("Navigate to / انتقل إلى", section_list, index=0)

# -------------------- Demo data --------------------
np.random.seed(1)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})

# ========== MAIN SECTIONS ==========
if section == t["side_sections"][0]:  # Dashboard
    st.markdown(f'<div class="main-header">{t["dashboard_title"]}</div>', unsafe_allow_html=True)
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Temperature</div>
            <div class="kpi-value">{'٥٥' if lang == 'ar' else '55'}°C</div>
            <div style="color:#43A047;">✓ Normal</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Pressure</div>
            <div class="kpi-value">{'٧٫٢' if lang == 'ar' else '7.2'} bar</div>
            <div style="color:#43A047;">✓ Stable</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Methane Level</div>
            <div class="kpi-value">{'١٫٤' if lang == 'ar' else '1.4'}%</div>
            <div style="color:#43A047;">✓ Safe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">System Health</div>
            <div class="kpi-value">{'٩٦' if lang == 'ar' else '96'}%</div>
            <div style="color:#43A047;">✓ Optimal</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="sub-header">Temperature Trend</div>', unsafe_allow_html=True)
        fig = px.line(demo_df, x="time", y="Temperature", title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">Methane Level</div>', unsafe_allow_html=True)
        fig = px.line(demo_df, x="time", y="Methane", title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif section == t["side_sections"][1]:  # Predictive Analytics
    st.markdown(f'<div class="main-header">{t["prediction_title"]}</div>', unsafe_allow_html=True)
    
    # Generate forecast data
    days = pd.date_range(datetime.now(), periods=7)
    forecast_df = pd.DataFrame({
        "Day": days,
        "Methane": np.linspace(1.2, 4.5, 7) + np.random.normal(0, 0.2, 7),
        "Temperature": np.linspace(55, 63, 7) + np.random.normal(0, 1, 7)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="sub-header">Temperature Forecast</div>', unsafe_allow_html=True)
        fig = px.line(forecast_df, x="Day", y="Temperature", title="")
        fig.update_layout(height=350, yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">Methane Forecast</div>', unsafe_allow_html=True)
        fig = px.line(forecast_df, x="Day", y="Methane", title="")
        fig.update_layout(height=350, yaxis_title="Methane Level (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment
    st.markdown(f'<div class="sub-header">Risk Assessment</div>', unsafe_allow_html=True)
    risk_data = pd.DataFrame({
        "Component": ["Compressor A", "Reactor B", "Pump System", "Cooling Unit"],
        "Risk Level": [25, 65, 40, 15],
        "Status": ["Low", "High", "Medium", "Low"]
    })
    
    fig = px.bar(risk_data, x="Component", y="Risk Level", color="Status",
                 color_discrete_map={"Low": "#43A047", "Medium": "#FF9800", "High": "#F44336"})
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

elif section == t["side_sections"][2]:  # Live Monitoring
    st.markdown(f'<div class="main-header">{t["monitoring_title"]}</div>', unsafe_allow_html=True)
    
    # MQTT Temperature monitoring
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<div class="sub-header">Real-time Temperature</div>', unsafe_allow_html=True)
        
        # Simulate MQTT data if not available
        current_temp = st.session_state["mqtt_temp"] or np.random.normal(55, 2)
        display_temp = to_arabic_numerals(round(current_temp, 1)) if lang == "ar" else round(current_temp, 1)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_temp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [40, 70], 'tickwidth': 1},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [40, 55], 'color': "lightgray"},
                    {'range': [55, 65], 'color': "lightgreen"},
                    {'range': [65, 70], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 65}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">Current Reading</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Temperature</div>
            <div class="kpi-value">{display_temp}°C</div>
            <div style="color:{'#F44336' if current_temp > 65 else '#43A047'};">
                {'⚠️ High' if current_temp > 65 else '✓ Normal'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Last Update</div>
            <div style="font-size:1.2rem;">{st.session_state['mqtt_last'].strftime('%H:%M:%S') if st.session_state['mqtt_last'] else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if current_temp > 65 and not st.session_state["sms_sent"]:
            if st.button("Send Alert", key="alert_btn"):
                ok, msg = send_sms(TWILIO_TO, 
                    f"ALERT: High temperature detected! Current: {current_temp:.1f}°C" if lang=="en" 
                    else f"تنبيه: تم اكتشاف درجة حرارة مرتفعة! الحالية: {to_arabic_numerals(round(current_temp,1))}°م")
                st.session_state["sms_sent"] = True
                st.success("Alert sent!" if lang=="en" else "تم إرسال التنبيه!")

elif section == t["side_sections"][3]:  # Smart Solutions
    st.markdown(f'<div class="main-header">Smart Solutions</div>', unsafe_allow_html=True)
    
    solutions = t["solutions"]
    for i, sol in enumerate(solutions):
        steps_html = "".join([f"<li>{s}</li>" for s in sol["steps"]])
        
        st.markdown(f"""
        <div class="solution-card">
            <div style="font-size:2em;text-align:center;">{sol["icon"]}</div>
            <div class="solution-title" style="text-align:center;">{sol["title"]}</div>
            <div class="solution-detail" style="text-align:center;">{sol["desc"]}</div>
            
            <div style="display:flex;flex-wrap:wrap;gap:1em;justify-content:center;">
                <div class="solution-detail">
                    <span class="solution-label">Priority:</span> {sol['priority']}
                </div>
                <div class="solution-detail">
                    <span class="solution-label">Effectiveness:</span> {sol['effectiveness']}
                </div>
                <div class="solution-detail">
                    <span class="solution-label">Time:</span> {sol['time']}
                </div>
                <div class="solution-detail">
                    <span class="solution-label">Cost:</span> {sol['cost']}
                </div>
                <div class="solution-detail">
                    <span class="solution-label">Savings:</span> {sol['savings']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif section == t["side_sections"][4]:  # KPI Metrics
    st.markdown(f'<div class="main-header">KPI Metrics</div>', unsafe_allow_html=True)
    
    kpi_data = pd.DataFrame({
        "Metric": ["Production Efficiency", "Energy Consumption", "Equipment Availability", "Safety Incidents"],
        "Current": [96, 272, 98, 1],
        "Target": [98, 250, 99, 0],
        "Unit": ["%", "MWh", "%", "Count"]
    })
    
    for _, row in kpi_data.iterrows():
        current_val = to_arabic_numerals(row["Current"]) if lang == "ar" else row["Current"]
        target_val = to_arabic_numerals(row["Target"]) if lang == "ar" else row["Target"]
        status_color = "#43A047" if row["Current"] >= row["Target"] else "#F44336"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="kpi-label">{row["Metric"]}</div>
                    <div class="kpi-value">{current_val} {row["Unit"]}</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#666;">Target: {target_val}</div>
                    <div style="color:{status_color};font-weight:bold;">
                        {'✓ Achieved' if row["Current"] >= row["Target"] else '⚠️ Needs improvement'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown(f'<div class="sub-header">Performance Trends</div>', unsafe_allow_html=True)
    trend_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Efficiency": [88, 90, 92, 94, 95, 96],
        "Availability": [95, 96, 97, 97, 98, 98]
    })
    
    fig = px.line(trend_data, x="Month", y=["Efficiency", "Availability"], 
                  title="", labels={"value": "Percentage", "variable": "Metric"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif section == t["side_sections"][5]:  # 3D Visualization
    st.markdown(f'<div class="main-header">{t["live3d_header"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{t["live3d_intro"]}</div>', unsafe_allow_html=True)
    
    # Use an embedded iframe for 3D visualization
    try:
        st.components.v1.html("""
        <div style="width:100%; height:500px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:15px; display:flex; justify-content:center; align-items:center; color:white;">
            <div style="text-align:center;">
                <h2>Interactive 3D Plant Model</h2>
                <p>Rotate, zoom, and explore the digital twin</p>
                <div style="font-size:48px;">🏭</div>
            </div>
        </div>
        """, height=500)
    except Exception:
        st.image("https://cdn.pixabay.com/photo/2016/11/29/10/07/architecture-1868667_1280.jpg",
                 caption=t["static_3d_caption"])

elif section == t["side_sections"][6]:  # About
    st.markdown(f'<div class="main-header">{t["about_header"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="about-card">
        <div style="font-size:1.2rem; line-height:1.6; margin-bottom:2rem;">
            {t["about_story"]}
        </div>
        
        <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center; margin-bottom:2rem;">
            {"".join([f'''
            <div style="background:{color}; color:white; padding:1rem; border-radius:10px; text-align:center; min-width:150px;">
                <div style="font-weight:bold; margin-bottom:0.5rem;">{value}</div>
            </div>
            ''' for color, value in t["about_colorful"]])}
        </div>
        
        <div style="margin-bottom:2rem;">
            <h3>Features</h3>
            {"".join([f'''
            <div class="feature-item">
                <div style="display:flex; align-items:center;">
                    <div style="margin-right:10px;">✓</div>
                    <div>{feature}</div>
                </div>
            </div>
            ''' for feature in t["features"]])}
        </div>
        
        <div>
            <h3>{t["contact"]}</h3>
            {"".join([f'''
            <div style="margin:1rem 0;">
                <div style="font-weight:bold;">{name}</div>
                <div>📧 {mail}</div>
                <div>📞 {phone}</div>
            </div>
            ''' for name, mail, phone in t["developers"]])}
        </div>
        
        <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #ddd; color:#666;">
            {t["demo_note"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

elif section == t["side_sections"][7]:  # Smart Recommendations
    st.markdown(f'<div class="main-header">{t["smart_recommendations"]}</div>', unsafe_allow_html=True)
    
    # Generate solution button
    if st.button(t["generate_solution"], key="generate_solution_btn", use_container_width=True):
        with st.spinner("Generating smart solution..." if lang=="en" else "جاري توليد حل ذكي..."):
            time.sleep(1.5)  # Simulate processing time
            solution = generate_smart_solution(lang)
            st.session_state["generated_solutions"].append(solution)
            st.session_state["solution_generated"] = True
            st.success("Solution generated successfully!" if lang=="en" else "تم توليد الحل بنجاح!")
    
    # Display generated solutions
    if st.session_state["solution_generated"] and st.session_state["generated_solutions"]:
        latest_solution = st.session_state["generated_solutions"][-1]
        
        st.markdown(f"""
        <div class="solution-card">
            <div class="solution-title">{latest_solution["title"]}</div>
            
            <div class="solution-detail">
                <span class="solution-label">{t["solution_description"]}:</span> {latest_solution["description"]}
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 1em;">
                <div class="solution-detail" style="flex: 1; min-width: 200px;">
                    <span class="solution-label">{t["solution_priority"]}:</span> {latest_solution["priority"]}
                </div>
                
                <div class="solution-detail" style="flex: 1; min-width: 200px;">
                    <span class="solution-label">{t["solution_time"]}:</span> {latest_solution["time_required"]}
                </div>
                
                <div class="solution-detail" style="flex: 1; min-width: 200px;">
                    <span class="solution-label">{t["solution_impact"]}:</span> {latest_solution["impact"]}
                </div>
                
                <div class="solution-detail" style="flex: 1; min-width: 200px;">
                    <span class="solution-label">{t["solution_cost"]}:</span> {latest_solution["cost"]}
                </div>
                
                <div class="solution-detail" style="flex: 1; min-width: 200px;">
                    <span class="solution-label">{t["solution_savings"]}:</span> {latest_solution["savings"]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Implement Solution" if lang=="en" else "تنفيذ الحل", key="implement_btn"):
                st.success("Solution implementation started!" if lang=="en" else "بدأ تنفيذ الحل!")
        with col2:
            if st.button("Schedule for Later" if lang=="en" else "جدولة لوقت لاحق", key="schedule_btn"):
                st.info("Solution scheduled for tomorrow." if lang=="en" else "تم جدولة الحل للغد.")
        with col3:
            if st.button("Generate Alternative" if lang=="en" else "توليد بديل", key="alternative_btn"):
                st.session_state["solution_generated"] = False
                st.rerun()
    
    # Show previous solutions if any
    if len(st.session_state["generated_solutions"]) > 1:
        with st.expander("Previous Solutions" if lang=="en" else "الحلول السابقة"):
            for i, solution in enumerate(st.session_state["generated_solutions"][:-1]):
                st.markdown(f"""
                <div class="solution-card" style="opacity: 0.7;">
                    <div class="solution-title">{solution["title"]}</div>
                    <div class="solution-detail">
                        <span class="solution-label">{t["solution_priority"]}:</span> {solution["priority"]} | 
                        <span class="solution-label">{t["solution_time"]}:</span> {solution["time_required"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
