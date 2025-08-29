import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading
import paho.mqtt.client as mqtt
import os
import random
import time
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

# -------------------- App state Initialization --------------------
for key, default in [
    ("lang", "en"), ("scenario_step", 0), ("solution_idx", 0), ("theme", "light"),
    ("mqtt_temp", 55.0), ("mqtt_last", datetime.now()), ("mqtt_started", False), ("sms_sent", False),
    ("feedback_list", []), ("generated_solutions", []), ("solution_generated", False),
    ("ai_analysis_done", False), ("anomalies_detected", []), ("preprocessed_data", None),
    ("pi_connected", False), ("pi_status", "disconnected"), ("simulation_active", False),
    ("chat_history", []), ("twilio_enabled", True), ("alert_phone_number", "+966532559664")
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- MQTT Setup --------------------
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
        client.loop_start()
    except Exception:
        pass

if not st.session_state["mqtt_started"]:
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    st.session_state["mqtt_started"] = True

# -------------------- Secrets Management --------------------
def load_secrets():
    try:
        return {
            "twilio": {
                "account_sid": st.secrets.get("twilio", {}).get("account_sid", "demo_sid"),
                "auth_token": st.secrets.get("twilio", {}).get("auth_token", "demo_token"),
                "from_number": st.secrets.get("twilio", {}).get("from_number", "+15005550006")
            },
            "openai": {
                "api_key": st.secrets.get("openai", {}).get("api_key", "demo_key")
            }
        }
    except Exception:
        return {
            "twilio": {
                "account_sid": "demo_sid",
                "auth_token": "demo_token", 
                "from_number": "+15005550006"
            },
            "openai": {
                "api_key": "demo_key"
            }
        }

secrets = load_secrets()

# -------------------- Twilio Integration --------------------
def send_twilio_alert(message, phone_number):
    try:
        # محاكاة إرسال الرسالة (ستستخدم Twilio الفعلي في Production)
        st.success(f"✅ Alert sent to {phone_number}")
        print(f"TWILIO ALERT: {message} to {phone_number}")
        
        # كود Twilio الحقيقي (معلق حالياً):
        """
        from twilio.rest import Client
        client = Client(secrets['twilio']['account_sid'], secrets['twilio']['auth_token'])
        message = client.messages.create(
            body=message,
            from_=secrets['twilio']['from_number'],
            to=phone_number
        )
        """
        return True
    except Exception as e:
        st.error(f"❌ Failed to send alert: {str(e)}")
        return False

# -------------------- AI Copilot with General Knowledge --------------------
def generate_ai_response(prompt):
    """توليد رد الذكاء الاصطناعي مع دعم الأسئلة العامة"""
    prompt_lower = prompt.lower()
    
    # الأسئلة العامة
    if any(word in prompt_lower for word in ["weather", "temperature outside", "الطقس", "درجة الحرارة"]):
        return get_weather_info()
    
    elif any(word in prompt_lower for word in ["time", "date", "today", "الوقت", "التاريخ", "اليوم"]):
        return get_current_time_info()
    
    elif any(word in prompt_lower for word in ["hello", "hi", "مرحبا", "السلام"]):
        return "Hello! I'm your SNDT AI Copilot. I can help you with plant monitoring, predictions, and general questions. How can I assist you today?"
    
    # الأسئلة التنبؤية
    elif any(word in prompt_lower for word in ["predict", "forecast", "next", "future", "تنبأ", "توقع"]):
        return generate_time_based_prediction(prompt)
    
    # الأسئلة عن الحالة الحالية
    elif any(word in prompt_lower for word in ["current", "now", "live", "status", "حالي", "مباشر"]):
        return generate_current_status()
    
    else:
        return "I'm your SNDT AI assistant. I can help with plant monitoring, predictions, weather, time, and general questions. What would you like to know?"

def get_weather_info():
    """معلومات عن الطقس الحالي"""
    weather_data = {
        "temperature": random.randint(20, 35),
        "condition": random.choice(["Sunny", "Partly Cloudy", "Clear"]),
        "humidity": random.randint(30, 70)
    }
    return f"Current weather:\n• Temperature: {weather_data['temperature']}°C\n• Condition: {weather_data['condition']}\n• Humidity: {weather_data['humidity']}%"

def get_current_time_info():
    """معلومات عن الوقت والتاريخ"""
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.strftime('%H:%M:%S')}\nToday's date: {now.strftime('%Y-%m-%d')}\nDay of week: {now.strftime('%A')}"

def generate_time_based_prediction(prompt):
    """إنشاء تنبؤات زمنية ذكية"""
    time_keywords = {"hour": 1, "hours": 1, "day": 24, "days": 24, "week": 168, "weeks": 168}
    hours_ahead = 2
    
    for word, value in time_keywords.items():
        if word in prompt.lower():
            hours_ahead = value
            break
    
    predictions = []
    if "temperature" in prompt.lower():
        predictions.append(f"Temperature will increase by {random.randint(2, 8)}°C in {hours_ahead} hours")
    if "pressure" in prompt.lower():
        predictions.append(f"Pressure will rise by {random.uniform(0.5, 2.1):.1f} bar in {hours_ahead} hours")
    if "methane" in prompt.lower():
        predictions.append(f"Methane levels may reach {random.uniform(2.5, 4.8):.1f}% in {hours_ahead} hours")
    
    if predictions:
        return f"Based on current trends:\n\n" + "\n\n".join(f"• {pred}" for pred in predictions)
    else:
        return "I'll analyze the system and provide predictions. Please specify what you want me to predict."

# -------------------- Raspberry Pi Integration --------------------
class RaspberryPiController:
    def __init__(self):
        self.physical_components = {
            "leds": {"red": False, "green": False, "blue": False},
            "buzzer": False,
            "display": "ready"
        }
    
    def connect_to_pi(self):
        try:
            time.sleep(2)
            st.session_state['pi_connected'] = True
            st.session_state['pi_status'] = "connected"
            self.initialize_mockup()
            return True, "✅ Connected to Raspberry Pi successfully"
        except Exception as e:
            st.session_state['pi_connected'] = False
            st.session_state['pi_status'] = "disconnected"
            return False, f"❌ Connection failed: {str(e)}"
    
    def initialize_mockup(self):
        self.set_led("green", True)
        self.set_led("red", False)
        self.set_buzzer(False)
        self.set_display("System Ready")
    
    def simulate_disaster(self, disaster_type="methane_leak"):
        if not st.session_state['pi_connected']:
            return False, "Raspberry Pi not connected"
        
        st.session_state['simulation_active'] = True
        self.set_led("green", False)
        self.set_led("red", True)
        self.set_buzzer(True)
        self.set_display("DISASTER: " + disaster_type.upper())
        
        # إرسال إشعار Twilio
        disaster_message = self.generate_disaster_alert(disaster_type)
        if st.session_state.get('twilio_enabled', False):
            phone_number = st.session_state.get('alert_phone_number', '')
            if phone_number:
                send_twilio_alert(disaster_message, phone_number)
        
        disaster_data = self.generate_disaster_data(disaster_type)
        return True, disaster_data
    
    def generate_disaster_alert(self, disaster_type):
        time_remaining = random.randint(5, 45)
        alert_templates = {
            "methane_leak": f"⚠️ METHANE LEAK: Critical levels detected! Estimated time to danger: {time_remaining} minutes",
            "pressure_surge": f"⚠️ PRESSURE SURGE: System pressure critical! Estimated time to failure: {time_remaining} minutes",
            "overheating": f"⚠️ OVERHEATING: Temperature critical! Estimated time to meltdown: {time_remaining} minutes"
        }
        return alert_templates.get(disaster_type, "⚠️ EMERGENCY: Critical anomaly detected!")
    
    def generate_disaster_data(self, disaster_type):
        disaster_patterns = {
            "methane_leak": {"methane": lambda t: 1.4 + (t * 0.8), "temperature": lambda t: 55.0 + (t * 0.3)},
            "pressure_surge": {"pressure": lambda t: 7.2 + (t * 1.5), "vibration": lambda t: 4.2 + (t * 0.7)},
            "overheating": {"temperature": lambda t: 55.0 + (t * 2.1), "pressure": lambda t: 7.2 + (t * 0.4)}
        }
        return disaster_patterns.get(disaster_type, disaster_patterns["methane_leak"])
    
    def set_led(self, color, state):
        if color in self.physical_components["leds"]:
            self.physical_components["leds"][color] = state
    
    def set_buzzer(self, state):
        self.physical_components["buzzer"] = state
    
    def set_display(self, message):
        self.physical_components["display"] = message

pi_controller = RaspberryPiController()

# -------------------- Advanced AI Analysis --------------------
class AdvancedAIAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def generate_sensor_data(self):
        np.random.seed(42)
        time_index = pd.date_range(start='2023-01-01', periods=500, freq='H')
        base_temp = 50 + 10 * np.sin(2 * np.pi * np.arange(500) / 24)
        base_pressure = 100 + 5 * np.sin(2 * np.pi * np.arange(500) / 168)
        base_methane = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(500) / 84)
        
        data = pd.DataFrame({
            'timestamp': time_index,
            'temperature': base_temp + np.random.normal(0, 2, 500),
            'pressure': base_pressure + np.random.normal(0, 1, 500),
            'methane': np.clip(base_methane + np.random.normal(0, 0.1, 500), 0, 5),
            'vibration': np.random.normal(5, 1, 500)
        })
        return data.set_index('timestamp')
    
    def detect_anomalies(self, data):
        features = data[['temperature', 'pressure', 'methane', 'vibration']].copy()
        scaled_features = self.scaler.fit_transform(features)
        self.anomaly_detector.fit(scaled_features)
        predictions = self.anomaly_detector.predict(scaled_features)
        data['anomaly_predicted'] = [1 if x == 1 else -1 for x in predictions]
        return data
    
    def predict_future(self, data, hours=24):
        last_values = data.iloc[-24:].mean()
        future_time = pd.date_range(start=data.index[-1] + timedelta(hours=1), periods=hours, freq='H')
        future_data = pd.DataFrame(index=future_time)
        
        for column in ['temperature', 'pressure', 'methane', 'vibration']:
            trend = np.random.normal(0, 0.5)
            future_data[column] = last_values[column] + trend * np.arange(1, hours+1)
            
            if column == 'temperature':
                future_data[column] += 5 * np.sin(2 * np.pi * np.arange(hours) / 24)
            elif column == 'pressure':
                future_data[column] += 2 * np.sin(2 * np.pi * np.arange(hours) / 24)
        
        future_data['methane'] = np.clip(future_data['methane'], 0, 5)
        return future_data
    
    def generate_insights(self, data, future_data):
        insights = []
        current_temp = data['temperature'].iloc[-1]
        avg_temp = data['temperature'].mean()
        
        if current_temp > avg_temp + 5:
            insights.append("🌡️ High temperature detected. Consider checking cooling systems.")
        elif current_temp < avg_temp - 5:
            insights.append("🌡️ Low temperature detected. Verify heating systems.")
        
        current_methane = data['methane'].iloc[-1]
        if current_methane > 2.5:
            insights.append("⚠️ High methane levels detected. Potential leak possible.")
        
        pressure_std = data['pressure'].std()
        if pressure_std > 3:
            insights.append("📊 Pressure fluctuations detected. System may be unstable.")
        
        future_temp_trend = future_data['temperature'].iloc[-1] - future_data['temperature'].iloc[0]
        if abs(future_temp_trend) > 3:
            trend_dir = "increasing" if future_temp_trend > 0 else "decreasing"
            insights.append(f"📈 Temperature is {trend_dir}. Expected change: {abs(future_temp_trend):.1f}°C in 24h.")
        
        return insights

ai_analyzer = AdvancedAIAnalyzer()
sensor_data = ai_analyzer.generate_sensor_data()

# -------------------- Translations --------------------
texts = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "app_sub": "Intelligent Digital Plant Platform",
        "side_sections": [
            "Dashboard", "Predictive Analytics", "Live Monitoring", 
            "AI Copilot Chat", "Smart Solutions", "KPI Metrics",
            "SNDT Safety", "3D Visualization", "About", "Raspberry Pi Control"
        ],
        "lang_en": "English", "lang_ar": "Arabic",
        "solution_btn": "Next Solution", "logo_alt": "Smart Neural Digital Twin Logo",
        "about_header": "Our Vision", "contact": "Contact Our Team",
        "demo_note": "Demo version - Not for operational use",
        "live3d_header": "3D Plant Visualization",
        "live3d_intro": "Explore our interactive 3D plant model with real-time data overlay",
        "live3d_404": "3D model loading enhanced simulation",
        "static_3d_caption": "Advanced Plant Visualization",
        "dashboard_title": "Plant Overview Dashboard",
        "prediction_title": "Predictive Analytics",
        "monitoring_title": "Live Monitoring",
        "ai_analysis_title": "Advanced AI Analysis",
        "run_ai_analysis": "RUN AI ANALYSIS",
        "ai_insights": "AI Insights", "anomaly_detection": "Anomaly Detection",
        "future_prediction": "Future Prediction", "system_health": "System Health",
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
        "developers": [
            ("Rakan Almarri", "rakan.almarri.2@aramco.com", "+966532559664"),
            ("Abdulrahman Alzahrani", "abdulrahman.alzhrani.1@aramco.com", "+966549202574")
        ],
        "solutions": [{
            "title": "Automated Methane Response",
            "desc": "Integrated sensor network with automated shutdown protocols",
            "steps": ["Deploy IoT sensors", "AI detection algorithms", "Emergency response linkage", "Operator training"],
            "priority": "High", "effectiveness": "94%", "time": "3 days", "cost": "$4,000", "savings": "$25,000/year",
            "icon": "🔄"
        }]
    },
    "ar": {
        "app_title": "التوأم الرقمي العصبي الذكي",
        "app_sub": "منصة المصنع الذكي الرقمي",
        "side_sections": [
            "لوحة التحكم", "التحليلات التنبؤية", "المراقبة الحية", 
            "محادثة الذكاء الاصطناعي", "الحلول الذكية", "مقاييس الأداء",
            "أمان SNDT", "التصور ثلاثي الأبعاد", "حول", "تحكم Raspberry Pi"
        ],
        "lang_en": "الإنجليزية", "lang_ar": "العربية",
        "solution_btn": "الحل التالي", "logo_alt": "شعار التوأم الرقمي العصبي الذكي",
        "about_header": "رؤيتنا", "contact": "اتصل بفريقنا",
        "demo_note": "نسخة تجريبية - غير مخصصة للاستخدام التشغيلي",
        "live3d_header": "تصور المصنع ثلاثي الأبعاد",
        "live3d_intro": "استكشف نموذج المصنع التفاعلي ثلاثي الأبعاد مع تراكب البيانات في الوقت الحقيقي",
        "live3d_404": "تحميل النموذج ثلاثي الأبعاد مع محاكاة محسنة",
        "static_3d_caption": "تصور متقدم للمصنع",
        "dashboard_title": "نظرة عامة على المصنع",
        "prediction_title": "التحليلات التنبؤية",
        "monitoring_title": "المراقبة الحية",
        "ai_analysis_title": "تحليل الذكاء الاصطناعي المتقدم",
        "run_ai_analysis": "تشغيل التحليل",
        "ai_insights": "رؤى الذكاء الاصطناعي", "anomaly_detection": "كشف الشذوذ",
        "future_prediction": "التنبؤ المستقبلي", "system_health": "صحة النظام",
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
        "developers": [
            ("راكان المعاري", "rakan.almarri.2@aramco.com", "+966532559664"),
            ("عبدالرحمن الزهراني", "abdulrahman.alzhrani.1@aramco.com", "+966549202574")
        ],
        "solutions": [{
            "title": "استجابة آلية للميثان",
            "desc": "شبكة مستشعرات متكاملة مع بروتوكولات إيقاف تلقائية",
            "steps": ["نشر مستشعرات إنترنت الأشياء", "خوارزميات كشف بالذكاء الاصطناعي", "ربط الاستجابة الطارئة", "تدريب المشغلين"],
            "priority": "عالية", "effectiveness": "94%", "time": "٣ أيام", "cost": "$٤٬٠٠٠", "savings": "$٢٥٬٠٠٠/سنة",
            "icon": "🔄"
        }]
    }
}

# -------------------- Custom CSS --------------------
def apply_custom_css():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #1E88E5; --secondary: #FF6D00; --success: #43A047;
        --danger: #E53935; --dark: #263238; --light: #F5F5F5;
        --gradient-start: #43cea2; --gradient-end: #185a9d;
    }}
    .main {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: var(--dark); }}
    .stApp {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }}
    .sidebar .sidebar-content {{ background: linear-gradient(180deg, var(--gradient-start) 0%, var(--gradient-end) 100%); color: white; }}
    .main-header {{ font-size: 2.5rem; color: var(--gradient-end); text-align: center; margin-bottom: 1.5rem; font-weight: 700;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .sub-header {{ font-size: 1.6rem; color: var(--dark); margin-bottom: 1rem; font-weight: 600; border-left: 4px solid var(--gradient-start); padding-left: 1rem; }}
    .metric-card {{ background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%); border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease; border: 1px solid #e0e0e0; }}
    .metric-card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0,0,0,0.15); }}
    .kpi-value {{ font-size: 2rem; font-weight: 700; color: var(--gradient-end); }}
    .kpi-label {{ font-size: 0.9rem; color: var(--dark); font-weight: 500; }}
    .stButton>button {{ background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%); color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.3s ease; }}
    .stButton>button:hover {{ transform: scale(1.05); box-shadow: 0 4px 12px rgba(67, 206, 162, 0.3); }}
    .solution-card {{ background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 4px solid var(--success); }}
    .solution-title {{ font-size: 1.4rem; font-weight: 700; color: var(--dark); margin-bottom: 0.8rem; }}
    .solution-detail {{ margin-bottom: 0.8rem; font-size: 1rem; }}
    .solution-label {{ font-weight: 600; color: var(--dark); }}
    .generate-btn {{ background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%); color: white; border: none; border-radius: 8px;
        padding: 0.8rem 1.5rem; font-weight: 600; font-size: 1.1rem; cursor: pointer; transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(67, 206, 162, 0.3); width: 100%; margin-bottom: 1.5rem; }}
    .generate-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(67, 206, 162, 0.4); }}
    .about-card {{ background: linear-gradient(135deg, #ffffff 0%, #f3e5f5 100%); border-radius: 15px; padding: 2rem; margin: 1.5rem 0; box-shadow: 0 6px 20px rgba(0,0,0,0.1); }}
    .feature-item {{ background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid var(--primary); }}
    .ai-insight-card {{ background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid var(--success); }}
    .anomaly-card {{ background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid var(--danger); }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# -------------------- Helper Functions --------------------
def to_arabic_numerals(num):
    return str(num).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def rtl_wrap(txt):
    if st.session_state["lang"] == "ar":
        return f'<div style="direction:rtl;text-align:right">{txt}</div>'
    else:
        return f'<div style="direction:ltr;text-align:left">{txt}</div>'

def show_logo():
    st.markdown(f'<div style="text-align:center;padding-bottom:1.2em;">{logo_svg}</div>', unsafe_allow_html=True)

# -------------------- Demo Data --------------------
np.random.seed(1)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})

# -------------------- Section: Dashboard --------------------
def dashboard_section():
    st.markdown(f'<div class="main-header">{texts[lang]["dashboard_title"]}</div>', unsafe_allow_html=True)
    
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
    
    # Disaster Simulation
    st.markdown("---")
    st.markdown("### Disaster Simulation")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Methane Leak Simulation", use_container_width=True):
            success, data = pi_controller.simulate_disaster("methane_leak")
            if success:
                st.session_state['disaster_data'] = data
                st.session_state['disaster_type'] = "methane_leak"
                st.rerun()
    with col2:
        if st.button("Pressure Surge Simulation", use_container_width=True):
            success, data = pi_controller.simulate_disaster("pressure_surge")
            if success:
                st.session_state['disaster_data'] = data
                st.session_state['disaster_type'] = "pressure_surge"
                st.rerun()
    with col3:
        if st.button("Overheating Simulation", use_container_width=True):
            success, data = pi_controller.simulate_disaster("overheating")
            if success:
                st.session_state['disaster_data'] = data
                st.session_state['disaster_type'] = "overheating"
                st.rerun()
    
    if st.session_state['simulation_active']:
        if st.button("Stop Simulation", type="primary", use_container_width=True):
            st.session_state['simulation_active'] = False
            pi_controller.initialize_mockup()
            st.rerun()

# -------------------- Section: Predictive Analytics --------------------
def predictive_analytics_section():
    st.markdown(f'<div class="main-header">{texts[lang]["prediction_title"]}</div>', unsafe_allow_html=True)
    
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
    
    # Advanced Analytics
    st.markdown("---")
    st.markdown("### Advanced Predictive Analytics")
    
    if st.button(texts[lang]["run_ai_analysis"], key="run_ai_analysis", use_container_width=True):
        with st.spinner("Running advanced AI analysis..." if lang=="en" else "جاري تشغيل التحليل المتقدم..."):
            analyzed_data = ai_analyzer.detect_anomalies(sensor_data)
            future_predictions = ai_analyzer.predict_future(analyzed_data, hours=24)
            insights = ai_analyzer.generate_insights(analyzed_data, future_predictions)
            anomalies = analyzed_data[analyzed_data['anomaly_predicted'] == -1]
            
            st.session_state["ai_analysis_done"] = True
            st.session_state["analyzed_data"] = analyzed_data
            st.session_state["future_predictions"] = future_predictions
            st.session_state["ai_insights"] = insights
            st.session_state["anomalies_detected"] = anomalies
            
            st.success("AI analysis completed successfully!" if lang=="en" else "تم完成 التحليل بنجاح!")
    
    if st.session_state["ai_analysis_done"]:
        analyzed_data = st.session_state["analyzed_data"]
        future_predictions = st.session_state["future_predictions"]
        insights = st.session_state["ai_insights"]
        anomalies = st.session_state["anomalies_detected"]
        
        # Display AI Insights
        st.markdown(f'<div class="sub-header">{texts[lang]["ai_insights"]}</div>', unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f'<div class="ai-insight-card">📌 {insight}</div>', unsafe_allow_html=True)
        
        # Display Anomalies
        if not anomalies.empty:
            st.markdown(f'<div class="sub-header">{texts[lang]["anomaly_detection"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="anomaly-card">⚠️ {len(anomalies)} anomalies detected in historical data</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analyzed_data.index, y=analyzed_data['temperature'], 
                                    mode='lines', name='Temperature', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['temperature'], 
                                    mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
            fig.update_layout(title="Temperature with Anomalies", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display Future Predictions
        st.markdown(f'<div class="sub-header">{texts[lang]["future_prediction"]}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(future_predictions, y='temperature', title="Temperature Forecast")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(future_predictions, y='methane', title="Methane Forecast")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # System Health Assessment
        st.markdown(f'<div class="sub-header">{texts[lang]["system_health"]}</div>', unsafe_allow_html=True)
        health_score = max(0, 100 - len(anomalies) * 5)
        health_color = "#43A047" if health_score >= 80 else "#FF9800" if health_score >= 60 else "#F44336"
        health_status = "Good" if health_score >= 80 else "Fair" if health_score >= 60 else "Poor"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">System Health Score</div>
            <div class="kpi-value" style="color:{health_color};">{health_score}</div>
            <div style="color:{health_color};font-weight:bold;">{health_status}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Section: Live Monitoring --------------------
def live_monitoring_section():
    st.markdown(f'<div class="main-header">{texts[lang]["monitoring_title"]}</div>', unsafe_allow_html=True)
    
    # MQTT Temperature monitoring
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f'<div class="sub-header">Real-time Temperature</div>', unsafe_allow_html=True)
        current_temp = st.session_state["mqtt_temp"]
        display_temp = to_arabic_numerals(round(current_temp, 1)) if lang == "ar" else round(current_temp, 1)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = current_temp, domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [40, 70], 'tickwidth': 1}, 'bar': {'color': "#1E88E5"},
                    'steps': [{'range': [40, 55], 'color': "lightgray"}, {'range': [55, 65], 'color': "lightgreen"},
                             {'range': [65, 70], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 65}}
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
            <div style="font-size:1.2rem;">{st.session_state['mqtt_last'].strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if current_temp > 65 and not st.session_state["sms_sent"]:
            if st.button("Send Alert", key="alert_btn"):
                st.success("Alert sent!" if lang=="en" else "تم إرسال التنبيه!")
                st.session_state["sms_sent"] = True
    
    # Twilio Notification System
    st.markdown("---")
    st.markdown("### 📱 Twilio Notification System")
    twilio_enabled = st.checkbox("Enable Twilio Notifications", value=st.session_state.get('twilio_enabled', True))
    st.session_state['twilio_enabled'] = twilio_enabled
    
    phone_number = st.text_input("Your Phone Number", value=st.session_state.get('alert_phone_number', '+966532559664'))
    st.session_state['alert_phone_number'] = phone_number
    
    if st.button("Test Twilio Notification"):
        test_message = "🔔 Test alert from SNDT System: This is a test notification. System is working correctly."
        send_twilio_alert(test_message, phone_number)
    
    # Disaster Simulation Alerts
    st.markdown("---")
    st.markdown("### 🚨 Disaster Simulation Alerts")
    disaster_types = {"methane_leak": "Methane Leak Detection", "pressure_surge": "Pressure Surge Detection", "overheating": "Overheating Detection"}
    selected_disaster = st.selectbox("Select Disaster for Alert Test", list(disaster_types.keys()), format_func=lambda x: disaster_types[x])
    
    if st.button("Send Disaster Alert"):
        message = generate_disaster_alert_message(selected_disaster)
        send_twilio_alert(message, phone_number)

def generate_disaster_alert_message(disaster_type):
    alert_messages = {
        "methane_leak": f"🚨 CRITICAL: Methane leak detected! Levels at {random.uniform(3.5, 5.8):.1f}%. Evacuate area immediately!",
        "pressure_surge": f"⚠️ WARNING: Pressure surge detected! Current pressure {random.uniform(9.5, 12.3):.1f} bar. Stabilization needed!",
        "overheating": f"🔥 ALERT: Overheating detected! Temperature at {random.randint(75, 89)}°C. Cool down required!"
    }
    return alert_messages.get(disaster_type, "Alert: Anomaly detected in the system!")

# -------------------- Section: AI Copilot Chat --------------------
def ai_chat_section():
    st.markdown(f'<div class="main-header">AI Copilot Chat</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask AI about plant status or general questions..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_ai_response(prompt)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# -------------------- Section: Smart Solutions --------------------
def smart_solutions_section():
    st.markdown(f'<div class="main-header">Smart Solutions</div>', unsafe_allow_html=True)
    solutions = texts[lang]["solutions"]
    
    for sol in solutions:
        steps_html = "".join([f"<li>{s}</li>" for s in sol["steps"]])
        st.markdown(f"""
        <div class="solution-card">
            <div style="font-size:2em;text-align:center;">{sol["icon"]}</div>
            <div class="solution-title" style="text-align:center;">{sol["title"]}</div>
            <div class="solution-detail" style="text-align:center;">{sol["desc"]}</div>
            <ul style="margin-bottom:0.7em;">{steps_html}</ul>
            <div style="display:flex;flex-wrap:wrap;gap:1em;justify-content:center;">
                <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">Priority: {sol['priority']}</span>
                <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">Effectiveness: {sol['effectiveness']}</span>
                <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">Time: {sol['time']}</span>
                <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">Cost: {sol['cost']}</span>
                <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">Savings: {sol['savings']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Section: KPI Metrics --------------------
def kpi_metrics_section():
    st.markdown(f'<div class="main-header">KPI Metrics</div>', unsafe_allow_html=True)
    kpi_data = pd.DataFrame({
        "Metric": ["Production Efficiency", "Energy Consumption", "Equipment Availability", "Safety Incidents"],
        "Current": [96, 272, 98, 1], "Target": [98, 250, 99, 0], "Unit": ["%", "MWh", "%", "Count"]
    })
    
    for _, row in kpi_data.iterrows():
        current_val = to_arabic_numerals(row["Current"]) if lang == "ar" else row["Current"]
        target_val = to_arabic_numerals(row["Target"]) if lang == "ar" else row["Target"]
        status_color = "#43A047" if row["Current"] >= row["Target"] else "#F44336"
        status_text = "✓ Achieved" if row["Current"] >= row["Target"] else "⚠️ Needs improvement"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div><div class="kpi-label">{row["Metric"]}</div><div class="kpi-value">{current_val} {row["Unit"]}</div></div>
                <div style="text-align:right;"><div style="color:#666;">Target: {target_val}</div><div style="color:{status_color};font-weight:bold;">{status_text}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown(f'<div class="sub-header">Performance Trends</div>', unsafe_allow_html=True)
    trend_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Efficiency": [88, 90, 92, 94, 95, 96], "Availability": [95, 96, 97, 97, 98, 98]
    })
    
    fig = px.line(trend_data, x="Month", y=["Efficiency", "Availability"], title="", labels={"value": "Percentage", "variable": "Metric"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Section: SNDT Safety --------------------
def sndt_safety_section():
    st.markdown(f'<div class="main-header">SNDT Safety System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <h3>🧠 Neural-Powered Safety Protection</h3>
        <p>Our SNDT (Smart Neural Digital Twin) safety system uses advanced neural networks 
        to predict and prevent industrial accidents before they happen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety layers
    safety_layers = [
        {"name": "Physical Sensors", "status": "active", "description": "Real-time monitoring of all physical parameters"},
        {"name": "AI Prediction", "status": "active", "description": "Advanced neural networks for failure prediction"},
        {"name": "Human Verification", "status": "active", "description": "Human oversight for critical decisions"},
        {"name": "Automatic Shutdown", "status": "standby", "description": "Emergency shutdown protocols"},
        {"name": "Emergency Protocols", "status": "standby", "description": "Full emergency response system"}
    ]
    
    for layer in safety_layers:
        status_icon = "✅" if layer["status"] == "active" else "🟡" if layer["status"] == "standby" else "❌"
        st.markdown(f"{status_icon} **{layer['name']}** - *{layer['status'].upper()}*")
        st.caption(layer["description"])
    
    st.progress(1.0, text="Safety System Integrity")
    
    # Safety metrics
    st.markdown(f'<div class="sub-header">Safety Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Days Without Incident", "42", "7 days")
    with col2:
        st.metric("Prevented Accidents", "8", "2 this month")
    with col3:
        st.metric("System Uptime", "99.98%", "0.02% improvement")

# -------------------- Section: 3D Visualization --------------------
def visualization_3d_section():
    st.markdown(f'<div class="main-header">{texts[lang]["live3d_header"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{texts[lang]["live3d_intro"]}</div>', unsafe_allow_html=True)
    
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
                 caption=texts[lang]["static_3d_caption"])

# -------------------- Section: About --------------------
def about_section():
    st.markdown(f'<div class="main-header">{texts[lang]["about_header"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="about-card">
        <div style="font-size:1.2rem; line-height:1.6; margin-bottom:2rem;">
            {texts[lang]["about_story"]}
        </div>
        
        <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center; margin-bottom:2rem;">
            {"".join([f'''
            <div style="background:{color}; color:white; padding:1rem; border-radius:10px; text-align:center; min-width:150px;">
                <div style="font-weight:bold; margin-bottom:0.5rem;">{value}</div>
            </div>
            ''' for color, value in texts[lang]["about_colorful"]])}
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
            ''' for feature in texts[lang]["features"]])}
        </div>
        
        <div>
            <h3>{texts[lang]["contact"]}</h3>
            <div style="margin:1rem 0;">
                <div style="font-weight:bold;">Main Developer: Rakan Almarri</div>
                <div>📧 rakan.almarri.2@aramco.com</div>
                <div>📞 +966532559664</div>
            </div>
            <div style="margin:1rem 0;">
                <div style="font-weight:bold;">Main Developer: Abdulrahman Alzahrani</div>
                <div>📧 abdulrahman.alzhrani.1@aramco.com</div>
                <div>📞 +966549202574</div>
            </div>
        </div>
        
        <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #ddd; color:#666;">
            {texts[lang]["demo_note"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Section: Raspberry Pi Control --------------------
def raspberry_pi_section():
    st.markdown(f'<div class="main-header">Raspberry Pi Control</div>', unsafe_allow_html=True)
    
    connection_status = st.session_state.get('pi_status', 'disconnected')
    col1, col2 = st.columns(2)
    
    with col1:
        if connection_status == "disconnected":
            st.error("❌ Raspberry Pi Disconnected")
            if st.button("Connect to Raspberry Pi", key="connect_pi"):
                with st.spinner("Connecting to Raspberry Pi..."):
                    success, message = pi_controller.connect_to_pi()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        else:
            st.success("✅ Raspberry Pi Connected")
            if st.button("Disconnect", key="disconnect_pi"):
                st.session_state['pi_connected'] = False
                st.session_state['pi_status'] = "disconnected"
                st.rerun()
    
    with col2:
        st.info(f"**Mockup Status:** {'Active' if st.session_state.get('simulation_active', False) else 'Ready'}")
        if st.session_state.get('simulation_active', False):
            st.warning("⚠️ Disaster simulation in progress")
    
    if connection_status == "connected":
        st.markdown("---")
        st.markdown("### 🚨 Simulate Disaster Scenarios")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Methane Leak", use_container_width=True):
                success, data = pi_controller.simulate_disaster("methane_leak")
                if success:
                    st.session_state['disaster_data'] = data
                    st.session_state['disaster_type'] = "methane_leak"
                    st.rerun()
        with col2:
            if st.button("Pressure Surge", use_container_width=True):
                success, data = pi_controller.simulate_disaster("pressure_surge")
                if success:
                    st.session_state['disaster_data'] = data
                    st.session_state['disaster_type'] = "pressure_surge"
                    st.rerun()
        with col3:
            if st.button("Overheating", use_container_width=True):
                success, data = pi_controller.simulate_disaster("overheating")
                if success:
                    st.session_state['disaster_data'] = data
                    st.session_state['disaster_type'] = "overheating"
                    st.rerun()
        
        if st.session_state.get('simulation_active', False):
            if st.button("🛑 Stop Simulation", type="primary", use_container_width=True):
                st.session_state['simulation_active'] = False
                pi_controller.initialize_mockup()
                st.rerun()

# -------------------- Main Application --------------------
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"""<div style="color:white;font-size:24px;font-weight:bold;text-align:center;margin-bottom:10px;">
        {texts[st.session_state["lang"]]["app_title"]}</div>
        <div style="color:rgba(255,255,255,0.8);text-align:center;margin-bottom:30px;">
        {texts[st.session_state["lang"]]["app_sub"]}</div>""", unsafe_allow_html=True)
        
        lang_options = ["English", "Arabic"] if st.session_state["lang"] == "en" else ["الإنجليزية", "العربية"]
        lang_sel = st.radio("Language / اللغة", lang_options, index=0, key="lang_selector")
        st.session_state["lang"] = "en" if lang_sel == "English" or lang_sel == "الإنجليزية" else "ar"
        
        lang = st.session_state["lang"]
        t = texts[lang]
        section_list = t["side_sections"]
        section = st.radio("Navigate to / انتقل إلى", section_list, index=0)
    
    # Main Content
    if section == t["side_sections"][0]:  # Dashboard
        dashboard_section()
    elif section == t["side_sections"][1]:  # Predictive Analytics
        predictive_analytics_section()
    elif section == t["side_sections"][2]:  # Live Monitoring
        live_monitoring_section()
    elif section == t["side_sections"][3]:  # AI Copilot Chat
        ai_chat_section()
    elif section == t["side_sections"][4]:  # Smart Solutions
        smart_solutions_section()
    elif section == t["side_sections"][5]:  # KPI Metrics
        kpi_metrics_section()
    elif section == t["side_sections"][6]:  # SNDT Safety
        sndt_safety_section()
    elif section == t["side_sections"][7]:  # 3D Visualization
        visualization_3d_section()
    elif section == t["side_sections"][8]:  # About
        about_section()
    elif section == t["side_sections"][9]:  # Raspberry Pi Control
        raspberry_pi_section()

if __name__ == "__main__":
    main()
