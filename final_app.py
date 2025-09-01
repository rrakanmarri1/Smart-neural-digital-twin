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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import secrets
import bcrypt
import warnings
warnings.filterwarnings('ignore')

# -------------------- نظام التسجيل والمراقبة --------------------
def setup_logging():
    """إعداد نظام التسجيل والمراقبة"""
    logger = logging.getLogger('SNDT_Platform')
    logger.setLevel(logging.INFO)
    
    # إنشاء مجلد السجلات إذا لم يكن موجوداً
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # معالج للسجلات الدورية (10MB لكل ملف، 5 ملفات احتياطية)
    handler = RotatingFileHandler(
        'logs/sndt_platform.log', 
        maxBytes=10*1024*1024, 
        backupCount=5,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()

# -------------------- نظام الأمان --------------------
class SecurityManager:
    """مدير أمان متكامل للمنصة"""
    
    def __init__(self):
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self):
        """إنشاء مفتاح تشفير آمن"""
        if 'encryption_key' not in st.session_state:
            st.session_state.encryption_key = secrets.token_urlsafe(32)
        return st.session_state.encryption_key
    
    def encrypt_data(self, data):
        """تشفير البيانات (نموذج مبسط)"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data + self.encryption_key.encode()).hexdigest()
    
    def hash_password(self, password):
        """تشفير كلمات المرور"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)
    
    def check_password(self, password, hashed):
        """التحقق من كلمة المرور"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode())

security_manager = SecurityManager()

# -------------------- نظام التخزين المؤقت --------------------
def cache_data(key, data, expiry_minutes=10):
    """تخزين البيانات مؤقتاً لتحسين الأداء"""
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
    st.session_state.cache[key] = {
        'data': data,
        'expiry': expiry_time
    }
    
    logger.info(f"تم تخزين البيانات مؤقتاً للمفتاح: {key}")

def get_cached_data(key):
    """استرجاع البيانات المخزنة مؤقتاً"""
    if 'cache' not in st.session_state:
        return None
        
    cached = st.session_state.cache.get(key)
    if cached and datetime.now() < cached['expiry']:
        logger.info(f"تم استرجاع البيانات من التخزين المؤقت للمفتاح: {key}")
        return cached['data']
    
    return None

# -------------------- نظام الثيمات --------------------
class ThemeManager:
    """مدير الثيمات لوضع الضوء والداكن"""
    
    def __init__(self):
        self.themes = {
            'light': {
                'primary': '#1E88E5',
                'secondary': '#FF6D00',
                'background': '#FFFFFF',
                'surface': '#F5F5F5',
                'text': '#000000',
                'accent': '#43A047',
                'error': '#E53935'
            },
            'dark': {
                'primary': '#2196F3',
                'secondary': '#FF9800',
                'background': '#121212',
                'surface': '#1E1E1E',
                'text': '#FFFFFF',
                'accent': '#4CAF50',
                'error': '#F44336'
            }
        }
    
    def get_theme(self):
        """الحصول على الثيم الحالي"""
        return st.session_state.get('theme', 'light')
    
    def toggle_theme(self):
        """تبديل الثيم"""
        current = self.get_theme()
        new_theme = 'dark' if current == 'light' else 'light'
        st.session_state.theme = new_theme
        logger.info(f"تم تغيير الثيم إلى: {new_theme}")
        return new_theme
    
    def apply_theme_styles(self):
        """تطبيق أنماط الثيم الحالي"""
        theme = self.get_theme()
        colors = self.themes[theme]
        
        st.markdown(f"""
        <style>
        :root {{
            --primary: {colors['primary']};
            --secondary: {colors['secondary']};
            --background: {colors['background']};
            --surface: {colors['surface']};
            --text: {colors['text']};
            --accent: {colors['accent']};
            --error: {colors['error']};
            --gradient-start: #43cea2;
            --gradient-end: #185a9d;
        }}
        
        .stApp {{
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        .main-header {{
            color: {colors['primary']};
            border-bottom: 2px solid {colors['secondary']};
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
        }}
        
        .card {{
            background-color: {colors['surface']};
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border: 1px solid {colors['primary']}20;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {colors['primary']}20, {colors['accent']}20);
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid {colors['primary']}30;
        }}
        
        .status-simulation {{
            background: linear-gradient(135deg, {colors['secondary']}30, {colors['error']}30);
            color: {colors['secondary']};
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            border: 1px solid {colors['secondary']}50;
        }}
        
        .status-real {{
            background: linear-gradient(135deg, {colors['accent']}30, {colors['primary']}30);
            color: {colors['accent']};
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            border: 1px solid {colors['accent']}50;
        }}
        
        .btn-primary {{
            background-color: {colors['primary']};
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .btn-primary:hover {{
            background-color: {colors['primary']}DD;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .language-selector {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .lang-btn {{
            padding: 0.3rem 0.6rem;
            border-radius: 5px;
            border: 1px solid {colors['primary']};
            background: transparent;
            color: {colors['text']};
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .lang-btn.active {{
            background: {colors['primary']};
            color: white;
        }}
        
        .theme-selector {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .theme-btn {{
            padding: 0.3rem 0.6rem;
            border-radius: 5px;
            border: 1px solid {colors['primary']};
            background: transparent;
            color: {colors['text']};
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .theme-btn.active {{
            background: {colors['primary']};
            color: white;
        }}
        
        .notification {{
            padding: 0.8rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid;
        }}
        
        .notification.info {{
            background: {colors['primary']}20;
            border-left-color: {colors['primary']};
        }}
        
        .notification.warning {{
            background: {colors['secondary']}20;
            border-left-color: {colors['secondary']};
        }}
        
        .notification.error {{
            background: {colors['error']}20;
            border-left-color: {colors['error']};
        }}
        
        .notification.success {{
            background: {colors['accent']}20;
            border-left-color: {colors['accent']};
        }}
        </style>
        """, unsafe_allow_html=True)

theme_manager = ThemeManager()

# -------------------- نظام الترجمة متعدد اللغات --------------------
class TranslationSystem:
    """نظام متكامل للترجمة متعددة اللغات"""
    
    def __init__(self):
        self.texts = {
            'ar': {
                'app_title': 'منصة التوأم الرقمي العصبي الذكي',
                'app_sub': 'رقمنة صناعية ومراقبة مدعومة بالذكاء الاصطناعي',
                'side_sections': [
                    '🏠 لوحة التحكم', 
                    '📊 التحليلات والذكاء الاصطناعي', 
                    '🏭 العمليات', 
                    '🤖 التحكم بالأجهزة', 
                    '🛡️ السلامة والتنبيهات',
                    '🌱 الاستدامة',
                    '⚙️ الإعدادات والمساعدة'
                ],
                'system_status_simulation': 'وضع المحاكاة',
                'system_status_real': 'الأجهزة الحقيقية متصلة',
                'self_test': 'اختبار ذاتي',
                'send_test_alert': 'إرسال تنبيه اختباري',
                'temperature': 'درجة الحرارة',
                'pressure': 'الضغط',
                'methane': 'الميثان',
                'vibration': 'الاهتزاز',
                'flow_rate': 'معدل التدفق',
                'real_time_monitoring': 'المراقبة اللحظية',
                'ai_predictions': 'تنبؤات الذكاء الاصطناعي',
                'anomaly_detection': 'كشف الشذوذ',
                'equipment_control': 'التحكم بالمعدات',
                'emergency_protocols': 'بروتوكولات الطوارئ',
                'energy_management': 'إدارة الطاقة',
                'carbon_footprint': 'البصمة الكربونية',
                'language': 'اللغة',
                'theme': 'المظهر',
                'light': 'فاتح',
                'dark': 'داكن',
                'settings': 'الإعدادات',
                'help': 'المساعدة',
                'logout': 'تسجيل الخروج'
            },
            'en': {
                'app_title': 'Smart Neural Digital Twin Platform',
                'app_sub': 'Industrial Digitalization & AI-Powered Monitoring',
                'side_sections': [
                    '🏠 Dashboard', 
                    '📊 Analytics & AI', 
                    '🏭 Operations', 
                    '🤖 Hardware Control', 
                    '🛡️ Safety & Alerts',
                    '🌱 Sustainability',
                    '⚙️ Settings & Help'
                ],
                'system_status_simulation': 'Simulation Mode',
                'system_status_real': 'Real Hardware Connected',
                'self_test': 'Self Test',
                'send_test_alert': 'Send Test Alert',
                'temperature': 'Temperature',
                'pressure': 'Pressure',
                'methane': 'Methane',
                'vibration': 'Vibration',
                'flow_rate': 'Flow Rate',
                'real_time_monitoring': 'Real-time Monitoring',
                'ai_predictions': 'AI Predictions',
                'anomaly_detection': 'Anomaly Detection',
                'equipment_control': 'Equipment Control',
                'emergency_protocols': 'Emergency Protocols',
                'energy_management': 'Energy Management',
                'carbon_footprint': 'Carbon Footprint',
                'language': 'Language',
                'theme': 'Theme',
                'light': 'Light',
                'dark': 'Dark',
                'settings': 'Settings',
                'help': 'Help',
                'logout': 'Logout'
            }
        }
    
    def get_text(self, key, lang=None):
        """الحصول على النص المترجم"""
        if lang is None:
            lang = st.session_state.get('lang', 'ar')
        return self.texts[lang].get(key, key)
    
    def set_language(self, lang):
        """تغيير اللغة"""
        st.session_state.lang = lang
        logger.info(f"تم تغيير اللغة إلى: {lang}")

translator = TranslationSystem()

# -------------------- SVG Logo --------------------
logo_svg = """<svg width="64" height="64" viewBox="0 0 64 64" fill="none">
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
</svg>"""

# -------------------- MQTT Config --------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_TEMPERATURE = "sndt/temperature"
MQTT_TOPIC_PRESSURE = "sndt/pressure"
MQTT_TOPIC_METHANE = "sndt/methane"
MQTT_TOPIC_CONTROL = "sndt/control"

# -------------------- تهيئة حالة التطبيق --------------------
def init_session_state():
    """تهيئة حالة الجلسة مع جميع القيم المطلوبة"""
    defaults = {
        "lang": "ar",
        "scenario_step": 0,
        "solution_idx": 0,
        "theme": "light",
        "mqtt_temp": 55.0,
        "mqtt_last": datetime.now(),
        "mqtt_started": False,
        "sms_sent": False,
        "feedback_list": [],
        "generated_solutions": [],
        "solution_generated": False,
        "ai_analysis_done": False,
        "anomalies_detected": [],
        "preprocessed_data": None,
        "pi_connected": False,
        "pi_status": "disconnected",
        "simulation_active": False,
        "chat_history": [],
        "twilio_enabled": True,
        "alert_phone_number": "+966532559664",
        "operations_data": {},
        "energy_optimization": {},
        "incident_timeline": [],
        "lifelong_memory": [],
        "physical_twin_connected": False,
        "pressure": 7.2,
        "methane": 1.4,
        "vibration": 4.5,
        "flow_rate": 110.0,
        "mqtt_connected": False,
        "current_sensor_data": {},
        "show_advanced": False,
        "openai_enabled": False,
        "openai_api_key": "",
        "notification_history": [],
        "self_test_results": {},
        "system_status": "simulation",
        "last_emergency_alert": None,
        "emergency_protocols": {},
        "optimization_history": [],
        "maintenance_predictions": [],
        "carbon_footprint": {},
        "digital_threads": {},
        "cache": {},
        "user_authenticated": False,
        "login_attempts": 0,
        "last_login_attempt": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -------------------- نظام MQTT متين --------------------
class RobustMQTTClient:
    """نظام اتصال MQTT مع إدارة أخطاء متقدمة"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.connection_attempts = 0
        self.max_attempts = 5
    
    def on_connect(self, client, userdata, flags, rc):
        """دالة الاتصال"""
        if rc == 0:
            self.connected = True
            self.connection_attempts = 0
            st.session_state.mqtt_connected = True
            logger.info("تم الاتصال بنجاح بخادم MQTT")
            
            # الاشتراك في المواضيع
            client.subscribe([
                (MQTT_TOPIC_TEMPERATURE, 0),
                (MQTT_TOPIC_PRESSURE, 0),
                (MQTT_TOPIC_METHANE, 0)
            ])
        else:
            logger.error(f"فشل الاتصال بخادم MQTT، رمز الخطأ: {rc}")
    
    def on_message(self, client, userdata, msg):
        """معالجة الرسائل الواردة"""
        try:
            payload = msg.payload.decode()
            value = float(payload)
            
            if msg.topic == MQTT_TOPIC_TEMPERATURE:
                st.session_state.mqtt_temp = value
                st.session_state.current_sensor_data['temperature'] = value
            elif msg.topic == MQTT_TOPIC_PRESSURE:
                st.session_state.pressure = value
                st.session_state.current_sensor_data['pressure'] = value
            elif msg.topic == MQTT_TOPIC_METHANE:
                st.session_state.methane = value
                st.session_state.current_sensor_data['methane'] = value
            
            st.session_state.mqtt_last = datetime.now()
            logger.info(f"تم استقبال بيانات: {msg.topic} = {value}")
            
        except Exception as e:
            logger.error(f"خطأ في معالجة رسالة MQTT: {str(e)}")
    
    def on_disconnect(self, client, userdata, rc):
        """دالة انقطاع الاتصال"""
        self.connected = False
        st.session_state.mqtt_connected = False
        if rc != 0:
            logger.warning("انقطع الاتصال غير المتوقع بخادم MQTT")
    
    def connect_with_retry(self):
        """الاتصال مع إعادة المحاولة عند الفشل"""
        if self.connection_attempts >= self.max_attempts:
            logger.error("وصلت محاولات الاتصال إلى الحد الأقصى، التحويل إلى وضع المحاكاة")
            st.session_state.system_status = "simulation"
            return False
        
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            self.connection_attempts += 1
            logger.info(f"محاولة اتصال رقم {self.connection_attempts} إلى خادم MQTT")
            
            # الانتظار قليلاً للاتصال
            time.sleep(2)
            
            if self.connected:
                st.session_state.system_status = "real_hardware"
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"فشل الاتصال بخادم MQTT: {str(e)}")
            self.connection_attempts += 1
            return False
    
    def publish(self, topic, message):
        """نشر رسالة MQTT"""
        if self.connected and self.client:
            try:
                result = self.client.publish(topic, message)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"تم إرسال رسالة إلى {topic}: {message}")
                    return True
                else:
                    logger.error(f"فشل إرسال الرسالة إلى {topic}")
                    return False
            except Exception as e:
                logger.error(f"خطأ في إرسال رسالة MQTT: {str(e)}")
                return False
        else:
            logger.warning("محاولة إرسال رسالة MQTT بدون اتصال")
            return False
    
    def disconnect(self):
        """قطع الاتصال"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            st.session_state.mqtt_connected = False
            logger.info("تم قطع الاتصال بخادم MQTT")

# تهيئة عميل MQTT
mqtt_client = RobustMQTTClient()

# -------------------- محاكاة بيانات MQTT --------------------
def start_mqtt_simulation():
    """تشغيل محاكاة البيانات إذا تعذر الاتصال"""
    def simulate_data():
        anomaly_counter = 0
        while True:
            if not mqtt_client.connected:
                current_time = datetime.now()
                
                # محاكاة بيانات طبيعية مع شذوذ عشوائي
                base_temp = 55 + 5 * np.sin(current_time.hour / 24 * 2 * np.pi)
                temp_noise = np.random.normal(0, 1.5)
                temperature = base_temp + temp_noise
                
                # محاكاة شذوذ عشوائي (5% احتمال)
                if random.random() < 0.05:
                    temperature += random.uniform(8, 15)
                    anomaly_counter += 1
                    logger.warning(f"تم محاكاة شذوذ في درجة الحرارة: {temperature:.2f}°C")
                
                base_pressure = 7 + 0.5 * np.sin(current_time.hour / 12 * 2 * np.pi)
                pressure = base_pressure + np.random.normal(0, 0.3)
                
                base_methane = 1.2 + 0.3 * np.sin(current_time.hour / 8 * 2 * np.pi)
                methane = max(0, base_methane + np.random.normal(0, 0.2))
                
                # تحديث حالة الجلسة
                st.session_state.mqtt_temp = temperature
                st.session_state.pressure = pressure
                st.session_state.methane = methane
                st.session_state.mqtt_last = current_time
                
                st.session_state.current_sensor_data = {
                    'temperature': temperature,
                    'pressure': pressure,
                    'methane': methane,
                    'vibration': 4.5 + np.random.normal(0, 0.5),
                    'flow_rate': 110 + np.random.normal(0, 10)
                }
                
                # تسجيل بيانات المحاكاة
                if anomaly_counter > 0 and anomaly_counter % 5 == 0:
                    logger.info(f"محاكاة MQTT: {temperature:.2f}°C, {pressure:.2f}bar, {methane:.2f}ppm")
                
            time.sleep(2)
    
    # بدء محاكاة البيانات في خلفية مؤشرة
    if not st.session_state.get('simulation_thread_started', False):
        simulation_thread = threading.Thread(target=simulate_data, daemon=True)
        simulation_thread.start()
        st.session_state.simulation_thread_started = True
        logger.info("بدأت محاكاة بيانات MQTT")

# -------------------- التهيئة الرئيسية --------------------
if not st.session_state["mqtt_started"]:
    mqtt_success = mqtt_client.connect_with_retry()
    if not mqtt_success:
        start_mqtt_simulation()
    st.session_state["mqtt_started"] = True

# -------------------- تكامل حقيقي مع Raspberry Pi --------------------
class RealRaspberryPiController:
    """متحكم حقيقي بـ Raspberry Pi مع دعم GPIO"""
    
    def __init__(self):
        self.connected = False
        try:
            # محاولة استيراد مكتبة GPIO
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.connected = True
            st.session_state.pi_connected = True
            st.session_state.pi_status = "connected"
            logger.info("تم الكشف عن Raspberry Pi وتحميل مكتبة GPIO")
        except ImportError:
            logger.warning("لم يتم العثور على مكتبة RPi.GPIO، تشغيل وضع المحاكاة")
            self.connected = False
            st.session_state.pi_connected = False
            st.session_state.pi_status = "simulated"
    
    def set_output(self, pin, state):
        """ضبط حالة دبوس الإخراج"""
        if self.connected:
            try:
                self.GPIO.output(pin, state)
                logger.info(f"تم ضبط الدبوس {pin} إلى {'HIGH' if state else 'LOW'}")
                return True
            except Exception as e:
                logger.error(f"خطأ في ضبط دبوس الإخراج: {str(e)}")
                return False
        else:
            # محاكاة النجاح في وضع المحاكاة
            logger.info(f"محاكاة: تم ضبط الدبوس {pin} إلى {'HIGH' if state else 'LOW'}")
            return True
    
    def read_input(self, pin):
        """قراءة حالة دبوس الإدخال"""
        if self.connected:
            try:
                value = self.GPIO.input(pin)
                logger.info(f"تم قراءة الدبوس {pin}: {value}")
                return value
            except Exception as e:
                logger.error(f"خطأ في قراءة دبوس الإدخال: {str(e)}")
                return None
        else:
            # محاكاة قيمة عشوائية في وضع المحاكاة
            value = random.choice([0, 1])
            logger.info(f"محاكاة: تم قراءة الدبوس {pin}: {value}")
            return value

# تهيئة المتحكم الحقيقي
real_pi_controller = RealRaspberryPiController()

# -------------------- الذاكرة الدائمة --------------------
class LifelongLearningMemory:
    """نظام ذاكرة دائمة للتعلم من التجارب"""
    
    def __init__(self):
        self.memory_file = "lifelong_memory.json"
        self.load_memory()
    
    def load_memory(self):
        """تحميل الذاكرة من الملف"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    st.session_state.lifelong_memory = json.load(f)
                logger.info("تم تحميل الذاكرة الدائمة بنجاح")
            else:
                st.session_state.lifelong_memory = []
                logger.info("تم تهيئة ذاكرة دائمة جديدة")
        except Exception as e:
            logger.error(f"خطأ في تحميل الذاكرة الدائمة: {str(e)}")
            st.session_state.lifelong_memory = []
    
    def save_memory(self):
        """حفظ الذاكرة إلى الملف"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.lifelong_memory, f, ensure_ascii=False, indent=2)
            logger.info("تم حفظ الذاكرة الدائمة بنجاح")
        except Exception as e:
            logger.error(f"خطأ في حفظ الذاكرة الدائمة: {str(e)}")
    
    def add_experience(self, event_type, description, data=None):
        """إضافة تجربة جديدة إلى الذاكرة"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description,
            'data': data or {}
        }
        
        st.session_state.lifelong_memory.append(experience)
        
        # الحفاظ على حجم معقول للذاكرة
        if len(st.session_state.lifelong_memory) > 1000:
            st.session_state.lifelong_memory = st.session_state.lifelong_memory[-1000:]
        
        self.save_memory()
        logger.info(f"تم إضافة تجربة إلى الذاكرة: {event_type} - {description}")
    
    def get_relevant_experiences(self, event_type=None, limit=10):
        """الحصول على تجارب ذات صلة"""
        experiences = st.session_state.lifelong_memory
        
        if event_type:
            experiences = [e for e in experiences if e['type'] == event_type]
        
        return experiences[-limit:]

# تهيئة الذاكرة الدائمة
lifelong_memory = LifelongLearningMemory()

# -------------------- Advanced AI Analysis --------------------
class AdvancedAIAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=3, random_state=42)
        self.regressor = LinearRegression()
    
    def generate_sensor_data(self, hours=48):
        """إنشاء بيانات استشعار للتحليل"""
        np.random.seed(1)
        timestamps = pd.date_range(
            datetime.now() - timedelta(hours=hours), 
            periods=hours*2, 
            freq="30min"
        )
        
        data = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": np.random.normal(55, 6, hours*2),
            "pressure": np.random.normal(7, 1.2, hours*2),
            "methane": np.clip(np.random.normal(1.4, 0.7, hours*2), 0, 6),
            "vibration": np.random.normal(4.5, 1.0, hours*2),
            "flow_rate": np.random.normal(110, 15, hours*2)
        })
        
        # إضافة بعض الشذوذ
        anomaly_indices = np.random.choice(len(data), size=int(0.05 * len(data)), replace=False)
        for idx in anomaly_indices:
            data.loc[idx, 'temperature'] += np.random.uniform(10, 20)
            data.loc[idx, 'pressure'] += np.random.uniform(2, 4) * np.random.choice([-1, 1])
        
        return data
    
    def detect_anomalies(self, data):
        """كشف الشذوذ في البيانات"""
        try:
            # تحضير البيانات
            features = data[['temperature', 'pressure', 'methane', 'vibration', 'flow_rate']].copy()
            features_scaled = self.scaler.fit_transform(features)
            
            # تدريب وكشف الشذوذ
            anomalies = self.anomaly_detector.fit_predict(features_scaled)
            data['anomaly'] = anomalies
            
            # استخراج نقاط الشذوذ
            anomaly_points = data[data['anomaly'] == -1].copy()
            
            # حفظ في الذاكرة الدائمة
            if len(anomaly_points) > 0:
                lifelong_memory.add_experience(
                    'anomaly_detection',
                    f'تم كشف {len(anomaly_points)} شذوذ في بيانات الاستشعار',
                    {'anomaly_count': len(anomaly_points)}
                )
            
            logger.info(f"تم كشف {len(anomaly_points)} شذوذ في البيانات")
            return data, anomaly_points
        
        except Exception as e:
            logger.error(f"خطأ في كشف الشذوذ: {str(e)}")
            return data, pd.DataFrame()
    
    def cluster_data(self, data):
        """تجميع البيانات لاكتشاف الأنماط"""
        try:
            features = data[['temperature', 'pressure', 'methane']].copy()
            features_scaled = self.scaler.fit_transform(features)
            
            clusters = self.clusterer.fit_predict(features_scaled)
            data['cluster'] = clusters
            
            logger.info(f"تم تجميع البيانات إلى {len(set(clusters))} clusters")
            return data
        
        except Exception as e:
            logger.error(f"خطأ في تجميع البيانات: {str(e)}")
            return data
    
    def predict_trends(self, data, hours_ahead=6):
        """التنبؤ بالاتجاهات المستقبلية"""
        try:
            # تحضير البيانات للانحدار
            data = data.sort_values('timestamp').reset_index(drop=True)
            data['time_index'] = range(len(data))
            
            # التنبؤ بدرجة الحرارة
            X = data[['time_index']].values
            y_temp = data['temperature'].values
            
            self.regressor.fit(X, y_temp)
            future_indices = np.array(range(len(data), len(data) + hours_ahead * 2)).reshape(-1, 1)
            temp_predictions = self.regressor.predict(future_indices)
            
            # إنجاد تواريخ مستقبلية
            last_time = data['timestamp'].iloc[-1]
            future_times = [last_time + timedelta(minutes=30*i) for i in range(1, hours_ahead*2 + 1)]
            
            predictions_df = pd.DataFrame({
                'timestamp': future_times,
                'temperature_pred': temp_predictions,
                'type': 'prediction'
            })
            
            logger.info(f"تم إنشاء تنبؤات لـ {hours_ahead*2} فترات قادمة")
            return predictions_df
        
        except Exception as e:
            logger.error(f"خطأ في التنبؤ بالاتجاهات: {str(e)}")
            return pd.DataFrame()

# تهيئة محلل الذكاء الاصطناعي
ai_analyzer = AdvancedAIAnalyzer()

# -------------------- OpenAI Integration --------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def init_openai(api_key):
    """تهيئة OpenAI"""
    if api_key and OPENAI_AVAILABLE:
        try:
            openai.api_key = api_key
            # اختبار المفتاح بمحاولة بسيطة
            openai.Model.list()
            st.session_state["openai_api_key"] = api_key
            st.session_state["openai_enabled"] = True
            logger.info("تم تهيئة OpenAI بنجاح")
            return True
        except Exception as e:
            error_msg = f"❌ مفتاح OpenAI غير صالح: {str(e)}"
            st.session_state["notification_history"].append({
                "timestamp": datetime.now(),
                "type": "error",
                "message": error_msg
            })
            logger.error(f"فشل تهيئة OpenAI: {str(e)}")
            return False
    return False

def generate_openai_response(prompt):
    """إنشاء رد باستخدام OpenAI"""
    try:
        if not st.session_state.get("openai_enabled", False):
            return None
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an industrial AI assistant for a Smart Neural Digital Twin platform. Provide helpful, concise responses in Arabic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        logger.info("تم إنشاء رد باستخدام OpenAI")
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"خطأ في إنشاء رد OpenAI: {str(e)}")
        return None

# -------------------- Twilio Integration --------------------
def send_twilio_alert(message, phone_number):
    """إرسال تنبيه عبر Twilio"""
    try:
        from twilio.rest import Client
        
        # تخزين مؤقت لمعلومات Twilio (يجب تخزينها في متغيرات البيئة في الإنتاج)
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "test_sid")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "test_token")
        twilio_number = os.environ.get("TWILIO_PHONE_NUMBER", "+1234567890")
        
        # في وضع المحاكاة، نعود إلى True لمحاكاة النجاح
        if account_sid == "test_sid":
            logger.info(f"محاكاة: تم إرسال تنبيه إلى {phone_number}: {message}")
            return True
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body=message,
            from_=twilio_number,
            to=phone_number
        )
        
        lifelong_memory.add_experience(
            'twilio_alert',
            f'تم إرسال تنبيه إلى {phone_number}',
            {'message': message, 'phone_number': phone_number}
        )
        
        logger.info(f"تم إرسال تنبيه Twilio إلى {phone_number}")
        return True
    
    except Exception as e:
        error_msg = f"فشل إرسال التنبيه: {str(e)}"
        st.session_state["notification_history"].append({
            "timestamp": datetime.now(),
            "type": "error",
            "message": error_msg
        })
        logger.error(f"فشل إرسال تنبيه Twilio: {str(e)}")
        return False

# -------------------- 1. Digital Twin Optimization (تحسين التوأم الرقمي) --------------------
class DigitalTwinOptimizer:
    """نظام تحسين ذاتي للتوأم الرقمي باستخدام التعلم المعزز"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_parameters(self, current_data):
        """تحسين معاملات التشغيل بناءً على البيانات الحالية"""
        try:
            # محاكاة خوارزمية تحسين (في الواقع الفعلي، ستكون خوارزمية أكثر تعقيداً)
            temp = current_data.get('temperature', 55)
            pressure = current_data.get('pressure', 7)
            methane = current_data.get('methane', 1.4)
            
            # تحسين بسيط لمحاكاة الخوارزمية
            optimized_temp = max(40, min(65, temp * 0.95 if temp > 58 else temp * 1.05))
            optimized_pressure = max(5, min(10, pressure * 0.97 if pressure > 7.5 else pressure * 1.03))
            
            optimization = {
                'timestamp': datetime.now(),
                'original_temp': temp,
                'optimized_temp': optimized_temp,
                'original_pressure': pressure,
                'optimized_pressure': optimized_pressure,
                'methane_level': methane,
                'estimated_savings': random.uniform(2.5, 5.7)
            }
            
            self.optimization_history.append(optimization)
            st.session_state.optimization_history = self.optimization_history
            
            lifelong_memory.add_experience(
                'optimization',
                'تم تحسين معاملات التشغيل للتوأم الرقمي',
                optimization
            )
            
            logger.info("تم تحسين معاملات التشغيل للتوأم الرقمي")
            return optimization
        
        except Exception as e:
            logger.error(f"خطأ في تحسين المعاملات: {str(e)}")
            return None

# -------------------- 2. AI-Powered Predictive Maintenance (الصيانة التنبؤية) --------------------
class PredictiveMaintenance:
    """نظام الصيانة التنبؤية المدعوم بالذكاء الاصطناعي"""
    
    def __init__(self):
        self.maintenance_history = []
    
    def predict_maintenance(self, sensor_data):
        """التنبؤ باحتياجات الصيانة"""
        try:
            # محاكاة تنبؤات الصيانة
            temp = sensor_data.get('temperature', 55)
            vibration = sensor_data.get('vibration', 4.5)
            hours_until_maintenance = max(0, 500 - (temp - 50) * 10 - vibration * 20)
            
            # تحديد مستوى الأولوية
            if hours_until_maintenance < 24:
                priority = "high"
                message = "الصيانة العاجلة مطلوبة في خلال 24 ساعة"
            elif hours_until_maintenance < 72:
                priority = "medium"
                message = "الصيانة مطلوبة في خلال 3 أيام"
            else:
                priority = "low"
                message = "لا توجد صيانة عاجلة مطلوبة"
            
            prediction = {
                'timestamp': datetime.now(),
                'hours_until_maintenance': hours_until_maintenance,
                'priority': priority,
                'message': message,
                'sensor_data': sensor_data
            }
            
            self.maintenance_history.append(prediction)
            st.session_state.maintenance_predictions = self.maintenance_history
            
            lifelong_memory.add_experience(
                'maintenance_prediction',
                f'تنبؤ بالصيانة: {message}',
                prediction
            )
            
            logger.info(f"تم إنشاء تنبؤ بالصيانة: {message}")
            return prediction
        
        except Exception as e:
            logger.error(f"خطأ في التنبؤ بالصيانة: {str(e)}")
            return None

# -------------------- 6. Advanced Emergency Response (استجابة طوارئ متقدمة) --------------------
class EmergencyResponseSystem:
    """نظام متقدم للاستجابة للطوارئ والكوارث"""
    
    def __init__(self):
        self.protocols = {
            'high_temperature': {
                'name': 'ارتفاع درجة الحرارة',
                'actions': [
                    'تشغيل نظام التبريد الاحتياطي',
                    'تقليل حمل التشغيل',
                    'إخطار فريق الصيانة'
                ],
                'threshold': 65
            },
            'high_methane': {
                'name': 'ارتفاع مستوى الميثان',
                'actions': [
                    'تشغيل نظام التهوية',
                    'إيقاف المعدات غير الضرورية',
                    'إخلاء المنطقة إذا لزم الأمر'
                ],
                'threshold': 3.0
            },
            'high_pressure': {
                'name': 'ارتفاع الضغط',
                'actions': [
                    'تشغيل صمامات الأمان',
                    'تقليل ضغط التشغيل',
                    'فحص نظام الاحتواء'
                ],
                'threshold': 9.0
            }
        }
    
    def check_emergency(self, sensor_data):
        """فحص حالات الطوارئ المحتملة"""
        emergencies = []
        
        # فحص ارتفاع درجة الحرارة
        if sensor_data.get('temperature', 0) > self.protocols['high_temperature']['threshold']:
            emergencies.append('high_temperature')
        
        # فحص ارتفاع الميثان
        if sensor_data.get('methane', 0) > self.protocols['high_methane']['threshold']:
            emergencies.append('high_methane')
        
        # فحص ارتفاع الضغط
        if sensor_data.get('pressure', 0) > self.protocols['high_pressure']['threshold']:
            emergencies.append('high_pressure')
        
        # إذا تم اكتشاف حالات طوارئ
        if emergencies:
            for emergency in emergencies:
                protocol = self.protocols[emergency]
                
                # إرسال تنبيه
                alert_message = f"تحذير: {protocol['name']} تم اكتشافه. الإجراءات: {', '.join(protocol['actions'])}"
                
                if st.session_state.twilio_enabled:
                    send_twilio_alert(alert_message, st.session_state.alert_phone_number)
                
                # تسجيل في الذاكرة الدائمة
                lifelong_memory.add_experience(
                    'emergency_alert',
                    f'تم اكتشاف حالة طوارئ: {protocol["name"]}',
                    {
                        'protocol': protocol,
                        'sensor_data': sensor_data,
                        'alert_sent': st.session_state.twilio_enabled
                    }
                )
                
                logger.warning(f"تم اكتشاف حالة طوارئ: {protocol['name']}")
            
            return emergencies
        
        return []

# -------------------- 7. Digital Thread Implementation (الخيط الرقمي) --------------------
class DigitalThread:
    """تنفيذ الخيط الرقمي لربط دورة الحياة الكاملة"""
    
    def __init__(self):
        self.threads = {}
    
    def create_thread(self, component_id, component_data):
        """إنشاء خيط رقمي للمكون"""
        thread = {
            'component_id': component_id,
            'created': datetime.now(),
            'last_updated': datetime.now(),
            'data': component_data,
            'history': []
        }
        
        self.threads[component_id] = thread
        st.session_state.digital_threads = self.threads
        
        logger.info(f"تم إنشاء خيط رقمي للمكون: {component_id}")
        return thread
    
    def update_thread(self, component_id, update_data):
        """تحديث الخيط الرقمي"""
        if component_id in self.threads:
            self.threads[component_id]['data'].update(update_data)
            self.threads[component_id]['last_updated'] = datetime.now()
            self.threads[component_id]['history'].append({
                'timestamp': datetime.now(),
                'update': update_data
            })
            
            st.session_state.digital_threads = self.threads
            logger.info(f"تم تحديث الخيط الرقمي للمكون: {component_id}")
            return True
        
        logger.warning(f"لم يتم العثور على خيط رقمي للمكون: {component_id}")
        return False

# -------------------- 8. Self-Healing System (نظام الإصلاح الذاتي) --------------------
class SelfHealingSystem:
    """نظام الإصلاح الذاتي التلقائي"""
    
    def __init__(self):
        self.healing_actions = []
    
    def diagnose_and_heal(self, sensor_data, anomalies):
        """تشخيص المشاكل وتطبيق الإصلاحات التلقائية"""
        healing_applied = False
        
        # إذا كان هناك شذوذ في درجة الحرارة
        if not anomalies.empty and 'temperature' in anomalies.columns:
            high_temp_anomalies = anomalies[anomalies['temperature'] > 65]
            if not high_temp_anomalies.empty:
                # تطبيق إجراء الإصلاح (محاكاة)
                healing_action = {
                    'timestamp': datetime.now(),
                    'issue': 'ارتفاع درجة الحرارة',
                    'action': 'تشغيل نظام التبريد الإضافي تلقائياً',
                    'result': 'success'
                }
                
                self.healing_actions.append(healing_action)
                healing_applied = True
                
                lifelong_memory.add_experience(
                    'self_healing',
                    'تم تطبيق إصلاح تلقائي لارتفاع درجة الحرارة',
                    healing_action
                )
                
                logger.info("تم تطبيق إصلاح تلقائي لارتفاع درجة الحرارة")
        
        return healing_applied

# -------------------- 9. Sustainability Analytics (تحليلات الاستدامة) --------------------
class SustainabilityMonitor:
    """مراقبة وتحليل استدامة العمليات"""
    
    def __init__(self):
        self.carbon_data = {}
    
    def calculate_carbon_footprint(self, sensor_data):
        """حساب البصمة الكربونية بناءً على بيانات الاستشعار"""
        try:
            # معادلة مبسطة لحساب البصمة الكربونية
            temp = sensor_data.get('temperature', 55)
            energy_consumption = temp * 0.5  # محاكاة استهلاك الطاقة
            
            carbon_footprint = energy_consumption * 0.8  # محاكاة انبعاثات الكربون
            
            sustainability_data = {
                'timestamp': datetime.now(),
                'energy_consumption': energy_consumption,
                'carbon_footprint': carbon_footprint,
                'efficiency_score': max(0, 100 - (temp - 55) * 2)
            }
            
            self.carbon_data = sustainability_data
            st.session_state.carbon_footprint = sustainability_data
            
            lifelong_memory.add_experience(
                'sustainability_calculation',
                'تم حساب مقاييس الاستدامة',
                sustainability_data
            )
            
            logger.info("تم حساب البصمة الكربونية ومقاييس الاستدامة")
            return sustainability_data
        
        except Exception as e:
            logger.error(f"خطأ في حساب البصمة الكربونية: {str(e)}")
            return None

# تهيئة الأنظمة الجديدة
digital_twin_optimizer = DigitalTwinOptimizer()
predictive_maintenance = PredictiveMaintenance()
emergency_response = EmergencyResponseSystem()
digital_thread = DigitalThread()
self_healing = SelfHealingSystem()
sustainability_monitor = SustainabilityMonitor()

# -------------------- المساعد الذكي --------------------
def generate_ai_response(prompt):
    """مساعد ذكي مدعوم بالذاكرة الدائمة وOpenAI"""
    prompt_lower = prompt.lower()
    
    # أولاً، محاولة استخدام OpenAI إذا كان متاحاً
    if st.session_state.get("openai_enabled", False):
        openai_response = generate_openai_response(prompt)
        if openai_response:
            lifelong_memory.add_experience(
                'ai_assistant',
                'استجابة من المساعد الذكي باستخدام OpenAI',
                {'prompt': prompt, 'response': openai_response}
            )
            return openai_response
    
    # إذا لم يكن OpenAI متاحاً، استخدام الردود الأساسية
    return generate_fallback_response(prompt_lower)

def generate_fallback_response(prompt_lower):
    """إنشاء رد عند عدم توفر OpenAI"""
    response = ""
    if any(word in prompt_lower for word in ["الطقس", "درجة الحرارة", "weather", "temperature"]):
        response += get_weather_info()
    elif any(word in prompt_lower for word in ["الوقت", "التاريخ", "اليوم", "time", "date", "today"]):
        response += get_current_time_info()
    elif any(word in prompt_lower for word in ["مرحبا", "السلام", "hello", "hi"]):
        response += "مرحباً! أنا المساعد الذكي لمنصة التوأم الرقمي. كيف يمكنني مساعدتك اليوم؟"
    elif any(word in prompt_lower for word in ["تنبأ", "توقع", "predict", "forecast"]):
        response += generate_time_based_prediction(prompt_lower)
    elif any(word in prompt_lower for word in ["حالي", "مباشر", "current", "now"]):
        response += generate_current_status()
    else:
        response += "أنا المساعد الذكي للمنصة. يمكنني مساعدتك في مراقبة المصنع، التنبؤات، الطقس، الوقت، وأسئلة عامة أخرى."
    
    lifelong_memory.add_experience(
        'ai_assistant',
        'استجابة من المساعد الذكي (الوضع الأساسي)',
        {'prompt': prompt_lower, 'response': response}
    )
    
    return response

def get_weather_info():
    """معلومات الطقس"""
    weather_data = {
        "temperature": random.randint(20, 35),
        "condition": random.choice(["مشمس", "غائم جزئياً", "صافي"]),
        "humidity": random.randint(30, 70)
    }
    return f"حالة الطقس الحالية:\n• درجة الحرارة: {weather_data['temperature']}°م\n• الحالة: {weather_data['condition']}\n• الرطوبة: {weather_data['humidity']}%"

def get_current_time_info():
    """معلومات الوقت والتاريخ"""
    now = datetime.now()
    return f"الوقت الحالي: {now.strftime('%H:%M:%S')}\nتاريخ اليوم: {now.strftime('%Y-%m-%d')}\nاليوم: {now.strftime('%A')}"

def generate_time_based_prediction(prompt):
    """إنشاء تنبؤات زمنية"""
    time_keywords = {"ساعة": 1, "ساعات": 1, "يوم": 24, "أيام": 24, "أسبوع": 168, "أسابيع": 168}
    hours_ahead = 2
    
    for keyword, hours in time_keywords.items():
        if keyword in prompt:
            hours_ahead = hours
            break
    
    # محاكاة تنبؤات بناءً على البيانات الحالية
    current_temp = st.session_state.get('mqtt_temp', 55)
    predicted_temp = current_temp + np.random.normal(0, 1.5)
    
    return f"التنبؤ لـ {hours_ahead} ساعة القادمة:\n• درجة الحرارة المتوقعة: {predicted_temp:.1f}°م\n• الضغط المتوقع: {st.session_state.get('pressure', 7.2) + np.random.normal(0, 0.2):.1f} بار\n• مستوى الميثان المتوقع: {max(0, st.session_state.get('methane', 1.4) + np.random.normal(0, 0.1)):.2f} ppm"

def generate_current_status():
    """الحالة الحالية للنظام"""
    return f"الحالة الحالية للنظام:\n• درجة الحرارة: {st.session_state.get('mqtt_temp', 55)}°م\n• الضغط: {st.session_state.get('pressure', 7.2)} بار\n• الميثان: {st.session_state.get('methane', 1.4)} ppm\n• آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}\n• صحة النظام: جيدة"

# -------------------- وظائف مساعدة --------------------
def to_arabic_numerals(num):
    """تحويل الأرقام إلى العربية"""
    return str(num).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def rtl_wrap(txt):
    """تفعيل النص من اليمين لليسار للغة العربية"""
    if st.session_state.get("lang", "ar") == "ar":
        return f'<div style="direction:rtl;text-align:right">{txt}</div>'
    else:
        return f'<div style="direction:ltr;text-align:left">{txt}</div>'

def show_logo():
    """عرض الشعار"""
    st.markdown(f'<div style="text-align:center;padding-bottom:1.2em;">{logo_svg}</div>', unsafe_allow_html=True)

def show_system_status_banner():
    """عرض لافتة حالة النظام"""
    status = st.session_state.get("system_status", "simulation")
    status_text = translator.get_text("system_status_simulation") if status == "simulation" else translator.get_text("system_status_real")
    status_class = "status-simulation" if status == "simulation" else "status-real"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-bottom: 1.5rem;">
        <div class="{status_class}">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

def show_notification_history():
    """عرض سجل الإشعارات"""
    notifications = st.session_state.get("notification_history", [])
    
    if notifications:
        st.markdown("#### 📋 سجل الإشعارات")
        
        for notification in reversed(notifications[-10:]):  # عرض آخر 10 إشعارات
            ntype = notification.get("type", "info")
            message = notification.get("message", "")
            timestamp = notification.get("timestamp", datetime.now()).strftime("%H:%M:%S")
            
            st.markdown(f"""
            <div class="notification {ntype}">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{message}</strong>
                    <span style="font-size: 0.8em; opacity: 0.7;">{timestamp}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------- بيانات العرض --------------------
np.random.seed(1)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})

# -------------------- أقسام التطبيق --------------------
def dashboard_section():
    """قسم لوحة التحكم"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[0]}</div>', unsafe_allow_html=True)
    
    show_system_status_banner()
    
    # بطاقات المقاييس الرئيسية
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('temperature')}</h3>
            <h2>{st.session_state.get('mqtt_temp', 55):.1f}°C</h2>
            <p>آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('pressure')}</h3>
            <h2>{st.session_state.get('pressure', 7.2):.1f} bar</h2>
            <p>الحالة: طبيعية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('methane')}</h3>
            <h2>{st.session_state.get('methane', 1.4):.2f} ppm</h2>
            <p>الحالة: آمنة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('flow_rate')}</h3>
            <h2>{st.session_state.get('flow_rate', 110):.0f} L/min</h2>
            <p>الكفاءة: 92%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # المخططات
    st.markdown("### 📈 المراقبة اللحظية")
    
    # إنشاء مخطط درجة الحرارة
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=demo_df['time'], 
        y=demo_df['Temperature'],
        mode='lines',
        name='درجة الحرارة',
        line=dict(color='#FF6D00', width=2)
    ))
    fig_temp.update_layout(
        title='تغير درجة الحرارة خلال 24 ساعة',
        xaxis_title='الوقت',
        yaxis_title='درجة الحرارة (°C)',
        height=400
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # مخططات أخرى
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pressure = go.Figure()
        fig_pressure.add_trace(go.Scatter(
            x=demo_df['time'], 
            y=demo_df['Pressure'],
            mode='lines',
            name='الضغط',
            line=dict(color='#1E88E5', width=2)
        ))
        fig_pressure.update_layout(
            title='تغير الضغط خلال 24 ساعة',
            xaxis_title='الوقت',
            yaxis_title='الضغط (bar)',
            height=300
        )
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col2:
        fig_methane = go.Figure()
        fig_methane.add_trace(go.Scatter(
            x=demo_df['time'], 
            y=demo_df['Methane'],
            mode='lines',
            name='الميثان',
            line=dict(color='#43A047', width=2)
        ))
        fig_methane.update_layout(
            title='تغير مستوى الميثان خلال 24 ساعة',
            xaxis_title='الوقت',
            yaxis_title='الميثان (ppm)',
            height=300
        )
        st.plotly_chart(fig_methane, use_container_width=True)
    
    # التنبيهات والإشعارات
    show_notification_history()

def analytics_ai_section():
    """قسم التحليلات والذكاء الاصطناعي"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[1]}</div>', unsafe_allow_html=True)
    
    # تحليل البيانات
    st.markdown("### 🔍 تحليل البيانات والكشف عن الشذوذ")
    
    if st.button("تشغيل التحليل", key="run_analysis"):
        with st.spinner("جاري تحليل البيانات والكشف عن الشذوذ..."):
            # توليد بيانات استشعار
            sensor_data = ai_analyzer.generate_sensor_data()
            
            # الكشف عن الشذوذ
            analyzed_data, anomalies = ai_analyzer.detect_anomalies(sensor_data)
            
            # تجميع البيانات
            clustered_data = ai_analyzer.cluster_data(analyzed_data)
            
            # التنبؤ بالاتجاهات
            predictions = ai_analyzer.predict_trends(clustered_data)
            
            # حفظ النتائج في حالة الجلسة
            st.session_state.analyzed_data = analyzed_data
            st.session_state.anomalies = anomalies
            st.session_state.clustered_data = clustered_data
            st.session_state.predictions = predictions
            st.session_state.ai_analysis_done = True
            
            st.success("تم تحليل البيانات بنجاح!")
    
    if st.session_state.get("ai_analysis_done", False):
        analyzed_data = st.session_state.analyzed_data
        anomalies = st.session_state.anomalies
        clustered_data = st.session_state.clustered_data
        predictions = st.session_state.predictions
        
        # عرض نتائج التحليل
        st.markdown("#### نتائج التحليل")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("إجمالي نقاط البيانات", len(analyzed_data))
            st.metric("نقاط الشذوذ المكتشفة", len(anomalies))
        
        with col2:
            st.metric("عدد المجموعات", clustered_data['cluster'].nunique())
            st.metric("فترات التنبؤ", len(predictions) if predictions is not None else 0)
        
        # مخطط الشذوذ
        st.markdown("##### نقاط الشذوذ في درجة الحرارة")
        if not anomalies.empty:
            fig_anomalies = px.scatter(
                anomalies, 
                x='timestamp', 
                y='temperature',
                color='anomaly',
                title='نقاط الشذوذ في درجة الحرارة'
            )
            st.plotly_chart(fig_anomalies, use_container_width=True)
        else:
            st.info("لم يتم اكتشاف أي شذوذ في البيانات")
        
        # مخطط التجمعات
        st.markdown("##### تجميع بيانات الاستشعار")
        fig_clusters = px.scatter(
            clustered_data,
            x='temperature',
            y='pressure',
            color='cluster',
            title='تجميع بيانات درجة الحرارة والضغط'
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # التنبؤات
        if not predictions.empty:
            st.markdown("##### تنبؤات درجة الحرارة المستقبلية")
            fig_predictions = px.line(
                predictions,
                x='timestamp',
                y='temperature_pred',
                title='تنبؤات درجة الحرارة للـ 6 ساعات القادمة'
            )
            st.plotly_chart(fig_predictions, use_container_width=True)
    
    # الصيانة التنبؤية
    st.markdown("### 🔮 الصيانة التنبؤية")
    
    if st.button("توليد تنبؤات الصيانة", key="generate_maintenance"):
        with st.spinner("جاري تحليل بيانات الصيانة..."):
            prediction = predictive_maintenance.predict_maintenance(
                st.session_state.current_sensor_data
            )
            
            if prediction:
                st.session_state.maintenance_prediction = prediction
                
                # عرض نتائج الصيانة
                st.markdown("#### نتائج تحليل الصيانة")
                
                priority_colors = {
                    "high": "#E53935",
                    "medium": "#FF6D00",
                    "low": "#43A047"
                }
                
                priority = prediction['priority']
                color = priority_colors.get(priority, "#000000")
                
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 8px; border-left: 4px solid {color}; background-color: {color}20;">
                    <h4 style="margin: 0; color: {color};">{prediction['message']}</h4>
                    <p>الأولوية: <strong>{priority}</strong></p>
                    <p>الساعات المتبقية للصيانة: <strong>{prediction['hours_until_maintenance']:.1f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("فشل في توليد تنبؤات الصيانة")

def operations_section():
    """قسم العمليات"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[2]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🏭 إدارة العمليات")
    
    # محاكاة بيانات العمليات
    operations_data = {
        "الإنتاجية": {"القيمة": 87, "الوحدة": "%", "الاتجاه": "↑"},
        "الكفاءة": {"القيمة": 92, "الوحدة": "%", "الاتجاه": "→"},
        "الجودة": {"القيمة": 94, "الوحدة": "%", "الاتجاه": "↑"},
        "الهدر": {"القيمة": 5.2, "الوحدة": "%", "الاتجاه": "↓"}
    }
    
    # عرض مقاييس العمليات
    cols = st.columns(4)
    for i, (metric, data) in enumerate(operations_data.items()):
        with cols[i]:
            trend_icon = {"↑": "📈", "↓": "📉", "→": "➡️"}.get(data["الاتجاه"], "➡️")
            st.markdown(f"""
            <div class="metric-card">
                <h3>{metric}</h3>
                <h2>{data['القيمة']}{data['الوحدة']} {trend_icon}</h2>
                <p>الاتجاه: {data['الاتجاه']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # تحسين التوأم الرقمي
    st.markdown("### ⚙️ تحسين أداء التوأم الرقمي")
    
    if st.button("تحسين المعاملات", key="optimize_params"):
        with st.spinner("جاري تحسين معاملات التشغيل..."):
            optimization = digital_twin_optimizer.optimize_parameters(
                st.session_state.current_sensor_data
            )
            
            if optimization:
                st.session_state.last_optimization = optimization
                
                st.markdown("#### نتائج التحسين")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "درجة الحرارة الحالية", 
                        f"{optimization['original_temp']:.1f}°C",
                        f"{optimization['optimized_temp'] - optimization['original_temp']:.1f}°C"
                    )
                
                with col2:
                    st.metric(
                        "الضغط الحالي", 
                        f"{optimization['original_pressure']:.1f} bar",
                        f"{optimization['optimized_pressure'] - optimization['original_pressure']:.1f} bar"
                    )
                
                st.metric(
                    "التوفير المتوقع في الطاقة",
                    f"{optimization['estimated_savings']:.2f}%"
                )
                
                st.success("تم تحسين معاملات التشغيل بنجاح!")
            else:
                st.error("فشل في تحسين معاملات التشغيل")
    
    # الخيط الرقمي
    st.markdown("### 🧵 إدارة الخيوط الرقمية")
    
    if st.button("إنشاء خيط رقمي جديد", key="create_digital_thread"):
        component_id = f"comp_{int(time.time())}"
        component_data = {
            "نوع": "مولد",
            "الطراز": "GEN-5000",
           الحالة": "نشط",
            "الموقع": "الخط 2"
        }
        
        digital_thread.create_thread(component_id, component_data)
        st.success(f"تم إنشاء الخيط الرقمي للمكون {component_id}")

def hardware_control_section():
    """قسم التحكم بالأجهزة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[3]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🤖 التحكم بالمعدات والأجهزة")
    
    # حالة الاتصال بـ Raspberry Pi
    pi_status = st.session_state.get("pi_status", "disconnected")
    status_text = "متصل" if pi_status == "connected" else "محاكاة" if pi_status == "simulated" else "غير متصل"
    
    st.markdown(f"**حالة وحدة التحكم:** {status_text}")
    
    # عناصر التحكم
    st.markdown("#### عناصر التحكم")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("تشغيل المضخة الرئيسية", key="pump_on"):
            result = real_pi_controller.set_output(18, True)  # GPIO 18
            if result:
                st.success("تم تشغيل المضخة الرئيسية")
                lifelong_memory.add_experience(
                    'control_action',
                    'تم تشغيل المضخة الرئيسية',
                    {'component': 'main_pump', 'action': 'on'}
                )
            else:
                st.error("فشل في تشغيل المضخة")
        
        if st.button("إيقاف المضخة الرئيسية", key="pump_off"):
            result = real_pi_controller.set_output(18, False)  # GPIO 18
            if result:
                st.success("تم إيقاف المضخة الرئيسية")
                lifelong_memory.add_experience(
                    'control_action',
                    'تم إيقاف المضخة الرئيسية',
                    {'component': 'main_pump', 'action': 'off'}
                )
            else:
                st.error("فشل في إيقاف المضخة")
    
    with col2:
        if st.button("تشغيل النظام الاحتياطي", key="backup_on"):
            result = real_pi_controller.set_output(23, True)  # GPIO 23
            if result:
                st.success("تم تشغيل النظام الاحتياطي")
                lifelong_memory.add_experience(
                    'control_action',
                    'تم تشغيل النظام الاحتياطي',
                    {'component': 'backup_system', 'action': 'on'}
                )
            else:
                st.error("فشل في تشغيل النظام الاحتياطي")
        
        if st.button("إيقاف النظام الاحتياطي", key="backup_off"):
            result = real_pi_controller.set_output(23, False)  # GPIO 23
            if result:
                st.success("تم إيقاف النظام الاحتياطي")
                lifelong_memory.add_experience(
                    'control_action',
                    'تم إيقاف النظام الاحتياطي',
                    {'component': 'backup_system', 'action': 'off'}
                )
            else:
                st.error("فشل في إيقاف النظام الاحتياطي")
    
    # قراءة مدخلات
    st.markdown("#### قراءة حالة المستشعرات")
    
    if st.button("قراءة حالة المستشعرات", key="read_sensors"):
        # محاكاة قراءة المستشعرات
        sensor1 = real_pi_controller.read_input(24)  # GPIO 24
        sensor2 = real_pi_controller.read_input(25)  # GPIO 25
        
        col1, col2 = st.columns(2)
        
        with col1:
            status = "نشط" if sensor1 == 1 else "غير نشط" if sensor1 == 0 else "غير متوفر"
            st.markdown(f"**المستشعر 1:** {status}")
        
        with col2:
            status = "نشط" if sensor2 == 1 else "غير نشط" if sensor2 == 0 else "غير متوفر"
            st.markdown(f"**المستشعر 2:** {status}")
        
        lifelong_memory.add_experience(
            'sensor_reading',
            'تم قراءة حالة المستشعرات',
            {'sensor1': sensor1, 'sensor2': sensor2}
        )

def safety_alerts_section():
    """قسم السلامة والتنبيهات"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[4]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🛡️ نظام السلامة والإنذار")
    
    # فحص الطوارئ
    if st.button("فحص حالات الطوارئ", key="check_emergencies"):
        emergencies = emergency_response.check_emergency(
            st.session_state.current_sensor_data
        )
        
        if emergencies:
            for emergency in emergencies:
                protocol = emergency_response.protocols[emergency]
                st.error(f"تحذير: {protocol['name']}")
                st.write("الإجراءات المطلوبة:")
                for action in protocol['actions']:
                    st.write(f"- {action}")
        else:
            st.success("لا توجد حالات طوارئ حالياً")
    
    # إعدادات التنبيهات
    st.markdown("#### إعدادات نظام التنبيهات")
    
    st.session_state.twilio_enabled = st.checkbox(
        "تفعيل تنبيهات SMS",
        value=st.session_state.twilio_enabled,
        key="twilio_enable"
    )
    
    st.session_state.alert_phone_number = st.text_input(
        "رقم الهاتف للتنبيهات",
        value=st.session_state.alert_phone_number,
        key="alert_phone"
    )
    
    # اختبار التنبيه
    if st.button("إرسال تنبيه اختباري", key="test_alert"):
        if st.session_state.twilio_enabled and st.session_state.alert_phone_number:
            success = send_twilio_alert(
                "هذا تنبيه اختباري من منصة التوأم الرقمي العصبي الذكي",
                st.session_state.alert_phone_number
            )
            
            if success:
                st.success("تم إرسال التنبيه الاختباري بنجاح")
            else:
                st.error("فشل في إرسال التنبيه الاختباري")
        else:
            st.warning("يجب تفعيل تنبيهات SMS وإدخال رقم هاتف صحيح")
    
    # نظام الإصلاح الذاتي
    st.markdown("### 🔧 نظام الإصلاح الذاتي")
    
    if st.button("فحص وعلاج الأعطال تلقائياً", key="self_heal"):
        if st.session_state.get("anomalies") is not None:
            healing_applied = self_healing.diagnose_and_heal(
                st.session_state.current_sensor_data,
                st.session_state.anomalies
            )
            
            if healing_applied:
                st.success("تم تطبيق إصلاح تلقائي بنجاح")
            else:
                st.info("لا توجد أعطال تحتاج إلى إصلاح تلقائي")
        else:
            st.warning("يجب تشغيل تحليل البيانات أولاً من قسم التحليلات")

def sustainability_section():
    """قسم الاستدامة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[5]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌱 تحليلات الاستدامة")
    
    # حساب البصمة الكربونية
    if st.button("حساب البصمة الكربونية", key="calculate_carbon"):
        carbon_data = sustainability_monitor.calculate_carbon_footprint(
            st.session_state.current_sensor_data
        )
        
        if carbon_data:
            st.session_state.carbon_data = carbon_data
            
            st.markdown("#### نتائج تحليل الاستدامة")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "استهلاك الطاقة",
                    f"{carbon_data['energy_consumption']:.1f} kWh",
                    "2.5% أقل من الشهر الماضي"
                )
            
            with col2:
                st.metric(
                    "البصمة الكربونية",
                    f"{carbon_data['carbon_footprint']:.1f} kg CO₂",
                    "3.1% أقل من الشهر الماضي"
                )
            
            with col3:
                st.metric(
                    "معدل الكفاءة",
                    f"{carbon_data['efficiency_score']:.1f}%",
                    "1.8% أفضل من الشهر الماضي"
                )
            
            # مخطط البصمة الكربونية
            carbon_history = pd.DataFrame({
                'الشهر': ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو'],
                'البصمة الكربونية': [45.2, 43.8, 42.1, 40.5, 39.2, carbon_data['carbon_footprint']]
            })
            
            fig_carbon = px.line(
                carbon_history,
                x='الشهر',
                y='البصمة الكربونية',
                title='تطور البصمة الكربونية خلال الأشهر الستة الماضية',
                markers=True
            )
            st.plotly_chart(fig_carbon, use_container_width=True)
            
            st.success("تم حساب مقاييس الاستدامة بنجاح!")
        else:
            st.error("فشل في حساب مقاييس الاستدامة")
    
    # نصائح لتحسين الاستدامة
    st.markdown("#### 💡 نصائح لتحسين الاستدامة")
    
    tips = [
        "تحسين عزل الأنابيب لتقليل فقدان الحرارة",
        "برمجة أوقات التشغيل لتتوافق مع فترات انخفاض استهلاك الطاقة",
        "الصيانة الدورية للمعدات لضمان الكفاءة القصوى",
        "استخدام مصادر الطاقة المتجددة حيثما أمكن",
        "إعادة تدوير المياه المستخدمة في عمليات التبريد"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.markdown(f"{i}. {tip}")

def settings_help_section():
    """قسم الإعدادات والمساعدة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[6]}</div>', unsafe_allow_html=True)
    
    # إعدادات اللغة والمظهر
    st.markdown("### ⚙️ الإعدادات")
    
    # منتقي اللغة
    st.markdown(f"**{translator.get_text('language')}**")
    current_lang = st.session_state.get("lang", "ar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("عربي", key="lang_ar", use_container_width=True):
            translator.set_language("ar")
            st.rerun()
    
    with col2:
        if st.button("English", key="lang_en", use_container_width=True):
            translator.set_language("en")
            st.rerun()
    
    # منتقي الثيم
    st.markdown(f"**{translator.get_text('theme')}**")
    current_theme = st.session_state.get("theme", "light")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(translator.get_text("light"), key="theme_light", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    
    with col2:
        if st.button(translator.get_text("dark"), key="theme_dark", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    
    # إعدادات OpenAI
    st.markdown("#### إعدادات الذكاء الاصطناعي")
    
    api_key = st.text_input(
        "مفتاح OpenAI API",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help="أدخل مفتاح OpenAI API لتمكين الميزات المتقدمة للمساعد الذكي"
    )
    
    if st.button("حفظ مفتاح API", key="save_api_key"):
        if api_key:
            success = init_openai(api_key)
            if success:
                st.success("تم حفظ مفتاح API بنجاح وتمكين ميزات الذكاء الاصطناعي المتقدمة")
            else:
                st.error("فشل في تهيئة OpenAI. يرجى التحقق من المفتاح والمحاولة مرة أخرى")
        else:
            st.session_state.openai_enabled = False
            st.info("تم تعطيل ميزات الذكاء الاصطناعي المتقدمة")
    
    # المساعد الذكي
    st.markdown("### 💬 المساعد الذكي")
    
    # عرض سجل المحادثة
    chat_history = st.session_state.get("chat_history", [])
    
    for message in chat_history:
        if message["role"] == "user":
            st.markdown(f"**أنت:** {message['content']}")
        else:
            st.markdown(f"**المساعد:** {message['content']}")
    
    # مدخل الرسالة الجديدة
    user_input = st.text_input("اكتب رسالتك هنا...", key="chat_input")
    
    if st.button("إرسال", key="send_message") and user_input:
        # إضافة رسالة المستخدم إلى السجل
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # توليد الرد
        with st.spinner("جاري توليد الرد..."):
            response = generate_ai_response(user_input)
            
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.error("فشل في توليد الرد. يرجى المحاولة مرة أخرى")
    
    # معلومات النظام
    st.markdown("### ℹ️ معلومات النظام")
    
    st.markdown(f"""
    - **إصدار المنصة:** 2.1.0
    - **حالة اتصال MQTT:** {'متصل' if st.session_state.mqtt_connected else 'غير متصل'}
    - **حالة اتصال Raspberry Pi:** {st.session_state.pi_status}
    - **وضع التشغيل:** {'أجهزة حقيقية' if st.session_state.system_status == 'real_hardware' else 'محاكاة'}
    - **عدد التجارب في الذاكرة:** {len(st.session_state.lifelong_memory)}
    - **وقت التشغيل:** {(datetime.now() - st.session_state.mqtt_last).seconds // 60} دقيقة
    """)
    
    # أزرار التحكم بالنظام
    st.markdown("#### أدوات التحكم بالنظام")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("اختبار ذاتي للنظام", key="system_self_test"):
            with st.spinner("جاري الاختبار الذاتي للنظام..."):
                # محاكاة الاختبار الذاتي
                time.sleep(2)
                
                test_results = {
                    "الاتصال بالإنترنت": "ناجح",
                    "اتصال MQTT": "ناجح" if st.session_state.mqtt_connected else "فاشل",
                    "قاعدة البيانات": "ناجح",
                    "خدمة الذكاء الاصطناعي": "ناجح" if st.session_state.openai_enabled else "معطل",
                    "خدمة التنبيهات": "ناجح" if st.session_state.twilio_enabled else "معطل"
                }
                
                st.session_state.self_test_results = test_results
                st.success("تم الانتهاء من الاختبار الذاتي")
    
    with col2:
        if st.button("تصفير الذاكرة", key="clear_memory"):
            st.session_state.lifelong_memory = []
            st.session_state.chat_history = []
            st.session_state.notification_history = []
            st.success("تم تصفير الذاكرة وسجل المحادثة")
    
    # عرض نتائج الاختبار الذاتي إذا كانت متاحة
    if st.session_state.self_test_results:
        st.markdown("##### نتائج الاختبار الذاتي")
        
        for test, result in st.session_state.self_test_results.items():
            color = "green" if result == "ناجح" else "orange" if result == "معطل" else "red"
            st.markdown(f"- **{test}:** :{color}[{result}]")

# -------------------- التطبيق الرئيسي --------------------
def main():
    # تطبيق أنماط الثيم
    theme_manager.apply_theme_styles()
    
    # الشريط الجانبي
    with st.sidebar:
        show_logo()
        
        st.markdown(f"### {translator.get_text('app_title')}")
        st.markdown(f"_{translator.get_text('app_sub')}_")
        
        # منتقي اللغة في الشريط الجانبي
        st.markdown("---")
        st.markdown(f"**{translator.get_text('language')}**")
        
        lang_col1, lang_col2 = st.columns(2)
        with lang_col1:
            if st.button("عربي", key="sidebar_lang_ar", use_container_width=True):
                translator.set_language("ar")
                st.rerun()
        with lang_col2:
            if st.button("English", key="sidebar_lang_en", use_container_width=True):
                translator.set_language("en")
                st.rerun()
        
        # منتقي الثيم في الشريط الجانبي
        st.markdown(f"**{translator.get_text('theme')}**")
        
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button(translator.get_text("light"), key="sidebar_theme_light", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
        with theme_col2:
            if st.button(translator.get_text("dark"), key="sidebar_theme_dark", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()
        
        st.markdown("---")
        
        # قائمة الأقسام
        section = st.radio(
            "اختر القسم:",
            translator.get_text("side_sections"),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # معلومات إضافية في الشريط الجانبي
        st.markdown("#### حالة النظام")
        status = st.session_state.get("system_status", "simulation")
        status_text = translator.get_text("system_status_simulation") if status == "simulation" else translator.get_text("system_status_real")
        
        st.markdown(f"**الحالة:** {status_text}")
        st.markdown(f"**اللغة:** {'العربية' if st.session_state.get('lang', 'ar') == 'ar' else 'English'}")
        st.markdown(f"**المظهر:** {'الفاتح' if st.session_state.get('theme', 'light') == 'light' else 'الداكن'}")
        
        # زر تسجيل الخروج
        st.markdown("---")
        if st.button(translator.get_text("logout"), key="logout_btn", use_container_width=True):
            st.session_state.user_authenticated = False
            st.success("تم تسجيل الخروج بنجاح")
            st.rerun()
    
    # عرض القسم المحدد
    sections = {
        translator.get_text("side_sections")[0]: dashboard_section,
        translator.get_text("side_sections")[1]: analytics_ai_section,
        translator.get_text("side_sections")[2]: operations_section,
        translator.get_text("side_sections")[3]: hardware_control_section,
        translator.get_text("side_sections")[4]: safety_alerts_section,
        translator.get_text("side_sections")[5]: sustainability_section,
        translator.get_text("side_sections")[6]: settings_help_section
    }
    
    if section in sections:
        sections[section]()

if __name__ == "__main__":
    main()
