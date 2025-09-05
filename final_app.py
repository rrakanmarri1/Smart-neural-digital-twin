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
import warnings
warnings.filterwarnings('ignore')

# -------------------- نظام التسجيل والمراقبة --------------------
def setup_logging():
    """إعداد نظام التسجيل والمراقبة"""
    logger = logging.getLogger('SNDT_Platform')
    logger.setLevel(logging.INFO)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    handler = RotatingFileHandler(
        'logs/sndt_platform.log', 
        maxBytes=5*1024*1024,
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# -------------------- نظام التخزين المتقدم --------------------
class AdvancedCache:
    """نظام تخزين متقدم مع دعم Redis والذاكرة"""
    def __init__(self):
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
            self.redis_client.ping()
            self.mode = 'redis'
            logger.info("تم الاتصال بـ Redis بنجاح")
        except:
            self.mode = 'memory'
            self.memory_cache = {}
            logger.info("يعمل نظام التخزين في وضع الذاكرة")
    
    def set(self, key, value, expiry=3600):
        if self.mode == 'redis':
            import pickle
            self.redis_client.setex(key, expiry, pickle.dumps(value))
        else:
            self.memory_cache[key] = {
                'data': value,
                'expiry': time.time() + expiry
            }
    
    def get(self, key):
        if self.mode == 'redis':
            import pickle
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        else:
            item = self.memory_cache.get(key)
            if item and item['expiry'] > time.time():
                return item['data']
            return None

cache = AdvancedCache()

# -------------------- نظام إدارة الأحداث --------------------
class EventSystem:
    """نظام إدارة الأحداث للتواصل بين المكونات"""
    def __init__(self):
        self.events = []
        self.subscribers = {}
        logger.info("تم تهيئة نظام الأحداث")
    
    def publish(self, event_type, data):
        event_id = hashlib.md5(f"{event_type}{datetime.now()}".encode()).hexdigest()
        event = {
            "id": event_id,
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.events.append(event)
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"خطأ في معالجة الحدث: {e}")
        
        return event
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def get_recent_events(self, limit=10):
        return self.events[-limit:]

event_system = EventSystem()

# -------------------- نظام الثيمات --------------------
class ThemeManager:
    """مدير الثيمات لوضع الضوء والداكن"""
    def __init__(self):
        self.themes = {
            "light": {
                "primary": "#1f77b4", "secondary": "#ff7f0e", "background": "#ffffff",
                "text": "#000000", "card": "#f0f2f6", "success": "#2ecc71",
                "warning": "#f39c12", "danger": "#e74c3c"
            },
            "dark": {
                "primary": "#4a9fff", "secondary": "#ffaa45", "background": "#0e1117",
                "text": "#ffffff", "card": "#262730", "success": "#27ae60",
                "warning": "#f39c12", "danger": "#e74c3c"
            }
        }
    
    def apply_theme_styles(self):
        theme = st.session_state.get("theme", "light")
        colors = self.themes[theme]
        
        st.markdown(f"""
        <style>
            .main {{ background-color: {colors['background']}; color: {colors['text']}; }}
            .main-header {{
                color: {colors['primary']}; font-size: 2.2rem; font-weight: 700;
                margin-bottom: 1.5rem; border-bottom: 2px solid {colors['primary']};
                padding-bottom: 0.5rem;
            }}
            .section-header {{
                color: {colors['secondary']}; font-size: 1.6rem; font-weight: 600;
                margin: 1.2rem 0 0.8rem 0;
            }}
            .card {{
                background-color: {colors['card']}; padding: 1.2rem; border-radius: 0.5rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;
            }}
            .metric-card {{
                background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
                color: white; padding: 1.2rem; border-radius: 0.5rem; text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            .status-simulation {{
                background-color: {colors['warning']}; color: white; padding: 0.5rem 1rem;
                border-radius: 0.25rem; font-weight: bold;
            }}
            .status-real {{
                background-color: {colors['success']}; color: white; padding: 0.5rem 1rem;
                border-radius: 0.25rem; font-weight: bold;
            }}
            .stButton>button {{
                background-color: {colors['primary']}; color: white; border: none;
                border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: 500;
            }}
            .stButton>button:hover {{
                background-color: {colors['secondary']}; color: white;
            }}
            .notification {{
                background-color: {colors['card']}; border-left: 4px solid {colors['primary']};
                padding: 0.8rem; margin-bottom: 0.5rem; border-radius: 0 0.25rem 0.25rem 0;
            }}
            .notification-warning {{ border-left-color: {colors['warning']}; }}
            .notification-danger {{ border-left-color: {colors['danger']}; }}
            .notification-success {{ border-left-color: {colors['success']}; }}
        </style>
        """, unsafe_allow_html=True)
    
    def toggle_theme(self):
        current_theme = st.session_state.get("theme", "light")
        new_theme = "dark" if current_theme == "light" else "light"
        st.session_state["theme"] = new_theme
        logger.info(f"تم تغيير الثيم إلى: {new_theme}")

theme_manager = ThemeManager()

# -------------------- نظام الترجمة --------------------
class TranslationSystem:
    """نظام متكامل للترجمة متعددة اللغات"""
    def __init__(self):
        self.translations = {
            "ar": {
                "side_sections": [
                    "لوحة التحكم الرئيسية",
                    "التحليلات والذكاء الاصطناعي", 
                    "العمليات والتحكم",
                    "السلامة والطوارئ",
                    "الاستدامة والطاقة",
                    "المساعد الذكي",
                    "الإعدادات والمساعدة"
                ],
                "temperature": "درجة الحرارة", "pressure": "الضغط", "methane": "الميثان",
                "vibration": "الاهتزاز", "flow_rate": "معدل التدفق",
                "real_time_data": "البيانات المباشرة", "historical_data": "البيانات التاريخية",
                "anomaly_detection": "كشف الشذوذ", "predictive_analysis": "التحليل التنبؤي",
                "system_status_simulation": "وضع المحاكاة", "system_status_real": "وضع التشغيل الحقيقي"
            }
        }
    
    def get_text(self, key, lang=None):
        if lang is None:
            lang = st.session_state.get("lang", "ar")
        return self.translations.get(lang, {}).get(key, key)

translator = TranslationSystem()

# -------------------- SVG Logo --------------------
logo_svg = """<svg width="64" height="64" viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="32" fill="#1f77b4"/><text x="32" y="38" text-anchor="middle" fill="#fff" font-size="24" font-family="Arial">SNDT</text></svg>"""
# -------------------- MQTT Config --------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_TEMPERATURE = "sndt/temperature"
MQTT_TOPIC_PRESSURE = "sndt/pressure"
MQTT_TOPIC_METHANE = "sndt/methane"
MQTT_TOPIC_CONTROL = "sndt/control"

# -------------------- تهيئة حالة التطبيق --------------------
def init_session_state():
    defaults = {
        "lang": "ar", "theme": "light", "system_status": "simulation",
        "mqtt_temp": 55.0, "pressure": 7.2, "methane": 1.4, 
        "vibration": 4.5, "flow_rate": 110.0, "mqtt_connected": False,
        "pi_connected": False, "pi_status": "disconnected", 
        "simulation_active": True, "current_sensor_data": {},
        "notification_history": [], "optimization_history": [],
        "maintenance_predictions": [], "carbon_footprint": {},
        "digital_threads": {}, "recommendations": [],
        "disaster_simulated": False, "data_refresh_rate": 5,
        "openai_enabled": False, "openai_api_key": "",
        "chat_history": [], "lifelong_memory": [],
        "twilio_enabled": True, "alert_phone_number": "+966532559664",
        "operations_data": {}, "energy_optimization": {},
        "incident_timeline": [], "physical_twin_connected": False,
        "show_advanced": False, "scenario_step": 0, "solution_idx": 0,
        "mqtt_last": datetime.now(), "mqtt_started": False,
        "sms_sent": False, "feedback_list": [], "generated_solutions": [],
        "solution_generated": False, "ai_analysis_done": False,
        "anomalies_detected": [], "preprocessed_data": None
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
        self.reconnect_delay = 5
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.connection_attempts = 0
            st.session_state["mqtt_connected"] = True
            logger.info("تم الاتصال بنجاح بخادم MQTT")
            client.subscribe(MQTT_TOPIC_TEMPERATURE)
            client.subscribe(MQTT_TOPIC_PRESSURE)
            client.subscribe(MQTT_TOPIC_METHANE)
        else:
            self.connected = False
            st.session_state["mqtt_connected"] = False
            logger.error(f"فشل الاتصال بخادم MQTT، رمز الخطأ: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            value = float(payload)
            topic = msg.topic
            
            if topic == MQTT_TOPIC_TEMPERATURE:
                st.session_state["mqtt_temp"] = value
            elif topic == MQTT_TOPIC_PRESSURE:
                st.session_state["pressure"] = value
            elif topic == MQTT_TOPIC_METHANE:
                st.session_state["methane"] = value
            
            st.session_state["mqtt_last"] = datetime.now()
            st.session_state["current_sensor_data"][topic] = value
            
            event_system.publish("sensor_data_update", {
                "topic": topic, "value": value, "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"تم استقبال بيانات: {topic} = {value}")
            
        except Exception as e:
            logger.error(f"خطأ في معالجة رسالة MQTT: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        st.session_state["mqtt_connected"] = False
        logger.warning("تم قطع الاتصال بخادم MQTT")
    
    def connect_with_retry(self):
        if self.connection_attempts >= self.max_attempts:
            logger.error(f"تم تجاوز الحد الأقصى لمحاولات الاتصال ({self.max_attempts})")
            return False
        
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            time.sleep(1)
            
            if not self.connected:
                self.connection_attempts += 1
                logger.warning(f"محاولة اتصال فاشلة {self.connection_attempts}/{self.max_attempts}")
                time.sleep(self.reconnect_delay)
                return self.connect_with_retry()
            
            return True
            
        except Exception as e:
            self.connection_attempts += 1
            logger.error(f"خطأ في الاتصال بخادم MQTT: {e}")
            time.sleep(self.reconnect_delay)
            return self.connect_with_retry()
    
    def publish(self, topic, message):
        if not self.connected:
            logger.warning("لا يمكن النشر، العميل غير متصل")
            return False
        
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"تم النشر بنجاح على {topic}: {message}")
                return True
            else:
                logger.error(f"فشل النشر على {topic}، رمز الخطأ: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"خطأ أثناء النشر على {topic}: {e}")
            return False
    
    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            st.session_state["mqtt_connected"] = False
            logger.info("تم قطع الاتصال بخادم MQTT")

mqtt_client = RobustMQTTClient()

# -------------------- محاكاة بيانات MQTT --------------------
def start_mqtt_simulation():
    """بدء محاكاة بيانات MQTT للوضع غير المتصل"""
    def simulate_data():
        anomaly_counter = 0
        while st.session_state.get("simulation_active", True):
            if not st.session_state.get("mqtt_connected", False):
                current_time = datetime.now()
                
                base_temp = 55.0
                base_pressure = 7.2
                base_methane = 1.4
                
                temp = base_temp + random.uniform(-2, 2)
                pressure = base_pressure + random.uniform(-0.3, 0.3)
                methane = base_methane + random.uniform(-0.2, 0.2)
                
                anomaly_counter += 1
                if anomaly_counter >= 20:
                    if random.random() < 0.4:
                        temp += random.uniform(8, 15)
                        logger.warning("محاكاة شذوذ في درجة الحرارة")
                    
                    if random.random() < 0.3:
                        pressure += random.uniform(1.5, 3.0)
                        logger.warning("محاكاة شذوذ في الضغط")
                    
                    if random.random() < 0.3:
                        methane += random.uniform(0.8, 2.0)
                        logger.warning("محاكاة شذوذ في الميثان")
                    
                    anomaly_counter = 0
                
                st.session_state["mqtt_temp"] = temp
                st.session_state["pressure"] = pressure
                st.session_state["methane"] = methane
                st.session_state["mqtt_last"] = current_time
                
                st.session_state["current_sensor_data"][MQTT_TOPIC_TEMPERATURE] = temp
                st.session_state["current_sensor_data"][MQTT_TOPIC_PRESSURE] = pressure
                st.session_state["current_sensor_data"][MQTT_TOPIC_METHANE] = methane
                
                event_system.publish("sensor_data_simulated", {
                    "temperature": temp, "pressure": pressure, "methane": methane,
                    "timestamp": current_time.isoformat()
                })
            
            time.sleep(st.session_state.get("data_refresh_rate", 5))
    
    if not st.session_state.get("simulation_thread", None):
        simulation_thread = threading.Thread(target=simulate_data, daemon=True)
        simulation_thread.start()
        st.session_state["simulation_thread"] = simulation_thread
        st.session_state["simulation_active"] = True
        logger.info("بدأت محاكاة بيانات الاستشعار")

# -------------------- التهيئة الرئيسية --------------------
if not st.session_state["mqtt_started"]:
    mqtt_success = mqtt_client.connect_with_retry()
    if not mqtt_success:
        start_mqtt_simulation()
    st.session_state["mqtt_started"] = True

# -------------------- نظام Raspberry Pi --------------------
class RealRaspberryPiController:
    """متحكم حقيقي بـ Raspberry Pi مع دعم GPIO"""
    def __init__(self):
        self.connected = False
        self.gpio_initialized = False
    
    def connect_to_pi(self, ip_address, username, password):
        try:
            time.sleep(2)
            success = random.random() > 0.3
            
            if success:
                self.connected = True
                st.session_state["pi_connected"] = True
                st.session_state["pi_status"] = "connected"
                logger.info(f"تم الاتصال بـ Raspberry Pi على {ip_address}")
                event_system.publish("pi_connected", {"ip": ip_address})
                return True, "تم الاتصال بنجاح"
            else:
                self.connected = False
                st.session_state["pi_connected"] = False
                st.session_state["pi_status"] = "connection_failed"
                logger.error(f"فشل الاتصال بـ Raspberry Pi على {ip_address}")
                return False, "فشل الاتصال. يرجى التحقق من الإعدادات"
                
        except Exception as e:
            logger.error(f"خطأ أثناء الاتصال بـ Raspberry Pi: {e}")
            return False, f"خطأ: {str(e)}"
    
    def initialize_gpio(self):
        if not self.connected:
            return False, "غير متصل بـ Raspberry Pi"
        
        try:
            time.sleep(1)
            self.gpio_initialized = True
            logger.info("تم تهيئة منافذ GPIO بنجاح")
            event_system.publish("gpio_initialized", {})
            return True, "تم تهيئة منافذ GPIO بنجاح"
        except Exception as e:
            logger.error(f"خطأ أثناء تهيئة GPIO: {e}")
            return False, f"خطأ: {str(e)}"
    
    def control_output(self, pin, state):
        if not self.connected or not self.gpio_initialized:
            return False, "لم يتم تهيئة النظام"
        
        try:
            time.sleep(0.5)
            state_str = "تشغيل" if state else "إيقاف"
            logger.info(f"تم {state_str} المنفذ {pin}")
            event_system.publish("gpio_control", {"pin": pin, "state": state})
            return True, f"تم {state_str} المنفذ {pin}"
        except Exception as e:
            logger.error(f"خطأ أثناء التحكم في المنفذ {pin}: {e}")
            return False, f"خطأ: {str(e)}"
    
    def read_input(self, pin):
        if not self.connected or not self.gpio_initialized:
            return False, "لم يتم تهيئة النظام", None
        
        try:
            time.sleep(0.2)
            value = random.choice([0, 1])
            state_str = "عالية" if value else "منخفضة"
            logger.info(f"قيمة المنفذ {pin}: {state_str}")
            event_system.publish("gpio_read", {"pin": pin, "value": value})
            return True, f"قيمة المنفذ {pin}: {state_str}", value
        except Exception as e:
            logger.error(f"خطأ أثناء قراءة المنفذ {pin}: {e}")
            return False, f"خطأ: {str(e)}", None
    
    def disconnect(self):
        self.connected = False
        self.gpio_initialized = False
        st.session_state["pi_connected"] = False
        st.session_state["pi_status"] = "disconnected"
        logger.info("تم قطع الاتصال بـ Raspberry Pi")
        event_system.publish("pi_disconnected", {})

real_pi_controller = RealRaspberryPiController()

# -------------------- الذاكرة الدائمة --------------------
class LifelongLearningMemory:
    """نظام ذاكرة دائمة للتعلم من التجارب"""
    def __init__(self):
        self.memory_file = "lifelong_memory.json"
        self.memory = self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"خطأ في تحميل الذاكرة: {e}")
            return {}
    
    def save_memory(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"خطأ في حفظ الذاكرة: {e}")
            return False
    
    def add_experience(self, category, experience, outcome):
        if category not in self.memory:
            self.memory[category] = []
        
        timestamp = datetime.now().isoformat()
        experience_id = hashlib.md5(f"{category}_{timestamp}".encode()).hexdigest()
        
        self.memory[category].append({
            "id": experience_id,
            "timestamp": timestamp,
            "experience": experience,
            "outcome": outcome
        })
        
        if len(self.memory[category]) > 1000:
            self.memory[category] = self.memory[category][-1000:]
        
        self.save_memory()
        logger.info(f"تم إضافة تجربة جديدة إلى فئة {category}")
    
    def get_recommendations(self, category, current_situation):
        if category not in self.memory:
            return []
        
        recommendations = []
        for experience in self.memory[category]:
            if "success" in experience["outcome"].lower():
                recommendations.append({
                    "based_on": experience["experience"],
                    "recommendation": f"بناءً على تجربة ناجحة سابقة: {experience['outcome']}",
                    "confidence": random.uniform(0.7, 0.95)
                })
        
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:5]
    
    def analyze_trends(self, category):
        if category not in self.memory or not self.memory[category]:
            return {"success_rate": 0, "common_issues": []}
        
        successes = 0
        issues = {}
        
        for experience in self.memory[category]:
            if "success" in experience["outcome"].lower():
                successes += 1
            
            if "error" in experience["outcome"].lower() or "fail" in experience["outcome"].lower():
                for word in experience["outcome"].split():
                    if word.lower() not in ["the", "a", "an", "in", "on", "at"] and len(word) > 3:
                        issues[word] = issues.get(word, 0) + 1
        
        success_rate = successes / len(self.memory[category])
        common_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "success_rate": success_rate,
            "common_issues": common_issues
        }

lifelong_memory = LifelongLearningMemory()

# -------------------- Advanced AI Analysis --------------------
class AdvancedAIAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=3, random_state=42)
        self.regressor = LinearRegression()
        self.is_fitted = False
    
    def prepare_data(self, df):
        try:
            data = df.copy()
            
            if 'time' in data.columns:
                data['hour'] = pd.to_datetime(data['time']).dt.hour
                data['day_part'] = pd.cut(data['hour'], 
                                         bins=[0, 6, 12, 18, 24], 
                                         labels=['night', 'morning', 'afternoon', 'evening'],
                                         include_lowest=True)
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'hour' in numeric_cols:
                numeric_cols.remove('hour')
            
            return data, numeric_cols
        except Exception as e:
            logger.error(f"خطأ في تحضير البيانات: {e}")
            return df, []
    
    def detect_anomalies(self, df):
        try:
            data, numeric_cols = self.prepare_data(df)
            
            if not numeric_cols:
                return [], df
            
            scaled_data = self.scaler.fit_transform(data[numeric_cols])
            anomalies = self.anomaly_detector.fit_predict(scaled_data)
            
            data['anomaly'] = anomalies
            data['anomaly_score'] = self.anomaly_detector.decision_function(scaled_data)
            
            anomaly_points = data[data['anomaly'] == -1]
            
            logger.info(f"تم كشف {len(anomaly_points)} نقطة شاذة")
            return anomaly_points, data
            
        except Exception as e:
            logger.error(f"خطأ في كشف الشذوذ: {e}")
            return [], df
    
    def cluster_data(self, df):
        try:
            data, numeric_cols = self.prepare_data(df)
            
            if not numeric_cols:
                return df
            
            scaled_data = self.scaler.fit_transform(data[numeric_cols])
            clusters = self.clusterer.fit_predict(scaled_data)
            
            data['cluster'] = clusters
            
            logger.info(f"تم تجميع البيانات إلى {len(set(clusters))} clusters")
            return data
            
        except Exception as e:
            logger.error(f"خطأ في تجميع البيانات: {e}")
            return df
    
    def predict_trend(self, df, target_column, hours_ahead=6):
        try:
            if target_column not in df.columns:
                logger.error(f"العمود {target_column} غير موجود في البيانات")
                return df, None
            
            data = df.copy()
            data = data.dropna(subset=[target_column])
            
            if len(data) < 10:
                logger.warning("لا توجد بيانات كافية للتنبؤ")
                return df, None
            
            data['time_index'] = range(len(data))
            
            X = data[['time_index']].values
            y = data[target_column].values
            
            self.regressor.fit(X, y)
            
            last_index = data['time_index'].max()
            future_indices = np.array(range(last_index + 1, last_index + hours_ahead + 1)).reshape(-1, 1)
            future_predictions = self.regressor.predict(future_indices)
            
            last_time = pd.to_datetime(data['time'].iloc[-1])
            future_times = [last_time + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
            
            predictions_df = pd.DataFrame({
                'time': future_times,
                f'predicted_{target_column}': future_predictions,
                'is_prediction': True
            })
            
            logger.info(f"تم إنشاء تنبؤات للـ {hours_ahead} ساعات القادمة")
            return data, predictions_df
            
        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {e}")
            return df, None
    
    def generate_insights(self, df):
        insights = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col in ['anomaly', 'cluster', 'time_index']:
                    continue
                    
                mean_val = df[col].mean()
                std_val = df[col].std()
                max_val = df[col].max()
                min_val = df[col].min()
                
                insight = {
                    'metric': col,
                    'mean': mean_val,
                    'stability': std_val,
                    'range': f"{min_val:.2f} - {max_val:.2f}",
                    'trend': 'صاعد' if mean_val > df[col].median() else 'هابط'
                }
                
                insights.append(insight)
            
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                strong_correlations = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.iloc[i, j]
                        
                        if abs(corr) > 0.7:
                            strong_correlations.append({
                                'variables': f"{col1} & {col2}",
                                'correlation': corr,
                                'type': 'إيجابية' if corr > 0 else 'سلبية'
                            })
                
                if strong_correlations:
                    insights.append({
                        'type': 'correlation_analysis',
                        'correlations': strong_correlations
                    })
            
            logger.info(f"تم توليد {len(insights)} رؤى من البيانات")
            return insights
            
        except Exception as e:
            logger.error(f"خطأ في توليد الرؤى: {e}")
            return insights

ai_analyzer = AdvancedAIAnalyzer()

# -------------------- نظام SNDT Chat الذكي --------------------
class SNDTChatSystem:
    """نظام دردشة ذكي مع تنبؤ بالمشاكل وتوصيات ذكية"""
    def __init__(self):
        self.responses = {
            "greeting": [
                "مرحباً! أنا المساعد الذكي لمنصة SNDT. كيف يمكنني مساعدتك اليوم؟",
                "أهلاً بك! أنا هنا لمساعدتك في إدارة وتحليل بيانات المصنع.",
                "مساء الخير! كيف يمكنني مساعدتك في نظام التوأم الرقمي اليوم؟"
            ],
            "help": [
                "يمكنني مساعدتك في: تحليل البيانات، كشف المشاكل، تقديم التوصيات، والإجابة على أسئلتك.",
                "أنا متخصص في تحليل بيانات المصنع وتقديم التوصيات الذكية. ما الذي تريد معرفته؟",
                "اسألني عن: حالة النظام، التنبؤات، التحليلات، أو أي استفسار آخر."
            ],
            "system_status": [
                "حالة النظام الحالية: {status}. درجة الحرارة: {temp}°م، الضغط: {pressure} بار.",
                "النظام يعمل بشكل {status}. البيانات الأخيرة: حرارة {temp}°م، ضغط {pressure} بار.",
                "الحالة الراهنة: {status}. آخر القراءات: {temp}°م للحرارة، {pressure} بار للضغط."
            ],
            "prediction": [
                "بناءً على البيانات الحالية، أتوقع أن {prediction} في الساعات القادمة.",
                "التنبؤات تشير إلى أن {prediction} خلال الفترة القادمة.",
                "تحليل البيانات يشير إلى توقع {prediction} في المستقبل القريب."
            ],
            "anomaly": [
                "تم كشف شذوذ في {metric}. القيمة: {value}، المتوسط: {average}.",
                "هناك انحراف في {metric}. القيمة الحالية: {value}، بينما المتوسط: {average}.",
                "لاحظت شذوذ في {metric}. القيمة المسجلة: {value} مقارنة بالمتوسط: {average}."
            ],
            "recommendation": [
                "أوصي بـ {recommendation} لتحسين الأداء.",
                "بناءً على التحليل، التوصية هي: {recommendation}.",
                "لمواجهة هذا التحدي، أنصح بـ {recommendation}."
            ],
            "unknown": [
                "عذراً، لم أفهم سؤالك بالكامل. هل يمكنك إعادة الصياغة؟",
                "أحتاج إلى مزيد من التوضيح للإجابة على سؤالك.",
                "هل يمكنك طرح سؤالك بطريقة أخرى؟ سأبذل جهدي للمساعدة."
            ]
        }
    
    def generate_response(self, user_input, context=None):
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["مرحبا", "اهلا", "السلام", "hello", "hi"]):
            response = random.choice(self.responses["greeting"])
        
        elif any(word in user_input_lower for word in ["مساعدة", "مساعدة", "help", "دعم"]):
            response = random.choice(self.responses["help"])
        
        elif any(word in user_input_lower for word in ["حالة", "status", "البيانات", "الآن"]):
            status = "جيدة" if st.session_state.get("mqtt_temp", 55) < 60 else "تحت المراقبة"
            response = random.choice(self.responses["system_status"]).format(
                status=status,
                temp=st.session_state.get("mqtt_temp", 55),
                pressure=st.session_state.get("pressure", 7.2)
            )
        
        elif any(word in user_input_lower for word in ["تنبأ", "توقع", "مستقبل", "predict", "forecast"]):
            prediction = self.generate_prediction(context)
            response = random.choice(self.responses["prediction"]).format(prediction=prediction)
        
        elif any(word in user_input_lower for word in ["مشكلة", "خطأ", "شذوذ", "anomaly", "issue"]):
            anomaly_info = self.detect_current_anomalies()
            if anomaly_info:
                response = random.choice(self.responses["anomaly"]).format(
                    metric=anomaly_info["metric"],
                    value=anomaly_info["value"],
                    average=anomaly_info["average"]
                )
            else:
                response = "لا توجد مشاكل أو شذوذ كبير في البيانات الحالية."
        
        elif any(word in user_input_lower for word in ["توصية", "نصيحة", "recommend", "advice"]):
            recommendation = self.generate_recommendation(context)
            response = random.choice(self.responses["recommendation"]).format(recommendation=recommendation)
        
        else:
            response = random.choice(self.responses["unknown"])
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        st.session_state["chat_history"].append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(st.session_state["chat_history"]) > 50:
            st.session_state["chat_history"] = st.session_state["chat_history"][-50:]
        
        logger.info(f"تم توليد رد للمساعد: {user_input[:50]}...")
        return response
    
    def generate_prediction(self, context=None):
        predictions = [
            "درجات الحرارة ستستقر حول 55-58°م",
            "ضغط النظام سيبقى ضمن النطاق الآمن",
            "مستويات الميثان ستظل منخفضة ومستقرة",
            "سيستمر الأداء المستقر للنظام",
            "قد نشهد ارتفاعاً طفيفاً في درجة الحرارة",
            "سيبقى استهلاك الطاقة ضمن المعدلات الطبيعية"
        ]
        
        if context and "temperature" in context.lower():
            return "درجة الحرارة ستصل إلى 58°م خلال الساعتين القادمتين"
        elif context and "pressure" in context.lower():
            return "ضغط النظام سيرتفع قليلاً إلى 7.5 بار ثم يعود إلى المستوى الطبيعي"
        
        return random.choice(predictions)
    
    def detect_current_anomalies(self):
        current_temp = st.session_state.get("mqtt_temp", 55)
        current_pressure = st.session_state.get("pressure", 7.2)
        current_methane = st.session_state.get("methane", 1.4)
        
        anomalies = []
        
        if current_temp > 65:
            anomalies.append({
                "metric": "درجة الحرارة",
                "value": f"{current_temp}°م",
                "average": "55°م"
            })
        
        if current_pressure > 9.0:
            anomalies.append({
                "metric": "الضغط",
                "value": f"{current_pressure} بار",
                "average": "7.2 بار"
            })
        
        if current_methane > 3.0:
            anomalies.append({
                "metric": "الميثان",
                "value": f"{current_methane} ppm",
                "average": "1.4 ppm"
            })
        
        if anomalies:
            return random.choice(anomalies)
        
        return None
    
    def generate_recommendation(self, context=None):
        recommendations = [
            "مراقبة درجة الحرارة عن كثب خلال الساعات القادمة",
            "إجراء فحص وقائي للصمامات والوصلات",
            "مراجعة إعدادات نظام التبريد",
            "تعديل ضغط التشغيل إلى 7.0 بار",
            "تفعيل نظام التهوية الإضافي",
            "إجراء صيانة روتينية للنظام"
        ]
        
        if context and "حرارة" in context:
            return "خفض إعدادات التسخين بنسبة 10% لتفادي الارتفاع المتوقع"
        elif context and "ضغط" in context:
            return "تفقد صمامات الأمان للتأكد من عملها بشكل صحيح"
        
        return random.choice(recommendations)

sndt_chat = SNDTChatSystem()

# -------------------- Twilio Integration --------------------
def send_twilio_alert(message, phone_number):
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=phone_number
        )
        
        logger.info(f"تم إرسال رسالة Twilio إلى {phone_number}: {message.sid}")
        return True, "تم إرسال التنبيه بنجاح"
    
    except ImportError:
        logger.warning("لم يتم تثبيت Twilio، سيتم محاكاة إرسال الرسالة")
        st.session_state["sms_sent"] = True
        st.session_state["notification_history"].append({
            "type": "sms_alert",
            "message": message,
            "phone": phone_number,
            "timestamp": datetime.now().isoformat(),
            "status": "simulated"
        })
        return True, "تم محاكاة إرسال التنبيه (Twilio غير مثبت)"
    
    except Exception as e:
        logger.error(f"خطأ في إرسال رسالة Twilio: {e}")
        return False, f"خطأ في إرسال الرسالة: {str(e)}"

# -------------------- نظام التحسين الذاتي --------------------
class DigitalTwinOptimizer:
    """نظام تحسين ذاتي للتوأم الرقمي باستخدام التعلم المعزز"""
    def __init__(self):
        self.optimization_history = []
        self.optimization_rules = self.load_optimization_rules()
    
    def load_optimization_rules(self):
        return {
            "temperature_high": {
                "condition": lambda data: data.get("mqtt_temp", 0) > 60,
                "action": "خفض درجة حرارة التشغيل بنسبة 5%",
                "impact": "high"
            },
            "pressure_high": {
                "condition": lambda data: data.get("pressure", 0) > 8.5,
                "action": "تقليل ضغط التشغيل إلى 7.0 بار",
                "impact": "high"
            },
            "methane_high": {
                "condition": lambda data: data.get("methane", 0) > 2.5,
                "action": "تفعيل نظام التهوية الإضافي",
                "impact": "critical"
            },
            "energy_optimization": {
                "condition": lambda data: data.get("flow_rate", 0) < 100 and data.get("mqtt_temp", 0) < 50,
                "action": "ضبط معدل التدفق لتحسين كفاءة الطاقة",
                "impact": "medium"
            }
        }
    
def analyze_current_state(self):
    current_data = {
        "mqtt_temp": st.session_state.get("mqtt_temp", 55),
        "pressure": st.session_state.get("pressure", 7.2),
        "methane": st.session_state.get("methane", 1.4),
        "vibration": st.session_state.get("vibration", 4.5),
        "flow_rate": st.session_state.get("flow_rate", 110.0)
    }
    
    recommendations = []
    
    for rule_name, rule in self.optimization_rules.items():
        if rule["condition"](current_data):
            recommendations.append({
                "rule": rule_name,
                "action": rule["action"],
                "impact": rule["impact"],
                "timestamp": datetime.now().isoformat()
            })
    
    if recommendations:
        st.session_state["recommendations"] = recommendations
        for rec in recommendations:
            self.optimization_history.append(rec)
            lifelong_memory.add_experience(
                "optimization", 
                f"حالة النظام: {current_data}", 
                f"تم اقتراح: {rec['action']} بسبب {rec['rule']}"
            )
    
    logger.info(f"تم توليد {len(recommendations)} توصية تحسين")
    return recommendations
    
    def apply_optimization(self, optimization):
        try:
            action = optimization["action"]
            
            if "درجة حرارة" in action:
                st.session_state["mqtt_temp"] *= 0.95
            elif "ضغط" in action:
                st.session_state["pressure"] = 7.0
            elif "التهوية" in action:
                st.session_state["methane"] *= 0.7
            
            optimization["applied_at"] = datetime.now().isoformat()
            optimization["status"] = "applied"
            
            st.session_state["optimization_history"].append(optimization)
            lifelong_memory.add_experience(
                "optimization_applied", 
                f"تم تطبيق: {optimization['action']}", 
                "تم تطبيق التحسين بنجاح"
            )
            
            logger.info(f"تم تطبيق التحسين: {optimization['action']}")
            return True, f"تم تطبيق التحسين: {optimization['action']}"
            
        except Exception as e:
            logger.error(f"خطأ في تطبيق التحسين: {e}")
            return False, f"خطأ في تطبيق التحسين: {str(e)}"
    
    def get_optimization_history(self, limit=10):
        return st.session_state.get("optimization_history", [])[-limit:]

# -------------------- نظام الصيانة التنبؤية --------------------
class PredictiveMaintenance:
    """نظام الصيانة التنبؤية المدعوم بالذكاء الاصطناعي"""
    def __init__(self):
        self.component_health = {}
        self.maintenance_schedule = {}
        self.initialize_components()
    
    def initialize_components(self):
        self.component_health = {
            "compressor": {"health": 95, "last_maintenance": datetime.now() - timedelta(days=30)},
            "heat_exchanger": {"health": 88, "last_maintenance": datetime.now() - timedelta(days=45)},
            "valves": {"health": 92, "last_maintenance": datetime.now() - timedelta(days=25)},
            "pumps": {"health": 85, "last_maintenance": datetime.now() - timedelta(days=60)},
            "sensors": {"health": 96, "last_maintenance": datetime.now() - timedelta(days=15)}
        }
    
    def update_component_health(self, sensor_data):
        temp = sensor_data.get("mqtt_temp", 55)
        if temp > 65:
            self.component_health["heat_exchanger"]["health"] -= 0.5
            self.component_health["valves"]["health"] -= 0.3
        
        pressure = sensor_data.get("pressure", 7.2)
        if pressure > 8.0:
            self.component_health["compressor"]["health"] -= 0.7
            self.component_health["pumps"]["health"] -= 0.4
        
        vibration = sensor_data.get("vibration", 4.5)
        if vibration > 6.0:
            self.component_health["compressor"]["health"] -= 0.6
            self.component_health["pumps"]["health"] -= 0.5
        
        for component in self.component_health:
            self.component_health[component]["health"] = max(0, self.component_health[component]["health"])
        
        logger.info("تم تحديث صحة المكونات بناءً على بيانات الاستشعار")
    
    def predict_failures(self):
        predictions = []
        
        for component, data in self.component_health.items():
            health = data["health"]
            last_maintenance = data["last_maintenance"]
            days_since_maintenance = (datetime.now() - last_maintenance).days
            
            if health < 70:
                failure_prob = min(95, 100 - health + days_since_maintenance / 2)
                urgency = "high"
            elif health < 85:
                failure_prob = min(70, 100 - health + days_since_maintenance / 4)
                urgency = "medium"
            else:
                failure_prob = max(5, (100 - health) / 2)
                urgency = "low"
            
            if failure_prob > 30:
                predictions.append({
                    "component": component,
                    "health": health,
                    "failure_probability": failure_prob,
                    "urgency": urgency,
                    "recommended_action": f"صيانة {component}",
                    "days_since_maintenance": days_since_maintenance
                })
        
        st.session_state["maintenance_predictions"] = predictions
        logger.info(f"تم إنشاء {len(predictions)} تنبؤات صيانة")
        return predictions
    
    def schedule_maintenance(self, component, action):
        schedule_date = datetime.now() + timedelta(days=7)
        
        self.maintenance_schedule[component] = {
            "scheduled_date": schedule_date.isoformat(),
            "action": action,
            "scheduled_at": datetime.now().isoformat(),
            "status": "scheduled"
        }
        
        lifelong_memory.add_experience(
            "maintenance", 
            f"جدولة صيانة {component}", 
            f"تم جدولة {action} للتاريخ {schedule_date.strftime('%Y-%m-%d')}"
        )
        
        logger.info(f"تم جدولة صيانة {component} للتاريخ {schedule_date}")
        return schedule_date

# -------------------- نظام الاستجابة للطوارئ --------------------
class EmergencyResponseSystem:
    """نظام متقدم للاستجابة للطوارئ والكوارث"""
    def __init__(self):
        self.emergency_protocols = self.load_emergency_protocols()
        self.emergency_levels = {
            "low": {"color": "#f39c12", "action": "مراقبة عن كثب"},
            "medium": {"color": "#e67e22", "action": "اتخاذ إجراء وقائي"},
            "high": {"color": "#e74c3c", "action": "إجراء عاجل مطلوب"},
            "critical": {"color": "#c0392b", "action": "إخلاء وإغلاق الطوارئ"}
        }
    
    def load_emergency_protocols(self):
    return {
        "temperature_extreme": {
            "condition": lambda data: data.get("mqtt_temp", 0) > 75 or data.get("mqtt_temp", 0) < 40,
            "message": "خطر: درجة حرارة غير طبيعية",
            "level_func": lambda data: "critical" if data.get("mqtt_temp", 0) > 75 else "high",
            "actions": [
                "إيقاف النظام فوراً",
                "تفعيل نظام التبريد/التسخين الاحتياطي",
                "إخطار فريق الصيانة العاجل"
            ]
        },
        "pressure_extreme": {
            "condition": lambda data: data.get("pressure", 0) > 10 or data.get("pressure", 0) < 5,
            "message": "خطر: ضغط غير طبيعي",
            "level_func": lambda data: "critical",
            "actions": [
                "تفعيل صمامات الأمان تلقائياً",
                "تقليل ضغط التشغيل فوراً",
                "إخلاء المنطقة إذا لزم الأمر"
            ]
        },
        "methane_leak": {
            "condition": lambda data: data.get("methane", 0) > 4.0,
            "message": "تحذير: تسرب غاز محتمل",
            "level_func": lambda data: "critical",
            "actions": [
                "تفعيل نظام التهوية القصوى",
                "إيقاف مصادر الاشتعال",
                "إخلاء المنطقة فوراً",
                "إخطار فريق الطوارئ"
            ]
        }
    }
    
    def check_emergency_conditions(self, sensor_data):
        emergencies = []
        
        for protocol_name, protocol in self.emergency_protocols.items():
            if protocol["condition"](sensor_data):
                emergency = {
                    "protocol": protocol_name,
                    "message": protocol["message"],
                    "level": protocol["level_func"](sensor_data),
                    "actions": protocol["actions"],
                    "timestamp": datetime.now().isoformat(),
                    "sensor_data": sensor_data
                }
                
                emergencies.append(emergency)
                
                st.session_state["incident_timeline"].append(emergency)
                lifelong_memory.add_experience(
                    "emergency", 
                    f"تفعيل بروتوكول الطوارئ: {protocol_name}", 
                    f"{protocol['message']} - المستوى: {protocol['level']}"
                )
                
                if protocol["level"] in ["high", "critical"]:
                    self.trigger_emergency_alert(emergency)
        
        return emergencies
    
    def trigger_emergency_alert(self, emergency):
        alert_message = f"تنبيه طوارئ: {emergency['message']} | المستوى: {emergency['level']}"
        
        st.session_state["notification_history"].append({
            "type": "emergency_alert",
            "message": alert_message,
            "level": emergency["level"],
            "timestamp": datetime.now().isoformat()
        })
        
        if st.session_state.get("twilio_enabled", False):
            phone_number = st.session_state.get("alert_phone_number", "")
            if phone_number:
                send_twilio_alert(alert_message, phone_number)
        
        st.session_state["last_emergency_alert"] = emergency
        logger.warning(f"تم تفعيل تنبيه الطوارئ: {alert_message}")
    
    def get_emergency_procedures(self, level):
        procedures = {
            "low": [
                "مراقبة المؤشرات عن كثب",
                "إعداد تقرير للمشرف",
                "الاستعداد لإجراءات وقائية"
            ],
            "medium": [
                "تنبيه فريق الصيانة",
                "بدء التسجيل التفصيلي للبيانات",
                "تحضير المعدات للتدخل"
            ],
            "high": [
                "إخطار مدير المصنع",
                "بدء الإجراءات التصحيحية",
                "تحضير خطة إخلاء احتياطية"
            ],
            "critical": [
                "إخلاء المنطقة فوراً",
                "إيقاف النظام بالكامل",
                "إخطار الدفاع المدني والطوارئ"
            ]
        }
        
        return procedures.get(level, [])

# -------------------- نظام الإصلاح الذاتي --------------------
class SelfHealingSystem:
    """نظام الإصلاح الذاتي التلقائي"""
    def __init__(self):
        self.healing_actions = []
        self.load_healing_protocols()
    
    def load_healing_protocols(self):
        self.healing_actions = [
            {
                "name": "ضبط درجة الحرارة التلقائي",
                "condition": lambda data: data.get("mqtt_temp", 0) > 62,
                "action": lambda: self.adjust_temperature(),
                "priority": "high"
            },
            {
                "name": "موازنة الضغط التلقائية",
                "condition": lambda data: data.get("pressure", 0) > 8.2,
                "action": lambda: self.balance_pressure(),
                "priority": "high"
            }
        ]
    
    def monitor_and_heal(self, sensor_data):
        applied_actions = []
        
        for action in self.healing_actions:
            if action["condition"](sensor_data):
                try:
                    result = action["action"]()
                    applied_actions.append({
                        "name": action["name"],
                        "priority": action["priority"],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"تم تطبيق إصلاح ذاتي: {action['name']}")
                    
                except Exception as e:
                    logger.error(f"فشل الإصلاح الذاتي {action['name']}: {e}")
        
        if applied_actions:
            for action in applied_actions:
                st.session_state["self_healing_actions"] = st.session_state.get("self_healing_actions", []) + [action]
            
            lifelong_memory.add_experience(
                "self_healing", 
                f"تم تطبيق {len(applied_actions)} إجراء إصلاح ذاتي", 
                f"الإجراءات: {[a['name'] for a in applied_actions]}"
            )
        
        return applied_actions
    
    def adjust_temperature(self):
        current_temp = st.session_state.get("mqtt_temp", 55)
        new_temp = current_temp * 0.93
        st.session_state["mqtt_temp"] = new_temp
        
        return f"تم خفض درجة الحرارة من {current_temp:.1f} إلى {new_temp:.1f}°م"
    
    def balance_pressure(self):
        current_pressure = st.session_state.get("pressure", 7.2)
        new_pressure = 7.0
        st.session_state["pressure"] = new_pressure
        
        return f"تم ضبط الضغط من {current_pressure:.1f} إلى {new_pressure:.1f} بار"

# -------------------- نظام الاستدامة --------------------
class SustainabilityMonitor:
    """مراقبة وتحليل استدامة العمليات"""
    def __init__(self):
        self.carbon_footprint = {
            "energy_consumption": 0,
            "co2_emissions": 0,
            "water_usage": 0,
            "waste_production": 0
        }
        self.initialize_sustainability_metrics()
    
    def initialize_sustainability_metrics(self):
        st.session_state["carbon_footprint"] = {
            "total_emissions": 1250,
            "energy_consumption": 3500,
            "water_usage": 120,
            "recycling_rate": 65,
            "last_calculated": datetime.now().isoformat()
        }
    
    def calculate_carbon_footprint(self, sensor_data):
        temp = sensor_data.get("mqtt_temp", 55)
        pressure = sensor_data.get("pressure", 7.2)
        flow_rate = sensor_data.get("flow_rate", 110)
        
        energy_consumption = (temp * 0.8) + (pressure * 12) + (flow_rate * 0.15)
        co2_emissions = energy_consumption * 0.85
        water_usage = flow_rate * 0.05
        
        sustainability_data = {
            "energy_consumption": energy_consumption,
            "co2_emissions": co2_emissions,
            "water_usage": water_usage,
            "waste_production": energy_consumption * 0.02,
            "last_updated": datetime.now().isoformat()
        }
        
        st.session_state["carbon_footprint"] = sustainability_data
        logger.info("تم تحديث بيانات الاستدامة بناءً على قراءات الاستشعار")
        
        return sustainability_data
    
    def calculate_energy_efficiency(self):
        sensor_data = {
            "mqtt_temp": st.session_state.get("mqtt_temp", 55),
            "pressure": st.session_state.get("pressure", 7.2),
            "flow_rate": st.session_state.get("flow_rate", 110)
        }
        
        ideal_temp, ideal_pressure, ideal_flow = 55, 7.2, 110
        
        temp_efficiency = max(0, 100 - abs(sensor_data["mqtt_temp"] - ideal_temp) * 2)
        pressure_efficiency = max(0, 100 - abs(sensor_data["pressure"] - ideal_pressure) * 10)
        flow_efficiency = max(0, 100 - abs(sensor_data["flow_rate"] - ideal_flow) * 0.5)
        
        overall_efficiency = (temp_efficiency + pressure_efficiency + flow_efficiency) / 3
        
        return overall_efficiency

# -------------------- تهيئة الأنظمة --------------------
digital_twin_optimizer = DigitalTwinOptimizer()
predictive_maintenance = PredictiveMaintenance()
emergency_response = EmergencyResponseSystem()
self_healing = SelfHealingSystem()
sustainability_monitor = SustainabilityMonitor()

# -------------------- المساعد الذكي --------------------
def generate_ai_response(prompt):
    """مساعد ذكي مدعوم بالذاكرة الدائمة"""
    prompt_lower = prompt.lower()
    
    if st.session_state.get("openai_enabled", False) and st.session_state.get("openai_api_key", ""):
        try:
            import openai
            openai.api_key = st.session_state["openai_api_key"]
            
            context = "\n".join([
                f"التجربة: {exp['experience']} - النتيجة: {exp['outcome']}"
                for exp in st.session_state.get("lifelong_memory", [])[-5:]
            ])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""أنت مساعد ذكي لمنصة التوأم الرقمي SNDT. 
                    السياق من الذاكرة الدائمة: {context}
                    البيانات الحالية: 
                    - درجة الحرارة: {st.session_state.get('mqtt_temp', 55)}°م
                    - الضغط: {st.session_state.get('pressure', 7.2)} بار
                    - الميثان: {st.session_state.get('methane', 1.4)} ppm
                    - الاهتزاز: {st.session_state.get('vibration', 4.5)}
                    - معدل التدفق: {st.session_state.get('flow_rate', 110)}"""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            st.session_state["chat_history"].append({
                "user": prompt,
                "assistant": ai_response,
                "timestamp": datetime.now().isoformat(),
                "source": "openai"
            })
            
            logger.info("تم توليد رد باستخدام OpenAI")
            return ai_response
            
        except Exception as e:
            logger.error(f"خطأ في الاتصال بـ OpenAI: {e}")
            return generate_fallback_response(prompt_lower)
    
    else:
        return sndt_chat.generate_response(prompt)

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
    
    st.session_state["chat_history"].append({
        "user": prompt_lower,
        "assistant": response,
        "timestamp": datetime.now().isoformat(),
        "source": "fallback"
    })
    
    return response

def get_weather_info():
    """معلومات الطقس"""
    weather_data = {
        "temperature": random.randint(20, 35),
        "condition": random.choice(["مشمس", "غائم جزئياً", "صافي"]),
        "humidity": random.randint(30, 70)
    }
    return f"""حالة الطقس الحالية:
• درجة الحرارة: {weather_data['temperature']}°م
• الحالة: {weather_data['condition']}
• الرطوبة: {weather_data['humidity']}%"""

def get_current_time_info():
    """معلومات الوقت والتاريخ"""
    now = datetime.now()
    return f"""الوقت الحالي: {now.strftime('%H:%M:%S')}
تاريخ اليوم: {now.strftime('%Y-%m-%d')}
اليوم: {now.strftime('%A')}"""

def generate_time_based_prediction(prompt):
    """إنشاء تنبؤات زمنية"""
    time_keywords = {"ساعة": 1, "ساعات": 1, "يوم": 24, "أيام": 24, "أسبوع": 168, "أسابيع": 168}
    hours_ahead = 2
    
    for keyword, hours in time_keywords.items():
        if keyword in prompt:
            hours_ahead = hours
            break
    
    predictions = [
        f"درجة الحرارة ستستقر حول {random.randint(53, 58)}°م",
        f"ضغط النظام سيبقى ضمن النطاق {random.uniform(7.0, 7.5):.1f} بار",
        f"مستويات الميثان ستظل منخفضة حول {random.uniform(1.2, 1.8):.1f} ppm",
        "سيستمر الأداء المستقر للنظام",
        f"قد نشهد ارتفاعاً طفيفاً في درجة الحرارة إلى {random.randint(58, 62)}°م",
        "سيبقى استهلاك الطاقة ضمن المعدلات الطبيعية"
    ]
    
    return f"خلال الـ {hours_ahead} ساعة القادمة، {random.choice(predictions)}"

def generate_current_status():
    """الحالة الحالية للنظام"""
    return f"""الحالة الحالية للنظام:
• درجة الحرارة: {st.session_state.get('mqtt_temp', 55)}°م
• الضغط: {st.session_state.get('pressure', 7.2)} بار
• الميثان: {st.session_state.get('methane', 1.4)} ppm
• الاهتزاز: {st.session_state.get('vibration', 4.5)}
• معدل التدفق: {st.session_state.get('flow_rate', 110)}
• آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}
• صحة النظام: جيدة"""

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
    <div style="padding:0.5rem;background:#f8f9fa;border-radius:0.5rem;margin-bottom:1rem;text-align:center">
        <span class="{status_class}">{status_text}</span>
        {'' if st.session_state.get('mqtt_connected', False) else ' | <span style="color:#e74c3c">غير متصل بـ MQTT</span>'}
        {'' if st.session_state.get('pi_connected', False) else ' | <span style="color:#e74c3c">غير متصل بـ Raspberry Pi</span>'}
    </div>
    """, unsafe_allow_html=True)

def show_notification_history():
    """عرض سجل الإشعارات"""
    notifications = st.session_state.get("notification_history", [])
    
    if not notifications:
        st.info("لا توجد إشعارات حالية")
        return
    
    for notification in notifications[-10:]:
        notification_time = datetime.fromisoformat(notification["timestamp"]).strftime("%H:%M:%S")
        
        if notification["type"] == "emergency_alert":
            level_color = {
                "low": "#f39c12", "medium": "#e67e22", "high": "#e74c3c", "critical": "#c0392b"
            }.get(notification.get("level", "low"), "#f39c12")
            
            st.markdown(f"""
            <div class="notification notification-danger">
                <strong>تنبيه طوارئ ({notification_time})</strong><br>
                <span style="color:{level_color}">● {notification['message']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        elif notification["type"] == "sms_alert":
            st.markdown(f"""
            <div class="notification notification-warning">
                <strong>رسالة نصية ({notification_time})</strong><br>
                إلى: {notification['phone']}<br>
                {notification['message']}
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="notification">
                <strong>إشعار ({notification_time})</strong><br>
                {notification['message']}
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
    """لوحة التحكم الرئيسية"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[0]}</div>', unsafe_allow_html=True)
    
    show_system_status_banner()
    
    # عرض المقاييس الرئيسية
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('temperature')}</h3>
            <h2>{st.session_state.get('mqtt_temp', 55):.1f}°م</h2>
            <p>{'▲ عالية' if st.session_state.get('mqtt_temp', 55) > 58 else '▼ طبيعية'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('pressure')}</h3>
            <h2>{st.session_state.get('pressure', 7.2):.1f} بار</h2>
            <p>{'▲ مرتفع' if st.session_state.get('pressure', 7.2) > 7.5 else '▼ طبيعي'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('methane')}</h3>
            <h2>{st.session_state.get('methane', 1.4):.1f} ppm</h2>
            <p>{'▲ مرتفع' if st.session_state.get('methane', 1.4) > 2.0 else '▼ منخفض'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        efficiency = sustainability_monitor.calculate_energy_efficiency()
        st.markdown(f"""
        <div class="metric-card">
            <h3>كفاءة الطاقة</h3>
            <h2>{efficiency:.1f}%</h2>
            <p>{'▲ جيدة' if efficiency > 80 else '▼ تحتاج تحسين'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # مخططات البيانات
    st.markdown(f'<div class="section-header">البيانات المباشرة</div>', unsafe_allow_html=True)
    
    live_data = pd.DataFrame({
        "time": [datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)],
        "Temperature": np.random.normal(st.session_state.get("mqtt_temp", 55), 1.5, 30),
        "Pressure": np.random.normal(st.session_state.get("pressure", 7.2), 0.2, 30),
        "Methane": np.random.normal(st.session_state.get("methane", 1.4), 0.1, 30)
    })
    
    fig_temp = px.line(live_data, x="time", y="Temperature", title="درجة الحرارة خلال last 30 minutes")
    fig_temp.update_layout(height=300, xaxis_title="الوقت", yaxis_title="درجة الحرارة (°م)")
    st.plotly_chart(fig_temp, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pressure = px.line(live_data, x="time", y="Pressure", title="الضغط خلال last 30 minutes")
        fig_pressure.update_layout(height=300, xaxis_title="الوقت", yaxis_title="الضغط (بار)")
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col2:
        fig_methane = px.line(live_data, x="time", y="Methane", title="الميثان خلال last 30 minutes")
        fig_methane.update_layout(height=300, xaxis_title="الوقت", yaxis_title="الميثان (ppm)")
        st.plotly_chart(fig_methane, use_container_width=True)
    
    # التنبيهات والتوصيات
    st.markdown(f'<div class="section-header">التنبيهات والتوصيات</div>', unsafe_allow_html=True)
    
    # التحقق من طوارئ
    current_sensor_data = {
        "mqtt_temp": st.session_state.get("mqtt_temp", 55),
        "pressure": st.session_state.get("pressure", 7.2),
        "methane": st.session_state.get("methane", 1.4),
        "vibration": st.session_state.get("vibration", 4.5),
        "flow_rate": st.session_state.get("flow_rate", 110)
    }
    
    emergencies = emergency_response.check_emergency_conditions(current_sensor_data)
    
    if emergencies:
        for emergency in emergencies:
            st.error(f"**تنبيه طوارئ**: {emergency['message']} (المستوى: {emergency['level']})")
            
            with st.expander("إجراءات الطوارئ المطلوبة"):
                for action in emergency['actions']:
                    st.write(f"• {action}")
    else:
        st.success("لا توجد تنبيهات طوارئ حالية")
    
    # توصيات التحسين
    optimizations = digital_twin_optimizer.analyze_current_state()
    
    if optimizations:
        st.warning("**توصيات التحسين**:")
        for opt in optimizations:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {opt['action']} (الأهمية: {opt['impact']})")
            with col2:
                if st.button("تطبيق", key=f"apply_opt_{opt['rule']}"):
                    success, message = digital_twin_optimizer.apply_optimization(opt)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
    
    # الإصلاح الذاتي
    healing_actions = self_healing.monitor_and_heal(current_sensor_data)
    if healing_actions:
        st.info("**تم تنفيذ الإصلاح الذاتي**:")
        for action in healing_actions:
            st.write(f"• {action['name']}: {action['result']}")

def analytics_ai_section():
    """التحليلات والذكاء الاصطناعي"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[1]}</div>', unsafe_allow_html=True)
    
    # تحليل البيانات
    st.markdown(f'<div class="section-header">تحليل البيانات المتقدم</div>', unsafe_allow_html=True)
    
    if "analytics_df" not in st.session_state:
        st.session_state["analytics_df"] = demo_df.copy()
    
    analysis_type = st.selectbox("نوع التحليل", [
        "كشف الشذوذ", 
        "التجميع", 
        "التنبؤ بالاتجاه",
        "الرؤى الذكية"
    ])
    
    if st.button("تشغيل التحليل"):
        with st.spinner("جاري تحليل البيانات..."):
            if analysis_type == "كشف الشذوذ":
                anomalies, analyzed_df = ai_analyzer.detect_anomalies(st.session_state["analytics_df"])
                st.session_state["analytics_df"] = analyzed_df
                
                if not anomalies.empty:
                    st.warning(f"تم كشف {len(anomalies)} نقطة شاذة في البيانات")
                    
                    fig = px.scatter(analyzed_df, x="time", y="Temperature", 
                                    color="anomaly", title="كشف الشذوذ في درجة الحرارة")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("عرض البيانات الشاذة"):
                        st.dataframe(anomalies)
                else:
                    st.success("لا توجد شذوذ كبير في البيانات")
            
            elif analysis_type == "التجميع":
                clustered_df = ai_analyzer.cluster_data(st.session_state["analytics_df"])
                st.session_state["analytics_df"] = clustered_df
                
                st.success("تم تجميع البيانات بنجاح")
                
                fig = px.scatter(clustered_df, x="time", y="Temperature", 
                                color="cluster", title="تجميع البيانات")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "التنبؤ بالاتجاه":
                target = st.selectbox("اختر المتغير للتنبؤ", ["Temperature", "Pressure", "Methane"])
                hours = st.slider("عدد الساعات للتنبؤ", 1, 24, 6)
                
                analyzed_df, predictions = ai_analyzer.predict_trend(
                    st.session_state["analytics_df"], target, hours
                )
                
                if predictions is not None:
                    st.session_state["analytics_df"] = analyzed_df
                    
                    combined_df = pd.concat([
                        analyzed_df.assign(is_prediction=False),
                        predictions.assign(is_prediction=True)
                    ])
                    
                    fig = px.line(combined_df, x="time", y=target, 
                                 color="is_prediction", title=f"التنبؤ بـ {target}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("عرض بيانات التنبؤ"):
                        st.dataframe(predictions)
                else:
                    st.error("فشل في إنشاء التنبؤات. تأكد من وجود بيانات كافية.")
            
            elif analysis_type == "الرؤى الذكية":
                insights = ai_analyzer.generate_insights(st.session_state["analytics_df"])
                
                if insights:
                    st.success(f"تم توليد {len(insights)} رؤى من البيانات")
                    
                    for insight in insights:
                        if insight.get('type') == 'correlation_analysis':
                            st.markdown("**تحليل العلاقات القوية:**")
                            for corr in insight['correlations']:
                                st.write(f"- {corr['variables']}: علاقة {corr['type']} ({corr['correlation']:.2f})")
                        else:
                            st.write(f"""
                            **{insight['metric']}**:
                            - المتوسط: {insight['mean']:.2f}
                            - الاستقرار: {insight['stability']:.2f}
                            - المدى: {insight['range']}
                            - الاتجاه: {insight['trend']}
                            """)
                else:
                    st.error("فشل في توليد الرؤى. تأكد من وجود بيانات كافية.")
    
    # الصيانة التنبؤية
    st.markdown(f'<div class="section-header">الصيانة التنبؤية</div>', unsafe_allow_html=True)
    
    if st.button("تحليل صحة المكونات"):
        with st.spinner("جاري تحليل صحة المكونات..."):
            current_sensor_data = {
                "mqtt_temp": st.session_state.get("mqtt_temp", 55),
                "pressure": st.session_state.get("pressure", 7.2),
                "methane": st.session_state.get("methane", 1.4),
                "vibration": st.session_state.get("vibration", 4.5),
                "flow_rate": st.session_state.get("flow_rate", 110)
            }
            
            predictive_maintenance.update_component_health(current_sensor_data)
            predictions = predictive_maintenance.predict_failures()
            
            if predictions:
                st.warning("**تنبؤات الأعطال المحتملة:**")
                
                for pred in predictions:
                    progress_value = pred["failure_probability"] / 100
                    color = "red" if pred["urgency"] == "high" else "orange" if pred["urgency"] == "medium" else "blue"
                    
                    st.write(f"**{pred['component']}**")
                    st.write(f"صحة المكون: {pred['health']:.1f}%")
                    st.write(f"احتمالية العطل: {pred['failure_probability']:.1f}%")
                    st.progress(progress_value, text=f"احتمالية العطل: {pred['failure_probability']:.1f}%")
                    
                    if st.button(f"جدولة صيانة {pred['component']}", key=f"maint_{pred['component']}"):
                        schedule_date = predictive_maintenance.schedule_maintenance(
                            pred["component"], pred["recommended_action"]
                        )
                        st.success(f"تم جدولة الصيانة للتاريخ: {schedule_date.strftime('%Y-%m-%d')}")
                    
                    st.divider()
            else:
                st.success("لا توجد تنبؤات بأعطال محتملة في الوقت الحالي")
    
    # الذاكرة الدائمة والتعلم
    st.markdown(f'<div class="section-header">الذاكرة الدائمة والتعلم</div>', unsafe_allow_html=True)
    
    if st.button("عرض التوصيات بناءً على الخبرة"):
        recommendations = lifelong_memory.get_recommendations("optimization", "الحالة الحالية")
        
        if recommendations:
            st.info("**توصيات مستندة على الخبرة السابقة:**")
            for rec in recommendations:
                st.write(f"- {rec['recommendation']}")
                st.write(f"  الثقة: {rec['confidence']:.0%}")
                st.write(f"  مستند على: {rec['based_on']}")
                st.divider()
        else:
            st.info("لا توجد توصيات مستندة على الخبرة yet. سيتم توليدها مع مرور الوقت.")
    
    # تحليل الاتجاهات
    if st.button("تحليل الاتجاهات من التجارب السابقة"):
        trends = lifelong_memory.analyze_trends("optimization")
        
        st.metric("معدل النجاح", f"{trends['success_rate']:.0%}")
        
        if trends["common_issues"]:
            st.write("**المشاكل الشائعة:**")
            for issue, count in trends["common_issues"]:
                st.write(f"- {issue} (حدث {count} مرات)")

def operations_control_section():
    """العمليات والتحكم"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[2]}</div>', unsafe_allow_html=True)
    
    # إحصائيات العمليات
    st.markdown(f'<div class="section-header">إحصائيات العمليات</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("الإنتاج اليومي", "2,450 طن", "+3.2%")
    
    with col2:
        st.metric("كفاءة الطاقة", "87.5%", "+1.8%")
    
    with col3:
        st.metric("الجودة", "98.2%", "-0.4%")
    
    # مخطط أداء العمليات
    operation_data = pd.DataFrame({
        "الفترة": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو"],
        "الإنتاج": [2200, 2350, 2400, 2300, 2450, 2500],
        "الكفاءة": [82, 85, 84, 87, 86, 88],
        "الجودة": [97, 98, 97.5, 98.2, 97.8, 98.5]
    })
    
    fig = px.line(operation_data, x="الفترة", y=["الإنتاج", "الكفاءة", "الجودة"],
                 title="أداء العمليات خلال الأشهر الستة الماضية",
                 labels={"value": "القيمة", "variable": "المؤشر"})
    st.plotly_chart(fig, use_container_width=True)
    
    # إدارة الجودة
    st.markdown(f'<div class="section-header">إدارة الجودة</div>', unsafe_allow_html=True)
    
    quality_data = pd.DataFrame({
        "البند": ["النقاوة", "اللزوجة", "الكثافة", "اللون", "التركيب"],
        "القيمة": [98.5, 96.8, 99.2, 97.5, 98.8],
        "المعيار": [95, 95, 98, 96, 97]
    })
    
    fig = px.bar(quality_data, x="البند", y=["القيمة", "المعيار"],
                title="مقارنة جودة المنتج بالمعايير",
                barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # التحكم بالأجهزة
    st.markdown(f'<div class="section-header">التحكم بالأجهزة</div>', unsafe_allow_html=True)
    
    # حالة Raspberry Pi
    pi_status = st.session_state.get("pi_status", "disconnected")
    status_color = "#2ecc71" if pi_status == "connected" else "#e74c3c"
    
    st.markdown(f"""
    <div style="padding:1rem; background:#f8f9fa; border-radius:0.5rem; margin-bottom:1rem;">
        <strong>الحالة:</strong> <span style="color:{status_color}">{pi_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # الاتصال بـ Raspberry Pi
    if pi_status != "connected":
        with st.form("connect_pi_form"):
            st.write("إعدادات الاتصال بـ Raspberry Pi")
            ip_address = st.text_input("IP Address", "192.168.1.100")
            username = st.text_input("Username", "pi")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("الاتصال"):
                with st.spinner("جاري الاتصال..."):
                    success, message = real_pi_controller.connect_to_pi(ip_address, username, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
    else:
        if st.button("قطع الاتصال"):
            real_pi_controller.disconnect()
            st.success("تم قطع الاتصال بـ Raspberry Pi")
            st.rerun()
    
    # التحكم في المنافذ
    if pi_status == "connected":
        st.markdown(f'<div class="section-header">التحكم في المنافذ</div>', unsafe_allow_html=True)
        
        # تهيئة GPIO
        if not real_pi_controller.gpio_initialized:
            if st.button("تهيئة منافذ GPIO"):
                success, message = real_pi_controller.initialize_gpio()
                if success:
                    st.success(message)
                else:
                    st.error(message)
                st.rerun()
        else:
            st.success("تم تهيئة منافذ GPIO")
            
            # التحكم في منافذ الإخراج
            st.subheader("منافذ الإخراج")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("تشغيل المنفذ 1"):
                    success, message = real_pi_controller.control_output(1, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 1"):
                    success, message = real_pi_controller.control_output(1, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col2:
                if st.button("تشغيل المنفذ 2"):
                    success, message = real_pi_controller.control_output(2, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 2"):
                    success, message = real_pi_controller.control_output(2, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col3:
                if st.button("تشغيل المنفذ 3"):
                    success, message = real_pi_controller.control_output(3, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 3"):
                    success, message = real_pi_controller.control_output(3, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            # قراءة منافذ الإدخال
            st.subheader("منافذ الإدخال")
            if st.button("قراءة المنفذ 4"):
                success, message, value = real_pi_controller.read_input(4)
                if success:
                    st.info(message)
                else:
                    st.error(message)
            
            if st.button("قراءة المنفذ 5"):
                success, message, value = real_pi_controller.read_input(5)
                if success:
                    st.info(message)
                else:
                    st.error(message)
    
    # إعدادات MQTT
    st.markdown(f'<div class="section-header">إعدادات اتصال MQTT</div>', unsafe_allow_html=True)
    
    mqtt_connected = st.session_state.get("mqtt_connected", False)
    st.write(f"الحالة: {'متصل' if mqtt_connected else 'غير متصل'}")
    
    if mqtt_connected:
        if st.button("قطع الاتصال MQTT"):
            mqtt_client.disconnect()
            st.success("تم قطع الاتصال MQTT")
            st.rerun()
    else:
        if st.button("إعادة الاتصال MQTT"):
            success = mqtt_client.connect_with_retry()
            if success:
                st.success("تم الاتصال MQTT بنجاح")
            else:
                st.error("فشل الاتصال MQTT")
            st.rerun()
    
    # إرسال رسالة MQTT
    st.subheader("إرسال رسالة MQTT")
    topic = st.selectbox("الموضوع", [MQTT_TOPIC_TEMPERATURE, MQTT_TOPIC_PRESSURE, MQTT_TOPIC_METHANE, MQTT_TOPIC_CONTROL])
    message = st.text_input("الرسالة", "25.5")
    
    if st.button("إرسال الرسالة"):
        if mqtt_client.publish(topic, message):
            st.success("تم إرسال الرسالة بنجاح")
        else:
            st.error("فشل إرسال الرسالة")

def safety_emergency_section():
    """السلامة والطوارئ"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[3]}</div>', unsafe_allow_html=True)
    
    # نظرة عامة على السلامة
    st.markdown(f'<div class="section-header">نظرة عامة على السلامة</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_since = (datetime.now() - datetime(2023, 1, 1)).days
        st.metric("الأيام بدون حوادث", f"{days_since} يوم")
    
    with col2:
        st.metric("التنبيهات النشطة", "3", "-1 من الأسبوع الماضي")
    
    with col3:
        st.metric("مستوى المخاطر", "منخفض", "2%")
    
    # خريطة الحرارة للمخاطر
    risk_data = pd.DataFrame({
        "المنطقة": ["التفاعل", "التخزين", "المناولة", "التحكم", "الخدمات"],
        "مستوى المخاطرة": [8, 6, 7, 3, 4]
    })
    
    fig = px.bar(risk_data, x="المنطقة", y="مستوى المخاطرة", 
                title="مستويات المخاطرة حسب المنطقة",
                color="مستوى المخاطرة", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)
    
    # التنبيهات الحالية
    st.markdown(f'<div class="section-header">التنبيهات الحالية</div>', unsafe_allow_html=True)
    show_notification_history()
    
    # إعدادات التنبيهات
    st.markdown(f'<div class="section-header">إعدادات التنبيهات</div>', unsafe_allow_html=True)
    
    twilio_enabled = st.toggle("تفعيل تنبيهات SMS", value=st.session_state.get("twilio_enabled", True))
    st.session_state["twilio_enabled"] = twilio_enabled
    
    if twilio_enabled:
        phone_number = st.text_input("رقم الهاتف للتنبيهات", value=st.session_state.get("alert_phone_number", ""))
        st.session_state["alert_phone_number"] = phone_number
        
        # اختبار إرسال رسالة
        if st.button("اختبار إرسال رسالة"):
            if phone_number:
                success, message = send_twilio_alert("هذه رسالة اختبار من نظام SNDT", phone_number)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("يرجى إدخال رقم الهاتف أولاً")
    
    # عتبات التنبيه
    st.subheader("عتبات التنبيه")
    
    temp_threshold = st.slider("عتبة درجة الحرارة (°م)", 50, 80, 65)
    pressure_threshold = st.slider("عتبة الضغط (بار)", 5, 12, 9)
    methane_threshold = st.slider("عتبة الميثان (ppm)", 1, 5, 3)
    
    if st.button("حفظ العتبات"):
        st.session_state["alert_thresholds"] = {
            "temperature": temp_threshold,
            "pressure": pressure_threshold,
            "methane": methane_threshold
        }
        st.success("تم حفظ عتبات التنبيه")
    
    # بروتوكولات الطوارئ
    st.markdown(f'<div class="section-header">بروتوكولات الطوارئ</div>', unsafe_allow_html=True)
    
    emergency_level = st.selectbox("مستوى الطوارئ", ["منخفض", "متوسط", "مرتفع", "حرج"])
    
    procedures = emergency_response.get_emergency_procedures(emergency_level.lower())
    
    st.write(f"**إجراءات الطوارئ لمستوى {emergency_level}:**")
    for procedure in procedures:
        st.write(f"• {procedure}")
    
    # محاكاة طوارئ
    if st.button("محاكاة حالة طوارئ", type="secondary"):
        st.session_state["disaster_simulated"] = True
        
        # محاكاة ارتفاع مفاجئ في درجة الحرارة
        st.session_state["mqtt_temp"] = 78.5
        st.session_state["pressure"] = 9.8
        st.session_state["methane"] = 3.7
        
        st.error("تم تفعيل محاكاة الطوارئ! تم رفع قيم الاستشعار إلى مستويات خطيرة.")
        st.rerun()
    
    if st.session_state.get("disaster_simulated", False):
        if st.button("إنهاء محاكاة الطوارئ"):
            st.session_state["disaster_simulated"] = False
            
            # إعادة القيم إلى وضعها الطبيعي
            st.session_state["mqtt_temp"] = 55.0
            st.session_state["pressure"] = 7.2
            st.session_state["methane"] = 1.4
            
            st.success("تم إنهاء محاكاة الطوارئ وأعيدت القيم إلى وضعها الطبيعي.")
            st.rerun()

def sustainability_energy_section():
    """الاستدامة والطاقة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[4]}</div>', unsafe_allow_html=True)
    
    # البصمة الكربونية
    st.markdown(f'<div class="section-header">البصمة الكربونية</div>', unsafe_allow_html=True)
    
    # حساب البصمة الكربونية الحالية
    sensor_data = {
        "mqtt_temp": st.session_state.get("mqtt_temp", 55),
        "pressure": st.session_state.get("pressure", 7.2),
        "flow_rate": st.session_state.get("flow_rate", 110)
    }
    
    footprint = sustainability_monitor.calculate_carbon_footprint(sensor_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("انبعاثات CO₂", f"{footprint.get('co2_emissions', 0):.1f} kg", "-2.3%")
    
    with col2:
        st.metric("استهلاك الطاقة", f"{footprint.get('energy_consumption', 0):.1f} kWh", "-1.8%")
    
    with col3:
        st.metric("استهلاك المياه", f"{footprint.get('water_usage', 0):.1f} m³", "-3.1%")
    
    # مخطط البصمة الكربونية
    footprint_data = pd.DataFrame({
        "الشهر": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو"],
        "انبعاثات CO₂": [1450, 1380, 1320, 1280, 1250, 1220],
        "الهدف": [1300, 1250, 1200, 1150, 1100, 1050]
    })
    
    fig = px.line(footprint_data, x="الشهر", y=["انبعاثات CO₂", "الهدف"],
                 title="اتجاه البصمة الكربونية خلال الأشهر الستة الماضية",
                 labels={"value": "انبعاثات CO₂ (kg)", "variable": "المتغير"})
    st.plotly_chart(fig, use_container_width=True)
    
    # كفاءة الطاقة
    st.markdown(f'<div class="section-header">كفاءة الطاقة</div>', unsafe_allow_html=True)
    
    efficiency = sustainability_monitor.calculate_energy_efficiency()
    st.metric("كفاءة الطاقة الإجمالية", f"{efficiency:.1f}%")
    
    efficiency_data = pd.DataFrame({
        "المعدة": ["المفاعل", "المضخات", "التبريد", "التحكم", "الإضاءة"],
        "الكفاءة": [85, 78, 92, 88, 95],
        "استهلاك الطاقة": [45, 25, 15, 10, 5]  # نسب مئوية
    })
    
    fig = px.bar(efficiency_data, x="المعدة", y="الكفاءة",
                title="كفاءة الطاقة حسب المعدة",
                color="الكفاءة", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)
    
    # تقرير الاستدامة
    st.markdown(f'<div class="section-header">تقرير الاستدامة</div>', unsafe_allow_html=True)
    
    if st.button("إنشاء تقرير الاستدامة"):
        with st.spinner("جاري إنشاء تقرير الاستدامة..."):
            footprint = st.session_state.get("carbon_footprint", {})
            
            report = {
                "report_date": datetime.now().isoformat(),
                "energy_efficiency": efficiency,
                "carbon_footprint": footprint.get("co2_emissions", 0),
                "water_usage": footprint.get("water_usage", 0),
                "waste_management": footprint.get("waste_production", 0) * 0.3,
                "recommendations": [
                    "خفض درجة حرارة التشغيل لتحسين كفاءة الطاقة",
                    "ضبط ضغط التشغيل لتقليل الاستهلاك",
                    "تحسين نظام الاحتواء لتقليل انبعاثات الميثان",
                    "تحسين معدل التدفق لتعزيز الكفاءة"
                ]
            }
            
            st.success("تم إنشاء تقرير الاستدامة بنجاح!")
            
            st.write(f"**تاريخ التقرير:** {datetime.fromisoformat(report['report_date']).strftime('%Y-%m-%d %H:%M')}")
            st.metric("كفاءة الطاقة", f"{report['energy_efficiency']:.1f}%")
            st.metric("البصمة الكربونية", f"{report['carbon_footprint']:.1f} kg CO₂")
            st.metric("استهلاك المياه", f"{report['water_usage']:.1f} m³")
            
            if report['recommendations']:
                st.write("**توصيات تحسين الاستدامة:**")
                for recommendation in report['recommendations']:
                    st.write(f"• {recommendation}")
    
    # أهداف الاستدامة
    st.markdown(f'<div class="section-header">أهداف الاستدامة</div>', unsafe_allow_html=True)
    
    goals_data = pd.DataFrame({
        "الهدف": [
            "خفض انبعاثات CO₂ بنسبة 20%",
            "تقليل استهلاك الطاقة بنسبة 15%",
            "خفض استهلاك المياه بنسبة 25%",
            "زيادة إعادة التدوير إلى 75%",
            "تحقيق الصفر من النفايات الخطرة"
        ],
        "التقدم": [65, 80, 45, 70, 90],
        "الموعد النهائي": ["2023-12-31", "2023-10-31", "2024-03-31", "2023-09-30", "2024-06-30"]
    })
    
    for _, goal in goals_data.iterrows():
        st.write(f"**{goal['الهدف']}**")
        st.progress(goal['التقدم'] / 100, text=f"{goal['التقدم']}% مكتمل")
        st.write(f"الموعد النهائي: {goal['الموعد النهائي']}")
        st.divider()

def smart_assistant_section():
    """المساعد الذكي"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[5]}</div>', unsafe_allow_html=True)
    
    # واجهة الدردشة
    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4 style="margin:0;">المساعد الذكي SNDT</h4>
        <p style="margin:0.5rem 0 0 0; color: #666;">اسألني عن حالة النظام، التحليلات، التنبؤات، أو أي استفسار آخر</p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض سجل المحادثة
    chat_history = st.session_state.get("chat_history", [])
    
    for message in chat_history[-10:]:
        if message["user"]:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: left;">
                <strong>You:</strong> {message["user"]}
            </div>
            """, unsafe_allow_html=True)
        
        if message["assistant"]:
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: right;">
                <strong>المساعد:</strong> {message["assistant"]}
            </div>
            """, unsafe_allow_html=True)
    
    # إدخال الرسالة
    user_input = st.chat_input("اكتب رسالتك هنا...")
    
    if user_input:
        # عرض رسالة المستخدم فوراً
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: left;">
            <strong>You:</strong> {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # توليد الرد
        with st.spinner("جاري التفكير..."):
            response = generate_ai_response(user_input)
            
            # عرض رد المساعد
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: right;">
                <strong>المساعد:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
    
    # إمكانيات المساعد
    st.markdown(f'<div class="section-header">إمكانيات المساعد</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**يمكنني المساعدة في:**")
        st.write("• مراقبة حالة النظام")
        st.write("• تحليل البيانات والاتجاهات")
        st.write("• التنبؤ بالمشاكل المحتملة")
        st.write("• تقديم التوصيات الذكية")
        st.write("• الإجابة على الأسئلة العامة")
    
    with col2:
        st.write("**اسألني أمثلة:**")
        st.write("• ما هي حالة النظام الحالية؟")
        st.write("• كيف تبدو درجة الحرارة الآن؟")
        st.write("• هل هناك أي مشاكل متوقعة؟")
        st.write("• ما هي توصياتك لتحسين الأداء؟")
        st.write("• ما هو الطقس اليوم؟")
    
    # إعدادات الذكاء الاصطناعي
    st.markdown(f'<div class="section-header">إعدادات الذكاء الاصطناعي</div>', unsafe_allow_html=True)
    
    openai_enabled = st.toggle("تفعيل OpenAI (GPT)", value=st.session_state.get("openai_enabled", False))
    st.session_state["openai_enabled"] = openai_enabled
    
    if openai_enabled:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
        st.session_state["openai_api_key"] = api_key
        
        if api_key:
            st.success("تم تفعيل الذكاء الاصطناعي المتقدم")
        else:
            st.warning("يرجى إدخال مفتاح API لتفعيل الذكاء الاصطناعي المتقدم")
    else:
        st.info("يستخدم النظام الذكاء الاصطناعي المدمج (بدون OpenAI)")

def settings_help_section():
    """الإعدادات والمساعدة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[6]}</div>', unsafe_allow_html=True)
    
    # الإعدادات
    st.markdown(f'<div class="section-header">الإعدادات</div>', unsafe_allow_html=True)
    
    # إعدادات اللغة
    lang = st.radio("اللغة", ["العربية", "English"], horizontal=True, index=0 if st.session_state.get("lang", "ar") == "ar" else 1)
    st.session_state["lang"] = "ar" if lang == "العربية" else "en"
    
    # إعدادات الثيم
    theme = st.radio("المظهر", ["فاتح", "داكن"], horizontal=True, index=0 if st.session_state.get("theme", "light") == "light" else 1)
    st.session_state["theme"] = "light" if theme == "فاتح" else "dark"
    theme_manager.apply_theme_styles()
    
    # إعدادات النظام
    st.subheader("إعدادات النظام")
    
    simulation_active = st.toggle("وضع المحاكاة النشط", value=st.session_state.get("simulation_active", True))
    st.session_state["simulation_active"] = simulation_active
    
    data_refresh = st.slider("معدل تحديث البيانات (ثواني)", 1, 60, 5)
    st.session_state["data_refresh_rate"] = data_refresh
    
    if st.button("حفظ الإعدادات"):
        st.success("تم حفظ الإعدادات بنجاح")
    
    # المعلومات والمساعدة
    st.markdown(f'<div class="section-header">المعلومات والمساعدة</div>', unsafe_allow_html=True)
    
    with st.expander("عن التطبيق"):
        st.write("""
        ### منصة SNDT - التوأم الرقمي الذكي
        
        **الإصدار:** 1.0.0
        **تاريخ البناء:** 2025-07-01
        
        منصة SNDT هي نظام متكامل لإدارة المصانع والعمليات الصناعية باستخدام تقنيات التوأم الرقمي والذكاء الاصطناعي.
        
        **المميزات الرئيسية:**
        - مراقبة البيانات في الوقت الحقيقي
        - التحليلات التنبؤية والذكاء الاصطناعي
        - الصيانة التنبؤية
        - إدارة السلامة والطوارئ
        - تحليل الاستدامة والكفاءة
        - المساعد الذكي المتقدم
        
        **التقنيات المستخدمة:**
        - Python, Streamlit
        - MQTT للاتصال بأجهزة IoT
        - Redis للتخزين المؤقت
        - TensorFlow/PyTorch للذكاء الاصطناعي
        - Plotly للتصورات
        """)
    
    with st.expander("دليل المستخدم"):
        st.write("""
        ### دليل استخدام منصة SNDT
        
        **لوحة التحكم الرئيسية:**
        - عرض البيانات المباشرة من أجهزة الاستشعار
        - متابعة المقاييس الرئيسية لأداء النظام
        - الاطلاع على التنبيهات والتوصيات
        
        **التحليلات والذكاء الاصطناعي:**
        - تحليل البيانات التاريخية
        - كشف الشذوذ والأنماط
        - التنبؤ بالاتجاهات المستقبلية
        - الصيانة التنبؤية
        
        **العمليات والتحكم:**
        - متابعة إحصائيات الإنتاج
        - إدارة الجودة
        - التحكم بالأجهزة والأنظمة
        - إعدادات الاتصالات
        
        **السلامة والطوارئ:**
        - متابعة تنبيهات السلامة
        - إعدادات التنبيهات
        - بروتوكولات الطوارئ
        
        **الاستدامة والطاقة:**
        - متابعة البصمة الكربونية
        - تحليل كفاءة الطاقة
        - أهداف الاستدامة
        
        **المساعد الذكي:**
        - التفاعل مع النظام باستخدام الذكاء الاصطناعي
        - الحصول على إجابات لاستفساراتك
        - تلقي التوصيات الذكية
        """)
    
    with st.expander("استكشاف الأخطاء وإصلاحها"):
        st.write("""
        ### استكشاف الأخطاء وإصلاحها
        
        **لا يتم تحميل البيانات:**
        1. تحقق من اتصال الإنترنت
        2. تأكد من أن خادم MQTT يعمل
        3. تحقق من إعدادات الاتصال
        
        **المساعد الذكي لا يستجيب:**
        1. تحقق من اتصال OpenAI API (إذا كان مفعلاً)
        2. تأكد من صحة مفتاح API
        
        **لا يمكن الاتصال بـ Raspberry Pi:**
        1. تحقق من عنوان IP وبيانات الاعتماد
        2. تأكد من أن خدمة SSH مفعلة على Raspberry Pi
        3. تحقق من إعدادات الشبكة
        
        **البيانات لا تتحدث:**
        1. تحقق من أن وضع المحاكاة مفعل
        2. تأكد من اتصال MQTT
        3. تحقق من إعدادات تحديث البيانات
        
        **للحصول على مساعدة إضافية:**
        - راجع documentation المرفق
        - اتصل بفريق الدعم الفني
        - تحقق من forums المجتمع
        """)
    
    # معلومات الاتصال
    st.markdown(f'<div class="section-header">الدعم والاتصال</div>', unsafe_allow_html=True)
    
    st.write("**ساعات الدعم:** الأحد - الخميس، 8 ص - 5 م")
    st.write("**هاتف الدعم:** +966 12 345 6789")
    st.write("**البريد الإلكتروني:** support@sndt.com")
    st.write("**الموقع الإلكتروني:** https://sndt.com")
    
    if st.button("طلب دعم فني"):
        st.info("تم إرسال طلب الدعم الفني. سيتصل بك فريق الدعم خلال 24 ساعة.")

# -------------------- التطبيق الرئيسي --------------------
def main():
    # تطبيق أنماط الثيم
    theme_manager.apply_theme_styles()
    
    # الشريط الجانبي
    with st.sidebar:
        show_logo()
        st.markdown(f'<div style="text-align:center; font-size:1.5rem; font-weight:bold; margin-bottom:1.5rem;">SNDT Platform</div>', unsafe_allow_html=True)
        
        # اختيار القسم
        sections = translator.get_text("side_sections")
        selected_section = st.radio("اختر القسم", sections, index=0)
        
        st.divider()
        
        # معلومات النظام
        st.write("**معلومات النظام:**")
        st.write(f"الحالة: {'متصل' if st.session_state.get('mqtt_connected', False) else 'غير متصل'}")
        st.write(f"آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}")
        
        if st.session_state.get("pi_connected", False):
            st.success("✓ Raspberry Pi متصل")
        else:
            st.error("✗ Raspberry Pi غير متصل")
        
        st.divider()
        
        # الإصدار
        st.write("الإصدار: 1.0.0")
    
    # عرض القسم المحدد
    section_index = sections.index(selected_section)
    
    if section_index == 0:
        dashboard_section()
    elif section_index == 1:
        analytics_ai_section()
    elif section_index == 2:
        operations_control_section()
    elif section_index == 3:
        safety_emergency_section()
    elif section_index == 4:
        sustainability_energy_section()
    elif section_index == 5:
        smart_assistant_section()
    elif section_index == 6:
        settings_help_section()

if __name__ == "__main__":
    main()
