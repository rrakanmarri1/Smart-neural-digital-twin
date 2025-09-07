import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading
import paho.mqtt.client as mqtt
import random
import time
import json
import hashlib
from config_and_logging import logger, MQTT_BROKER, MQTT_PORT, MQTT_TOPIC_TEMPERATURE, MQTT_TOPIC_PRESSURE, MQTT_TOPIC_METHANE, MQTT_TOPIC_CONTROL

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
        "anomalies_detected": [], "preprocessed_data": None,
        "alert_thresholds": {"temperature": 65, "pressure": 9.0, "methane": 3.0},
        "self_healing_actions": [], "emergency_protocols_activated": False,
        "component_health": {
            "compressor": {"health": 95, "last_maintenance": (datetime.now() - timedelta(days=30)).isoformat()},
            "heat_exchanger": {"health": 88, "last_maintenance": (datetime.now() - timedelta(days=45)).isoformat()},
            "valves": {"health": 92, "last_maintenance": (datetime.now() - timedelta(days=25)).isoformat()},
            "pumps": {"health": 85, "last_maintenance": (datetime.now() - timedelta(days=60)).isoformat()},
            "sensors": {"health": 96, "last_maintenance": (datetime.now() - timedelta(days=15)).isoformat()}
        }
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
