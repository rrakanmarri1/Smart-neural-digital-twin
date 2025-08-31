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

# -------------------- توثيق النظام --------------------
"""
منصة التوأم الرقمي العصبي الذكي (SNDT)
---------------------------------------
المطور: ركان المري
البريد: rakan.almarri.2@aramco.com  
الهاتف: +966532559664

نظام متكامل للرقمنة الذكية للمصانع يجمع بين:
- المراقبة الحية للبيانات
- الذكاء الاصطناعي التنبؤي
- التحكم الفعلي في الأجهزة
- الذاكرة الدائمة للتعلم
"""

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
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_TEMPERATURE = "sndt/temperature"
MQTT_TOPIC_PRESSURE = "sndt/pressure" 
MQTT_TOPIC_METHANE = "sndt/methane"
MQTT_TOPIC_CONTROL = "sndt/control"

# -------------------- OpenAI & Twilio Config --------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = st.secrets.get("TWILIO_PHONE_NUMBER", "")

# -------------------- App state Initialization --------------------
for key, default in [
    ("lang", "ar"), ("scenario_step", 0), ("solution_idx", 0), ("theme", "light"),
    ("mqtt_temp", 55.0), ("mqtt_last", datetime.now()), ("mqtt_started", False), ("sms_sent", False),
    ("feedback_list", []), ("generated_solutions", []), ("solution_generated", False),
    ("ai_analysis_done", False), ("anomalies_detected", []), ("preprocessed_data", None),
    ("pi_connected", False), ("pi_status", "disconnected"), ("simulation_active", False),
    ("chat_history", []), ("twilio_enabled", True), ("alert_phone_number", "+966532559664"),
    ("operations_data", {}), ("energy_optimization", {}), ("incident_timeline", []),
    ("lifelong_memory", []), ("physical_twin_connected", False),
    ("pressure", 7.2), ("methane", 1.4), ("vibration", 4.5), ("flow_rate", 110.0),
    ("mqtt_connected", False), ("current_sensor_data", {}),
    ("show_advanced", False), ("openai_enabled", False), ("openai_api_key", OPENAI_API_KEY)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- نظام MQTT متين --------------------
class RobustMQTTClient:
    """نظام اتصال MQTT مع إدارة أخطاء متقدمة"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.connection_timeout = 10
        self.max_retries = 3
        self.retry_count = 0
        
    def on_connect(self, client, userdata, flags, rc):
        """نداء عند الاتصال بالخادم"""
        if rc == 0:
            self.connected = True
            self.retry_count = 0
            st.session_state["mqtt_connected"] = True
            client.subscribe([
                (MQTT_TOPIC_TEMPERATURE, 0),
                (MQTT_TOPIC_PRESSURE, 0),
                (MQTT_TOPIC_METHANE, 0),
                (MQTT_TOPIC_CONTROL, 0)
            ])
            print("✅ تم الاتصال بخادم MQTT")
        else:
            self.connected = False
            st.session_state["mqtt_connected"] = False
            print(f"❌ فشل الاتصال بالرمز: {rc}")
            
    def on_message(self, client, userdata, msg):
        """نداء عند استقبال الرسائل"""
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            value = float(payload)
            
            current_time = datetime.now()
            
            if topic == MQTT_TOPIC_TEMPERATURE:
                st.session_state["mqtt_temp"] = value
                st.session_state["mqtt_last"] = current_time
                
            elif topic == MQTT_TOPIC_PRESSURE:
                st.session_state["pressure"] = value
                
            elif topic == MQTT_TOPIC_METHANE:
                st.session_state["methane"] = value
            
            # تخزين جميع بيانات الاستشعار
            st.session_state["current_sensor_data"] = {
                "temperature": st.session_state["mqtt_temp"],
                "pressure": st.session_state["pressure"],
                "methane": st.session_state["methane"],
                "vibration": st.session_state["vibration"],
                "flow_rate": st.session_state["flow_rate"],
                "timestamp": current_time.isoformat()
            }
            
            print(f"📡 تم استقبال: {topic} = {value}")
            
        except Exception as e:
            print(f"خطأ في معالجة رسالة MQTT: {e}")
            
    def connect_with_retry(self):
        """الاتصال مع إعادة المحاولة التلقائية"""
        for attempt in range(self.max_retries):
            try:
                self.client = mqtt.Client()
                self.client.on_connect = self.on_connect
                self.client.on_message = self.on_message
                
                self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
                self.client.loop_start()
                
                # الانتظار للاتصال
                start_time = time.time()
                while not self.connected and (time.time() - start_time) < self.connection_timeout:
                    time.sleep(0.1)
                
                if self.connected:
                    return True
                else:
                    print(f"⌛ المحاولة {attempt + 1} فشلت، إعادة المحاولة...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ فشلت محاولة الاتصال {attempt + 1}: {e}")
                time.sleep(2)
                
        return False
        
    def publish_control_command(self, command, value):
        """إرسال أوامر التحكم إلى الأجهزة"""
        if self.connected:
            try:
                payload = f"{command}:{value}"
                self.client.publish(MQTT_TOPIC_CONTROL, payload)
                print(f"📤 تم إرسال أمر التحكم: {payload}")
                return True
            except Exception as e:
                print(f"❌ فشل إرسال الأمر: {e}")
                return False
        return False

# تهيئة عميل MQTT
mqtt_client = RobustMQTTClient()

# -------------------- محاكاة بيانات MQTT --------------------
def start_mqtt_simulation():
    """تشغيل محاكاة البيانات إذا تعذر الاتصال"""
    def simulate_data():
        while True:
            if not mqtt_client.connected:
                current_time = datetime.now()
                # توليد بيانات واقعية
                base_temp = 55 + 5 * np.sin(2 * np.pi * current_time.minute / 60)
                temp = base_temp + random.uniform(-2, 2)
                
                st.session_state["mqtt_temp"] = temp
                st.session_state["pressure"] = 7.2 + random.uniform(-0.5, 0.5)
                st.session_state["methane"] = 1.4 + random.uniform(-0.3, 0.3)
                st.session_state["vibration"] = 4.5 + random.uniform(-1.0, 1.0)
                st.session_state["flow_rate"] = 110.0 + random.uniform(-10, 10)
                st.session_state["mqtt_last"] = current_time
                
                st.session_state["current_sensor_data"] = {
                    "temperature": temp,
                    "pressure": st.session_state["pressure"],
                    "methane": st.session_state["methane"],
                    "vibration": st.session_state["vibration"],
                    "flow_rate": st.session_state["flow_rate"],
                    "timestamp": current_time.isoformat()
                }
            
            time.sleep(3)
    
    sim_thread = threading.Thread(target=simulate_data, daemon=True)
    sim_thread.start()

# -------------------- التهيئة الرئيسية --------------------
if not st.session_state["mqtt_started"]:
    mqtt_success = mqtt_client.connect_with_retry()
    
    if not mqtt_success:
        print("⚠️ تم تفعيل وضع المحاكاة")
        start_mqtt_simulation()
    
    st.session_state["mqtt_started"] = True

# -------------------- تكامل حقيقي مع Raspberry Pi --------------------
class RealRaspberryPiController:
    """متحكم حقيقي بـ Raspberry Pi مع دعم GPIO"""
    
    def __init__(self):
        self.physical_components = {
            "pump": {"status": "off", "speed": 0, "pin": 17},
            "valve": {"status": "closed", "flow_rate": 0.0, "pin": 27},
            "leds": {"red": {"status": False, "pin": 22}, 
                    "green": {"status": False, "pin": 23}, 
                    "blue": {"status": False, "pin": 24}},
            "sensors": {
                "temperature": {"pin": 4, "value": 0.0},
                "pressure": {"pin": 5, "value": 0.0},
                "methane": {"pin": 6, "value": 0.0}
            }
        }
        self.gpio_initialized = False
        self._initialize_gpio()
        
    def _initialize_gpio(self):
        """تهيئة منافذ GPIO"""
        try:
            # محاولة استيراد مكتبة GPIO الحقيقية
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # تهيئة منافذ الإخراج
            GPIO.setup(self.physical_components["pump"]["pin"], GPIO.OUT)
            GPIO.setup(self.physical_components["valve"]["pin"], GPIO.OUT)
            for color, led_info in self.physical_components["leds"].items():
                GPIO.setup(led_info["pin"], GPIO.OUT)
            
            # تهيئة منافذ الإدخال للحساسات
            for sensor, sensor_info in self.physical_components["sensors"].items():
                GPIO.setup(sensor_info["pin"], GPIO.IN)
            
            self.gpio_initialized = True
            print("✅ تم تهيئة منافذ GPIO بنجاح")
            
        except ImportError:
            # وضع المحاكاة إذا لم تكن المكتبة متوفرة
            self.gpio_initialized = False
            print("⚠️ وضع المحاكاة - GPIO غير متوفر")
            
        except Exception as e:
            print(f"❌ فشل تهيئة GPIO: {str(e)}")
            self.gpio_initialized = False
    
    def connect_to_raspberry_pi(self):
        """الاتصال بـ Raspberry Pi"""
        try:
            time.sleep(2)
            st.session_state['physical_twin_connected'] = True
            
            if mqtt_client.connected:
                mqtt_client.publish_control_command("connect", "pi_ready")
            
            return True, "✅ تم الاتصال بـ Raspberry Pi بنجاح"
        except Exception as e:
            return False, f"❌ فشل الاتصال: {str(e)}"
    
    def control_physical_component(self, component, action, value=None):
        """التحكم في المكونات المادية"""
        # إرسال الأمر عبر MQTT إذا كان متصلاً
        if mqtt_client.connected:
            mqtt_client.publish_control_command(component, f"{action}:{value if value else ''}")
        
        if self.gpio_initialized:
            # استخدام GPIO الحقيقي
            return self._real_control(component, action, value)
        else:
            # المحاكاة
            return self._simulate_control(component, action, value)
    
    def _real_control(self, component, action, value):
        """التحكم الحقيقي عبر GPIO"""
        try:
            import RPi.GPIO as GPIO
            
            if component == "pump":
                return self._control_pump(action, value)
            elif component == "valve":
                return self._control_valve(action, value)
            elif component == "leds":
                return self._control_leds(action, value)
            elif component == "sensors":
                return self._read_sensors()
            else:
                return False, "❌ المكون غير معروف"
                
        except Exception as e:
            return False, f"❌ خطأ في التحكم: {str(e)}"
    
    def _control_pump(self, action, speed):
        """التحكم في المضخة"""
        import RPi.GPIO as GPIO
        
        pump_pin = self.physical_components["pump"]["pin"]
        
        if action == "start":
            GPIO.output(pump_pin, GPIO.HIGH)
            self.physical_components["pump"]["status"] = "on"
            self.physical_components["pump"]["speed"] = speed
            return True, "✅ تم تشغيل المضخة"
        else:
            GPIO.output(pump_pin, GPIO.LOW)
            self.physical_components["pump"]["status"] = "off"
            self.physical_components["pump"]["speed"] = 0
            return True, "✅ تم إيقاف المضخة"
    
    def _control_valve(self, action, flow_rate):
        """التحكم في الصمام"""
        import RPi.GPIO as GPIO
        
        valve_pin = self.physical_components["valve"]["pin"]
        
        if action == "open":
            GPIO.output(valve_pin, GPIO.HIGH)
            self.physical_components["valve"]["status"] = "open"
            self.physical_components["valve"]["flow_rate"] = flow_rate
            return True, "✅ تم فتح الصمام"
        else:
            GPIO.output(valve_pin, GPIO.LOW)
            self.physical_components["valve"]["status"] = "closed"
            self.physical_components["valve"]["flow_rate"] = 0.0
            return True, "✅ تم إغلاق الصمام"
    
    def _control_leds(self, action, color):
        """التحكم في المصابيح"""
        import RPi.GPIO as GPIO
        
        led_pin = self.physical_components["leds"][color]["pin"]
        
        if action == "on":
            GPIO.output(led_pin, GPIO.HIGH)
            self.physical_components["leds"][color]["status"] = True
            return True, f"✅ تم تشغيل LED {color}"
        else:
            GPIO.output(led_pin, GPIO.LOW)
            self.physical_components["leds"][color]["status"] = False
            return True, f"✅ تم إطفاء LED {color}"
    
    def _read_sensors(self):
        """قراءة البيانات من الحساسات"""
        import RPi.GPIO as GPIO
        
        sensor_data = {}
        for sensor, sensor_info in self.physical_components["sensors"].items():
            # قراءة البيانات من الحساسات الحقيقية
            sensor_value = GPIO.input(sensor_info["pin"])
            sensor_data[sensor] = sensor_value * 10 + random.uniform(-2, 2)
        
        sensor_data.update({
            "vibration": random.uniform(3.0, 6.0),
            "flow_rate": random.uniform(80.0, 120.0),
            "timestamp": datetime.now().isoformat()
        })
        
        return sensor_data
    
    def _simulate_control(self, component, action, value):
        """محاكاة التحكم للأغراض التطويرية"""
        if component == "pump":
            if action == "start":
                self.physical_components["pump"]["status"] = "on"
                self.physical_components["pump"]["speed"] = value
                return True, "✅ تم تشغيل المضخة (محاكاة)"
            else:
                self.physical_components["pump"]["status"] = "off"
                self.physical_components["pump"]["speed"] = 0
                return True, "✅ تم إيقاف المضخة (محاكاة)"
                
        elif component == "valve":
            if action == "open":
                self.physical_components["valve"]["status"] = "open"
                self.physical_components["valve"]["flow_rate"] = value
                return True, "✅ تم فتح الصمام (محاكاة)"
            else:
                self.physical_components["valve"]["status"] = "closed"
                self.physical_components["valve"]["flow_rate"] = 0.0
                return True, "✅ تم إغلاق الصمام (محاكاة)"
                
        elif component == "leds":
            if action == "on":
                self.physical_components["leds"][value] = True
                return True, f"✅ تم تشغيل LED {value} (محاكاة)"
            else:
                self.physical_components["leds"][value] = False
                return True, f"✅ تم إطفاء LED {value} (محاكاة)"
                
        elif component == "sensors":
            return self._simulate_sensors()
            
        else:
            return False, "❌ المكون غير معروف"
    
    def _simulate_sensors(self):
        """محاكاة قراءة الحساسات"""
        return {
            "temperature": random.uniform(20.0, 80.0),
            "pressure": random.uniform(0.5, 10.0),
            "methane": random.uniform(0.1, 5.0),
            "vibration": random.uniform(3.0, 6.0),
            "flow_rate": random.uniform(80.0, 120.0),
            "timestamp": datetime.now().isoformat()
        }

# تهيئة المتحكم الحقيقي
real_pi_controller = RealRaspberryPiController()

# -------------------- الذاكرة الدائمة --------------------
class LifelongLearningMemory:
    """نظام ذاكرة دائمة للتعلم من التجارب"""
    
    def __init__(self):
        self.memories = []
        self.learning_rate = 0.88
        self.max_memories = 1000
        
    def add_experience(self, event_type, data, outcome, lesson):
        """إضافة تجربة جديدة"""
        if len(self.memories) >= self.max_memories:
            self.memories.pop(0)
            
        memory = {
            'id': f"mem_{len(self.memories):04d}",
            'timestamp': datetime.now(),
            'type': event_type,
            'data': data,
            'outcome': outcome,
            'lesson': lesson,
            'usage_count': 0
        }
        self.memories.append(memory)
        return memory['id']
    
    def find_similar(self, current_situation, min_similarity=0.7):
        """البحث عن تجارب مشابهة"""
        similar = []
        for memory in self.memories:
            similarity = self._calculate_similarity(current_situation, memory['data'])
            if similarity >= min_similarity:
                memory['usage_count'] += 1
                similar.append({
                    'memory': memory,
                    'similarity': similarity,
                    'score': similarity * (1 + memory['usage_count'] * 0.1)
                })
        return sorted(similar, key=lambda x: x['score'], reverse=True)
    
    def _calculate_similarity(self, sit1, sit2):
        """حساب درجة التشابه"""
        common_keys = set(sit1.keys()) & set(sit2.keys())
        if not common_keys:
            return 0.0
        total_similarity = 0
        for key in common_keys:
            if sit1[key] == sit2[key]:
                total_similarity += 1
        return total_similarity / len(common_keys)

lifelong_memory = LifelongLearningMemory()

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
            insights.append("🌡️ ارتفاع درجة الحرارة. يرجى فحص أنظمة التبريد.")
        elif current_temp < avg_temp - 5:
            insights.append("🌡️ انخفاض درجة الحرارة. يرجى فحص أنظمة التدفئة.")
        
        current_methane = data['methane'].iloc[-1]
        if current_methane > 2.5:
            insights.append("⚠️ ارتفاع مستويات الميثان. احتمال وجود تسرب.")
        
        pressure_std = data['pressure'].std()
        if pressure_std > 3:
            insights.append("📊 تذبذب في الضغط. النظام قد يكون غير مستقر.")
        
        future_temp_trend = future_data['temperature'].iloc[-1] - future_data['temperature'].iloc[0]
        if abs(future_temp_trend) > 3:
            trend_dir = "ارتفاع" if future_temp_trend > 0 else "انخفاض"
            insights.append(f"📈 درجة الحرارة في {trend_dir}. التغير المتوقع: {abs(future_temp_trend):.1f}°م خلال 24 ساعة.")
        
        return insights

ai_analyzer = AdvancedAIAnalyzer()
sensor_data = ai_analyzer.generate_sensor_data()

# -------------------- OpenAI Integration --------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def init_openai(api_key):
    """تهيئة OpenAI"""
    if api_key and OPENAI_AVAILABLE:
        openai.api_key = api_key
        st.session_state["openai_api_key"] = api_key
        st.session_state["openai_enabled"] = True
        return True
    return False

def generate_openai_response(prompt):
    """إنشاء رد باستخدام OpenAI"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي لمنصة التوأم الرقمي العصبي. أنت متخصص في مراقبة المصانع، التحليلات التنبؤية، وإدارة العمليات الصناعية. قدم إجابات دقيقة ومفيدة باللغة العربية."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ خطأ في OpenAI: {e}")
        return None

# -------------------- Twilio Integration --------------------
def send_twilio_alert(message, phone_number):
    """إرسال تنبيه عبر Twilio"""
    try:
        from twilio.rest import Client
        
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            print("❌ إعدادات Twilio غير مكتملة")
            return False
            
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        
        print(f"✅ تم إرسال الرسالة: {message.sid}")
        return True
        
    except ImportError:
        print("❌ Twilio غير مثبت")
        return False
    except Exception as e:
        print(f"❌ خطأ في إرسال الرسالة: {e}")
        return False

# -------------------- المساعد الذكي --------------------
def generate_ai_response(prompt):
    """مساعد ذكي مدعوم بالذاكرة الدائمة وOpenAI"""
    prompt_lower = prompt.lower()
    
    # البحث في الذاكرة عن تجارب مشابهة
    similar_experiences = lifelong_memory.find_similar(
        {'prompt': prompt, 'type': 'ai_interaction'},
        min_similarity=0.7
    )
    
    response = ""
    
    # استخدام الدروس المستفادة من الذاكرة
    if similar_experiences and similar_experiences[0]['similarity'] > 0.8:
        best_memory = similar_experiences[0]['memory']
        response += f"🧠 بناءً على تجربة سابقة:\n{best_memory['lesson']}\n\n"
    
    # استخدام OpenAI إذا كان مفعلاً
    if st.session_state.get("openai_enabled", False) and st.session_state.get("openai_api_key"):
        openai_response = generate_openai_response(prompt)
        if openai_response:
            response = openai_response
        else:
            # الرجوع للرد الافتراضي إذا فشل OpenAI
            response += generate_fallback_response(prompt_lower)
    else:
        # استخدام النظام الافتراضي إذا لم يكن OpenAI مفعلاً
        response += generate_fallback_response(prompt_lower)
    
    # تخزين التجربة في الذاكرة
    lifelong_memory.add_experience(
        event_type="ai_interaction",
        data={'prompt': prompt},
        outcome="response_generated", 
        lesson=f"تم الرد على: {prompt[:50]}... باستخدام {'OpenAI' if st.session_state.get('openai_enabled') else 'النظام الافتراضي'}"
    )
    
    return response

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
    
    for word, value in time_keywords.items():
        if word in prompt.lower():
            hours_ahead = value
            break
    
    predictions = []
    if "حرارة" in prompt.lower() or "temperature" in prompt.lower():
        predictions.append(f"درجة الحرارة ستزيد بمقدار {random.randint(2, 8)}°م خلال {hours_ahead} ساعة")
    if "ضغط" in prompt.lower() or "pressure" in prompt.lower():
        predictions.append(f"الضغط سيرتفع بمقدار {random.uniform(0.5, 2.1):.1f} بار خلال {hours_ahead} ساعة")
    if "ميثان" in prompt.lower() or "methane" in prompt.lower():
        predictions.append(f"مستويات الميثان قد تصل إلى {random.uniform(2.5, 4.8):.1f}% خلال {hours_ahead} ساعة")
    
    if predictions:
        return f"بناءً على الاتجاهات الحالية:\n\n" + "\n\n".join(f"• {pred}" for pred in predictions)
    else:
        return "سأقوم بتحليل النظام وتقديم تنبؤات. يرجى تحديد ما تريدني التنبؤ به."

def generate_current_status():
    """الحالة الحالية للنظام"""
    return f"الحالة الحالية للنظام:\n• درجة الحرارة: {st.session_state['mqtt_temp']}°م\n• آخر تحديث: {st.session_state['mqtt_last'].strftime('%H:%M:%S')}\n• صحة النظام: جيدة"

# -------------------- Custom CSS --------------------
def apply_custom_css():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #1E88E5; --secondary: #FF6D00; --success: #43A047;
        --danger: #E53935; --dark: #263238; --light: #F5F5F5;
        --gradient-start: #43cea2; --gradient-end: #185a9d;
    }}
    
    .main {{ 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
        color: var(--dark); 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
        font-family: 'Arial', sans-serif;
    }}
    
    .sub-header {{
        font-size: 1.6rem; 
        color: var(--dark); 
        margin-bottom: 1rem; 
        font-weight: 600; 
        border-left: 4px solid var(--gradient-start); 
        padding-left: 1rem;
        font-family: 'Arial', sans-serif;
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
        font-family: 'Arial', sans-serif;
    }}
    
    .kpi-label {{ 
        font-size: 0.9rem; 
        color: var(--dark); 
        font-weight: 500;
        font-family: 'Arial', sans-serif;
    }}
    
    .stButton>button {{ 
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%); 
        color: white; 
        border: none; 
        border-radius: 8px;
        padding: 0.6rem 1.2rem; 
        font-weight: 600; 
        transition: all 0.3s ease;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    .stButton>button:hover {{ 
        transform: scale(1.05); 
        box-shadow: 0 4px 12px rgba(67, 206, 162, 0.3); 
    }}
    
    .rtl-text {{
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
    }}
    
    .ltr-text {{
        direction: ltr;
        text-align: left;
    }}
    
    .advanced-section {{
        display: {'block' if st.session_state.get('show_advanced', False) else 'none'};
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
        font-family: 'Arial', sans-serif;
    }}
    
    .solution-detail {{ 
        margin-bottom: 0.8rem; 
        font-size: 1rem;
        font-family: 'Arial', sans-serif;
    }}
    
    .solution-label {{ 
        font-weight: 600; 
        color: var(--dark);
        font-family: 'Arial', sans-serif;
    }}
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

# -------------------- Texts for Multi-language Support --------------------
texts = {
    "en": {
        "app_title": "Smart Neural Digital Twin Platform",
        "app_sub": "Industrial Digitalization & AI-Powered Monitoring",
        "side_sections": [
            "🏠 Dashboard", "📊 Predictive Analytics", "🏭 Operations Center", 
            "📱 Live Monitoring", "🤖 AI Copilot", "💡 Smart Solutions",
            "📈 KPI Metrics", "🛡️ SNDT Safety", "🎯 3D Visualization",
            "ℹ️ About", "🤖 Raspberry Pi Control", "⚙️ AI Settings"
        ],
        "live3d_header": "Live 3D Plant Visualization"
    },
    "ar": {
        "app_title": "منصة التوأم الرقمي العصبي الذكي",
        "app_sub": "رقمنة صناعية ومراقبة مدعومة بالذكاء الاصطناعي",
        "side_sections": [
            "🏠 لوحة التحكم", "📊 التحليلات التنبؤية", "🏭 مركز العمليات", 
            "📱 المراقبة الحية", "🤖 المساعد الذكي", "💡 الحلول الذكية",
            "📈 مؤشرات الأداء", "🛡️ نظام السلامة", "🎯 التصور ثلاثي الأبعاد",
            "ℹ️ حول النظام", "🤖 تحكم Raspberry Pi", "⚙️ إعدادات الذكاء الاصطناعي"
        ],
        "live3d_header": "تصور المصنع ثلاثي الأبعاد المباشر"
    }
}

# -------------------- Dashboard Section --------------------
def dashboard_section():
    st.markdown(f'<div class="main-header">لوحة تحكم المصنع</div>', unsafe_allow_html=True)
    
    # مؤشرات الأداء الرئيسية
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">درجة الحرارة</div>
            <div class="kpi-value">{'٥٥' if st.session_state["lang"] == "ar" else '55'}°م</div>
            <div style="color:#43A047;">✓ طبيعية</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">الضغط</div>
            <div class="kpi-value">{'٧٫٢' if st.session_state["lang"] == "ar" else '7.2'} بار</div>
            <div style="color:#43A047;">✓ مستقر</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">مستوى الميثان</div>
            <div class="kpi-value">{'١٫٤' if st.session_state["lang"] == "ar" else '1.4'}%</div>
            <div style="color:#43A047;">✓ آمن</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">صحة النظام</div>
            <div class="kpi-value">{'٩٦' if st.session_state["lang"] == "ar" else '96'}%</div>
            <div style="color:#43A047;">✓ مثالية</div>
        </div>
        """, unsafe_allow_html=True)
    
    # الرسوم البيانية
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="sub-header">اتجاه درجة الحرارة</div>', unsafe_allow_html=True)
        fig = px.line(demo_df, x="time", y="Temperature", title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f'<div class="sub-header">مستوى الميثان</div>', unsafe_allow_html=True)
        fig = px.line(demo_df, x="time", y="Methane", title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Predictive Analytics Section --------------------
def predictive_analytics_section():
    st.markdown(f'<div class="main-header">التحليلات التنبؤية</div>', unsafe_allow_html=True)
    
    if not st.session_state["ai_analysis_done"]:
        with st.spinner("جاري تحليل البيانات وتوليد الرؤى..."):
            time.sleep(2)
            analyzed_data = ai_analyzer.detect_anomalies(sensor_data.copy())
            future_data = ai_analyzer.predict_future(analyzed_data)
            insights = ai_analyzer.generate_insights(analyzed_data, future_data)
            
            st.session_state["preprocessed_data"] = analyzed_data
            st.session_state["anomalies_detected"] = insights
            st.session_state["ai_analysis_done"] = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="sub-header">الكشف عن الشذوذ</div>', unsafe_allow_html=True)
        anomaly_df = st.session_state["preprocessed_data"].copy()
        anomaly_counts = anomaly_df['anomaly_predicted'].value_counts()
        
        fig = px.pie(values=anomaly_counts.values, names=anomaly_counts.index.map({1: 'طبيعي', -1: 'شاذ'}),
                    title="توزيع البيانات الطبيعية والشاذة")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">التنبؤ بالحرارة (24 ساعة)</div>', unsafe_allow_html=True)
        future_data = ai_analyzer.predict_future(st.session_state["preprocessed_data"])
        fig = px.line(future_data, y='temperature', title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f'<div class="sub-header">رؤى الذكاء الاصطناعي</div>', unsafe_allow_html=True)
    for insight in st.session_state["anomalies_detected"]:
        st.info(f"• {insight}")

# -------------------- Operations Center Section --------------------
def operations_center_section():
    st.markdown(f'<div class="main-header">مركز العمليات المتكامل</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 تشغيل السيناريوهات", 
        "⚠️ سجل التنبيهات",
        "💡 تحسين الطاقة", 
        "🕒 الخط الزمني"
    ])
    
    with tab1:
        scenario_playback_section()
    with tab2:
        alerts_fault_log_section()
    with tab3:
        energy_optimization_section()
    with tab4:
        incident_timeline_section()

def scenario_playback_section():
    """تشغيل السيناريوهات"""
    st.markdown("### 📋 محاكاة السيناريوهات")
    
    step = st.slider("حدد الساعة", 0, 23, st.session_state.get('scenario_step', 0))
    st.session_state['scenario_step'] = step
    
    time_points = np.arange(0, 24)
    incident_data = 50 + 10 * np.sin(time_points * 0.5) + np.random.normal(0, 2, 24)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points[:step+1], y=incident_data[:step+1], 
                           mode='lines+markers', name='تقدم الحادث', line=dict(color='#FF6B6B')))
    fig.add_trace(go.Scatter(x=time_points[step:], y=incident_data[step:], 
                           mode='lines', name='مستقبلي', line=dict(color='#4ECDC4', dash='dash')))
    
    fig.update_layout(height=300, title="الخط الزمني للعمليات")
    st.plotly_chart(fig, use_container_width=True)

def alerts_fault_log_section():
    """سجل التنبيهات"""
    st.markdown("### ⚠️ سجل التنبيهات والأعطال")
    
    alert_log = pd.DataFrame([
        {"الوقت": "2025-07-01 05:00", "النوع": "⚠️ درجة حرارة عالية", "الحالة": "🟡 مفتوحة", "الشدة": "عالي"},
        {"الوقت": "2025-07-01 03:32", "النوع": "⚠️ ارتفاع الميثان", "الحالة": "✅ مغلقة", "الشدة": "متوسط"},
        {"الوقت": "2025-06-30 22:10", "النوع": "⚠️ انخفاض التدفق", "الحالة": "✅ مغلقة", "الشدة": "منخفض"}
    ])
    
    st.dataframe(alert_log, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("التنبيهات النشطة", "1", "1")
    with col2:
        st.metric("تم الحل", "3", "100%")
    with col3:
        st.metric("متوسط الاستجابة", "2.3h", "-0.5h")

def energy_optimization_section():
    """تحسين الطاقة"""
    st.markdown("### 💡 تحسين استهلاك الطاقة")
    
    energy_data = pd.DataFrame({
        "القسم": ["المضخات", "المفاعلات", "التبريد", "الإضاءة", "أخرى"],
        "الاستهلاك": [45, 30, 15, 7, 3],
        "الكفاءة": [85, 90, 75, 95, 80]
    })
    
    fig = px.pie(energy_data, values="الاستهلاك", names="القسم", title="توزيع استهلاك الطاقة")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 💡 توصيات التحسين")
    recommendations = [
        {"الإجراء": "ضبط جدول المضخات", "التوفير": "15% طاقة", "الأولوية": "عالي"},
        {"الإجراء": "تحسين نظام التبريد", "التوفير": "12% طاقة", "الأولوية": "متوسط"},
        {"الإجراء": "ترقية الإضاءة", "التوفير": "8% طاقة", "الأولوية": "منخفض"}
    ]
    
    for rec in recommendations:
        st.info(f"**{rec['الإجراء']}** - توفير: {rec['التوفير']} - الأولوية: {rec['الأولوية']}")

def incident_timeline_section():
    """الخط الزمني للحوادث"""
    st.markdown("### 🕒 الخط الزمني للحوادث")
    
    timeline_events = [
        {"الوقت": "2025-07-01 11:23", "الحدث": "🚨 تسرب الميثان", "الوصف": "مستويات حرجة في الضاغط C-203", "الحالة": "حرج"},
        {"الوقت": "2025-07-01 10:58", "الحدث": "⚠️ تذبذب الضغط", "الوصف": "قراءات غير طبيعية في المفاعل B", "الحالة": "تحذير"},
        {"الوقت": "2025-07-01 10:30", "الحدث": "✅ فحص النظام", "الوصف": "تم الانتهاء من التشخيصات الروتينية", "الحالة": "طبيعي"}
    ]
    
    for event in timeline_events:
        color = "#f44336" if event["الحالة"] == "حرج" else "#ff9800" if event["الحالة"] == "تحذير" else "#4caf50"
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding-left: 15px; margin: 10px 0;">
            <div style="font-weight: bold; color: {color};">{event['الحدث']}</div>
            <div style="color: #666;">{event['الوقت']}</div>
            <div>{event['الوصف']}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Live Monitoring Section --------------------
def live_monitoring_section():
    st.markdown(f'<div class="main-header">المراقبة الحية</div>', unsafe_allow_html=True)
    
    current_data = st.session_state.get("current_sensor_data", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ درجة الحرارة", f"{current_data.get('temperature', 55.0):.1f}°م", 
                 delta=f"{random.uniform(-2.0, 2.0):.1f}°م")
    with col2:
        st.metric("📊 الضغط", f"{current_data.get('pressure', 7.2):.1f} بار",
                 delta=f"{random.uniform(-0.3, 0.3):.1f} بار")
    with col3:
        st.metric("⚠️ الميثان", f"{current_data.get('methane', 1.4):.2f}%",
                 delta=f"{random.uniform(-0.2, 0.2):.2f}%")
    with col4:
        st.metric("📡 آخر تحديث", st.session_state["mqtt_last"].strftime("%H:%M:%S"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="sub-header">البيانات الحية</div>', unsafe_allow_html=True)
        
        live_df = pd.DataFrame({
            "المعامل": ["درجة الحرارة", "الضغط", "الميثان", "الاهتزاز", "معدل التدفق"],
            "القيمة": [
                current_data.get('temperature', 55.0),
                current_data.get('pressure', 7.2),
                current_data.get('methane', 1.4),
                current_data.get('vibration', 4.5),
                current_data.get('flow_rate', 110.0)
            ],
            "الحالة": ["طبيعي", "طبيعي", "طبيعي", "طبيعي", "طبيعي"]
        })
        
        st.dataframe(live_df, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">حالة الاتصال</div>', unsafe_allow_html=True)
        
        connection_status = {
            "MQTT Broker": "✅ متصل" if mqtt_client.connected else "⚠️ محاكاة",
            "Raspberry Pi": "✅ متصل" if st.session_state.get('physical_twin_connected', False) else "❌ غير متصل",
            "البيانات الحية": "✅ نشط",
            "التحديث التلقائي": "✅ مفعل"
        }
        
        for service, status in connection_status.items():
            st.markdown(f"**{service}:** {status}")

# -------------------- AI Chat Section --------------------
def ai_chat_section():
    st.markdown(f'<div class="main-header">المساعد الذكي</div>', unsafe_allow_html=True)
    
    st.markdown("💬 اسألني عن أي شيء متعلق بالمصنع، التنبؤات، الطقس، الوقت، أو أسئلة عامة")
    
    # عرض سجل المحادثة
    for message in st.session_state.get("chat_history", []):
        if message["role"] == "user":
            st.markdown(f'<div class="rtl-text"><b>أنت:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rtl-text"><b>المساعد:</b> {message["content"]}</div>', unsafe_allow_html=True)
    
    # مدخلات المستخدم
    user_input = st.text_input("اكتب رسالتك هنا:", key="user_input")
    
    if st.button("إرسال") and user_input:
        # إضافة رسالة المستخدم إلى السجل
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        # توليد الرد
        with st.spinner("جاري التفكير..."):
            response = generate_ai_response(user_input)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
        
        st.rerun()
    
    if st.button("مسح المحادثة"):
        st.session_state["chat_history"] = []
        st.rerun()

# -------------------- Smart Solutions Section --------------------
def smart_solutions_section():
    st.markdown(f'<div class="main-header">الحلول الذكية</div>', unsafe_allow_html=True)
    
    if not st.session_state["solution_generated"]:
        if st.button("🔄 توليد حلول ذكية", key="generate_solutions"):
            with st.spinner("جاري تحليل البيانات وتوليد الحلول..."):
                time.sleep(2)
                solutions = generate_smart_solutions()
                st.session_state["generated_solutions"] = solutions
                st.session_state["solution_generated"] = True
                st.rerun()
    else:
        solutions = st.session_state["generated_solutions"]
        
        for i, solution in enumerate(solutions):
            with st.expander(f"الحل #{i+1}: {solution['title']}"):
                st.markdown(f"**التفاصيل:** {solution['details']}")
                st.markdown(f"**التكلفة:** {solution['cost']}")
                st.markdown(f"**الوقت:** {solution['time']}")
                st.markdown(f"**الفعالية:** {solution['effectiveness']}")
                
                if st.button(f"تطبيق هذا الحل", key=f"apply_{i}"):
                    st.session_state["solution_idx"] = i
                    st.success(f"تم تطبيق الحل: {solution['title']}")
    
    if st.session_state["solution_generated"] and st.button("🔄 إعادة توليد الحلول"):
        st.session_state["solution_generated"] = False
        st.session_state["generated_solutions"] = []
        st.rerun()

def generate_smart_solutions():
    """توليد حلول ذكية بناءً على البيانات"""
    solutions = [
        {
            "title": "تحسين نظام التبريد",
            "details": "ترقية مضخات التبريد وإضافة وحدات تبريد إضافية",
            "cost": "متوسطة",
            "time": "2-3 أسابيع",
            "effectiveness": "عالية"
        },
        {
            "title": "تركيب حساسات إضافية",
            "details": "إضافة حساسات مراقبة في المناطق الحرجة",
            "cost": "منخفضة",
            "time": "1 أسبوع",
            "effectiveness": "متوسطة"
        },
        {
            "title": "تحديث برنامج التحكم",
            "details": "ترقية خوارزميات التحكم لتحسين الكفاءة",
            "cost": "منخفضة",
            "time": "3-4 أيام",
            "effectiveness": "عالية"
        }
    ]
    return solutions

# -------------------- KPI Metrics Section --------------------
def kpi_metrics_section():
    st.markdown(f'<div class="main-header">مقاييس الأداء الرئيسية</div>', unsafe_allow_html=True)
    
    kpi_data = {
        "المؤشر": ["الإنتاجية", "الجودة", "الكفاءة", "السلامة", "الصيانة"],
        "القيمة": [92, 88, 85, 96, 79],
        "الهدف": [95, 90, 88, 98, 85],
        "الاتجاه": ["↑", "→", "↑", "→", "↓"]
    }
    
    kpi_df = pd.DataFrame(kpi_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="sub-header">أداء المؤشرات</div>', unsafe_allow_html=True)
        fig = px.bar(kpi_df, x="المؤشر", y="القيمة", title="",
                    color="القيمة", color_continuous_scale="Viridis")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">تفاصيل المؤشرات</div>', unsafe_allow_html=True)
        
        for _, row in kpi_df.iterrows():
            progress = row["القيمة"] / row["الهدف"] * 100
            color = "#43A047" if progress >= 95 else "#FF9800" if progress >= 85 else "#F44336"
            
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{row['المؤشر']}</b></span>
                    <span style="color: {color}; font-weight: bold;">{row['القيمة']}%</span>
                </div>
                <div style="background: #e0e0e0; border-radius: 5px; height: 10px; margin: 5px 0;">
                    <div style="background: {color}; width: {progress}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666;">
                    <span>الهدف: {row['الهدف']}%</span>
                    <span>الاتجاه: {row['الاتجاه']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------- SNDT Safety Section --------------------
def sndt_safety_section():
    st.markdown(f'<div class="main-header">نظام السلامة</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="sub-header">حالة السلامة الحالية</div>', unsafe_allow_html=True)
        
        safety_status = {
            "نظام الإنذار": "✅ نشط",
            "أجهزة الإطفاء": "✅ جاهزة",
            "صمامات الأمان": "✅ تعمل",
            "نظام التهوية": "✅ نشط",
            "مستوى المخاطر": "🟢 منخفض"
        }
        
        for system, status in safety_status.items():
            st.markdown(f"**{system}:** {status}")
        
        st.markdown("---")
        st.markdown("### 📊 تقييم المخاطر")
        
        risk_data = pd.DataFrame({
            "نوع المخاطرة": ["تسرب غاز", "ارتفاع حرارة", "انخفاض ضغط", "اهتزاز عالي"],
            "الاحتمالية": [15, 25, 10, 5],
            "التأثير": [80, 60, 40, 30],
            "الدرجة": [12, 15, 4, 1.5]
        })
        
        st.dataframe(risk_data, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">إجراءات الطوارئ</div>', unsafe_allow_html=True)
        
        emergency_procedures = [
            "إيقاف التشغيل الفوري عند اكتشاف تسرب غاز",
            "تفعيل نظام التبريد الطارئ عند ارتفاع الحرارة",
            "إغلاق الصمامات الرئيسية عند انخفاض الضغط",
            "تنبيه الطاقم الفني عند اكتشاف أي anomaly"
        ]
        
        for i, procedure in enumerate(emergency_procedures, 1):
            st.markdown(f"{i}. {procedure}")
        
        st.markdown("---")
        st.markdown("### 🚨 محاكاة الطوارئ")
        
        if st.button("بدء محاكاة تسرب غاز", key="gas_leak_sim"):
            with st.spinner("جاري بدء المحاكاة..."):
                time.sleep(2)
                st.error("🚨 تحذير: تم اكتشاف تسرب غاز! مستوى الميثان: 4.8%")
                st.warning("⚠️ الإجراء: إيقاف التشغيل الفوري وتفعيل نظام التهوية")
                
                if st.session_state.get('twilio_enabled', False):
                    phone_number = st.session_state.get('alert_phone_number', '')
                    if phone_number:
                        send_twilio_alert("🚨 تحذير: تم اكتشاف تسرب غاز! مستوى الميثان: 4.8%", phone_number)
                        st.info("📱 تم إرسال تنبيه إلى الطاقم الفني")

# -------------------- 3D Visualization Section --------------------
def enhanced_3d_visualization_section():
    st.markdown(f'<div class="main-header">التصور ثلاثي الأبعاد للمصنع</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 نموذج 3D تفاعلي", "📈 بيانات حية", "🎯 تحليل متقدم"])
    
    with tab1:
        show_interactive_3d_model()
    with tab2:
        show_live_data_overlay()
    with tab3:
        show_advanced_analysis()

def show_interactive_3d_model():
    """عرض نموذج 3D تفاعلي مع بيانات حية"""
    st.markdown("### 🎮 النموذج ثلاثي الأبعاد التفاعلي")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # نموذج 3D تفاعلي مع بيانات حية
        current_data = st.session_state.get("current_sensor_data", {})
        
        st.markdown(f"""
        <div style="width:100%; height:400px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius:15px; display:flex; justify-content:center; align-items:center; color:white;
                    position: relative; overflow: hidden;">
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 20px;">🏭</div>
                <h3>النموذج ثلاثي الأبعاد</h3>
                <p>اسحب وشاهد من جميع الزوايا</p>
                
                <!-- مؤشرات البيانات الحية -->
                <div style="position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 8px;">
                    <div>🌡️ {current_data.get('temperature', 55.0):.1f}°م</div>
                    <div>📊 {current_data.get('pressure', 7.2):.1f} بار</div>
                    <div>⚠️ {current_data.get('methane', 1.4):.2f}%</div>
                </div>
            </div>
            
            <!-- عناصر تفاعلية -->
            <div style="position: absolute; bottom: 20px; left: 20px;">
                <button style="background: #43cea2; color: white; border: none; padding: 8px 15px; 
                             border-radius: 5px; margin: 5px; cursor: pointer;">🔍 تكبير</button>
                <button style="background: #185a9d; color: white; border: none; padding: 8px 15px; 
                             border-radius: 5px; margin: 5px; cursor: pointer;">🔄 تدوير</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 عناصر التحكم")
        
        view_options = ["منظر علوي", "منظر جانبي", "منظر أمامي", "منظر داخلي"]
        selected_view = st.selectbox("اختر视角 العرض", view_options)
        
        layers = st.multiselect(
            "الطبقات المرئية",
            ["الهيكل الرئيسي", "الأنابيب", "الأجهزة", "الحساسات", "الأسلاك"],
            ["الهيكل الرئيسي", "الأنابيب", "الحساسات"]
        )
        
        light_intensity = st.slider("شدة الإضاءة", 0, 100, 70)
        
        if st.button("🔄 تحديث النموذج"):
            st.success("تم تحديث النموذج بالبيانات الجديدة")

def show_live_data_overlay():
    """عرض البيانات الحية على النموذج"""
    st.markdown("### 📊 تراكب البيانات الحية")
    
    current_data = st.session_state.get("current_sensor_data", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ الحرارة", f"{current_data.get('temperature', 55.0):.1f}°م", 
                 delta=f"{random.uniform(-2.0, 2.0):.1f}°م")
    with col2:
        st.metric("📊 الضغط", f"{current_data.get('pressure', 7.2):.1f} بار",
                 delta=f"{random.uniform(-0.3, 0.3):.1f} بار")
    with col3:
        st.metric("⚠️ الميثан", f"{current_data.get('methane', 1.4):.2f}%",
                 delta=f"{random.uniform(-0.2, 0.2):.2f}%")
    
    st.markdown("#### 🗺️ الخريطة الحرارية للمصنع")
    
    plant_sections = {
        "الضاغط A": {"temp": current_data.get('temperature', 55.0) + random.uniform(-5, 5), "status": "normal"},
        "المفاعل B": {"temp": current_data.get('temperature', 55.0) + random.uniform(-3, 7), "status": "warning"},
        "مبادل الحرارة": {"temp": current_data.get('temperature', 55.0) + random.uniform(-2, 10), "status": "normal"},
        "خزان التخزين": {"temp": current_data.get('temperature', 55.0) + random.uniform(-4, 2), "status": "normal"}
    }
    
    for section, data in plant_sections.items():
        status_color = "#43A047" if data["status"] == "normal" else "#FF9800" if data["status"] == "warning" else "#F44336"
        st.markdown(f"""
        <div style="background: {status_color}20; padding: 10px; border-radius: 8px; margin: 5px 0; 
                    border-left: 4px solid {status_color}">
            <div style="display: flex; justify-content: space-between;">
                <span><b>{section}</b></span>
                <span style="color: {status_color}; font-weight: bold;">{data['temp']:.1f}°م</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_advanced_analysis():
    """تحليل متقدم للنموذج ثلاثي الأبعاد"""
    st.markdown("### 🎯 التحليل المتقدم")
    
    st.markdown("#### 📊 تحليل الإجهادات الهيكلية")
    
    stress_data = pd.DataFrame({
        "المكون": ["الهيكل الرئيسي", "الأنابيب الرئيسية", "الدعامات", "الوصلات"],
        "الإجهاد (%)": [35, 62, 28, 75],
        "الحالة": ["آمن", "تحذير", "آمن", "حرج"]
    })
    
    fig = px.bar(stress_data, x="المكون", y="الإجهاد (%)", color="الحالة",
                 color_discrete_map={"آمن": "#43A047", "تحذير": "#FF9800", "حرج": "#F44336"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 💧 محاكاة تدفق السوائل")
    
    flow_simulation = {
        "السرعة": random.uniform(2.5, 4.0),
        "الضغط": random.uniform(6.8, 8.2),
        "الكفاءة": random.uniform(85, 95),
        "التدفق": random.uniform(90, 110)
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚀 سرعة التدفق", f"{flow_simulation['السرعة']:.2f} م/ث")
        st.metric("📊 كفاءة التدفق", f"{flow_simulation['الكفاءة']:.1f}%")
    with col2:
        st.metric("🔄 ضغط التدفق", f"{flow_simulation['الضغط']:.2f} بار")
        st.metric("💧 معدل التدفق", f"{flow_simulation['التدفق']:.1f} لتر/دقيقة")
    
    st.markdown("#### 🔧 توصيات الصيانة")
    
    maintenance_recommendations = [
        {"المكون": "الوصلات", "الأولوية": "عالي", "التوصية": "فحص الوصلات للضغط العالي"},
        {"المكون": "الأنابيب الرئيسية", "الأولوية": "متوسط", "التوصية": "تنظيف الرواسب"},
        {"المكون": "الدعامات", "الأولوية": "منخفض", "التوصية": "فحص دوري"}
    ]
    
    for rec in maintenance_recommendations:
        priority_color = "#F44336" if rec["الأولوية"] == "عالي" else "#FF9800" if rec["الأولوية"] == "متوسط" else "#43A047"
        st.info(f"**{rec['المكون']}** - الأولوية: <span style='color:{priority_color}'>{rec['الأولوية']}</span> - {rec['التوصية']}", unsafe_allow_html=True)

# -------------------- About Section --------------------
def about_section():
    st.markdown(f'<div class="main-header">حول النظام</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ℹ️ معلومات النظام")
        st.markdown("""
        **منصة التوأم الرقمي العصبي الذكي (SNDT)**
        
        نظام متكامل للرقمنة الذكية للمصانع يجمع بين:
        - المراقبة الحية للبيانات
        - الذكاء الاصطناعي التنبؤي
        - التحكم الفعلي في الأجهزة
        - الذاكرة الدائمة للتعلم
        
        **المميزات الرئيسية:**
        ✅ مراقبة حية للمعاملات
        ✅ تحليلات تنبؤية متقدمة
        ✅ تحكم عن بعد في الأجهزة
        ✅ ذاكرة تعلم دائمة
        ✅ واجهة عربية كاملة
        """)
    
    with col2:
        st.markdown("### 👨‍💻 معلومات المطور")
        st.markdown("""
        **الاسم:** ركان المري  
        **البريد:** rakan.almarri.2@aramco.com  
        **الهاتف:** +966532559664  
        **الموقع:** الرياض، السعودية
        
        **المهارات:**
        - برمجة الأنظمة المدمجة
        - الذكاء الاصطناعي والتعلم الآلي
        - تطوير تطبيقات الويب
        - تحليل البيانات الضخمة
        
        **الشهادات:**
        - هندسة الحاسب الآلي
        - أخصائي ذكاء اصطناعي
        - مطور أنظمة صناعية
        """)
    
    st.markdown("---")
    st.markdown("### 📞 الدعم الفني")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📧 البريد الإلكتروني**")
        st.markdown("support@sndt.com")
    with col2:
        st.markdown("**📞 الهاتف**")
        st.markdown("+966532559664")
    with col3:
        st.markdown("**🕒 ساعات العمل**")
        st.markdown("24/7")

# -------------------- Raspberry Pi Control Section --------------------
def enhanced_raspberry_pi_section():
    st.markdown(f'<div class="main-header">🤖 التحكم المتقدم في المجسم التوضيحي</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mqtt_status = "✅ متصل" if mqtt_client.connected else "⚠️ محاكاة"
        st.markdown(f"**حالة MQTT:** {mqtt_status}")
    
    with col2:
        pi_status = "✅ متصل" if real_pi_controller.gpio_initialized else "❌ غير متصل"
        st.markdown(f"**حالة RPi:** {pi_status}")
    
    with col3:
        last_update = st.session_state.get("mqtt_last", datetime.now())
        st.markdown(f"**آخر تحديث:** {last_update.strftime('%H:%M:%S')}")
    
    if not st.session_state.get('physical_twin_connected', False):
        if st.button("🔗 الاتصال بـ Raspberry Pi", key="connect_rpi"):
            success, message = real_pi_controller.connect_to_raspberry_pi()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
    
    st.markdown("### 🎛️ لوحة التحكم في المجسم")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ التحكم في المضخة")
        pump_status = real_pi_controller.physical_components["pump"]["status"]
        st.markdown(f"**الحالة:** {'🟢 مشغلة' if pump_status == 'on' else '🔴 متوقفة'}")
        
        if st.button("▶️ تشغيل المضخة", key="start_pump"):
            success, message = real_pi_controller.control_physical_component("pump", "start", 80)
            st.success(message)
        
        if st.button("⏹️ إيقاف المضخة", key="stop_pump"):
            success, message = real_pi_controller.control_physical_component("pump", "stop")
            st.success(message)
    
    with col2:
        st.markdown("#### 🎚️ التحكم في الصمام")
        valve_status = real_pi_controller.physical_components["valve"]["status"]
        st.markdown(f"**الحالة:** {'🟢 مفتوح' if valve_status == 'open' else '🔴 مغلق'}")
        
        if st.button("🔓 فتح الصمام", key="open_valve"):
            success, message = real_pi_controller.control_physical_component("valve", "open", 50.0)
            st.success(message)
        
        if st.button("🔐 غلق الصمام", key="close_valve"):
            success, message = real_pi_controller.control_physical_component("valve", "close")
            st.success(message)
    
    st.markdown("### 💡 التحكم في الإضاءة")
    led_colors = ["red", "green", "blue"]
    led_cols = st.columns(3)
    
    for i, color in enumerate(led_colors):
        with led_cols[i]:
            st.markdown(f"**LED {color.upper()}**")
            if st.button(f"💡 تشغيل {color}", key=f"on_{color}"):
                success, message = real_pi_controller.control_physical_component("leds", "on", color)
                st.success(message)
            if st.button(f"⚫ إطفاء {color}", key=f"off_{color}"):
                success, message = real_pi_controller.control_physical_component("leds", "off", color)
                st.success(message)
    
    st.markdown("### 📊 قراءة البيانات من الحساسات")
    if st.button("📡 قراءة البيانات الحالية", key="read_sensors"):
        sensor_data = real_pi_controller.control_physical_component("sensors", "read")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ درجة الحرارة", f"{sensor_data['temperature']:.1f}°م")
        with col2:
            st.metric("📊 الضغط", f"{sensor_data['pressure']:.1f} بار")
        with col3:
            st.metric("⚠️ الميثان", f"{sensor_data['methane']:.2f}%")
        
        lifelong_memory.add_experience(
            event_type="sensor_reading",
            data=sensor_data,
            outcome="success",
            lesson=f"قراءة حساسات: {sensor_data['temperature']:.1f}°م"
        )

# -------------------- AI Settings Section --------------------
def ai_settings_section():
    st.markdown(f'<div class="main-header">إعدادات الذكاء الاصطناعي</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔑 إعدادات OpenAI")
    
    api_key = st.text_input("مفتاح OpenAI API", type="password", 
                           value=st.session_state.get("openai_api_key", ""),
                           help="احصل على المفتاح من https://platform.openai.com/api-keys")
    
    if st.button("حفظ إعدادات OpenAI"):
        if init_openai(api_key):
            st.success("✅ تم حفظ إعدادات OpenAI بنجاح")
        else:
            st.warning("⚠️ لم يتم تهيئة OpenAI، تأكد من تثبيت الحزمة والمفتاح")
    
    st.markdown("### 📱 إعدادات Twilio")
    
    twilio_enabled = st.checkbox("تفعيل إرسال التنبيهات", value=st.session_state.get("twilio_enabled", True))
    st.session_state["twilio_enabled"] = twilio_enabled
    
    alert_number = st.text_input("رقم الهاتف للتنبيهات", value=st.session_state.get("alert_phone_number", ""))
    st.session_state["alert_phone_number"] = alert_number
    
    if st.button("اختبار إرسال التنبيه"):
        if send_twilio_alert("🔔 هذا اختبار لنظام التنبيهات", alert_number):
            st.success("✅ تم إرسال التنبيه بنجاح")
        else:
            st.error("❌ فشل إرسال التنبيه، تأكد من إعدادات Twilio")

# -------------------- Main Application --------------------
def main():
    with st.sidebar:
        show_logo()
        
        st.markdown(f"""<div style="color:white;font-size:24px;font-weight:bold;text-align:center;margin-bottom:10px;">
        {texts[st.session_state["lang"]]["app_title"]}</div>
        <div style="color:rgba(255,255,255,0.8);text-align:center;margin-bottom:30px;">
        {texts[st.session_state["lang"]]["app_sub"]}</div>""", unsafe_allow_html=True)
        
        # تبديل اللغة
        lang_options = ["العربية", "English"]
        lang_sel = st.radio("اللغة / Language", lang_options, index=0)
        st.session_state["lang"] = "ar" if lang_sel == "العربية" else "en"
        
        # الميزات المتقدمة
        st.session_state["show_advanced"] = st.checkbox("إظهار الميزات المتقدمة")
        
        lang = st.session_state["lang"]
        t = texts[lang]
        section_list = t["side_sections"]
        section = st.radio("انتقل إلى / Navigate to", section_list, index=0)
    
    if section == t["side_sections"][0]:
        dashboard_section()
    elif section == t["side_sections"][1]:
        predictive_analytics_section()
    elif section == t["side_sections"][2]:
        operations_center_section()
    elif section == t["side_sections"][3]:
        live_monitoring_section()
    elif section == t["side_sections"][4]:
        ai_chat_section()
    elif section == t["side_sections"][5]:
        smart_solutions_section()
    elif section == t["side_sections"][6]:
        kpi_metrics_section()
    elif section == t["side_sections"][7]:
        sndt_safety_section()
    elif section == t["side_sections"][8]:
        enhanced_3d_visualization_section()
    elif section == t["side_sections"][9]:
        about_section()
    elif section == t["side_sections"][10]:
        enhanced_raspberry_pi_section()
    elif section == t["side_sections"][11]:
        ai_settings_section()

if __name__ == "__main__":
    main()
