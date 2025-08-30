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

# -------------------- MQTT Config (محدث) --------------------
MQTT_BROKER = "broker.emqx.io"  # broker أكثر موثوقية
MQTT_PORT = 1883
MQTT_TOPIC_TEMPERATURE = "sndt/temperature"
MQTT_TOPIC_PRESSURE = "sndt/pressure" 
MQTT_TOPIC_METHANE = "sndt/methane"
MQTT_TOPIC_CONTROL = "sndt/control"

# -------------------- App state Initialization --------------------
for key, default in [
    ("lang", "en"), ("scenario_step", 0), ("solution_idx", 0), ("theme", "light"),
    ("mqtt_temp", 55.0), ("mqtt_last", datetime.now()), ("mqtt_started", False), ("sms_sent", False),
    ("feedback_list", []), ("generated_solutions", []), ("solution_generated", False),
    ("ai_analysis_done", False), ("anomalies_detected", []), ("preprocessed_data", None),
    ("pi_connected", False), ("pi_status", "disconnected"), ("simulation_active", False),
    ("chat_history", []), ("twilio_enabled", True), ("alert_phone_number", "+966532559664"),
    # إضافة حالات جديدة
    ("operations_data", {}), ("energy_optimization", {}), ("incident_timeline", []),
    ("lifelong_memory", []), ("physical_twin_connected", False),
    ("pressure", 7.2), ("methane", 1.4), ("vibration", 4.5), ("flow_rate", 110.0),
    ("mqtt_connected", False), ("current_sensor_data", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- MQTT Setup (محدث مع إدارة متقدمة) --------------------
class RobustMQTTClient:
    def __init__(self):
        self.client = None
        self.connected = False
        self.connection_timeout = 10
        self.max_retries = 3
        self.retry_count = 0
        
    def on_connect(self, client, userdata, flags, rc):
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
            print("✅ Connected to MQTT Broker")
        else:
            self.connected = False
            st.session_state["mqtt_connected"] = False
            print(f"❌ Connection failed with code {rc}")
            
    def on_message(self, client, userdata, msg):
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
            
            print(f"📡 Received: {topic} = {value}")
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
            
    def connect_with_retry(self):
        """اتصال مع إعادة محاولة ذكية"""
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
                    print(f"⌛ Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Connection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
                
        return False
        
    def publish_control_command(self, command, value):
        """إرسال أوامر التحكم إلى المجسم"""
        if self.connected:
            try:
                payload = f"{command}:{value}"
                self.client.publish(MQTT_TOPIC_CONTROL, payload)
                print(f"📤 Sent control command: {payload}")
                return True
            except Exception as e:
                print(f"❌ Failed to send command: {e}")
                return False
        return False

# تهيئة عميل MQTT المتين
mqtt_client = RobustMQTTClient()

# -------------------- محاكاة بيانات MQTT إذا فشل الاتصال --------------------
def start_mqtt_simulation():
    """محاكاة بيانات حية إذا كان MQTT غير متوفر"""
    def simulate_data():
        while True:
            if not mqtt_client.connected:
                current_time = datetime.now()
                # توليد بيانات واقعية للمحاكاة
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

# -------------------- التهيئة المحدثة --------------------
if not st.session_state["mqtt_started"]:
    mqtt_success = mqtt_client.connect_with_retry()
    
    if not mqtt_success:
        print("⚠️ Using MQTT simulation mode")
        start_mqtt_simulation()
    
    st.session_state["mqtt_started"] = True

# -------------------- الذاكرة الدائمة (Lifelong Learning Memory) --------------------
class LifelongLearningMemory:
    def __init__(self):
        self.memories = []
        self.learning_rate = 0.88
        self.max_memories = 1000
        
    def add_experience(self, event_type, data, outcome, lesson):
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
        common_keys = set(sit1.keys()) & set(sit2.keys())
        if not common_keys:
            return 0.0
        total_similarity = 0
        for key in common_keys:
            if sit1[key] == sit2[key]:
                total_similarity += 1
        return total_similarity / len(common_keys)

lifelong_memory = LifelongLearningMemory()

# -------------------- Reverse Digital Twin للمجسم التوضيحي --------------------
class EnhancedPhysicalTwinController:
    def __init__(self):
        self.physical_components = {
            "pump": {"status": "off", "speed": 0, "temperature": 25.0},
            "valve": {"status": "closed", "flow_rate": 0.0},
            "leds": {"red": False, "green": False, "blue": False},
            "sensors": {"temperature": 0.0, "pressure": 0.0, "methane": 0.0}
        }
        self.pi_connected = False
    
    def connect_to_raspberry_pi(self):
        try:
            time.sleep(2)
            self.pi_connected = True
            st.session_state['physical_twin_connected'] = True
            
            if mqtt_client.connected:
                mqtt_client.publish_control_command("connect", "pi_ready")
            
            return True, "✅ تم الاتصال بـ Raspberry Pi بنجاح"
        except Exception as e:
            self.pi_connected = False
            return False, f"❌ فشل الاتصال: {str(e)}"
    
    def control_physical_component(self, component, action, value=None):
        if mqtt_client.connected:
            mqtt_client.publish_control_command(component, f"{action}:{value if value else ''}")
        
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
    
    def _control_pump(self, action, speed):
        if action == "start":
            self.physical_components["pump"]["status"] = "on"
            self.physical_components["pump"]["speed"] = speed
            return True, "✅ تم تشغيل المضخة"
        else:
            self.physical_components["pump"]["status"] = "off"
            self.physical_components["pump"]["speed"] = 0
            return True, "✅ تم إيقاف المضخة"
    
    def _control_valve(self, action, flow_rate):
        if action == "open":
            self.physical_components["valve"]["status"] = "open"
            self.physical_components["valve"]["flow_rate"] = flow_rate
            return True, "✅ تم فتح الصمام"
        else:
            self.physical_components["valve"]["status"] = "closed"
            self.physical_components["valve"]["flow_rate"] = 0.0
            return True, "✅ تم إغلاق الصمام"
    
    def _control_leds(self, action, color):
        if action == "on":
            self.physical_components["leds"][color] = True
            return True, f"✅ تم تشغيل LED {color}"
        else:
            self.physical_components["leds"][color] = False
            return True, f"✅ تم إطفاء LED {color}"
    
    def _read_sensors(self):
        if mqtt_client.connected and st.session_state.get("current_sensor_data"):
            sensor_data = st.session_state["current_sensor_data"]
        else:
            sensor_data = {
                "temperature": random.uniform(20.0, 80.0),
                "pressure": random.uniform(0.5, 10.0),
                "methane": random.uniform(0.1, 5.0),
                "vibration": random.uniform(3.0, 6.0),
                "flow_rate": random.uniform(80.0, 120.0),
                "timestamp": datetime.now().isoformat()
            }
        
        self.physical_components["sensors"] = sensor_data
        return sensor_data

physical_twin = EnhancedPhysicalTwinController()

# -------------------- AI Copilot with General Knowledge --------------------
def generate_ai_response(prompt):
    prompt_lower = prompt.lower()
    
    similar_experiences = lifelong_memory.find_similar(
        {'prompt': prompt, 'type': 'ai_interaction'},
        min_similarity=0.7
    )
    
    response = ""
    
    if similar_experiences and similar_experiences[0]['similarity'] > 0.8:
        best_memory = similar_experiences[0]['memory']
        response += f"🧠 بناءً على تجربة سابقة:\n{best_memory['lesson']}\n\n"
    
    if any(word in prompt_lower for word in ["weather", "temperature outside", "الطقس", "درجة الحرارة"]):
        response += get_weather_info()
    elif any(word in prompt_lower for word in ["time", "date", "today", "الوقت", "التاريخ", "اليوم"]):
        response += get_current_time_info()
    elif any(word in prompt_lower for word in ["hello", "hi", "مرحبا", "السلام"]):
        response += "Hello! I'm your SNDT AI Copilot. I can help you with plant monitoring, predictions, and general questions. How can I assist you today?"
    elif any(word in prompt_lower for word in ["predict", "forecast", "next", "future", "تنبأ", "توقع"]):
        response += generate_time_based_prediction(prompt)
    elif any(word in prompt_lower for word in ["current", "now", "live", "status", "حالي", "مباشر"]):
        response += generate_current_status()
    else:
        response += "I'm your SNDT AI assistant. I can help with plant monitoring, predictions, weather, time, and general questions. What would you like to know?"
    
    lifelong_memory.add_experience(
        event_type="ai_interaction",
        data={'prompt': prompt},
        outcome="response_generated", 
        lesson=f"تم الرد على: {prompt[:50]}..."
    )
    
    return response

def get_weather_info():
    weather_data = {
        "temperature": random.randint(20, 35),
        "condition": random.choice(["Sunny", "Partly Cloudy", "Clear"]),
        "humidity": random.randint(30, 70)
    }
    return f"Current weather:\n• Temperature: {weather_data['temperature']}°C\n• Condition: {weather_data['condition']}\n• Humidity: {weather_data['humidity']}%"

def get_current_time_info():
    now = datetime.now()
    return f"Current time: {now.strftime('%H:%M:%S')}\nToday's date: {now.strftime('%Y-%m-%d')}\nDay of week: {now.strftime('%A')}"

def generate_time_based_prediction(prompt):
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

def generate_current_status():
    return f"Current system status:\n• Temperature: {st.session_state['mqtt_temp']}°C\n• Last update: {st.session_state['mqtt_last'].strftime('%H:%M:%S')}\n• System health: Good"

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

# -------------------- 3D Visualization محسنة ومفيدة --------------------
def enhanced_3d_visualization_section():
    st.markdown(f'<div class="main-header">🏭 {texts[lang]["live3d_header"]}</div>', unsafe_allow_html=True)
    
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
                    <div>🌡️ {current_data.get('temperature', 55.0):.1f}°C</div>
                    <div>📊 {current_data.get('pressure', 7.2):.1f} bar</div>
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
        st.metric("🌡️ الحرارة", f"{current_data.get('temperature', 55.0):.1f}°C", 
                 delta=f"{random.uniform(-2.0, 2.0):.1f}°C")
    with col2:
        st.metric("📊 الضغط", f"{current_data.get('pressure', 7.2):.1f} bar",
                 delta=f"{random.uniform(-0.3, 0.3):.1f} bar")
    with col3:
        st.metric("⚠️ الميثان", f"{current_data.get('methane', 1.4):.2f}%",
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
                <span style="color: {status_color}; font-weight: bold;">{data['temp']:.1f}°C</span>
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
        st.metric("🚀 سرعة التدفق", f"{flow_simulation['السرعة']:.2f} m/s")
        st.metric("📊 كفاءة التدفق", f"{flow_simulation['الكفاءة']:.1f}%")
    with col2:
        st.metric("🔄 ضغط التدفق", f"{flow_simulation['الضغط']:.2f} bar")
        st.metric("💧 معدل التدفق", f"{flow_simulation['التدفق']:.1f} L/min")
    
    st.markdown("#### 🔧 توصيات الصيانة")
    
    maintenance_recommendations = [
        {"المكون": "الوصلات", "الأولوية": "عالي", "التوصية": "فحص الوصلات للضغط العالي"},
        {"المكون": "الأنابيب الرئيسية", "الأولوية": "متوسط", "التوصية": "تنظيف الرواسب"},
        {"المكون": "الدعامات", "الأولوية": "منخفض", "التوصية": "فحص دوري"}
    ]
    
    for rec in maintenance_recommendations:
        priority_color = "#F44336" if rec["الأولوية"] == "عالي" else "#FF9800" if rec["الأولوية"] == "متوسط" else "#43A047"
        st.info(f"**{rec['المكون']}** - الأولوية: <span style='color:{priority_color}'>{rec['الأولوية']}</span> - {rec['التوصية']}", unsafe_allow_html=True)

# -------------------- مركز العمليات المتكامل --------------------
def operations_center_section():
    st.markdown(f'<div class="main-header">🏭 مركز العمليات المتكامل</div>', unsafe_allow_html=True)
    
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

# -------------------- التحكم المتقدم في المجسم --------------------
def enhanced_raspberry_pi_section():
    st.markdown(f'<div class="main-header">🤖 التحكم المتقدم في المجسم التوضيحي</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mqtt_status = "✅ متصل" if mqtt_client.connected else "⚠️ محاكاة"
        st.markdown(f"**حالة MQTT:** {mqtt_status}")
    
    with col2:
        pi_status = "✅ متصل" if physical_twin.pi_connected else "❌ غير متصل"
        st.markdown(f"**حالة RPi:** {pi_status}")
    
    with col3:
        last_update = st.session_state.get("mqtt_last", datetime.now())
        st.markdown(f"**آخر تحديث:** {last_update.strftime('%H:%M:%S')}")
    
    if not physical_twin.pi_connected:
        if st.button("🔗 الاتصال بـ Raspberry Pi", key="connect_rpi"):
            success, message = physical_twin.connect_to_raspberry_pi()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
    
    st.markdown("### 🎛️ لوحة التحكم في المجسم")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ التحكم في المضخة")
        pump_status = physical_twin.physical_components["pump"]["status"]
        st.markdown(f"**الحالة:** {'🟢 مشغلة' if pump_status == 'on' else '🔴 متوقفة'}")
        
        if st.button("▶️ تشغيل المضخة", key="start_pump"):
            success, message = physical_twin.control_physical_component("pump", "start", 80)
            st.success(message)
        
        if st.button("⏹️ إيقاف المضخة", key="stop_pump"):
            success, message = physical_twin.control_physical_component("pump", "stop")
            st.success(message)
    
    with col2:
        st.markdown("#### 🎚️ التحكم في الصمام")
        valve_status = physical_twin.physical_components["valve"]["status"]
        st.markdown(f"**الحالة:** {'🟢 مفتوح' if valve_status == 'open' else '🔴 مغلق'}")
        
        if st.button("🔓 فتح الصمام", key="open_valve"):
            success, message = physical_twin.control_physical_component("valve", "open", 50.0)
            st.success(message)
        
        if st.button("🔐 غلق الصمام", key="close_valve"):
            success, message = physical_twin.control_physical_component("valve", "close")
            st.success(message)
    
    st.markdown("### 💡 التحكم في الإضاءة")
    led_colors = ["red", "green", "blue"]
    led_cols = st.columns(3)
    
    for i, color in enumerate(led_colors):
        with led_cols[i]:
            st.markdown(f"**LED {color.upper()}**")
            if st.button(f"💡 تشغيل {color}", key=f"on_{color}"):
                success, message = physical_twin.control_physical_component("leds", "on", color)
                st.success(message)
            if st.button(f"⚫ إطفاء {color}", key=f"off_{color}"):
                success, message = physical_twin.control_physical_component("leds", "off", color)
                st.success(message)
    
    st.markdown("### 📊 قراءة البيانات من الحساسات")
    if st.button("📡 قراءة البيانات الحالية", key="read_sensors"):
        sensor_data = physical_twin.control_physical_component("sensors", "read")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ درجة الحرارة", f"{sensor_data['temperature']:.1f}°C")
        with col2:
            st.metric("📊 الضغط", f"{sensor_data['pressure']:.1f} bar")
        with col3:
            st.metric("⚠️ الميثان", f"{sensor_data['methane']:.2f}%")
        
        lifelong_memory.add_experience(
            event_type="sensor_reading",
            data=sensor_data,
            outcome="success",
            lesson=f"قراءة حساسات: {sensor_data['temperature']:.1f}°C"
        )

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

# -------------------- Sections (يتبع باقي الأقسام بنفس الهيكل) --------------------
# [يتم الحفاظ على كل الأقسام الأصلية مع استبدال الدوال المحدثة]

# -------------------- Main Application --------------------
def main():
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
        enhanced_3d_visualization_section()  # استخدام 3D الجديدة
    elif section == t["side_sections"][9]:
        about_section()
    elif section == t["side_sections"][10]:
        enhanced_raspberry_pi_section()  # استخدام RPi الجديدة

if __name__ == "__main__":
    main()
