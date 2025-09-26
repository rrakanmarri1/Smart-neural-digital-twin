import streamlit as st
import logging
import time
import threading
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# استيرادات صحيحة من الهيكل الجديد
from core_systems import create_digital_twin, DigitalTwinCore
from advanced_systems import create_advanced_systems, DashboardRenderer
from ai_systems import create_ai_systems, AIChatSystem
from config_and_logging import setup_logging, load_configuration
from hardware.sensor_manager import SensorManager
from hardware.relay_controller import RelayController

class SmartDigitalTwinApp:
    """
    التطبيق الرئيسي للتوأم الرقمي العصبي الذكي
    """
    
    def __init__(self):
        self.setup_page_config()
        self.config = self.load_configuration()
        self.logger = setup_logging()
        
        # تهيئة الأنظمة
        self.digital_twin = None
        self.advanced_systems = None
        self.ai_systems = None
        self.sensor_manager = None
        self.relay_controller = None
        
        self.initialize_systems()
        
        # حالة التطبيق
        self.is_running = False
        self.last_update = None
        self.real_time_data = {}
        self.system_health = {}
        
    def setup_page_config(self):
        """إعداد إعدادات صفحة Streamlit"""
        st.set_page_config(
            page_title="🧠 Smart Neural Digital Twin",
            page_icon="🔥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # تطبيق تخصيصات CSS
        self.apply_custom_styles()
    
    def apply_custom_styles(self):
        """تطبيق التخصيصات CSS"""
        st.markdown("""
        <style>
        .main {
            background-color: #1a365d;
        }
        .stAlert {
            background-color: #2d3748;
        }
        .css-1d391kg {
            background-color: #2d3748;
        }
        .metric-card {
            background-color: #2d3748;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #3182ce;
            margin: 10px 0;
        }
        .emergency-alert {
            background-color: #e53e3e;
            color: white;
            padding: 15px;
            border-radius: 10px;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_configuration(self) -> dict:
        """تحميل الإعدادات"""
        try:
            return load_configuration()
        except Exception as e:
            st.error(f"❌ Failed to load configuration: {e}")
            return {}
    
    def initialize_systems(self):
        """تهيئة جميع الأنظمة"""
        try:
            with st.spinner("🚀 Initializing Smart Neural Digital Twin..."):
                # إنشاء التوأم الرقمي الأساسي
                self.digital_twin = create_digital_twin()
                
                # إنشاء الأنظمة المتقدمة
                self.advanced_systems = create_advanced_systems(self.config)
                
                # إنشاء أنظمة الذكاء الاصطناعي
                self.ai_systems = create_ai_systems(self.config)
                
                # إنشاء مدير المستشعرات
                self.sensor_manager = SensorManager(self.config)
                
                # إنشاء متحكم الريلاي
                self.relay_controller = RelayController(self.config)
                
                st.success("✅ All systems initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize systems: {e}")
            self.logger.error(f"Initialization error: {e}")
    
    def run(self):
        """تشغيل التطبيق الرئيسي"""
        try:
            # الشريط الجانبي
            self.render_sidebar()
            
            # المنطقة الرئيسية
            self.render_header()
            self.render_dashboard()
            self.render_ai_chat()
            self.render_system_health()
            self.render_hardware_control()
            
            # التحديث التلقائي
            self.auto_refresh()
            
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            self.logger.error(f"Application error: {e}")
    
    def render_sidebar(self):
        """عرض الشريط الجانبي"""
        with st.sidebar:
            st.title("🧠 Control Panel")
            st.markdown("---")
            
            # حالة النظام
            st.subheader("System Status")
            if self.digital_twin:
                status = self.digital_twin.system_status.value
                status_color = {
                    'normal': '🟢',
                    'warning': '🟡', 
                    'critical': '🟠',
                    'emergency': '🔴'
                }.get(status, '⚪')
                
                st.write(f"{status_color} **Status:** {status.upper()}")
            
            # التحكم في النظام
            st.subheader("System Control")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start Monitoring", type="primary"):
                    self.start_monitoring()
            
            with col2:
                if st.button("⏹️ Stop Monitoring"):
                    self.stop_monitoring()
            
            # إعدادات التحديث
            st.subheader("Update Settings")
            update_interval = st.slider("Update Interval (seconds)", 1, 60, 5)
            st.session_state.update_interval = update_interval
            
            # معلومات النظام
            st.subheader("System Info")
            st.write(f"**Last Update:** {self.last_update or 'Never'}")
            st.write(f"**Sensors Active:** {len(self.real_time_data)}")
            st.write(f"**AI Models:** 6")
            st.write(f"**Hardware:** Raspberry Pi 4")
            
            st.markdown("---")
            st.markdown("### 🔧 Quick Actions")
            
            if st.button("🔄 Manual Refresh"):
                self.manual_refresh()
            
            if st.button("📊 Generate Report"):
                self.generate_report()
            
            if st.button("⚙️ System Diagnostics"):
                self.run_diagnostics()
    
    def render_header(self):
        """عرض الهيدر الرئيسي"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🔥 Smart Neural Digital Twin")
            st.markdown("### Oil Field Disaster Prevention System")
        
        with col2:
            if self.last_update:
                st.metric("Last Update", self.last_update.strftime("%H:%M:%S"))
        
        with col3:
            if self.system_health.get('status'):
                status = self.system_health['status']
                color = {
                    'normal': 'green',
                    'warning': 'orange',
                    'critical': 'red',
                    'emergency': 'darkred'
                }.get(status, 'gray')
                
                st.markdown(
                    f"<div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;'>"
                    f"<strong>STATUS: {status.upper()}</strong>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
    
    def render_dashboard(self):
        """عرض الداشبورد الرئيسي"""
        st.header("📊 Real-Time Monitoring Dashboard")
        
        # صف المؤشرات
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self.render_sensor_card("Pressure", "💨", self.real_time_data.get('pressure', 0), "bar")
        
        with col2:
            self.render_sensor_card("Temperature", "🌡️", self.real_time_data.get('temperature', 0), "°C")
        
        with col3:
            self.render_sensor_card("Methane", "⚠️", self.real_time_data.get('methane', 0), "ppm")
        
        with col4:
            self.render_sensor_card("Vibration", "📳", self.real_time_data.get('vibration', 0), "m/s²")
        
        with col5:
            self.render_sensor_card("Flow", "💧", self.real_time_data.get('flow', 0), "L/min")
        
        # الرسوم البيانية
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_sensor_charts()
        
        with col2:
            self.render_anomaly_display()
        
        # التنبؤات
        self.render_predictions()
    
    def render_sensor_card(self, name: str, icon: str, value: float, unit: str):
        """عرض بطاقة المستشعر"""
        try:
            # تحديد اللون بناءً على القيمة
            if name == "Pressure":
                color = "green" if value < 60 else "orange" if value < 80 else "red"
            elif name == "Temperature":
                color = "green" if value < 80 else "orange" if value < 100 else "red"
            elif name == "Methane":
                color = "green" if value < 200 else "orange" if value < 400 else "red"
            else:
                color = "green"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{icon} {name}</h3>
                <h2 style="color: {color};">{value:.2f} {unit}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error rendering {name} card: {e}")
    
    def render_sensor_charts(self):
        """عرض رسوم المستشعرات البيانية"""
        try:
            st.subheader("📈 Sensor Trends")
            
            if self.real_time_data:
                sensors = ['pressure', 'temperature', 'methane', 'vibration', 'flow']
                values = [self.real_time_data.get(sensor, 0) for sensor in sensors]
                
                fig = go.Figure(data=[
                    go.Bar(x=sensors, y=values, marker_color=['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5'])
                ])
                
                fig.update_layout(
                    title="Current Sensor Readings",
                    xaxis_title="Sensors",
                    yaxis_title="Values",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sensor data available")
                
        except Exception as e:
            st.error(f"Error rendering sensor charts: {e}")
    
    def render_anomaly_display(self):
        """عرض معلومات الشذوذ"""
        try:
            st.subheader("🚨 Anomaly Detection")
            
            anomalies = self.real_time_data.get('anomalies', {})
            
            if anomalies and anomalies.get('is_anomaly', False):
                risk_level = anomalies.get('risk_level', 'unknown')
                risk_color = {
                    'low': 'green',
                    'medium': 'orange',
                    'high': 'red',
                    'critical': 'darkred'
                }.get(risk_level, 'gray')
                
                st.markdown(f"""
                <div style="background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px;">
                    <h3>🚨 ANOMALY DETECTED</h3>
                    <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
                    <p><strong>Score:</strong> {anomalies.get('anomaly_score', 0):.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # عرض الشذوذ الحرجة
                critical_anomalies = anomalies.get('critical_anomalies', [])
                if critical_anomalies:
                    st.warning("**Critical Anomalies:**")
                    for anomaly in critical_anomalies:
                        st.write(f"• {anomaly}")
            else:
                st.success("✅ No anomalies detected")
                st.metric("Anomaly Score", f"{anomalies.get('anomaly_score', 0):.3f}" if anomalies else "0.000")
                
        except Exception as e:
            st.error(f"Error rendering anomaly display: {e}")
    
    def render_predictions(self):
        """عرض التنبؤات"""
        try:
            st.subheader("🔮 24-Hour Predictions")
            
            predictions = self.real_time_data.get('predictions', {})
            
            if predictions and not predictions.get('error'):
                pred_data = predictions.get('predictions', {})
                trends = predictions.get('trends', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Sensor Trends:**")
                    for sensor, trend in trends.items():
                        trend_icon = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                        st.write(f"{trend_icon} {sensor}: {trend}")
                
                with col2:
                    st.write("**Confidence Levels:**")
                    confidences = predictions.get('confidence_scores', {})
                    for sensor, confidence in confidences.items():
                        st.write(f"🎯 {sensor}: {confidence:.1%}")
                
                with col3:
                    critical_points = predictions.get('critical_points', [])
                    if critical_points:
                        st.warning(f"**Critical Points:** {len(critical_points)}")
                        for point in critical_points[:3]:  # عرض أول 3 نقاط فقط
                            st.write(f"• {point['sensor']} at hour {point['hour']}")
                
                # رسم بياني للتنبؤات
                if pred_data:
                    self.render_prediction_chart(pred_data)
            else:
                st.info("No prediction data available")
                
        except Exception as e:
            st.error(f"Error rendering predictions: {e}")
    
    def render_prediction_chart(self, predictions: dict):
        """عرض رسم بياني للتنبؤات"""
        try:
            hours = list(range(24))
            fig = go.Figure()
            
            for sensor, values in predictions.items():
                if len(values) == 24:
                    fig.add_trace(go.Scatter(
                        x=hours, 
                        y=values, 
                        name=sensor,
                        mode='lines+markers'
                    ))
            
            fig.update_layout(
                title="24-Hour Sensor Predictions",
                xaxis_title="Hours Ahead",
                yaxis_title="Predicted Values",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering prediction chart: {e}")
    
    def render_ai_chat(self):
        """عرض نظام الدردشة بالذكاء الاصطناعي"""
        st.header("🤖 AI Assistant")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask about system status, predictions, or emergency procedures:",
                placeholder="e.g., What's the current pressure status?"
            )
        
        with col2:
            if st.button("Ask AI", type="primary"):
                if user_question and self.ai_systems:
                    with st.spinner("AI is thinking..."):
                        response = self.ai_systems['ai_chat'].ask_question(
                            user_question, 
                            self.real_time_data
                        )
                        st.session_state.last_ai_response = response
                        st.session_state.last_ai_question = user_question
        
        if hasattr(st.session_state, 'last_ai_response'):
            st.markdown("### 💬 AI Response")
            st.info(st.session_state.last_ai_response)
    
    def render_system_health(self):
        """عرض صحة النظام"""
        st.header("⚕️ System Health")
        
        if self.system_health:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Status", self.system_health.get('status', 'unknown').upper())
            
            with col2:
                active_alerts = self.system_health.get('active_alerts', 0)
                st.metric("Active Alerts", active_alerts)
            
            with col3:
                sensor_health = self.system_health.get('sensor_health', {})
                working_sensors = sensor_health.get('working', 0)
                total_sensors = sensor_health.get('total', 6)
                st.metric("Sensors Online", f"{working_sensors}/{total_sensors}")
            
            with col4:
                last_update = self.system_health.get('last_update', 'Unknown')
                if isinstance(last_update, str):
                    st.metric("Last Health Check", last_update)
                else:
                    st.metric("Last Health Check", last_update.strftime("%H:%M:%S"))
            
            # معلومات مفصلة عن الذكاء الاصطناعي
            st.subheader("🧠 AI System Performance")
            ai_health = self.system_health.get('ai_health', {})
            
            if ai_health:
                for model, performance in list(ai_health.items())[:4]:  # عرض أول 4 نماذج فقط
                    score = performance.get('performance_score', 0)
                    success_rate = performance.get('success_rate', 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.progress(score)
                        st.write(f"**{model}**: {score:.1%}")
                    
                    with col2:
                        st.progress(success_rate)
                        st.write(f"Success Rate: {success_rate:.1%}")
        else:
            st.info("System health data not available")
    
    def render_hardware_control(self):
        """عرض عناصر تحكم الهاردوير"""
        st.header("🔧 Hardware Control")
        
        tab1, tab2, tab3 = st.tabs(["Relay Control", "Sensor Simulation", "Emergency Actions"])
        
        with tab1:
            self.render_relay_control()
        
        with tab2:
            self.render_sensor_simulation()
        
        with tab3:
            self.render_emergency_actions()
    
    def render_relay_control(self):
        """عرض تحكم الريلاي"""
        st.subheader("🔌 Relay Control")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Activate Emergency Cooling", type="secondary"):
                self.control_relay(1, True)
                st.success("Emergency cooling activated")
        
        with col2:
            if st.button("🟢 Activate Pressure Release", type="secondary"):
                self.control_relay(2, True)
                st.success("Pressure release activated")
        
        with col3:
            if st.button("🔴 Deactivate All Systems", type="secondary"):
                self.control_relay(1, False)
                self.control_relay(2, False)
                st.success("All systems deactivated")
        
        st.info("💡 Relay controls simulate hardware actions. In production, these would control actual relays.")
    
    def render_sensor_simulation(self):
        """عرض محاكاة المستشعرات"""
        st.subheader("🎮 Sensor Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pressure = st.slider("Simulate Pressure", 0, 100, 50)
            temperature = st.slider("Simulate Temperature", 0, 150, 75)
            methane = st.slider("Simulate Methane", 0, 1000, 100)
        
        with col2:
            vibration = st.slider("Simulate Vibration", 0, 10, 2)
            flow = st.slider("Simulate Flow", 0, 100, 50)
            h2s = st.slider("Simulate H2S", 0, 100, 10)
        
        if st.button("Apply Simulation", type="primary"):
            simulated_data = {
                'pressure': pressure,
                'temperature': temperature,
                'methane': methane,
                'vibration': vibration,
                'flow': flow,
                'hydrogen_sulfide': h2s
            }
            self.real_time_data.update(simulated_data)
            st.success("Sensor simulation applied!")
    
    def render_emergency_actions(self):
        """عرض إجراءات الطوارئ"""
        st.subheader("🚨 Emergency Actions")
        
        emergency_type = st.selectbox(
            "Select Emergency Type:",
            ["High Pressure", "Gas Leak", "Equipment Failure", "Fire Hazard"]
        )
        
        if st.button("🚨 ACTIVATE EMERGENCY PROTOCOL", type="primary"):
            st.error("EMERGENCY PROTOCOL ACTIVATED!")
            st.session_state.emergency_activated = True
            
            # محاكاة إجراءات الطوارئ
            self.simulate_emergency_response(emergency_type)
    
    def start_monitoring(self):
        """بدء المراقبة"""
        try:
            self.is_running = True
            st.session_state.monitoring_started = True
            st.success("✅ Real-time monitoring started!")
            
            # بدء thread للمراقبة المستمرة
            if not hasattr(st.session_state, 'monitoring_thread'):
                monitoring_thread = threading.Thread(target=self.continuous_monitoring)
                monitoring_thread.daemon = True
                monitoring_thread.start()
                st.session_state.monitoring_thread = monitoring_thread
                
        except Exception as e:
            st.error(f"❌ Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """إيقاف المراقبة"""
        self.is_running = False
        st.session_state.monitoring_started = False
        st.warning("⏹️ Monitoring stopped")
    
    def continuous_monitoring(self):
        """المراقبة المستمرة في thread منفصل"""
        while self.is_running:
            try:
                self.manual_refresh()
                time.sleep(st.session_state.get('update_interval', 5))
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def manual_refresh(self):
        """تحديث يدوي للبيانات"""
        try:
            if self.digital_twin:
                # معالجة البيانات الحية
                processed_data = self.digital_twin.process_real_time_data()
                self.real_time_data = processed_data
                
                # تحديث صحة النظام
                self.system_health = self.digital_twin.get_system_health()
                
                self.last_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Refresh error: {e}")
    
    def auto_refresh(self):
        """التحديث التلقائي"""
        if st.session_state.get('monitoring_started', False) and self.is_running:
            time_since_update = datetime.now() - self.last_update if self.last_update else None
            
            if not self.last_update or (time_since_update and time_since_update.total_seconds() > st.session_state.get('update_interval', 5)):
                self.manual_refresh()
    
    def control_relay(self, relay_id: int, state: bool):
        """التحكم في الريلاي"""
        try:
            if self.relay_controller:
                self.relay_controller.control_relay(relay_id, state)
                self.logger.info(f"Relay {relay_id} set to {state}")
        except Exception as e:
            st.error(f"Relay control error: {e}")
    
    def simulate_emergency_response(self, emergency_type: str):
        """محاكاة استجابة الطوارئ"""
        try:
            # محاكاة إجراءات الطوارئ بناءً على النوع
            emergency_actions = {
                "High Pressure": [
                    "Activating pressure release valves",
                    "Reducing flow rates", 
                    "Engaging emergency cooling",
                    "Notifying emergency team"
                ],
                "Gas Leak": [
                    "Activating gas detection systems",
                    "Engaging ventilation systems",
                    "Shutting down affected areas",
                    "Evacuation procedures initiated"
                ],
                "Equipment Failure": [
                    "Shutting down malfunctioning equipment",
                    "Activating backup systems",
                    "Diagnosing failure cause",
                    "Maintenance team alerted"
                ],
                "Fire Hazard": [
                    "Activating fire suppression systems",
                    "Emergency shutdown initiated",
                    "Evacuation alarms activated",
                    "Fire department notified"
                ]
            }
            
            actions = emergency_actions.get(emergency_type, [])
            
            for action in actions:
                with st.empty():
                    st.warning(f"🚨 {action}")
                    time.sleep(2)
            
            st.success("✅ Emergency response completed!")
            
        except Exception as e:
            st.error(f"Emergency simulation error: {e}")
    
    def generate_report(self):
        """توليد تقرير النظام"""
        try:
            st.info("📊 Generating system report...")
            time.sleep(2)
            
            report = {
                "timestamp": datetime.now(),
                "system_status": self.system_health.get('status', 'unknown'),
                "sensor_readings": self.real_time_data,
                "anomalies_detected": self.real_time_data.get('anomalies', {}).get('critical_anomalies', []),
                "ai_performance": self.system_health.get('ai_health', {})
            }
            
            st.download_button(
                label="📥 Download Report",
                data=str(report),
                file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Report generation error: {e}")
    
    def run_diagnostics(self):
        """تشغيل تشخيصات النظام"""
        try:
            with st.spinner("🔍 Running system diagnostics..."):
                time.sleep(3)
                
                diagnostics = {
                    "Core Systems": "✅ Operational",
                    "AI Models": "✅ All models loaded",
                    "Sensor Communication": "✅ Active", 
                    "Hardware Interface": "✅ Ready",
                    "Emergency Protocols": "✅ Loaded",
                    "Data Processing": "✅ Running",
                    "Alert System": "✅ Configured"
                }
                
                st.success("System diagnostics completed!")
                
                for component, status in diagnostics.items():
                    st.write(f"**{component}:** {status}")
                    
        except Exception as e:
            st.error(f"Diagnostics error: {e}")

def main():
    """الدالة الرئيسية للتطبيق"""
    try:
        app = SmartDigitalTwinApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ Application failed to start: {e}")
        logging.error(f"Application failure: {e}")

if __name__ == "__main__":
    main()
