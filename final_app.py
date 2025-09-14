import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging

from config.settings import AdvancedConfig
from core_systems import AdvancedCoreSystem, create_core_system
from advanced_systems import AdvancedSystems
from twilio_integration import TwilioIntegration, create_twilio_integration
from ai_chat_system import AIChatSystem, create_ai_chat

class CompleteDashboard:
    def __init__(self):
        self.config = AdvancedConfig()
        self.setup_page()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page(self):
        """تهيئة صفحة Streamlit"""
        st.set_page_config(
            page_title="Oil Field Neural Digital Twin",
            page_icon="🛢️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
            <style>
            .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; }
            .emergency-alert { background-color: #ff4b4b; color: white; padding: 1rem; border-radius: 0.5rem; }
            </style>
        """, unsafe_allow_html=True)
    
    def setup_session_state(self):
        """تهيئة حالة الجلسة"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'anomalies' not in st.session_state:
            st.session_state.anomalies = []
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = []
    
    def initialize_components(self):
        """تهيئة جميع المكونات"""
        try:
            self.core_system = create_core_system(self.config)
            self.advanced_systems = AdvancedSystems(self.core_system, self.config)
            self.twilio = create_twilio_integration(self.config)
            self.ai_chat = create_ai_chat(self.config)
            
            st.success("✅ System initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            logging.exception("Initialization error")
    
    def run(self):
        """تشغيل لوحة التحكم"""
        st.title("🛢️ Oil Field Neural Digital Twin")
        st.markdown("---")
        
        # الشريط الجانبي
        with st.sidebar:
            self.render_sidebar()
        
        # علامات التبويب الرئيسية
        tabs = st.tabs(["Dashboard", "AI Chat", "Predictions", "Interventions", "Reverse Twin", "Settings"])
        
        with tabs[0]:
            self.render_dashboard()
        with tabs[1]:
            self.render_ai_chat()
        with tabs[2]:
            self.render_predictions()
        with tabs[3]:
            self.render_interventions()
        with tabs[4]:
            self.render_reverse_twin()
        with tabs[5]:
            self.render_settings()
    
    def render_sidebar(self):
        """عرض الشريط الجانبي"""
        st.header("🚨 Emergency System")
        
        # اختبار Twilio
        if st.button("📱 Test Twilio SMS", type="secondary"):
            result = self.twilio.test_connection()
            if result['success']:
                st.success("✅ SMS sent successfully!")
            else:
                st.error(f"❌ Failed: {result['error']}")
        
        st.header("📊 Live Data")
        status = self.core_system.get_system_status()
        st.metric("System Health", f"{status['health']}%")
        st.metric("Active Sensors", len(self.core_system.sensor_readings))
        st.metric("Emergency Mode", "🟢 Normal" if not status['emergency_mode'] else "🔴 Active")
    
    def render_dashboard(self):
        """لوحة التحكم الرئيسية"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Real-time Sensor Data")
            self.render_sensor_charts()
        
        with col2:
            st.subheader("⚠️ Anomaly Detection")
            self.render_anomalies()
        
        st.subheader("🤖 AI Recommendations")
        self.render_ai_recommendations()
    
    def render_ai_chat(self):
        """دردشة الذكاء الاصطناعي"""
        st.header("🤖 AI Chat Assistant")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about the oil field system..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.ai_chat.ask_question(prompt, self.get_chat_context())
                    st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    
    def render_reverse_twin(self):
        """التوأم الرقمي العكسي"""
        st.header("🔄 Reverse Digital Twin")
        
        with st.form("simulation_form"):
            st.subheader("Run Reverse Simulation")
            scenario_type = st.selectbox("Scenario Type", ["gas_leak", "pressure_surge", "equipment_failure"])
            duration = st.slider("Duration (hours)", 1, 24, 6)
            
            if st.form_submit_button("Run Simulation"):
                with st.spinner("Running reverse simulation..."):
                    scenario = {
                        "type": scenario_type,
                        "duration": duration,
                        "timestamp": datetime.now()
                    }
                    result = self.advanced_systems.handle_advanced_scenarios("reverse_simulation", scenario)
                    st.session_state.simulation_results.append(result)
                    st.success("✅ Simulation completed!")
        
        if st.session_state.simulation_results:
            st.subheader("Simulation Results")
            for result in st.session_state.simulation_results[-3:]:
                st.json(result)
    
    def get_chat_context(self) -> Dict[str, Any]:
        """الحصول على سياق المحادثة"""
        return {
            "system_status": self.core_system.get_system_status(),
            "recent_anomalies": st.session_state.anomalies[-5:] if st.session_state.anomalies else [],
            "current_time": datetime.now().isoformat()
        }

def main():
    """الدالة الرئيسية"""
    try:
        dashboard = CompleteDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.exception("Application crashed")

if __name__ == "__main__":
    main()
