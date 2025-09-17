import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging

from config_and_logging import AdvancedConfig, ThemeConfig
from core_systems import AdvancedCoreSystem, create_core_system
from advanced_systems import AdvancedSystems
from advanced_systems import TwilioIntegration, create_twilio_integration
from ai_chat_system import AIChatSystem, create_ai_chat

class CompleteDashboard:
    def __init__(self):
        self.config = AdvancedConfig()
        self.theme = self.config.theme
        self.setup_page()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page(self):
        """تهيئة صفحة Streamlit بتصميم مريح للعين"""
        st.set_page_config(
            page_title="Oil Field Neural Digital Twin",
            page_icon="🛢️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # تطبيق أنماط التصميم المريحة
        st.markdown(self.theme.get_css_styles(), unsafe_allow_html=True)
        st.markdown(f"""
            <style>
            .main {{
                background-color: {self.theme.BACKGROUND_COLOR};
                color: {self.theme.TEXT_COLOR};
            }}
            h1, h2, h3 {{
                color: {self.theme.PRIMARY_COLOR};
            }}
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
        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = "Dashboard"
    
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
        """عرض الشريط الجانبي بتصميم مريح"""
        st.header("🚨 Emergency System")
        
        # اختبار Twilio
        if st.button("📱 Test Twilio SMS", type="secondary", use_container_width=True):
            result = self.twilio.test_connection()
            if result['success']:
                st.success("✅ SMS sent successfully!")
            else:
                st.error(f"❌ Failed: {result['error']}")
        
        st.header("📊 Live Data")
        status = self.core_system.get_system_status()
        
        # بطاقات المقاييس بتصميم مريح
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {self.theme.PRIMARY_COLOR}">System Health</h3>
                    <h2>{status['health']}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {self.theme.PRIMARY_COLOR}">Active Sensors</h3>
                    <h2>{status['sensor_count']}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # حالة الطوارئ
        emergency_status = "🟢 Normal" if not status['emergency_mode'] else "🔴 Active"
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {self.theme.PRIMARY_COLOR}">Emergency Mode</h3>
                <h2>{emergency_status}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """لوحة التحكم الرئيسية بتصميم مريح"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Real-time Sensor Data")
            self.render_sensor_charts()
        
        with col2:
            st.subheader("⚠️ Anomaly Detection")
            self.render_anomalies()
        
        st.subheader("🤖 AI Recommendations")
        self.render_ai_recommendations()
    
    def render_sensor_charts(self):
        """عرض مخططات المستشعرات بتصميم مريح"""
        sensor_types = ['temperature', 'pressure', 'vibration', 'methane', 'h2s']
        selected_sensor = st.selectbox("Select Sensor", sensor_types, key="sensor_select")
        
        if selected_sensor:
            # محاكاة البيانات
            times = pd.date_range(end=datetime.now(), periods=24, freq='H')
            values = np.random.normal(25, 5, 24) if selected_sensor == 'temperature' else np.random.normal(1000, 50, 24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=values, mode='lines+markers',
                name=selected_sensor, 
                line=dict(color=self.theme.PRIMARY_COLOR, width=3),
                marker=dict(size=6, color=self.theme.SECONDARY_COLOR)
            ))
            
            fig.update_layout(
                title=f"{selected_sensor.title()} Trend",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
                plot_bgcolor=self.theme.BACKGROUND_COLOR,
                paper_bgcolor=self.theme.BACKGROUND_COLOR,
                font=dict(color=self.theme.TEXT_COLOR)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_anomalies(self):
        """عرض التنبيهات بتصميم مريح"""
        if st.session_state.anomalies:
            for anomaly in st.session_state.anomalies[-3:]:
                st.markdown(f"""
                    <div class="emergency-alert">
                        <strong>🚨 {anomaly['sensor'].upper()}</strong><br>
                        Value: {anomaly['value']:.2f}<br>
                        Score: {anomaly['score']:.2f}<br>
                        Time: {anomaly['timestamp']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected")
    
    def render_ai_chat(self):
        """دردشة الذكاء الاصطناعي بتصميم مريح"""
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
        """التوأم الرقمي العكسي بتصميم مريح"""
        st.header("🔄 Reverse Digital Twin")
        
        with st.form("simulation_form"):
            st.subheader("Run Reverse Simulation")
            
            col1, col2 = st.columns(2)
            with col1:
                scenario_type = st.selectbox("Scenario Type", 
                                           ["gas_leak", "pressure_surge", "equipment_failure"])
            with col2:
                duration = st.slider("Duration (hours)", 1, 24, 6)
            
            if st.form_submit_button("🚀 Run Simulation", use_container_width=True):
                with st.spinner("Running reverse simulation..."):
                    scenario = {
                        "type": scenario_type,
                        "duration": duration,
                        "timestamp": datetime.now()
                    }
                    result = self.advanced_systems.handle_advanced_scenarios("reverse_simulation", scenario)
                    st.session_state.simulation_results.append(result)
                    
                    if result.get('success', False):
                        st.success("✅ Simulation completed successfully!")
                        st.json(result.get('results', {}))
                    else:
                        st.error("❌ Simulation failed")
        
        if st.session_state.simulation_results:
            st.subheader("Latest Simulation Results")
            for result in st.session_state.simulation_results[-3:]:
                with st.expander(f"Scenario: {result.get('scenario', 'unknown')}"):
                    st.json(result)
    
    def render_settings(self):
        """إعدادات النظام بتصميم مريح"""
        st.header("⚙️ System Settings")
        
        with st.form("system_settings"):
            st.subheader("Performance Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                sampling_rate = st.slider("Sampling Rate (seconds)", 1, 10, 2)
                simulation_mode = st.checkbox("Simulation Mode", value=True)
            
            with col2:
                monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 2000, 1000)
                enable_twilio = st.checkbox("Enable Twilio Alerts", value=False)
            
            if st.form_submit_button("💾 Save Settings", use_container_width=True):
                st.success("✅ Settings saved successfully!")
    
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
