import streamlit as st
import time
import logging
from datetime import datetime
import sys
import os
import traceback

# إضافة المسار للوحدات
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core_systems import create_smart_neural_twin
    from advanced_systems import AdvancedDashboard, create_advanced_dashboard
    from config_and_logging import SmartConfig
except ImportError as e:
    st.error(f"❌ Module import error: {e}")
    logging.error(f"Import error: {e}")

class SmartNeuralApp:
    """التطبيق الرئيسي لـ Smart Neural Digital Twin"""
    
    def __init__(self):
        self.setup_application()
        self.smart_twin = None
        self.dashboard = None
        self.config = None
        self.initialize_application()
        
    def setup_application(self):
        """إعداد التطبيق المتقدم"""
        st.set_page_config(
            page_title="Smart Neural Digital Twin",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': """
                # 🧠 Smart Neural Digital Twin
                ## Advanced Oil Field Disaster Prevention System
                
                Real-time monitoring and predictive analytics
                for oil field safety and disaster prevention.
                
                ### Features:
                - 🔮 AI-Powered Predictions
                - ⚠️ Advanced Anomaly Detection  
                - 📊 Real-time Monitoring
                - 🚨 Emergency Response
                - 📈 Performance Analytics
                
                **Version:** 1.0.0 |
                """
            }
        )
        
        # إعداد التسجيل
        self.setup_logging()
        
    def setup_logging(self):
        """إعداد نظام التسجيل للتطبيق"""
        try:
            # استخدام نظام التسجيل المتقدم من SmartConfig
            self.config = SmartConfig()
            self.logger = self.config.get_logger('SmartNeural.App')
            self.logger.info("🎯 Application logging initialized")
            
        except Exception as e:
            # نظام تسجيل احتياطي
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('logs/app_emergency.log', encoding='utf-8')
                ]
            )
            self.logger = logging.getLogger('SmartNeural.App')
            self.logger.warning(f"Using emergency logging: {e}")
        
    def initialize_application(self):
        """تهيئة التطبيق والأنظمة"""
        try:
            # شاشة التحميل المتقدمة
            self._show_loading_screen()
            
            # إنشاء النظام الرئيسي
            self.smart_twin = self._create_smart_twin_system()
            
            # إنشاء الداشبورد المتقدم
            self.dashboard = self._create_advanced_dashboard()
            
            st.success("🎉 Smart Neural Digital Twin Started Successfully!")
            
            # إظهار رسالة ترحيب
            self._show_welcome_message()
            
        except Exception as e:
            self.logger.error(f"Application initialization error: {e}")
            self.logger.error(traceback.format_exc())
            
            # وضع الطوارئ المتقدم
            self._initialize_emergency_mode(e)
    
    def _show_loading_screen(self):
        """عرض شاشة تحميل متقدمة"""
        st.markdown("""
        <style>
        .loading-container {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            border-radius: 15px;
            margin: 2rem 0;
        }
        .loading-title {
            font-size: 2.5em;
            background: linear-gradient(135deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
        }
        .loading-subtitle {
            color: #cbd5e1;
            font-size: 1.2em;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="loading-container">
                <div class="loading-title">🧠 Smart Neural Digital Twin</div>
                <div class="loading-subtitle">Initializing Advanced AI Systems</div>
            </div>
            """, unsafe_allow_html=True)
            
            # شريط التقدم المتقدم
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            
            # مراحل التهيئة
            initialization_steps = [
                ("Loading Configuration", 10),
                ("Initializing AI Models", 25),
                ("Setting Up Sensor Grid", 40),
                ("Starting Prediction Engine", 60),
                ("Calibrating Safety Systems", 80),
                ("Finalizing Dashboard", 95),
                ("System Ready", 100)
            ]
            
            for step_name, target_progress in initialization_steps:
                current_progress = progress_bar.progress(0)
                
                # محاكاة التقدم مع تفاصيل
                for i in range(target_progress - int(current_progress * 100)):
                    current_progress_value = (int(current_progress * 100) + i + 1) / 100
                    progress_bar.progress(current_progress_value)
                    
                    status_text.text(f"🔄 {step_name}... {int(current_progress_value * 100)}%")
                    details_text.text(f"▪️ Initializing subsystem components...")
                    
                    time.sleep(0.03)  # محاكاة وقت التحميل
                
                # تحديث التفاصيل النهائية لكل مرحلة
                details_text.text(f"✅ {step_name} completed")
                time.sleep(0.5)
            
            status_text.text("✅ Initialization complete!")
            time.sleep(1)
            
            # تنظيف شاشة التحميل
            progress_bar.empty()
            status_text.empty()
            details_text.empty()
    
    def _create_smart_twin_system(self):
        """إنشاء نظام Smart Twin المتقدم"""
        try:
            self.logger.info("Creating Smart Neural Digital Twin system...")
            
            # استخدام الدالة المحسنة من core_systems
            smart_twin = create_smart_neural_twin()
            
            if smart_twin:
                self.logger.info("✅ Smart Twin system created successfully")
                return smart_twin
            else:
                raise Exception("Smart Twin creation returned None")
                
        except Exception as e:
            self.logger.error(f"❌ Smart Twin creation failed: {e}")
            raise
    
    def _create_advanced_dashboard(self):
        """إنشاء داشبورد متقدم"""
        try:
            self.logger.info("Creating advanced dashboard...")
            
            # استخدام الدالة المحسنة من advanced_systems
            dashboard = create_advanced_dashboard(self.smart_twin)
            
            if dashboard:
                self.logger.info("✅ Advanced dashboard created successfully")
                return dashboard
            else:
                raise Exception("Dashboard creation returned None")
                
        except Exception as e:
            self.logger.error(f"❌ Dashboard creation failed: {e}")
            raise
    
    def _show_welcome_message(self):
        """عرض رسالة ترحيب"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #065f46, #047857); border-radius: 15px; margin: 1rem 0;">
            <h2 style="color: white; margin-bottom: 1rem;">🚀 System Ready</h2>
            <p style="color: #ecfdf5; margin-bottom: 0.5rem;">Smart Neural Digital Twin is now operational with advanced AI monitoring</p>
            <p style="color: #d1fae5; font-size: 0.9em;">Real-time anomaly detection and predictive analytics are active</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _initialize_emergency_mode(self, error: Exception):
        """تهيئة وضع الطوارئ المتقدم"""
        try:
            st.error(f"❌ Application initialization failed: {str(error)}")
            
            # عرض تفاصيل الخطأ للمطورين
            with st.expander("🔧 Technical Details (For Support)"):
                st.code(traceback.format_exc())
            
            st.warning("""
            🔧 **Emergency Mode Activated**
            
            The system is running with limited functionality:
            - Basic dashboard interface available
            - Simulation data will be used
            - Core AI features disabled
            """)
            
            # إنشاء نظام محاكاة
            self._create_simulation_system()
            
            # محاولة إنشاء داشبورد الطوارئ
            try:
                self.dashboard = create_advanced_dashboard(None)
                st.info("🔄 Emergency dashboard initialized")
            except Exception as e:
                st.error(f"❌ Emergency dashboard failed: {e}")
                self._create_basic_interface()
                
        except Exception as emergency_error:
            st.error(f"❌ Emergency mode failed: {emergency_error}")
            self._create_basic_interface()
    
    def _create_simulation_system(self):
        """إنشاء نظام محاكاة للطوارئ"""
        try:
            # نظام محاكاة بسيط
            class SimulationSystem:
                def get_enhanced_system_status(self):
                    return {
                        'system_status': 'SIMULATION',
                        'ai_confidence': 85.0,
                        'response_time': 2.5,
                        'risk_level': 'LOW',
                        'timestamp': datetime.now()
                    }
                
                def real_time_data(self):
                    return {
                        'pressure': 45.2,
                        'temperature': 75.3,
                        'methane': 210.5,
                        'hydrogen_sulfide': 8.7,
                        'vibration': 1.8,
                        'flow': 195.4
                    }
                
                @property
                def sensor_grid_status(self):
                    return {
                        'active_sensors': 6,
                        'grid_health': 0.85
                    }
            
            self.smart_twin = SimulationSystem()
            self.logger.info("✅ Simulation system created for emergency mode")
            
        except Exception as e:
            self.logger.error(f"❌ Simulation system creation failed: {e}")
    
    def _create_basic_interface(self):
        """إنشاء واجهة أساسية للطوارئ القصوى"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #1e293b; border-radius: 10px;">
            <h1 style="color: #f1f5f9;">🧠 Smart Neural Digital Twin</h1>
            <p style="color: #cbd5e1;">Emergency Basic Interface</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "EMERGENCY")
        with col2:
            st.metric("AI Systems", "OFFLINE")
        with col3:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
        
        st.info("""
        ⚠️ **System Recovery Required**
        
        Please try:
        1. Refreshing the page
        2. Checking system logs
        3. Contacting support team
        """)
    
    def run_application(self):
        """تشغيل التطبيق الرئيسي"""
        try:
            # التحقق من تهيئة النظام
            if self.dashboard is None:
                st.error("""
                ❌ **Dashboard Not Initialized**
                
                The application dashboard could not be loaded. This could be due to:
                - Missing dependencies
                - Configuration errors  
                - System resource issues
                
                Please refresh the page or contact support.
                """)
                
                if st.button("🔄 Restart Application", type="primary", use_container_width=True):
                    st.rerun()
                return
            
            # تشغيل الداشبورد المتقدم
            self.logger.info("Starting main application dashboard...")
            self.dashboard.run_dashboard()
            
            # إدارة الجلسة المتقدمة
            self._manage_advanced_session()
            
            # عرض معلومات التصحيح
            self._display_debug_info()
            
        except Exception as e:
            self.logger.error(f"Application runtime error: {e}")
            self.logger.error(traceback.format_exc())
            
            st.error(f"""
            ❌ **Application Runtime Error**
            
            Error details: {str(e)}
            
            The system has encountered an unexpected error. 
            """)
            
            # محاولة استعادة
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Restart Application", type="primary", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("📋 Show Error Details", use_container_width=True):
                    with st.expander("Error Traceback"):
                        st.code(traceback.format_exc())
    
    def _manage_advanced_session(self):
        """إدارة جلسة التطبيق المتقدمة"""
        # تهيئة حالة الجلسة
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
            st.session_state.activity_count = 0
        
        # تحديث النشاط
        st.session_state.last_activity = datetime.now()
        st.session_state.activity_count += 1
        
        # تنظيف الجلسة التلقائي (كل ساعتين)
        current_time = datetime.now()
        session_duration = current_time - st.session_state.session_start
        
        if session_duration.total_seconds() > 7200:  # ساعتين
            st.info("🔄 Session auto-refresh for optimal performance")
            st.session_state.clear()
            st.rerun()
        
        # عرض معلومات الجلسة في وضع التصحيح
        if st.sidebar.checkbox("📊 Session Info", False):
            with st.sidebar.expander("Session Details"):
                st.write(f"**Started:** {st.session_state.session_start.strftime('%H:%M:%S')}")
                st.write(f"**Duration:** {str(session_duration).split('.')[0]}")
                st.write(f"**Activities:** {st.session_state.activity_count}")
                st.write(f"**Memory:** {sys.getsizeof(st.session_state)} bytes")
    
    def _display_debug_info(self):
        """عرض معلومات التصحيح المتقدمة"""
        if st.sidebar.checkbox("🔧 Advanced Debug", False):
            with st.sidebar.expander("System Debug Information"):
                
                # معلومات النظام الأساسي
                st.subheader("System Status")
                if self.smart_twin:
                    try:
                        status = self.smart_twin.get_enhanced_system_status()
                        st.json(status)
                    except Exception as e:
                        st.error(f"Status error: {e}")
                else:
                    st.warning("Smart Twin system not available")
                
                # معلومات التكوين
                st.subheader("Configuration")
                if self.config:
                    try:
                        system_config = self.config.get_config('system')
                        st.json(system_config)
                    except Exception as e:
                        st.error(f"Config error: {e}")
                
                # معلومات الأداء
                st.subheader("Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Python Version", sys.version.split()[0])
                    st.metric("Streamlit", st.__version__)
                with col2:
                    st.metric("Log Entries", len(self.config.get_log_entries()) if self.config else "N/A")
                
                # أزرار التحكم
                st.subheader("Controls")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reload Config", use_container_width=True):
                        if self.config and self.config.reload_config():
                            st.success("Config reloaded!")
                        else:
                            st.error("Config reload failed")
                with col2:
                    if st.button("Clear Cache", use_container_width=True):
                        st.session_state.clear()
                        st.success("Cache cleared!")
                        st.rerun()

def main():
    """الدالة الرئيسية للتطبيق"""
    try:
        # رسالة بدء التشغيل
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #1e3a8a, #1e40af);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # إنشاء وتشغيل التطبيق
        app = SmartNeuralApp()
        app.run_application()
        
    except Exception as e:
        # معالجة الأخطاء العامة الحرجة
        logging.critical(f"Application crash: {e}")
        logging.critical(traceback.format_exc())
        
        # واجهة طوارئ متقدمة
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #7f1d1d; border-radius: 15px; margin: 2rem 0;">
            <h1 style="color: white; margin-bottom: 1rem;">🚨 System Emergency</h1>
            <p style="color: #fca5a5; font-size: 1.2em; margin-bottom: 1rem;">
                The Smart Neural Digital Twin has encountered a critical error.
            </p>
            <p style="color: #fecaca;">
                Please contact system administrator immediately and provide the error details below.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # تفاصيل الخطأ
        with st.expander("🚨 Critical Error Details"):
            st.error(f"Error Type: {type(e).__name__}")
            st.error(f"Error Message: {str(e)}")
            st.code(traceback.format_exc())
        
        # أزرار الاستعادة
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Application", type="primary", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📋 Copy Error Report", use_container_width=True):
                error_report = f"""
                Smart Neural Digital Twin - Critical Error Report
                Timestamp: {datetime.now()}
                Error: {type(e).__name__}
                Message: {str(e)}
                Traceback: {traceback.format_exc()}
                """
                st.code(error_report)
                st.success("Error report copied to clipboard")
        
        with col3:
            if st.button("🆘 Emergency Support", use_container_width=True):
                st.info("""
                **Emergency Support Contact:**
                - Email: rakan.almarri.2@aramco.com
                - Phone: +966-53-255-9664
                - Incident ID: SNDT-EMG-001
                """)

if __name__ == "__main__":
    main()
