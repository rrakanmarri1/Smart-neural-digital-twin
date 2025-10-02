import streamlit as st
import time
import logging
from datetime import datetime
import sys
import os

# إضافة المسار للوحدات
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_systems import create_smart_neural_twin
from advanced_systems import AdvancedDashboard
from config_and_logging import SmartConfig

class SmartNeuralApp:
    """التطبيق الرئيسي لـ Smart Neural Digital Twin - SS Rating"""
    
    def __init__(self):
        self.setup_application()
        self.smart_twin = None
        self.dashboard = None
        self.initialize_application()
        
    def setup_application(self):
        """إعداد التطبيق المتقدم"""
        st.set_page_config(
            page_title="Smart Neural Digital Twin - SS Rating",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/rrakanmarri1/Smart-neural-digital-twin',
                'Report a bug': "https://github.com/rrakanmarri1/Smart-neural-digital-twin/issues",
                'About': "# Smart Neural Digital Twin\nAdvanced Oil Field Disaster Prevention System"
            }
        )
        
        # إعداد التسجيل
        self.setup_logging()
        
    def setup_logging(self):
        """إعداد نظام التسجيل للتطبيق"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SmartNeural.App')
        
    def initialize_application(self):
        """تهيئة التطبيق والأنظمة"""
        try:
            # شاشة التحميل
            with st.spinner("🚀 Initializing Smart Neural Digital Twin..."):
                
                # محاكاة وقت التهيئة
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Initializing systems... {i+1}%")
                    time.sleep(0.02)  # محاكاة وقت التحميل
                
                # إنشاء النظام الرئيسي
                self.smart_twin = create_smart_neural_twin()
                
                # إنشاء الداشبورد
                self.dashboard = AdvancedDashboard(self.smart_twin)
                
                status_text.text("✅ Initialization complete!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
            st.success("🎉 Smart Neural Digital Twin Started Successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Application initialization failed: {e}")
            self.logger.error(f"Application initialization error: {e}")
            
            # وضع الطوارئ
            st.warning("🔧 Running in emergency mode with limited functionality")
            self._initialize_emergency_mode()
    
    def _initialize_emergency_mode(self):
        """تهيئة وضع الطوارئ"""
        try:
            # تحميل إعدادات أساسية
            config = SmartConfig()
            self.dashboard = AdvancedDashboard(None)
            st.info("🔄 Emergency mode activated - Basic UI available")
            
        except Exception as e:
            st.error(f"❌ Emergency mode failed: {e}")
    
    def run_application(self):
        """تشغيل التطبيق الرئيسي"""
        try:
            # التحقق من تهيئة النظام
            if self.dashboard is None:
                st.error("❌ Dashboard not initialized. Please refresh the page.")
                return
            
            # تشغيل الداشبورد
            self.dashboard.run_dashboard()
            
            # إدارة الجلسة
            self._manage_session()
            
        except Exception as e:
            st.error(f"❌ Application runtime error: {e}")
            self.logger.error(f"Application runtime error: {e}")
            
            # محاولة استعادة
            if st.button("🔄 Restart Application"):
                st.rerun()
    
    def _manage_session(self):
        """إدارة جلسة التطبيق"""
        # تحديث وقت الجلسة الأخير
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.now()
        else:
            st.session_state.last_activity = datetime.now()
        
        # تنظيف الجلسة التلقائي (كل ساعة)
        current_time = datetime.now()
        last_activity = st.session_state.last_activity
        
        if (current_time - last_activity).total_seconds() > 3600:  # ساعة واحدة
            st.info("🔄 Session refreshed due to inactivity")
            st.session_state.clear()
            st.rerun()
    
    def display_system_info(self):
        """عرض معلومات النظام (للتصحيح)"""
        if st.sidebar.checkbox("🔧 Debug Info", False):
            with st.sidebar.expander("System Information"):
                if self.smart_twin:
                    status = self.smart_twin.get_enhanced_system_status()
                    st.json(status)
                else:
                    st.warning("System not initialized")

def main():
    """الدالة الرئيسية للتطبيق"""
    try:
        # إنشاء وتشغيل التطبيق
        app = SmartNeuralApp()
        app.run_application()
        
    except Exception as e:
        # معالجة الأخطاء العامة
        st.error(f"🚨 Critical application error: {e}")
        logging.critical(f"Application crash: {e}")
        
        # رسالة طوارئ
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #7f1d1d; border-radius: 10px;">
            <h1 style="color: white;">🚨 System Emergency</h1>
            <p style="color: #fca5a5;">The Smart Neural Digital Twin has encountered a critical error.</p>
            <p style="color: #fca5a5;">Please contact system administrator immediately.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # زر إعادة التشغيل
        if st.button("🔄 Restart Application", type="primary"):
            st.rerun()

if __name__ == "__main__":
    main()
