import logging
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import threading
import time

class TwilioIntegration:
    """
    تكامل Twilio لإرسال رسائل SMS للطوارئ
    """
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        self.logger = logging.getLogger(__name__)
    
    def send_sms(self, to_number: str, message: str) -> bool:
        """إرسال رسالة SMS"""
        try:
            # استخدام secrets إذا كان في بيئة Streamlit
            if hasattr(st, 'secrets') and st.secrets.get('twilio', {}):
                twilio_secrets = st.secrets['twilio']
                self.account_sid = twilio_secrets.get('account_sid', self.account_sid)
                self.auth_token = twilio_secrets.get('auth_token', self.auth_token)
                self.from_number = twilio_secrets.get('from_number', self.from_number)
            
            payload = {
                'Body': message,
                'From': self.from_number,
                'To': to_number
            }
            
            response = requests.post(
                self.base_url,
                auth=(self.account_sid, self.auth_token),
                data=payload
            )
            
            if response.status_code == 201:
                self.logger.info(f"✅ SMS sent to {to_number}")
                return True
            else:
                self.logger.error(f"❌ Failed to send SMS: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error sending SMS: {e}")
            return False
    
    def send_emergency_alert(self, to_numbers: List[str], emergency_data: Dict[str, Any]) -> bool:
        """إرسال تنبيه طوارئ"""
        try:
            message = self._format_emergency_message(emergency_data)
            results = []
            
            for number in to_numbers:
                success = self.send_sms(number, message)
                results.append(success)
            
            return all(results)
            
        except Exception as e:
            self.logger.error(f"❌ Error sending emergency alert: {e}")
            return False
    
    def _format_emergency_message(self, data: Dict[str, Any]) -> str:
        """تنسيق رسالة الطوارئ"""
        risk_level = data.get('risk_level', 0)
        anomalies = data.get('anomalies', {})
        
        message = f"🚨 EMERGENCY ALERT - Oil Field Monitoring\n"
        message += f"Risk Level: {risk_level:.1%}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if anomalies:
            message += "Critical Anomalies Detected:\n"
            for anomaly in anomalies.get('critical', []):
                message += f"- {anomaly}\n"
        
        message += "\n⚠️ Emergency protocols activated. Please check system immediately."
        return message

class EmailNotifier:
    """
    نظام إشعارات البريد الإلكتروني
    """
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, to_emails: List[str], subject: str, body: str) -> bool:
        """إرسال بريد إلكتروني"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"✅ Email sent to {to_emails}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error sending email: {e}")
            return False

class DashboardRenderer:
    """
    محرك عرض الداشبورد المتقدم
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_scheme = {
            'primary': '#1a365d',
            'secondary': '#2d3748', 
            'accent': '#3182ce',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e'
        }
    
    def render_main_dashboard(self, real_time_data: Dict[str, Any], system_health: Dict[str, Any]):
        """عرض الداشبورد الرئيسي"""
        try:
            # استخدام Streamlit لعرض الواجهة
            st.set_page_config(
                page_title="Smart Neural Digital Twin",
                page_icon="🔥",
                layout="wide"
            )
            
            # تطبيق تخصيصات CSS
            self._apply_custom_styles()
            
            # الهيدر
            st.title("🧠 Smart Neural Digital Twin - Oil Field Monitoring")
            st.markdown("---")
            
            # صف المؤشرات
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self._render_status_card(system_health['status'])
            
            with col2:
                self._render_sensor_card("Pressure", real_time_data.get('pressure', 0))
            
            with col3:
                self._render_sensor_card("Temperature", real_time_data.get('temperature', 0))
            
            with col4:
                self._render_sensor_card("Methane", real_time_data.get('methane', 0))
            
            # صف الرسوم البيانية
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_sensor_charts(real_time_data)
            
            with col2:
                self._render_anomaly_charts(real_time_data.get('anomalies', {}))
            
            # صف التنبؤات
            st.subheader("📊 24-Hour Predictions")
            self._render_predictions(real_time_data.get('predictions', {}))
            
        except Exception as e:
            self.logger.error(f"❌ Error rendering dashboard: {e}")
            st.error(f"Error rendering dashboard: {e}")
    
    def _apply_custom_styles(self):
        """تطبيق التخصيصات CSS"""
        st.markdown(f"""
        <style>
        .main {{
            background-color: {self.color_scheme['primary']};
        }}
        .stAlert {{
            background-color: {self.color_scheme['secondary']};
        }}
        .css-1d391kg {{
            background-color: {self.color_scheme['secondary']};
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def _render_status_card(self, status: str):
        """عرض بطاقة الحالة"""
        status_colors = {
            'normal': self.color_scheme['success'],
            'warning': self.color_scheme['warning'], 
            'critical': self.color_scheme['danger'],
            'emergency': '#ff0000'
        }
        
        color = status_colors.get(status, self.color_scheme['secondary'])
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">System Status</h3>
            <h2 style="color: white; margin: 0;">{status.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sensor_card(self, sensor_name: str, value: float):
        """عرض بطاقة المستشعر"""
        st.metric(
            label=sensor_name,
            value=f"{value:.2f}",
            delta=None
        )
    
    def _render_sensor_charts(self, data: Dict[str, Any]):
        """عرض رسوم المستشعرات البيانية"""
        try:
            import plotly.graph_objects as go
            
            sensors = ['pressure', 'temperature', 'methane', 'vibration', 'flow']
            values = [data.get(sensor, 0) for sensor in sensors]
            
            fig = go.Figure(data=[go.Bar(x=sensors, y=values)])
            fig.update_layout(title="Sensor Readings")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Chart error: {e}")
    
    def _render_anomaly_charts(self, anomalies: Dict[str, Any]):
        """عرض رسوم الشذوذ البيانية"""
        try:
            import plotly.express as px
            
            if anomalies:
                anomaly_types = list(anomalies.keys())
                counts = [len(anomalies[atype]) for atype in anomaly_types]
                
                fig = px.pie(values=counts, names=anomaly_types, title="Anomaly Distribution")
                st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Anomaly chart error: {e}")
    
    def _render_predictions(self, predictions: Dict[str, Any]):
        """عرض التنبؤات"""
        try:
            if predictions:
                for sensor, prediction in predictions.items():
                    st.write(f"**{sensor}**: {prediction.get('trend', 'stable')}")
            else:
                st.info("No prediction data available")
                
        except Exception as e:
            st.error(f"Prediction display error: {e}")

class ReverseDigitalTwin:
    """
    التوأم الرقمي العكسي - محاكاة كاملة للعمليات الفعلية
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.simulation_data = {}
    
    def simulate_physical_processes(self, input_actions: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة العمليات الفعلية بناءً على الإجراءات"""
        try:
            # محاكاة تأثير الإجراءات على النظام الفعلي
            simulation_results = {
                'pressure_change': self._simulate_pressure_effect(input_actions),
                'temperature_change': self._simulate_temperature_effect(input_actions),
                'flow_change': self._simulate_flow_effect(input_actions),
                'vibration_change': self._simulate_vibration_effect(input_actions),
                'gas_levels_change': self._simulate_gas_effect(input_actions),
                'timestamp': datetime.now()
            }
            
            self.simulation_data = simulation_results
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating physical processes: {e}")
            return {}
    
    def _simulate_pressure_effect(self, actions: Dict[str, Any]) -> float:
        """محاكاة تأثير الإجراءات على الضغط"""
        pressure_change = 0.0
        
        for action in actions.get('valve_actions', []):
            if action['type'] == 'pressure_release':
                pressure_change -= action.get('intensity', 0) * 2.5
        
        return max(-10.0, min(10.0, pressure_change))
    
    def _simulate_temperature_effect(self, actions: Dict[str, Any]) -> float:
        """محاكاة تأثير الإجراءات على الحرارة"""
        temp_change = 0.0
        
        for action in actions.get('cooling_actions', []):
            if action['type'] == 'emergency_cooling':
                temp_change -= action.get('intensity', 0) * 5.0
        
        return max(-15.0, min(15.0, temp_change))
    
    def _simulate_flow_effect(self, actions: Dict[str, Any]) -> float:
        """محاكاة تأثير الإجراءات على التدفق"""
        flow_change = 0.0
        
        for action in actions.get('flow_actions', []):
            if action['type'] == 'flow_adjustment':
                flow_change += action.get('adjustment', 0)
        
        return max(-20.0, min(20.0, flow_change))
    
    def _simulate_vibration_effect(self, actions: Dict[str, Any]) -> float:
        """محاكاة تأثير الإجراءات على الاهتزاز"""
        vibration_change = 0.0
        
        for action in actions.get('stabilization_actions', []):
            if action['type'] == 'vibration_dampening':
                vibration_change -= action.get('effectiveness', 0) * 0.5
        
        return max(-2.0, min(2.0, vibration_change))
    
    def _simulate_gas_effect(self, actions: Dict[str, Any]) -> Dict[str, float]:
        """محاكاة تأثير الإجراءات على مستويات الغاز"""
        gas_changes = {
            'methane': 0.0,
            'hydrogen_sulfide': 0.0,
            'carbon_dioxide': 0.0
        }
        
        for action in actions.get('ventilation_actions', []):
            if action['type'] == 'gas_venting':
                for gas in gas_changes.keys():
                    gas_changes[gas] -= action.get('venting_rate', 0) * 0.3
        
        return gas_changes

# دالة إنشاء التكامل المتقدم
def create_advanced_systems(config: Dict[str, Any]) -> Dict[str, Any]:
    """إنشاء جميع الأنظمة المتقدمة"""
    try:
        twilio_config = config.get('twilio', {})
        email_config = config.get('email', {})
        
        systems = {
            'twilio': TwilioIntegration(
                twilio_config.get('account_sid', ''),
                twilio_config.get('auth_token', ''),
                twilio_config.get('from_number', '')
            ),
            'email': EmailNotifier(
                email_config.get('smtp_server', ''),
                email_config.get('smtp_port', 587),
                email_config.get('username', ''),
                email_config.get('password', '')
            ),
            'dashboard': DashboardRenderer(),
            'reverse_twin': ReverseDigitalTwin(config)
        }
        
        logging.info("✅ Advanced systems created successfully")
        return systems
        
    except Exception as e:
        logging.error(f"❌ Failed to create advanced systems: {e}")
        raise

if __name__ == "__main__":
    # اختبار الأنظمة المتقدمة
    test_config = {
        'twilio': {
            'account_sid': 'test',
            'auth_token': 'test', 
            'from_number': '+1234567890'
        },
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'test@gmail.com',
            'password': 'test'
        }
    }
    
    systems = create_advanced_systems(test_config)
    print("✅ Advanced systems tested successfully!")
