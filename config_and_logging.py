import logging
import json
import os
import streamlit as st
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import RPi.GPIO as GPIO
import sys

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SmartTheme:
    """ثيم Smart Neural Digital Twin المتقدم"""
    primary: str = "#1a365d"
    secondary: str = "#2d3748"
    accent: str = "#3182ce"
    background: str = "#0f172a"
    card: str = "#1e293b"
    text: str = "#f7fafc"
    success: str = "#38a169"
    warning: str = "#d69e2e"
    danger: str = "#e53e3e"
    
    def apply_theme(self):
        """تطبيق الثيم المتقدم على Streamlit"""
        st.markdown(f"""
        <style>
        .main {{
            background: linear-gradient(135deg, {self.background}, #1e293b);
            color: {self.text};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        .stSidebar {{
            background: linear-gradient(180deg, {self.secondary}, {self.primary});
        }}
        
        .stAlert {{
            background: rgba(30, 41, 59, 0.95) !important;
            backdrop-filter: blur(10px);
            border-left: 4px solid {self.accent};
            border-radius: 10px;
            padding: 1rem;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {self.card}, {self.secondary});
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border: 1px solid {self.accent}40;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        }}
        
        .emergency-glowing {{
            animation: emergency-pulse 1.5s infinite;
            border: 2px solid #ef4444;
            background: linear-gradient(45deg, rgba(239, 68, 68, 0.1), transparent);
        }}
        
        @keyframes emergency-pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
            70% {{ box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        
        .section-header {{
            background: linear-gradient(90deg, {self.primary}, {self.accent});
            padding: 0.75rem 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: white;
            font-weight: bold;
        }}
        
        .smart-recommendation {{
            background: rgba(56, 161, 105, 0.1);
            border-left: 4px solid {self.success};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }}
        </style>
        """, unsafe_allow_html=True)

class SmartConfig:
    """نظام إعدادات Smart Neural Digital Twin المتقدم"""
    
    def __init__(self, config_path: str = "config/smart_neural_config.json"):
        self.config_path = config_path
        self.logger = self.setup_advanced_logging()
        self.config = self.load_advanced_config()
        self.theme = SmartTheme()
        
        # إعداد GPIO للـ Raspberry Pi
        self.setup_raspberry_pi()
        
        self.logger.info("🎯 Smart Neural Digital Twin Config Initialized")
    
    def setup_advanced_logging(self) -> logging.Logger:
        """إعداد نظام تسجيل متقدم"""
        # إنشاء مجلدات النظام
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # تنسيق متقدم للـLogs
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-25s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # معالجات متعددة
        file_handler = logging.FileHandler('logs/smart_neural_system.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # إنشاء اللوجر الرئيسي
        logger = logging.getLogger('SmartNeuralTwin')
        logger.setLevel(logging.INFO)
        
        # إزالة المعالجات القديمة
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # إضافة المعالجات الجديدة
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # إعداد لوجرات فرعية
        self.setup_subsystem_loggers()
        
        return logger
    
    def setup_subsystem_loggers(self):
        """إعداد لوجرات الأنظمة الفرعية"""
        subsystems = ['AI', 'Sensors', 'Hardware', 'Prediction', 'Anomaly', 'Memory']
        for subsystem in subsystems:
            logger = logging.getLogger(f'SmartNeural.{subsystem}')
            logger.setLevel(logging.INFO)
    
    def setup_raspberry_pi(self):
        """إعداد Raspberry Pi مع GPIO"""
        try:
            if self.config['system']['raspberry_pi']['active']:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # إعداد أطراف الـ Relay
                relay_pins = self.config['system']['raspberry_pi']['relay_pins']
                for pin_name, pin_number in relay_pins.items():
                    GPIO.setup(pin_number, GPIO.OUT)
                    GPIO.output(pin_number, GPIO.LOW)  # إيقاف افتراضي
                
                self.logger.info("✅ Raspberry Pi GPIO initialized successfully")
            else:
                self.logger.info("🔄 Raspberry Pi simulation mode activated")
                
        except Exception as e:
            self.logger.error(f"❌ Raspberry Pi setup failed: {e}")
            self.logger.info("🔧 Continuing in simulation mode")
    
    def load_advanced_config(self) -> Dict[str, Any]:
        """تحميل إعدادات متقدمة"""
        default_config = {
            "system": {
                "name": "Smart Neural Digital Twin - SS Rating",
                "version": "2.0.0",
                "description": "Advanced Oil Field Disaster Prevention System with AI",
                "update_interval": 2,
                "max_memory_usage": "2GB",
                "data_retention_days": 30
            },
            "raspberry_pi": {
                "active": True,
                "gpio_mode": "BCM",
                "relay_pins": {
                    "emergency_cooling": 17,
                    "pressure_release": 18,
                    "gas_venting": 22,
                    "main_shutdown": 23,
                    "backup_pump": 24,
                    "safety_valve": 25
                },
                "sensor_pins": {
                    "temperature": 2,
                    "pressure": 3,
                    "methane": 4,
                    "vibration": 5
                }
            },
            "foresight_engine": {
                "scenarios_per_second": {
                    "min": 100,
                    "max": 1000,
                    "default": 500
                },
                "prediction_horizons": {
                    "short_term": 6,
                    "medium_term": 24,
                    "long_term": 168
                },
                "confidence_thresholds": {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.5
                },
                "monte_carlo_simulations": 1000
            },
            "ai_models": {
                "isolation_forest": {
                    "contamination": 0.1,
                    "n_estimators": 200,
                    "max_features": 1.0,
                    "bootstrap": True
                },
                "lstm_models": {
                    "short_term": {"units": 128, "layers": 3, "dropout": 0.2},
                    "medium_term": {"units": 256, "layers": 4, "dropout": 0.3},
                    "long_term": {"units": 512, "layers": 5, "dropout": 0.4}
                },
                "anomaly_detection": {
                    "sensitivity": 0.85,
                    "window_size": 100,
                    "retrain_interval": 3600
                },
                "autoencoder": {
                    "encoding_dim": 32,
                    "epochs": 100,
                    "batch_size": 32
                }
            },
            "sensors": {
                "pressure": {"min": 0, "max": 200, "critical": 150, "unit": "bar", "pin": 2},
                "temperature": {"min": -50, "max": 300, "critical": 200, "unit": "°C", "pin": 3},
                "methane": {"min": 0, "max": 5000, "critical": 1000, "unit": "ppm", "pin": 4},
                "hydrogen_sulfide": {"min": 0, "max": 500, "critical": 50, "unit": "ppm", "pin": 5},
                "vibration": {"min": 0, "max": 20, "critical": 8, "unit": "m/s²", "pin": 6},
                "flow": {"min": 0, "max": 500, "critical": 400, "unit": "L/min", "pin": 7}
            },
            "emergency_protocols": {
                "auto_response": True,
                "response_timeout": 30,
                "notification_levels": ["warning", "critical", "emergency"],
                "escalation_procedures": {
                    "level_1": ["alert_team", "increase_monitoring"],
                    "level_2": ["activate_safety", "reduce_pressure"],
                    "level_3": ["emergency_shutdown", "notify_authorities"]
                }
            },
            "data_processing": {
                "preprocessing": {
                    "normalization": True,
                    "outlier_detection": True,
                    "feature_scaling": "standard",
                    "window_size": 50
                },
                "storage": {
                    "real_time_buffer": 1000,
                    "historical_days": 30,
                    "compression": True
                }
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                return self.deep_merge(default_config, user_config)
            else:
                self.create_default_config(default_config)
                return default_config
                
        except Exception as e:
            self.logger.error(f"❌ Config loading failed: {e}")
            return default_config
    
    def deep_merge(self, default: Dict, user: Dict) -> Dict:
        """دمج متعمق للإعدادات"""
        result = default.copy()
        
        for key, value in user.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_default_config(self, config: Dict):
        """إنشاء ملف الإعدادات الافتراضي"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        self.logger.info(f"✅ Default config created at {self.config_path}")
    
    def save_config(self, config: Dict):
        """حفظ الإعدادات"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.logger.info("✅ Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"❌ Config save failed: {e}")

class RelayController:
    """متحكم متقدم في الريلايات للـ Raspberry Pi"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.Hardware')
        self.relay_states = {}
        self.relay_history = []
        self.setup_relays()
    
    def setup_relays(self):
        """إعداد الريلايات مع تحكم متقدم"""
        try:
            relay_pins = self.config['raspberry_pi']['relay_pins']
            
            for relay_name, pin in relay_pins.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
                self.relay_states[relay_name] = {
                    'state': False,
                    'pin': pin,
                    'last_activated': None,
                    'activation_count': 0
                }
            
            self.logger.info(f"✅ {len(relay_pins)} relays initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Relay setup failed: {e}")
            # وضع المحاكاة
            relay_pins = self.config['raspberry_pi']['relay_pins']
            for relay_name in relay_pins.keys():
                self.relay_states[relay_name] = {
                    'state': False,
                    'pin': None,
                    'last_activated': None,
                    'activation_count': 0,
                    'simulated': True
                }
    
    def control_relay(self, relay_name: str, state: bool, reason: str = "Manual control"):
        """التحكم المتقدم في الريلاي"""
        try:
            if relay_name not in self.relay_states:
                self.logger.error(f"❌ Relay {relay_name} not found")
                return False
            
            relay_info = self.relay_states[relay_name]
            
            if not relay_info.get('simulated', False):
                # تحكم حقيقي
                GPIO.output(relay_info['pin'], GPIO.HIGH if state else GPIO.LOW)
            
            # تحديث الحالة
            old_state = relay_info['state']
            relay_info['state'] = state
            
            if state and not old_state:  # عند التنشيط
                relay_info['last_activated'] = datetime.now()
                relay_info['activation_count'] += 1
            
            # تسجيل في السجل
            log_entry = {
                'timestamp': datetime.now(),
                'relay': relay_name,
                'state': state,
                'reason': reason,
                'simulated': relay_info.get('simulated', False)
            }
            self.relay_history.append(log_entry)
            
            action = "activated" if state else "deactivated"
            self.logger.info(f"✅ Relay {relay_name} {action} - {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Relay control failed for {relay_name}: {e}")
            return False
    
    def emergency_shutdown(self):
        """إيقاف طوارئ لجميع الريلايات"""
        self.logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        for relay_name in self.relay_states.keys():
            self.control_relay(relay_name, False, "Emergency shutdown")
        
        self.logger.info("✅ All relays deactivated for emergency shutdown")
    
    def get_relay_status(self) -> Dict[str, Any]:
        """الحصول على حالة الريلايات"""
        status = {}
        for relay_name, info in self.relay_states.items():
            status[relay_name] = {
                'state': info['state'],
                'last_activated': info['last_activated'],
                'activation_count': info['activation_count'],
                'simulated': info.get('simulated', False)
            }
        return status

# اختبار النظام
if __name__ == "__main__":
    config_system = SmartConfig()
    print("✅ Smart Neural Digital Twin Config System Ready")
