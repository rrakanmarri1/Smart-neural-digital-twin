import json
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import streamlit as st
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ThemeConfig:
    """إعدادات التصميم المرئي المريح للعين"""
    PRIMARY_COLOR = "#1f4e79"  # أزرق غامق مريح
    SECONDARY_COLOR = "#2e75b6"  # أزرق متوسط
    ACCENT_COLOR = "#8faadc"  # أزرق فاتح
    BACKGROUND_COLOR = "#f0f4f8"  # خلفية فاتحة مريحة
    TEXT_COLOR = "#2d2d2d"  # نص غامق مريح
    SUCCESS_COLOR = "#107c10"  # أخضر مريح
    WARNING_COLOR = "#d83b01"  # برتقالي تحذيري
    ERROR_COLOR = "#e81123"  # أحمر طوارئ
    
    @classmethod
    def get_css_styles(cls):
        """إرجاع أنماط CSS مخصصة"""
        return f"""
            <style>
            .main {{
                background-color: {cls.BACKGROUND_COLOR};
                color: {cls.TEXT_COLOR};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .stButton>button {{
                background-color: {cls.PRIMARY_COLOR};
                color: white;
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: 500;
            }}
            .stButton>button:hover {{
                background-color: {cls.SECONDARY_COLOR};
            }}
            .metric-card {{
                background: linear-gradient(135deg, {cls.BACKGROUND_COLOR}, #ffffff);
                border-radius: 12px;
                padding: 20px;
                border-left: 4px solid {cls.PRIMARY_COLOR};
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .emergency-alert {{
                background: linear-gradient(135deg, #ffe6e6, #ffcccc);
                border: 2px solid {cls.ERROR_COLOR};
                border-radius: 10px;
                padding: 15px;
                color: #d13438;
                font-weight: bold;
            }}
            </style>
        """

class AdvancedConfig:
    def __init__(self, config_file: str = "config/system_config.json"):
        self.config_file = config_file
        self.theme = ThemeConfig()
        self.config = self.load_config()
        self.setup_logging()
        self.last_modified = datetime.now()
        
    def load_config(self) -> Dict[str, Any]:
        """تحميل الإعدادات من Streamlit secrets أو ملف JSON"""
        default_config = {
            "system": {
                "name": "Smart Oil Field Neural Digital Twin",
                "version": "8.0.0",
                "description": "Advanced AI-Powered Disaster Prevention with Dynamic Model Selection",
                "log_level": "INFO",
                "data_retention_days": 365,
                "timezone": "Asia/Riyadh",
                "simulation_mode": True,
                "emergency_contacts": [],
                "max_anomaly_score": 0.85,
                "raspberry_pi_mode": True
            },
            "theme": {
                "primary_color": self.theme.PRIMARY_COLOR,
                "secondary_color": self.theme.SECONDARY_COLOR,
                "accent_color": self.theme.ACCENT_COLOR,
                "background_color": self.theme.BACKGROUND_COLOR,
                "text_color": self.theme.TEXT_COLOR
            },
            "twilio": {
                "enabled": False,
                "account_sid": "",
                "auth_token": "",
                "phone_number": "",
                "emergency_contacts": []
            },
            "openai": {
                "enabled": False,
                "api_key": "",
                "model": "gpt-4",
                "max_tokens": 500
            },
            "hardware": {
                "sampling_interval": 2.0,
                "sensor_update_frequency": 5.0,
                "sensors": {
                    "temperature": {"pin": 22, "min": -10, "max": 85, "unit": "°C"},
                    "pressure": {"pin": 23, "min": 800, "max": 1200, "unit": "hPa"},
                    "vibration": {"pin": 24, "min": 0, "max": 10, "unit": "g"},
                    "methane": {"pin": 25, "min": 0, "max": 2000, "unit": "ppm"},
                    "h2s": {"pin": 26, "min": 0, "max": 200, "unit": "ppm"},
                    "flow": {"pin": 27, "min": 0, "max": 500, "unit": "L/min"}
                },
                "actuators": {
                    "valve_1": {"pin": 17, "type": "servo", "min_angle": 0, "max_angle": 180},
                    "valve_2": {"pin": 18, "type": "servo", "min_angle": 0, "max_angle": 180},
                    "pump_1": {"pin": 19, "type": "pwm", "min_speed": 0, "max_speed": 100},
                    "emergency_shutdown": {"pin": 20, "type": "digital", "active_low": True}
                }
            },
            "ai": {
                "dynamic_model_selection": {
                    "enabled": True,
                    "max_models": 5,
                    "selection_interval": 300,
                    "performance_threshold": 0.8
                },
                "anomaly_detection": {
                    "enabled": True,
                    "confidence_threshold": 0.8,
                    "update_interval": 3600,
                    "max_training_samples": 5000
                },
                "prediction": {
                    "enabled": True,
                    "horizon_hours": 24,
                    "monte_carlo_simulations": 1000,
                    "prediction_interval": 300
                },
                "optimization": {
                    "enabled": True,
                    "target_efficiency": 0.85,
                    "max_iterations": 500
                }
            },
            "memory": {
                "max_patterns": 1000,
                "pattern_min_confidence": 0.7,
                "auto_cleanup": True,
                "cleanup_interval": 3600,
                "playbook_enabled": True,
                "lifelong_learning": True
            },
            "emergency": {
                "auto_shutdown": True,
                "response_timeout": 30,
                "escalation_levels": {
                    "level_1": {"threshold": 70, "actions": ["alert", "log"]},
                    "level_2": {"threshold": 85, "actions": ["shutdown", "notify"]},
                    "level_3": {"threshold": 95, "actions": ["evacuate", "emergency_services"]}
                }
            },
            "performance": {
                "optimized_for_pi": True,
                "max_threads": 4,
                "memory_limit_mb": 512,
                "cpu_usage_limit": 0.7,
                "cache_enabled": True,
                "cache_size": 100,
                "data_compression": True,
                "batch_processing": True
            },
            "security": {
                "encryption_enabled": True,
                "authentication_required": True,
                "rate_limiting": True,
                "max_login_attempts": 3,
                "session_timeout": 3600,
                "data_backup": {
                    "enabled": True,
                    "interval": 86400,
                    "retention_days": 7
                },
                "audit_logging": True
            }
        }
        
        try:
            # محاولة تحميل من Streamlit secrets أولاً
            if hasattr(st, 'secrets'):
                secrets = st.secrets
                if 'twilio' in secrets:
                    default_config['twilio'].update({
                        'account_sid': secrets['twilio']['account_sid'],
                        'auth_token': secrets['twilio']['auth_token'],
                        'phone_number': secrets['twilio']['phone_number'],
                        'enabled': True
                    })
                if 'openai' in secrets:
                    default_config['openai'].update({
                        'api_key': secrets['openai']['api_key'],
                        'enabled': True
                    })
                if 'emergency_contacts' in secrets:
                    default_config['system']['emergency_contacts'] = secrets['emergency_contacts']
            
            # ثم تحميل من ملف الإعدادات إذا وجد
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    return self.deep_merge(default_config, file_config)
            
            return default_config
            
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config
    
    def deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """دمج عميق للإعدادات"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self.deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def setup_logging(self):
        """تهيئة نظام التسجيل"""
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """الحصول على إعداد"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """تعيين إعداد"""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            return True
        except Exception as e:
            logging.error(f"Error setting config: {e}")
            return False

# إنشاء instance عالمي
config = AdvancedConfig()
