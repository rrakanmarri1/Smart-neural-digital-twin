import logging
import json
import os
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_file: str = "digital_twin.log"):
    """
    إعداد نظام التسجيل (Logging) المتقدم
    """
    try:
        # إنشاء مجلد اللوجات إذا لم يكن موجوداً
        os.makedirs('logs', exist_ok=True)
        
        log_path = os.path.join('logs', log_file)
        
        # تنسيق اللوجات
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # معالج الملفات
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # معهد وحدة التحكم
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # إعداد اللوجر الرئيسي
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # إزالة المعالجات القديمة إذا وجدت
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # إضافة المعالجات الجديدة
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # إعداد لوجرات محددة
        setup_specific_loggers()
        
        logging.info("✅ Logging system initialized successfully")
        return logger
        
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}")
        raise

def setup_specific_loggers():
    """
    إعداد لوجرات محددة لأجزاء النظام
    """
    # لوجر الذكاء الاصطناعي
    ai_logger = logging.getLogger('ai_systems')
    ai_logger.setLevel(logging.INFO)
    
    # لوجر الهاردوير
    hw_logger = logging.getLogger('hardware')
    hw_logger.setLevel(logging.INFO)
    
    # لوجر المستشعرات
    sensor_logger = logging.getLogger('sensors')
    sensor_logger.setLevel(logging.INFO)

class ConfigLoader:
    """
    محمل الإعدادات المتقدم مع معالجة الأخطاء
    """
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self.default_config = self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """الإعدادات الافتراضية للنظام"""
        return {
            "system": {
                "name": "Smart Neural Digital Twin",
                "version": "1.0.0",
                "description": "Oil Field Disaster Prevention System",
                "update_interval": 5  # ثواني
            },
            "sensors": {
                "pressure": {"min": 0, "max": 100, "unit": "bar"},
                "temperature": {"min": 0, "max": 150, "unit": "celsius"},
                "methane": {"min": 0, "max": 1000, "unit": "ppm"},
                "hydrogen_sulfide": {"min": 0, "max": 100, "unit": "ppm"},
                "vibration": {"min": 0, "max": 10, "unit": "m/s²"},
                "flow": {"min": 0, "max": 100, "unit": "L/min"}
            },
            "ai_models": {
                "anomaly_detection": {
                    "sensitivity": 0.8,
                    "window_size": 100,
                    "retrain_interval": 3600  # ثانية
                },
                "prediction": {
                    "horizon": 24,  # ساعة
                    "confidence_threshold": 0.7
                }
            },
            "hardware": {
                "raspberry_pi": {
                    "gpio_pins": {
                        "relay_1": 17,
                        "relay_2": 27,
                        "servo_1": 22,
                        "led_green": 5,
                        "led_red": 6
                    }
                },
                "i2c_addresses": {
                    "ads1115": 0x48,
                    "bmp280": 0x76
                }
            },
            "emergency": {
                "risk_thresholds": {
                    "warning": 0.4,
                    "critical": 0.6,
                    "emergency": 0.8
                },
                "notification": {
                    "sms_enabled": True,
                    "email_enabled": True,
                    "phone_numbers": [],
                    "emails": []
                }
            },
            "api_keys": {
                "twilio": {
                    "account_sid": "",
                    "auth_token": "",
                    "from_number": ""
                },
                "openai": {
                    "api_key": ""
                }
            }
        }
    
    def load(self) -> Dict[str, Any]:
        """
        تحميل الإعدادات من الملف مع الدمج مع الإعدادات الافتراضية
        """
        try:
            # تحميل الإعدادات من الملف إذا وجد
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # الدمج مع الإعدادات الافتراضية
                merged_config = self._deep_merge(self.default_config, file_config)
                self.logger.info(f"✅ Configuration loaded from {self.config_path}")
                return merged_config
            else:
                # إنشاء ملف الإعدادات إذا لم يكن موجوداً
                self._create_default_config_file()
                self.logger.warning(f"⚠️ Using default config, created {self.config_path}")
                return self.default_config
                
        except Exception as e:
            self.logger.error(f"❌ Error loading configuration: {e}")
            self.logger.info("🔄 Using default configuration")
            return self.default_config
    
    def _deep_merge(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """دمج القواميس بشكل متعمق"""
        result = default.copy()
        
        for key, value in custom.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_default_config_file(self):
        """إنشاء ملف الإعدادات الافتراضي"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"✅ Default config file created at {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating config file: {e}")
    
    def save(self, config: Dict[str, Any]):
        """حفظ الإعدادات إلى الملف"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"✅ Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving configuration: {e}")
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """التحقق من صحة الإعدادات"""
        try:
            required_sections = ['system', 'sensors', 'ai_models', 'emergency']
            
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"❌ Missing required section: {section}")
                    return False
            
            # التحقق من قيم المستشعرات
            sensors = config.get('sensors', {})
            for sensor, params in sensors.items():
                if 'min' not in params or 'max' not in params:
                    self.logger.error(f"❌ Invalid sensor config for {sensor}")
                    return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

# دالة مساعدة لتحميل الإعدادات
def load_configuration(config_path: str = "config/settings.json") -> Dict[str, Any]:
    """
    دالة مساعدة لتحميل الإعدادات مع معالجة الأخطاء
    """
    try:
        loader = ConfigLoader(config_path)
        config = loader.load()
        
        if loader.validate(config):
            return config
        else:
            raise ValueError("Configuration validation failed")
            
    except Exception as e:
        logging.error(f"❌ Failed to load configuration: {e}")
        raise

if __name__ == "__main__":
    # اختبار نظام الإعدادات والتسجيل
    setup_logging()
    
    config = load_configuration()
    print("✅ Config and logging tested successfully!")
    print(f"System: {config['system']['name']} v{config['system']['version']}")
