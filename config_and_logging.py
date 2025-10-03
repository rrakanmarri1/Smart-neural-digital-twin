import logging
import json
import os
import sys
import inspect
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import time

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """هيكل مدخل السجل"""
    timestamp: datetime
    level: str
    logger: str
    message: str
    file: str
    line: int
    function: str
    thread: str

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
    info: str = "#4299e1"
    
    def to_dict(self) -> Dict[str, str]:
        """تحويل الثيم إلى قاموس"""
        return asdict(self)

class AdvancedLogHandler(logging.Handler):
    """معالج سجل متقدم مع إدارة الذاكرة"""
    
    def __init__(self, max_entries: int = 10000):
        super().__init__()
        self.max_entries = max_entries
        self.log_entries: List[LogEntry] = []
        self.lock = threading.RLock()
        
    def emit(self, record):
        """تسجيل المدخلات مع معلومات متقدمة"""
        try:
            # استخراج معلومات الإطار
            frame = inspect.currentframe()
            for _ in range(6):  # البحث في 6 إطارات للخلف
                if frame is None:
                    break
                if frame.f_code.co_name == self.emit.__name__:
                    frame = frame.f_back
                    continue
                break
            
            file_name = record.filename if frame is None else frame.f_code.co_filename
            line_no = record.lineno if frame is None else frame.f_lineno
            function_name = record.funcName if frame is None else frame.f_code.co_name
            
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger=record.name,
                message=self.format(record),
                file=file_name,
                line=line_no,
                function=function_name,
                thread=record.threadName
            )
            
            with self.lock:
                self.log_entries.append(log_entry)
                # إدارة الذاكرة - الاحتفاظ بأحدث المدخلات فقط
                if len(self.log_entries) > self.max_entries:
                    self.log_entries = self.log_entries[-self.max_entries:]
                    
        except Exception as e:
            print(f"Logging error: {e}", file=sys.stderr)

class SmartConfig:
    """نظام إعدادات Smart Neural Digital Twin المتقدم - SS Rating"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: str = "config/smart_neural_config.json"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SmartConfig, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, config_path: str = "config/smart_neural_config.json"):
        if not hasattr(self, '_initialized'):
            self.config_path = Path(config_path)
            self._config = {}
            self._last_modified = 0
            self._theme = SmartTheme()
            self._log_handler = None
            self._system_logger = None
            
            # إنشاء الهيكل الأساسي للمجلدات
            self._create_directory_structure()
            
            # إعداد التسجيل أولاً
            self._setup_advanced_logging()
            
            # ثم تحميل الإعدادات
            self._load_advanced_config()
            
            # إعداد النظام
            self._setup_system_components()
            
            self._initialized = True
            self.get_logger().info("🎯 Smart Neural Digital Twin Config Initialized - SS Rating")
    
    def _create_directory_structure(self):
        """إنشاء هيكل المجلدات المتقدم"""
        directories = [
            'logs/system',
            'logs/performance',
            'logs/security',
            'models/ai',
            'models/anomaly',
            'models/prediction',
            'data/real_time',
            'data/historical',
            'data/backup',
            'config/backups',
            'reports/daily',
            'reports/incidents',
            'cache/temp'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_advanced_logging(self):
        """إعداد نظام تسجيل متقدم مع تعدد المسارات"""
        try:
            # تنسيق متقدم للـLogs
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)-30s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(threadName)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            simple_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # معالجات متعددة للمسارات المختلفة
            handlers = [
                # السجل الرئيسي
                logging.FileHandler('logs/system/main.log', encoding='utf-8', delay=True),
                # سجل الأداء
                logging.FileHandler('logs/performance/performance.log', encoding='utf-8', delay=True),
                # سجل الأخطاء
                logging.FileHandler('logs/system/errors.log', encoding='utf-8', delay=True),
                # وحدة التحكم
                logging.StreamHandler(sys.stdout)
            ]
            
            # تكوين المعالجات
            handlers[0].setFormatter(detailed_formatter)
            handlers[0].setLevel(logging.INFO)
            
            handlers[1].setFormatter(simple_formatter)
            handlers[1].setLevel(logging.INFO)
            handlers[1].addFilter(lambda record: record.levelno >= logging.INFO)
            
            handlers[2].setFormatter(detailed_formatter)
            handlers[2].setLevel(logging.ERROR)
            
            handlers[3].setFormatter(simple_formatter)
            handlers[3].setLevel(logging.INFO)
            
            # إنشاء اللوجر الرئيسي
            self._system_logger = logging.getLogger('SmartNeuralTwin')
            self._system_logger.setLevel(logging.INFO)
            
            # إزالة المعالجات القديمة
            for handler in self._system_logger.handlers[:]:
                self._system_logger.removeHandler(handler)
            
            # إضافة المعالجات الجديدة
            for handler in handlers:
                self._system_logger.addHandler(handler)
            
            # معالج مخصص للذاكرة
            self._log_handler = AdvancedLogHandler()
            self._log_handler.setFormatter(detailed_formatter)
            self._log_handler.setLevel(logging.INFO)
            self._system_logger.addHandler(self._log_handler)
            
            # إعداد لوجرات الأنظمة الفرعية
            self._setup_subsystem_loggers()
            
            # منع انتشار الـLogs إلى الـroot logger
            self._system_logger.propagate = False
            
            self._system_logger.info("✅ Advanced logging system initialized")
            
        except Exception as e:
            print(f"❌ Critical logging setup failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _setup_subsystem_loggers(self):
        """إعداد لوجرات الأنظمة الفرعية المتخصصة"""
        subsystems = {
            'AI': logging.INFO,
            'Sensors': logging.INFO,
            'Hardware': logging.INFO,
            'Prediction': logging.DEBUG,
            'Anomaly': logging.INFO,
            'Memory': logging.INFO,
            'UI': logging.INFO,
            'Security': logging.WARNING,
            'Performance': logging.INFO
        }
        
        for subsystem, level in subsystems.items():
            logger = logging.getLogger(f'SmartNeural.{subsystem}')
            logger.setLevel(level)
            # منع الانتشار لتجنب التكرار
            logger.propagate = False
    
    def _load_advanced_config(self):
        """تحميل الإعدادات المتقدمة مع التحقق من الصحة"""
        default_config = self._get_default_config()
        
        try:
            if self.config_path.exists():
                file_mtime = self.config_path.stat().st_mtime
                if file_mtime > self._last_modified:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        user_config = json.load(f)
                    
                    self._config = self._deep_merge(default_config, user_config)
                    self._last_modified = file_mtime
                    
                    if self._validate_config():
                        self.get_logger().info(f"✅ Configuration loaded from {self.config_path}")
                    else:
                        self.get_logger().warning("⚠️ Configuration loaded with validation warnings")
                else:
                    self._config = default_config
            else:
                self._create_default_config(default_config)
                self._config = default_config
                self.get_logger().info("✅ Default configuration created and loaded")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ Config JSON error: {e}")
            self._config = default_config
        except Exception as e:
            self.get_logger().error(f"❌ Config loading failed: {e}")
            self._config = default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """الحصول على الإعدادات الافتراضية المتقدمة"""
        return {
            "system": {
                "name": "Smart Neural Digital Twin - SS Rating",
                "version": "3.0.0",
                "description": "Advanced Oil Field Disaster Prevention System with AI - SS Rating",
                "update_interval": 2.0,
                "max_memory_usage": "2GB",
                "data_retention_days": 30,
                "performance_mode": "SS_RATING",
                "timezone": "Asia/Riyadh",
                "language": "ar"
            },
            "raspberry_pi": {
                "active": False,
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
                },
                "simulation_mode": True
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
                "monte_carlo_simulations": 1000,
                "adaptive_learning": True
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
                    "retrain_interval": 3600,
                    "ensemble_weights": [0.4, 0.3, 0.3]
                },
                "autoencoder": {
                    "encoding_dim": 32,
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            },
            "sensors": {
                "pressure": {"min": 0, "max": 200, "critical": 150, "unit": "bar", "weight": 0.25},
                "temperature": {"min": -50, "max": 300, "critical": 200, "unit": "°C", "weight": 0.20},
                "methane": {"min": 0, "max": 5000, "critical": 1000, "unit": "ppm", "weight": 0.25},
                "hydrogen_sulfide": {"min": 0, "max": 500, "critical": 50, "unit": "ppm", "weight": 0.15},
                "vibration": {"min": 0, "max": 20, "critical": 8, "unit": "m/s²", "weight": 0.10},
                "flow": {"min": 0, "max": 500, "critical": 400, "unit": "L/min", "weight": 0.05}
            },
            "emergency_protocols": {
                "auto_response": True,
                "response_timeout": 30,
                "notification_levels": ["warning", "critical", "emergency"],
                "escalation_procedures": {
                    "level_1": ["alert_team", "increase_monitoring"],
                    "level_2": ["activate_safety", "reduce_pressure"],
                    "level_3": ["emergency_shutdown", "notify_authorities"]
                },
                "confirmation_required": True
            },
            "data_processing": {
                "preprocessing": {
                    "normalization": True,
                    "outlier_detection": True,
                    "feature_scaling": "standard",
                    "window_size": 50,
                    "sequence_length": 50
                },
                "storage": {
                    "real_time_buffer": 1000,
                    "historical_days": 30,
                    "compression": True,
                    "backup_interval": 3600
                }
            },
            "performance": {
                "target_processing_time": 0.1,
                "max_memory_usage": "2GB",
                "cpu_utilization_limit": 0.8,
                "gpu_acceleration": True,
                "cache_size": "500MB"
            },
            "security": {
                "encryption_enabled": True,
                "access_logging": True,
                "max_login_attempts": 3,
                "session_timeout": 3600
            }
        }
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """دمج متعمق وآمن للإعدادات"""
        result = default.copy()
        
        for key, value in user.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                # التحقق من أنواع البيانات الأساسية
                if key in result and type(result[key]) != type(value) and value is not None:
                    self.get_logger().warning(f"⚠️ Type mismatch for key '{key}': {type(result[key])} vs {type(value)}")
                    try:
                        # محاولة التحويل للنوع الأصلي
                        if isinstance(result[key], bool):
                            value = str(value).lower() in ('true', '1', 'yes')
                        elif isinstance(result[key], int):
                            value = int(value)
                        elif isinstance(result[key], float):
                            value = float(value)
                        elif isinstance(result[key], str):
                            value = str(value)
                    except (ValueError, TypeError):
                        self.get_logger().error(f"❌ Cannot convert value for key '{key}', using default")
                        continue
                
                result[key] = value
        
        return result
    
    def _create_default_config(self, config: Dict):
        """إنشاء ملف الإعدادات الافتراضي مع النسخ الاحتياطي"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # حفظ النسخة الافتراضية
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False, default=str)
            
            # إنشاء نسخة احتياطية مؤرخة
            backup_path = self.config_path.parent / 'backups' / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False, default=str)
            
            self.get_logger().info(f"✅ Default config created at {self.config_path}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to create default config: {e}")
            raise
    
    def _validate_config(self) -> bool:
        """التحقق من صحة الإعدادات بشكل شامل"""
        try:
            required_sections = ['system', 'raspberry_pi', 'foresight_engine', 'ai_models', 'sensors']
            for section in required_sections:
                if section not in self._config:
                    self.get_logger().error(f"❌ Missing config section: {section}")
                    return False
            
            # التحقق من المستشعرات
            required_sensors = ['pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow']
            sensor_config = self._config.get('sensors', {})
            for sensor in required_sensors:
                if sensor not in sensor_config:
                    self.get_logger().error(f"❌ Missing sensor config: {sensor}")
                    return False
                
                sensor_params = sensor_config[sensor]
                required_params = ['min', 'max', 'critical', 'unit', 'weight']
                for param in required_params:
                    if param not in sensor_params:
                        self.get_logger().error(f"❌ Missing parameter '{param}' for sensor '{sensor}'")
                        return False
            
            # التحقق من القيم العددية
            system_config = self._config.get('system', {})
            if system_config.get('update_interval', 0) <= 0:
                self.get_logger().error("❌ Invalid update interval")
                return False
            
            # التحقق من إعدادات الأداء
            performance_config = self._config.get('performance', {})
            if performance_config.get('cpu_utilization_limit', 0) <= 0 or performance_config.get('cpu_utilization_limit', 0) > 1:
                self.get_logger().error("❌ Invalid CPU utilization limit")
                return False
            
            self.get_logger().info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Config validation failed: {e}")
            return False
    
    def _setup_system_components(self):
        """إعداد مكونات النظام المتقدمة"""
        try:
            # إعداد Raspberry Pi إذا كان مفعلاً
            if self._config.get('raspberry_pi', {}).get('active', False):
                self._setup_raspberry_pi()
            else:
                self.get_logger().info("🔧 Raspberry Pi simulation mode activated")
                
            # إعداد مراقبة الأداء
            self._setup_performance_monitoring()
            
        except Exception as e:
            self.get_logger().error(f"❌ System components setup failed: {e}")
    
    def _setup_raspberry_pi(self):
        """إعداد Raspberry Pi مع GPIO - إصدار آمن ومتطور"""
        try:
            # التحقق من وجود RPi.GPIO بشكل آمن
            try:
                import RPi.GPIO as GPIO
                self._gpio_available = True
                
                # استخدام الإعدادات من التكوين
                gpio_mode = self._config['raspberry_pi']['gpio_mode']
                if gpio_mode.upper() == 'BCM':
                    GPIO.setmode(GPIO.BCM)
                else:
                    GPIO.setmode(GPIO.BOARD)
                
                GPIO.setwarnings(False)
                
                # إعداد أطراف الـ Relay
                relay_pins = self._config['raspberry_pi']['relay_pins']
                for pin_name, pin_number in relay_pins.items():
                    GPIO.setup(pin_number, GPIO.OUT)
                    GPIO.output(pin_number, GPIO.LOW)  # إيقاف افتراضي آمن
                
                self.get_logger().info("✅ Raspberry Pi GPIO initialized successfully")
                
            except (ImportError, RuntimeError) as e:
                self._gpio_available = False
                self._config['raspberry_pi']['simulation_mode'] = True
                self.get_logger().info(f"🔧 Raspberry Pi GPIO not available: {e}")
                
        except Exception as e:
            self.get_logger().error(f"❌ Raspberry Pi setup failed: {e}")
            self._gpio_available = False
            self._config['raspberry_pi']['simulation_mode'] = True
    
    def _setup_performance_monitoring(self):
        """إعداد مراقبة الأداء"""
        self._performance_stats = {
            'start_time': datetime.now(),
            'config_reloads': 0,
            'errors_count': 0,
            'warnings_count': 0
        }
    
    def get_logger(self, name: str = 'SmartNeuralTwin') -> logging.Logger:
        """الحصول على logger للنظام"""
        if name == 'SmartNeuralTwin':
            return self._system_logger
        return logging.getLogger(name)
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """الحصول على الإعدادات بشكل آمن"""
        try:
            if key is None:
                return self._config.copy()
            
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.get_logger().warning(f"⚠️ Config access error for key '{key}': {e}")
            return default
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """تحديث الإعدادات ديناميكياً"""
        try:
            with threading.Lock():
                # دمج التحديثات
                self._config = self._deep_merge(self._config, updates)
                
                # التحقق من الصحة
                if not self._validate_config():
                    self.get_logger().error("❌ Config update validation failed")
                    return False
                
                # الحفظ إذا مطلوب
                if save:
                    self.save_config()
                
                self.get_logger().info("✅ Configuration updated successfully")
                return True
                
        except Exception as e:
            self.get_logger().error(f"❌ Config update failed: {e}")
            return False
    
    def save_config(self) -> bool:
        """حفظ الإعدادات الحالية"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False, default=str)
            
            self._last_modified = self.config_path.stat().st_mtime
            self.get_logger().info("✅ Configuration saved successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Config save failed: {e}")
            return False
    
    def reload_config(self) -> bool:
        """إعادة تحميل الإعدادات من الملف"""
        try:
            self._load_advanced_config()
            self._performance_stats['config_reloads'] += 1
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Config reload failed: {e}")
            return False
    
    def get_theme(self) -> SmartTheme:
        """الحصول على الثيم الحالي"""
        return self._theme
    
    def get_log_entries(self, level: str = None, limit: int = 100) -> List[LogEntry]:
        """الحصول على مدخلات السجل"""
        if self._log_handler is None:
            return []
        
        with self._log_handler.lock:
            entries = self._log_handler.log_entries.copy()
        
        if level:
            entries = [entry for entry in entries if entry.level == level.upper()]
        
        return entries[-limit:] if limit else entries
    
    def get_system_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النظام"""
        return {
            'config_path': str(self.config_path),
            'last_modified': datetime.fromtimestamp(self._last_modified) if self._last_modified else None,
            'performance_stats': self._performance_stats.copy(),
            'gpio_available': getattr(self, '_gpio_available', False),
            'system_uptime': datetime.now() - self._performance_stats['start_time'],
            'log_entries_count': len(self._log_handler.log_entries) if self._log_handler else 0
        }

# اختبار النظام المحسن
if __name__ == "__main__":
    try:
        # اختبار نظام الإعدادات
        config_system = SmartConfig()
        
        # اختبار الوظائف الأساسية
        logger = config_system.get_logger()
        logger.info("🧪 Testing configuration system...")
        
        # اختبار الحصول على الإعدادات
        system_name = config_system.get_config('system.name')
        logger.info(f"System Name: {system_name}")
        
        # اختبار تحديث الإعدادات
        test_update = {'system': {'update_interval': 3.0}}
        if config_system.update_config(test_update, save=False):
            logger.info("✅ Config update test passed")
        
        # عرض معلومات النظام
        system_info = config_system.get_system_info()
        logger.info(f"System Uptime: {system_info['system_uptime']}")
        logger.info(f"Config Reloads: {system_info['performance_stats']['config_reloads']}")
        
        logger.info("🎯 Smart Neural Digital Twin Config System Ready - SS Rating")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        sys.exit(1)
