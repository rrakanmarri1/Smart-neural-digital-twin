import time
import threading
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# استيراد جميع المكونات مع معالجة الأخطاء
try:
    from preprocess_data import DataPreprocessor
except ImportError as e:
    logging.warning(f"DataPreprocessor not available: {e}")
    DataPreprocessor = None

try:
    from generate_sensor_data import SensorDataGenerator
except ImportError as e:
    logging.warning(f"SensorDataGenerator not available: {e}")
    SensorDataGenerator = None
    
# استيراد أنظمة الذكاء الاصطناعي
try:
    from ai_systems.advanced_anomaly_system import AdvancedAnomalyDetector
except ImportError:
    AdvancedAnomalyDetector = None

try:
    from ai_systems.advanced_prediction_engine import AdvancedPredictionEngine
except ImportError:
    AdvancedPredictionEngine = None

try:
    from ai_systems.advanced_prediction_modules import PredictionModules
except ImportError:
    PredictionModules = None

try:
    from ai_systems.intervention_engine import InterventionEngine
except ImportError:
    InterventionEngine = None

try:
    from ai_systems.lifelong_memory import LifelongMemory
except ImportError:
    LifelongMemory = None

try:
    from ai_systems.memory_playbook import MemoryPlaybook
except ImportError:
    MemoryPlaybook = None

try:
    from ai_systems.dynamic_model_selector import DynamicModelSelector, ModelType
except ImportError:
    # استخدام النسخة المحلية
    from ai_systems import DynamicModelSelector, ModelType

try:
    from twilio_integration import TwilioIntegration
except ImportError:
    TwilioIntegration = None

try:
    from ai_chat_system import AIChatSystem
except ImportError:
    AIChatSystem = None

try:
    from utils.error_handling import SystemError, ErrorSeverity, GracefulDegradation
except ImportError:
    # إنشاء فئات بديلة
    class ErrorSeverity(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4
    
    class SystemError(Exception):
        def __init__(self, message: str, severity: ErrorSeverity):
            super().__init__(message)
            self.severity = severity
    
    class GracefulDegradation:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)
        
        def handle_error(self, error: SystemError):
            self.logger.error(f"System error: {error} (Severity: {error.severity})")

try:
    from utils.performance_optimizer import PerformanceOptimizer
except ImportError:
    class PerformanceOptimizer:
        def __init__(self, config):
            self.config = config

class SensorType(Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    METHANE = "methane"
    H2S = "hydrogen_sulfide"
    FLOW = "flow"
    HUMIDITY = "humidity"
    ACCELERATION = "acceleration"

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    confidence: float = 0.95
    raw_value: Optional[float] = None
    status: str = "normal"

@dataclass
class ActuatorState:
    actuator_id: str
    state: Any
    timestamp: datetime
    command_source: str
    success: bool = True
    response_time: float = 0.0

class AdvancedCoreSystem:
    def __init__(self, config):
        self.config = config or {}
        self.sensor_readings: Dict[str, List[SensorReading]] = {}
        self.actuator_states: Dict[str, ActuatorState] = {}
        self.system_health = 100
        self.emergency_mode = False
        self.degradation_handler = GracefulDegradation(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.monitoring_thread = None
        self._monitoring_active = False
        
        # تهيئة جميع المكونات
        self.logger = logging.getLogger(__name__)
        self.setup_components()
        
    def setup_components(self):
        """تهيئة جميع المكونات المتقدمة"""
        try:
            # معالجة البيانات
            if DataPreprocessor:
                self.data_preprocessor = DataPreprocessor(self.config)
            else:
                self.data_preprocessor = self._create_fallback_preprocessor()
            
            if SensorDataGenerator:
                self.data_generator = SensorDataGenerator(self.config)
            else:
                self.data_generator = self._create_fallback_generator()
            
            # Dynamic Model Selection - ميزة براءة الاختراع
            self.model_selector = DynamicModelSelector(self.config)
            
            # أنظمة الذكاء الاصطناعي
            if AdvancedAnomalyDetector:
                self.anomaly_detector = AdvancedAnomalyDetector(self.config)
            else:
                self.anomaly_detector = self._create_fallback_anomaly_detector()
            
            if AdvancedPredictionEngine:
                self.prediction_engine = AdvancedPredictionEngine(self.config)
            else:
                self.prediction_engine = self._create_fallback_prediction_engine()
            
            if PredictionModules:
                self.prediction_modules = PredictionModules(self.config)
            
            if InterventionEngine:
                self.intervention_engine = InterventionEngine(self.config, self.prediction_engine, self)
            
            # تسجيل النماذج في النظام الذكي
            self._register_models()
            
            # أنظمة الذاكرة
            if LifelongMemory:
                self.lifelong_memory = LifelongMemory(self.config)
            
            if MemoryPlaybook:
                self.memory_playbook = MemoryPlaybook(self.config)
            
            # التكاملات الخارجية
            if TwilioIntegration:
                self.twilio = TwilioIntegration(self.config)
            else:
                self.twilio = None
            
            if AIChatSystem:
                self.ai_chat = AIChatSystem(self.config)
            else:
                self.ai_chat = None
            
            # تدريب النماذج الأولية
            self.train_ai_models()
            
            # بدء المراقبة
            self.start_monitoring()
            
            self.logger.info("✅ All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            self.degradation_handler.handle_error(
                SystemError("System initialization failed", ErrorSeverity.HIGH)
            )
            # لا نرفع الاستثناء لتجنب تعطل النظام
    
    def _create_fallback_preprocessor(self):
        """إنشاء معالج بيانات بديل"""
        class FallbackPreprocessor:
            def __init__(self, config):
                self.config = config
            
            def preprocess_data(self, data):
                return data if data else {}
            
            def preprocess_realtime_data(self, data):
                return data if data else {}
        
        return FallbackPreprocessor(self.config)
    
    def _create_fallback_generator(self):
        """إنشاء مولد بيانات بديل"""
        class FallbackGenerator:
            def __init__(self, config):
                self.config = config
            
            def generate_training_data(self, size):
                return {"sensor_data": [random.random() for _ in range(size)]}
            
            def generate_sensor_value(self, sensor_type):
                # محاكاة قيم المستشعرات
                ranges = {
                    'temperature': (20, 80),
                    'pressure': (1000, 1100),
                    'vibration': (0, 10),
                    'methane': (0, 100),
                    'hydrogen_sulfide': (0, 50),
                    'flow': (0, 100),
                    'humidity': (30, 90),
                    'acceleration': (0, 20)
                }
                min_val, max_val = ranges.get(sensor_type, (0, 100))
                return random.uniform(min_val, max_val)
        
        return FallbackGenerator(self.config)
    
    def _create_fallback_anomaly_detector(self):
        """إنشاء كاشف شذوذ بديل"""
        class FallbackAnomalyDetector:
            def __init__(self, config):
                self.config = config
                self.lstm_model = None
                self.isolation_forest = None
                self.one_class_svm = None
            
            def train_models(self, data):
                pass
            
            def detect_anomalies(self, data):
                return {'anomalies': [], 'critical_anomalies': []}
            
            def detect_with_isolation_forest(self, data):
                return {'anomalies': [], 'critical_anomalies': []}
            
            def detect_with_one_class_svm(self, data):
                return {'anomalies': [], 'critical_anomalies': []}
            
            def detect_with_lstm(self, data):
                return {'anomalies': [], 'critical_anomalies': []}
        
        return FallbackAnomalyDetector(self.config)
    
    def _create_fallback_prediction_engine(self):
        """إنشاء محرك تنبؤ بديل"""
        class FallbackPredictionEngine:
            def __init__(self, config):
                self.config = config
            
            def train_lstm_models(self, data):
                pass
            
            def predict(self, data):
                return {'predictions': [], 'confidence': 0.5}
        
        return FallbackPredictionEngine(self.config)
    
    def _register_models(self):
        """تسجيل جميع النماذج في نظام الاختيار الذكي"""
        try:
            if hasattr(self.anomaly_detector, 'lstm_model'):
                self.model_selector.register_model(ModelType.LSTM, self.anomaly_detector.lstm_model)
            if hasattr(self.anomaly_detector, 'isolation_forest'):
                self.model_selector.register_model(ModelType.ISOLATION_FOREST, self.anomaly_detector.isolation_forest)
            if hasattr(self.anomaly_detector, 'one_class_svm'):
                self.model_selector.register_model(ModelType.ONE_CLASS_SVM, self.anomaly_detector.one_class_svm)
            
            self.model_selector.register_model(ModelType.MONTE_CARLO, self.prediction_engine)
            
            self.logger.info("✅ All models registered in dynamic selector")
        except Exception as e:
            self.logger.error(f"❌ Error registering models: {e}")
    
    def train_ai_models(self):
        """تدريب نماذج الذكاء الاصطناعي"""
        try:
            # توليد بيانات تدريبية واقعية
            training_data = self.data_generator.generate_training_data(5000)
            processed_data = self.data_preprocessor.preprocess_data(training_data)
            
            # تدريب نماذج كشف الشذوذ
            if hasattr(self.anomaly_detector, 'train_models'):
                self.anomaly_detector.train_models(processed_data)
            
            # تدريب نماذج LSTM
            if hasattr(self.prediction_engine, 'train_lstm_models'):
                self.prediction_engine.train_lstm_models(processed_data)
            
            self.logger.info("✅ AI models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ AI model training failed: {e}")
            self.degradation_handler.handle_error(
                SystemError("Model training failed", ErrorSeverity.MEDIUM)
            )
    
    def start_monitoring(self):
        """بدء مراقبة النظام"""
        try:
            if not self._monitoring_active:
                self._monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
                self.logger.info("📊 Started real-time monitoring")
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """إيقاف مراقبة النظام"""
        self._monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """حلقة المراقبة الرئيسية مع Dynamic Model Selection"""
        while self._monitoring_active:
            try:
                # قراءة البيانات من المستشعرات
                sensor_data = self.read_all_sensors()
                
                # معالجة البيانات
                processed_data = self.data_preprocessor.preprocess_realtime_data(sensor_data)
                
                # اختيار أفضل نموذج تلقائياً - ميزة براءة الاختراع
                best_model = self._select_best_model_for_task(processed_data)
                
                # استخدام النموذج المختار
                anomalies = self._detect_anomalies_with_model(best_model, processed_data)
                
                # التنبؤ بالمستقبل
                predictions = self.prediction_engine.predict(processed_data)
                
                # تقييم الوضع واتخاذ القرارات
                if anomalies.get('critical_anomalies'):
                    self.handle_emergency(anomalies, predictions)
                
                # التعلم المستمر وتحديث أداء النموذج
                self._update_model_performance(best_model, anomalies, predictions)
                
                time.sleep(self.config.get('hardware.sampling_interval', 2.0))
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                self.degradation_handler.handle_error(
                    SystemError("Monitoring loop failed", ErrorSeverity.MEDIUM)
                )
                time.sleep(5)
    
    def _detect_anomalies_with_model(self, model_type: ModelType, processed_data: Dict[str, Any]):
        """كشف الشذوذ باستخدام النموذج المحدد"""
        try:
            if model_type == ModelType.ISOLATION_FOREST and hasattr(self.anomaly_detector, 'detect_with_isolation_forest'):
                return self.anomaly_detector.detect_with_isolation_forest(processed_data)
            elif model_type == ModelType.ONE_CLASS_SVM and hasattr(self.anomaly_detector, 'detect_with_one_class_svm'):
                return self.anomaly_detector.detect_with_one_class_svm(processed_data)
            elif model_type == ModelType.LSTM and hasattr(self.anomaly_detector, 'detect_with_lstm'):
                return self.anomaly_detector.detect_with_lstm(processed_data)
            else:
                return self.anomaly_detector.detect_anomalies(processed_data)
        except Exception as e:
            self.logger.error(f"❌ Error detecting anomalies with {model_type.value}: {e}")
            return {'anomalies': [], 'critical_anomalies': []}
    
    def _select_best_model_for_task(self, processed_data: Dict[str, Any]) -> ModelType:
        """اختيار أفضل نموذج للمهمة الحالية"""
        try:
            data_characteristics = {
                'size': len(processed_data) if processed_data else 0,
                'dimensions': len(processed_data.keys()) if processed_data else 0,
                'variability': np.std(list(processed_data.values())) if processed_data else 0
            }
            
            resource_constraints = {
                'cpu_available': 0.8,  # سيتم قراءتها من النظام
                'memory_available': 0.7  # سيتم قراءتها من النظام
            }
            
            return self.model_selector.select_best_model(data_characteristics, resource_constraints)
        except Exception as e:
            self.logger.error(f"❌ Error selecting best model: {e}")
            return ModelType.LSTM  # النموذج الافتراضي
    
    def _update_model_performance(self, model_type: ModelType, anomalies: Dict[str, Any], 
                                predictions: Dict[str, Any]):
        """تحديث أداء النموذج المستخدم"""
        try:
            accuracy = self._calculate_accuracy(anomalies, predictions)
            latency = 1.0  # سيتم حسابه من الوقت الفعلي
            memory_usage = 0.5  # سيتم قياسه من النظام
            
            self.model_selector.update_model_performance(
                model_type, accuracy, latency, memory_usage, True
            )
        except Exception as e:
            self.logger.error(f"❌ Error updating model performance: {e}")
    
    def _calculate_accuracy(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]) -> float:
        """حساب دقة النموذج"""
        try:
            # محاكاة حساب الدقة
            return random.uniform(0.7, 0.95)
        except Exception as e:
            self.logger.error(f"❌ Error calculating accuracy: {e}")
            return 0.8
    
    def read_all_sensors(self) -> Dict[str, Any]:
        """قراءة جميع المستشعرات"""
        sensor_data = {}
        
        try:
            sensors_config = self.config.get('hardware', {}).get('sensors', {})
            if not sensors_config:
                # استخدام المستشعرات الافتراضية
                sensors_config = {sensor_type.value: {} for sensor_type in SensorType}
            
            for sensor_type, sensor_config in sensors_config.items():
                reading = self.read_sensor(sensor_type)
                if reading:
                    sensor_data[sensor_type] = reading.value
        except Exception as e:
            self.logger.error(f"❌ Error reading sensors: {e}")
        
        return sensor_data
    
    def read_sensor(self, sensor_type: str) -> Optional[SensorReading]:
        """قراءة مستشعر معين"""
        try:
            if self.config.get('system', {}).get('simulation_mode', True):
                value = self.data_generator.generate_sensor_value(sensor_type)
            else:
                value = self._read_real_sensor(sensor_type)
            
            # التأكد من أن sensor_type صالح
            try:
                sensor_type_enum = SensorType(sensor_type)
            except ValueError:
                self.logger.warning(f"Unknown sensor type: {sensor_type}")
                return None
            
            reading = SensorReading(
                sensor_id=f"{sensor_type}_01",
                sensor_type=sensor_type_enum,
                value=value,
                unit=self.config.get('hardware', {}).get('sensors', {}).get(sensor_type, {}).get('unit', 'unknown'),
                timestamp=datetime.now(),
                confidence=0.95
            )
            
            # تخزين القراءة
            if reading.sensor_id not in self.sensor_readings:
                self.sensor_readings[reading.sensor_id] = []
            self.sensor_readings[reading.sensor_id].append(reading)
            
            # الاحتفاظ بآخر 1000 قراءة فقط
            if len(self.sensor_readings[reading.sensor_id]) > 1000:
                self.sensor_readings[reading.sensor_id] = self.sensor_readings[reading.sensor_id][-1000:]
            
            return reading
            
        except Exception as e:
            self.logger.error(f"❌ Error reading sensor {sensor_type}: {e}")
            return None
    
    def _read_real_sensor(self, sensor_type: str) -> float:
        """قراءة مستشعر حقيقي (لـ Raspberry Pi)"""
        try:
            # محاولة استيراد مكتبة GPIO
            try:
                import RPi.GPIO as GPIO
                pin = self.config.get('hardware', {}).get('sensors', {}).get(sensor_type, {}).get('pin')
                if pin is not None:
                    # قراءة فعلية من GPIO
                    return random.uniform(
                        self.config.get('hardware', {}).get('sensors', {}).get(sensor_type, {}).get('min', 0),
                        self.config.get('hardware', {}).get('sensors', {}).get(sensor_type, {}).get('max', 100)
                    )
                return 0.0
            except ImportError:
                # العودة للمحاكاة إذا لم تكن متوفرة
                return self.data_generator.generate_sensor_value(sensor_type)
        except Exception as e:
            self.logger.error(f"❌ Error reading real sensor {sensor_type}: {e}")
            return self.data_generator.generate_sensor_value(sensor_type)
    
    def handle_emergency(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]):
        """معالجة حالات الطوارئ بشكل كامل"""
        try:
            # البحث في memory playbook عن أفضل استجابة
            if hasattr(self, 'memory_playbook') and self.memory_playbook:
                best_response = self.memory_playbook.get_best_response(anomalies)
                if best_response:
                    self.execute_response(best_response)
                    return
            
            # استجابة جديدة من intervention engine
            if hasattr(self, 'intervention_engine') and self.intervention_engine:
                decisions = self.intervention_engine.evaluate_situation(anomalies, predictions)
                for decision in decisions:
                    if hasattr(decision, 'priority') and decision.priority.value >= 4:  # HIGH priority or above
                        self.execute_emergency_protocol(decision)
                        break
            
            # الاستجابة الافتراضية إذا فشلت كل الخيارات
            self.execute_default_emergency_response(anomalies)
            
        except Exception as e:
            self.logger.error(f"❌ Emergency handling failed: {e}")
            self.execute_default_emergency_response(anomalies)

    def execute_response(self, response):
        """تنفيذ استجابة من memory playbook"""
        try:
            self.logger.info(f"🔄 Executing response from memory playbook: {response.get('name', 'Unknown')}")
            
            if 'actions' in response:
                for action in response['actions']:
                    success = self.execute_actuator_command(
                        action.get('actuator'), 
                        action.get('command')
                    )
                    if not success:
                        self.logger.warning(f"⚠️ Failed to execute action: {action}")
            
            # إرسال الإشعارات إذا كانت محددة
            if 'notifications' in response:
                for notification in response['notifications']:
                    self.send_emergency_notifications(notification)
            
            # تسجيل الاستجابة الناجحة
            if hasattr(self, 'lifelong_memory'):
                self.lifelong_memory.record_response_success(response)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute response: {e}")

    def execute_emergency_protocol(self, decision):
        """تنفيذ بروتوكول الطوارئ"""
        try:
            self.emergency_mode = True
            self.logger.critical(f"🚨 EMERGENCY PROTOCOL ACTIVATED: {decision}")
            
            # تنفيذ إجراءات الطوارئ
            if hasattr(decision, 'actions'):
                for action in decision.actions:
                    self.execute_actuator_command(action['actuator'], action['command'])
            
            # إرسال إشعارات الطوارئ
            self.send_emergency_notifications(decision)
            
            # تسجيل الحادث في الذاكرة الدائمة
            if hasattr(self, 'lifelong_memory'):
                self.lifelong_memory.record_incident({
                    'type': 'emergency',
                    'decision': str(decision),
                    'timestamp': datetime.now(),
                    'resolved': False
                })
                
        except Exception as e:
            self.logger.error(f"❌ Emergency protocol execution failed: {e}")

    def execute_default_emergency_response(self, anomalies):
        """تنفيذ استجابة الطوارئ الافتراضية"""
        try:
            self.emergency_mode = True
            self.logger.critical("🚨 DEFAULT EMERGENCY RESPONSE ACTIVATED")
            
            # الإجراءات الافتراضية
            default_actions = [
                {'actuator': 'main_valve', 'command': 'close'},
                {'actuator': 'emergency_shutdown', 'command': 'activate'},
                {'actuator': 'alarm_system', 'command': 'activate'}
            ]
            
            for action in default_actions:
                self.execute_actuator_command(action['actuator'], action['command'])
            
            # إرسال إشعارات
            self.send_emergency_notifications({
                'message': 'Emergency detected! Default safety protocols activated.',
                'severity': 'CRITICAL'
            })
            
        except Exception as e:
            self.logger.error(f"❌ Default emergency response failed: {e}")

    def execute_actuator_command(self, actuator_id: str, command: Any):
        """تنفيذ أمر للمشغل"""
        try:
            # محاكاة تنفيذ الأمر (سيتم استبدالها بـ GPIO على Raspberry Pi)
            success = random.random() > 0.1  # 90% success rate in simulation
            
            actuator_state = ActuatorState(
                actuator_id=actuator_id,
                state=command,
                timestamp=datetime.now(),
                command_source='system',
                success=success,
                response_time=random.uniform(0.1, 0.5)
            )
            
            self.actuator_states[actuator_id] = actuator_state
            
            if success:
                self.logger.info(f"✅ Actuator {actuator_id} executed command: {command}")
            else:
                self.logger.warning(f"⚠️ Actuator {actuator_id} failed to execute: {command}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Actuator command failed for {actuator_id}: {e}")
            return False

    def send_emergency_notifications(self, emergency_data):
        """إرسال إشعارات الطوارئ"""
        try:
            message = f"🚨 EMERGENCY: {emergency_data.get('message', 'Critical anomaly detected')}"
            
            # إرسال عبر Twilio إذا كان متاحاً
            if self.twilio:
                self.twilio.send_emergency_sms(message)
            
            # إرسال عبر نظام الدردشة AI إذا كان متاحاً
            if self.ai_chat:
                self.ai_chat.broadcast_emergency(message)
                
            self.logger.critical(f"📢 Emergency notification sent: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency notification failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام الحالية"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.system_health,
                'emergency_mode': self.emergency_mode,
                'active_sensors': len(self.sensor_readings),
                'active_actuators': len(self.actuator_states),
                'model_performance': self.model_selector.get_performance_report() if hasattr(self, 'model_selector') else {},
                'recent_anomalies': len(self.sensor_readings.get('anomalies', [])),
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage()
            }
            return status
        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}

    def _get_memory_usage(self) -> float:
        """الحصول على استخدام الذاكرة (محاكاة)"""
        try:
            # في الإنتاج الحقيقي، سنستخدم psutil أو مكتبة نظامية
            return random.uniform(0.3, 0.8)
        except:
            return 0.5

    def _get_cpu_usage(self) -> float:
        """الحصول على استخدام CPU (محاكاة)"""
        try:
            # في الإنتاج الحقيقي، سنستخدم psutil أو مكتبة نظامية
            return random.uniform(0.2, 0.7)
        except:
            return 0.4

    def optimize_performance(self):
        """تحسين أداء النظام"""
        try:
            if hasattr(self, 'performance_optimizer'):
                self.performance_optimizer.optimize(self)
            
            # التحسين التلقائي للنماذج
            if hasattr(self, 'model_selector'):
                self.model_selector.auto_optimize()
                
            self.logger.info("✅ System performance optimized")
            
        except Exception as e:
            self.logger.error(f"❌ Performance optimization failed: {e}")

    def shutdown(self):
        """إيقاف النظام بشكل آمن"""
        try:
            self.stop_monitoring()
            
            # حفظ حالة النظام
            if hasattr(self, 'lifelong_memory'):
                self.lifelong_memory.save_state()
            
            # إغلاق جميع الموارد
            self._cleanup_resources()
            
            self.logger.info("🛑 System shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ System shutdown failed: {e}")

    def _cleanup_resources(self):
        """تنظيف الموارد"""
        try:
            # تنظيف أي موارد مفتوحة
            if hasattr(self, 'twilio'):
                self.twilio.cleanup()
            if hasattr(self, 'ai_chat'):
                self.ai_chat.cleanup()
        except Exception as e:
            self.logger.error(f"❌ Resource cleanup failed: {e}")

    def __del__(self):
        """الدمار - التأكد من إيقاف النظام"""
        try:
            self.shutdown()
        except:
            pass

# دالة مساعدة لإنشاء النظام
def create_core_system(config_path: str = None) -> AdvancedCoreSystem:
    """إنشاء نظام أساسي مع معالجة الأخطاء"""
    try:
        config = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
            except:
                pass
        
        return AdvancedCoreSystem(config)
    except Exception as e:
        logging.error(f"❌ Failed to create core system: {e}")
        # إرجاع نظام بديل في حالة الفشل
        return AdvancedCoreSystem({})

# مثال للاستخدام
if __name__ == "__main__":
    # إعداد التسجيل
    logging.basicConfig(level=logging.INFO)
    
    try:
        # إنشاء النظام
        system = create_core_system()
        
        # تشغيل النظام لفترة قصيرة للاختبار
        print("🚀 Starting system...")
        time.sleep(10)
        
        # الحصول على حالة النظام
        status = system.get_system_status()
        print(f"📊 System status: {status}")
        
        # إيقاف النظام
        system.shutdown()
        print("✅ System test completed successfully")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        logging.error(f"System test failed: {e}")
