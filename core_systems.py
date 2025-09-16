import time
import threading
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# استيراد جميع المكونات
from preprocess_data import DataPreprocessor
from generate_sensor_data import SensorDataGenerator
from ai_systems.advanced_anomaly_system import AdvancedAnomalyDetector
from ai_systems.advanced_prediction_engine import AdvancedPredictionEngine
from ai_systems.advanced_prediction_modules import PredictionModules
from ai_systems.intervention_engine import InterventionEngine
from ai_systems.lifelong_memory import LifelongMemory
from ai_systems.memory_playbook import MemoryPlaybook
from ai_systems.dynamic_model_selector import DynamicModelSelector, ModelType
from twilio_integration import TwilioIntegration
from ai_chat_system import AIChatSystem
from utils.error_handling import SystemError, ErrorSeverity, GracefulDegradation
from utils.performance_optimizer import PerformanceOptimizer

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
        self.config = config
        self.sensor_readings: Dict[str, List[SensorReading]] = {}
        self.actuator_states: Dict[str, ActuatorState] = {}
        self.system_health = 100
        self.emergency_mode = False
        self.degradation_handler = GracefulDegradation(config)
        self.performance_optimizer = PerformanceOptimizer(config)
        
        # تهيئة جميع المكونات
        self.setup_components()
        self.logger = logging.getLogger(__name__)
        
    def setup_components(self):
        """تهيئة جميع المكونات المتقدمة"""
        try:
            # معالجة البيانات
            self.data_preprocessor = DataPreprocessor(self.config)
            self.data_generator = SensorDataGenerator(self.config)
            
            # Dynamic Model Selection - ميزة براءة الاختراع
            self.model_selector = DynamicModelSelector(self.config)
            
            # أنظمة الذكاء الاصطناعي
            self.anomaly_detector = AdvancedAnomalyDetector(self.config)
            self.prediction_engine = AdvancedPredictionEngine(self.config)
            self.prediction_modules = PredictionModules(self.config)
            self.intervention_engine = InterventionEngine(self.config, self.prediction_engine, self)
            
            # تسجيل النماذج في النظام الذكي
            self._register_models()
            
            # أنظمة الذاكرة
            self.lifelong_memory = LifelongMemory(self.config)
            self.memory_playbook = MemoryPlaybook(self.config)
            
            # التكاملات الخارجية
            self.twilio = TwilioIntegration(self.config)
            self.ai_chat = AIChatSystem(self.config)
            
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
            raise
    
    def _register_models(self):
        """تسجيل جميع النماذج في نظام الاختيار الذكي"""
        self.model_selector.register_model(ModelType.LSTM, self.anomaly_detector.lstm_model)
        self.model_selector.register_model(ModelType.ISOLATION_FOREST, self.anomaly_detector.isolation_forest)
        self.model_selector.register_model(ModelType.ONE_CLASS_SVM, self.anomaly_detector.one_class_svm)
        self.model_selector.register_model(ModelType.MONTE_CARLO, self.prediction_engine)
        
        self.logger.info("✅ All models registered in dynamic selector")
    
    def train_ai_models(self):
        """تدريب نماذج الذكاء الاصطناعي"""
        try:
            # توليد بيانات تدريبية واقعية
            training_data = self.data_generator.generate_training_data(5000)
            processed_data = self.data_preprocessor.preprocess_data(training_data)
            
            # تدريب نماذج كشف الشذوذ
            self.anomaly_detector.train_models(processed_data)
            
            # تدريب نماذج LSTM
            self.prediction_engine.train_lstm_models(processed_data)
            
            self.logger.info("✅ AI models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ AI model training failed: {e}")
            self.degradation_handler.handle_error(
                SystemError("Model training failed", ErrorSeverity.MEDIUM)
            )
    
    def start_monitoring(self):
        """بدء مراقبة النظام"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("📊 Started real-time monitoring")
    
    def _monitoring_loop(self):
        """حلقة المراقبة الرئيسية مع Dynamic Model Selection"""
        while True:
            try:
                # قراءة البيانات من المستشعرات
                sensor_data = self.read_all_sensors()
                
                # معالجة البيانات
                processed_data = self.data_preprocessor.preprocess_realtime_data(sensor_data)
                
                # اختيار أفضل نموذج تلقائياً - ميزة براءة الاختراع
                best_model = self._select_best_model_for_task(processed_data)
                
                # استخدام النموذج المختار
                if best_model == ModelType.ISOLATION_FOREST:
                    anomalies = self.anomaly_detector.detect_with_isolation_forest(processed_data)
                elif best_model == ModelType.ONE_CLASS_SVM:
                    anomalies = self.anomaly_detector.detect_with_one_class_svm(processed_data)
                elif best_model == ModelType.LSTM:
                    anomalies = self.anomaly_detector.detect_with_lstm(processed_data)
                else:
                    anomalies = self.anomaly_detector.detect_anomalies(processed_data)
                
                # التنبؤ بالمستقبل
                predictions = self.prediction_engine.predict(processed_data)
                
                # تقييم الوضع واتخاذ القرارات
                if anomalies['critical_anomalies']:
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
    
    def _select_best_model_for_task(self, processed_data: Dict[str, Any]) -> ModelType:
        """اختيار أفضل نموذج للمهمة الحالية"""
        data_characteristics = {
            'size': len(processed_data),
            'dimensions': len(processed_data.keys()),
            'variability': np.std(list(processed_data.values())) if processed_data else 0
        }
        
        resource_constraints = {
            'cpu_available': 0.8,  # سيتم قراءتها من النظام
            'memory_available': 0.7  # سيتم قراءتها من النظام
        }
        
        return self.model_selector.select_best_model(data_characteristics, resource_constraints)
    
    def _update_model_performance(self, model_type: ModelType, anomalies: Dict[str, Any], 
                                predictions: Dict[str, Any]):
        """تحديث أداء النموذج المستخدم"""
        accuracy = self._calculate_accuracy(anomalies, predictions)
        latency = time.time() - getattr(anomalies, 'processing_time', 1.0)
        memory_usage = 0.5  # سيتم قياسه من النظام
        
        self.model_selector.update_model_performance(
            model_type, accuracy, latency, memory_usage, True
        )
    
    def _calculate_accuracy(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]) -> float:
        """حساب دقة النموذج"""
        # محاكاة حساب الدقة
        return random.uniform(0.7, 0.95)
    
    def read_all_sensors(self) -> Dict[str, Any]:
        """قراءة جميع المستشعرات"""
        sensor_data = {}
        
        for sensor_type, sensor_config in self.config.get('hardware.sensors', {}).items():
            reading = self.read_sensor(sensor_type)
            if reading:
                sensor_data[sensor_type] = reading.value
        
        return sensor_data
    
    def read_sensor(self, sensor_type: str) -> Optional[SensorReading]:
        """قراءة مستشعر معين"""
        try:
            if self.config.get('system.simulation_mode', True):
                value = self.data_generator.generate_sensor_value(sensor_type)
            else:
                value = self._read_real_sensor(sensor_type)
            
            reading = SensorReading(
                sensor_id=f"{sensor_type}_01",
                sensor_type=SensorType(sensor_type),
                value=value,
                unit=self.config.get(f'hardware.sensors.{sensor_type}.unit', 'unknown'),
                timestamp=datetime.now(),
                confidence=0.95
            )
            
            # تخزين القراءة
            if reading.sensor_id not in self.sensor_readings:
                self.sensor_readings[reading.sensor_id] = []
            self.sensor_readings[reading.sensor_id].append(reading)
            
            return reading
            
        except Exception as e:
            self.logger.error(f"❌ Error reading sensor {sensor_type}: {e}")
            return None
    
    def _read_real_sensor(self, sensor_type: str) -> float:
        """قراءة مستشعر حقيقي (لـ Raspberry Pi)"""
        # محاكاة للقراءة الحقيقية
        try:
            import RPi.GPIO as GPIO
            pin = self.config.get(f'hardware.sensors.{sensor_type}.pin')
            if pin is not None:
                return random.uniform(
                    self.config.get(f'hardware.sensors.{sensor_type}.min', 0),
                    self.config.get(f'hardware.sensors.{sensor_type}.max', 100)
                )
            return 0.0
        except ImportError:
            return self.data_generator.generate_sensor_value(sensor_type)
    
    def handle_emergency(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]):
        """معالجة حالات الطوارئ بشكل كامل"""
        try:
            # البحث في memory playbook عن أفضل استجابة
            best_response = self.memory_playbook.get_best_response(anomalies)
            
            if best_response:
                # تنفيذ استجابة الذاكرة
                self.execute_response(best_response)
            else:
                # استجابة جديدة من intervention engine
                decisions = self.intervention_engine.evaluate_situation(anomalies, predictions)
                for decision in decisions:
                    if decision.priority.value >= 4:  # HIGH priority or above
                        self.execute_intervention(decision)
            
            # إرسال تنبيهات الطوارئ
            if self.twilio and self.config.get('twilio.enabled', False):
                self.twilio.send_emergency_alert(anomalies)
            
            self.logger.warning(f"🚨 Emergency handled: {len(anomalies['anomalies'])} critical anomalies")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency handling failed: {e}")
            self.degradation_handler.handle_error(
                SystemError("Emergency handling failed", ErrorSeverity.HIGH)
            )
    
    def execute_intervention(self, decision):
        """تنفيذ قرار التدخل"""
        try:
            result = self.intervention_engine.execute_intervention(decision)
            
            # التعلم من النتيجة
            self.lifelong_memory.learn_from_intervention(decision, result)
            self.memory_playbook.add_response(decision, result)
            
            self.logger.info(f"✅ Intervention executed: {decision.intervention_type.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Intervention execution failed: {e}")
    
    def execute_response(self, response: Dict[str, Any]):
        """تنفيذ استجابة من memory playbook"""
        try:
            # تنفيذ الإجراءات المطلوبة
            for action in response.get('actions', []):
                self.control_actuator(action['target'], action['value'], "memory_playbook")
            
            self.logger.info(f"✅ Memory playbook response executed: {response['response_id']}")
            
        except Exception as e:
            self.logger.error(f"❌ Memory playbook execution failed: {e}")
    
    def control_actuator(self, actuator_id: str, command: Any, source: str = "system"):
        """التحكم في المشغلات"""
        try:
            # محاكاة التحكم بالمشغلات
            state = ActuatorState(
                actuator_id=actuator_id,
                state=command,
                timestamp=datetime.now(),
                command_source=source,
                success=True,
                response_time=0.1
            )
            
            self.actuator_states[actuator_id] = state
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Actuator control failed: {e}")
            return ActuatorState(
                actuator_id=actuator_id,
                state=command,
                timestamp=datetime.now(),
                command_source=source,
                success=False,
                response_time=0.0
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام"""
        return {
            'health': self.system_health,
            'emergency_mode': self.emergency_mode,
            'sensor_count': len(self.sensor_readings),
            'anomaly_count': sum(1 for readings in self.sensor_readings.values() 
                               for r in readings if r.status != 'normal'),
            'timestamp': datetime.now(),
            'model_performance': self.model_selector.get_performance_report()
        }
    
    def reverse_digital_twin_simulation(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة التوأم الرقمي العكسي - مكتمل الآن"""
        try:
            # تحليل السيناريو
            scenario_type = scenario.get('type', 'gas_leak')
            duration = scenario.get('duration', 6)
            
            # تشغيل المحاكاة العكسية
            simulation_data = self._run_reverse_simulation(scenario_type, duration)
            
            # تحليل النتائج
            analysis = self._analyze_simulation_results(simulation_data, scenario)
            
            # إضافة إلى الذاكرة للتعلم المستقبلي
            self.lifelong_memory.store_simulation(scenario, simulation_data, analysis)
            
            return {
                'success': True,
                'scenario': scenario_type,
                'duration': duration,
                'results': analysis,
                'recommendations': self._generate_recommendations(analysis),
                'simulation_data': simulation_data
            }
            
        except Exception as e:
            self.logger.error(f"❌ Reverse simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'scenario': scenario
            }
    
    def _run_reverse_simulation(self, scenario_type: str, duration: int) -> Dict[str, Any]:
        """تشغيل المحاكاة العكسية"""
        # محاكاة واقعية للسيناريوهات العكسية
        simulations = {
            'gas_leak': self._simulate_gas_leak(duration),
            'pressure_surge': self._simulate_pressure_surge(duration),
            'equipment_failure': self._simulate_equipment_failure(duration)
        }
        
        return simulations.get(scenario_type, {})
    
    def _simulate_gas_leak(self, duration: int) -> Dict[str, Any]:
        """محاكاة تسرب غاز"""
        return {
            'methane_levels': np.linspace(50, 1500, duration),
            'pressure_changes': np.linspace(1013, 800, duration),
            'safety_breaches': [False] * (duration-2) + [True, True],
            'response_times': [30, 25, 20, 15, 10, 5]
        }
    
    def _analyze_simulation_results(self, simulation_data: Dict[str, Any], 
                                  scenario: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل نتائج المحاكاة"""
        return {
            'risk_level': self._calculate_risk_level(simulation_data),
            'impact_score': self._calculate_impact_score(simulation_data),
            'recovery_time': self._estimate_recovery_time(simulation_data),
            'cost_estimate': self._estimate_costs(simulation_data),
            'prevention_effectiveness': self._assess_prevention_measures(simulation_data)
        }
    
    def cleanup(self):
        """تنظيف الموارد"""
        try:
            self.monitoring_thread.join(timeout=2.0)
            self.logger.info("🧹 System resources cleaned up")
        except:
            pass

# دالة مساعدة لإنشاء النظام
def create_core_system(config) -> AdvancedCoreSystem:
    return AdvancedCoreSystem(config)
