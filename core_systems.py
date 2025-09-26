import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from enum import Enum

# استيرادات صحيحة من الهيكل الجديد
from hardware.sensor_manager import SensorManager
from hardware.relay_controller import RelayController
from ai_systems.advanced_anomaly_system import AdvancedAnomalyDetector
from ai_systems.advanced_prediction_engine import AdvancedPredictionEngine
from ai_systems.intervention_engine import InterventionEngine
from ai_systems.lifelong_memory import LifelongMemory
from ai_systems.memory_playbook import MemoryPlaybook
from ai_systems.dynamic_model_selector import DynamicModelSelector, ModelType
from utils.helpers import setup_logging, load_config
from utils.config_loader import ConfigLoader

class SystemStatus(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DigitalTwinCore:
    """
    القلب الرئيسي للتوأم الرقمي - يدير كل العمليات
    """
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.logger = setup_logging()
        self.config = ConfigLoader(config_path).load()
        
        # تهيئة الأنظمة
        self.sensor_manager = SensorManager(self.config)
        self.relay_controller = RelayController(self.config)
        self.anomaly_detector = AdvancedAnomalyDetector(self.config)
        self.prediction_engine = AdvancedPredictionEngine(self.config)
        self.intervention_engine = InterventionEngine(self.config)
        self.memory_system = LifelongMemory(self.config)
        self.playbook_system = MemoryPlaybook(self.config)
        self.model_selector = DynamicModelSelector(self.config)
        
        self.system_status = SystemStatus.NORMAL
        self.real_time_data = {}
        self.prediction_cache = {}
        self.emergency_protocols = {}
        
        self._initialize_systems()
        self.logger.info("✅ Digital Twin Core initialized successfully")
    
    def _initialize_systems(self):
        """تهيئة جميع الأنظمة الفرعية"""
        try:
            # تحميل خطط الطوارئ
            self.emergency_protocols = self.playbook_system.load_emergency_playbooks()
            
            # بدء مراقبة المستشعرات
            self.sensor_manager.start_monitoring()
            
            # تهيئة نماذج الذكاء الاصطناعي
            self._initialize_ai_models()
            
            self.logger.info("✅ All subsystems initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize systems: {e}")
            raise
    
    def _initialize_ai_models(self):
        """تهيئة نماذج الذكاء الاصطناعي"""
        try:
            # تسجيل النماذج في النظام الذكي
            self.model_selector.register_model(ModelType.LSTM, self.prediction_engine)
            self.model_selector.register_model(ModelType.ISOLATION_FOREST, self.anomaly_detector)
            self.model_selector.register_model(ModelType.MONTE_CARLO, self.intervention_engine)
            
            self.logger.info("✅ AI Models registered in dynamic selector")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AI models: {e}")
    
    def process_real_time_data(self) -> Dict[str, Any]:
        """
        معالجة البيانات الحية من المستشعرات
        """
        try:
            # جمع البيانات من المستشعرات
            sensor_data = self.sensor_manager.get_all_sensor_data()
            self.real_time_data = sensor_data
            
            # تحليل الشذوذ
            anomaly_results = self.anomaly_detector.detect_anomalies(sensor_data)
            
            # التنبؤ بالمستقبل (24 ساعة)
            predictions = self.prediction_engine.predict_next_24_hours(sensor_data)
            
            # تحديث الذاكرة
            self.memory_system.store_experience(sensor_data, anomaly_results, predictions)
            
            # اختيار أفضل نموذج للجولة القادمة
            best_model = self._select_best_model(sensor_data)
            
            result = {
                'timestamp': datetime.now(),
                'sensor_data': sensor_data,
                'anomalies': anomaly_results,
                'predictions': predictions,
                'selected_model': best_model.value,
                'system_status': self.system_status.value
            }
            
            # التحقق من حالات الطوارئ
            self._check_emergency_conditions(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing real-time data: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def _select_best_model(self, sensor_data: Dict[str, Any]) -> ModelType:
        """اختيار أفضل نموذج بناءً على البيانات الحالية"""
        try:
            data_characteristics = {
                'size': len(sensor_data),
                'dimensions': len(sensor_data.keys()),
                'variability': np.std(list(sensor_data.values())) if sensor_data else 0
            }
            
            resource_constraints = {
                'cpu_available': 0.8,  # محاكاة - في الواقع يتم قياسه
                'memory_available': 0.7
            }
            
            return self.model_selector.select_best_model(data_characteristics, resource_constraints)
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting best model: {e}")
            return ModelType.LSTM
    
    def _check_emergency_conditions(self, processed_data: Dict[str, Any]):
        """التحقق من شروط الطوارئ وتنفيذ الإجراءات"""
        try:
            anomalies = processed_data['anomalies']
            sensor_data = processed_data['sensor_data']
            
            # تحليل مستوى الخطورة
            risk_level = self._calculate_risk_level(anomalies, sensor_data)
            
            if risk_level >= 0.8:
                self.system_status = SystemStatus.EMERGENCY
                self._execute_emergency_protocol(risk_level, processed_data)
            elif risk_level >= 0.6:
                self.system_status = SystemStatus.CRITICAL
            elif risk_level >= 0.4:
                self.system_status = SystemStatus.WARNING
            else:
                self.system_status = SystemStatus.NORMAL
                
        except Exception as e:
            self.logger.error(f"❌ Error checking emergency conditions: {e}")
    
    def _calculate_risk_level(self, anomalies: Dict[str, Any], sensor_data: Dict[str, Any]) -> float:
        """حساب مستوى الخطورة"""
        try:
            risk_score = 0.0
            weights = {
                'pressure': 0.3,
                'temperature': 0.25,
                'methane': 0.2,
                'vibration': 0.15,
                'flow': 0.1
            }
            
            for sensor, value in sensor_data.items():
                if sensor in weights:
                    # تحويل القيمة إلى مستوى خطورة (0-1)
                    normalized_value = self._normalize_sensor_value(sensor, value)
                    risk_score += normalized_value * weights[sensor]
            
            # إضافة تأثير الشذوذ
            if anomalies.get('critical_anomalies', 0) > 0:
                risk_score = min(1.0, risk_score + 0.3)
            
            return min(1.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk level: {e}")
            return 0.0
    
    def _normalize_sensor_value(self, sensor_type: str, value: float) -> float:
        """تطبيع قيم المستشعرات إلى نطاق 0-1"""
        try:
            ranges = {
                'pressure': (0, 100),  # بار
                'temperature': (0, 150),  # درجة مئوية
                'methane': (0, 1000),  # جزء في المليون
                'vibration': (0, 10),  # م/ث²
                'flow': (0, 100)  # لتر/دقيقة
            }
            
            if sensor_type in ranges:
                min_val, max_val = ranges[sensor_type]
                return (value - min_val) / (max_val - min_val)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error normalizing sensor value: {e}")
            return 0.0
    
    def _execute_emergency_protocol(self, risk_level: float, data: Dict[str, Any]):
        """تنفيذ بروتوكول الطوارئ"""
        try:
            self.logger.critical(f"🚨 EMERGENCY PROTOCOL ACTIVATED - Risk Level: {risk_level}")
            
            # تنفيذ قرار مستحيل باستخدام مونتي كارلو
            decision = self.intervention_engine.make_impossible_decision(data, risk_level)
            
            # تنفيذ الإجراءات على الهاردوير
            self._execute_hardware_actions(decision['actions'])
            
            # إرسال تنبيهات
            self._send_emergency_alerts(decision)
            
            self.logger.info("✅ Emergency protocol executed")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing emergency protocol: {e}")
    
    def _execute_hardware_actions(self, actions: List[Dict[str, Any]]):
        """تنفيذ الإجراءات على الهاردوير"""
        try:
            for action in actions:
                if action['type'] == 'relay_control':
                    self.relay_controller.control_relay(
                        action['relay_id'], 
                        action['state']
                    )
                elif action['type'] == 'valve_control':
                    # التحكم في الصمامات عبر السيرفو
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Error executing hardware actions: {e}")
    
    def _send_emergency_alerts(self, decision: Dict[str, Any]):
        """إرسال تنبيهات الطوارئ"""
        try:
            # سيتم تنفيذ هذا في advanced_systems.py
            pass
        except Exception as e:
            self.logger.error(f"❌ Error sending emergency alerts: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """الحصول على صحة النظام"""
        try:
            return {
                'status': self.system_status.value,
                'sensor_health': self.sensor_manager.get_health_status(),
                'ai_health': self.model_selector.get_performance_report(),
                'last_update': datetime.now(),
                'active_alerts': len(self.emergency_protocols)
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting system health: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """إيقاف النظام بأمان"""
        try:
            self.sensor_manager.stop_monitoring()
            self.relay_controller.safe_shutdown()
            self.logger.info("✅ Digital Twin Core shutdown safely")
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")

# دالة إنشاء النظام الرئيسي
def create_digital_twin(config_path: str = "config/settings.json") -> DigitalTwinCore:
    """إنشاء وتهيئة التوأم الرقمي"""
    try:
        return DigitalTwinCore(config_path)
    except Exception as e:
        logging.error(f"❌ Failed to create digital twin: {e}")
        raise

if __name__ == "__main__":
    # اختبار النظام
    twin = create_digital_twin()
    print("✅ Digital Twin Core is running!")
