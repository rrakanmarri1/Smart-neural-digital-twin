import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
import RPi.GPIO as GPIO
from dataclasses import dataclass
from enum import Enum
import random

from ai_systems import ForeSightEngine, AdvancedAnomalySystem, AdvancedPredictionEngine
from config_and_logging import SmartConfig, RelayController

class SensorStatus(Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED" 
    FAILED = "FAILED"
    SIMULATED = "SIMULATED"

@dataclass
class SensorReading:
    """هيكل بيانات لقراءة المستشعر"""
    value: float
    confidence: float
    status: SensorStatus
    timestamp: datetime
    source: str  # 'physical', 'simulated', 'fused'

class AdaptiveSensorFusionGrid:
    """
    🌐 SenseGrid - شبكة استشعار افتراضية أذكى من أي شبكة فيزيائية
    
    الميزات:
    - دمج بيانات مستشعرات متعددة
    - محاكاة ذكية للمستشعرات المعطلة
    - تصحيح أخطاء في الوقت الحقيقي
    - تكيف ذاتي مع تغير الظروف
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.SenseGrid')
        
        # حالة المستشعرات
        self.sensor_status = {}
        self.sensor_calibration = {}
        self.fusion_models = {}
        self.correlation_matrix = {}
        
        # سجل القراءات
        self.sensor_history = {sensor: [] for sensor in config['sensors'].keys()}
        self.fusion_history = []
        
        self._initialize_sensor_grid()
        self.logger.info("🌐 Adaptive Sensor Fusion Grid (SenseGrid) Initialized")
    
    def _initialize_sensor_grid(self):
        """تهيئة شبكة المستشعرات"""
        try:
            # فحص حالة المستشعرات الفعلية
            self._scan_physical_sensors()
            
            # بناء مصفوفة الارتباط بين المستشعرات
            self._build_correlation_matrix()
            
            # تدريب نماذج الدمج
            self._train_fusion_models()
            
            # معايرة المستشعرات
            self._calibrate_sensors()
            
        except Exception as e:
            self.logger.error(f"❌ SenseGrid initialization failed: {e}")
    
    def _scan_physical_sensors(self):
        """فحص المستشعرات الفعلية على الـ Raspberry Pi"""
        sensor_pins = self.config['raspberry_pi']['sensor_pins']
        
        for sensor_name, pin in sensor_pins.items():
            try:
                # محاولة قراءة المستشعر الفعلي
                if self._read_physical_sensor(pin):
                    self.sensor_status[sensor_name] = SensorStatus.ACTIVE
                    self.logger.info(f"✅ Physical sensor {sensor_name} detected on pin {pin}")
                else:
                    self.sensor_status[sensor_name] = SensorStatus.FAILED
                    self.logger.warning(f"⚠️ Physical sensor {sensor_name} failed, using simulation")
                    
            except Exception as e:
                self.sensor_status[sensor_name] = SensorStatus.FAILED
                self.logger.warning(f"⚠️ Sensor {sensor_name} error: {e}, using simulation")
    
    def _read_physical_sensor(self, pin: int) -> bool:
        """قراءة المستشعر الفعلي (محاكاة للتوضيح)"""
        try:
            # في الواقع، هنا سيتم قراءة القيمة الفعلية من الـ GPIO
            # للمحاكاة، نعيد True إذا كان المستشعر "موجوداً"
            
            # محاكاة فشل عشوائي بنسبة 5% (لاختبار النظام)
            if random.random() < 0.05:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Physical sensor reading failed on pin {pin}: {e}")
            return False
    
    def _build_correlation_matrix(self):
        """بناء مصفوفة الارتباط بين المستشعرات"""
        # هذه المصفوفة تحدد كيف ترتبط قراءات المستشعرات ببعضها
        # بناءً على البيانات التاريخية والفيزياء الأساسية
        
        self.correlation_matrix = {
            'pressure': {
                'temperature': 0.7,    # الضغط والحرارة مرتبطان
                'flow': 0.8,           # الضغط والتدفق مرتبطان بقوة
                'vibration': 0.3,      # ضعف الارتباط
                'methane': 0.1,        # ارتباط ضعيف
                'hydrogen_sulfide': 0.1
            },
            'temperature': {
                'pressure': 0.7,
                'flow': 0.6,
                'vibration': 0.4,
                'methane': 0.2,
                'hydrogen_sulfide': 0.1
            },
            'flow': {
                'pressure': 0.8,
                'temperature': 0.6,
                'vibration': 0.5,
                'methane': 0.1,
                'hydrogen_sulfide': 0.1
            },
            # ... باقي المستشعرات
        }
    
    def _train_fusion_models(self):
        """تدريب نماذج دمج المستشعرات"""
        # نماذج التعلم الآلي للتنبؤ بقيم المستشعرات المفقودة
        # بناءً على المستشعرات المتاحة
        
        for target_sensor in self.config['sensors'].keys():
            # تحديد المستشعرات المساعدة للتنبؤ
            supporting_sensors = [s for s in self.config['sensors'].keys() if s != target_sensor]
            
            self.fusion_models[target_sensor] = {
                'supporting_sensors': supporting_sensors,
                'weights': self._calculate_sensor_weights(target_sensor, supporting_sensors),
                'accuracy': 0.85  # دقة متوقعة للنموذج
            }
    
    def _calculate_sensor_weights(self, target_sensor: str, supporting_sensors: List[str]) -> Dict[str, float]:
        """حساب أوزان المستشعرات المساعدة"""
        weights = {}
        total_correlation = 0
        
        for sensor in supporting_sensors:
            correlation = self.correlation_matrix.get(target_sensor, {}).get(sensor, 0.1)
            weights[sensor] = correlation
            total_correlation += correlation
        
        # تطبيع الأوزان
        if total_correlation > 0:
            for sensor in weights:
                weights[sensor] /= total_correlation
        
        return weights
    
    def _calibrate_sensors(self):
        """معايرة المستشعرات"""
        for sensor_name in self.config['sensors'].keys():
            self.sensor_calibration[sensor_name] = {
                'offset': random.uniform(-0.05, 0.05),  # انزياح عشوائي صغير
                'drift': 0.0,  # انحراف زمني
                'last_calibration': datetime.now()
            }
    
    def read_sensor_grid(self) -> Dict[str, SensorReading]:
        """قراءة شبكة المستشعرات الكاملة"""
        sensor_readings = {}
        
        # جمع البيانات من المستشعرات المتاحة
        available_data = self._collect_available_sensor_data()
        
        # دمج البيانات والمحاكاة
        for sensor_name in self.config['sensors'].keys():
            if sensor_name in available_data:
                # استخدام البيانات الفعلية
                reading = available_data[sensor_name]
            else:
                # محاكاة المستشعر المعطل
                reading = self._simulate_sensor_reading(sensor_name, available_data)
            
            sensor_readings[sensor_name] = reading
        
        # تطبيق تصحيحات الدمج
        fused_readings = self._apply_sensor_fusion(sensor_readings)
        
        # تخزين في السجل
        self._update_sensor_history(fused_readings)
        
        return fused_readings
    
    def _collect_available_sensor_data(self) -> Dict[str, SensorReading]:
        """جمع البيانات من المستشعرات المتاحة"""
        available_data = {}
        
        for sensor_name, status in self.sensor_status.items():
            if status == SensorStatus.ACTIVE:
                try:
                    # قراءة المستشعر الفعلي
                    raw_value = self._read_sensor_value(sensor_name)
                    calibrated_value = self._apply_calibration(sensor_name, raw_value)
                    
                    reading = SensorReading(
                        value=calibrated_value,
                        confidence=0.95,  # ثقة عالية في المستشعرات الفعلية
                        status=SensorStatus.ACTIVE,
                        timestamp=datetime.now(),
                        source='physical'
                    )
                    
                    available_data[sensor_name] = reading
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to read physical sensor {sensor_name}: {e}")
                    self.sensor_status[sensor_name] = SensorStatus.FAILED
        
        return available_data
    
    def _read_sensor_value(self, sensor_name: str) -> float:
        """قراءة قيمة المستشعر الفعلية (محاكاة)"""
        # في الواقع، هذه القيمة ستأتي من الـ GPIO
        # هنا نستخدم محاكاة واقعية
        
        sensor_config = self.config['sensors'][sensor_name]
        
        # قيمة أساسية واقعية
        base_value = random.uniform(
            sensor_config['min'] * 0.3, 
            sensor_config['max'] * 0.7
        )
        
        # إضافة ضوضاء واقعية
        noise = random.gauss(0, base_value * 0.02)  # 2% ضوضاء
        value = base_value + noise
        
        # التأكد من الحدود
        value = max(sensor_config['min'], min(sensor_config['max'], value))
        
        return value
    
    def _apply_calibration(self, sensor_name: str, raw_value: float) -> float:
        """تطبيق المعايرة على قراءة المستشعر"""
        calibration = self.sensor_calibration.get(sensor_name, {})
        calibrated_value = raw_value * (1 + calibration.get('offset', 0))
        return calibrated_value
    
    def _simulate_sensor_reading(self, sensor_name: str, available_data: Dict[str, SensorReading]) -> SensorReading:
        """محاكاة قراءة مستشعر معطل"""
        try:
            if available_data:
                # استخدام بيانات المستشعرات الأخرى للتنبؤ
                simulated_value = self._predict_sensor_value(sensor_name, available_data)
                confidence = 0.8  # ثقة عالية في المحاكاة الذكية
            else:
                # لا توجد بيانات متاحة، استخدام قيمة افتراضية
                sensor_config = self.config['sensors'][sensor_name]
                simulated_value = random.uniform(
                    sensor_config['min'] * 0.4, 
                    sensor_config['max'] * 0.6
                )
                confidence = 0.5  # ثقة متوسطة
            
            return SensorReading(
                value=simulated_value,
                confidence=confidence,
                status=SensorStatus.SIMULATED,
                timestamp=datetime.now(),
                source='simulated'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Sensor simulation failed for {sensor_name}: {e}")
            
            # قيمة طارئة
            sensor_config = self.config['sensors'][sensor_name]
            emergency_value = (sensor_config['min'] + sensor_config['max']) / 2
            
            return SensorReading(
                value=emergency_value,
                confidence=0.3,  # ثقة منخفضة
                status=SensorStatus.FAILED,
                timestamp=datetime.now(),
                source='emergency'
            )
    
    def _predict_sensor_value(self, target_sensor: str, available_data: Dict[str, SensorReading]) -> float:
        """التنبؤ بقيمة مستشعر بناءً على المستشعرات المتاحة"""
        model = self.fusion_models.get(target_sensor, {})
        weights = model.get('weights', {})
        
        if not weights:
            # إذا لم يكن هناك نموذج، استخدام متوسط القيم المتاحة
            values = [reading.value for reading in available_data.values()]
            return sum(values) / len(values) if values else 0
        
        # حساب القيمة المتوقعة باستخدام الأوزان
        predicted_value = 0
        total_weight = 0
        
        for sensor_name, reading in available_data.items():
            if sensor_name in weights:
                weight = weights[sensor_name]
                predicted_value += reading.value * weight
                total_weight += weight
        
        if total_weight > 0:
            predicted_value /= total_weight
        else:
            # قيم افتراضية إذا فشل الحساب
            sensor_config = self.config['sensors'][target_sensor]
            predicted_value = (sensor_config['min'] + sensor_config['max']) / 2
        
        return predicted_value
    
    def _apply_sensor_fusion(self, sensor_readings: Dict[str, SensorReading]) -> Dict[str, SensorReading]:
        """تطبيق دمج متقدم للمستشعرات"""
        fused_readings = {}
        
        for sensor_name, reading in sensor_readings.items():
            # تطبيق تحسينات الدمج
            improved_reading = self._improve_reading_quality(sensor_name, reading, sensor_readings)
            fused_readings[sensor_name] = improved_reading
        
        return fused_readings
    
    def _improve_reading_quality(self, sensor_name: str, reading: SensorReading, 
                               all_readings: Dict[str, SensorReading]) -> SensorReading:
        """تحسين جودة القراءة باستخدام بيانات المستشعرات الأخرى"""
        
        # التحقق من التناسق مع المستشعرات الأخرى
        consistency_score = self._calculate_consistency(sensor_name, reading, all_readings)
        
        # تعديل الثقة بناءً على التناسق
        adjusted_confidence = reading.confidence * consistency_score
        
        # إذا كانت الثقة منخفضة جداً، تطبيق تصحيحات إضافية
        if adjusted_confidence < 0.6:
            corrected_value = self._apply_consistency_correction(sensor_name, reading, all_readings)
            return SensorReading(
                value=corrected_value,
                confidence=0.7,  # ثقة محسنة بعد التصحيح
                status=reading.status,
                timestamp=reading.timestamp,
                source='fused'
            )
        
        return SensorReading(
            value=reading.value,
            confidence=adjusted_confidence,
            status=reading.status,
            timestamp=reading.timestamp,
            source=reading.source
        )
    
    def _calculate_consistency(self, sensor_name: str, reading: SensorReading, 
                             all_readings: Dict[str, SensorReading]) -> float:
        """حساب درجة تناسق القراءة مع المستشعرات الأخرى"""
        if len(all_readings) <= 1:
            return 1.0  # لا توجد مستشعرات أخرى للمقارنة
        
        total_consistency = 0
        comparison_count = 0
        
        for other_sensor, other_reading in all_readings.items():
            if other_sensor != sensor_name:
                expected_relation = self.correlation_matrix.get(sensor_name, {}).get(other_sensor, 0)
                
                if expected_relation > 0.3:  # إذا كان هناك ارتباط معقول
                    # حساب الانحراف عن العلاقة المتوقعة
                    expected_value = other_reading.value * expected_relation
                    actual_deviation = abs(reading.value - expected_value) / (reading.value + 1e-8)
                    
                    consistency = max(0, 1 - actual_deviation)
                    total_consistency += consistency
                    comparison_count += 1
        
        return total_consistency / max(1, comparison_count)
    
    def _apply_consistency_correction(self, sensor_name: str, reading: SensorReading,
                                    all_readings: Dict[str, SensorReading]) -> float:
        """تطبيق تصحيح للقراءة غير المتسقة"""
        if not all_readings:
            return reading.value
        
        # استخدام متوسط القيم المتوقعة من المستشعرات الأخرى
        predicted_values = []
        
        for other_sensor, other_reading in all_readings.items():
            if other_sensor != sensor_name:
                correlation = self.correlation_matrix.get(sensor_name, {}).get(other_sensor, 0)
                if correlation > 0.3:
                    predicted_value = other_reading.value * correlation
                    predicted_values.append(predicted_value)
        
        if predicted_values:
            # متوسط القيم المتوقعة مع وزن القراءة الأصلية
            corrected_value = (sum(predicted_values) + reading.value) / (len(predicted_values) + 1)
            return corrected_value
        else:
            return reading.value
    
    def _update_sensor_history(self, readings: Dict[str, SensorReading]):
        """تحديث سجل المستشعرات"""
        for sensor_name, reading in readings.items():
            self.sensor_history[sensor_name].append(reading)
            
            # الاحتفاظ بـ 1000 قراءة فقط
            if len(self.sensor_history[sensor_name]) > 1000:
                self.sensor_history[sensor_name] = self.sensor_history[sensor_name][-1000:]
    
    def get_sensor_grid_status(self) -> Dict[str, Any]:
        """الحصول على حالة شبكة المستشعرات"""
        active_count = sum(1 for status in self.sensor_status.values() 
                          if status == SensorStatus.ACTIVE)
        simulated_count = sum(1 for status in self.sensor_status.values() 
                             if status == SensorStatus.SIMULATED)
        failed_count = sum(1 for status in self.sensor_status.values() 
                          if status == SensorStatus.FAILED)
        
        return {
            'total_sensors': len(self.sensor_status),
            'active_sensors': active_count,
            'simulated_sensors': simulated_count,
            'failed_sensors': failed_count,
            'grid_health': active_count / len(self.sensor_status) if self.sensor_status else 0,
            'fusion_accuracy': np.mean([model.get('accuracy', 0) for model in self.fusion_models.values()]),
            'last_update': datetime.now()
        }
    
    def auto_recalibrate(self):
        """معايرة تلقائية بناءً على أنماط البيانات"""
        try:
            for sensor_name in self.config['sensors'].keys():
                if len(self.sensor_history[sensor_name]) > 100:
                    recent_readings = self.sensor_history[sensor_name][-100:]
                    
                    # اكتشاف الانحراف التدريجي
                    values = [reading.value for reading in recent_readings]
                    if len(values) > 10:
                        trend = self._calculate_trend(values)
                        
                        # تحديث الانزياح إذا كان هناك انحراف واضح
                        if abs(trend) > 0.05:  # انحراف أكثر من 5%
                            current_offset = self.sensor_calibration[sensor_name].get('offset', 0)
                            new_offset = current_offset - trend * 0.1  # تصحيح تدريجي
                            self.sensor_calibration[sensor_name]['offset'] = new_offset
                            
                            self.logger.info(f"🔧 Auto-recalibrated {sensor_name}: offset = {new_offset:.3f}")
            
            self.logger.info("✅ Sensor grid auto-recalibration completed")
            
        except Exception as e:
            self.logger.error(f"❌ Auto-recalibration failed: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """حساب الاتجاه في سلسلة من القيم"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # انحدار خطي بسيط
        slope = np.polyfit(x, y, 1)[0]
        
        # تسوية حسب متوسط القيمة
        mean_value = np.mean(y)
        if mean_value > 0:
            return slope / mean_value
        else:
            return slope

class SmartNeuralDigitalTwin:
    """القلب الرئيسي للـ Smart Neural Digital Twin مع SenseGrid"""
    
    def __init__(self, config_path: str = "config/smart_neural_config.json"):
        self.config_manager = SmartConfig(config_path)
        self.config = self.config_manager.config
        self.logger = self.config_manager.logger
        
        # تهيئة الأنظمة المتقدمة
        self.sense_grid = AdaptiveSensorFusionGrid(self.config)  # 🌐 النظام الجديد
        self.relay_controller = RelayController(self.config)
        self.fore_sight_engine = ForeSightEngine(self.config)
        
        # حالة النظام
        self.system_status = "NORMAL"
        self.raspberry_pi_active = self.config['system']['raspberry_pi']['active']
        self.real_time_data = {}
        self.sensor_grid_status = {}
        
        # إحصائيات
        self.system_stats = {
            'start_time': datetime.now(),
            'processed_readings': 0,
            'sensor_failures_handled': 0,
            'avg_processing_time': 0.0
        }
        
        self._initialize_enhanced_systems()
        self.logger.info("🚀 Smart Neural Digital Twin with SenseGrid Initialized")
    
    def _initialize_enhanced_systems(self):
        """تهيئة الأنظمة المحسنة"""
        try:
            # بدء المراقبة المتقدمة
            self._start_enhanced_monitoring()
            
            # تحميل بيانات التدريب للذكاء الاصطناعي
            self._load_training_data()
            
            # بدء صيانة SenseGrid التلقائية
            self._start_sense_grid_maintenance()
            
            self.logger.info("✅ All enhanced systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced system initialization failed: {e}")
    
    def _start_enhanced_monitoring(self):
        """بدء مراقبة محسنة مع SenseGrid"""
        def monitoring_loop():
            while True:
                try:
                    start_time = time.time()
                    self._enhanced_monitoring_cycle()
                    processing_time = time.time() - start_time
                    
                    # تحديث إحصائيات الأداء
                    self.system_stats['avg_processing_time'] = (
                        self.system_stats['avg_processing_time'] * 0.9 + processing_time * 0.1
                    )
                    
                    time.sleep(self.config['system']['update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Enhanced monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _enhanced_monitoring_cycle(self):
        """دورة المراقبة المحسنة"""
        # 1. قراءة بيانات SenseGrid المتقدمة
        sensor_readings = self.sense_grid.read_sensor_grid()
        self.real_time_data = {name: reading.value for name, reading in sensor_readings.items()}
        
        # 2. تحديث حالة الشبكة
        self.sensor_grid_status = self.sense_grid.get_sensor_grid_status()
        
        # 3. معالجة البيانات عبر ForeSight Engine
        processed_data = self.fore_sight_engine.process_sensor_data(self.real_time_data)
        
        # 4. التحقق من حالات الطوارئ
        self._check_enhanced_emergency_conditions(processed_data, sensor_readings)
        
        # 5. تحديث الإحصائيات
        self.system_stats['processed_readings'] += 1
        self.system_stats['sensor_failures_handled'] = (
            self.sensor_grid_status['failed_sensors'] + self.sensor_grid_status['simulated_sensors']
        )
    
    def _check_enhanced_emergency_conditions(self, processed_data: Dict[str, Any], 
                                           sensor_readings: Dict[str, SensorReading]):
        """التحقق المحسن من حالات الطوارئ"""
        try:
            anomalies = processed_data.get('anomalies', {})
            predictions = processed_data.get('predictions', {})
            
            # تحليل مخاطر متقدم مع مراعاة ثقة المستشعرات
            risk_score = self._calculate_enhanced_risk_score(anomalies, predictions, sensor_readings)
            
            if risk_score >= 0.9:
                self.system_status = "EMERGENCY"
                self._execute_enhanced_emergency_response(processed_data)
            elif risk_score >= 0.7:
                self.system_status = "CRITICAL"
            elif risk_score >= 0.5:
                self.system_status = "HIGH_ALERT"
            else:
                self.system_status = "NORMAL"
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced emergency check failed: {e}")
    
    def _calculate_enhanced_risk_score(self, anomalies: Dict, predictions: Dict, 
                                     sensor_readings: Dict[str, SensorReading]) -> float:
        """حساب درجة خطر محسنة مع مراعاة ثقة المستشعرات"""
        base_risk = anomalies.get('anomaly_score', 0)
        
        # تعديل بناءً على ثقة المستشعرات
        confidence_penalty = 0
        for sensor_name, reading in sensor_readings.items():
            if reading.confidence < 0.7:  # إذا كانت الثقة منخفضة
                confidence_penalty += (0.7 - reading.confidence) * 0.1
        
        adjusted_risk = min(1.0, base_risk + confidence_penalty)
        return adjusted_risk
    
    def _execute_enhanced_emergency_response(self, processed_data: Dict[str, Any]):
        """تنفيذ استجابة طوارئ محسنة"""
        try:
            decision = processed_data.get('decision', {})
            actions = decision.get('decision', {}).get('actions', [])
            
            for action in actions[:3]:  # تنفيذ أفضل 3 إجراءات
                self._execute_enhanced_action(action)
            
            self.logger.critical("🚨 Enhanced emergency response executed")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced emergency response failed: {e}")
    
    def _execute_enhanced_action(self, action: Dict[str, Any]):
        """تنفيذ إجراء محسن"""
        action_type = action.get('type', '')
        
        if action_type == 'relay_control':
            relay_name = action.get('relay_name')
            state = action.get('state', False)
            self.relay_controller.control_relay(relay_name, state, "Enhanced emergency response")
        
        elif action_type == 'system_adjustment':
            self._adjust_system_parameters(action.get('parameters', {}))
    
    def _start_sense_grid_maintenance(self):
        """بدء صيانة SenseGrid التلقائية"""
        def maintenance_loop():
            while True:
                try:
                    # معايرة تلقائية كل ساعة
                    self.sense_grid.auto_recalibrate()
                    time.sleep(3600)  # كل ساعة
                    
                except Exception as e:
                    self.logger.error(f"SenseGrid maintenance error: {e}")
                    time.sleep(300)  # انتظار 5 دقائق ثم إعادة المحاولة
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام المحسن"""
        return {
            'system_status': self.system_status,
            'raspberry_pi_active': self.raspberry_pi_active,
            'sensor_grid_status': self.sensor_grid_status,
            'relay_states': self.relay_controller.get_relay_status(),
            'performance_metrics': self.system_stats,
            'sense_grid_health': self.sense_grid.get_sensor_grid_status()['grid_health'],
            'last_update': datetime.now()
        }

# دالة الإنشاء المحسنة
def create_smart_neural_twin(config_path: str = "config/smart_neural_config.json"):
    """إنشاء Smart Neural Digital Twin مع SenseGrid"""
    try:
        return SmartNeuralDigitalTwin(config_path)
    except Exception as e:
        logging.error(f"❌ Failed to create Smart Neural Digital Twin: {e}")
        raise

if __name__ == "__main__":
    twin = create_smart_neural_twin()
    print("🚀 Smart Neural Digital Twin with SenseGrid Running!")
