import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass
import math

class InterventionLevel(Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class InterventionType(Enum):
    SHUTDOWN = "shutdown"
    REDUCE_LOAD = "reduce_load"
    ALERT = "alert"
    NOTIFY = "notify"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    ACTIVATE_SAFETY = "activate_safety"

@dataclass
class InterventionDecision:
    level: InterventionLevel
    type: InterventionType
    confidence: float
    recommended_actions: List[str]
    expected_impact: str
    urgency: int
    timestamp: datetime

class AdvancedInterventionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = self._load_thresholds()
        self.intervention_history = []
        self.risk_patterns = {}
        self.context_factors = self._initialize_context_factors()
        self.setup_logging()
        
    def _load_thresholds(self) -> Dict[str, Dict]:
        """تحميل الحدود من الإعدادات"""
        return {
            'temperature': {'normal': 60, 'warning': 70, 'critical': 85, 'emergency': 90},
            'pressure': {'normal': 900, 'warning': 950, 'critical': 1000, 'emergency': 1100},
            'vibration': {'normal': 1.5, 'warning': 2.5, 'critical': 3.5, 'emergency': 5.0},
            'methane': {'normal': 50, 'warning': 100, 'critical': 200, 'emergency': 500},
            'h2s': {'normal': 5, 'warning': 10, 'critical': 20, 'emergency': 50},
            'flow': {'normal': 300, 'warning': 400, 'critical': 450, 'emergency': 500}
        }
    
    def _initialize_context_factors(self) -> Dict[str, float]:
        """تهيئة عوامل السياق"""
        return {
            'time_of_day': 1.0,  # وزن وقت اليوم
            'equipment_age': 1.2,  # وزن عمر المعدة
            'maintenance_history': 0.8,  # وزن تاريخ الصيانة
            'weather_conditions': 1.1,  # وزن حالة الطقس
            'human_presence': 1.3  # وزن وجود بشر
        }
    
    def setup_logging(self):
        """تهيئة التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def assess_risk(self, sensor_data: Dict[str, float], context: Optional[Dict] = None) -> Dict[str, InterventionLevel]:
        """تقييم المخاطر المتقدم مع السياق"""
        risks = {}
        
        for sensor, value in sensor_data.items():
            if sensor in self.thresholds:
                base_risk = self._calculate_base_risk(value, self.thresholds[sensor])
                
                # تطبيق عوامل السياق إذا موجودة
                if context:
                    contextual_risk = self._apply_context_factors(base_risk, context, sensor)
                    risks[sensor] = contextual_risk
                else:
                    risks[sensor] = base_risk
        
        return risks
    
    def _calculate_base_risk(self, value: float, thresholds: Dict) -> InterventionLevel:
        """حساب الخطر الأساسي"""
        if value >= thresholds['emergency']:
            return InterventionLevel.EMERGENCY
        elif value >= thresholds['critical']:
            return InterventionLevel.CRITICAL
        elif value >= thresholds['warning']:
            return InterventionLevel.WARNING
        elif value >= thresholds['normal']:
            return InterventionLevel.MONITOR
        else:
            return InterventionLevel.NORMAL
    
    def _apply_context_factors(self, base_risk: InterventionLevel, context: Dict, sensor: str) -> InterventionLevel:
        """تطبيق عوامل السياق على الخطر"""
        risk_value = self._risk_level_to_value(base_risk)
        
        # تطبيق عوامل السياق
        context_factor = 1.0
        for factor, weight in self.context_factors.items():
            if factor in context:
                context_factor *= weight
        
        # تعديل مستوى الخطر بناءً على السياق
        adjusted_risk_value = risk_value * context_factor
        return self._value_to_risk_level(adjusted_risk_value)
    
    def _risk_level_to_value(self, risk_level: InterventionLevel) -> float:
        """تحويل مستوى الخطر إلى قيمة رقمية"""
        risk_values = {
            InterventionLevel.NORMAL: 1.0,
            InterventionLevel.MONITOR: 2.0,
            InterventionLevel.WARNING: 3.0,
            InterventionLevel.CRITICAL: 4.0,
            InterventionLevel.EMERGENCY: 5.0
        }
        return risk_values.get(risk_level, 1.0)
    
    def _value_to_risk_level(self, risk_value: float) -> InterventionLevel:
        """تحويل القيمة الرقمية إلى مستوى خطر"""
        if risk_value >= 4.5:
            return InterventionLevel.EMERGENCY
        elif risk_value >= 3.5:
            return InterventionLevel.CRITICAL
        elif risk_value >= 2.5:
            return InterventionLevel.WARNING
        elif risk_value >= 1.5:
            return InterventionLevel.MONITOR
        else:
            return InterventionLevel.NORMAL
    
    def recommend_intervention(self, risks: Dict[str, InterventionLevel], 
                              sensor_data: Dict[str, float]) -> InterventionDecision:
        """توصية تدخل متقدمة"""
        # تحديد أعلى مستوى خطر
        max_risk_level = max(risks.values(), key=lambda x: self._risk_level_to_value(x))
        
        # تحديد نوع التدخل المناسب
        intervention_type = self._determine_intervention_type(max_risk_level, risks, sensor_data)
        
        # حساب الثقة في القرار
        confidence = self._calculate_confidence(risks, sensor_data)
        
        # توليد الإجراءات الموصى بها
        recommended_actions = self._generate_recommended_actions(risks, intervention_type)
        
        # تقدير الأثر المتوقع
        expected_impact = self._estimate_impact(intervention_type, risks)
        
        # تحديد مستوى الاستعجال
        urgency = self._determine_urgency(max_risk_level)
        
        return InterventionDecision(
            level=max_risk_level,
            type=intervention_type,
            confidence=confidence,
            recommended_actions=recommended_actions,
            expected_impact=expected_impact,
            urgency=urgency,
            timestamp=datetime.now()
        )
    
    def _determine_intervention_type(self, max_risk: InterventionLevel, 
                                   risks: Dict[str, InterventionLevel],
                                   sensor_data: Dict[str, float]) -> InterventionType:
        """تحديد نوع التدخل المناسب"""
        if max_risk == InterventionLevel.EMERGENCY:
            return InterventionType.SHUTDOWN
        
        elif max_risk == InterventionLevel.CRITICAL:
            # إذا كان هناك multiple critical risks → shutdown
            critical_count = sum(1 for r in risks.values() if r == InterventionLevel.CRITICAL)
            if critical_count >= 2:
                return InterventionType.SHUTDOWN
            return InterventionType.REDUCE_LOAD
        
        elif max_risk == InterventionLevel.WARNING:
            # تحليل إذا كان هناك trend تصاعدي
            if self._is_rising_trend(sensor_data):
                return InterventionType.ACTIVATE_SAFETY
            return InterventionType.ALERT
        
        else:
            return InterventionType.MONITOR
    
    def _is_rising_trend(self, sensor_data: Dict[str, float]) -> bool:
        """الكشف عن الاتجاه التصاعدي"""
        # في التنفيذ الحقيقي، سيتم تحليل البيانات التاريخية
        # هنا نعود قيمة افتراضية للتوضيح
        return any(value > self.thresholds[sensor]['warning'] * 0.8 
                  for sensor, value in sensor_data.items() 
                  if sensor in self.thresholds)
    
    def _calculate_confidence(self, risks: Dict[str, InterventionLevel], 
                            sensor_data: Dict[str, float]) -> float:
        """حساب ثقة القرار"""
        # عدد المخاطر الحرجة
        critical_risks = sum(1 for r in risks.values() 
                           if r in [InterventionLevel.CRITICAL, InterventionLevel.EMERGENCY])
        
        # تناسق البيانات
        data_consistency = self._check_data_consistency(sensor_data)
        
        # حساب الثقة النهائية
        base_confidence = 0.7 + (critical_risks * 0.1)
        return min(0.95, base_confidence * data_consistency)
    
    def _check_data_consistency(self, sensor_data: Dict[str, float]) -> float:
        """فحص تناسق البيانات"""
        # في التنفيذ الحقيقي، يتم فحص جودة واتساق البيانات
        return 0.9  # قيمة افتراضية
    
    def _generate_recommended_actions(self, risks: Dict[str, InterventionLevel],
                                    intervention_type: InterventionType) -> List[str]:
        """توليد الإجراءات الموصى بها"""
        actions = []
        
        for sensor, risk in risks.items():
            if risk in [InterventionLevel.CRITICAL, InterventionLevel.EMERGENCY]:
                actions.append(f"Immediate attention required for {sensor}")
            
            elif risk == InterventionLevel.WARNING:
                actions.append(f"Close monitoring needed for {sensor}")
        
        # إضافة إجراءات عامة بناءً على نوع التدخل
        if intervention_type == InterventionType.SHUTDOWN:
            actions.extend([
                "Initiate emergency shutdown procedure",
                "Notify all personnel",
                "Activate safety systems"
            ])
        elif intervention_type == InterventionType.REDUCE_LOAD:
            actions.extend([
                "Reduce operational load by 50%",
                "Increase monitoring frequency",
                "Prepare for possible shutdown"
            ])
        
        return actions
    
    def _estimate_impact(self, intervention_type: InterventionType,
                        risks: Dict[str, InterventionLevel]) -> str:
        """تقدير الأثر المتوقع للتدخل"""
        impact_levels = {
            InterventionType.SHUTDOWN: "High impact: Production stoppage, safety ensured",
            InterventionType.REDUCE_LOAD: "Medium impact: Reduced efficiency, maintained safety",
            InterventionType.ALERT: "Low impact: Operational with increased vigilance",
            InterventionType.NOTIFY: "Minimal impact: Information dissemination",
            InterventionType.MONITOR: "No immediate impact: Continuous observation"
        }
        return impact_levels.get(intervention_type, "Impact assessment pending")
    
    def _determine_urgency(self, risk_level: InterventionLevel) -> int:
        """تحديد مستوى الاستعجال"""
        urgency_map = {
            InterventionLevel.EMERGENCY: 5,
            InterventionLevel.CRITICAL: 4,
            InterventionLevel.WARNING: 3,
            InterventionLevel.MONITOR: 2,
            InterventionLevel.NORMAL: 1
        }
        return urgency_map.get(risk_level, 1)
    
    def execute_intervention(self, decision: InterventionDecision,
                           sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """تنفيذ التدخل"""
        try:
            # تسجيل التدخل
            self.log_intervention(decision, sensor_data)
            
            # محاكاة التنفيذ (في الواقع ستكون استدعاءات للأنظمة)
            execution_result = {
                'success': True,
                'intervention_id': f"int_{datetime.now().timestamp()}",
                'decision': decision,
                'execution_time': datetime.now(),
                'actions_taken': self._simulate_actions(decision.recommended_actions)
            }
            
            self.logger.info(f"Intervention executed: {execution_result}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Intervention execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _simulate_actions(self, actions: List[str]) -> List[Dict]:
        """محاكاة تنفيذ الإجراءات"""
        executed_actions = []
        for action in actions:
            executed_actions.append({
                'action': action,
                'status': 'completed',
                'timestamp': datetime.now(),
                'execution_time_ms': np.random.randint(100, 500)
            })
        return executed_actions
    
    def log_intervention(self, decision: InterventionDecision,
                        sensor_data: Dict[str, float]) -> None:
        """تسجيل التدخل في السجل"""
        log_entry = {
            'timestamp': datetime.now(),
            'decision': {
                'level': decision.level.value,
                'type': decision.type.value,
                'confidence': decision.confidence,
                'urgency': decision.urgency
            },
            'sensor_data': sensor_data,
            'recommended_actions': decision.recommended_actions,
            'expected_impact': decision.expected_impact
        }
        
        self.intervention_history.append(log_entry)
        self.logger.info(f"Intervention logged: {log_entry}")
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التدخل"""
        if not self.intervention_history:
            return {'total_interventions': 0}
        
        total = len(self.intervention_history)
        emergency_count = sum(1 for i in self.intervention_history 
                            if i['decision']['level'] == 'emergency')
        critical_count = sum(1 for i in self.intervention_history 
                           if i['decision']['level'] == 'critical')
        
        return {
            'total_interventions': total,
            'emergency_interventions': emergency_count,
            'critical_interventions': critical_count,
            'success_rate': 0.95,  # في الواقع يتم حسابها من النتائج
            'last_intervention': self.intervention_history[-1]['timestamp'] if total > 0 else None
        }

# دالة مساعدة للاستيراد
def create_intervention_engine(config: Dict[str, Any]) -> AdvancedInterventionEngine:
    return AdvancedInterventionEngine(config)
