import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from enum import Enum
import logging

class AnomalyType(Enum):
    POINT_ANOMALY = "point"       # 📍 شذوذ نقطي
    CONTEXTUAL_ANOMALY = "contextual" # 🎯 شذوذ سياقي  
    COLLECTIVE_ANOMALY = "collective" # 👥 شذوذ جماعي

class AnomalySeverity(Enum):
    LOW = "low"           # 📉 منخفض
    MEDIUM = "medium"     # ⚠️ متوسط  
    HIGH = "high"         # 🚨 عالي
    CRITICAL = "critical" # 🔥 حرج

@dataclass
class AnomalyReport:
    timestamp: datetime
    sensor_type: str
    value: float
    expected_range: Tuple[float, float]
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    explanation: str
    recommended_action: str

class AdvancedAnomalySystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.existing_models = self._load_existing_models()
        self.anomaly_history: List[AnomalyReport] = []
        self.performance_metrics = {}
        self.setup_logging()
        
    def _load_existing_models(self) -> Dict[str, Any]:
        """تحميل النماذج الموجودة من ملف .pkl"""
        try:
            pkl_path = self.config.get('model_path', 'advanced_prediction_models.pkl')
            if Path(pkl_path).exists():
                models = joblib.load(pkl_path)
                self.logger.info(f"✅ تم تحميل {len(models)} نماذج مدربة مسبقاً")
                return models
            else:
                self.logger.warning("⚠️ ملف النماذج غير موجود - سيتم استخدام القواعد الأساسية")
                return {}
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل النماذج: {e}")
            return {}
    
    def setup_logging(self):
        """تهيئة نظام التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, sensor_data: Dict[str, float], 
                        context: Optional[Dict] = None) -> List[AnomalyReport]:
        """كشف الشذوذ المتقدم مع السياق"""
        anomalies = []
        
        for sensor_type, value in sensor_data.items():
            anomaly_report = self._analyze_sensor(sensor_type, value, context)
            if anomaly_report:
                anomalies.append(anomaly_report)
                self._log_anomaly(anomaly_report)
        
        return anomalies
    
    def _analyze_sensor(self, sensor_type: str, value: float, 
                       context: Optional[Dict]) -> Optional[AnomalyReport]:
        """تحليل متقدم لمستشعر individual"""
        
        # 1. الكشف باستخدام النماذج المدربة (إذا موجودة)
        if sensor_type in self.existing_models:
            model_anomaly = self._detect_with_model(sensor_type, value)
            if model_anomaly:
                return model_anomaly
        
        # 2. الكشف باستخدام القواعد (fallback)
        return self._detect_with_rules(sensor_type, value, context)
    
    def _detect_with_model(self, sensor_type: str, value: float) -> Optional[AnomalyReport]:
        """الكشف باستخدام نماذج ML"""
        try:
            model = self.existing_models[sensor_type]['model']
            scaler = self.existing_models[sensor_type]['scaler']
            
            # تطبيق المطبيع والتنبؤ
            value_scaled = scaler.transform([[value]])
            is_anomaly = model.predict(value_scaled)[0] == -1
            
            if is_anomaly:
                return AnomalyReport(
                    timestamp=datetime.now(),
                    sensor_type=sensor_type,
                    value=value,
                    expected_range=self._get_expected_range(sensor_type),
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=self._calculate_severity(sensor_type, value),
                    confidence=0.85,
                    explanation=f"Anomaly detected by ML model",
                    recommended_action="Check sensor and review recent readings"
                )
        except Exception as e:
            self.logger.error(f"Model detection failed for {sensor_type}: {e}")
        
        return None
    
    def _detect_with_rules(self, sensor_type: str, value: float, 
                          context: Optional[Dict]) -> Optional[AnomalyReport]:
        """الكشف باستخدام قواعد ذكية"""
        # الحدود الآمنة من الإعدادات
        thresholds = self.config.get('safety_thresholds', {}).get(sensor_type, {})
        
        if not thresholds:
            return None
        
        # التحقق من الحدود مع مراعاة السياق
        severity = None
        if value >= thresholds.get('emergency', float('inf')):
            severity = AnomalySeverity.CRITICAL
        elif value >= thresholds.get('critical', float('inf')):
            severity = AnomalySeverity.HIGH
        elif value >= thresholds.get('warning', float('inf')):
            severity = AnomalySeverity.MEDIUM
        
        if severity:
            return AnomalyReport(
                timestamp=datetime.now(),
                sensor_type=sensor_type,
                value=value,
                expected_range=(0, thresholds.get('warning', 100)),
                anomaly_type=AnomalyType.POINT_ANOMALY,
                severity=severity,
                confidence=0.9,
                explanation=f"Value exceeded {severity.value} threshold",
                recommended_action=self._get_action_for_severity(severity, sensor_type)
            )
        
        return None
    
    def _get_expected_range(self, sensor_type: str) -> Tuple[float, float]:
        """الحصول على المدى المتوقع للقيم الطبيعية"""
        thresholds = self.config.get('safety_thresholds', {}).get(sensor_type, {})
        return (0, thresholds.get('warning', 100))
    
    def _calculate_severity(self, sensor_type: str, value: float) -> AnomalySeverity:
        """حساب خطورة الشذوذ"""
        thresholds = self.config.get('safety_thresholds', {}).get(sensor_type, {})
        
        if value >= thresholds.get('emergency', float('inf')):
            return AnomalySeverity.CRITICAL
        elif value >= thresholds.get('critical', float('inf')):
            return AnomalySeverity.HIGH
        elif value >= thresholds.get('warning', float('inf')):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _get_action_for_severity(self, severity: AnomalySeverity, sensor_type: str) -> str:
        """الحصول على إجراء موصى به"""
        actions = {
            AnomalySeverity.CRITICAL: f"Immediate shutdown of {sensor_type} system. Evacuate area if necessary.",
            AnomalySeverity.HIGH: f"Emergency protocol for {sensor_type}. Prepare for shutdown.",
            AnomalySeverity.MEDIUM: f"Close monitoring of {sensor_type}. Schedule maintenance check.",
            AnomalySeverity.LOW: f"Watch {sensor_type} trends. No immediate action required."
        }
        return actions.get(severity, "Monitor situation")
    
    def _log_anomaly(self, report: AnomalyReport) -> None:
        """تسجيل الشذوذ في السجل"""
        self.anomaly_history.append(report)
        self.logger.warning(
            f"🚨 Anomaly detected: {report.sensor_type} = {report.value} "
            f"({report.severity.value} - {report.explanation})"
        )
    
    def get_anomaly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """الحصول على إحصائيات الشذوذ"""
        recent_anomalies = [
            a for a in self.anomaly_history 
            if (datetime.now() - a.timestamp).total_seconds() <= hours * 3600
        ]
        
        if not recent_anomalies:
            return {'total_anomalies': 0}
        
        severity_count = {sev.value: 0 for sev in AnomalySeverity}
        for anomaly in recent_anomalies:
            severity_count[anomaly.severity.value] += 1
        
        return {
            'total_anomalies': len(recent_anomalies),
            'by_severity': severity_count,
            'by_sensor': pd.Series([a.sensor_type for a in recent_anomalies]).value_counts().to_dict(),
            'time_period_hours': hours
        }
    
    def adapt_thresholds(self, new_data: Dict[str, List[float]]) -> None:
        """تكييف الحدود بناءً على البيانات الجديدة"""
        for sensor_type, values in new_data.items():
            if values:
                current_threshold = self.config.get('safety_thresholds', {}).get(sensor_type, {}).get('warning', 100)
                new_avg = np.mean(values)
                
                # تكييف تدريجي (10% من الفرق)
                adjustment = 0.1 * (new_avg - current_threshold)
                new_threshold = current_threshold + adjustment
                
                # تحديث الإعدادات
                if sensor_type not in self.config['safety_thresholds']:
                    self.config['safety_thresholds'][sensor_type] = {}
                
                self.config['safety_thresholds'][sensor_type]['warning'] = new_threshold
                self.logger.info(f"Adjusted {sensor_type} threshold to {new_threshold:.2f}")

# دالة مساعدة
def create_anomaly_system(config: Dict[str, Any]) -> AdvancedAnomalySystem:
    return AdvancedAnomalySystem(config)
