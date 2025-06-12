import numpy as np
import pandas as pd
from typing import Dict, List, Any

class InterventionEngine:
    """محرك محاكاة التدخلات الذكي"""
    
    def __init__(self):
        self.intervention_effects = {
            'activate_cooling': {
                'Temperature (°C)': -15,  # تقليل درجة الحرارة
                'Pressure (psi)': -5,     # تقليل الضغط قليلاً
                'description': 'تفعيل نظام التبريد الطارئ',
                'duration_hours': 4,
                'effectiveness': 0.9
            },
            'reduce_pressure': {
                'Pressure (psi)': -20,    # تقليل الضغط بشكل كبير
                'Temperature (°C)': -3,   # تقليل درجة الحرارة قليلاً
                'description': 'تقليل الضغط في الأنابيب',
                'duration_hours': 6,
                'effectiveness': 0.95
            },
            'increase_ventilation': {
                'Methane (CH₄ ppm)': -8,  # تقليل الميثان
                'H₂S (ppm)': -2,          # تقليل كبريتيد الهيدروجين
                'Temperature (°C)': -2,   # تقليل درجة الحرارة قليلاً
                'description': 'زيادة التهوية في المنطقة',
                'duration_hours': 3,
                'effectiveness': 0.85
            },
            'emergency_shutdown': {
                'Temperature (°C)': -25,  # تقليل كبير في درجة الحرارة
                'Pressure (psi)': -30,    # تقليل كبير في الضغط
                'Methane (CH₄ ppm)': -10, # تقليل الميثان
                'Vibration (g)': -0.3,    # تقليل الاهتزاز
                'description': 'إيقاف طارئ للعمليات',
                'duration_hours': 8,
                'effectiveness': 0.98
            },
            'stabilize_system': {
                'Vibration (g)': -0.2,    # تقليل الاهتزاز
                'Pressure (psi)': -8,     # استقرار الضغط
                'description': 'تثبيت النظام وتقليل الاهتزاز',
                'duration_hours': 5,
                'effectiveness': 0.8
            }
        }
    
    def apply_intervention(self, predictions: Dict, intervention_type: str) -> Dict:
        """تطبيق التدخل على التنبؤات المستقبلية"""
        if intervention_type not in self.intervention_effects:
            return predictions
        
        intervention = self.intervention_effects[intervention_type]
        modified_predictions = {}
        
        for sensor, pred_list in predictions.items():
            modified_predictions[sensor] = []
            
            for pred in pred_list:
                modified_pred = pred.copy()
                
                # تطبيق التأثير إذا كان ضمن مدة التدخل
                if pred['hours_ahead'] <= intervention['duration_hours']:
                    if sensor in intervention:
                        # حساب التأثير مع مراعاة الفعالية والتدهور مع الوقت
                        time_factor = 1 - (pred['hours_ahead'] - 1) / intervention['duration_hours']
                        effect = intervention[sensor] * intervention['effectiveness'] * time_factor
                        
                        # تطبيق التأثير
                        modified_pred['value'] = max(0, pred['value'] + effect)
                        modified_pred['intervention_applied'] = True
                        modified_pred['intervention_effect'] = effect
                    else:
                        modified_pred['intervention_applied'] = False
                        modified_pred['intervention_effect'] = 0
                else:
                    modified_pred['intervention_applied'] = False
                    modified_pred['intervention_effect'] = 0
                
                modified_predictions[sensor].append(modified_pred)
        
        return modified_predictions
    
    def calculate_risk_reduction(self, original_predictions: Dict, modified_predictions: Dict) -> Dict:
        """حساب تقليل المخاطر بعد التدخل"""
        risk_reduction = {}
        
        for sensor in original_predictions:
            original_avg = np.mean([p['value'] for p in original_predictions[sensor]])
            modified_avg = np.mean([p['value'] for p in modified_predictions[sensor]])
            
            # حساب نسبة التحسن
            if original_avg > 0:
                improvement = ((original_avg - modified_avg) / original_avg) * 100
            else:
                improvement = 0
            
            risk_reduction[sensor] = {
                'original_avg': original_avg,
                'modified_avg': modified_avg,
                'improvement_percent': max(0, improvement)
            }
        
        return risk_reduction
    
    def get_intervention_recommendations(self, current_data: Dict, risk_analysis: Dict) -> List[Dict]:
        """الحصول على توصيات التدخل بناءً على البيانات الحالية"""
        recommendations = []
        
        # توصيات بناءً على درجة الحرارة
        if current_data.get('temperature', 0) > 85:
            recommendations.append({
                'type': 'activate_cooling',
                'priority': 'high',
                'reason': 'درجة حرارة عالية تتطلب تبريد فوري'
            })
        
        # توصيات بناءً على الضغط
        if current_data.get('pressure', 0) > 220:
            recommendations.append({
                'type': 'reduce_pressure',
                'priority': 'high',
                'reason': 'ضغط عالي يتطلب تخفيف فوري'
            })
        
        # توصيات بناءً على الغازات
        if current_data.get('methane', 0) > 10:
            recommendations.append({
                'type': 'increase_ventilation',
                'priority': 'medium',
                'reason': 'مستوى ميثان مرتفع يتطلب تهوية'
            })
        
        # توصيات بناءً على الاهتزاز
        if current_data.get('vibration', 0) > 0.8:
            recommendations.append({
                'type': 'stabilize_system',
                'priority': 'medium',
                'reason': 'اهتزاز عالي يتطلب تثبيت النظام'
            })
        
        # توصية الإيقاف الطارئ في حالة المخاطر العالية
        if risk_analysis.get('total_risk', 0) > 0.8:
            recommendations.append({
                'type': 'emergency_shutdown',
                'priority': 'critical',
                'reason': 'مستوى مخاطر حرج يتطلب إيقاف طارئ'
            })
        
        return recommendations
    
    def get_intervention_info(self, intervention_type: str) -> Dict:
        """الحصول على معلومات التدخل"""
        if intervention_type in self.intervention_effects:
            return self.intervention_effects[intervention_type]
        return {}

# Test the intervention engine
if __name__ == "__main__":
    engine = InterventionEngine()
    
    # Test data
    test_predictions = {
        'Temperature (°C)': [
            {'time': pd.Timestamp.now(), 'value': 95, 'hours_ahead': 1},
            {'time': pd.Timestamp.now(), 'value': 98, 'hours_ahead': 2}
        ]
    }
    
    modified = engine.apply_intervention(test_predictions, 'activate_cooling')
    print("Original:", test_predictions)
    print("Modified:", modified)

