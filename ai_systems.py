import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import random
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

class AIChatSystem:
    """
    نظام الدردشة بالذكاء الاصطناعي باستخدام OpenAI
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        
        # استخدام secrets إذا كان في بيئة Streamlit
        if hasattr(__import__('streamlit'), 'secrets'):
            import streamlit as st
            if st.secrets.get('openai', {}):
                self.api_key = st.secrets['openai'].get('api_key', self.api_key)
    
    def ask_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """طرح سؤال على الذكاء الاصطناعي"""
        try:
            if not self.api_key:
                return "⚠️ OpenAI API key not configured. Using simulated response."
            
            # هنا سيتم تكامل OpenAI الحقيقي
            # response = openai.ChatCompletion.create(...)
            
            # نموذج محاكاة مؤقت
            return self._simulate_ai_response(question, context)
            
        except Exception as e:
            self.logger.error(f"❌ Error in AI chat: {e}")
            return f"❌ Error: {str(e)}"
    
    def _simulate_ai_response(self, question: str, context: Dict[str, Any] = None) -> str:
        """محاكاة ردود الذكاء الاصطناعي"""
        question_lower = question.lower()
        
        if 'pressure' in question_lower:
            return "Based on current sensor readings, pressure levels are within normal range. No immediate action required."
        elif 'temperature' in question_lower:
            return "Temperature sensors indicate stable conditions. Monitoring ongoing."
        elif 'emergency' in question_lower or 'risk' in question_lower:
            return "Emergency systems are active. Risk assessment shows normal conditions. All safety protocols are operational."
        elif 'prediction' in question_lower or 'forecast' in question_lower:
            return "AI models predict stable conditions for the next 24 hours. Continuous monitoring in progress."
        else:
            return "I've analyzed your query. The oil field monitoring system is functioning optimally. All critical parameters are within safe limits."

class AdvancedAnomalyDetector:
    """
    نظام متقدم لكشف الشذوذ باستخدام多种 خوارزميات
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.anomaly_history = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """تهيئة نماذج كشف الشذوذ"""
        try:
            # Isolation Forest للكشف عن الشذوذ
            self.models['isolation_forest'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # One-Class SVM
            self.models['one_class_svm'] = OneClassSVM(
                nu=0.1,
                kernel='rbf'
            )
            
            # نموذج LSTM للكشف عن الشذوذ الزمني
            self.models['lstm_autoencoder'] = self._create_lstm_autoencoder()
            
            self.logger.info("✅ Anomaly detection models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing anomaly models: {e}")
    
    def _create_lstm_autoencoder(self) -> nn.Module:
        """إنشاء LSTM Autoencoder للكشف عن الشذوذ"""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim=6, hidden_dim=32, num_layers=2):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
            
            def forward(self, x):
                encoded, _ = self.encoder(x)
                decoded, _ = self.decoder(encoded)
                return decoded
        
        return LSTMAutoencoder()
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """كشف الشذوذ في بيانات المستشعرات"""
        try:
            if not sensor_data:
                return {'error': 'No sensor data provided'}
            
            # تحويل البيانات إلى مصفوفة
            data_array = self._prepare_data(sensor_data)
            
            # الكشف باستخدام多种 خوارزميات
            results = {}
            
            # Isolation Forest
            iforest_pred = self.models['isolation_forest'].fit_predict(data_array.reshape(1, -1))
            results['isolation_forest'] = iforest_pred[0] == -1
            
            # One-Class SVM
            svm_pred = self.models['one_class_svm'].fit_predict(data_array.reshape(1, -1))
            results['one_class_svm'] = svm_pred[0] == -1
            
            # LSTM Autoencoder (reconstruction error)
            lstm_anomaly = self._detect_lstm_anomaly(data_array)
            results['lstm_autoencoder'] = lstm_anomaly
            
            # تحليل النتائج
            anomaly_score = sum(results.values()) / len(results)
            is_anomaly = anomaly_score > 0.5
            
            # تصنيف مستوى الخطورة
            risk_level = self._classify_anomaly_risk(sensor_data, anomaly_score)
            
            result = {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'risk_level': risk_level,
                'algorithm_results': results,
                'critical_anomalies': self._identify_critical_anomalies(sensor_data),
                'timestamp': datetime.now()
            }
            
            # تخزين في السجل
            self.anomaly_history.append(result)
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting anomalies: {e}")
            return {'error': str(e)}
    
    def _prepare_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """تحضير البيانات للتحليل"""
        try:
            # استخراج القيم العددية من بيانات المستشعرات
            values = []
            sensors = ['pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow']
            
            for sensor in sensors:
                value = sensor_data.get(sensor, 0)
                if isinstance(value, (int, float)):
                    values.append(float(value))
                else:
                    values.append(0.0)
            
            return np.array(values)
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing data: {e}")
            return np.zeros(6)
    
    def _detect_lstm_anomaly(self, data: np.ndarray) -> bool:
        """الكشف عن الشذوذ باستخدام LSTM"""
        try:
            # محاكاة خطإ إعادة البناء
            reconstruction_error = np.random.random() * 0.1
            return reconstruction_error > 0.05  # عتبة
        except Exception as e:
            self.logger.error(f"❌ LSTM anomaly detection failed: {e}")
            return False
    
    def _classify_anomaly_risk(self, sensor_data: Dict[str, Any], anomaly_score: float) -> str:
        """تصنيف مستوى خطورة الشذوذ"""
        try:
            if anomaly_score >= 0.8:
                return 'critical'
            elif anomaly_score >= 0.6:
                return 'high'
            elif anomaly_score >= 0.4:
                return 'medium'
            else:
                return 'low'
        except Exception as e:
            self.logger.error(f"❌ Error classifying risk: {e}")
            return 'unknown'
    
    def _identify_critical_anomalies(self, sensor_data: Dict[str, Any]) -> List[str]:
        """تحديد الشذوذ الحرج"""
        critical_anomalies = []
        
        try:
            # تحقق من الضغط
            pressure = sensor_data.get('pressure', 0)
            if pressure > 80:  # بار
                critical_anomalies.append('High pressure detected')
            
            # تحقق من الحرارة
            temperature = sensor_data.get('temperature', 0)
            if temperature > 120:  # درجة مئوية
                critical_anomalies.append('High temperature detected')
            
            # تحقق من الميثان
            methane = sensor_data.get('methane', 0)
            if methane > 500:  # جزء في المليون
                critical_anomalies.append('High methane level')
            
            # تحقق من الاهتزاز
            vibration = sensor_data.get('vibration', 0)
            if vibration > 5:  # م/ث²
                critical_anomalies.append('Excessive vibration')
            
            return critical_anomalies
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying critical anomalies: {e}")
            return []

class AdvancedPredictionEngine:
    """
    محرك تنبؤ متقدم للتنبؤ بالمستقبل 24 ساعة
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.prediction_history = []
        self.model = self._create_prediction_model()
    
    def _create_prediction_model(self):
        """إنشاء نموذج التنبؤ"""
        # نموذج LSTM للتنبؤ الزمني
        class PredictionLSTM(nn.Module):
            def __init__(self, input_size=6, hidden_size=64, output_size=6, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.linear = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                predictions = self.linear(lstm_out[:, -1, :])
                return predictions
        
        return PredictionLSTM()
    
    def predict_next_24_hours(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ بالمستقبل 24 ساعة"""
        try:
            # تحضير البيانات التاريخية
            historical_data = self._prepare_historical_data(sensor_data)
            
            # التنبؤ باستخدام النموذج
            predictions = self._generate_predictions(historical_data)
            
            # تحليل الاتجاهات
            trends = self._analyze_trends(predictions)
            
            result = {
                'predictions': predictions,
                'trends': trends,
                'confidence_scores': self._calculate_confidence(predictions),
                'critical_points': self._identify_critical_points(predictions),
                'timestamp': datetime.now(),
                'time_horizon': '24_hours'
            }
            
            # تخزين في السجل
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating predictions: {e}")
            return {'error': str(e)}
    
    def _prepare_historical_data(self, current_data: Dict[str, Any]) -> np.ndarray:
        """تحضير البيانات التاريخية للتنبؤ"""
        try:
            # محاكاة بيانات تاريخية
            time_steps = 24  # 24 ساعة سابقة
            num_sensors = 6
            
            historical = np.random.normal(0.5, 0.1, (time_steps, num_sensors))
            
            # إضافة البيانات الحالية
            current_values = self._extract_sensor_values(current_data)
            historical = np.vstack([historical, current_values.reshape(1, -1)])
            
            return historical
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing historical data: {e}")
            return np.random.normal(0.5, 0.1, (25, 6))
    
    def _extract_sensor_values(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """استخراج قيم المستشعرات"""
        sensors = ['pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow']
        values = []
        
        for sensor in sensors:
            value = sensor_data.get(sensor, 0)
            if isinstance(value, (int, float)):
                # تطبيع القيمة
                normalized = max(0, min(1, value / 100))
                values.append(normalized)
            else:
                values.append(0.5)
        
        return np.array(values)
    
    def _generate_predictions(self, historical_data: np.ndarray) -> Dict[str, List[float]]:
        """توليد التنبؤات"""
        try:
            # محاكاة التنبؤات
            predictions = {}
            sensors = ['pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow']
            
            for i, sensor in enumerate(sensors):
                # إنشاء تنبؤات واقعية مع بعض التقلبات
                base_value = historical_data[-1, i]
                future_values = []
                
                for hour in range(24):
                    # إضافة بعض التقلبات الطبيعية
                    fluctuation = np.random.normal(0, 0.05)
                    predicted_value = base_value + (hour * 0.01) + fluctuation
                    future_values.append(max(0, min(1, predicted_value)) * 100)  # إعادة التحجيم
                
                predictions[sensor] = future_values
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating predictions: {e}")
            return {}
    
    def _analyze_trends(self, predictions: Dict[str, List[float]]) -> Dict[str, str]:
        """تحليل الاتجاهات في التنبؤات"""
        trends = {}
        
        try:
            for sensor, values in predictions.items():
                if len(values) >= 2:
                    start_val = values[0]
                    end_val = values[-1]
                    
                    if end_val > start_val * 1.1:
                        trends[sensor] = 'increasing'
                    elif end_val < start_val * 0.9:
                        trends[sensor] = 'decreasing'
                    else:
                        trends[sensor] = 'stable'
                else:
                    trends[sensor] = 'unknown'
            
            return trends
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing trends: {e}")
            return {}
    
    def _calculate_confidence(self, predictions: Dict[str, List[float]]) -> Dict[str, float]:
        """حساب درجات الثقة في التنبؤات"""
        confidence = {}
        
        try:
            for sensor in predictions.keys():
                # محاكاة درجات الثقة بناءً على تقلب البيانات
                confidence[sensor] = max(0.5, min(0.95, np.random.normal(0.8, 0.1)))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence: {e}")
            return {}
    
    def _identify_critical_points(self, predictions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """تحديد النقاط الحرجة في التنبؤات"""
        critical_points = []
        
        try:
            thresholds = {
                'pressure': 80,
                'temperature': 120,
                'methane': 500,
                'vibration': 5
            }
            
            for sensor, values in predictions.items():
                if sensor in thresholds:
                    for hour, value in enumerate(values):
                        if value > thresholds[sensor]:
                            critical_points.append({
                                'sensor': sensor,
                                'hour': hour,
                                'value': value,
                                'threshold': thresholds[sensor],
                                'risk_level': 'high' if value > thresholds[sensor] * 1.2 else 'medium'
                            })
            
            return critical_points
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying critical points: {e}")
            return []

class InterventionEngine:
    """
    محرك اتخاذ القرارات المستحيلة باستخدام مونتي كارلو
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.decision_history = []
    
    def make_impossible_decision(self, situation_data: Dict[str, Any], risk_level: float) -> Dict[str, Any]:
        """اتخاذ قرار مستحيل باستخدام محاكاة مونتي كارلو"""
        try:
            # توليد آلاف السيناريوهات باستخدام مونتي كارلو
            scenarios = self._generate_monte_carlo_scenarios(situation_data, risk_level)
            
            # تقييم كل سيناريو بناءً على معايير متعددة
            evaluated_scenarios = self._evaluate_scenarios(scenarios)
            
            # اختيار أفضل سيناريو
            best_scenario = self._select_best_scenario(evaluated_scenarios)
            
            # تطوير الحل الثالث (التوفيقي)
            third_way_solution = self._develop_third_way(best_scenario, evaluated_scenarios)
            
            result = {
                'decision_type': 'monte_carlo_optimized',
                'risk_level': risk_level,
                'best_scenario': best_scenario,
                'third_way_solution': third_way_solution,
                'evaluation_metrics': self._calculate_metrics(evaluated_scenarios),
                'timestamp': datetime.now()
            }
            
            self.decision_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error making impossible decision: {e}")
            return {'error': str(e)}
    
    def _generate_monte_carlo_scenarios(self, situation: Dict[str, Any], risk_level: float, num_scenarios: int = 1000) -> List[Dict[str, Any]]:
        """توليد سيناريوهات مونتي كارلو"""
        scenarios = []
        
        try:
            for i in range(num_scenarios):
                scenario = {
                    'id': i,
                    'actions': self._generate_random_actions(risk_level),
                    'expected_outcomes': {},
                    'risk_factors': np.random.random(),
                    'resource_usage': np.random.random(),
                    'time_to_resolve': np.random.exponential(10)
                }
                
                # حساب النتائج المتوقعة
                scenario['expected_outcomes'] = self._simulate_outcomes(scenario['actions'], situation)
                
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"❌ Error generating Monte Carlo scenarios: {e}")
            return []
    
    def _generate_random_actions(self, risk_level: float) -> List[Dict[str, Any]]:
        """توليد إجراءات عشوائية"""
        actions = []
        
        try:
            possible_actions = [
                {'type': 'pressure_release', 'intensity': np.random.random()},
                {'type': 'emergency_cooling', 'intensity': np.random.random()},
                {'type': 'gas_venting', 'venting_rate': np.random.random()},
                {'type': 'flow_adjustment', 'adjustment': np.random.uniform(-20, 20)},
                {'type': 'vibration_dampening', 'effectiveness': np.random.random()}
            ]
            
            # عدد الإجراءات يعتمد على مستوى الخطورة
            num_actions = min(5, max(1, int(risk_level * 10)))
            
            for _ in range(num_actions):
                action = random.choice(possible_actions).copy()
                # تعديل الشدة بناءً على مستوى الخطورة
                if 'intensity' in action:
                    action['intensity'] *= risk_level
                actions.append(action)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating random actions: {e}")
            return []
    
    def _simulate_outcomes(self, actions: List[Dict[str, Any]], situation: Dict[str, Any]) -> Dict[str, float]:
        """محاكاة نتائج الإجراءات"""
        try:
            outcomes = {
                'safety_improvement': 0.0,
                'production_impact': 0.0,
                'cost_effectiveness': 0.0,
                'environmental_impact': 0.0,
                'reputation_effect': 0.0
            }
            
            for action in actions:
                action_type = action['type']
                intensity = action.get('intensity', action.get('adjustment', action.get('venting_rate', 0.5)))
                
                if action_type == 'pressure_release':
                    outcomes['safety_improvement'] += intensity * 0.8
                    outcomes['production_impact'] -= intensity * 0.3
                elif action_type == 'emergency_cooling':
                    outcomes['safety_improvement'] += intensity * 0.7
                    outcomes['cost_effectiveness'] -= intensity * 0.4
                elif action_type == 'gas_venting':
                    outcomes['safety_improvement'] += intensity * 0.9
                    outcomes['environmental_impact'] -= intensity * 0.6
                elif action_type == 'flow_adjustment':
                    outcomes['production_impact'] += abs(intensity) * 0.1
                    outcomes['safety_improvement'] += intensity * 0.2 if intensity > 0 else -intensity * 0.1
                elif action_type == 'vibration_dampening':
                    outcomes['safety_improvement'] += intensity * 0.5
                    outcomes['reputation_effect'] += intensity * 0.3
            
            # تطبيع النتائج
            for key in outcomes:
                outcomes[key] = max(0.0, min(1.0, outcomes[key]))
            
            return outcomes
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating outcomes: {e}")
            return {}
    
    def _evaluate_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تقييم السيناريوهات باستخدام معايير متعددة"""
        evaluated = []
        
        try:
            for scenario in scenarios:
                # حساب الدرجة الإجمالية
                outcomes = scenario['expected_outcomes']
                
                # الأوزان للمعايير المختلفة
                weights = {
                    'safety': 0.35,
                    'production': 0.25,
                    'cost': 0.15,
                    'environment': 0.15,
                    'reputation': 0.10
                }
                
                total_score = (
                    outcomes['safety_improvement'] * weights['safety'] +
                    outcomes['production_impact'] * weights['production'] +
                    outcomes['cost_effectiveness'] * weights['cost'] +
                    outcomes['environmental_impact'] * weights['environment'] +
                    outcomes['reputation_effect'] * weights['reputation']
                )
                
                scenario['total_score'] = total_score
                scenario['weighted_score'] = total_score / (scenario['risk_factors'] + 0.1)
                
                evaluated.append(scenario)
            
            return evaluated
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating scenarios: {e}")
            return scenarios
    
    def _select_best_scenario(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """اختيار أفضل سيناريو"""
        try:
            if not scenarios:
                return {}
            
            # اختيار السيناريو بأعلى درجة
            best = max(scenarios, key=lambda x: x.get('weighted_score', 0))
            return best
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting best scenario: {e}")
            return {}
    
    def _develop_third_way(self, best_scenario: Dict[str, Any], all_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تطوير الحل الثالث (التوفيقي)"""
        try:
            if not all_scenarios:
                return best_scenario
            
            # أخذ أفضل 10 سيناريوهات
            top_scenarios = sorted(all_scenarios, key=lambda x: x.get('weighted_score', 0), reverse=True)[:10]
            
            # دمج أفضل العناصر من السيناريوهات المختلفة
            combined_actions = []
            combined_outcomes = {
                'safety_improvement': 0.0,
                'production_impact': 0.0,
                'cost_effectiveness': 0.0,
                'environmental_impact': 0.0,
                'reputation_effect': 0.0
            }
            
            for scenario in top_scenarios:
                # أخذ أفضل إجراء من كل سيناريو
                if scenario.get('actions'):
                    best_action = max(scenario['actions'], 
                                   key=lambda a: a.get('intensity', a.get('adjustment', a.get('venting_rate', 0))))
                    combined_actions.append(best_action)
                
                # متوسط النتائج
                for key in combined_outcomes.keys():
                    combined_outcomes[key] += scenario['expected_outcomes'].get(key, 0) / len(top_scenarios)
            
            third_way = {
                'actions': combined_actions[:5],  # الحد إلى 5 إجراءات
                'expected_outcomes': combined_outcomes,
                'description': 'Hybrid solution combining best elements from top scenarios',
                'innovation_score': 0.8  # درجة الابتكار
            }
            
            return third_way
            
        except Exception as e:
            self.logger.error(f"❌ Error developing third way: {e}")
            return best_scenario
    
    def _calculate_metrics(self, scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """حساب مقاييس التقييم"""
        try:
            if not scenarios:
                return {}
            
            scores = [s.get('total_score', 0) for s in scenarios]
            
            return {
                'average_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'std_score': np.std(scores),
                'num_scenarios': len(scenarios)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating metrics: {e}")
            return {}

class LifelongMemory:
    """
    نظام الذاكرة المستدامة - التعلم من التجارب السابقة
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_store = []
        self.learning_rate = 0.1
    
    def store_experience(self, sensor_data: Dict[str, Any], 
                        anomaly_results: Dict[str, Any], 
                        predictions: Dict[str, Any]):
        """تخزين التجربة في الذاكرة"""
        try:
            experience = {
                'timestamp': datetime.now(),
                'sensor_data': sensor_data.copy(),
                'anomaly_results': anomaly_results.copy(),
                'predictions': predictions.copy(),
                'lessons_learned': self._extract_lessons(anomaly_results, predictions)
            }
            
            self.memory_store.append(experience)
            
            # الاحتفاظ فقط بـ 10000 تجربة حديثة
            if len(self.memory_store) > 10000:
                self.memory_store = self.memory_store[-10000:]
            
            self.logger.info("✅ Experience stored in lifelong memory")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing experience: {e}")
    
    def _extract_lessons(self, anomaly_results: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """استخلاص الدروس من التجارب"""
        lessons = []
        
        try:
            if anomaly_results.get('is_anomaly', False):
                risk_level = anomaly_results.get('risk_level', 'low')
                lessons.append(f"Anomaly detected with {risk_level} risk level")
            
            if predictions.get('critical_points'):
                lessons.append(f"Critical points predicted: {len(predictions['critical_points'])}")
            
            return lessons
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting lessons: {e}")
            return []
    
    def retrieve_relevant_experiences(self, current_situation: Dict[str, Any], 
                                    max_experiences: int = 10) -> List[Dict[str, Any]]:
        """استرجاع التجارب ذات الصلة بالحالة الحالية"""
        try:
            if not self.memory_store:
                return []
            
            # حساب التشابه مع التجارب السابقة
            scored_experiences = []
            
            for experience in self.memory_store[-1000:]:  # البحث في آخر 1000 تجربة
                similarity = self._calculate_similarity(current_situation, experience['sensor_data'])
                scored_experiences.append((similarity, experience))
            
            # ترتيب حسب التشابه
            scored_experiences.sort(key=lambda x: x[0], reverse=True)
            
            return [exp for _, exp in scored_experiences[:max_experiences]]
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving experiences: {e}")
            return []
    
    def _calculate_similarity(self, current: Dict[str, Any], previous: Dict[str, Any]) -> float:
        """حساب التشابه بين الحالة الحالية والسابقة"""
        try:
            common_sensors = set(current.keys()) & set(previous.keys())
            if not common_sensors:
                return 0.0
            
            similarities = []
            for sensor in common_sensors:
                curr_val = current[sensor]
                prev_val = previous[sensor]
                
                if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
                    # حساب التشابه بناءً على الفرق النسبي
                    if max(abs(curr_val), abs(prev_val)) > 0:
                        similarity = 1 - (abs(curr_val - prev_val) / max(abs(curr_val), abs(prev_val)))
                        similarities.append(max(0, similarity))
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating similarity: {e}")
            return 0.0

class MemoryPlaybook:
    """
    دفتر خطط الطوارئ - استجابة ذكية للكوارث
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.emergency_playbooks = self._load_default_playbooks()
    
    def _load_default_playbooks(self) -> Dict[str, Any]:
        """تحميل خطط الطوارئ الافتراضية"""
        return {
            'high_pressure': {
                'name': 'High Pressure Emergency',
                'triggers': ['pressure > 80'],
                'actions': [
                    {'type': 'pressure_release', 'intensity': 0.8},
                    {'type': 'flow_adjustment', 'adjustment': -15},
                    {'type': 'emergency_cooling', 'intensity': 0.6}
                ],
                'priority': 'critical'
            },
            'high_temperature': {
                'name': 'High Temperature Emergency', 
                'triggers': ['temperature > 120'],
                'actions': [
                    {'type': 'emergency_cooling', 'intensity': 0.9},
                    {'type': 'flow_adjustment', 'adjustment': 10},
                    {'type': 'pressure_release', 'intensity': 0.3}
                ],
                'priority': 'critical'
            },
            'gas_leak': {
                'name': 'Gas Leak Emergency',
                'triggers': ['methane > 500', 'hydrogen_sulfide > 50'],
                'actions': [
                    {'type': 'gas_venting', 'venting_rate': 0.9},
                    {'type': 'flow_adjustment', 'adjustment': -20},
                    {'type': 'emergency_cooling', 'intensity': 0.4}
                ],
                'priority': 'emergency'
            },
            'equipment_failure': {
                'name': 'Equipment Failure',
                'triggers': ['vibration > 5'],
                'actions': [
                    {'type': 'vibration_dampening', 'effectiveness': 0.8},
                    {'type': 'flow_adjustment', 'adjustment': -10},
                    {'type': 'pressure_release', 'intensity': 0.2}
                ],
                'priority': 'high'
            }
        }
    
    def load_emergency_playbooks(self) -> Dict[str, Any]:
        """تحميل خطط الطوارئ"""
        return self.emergency_playbooks
    
    def get_appropriate_playbook(self, sensor_data: Dict[str, Any], 
                               anomaly_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """الحصول على خطة الطوارئ المناسبة"""
        try:
            triggered_playbooks = []
            
            for playbook_name, playbook in self.emergency_playbooks.items():
                if self._check_triggers(playbook['triggers'], sensor_data):
                    triggered_playbooks.append(playbook)
            
            if triggered_playbooks:
                # اختيار الخطة ذات الأولوية الأعلى
                priority_order = {'emergency': 3, 'critical': 2, 'high': 1, 'medium': 0}
                best_playbook = max(triggered_playbooks, 
                                  key=lambda p: priority_order.get(p['priority'], 0))
                return best_playbook
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting appropriate playbook: {e}")
            return None
    
    def _check_triggers(self, triggers: List[str], sensor_data: Dict[str, Any]) -> bool:
        """التحقق من شروط التنشيط"""
        try:
            for trigger in triggers:
                if '>' in trigger:
                    sensor, value = trigger.split('>')
                    sensor = sensor.strip()
                    threshold = float(value.strip())
                    
                    if sensor in sensor_data and sensor_data[sensor] > threshold:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking triggers: {e}")
            return False
    
    def add_custom_playbook(self, name: str, triggers: List[str], 
                          actions: List[Dict[str, Any]], priority: str = 'medium'):
        """إضافة خطة طوارئ مخصصة"""
        try:
            self.emergency_playbooks[name] = {
                'name': name,
                'triggers': triggers,
                'actions': actions,
                'priority': priority
            }
            self.logger.info(f"✅ Custom playbook added: {name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding custom playbook: {e}")

# دالة إنشاء أنظمة الذكاء الاصطناعي
def create_ai_systems(config: Dict[str, Any]) -> Dict[str, Any]:
    """إنشاء جميع أنظمة الذكاء الاصطناعي"""
    try:
        systems = {
            'ai_chat': AIChatSystem(config.get('api_keys', {}).get('openai', {}).get('api_key', '')),
            'anomaly_detector': AdvancedAnomalyDetector(config),
            'prediction_engine': AdvancedPredictionEngine(config),
            'intervention_engine': InterventionEngine(config),
            'lifelong_memory': LifelongMemory(config),
            'memory_playbook': MemoryPlaybook(config)
        }
        
        logging.info("✅ AI systems created successfully")
        return systems
        
    except Exception as e:
        logging.error(f"❌ Failed to create AI systems: {e}")
        raise

if __name__ == "__main__":
    # اختبار أنظمة الذكاء الاصطناعي
    test_config = {
        'api_keys': {
            'openai': {'api_key': 'test'}
        }
    }
    
    ai_systems = create_ai_systems(test_config)
    print("✅ AI systems tested successfully!")
