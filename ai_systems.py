import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedPreprocessor:
    """معالج بيانات متقدم للذكاء الاصطناعي"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Preprocessor')
        self.scaler = StandardScaler()
        self.feature_names = ['pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow']
        
    def preprocess_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """معالجة متقدمة لبيانات المستشعرات"""
        try:
            # استخراج القيم بالترتيب الصحيح
            values = [sensor_data.get(sensor, 0.0) for sensor in self.feature_names]
            array_data = np.array(values).reshape(1, -1)
            
            # تطبيع البيانات
            normalized_data = self.scaler.fit_transform(array_data)
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 50) -> np.ndarray:
        """إنشاء متواليات زمنية للتدريب"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)

class AdvancedAnomalySystem:
    """نظام كشف الشذوذ المتقدم باستخدام多种 خوارزميات"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Anomaly')
        
        # تهيئة نماذج كشف الشذوذ المتقدمة
        self.models = {
            'isolation_forest': IsolationForest(
                n_estimators=config['ai_models']['isolation_forest']['n_estimators'],
                contamination=config['ai_models']['isolation_forest']['contamination'],
                max_features=config['ai_models']['isolation_forest']['max_features'],
                bootstrap=config['ai_models']['isolation_forest']['bootstrap'],
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=0.1,
                kernel='rbf',
                gamma='scale'
            ),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        
        self.autoencoder = self._build_advanced_autoencoder()
        self.preprocessor = AdvancedPreprocessor(config)
        self.anomaly_history = []
        self.is_trained = False
        
        self.logger.info("✅ Advanced Anomaly Detection System Initialized")
    
    def _build_advanced_autoencoder(self) -> nn.Module:
        """بناء Autoencoder متقدم للكشف عن الشذوذ"""
        class AdvancedAutoencoder(nn.Module):
            def __init__(self, input_dim=6, encoding_dim=32):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, encoding_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(encoding_dim)
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, input_dim),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return AdvancedAutoencoder(
            input_dim=6,
            encoding_dim=self.config['ai_models']['autoencoder']['encoding_dim']
        )
    
    def train_anomaly_models(self, training_data: List[Dict[str, Any]]):
        """تدريب جميع نماذج كشف الشذوذ"""
        try:
            self.logger.info("🔄 Training anomaly detection models...")
            
            # تحضير البيانات
            X = np.array([list(data.values()) for data in training_data])
            
            # تدريب Isolation Forest
            self.models['isolation_forest'].fit(X)
            
            # تدريب OneClass SVM
            self.models['one_class_svm'].fit(X)
            
            # تدريب Autoencoder
            self._train_autoencoder(X)
            
            self.is_trained = True
            self.logger.info("✅ All anomaly models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Anomaly models training failed: {e}")
    
    def _train_autoencoder(self, X: np.ndarray):
        """تدريب الـAutoencoder"""
        try:
            # تحويل البيانات إلى Tensor
            X_tensor = torch.FloatTensor(X)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # خيارات التدريب
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
            
            # التدريب
            self.autoencoder.train()
            for epoch in range(self.config['ai_models']['autoencoder']['epochs']):
                total_loss = 0
                for batch in dataloader:
                    data = batch[0]
                    optimizer.zero_grad()
                    output = self.autoencoder(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    self.logger.info(f"Autoencoder Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
                    
        except Exception as e:
            self.logger.error(f"❌ Autoencoder training failed: {e}")
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """كشف شذوذ متقدم باستخدام多种 خوارزميات"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained', 'is_anomaly': False}
            
            # معالجة البيانات
            processed_data = self.preprocessor.preprocess_data(sensor_data)
            
            # الكشف باستخدام جميع النماذج
            results = {}
            
            # Isolation Forest
            iforest_pred = self.models['isolation_forest'].predict(processed_data)[0]
            results['isolation_forest'] = iforest_pred == -1
            
            # OneClass SVM
            svm_pred = self.models['one_class_svm'].predict(processed_data)[0]
            results['one_class_svm'] = svm_pred == -1
            
            # Autoencoder Reconstruction Error
            autoencoder_anomaly = self._detect_autoencoder_anomaly(processed_data)
            results['autoencoder'] = autoencoder_anomaly
            
            # حساب درجة الشذوذ الإجمالية
            anomaly_score = sum(results.values()) / len(results)
            
            # تحليل متقدم
            critical_anomalies = self._analyze_critical_anomalies(sensor_data)
            risk_level = self._calculate_risk_level(anomaly_score, critical_anomalies)
            
            result = {
                'is_anomaly': anomaly_score > 0.5,
                'anomaly_score': float(anomaly_score),
                'risk_level': risk_level,
                'algorithm_results': results,
                'critical_anomalies': critical_anomalies,
                'confidence': self._calculate_confidence(results),
                'timestamp': datetime.now()
            }
            
            # تخزين في السجل
            self.anomaly_history.append(result)
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Anomaly detection failed: {e}")
            return {'error': str(e), 'is_anomaly': False}
    
    def _detect_autoencoder_anomaly(self, data: np.ndarray) -> bool:
        """كشف الشذوذ باستخدام Autoencoder"""
        try:
            self.autoencoder.eval()
            with torch.no_grad():
                data_tensor = torch.FloatTensor(data)
                reconstructed = self.autoencoder(data_tensor)
                reconstruction_error = torch.mean((data_tensor - reconstructed) ** 2).item()
                
            return reconstruction_error > 0.1  # عتبة الشذوذ
            
        except Exception as e:
            self.logger.error(f"❌ Autoencoder detection failed: {e}")
            return False
    
    def _analyze_critical_anomalies(self, sensor_data: Dict[str, Any]) -> List[str]:
        """تحليل الشذوذ الحرج"""
        critical_anomalies = []
        thresholds = {
            'pressure': 150,
            'temperature': 200,
            'methane': 1000,
            'hydrogen_sulfide': 50,
            'vibration': 8,
            'flow': 400
        }
        
        for sensor, value in sensor_data.items():
            if sensor in thresholds and value > thresholds[sensor]:
                critical_anomalies.append(f"{sensor} exceeded threshold: {value} > {thresholds[sensor]}")
        
        return critical_anomalies
    
    def _calculate_risk_level(self, anomaly_score: float, critical_anomalies: List[str]) -> str:
        """حساب مستوى الخطورة"""
        if anomaly_score >= 0.8 or len(critical_anomalies) >= 2:
            return 'CRITICAL'
        elif anomaly_score >= 0.6 or len(critical_anomalies) >= 1:
            return 'HIGH'
        elif anomaly_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_confidence(self, results: Dict[str, bool]) -> float:
        """حساب درجة الثقة في الكشف"""
        agreement = sum(results.values())
        total = len(results)
        return agreement / total if total > 0 else 0.0

class AdvancedPredictionEngine:
    """محرك تنبؤ متقدم باستخدام LSTM متعدد المستويات"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Prediction')
        
        # نماذج LVM للمستويات الزمنية المختلفة
        self.models = {
            'short_term': self._build_lstm_model('short_term'),
            'medium_term': self._build_lstm_model('medium_term'),
            'long_term': self._build_lstm_model('long_term')
        }
        
        self.preprocessor = AdvancedPreprocessor(config)
        self.prediction_history = []
        self.is_trained = False
        
        self.logger.info("✅ Advanced Prediction Engine Initialized")
    
    def _build_lstm_model(self, timeframe: str) -> nn.Module:
        """بناء نموذج LSTM للمستوى الزمني المحدد"""
        config = self.config['ai_models']['lstm_models'][timeframe]
        
        class TimeframeLSTM(nn.Module):
            def __init__(self, input_size=6, hidden_size=128, output_size=6, num_layers=2, dropout=0.2):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.attention = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                # LSTM
                lstm_out, (hidden, cell) = self.lstm(x)
                
                # Attention mechanism
                attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
                context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
                
                # التنبؤ
                output = self.fc(self.dropout(context_vector))
                return output
        
        return TimeframeLSTM(
            input_size=6,
            hidden_size=config['units'],
            num_layers=config['layers'],
            dropout=config['dropout']
        )
    
    def train_prediction_models(self, historical_data: List[Dict[str, Any]]):
        """تدريب نماذج التنبؤ"""
        try:
            self.logger.info("🔄 Training prediction models...")
            
            # تحضير البيانات
            sequences, targets = self._prepare_training_data(historical_data)
            
            # تدريب كل نموذج
            for timeframe, model in self.models.items():
                self._train_single_model(model, sequences, targets, timeframe)
            
            self.is_trained = True
            self.logger.info("✅ All prediction models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Prediction models training failed: {e}")
    
    def _prepare_training_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """تحضير بيانات التدريب"""
        # تحويل البيانات إلى مصفوفة
        values = np.array([list(d.values()) for d in data])
        
        # إنشاء متواليات
        sequence_length = 50
        sequences = []
        targets = []
        
        for i in range(len(values) - sequence_length):
            sequences.append(values[i:i + sequence_length])
            targets.append(values[i + sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def _train_single_model(self, model: nn.Module, sequences: np.ndarray, 
                          targets: np.ndarray, timeframe: str):
        """تدريب نموذج فردي"""
        try:
            # تحويل إلى Tensor
            X_tensor = torch.FloatTensor(sequences)
            y_tensor = torch.FloatTensor(targets)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # خيارات التدريب
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # التدريب
            model.train()
            for epoch in range(100):  # 100 epoch للتدريب
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    self.logger.info(f"{timeframe} LSTM Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
                    
        except Exception as e:
            self.logger.error(f"❌ {timeframe} model training failed: {e}")
    
    def predict(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ متعدد المستويات"""
        try:
            if not self.is_trained:
                return self._generate_simulated_predictions(sensor_data)
            
            # معالجة البيانات
            processed_data = self.preprocessor.preprocess_data(sensor_data)
            
            # التنبؤ لكل مستوى زمني
            predictions = {}
            horizons = self.config['foresight_engine']['prediction_horizons']
            
            for timeframe, horizon in horizons.items():
                predictions[timeframe] = self._predict_timeframe(
                    self.models[timeframe], processed_data, horizon
                )
            
            # تحليل الاتجاهات والمخاطر
            trend_analysis = self._analyze_trends(predictions)
            risk_assessment = self._assess_risks(predictions)
            
            result = {
                'predictions': predictions,
                'trend_analysis': trend_analysis,
                'risk_assessment': risk_assessment,
                'confidence_scores': self._calculate_prediction_confidence(predictions),
                'critical_points': self._identify_critical_points(predictions),
                'timestamp': datetime.now()
            }
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return self._generate_simulated_predictions(sensor_data)
    
    def _predict_timeframe(self, model: nn.Module, data: np.ndarray, horizon: int) -> Dict[str, Any]:
        """التنبؤ لمستوى زمني محدد"""
        try:
            model.eval()
            with torch.no_grad():
                # تحضير بيانات الإدخال
                input_seq = torch.FloatTensor(data).unsqueeze(0)
                
                # التنبؤ
                prediction = model(input_seq).numpy().flatten()
                
                return {
                    'horizon': horizon,
                    'values': prediction.tolist(),
                    'time_unit': 'hours',
                    'model': 'Advanced_LSTM'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Timeframe prediction failed: {e}")
            return self._simulate_prediction(horizon)
    
    def _generate_simulated_predictions(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """توليد تنبؤات محاكاة (للاختبار)"""
        predictions = {}
        horizons = self.config['foresight_engine']['prediction_horizons']
        
        for timeframe, horizon in horizons.items():
            predictions[timeframe] = self._simulate_prediction(horizon)
        
        return {
            'predictions': predictions,
            'trend_analysis': {'overall': 'stable'},
            'risk_assessment': {'overall_risk': 'low'},
            'confidence_scores': {'overall': 0.8},
            'critical_points': [],
            'timestamp': datetime.now(),
            'simulated': True
        }
    
    def _simulate_prediction(self, horizon: int) -> Dict[str, Any]:
        """محاكاة تنبؤ واقعي"""
        base_values = np.random.uniform(0.3, 0.7, 6)
        trend = np.linspace(0, 0.2, horizon)
        
        predicted_values = []
        for i in range(horizon):
            values = base_values * (1 + trend[i] * np.random.normal(1, 0.1))
            predicted_values.append(values.tolist())
        
        return {
            'horizon': horizon,
            'values': predicted_values,
            'time_unit': 'hours',
            'model': 'Simulated_LSTM'
        }

class InterventionEngine:
    """محرك التدخل المتقدم - بديل Monte Carlo"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Intervention')
        self.scenarios_per_second = config['foresight_engine']['scenarios_per_second']['default']
        self.decision_history = []
        
        self.logger.info("✅ Advanced Intervention Engine Initialized")
    
    def make_decision(self, sensor_data: Dict, predictions: Dict, anomalies: Dict) -> Dict[str, Any]:
        """اتخاذ قرار تدخل متقدم"""
        try:
            # تحليل الوضع الحالي
            situation_analysis = self._analyze_current_situation(sensor_data, predictions, anomalies)
            
            # توليد سيناريوهات
            scenarios = self._generate_scenarios(situation_analysis)
            
            # تقييم السيناريوهات
            evaluated_scenarios = self._evaluate_scenarios(scenarios)
            
            # دمج القرارات
            final_decision = self._make_final_decision(evaluated_scenarios)
            
            result = {
                'decision': final_decision,
                'situation_analysis': situation_analysis,
                'scenarios_generated': len(scenarios),
                'scenarios_per_second': self.scenarios_per_second,
                'confidence': final_decision.get('confidence', 0.7),
                'reasoning': final_decision.get('reasoning', ''),
                'timestamp': datetime.now()
            }
            
            self.decision_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Decision making failed: {e}")
            return {'error': str(e)}
    
    def _analyze_current_situation(self, sensor_data: Dict, predictions: Dict, anomalies: Dict) -> Dict[str, Any]:
        """تحليل الوضع الحالي المتقدم"""
        risk_factors = {}
        
        # تحليل بيانات المستشعرات
        for sensor, value in sensor_data.items():
            risk_factors[sensor] = self._calculate_sensor_risk(sensor, value)
        
        # تحليل التنبؤات
        pred_risk = self._analyze_prediction_risk(predictions)
        
        # تحليل الشذوذ
        anomaly_risk = self._analyze_anomaly_risk(anomalies)
        
        return {
            'sensor_risks': risk_factors,
            'prediction_risks': pred_risk,
            'anomaly_risks': anomaly_risk,
            'overall_risk': max(list(risk_factors.values()) + [pred_risk, anomaly_risk])
        }
    
    def _generate_scenarios(self, situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """توليد سيناريوهات متعددة"""
        scenarios = []
        num_scenarios = min(self.scenarios_per_second, 1000)
        
        for i in range(num_scenarios):
            scenario = {
                'id': i,
                'actions': self._generate_random_actions(situation['overall_risk']),
                'expected_outcomes': {},
                'risk_reduction': np.random.random(),
                'cost': np.random.random() * 1000,
                'time_to_implement': np.random.exponential(10)
            }
            
            # حساب النتائج المتوقعة
            scenario['expected_outcomes'] = self._simulate_outcomes(scenario['actions'])
            scenarios.append(scenario)
        
        return scenarios
    
    def _evaluate_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تقييم السيناريوهات باستخدام معايير متعددة"""
        evaluated = []
        
        for scenario in scenarios:
            # حساب الدرجة الإجمالية
            score = self._calculate_scenario_score(scenario)
            scenario['score'] = score
            evaluated.append(scenario)
        
        # ترتيب حسب الدرجة
        evaluated.sort(key=lambda x: x['score'], reverse=True)
        return evaluated
    
    def _make_final_decision(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """اتخاذ القرار النهائي"""
        if not scenarios:
            return {'actions': [], 'confidence': 0.0, 'reasoning': 'No scenarios available'}
        
        # اختيار أفضل سيناريو
        best_scenario = scenarios[0]
        
        return {
            'actions': best_scenario['actions'][:3],  # أفضل 3 إجراءات
            'confidence': best_scenario['score'],
            'reasoning': f"Selected scenario {best_scenario['id']} with score {best_scenario['score']:.3f}",
            'expected_risk_reduction': best_scenario['risk_reduction'],
            'implementation_time': best_scenario['time_to_implement']
        }

class AdaptiveLifelongMemory:
    """ذاكرة مستدامة تكيفية متقدمة"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Memory')
        self.memory_store = []
        self.pattern_library = {}
        self.experience_weights = {}
        self.max_memories = 10000
        
        self.logger.info("✅ Adaptive Lifelong Memory Initialized")
    
    def store_experience(self, experience: Dict[str, Any]):
        """تخزين تجربة مع تحليل أنماط"""
        try:
            # إضافة معلومات زمنية
            experience['timestamp'] = datetime.now()
            experience['memory_id'] = len(self.memory_store)
            experience['pattern_hash'] = self._generate_pattern_hash(experience)
            
            # التخزين الأساسي
            self.memory_store.append(experience)
            
            # تحديث مكتبة الأنماط
            self._update_pattern_library(experience)
            
            # التعلم التكيفي
            self._adaptive_learning(experience)
            
            # إدارة الذاكرة
            self._manage_memory_size()
            
            self.logger.debug(f"💾 Experience stored: {experience['memory_id']}")
            
        except Exception as e:
            self.logger.error(f"❌ Experience storage failed: {e}")
    
    def retrieve_relevant_memories(self, current_situation: Dict[str, Any], 
                                 max_memories: int = 10) -> List[Dict[str, Any]]:
        """استرجاع ذكريات ذات صلة مع ترجيح"""
        try:
            scored_memories = []
            
            for memory in self.memory_store[-1000:]:  # البحث في الذاكرة الحديثة
                similarity = self._calculate_similarity(current_situation, memory)
                relevance = self._calculate_temporal_relevance(memory)
                pattern_match = self._calculate_pattern_match(current_situation, memory)
                
                # حساب النقاط الإجمالية
                total_score = (similarity * 0.4 + relevance * 0.3 + pattern_match * 0.3)
                
                scored_memories.append((total_score, memory))
            
            # ترتيب حسب النقاط
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            
            return [memory for score, memory in scored_memories[:max_memories]]
            
        except Exception as e:
            self.logger.error(f"❌ Memory retrieval failed: {e}")
            return []
    
    def _generate_pattern_hash(self, experience: Dict[str, Any]) -> str:
        """توليد هاش للنمط"""
        import hashlib
        pattern_data = str(experience.get('sensor_data', {}))
        return hashlib.md5(pattern_data.encode()).hexdigest()[:16]
    
    def _update_pattern_library(self, experience: Dict[str, Any]):
        """تحديث مكتبة الأنماط"""
        pattern_hash = experience['pattern_hash']
        
        if pattern_hash in self.pattern_library:
            self.pattern_library[pattern_hash]['count'] += 1
            self.pattern_library[pattern_hash]['last_seen'] = datetime.now()
        else:
            self.pattern_library[pattern_hash] = {
                'count': 1,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'example': experience
            }
    
    def _adaptive_learning(self, experience: Dict[str, Any]):
        """تعلم تكيفي من التجارب"""
        # تحديث أوزان التجارب المماثلة
        similar_memories = self.retrieve_relevant_memories(experience, 5)
        
        for _, memory in similar_memories:
            mem_id = memory['memory_id']
            if mem_id in self.experience_weights:
                self.experience_weights[mem_id] *= 1.1  # تعزيز الوزن
            else:
                self.experience_weights[mem_id] = 1.0
    
    def _manage_memory_size(self):
        """إدارة حجم الذاكرة"""
        if len(self.memory_store) > self.max_memories:
            # إزالة الذكريات الأقل أهمية
            self.memory_store.sort(key=lambda x: self.experience_weights.get(x['memory_id'], 0))
            self.memory_store = self.memory_store[-self.max_memories:]

class ForeSightEngine:
    """محرك التوقع الشامل - يجمع كل المكونات"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.ForeSight')
        
        # تهيئة جميع المكونات
        self.anomaly_system = AdvancedAnomalySystem(config)
        self.prediction_engine = AdvancedPredictionEngine(config)
        self.intervention_engine = InterventionEngine(config)
        self.memory_system = AdaptiveLifelongMemory(config)
        
        # إعدادات المحرك
        self.scenarios_per_second = config['foresight_engine']['scenarios_per_second']['default']
        self.processing_history = []
        
        self.logger.info("🚀 ForeSight Engine Initialized - All Systems Ready")
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة كاملة لبيانات المستشعرات"""
        try:
            # 1. كشف الشذوذ
            anomalies = self.anomaly_system.detect_anomalies(sensor_data)
            
            # 2. التنبؤ بالمستقبل
            predictions = self.prediction_engine.predict(sensor_data)
            
            # 3. اتخاذ قرار التدخل
            decision = self.intervention_engine.make_decision(sensor_data, predictions, anomalies)
            
            # 4. تخزين في الذاكرة التكيفية
            experience = {
                'sensor_data': sensor_data,
                'anomalies': anomalies,
                'predictions': predictions,
                'decision': decision,
                'processing_time': datetime.now()
            }
            self.memory_system.store_experience(experience)
            
            # 5. تجميع النتائج
            result = {
                'anomalies': anomalies,
                'predictions': predictions,
                'decision': decision,
                'memory_usage': len(self.memory_system.memory_store),
                'scenarios_per_second': self.scenarios_per_second,
                'processing_timestamp': datetime.now(),
                'engine_status': 'OPTIMAL'
            }
            
            self.processing_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Sensor data processing failed: {e}")
            return {'error': str(e), 'engine_status': 'ERROR'}
    
    def update_engine_settings(self, scenarios_per_second: int):
        """تحديث إعدادات المحرك"""
        self.scenarios_per_second = scenarios_per_second
        self.intervention_engine.scenarios_per_second = scenarios_per_second
        
        self.logger.info(f"✅ ForeSight Engine settings updated: {scenarios_per_second} scenarios/sec")

# اختبار النظام
if __name__ == "__main__":
    print("🧠 Smart Neural Digital Twin AI Systems - SS Rating Activated")
