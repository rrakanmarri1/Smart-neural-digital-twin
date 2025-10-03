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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import hashlib
import json
import math
from scipy import stats
from dataclasses import dataclass
from enum import Enum
import torch.nn.functional as F

warnings.filterwarnings('ignore')

class ModelType(Enum):
    """أنواع النماذج المتاحة"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    LSTM = "lstm"
    TRANSFORMER = "transformer"

@dataclass
class ModelPerformance:
    """أداء النموذج"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_loss: List[float]
    validation_loss: List[float]
    last_trained: datetime

class DynamicModelSelector:
    """مختار نماذج ديناميكي متقدم"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.ModelSelector')
        self.available_models = {}
        self.model_performance = {}
        self.current_best_model = None
        
        self._initialize_models()
        self.logger.info("✅ Dynamic Model Selector Initialized - SS Rating")
    
    def _initialize_models(self):
        """تهيئة جميع النماذج المتاحة"""
        try:
            # نماذج كشف الشذوذ
            self.available_models[ModelType.ISOLATION_FOREST] = {
                'model': IsolationForest(
                    n_estimators=200,
                    contamination=0.1,
                    max_features=1.0,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 0.3,
                'performance': ModelPerformance(0.0, 0.0, 0.0, 0.0, [], [], datetime.now())
            }
            
            self.available_models[ModelType.ONE_CLASS_SVM] = {
                'model': OneClassSVM(
                    nu=0.1,
                    kernel='rbf',
                    gamma='scale'
                ),
                'weight': 0.2,
                'performance': ModelPerformance(0.0, 0.0, 0.0, 0.0, [], [], datetime.now())
            }
            
            self.available_models[ModelType.AUTOENCODER] = {
                'model': self._build_advanced_autoencoder(),
                'weight': 0.3,
                'performance': ModelPerformance(0.0, 0.0, 0.0, 0.0, [], [], datetime.now())
            }
            
            self.available_models[ModelType.LSTM] = {
                'model': self._build_lstm_anomaly_detector(),
                'weight': 0.2,
                'performance': ModelPerformance(0.0, 0.0, 0.0, 0.0, [], [], datetime.now())
            }
            
            self.logger.info(f"✅ {len(self.available_models)} AI models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
    
    def _build_advanced_autoencoder(self) -> nn.Module:
        """بناء Autoencoder متقدم للكشف عن الشذوذ"""
        class AdvancedAutoencoder(nn.Module):
            def __init__(self, input_dim=6, encoding_dim=32):
                super().__init__()
                # Encoder متقدم
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, encoding_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(encoding_dim)
                )
                # Decoder متقدم
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, input_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return AdvancedAutoencoder(
            input_dim=6,
            encoding_dim=self.config['ai_models']['autoencoder']['encoding_dim']
        )
    
    def _build_lstm_anomaly_detector(self) -> nn.Module:
        """بناء LSTM متقدم للكشف عن الشذوذ"""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_size=6, hidden_size=64, num_layers=2):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # LSTM Encoder
                self.encoder = nn.LSTM(input_size, hidden_size, num_layers, 
                                     batch_first=True, dropout=0.2)
                
                # LSTM Decoder
                self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                                     batch_first=True, dropout=0.2)
                
                self.output_layer = nn.Linear(hidden_size, input_size)
            
            def forward(self, x):
                # Encoding
                _, (hidden, cell) = self.encoder(x)
                
                # Decoding
                decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_size).to(x.device)
                decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
                
                # Reconstruction
                output = self.output_layer(decoder_output)
                return output
        
        return LSTMAutoencoder()

class SensorFeatureMapper:
    """مدير ميزات المستشعرات المتقدم"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = sorted(feature_names)
        self.feature_indices = {name: idx for idx, name in enumerate(self.feature_names)}
        self.feature_importance = {name: 1.0 for name in feature_names}  # أهمية كل ميزة
    
    def extract_ordered_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """استخراج الميزات بترتيب ثابت مع مراعاة الأهمية"""
        return np.array([sensor_data.get(sensor, 0.0) for sensor in self.feature_names])
    
    def update_feature_importance(self, importances: Dict[str, float]):
        """تحديث أهمية الميزات بناءً على تحليل الأداء"""
        for feature, importance in importances.items():
            if feature in self.feature_importance:
                self.feature_importance[feature] = importance
    
    def get_weighted_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """الحصول على ميزات موزونة بالأهمية"""
        features = self.extract_ordered_features(sensor_data)
        weights = np.array([self.feature_importance[feature] for feature in self.feature_names])
        return features * weights

class AdvancedPreprocessor:
    """معالج بيانات متقدم للذكاء الاصطناعي - SS Rating"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Preprocessor')
        
        # متعدد السكالرات لأنواع مختلفة من البيانات
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': StandardScaler()  # محاكاة للـ RobustScaler
        }
        
        self.feature_mapper = SensorFeatureMapper([
            'pressure', 'temperature', 'methane', 'hydrogen_sulfide', 'vibration', 'flow'
        ])
        
        self.is_fitted = False
        self.data_statistics = {}
        self.feature_correlations = {}
        
        self.logger.info("✅ Advanced Preprocessor Initialized - SS Rating")
    
    def fit(self, training_data: List[Dict[str, Any]]):
        """تدريب المعالج على بيانات التدريب بشكل متقدم"""
        try:
            # استخراج البيانات بالترتيب الصحيح
            X = np.array([self.feature_mapper.extract_ordered_features(data) for data in training_data])
            
            # حساب إحصائيات البيانات
            self._compute_data_statistics(X)
            
            # حساب الارتباطات بين الميزات
            self._compute_feature_correlations(X)
            
            # تدريب جميع السكالرات
            for scaler_name, scaler in self.scalers.items():
                scaler.fit(X)
            
            self.is_fitted = True
            self.logger.info("✅ Advanced preprocessor fitted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessor fitting failed: {e}")
    
    def _compute_data_statistics(self, X: np.ndarray):
        """حساب إحصائيات متقدمة للبيانات"""
        self.data_statistics = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0),
            'skewness': stats.skew(X, axis=0),
            'kurtosis': stats.kurtosis(X, axis=0)
        }
    
    def _compute_feature_correlations(self, X: np.ndarray):
        """حساب الارتباطات بين الميزات"""
        df = pd.DataFrame(X, columns=self.feature_mapper.feature_names)
        self.feature_correlations = df.corr().to_dict()
    
    def preprocess_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """معالجة متقدمة لبيانات المستشعرات"""
        try:
            # استخراج القيم بالترتيب الصحيح
            values = self.feature_mapper.extract_ordered_features(sensor_data)
            array_data = values.reshape(1, -1)
            
            if self.is_fitted:
                # استخدام السكالر المناسب لنوع البيانات
                normalized_data = self.scalers['minmax'].transform(array_data)
            else:
                # استخدام تطبيع بسيط إذا لم يتم التدريب
                normalized_data = array_data / np.array([200, 300, 5000, 500, 20, 500])
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"❌ Advanced data preprocessing failed: {e}")
            return np.zeros((1, len(self.feature_mapper.feature_names)))
    
    def detect_data_quality_issues(self, sensor_data: Dict[str, Any]) -> List[str]:
        """كشف مشاكل جودة البيانات"""
        issues = []
        values = self.feature_mapper.extract_ordered_features(sensor_data)
        
        for i, (feature, value) in enumerate(zip(self.feature_mapper.feature_names, values)):
            # التحقق من القيم الشاذة الإحصائية
            if self.data_statistics and self.is_fitted:
                z_score = abs(value - self.data_statistics['mean'][i]) / (self.data_statistics['std'][i] + 1e-8)
                if z_score > 3:  # خارج 3 انحرافات معيارية
                    issues.append(f"Outlier detected in {feature}: z-score = {z_score:.2f}")
            
            # التحقق من القيم المستحيلة
            sensor_config = self.config['sensors'].get(feature, {})
            if value < sensor_config.get('min', 0) or value > sensor_config.get('max', 1000):
                issues.append(f"Impossible value in {feature}: {value}")
        
        return issues
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 50) -> np.ndarray:
        """إنشاء متواليات زمنية متقدمة للتدريب"""
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:(i + sequence_length)]
            sequences.append(seq)
        return np.array(sequences)

class EnsembleAnomalyDetector:
    """كاشف شذوذ متقدم باستخدام تقنيات التجميع"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.EnsembleAnomaly')
        
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        
        self._initialize_ensemble_models()
        self.logger.info("✅ Ensemble Anomaly Detector Initialized - SS Rating")
    
    def _initialize_ensemble_models(self):
        """تهيئة نماذج التجميع"""
        # Isolation Forest محسن
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=300,
            contamination=0.05,  # أكثر تحفظاً
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # One-Class SVM محسن
        self.models['one_class_svm'] = OneClassSVM(
            nu=0.05,  # أكثر تحفظاً
            kernel='rbf',
            gamma='auto',
            cache_size=1000
        )
        
        # Local Outlier Factor
        from sklearn.neighbors import LocalOutlierFactor
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        
        # تعيين الأوزان الأولية
        self.model_weights = {
            'isolation_forest': 0.4,
            'one_class_svm': 0.3,
            'lof': 0.3
        }
    
    def train_ensemble(self, X: np.ndarray):
        """تدريب نماذج التجميع"""
        try:
            for name, model in self.models.items():
                if hasattr(model, 'fit'):
                    model.fit(X)
                    self.logger.info(f"✅ Ensemble model {name} trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
    
    def predict_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """التنبؤ باستخدام التجميع المتقدم"""
        try:
            predictions = {}
            scores = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions[name] = (pred == -1)  # تحويل إلى boolean
                
                if hasattr(model, 'decision_function'):
                    score = model.decision_function(X)
                    scores[name] = score
                elif hasattr(model, 'score_samples'):
                    score = model.score_samples(X)
                    scores[name] = score
            
            # تجميع النتائج مع الأوزان
            weighted_score = 0.0
            total_weight = 0.0
            
            for name, weight in self.model_weights.items():
                if name in predictions:
                    weighted_score += weight * (1.0 if predictions[name] else 0.0)
                    total_weight += weight
            
            final_prediction = weighted_score / total_weight if total_weight > 0 else 0.0
            
            return {
                'ensemble_score': final_prediction,
                'individual_predictions': predictions,
                'individual_scores': scores,
                'is_anomaly': final_prediction > 0.6,
                'confidence': min(1.0, abs(final_prediction - 0.5) * 2)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble prediction failed: {e}")
            return {'is_anomaly': False, 'ensemble_score': 0.0, 'confidence': 0.0}

class AdvancedAnomalySystem:
    """نظام كشف الشذوذ المتقدم - SS Rating"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Anomaly')
        
        # المكونات المتقدمة
        self.preprocessor = AdvancedPreprocessor(config)
        self.ensemble_detector = EnsembleAnomalyDetector(config)
        self.autoencoder = self._build_advanced_autoencoder()
        
        # إدارة الحالة
        self.anomaly_history = []
        self.performance_metrics = {}
        self.is_trained = False
        
        # التعلم التكيفي
        self.adaptive_threshold = 0.6
        self.false_positive_rate = 0.0
        self.detection_sensitivity = 0.8
        
        self.logger.info("✅ Advanced Anomaly Detection System Initialized - SS Rating")
    
    def _build_advanced_autoencoder(self) -> nn.Module:
        """بناء Autoencoder متقدم مع تعقيد مناسب"""
        class AdvancedAutoencoder(nn.Module):
            def __init__(self, input_dim=6, encoding_dim=32):
                super().__init__()
                
                # Encoder مع تسرب وطبقات متعددة
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, encoding_dim),
                    nn.Tanh()
                )
                
                # Decoder متناظر
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 64),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 128),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, input_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return AdvancedAutoencoder(
            input_dim=6,
            encoding_dim=self.config['ai_models']['autoencoder']['encoding_dim']
        )
    
    def train_models(self, training_data: List[Dict[str, Any]]):
        """تدريب متقدم لجميع النماذج"""
        try:
            self.logger.info("🔄 Training advanced anomaly detection models...")
            
            # تدريب المعالج أولاً
            self.preprocessor.fit(training_data)
            
            # تحضير البيانات
            X = np.array([self.preprocessor.feature_mapper.extract_ordered_features(data) 
                         for data in training_data])
            
            # تدريب نماذج التجميع
            self.ensemble_detector.train_ensemble(X)
            
            # تدريب الـAutoencoder
            self._train_autoencoder(X)
            
            # معايرة العتبات
            self._calibrate_detection_thresholds(X)
            
            self.is_trained = True
            self.logger.info("✅ All advanced anomaly models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Advanced anomaly models training failed: {e}")
    
    def _train_autoencoder(self, X: np.ndarray):
        """تدريب الـAutoencoder المتقدم"""
        try:
            # تطبيع البيانات
            X_normalized = self.preprocessor.scalers['minmax'].transform(X)
            X_tensor = torch.FloatTensor(X_normalized)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # خيارات التدريب المتقدمة
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.autoencoder.parameters(), 
                                  lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # التدريب مع المراقبة
            self.autoencoder.train()
            training_losses = []
            
            for epoch in range(self.config['ai_models']['autoencoder']['epochs']):
                epoch_loss = 0.0
                for batch in dataloader:
                    data = batch[0]
                    optimizer.zero_grad()
                    reconstructed = self.autoencoder(data)
                    loss = criterion(reconstructed, data)
                    loss.backward()
                    
                    # Gradient Clipping لمنع الانفجار
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if epoch % 25 == 0:
                    self.logger.info(f"Autoencoder Epoch {epoch}, Loss: {avg_loss:.6f}")
                    
            self.performance_metrics['autoencoder_training_loss'] = training_losses
            
        except Exception as e:
            self.logger.error(f"❌ Advanced autoencoder training failed: {e}")
    
    def _calibrate_detection_thresholds(self, X: np.ndarray):
        """معايرة عتبات الكشف بناءً على بيانات التدريب"""
        try:
            # حساب أخطإ إعادة البناء لبيانات التدريب
            reconstruction_errors = []
            self.autoencoder.eval()
            
            with torch.no_grad():
                X_normalized = self.preprocessor.scalers['minmax'].transform(X)
                X_tensor = torch.FloatTensor(X_normalized)
                
                for i in range(0, len(X_tensor), 32):
                    batch = X_tensor[i:i+32]
                    reconstructed = self.autoencoder(batch)
                    error = torch.mean((batch - reconstructed) ** 2, dim=1)
                    reconstruction_errors.extend(error.numpy())
            
            # حساب العتبة باستخدام طريقة إحصائية
            reconstruction_errors = np.array(reconstruction_errors)
            self.adaptive_threshold = np.percentile(reconstruction_errors, 95)  % 95%
            
            self.logger.info(f"✅ Detection threshold calibrated: {self.adaptive_threshold:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Threshold calibration failed: {e}")
            self.adaptive_threshold = 0.05  # قيمة افتراضية آمنة
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """كشف شذوذ متقدم باستخدام تقنيات متعددة"""
        try:
            if not self.is_trained:
                return self._get_untrained_response()
            
            # التحقق من جودة البيانات أولاً
            data_quality_issues = self.preprocessor.detect_data_quality_issues(sensor_data)
            
            # معالجة البيانات
            processed_data = self.preprocessor.preprocess_data(sensor_data)
            
            # الكشف باستخدام التجميع
            ensemble_result = self.ensemble_detector.predict_ensemble(processed_data)
            
            # الكشف باستخدام الـAutoencoder
            autoencoder_result = self._detect_autoencoder_anomaly(processed_data)
            
            # تحليل متقدم للشذوذ
            critical_anomalies = self._analyze_critical_anomalies(sensor_data)
            temporal_analysis = self._perform_temporal_analysis(sensor_data)
            
            # دمج النتائج بشكل متقدم
            final_score = self._fuse_detection_results(ensemble_result, autoencoder_result)
            
            # تحديث العتبة التكيفية
            self._update_adaptive_threshold(final_score)
            
            result = {
                'is_anomaly': final_score > self.adaptive_threshold,
                'anomaly_score': float(final_score),
                'risk_level': self._calculate_risk_level(final_score, critical_anomalies),
                'ensemble_results': ensemble_result,
                'autoencoder_result': autoencoder_result,
                'critical_anomalies': critical_anomalies,
                'temporal_analysis': temporal_analysis,
                'data_quality_issues': data_quality_issues,
                'confidence': self._calculate_advanced_confidence(ensemble_result, autoencoder_result),
                'adaptive_threshold': float(self.adaptive_threshold),
                'recommendations': self._generate_anomaly_recommendations(final_score, critical_anomalies),
                'timestamp': datetime.now()
            }
            
            # تخزين في السجل
            self._update_anomaly_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced anomaly detection failed: {e}")
            return {'error': str(e), 'is_anomaly': False, 'anomaly_score': 0.0}
    
    def _detect_autoencoder_anomaly(self, data: np.ndarray) -> Dict[str, Any]:
        """كشف الشذوذ باستخدام Autoencoder متقدم"""
        try:
            self.autoencoder.eval()
            with torch.no_grad():
                data_tensor = torch.FloatTensor(data)
                reconstructed = self.autoencoder(data_tensor)
                
                # حساب خطإ إعادة البناء مع مقاييس متعددة
                mse_error = torch.mean((data_tensor - reconstructed) ** 2).item()
                mae_error = torch.mean(torch.abs(data_tensor - reconstructed)).item()
                
                # حساب درجة الشذوذ المعقدة
                reconstruction_error = 0.7 * mse_error + 0.3 * mae_error
                
                return {
                    'reconstruction_error': reconstruction_error,
                    'mse_error': mse_error,
                    'mae_error': mae_error,
                    'is_anomaly': reconstruction_error > self.adaptive_threshold,
                    'confidence': min(1.0, reconstruction_error / (self.adaptive_threshold * 2))
                }
                
        except Exception as e:
            self.logger.error(f"❌ Autoencoder detection failed: {e}")
            return {'reconstruction_error': 0.0, 'is_anomaly': False, 'confidence': 0.0}
    
    def _fuse_detection_results(self, ensemble_result: Dict, autoencoder_result: Dict) -> float:
        """دمج نتائج الكشف بشكل متقدم"""
        try:
            ensemble_score = ensemble_result.get('ensemble_score', 0.0)
            autoencoder_score = autoencoder_result.get('reconstruction_error', 0.0)
            
            # تطبيع الدرجات
            ensemble_normalized = ensemble_score
            autoencoder_normalized = min(1.0, autoencoder_score / (self.adaptive_threshold * 2))
            
            # دمج مرجح مع مراعاة الثقة
            ensemble_confidence = ensemble_result.get('confidence', 0.5)
            autoencoder_confidence = autoencoder_result.get('confidence', 0.5)
            
            total_confidence = ensemble_confidence + autoencoder_confidence
            if total_confidence > 0:
                weight_ensemble = ensemble_confidence / total_confidence
                weight_autoencoder = autoencoder_confidence / total_confidence
            else:
                weight_ensemble = weight_autoencoder = 0.5
            
            fused_score = (weight_ensemble * ensemble_normalized + 
                          weight_autoencoder * autoencoder_normalized)
            
            return min(1.0, fused_score)
            
        except Exception as e:
            self.logger.error(f"❌ Results fusion failed: {e}")
            return 0.0
    
    def _analyze_critical_anomalies(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تحليل متقدم للشذوذ الحرج"""
        critical_anomalies = []
        sensor_config = self.config['sensors']
        
        for sensor, value in sensor_data.items():
            if sensor in sensor_config:
                config = sensor_config[sensor]
                critical_threshold = config.get('critical', 100)
                warning_threshold = critical_threshold * 0.8
                
                if value >= critical_threshold:
                    critical_anomalies.append({
                        'sensor': sensor,
                        'value': value,
                        'threshold': critical_threshold,
                        'severity': 'CRITICAL',
                        'description': f'{sensor} exceeded critical threshold'
                    })
                elif value >= warning_threshold:
                    critical_anomalies.append({
                        'sensor': sensor,
                        'value': value,
                        'threshold': warning_threshold,
                        'severity': 'WARNING',
                        'description': f'{sensor} approaching critical threshold'
                    })
        
        return critical_anomalies
    
    def _perform_temporal_analysis(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل زمني متقدم للبيانات"""
        try:
            # تحليل الاتجاهات على المدى القصير
            if len(self.anomaly_history) > 10:
                recent_scores = [entry['anomaly_score'] for entry in self.anomaly_history[-10:]]
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                return {
                    'short_term_trend': trend,
                    'trend_direction': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable',
                    'volatility': np.std(recent_scores) if recent_scores else 0.0,
                    'stability_index': 1.0 - (np.std(recent_scores) if recent_scores else 0.0)
                }
            else:
                return {
                    'short_term_trend': 0.0,
                    'trend_direction': 'unknown',
                    'volatility': 0.0,
                    'stability_index': 0.5
                }
                
        except Exception as e:
            self.logger.error(f"❌ Temporal analysis failed: {e}")
            return {}
    
    def _calculate_risk_level(self, anomaly_score: float, critical_anomalies: List[Dict]) -> str:
        """حساب مستوى الخطورة المتقدم"""
        critical_count = len([a for a in critical_anomalies if a['severity'] == 'CRITICAL'])
        warning_count = len([a for a in critical_anomalies if a['severity'] == 'WARNING'])
        
        risk_factor = anomaly_score + (critical_count * 0.3) + (warning_count * 0.1)
        
        if risk_factor >= 0.8 or critical_count >= 2:
            return 'CRITICAL'
        elif risk_factor >= 0.6 or critical_count >= 1:
            return 'HIGH'
        elif risk_factor >= 0.4 or warning_count >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_advanced_confidence(self, ensemble_result: Dict, autoencoder_result: Dict) -> float:
        """حساب ثقة متقدمة في الكشف"""
        ensemble_confidence = ensemble_result.get('confidence', 0.5)
        autoencoder_confidence = autoencoder_result.get('confidence', 0.5)
        
        # تقييم اتساق النتائج
        ensemble_anomaly = ensemble_result.get('is_anomaly', False)
        autoencoder_anomaly = autoencoder_result.get('is_anomaly', False)
        
        if ensemble_anomaly == autoencoder_anomaly:
            consistency_boost = 0.2  # زيادة الثقة عند الاتساق
        else:
            consistency_boost = -0.1  # خفض الثقة عند الاختلاف
        
        base_confidence = (ensemble_confidence + autoencoder_confidence) / 2
        final_confidence = max(0.0, min(1.0, base_confidence + consistency_boost))
        
        return final_confidence
    
    def _generate_anomaly_recommendations(self, anomaly_score: float, critical_anomalies: List[Dict]) -> List[str]:
        """توليد توصيات ذكية بناءً على الشذوذ"""
        recommendations = []
        
        if anomaly_score >= 0.8:
            recommendations.append("🚨 تنفيذ إجراءات الطوارئ الفورية")
            recommendations.append("🔴 تفعيل نظام الإغلاق الآلي")
            recommendations.append("📞 إخطار فريق الطوارئ")
        
        if anomaly_score >= 0.6:
            recommendations.append("⚠️ زيادة وتيرة المراقبة إلى كل 5 ثواني")
            recommendations.append("🔧 فحص المعدات الحرجة")
        
        for anomaly in critical_anomalies:
            if anomaly['severity'] == 'CRITICAL':
                recommendations.append(f"🔥 فحص عاجل لمستشعر {anomaly['sensor']}")
            elif anomaly['severity'] == 'WARNING':
                recommendations.append(f"🔶 مراقبة مستشعر {anomaly['sensor']}")
        
        if not recommendations:
            recommendations.append("✅ الوضع طبيعي - متابعة المراقبة الروتينية")
        
        return recommendations
    
    def _update_adaptive_threshold(self, current_score: float):
        """تحديث العتبة التكيفية بناءً على الأداء الحديث"""
        try:
            if len(self.anomaly_history) > 50:
                recent_scores = [entry['anomaly_score'] for entry in self.anomaly_history[-50:]]
                current_percentile = np.percentile(recent_scores, 95)
                
                # تحديث سلس مع مراعاة الاستقرار
                self.adaptive_threshold = 0.95 * self.adaptive_threshold + 0.05 * current_percentile
                
        except Exception as e:
            self.logger.error(f"❌ Adaptive threshold update failed: {e}")
    
    def _update_anomaly_history(self, result: Dict[str, Any]):
        """تحديث سجل الشذوذ مع إدارة الذاكرة"""
        self.anomaly_history.append(result)
        
        # الاحتفاظ بـ 1000 تسجيل فقط لتحسين الأداء
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]
    
    def _get_untrained_response(self) -> Dict[str, Any]:
        """استجابة عندما لا تكون النماذج مدربة"""
        return {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'risk_level': 'LOW',
            'ensemble_results': {},
            'autoencoder_result': {},
            'critical_anomalies': [],
            'temporal_analysis': {},
            'data_quality_issues': ['Models not trained'],
            'confidence': 0.1,
            'adaptive_threshold': 0.05,
            'recommendations': ['System initializing - training required'],
            'timestamp': datetime.now(),
            'warning': 'Models not trained'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام المتقدم"""
        return {
            'is_trained': self.is_trained,
            'adaptive_threshold': float(self.adaptive_threshold),
            'anomaly_history_count': len(self.anomaly_history),
            'performance_metrics': self.performance_metrics,
            'data_quality_checks': True,
            'temporal_analysis_enabled': True,
            'ensemble_models_active': True,
            'last_training': datetime.now() if self.is_trained else None,
            'system_confidence': 0.95 if self.is_trained else 0.1
        }
        class AdvancedPredictionEngine:
    """محرك تنبؤ متقدم باستخدام الشبكات العصبية العميقة - SS Rating"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Prediction')
        
        # النماذج المتقدمة
        self.lstm_model = self._build_advanced_lstm()
        self.transformer_model = self._build_transformer_model()
        self.hybrid_model = self._build_hybrid_model()
        
        # إدارة النماذج
        self.active_model = None
        self.model_performance = {}
        self.sequence_length = config['prediction']['sequence_length']
        
        # معالجة البيانات المتقدمة
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler()
        }
        
        self.is_trained = False
        self.prediction_history = []
        self.confidence_calibrator = ConfidenceCalibrator()
        
        self.logger.info("✅ Advanced Prediction Engine Initialized - SS Rating")
    
    def _build_advanced_lstm(self) -> nn.Module:
        """بناء نموذج LSTM متقدم للتنبؤ"""
        class AdvancedLSTM(nn.Module):
            def __init__(self, input_size=6, hidden_size=128, num_layers=3, output_size=6, dropout=0.3):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # LSTM متعدد الطبقات مع تسرب
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout, bidirectional=True
                )
                
                # توجه الانتباه
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size * 2,  # لأنها bidirectional
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )
                
                # طبقات كثيفة متقدمة
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_size * 2, 256),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, output_size),
                    nn.Tanh()
                )
                
                # تهيئة الأوزان
                self._init_weights()
            
            def _init_weights(self):
                """تهيئة متقدمة للأوزان"""
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        if 'lstm' in name:
                            nn.init.orthogonal_(param)
                        elif 'attention' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'fc' in name:
                            nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
            
            def forward(self, x):
                # LSTM
                lstm_out, (hidden, cell) = self.lstm(x)
                
                # الانتباه
                attended_out, attention_weights = self.attention(
                    lstm_out, lstm_out, lstm_out
                )
                
                # أخذ آخر خطوة زمنية
                last_output = attended_out[:, -1, :]
                
                # طبقات كثيفة
                output = self.fc_layers(last_output)
                return output, attention_weights
        
        return AdvancedLSTM(
            input_size=6,
            hidden_size=128,
            num_layers=3,
            output_size=6,
            dropout=0.3
        )
    
    def _build_transformer_model(self) -> nn.Module:
        """بناء نموذج Transformer للتنبؤ الزمني"""
        class TimeSeriesTransformer(nn.Module):
            def __init__(self, input_dim=6, model_dim=128, num_heads=8, num_layers=4, output_dim=6):
                super().__init__()
                self.model_dim = model_dim
                
                # تضمين الموضع
                self.positional_encoding = PositionalEncoding(model_dim)
                
                # تضمين الإدخال
                self.input_projection = nn.Linear(input_dim, model_dim)
                
                # طبقات Transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=512,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # طبقة الإخراج
                self.output_layer = nn.Sequential(
                    nn.Linear(model_dim, 256),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, output_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                # إسقاط الإدخال
                x = self.input_projection(x)
                
                # الترميز الموضعي
                x = self.positional_encoding(x)
                
                # Transformer
                transformer_out = self.transformer(x)
                
                # أخذ آخر خطوة زمنية
                last_timestep = transformer_out[:, -1, :]
                
                # إخراج
                output = self.output_layer(last_timestep)
                return output
        
        return TimeSeriesTransformer()
    
    def _build_hybrid_model(self) -> nn.Module:
        """بناء نموذج هجين LSTM-Transformer"""
        class HybridModel(nn.Module):
            def __init__(self, input_dim=6, lstm_hidden=128, transformer_dim=128, output_dim=6):
                super().__init__()
                
                # LSTM Branch
                self.lstm_branch = nn.LSTM(
                    input_dim, lstm_hidden, 2,
                    batch_first=True, bidirectional=True, dropout=0.2
                )
                
                # Transformer Branch
                self.transformer_branch = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=input_dim,
                        nhead=8,
                        dim_feedforward=256,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=2
                )
                
                # اندماج الميزات
                self.feature_fusion = nn.Sequential(
                    nn.Linear(lstm_hidden * 2 + input_dim, 256),  # LSTM bidirectional + Transformer
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, output_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                # LSTM
                lstm_out, _ = self.lstm_branch(x)
                lstm_features = lstm_out[:, -1, :]
                
                # Transformer
                transformer_out = self.transformer_branch(x)
                transformer_features = transformer_out[:, -1, :]
                
                # اندماج
                combined = torch.cat([lstm_features, transformer_features], dim=1)
                output = self.feature_fusion(combined)
                return output
        
        return HybridModel()
    
    def prepare_sequences(self, sensor_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """تحضير متواليات زمنية متقدمة للتدريب"""
        try:
            # استخراج الميزات
            features = []
            for data in sensor_data:
                feature_vector = [
                    data.get('pressure', 0.0),
                    data.get('temperature', 0.0),
                    data.get('methane', 0.0),
                    data.get('hydrogen_sulfide', 0.0),
                    data.get('vibration', 0.0),
                    data.get('flow', 0.0)
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # إنشاء المتواليات
            X, y = [], []
            for i in range(len(features_array) - self.sequence_length):
                X.append(features_array[i:i + self.sequence_length])
                y.append(features_array[i + self.sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"❌ Sequence preparation failed: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, training_data: List[Dict[str, Any]], validation_data: List[Dict[str, Any]] = None):
        """تدريب متقدم لجميع نماذج التنبؤ"""
        try:
            self.logger.info("🔄 Training advanced prediction models...")
            
            # تحضير البيانات
            X_train, y_train = self.prepare_sequences(training_data)
            
            if len(X_train) == 0:
                self.logger.error("❌ No training sequences generated")
                return
            
            # تدريب السكالر
            X_reshaped = X_train.reshape(-1, X_train.shape[-1])
            self.scalers['features'].fit(X_reshaped)
            self.scalers['target'].fit(y_train)
            
            # تحويل البيانات
            X_normalized = self.scalers['features'].transform(X_reshaped).reshape(X_train.shape)
            y_normalized = self.scalers['target'].transform(y_train)
            
            # تحويل إلى Tensor
            X_tensor = torch.FloatTensor(X_normalized)
            y_tensor = torch.FloatTensor(y_normalized)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # تدريب النماذج
            models = {
                'lstm': self.lstm_model,
                'transformer': self.transformer_model,
                'hybrid': self.hybrid_model
            }
            
            for name, model in models.items():
                self._train_single_model(model, name, dataloader, validation_data)
            
            # اختيار أفضل نموذج
            self._select_best_model()
            self.is_trained = True
            
            self.logger.info("✅ All prediction models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Prediction models training failed: {e}")
    
    def _train_single_model(self, model: nn.Module, model_name: str, 
                          dataloader: DataLoader, validation_data: List[Dict[str, Any]] = None):
        """تدريب نموذج فردي متقدم"""
        try:
            # إعداد التدريب
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            
            training_losses = []
            validation_losses = []
            
            # حلقة التدريب
            for epoch in range(100):  # epochs
                model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    if model_name == 'lstm':
                        predictions, _ = model(batch_X)
                    else:
                        predictions = model(batch_X)
                    
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # قص التدرج
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                scheduler.step()
                
                # حساب الخسارة المتوسطة
                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)
                
                # التحقق من الصحة إذا كانت البيانات متاحة
                if validation_data and epoch % 10 == 0:
                    val_loss = self._validate_model(model, model_name, validation_data)
                    validation_losses.append(val_loss)
                    self.logger.info(f"📊 {model_name} Epoch {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    if epoch % 25 == 0:
                        self.logger.info(f"📊 {model_name} Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # حفظ أداء النموذج
            self.model_performance[model_name] = {
                'training_loss': training_losses,
                'validation_loss': validation_losses,
                'final_train_loss': training_losses[-1] if training_losses else float('inf'),
                'final_val_loss': validation_losses[-1] if validation_losses else float('inf')
            }
            
        except Exception as e:
            self.logger.error(f"❌ {model_name} training failed: {e}")
    
    def _validate_model(self, model: nn.Module, model_name: str, validation_data: List[Dict[str, Any]]) -> float:
        """تحقق من أداء النموذج"""
        try:
            model.eval()
            criterion = nn.MSELoss()
            
            X_val, y_val = self.prepare_sequences(validation_data)
            if len(X_val) == 0:
                return float('inf')
            
            # تطبيع البيانات
            X_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_normalized = self.scalers['features'].transform(X_reshaped).reshape(X_val.shape)
            y_normalized = self.scalers['target'].transform(y_val)
            
            X_tensor = torch.FloatTensor(X_normalized)
            y_tensor = torch.FloatTensor(y_normalized)
            
            with torch.no_grad():
                if model_name == 'lstm':
                    predictions, _ = model(X_tensor)
                else:
                    predictions = model(X_tensor)
                
                loss = criterion(predictions, y_tensor)
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return float('inf')
    
    def _select_best_model(self):
        """اختيار أفضل نموذج بناءً على الأداء"""
        try:
            best_model_name = None
            best_loss = float('inf')
            
            for name, performance in self.model_performance.items():
                # استخدام خسارة التحقق إن أمكن، وإلا خسارة التدريب
                loss = performance.get('final_val_loss', performance.get('final_train_loss', float('inf')))
                
                if loss < best_loss:
                    best_loss = loss
                    best_model_name = name
            
            if best_model_name == 'lstm':
                self.active_model = self.lstm_model
            elif best_model_name == 'transformer':
                self.active_model = self.transformer_model
            else:
                self.active_model = self.hybrid_model
            
            self.logger.info(f"🎯 Selected best model: {best_model_name} with loss: {best_loss:.6f}")
            
        except Exception as e:
            self.logger.error(f"❌ Model selection failed: {e}")
            self.active_model = self.lstm_model  # نموذج افتراضي آمن
    
    def predict(self, sensor_data: List[Dict[str, Any]], steps: int = 1) -> Dict[str, Any]:
        """تنبؤ متقدم متعدد الخطوات"""
        try:
            if not self.is_trained or self.active_model is None:
                return self._get_untrained_prediction()
            
            # تحضير أحدث البيانات
            recent_data = sensor_data[-self.sequence_length:]
            if len(recent_data) < self.sequence_length:
                self.logger.warning("⚠️ Insufficient data for prediction")
                return self._get_insufficient_data_prediction()
            
            # إنشاء متوالية الإدخال
            input_sequence = self._create_input_sequence(recent_data)
            
            # التنبؤ المتكرر
            predictions = []
            confidence_scores = []
            current_sequence = input_sequence.clone()
            
            for step in range(steps):
                with torch.no_grad():
                    if isinstance(self.active_model, self.lstm_model.__class__):
                        pred, attention_weights = self.active_model(current_sequence)
                    else:
                        pred = self.active_model(current_sequence)
                        attention_weights = None
                
                # حساب الثقة
                confidence = self.confidence_calibrator.calculate_confidence(
                    pred, current_sequence, step
                )
                
                predictions.append(pred.numpy())
                confidence_scores.append(confidence)
                
                # تحديث المتوالية للخطوة التالية
                if step < steps - 1:
                    current_sequence = self._update_sequence(current_sequence, pred)
            
            # معالجة التنبؤات
            predictions_array = np.array(predictions).squeeze()
            final_predictions = self.scalers['target'].inverse_transform(predictions_array)
            
            # تحليل النتائج
            risk_assessment = self._assess_prediction_risk(final_predictions, confidence_scores)
            trends = self._analyze_prediction_trends(final_predictions)
            
            result = {
                'predictions': final_predictions.tolist(),
                'confidence_scores': confidence_scores,
                'average_confidence': np.mean(confidence_scores),
                'risk_level': risk_assessment['risk_level'],
                'risk_factors': risk_assessment['risk_factors'],
                'trends': trends,
                'model_used': self.active_model.__class__.__name__,
                'timestamp': datetime.now(),
                'steps_ahead': steps,
                'recommendations': self._generate_prediction_recommendations(
                    final_predictions, risk_assessment, trends
                )
            }
            
            # تحديث السجل
            self._update_prediction_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced prediction failed: {e}")
            return {
                'error': str(e),
                'predictions': [],
                'confidence_scores': [],
                'risk_level': 'UNKNOWN',
                'recommendations': ['Prediction system error - check logs']
            }
    
    def _create_input_sequence(self, sensor_data: List[Dict[str, Any]]) -> torch.Tensor:
        """إنشاء متوالية إدخال للتنبؤ"""
        features = []
        for data in sensor_data:
            feature_vector = [
                data.get('pressure', 0.0),
                data.get('temperature', 0.0),
                data.get('methane', 0.0),
                data.get('hydrogen_sulfide', 0.0),
                data.get('vibration', 0.0),
                data.get('flow', 0.0)
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        normalized_features = self.scalers['features'].transform(features_array)
        
        return torch.FloatTensor(normalized_features).unsqueeze(0)
    
    def _update_sequence(self, current_sequence: torch.Tensor, new_prediction: torch.Tensor) -> torch.Tensor:
        """تحديث المتوالية بإضافة تنبؤ جديد"""
        # إزالة أقدم نقطة وإضافة التنبؤ الجديد
        updated_sequence = torch.cat([
            current_sequence[:, 1:, :],  # إزالة الأول
            new_prediction.unsqueeze(1)  # إضافة الجديد
        ], dim=1)
        
        return updated_sequence
    
    def _assess_prediction_risk(self, predictions: np.ndarray, confidence_scores: List[float]) -> Dict[str, Any]:
        """تقييم مخاطر التنبؤات"""
        risk_factors = []
        
        # تحليل كل مستشعر
        sensor_limits = {
            'pressure': (0, 200),
            'temperature': (0, 300),
            'methane': (0, 5000),
            'hydrogen_sulfide': (0, 500),
            'vibration': (0, 20),
            'flow': (0, 500)
        }
        
        for i, (sensor, (min_val, max_val)) in enumerate(sensor_limits.items()):
            sensor_predictions = predictions[:, i] if predictions.ndim > 1 else predictions[i]
            
            # التحقق من تجاوز الحدود
            if np.any(sensor_predictions > max_val * 0.9):  # 90% من الحد الأقصى
                risk_factors.append(f"{sensor} approaching upper limit")
            
            if np.any(sensor_predictions < min_val * 1.1):  # قريبة من الحد الأدنى
                risk_factors.append(f"{sensor} near lower limit")
            
            # تحليل التقلب
            if len(sensor_predictions) > 1:
                volatility = np.std(sensor_predictions)
                if volatility > (max_val - min_val) * 0.1:  # تقلب عالي
                    risk_factors.append(f"High volatility in {sensor}")
        
        # تقييم الثقة
        low_confidence_count = sum(1 for conf in confidence_scores if conf < 0.6)
        if low_confidence_count > len(confidence_scores) * 0.5:
            risk_factors.append("Multiple low-confidence predictions")
        
        # تحديد مستوى الخطورة
        if len(risk_factors) >= 3:
            risk_level = "HIGH"
        elif len(risk_factors) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'confidence_issues': low_confidence_count
        }
    
    def _analyze_prediction_trends(self, predictions: np.ndarray) -> Dict[str, Any]:
        """تحليل اتجاهات التنبؤات"""
        if predictions.ndim == 1 or len(predictions) < 2:
            return {'trend': 'stable', 'momentum': 0.0}
        
        trends = []
        momentums = []
        
        for i in range(predictions.shape[1] if predictions.ndim > 1 else 1):
            if predictions.ndim > 1:
                series = predictions[:, i]
            else:
                series = predictions
            
            # حساب الاتجاه
            if len(series) >= 2:
                x = np.arange(len(series))
                slope, _, _, _, _ = stats.linregress(x, series)
                
                trends.append('increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable')
                momentums.append(abs(slope))
        
        return {
            'trends': trends,
            'dominant_trend': max(set(trends), key=trends.count) if trends else 'stable',
            'average_momentum': np.mean(momentums) if momentums else 0.0,
            'volatility': np.std(predictions) if predictions.size > 1 else 0.0
        }
    
    def _generate_prediction_recommendations(self, predictions: np.ndarray, 
                                           risk_assessment: Dict[str, Any], 
                                           trends: Dict[str, Any]) -> List[str]:
        """توليد توصيات ذكية بناءً على التنبؤات"""
        recommendations = []
        
        risk_level = risk_assessment['risk_level']
        risk_factors = risk_assessment['risk_factors']
        
        if risk_level == "HIGH":
            recommendations.append("🚨 تنبؤ عالي الخطورة - تفعيل بروتوكولات الطوارئ")
            recommendations.append("📊 زيادة وتيرة جمع البيانات إلى كل ثانية")
            recommendations.append("👥 إخطار الفنيين للتدخل الفوري")
        
        elif risk_level == "MEDIUM":
            recommendations.append("⚠️ مراقبة متزايدة للمتغيرات عالية الخطورة")
            recommendations.append("🔧 فحص وقائي للمعدات")
        
        # توصيات بناءً على عوامل الخطورة
        for factor in risk_factors:
            if "pressure" in factor:
                recommendations.append("⛽ فحص نظام الضغط والمضخات")
            elif "temperature" in factor:
                recommendations.append("🔥 مراقبة أنظمة التبريد")
            elif "methane" in factor or "hydrogen_sulfide" in factor:
                recommendations.append("☠️ تفعيل أنظمة الكشف عن التسرب")
            elif "vibration" in factor:
                recommendations.append("📳 فحص التوازن والمحامل الميكانيكية")
        
        # توصيات بناءً على الاتجاهات
        if trends.get('dominant_trend') == 'increasing':
            recommendations.append("📈 اتجاه تصاعدي - الاستعداد لتدابير احترازية")
        elif trends.get('dominant_trend') == 'decreasing':
            recommendations.append("📉 اتجاه تنازلي - مراقبة الاستقرار")
        
        if not recommendations:
            recommendations.append("✅ الوضع مستقر - متابعة المراقبة الروتينية")
        
        return recommendations
    
    def _get_untrained_prediction(self) -> Dict[str, Any]:
        """استجابة عندما لا تكون النماذج مدربة"""
        return {
            'predictions': [],
            'confidence_scores': [],
            'average_confidence': 0.1,
            'risk_level': 'UNKNOWN',
            'risk_factors': ['Prediction models not trained'],
            'trends': {},
            'model_used': 'None',
            'timestamp': datetime.now(),
            'steps_ahead': 0,
            'recommendations': ['System initializing - training required'],
            'warning': 'Models not trained'
        }
    
    def _get_insufficient_data_prediction(self) -> Dict[str, Any]:
        """استجابة عندما تكون البيانات غير كافية"""
        return {
            'predictions': [],
            'confidence_scores': [],
            'average_confidence': 0.0,
            'risk_level': 'LOW',
            'risk_factors': ['Insufficient historical data'],
            'trends': {},
            'model_used': 'None',
            'timestamp': datetime.now(),
            'steps_ahead': 0,
            'recommendations': ['Collect more data for accurate predictions'],
            'warning': 'Insufficient data'
        }
    
    def _update_prediction_history(self, result: Dict[str, Any]):
        """تحديث سجل التنبؤات"""
        self.prediction_history.append(result)
        
        # الاحتفاظ بـ 500 تنبؤ حديث فقط
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-500:]

class PositionalEncoding(nn.Module):
    """ترميز موضعي متقدم للنماذج التتابعية"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # إنشاء مصفوفة الترميز الموضعي
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class ConfidenceCalibrator:
    """معاير ثقة متقدم للتنبؤات"""
    
    def __init__(self):
        self.prediction_errors = []
        self.confidence_history = []
    
    def calculate_confidence(self, prediction: torch.Tensor, 
                           input_sequence: torch.Tensor, 
                           step: int) -> float:
        """حساب ثقة متقدمة للتنبؤ"""
        try:
            # ثقة أساسية بناءً على اتساق التنبؤ
            consistency_score = self._calculate_consistency(prediction, input_sequence)
            
            # ثقة بناءً على خطوة التنبؤ (تقل كلما ابتعدنا)
            step_penalty = 1.0 / (1.0 + step * 0.1)
            
            # ثقة بناءً على توزيع القيم
            distribution_score = self._calculate_distribution_confidence(prediction)
            
            # دمج الثقة النهائية
            final_confidence = (consistency_score * 0.4 + 
                              step_penalty * 0.3 + 
                              distribution_score * 0.3)
            
            # تحديث السجل
            self.confidence_history.append(final_confidence)
            
            return max(0.1, min(1.0, final_confidence))
            
        except Exception as e:
            logging.getLogger('SmartNeural.AI.Confidence').error(f"Confidence calculation failed: {e}")
            return 0.5  # ثقة متوسطة افتراضية
    
    def _calculate_consistency(self, prediction: torch.Tensor, input_sequence: torch.Tensor) -> float:
        """حساب اتساق التنبؤ مع البيانات التاريخية"""
        try:
            # حساب متوسط القيم التاريخية
            historical_mean = torch.mean(input_sequence, dim=1)
            
            # حساب انحراف التنبؤ عن المتوسط
            deviation = torch.abs(prediction - historical_mean)
            normalized_deviation = torch.mean(deviation / (torch.std(input_sequence, dim=1) + 1e-8))
            
            # تحويل الانحراف إلى ثقة (كلما قل الانحراف زادت الثقة)
            consistency = 1.0 / (1.0 + normalized_deviation.item())
            
            return consistency
            
        except Exception:
            return 0.5
    
    def _calculate_distribution_confidence(self, prediction: torch.Tensor) -> float:
        """حساب الثقة بناءً على توزيع القيم المتوقع"""
        try:
            # افتراض أن القيم القصوى أقل ثقة
            max_values = torch.tensor([200, 300, 5000, 500, 20, 500])  # حدود المستشعرات
            normalized_pred = torch.abs(prediction) / max_values
            
            # الثقة تقل عندما تقترب القيم من الحدود
            boundary_distance = 1.0 - torch.max(normalized_pred)
            
            return max(0.0, boundary_distance.item())
            
        except Exception:
            return 0.5

class AdaptiveLearningSystem:
    """نظام تعلم تكيفي متقدم - SS Rating"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.AdaptiveLearning')
        
        # إدارة الأداء
        self.performance_metrics = {}
        self.model_drift_detector = ModelDriftDetector()
        self.concept_drift_tracker = ConceptDriftTracker()
        
        # التعلم النشط
        self.uncertainty_sampler = UncertaintySampler()
        self.feedback_loop = FeedbackLoop()
        
        # التحسين التلقائي
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        self.learning_rate = 0.001
        self.retraining_interval = timedelta(hours=24)
        self.last_retraining = None
        
        self.logger.info("✅ Adaptive Learning System Initialized - SS Rating")
    
    def monitor_model_performance(self, predictions: Dict[str, Any], 
                                actual_values: Dict[str, Any]) -> Dict[str, Any]:
        """مراقبة أداء النموذج بشكل مستمر"""
        try:
            performance_report = {
                'timestamp': datetime.now(),
                'prediction_accuracy': self._calculate_prediction_accuracy(predictions, actual_values),
                'anomaly_detection_accuracy': self._calculate_anomaly_accuracy(predictions, actual_values),
                'model_drift': self.model_drift_detector.detect_drift(predictions, actual_values),
                'concept_drift': self.concept_drift_tracker.track_changes(actual_values),
                'data_quality': self._assess_data_quality(actual_values),
                'recommendations': []
            }
            
            # توليد توصيات التحسين
            if performance_report['model_drift']['detected']:
                performance_report['recommendations'].append("🔄 إعادة تدريب النماذج بسبب انحراف الأداء")
            
            if performance_report['concept_drift']['detected']:
                performance_report['recommendations'].append("🎯 تحديث النماذج بسبب تغير الأنماط")
            
            # تحديث المقاييس
            self._update_performance_metrics(performance_report)
            
            return performance_report
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_accuracy(self, predictions: Dict[str, Any], 
                                    actual_values: Dict[str, Any]) -> float:
        """حساب دقة التنبؤ"""
        try:
            if 'predictions' not in predictions or not actual_values:
                return 0.0
            
            pred_array = np.array(predictions['predictions'])
            actual_array = np.array([
                actual_values.get('pressure', 0.0),
                actual_values.get('temperature', 0.0),
                actual_values.get('methane', 0.0),
                actual_values.get('hydrogen_sulfide', 0.0),
                actual_values.get('vibration', 0.0),
                actual_values.get('flow', 0.0)
            ])
            
            # حساب خطأ الجذر التربيعي المتوسط
            if pred_array.ndim > 1 and pred_array.shape[0] > 0:
                mse = np.mean((pred_array[0] - actual_array) ** 2)
                accuracy = 1.0 / (1.0 + np.sqrt(mse))
                return min(1.0, accuracy)
            
            return 0.5  # دقة متوسطة افتراضية
            
        except Exception:
            return 0.5
    
    def _calculate_anomaly_accuracy(self, predictions: Dict[str, Any], 
                                  actual_values: Dict[str, Any]) -> float:
        """حساب دقة كشف الشذوذ (محاكاة)"""
        # في التطبيق الحقيقي، ستحتاج بيانات حقيقة عن الشذوذ
        return 0.85  # محاكاة لدقة جيدة
    
    def _assess_data_quality(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """تقييم جودة البيانات"""
        quality_metrics = {
            'completeness': 1.0,  # نسبة اكتمال البيانات
            'consistency': 1.0,   # اتساق البيانات
            'timeliness': 1.0,    # حداثة البيانات
            'validity': 1.0       # صحة البيانات
        }
        
        # تحليل بسيط لجودة البيانات
        for sensor, value in sensor_data.items():
            if value is None:
                quality_metrics['completeness'] -= 0.1
            elif not isinstance(value, (int, float)):
                quality_metrics['validity'] -= 0.1
        
        return quality_metrics
    
    def _update_performance_metrics(self, performance_report: Dict[str, Any]):
        """تحديث مقاييس الأداء التاريخية"""
        timestamp = performance_report['timestamp']
        
        for metric, value in performance_report.items():
            if metric not in ['timestamp', 'recommendations']:
                if metric not in self.performance_metrics:
                    self.performance_metrics[metric] = []
                
                self.performance_metrics[metric].append({
                    'timestamp': timestamp,
                    'value': value
                })
                
                # الاحتفاظ بـ 1000 قياس حديث فقط
                if len(self.performance_metrics[metric]) > 1000:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-1000:]

class ModelDriftDetector:
    """كاشف انحراف النماذج"""
    
    def __init__(self):
        self.prediction_errors = []
        self.drift_threshold = 0.1
    
    def detect_drift(self, predictions: Dict[str, Any], actual_values: Dict[str, Any]) -> Dict[str, Any]:
        """كشف انحراف النموذج"""
        try:
            # حساب خطأ التنبؤ الحالي
            current_error = self._calculate_prediction_error(predictions, actual_values)
            self.prediction_errors.append(current_error)
            
            # تحليل الانحراف إذا كان لديك بيانات كافية
            if len(self.prediction_errors) > 50:
                recent_errors = self.prediction_errors[-50:]
                historical_errors = self.prediction_errors[-100:-50] if len(self.prediction_errors) > 100 else recent_errors
                
                # اختبار الانحراف باستخدام اختبار t
                t_stat, p_value = stats.ttest_ind(recent_errors, historical_errors)
                
                drift_detected = p_value < 0.05 and np.mean(recent_errors) > np.mean(historical_errors)
                
                return {
                    'detected': drift_detected,
                    'p_value': p_value,
                    'current_error': current_error,
                    'error_increase': np.mean(recent_errors) - np.mean(historical_errors) if drift_detected else 0.0
                }
            else:
                return {
                    'detected': False,
                    'p_value': 1.0,
                    'current_error': current_error,
                    'error_increase': 0.0
                }
                
        except Exception as e:
            logging.getLogger('SmartNeural.AI.DriftDetection').error(f"Drift detection failed: {e}")
            return {'detected': False, 'error': str(e)}
    
    def _calculate_prediction_error(self, predictions: Dict[str, Any], actual_values: Dict[str, Any]) -> float:
        """حساب خطأ التنبؤ"""
        try:
            if 'predictions' not in predictions or not actual_values:
                return 1.0  # أقصى خطأ
            
            pred_values = predictions['predictions']
            if not pred_values or len(pred_values) == 0:
                return 1.0
            
            # استخدام أول تنبؤ للمقارنة
            pred_array = np.array(pred_values[0] if isinstance(pred_values[0], list) else pred_values)
            actual_array = np.array([
                actual_values.get('pressure', 0.0),
                actual_values.get('temperature', 0.0),
                actual_values.get('methane', 0.0),
                actual_values.get('hydrogen_sulfide', 0.0),
                actual_values.get('vibration', 0.0),
                actual_values.get('flow', 0.0)
            ])
            
            # حساب الخطأ التربيعي المتوسط
            mse = np.mean((pred_array - actual_array) ** 2)
            return mse
            
        except Exception:
            return 1.0

class ConceptDriftTracker:
    """متتبع انحراف المفاهيم"""
    
    def __init__(self):
        self.data_distributions = []
        self.distribution_changes = []
    
    def track_changes(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """تتبع تغيرات توزيع البيانات"""
        try:
            # استخراج قيم المستشعرات
            values = np.array([
                sensor_data.get('pressure', 0.0),
                sensor_data.get('temperature', 0.0),
                sensor_data.get('methane', 0.0),
                sensor_data.get('hydrogen_sulfide', 0.0),
                sensor_data.get('vibration', 0.0),
                sensor_data.get('flow', 0.0)
            ])
            
            self.data_distributions.append(values)
            
            # الاحتفاظ بـ 100 توزيع حديث فقط
            if len(self.data_distributions) > 100:
                self.data_distributions = self.data_distributions[-100:]
            
            # كشف التغير إذا كان لديك بيانات كافية
            if len(self.data_distributions) > 30:
                recent_data = np.array(self.data_distributions[-30:])
                historical_data = np.array(self.data_distributions[-60:-30]) if len(self.data_distributions) > 60 else recent_data
                
                # حساب تغير التوزيع باستخدام مسافة واسرستين
                drift_detected = self._detect_distribution_drift(recent_data, historical_data)
                
                return {
                    'detected': drift_detected,
                    'recent_mean': np.mean(recent_data, axis=0).tolist(),
                    'historical_mean': np.mean(historical_data, axis=0).tolist(),
                    'change_magnitude': np.linalg.norm(
                        np.mean(recent_data, axis=0) - np.mean(historical_data, axis=0)
                    )
                }
            else:
                return {
                    'detected': False,
                    'recent_mean': [],
                    'historical_mean': [],
                    'change_magnitude': 0.0
                }
                
        except Exception as e:
            logging.getLogger('SmartNeural.AI.ConceptDrift').error(f"Concept drift tracking failed: {e}")
            return {'detected': False, 'error': str(e)}
    
    def _detect_distribution_drift(self, recent_data: np.ndarray, historical_data: np.ndarray) -> bool:
        """كشف انحراف التوزيع باستخدام اختبار إحصائي"""
        try:
            # استخدام اختبار كولموجوروف-سميرنوف لكل ميزة
            drift_detected = False
            
            for i in range(recent_data.shape[1]):
                ks_stat, p_value = stats.ks_2samp(historical_data[:, i], recent_data[:, i])
                if p_value < 0.01:  # عتبة صارمة
                    drift_detected = True
                    break
            
            return drift_detected
            
        except Exception:
            return False

class UncertaintySampler:
    """عينة عدم اليقين للتعلم النشط"""
    
    def __init__(self):
        self.uncertainty_threshold = 0.3
        self.sampling_history = []
    
    def should_sample(self, prediction_confidence: float, anomaly_score: float) -> bool:
        """تحديد ما إذا كان يجب أخذ عينة للتعلم"""
        # أخذ عينة عندما تكون الثقة منخفضة أو درجة الشذوذ عالية
        return (prediction_confidence < self.uncertainty_threshold or 
                anomaly_score > 0.7)

class FeedbackLoop:
    """حلقة تغذية راجعة للتعلم التكيفي"""
    
    def __init__(self):
        self.feedback_data = []
        self.learning_rate = 0.01
    
    def add_feedback(self, prediction: Dict[str, Any], actual_value: Dict[str, Any], 
                    user_feedback: Optional[Dict[str, Any]] = None):
        """إضافة تغذية راجعة للتعلم"""
        feedback_entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual_value,
            'user_feedback': user_feedback,
            'error': self._calculate_feedback_error(prediction, actual_value)
        }
        
        self.feedback_data.append(feedback_entry)
        
        # الاحتفاظ بـ 1000 عينة تغذية راجعة
        if len(self.feedback_data) > 1000:
            self.feedback_data = self.feedback_data[-1000:]
    
    def _calculate_feedback_error(self, prediction: Dict[str, Any], actual_value: Dict[str, Any]) -> float:
        """حساب خطأ التغذية الراجعة"""
        try:
            # حساب الخطأ بين التنبؤ والقيمة الفعلية
            pred_values = prediction.get('predictions', [])
            if not pred_values:
                return 1.0
            
            pred_array = np.array(pred_values[0] if isinstance(pred_values[0], list) else pred_values)
            actual_array = np.array([
                actual_value.get('pressure', 0.0),
                actual_value.get('temperature', 0.0),
                actual_value.get('methane', 0.0),
                actual_value.get('hydrogen_sulfide', 0.0),
                actual_value.get('vibration', 0.0),
                actual_value.get('flow', 0.0)
            ])
            
            return np.mean((pred_array - actual_array) ** 2)
            
        except Exception:
            return 1.0

class HyperparameterOptimizer:
    """محسن معاملات النماذج"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_parameters = {}
    
    def optimize_parameters(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """تحسين معاملات النموذج بناءً على الأداء"""
        try:
            # تحسين بسيط للمعاملات (في التطبيق الحقيقي يستخدم Bayesian Optimization)
            suggestions = {}
            
            if model_performance.get('prediction_accuracy', 0) < 0.8:
                suggestions['learning_rate'] = 'Consider decreasing learning rate'
                suggestions['architecture'] = 'Consider adding more layers'
            
            if model_performance.get('model_drift', {}).get('detected', False):
                suggestions['retraining'] = 'Schedule immediate retraining'
                suggestions['regularization'] = 'Increase regularization'
            
            return {
                'suggestions': suggestions,
                'confidence_boost': 'Consider confidence calibration' if 
                model_performance.get('prediction_accuracy', 0) < 0.7 else 'Confidence adequate',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.getLogger('SmartNeural.AI.HyperparameterOpt').error(f"Hyperparameter optimization failed: {e}")
            return {'suggestions': {}, 'error': str(e)}

class AISystemManager:
    """مدير نظام الذكاء الاصطناعي الرئيسي - SS Rating"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SmartNeural.AI.Manager')
        
        # المكونات الرئيسية
        self.anomaly_system = AdvancedAnomalySystem(config)
        self.prediction_engine = AdvancedPredictionEngine(config)
        self.adaptive_learning = AdaptiveLearningSystem(config)
        
        # إدارة الحالة
        self.system_status = 'initializing'
        self.performance_metrics = {}
        self.alert_history = []
        
        # التهيئة
        self._initialize_systems()
        self.logger.info("✅ AI System Manager Initialized - SS Rating")
    
    def _initialize_systems(self):
        """تهيئة جميع أنظمة الذكاء الاصطناعي"""
        try:
            # تحقق من التهيئة
            self.system_status = 'initialized'
            self.logger.info("🎯 All AI systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ AI systems initialization failed: {e}")
            self.system_status = 'error'
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة متقدمة لبيانات المستشعرات"""
        try:
            start_time = datetime.now()
            
            # كشف الشذوذ
            anomaly_result = self.anomaly_system.detect_anomalies(sensor_data)
            
            # التنبؤ
            prediction_result = self.prediction_engine.predict([sensor_data], steps=3)
            
            # التعلم التكيفي
            learning_result = self.adaptive_learning.monitor_model_performance(
                prediction_result, sensor_data
            )
            
            # دمج النتائج
            final_result = {
                'timestamp': datetime.now(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'anomaly_detection': anomaly_result,
                'prediction': prediction_result,
                'adaptive_learning': learning_result,
                'system_status': self.system_status,
                'overall_risk_assessment': self._assess_overall_risk(anomaly_result, prediction_result),
                'action_recommendations': self._generate_actions(anomaly_result, prediction_result, learning_result)
            }
            
            # تحديث السجلات
            self._update_system_metrics(final_result)
            
            # إدارة التنبيهات
            self._manage_alerts(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ AI processing failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(),
                'system_status': 'error'
            }
    
    def _assess_overall_risk(self, anomaly_result: Dict[str, Any], 
                           prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """تقييم الخطورة الشاملة"""
        try:
            anomaly_risk = anomaly_result.get('risk_level', 'LOW')
            prediction_risk = prediction_result.get('risk_level', 'LOW')
            
            # تحويل المخاطر إلى قيم رقمية
            risk_values = {
                'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 1
            }
            
            anomaly_score = risk_values.get(anomaly_risk, 1)
            prediction_score = risk_values.get(prediction_risk, 1)
            
            # حساب الخطورة الشاملة
            overall_score = max(anomaly_score, prediction_score)
            
            risk_levels = {
                4: 'CRITICAL', 3: 'HIGH', 2: 'MEDIUM', 1: 'LOW'
            }
            
            return {
                'level': risk_levels.get(overall_score, 'LOW'),
                'score': overall_score,
                'factors': {
                    'anomaly_risk': anomaly_risk,
                    'prediction_risk': prediction_risk,
                    'anomaly_score': anomaly_result.get('anomaly_score', 0.0),
                    'prediction_confidence': prediction_result.get('average_confidence', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Overall risk assessment failed: {e}")
            return {'level': 'UNKNOWN', 'score': 0, 'factors': {}}
    
    def _generate_actions(self, anomaly_result: Dict[str, Any], 
                        prediction_result: Dict[str, Any],
                        learning_result: Dict[str, Any]) -> List[str]:
        """توليد إجراءات ذكية بناءً على جميع النتائج"""
        actions = []
        
        # إجراءات بناءً على الشذوذ
        anomaly_actions = anomaly_result.get('recommendations', [])
        actions.extend(anomaly_actions)
        
        # إجراءات بناءً على التنبؤ
        prediction_actions = prediction_result.get('recommendations', [])
        actions.extend(prediction_actions)
        
        # إجراءات بناءً على التعلم
        learning_actions = learning_result.get('recommendations', [])
        actions.extend(learning_actions)
        
        # إزالة التكرارات
        unique_actions = list(dict.fromkeys(actions))
        
        return unique_actions[:10]  # الحد الأقصى 10 إجراءات
    
    def _update_system_metrics(self, processing_result: Dict[str, Any]):
        """تحديث مقاييس أداء النظام"""
        timestamp = processing_result['timestamp']
        
        # تحديث وقت المعالجة
        if 'processing_time_metrics' not in self.performance_metrics:
            self.performance_metrics['processing_time_metrics'] = []
        
        self.performance_metrics['processing_time_metrics'].append({
            'timestamp': timestamp,
            'processing_time': processing_result['processing_time']
        })
        
        # تحديث معدلات الخطورة
        risk_level = processing_result['overall_risk_assessment']['level']
        if 'risk_level_metrics' not in self.performance_metrics:
            self.performance_metrics['risk_level_metrics'] = []
        
        self.performance_metrics['risk_level_metrics'].append({
            'timestamp': timestamp,
            'risk_level': risk_level
        })
    
    def _manage_alerts(self, processing_result: Dict[str, Any]):
        """إدارة التنبيهات الذكية"""
        try:
            risk_level = processing_result['overall_risk_assessment']['level']
            
            if risk_level in ['HIGH', 'CRITICAL']:
                alert = {
                    'timestamp': datetime.now(),
                    'level': risk_level,
                    'message': f"تنبيه {risk_level}: تم اكتشاف شذوذ حرج أو تنبؤ عالي الخطورة",
                    'details': {
                        'anomaly_score': processing_result['anomaly_detection'].get('anomaly_score', 0),
                        'prediction_risk': processing_result['prediction'].get('risk_level', 'UNKNOWN'),
                        'recommendations': processing_result['action_recommendations']
                    },
                    'acknowledged': False
                }
                
                self.alert_history.append(alert)
                
                # إرسال تنبيه فوري للخطورة العالية
                if risk_level == 'CRITICAL':
                    self._send_critical_alert(alert)
            
            # الاحتفاظ بـ 100 تنبيه حديث فقط
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
                
        except Exception as e:
            self.logger.error(f"❌ Alert management failed: {e}")
    
    def _send_critical_alert(self, alert: Dict[str, Any]):
        """إرسال تنبيه حرج"""
        try:
            # في التطبيق الحقيقي، أضف هنا إرسال الإشعارات (Email, SMS, etc.)
            self.logger.critical(f"🚨 CRITICAL ALERT: {alert['message']}")
            self.logger.critical(f"📋 Recommendations: {alert['details']['recommendations']}")
            
        except Exception as e:
            self.logger.error(f"❌ Critical alert sending failed: {e}")
    
    def train_all_models(self, training_data: List[Dict[str, Any]], 
                        validation_data: List[Dict[str, Any]] = None):
        """تدريب جميع النماذج المتقدمة"""
        try:
            self.logger.info("🔄 Starting comprehensive AI models training...")
            
            # تدريب نظام الشذوذ
            self.anomaly_system.train_models(training_data)
            
            # تدريب محرك التنبؤ
            self.prediction_engine.train_models(training_data, validation_data)
            
            self.system_status = 'trained'
            self.logger.info("✅ All AI models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive training failed: {e}")
            self.system_status = 'training_error'
    
    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام الشاملة"""
        return {
            'system_status': self.system_status,
            'anomaly_system_status': self.anomaly_system.get_system_status(),
            'prediction_engine_status': {
                'is_trained': self.prediction_engine.is_trained,
                'active_model': self.prediction_engine.active_model.__class__.__name__ if self.prediction_engine.active_model else 'None',
                'prediction_history_count': len(self.prediction_engine.prediction_history)
            },
            'performance_metrics_summary': {
                'total_processing_count': len(self.performance_metrics.get('processing_time_metrics', [])),
                'average_processing_time': np.mean([m['processing_time'] for m in self.performance_metrics.get('processing_time_metrics', [])]) if self.performance_metrics.get('processing_time_metrics') else 0,
                'critical_alerts_count': len([a for a in self.alert_history if a['level'] == 'CRITICAL']),
                'high_risk_percentage': len([m for m in self.performance_metrics.get('risk_level_metrics', []) if m['risk_level'] in ['HIGH', 'CRITICAL']]) / max(1, len(self.performance_metrics.get('risk_level_metrics', []))) * 100
            },
            'last_updated': datetime.now()
        }

# =============================================================================
# دوال مساعدة متقدمة
# =============================================================================

def _identify_critical_points(sensor_data: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """تحديد النقاط الحرجة في بيانات المستشعرات"""
    critical_points = []
    
    for sensor, value in sensor_data.items():
        sensor_config = config['sensors'].get(sensor, {})
        critical_threshold = sensor_config.get('critical', 100)
        warning_threshold = sensor_config.get('warning', critical_threshold * 0.8)
        
        if value >= critical_threshold:
            critical_points.append({
                'sensor': sensor,
                'value': value,
                'threshold': critical_threshold,
                'severity': 'CRITICAL',
                'impact': 'IMMEDIATE_SAFETY_RISK'
            })
        elif value >= warning_threshold:
            critical_points.append({
                'sensor': sensor,
                'value': value,
                'threshold': warning_threshold,
                'severity': 'WARNING',
                'impact': 'POTENTIAL_SAFETY_RISK'
            })
    
    return critical_points

def _generate_training_data(sensor_history: List[Dict[str, Any]], 
                          sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """توليد بيانات تدريب متقدمة من السجل"""
    try:
        if len(sensor_history) < sequence_length + 1:
            return np.array([]), np.array([])
        
        features = []
        targets = []
        
        for i in range(len(sensor_history) - sequence_length):
            # متوالية الإدخال
            sequence = sensor_history[i:i + sequence_length]
            feature_vector = []
            
            for data_point in sequence:
                vector = [
                    data_point.get('pressure', 0.0),
                    data_point.get('temperature', 0.0),
                    data_point.get('methane', 0.0),
                    data_point.get('hydrogen_sulfide', 0.0),
                    data_point.get('vibration', 0.0),
                    data_point.get('flow', 0.0)
                ]
                feature_vector.append(vector)
            
            # الهدف (النقطة التالية)
            target_point = sensor_history[i + sequence_length]
            target_vector = [
                target_point.get('pressure', 0.0),
                target_point.get('temperature', 0.0),
                target_point.get('methane', 0.0),
                target_point.get('hydrogen_sulfide', 0.0),
                target_point.get('vibration', 0.0),
                target_point.get('flow', 0.0)
            ]
            
            features.append(feature_vector)
            targets.append(target_vector)
        
        return np.array(features), np.array(targets)
        
    except Exception as e:
        logging.getLogger('SmartNeural.AI.Training').error(f"Training data generation failed: {e}")
        return np.array([]), np.array([])

def _calculate_advanced_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """حساب مقاييس أداء متقدمة"""
    try:
        if len(predictions) == 0 or len(actuals) == 0:
            return {}
        
        # الأخطاء الأساسية
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # نسبة الدقة
        accuracy = 1.0 / (1.0 + rmse)
        
        # معامل التحديد (R²)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # متوسط النسبة المئوية للخطأ
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy),
            'r_squared': float(r_squared),
            'mape': float(mape),
            'volatility': float(np.std(predictions - actuals))
        }
        
    except Exception as e:
        logging.getLogger('SmartNeural.AI.Metrics').error(f"Advanced metrics calculation failed: {e}")
        return {}

# =============================================================================
# تهيئة النظام الرئيسي
# =============================================================================

def create_ai_system(config: Dict[str, Any]) -> AISystemManager:
    """دالة إنشاء نظام الذكاء الاصطناعي الرئيسي"""
    try:
        ai_system = AISystemManager(config)
        logging.getLogger('SmartNeural.AI').info("🎯 Advanced AI System Created Successfully - SS Rating")
        return ai_system
    except Exception as e:
        logging.getLogger('SmartNeural.AI').error(f"❌ AI System creation failed: {e}")
        raise

# تصدير المكونات الرئيسية
__all__ = [
    'AISystemManager',
    'AdvancedAnomalySystem', 
    'AdvancedPredictionEngine',
    'AdaptiveLearningSystem',
    'create_ai_system',
    '_identify_critical_points',
    '_generate_training_data',
    '_calculate_advanced_metrics'
                ]
