import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import joblib
from pathlib import Path

# استيراد مكتبات التعلم الآلي
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logging.warning("ML libraries not available - prediction capabilities limited")

class PredictionModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

@dataclass
class PredictionResult:
    value: float
    confidence: float
    upper_bound: float
    lower_bound: float
    model_type: str
    timestamp: datetime
    features_used: List[str]

@dataclass
class ModelPerformance:
    mae: float
    mse: float
    r2: float
    last_trained: datetime
    training_samples: int

class AdvancedPredictionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importances: Dict[str, List] = {}
        self.prediction_history: List[Dict] = []
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.setup_logging()
        
        # إعدادات النماذج
        self.model_config = {
            'random_forest': {'n_estimators': 200, 'max_depth': 20},
            'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.1},
            'xgboost': {'n_estimators': 300, 'max_depth': 10},
            'lstm': {'units': 50, 'epochs': 100, 'batch_size': 32}
        }
    
    def setup_logging(self):
        """تهيئة التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, sensor_type: str, historical_data: pd.DataFrame, 
                   model_type: PredictionModelType = PredictionModelType.ENSEMBLE) -> bool:
        """تدريب نموذج متقدم للتنبؤ"""
        try:
            if not HAS_ML_LIBS:
                self.logger.warning("ML libraries not available - using simple prediction")
                return self._train_simple_model(sensor_type, historical_data)
            
            # تحضير البيانات
            X, y, feature_names = self.prepare_advanced_training_data(historical_data, sensor_type)
            
            if model_type == PredictionModelType.ENSEMBLE:
                # استخدام ensemble من النماذج
                model = self._train_ensemble_model(X, y)
            else:
                # تدريب نموذج فردي
                model = self._train_single_model(X, y, model_type)
            
            # تقييم الأداء
            performance = self.evaluate_model(model, X, y, model_type)
            
            # حفظ النموذج
            self.models[sensor_type] = {
                'model': model,
                'model_type': model_type,
                'feature_names': feature_names,
                'performance': performance,
                'last_trained': datetime.now()
            }
            
            self.logger.info(f"Model trained for {sensor_type} with {model_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model for {sensor_type}: {e}")
            return False
    
    def _train_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """تدريب ensemble model"""
        models = {}
        
        # تدريب multiple models
        if len(X) > 100:  #需要有足够的数据
            models['random_forest'] = RandomForestRegressor(
                **self.model_config['random_forest']
            ).fit(X, y)
            
            models['gradient_boosting'] = GradientBoostingRegressor(
                **self.model_config['gradient_boosting']
            ).fit(X, y)
        
        return {'ensemble': models, 'weights': {'random_forest': 0.4, 'gradient_boosting': 0.6}}
    
    def _train_single_model(self, X: np.ndarray, y: np.ndarray, 
                          model_type: PredictionModelType) -> Any:
        """تدريب نموذج فردي"""
        if model_type == PredictionModelType.RANDOM_FOREST:
            return RandomForestRegressor(**self.model_config['random_forest']).fit(X, y)
        
        elif model_type == PredictionModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(**self.model_config['gradient_boosting']).fit(X, y)
        
        elif model_type == PredictionModelType.XGBOOST and HAS_ML_LIBS:
            return xgb.XGBRegressor(**self.model_config['xgboost']).fit(X, y)
        
        elif model_type == PredictionModelType.LSTM and HAS_ML_LIBS:
            return self._train_lstm_model(X, y)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Sequential:
        """تدريب نموذج LSTM"""
        # إعادة تشكيل البيانات للـLSTM
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(30, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_reshaped, y, epochs=100, batch_size=32, verbose=0)
        
        return model
    
    def prepare_advanced_training_data(self, historical_data: pd.DataFrame, 
                                     target_sensor: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """تحضير بيانات تدريب متقدمة"""
        # إضافة features زمنية
        data = historical_data.copy()
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        # إضافة statistical features
        for sensor in data.columns:
            if sensor != target_sensor:
                data[f'{sensor}_rolling_mean_6'] = data[sensor].rolling(6).mean()
                data[f'{sensor}_rolling_std_6'] = data[sensor].rolling(6).std()
        
        # إضافة lag features
        for lag in [1, 2, 3, 6, 12]:
            data[f'{target_sensor}_lag_{lag}'] = data[target_sensor].shift(lag)
        
        # تنظيف البيانات
        data = data.dropna()
        
        # فصل features وtarget
        feature_columns = [col for col in data.columns if col != target_sensor]
        X = data[feature_columns].values
        y = data[target_sensor].values
        
        return X, y, feature_columns
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      model_type: PredictionModelType) -> ModelPerformance:
        """تقييم أداء النموذج"""
        if model_type == PredictionModelType.ENSEMBLE:
            predictions = self._ensemble_predict(model, X)
        else:
            predictions = model.predict(X)
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return ModelPerformance(
            mae=mae,
            mse=mse,
            r2=r2,
            last_trained=datetime.now(),
            training_samples=len(y)
        )
    
    def predict(self, sensor_type: str, recent_data: pd.DataFrame, 
               hours_ahead: int = 1) -> PredictionResult:
        """تنبؤ متقدم مع فترات زمنية متعددة"""
        try:
            if sensor_type not in self.models:
                raise ValueError(f"No model trained for {sensor_type}")
            
            model_info = self.models[sensor_type]
            model = model_info['model']
            model_type = model_info['model_type']
            
            # تحضير بيانات التنبؤ
            X_pred, feature_names = self.prepare_prediction_data(recent_data, sensor_type, model_info)
            
            if model_type == PredictionModelType.ENSEMBLE:
                prediction = self._ensemble_predict(model, X_pred)
                confidence = 0.85
            else:
                prediction = model.predict(X_pred.reshape(1, -1))[0]
                confidence = self._calculate_confidence(model, X_pred, model_type)
            
            # حساب فترات الثقة
            upper_bound, lower_bound = self._calculate_confidence_interval(prediction, confidence)
            
            result = PredictionResult(
                value=float(prediction),
                confidence=confidence,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                model_type=model_type.value,
                timestamp=datetime.now(),
                features_used=feature_names
            )
            
            self._save_prediction_history(sensor_type, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error for {sensor_type}: {e}")
            # Fallback to simple prediction
            return self._simple_fallback_prediction(sensor_type, recent_data)
    
    def _ensemble_predict(self, ensemble_model: Dict[str, Any], X: np.ndarray) -> float:
        """تنبؤ ensemble model"""
        predictions = []
        for model_name, model in ensemble_model['ensemble'].items():
            pred = model.predict(X.reshape(1, -1))[0]
            predictions.append(pred * ensemble_model['weights'][model_name])
        
        return sum(predictions)
    
    def _calculate_confidence(self, model: Any, X: np.ndarray, 
                            model_type: PredictionModelType) -> float:
        """حساب ثقة التنبؤ"""
        if model_type == PredictionModelType.RANDOM_FOREST:
            # استخدام out-of-bag score للثقة
            return min(0.95, model.oob_score_ if hasattr(model, 'oob_score_') else 0.8)
        
        elif hasattr(model, 'score'):
            # استخدام model score
            return min(0.95, max(0.6, model.score(X, X) if hasattr(model, 'score') else 0.7))
        
        else:
            return 0.75  # ثقة افتراضية
    
    def _calculate_confidence_interval(self, prediction: float, confidence: float) -> Tuple[float, float]:
        """حساب فترات الثقة"""
        margin = (1 - confidence) * prediction * 0.5
        upper_bound = prediction + margin
        lower_bound = prediction - margin
        return upper_bound, lower_bound
    
    def prepare_prediction_data(self, recent_data: pd.DataFrame, sensor_type: str,
                              model_info: Dict) -> Tuple[np.ndarray, List[str]]:
        """تحضير بيانات التنبؤ"""
        # إضافة features زمنية
        current_time = datetime.now()
        recent_data = recent_data.copy()
        recent_data['hour'] = current_time.hour
        recent_data['day_of_week'] = current_time.weekday()
        recent_data['month'] = current_time.month
        
        # استخدام features التي تم تدريب النموذج عليها
        feature_columns = model_info['feature_names']
        X_pred = recent_data[feature_columns].values.flatten()
        
        return X_pred, feature_columns
    
    def _save_prediction_history(self, sensor_type: str, result: PredictionResult) -> None:
        """حفظ سجل التنبؤات"""
        history_entry = {
            'sensor_type': sensor_type,
            'prediction': result.value,
            'confidence': result.confidence,
            'upper_bound': result.upper_bound,
            'lower_bound': result.lower_bound,
            'timestamp': result.timestamp,
            'model_type': result.model_type
        }
        
        self.prediction_history.append(history_entry)
        self.logger.info(f"Prediction saved: {history_entry}")
    
    def _simple_fallback_prediction(self, sensor_type: str, 
                                  recent_data: pd.DataFrame) -> PredictionResult:
        """تنبؤ بسيط fallback"""
        # متوسط القيم الأخيرة
        last_values = recent_data[sensor_type].tail(6).values
        prediction = np.mean(last_values)
        
        return PredictionResult(
            value=float(prediction),
            confidence=0.6,
            upper_bound=float(prediction * 1.2),
            lower_bound=float(prediction * 0.8),
            model_type="simple_average",
            timestamp=datetime.now(),
            features_used=['last_6_values']
        )
    
    def get_model_performance(self, sensor_type: str) -> Optional[ModelPerformance]:
        """الحصول على أداء النموذج"""
        if sensor_type in self.models:
            return self.models[sensor_type]['performance']
        return None
    
    def save_model(self, sensor_type: str, filepath: str) -> bool:
        """حفظ النموذج trained"""
        try:
            if sensor_type in self.models:
                model_info = self.models[sensor_type]
                joblib.dump(model_info, filepath)
                self.logger.info(f"Model for {sensor_type} saved to {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
        return False
    
    def load_model(self, sensor_type: str, filepath: str) -> bool:
        """تحميل النموذج trained"""
        try:
            if Path(filepath).exists():
                model_info = joblib.load(filepath)
                self.models[sensor_type] = model_info
                self.logger.info(f"Model for {sensor_type} loaded from {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        return False

# دالة مساعدة للاستيراد
def create_prediction_engine(config: Dict[str, Any]) -> AdvancedPredictionEngine:
    return AdvancedPredictionEngine(config)
