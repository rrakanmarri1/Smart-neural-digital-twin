import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import joblib
import optuna
from optuna.samplers import TPESampler

# استيراد TensorFlow بشكل conditional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available - LSTM training disabled")

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EnhancedLSTMTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.training_history: List[Dict] = []
        self.study: Optional[optuna.study.Study] = None
        self.setup_logging()
        
    def setup_logging(self):
        """تهيئة نظام التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not HAS_TENSORFLOW:
            self.logger.warning("TensorFlow is not available. LSTM functionality will be limited.")
    
    def prepare_advanced_data(self, df: pd.DataFrame, target_column: str,
                            sequence_length: int = 20, test_size: float = 0.2) -> Tuple:
        """تحضير بيانات متقدم للـLSTM"""
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in dataframe")
        
        # استخراج البيانات المستهدفة
        target_data = df[target_column].values
        
        # اختيار طريقة التطبيع المناسبة
        if self._has_outliers(target_data):
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        data_scaled = scaler.fit_transform(target_data.reshape(-1, 1))
        
        # إنشاء sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i+sequence_length])
            y.append(data_scaled[i+sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # تقسيم البيانات مع الحفاظ على الترتيب الزمني
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def _has_outliers(self, data: np.ndarray) -> bool:
        """التحقق من وجود قيم متطرفة"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return np.any((data < lower_bound) | (data > upper_bound))
    
    def build_optimized_model(self, input_shape: Tuple[int, int],
                            units: int = 50, dropout_rate: float = 0.2) -> Sequential:
        """بناء نموذج LSTM مُحسَّن"""
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required to build LSTM models")
        
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dropout(dropout_rate // 2),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mape'])
        
        return model
    
    def train_with_optimization(self, df: pd.DataFrame, target_column: str,
                              n_trials: int = 20) -> Dict:
        """تدريب مع optimization تلقائي"""
        if not HAS_TENSORFLOW:
            return {'error': 'TensorFlow not available'}
        
        def objective(trial):
            # معلمات لل optimization
            sequence_length = trial.suggest_int('sequence_length', 10, 50)
            units = trial.suggest_int('units', 32, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            # تحضير البيانات
            X_train, X_test, y_train, y_test, scaler = self.prepare_advanced_data(
                df, target_column, sequence_length
            )
            
            # بناء النموذج
            model = self.build_optimized_model(
                (X_train.shape[1], X_train.shape[2]), units, dropout_rate
            )
            
            # callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # التدريب
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # التقييم
            test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
            return test_loss
        
        # دراسة Optuna
        self.study = optuna.create_study(direction='minimize', sampler=TPESampler())
        self.study.optimize(objective, n_trials=n_trials)
        
        # أفضل معلمات
        best_params = self.study.best_params
        self.logger.info(f"🎯 Best parameters: {best_params}")
        
        # التدريب النهائي بأفضل المعلمات
        return self.train_final_model(df, target_column, best_params)
    
    def train_final_model(self, df: pd.DataFrame, target_column: str,
                        best_params: Dict) -> Dict:
        """التدريب النهائي بأفضل المعلمات"""
        X_train, X_test, y_train, y_test, scaler = self.prepare_advanced_data(
            df, target_column, best_params['sequence_length']
        )
        
        model = self.build_optimized_model(
            (X_train.shape[1], X_train.shape[2]),
            best_params['units'],
            best_params['dropout_rate']
        )
        
        # callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(
                f'best_{target_column}_model.keras',
                save_best_only=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(factor=0.5, patience=8)
        ]
        
        # التدريب
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # التقييم
        test_predictions = model.predict(X_test)
        test_metrics = self._calculate_metrics(y_test, test_predictions)
        
        # حفظ النموذج
        self.models[target_column] = model
        self.scalers[target_column] = scaler
        
        # حفظ سجل التدريب
        training_record = {
            'timestamp': datetime.now(),
            'target_column': target_column,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'training_loss': history.history['loss'][-1],
            'validation_loss': history.history['val_loss'][-1]
        }
        self.training_history.append(training_record)
        
        return training_record
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """حساب مقاييس الأداء"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
    
    def predict(self, df: pd.DataFrame, target_column: str,
               steps: int = 10) -> np.ndarray:
        """التنبؤ بالقيم المستقبلية"""
        if target_column not in self.models:
            raise ValueError(f"No trained model for {target_column}")
        
        model = self.models[target_column]
        scaler = self.scalers[target_column]
        
        # أخذ آخر sequence للتنبؤ
        last_sequence = df[target_column].values[-model.input_shape[1]:]
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(steps):
            # التنبؤ بالخطوة التالية
            next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            
            # تحديث sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # إلغاء التطبيع
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    def save_model(self, target_column: str, filepath: str) -> bool:
        """حفظ النموذج trained"""
        if target_column not in self.models:
            return False
        
        try:
            model = self.models[target_column]
            scaler = self.scalers[target_column]
            
            # حفظ النموذج
            model.save(filepath)
            
            # حفظ scaler
            scaler_path = filepath.replace('.keras', '_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            
            self.logger.info(f"💾 Saved model and scaler for {target_column}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, target_column: str, model_path: str) -> bool:
        """تحميل النموذج trained"""
        try:
            if not HAS_TENSORFLOW:
                return False
            
            # تحميل النموذج
            model = load_model(model_path)
            
            # تحميل scaler
            scaler_path = model_path.replace('.keras', '_scaler.joblib')
            scaler = joblib.load(scaler_path)
            
            self.models[target_column] = model
            self.scalers[target_column] = scaler
            
            self.logger.info(f"📂 Loaded model and scaler for {target_column}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False

# دالة مساعدة
def create_lstm_trainer(config: Dict[str, Any]) -> EnhancedLSTMTrainer:
    return EnhancedLSTMTrainer(config)
