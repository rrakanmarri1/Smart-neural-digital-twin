import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum
from dataclasses import dataclass
import random

class DataPattern(Enum):
    NORMAL = "normal"
    SEASONAL = "seasonal"
    TRENDING = "trending"
    NOISY = "noisy"
    ANOMALOUS = "anomalous"

@dataclass
class DataGenerationConfig:
    pattern: DataPattern
    noise_level: float
    anomaly_probability: float
    trend_strength: float
    seasonal_amplitude: float

class AdvancedSensorDataGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensor_config = self._load_sensor_config()
        self.generation_history = []
        self.setup_logging()
        
    def _load_sensor_config(self) -> Dict[str, Dict]:
        """تحميل إعدادات المستشعرات"""
        return {
            'temperature': {'mean': 25, 'std': 5, 'min': -10, 'max': 85},
            'pressure': {'mean': 1013, 'std': 50, 'min': 300, 'max': 1100},
            'vibration': {'mean': 0.5, 'std': 0.3, 'min': 0, 'max': 10},
            'flow': {'mean': 15, 'std': 5, 'min': 0, 'max': 100},
            'methane': {'mean': 50, 'std': 30, 'min': 0, 'max': 2000},
            'h2s': {'mean': 5, 'std': 3, 'min': 0, 'max': 200}
        }
    
    def setup_logging(self):
        """تهيئة التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_realistic_data(self, hours: int = 24, frequency: str = '1H',
                              pattern: DataPattern = DataPattern.SEASONAL,
                              noise_level: float = 0.1) -> pd.DataFrame:
        """توليد بيانات واقعية مع أنماط"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq=frequency
        )
        
        data = {'timestamp': dates}
        total_points = len(dates)
        
        for sensor, config in self.sensor_config.items():
            base_signal = self._generate_base_signal(
                total_points, config, pattern, dates
            )
            
            # إضافة ضوضاء واقعية
            noisy_signal = self._add_realistic_noise(base_signal, noise_level)
            
            # إضافة أنماط واقعية
            patterned_signal = self._apply_patterns(noisy_signal, pattern, dates)
            
            #确保数据在 الحدود المعقولة
            bounded_signal = np.clip(patterned_signal, config['min'], config['max'])
            
            data[sensor] = bounded_signal
        
        df = pd.DataFrame(data)
        self._log_generation(df, pattern, noise_level)
        return df
    
    def _generate_base_signal(self, n_points: int, config: Dict,
                            pattern: DataPattern, dates: pd.DatetimeIndex) -> np.ndarray:
        """توليد الإشارة الأساسية"""
        if pattern == DataPattern.NORMAL:
            return np.random.normal(config['mean'], config['std'], n_points)
        
        elif pattern == DataPattern.SEASONAL:
            # إشارة موسمية مع تواتر يومي
            base = config['mean'] + 0.5 * config['std'] * np.sin(
                2 * np.pi * dates.hour / 24
            )
            noise = np.random.normal(0, config['std'] * 0.3, n_points)
            return base + noise
        
        elif pattern == DataPattern.TRENDING:
            # إشارة مع اتجاه تدريجي
            trend = np.linspace(0, config['std'] * 0.5, n_points)
            base = np.random.normal(config['mean'], config['std'] * 0.7, n_points)
            return base + trend
        
        else:  # NOISY
            return np.random.normal(config['mean'], config['std'] * 1.5, n_points)
    
    def _add_realistic_noise(self, signal: np.ndarray, noise_level: float) -> np.ndarray:
        """إضافة ضوضاء واقعية"""
        noise = np.random.normal(0, np.std(signal) * noise_level, len(signal))
        return signal + noise
    
    def _apply_patterns(self, signal: np.ndarray, pattern: DataPattern,
                       dates: pd.DatetimeIndex) -> np.ndarray:
        """تطبيق الأنماط على الإشارة"""
        if pattern == DataPattern.SEASONAL:
            # تأثير موسمي إضافي
            seasonal_effect = 0.2 * np.std(signal) * np.sin(
                2 * np.pi * dates.dayofweek / 7
            )
            return signal + seasonal_effect
        
        return signal
    
    def add_contextual_anomalies(self, df: pd.DataFrame, anomaly_config: Dict) -> pd.DataFrame:
        """إضافة شذوذ سياقي واقعي"""
        df_anomalous = df.copy()
        n_anomalies = int(len(df) * anomaly_config.get('probability', 0.05))
        
        if n_anomalies == 0:
            return df_anomalous
        
        # إضافة شذوذ مترابط بين المستشعرات
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(['single', 'correlated', 'propagating'])
            
            if anomaly_type == 'single':
                # شذوذ فردي في مستشعر عشوائي
                sensor = random.choice(list(self.sensor_config.keys()))
                self._add_single_anomaly(df_anomalous, sensor, idx, anomaly_config)
            
            elif anomaly_type == 'correlated':
                # شذوذ مترابط بين مستشعرات مرتبطة
                self._add_correlated_anomalies(df_anomalous, idx, anomaly_config)
            
            elif anomaly_type == 'propagating':
                # شذوذ مت propagates مع الوقت
                self._add_propagating_anomaly(df_anomalous, idx, anomaly_config)
        
        return df_anomalous
    
    def _add_single_anomaly(self, df: pd.DataFrame, sensor: str,
                           index: int, config: Dict):
        """إضافة شذوذ فردي"""
        current_value = df.at[index, sensor]
        sensor_config = self.sensor_config[sensor]
        
        # شذوذ واقعي (ليس مجرد قفزة)
        anomaly_magnitude = config.get('magnitude', 3.0)
        anomaly_direction = random.choice([-1, 1])
        
        new_value = current_value + anomaly_direction * anomaly_magnitude * sensor_config['std']
        new_value = np.clip(new_value, sensor_config['min'], sensor_config['max'])
        
        df.at[index, sensor] = new_value
    
    def _add_correlated_anomalies(self, df: pd.DataFrame, index: int, config: Dict):
        """إضافة شذوذ مترابط"""
        # مجموعة مستشعرات مرتبطة (مثلاً درجة الحرارة والضغط)
        correlated_groups = [
            ['temperature', 'pressure'],
            ['vibration', 'flow'],
            ['methane', 'h2s']
        ]
        
        group = random.choice(correlated_groups)
        for sensor in group:
            self._add_single_anomaly(df, sensor, index, config)
    
    def _add_propagating_anomaly(self, df: pd.DataFrame, start_index: int, config: Dict):
        """إضافة شذوذ مت propagates"""
        propagation_length = random.randint(3, 10)
        sensor = random.choice(list(self.sensor_config.keys()))
        
        for i in range(propagation_length):
            if start_index + i < len(df):
                # شذوذ يقل تدريجياً مع الوقت
                decay = 1.0 - (i / propagation_length)
                current_config = config.copy()
                current_config['magnitude'] = config.get('magnitude', 3.0) * decay
                
                self._add_single_anomaly(df, sensor, start_index + i, current_config)
    
    def _log_generation(self, df: pd.DataFrame, pattern: DataPattern, noise_level: float):
        """تسجيل عملية التوليد"""
        log_entry = {
            'timestamp': datetime.now(),
            'data_points': len(df),
            'pattern': pattern.value,
            'noise_level': noise_level,
            'sensors': list(self.sensor_config.keys()),
            'data_range': {sensor: (df[sensor].min(), df[sensor].max()) 
                          for sensor in self.sensor_config.keys()}
        }
        self.generation_history.append(log_entry)
        self.logger.info(f"Generated {len(df)} data points with pattern {pattern.value}")
    
    def save_generated_data(self, df: pd.DataFrame, filepath: str):
        """حفظ البيانات المولدة"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if filepath.suffix == '.csv':
                df.to_csv(filepath, index=False)
            elif filepath.suffix == '.parquet':
                df.to_parquet(filepath, index=False)
            
            self.logger.info(f"💾 Saved generated data to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving generated data: {e}")
            return False

# دالة مساعدة
def create_data_generator(config: Dict[str, Any]) -> AdvancedSensorDataGenerator:
    return AdvancedSensorDataGenerator(config)
