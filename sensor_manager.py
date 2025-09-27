import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Union
import logging

class DataPattern(Enum):
    NORMAL = "normal"
    SEASONAL = "seasonal"
    TRENDING = "trending"
    CYCLIC = "cyclic"
    RANDOM = "random"

class SensorDataManager:
    def __init__(self):
        self.setup_logging()
        self.sensor_ranges = {
            'pressure': {'min': 0, 'max': 100, 'normal': (20, 60)},
            'temperature': {'min': 0, 'max': 150, 'normal': (50, 90)},
            'methane': {'min': 0, 'max': 1000, 'normal': (50, 200)},
            'vibration': {'min': 0, 'max': 10, 'normal': (1, 3)},
            'flow': {'min': 0, 'max': 100, 'normal': (30, 70)},
            'hydrogen_sulfide': {'min': 0, 'max': 100, 'normal': (5, 20)}
        }
        
        self.sensor_config = self._load_sensor_config()
    
    def _load_sensor_config(self) -> Dict[str, Dict]:
        """تحميل إعدادات المستشعرات للنسخة المتقدمة"""
        return {
            'temperature': {'mean': 75, 'std': 15, 'min': 0, 'max': 150},
            'pressure': {'mean': 50, 'std': 20, 'min': 0, 'max': 100},
            'vibration': {'mean': 2, 'std': 1.5, 'min': 0, 'max': 10},
            'flow': {'mean': 50, 'std': 20, 'min': 0, 'max': 100},
            'methane': {'mean': 100, 'std': 80, 'min': 0, 'max': 1000},
            'hydrogen_sulfide': {'mean': 10, 'std': 8, 'min': 0, 'max': 100}
        }
    
    def setup_logging(self):
        """تهيئة التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_realistic_data(self, hours: int = 24, frequency: str = '1H',
                              pattern: DataPattern = DataPattern.SEASONAL,
                              noise_level: float = 0.1,
                              return_type: str = 'dataframe') -> Union[pd.DataFrame, dict]:
        """
        🎯 الدالة الرئيسية المرنة - تحل محل الدالتين القديمتين
        
        Args:
            hours: عدد الساعات للبيانات التاريخية
            frequency: التكرار (1H, 30min, 1min)
            pattern: نمط البيانات
            noise_level: مستوى الضوضاء
            return_type: 'dataframe' أو 'dict'
        
        Returns:
            بيانات إما كـ DataFrame للتحليل أو dict للمراقبة الحية
        """
        
        # إذا طلب بيانات فورية (بديل الدالة البسيطة)
        if hours == 1 and frequency in ['1min', '5min'] and return_type == 'dict':
            return self._generate_real_time_data(pattern, noise_level)
        
        # البيانات التاريخية المتقدمة
        return self._generate_historical_data(hours, frequency, pattern, noise_level, return_type)
    
    def _generate_real_time_data(self, pattern: DataPattern, noise_level: float) -> Dict[str, float]:
        """توليد بيانات فورية للرصد الحي (بديل الدالة البسيطة)"""
        data = {}
        
        for sensor, config in self.sensor_config.items():
            # قيمة أساسية واقعية
            base_value = config['mean']
            
            # إضافة تأثير النمط
            if pattern == DataPattern.SEASONAL:
                # تغيير موسمي بسيط (ليل/نهار)
                hour = datetime.now().hour
                seasonal_factor = np.sin(hour * np.pi / 12) * 0.2
                base_value *= (1 + seasonal_factor)
            
            elif pattern == DataPattern.TRENDING:
                # اتجاه تدريجي
                trend_factor = random.uniform(-0.1, 0.1)
                base_value *= (1 + trend_factor)
            
            # إضافة ضوضاء واقعية
            noise = random.gauss(0, config['std'] * noise_level)
            final_value = base_value + noise
            
            # التأكد من الحدود
            final_value = max(config['min'], min(config['max'], final_value))
            
            data[sensor] = round(final_value, 2)
        
        # تسجيل البيانات
        self.logger.info(f"📊 Generated real-time data: {data}")
        return data
    
    def _generate_historical_data(self, hours: int, frequency: str, pattern: DataPattern,
                                noise_level: float, return_type: str) -> Union[pd.DataFrame, dict]:
        """توليد بيانات تاريخية للتحليل المتقدم"""
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq=frequency
        )
        
        data = {'timestamp': dates}
        total_points = len(dates)
        
        for sensor, config in self.sensor_config.items():
            base_signal = self._generate_base_signal(total_points, config, pattern, dates)
            noisy_signal = self._add_realistic_noise(base_signal, noise_level, config)
            bounded_signal = np.clip(noisy_signal, config['min'], config['max'])
            
            data[sensor] = bounded_signal
        
        df = pd.DataFrame(data)
        self._log_generation(df, pattern, noise_level)
        
        if return_type == 'dict' and len(df) > 0:
            return df.iloc[-1].to_dict()  # آخر نقطة كـ dict
        
        return df
    
    def _generate_base_signal(self, total_points: int, config: Dict, 
                            pattern: DataPattern, dates: pd.DatetimeIndex) -> np.ndarray:
        """توليد الإشارة الأساسية بناءً على النمط"""
        
        if pattern == DataPattern.NORMAL:
            return np.full(total_points, config['mean'])
        
        elif pattern == DataPattern.SEASONAL:
            # نمط موسمي (ساعات اليوم)
            hours = dates.hour.values
            seasonal = np.sin(hours * 2 * np.pi / 24) * 0.3
            return config['mean'] * (1 + seasonal)
        
        elif pattern == DataPattern.TRENDING:
            # نمط اتجاهي
            trend = np.linspace(-0.2, 0.2, total_points)
            return config['mean'] * (1 + trend)
        
        elif pattern == DataPattern.CYCLIC:
            # نمط دوري
            cyclic = np.sin(np.arange(total_points) * 2 * np.pi / 12) * 0.4
            return config['mean'] * (1 + cyclic)
        
        else:  # RANDOM
            return np.random.normal(config['mean'], config['std'] * 0.5, total_points)
    
    def _add_realistic_noise(self, signal: np.ndarray, noise_level: float, 
                           config: Dict) -> np.ndarray:
        """إضافة ضوضاء واقعية للإشارة"""
        noise = np.random.normal(0, config['std'] * noise_level, len(signal))
        return signal + noise
    
    def _log_generation(self, df: pd.DataFrame, pattern: DataPattern, noise_level: float):
        """تسجيل عملية توليد البيانات"""
        self.logger.info(f"✅ Generated {len(df)} data points with pattern {pattern.value}, "
                        f"noise {noise_level}")
    
    def get_current_reading(self, scenario: str = 'normal') -> Dict[str, float]:
        """
        🚀 دالة مساعدة مبسطة للحصول على قراءة حالية
        
        Args:
            scenario: 'normal', 'warning', 'danger'
        
        Returns:
            قراءة واحدة واقعية للمستشعرات
        """
        
        # الحصول على بيانات عادية أولاً
        base_data = self._generate_real_time_data(DataPattern.NORMAL, noise_level=0.05)
        
        # تعديل بناءً على السيناريو
        if scenario == 'warning':
            multiplier = random.uniform(1.2, 1.5)
        elif scenario == 'danger':
            multiplier = random.uniform(1.6, 2.0)
        else:  # normal
            multiplier = 1.0
        
        adjusted_data = {}
        for sensor, value in base_data.items():
            config = self.sensor_config[sensor]
            new_value = value * multiplier
            # التأكد من عدم تجاوز الحد الأقصى
            adjusted_data[sensor] = min(new_value, config['max'] * 0.95)
        
        self.logger.info(f"🚨 Generated {scenario} scenario data")
        return adjusted_data
    
    def simulate_emergency_pattern(self, emergency_type: str) -> Dict[str, float]:
        """
        محاكاة أنماط الطوارئ الواقعية
        """
        patterns = {
            'pressure_spike': {'pressure': 2.5, 'temperature': 1.3, 'vibration': 1.8},
            'gas_leak': {'methane': 3.0, 'hydrogen_sulfide': 2.5, 'flow': 0.7},
            'equipment_failure': {'vibration': 2.5, 'temperature': 1.5, 'pressure': 0.8},
            'normal': {}  # لا تغيير
        }
        
        base_data = self.get_current_reading('normal')
        pattern = patterns.get(emergency_type, {})
        
        for sensor, multiplier in pattern.items():
            if sensor in base_data:
                base_data[sensor] = base_data[sensor] * multiplier
        
        return base_data

# 🔧 اختبار الدالة
if __name__ == "__main__":
    sensor_manager = SensorDataManager()
    
    # اختبار 1: بيانات فورية (بديل الدالة البسيطة)
    print("📊 بيانات فورية:")
    real_time_data = sensor_manager.get_current_reading('normal')
    print(real_time_data)
    
    # اختبار 2: بيانات تاريخية (الدالة المتقدمة)
    print("\n📈 بيانات تاريخية:")
    historical_data = sensor_manager.generate_realistic_data(hours=6, frequency='30min')
    print(historical_data.head())
    
    # اختبار 3: محاكاة طوارئ
    print("\n🚨 محاكاة طوارئ:")
    emergency_data = sensor_manager.simulate_emergency_pattern('gas_leak')
    print(emergency_data)    
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
