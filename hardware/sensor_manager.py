import logging
import random
import time
from typing import Dict, List, Any

class SensorManager:
    """مدير المستشعرات - يتعامل مع جميع أجهزة الاستشعار"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sensors = self._initialize_sensors()
        self.is_monitoring = False
        
    def _initialize_sensors(self) -> Dict[str, Any]:
        """تهيئة جميع المستشعرات"""
        sensors = {
            'pressure': {'pin': 17, 'value': 50.0, 'unit': 'bar'},
            'temperature': {'pin': 18, 'value': 75.0, 'unit': 'celsius'},
            'methane': {'pin': 19, 'value': 100.0, 'unit': 'ppm'},
            'hydrogen_sulfide': {'pin': 20, 'value': 10.0, 'unit': 'ppm'},
            'vibration': {'pin': 21, 'value': 2.0, 'unit': 'm/s²'},
            'flow': {'pin': 22, 'value': 50.0, 'unit': 'L/min'}
        }
        return sensors
    
    def start_monitoring(self):
        """بدء مراقبة المستشعرات"""
        self.is_monitoring = True
        self.logger.info("✅ Sensor monitoring started")
    
    def stop_monitoring(self):
        """إيقاف مراقبة المستشعرات"""
        self.is_monitoring = False
        self.logger.info("⏹️ Sensor monitoring stopped")
    
    def get_all_sensor_data(self) -> Dict[str, float]:
        """الحصول على بيانات جميع المستشعرات"""
        try:
            data = {}
            for sensor_name, sensor_info in self.sensors.items():
                # محاكاة قراءات واقعية مع بعض التقلبات
                base_value = sensor_info['value']
                fluctuation = random.uniform(-0.1, 0.1) * base_value
                current_value = max(0, base_value + fluctuation)
                
                data[sensor_name] = current_value
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error reading sensor data: {e}")
            return {}
    
    def get_sensor_health(self) -> Dict[str, Any]:
        """الحصول على صحة المستشعرات"""
        return {
            'total': len(self.sensors),
            'working': len(self.sensors),  # في المحاكاة، جميعها تعمل
            'faulty': 0
        }
