import os
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')

# -------------------- نظام التسجيل والمراقبة --------------------
def setup_logging():
    """إعداد نظام التسجيل والمراقبة"""
    logger = logging.getLogger('SNDT_Platform')
    logger.setLevel(logging.INFO)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    handler = RotatingFileHandler(
        'logs/sndt_platform.log', 
        maxBytes=5*1024*1024,
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# -------------------- MQTT Config --------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_TEMPERATURE = "sndt/temperature"
MQTT_TOPIC_PRESSURE = "sndt/pressure"
MQTT_TOPIC_METHANE = "sndt/methane"
MQTT_TOPIC_CONTROL = "sndt/control"
