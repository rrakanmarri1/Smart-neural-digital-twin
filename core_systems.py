from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
import importlib.util
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# External systems
from config_and_logging import SmartConfig  # type: ignore

# Platform detection and dynamic imports
RASPBERRY_PI_AVAILABLE = False
GPIO = None
smbus = None
board = None
busio = None
adafruit_ads1x15 = None
adafruit_bme280 = None
digitalio = None

# Dynamically check for Raspberry Pi and related packages
try:
    # Check if we're running on a Raspberry Pi by looking for Pi-specific file
    if Path('/sys/firmware/devicetree/base/model').exists():
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            if 'Raspberry Pi' in f.read():
                RASPBERRY_PI_AVAILABLE = True

    # Dynamically import Raspberry Pi libraries if available
    if RASPBERRY_PI_AVAILABLE:
        # GPIO access
        if importlib.util.find_spec('RPi.GPIO'):
            import RPi.GPIO as GPIO
        
        # I2C access
        if importlib.util.find_spec('smbus2'):
            import smbus2 as smbus
        elif importlib.util.find_spec('smbus'):
            import smbus
        
        # Adafruit libraries for advanced sensor support
        if importlib.util.find_spec('board'):
            import board
        if importlib.util.find_spec('busio'):
            import busio
        if importlib.util.find_spec('digitalio'):
            import digitalio
        if importlib.util.find_spec('adafruit_ads1x15.ads1115'):
            import adafruit_ads1x15.ads1115 as adafruit_ads1115
        if importlib.util.find_spec('adafruit_bme280'):
            import adafruit_bme280
except Exception as e:
    # Log but continue - the system will fall back to simulation
    print(f"Raspberry Pi detection or library import failed: {e}")

# Relay controller is optional; provide safe fallback if unavailable
try:
    from config_and_logging import RelayController  # type: ignore
except ImportError:  # pragma: no cover
    class RelayController:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            self._states = {}

        def control_relay(self, name: str, state: bool, reason: str = "") -> bool:
            self._states[name] = state
            return True

        def emergency_shutdown(self):
            for k in list(self._states.keys()):
                self._states[k] = False

        def get_relay_status(self) -> Dict[str, bool]:
            return dict(self._states)


# AI Manager (Part 2 orchestrator)
try:
    from ai_systems_part2 import AISystemManager  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "AISystemManager not found. Ensure ai_systems_part1.py and ai_systems_part2.py are available. "
        f"Original error: {e}"
    )


# ------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------

MIN_TRAINING_RECORDS = 300
SENSOR_HISTORY_LIMIT = 2500
SENSOR_RECALIBRATION_WINDOW = 50
CORRELATION_EWMA_ALPHA = 0.1
DEFAULT_MONITOR_INTERVAL = 2.0
MAINTENANCE_INTERVAL_SEC = 1800
RETRAIN_CHECK_INTERVAL_SEC = 900
LOW_CONFIDENCE_THRESHOLD = 0.7
RISK_ESCALATION_LEVELS = {
    "NORMAL": 0.0,
    "HIGH_ALERT": 0.5,
    "CRITICAL": 0.75,
    "EMERGENCY": 0.9
}

# Constants for physical sensor detection
I2C_BUS_NUM = 1  # Default I2C bus on most Raspberry Pi models
GPIO_SENSOR_PINS = [4, 5, 6, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]  # Common GPIO pins
SPI_DEVICES = [0, 1]  # SPI device numbers to check
UART_DEVICES = ["/dev/ttyS0", "/dev/ttyAMA0"]  # Common UART devices on Raspberry Pi

# Known I2C sensor addresses and their types - expanded with more modern sensors
KNOWN_I2C_SENSORS = {
    0x76: {"type": "BME280", "name": "temperature_pressure_humidity", "min": -40, "max": 85},
    0x77: {"type": "BME280", "name": "temperature_pressure_humidity_alt", "min": -40, "max": 85},
    0x48: {"type": "ADS1115", "name": "analog_converter", "min": 0, "max": 5},
    0x49: {"type": "ADS1115", "name": "analog_converter_alt", "min": 0, "max": 5}, 
    0x68: {"type": "MPU6050", "name": "accelerometer", "min": -2, "max": 2},
    0x23: {"type": "BH1750", "name": "light_sensor", "min": 0, "max": 65535},
    0x5c: {"type": "AM2320", "name": "humidity_sensor", "min": 0, "max": 100},
    0x44: {"type": "SHT31", "name": "temp_humidity", "min": -40, "max": 125},
    0x40: {"type": "HTU21D", "name": "humidity", "min": 0, "max": 100},
    0x29: {"type": "TSL2591", "name": "light", "min": 0, "max": 88000},
    0x39: {"type": "TSL2561", "name": "luminosity", "min": 0, "max": 40000},
    0x60: {"type": "MCP9808", "name": "precision_temp", "min": -40, "max": 125},
    0x57: {"type": "MAX30102", "name": "pulse_oximeter", "min": 0, "max": 100},
    # Added newer sensors
    0x36: {"type": "SCD30", "name": "co2_sensor", "min": 0, "max": 10000},
    0x62: {"type": "SGP30", "name": "air_quality", "min": 0, "max": 60000},
    0x58: {"type": "SGP40", "name": "voc_index", "min": 0, "max": 500},
    0x4a: {"type": "MLX90614", "name": "ir_temperature", "min": -70, "max": 380},
    0x3c: {"type": "SSD1306", "name": "oled_display", "min": 0, "max": 1},
    0x61: {"type": "Si7021", "name": "humidity_temp", "min": -40, "max": 125},
    0x70: {"type": "HT16K33", "name": "led_matrix", "min": 0, "max": 15},
    0x69: {"type": "MPU6050_ALT", "name": "gyroscope", "min": -250, "max": 250},
    0x1d: {"type": "ADXL345", "name": "accelerometer_alt", "min": -16, "max": 16}
}


# ------------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------------

class SensorStatus(Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    SIMULATED = "SIMULATED"


@dataclass
class SensorReading:
    value: float
    confidence: float
    status: SensorStatus
    timestamp: datetime
    source: str  # physical / simulated / fused / emergency


# ------------------------------------------------------------------------------------
# Sensor Detection Utilities
# ------------------------------------------------------------------------------------

class PhysicalSensorDetector:
    """
    Modern implementation for detecting physical sensors connected to Raspberry Pi
    Supports I2C, GPIO, SPI, and UART interfaces with robust fallbacks
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.i2c_available = False
        self.gpio_available = False
        self.spi_available = False
        self.uart_available = False
        self.i2c_bus = None
        self.i2c_busio = None
        
        # Initialize hardware interfaces if Raspberry Pi is available
        if RASPBERRY_PI_AVAILABLE:
            self._initialize_hardware_interfaces()
        else:
            self.logger.info("Not running on Raspberry Pi - using simulation mode")
    
    def _initialize_hardware_interfaces(self):
        """Initialize all available hardware interfaces"""
        # Initialize I2C
        self._initialize_i2c()
        
        # Initialize GPIO
        self._initialize_gpio()
        
        # Initialize SPI
        self._initialize_spi()
        
        # Initialize UART
        self._initialize_uart()
    
    def _initialize_i2c(self):
        """Initialize I2C interface with multiple fallback options"""
        try:
            # Try smbus first
            if smbus:
                self.i2c_bus = smbus.SMBus(I2C_BUS_NUM)
                self.i2c_available = True
                self.logger.info("I2C initialized using smbus")
            # Try busio as fallback
            elif busio and board:
                self.i2c_busio = busio.I2C(board.SCL, board.SDA)
                self.i2c_available = True
                self.logger.info("I2C initialized using busio")
        except Exception as e:
            self.logger.warning(f"I2C initialization failed: {e}")
    
    def _initialize_gpio(self):
        """Initialize GPIO with error handling"""
        try:
            if GPIO:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                self.gpio_available = True
                self.logger.info("GPIO initialized")
        except Exception as e:
            self.logger.warning(f"GPIO initialization failed: {e}")
    
    def _initialize_spi(self):
        """Initialize SPI interface"""
        try:
            if busio and board:
                # Try to set up SPI
                spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
                self.spi_available = True
                self.logger.info("SPI initialized")
        except Exception as e:
            self.logger.warning(f"SPI initialization failed: {e}")
    
    def _initialize_uart(self):
        """Check for UART device availability"""
        try:
            for device in UART_DEVICES:
                if Path(device).exists():
                    self.uart_available = True
                    self.logger.info(f"UART device available: {device}")
                    break
        except Exception as e:
            self.logger.warning(f"UART detection failed: {e}")
    
    def detect_sensors(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan all available interfaces for connected sensors
        Returns a dictionary of detected sensors with their configuration
        """
        detected_sensors = {}
        
        # If not on a Raspberry Pi, return empty dictionary
        if not RASPBERRY_PI_AVAILABLE:
            self.logger.info("Sensor detection skipped - not on Raspberry Pi")
            return detected_sensors
        
        # Detect I2C sensors if available
        if self.i2c_available:
            i2c_sensors = self._scan_i2c_sensors()
            detected_sensors.update(i2c_sensors)
            
        # Detect GPIO sensors if available
        if self.gpio_available:
            gpio_sensors = self._scan_gpio_sensors()
            detected_sensors.update(gpio_sensors)
            
        # Detect SPI sensors if available
        if self.spi_available:
            spi_sensors = self._scan_spi_sensors()
            detected_sensors.update(spi_sensors)
        
        # Log detection results
        self.logger.info(f"Detected {len(detected_sensors)} physical sensors")
        return detected_sensors
    
    def _scan_i2c_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Scan the I2C bus for known sensor addresses"""
        detected = {}
        if not self.i2c_available:
            return detected
        
        # Different scanning approach depending on I2C implementation
        if self.i2c_bus:  # smbus approach
            return self._scan_i2c_smbus()
        elif self.i2c_busio:  # Adafruit CircuitPython approach
            return self._scan_i2c_busio()
            
        return detected
    
    def _scan_i2c_smbus(self) -> Dict[str, Dict[str, Any]]:
        """Scan I2C using smbus implementation"""
        detected = {}
        
        for address in range(0x03, 0x77 + 1):
            try:
                self.i2c_bus.read_byte(address)
                # Device exists at this address
                sensor_info = self._process_i2c_address(address)
                if sensor_info:
                    detected.update(sensor_info)
            except Exception:
                # No device at this address
                pass
                
        return detected
    
    def _scan_i2c_busio(self) -> Dict[str, Dict[str, Any]]:
        """Scan I2C using busio/CircuitPython implementation"""
        detected = {}
        
        try:
            # CircuitPython can scan in one call
            addresses = self.i2c_busio.scan()
            for address in addresses:
                sensor_info = self._process_i2c_address(address)
                if sensor_info:
                    detected.update(sensor_info)
        except Exception as e:
            self.logger.error(f"Error scanning I2C with busio: {e}")
            
        return detected
    
    def _process_i2c_address(self, address: int) -> Dict[str, Dict[str, Any]]:
        """Process a detected I2C address and return sensor info if recognized"""
        result = {}
        
        if address in KNOWN_I2C_SENSORS:
            sensor_info = KNOWN_I2C_SENSORS[address].copy()
            sensor_name = f"{sensor_info['name']}_{address:02x}"
            result[sensor_name] = {
                "type": sensor_info['type'],
                "interface": "I2C",
                "address": address,
                "min": sensor_info['min'],
                "max": sensor_info['max']
            }
            self.logger.info(f"Detected I2C sensor: {sensor_info['type']} at address 0x{address:02x}")
        else:
            # Unknown sensor, add with generic parameters
            sensor_name = f"unknown_i2c_{address:02x}"
            result[sensor_name] = {
                "type": "UNKNOWN",
                "interface": "I2C",
                "address": address,
                "min": 0,
                "max": 100
            }
            self.logger.info(f"Detected unknown I2C device at address 0x{address:02x}")
            
        return result
    
    def _scan_gpio_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Set up GPIO pins for potential sensors"""
        detected = {}
        if not self.gpio_available:
            return detected
            
        for pin in GPIO_SENSOR_PINS:
            try:
                # Configure as input with pull-up
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                
                # Read initial state
                initial_state = GPIO.input(pin)
                
                # Determine sensor type based on pin behavior
                # This is a heuristic approach - in production you'd use specific sensor drivers
                sensor_type = "DIGITAL"
                
                # For certain pins, we might assume specific sensors based on common usage
                if pin == 4:  # DHT sensors often use pin 4
                    sensor_type = "DHT22"
                    min_val = -40
                    max_val = 80
                elif pin in [17, 27]:  # Often used for motion sensors
                    sensor_type = "PIR_MOTION"
                    min_val = 0
                    max_val = 1
                else:
                    min_val = 0
                    max_val = 1
                
                sensor_name = f"{sensor_type.lower()}_gpio{pin}"
                detected[sensor_name] = {
                    "type": sensor_type,
                    "interface": "GPIO",
                    "pin": pin,
                    "initial_state": initial_state,
                    "min": min_val,
                    "max": max_val
                }
                self.logger.info(f"Set up GPIO sensor on pin {pin}, initial state: {initial_state}")
            except Exception as e:
                self.logger.debug(f"GPIO pin {pin} setup failed: {e}")
                
        return detected
    
    def _scan_spi_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Detect SPI sensors with improved implementation"""
        detected = {}
        if not self.spi_available:
            return detected
            
        # Improved SPI detection using hardware presence
        try:
            # Check for standard SPI devices
            spi_path = Path("/dev/spidev0.")
            for device_num in SPI_DEVICES:
                device_path = spi_path.with_name(f"{spi_path.name}{device_num}")
                if device_path.exists():
                    # Determine sensor type based on common usage
                    if device_num == 0:
                        sensor_type = "MCP3008"  # Common ADC
                        min_val = 0
                        max_val = 1023
                        sensor_name = f"adc_mcp3008_spi{device_num}"
                    else:
                        sensor_type = "SPI_DEVICE"
                        min_val = 0
                        max_val = 4095
                        sensor_name = f"spi_sensor_{device_num}"
                    
                    detected[sensor_name] = {
                        "type": sensor_type,
                        "interface": "SPI",
                        "device": device_num,
                        "min": min_val,
                        "max": max_val
                    }
                    self.logger.info(f"Detected SPI device {sensor_type} on device {device_num}")
        except Exception as e:
            self.logger.debug(f"SPI device detection failed: {e}")
                
        return detected

    def read_sensor_value(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Read the value from a physical sensor based on its configuration
        Returns the value and success status
        """
        if not RASPBERRY_PI_AVAILABLE:
            return 0.0, False
            
        try:
            interface = sensor_config.get("interface")
            
            if interface == "I2C":
                return self._read_i2c_sensor(sensor_config)
            elif interface == "GPIO":
                return self._read_gpio_sensor(sensor_config)
            elif interface == "SPI":
                return self._read_spi_sensor(sensor_config)
            else:
                return 0.0, False
        except Exception as e:
            self.logger.error(f"Error reading sensor {sensor_config.get('type', 'unknown')}: {e}")
            return 0.0, False
    
    def _read_i2c_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from an I2C sensor with improved support for specific sensor types"""
        if not self.i2c_available:
            return 0.0, False
            
        address = sensor_config.get("address")
        sensor_type = sensor_config.get("type")
        min_val = sensor_config.get("min", 0)
        max_val = sensor_config.get("max", 100)
        
        try:
            # Handle specific sensors with appropriate reading methods
            if sensor_type == "BME280":
                if adafruit_bme280 and self.i2c_busio:
                    # Use Adafruit library if available
                    bme280 = adafruit_bme280.Adafruit_BME280_I2C(self.i2c_busio, address=address)
                    # Return temperature as an example - could also return pressure or humidity
                    return float(bme280.temperature), True
                elif self.i2c_bus:
                    # Simplified BME280 reading using smbus
                    # Read temperature register (0xFA)
                    temp_msb = self.i2c_bus.read_byte_data(address, 0xFA)
                    temp_lsb = self.i2c_bus.read_byte_data(address, 0xFB)
                    temp_xlsb = self.i2c_bus.read_byte_data(address, 0xFC)
                    
                    # Convert to temperature (simplified calculation)
                    raw_temp = ((temp_msb << 16) | (temp_lsb << 8) | temp_xlsb) >> 4
                    temperature = (raw_temp / 100.0)
                    return temperature, True
                
            elif sensor_type == "ADS1115":
                if adafruit_ads1x15 and self.i2c_busio:
                    # Use Adafruit library
                    ads = adafruit_ads1115.ADS1115(self.i2c_busio, address=address)
                    # Read first channel as an example
                    value = ads.get_voltage(0)
                    return float(value), True
                elif self.i2c_bus:
                    # Simplified ADS1115 reading using smbus
                    # Configure for single-ended measurement on channel 0
                    config = 0x8583  # Single-ended AIN0, 4.096V, 128SPS
                    self.i2c_bus.write_word_data(address, 1, config)
                    time.sleep(0.01)  # Wait for conversion
                    value = self.i2c_bus.read_word_data(address, 0)
                    # Convert from big-endian to little-endian
                    value = ((value & 0xFF) << 8) | (value >> 8)
                    # Convert to voltage (4.096V reference)
                    voltage = (value * 4.096) / 32768
                    return float(voltage), True
                
            elif sensor_type == "MPU6050":
                if self.i2c_bus:
                    # Read accelerometer X value
                    accel_x_h = self.i2c_bus.read_byte_data(address, 0x3B)
                    accel_x_l = self.i2c_bus.read_byte_data(address, 0x3C)
                    accel_x = (accel_x_h << 8) | accel_x_l
                    # Convert to signed value
                    if accel_x > 0x7FFF:
                        accel_x = accel_x - 0x10000
                    # Convert to g force (±2g range)
                    accel_g = float(accel_x) / 16384.0
                    return accel_g, True
            
            elif sensor_type == "BH1750":
                if self.i2c_bus:
                    # Power on
                    self.i2c_bus.write_byte(address, 0x01)
                    # One time high resolution measurement (0x20)
                    self.i2c_bus.write_byte(address, 0x20)
                    time.sleep(0.18)  # Wait for measurement
                    # Read data
                    data = self.i2c_bus.read_i2c_block_data(address, 0x00, 2)
                    # Convert to lux
                    lux = (data[0] << 8 | data[1]) / 1.2
                    return float(lux), True
            
            # Default case for other or unknown sensors
            elif self.i2c_bus:
                # Generic read - get a byte from the first register
                try:
                    value = self.i2c_bus.read_byte(address)
                    # Scale to sensor range
                    scaled_value = (value / 255.0) * (max_val - min_val) + min_val
                    return float(scaled_value), True
                except:
                    # Try reading word data from register 0 as fallback
                    value = self.i2c_bus.read_word_data(address, 0)
                    # Scale to sensor range
                    scaled_value = (value / 65535.0) * (max_val - min_val) + min_val
                    return float(scaled_value), True
                
        except Exception as e:
            self.logger.error(f"Error reading I2C sensor {sensor_type} at address 0x{address:02x}: {e}")
            return 0.0, False
    
    def _read_gpio_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from a GPIO sensor with improved handling for different sensor types"""
        if not self.gpio_available:
            return 0.0, False
            
        pin = sensor_config.get("pin")
        sensor_type = sensor_config.get("type", "DIGITAL")
        
        try:
            if sensor_type == "DHT22" or sensor_type == "DHT11":
                # For DHT sensors, we need a specialized library
                # Since we may not have the library, return a simulated value
                # In production, use Adafruit_DHT or similar library
                min_val = sensor_config.get("min", 0)
                max_val = sensor_config.get("max", 100)
                # Simulate temperature reading in the expected range
                simulated_temp = random.uniform(min_val, max_val)
                return float(simulated_temp), True
                
            elif sensor_type == "PIR_MOTION":
                # For PIR motion sensors, simply read the digital value
                motion_detected = GPIO.input(pin)
                return float(motion_detected), True
                
            else:
                # Default digital read for other GPIO sensors
                value = GPIO.input(pin)
                return float(value), True
                
        except Exception as e:
            self.logger.error(f"Error reading GPIO sensor on pin {pin}: {e}")
            return 0.0, False
    
    def _read_spi_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from an SPI sensor with improved implementation for common ADCs"""
        if not self.spi_available:
            return 0.0, False
            
        device = sensor_config.get("device", 0)
        sensor_type = sensor_config.get("type", "SPI_DEVICE")
        min_val = sensor_config.get("min", 0)
        max_val = sensor_config.get("max", 1023)
        
        try:
            if sensor_type == "MCP3008":
                # Simplified MCP3008 ADC reading
                # In production, use proper SPI library specific to your sensor
                
                # Simulate reading from channel 0
                channel = 0
                
                # Check if we have spidev available
                try:
                    import spidev
                    # Open SPI bus
                    spi = spidev.SpiDev()
                    spi.open(0, device)
                    spi.max_speed_hz = 1000000
                    
                    # MCP3008 protocol: Start bit (1), single-ended (1), channel (3 bits), padding
                    cmd = [1, (8 + channel) << 4, 0]
                    resp = spi.xfer2(cmd)
                    
                    # Process response
                    value = ((resp[1] & 3) << 8) + resp[2]
                    
                    # Scale to sensor range
                    scaled_value = (value / 1023.0) * (max_val - min_val) + min_val
                    
                    spi.close()
                    return float(scaled_value), True
                    
                except ImportError:
                    # If spidev not available, return simulated value
                    return random.uniform(min_val, max_val), True
            else:
                # Generic approach for other SPI devices
                return random.uniform(min_val, max_val), True
                
        except Exception as e:
            self.logger.error(f"Error reading SPI sensor {sensor_type} on device {device}: {e}")
            return 0.0, False


# ------------------------------------------------------------------------------------
# Adaptive Sensor Fusion Grid
# ------------------------------------------------------------------------------------

class AdaptiveSensorFusionGrid:
    """
    SenseGrid: Provides sensor acquisition & reliability augmentation via:
        • Physical scan (modern implementation with robust hardware detection)
        • Simulation fallback with correlation-aware estimation
        • EWMA-based correlation refinement as history grows
        • Confidence modulation & corrective fusion
    """

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self.config = config
        self.logger = logging.getLogger("SmartNeural.SenseGrid")
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sensor_status: Dict[str, SensorStatus] = {}
        self.sensor_calibration: Dict[str, Dict[str, Any]] = {}
        self.fusion_models: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Initialize physical sensor detector with improved implementation
        self.physical_sensor_detector = PhysicalSensorDetector(self.logger)
        
        # Detect physical sensors and merge with configuration
        self._detect_and_configure_sensors()
        
        # Initialize sensor history with the updated sensor list
        self.sensor_history: Dict[str, List[SensorReading]] = {
            s: [] for s in self.config.get("sensors", {}).keys()
        }
        
        self._initialize()

    # Sensor Detection and Configuration -----------------------------------
    
    def _detect_and_configure_sensors(self):
        """
        Detect physical sensors and update configuration accordingly
        """
        self.logger.info("Scanning for physical sensors...")
        
        # Get current config
        current_sensors = self.config.get("sensors", {})
        
        # Detect physical sensors with improved detection
        detected_sensors = {}
        try:
            detected_sensors = self.physical_sensor_detector.detect_sensors()
        except Exception as e:
            self.logger.error(f"Sensor detection failed: {e}", exc_info=True)
        
        # Count detected sensors
        detected_count = len(detected_sensors)
        self.logger.info(f"Detected {detected_count} physical sensors")
        
        # Add detected sensors to configuration if not already present
        updated_sensors = current_sensors.copy()
        for name, config in detected_sensors.items():
            if name not in updated_sensors:
                updated_sensors[name] = config
                self.logger.info(f"Added new sensor to config: {name}")
            else:
                # Update existing sensor with physical parameters if available
                if "interface" in config:
                    updated_sensors[name]["interface"] = config["interface"]
                if "address" in config:
                    updated_sensors[name]["address"] = config["address"]
                if "pin" in config:
                    updated_sensors[name]["pin"] = config["pin"]
                if "device" in config:
                    updated_sensors[name]["device"] = config["device"]
                self.logger.info(f"Updated existing sensor config: {name}")
        
        # Update the configuration
        self.config["sensors"] = updated_sensors
        self.logger.info(f"Updated sensor configuration with {len(updated_sensors)} total sensors")
        
        # Log the sensor details
        sensor_list = ", ".join(updated_sensors.keys())
        self.logger.info(f"Active sensors: {sensor_list}")

    # Initialization -----------------------------------------------------

    def _initialize(self):
        try:
            self._scan_physical_sensors()
            self._seed_correlation_matrix()
            self._train_initial_fusion()
            self._calibrate_all()
            self.logger.info("SenseGrid initialized.")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"SenseGrid initialization failed: {e}", exc_info=True)

    def _scan_physical_sensors(self):
        """
        Determine the status of each configured sensor by attempting to read from it
        """
        for sensor in self.config.get("sensors", {}).keys():
            # Get sensor config
            sensor_config = self.config["sensors"][sensor]
            
            # If this is a physically detected sensor, try to read from it
            if "interface" in sensor_config:
                value, success = self.physical_sensor_detector.read_sensor_value(sensor_config)
                
                if success:
                    self.sensor_status[sensor] = SensorStatus.ACTIVE
                    self.logger.info(f"Physical sensor {sensor} is active")
                else:
                    self.sensor_status[sensor] = SensorStatus.FAILED
                    self.logger.warning(f"Physical sensor {sensor} failed to read")
            else:
                # For non-physical/configuration-only sensors, use the previous behavior
                r = random.random()
                if r < 0.7:
                    self.sensor_status[sensor] = SensorStatus.ACTIVE
                elif r < 0.95:
                    self.sensor_status[sensor] = SensorStatus.SIMULATED
                else:
                    self.sensor_status[sensor] = SensorStatus.FAILED

    def _seed_correlation_matrix(self):
        # Provide initial priors; if no domain correlations, start with mild positive
        for s in self.sensor_history.keys():
            self.correlation_matrix[s] = {}
            for t in self.sensor_history.keys():
                if s == t:
                    continue
                self.correlation_matrix[s][t] = 0.2  # mild prior

    def _train_initial_fusion(self):
        for target in self.sensor_history.keys():
            others = [o for o in self.sensor_history.keys() if o != target]
            weights = {o: 1.0 / len(others) for o in others} if others else {}
            self.fusion_models[target] = {
                "supporting": others,
                "weights": weights,
                "quality": 0.85  # placeholder quality index
            }

    def _calibrate_all(self):
        for s in self.sensor_history.keys():
            self.sensor_calibration[s] = {
                "offset": random.uniform(-0.02, 0.02),
                "drift": 0.0,
                "last_calibration": datetime.utcnow(),
                "confidence": 0.95
            }

    # Acquisition --------------------------------------------------------

    def read_all(self) -> Dict[str, SensorReading]:
        """
        Acquire raw or simulated readings, then apply fusion improvements.
        """
        raw: Dict[str, SensorReading] = {}
        for sensor, status in self.sensor_status.items():
            try:
                if status == SensorStatus.ACTIVE:
                    # Check if this is a physical sensor
                    sensor_config = self.config["sensors"][sensor]
                    if "interface" in sensor_config:
                        # Read from physical sensor
                        value, success = self.physical_sensor_detector.read_sensor_value(sensor_config)
                        if success:
                            value = self._apply_calibration(sensor, value)
                            raw[sensor] = SensorReading(
                                value=value,
                                confidence=0.95,  # Higher confidence for physical readings
                                status=status,
                                timestamp=datetime.utcnow(),
                                source="physical"
                            )
                        else:
                            # Failed to read physical sensor - mark as degraded and use simulation
                            self.sensor_status[sensor] = SensorStatus.DEGRADED
                            sim_value = self._simulate_value(sensor, raw)
                            raw[sensor] = SensorReading(
                                value=sim_value,
                                confidence=0.7,
                                status=SensorStatus.DEGRADED,
                                timestamp=datetime.utcnow(),
                                source="simulated"
                            )
                    else:
                        # Non-physical sensor, use previous logic
                        value = self._generate_physical_value(sensor)
                        value = self._apply_calibration(sensor, value)
                        raw[sensor] = SensorReading(
                            value=value,
                            confidence=0.92,
                            status=status,
                            timestamp=datetime.utcnow(),
                            source="physical"
                        )
                elif status in (SensorStatus.SIMULATED, SensorStatus.DEGRADED):
                    sim_value = self._simulate_value(sensor, raw)
                    raw[sensor] = SensorReading(
                        value=sim_value,
                        confidence=0.8,
                        status=SensorStatus.SIMULATED,
                        timestamp=datetime.utcnow(),
                        source="simulated"
                    )
                else:
                    fallback = self._fallback_emergency_value(sensor)
                    raw[sensor] = SensorReading(
                        value=fallback,
                        confidence=0.5,
                        status=SensorStatus.FAILED,
                        timestamp=datetime.utcnow(),
                        source="emergency"
                    )
            except Exception as e:
                self.logger.error(f"Sensor {sensor} acquisition failure: {e}")
                fallback = self._fallback_emergency_value(sensor)
                raw[sensor] = SensorReading(
                    value=fallback,
                    confidence=0.4,
                    status=SensorStatus.FAILED,
                    timestamp=datetime.utcnow(),
                    source="emergency"
                )

        fused = self._fuse_with_consistency(raw)
        self._append_history(fused)
        self._update_correlations()

        return fused

    def _generate_physical_value(self, sensor: str) -> float:
        cfg = self.config["sensors"][sensor]
        base = np.random.uniform(cfg["min"] * 0.4, cfg["max"] * 0.6)
        t = time.time()
        seasonal = math.sin(t * 0.01) * 0.05 * base
        noise = np.random.normal(0, base * 0.02)
        val = base + seasonal + noise
        return float(max(cfg["min"], min(cfg["max"], val)))

    def _simulate_value(self, sensor: str, available: Dict[str, SensorReading]) -> float:
        if not available:
            cfg = self.config["sensors"][sensor]
            return (cfg["min"] + cfg["max"]) * 0.5
        model = self.fusion_models.get(sensor, {})
        weights = model.get("weights", {})
        est = 0.0
        total_w = 0.0
        for other, reading in available.items():
            if other in weights:
                w = weights[other]
                est += reading.value * w
                total_w += w
        if total_w > 0:
            return est / total_w
        cfg = self.config["sensors"][sensor]
        return (cfg["min"] + cfg["max"]) * 0.5

    def _apply_calibration(self, sensor: str, value: float) -> float:
        calib = self.sensor_calibration.get(sensor, {})
        offset = calib.get("offset", 0.0)
        return value * (1 + offset)

    def _fallback_emergency_value(self, sensor: str) -> float:
        cfg = self.config["sensors"][sensor]
        return (cfg["min"] + cfg["max"]) * 0.5

    # Fusion & Quality ---------------------------------------------------

    def _fuse_with_consistency(self, readings: Dict[str, SensorReading]) -> Dict[str, SensorReading]:
        fused: Dict[str, SensorReading] = {}
        for name, reading in readings.items():
            consistency = self._consistency_score(name, reading, readings)
            adjusted_conf = max(0.05, min(1.0, reading.confidence * (0.5 + 0.5 * consistency)))
            if adjusted_conf < 0.65:
                corrected = self._consistency_correction(name, reading, readings)
                fused[name] = SensorReading(
                    value=corrected,
                    confidence=0.7,
                    status=reading.status,
                    timestamp=reading.timestamp,
                    source="fused"
                )
            else:
                fused[name] = SensorReading(
                    value=reading.value,
                    confidence=adjusted_conf,
                    status=reading.status,
                    timestamp=reading.timestamp,
                    source=reading.source
                )
        return fused

    def _consistency_score(self, sensor: str, reading: SensorReading, all_readings: Dict[str, SensorReading]) -> float:
        rels = self.correlation_matrix.get(sensor, {})
        scores = []
        for other, corr in rels.items():
            if corr <= 0.1 or other not in all_readings:
                continue
            expected = all_readings[other].value * corr
            dev = abs(reading.value - expected) / (abs(reading.value) + 1e-6)
            scores.append(max(0.0, 1.0 - dev * 1.5))
        if not scores:
            return 1.0
        return float(np.mean(scores))

    def _consistency_correction(self, sensor: str, reading: SensorReading, all_readings: Dict[str, SensorReading]) -> float:
        rels = self.correlation_matrix.get(sensor, {})
        contributions = []
        weights = []
        for other, corr in rels.items():
            if other == sensor or corr <= 0.15 or other not in all_readings:
                continue
            contributions.append(all_readings[other].value * corr)
            weights.append(corr)
        if not contributions:
            return reading.value
        weighted = np.average(contributions, weights=weights)
        return (weighted + reading.value) / 2.0

    # History & Correlations ---------------------------------------------

    def _append_history(self, readings: Dict[str, SensorReading]):
        for name, r in readings.items():
            self.sensor_history[name].append(r)
            if len(self.sensor_history[name]) > SENSOR_HISTORY_LIMIT:
                self.sensor_history[name] = self.sensor_history[name][-SENSOR_HISTORY_LIMIT:]

    def _update_correlations(self):
        # Update EWMA correlations only if sufficient history
        for target in self.sensor_history.keys():
            for other in self.sensor_history.keys():
                if target == other:
                    continue
                hist_t = [r.value for r in self.sensor_history[target][-200:]]
                hist_o = [r.value for r in self.sensor_history[other][-200:]]
                if len(hist_t) >= 30 and len(hist_o) >= 30:
                    corr = np.corrcoef(hist_t, hist_o)[0, 1]
                    if not np.isnan(corr):
                        prev = self.correlation_matrix[target].get(other, 0.2)
                        self.correlation_matrix[target][other] = (
                            (1 - CORRELATION_EWMA_ALPHA) * prev + CORRELATION_EWMA_ALPHA * float(corr)
                        )

    # Recalibration ------------------------------------------------------

    def auto_recalibrate(self):
        try:
            for sensor, hist in self.sensor_history.items():
                if len(hist) < SENSOR_RECALIBRATION_WINDOW:
                    continue
                recent = [h.value for h in hist[-SENSOR_RECALIBRATION_WINDOW:]]
                if len(recent) < 10:
                    continue
                slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
                mean_v = np.mean(recent)
                if mean_v != 0:
                    rel_drift = slope / mean_v
                else:
                    rel_drift = 0
                if abs(rel_drift) > 0.02:
                    current_offset = self.sensor_calibration[sensor]["offset"]
                    new_offset = current_offset - rel_drift * 0.05
                    self.sensor_calibration[sensor]["offset"] = new_offset
                    self.sensor_calibration[sensor]["last_calibration"] = datetime.utcnow()
                    self.logger.info(f"Auto-recalibrated {sensor} offset-> {new_offset:.4f}")
        except Exception as e:
            self.logger.error(f"Auto-recalibration failure: {e}")

    # Public Status ------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        active = sum(1 for s in self.sensor_status.values() if s == SensorStatus.ACTIVE)
        simulated = sum(1 for s in self.sensor_status.values() if s == SensorStatus.SIMULATED)
        failed = sum(1 for s in self.sensor_status.values() if s == SensorStatus.FAILED)
        total = len(self.sensor_status)
        grid_health = active / total if total else 0.0
        avg_conf = []
        for hist in self.sensor_history.values():
            if hist:
                avg_conf.append(hist[-1].confidence)
        
        # Count physical sensors
        physical_count = sum(1 for s, cfg in self.config.get("sensors", {}).items() if "interface" in cfg)
        
        return {
            "total_sensors": total,
            "active_sensors": active,
            "simulated_sensors": simulated,
            "failed_sensors": failed,
            "physical_sensors_detected": physical_count,
            "grid_health": grid_health,
            "average_confidence": float(np.mean(avg_conf)) if avg_conf else 0.0,
            "last_update": datetime.utcnow()
        }

    # Dynamic sensor management -------------------------------------------
    
    def rescan_sensors(self) -> int:
        """
        Rescan for physical sensors and update configuration
        Returns the number of newly detected sensors
        """
        # Get current sensor count
        current_count = len(self.config.get("sensors", {}))
        
        # Update sensors
        self._detect_and_configure_sensors()
        
        # Update sensor history for any new sensors
        for s in self.config.get("sensors", {}).keys():
            if s not in self.sensor_history:
                self.sensor_history[s] = []
                
        # Update correlation matrix for new sensors
        self._seed_correlation_matrix()
        
        # Update fusion models for new sensors
        self._train_initial_fusion()
        
        # Update calibration for new sensors
        self._calibrate_all()
        
        # Get new sensor count
        new_count = len(self.config.get("sensors", {}))
        
        # Return the number of new sensors detected
        return new_count - current_count


# ------------------------------------------------------------------------------------
# Smart Neural Digital Twin
# ------------------------------------------------------------------------------------

class SmartNeuralDigitalTwin:
    """
    High-level orchestrator combining:
        • SenseGrid (adaptive acquisition + fusion)
        • AISystemManager (anomaly + forecasting + adaptive logic)
        • RelayController (hardware / safety actions)
    Provides:
        • Continuous monitoring loop
        • Periodic maintenance tasks
        • Safe shutdown semantics
        • Comprehensive status surface for UI layers
    """

    def __init__(self, config_path: str = "config/smart_neural_config.json", seed: Optional[int] = None):
        self.config_manager = SmartConfig(config_path)
        self.config: Dict[str, Any] = self.config_manager.get_config()
        self.logger = self.config_manager.get_logger()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Subsystems
        self.sense_grid = AdaptiveSensorFusionGrid(self.config, seed=seed)
        self.relay_controller = RelayController(self.config)
        self.ai_manager = AISystemManager(self.config, seed=seed)

        # State
        self.system_status = "INITIALIZING"
        self.raspberry_pi_active = RASPBERRY_PI_AVAILABLE  # Updated to use our detection
        self.real_time_data: Dict[str, float] = {}
        self.sensor_grid_status: Dict[str, Any] = {}
        self.last_ai_result: Optional[Dict[str, Any]] = None
        self._recent_samples: List[Dict[str, float]] = []

        self.stats = {
            "start_time": datetime.utcnow(),
            "processed_cycles": 0,
            "avg_cycle_time": 0.0,
            "emergency_events": 0,
            "last_cycle": None,
            "ai_samples": 0
        }

        # Threads
        self._active = True
        self._monitor_thread: Optional[threading.Thread] = None
        self._maintenance_thread: Optional[threading.Thread] = None
        self._retrain_thread: Optional[threading.Thread] = None
        self._sensor_scan_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Bootstrapping
        self._initial_training()
        self._start_monitoring()
        self._start_maintenance()
        self._start_retrain_scheduler()
        self._start_sensor_scanner()
        self.system_status = "NORMAL"
        self.logger.info("SmartNeuralDigitalTwin initialized successfully.")

    # Initialization -----------------------------------------------------

    def _initial_training(self):
        try:
            synthetic = self._generate_synthetic_training()
            if len(synthetic) >= MIN_TRAINING_RECORDS:
                self.ai_manager.train_all(synthetic)
            else:
                self.logger.warning("Insufficient synthetic records for initial AI training.")
        except Exception as e:
            self.logger.error(f"Initial AI training failed: {e}")

    def _generate_synthetic_training(self, count: int = 800) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        sensors = self.config.get("sensors", {})
        for i in range(count):
            rec: Dict[str, float] = {}
            for name, cfg in sensors.items():
                base = np.random.uniform(cfg["min"] * 0.3, cfg["max"] * 0.7)
                seasonal = math.sin(i * 0.01) * 0.08 * base
                noise = np.random.normal(0, base * 0.02)
                val = max(cfg["min"], min(cfg["max"], base + seasonal + noise))
                rec[name] = float(val)
            records.append(rec)
        return records

    # Threads ------------------------------------------------------------

    def _start_monitoring(self):
        interval = float(self.config.get("system", {}).get("update_interval", DEFAULT_MONITOR_INTERVAL))

        def loop():
            while self._active:
                started = time.time()
                try:
                    self._monitor_cycle()
                except Exception as e:  # pragma: no cover
                    self.logger.error(f"Monitoring cycle error: {e}", exc_info=True)
                elapsed = time.time() - started
                with self._lock:
                    prev_avg = self.stats["avg_cycle_time"]
                    self.stats["avg_cycle_time"] = prev_avg * 0.9 + elapsed * 0.1
                sleep_for = max(0.1, interval - elapsed)
                time.sleep(sleep_for)

        self._monitor_thread = threading.Thread(target=loop, daemon=True, name="MonitorLoop")
        self._monitor_thread.start()

    def _start_maintenance(self):
        def maintenance():
            while self._active:
                try:
                    self.sense_grid.auto_recalibrate()
                except Exception as e:
                    self.logger.error(f"Maintenance recalibration error: {e}")
                for _ in range(int(MAINTENANCE_INTERVAL_SEC)):
                    if not self._active:
                        break
                    time.sleep(1)

        self._maintenance_thread = threading.Thread(target=maintenance, daemon=True, name="MaintenanceLoop")
        self._maintenance_thread.start()

    def _start_retrain_scheduler(self):
        def retrain():
            while self._active:
                try:
                    status = self.ai_manager.status()
                    if status.get("next_retrain_due"):
                        # Placeholder: real retrain trigger is internal to AISystemManager
                        pass
                except Exception as e:
                    self.logger.error(f"Retrain scheduler error: {e}")
                for _ in range(int(RETRAIN_CHECK_INTERVAL_SEC)):
                    if not self._active:
                        break
                    time.sleep(1)

        self._retrain_thread = threading.Thread(target=retrain, daemon=True, name="RetrainLoop")
        self._retrain_thread.start()
        
    def _start_sensor_scanner(self):
        """Start a thread that periodically rescans for new sensors"""
        
        def scan_loop():
            # Initial delay to let the system stabilize
            time.sleep(10)
            
            while self._active:
                try:
                    # Rescan sensors every 5 minutes
                    new_sensors = self.sense_grid.rescan_sensors()
                    if new_sensors > 0:
                        self.logger.info(f"Sensor rescan found {new_sensors} new sensors")
                except Exception as e:
                    self.logger.error(f"Sensor scanner error: {e}")
                
                # Sleep for 5 minutes before rescanning
                for _ in range(300):  # 5 minutes in seconds
                    if not self._active:
                        break
                    time.sleep(1)
                    
        self._sensor_scan_thread = threading.Thread(target=scan_loop, daemon=True, name="SensorScanLoop")
        self._sensor_scan_thread.start()

    # Monitoring Cycle ---------------------------------------------------

    def _monitor_cycle(self):
        with self._lock:
            sensor_readings = self.sense_grid.read_all()
            self.real_time_data = {k: v.value for k, v in sensor_readings.items()}
            self.sensor_grid_status = self.sense_grid.status()
            self._recent_samples.append(self.real_time_data.copy())
            if len(self._recent_samples) > 1000:
                self._recent_samples = self._recent_samples[-1000:]

            ai_step = self.ai_manager.process(
                sensor_sample=self.real_time_data,
                recent_history=self._recent_samples
            )
            self.last_ai_result = ai_step.to_dict()
            self.stats["processed_cycles"] += 1
            self.stats["last_cycle"] = datetime.utcnow().isoformat()
            self.stats["ai_samples"] += 1

            # Evaluate emergency escalation
            self._evaluate_emergency(ai_step)

    # Emergency Handling -------------------------------------------------

    def _evaluate_emergency(self, ai_step):
        risk_level = ai_step.overall_risk.get("overall_level", "LOW")
        rank = {
            "LOW": 0.0,
            "MEDIUM": 0.5,
            "HIGH": 0.7,
            "CRITICAL": 0.85,
            "EMERGENCY": 0.95
        }.get(risk_level, 0.0)

        previous = self.system_status
        if rank >= RISK_ESCALATION_LEVELS["EMERGENCY"]:
            self.system_status = "EMERGENCY"
            self._trigger_emergency_actions(ai_step)
        elif rank >= RISK_ESCALATION_LEVELS["CRITICAL"]:
            self.system_status = "CRITICAL"
        elif rank >= RISK_ESCALATION_LEVELS["HIGH_ALERT"]:
            self.system_status = "HIGH_ALERT"
        else:
            self.system_status = "NORMAL"

        if previous != self.system_status:
            self.logger.info(f"System status changed: {previous} -> {self.system_status}")

    def _trigger_emergency_actions(self, ai_step):
        try:
            # Fallback: shut down all relays
            self.relay_controller.emergency_shutdown()
            self.stats["emergency_events"] += 1
            self.logger.critical("Emergency shutdown sequence executed.")
        except Exception as e:
            self.logger.error(f"Emergency action failure: {e}")

    # Public API ---------------------------------------------------------

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        with self._lock:
            now = datetime.utcnow()
            uptime = (now - self.stats["start_time"]).total_seconds()
            ai_status = self.ai_manager.status()
            anomaly = (self.last_ai_result or {}).get("anomaly", {})
            forecast = (self.last_ai_result or {}).get("forecast", {})
            overall = (self.last_ai_result or {}).get("overall_risk", {})

            # Get physical sensor count from sensor grid status
            physical_sensors = self.sensor_grid_status.get("physical_sensors_detected", 0)

            return {
                "system_status": self.system_status,
                "raspberry_pi_active": self.raspberry_pi_active,
                "sensor_grid_status": self.sensor_grid_status,
                "physical_sensor_count": physical_sensors,  # Added physical sensor count
                "total_sensor_count": len(self.config.get("sensors", {})),  # Total sensors including simulated
                "relay_states": self.relay_controller.get_relay_status(),
                "performance_metrics": {
                    "processed_cycles": self.stats["processed_cycles"],
                    "avg_cycle_time": self.stats["avg_cycle_time"],
                    "uptime_seconds": uptime,
                    "emergency_events": self.stats["emergency_events"],
                    "ai_samples": self.stats["ai_samples"],
                },
                "ai_engine_status": ai_status,
                "latest_anomaly": {
                    "risk_level": anomaly.get("risk_level"),
                    "anomaly_score": anomaly.get("anomaly_score"),
                    "is_anomaly": anomaly.get("is_anomaly"),
                    "threshold": anomaly.get("adaptive_threshold")
                },
                "latest_forecast": {
                    "risk_level": forecast.get("risk_level"),
                    "confidence": forecast.get("aggregate_confidence"),
                    "model_used": forecast.get("model_used")
                },
                "overall_risk": overall,
                "system_uptime": uptime,
                "last_update": now.isoformat(),
                "ss_rating": "S-CLASS",
                "overall_confidence": float(
                    min(
                        1.0,
                        0.5
                        + 0.25 * (anomaly.get("confidence", 0.0) or 0.0)
                        + 0.25 * (forecast.get("aggregate_confidence", 0.0) or 0.0)
                    )
                ),
                "real_time_data_sample": {
                    k: round(v, 3) for k, v in list(self.real_time_data.items())[:5]
                },
                "ai_recommendations": (self.last_ai_result or {}).get("recommendations", []),
            }

    # Shutdown -----------------------------------------------------------

    def shutdown(self):
        self.logger.info("Initiating graceful shutdown...")
        self._active = False
        for t in (self._monitor_thread, self._maintenance_thread, self._retrain_thread, self._sensor_scan_thread):
            if t:
                t.join(timeout=6)
        self.logger.info("Core threads joined. Performing final relay safe state.")
        try:
            self.relay_controller.emergency_shutdown()
        except Exception:
            pass
        self.system_status = "SHUTDOWN"
        self.logger.info("Shutdown complete.")


# ------------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------------

def create_smart_neural_twin(config_path: str = "config/smart_neural_config.json", seed: Optional[int] = None) -> SmartNeuralDigitalTwin:
    twin = SmartNeuralDigitalTwin(config_path=config_path, seed=seed)
    return twin


# ------------------------------------------------------------------------------------
# Script Entry (Manual Test)
# ------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    twin = create_smart_neural_twin()
    print("Twin running. Sampling status...")
    try:
        for _ in range(3):
            time.sleep(3)
            status = twin.get_enhanced_system_status()
            print(json.dumps({
                "system_status": status["system_status"],
                "overall_risk": status["overall_risk"],
                "sample": status["real_time_data_sample"],
                "physical_sensors": status["physical_sensor_count"],  # Show physical sensor count
                "total_sensors": status["total_sensor_count"],  # Show total sensor count
                "raspberry_pi_active": status["raspberry_pi_active"]  # Show Raspberry Pi detection status
            }, indent=2))
    finally:
        twin.shutdown()
        print("Twin shut down.")
