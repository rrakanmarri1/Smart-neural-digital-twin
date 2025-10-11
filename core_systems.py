from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# External systems
from config_and_logging import SmartConfig  # type: ignore

# Raspberry Pi specific imports for physical sensor detection
try:
    import RPi.GPIO as GPIO
    import smbus2 as smbus
    import board
    import busio
    RASPBERRY_PI_AVAILABLE = True
except ImportError:  # pragma: no cover
    RASPBERRY_PI_AVAILABLE = False

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
GPIO_SENSOR_PINS = [4, 5, 6, 13, 16, 17, 18, 19, 20, 21, 22, 23]  # Example GPIO pins to check
SPI_DEVICES = [0, 1]  # SPI device numbers to check

# Known I2C sensor addresses and their types
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
    0x57: {"type": "MAX30102", "name": "pulse_oximeter", "min": 0, "max": 100}
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
    Detects physical sensors connected to Raspberry Pi
    Supports I2C, GPIO, and SPI interfaces
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.i2c_available = False
        self.gpio_available = False
        self.spi_available = False
        
        # Initialize interfaces if Raspberry Pi is available
        if RASPBERRY_PI_AVAILABLE:
            try:
                # Initialize I2C
                self.i2c_bus = smbus.SMBus(I2C_BUS_NUM)
                self.i2c_available = True
            except Exception as e:
                self.logger.warning(f"I2C initialization failed: {e}")
            
            try:
                # Initialize GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                self.gpio_available = True
            except Exception as e:
                self.logger.warning(f"GPIO initialization failed: {e}")
                
            try:
                # Initialize SPI via Adafruit's busio
                self.spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
                self.spi_available = True
            except Exception as e:
                self.logger.warning(f"SPI initialization failed: {e}")
    
    def detect_sensors(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan all available interfaces for connected sensors
        Returns a dictionary of detected sensors with their configuration
        """
        detected_sensors = {}
        
        # Detect I2C sensors if available
        if self.i2c_available:
            i2c_sensors = self._scan_i2c_sensors()
            detected_sensors.update(i2c_sensors)
            
        # Detect GPIO sensors if available
        if self.gpio_available:
            gpio_sensors = self._scan_gpio_sensors()
            detected_sensors.update(gpio_sensors)
            
        # Detect SPI sensors if available (simplified)
        if self.spi_available:
            spi_sensors = self._scan_spi_sensors()
            detected_sensors.update(spi_sensors)
            
        return detected_sensors
    
    def _scan_i2c_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Scan the I2C bus for known sensor addresses"""
        detected = {}
        if not self.i2c_available:
            return detected
            
        for address in range(0x03, 0x77 + 1):
            try:
                self.i2c_bus.read_byte(address)
                # Device exists at this address
                if address in KNOWN_I2C_SENSORS:
                    sensor_info = KNOWN_I2C_SENSORS[address].copy()
                    sensor_name = f"{sensor_info['name']}_{address:02x}"
                    detected[sensor_name] = {
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
                    detected[sensor_name] = {
                        "type": "UNKNOWN",
                        "interface": "I2C",
                        "address": address,
                        "min": 0,
                        "max": 100
                    }
                    self.logger.info(f"Detected unknown I2C device at address 0x{address:02x}")
            except Exception:
                # No device at this address
                pass
                
        return detected
    
    def _scan_gpio_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Set up GPIO pins for potential sensors"""
        detected = {}
        if not self.gpio_available:
            return detected
            
        for pin in GPIO_SENSOR_PINS:
            try:
                # Configure as input
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                
                # For demonstration, add as digital sensor
                # In practice, you would need to identify specific sensor types
                sensor_name = f"digital_sensor_gpio{pin}"
                detected[sensor_name] = {
                    "type": "DIGITAL",
                    "interface": "GPIO",
                    "pin": pin,
                    "min": 0,
                    "max": 1
                }
                self.logger.info(f"Set up GPIO sensor on pin {pin}")
            except Exception as e:
                self.logger.debug(f"GPIO pin {pin} setup failed: {e}")
                
        return detected
    
    def _scan_spi_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Detect SPI sensors (simplified implementation)"""
        detected = {}
        if not self.spi_available:
            return detected
            
        # This is a placeholder - actual SPI device detection 
        # would depend on specific sensor protocols
        for device in SPI_DEVICES:
            try:
                # In a real implementation, you would try to communicate
                # with the device using specific commands
                sensor_name = f"spi_sensor_{device}"
                detected[sensor_name] = {
                    "type": "SPI_SENSOR",
                    "interface": "SPI",
                    "device": device,
                    "min": 0,
                    "max": 1023
                }
                self.logger.info(f"Detected SPI device on device {device}")
            except Exception as e:
                self.logger.debug(f"SPI device {device} detection failed: {e}")
                
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
            self.logger.error(f"Error reading sensor {sensor_config}: {e}")
            return 0.0, False
    
    def _read_i2c_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from an I2C sensor"""
        if not self.i2c_available:
            return 0.0, False
            
        address = sensor_config.get("address")
        sensor_type = sensor_config.get("type")
        
        try:
            if sensor_type == "BME280":
                # Simplified BME280 reading - in real implementation use proper BME280 library
                temp_raw = self.i2c_bus.read_word_data(address, 0xFA)
                return (temp_raw / 100.0), True
                
            elif sensor_type == "ADS1115":
                # Simplified ADS1115 ADC reading
                value = self.i2c_bus.read_word_data(address, 0x00)
                return float(value), True
                
            elif sensor_type == "MPU6050":
                # Simplified accelerometer reading
                value = self.i2c_bus.read_word_data(address, 0x3B)
                return float(value) / 16384.0, True
                
            else:
                # Generic read for unknown sensors - read first register
                value = self.i2c_bus.read_byte(address)
                return float(value), True
                
        except Exception as e:
            self.logger.error(f"Error reading I2C sensor at address 0x{address:02x}: {e}")
            return 0.0, False
    
    def _read_gpio_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from a GPIO sensor"""
        if not self.gpio_available:
            return 0.0, False
            
        pin = sensor_config.get("pin")
        
        try:
            # For digital reading
            value = GPIO.input(pin)
            return float(value), True
        except Exception as e:
            self.logger.error(f"Error reading GPIO sensor on pin {pin}: {e}")
            return 0.0, False
    
    def _read_spi_sensor(self, sensor_config: Dict[str, Any]) -> Tuple[float, bool]:
        """Read data from an SPI sensor"""
        if not self.spi_available:
            return 0.0, False
            
        # Simplified SPI read - in real implementation, use proper SPI protocol for specific sensors
        try:
            # Placeholder for actual SPI reading code
            # Would use spidev or similar to communicate with specific sensor
            return random.uniform(sensor_config["min"], sensor_config["max"]), True
        except Exception as e:
            self.logger.error(f"Error reading SPI sensor {sensor_config.get('device')}: {e}")
            return 0.0, False


# ------------------------------------------------------------------------------------
# Adaptive Sensor Fusion Grid
# ------------------------------------------------------------------------------------

class AdaptiveSensorFusionGrid:
    """
    SenseGrid: Provides sensor acquisition & reliability augmentation via:
        • Physical scan (placeholder logic – extend with actual drivers)
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
        
        # Initialize physical sensor detector
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
        
        # Detect physical sensors
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
        self.raspberry_pi_active = bool(self.config.get("raspberry_pi", {}).get("active", False))
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
                "total_sensors": status["total_sensor_count"]  # Show total sensor count
            }, indent=2))
    finally:
        twin.shutdown()
        print("Twin shut down.")
