from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# -------------------------------------------------------------------------------------------------
# Log Data Structures
# -------------------------------------------------------------------------------------------------

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    logger: str
    message: str
    file: str
    line: int
    function: str
    thread: str


@dataclass
class SmartTheme:
    primary: str = "#1a365d"
    secondary: str = "#2d3748"
    accent: str = "#3182ce"
    background: str = "#0f172a"
    card: str = "#1e293b"
    text: str = "#f7fafc"
    success: str = "#38a169"
    warning: str = "#d69e2e"
    danger: str = "#e53e3e"
    info: str = "#4299e1"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# -------------------------------------------------------------------------------------------------
# Sensor Configuration Types
# -------------------------------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Configuration for a single sensor"""
    name: str
    min: float
    max: float
    critical: float
    unit: str
    weight: float = 1.0
    type: str = "UNKNOWN"
    interface: Optional[str] = None
    is_physical: bool = False
    is_simulated: bool = True
    pin: Optional[int] = None
    i2c_address: Optional[str] = None
    description: str = ""
    added_at: Optional[str] = None
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'SensorConfig':
        """Create a sensor configuration from a dictionary"""
        return cls(
            name=name,
            min=data.get("min", 0.0),
            max=data.get("max", 100.0),
            critical=data.get("critical", data.get("max", 100.0) * 0.9),
            unit=data.get("unit", ""),
            weight=data.get("weight", 1.0),
            type=data.get("type", "UNKNOWN"),
            interface=data.get("interface"),
            is_physical=data.get("is_physical", False),
            is_simulated=data.get("is_simulated", True),
            pin=data.get("pin"),
            i2c_address=data.get("i2c_address"),
            description=data.get("description", ""),
            added_at=data.get("added_at")
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for config storage"""
        result = asdict(self)
        # Remove None values for cleaner config
        return {k: v for k, v in result.items() if v is not None}


# -------------------------------------------------------------------------------------------------
# Advanced In-Memory Log Handler
# -------------------------------------------------------------------------------------------------

class AdvancedLogHandler(logging.Handler):
    """
    Memory ring buffer + metadata capture.
    Thread-safe append with size caps.
    """
    def __init__(self, max_entries: int = 10000):
        super().__init__()
        self.max_entries = max_entries
        self.log_entries: List[LogEntry] = []
        self.lock = threading.RLock()

    def emit(self, record: logging.LogRecord):
        try:
            # Attempt to extract a deeper frame (best effort)
            frame = inspect.currentframe()
            depth = 0
            while frame and depth < 6:
                if frame.f_code.co_name == "emit":
                    frame = frame.f_back
                else:
                    break
                depth += 1

            file_name = getattr(record, "filename", "")
            line_no = getattr(record, "lineno", 0)
            func_name = getattr(record, "funcName", "")
            if frame:
                file_name = frame.f_code.co_filename
                line_no = frame.f_lineno
                func_name = frame.f_code.co_name

            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger=record.name,
                message=self.format(record),
                file=file_name,
                line=line_no,
                function=func_name,
                thread=record.threadName
            )

            with self.lock:
                self.log_entries.append(entry)
                if len(self.log_entries) > self.max_entries:
                    self.log_entries = self.log_entries[-self.max_entries:]
        except Exception as e:  # pragma: no cover
            print(f"Logging error: {e}", file=sys.stderr)


# -------------------------------------------------------------------------------------------------
# Performance Counters Filter (counts warnings/errors centrally)
# -------------------------------------------------------------------------------------------------

class _StatsUpdatingFilter(logging.Filter):
    def __init__(self, config_ref_getter):
        super().__init__()
        self._config_ref_getter = config_ref_getter

    def filter(self, record: logging.LogRecord) -> bool:
        cfg = self._config_ref_getter()
        if not cfg:
            return True
        try:
            if record.levelno >= logging.ERROR:
                cfg._performance_stats["errors_count"] += 1
            elif record.levelno == logging.WARNING:
                cfg._performance_stats["warnings_count"] += 1
        except Exception:
            pass
        return True


# -------------------------------------------------------------------------------------------------
# SmartConfig Singleton
# -------------------------------------------------------------------------------------------------

class SmartConfig:
    """
    Central configuration & logging manager.
    Implements:
        - Deep merged config (default + user)
        - Validation & type coercion
        - Environment variable overrides
        - Structured & rotating logging handlers
        - Accessor & update APIs
        - Dynamic sensor registration and management
    """
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls, config_path: str = "config/smart_neural_config.json"):
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config/smart_neural_config.json"):
        if getattr(self, "_initialized", False):
            return
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._last_modified = 0.0
        self._theme = SmartTheme()
        self._log_handler: Optional[AdvancedLogHandler] = None
        self._system_logger: Optional[logging.Logger] = None
        self._gpio_available = False
        self._cfg_lock = threading.RLock()
        
        # Dynamic sensor tracking
        self._sensor_registry: Dict[str, SensorConfig] = {}
        self._sensor_discovery_history: List[Dict[str, Any]] = []
        self._last_sensor_scan = datetime.utcnow()

        # Performance stats counters
        self._performance_stats = {
            "start_time": datetime.utcnow(),
            "config_reloads": 0,
            "errors_count": 0,
            "warnings_count": 0,
            "sensor_discoveries": 0
        }

        self._create_directory_structure()
        self._setup_advanced_logging()
        self._load_advanced_config()
        self._apply_env_overrides()
        self._validate_config()
        self._setup_system_components()
        self._initialized = True
        self.get_logger().info("🎯 SmartConfig initialized (S‑Class with dynamic sensor support)")

    # ---------------------------------------------------------------------------------
    # Directory & Logging Setup
    # ---------------------------------------------------------------------------------

    def _create_directory_structure(self):
        dirs = [
            "logs/system",
            "logs/performance",
            "logs/security",
            "logs/json",
            "models/ai",
            "models/anomaly",
            "models/prediction",
            "data/real_time",
            "data/historical",
            "data/backup",
            "config/backups",
            "config/sensors",
            "reports/daily",
            "reports/incidents",
            "cache/temp"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _setup_advanced_logging(self):
        try:
            from logging.handlers import RotatingFileHandler

            detailed_fmt = logging.Formatter(
                "%(asctime)s | %(name)-32s | %(levelname)-8s | %(filename)s:%(lineno)d | "
                "%(funcName)s | %(threadName)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            simple_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )

            # Rotating handlers (limit size)
            main_handler = RotatingFileHandler(
                "logs/system/main.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
            )
            err_handler = RotatingFileHandler(
                "logs/system/errors.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            perf_handler = RotatingFileHandler(
                "logs/performance/performance.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            console_handler = logging.StreamHandler(sys.stdout)
            json_handler = RotatingFileHandler(
                "logs/json/structured.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )

            main_handler.setFormatter(detailed_fmt)
            main_handler.setLevel(logging.INFO)

            err_handler.setFormatter(detailed_fmt)
            err_handler.setLevel(logging.ERROR)

            perf_handler.setFormatter(simple_fmt)
            perf_handler.setLevel(logging.INFO)

            console_handler.setFormatter(simple_fmt)
            console_handler.setLevel(logging.INFO)

            # JSON structured logs (minimal)
            class _JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    payload = {
                        "ts": datetime.fromtimestamp(record.created).isoformat(),
                        "lvl": record.levelname,
                        "logger": record.name,
                        "msg": record.getMessage(),
                        "file": record.filename,
                        "line": record.lineno,
                        "func": record.funcName,
                        "thread": record.threadName
                    }
                    return json.dumps(payload, ensure_ascii=False)

            json_handler.setFormatter(_JsonFormatter())
            json_handler.setLevel(logging.INFO)

            root_name = "SmartNeuralTwin"
            self._system_logger = logging.getLogger(root_name)
            self._system_logger.setLevel(logging.INFO)
            self._system_logger.handlers.clear()

            for h in (main_handler, err_handler, perf_handler, console_handler, json_handler):
                self._system_logger.addHandler(h)

            # In-memory handler
            self._log_handler = AdvancedLogHandler(max_entries=15_000)
            self._log_handler.setFormatter(detailed_fmt)
            self._log_handler.setLevel(logging.INFO)
            self._system_logger.addHandler(self._log_handler)

            # Stats updating filter
            stats_filter = _StatsUpdatingFilter(lambda: self)
            self._system_logger.addFilter(stats_filter)

            # Subsystem loggers
            self._setup_subsystem_loggers(stats_filter, detailed_fmt)

            self._system_logger.propagate = False
            self._system_logger.info("✅ Advanced logging system initialized")
        except Exception as e:  # pragma: no cover
            print(f"❌ Critical logging setup failed: {e}", file=sys.stderr)
            sys.exit(1)

    def _setup_subsystem_loggers(self, stats_filter: logging.Filter, formatter: logging.Formatter):
        subsystems = {
            "SmartNeural.AI": logging.INFO,
            "SmartNeural.AI.AnomalyCore": logging.INFO,
            "SmartNeural.AI.Part2": logging.INFO,
            "SmartNeural.Sensors": logging.INFO,
            "SmartNeural.Hardware": logging.INFO,
            "SmartNeural.Prediction": logging.INFO,
            "SmartNeural.Anomaly": logging.INFO,
            "SmartNeural.UI": logging.INFO,
            "SmartNeural.Security": logging.WARNING,
            "SmartNeural.Performance": logging.INFO
        }
        for name, level in subsystems.items():
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = False
            # Provide at least one handler if none (inherit formatting)
            if not lg.handlers:
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(formatter)
                sh.setLevel(level)
                sh.addFilter(stats_filter)
                lg.addHandler(sh)

    # ---------------------------------------------------------------------------------
    # Configuration Loading & Environment Overrides
    # ---------------------------------------------------------------------------------

    def _load_advanced_config(self):
        default_cfg = self._get_default_config()
        if self.config_path.exists():
            try:
                file_mtime = self.config_path.stat().st_mtime
                if file_mtime > self._last_modified:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        user_cfg = json.load(f)
                    merged = self._deep_merge(default_cfg, user_cfg)
                    self._config = merged
                    self._last_modified = file_mtime
                    self.get_logger().info(f"✅ Configuration loaded from {self.config_path}")
                else:
                    # unchanged => still use present config (or default)
                    if not self._config:
                        self._config = default_cfg
            except json.JSONDecodeError as e:
                self.get_logger().error(f"❌ Config JSON error: {e}, using defaults.")
                self._config = default_cfg
            except Exception as e:
                self.get_logger().error(f"❌ Config load failed: {e}, using defaults.")
                self._config = default_cfg
        else:
            self._create_default_config(default_cfg)
            self._config = default_cfg
            self.get_logger().info("✅ Default configuration created and loaded")
            
        # Initialize sensor registry from config
        self._initialize_sensor_registry()

    def _create_default_config(self, config: Dict[str, Any]) -> None:
        """Create the default configuration file if it doesn't exist"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.get_logger().error(f"❌ Default config creation failed: {e}")

    def _apply_env_overrides(self):
        """
        Environment variable pattern:
            SMART_TWIN__SECTION__SUBKEY__SUBSUBKEY=value
        Hyphens in values: interpreted literally.
        Numeric coercion attempted.
        """
        prefix = "SMART_TWIN__"
        overrides = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        if not overrides:
            return
        applied = 0
        for key, val in overrides.items():
            path_parts = key[len(prefix):].split("__")
            if not path_parts:
                continue
            cur = self._config
            for p in path_parts[:-1]:
                p_lower = p.lower()
                if p_lower not in cur or not isinstance(cur[p_lower], dict):
                    cur[p_lower] = {}
                cur = cur[p_lower]
            leaf = path_parts[-1].lower()
            # Type coercion attempt
            coerced: Any = val
            for cast in (int, float):
                try:
                    coerced = cast(coerced)
                    break
                except ValueError:
                    continue
            if str(coerced).lower() in ("true", "false"):
                coerced = str(coerced).lower() == "true"
            cur[leaf] = coerced
            applied += 1
        if applied:
            self.get_logger().info(f"🔧 Applied {applied} env override(s) to configuration")

    # ---------------------------------------------------------------------------------
    # Default Configuration
    # ---------------------------------------------------------------------------------

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.1",  # Updated schema version for dynamic sensors
            "meta": {
                "config_version": "3.2.0",  # Updated for dynamic sensors
                "build": "s-class",
                "generated_at": datetime.utcnow().isoformat()
            },
            "system": {
                "name": "Smart Neural Digital Twin - SS Rating",
                "version": "3.1.0",  # Updated for dynamic sensors
                "description": "Advanced Oil Field Disaster Prevention System with AI - SS Rating",
                "update_interval": 2.0,
                "max_memory_usage": "2GB",
                "data_retention_days": 30,
                "performance_mode": "SS_RATING",
                "timezone": "Asia/Riyadh",
                "language": "ar",
                "dynamic_sensors_enabled": True,  # Enable dynamic sensor detection
                "sensor_scan_interval": 300  # Seconds between sensor scans
            },
            "raspberry_pi": {
                "active": False,
                "gpio_mode": "BCM",
                "auto_detect_sensors": True,  # Auto-detect physical sensors
                "i2c_enabled": True,  # Enable I2C bus scanning
                "i2c_bus": 1,  # Default I2C bus number
                "relay_pins": {
                    "emergency_cooling": 17,
                    "pressure_release": 18,
                    "gas_venting": 22,
                    "main_shutdown": 23,
                    "backup_pump": 24,
                    "safety_valve": 25
                },
                "sensor_pins": {
                    "temperature": 2,
                    "pressure": 3,
                    "methane": 4,
                    "vibration": 5
                },
                "simulation_mode": True
            },
            "foresight_engine": {
                "scenarios_per_second": {"min": 100, "max": 1000, "default": 500},
                "prediction_horizons": {
                    "short_term": 6,
                    "medium_term": 24,
                    "long_term": 168
                },
                "confidence_thresholds": {"high": 0.9, "medium": 0.7, "low": 0.5},
                "monte_carlo_simulations": 1000,
                "adaptive_learning": True,
                "dynamic_feature_adaptation": True  # Handle changing feature sets
            },
            "prediction": {
                "sequence_length": 50,
                "horizons": {"short": 6, "medium": 24, "long": 72},
                "train": {
                    "epochs": 100,
                    "patience": 12,
                    "val_split": 0.15,
                    "lr": 0.001,
                    "retrain_hours": 12,
                    "auto_retrain_on_new_sensors": True  # Auto-retrain when sensors change
                }
            },
            "ai_models": {
                "isolation_forest": {
                    "contamination": 0.1,
                    "n_estimators": 200,
                    "max_features": 1.0,
                    "bootstrap": True
                },
                "lstm_models": {
                    "short_term": {"units": 128, "layers": 3, "dropout": 0.2},
                    "medium_term": {"units": 256, "layers": 4, "dropout": 0.3},
                    "long_term": {"units": 512, "layers": 5, "dropout": 0.4}
                },
                "anomaly_detection": {
                    "sensitivity": 0.85,
                    "window_size": 100,
                    "retrain_interval": 3600,
                    "ensemble_weights": [0.4, 0.3, 0.3],
                    "dynamic_features": {
                        "enabled": True,  # Enable dynamic feature handling
                        "retraining_threshold": 0.25,  # Retrain if more than 25% of features changed
                        "preserve_order": True  # Maintain feature order compatibility
                    }
                },
                "autoencoder": {
                    "encoding_dim": 32,
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            },
            "sensors": {
                "pressure": {"min": 0, "max": 200, "critical": 150, "unit": "bar", "weight": 0.25, "type": "ANALOG"},
                "temperature": {"min": -50, "max": 300, "critical": 200, "unit": "°C", "weight": 0.20, "type": "DIGITAL"},
                "methane": {"min": 0, "max": 5000, "critical": 1000, "unit": "ppm", "weight": 0.25, "type": "ANALOG"},
                "hydrogen_sulfide": {"min": 0, "max": 500, "critical": 50, "unit": "ppm", "weight": 0.15, "type": "ANALOG"},
                "vibration": {"min": 0, "max": 20, "critical": 8, "unit": "m/s²", "weight": 0.10, "type": "DIGITAL"},
                "flow": {"min": 0, "max": 500, "critical": 400, "unit": "L/min", "weight": 0.05, "type": "ANALOG"}
            },
            "emergency_protocols": {
                "auto_response": True,
                "response_timeout": 30,
                "notification_levels": ["warning", "critical", "emergency"],
                "escalation_procedures": {
                    "level_1": ["alert_team", "increase_monitoring"],
                    "level_2": ["activate_safety", "reduce_pressure"],
                    "level_3": ["emergency_shutdown", "notify_authorities"]
                },
                "confirmation_required": True
            },
            "data_processing": {
                "preprocessing": {
                    "normalization": True,
                    "outlier_detection": True,
                    "feature_scaling": "standard",
                    "window_size": 50,
                    "sequence_length": 50
                },
                "storage": {
                    "real_time_buffer": 1000,
                    "historical_days": 30,
                    "compression": True,
                    "backup_interval": 3600
                }
            },
            "performance": {
                "target_processing_time": 0.1,
                "max_memory_usage": "2GB",
                "cpu_utilization_limit": 0.8,
                "gpu_acceleration": True,
                "cache_size": "500MB"
            },
            "security": {
                "encryption_enabled": True,
                "access_logging": True,
                "max_login_attempts": 3,
                "session_timeout": 3600
            },
            # New section for sensor discovery settings
            "sensor_discovery": {
                "enabled": True,
                "scan_interval_seconds": 300,  # 5 minutes between scans
                "i2c_scan_addresses": [0x76, 0x77, 0x48, 0x68, 0x23, 0x40],  # Common I2C addresses
                "gpio_scan_pins": [2, 3, 4, 5, 6, 13, 19, 26],  # GPIO pins to scan
                "known_sensor_types": {
                    "BME280": {"addresses": [0x76, 0x77], "description": "Temperature/Humidity/Pressure"},
                    "ADS1115": {"addresses": [0x48], "description": "16-bit ADC"},
                    "MPU6050": {"addresses": [0x68], "description": "Accelerometer/Gyroscope"},
                    "BH1750": {"addresses": [0x23], "description": "Light Sensor"},
                    "HTU21D": {"addresses": [0x40], "description": "Humidity Sensor"}
                }
            }
        }

    # ---------------------------------------------------------------------------------
    # Deep Merge & Validation
    # ---------------------------------------------------------------------------------

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k in set(base.keys()).union(override.keys()):
            b_val = base.get(k)
            o_val = override.get(k)
            if isinstance(b_val, dict) and isinstance(o_val, dict):
                result[k] = self._deep_merge(b_val, o_val)
            elif o_val is not None:
                # Basic type coercion attempt
                if b_val is not None and type(b_val) != type(o_val):
                    try:
                        if isinstance(b_val, bool):
                            o_val = str(o_val).lower() in ("true", "1", "yes")
                        elif isinstance(b_val, int):
                            o_val = int(o_val)
                        elif isinstance(b_val, float):
                            o_val = float(o_val)
                        elif isinstance(b_val, str):
                            o_val = str(o_val)
                    except Exception:
                        self.get_logger().warning(f"⚠️ Type mismatch for key '{k}', keeping default")
                        o_val = b_val
                result[k] = o_val
            else:
                result[k] = b_val
        return result

    def _validate_config(self) -> bool:
        try:
            required_sections = [
                "system",
                "raspberry_pi",
                "foresight_engine",
                "ai_models",
                "sensors",
                "prediction"
            ]
            for sec in required_sections:
                if sec not in self._config:
                    self.get_logger().error(f"❌ Missing config section: {sec}")
                    return False

            # Sensors - allow dynamic sensors, but ensure core sensors present
            required_sensors = ["pressure", "temperature"]
            sensors_cfg = self._config["sensors"]
            for s in required_sensors:
                if s not in sensors_cfg:
                    self.get_logger().error(f"❌ Missing core sensor: {s}")
                    return False
                    
            # Validate all sensors have required parameters
            for sensor_name, sensor_cfg in sensors_cfg.items():
                for p in ["min", "max"]:
                    if p not in sensor_cfg:
                        self.get_logger().error(f"❌ Sensor {sensor_name} missing required param {p}")
                        return False
                
                # Add critical if not defined (90% of max)
                if "critical" not in sensor_cfg:
                    sensor_cfg["critical"] = sensor_cfg["max"] * 0.9
                    
                # Add unit if missing
                if "unit" not in sensor_cfg:
                    sensor_cfg["unit"] = ""
                    
                # Add weight if missing
                if "weight" not in sensor_cfg:
                    sensor_cfg["weight"] = 1.0

            # Prediction section
            pred_cfg = self._config["prediction"]
            if "sequence_length" not in pred_cfg or pred_cfg["sequence_length"] <= 0:
                self.get_logger().error("❌ Invalid prediction.sequence_length")
                return False
            for k in ["horizons", "train"]:
                if k not in pred_cfg:
                    self.get_logger().error(f"❌ prediction.{k} section missing")
                    return False

            # System interval
            if self._config["system"].get("update_interval", 0) <= 0:
                self.get_logger().error("❌ Invalid system update interval")
                return False

            # Performance CPU utilization
            cpu_limit = self._config["performance"].get("cpu_utilization_limit", 0)
            if not (0 < cpu_limit <= 1):
                self.get_logger().error("❌ Invalid performance.cpu_utilization_limit")
                return False

            self.get_logger().info("✅ Configuration validation passed")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Config validation failed: {e}")
            return False

    # ---------------------------------------------------------------------------------
    # System Components Setup
    # ---------------------------------------------------------------------------------

    def _setup_system_components(self):
        try:
            if self._config.get("raspberry_pi", {}).get("active"):
                self._setup_raspberry_pi()
            else:
                self.get_logger().info("🔧 Raspberry Pi simulation mode")
        except Exception as e:
            self.get_logger().error(f"❌ System component setup failed: {e}")

    def _setup_raspberry_pi(self):
        try:
            try:
                import RPi.GPIO as GPIO  # pragma: no cover
                gpio_mode = self._config["raspberry_pi"]["gpio_mode"].upper()
                GPIO.setwarnings(False)
                if gpio_mode == "BCM":
                    GPIO.setmode(GPIO.BCM)
                else:
                    GPIO.setmode(GPIO.BOARD)
                relay_pins = self._config["raspberry_pi"]["relay_pins"]
                for _, pin in relay_pins.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                self._gpio_available = True
                self.get_logger().info("✅ Raspberry Pi GPIO initialized")
            except (ImportError, RuntimeError) as e:
                self._gpio_available = False
                self._config["raspberry_pi"]["simulation_mode"] = True
                self.get_logger().info(f"🔧 GPIO unavailable, simulation mode enabled: {e}")
        except Exception as e:  # pragma: no cover
            self._gpio_available = False
            self._config["raspberry_pi"]["simulation_mode"] = True
            self.get_logger().error(f"❌ GPIO setup failed: {e}")

    # ---------------------------------------------------------------------------------
    # Dynamic Sensor Management
    # ---------------------------------------------------------------------------------
    
    def _initialize_sensor_registry(self) -> None:
        """Initialize the sensor registry from configuration"""
        self._sensor_registry = {}
        if "sensors" not in self._config:
            return
            
        # Convert all sensors to SensorConfig objects
        for sensor_name, sensor_data in self._config["sensors"].items():
            # Skip empty or invalid sensors
            if not isinstance(sensor_data, dict):
                continue
                
            # Create sensor config
            sensor = SensorConfig.from_dict(sensor_name, sensor_data)
            self._sensor_registry[sensor_name] = sensor
            
        self.get_logger().info(f"✅ Initialized sensor registry with {len(self._sensor_registry)} sensors")
        
    def register_sensor(self, sensor_name: str, sensor_config: Dict[str, Any]) -> bool:
        """
        Register a new sensor in the configuration
        Returns True if sensor was new or updated, False otherwise
        """
        try:
            with self._cfg_lock:
                # Add timestamp to config if not present
                if "added_at" not in sensor_config:
                    sensor_config["added_at"] = datetime.utcnow().isoformat()
                    
                is_new = sensor_name not in self._config["sensors"]
                    
                # Update sensors section
                self._config["sensors"][sensor_name] = sensor_config
                
                # Update registry
                sensor = SensorConfig.from_dict(sensor_name, sensor_config)
                self._sensor_registry[sensor_name] = sensor
                
                # Record discovery
                self._record_sensor_discovery(sensor_name, sensor_config, is_new)
                
                if is_new:
                    self.get_logger().info(f"✅ Registered new sensor: {sensor_name}")
                else:
                    self.get_logger().info(f"✅ Updated existing sensor: {sensor_name}")
                
                # Save configuration if new sensor
                if is_new:
                    self.save_config()
                    
                return True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to register sensor {sensor_name}: {e}")
            return False
            
    def _record_sensor_discovery(self, sensor_name: str, config: Dict[str, Any], is_new: bool) -> None:
        """Record sensor discovery in history"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_name": sensor_name,
            "is_new": is_new,
            "is_physical": config.get("is_physical", False),
            "type": config.get("type", "UNKNOWN")
        }
        self._sensor_discovery_history.append(entry)
        self._performance_stats["sensor_discoveries"] += 1
        
        # Keep history bounded
        if len(self._sensor_discovery_history) > 100:
            self._sensor_discovery_history = self._sensor_discovery_history[-100:]
            
    def unregister_sensor(self, sensor_name: str) -> bool:
        """Remove a sensor from the configuration"""
        try:
            with self._cfg_lock:
                if sensor_name not in self._config["sensors"]:
                    return False
                    
                # Remove from config and registry
                del self._config["sensors"][sensor_name]
                if sensor_name in self._sensor_registry:
                    del self._sensor_registry[sensor_name]
                    
                self.get_logger().info(f"✅ Unregistered sensor: {sensor_name}")
                self.save_config()
                return True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to unregister sensor {sensor_name}: {e}")
            return False
            
    def get_sensor_registry(self) -> Dict[str, SensorConfig]:
        """Get a copy of the sensor registry"""
        return {k: v for k, v in self._sensor_registry.items()}
        
    def get_physical_sensors(self) -> Dict[str, SensorConfig]:
        """Get only physical sensors from registry"""
        return {k: v for k, v in self._sensor_registry.items() if v.is_physical}
        
    def get_simulated_sensors(self) -> Dict[str, SensorConfig]:
        """Get only simulated sensors from registry"""
        return {k: v for k, v in self._sensor_registry.items() if v.is_simulated}
        
    def get_sensor_discovery_history(self) -> List[Dict[str, Any]]:
        """Get sensor discovery history"""
        return self._sensor_discovery_history.copy()
        
    def get_sensor_config(self, sensor_name: str) -> Optional[SensorConfig]:
        """Get configuration for a specific sensor"""
        return self._sensor_registry.get(sensor_name)

    # ---------------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------------

    def get_logger(self, name: str = "SmartNeuralTwin") -> logging.Logger:
        if name == "SmartNeuralTwin":
            return self._system_logger  # type: ignore
        return logging.getLogger(name)

    def get_config(self, key: str = None, default: Any = None) -> Any:
        try:
            if key is None:
                return self._config.copy()
            val: Any = self._config
            for part in key.split("."):
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        except Exception:
            return default

    def update_config(self, updates: Dict[str, Any], save: bool = True) -> bool:
        try:
            with self._cfg_lock:
                merged = self._deep_merge(self._config, updates)
                old_config = self._config
                self._config = merged
                if not self._validate_config():
                    self._config = old_config
                    self.get_logger().error("❌ Config update reverted due to validation failure")
                    return False
                    
                # Update sensor registry if sensors changed
                if "sensors" in updates:
                    self._initialize_sensor_registry()
                    
                if save:
                    self.save_config()
                self.get_logger().info("✅ Configuration updated")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Config update failed: {e}")
            return False

    def save_config(self) -> bool:
        try:
            # Create a backup before saving
            if self.config_path.exists():
                backup_path = Path("config/backups") / f"config_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, "r", encoding="utf-8") as src:
                    with open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                        
            # Save the current config
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False, default=str)
            self._last_modified = self.config_path.stat().st_mtime
            self.get_logger().info("✅ Configuration saved")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Config save failed: {e}")
            return False

    def reload_config(self) -> bool:
        try:
            with self._cfg_lock:
                self._load_advanced_config()
                self._apply_env_overrides()
                if not self._validate_config():
                    self.get_logger().warning("⚠️ Reloaded config contains validation warnings.")
                self._performance_stats["config_reloads"] += 1
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Config reload failed: {e}")
            return False

    def get_theme(self) -> SmartTheme:
        return self._theme

    def get_log_entries(self, level: str = None, limit: int = 100) -> List[LogEntry]:
        if not self._log_handler:
            return []
        with self._log_handler.lock:
            entries = self._log_handler.log_entries[-limit:] if limit else self._log_handler.log_entries[:]
        if level:
            level_upper = level.upper()
            entries = [e for e in entries if e.level == level_upper]
        return entries

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "last_modified": datetime.fromtimestamp(self._last_modified) if self._last_modified else None,
            "performance_stats": self._performance_stats.copy(),
            "gpio_available": self._gpio_available,
            "simulation_mode": self._config.get("raspberry_pi", {}).get("simulation_mode", True),
            "system_uptime": (datetime.utcnow() - self._performance_stats["start_time"]).total_seconds(),
            "log_entries_count": len(self._log_handler.log_entries) if self._log_handler else 0,
            "schema_version": self._config.get("schema_version"),
            "config_version": self._config.get("meta", {}).get("config_version"),
            "sensor_count": len(self._sensor_registry),
            "physical_sensor_count": len(self.get_physical_sensors()),
            "simulated_sensor_count": len(self.get_simulated_sensors()),
            "sensor_discoveries": self._performance_stats.get("sensor_discoveries", 0)
        }

    # ---------------------------------------------------------------------------------
    # Convenience / Advanced Accessors
    # ---------------------------------------------------------------------------------

    def get_prediction_settings(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.get_config("prediction.sequence_length"),
            "horizons": self.get_config("prediction.horizons"),
            "train": self.get_config("prediction.train")
        }
        
    def get_sensor_discovery_settings(self) -> Dict[str, Any]:
        """Get sensor discovery settings"""
        return {
            "enabled": self.get_config("sensor_discovery.enabled", True),
            "scan_interval_seconds": self.get_config("sensor_discovery.scan_interval_seconds", 300),
            "i2c_scan_addresses": self.get_config("sensor_discovery.i2c_scan_addresses", []),
            "gpio_scan_pins": self.get_config("sensor_discovery.gpio_scan_pins", []),
            "known_sensor_types": self.get_config("sensor_discovery.known_sensor_types", {})
        }

    def dump_effective_config(self, path: Optional[str] = None) -> str:
        """
        Export current in-memory configuration (post merge + overrides).
        """
        snapshot = {
            "dumped_at": datetime.utcnow().isoformat(),
            "config": self._config,
            "sensors": {name: sensor.to_dict() for name, sensor in self._sensor_registry.items()},
            "system_info": self.get_system_info()
        }
        text = json.dumps(snapshot, indent=2, ensure_ascii=False)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            self.get_logger().info(f"Config snapshot dumped to {p}")
        return text

    def sanitize_for_export(self) -> Dict[str, Any]:
        """
        Remove or mask potentially sensitive or irrelevant internal fields
        for UI or API exposure.
        """
        cfg = self._config.copy()
        # Example of masking (extend if necessary)
        if "security" in cfg and isinstance(cfg["security"], dict):
            sec = cfg["security"].copy()
            # Keep high-level booleans, remove operational details if any appear later
            cfg["security"] = {
                "encryption_enabled": sec.get("encryption_enabled"),
                "access_logging": sec.get("access_logging")
            }
        return cfg
    
    def export_sensors_config(self, path: Optional[str] = None) -> str:
        """Export the sensor configuration to JSON"""
        sensors_data = {}
        for name, sensor in self._sensor_registry.items():
            sensors_data[name] = sensor.to_dict()
            
        text = json.dumps(sensors_data, indent=2, ensure_ascii=False)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            self.get_logger().info(f"Sensors config exported to {p}")
        return text


# -------------------------------------------------------------------------------------------------
# Self-Test (Manual)
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        cfg = SmartConfig()
        log = cfg.get_logger()
        log.info("🧪 Running SmartConfig self-test with dynamic sensor support...")

        # Display prediction settings
        log.info(f"Prediction settings: {cfg.get_prediction_settings()}")
        
        # Test registering a new sensor
        new_sensor = {
            "name": "test_sensor",
            "min": 0.0,
            "max": 100.0, 
            "critical": 80.0,
            "unit": "test_units",
            "weight": 1.0,
            "type": "TEST",
            "is_physical": True,
            "interface": "I2C",
            "i2c_address": "0x42"
        }
        
        cfg.register_sensor("test_sensor", new_sensor)
        
        # Check if sensor was registered
        test_sensor = cfg.get_sensor_config("test_sensor")
        log.info(f"Test sensor registered: {test_sensor is not None}")
        
        # Get sensor registry summary
        physical = len(cfg.get_physical_sensors())
        simulated = len(cfg.get_simulated_sensors())
        log.info(f"Sensor registry: {physical} physical, {simulated} simulated")

        # Test update with rollback scenario
        bad_update = {"system": {"update_interval": -5}}
        ok = cfg.update_config(bad_update, save=False)
        log.info(f"Bad update accepted? {ok}")

        good_update = {"system": {"update_interval": 1.5}}
        ok2 = cfg.update_config(good_update, save=False)
        log.info(f"Good update accepted? {ok2}")

        # Dump snapshot
        cfg.dump_effective_config("config/backup_effective_config.json")

        # Export sensors
        cfg.export_sensors_config("config/sensors/exported_sensors.json")

        # Sanitize preview
        sanitized = cfg.sanitize_for_export()
        log.info(f"Sanitized keys: {list(sanitized.keys())}")

        info = cfg.get_system_info()
        log.info(f"System uptime (s): {info['system_uptime']:.2f}")
        log.info("🎯 SmartConfig self-test complete.")
        
        # Clean up test sensor
        cfg.unregister_sensor("test_sensor")
        
    except Exception as e:
        print(f"❌ Self-test failed: {e}", file=sys.stderr)
        sys.exit(1)
