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
from typing import Any, Dict, List, Optional

import numpy as np

# External systems
from config_and_logging import SmartConfig  # type: ignore

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

        self.sensor_history: Dict[str, List[SensorReading]] = {
            s: [] for s in self.config.get("sensors", {}).keys()
        }
        self._initialize()

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
        for sensor in self.config.get("sensors", {}).keys():
            # Placeholder heuristic: 70% active, 25% simulated, 5% failed
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
        return {
            "total_sensors": total,
            "active_sensors": active,
            "simulated_sensors": simulated,
            "failed_sensors": failed,
            "grid_health": grid_health,
            "average_confidence": float(np.mean(avg_conf)) if avg_conf else 0.0,
            "last_update": datetime.utcnow()
        }


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
        self._lock = threading.RLock()

        # Bootstrapping
        self._initial_training()
        self._start_monitoring()
        self._start_maintenance()
        self._start_retrain_scheduler()
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

            return {
                "system_status": self.system_status,
                "raspberry_pi_active": self.raspberry_pi_active,
                "sensor_grid_status": self.sensor_grid_status,
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
        for t in (self._monitor_thread, self._maintenance_thread, self._retrain_thread):
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
                "sample": status["real_time_data_sample"]
            }, indent=2))
    finally:
        twin.shutdown()
        print("Twin shut down.")
