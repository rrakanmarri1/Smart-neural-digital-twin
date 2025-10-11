from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Set

import streamlit as st

# Ensure local modules resolvable
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

# Attempt imports with graceful fallback
IMPORT_ERRORS = []
try:
    from core_systems import create_smart_neural_twin, SmartNeuralDigitalTwin
except Exception as e:
    IMPORT_ERRORS.append(f"core_systems: {e}")
try:
    from advanced_systems import create_advanced_dashboard, AdvancedDashboard
except Exception as e:
    IMPORT_ERRORS.append(f"advanced_systems: {e}")
try:
    from config_and_logging import SmartConfig
except Exception as e:
    IMPORT_ERRORS.append(f"config_and_logging: {e}")

APP_VERSION = "2.1.0-ssclass-dynamic"  # Updated version with dynamic sensor support

# --------------------------------------------------------------------------------------------------
# Simulation Fallback (Emergency Mode)
# --------------------------------------------------------------------------------------------------

class _EmergencySimulationTwin:
    """Minimal interface for dashboard compatibility in emergency mode."""
    def __init__(self):
        self._start = datetime.utcnow()
        self._sample = {
            "pressure": 48.5,
            "temperature": 77.2,
            "methane": 230.4,
            "hydrogen_sulfide": 9.1,
            "vibration": 2.0,
            "flow": 192.3
        }

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        return {
            "system_status": "SIMULATION",
            "sensor_grid_status": {
                "active_sensors": 6,
                "simulated_sensors": 6,
                "failed_sensors": 0,
                "grid_health": 0.86,
                "average_confidence": 0.91,
                "last_update": datetime.utcnow().isoformat()
            },
            "physical_sensor_count": 0,
            "total_sensor_count": 6,
            "overall_risk": {"overall_level": "LOW"},
            "performance_metrics": {
                "processed_cycles": 0,
                "avg_cycle_time": 0.0,
                "uptime_seconds": (datetime.utcnow() - self._start).total_seconds(),
                "emergency_events": 0,
                "ai_samples": 0
            },
            "real_time_data_sample": self._sample,
            "raspberry_pi_active": False,
            "ss_rating": "SIM-CLASS",
            "overall_confidence": 0.85,
            "last_update": datetime.utcnow().isoformat()
        }

    @property
    def sensor_grid_status(self) -> Dict[str, Any]:
        return {
            "active_sensors": 6,
            "simulated_sensors": 6,
            "failed_sensors": 0,
            "grid_health": 0.86,
            "last_update": datetime.utcnow().isoformat()
        }

    @property
    def real_time_data(self) -> Dict[str, float]:
        return self._sample


# --------------------------------------------------------------------------------------------------
# Sensor Management
# --------------------------------------------------------------------------------------------------

class SensorRegistry:
    """
    Tracks detected sensors, their status, and configuration
    """
    def __init__(self):
        self.known_sensors: Set[str] = set()
        self.physical_sensors: Set[str] = set()
        self.simulated_sensors: Set[str] = set()
        self.sensor_types: Dict[str, str] = {}
        self.last_scan_time = datetime.utcnow()
        self.scan_interval = timedelta(minutes=5)  # Check for new sensors every 5 minutes
        self.new_sensors_detected = False
        
    def register_sensor(self, sensor_name: str, is_physical: bool = False, sensor_type: str = "UNKNOWN"):
        """Register a new sensor in the registry"""
        self.known_sensors.add(sensor_name)
        if is_physical:
            self.physical_sensors.add(sensor_name)
        else:
            self.simulated_sensors.add(sensor_name)
        self.sensor_types[sensor_name] = sensor_type
        
    def update_from_twin(self, twin: SmartNeuralDigitalTwin) -> bool:
        """Update registry from twin and return whether new sensors were detected"""
        if not twin:
            return False
            
        status = {}
        try:
            status = twin.get_enhanced_system_status()
        except:
            return False
            
        # Get all available sensors from real-time data
        data = getattr(twin, "real_time_data", {}) or {}
        
        # Get physical sensors
        physical_count = status.get("physical_sensor_count", 0)
        physical_sensors = set()
        for sensor, info in status.get("sensors", {}).items():
            if info.get("is_physical", False):
                physical_sensors.add(sensor)
                
        # Process all sensors
        new_detected = False
        for sensor in data.keys():
            is_physical = sensor in physical_sensors
            
            if sensor not in self.known_sensors:
                sensor_type = "PHYSICAL" if is_physical else "SIMULATED"
                self.register_sensor(sensor, is_physical, sensor_type)
                new_detected = True
                
        self.last_scan_time = datetime.utcnow()
        self.new_sensors_detected = new_detected
        return new_detected
        
    def should_scan(self) -> bool:
        """Check if it's time to scan for new sensors"""
        return datetime.utcnow() >= self.last_scan_time + self.scan_interval
        
    def summary(self) -> Dict[str, Any]:
        """Get summary information about detected sensors"""
        return {
            "total_sensors": len(self.known_sensors),
            "physical_sensors": len(self.physical_sensors),
            "simulated_sensors": len(self.simulated_sensors),
            "last_scan": self.last_scan_time.isoformat(),
            "new_detected": self.new_sensors_detected
        }


# --------------------------------------------------------------------------------------------------
# Application Class
# --------------------------------------------------------------------------------------------------

class SmartNeuralApp:
    """
    Main application class for the Smart Neural Digital Twin.
    Handles initialization, emergency fallback, and runtime execution.
    Enhanced with dynamic sensor detection and monitoring.
    """
    def __init__(self):
        self.config: Optional[SmartConfig] = None
        self.logger: Optional[logging.Logger] = None
        self.smart_twin: Optional[SmartNeuralDigitalTwin] = None
        self.dashboard: Optional[AdvancedDashboard] = None
        self._initialized = False
        self._emergency_mode = False
        self.sensor_registry = SensorRegistry()
        self._bootstrap()

    # ----------------------------------------------------------------------------------
    # Bootstrap / Initialization
    # ----------------------------------------------------------------------------------

    def _bootstrap(self):
        """Bootstrap the application: initialize logging, load components, and handle errors."""
        self._configure_page()
        self._init_logging()
        if IMPORT_ERRORS:
            self._enter_emergency_mode(
                RuntimeError("Module import errors: " + "; ".join(IMPORT_ERRORS))
            )
            return
        try:
            self._render_loading_sequence()
            self.smart_twin = self._create_twin()
            # Initialize sensor registry from twin
            if self.smart_twin:
                self.sensor_registry.update_from_twin(self.smart_twin)
            self.dashboard = self._create_dashboard()
            self._initialized = True
            self._welcome_banner()
        except Exception as e:
            self._enter_emergency_mode(e)

    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Smart Neural Digital Twin",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "About": f"""
                ### 🧠 Smart Neural Digital Twin
                Advanced Oil Field Disaster Prevention Platform with Dynamic Sensor Support

                Version: {APP_VERSION}
                """
            }
        )

    def _init_logging(self):
        """Initialize application logging."""
        try:
            self.config = SmartConfig()
            self.logger = self.config.get_logger("SmartNeural.App")
            self.logger.info("Application logging initialized.")
        except Exception as e:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                handlers=[logging.StreamHandler(sys.stdout),
                          logging.FileHandler("logs/app_emergency.log", encoding="utf-8")]
            )
            self.logger = logging.getLogger("SmartNeural.App")
            self.logger.warning(f"Fallback logging active: {e}")

    def _render_loading_sequence(self):
        """Render the loading sequence during application initialization."""
        st.markdown("""
        <style>
          .loading-box {
            background: linear-gradient(135deg,#0f172a,#1e293b);
            padding:2.5rem 1.5rem;
            border-radius:16px;
            text-align:center;
            border:1px solid #1e40af33;
            box-shadow:0 4px 18px rgba(0,0,0,0.35);
            margin-top:1.25rem;
          }
          .loading-title {
            font-size:2.1rem;
            background:linear-gradient(135deg,#3b82f6,#60a5fa);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            margin:0 0 0.75rem 0;
          }
          .loading-sub {
            color:#cbd5e1;
            font-size:.95rem;
            margin-bottom:1.25rem;
            letter-spacing:.5px;
          }
        </style>
        """, unsafe_allow_html=True)

        placeholder = st.empty()
        progress = st.progress(0)
        steps = [
            ("Loading Configuration", 10),
            ("Initializing Sensor Grid", 20),
            ("Scanning for Physical Sensors", 30),
            ("Bootstrapping AI Manager", 45),
            ("Warming Prediction Engines", 65),
            ("Establishing Safety Layer", 80),
            ("Preparing UI Components", 92),
            ("Finalizing Systems", 100)
        ]
        for title, pct in steps:
            placeholder.markdown(f"""
            <div class="loading-box">
                <div class="loading-title">🧠 Smart Neural Digital Twin</div>
                <div class="loading-sub">Initialization Phase</div>
                <div style="color:#93c5fd;font-size:.9rem;">{title}...</div>
            </div>
            """, unsafe_allow_html=True)
            progress.progress(pct / 100)
            time.sleep(0.15 if pct < 90 else 0.05)

        progress.empty()
        placeholder.empty()

    def _create_twin(self) -> SmartNeuralDigitalTwin:
        """Create the Smart Neural Digital Twin instance with dynamic sensor support."""
        if not self.logger:
            raise RuntimeError("Logger unavailable during twin creation.")
        self.logger.info("Creating Smart Neural Digital Twin with dynamic sensor support...")
        twin = create_smart_neural_twin()
        self.logger.info("Twin created.")
        atexit.register(self._safe_shutdown)
        return twin

    def _create_dashboard(self) -> AdvancedDashboard:
        """Create the Advanced Dashboard instance."""
        if not self.smart_twin:
            raise RuntimeError("Cannot create dashboard without twin.")
        if not self.logger:
            raise RuntimeError("Logger unavailable during dashboard creation.")
        self.logger.info("Creating dashboard...")
        dash = create_advanced_dashboard(self.smart_twin)
        self.logger.info("Dashboard ready.")
        return dash

    def _welcome_banner(self):
        """Display a welcome banner when the application successfully initializes."""
        # Get sensor information for the welcome banner
        sensor_info = self.sensor_registry.summary()
        physical_count = sensor_info.get("physical_sensors", 0)
        
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem 1rem;
             background:linear-gradient(135deg,#065f46,#047857);
             border-radius:14px;margin:1.25rem 0 0.75rem 0;
             border:1px solid #10b98133;">
          <h2 style="color:#fff;margin:.2rem 0 0.6rem 0;">🚀 System Ready</h2>
          <p style="color:#d1fae5;margin:0;font-size:.85rem;">
            Real-time anomaly detection & multi-horizon predictive analytics are active.
          </p>
          <p style="color:#a7f3d0;margin:.35rem 0 0;font-size:.7rem;">
            Version {APP_VERSION} | {physical_count} physical sensors detected
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------
    # Emergency Mode
    # ----------------------------------------------------------------------------------

    def _enter_emergency_mode(self, error: Exception):
        """Switch to emergency simulation mode when initialization fails."""
        self._emergency_mode = True
        err_text = f"{type(error).__name__}: {error}"
        if self.logger:
            self.logger.error(f"Entering emergency mode: {err_text}")
            self.logger.error(traceback.format_exc())
        st.error(f"❌ Initialization Failed: {err_text}")
        st.warning("Emergency fallback mode enabled. Core AI features disabled.")
        self.smart_twin = _EmergencySimulationTwin()
        try:
            self.dashboard = create_advanced_dashboard(self.smart_twin)
        except Exception:
            self.dashboard = None
        self._emergency_banner(err_text)

    def _emergency_banner(self, msg: str):
        """Display an emergency banner in the UI."""
        st.markdown(f"""
        <div style="text-align:center;padding:1.75rem;
             background:linear-gradient(135deg,#7f1d1d,#b91c1c);
             border-radius:14px;margin:1rem 0;
             border:1px solid #fecaca33;">
          <h2 style="color:#fff;margin:0 0 .75rem 0;">🚨 Emergency Mode</h2>
          <p style="color:#fecaca;font-size:.85rem;margin:0;">{msg}</p>
          <p style="color:#fca5a5;font-size:.7rem;margin:.5rem 0 0;">Limited simulation dashboard active.</p>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------
    # Runtime
    # ----------------------------------------------------------------------------------

    def run(self):
        """Run the application and render the dashboard."""
        if not self.dashboard:
            self._render_no_dashboard_state()
            return

        # Periodic sensor scan if appropriate
        self._check_for_new_sensors()
        
        # Dashboard main rendering
        self.dashboard.run_dashboard()

        # Session & Debug
        self._session_management()
        self._sensor_management_ui()
        self._debug_sidebar()

    def _render_no_dashboard_state(self):
        """Render a fallback state when the dashboard cannot be initialized."""
        st.error("Dashboard unavailable. Try restarting the application.")
        if st.button("🔄 Restart"):
            st.experimental_rerun()
        if self._emergency_mode:
            st.info("Emergency simulation active, but dashboard could not be initialized.")

    def _check_for_new_sensors(self):
        """Check for new sensors if scan interval has elapsed"""
        if not self._emergency_mode and self.sensor_registry.should_scan() and self.smart_twin:
            new_sensors = self.sensor_registry.update_from_twin(self.smart_twin)
            if new_sensors and self.logger:
                self.logger.info(f"New sensors detected during periodic scan")
                # Force refresh to show new sensors
                st.experimental_rerun()

    # ----------------------------------------------------------------------------------
    # Sensor Management UI
    # ----------------------------------------------------------------------------------

    def _sensor_management_ui(self):
        """Render the sensor management UI section"""
        if self._emergency_mode:
            return
            
        sensor_summary = self.sensor_registry.summary()
        
        with st.sidebar.expander("🔌 Sensor Management", expanded=False):
            st.markdown("### Sensor Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Physical Sensors", sensor_summary.get("physical_sensors", 0))
            with col2:
                st.metric("Simulated Sensors", sensor_summary.get("simulated_sensors", 0))
                
            st.markdown(f"Last Scan: {datetime.fromisoformat(sensor_summary.get('last_scan', datetime.utcnow().isoformat())).strftime('%H:%M:%S')}")
            
            if st.button("Scan for New Sensors"):
                if self.smart_twin:
                    with st.spinner("Scanning..."):
                        new_detected = self.sensor_registry.update_from_twin(self.smart_twin)
                    if new_detected:
                        st.success("New sensors detected!")
                    else:
                        st.info("No new sensors found.")
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("Twin not available for sensor scan.")
                    
            # Show sensors that have been detected
            with st.expander("Detected Sensors", expanded=False):
                if self.sensor_registry.known_sensors:
                    physical = sorted(list(self.sensor_registry.physical_sensors))
                    simulated = sorted(list(self.sensor_registry.simulated_sensors))
                    
                    if physical:
                        st.markdown("**Physical Sensors:**")
                        for sensor in physical:
                            st.markdown(f"- {sensor}")
                            
                    if simulated:
                        st.markdown("**Simulated Sensors:**")
                        for sensor in simulated:
                            st.markdown(f"- {sensor}")
                else:
                    st.info("No sensors detected yet.")

    # ----------------------------------------------------------------------------------
    # Session Management & Debug
    # ----------------------------------------------------------------------------------

    def _session_management(self):
        """Manage session state and auto-refresh."""
        if "session_start" not in st.session_state:
            st.session_state.session_start = datetime.utcnow()
            st.session_state.activity_count = 0
            st.session_state.auto_refresh_enabled = True
        st.session_state.activity_count += 1
        elapsed = (datetime.utcnow() - st.session_state.session_start).total_seconds()

        with st.sidebar.expander("⚙️ Session Control"):
            st.checkbox("Enable Auto Refresh (UI layer only)",
                        value=st.session_state.auto_refresh_enabled,
                        key="auto_refresh_enabled")
            st.write(f"Session Uptime: {int(elapsed)}s")
            st.write(f"Activity Count: {st.session_state.activity_count}")

        if elapsed > 7200:  # 2 hours
            st.info("🔄 Session auto-reset for stability.")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

    def _debug_sidebar(self):
        """Render advanced debug options in the sidebar."""
        with st.sidebar.expander("🔧 Advanced Debug", expanded=False):
            colA, colB = st.columns(2)
            with colA:
                if st.button("Reload Config"):
                    if self.config and self.config.reload_config():
                        st.success("Config reloaded.")
                    else:
                        st.error("Reload failed.")
            with colB:
                if st.button("Dump Config Snapshot"):
                    if self.config:
                        snapshot = self.config.dump_effective_config()
                        st.download_button("Download Config JSON",
                                           data=snapshot,
                                           file_name=f"config_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M:%S')}.json",
                                           mime="application/json")

            # Add system status with dynamic sensor information
            if st.checkbox("Show System Status JSON", value=False):
                try:
                    status = self.smart_twin.get_enhanced_system_status()  # type: ignore
                    st.json(status)
                except Exception as e:
                    st.error(f"Status error: {e}")
                    
            # Add sensor registry information
            if st.checkbox("Show Sensor Registry", value=False):
                st.json(self.sensor_registry.summary())

            if st.checkbox("Show Log Tail", value=False):
                if self.config:
                    entries = self.config.get_log_entries(limit=25)
                    for e in entries:
                        st.code(f"[{e.timestamp.strftime('%H:%M:%S')}] {e.level:<8} {e.logger}: {e.message}")

            if st.checkbox("Show Environment", value=False):
                env_preview = {k: v for k, v in os.environ.items() if k.startswith("SMART_TWIN__")}
                st.json(env_preview)
                
            # Add AI model information for dynamic sensors
            if not self._emergency_mode and st.checkbox("Show AI System Status", value=False):
                try:
                    # Get anomaly system status if available
                    if hasattr(self.smart_twin, "ai_manager") and hasattr(self.smart_twin.ai_manager, "anomaly_system"):
                        anomaly_status = self.smart_twin.ai_manager.anomaly_system.status()
                        st.subheader("Anomaly System")
                        st.json(anomaly_status)
                        
                    # Get prediction engine status if available
                    if hasattr(self.smart_twin, "ai_manager") and hasattr(self.smart_twin.ai_manager, "prediction_engine"):
                        prediction_status = self.smart_twin.ai_manager.prediction_engine.status()
                        st.subheader("Prediction Engine")
                        st.json(prediction_status)
                except Exception as e:
                    st.error(f"AI status error: {e}")

    # ----------------------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------------------

    def _safe_shutdown(self):
        """Perform safe application shutdown."""
        if getattr(self, "_shutdown_called", False):
            return
        setattr(self, "_shutdown_called", True)
        if self.logger:
            self.logger.info("Application shutdown hook triggered.")
        try:
            if self.smart_twin and hasattr(self.smart_twin, "shutdown"):
                self.smart_twin.shutdown()  # type: ignore
        except Exception:
            if self.logger:
                self.logger.error("Error during twin shutdown.", exc_info=True)


# --------------------------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------------------------

def main():
    """Main entry point for the application."""
    try:
        app = SmartNeuralApp()
        app.run()
    except Exception as e:
        logging.critical(f"Unhandled application crash: {e}")
        logging.critical(traceback.format_exc())
        _render_global_emergency(e)


def _render_global_emergency(exc: Exception):
    """Render a global emergency fallback UI."""
    st.markdown("""
    <div style="text-align:center;padding:2.5rem;
         background:#7f1d1d;border-radius:14px;margin:1.5rem 0;
         border:1px solid #fecaca55;">
      <h2 style="color:#fff;margin:0 0 .85rem 0;">🚨 Global Failure</h2>
      <p style="color:#fecaca;margin:0;font-size:.9rem;">
        The application encountered a critical, unrecoverable error.
      </p>
      <p style="color:#fca5a5;margin:.6rem 0 0;font-size:.7rem;">
        Please capture the stack trace below and notify support.
      </p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Stack Trace"):
        st.code(traceback.format_exc())
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Restart"):
            st.experimental_rerun()
    with c2:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()
        }
        st.download_button("Download Error Report",
                           data=json.dumps(report, indent=2),
                           file_name="critical_error_report.json",
                           mime="application/json")


if __name__ == "__main__":
    main()
