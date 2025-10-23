from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
import requests # Import the requests library

# Ensure local modules resolvable
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

# Attempt imports with graceful fallback
IMPORT_ERRORS = []
try:
    # We NO LONGER import core_systems here.
    from advanced_systems import create_advanced_dashboard, AdvancedDashboard
except Exception as e:
    IMPORT_ERRORS.append(f"advanced_systems: {e}")
try:
    from config_and_logging import SmartConfig
except Exception as e:
    IMPORT_ERRORS.append(f"config_and_logging: {e}")

APP_VERSION = "3.0.0-decoupled"  # New version reflecting architecture change

# --- API Configuration ---
API_BASE_URL = "http://127.0.0.1:8000" # The address of our new FastAPI server

# --------------------------------------------------------------------------------------------------
# Simulation Fallback (Emergency Mode) - This is now a simple API client
# --------------------------------------------------------------------------------------------------

class _EmergencySimulationClient:
    """Minimal interface for dashboard compatibility when the API is down."""
    def __init__(self):
        self._start = datetime.utcnow()
        self._sample = {
            "pressure": 48.5, "temperature": 77.2, "methane": 230.4,
            "hydrogen_sulfide": 9.1, "vibration": 2.0, "flow": 192.3
        }

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        return {
            "system_status": "API_DOWN",
            "sensor_grid_status": {"active_sensors": 0, "simulated_sensors": 6, "failed_sensors": 0, "grid_health": 0.0},
            "physical_sensor_count": 0, "total_sensor_count": 6, "ai_recommendations": ["Backend API is unreachable. Running in UI simulation mode."],
            "performance_metrics": {"uptime_seconds": (datetime.utcnow() - self._start).total_seconds()},
            "real_time_data_sample": self._sample, "raspberry_pi_active": False,
            "overall_confidence": 0.0, "last_update": datetime.utcnow().isoformat(),
            "latest_anomaly": {}, "latest_forecast": {}, "overall_risk": {"overall_level": "UNKNOWN"}
        }

    @property
    def real_time_data(self) -> Dict[str, float]:
        return self._sample

# --------------------------------------------------------------------------------------------------
# Application Class
# --------------------------------------------------------------------------------------------------

class SmartNeuralApp:
    """
    Main application class for the Smart Neural Digital Twin UI.
    This class is now a CLIENT that communicates with a backend API.
    """
    def __init__(self):
        self.config: Optional[SmartConfig] = None
        self.logger: Optional[logging.Logger] = None
        # The 'smart_twin' is now a client, not the real object.
        self.api_client: Any = None
        self.dashboard: Optional[AdvancedDashboard] = None
        self._initialized = False
        self._emergency_mode = False
        self._bootstrap()

    # ----------------------------------------------------------------------------------
    # API Communication Layer
    # ----------------------------------------------------------------------------------

    def _api_get(self, endpoint: str) -> Dict[str, Any]:
        """Helper function to make GET requests to the backend API."""
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
            response.raise_for_status()  # Raises an exception for 4xx/5xx errors
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.error(f"API GET request to {endpoint} failed: {e}")
            self._enter_emergency_mode(e)
            return {}
            
    def _api_post(self, endpoint: str) -> Dict[str, Any]:
        """Helper function to make POST requests to the backend API."""
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.error(f"API POST request to {endpoint} failed: {e}")
            # Don't enter emergency mode for POST failures, just show an error.
            st.error(f"Action failed: Could not communicate with backend.")
            return {}

    # --- This class now acts as the 'twin' for the dashboard ---
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        return self._api_get("/api/v1/system/status")

    @property
    def real_time_data(self) -> Dict[str, float]:
        return self._api_get("/api/v1/sensors/realtime_data")

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
            # The "twin" is now this app class itself, which acts as a client.
            self.api_client = self
            self.dashboard = self._create_dashboard()
            self._initialized = True
            # No welcome banner here, let the dashboard render first
        except Exception as e:
            self._enter_emergency_mode(e)

    def _configure_page(self):
        st.set_page_config(
            page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide",
            initial_sidebar_state="expanded",
            menu_items={"About": f"Version: {APP_VERSION}"}
        )

    def _init_logging(self):
        # UI logging can be simpler now
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | UI | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("SmartNeural.AppUI")
        self.logger.info("UI logging initialized.")


    def _render_loading_sequence(self):
        placeholder = st.empty()
        with placeholder.container():
            st.title("🧠 Smart Neural Digital Twin")
            st.info("Connecting to backend server...")
            # Check if backend is alive
            try:
                requests.get(API_BASE_URL, timeout=2)
                st.success("Connected to backend server. Initializing UI...")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API server.")
                st.warning(f"Please ensure the API server is running at {API_BASE_URL}")
                self._enter_emergency_mode(Exception("Backend connection failed."))
                return
            time.sleep(1)
        placeholder.empty()

    def _create_dashboard(self) -> AdvancedDashboard:
        """Create the Advanced Dashboard instance, passing the client."""
        self.logger.info("Creating dashboard...")
        # The dashboard gets a reference to this app instance, which will proxy API calls
        dash = create_advanced_dashboard(self)
        self.logger.info("Dashboard ready.")
        return dash

    # ----------------------------------------------------------------------------------
    # Emergency Mode
    # ----------------------------------------------------------------------------------

    def _enter_emergency_mode(self, error: Exception):
        """Switch to emergency simulation mode when initialization fails."""
        if self._emergency_mode: # Avoid recursive emergency mode
            return
        self._emergency_mode = True
        err_text = f"{type(error).__name__}: {error}"
        if self.logger:
            self.logger.error(f"Entering emergency mode: {err_text}")
            self.logger.error(traceback.format_exc())
        
        # Use a client that returns static data
        self.api_client = _EmergencySimulationClient()
        try:
            # Try to create dashboard with the emergency client
            self.dashboard = create_advanced_dashboard(self.api_client)
            self._emergency_banner(err_text)
        except Exception as dash_error:
            self.dashboard = None
            st.error(f"CRITICAL: Dashboard failed to initialize even in emergency mode: {dash_error}")


    def _emergency_banner(self, msg: str):
        st.error(f"**EMERGENCY MODE:** {msg}", icon="🚨")
        st.warning("The application could not connect to the backend. Displaying static UI with simulated data.")

    # ----------------------------------------------------------------------------------
    # Runtime
    # ----------------------------------------------------------------------------------

    def run(self):
        """Run the application and render the dashboard."""
        if not self.dashboard:
            st.error("Dashboard could not be initialized. Please check the logs.")
            return
        
        self.dashboard.run_dashboard()
        self._debug_sidebar()

    def _debug_sidebar(self):
        with st.sidebar.expander("🔧 Advanced Debug", expanded=False):
            if st.button("Check API Connection"):
                try:
                    response = requests.get(API_BASE_URL, timeout=2).json()
                    st.success("API is reachable.")
                    st.json(response)
                except Exception as e:
                    st.error(f"API connection failed: {e}")

            if st.checkbox("Show System Status JSON", value=False):
                status = self.get_enhanced_system_status()
                st.json(status)


# --------------------------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------------------------

def main():
    """Main entry point for the application."""
    if "final_app" not in st.session_state:
        st.session_state.app = SmartNeuralApp()
    
    st.session_state.app.run()


if __name__ == "__main__":
    main()
