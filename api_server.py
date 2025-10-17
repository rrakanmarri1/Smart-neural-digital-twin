import uvicorn
from fastapi import FastAPI
from typing import Any, Dict

# We will still import your core systems here, but NOWHERE else.
from core_systems import create_smart_neural_twin, SmartNeuralDigitalTwin

# --- Application Setup ---
app = FastAPI(
    title="Smart Neural Digital Twin API",
    description="Backend API for the Oil Field Disaster Prevention System.",
    version="1.0.0"
)

# --- Global Twin Instance ---
# This is where your main application logic will live.
# The API server creates one instance of it when it starts.
print("Creating SmartNeuralDigitalTwin instance...")
twin_instance: SmartNeuralDigitalTwin = create_smart_neural_twin()
print("Twin instance created successfully.")

# --- API Endpoints ---
# These are the "doors" into your backend. The Streamlit UI will "knock" on these doors.

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Smart Neural Digital Twin API is running."}


@app.get("/api/v1/system/status", response_model=Dict[str, Any])
def get_system_status():
    """
    Provides the full, enhanced status of the digital twin system.
    This replaces the direct call from the UI.
    """
    return twin_instance.get_enhanced_system_status()


@app.get("/api/v1/sensors/realtime_data", response_model=Dict[str, float])
def get_realtime_sensor_data():
    """
    Provides the latest dictionary of sensor readings.
    """
    return twin_instance.real_time_data


@app.get("/api/v1/sensors/configs", response_model=Dict[str, Any])
def get_sensor_configs():
    """
    Provides the configuration for all known sensors.
    """
    if not hasattr(twin_instance, "config"):
        return {}
    return twin_instance.config.get("sensors", {})


@app.post("/api/v1/actions/rescan_sensors")
def rescan_sensors():
    """
    Triggers a manual rescan for new physical sensors.
    """
    if hasattr(twin_instance, "sense_grid") and hasattr(twin_instance.sense_grid, "rescan_sensors"):
        new_sensors_found = twin_instance.sense_grid.rescan_sensors()
        return {"message": "Sensor rescan initiated.", "new_sensors_found": new_sensors_found}
    return {"message": "Sensor rescan not available."}


@app.post("/api/v1/actions/emergency_stop")
def trigger_emergency_stop():
    """
    Triggers the emergency stop sequence in the relay controller.
    """
    if hasattr(twin_instance, "relay_controller") and hasattr(twin_instance.relay_controller, "emergency_shutdown"):
        twin_instance.relay_controller.emergency_shutdown()
        return {"message": "Emergency stop sequence executed."}
    return {"message": "Relay controller not available."}


# --- Main Execution ---
if __name__ == "__main__":
    # This allows you to run the API server directly from the command line:
    # uvicorn api_server:app --reload
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
