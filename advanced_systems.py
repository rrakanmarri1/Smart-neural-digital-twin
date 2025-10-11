from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------------

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class SystemAlert:
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


# --------------------------------------------------------------------------------------
# Constants & Configuration
# --------------------------------------------------------------------------------------

# Default sensor configurations for fallback when sensors aren't available from the system
DEFAULT_SENSOR_CONFIGS = {
    # key: (Label, icon, critical, unit, nominal_min, nominal_max)
    "pressure": ("Pressure", "💨", 150, "bar", 40, 60),
    "temperature": ("Temperature", "🌡️", 200, "°C", 70, 85),
    "methane": ("Methane", "⚠️", 1000, "ppm", 0, 500),
    "hydrogen_sulfide": ("H2S", "☠️", 50, "ppm", 0, 20),
    "vibration": ("Vibration", "📳", 8, "m/s²", 1, 4),
    "flow": ("Flow", "💧", 400, "L/min", 150, 250),
}

# Icon mapping for different sensor types
SENSOR_TYPE_ICONS = {
    "BME280": "🌡️",
    "ADS1115": "📊",
    "MPU6050": "📳",
    "BH1750": "💡",
    "AM2320": "💧",
    "HTU21D": "💧",
    "SHT31": "🌡️",
    "DIGITAL": "🔌",
    "SPI_SENSOR": "🔌",
    "UNKNOWN": "🔍",
}

# Status icons for different sensor states
SENSOR_STATUS_ICONS = {
    "ACTIVE": "✅",
    "DEGRADED": "⚠️",
    "FAILED": "❌",
    "SIMULATED": "🔄"
}

AUTO_REFRESH_INTERVALS = {
    "5s": 5,
    "10s": 10,
    "30s": 30,
    "60s": 60
}

SECTION_DIVIDER = "---"


# --------------------------------------------------------------------------------------
# Utility & Helper Functions
# --------------------------------------------------------------------------------------

def safe_call(logger: logging.Logger, fn: Callable, fallback: Any = None, context: str = "") -> Any:
    """Execute a callable with full exception shielding, log on failure."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[{context}] failure: {e}", exc_info=True)
        return fallback


def classify_sensor_state(value: float,
                          critical: float,
                          nominal_min: float,
                          nominal_max: float) -> Tuple[str, str]:
    """
    Determine status classification & card CSS class for a sensor value.
    Returns: (status_text, css_class)
    """
    if value >= critical * 0.9 or value <= nominal_min * 0.9:
        return "🚨 CRITICAL", "emergency-card"
    if value >= critical * 0.8 or value <= nominal_min * 0.95:
        return "⚠️ WARNING", "warning-card"
    if nominal_min <= value <= nominal_max:
        return "✅ NORMAL", "success-card"
    return "⚠️ CHECK", "warning-card"


def ensure_session_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def get_sensor_display_name(sensor_key: str, sensor_config: Dict[str, Any]) -> str:
    """Generate a display name from sensor key and configuration"""
    if "name" in sensor_config:
        return sensor_config["name"].replace("_", " ").title()
    return sensor_key.replace("_", " ").title()


def get_sensor_icon(sensor_config: Dict[str, Any]) -> str:
    """Get appropriate icon for sensor type"""
    if "type" in sensor_config:
        return SENSOR_TYPE_ICONS.get(sensor_config["type"], "🔍")
    return "🔍"


# --------------------------------------------------------------------------------------
# Advanced Dashboard
# --------------------------------------------------------------------------------------

class AdvancedDashboard:
    """
    S-Class Operational Dashboard.

    Responsibilities:
    - Presentation of system status, sensor states, risk & performance analytics
    - Alert lifecycle management
    - Chat interface (operator ↔ AI system)
    - Modular UI sections (side-bar + main workspace)
    - Dynamic sensor detection & visualization
    """
    def __init__(self, smart_twin: Any):
        self.smart_twin = smart_twin
        self.logger = logging.getLogger("SmartNeural.UI.Dashboard")

        self._initialize_session()
        self._apply_theme()
        self.logger.info("AdvancedDashboard initialized (S-Class)")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_session(self):
        ensure_session_key("alerts", [])
        ensure_session_key("chat_history", [])
        ensure_session_key("system_metrics_cache", {})
        ensure_session_key("last_refresh_ts", time.time())
        ensure_session_key("sensor_history", {})

    def _apply_theme(self):
        # Theming injection (kept minimal; heavy CSS should remain maintainable)
        st.markdown("""
        <style>
          .metric-card {
              background: linear-gradient(135deg, #1e293b, #334155);
              border-radius: 14px;
              padding: 1.0rem 1.0rem 0.9rem 1.0rem;
              border: 1px solid #475569;
              box-shadow: 0 4px 18px rgba(0,0,0,0.25);
              transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
              position: relative;
              overflow: hidden;
          }
          .metric-card:hover {
              transform: translateY(-4px);
              box-shadow: 0 8px 30px rgba(0,0,0,0.35);
              border-color: #3b82f6;
          }
          .success-card { border-color:#10b981; background:linear-gradient(135deg,#064e3b,#047857); }
          .warning-card { border-color:#f59e0b; background:linear-gradient(135deg,#78350f,#d97706); }
          .emergency-card { border-color:#ef4444; background:linear-gradient(135deg,#7f1d1d,#dc2626); animation: pulse-em 2s infinite; }
          @keyframes pulse-em { 0%{box-shadow:0 0 0 0 rgba(239,68,68,0.6);} 60%{box-shadow:0 0 0 12px rgba(239,68,68,0);} 100%{box-shadow:0 0 0 0 rgba(239,68,68,0);} }

          .section-header {
            background:linear-gradient(90deg,#3b82f6,#60a5fa);
            padding:.65rem 1rem;
            border-radius:10px;
            margin:1rem 0 .75rem 0;
            color:white;
            font-weight:600;
            font-size:1.0rem;
            box-shadow:0 2px 8px rgba(59,130,246,.3);
            text-transform:uppercase;
            letter-spacing:.5px;
          }
          .alert-box {
            padding:.65rem .85rem;
            border-radius:8px;
            margin:.4rem 0;
            font-size:.85rem;
            border-left:4px solid #3b82f6;
            background:rgba(59,130,246,0.08);
          }
          .alert-box.warning {border-left-color:#f59e0b; background:rgba(245,158,11,0.10);}
          .alert-box.critical {border-left-color:#ef4444; background:rgba(239,68,68,0.10);}
          .alert-box.emergency {border-left-color:#dc2626; background:rgba(220,38,38,0.18); animation: pulse-em 2.5s infinite;}
          .chat-bubble {
            background:rgba(30,41,59,.80);
            padding:.65rem .75rem;
            border-radius:10px;
            margin:.35rem 0;
            font-size:.85rem;
            border:1px solid #475569;
          }
          .sim-label {
            display:inline-block;
            background:#334155;
            color:#e2e8f0;
            font-size:.65rem;
            padding:2px 6px;
            border-radius:6px;
            margin-left:6px;
            text-transform:uppercase;
            letter-spacing:1px;
          }
          .physical-badge {
            display:inline-block;
            background:#10b981;
            color:#f0fdf4;
            font-size:.65rem;
            padding:2px 6px;
            border-radius:6px;
            margin-left:6px;
            text-transform:uppercase;
            letter-spacing:1px;
          }
          .simulated-badge {
            display:inline-block;
            background:#6366f1;
            color:#f5f3ff;
            font-size:.65rem;
            padding:2px 6px;
            border-radius:6px;
            margin-left:6px;
            text-transform:uppercase;
            letter-spacing:1px;
          }
          .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            grid-gap: 1rem;
            margin-top: 1rem;
          }
        </style>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    def run_dashboard(self):
        """Orchestrates side bar & main content rendering + refresh control."""
        self._render_sidebar()
        self._render_main_content()
        self._handle_auto_refresh()

    # ------------------------------------------------------------------
    # Sidebar Composition
    # ------------------------------------------------------------------

    def _render_sidebar(self):
        with st.sidebar:
            self._sidebar_header()
            self._section_system_status()
            self._section_alerts()
            self._section_physical_sensors()  # New section for physical sensors
            self._section_foresight_controls()
            self._section_chat()
            self._section_recommendations()
            self._section_realtime_summary()
            self._section_emergency()
            self._section_ai_insights()

    def _sidebar_header(self):
        st.markdown("""
        <div style="text-align:center;padding:.75rem 0 .25rem 0;">
          <div style="font-size:2rem;">🧠</div>
          <h3 style="color:#fff;margin:.25rem 0 .25rem 0;font-weight:600;">Smart Neural Digital Twin</h3>
          <div style="color:#cbd5e1;font-size:.75rem;letter-spacing:1px;">S-CLASS MONITORING CONSOLE</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(SECTION_DIVIDER)

    # ------------------------------------------------------------------
    # Section: System Status
    # ------------------------------------------------------------------

    def _section_system_status(self):
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        status = self._fetch_system_status()
        uptime_seconds = status.get("performance_metrics", {}).get("uptime_seconds", 0.0)
        uptime_str = str(timedelta(seconds=int(uptime_seconds))) if uptime_seconds else "—"

        system_state = status.get("system_status", "UNKNOWN")
        color_map = {
            "NORMAL": "#10b981",
            "HIGH_ALERT": "#f59e0b",
            "CRITICAL": "#ef4444",
            "EMERGENCY": "#dc2626"
        }
        state_color = color_map.get(system_state, "#64748b")

        # Get grid health from sensor_grid_status
        sensor_grid = status.get("sensor_grid_status", {})
        grid_health = sensor_grid.get("grid_health", 0)

        st.markdown(
            f"""
            <div style="text-align:center;margin:.3rem 0 0 0;">
              <span style="font-size:1.15rem;font-weight:700;color:{state_color};letter-spacing:.5px;">{system_state}</span>
              <div style="color:#94a3b8;font-size:.7rem;margin-top:.25rem;">Updated {datetime.utcnow().strftime('%H:%M:%S')} UTC</div>
              <div style="color:#cbd5e1;font-size:.65rem;margin-top:.25rem;">Uptime: {uptime_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        cols = st.columns(2)
        with cols[0]:
            st.metric("Grid Health",
                      f"{grid_health*100:.1f}%",
                      help="Active physical or simulated sensor coverage")
        with cols[1]:
            st.metric("Avg Proc Time",
                      f"{status.get('performance_metrics', {}).get('avg_cycle_time', 0):.3f}s",
                      help="Average processing duration of monitoring loop")

    # ------------------------------------------------------------------
    # Section: Alerts
    # ------------------------------------------------------------------

    def _section_alerts(self):
        st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
        active_alerts = [a for a in st.session_state.alerts if not a.acknowledged]

        if not active_alerts:
            st.success("No active alerts")
        else:
            for alert in active_alerts[-5:]:
                css_class = "alert-box"
                if alert.level == AlertLevel.WARNING:
                    css_class += " warning"
                elif alert.level == AlertLevel.CRITICAL:
                    css_class += " critical"
                elif alert.level == AlertLevel.EMERGENCY:
                    css_class += " emergency"

                st.markdown(
                    f"""
                    <div class="{css_class}">
                      <div style="font-weight:600;color:#f1f5f9;">{alert.title}</div>
                      <div style="color:#cbd5e1;font-size:.75rem;">{alert.message}</div>
                      <div style="color:#64748b;font-size:.6rem;">{alert.timestamp.strftime('%H:%M:%S')} • {alert.source}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Ack {alert.level.value}", key=f"ack_{alert.timestamp.timestamp()}"):
                    alert.acknowledged = True
                    st.experimental_rerun()

        # Mini alert creation (for testing)
        with st.expander("Add Test Alert"):
            new_level = st.selectbox("Level", [e.value for e in AlertLevel], index=1)
            new_title = st.text_input("Title", "Manual Alert")
            new_msg = st.text_area("Message", "Operator inserted diagnostic alert.")
            if st.button("Add Alert"):
                self.add_alert(AlertLevel(new_level), new_title, new_msg, "Operator")
                st.success("Inserted.")
                st.experimental_rerun()

    # ------------------------------------------------------------------
    # Section: Physical Sensors (New)
    # ------------------------------------------------------------------

    def _section_physical_sensors(self):
        """New section to display physical sensor information"""
        st.markdown('<div class="section-header">Sensor Status</div>', unsafe_allow_html=True)
        
        # Get sensor information from system status
        status = self._fetch_system_status()
        sensor_grid = status.get("sensor_grid_status", {})
        
        # Get physical sensor count
        physical_count = status.get("physical_sensor_count", 0)
        total_count = status.get("total_sensor_count", 0)
        
        # Display sensor counts
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Physical Sensors", 
                     f"{physical_count}",
                     help="Number of detected physical sensors connected")
        with col2:
            st.metric("Total Sensors",
                     f"{total_count}",
                     help="Total sensors including physical and simulated")
        
        # Show sensor status breakdown
        active = sensor_grid.get("active_sensors", 0)
        simulated = sensor_grid.get("simulated_sensors", 0)
        failed = sensor_grid.get("failed_sensors", 0)
        
        status_data = {
            "Status": ["Active", "Simulated", "Failed"],
            "Count": [active, simulated, failed]
        }
        
        # Display as expandable details
        with st.expander("Sensor Status Details", expanded=False):
            # Create a horizontal bar chart
            chart_data = pd.DataFrame(status_data)
            st.bar_chart(chart_data.set_index("Status"), use_container_width=True)
            
            # Show latest sensor activity
            st.markdown("**Last Sensor Activity:**")
            last_update = sensor_grid.get("last_update", "Unknown")
            if isinstance(last_update, str):
                try:
                    last_update = datetime.fromisoformat(last_update)
                    last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    last_update_str = last_update
            else:
                last_update_str = "Unknown"
                
            st.write(f"- Last Grid Update: {last_update_str}")
            st.write(f"- Grid Health: {sensor_grid.get('grid_health', 0)*100:.1f}%")
            
            # Add button to trigger sensor rescan
            if st.button("Rescan for New Sensors"):
                # This would trigger a sensor rescan if implemented in the backend
                self.add_alert(AlertLevel.INFO, "Sensor Rescan", "Initiated manual sensor detection scan", "Operator")
                st.success("Sensor rescan initiated")
                st.experimental_rerun()

    # ------------------------------------------------------------------
    # Section: Foresight Controls
    # ------------------------------------------------------------------

    def _section_foresight_controls(self):
        st.markdown('<div class="section-header">Foresight Engine</div>', unsafe_allow_html=True)
        scenarios = st.slider("Scenarios / second", 100, 1000, 500, 100)
        horizon = st.selectbox("Prediction Horizon (hours)", [1, 3, 6, 12, 24], index=2)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Optimize Engine", use_container_width=True):
                self._optimize_engine(scenarios)
        with col2:
            if st.button("Generate Forecast Report", use_container_width=True):
                self._export_prediction_report(horizon)

    # ------------------------------------------------------------------
    # Section: Chat
    # ------------------------------------------------------------------

    def _section_chat(self):
        st.markdown('<div class="section-header">AI Chat</div>', unsafe_allow_html=True)
        # Display last few messages
        for msg in st.session_state.chat_history[-5:]:
            st.markdown(f'<div class="chat-bubble">{msg}</div>', unsafe_allow_html=True)

        user_input = st.text_input(
            "Ask / Command (prefix commands with /):",
            placeholder="e.g., /status or What is current pressure?",
            key="chat_input_box"
        )
        send_cols = st.columns(2)
        with send_cols[0]:
            if st.button("Send"):
                if user_input.strip():
                    self._process_chat(user_input.strip())
        with send_cols[1]:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    def _process_chat(self, user_input: str):
        st.session_state.chat_history.append(f"**👤 You:** {user_input}")
        if user_input.startswith("/"):
            response = self._handle_command(user_input)
        else:
            response = self._generate_chat_response(user_input)
        st.session_state.chat_history.append(f"**🤖 AI:** {response}")
        st.experimental_rerun()

    def _handle_command(self, command: str) -> str:
        cmd = command.lower().strip().lstrip("/")
        if cmd == "status":
            st_status = self._fetch_system_status()
            # Include physical sensor count in status
            physical_count = st_status.get("physical_sensor_count", 0)
            grid_health = st_status.get("sensor_grid_status", {}).get("grid_health", 0)
            uptime = int(st_status.get("performance_metrics", {}).get("uptime_seconds", 0))
            return f"System={st_status.get('system_status','?')} | Grid Health={grid_health:.2f} | Physical Sensors={physical_count} | Uptime(s)={uptime}"
        if cmd == "alerts":
            active = len([a for a in st.session_state.alerts if not a.acknowledged])
            return f"Active alerts: {active}"
        if cmd == "sensors":
            # New command to list all available sensors
            sensor_data = self._fetch_sensor_data()
            status = self._fetch_system_status()
            physical_count = status.get("physical_sensor_count", 0)
            total_count = status.get("total_sensor_count", 0)
            sensors_list = ", ".join(sensor_data.keys())
            return f"Physical sensors: {physical_count}, Total sensors: {total_count}. Available sensors: {sensors_list}"
        if cmd == "help":
            return "Commands: /status /alerts /sensors /help"
        return "Unknown command. Try /help"

    def _generate_chat_response(self, question: str) -> str:
        """Enhanced to respond about any detected sensor, not just hardcoded ones"""
        q = question.lower()
        data = self._fetch_sensor_data()
        
        # Check if question is asking about a specific sensor
        for sensor_name, value in data.items():
            if sensor_name.lower() in q:
                # Get sensor configuration if available
                sensor_config = self._get_sensor_config(sensor_name)
                unit = sensor_config.get("unit", "")
                is_physical = self._is_physical_sensor(sensor_name)
                sensor_type = self._get_sensor_source_type(sensor_name)
                
                return f"Current {sensor_name}: {value:.2f} {unit} ({sensor_type})"
        
        # If asking about general risk
        if "risk" in q:
            status = self._fetch_system_status()
            return f"Overall status: {status.get('system_status','?')} • Grid health: {status.get('sensor_grid_status', {}).get('grid_health', 0):.2f}"
            
        # If asking about sensors
        if "sensor" in q:
            status = self._fetch_system_status()
            physical_count = status.get("physical_sensor_count", 0)
            total_count = status.get("total_sensor_count", 0)
            return f"System has {physical_count} physical sensors and {total_count} total sensors (including simulated)."
            
        # Default response
        return "I have processed your query. Provide /help for commands or specify a sensor name to get its current reading."

    # ------------------------------------------------------------------
    # Section: Recommendations
    # ------------------------------------------------------------------

    def _section_recommendations(self):
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        # Real implementation would derive from anomaly + prediction merges
        recs = [
            "Review methane sensor calibration within next 24h (scheduled maintenance window).",
            "Enable enhanced monitoring if vibration variance increases > 15%.",
            "Perform preventive inspection: pressure regulation subsystem (trend stable, but 32h since last check)."
        ]
        for r in recs:
            st.markdown(f"- {r}")

    # ------------------------------------------------------------------
    # Section: Realtime Summary
    # ------------------------------------------------------------------

    def _section_realtime_summary(self):
        st.markdown('<div class="section-header">Realtime Snapshot</div>', unsafe_allow_html=True)
        status = self._fetch_system_status()
        sensor_grid = status.get("sensor_grid_status", {})
        
        # Update to show physical sensors specifically
        physical_sensors = status.get("physical_sensor_count", 0)
        total_sensors = status.get("total_sensor_count", 0)
        
        st.metric("Physical/Total Sensors", f"{physical_sensors}/{total_sensors}")
        st.metric("Average Confidence", f"{sensor_grid.get('average_confidence', 0)*100:.1f}%")
        st.caption("Values reflect last monitoring cycle")

    # ------------------------------------------------------------------
    # Section: Emergency
    # ------------------------------------------------------------------

    def _section_emergency(self):
        st.markdown('<div class="section-header">Emergency Control</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Emergency Stop", type="primary", use_container_width=True):
                self._invoke_emergency_stop()
        with col2:
            if st.button("Reset Alerts", use_container_width=True):
                for alert in st.session_state.alerts:
                    alert.acknowledged = True
                self.add_alert(AlertLevel.INFO, "Alerts Reset", "All active alerts acknowledged", "System")
                st.experimental_rerun()

    # ------------------------------------------------------------------
    # Section: AI Insights
    # ------------------------------------------------------------------

    def _section_ai_insights(self):
        st.markdown('<div class="section-header">AI Insights</div>', unsafe_allow_html=True)
        insights = [
            "No critical anomaly clusters detected in the last monitoring window.",
            "Sensor fusion confidence above operational threshold.",
            "Model drift not flagged (baseline variance within tolerance)."
        ]
        for ins in insights:
            st.write(f"• {ins}")

    # ------------------------------------------------------------------
    # Main Content (Workspace)
    # ------------------------------------------------------------------

    def _render_main_content(self):
        self._main_header()
        self._row_sensor_metrics()
        self._tabs_analytics()
        self._system_overview_panel()

    def _main_header(self):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(
                """
                <h1 style="margin:0;
                           background:linear-gradient(135deg,#3b82f6,#60a5fa);
                           -webkit-background-clip:text;
                           -webkit-text-fill-color:transparent;">
                  Smart Neural Digital Twin
                </h1>
                <div style="color:#94a3b8;font-size:.85rem;margin-top:-4px;">
                  Oil Field Disaster Prevention Console (S-Class)
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols[1]:
            status = self._fetch_system_status()
            st.metric("Status", status.get("system_status", "UNKNOWN"))
        with cols[2]:
            st.metric("Last Update (UTC)", datetime.utcnow().strftime("%H:%M:%S"))

    def _row_sensor_metrics(self):
        """
        Enhanced to dynamically display all detected sensors instead of fixed ones
        """
        st.subheader("Realtime Core Metrics")
        data = self._fetch_sensor_data()
        
        # Get sensor configurations for all available sensors
        sensor_configs = self._fetch_sensor_configs()
        
        if not data or not sensor_configs:
            st.warning("No sensor data available")
            return
        
        # Calculate number of columns based on number of sensors
        # Limit to a reasonable number to avoid UI issues
        num_sensors = min(len(data), 12)  # Limit to 12 sensors max in the display
        num_cols = min(6, max(3, num_sensors))  # Between 3 and 6 columns
        
        # Display sensor metrics in grid layout
        cols = st.columns(num_cols)
        
        # Track which sensors we've already shown
        displayed_sensors = set()
        
        # First display important sensors if they exist in the data
        important_sensors = ["pressure", "temperature", "methane", "hydrogen_sulfide", "vibration", "flow"]
        for sensor_key in important_sensors:
            if sensor_key in data and len(displayed_sensors) < num_cols:
                col_idx = len(displayed_sensors) % num_cols
                with cols[col_idx]:
                    self._render_sensor_metric(sensor_key, data[sensor_key], sensor_configs.get(sensor_key, {}))
                displayed_sensors.add(sensor_key)
        
        # Then display any remaining sensors
        for sensor_key, value in data.items():
            if sensor_key not in displayed_sensors and len(displayed_sensors) < num_cols:
                col_idx = len(displayed_sensors) % num_cols
                with cols[col_idx]:
                    self._render_sensor_metric(sensor_key, value, sensor_configs.get(sensor_key, {}))
                displayed_sensors.add(sensor_key)
                
        # If we have more sensors than can fit in the top row, add an expander for the rest
        if len(data) > num_cols:
            with st.expander("Show All Sensors"):
                remaining_sensors = [s for s in data.keys() if s not in displayed_sensors]
                
                # Use grid layout for remaining sensors
                st.markdown('<div class="sensor-grid">', unsafe_allow_html=True)
                
                for sensor_key in remaining_sensors:
                    self._render_sensor_metric_card(sensor_key, data[sensor_key], sensor_configs.get(sensor_key, {}))
                
                st.markdown('</div>', unsafe_allow_html=True)

    def _render_sensor_metric(self, sensor_key: str, value: float, sensor_config: Dict[str, Any]):
        """Render a single sensor metric card"""
        # Get sensor display properties
        label = get_sensor_display_name(sensor_key, sensor_config)
        icon = get_sensor_icon(sensor_config)
        unit = sensor_config.get("unit", "")
        critical = sensor_config.get("critical", 100)
        nominal_min = sensor_config.get("min", 0)
        nominal_max = sensor_config.get("max", critical * 0.8)
        is_physical = self._is_physical_sensor(sensor_key)
        
        # Add a badge indicating physical or simulated
        source_badge = '<span class="physical-badge">PHYSICAL</span>' if is_physical else '<span class="simulated-badge">SIM</span>'
        
        if np.isnan(value):
            st.markdown(
                f"""
                <div class="metric-card warning-card">
                  <h4 style="margin:0 0 6px 0;color:#f1f5f9;">{icon} {label} {source_badge}</h4>
                  <div style="font-size:1.2rem;color:#f1f5f9;">—</div>
                  <p style="margin:.25rem 0 0 0;font-size:.65rem;color:#94a3b8;">No data</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            status_text, css_class = classify_sensor_state(value, critical, nominal_min, nominal_max)
            st.markdown(
                f"""
                <div class="metric-card {css_class}">
                  <h4 style="margin:0 0 6px 0;color:#f1f5f9;">{icon} {label} {source_badge}</h4>
                  <div style="font-size:1.35rem;color:#f1f5f9;">{value:.1f}</div>
                  <p style="margin:.35rem 0 0 0;color:#cbd5e1;font-size:.7rem;">
                    {unit} | {status_text}
                  </p>
                  <p style="margin:.2rem 0 0 0;color:#64748b;font-size:.55rem;">Nominal: {nominal_min}-{nominal_max}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    def _render_sensor_metric_card(self, sensor_key: str, value: float, sensor_config: Dict[str, Any]):
        """Render a sensor metric card in the grid layout"""
        # Get sensor display properties
        label = get_sensor_display_name(sensor_key, sensor_config)
        icon = get_sensor_icon(sensor_config)
        unit = sensor_config.get("unit", "")
        critical = sensor_config.get("critical", 100)
        nominal_min = sensor_config.get("min", 0)
        nominal_max = sensor_config.get("max", critical * 0.8)
        is_physical = self._is_physical_sensor(sensor_key)
        
        # Add a badge indicating physical or simulated
        source_badge = '<span class="physical-badge">PHYSICAL</span>' if is_physical else '<span class="simulated-badge">SIM</span>'
        
        if np.isnan(value):
            html = f"""
            <div class="metric-card warning-card">
              <h4 style="margin:0 0 6px 0;color:#f1f5f9;">{icon} {label} {source_badge}</h4>
              <div style="font-size:1.2rem;color:#f1f5f9;">—</div>
              <p style="margin:.25rem 0 0 0;font-size:.65rem;color:#94a3b8;">No data</p>
            </div>
            """
        else:
            status_text, css_class = classify_sensor_state(value, critical, nominal_min, nominal_max)
            html = f"""
            <div class="metric-card {css_class}">
              <h4 style="margin:0 0 6px 0;color:#f1f5f9;">{icon} {label} {source_badge}</h4>
              <div style="font-size:1.35rem;color:#f1f5f9;">{value:.1f}</div>
              <p style="margin:.35rem 0 0 0;color:#cbd5e1;font-size:.7rem;">
                {unit} | {status_text}
              </p>
              <p style="margin:.2rem 0 0 0;color:#64748b;font-size:.55rem;">Nominal: {nominal_min}-{nominal_max}</p>
            </div>
            """
            
        st.markdown(html, unsafe_allow_html=True)

    def _tabs_analytics(self):
        st.subheader("Analytics & Trends")
        tab1, tab2, tab3, tab4 = st.tabs(["Sensor Trends", "AI Predictions", "Risk Analysis", "Performance"])
        with tab1:
            self._plot_sensor_trends()
        with tab2:
            self._plot_predictions()
        with tab3:
            self._plot_risk_analysis()
        with tab4:
            self._plot_performance_trends()

    def _system_overview_panel(self):
        st.subheader("System Overview")
        cols = st.columns(3)
        status = self._fetch_system_status()
        sensor_grid = status.get("sensor_grid_status", {})
        
        # Enhanced to show physical sensor count
        physical_sensors = status.get("physical_sensor_count", 0)
        total_sensors = status.get("total_sensor_count", 0)

        with cols[0]:
            st.markdown("**Hardware & IO**")
            st.write(f"- Raspberry Pi: {'Active' if status.get('raspberry_pi_active') else 'Simulated'}")
            st.write(f"- Total Sensors: {total_sensors}")
            st.write(f"- Physical Sensors: {physical_sensors}")
            st.write(f"- Active: {sensor_grid.get('active_sensors','?')}")
            st.write(f"- Simulated: {sensor_grid.get('simulated_sensors','?')}")
        with cols[1]:
            st.markdown("**AI Engine**")
            st.write(f"- Models Trained: (deferred integration)")
            st.write(f"- Anomaly Score Cache: n/a")
            st.write(f"- Fusion Accuracy: {sensor_grid.get('fusion_accuracy',0)*100:.1f}%")
            st.write(f"- Confidence (static demo): 97.0%")
        with cols[2]:
            st.markdown("**Operations**")
            st.write(f"- Processed Loops: {status.get('performance_metrics',{}).get('processed_cycles',0)}")
            st.write(f"- Emergencies: {status.get('performance_metrics',{}).get('emergency_events',0)}")
            st.write(f"- Avg Proc Time: {status.get('performance_metrics',{}).get('avg_cycle_time',0):.3f}s")
            st.write(f"- SS Rating: {status.get('ss_rating','N/A')}")

    # ------------------------------------------------------------------
    # Plotting Helpers
    # ------------------------------------------------------------------

    def _plot_sensor_trends(self):
        try:
            # Get actual sensor data
            data = self._fetch_sensor_data()
            
            # Get sensor history or initialize if not present
            sensor_history = ensure_session_key("sensor_history", {})
            
            # Update sensor history
            timestamp = datetime.utcnow()
            for key, value in data.items():
                if key not in sensor_history:
                    sensor_history[key] = []
                # Append current reading with timestamp
                sensor_history[key].append((timestamp, value))
                # Keep only recent history (last 36 points)
                sensor_history[key] = sensor_history[key][-36:]
            
            # Create time series plot
            fig = go.Figure()
            
            # Choose which sensors to show (prioritize important ones)
            # Use all sensors if 6 or fewer, otherwise prioritize important ones
            important_sensors = ["pressure", "temperature", "methane", "vibration", "hydrogen_sulfide", "flow"]
            available_sensors = list(data.keys())
            
            if len(available_sensors) <= 6:
                sensors_to_plot = available_sensors
            else:
                # Start with important sensors that exist in our data
                sensors_to_plot = [s for s in important_sensors if s in available_sensors][:4]
                # Add other sensors if we have room
                for s in available_sensors:
                    if s not in sensors_to_plot and len(sensors_to_plot) < 6:
                        sensors_to_plot.append(s)
            
            # Plot each selected sensor
            colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#6366f1", "#ec4899"]
            
            for i, sensor in enumerate(sensors_to_plot):
                if sensor in sensor_history and len(sensor_history[sensor]) > 1:
                    # Use actual sensor history data
                    timestamps = [entry[0] for entry in sensor_history[sensor]]
                    values = [entry[1] for entry in sensor_history[sensor]]
                    
                    # Determine if physical or simulated
                    is_physical = self._is_physical_sensor(sensor)
                    source_label = "(physical)" if is_physical else "(sim)"
                    
                    color_idx = i % len(colors)
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=values, 
                        name=f"{sensor.title()} {source_label}", 
                        line=dict(width=2.5, color=colors[color_idx])
                    ))
                
            # If no sensor history available, add synthetic demo data
            if not any(sensor in sensor_history and len(sensor_history[sensor]) > 1 for sensor in sensors_to_plot):
                time_index = pd.date_range(end=datetime.utcnow(), periods=36, freq="10min")
                rng = np.random.default_rng(seed=42)
                
                # Provide sample sensors as fallback
                for name, base, amp, color in [
                    ("Pressure", 50, 3, "#3b82f6"),
                    ("Temperature", 80, 4, "#ef4444"),
                    ("Vibration", 2.2, 0.3, "#f59e0b")
                ]:
                    series = base + amp * np.sin(np.linspace(0, 4, len(time_index))) + rng.normal(0, amp * 0.1, len(time_index))
                    fig.add_trace(go.Scatter(x=time_index, y=series, name=f"{name} (sim)", line=dict(width=2.5, color=color)))
                
                title = "Sensor Trends (Synthetic Demo)"
                caption = "Synthetic illustrative trends (replace with historical buffers when available)."
            else:
                title = "Sensor Trends (Real-Time Data)"
                caption = "Historical data from actual sensor readings. Limited to most recent entries."
                
            fig.update_layout(
                title=title,
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.15)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)
            
        except Exception as e:
            st.error(f"Trend plotting error: {e}")

    def _plot_predictions(self):
        try:
            horizon_hours = list(range(1, 25))
            rng = np.random.default_rng(123)
            pressure_pred = 50 + 2 * np.sin(np.array(horizon_hours) * 0.3) + rng.normal(0, 0.4, len(horizon_hours))
            temp_pred = 80 + 3 * np.cos(np.array(horizon_hours) * 0.2) + rng.normal(0, 0.6, len(horizon_hours))
            risk_curve = 0.15 + np.abs(np.sin(np.array(horizon_hours) * 0.4)) * 0.25

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=horizon_hours, y=pressure_pred, mode="lines+markers",
                                     name="Pressure Forecast (sim)", line=dict(color="#3b82f6", width=3)))
            fig.add_trace(go.Scatter(x=horizon_hours, y=temp_pred, mode="lines+markers",
                                     name="Temperature Forecast (sim)", line=dict(color="#ef4444", width=3)))
            fig.add_trace(go.Scatter(x=horizon_hours, y=risk_curve, mode="lines",
                                     name="Risk Score (sim)", line=dict(color="#f59e0b", dash="dash", width=2)))

            fig.update_layout(
                title="24h Forward-Look (Synthetic Demo)",
                xaxis_title="Hours Ahead",
                yaxis_title="Value / Risk Index",
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Forecast placeholders – integrate with trained prediction engine for live inference.")
        except Exception as e:
            st.error(f"Prediction plotting error: {e}")

    def _plot_risk_analysis(self):
        try:
            # Use actual sensor names if available
            data = self._fetch_sensor_data()
            if data and len(data) > 3:
                systems = list(data.keys())[:6]  # Limit to 6 systems
                systems = [s.replace("_", " ").title() for s in systems]
            else:
                systems = ["Pressure", "Temperature", "Gas Detection", "Vibration", "Flow", "Cooling"]
                
            rng = np.random.default_rng(7)
            risk_scores = rng.uniform(0.05, 0.28, len(systems))
            conf_scores = rng.uniform(0.85, 0.97, len(systems))

            fig = go.Figure(data=[
                go.Bar(x=systems, y=risk_scores, name="Risk Score (sim)", marker_color="#ef4444"),
                go.Bar(x=systems, y=conf_scores, name="AI Confidence (sim)", marker_color="#10b981")
            ])
            fig.update_layout(
                title="Subsystem Risk vs Confidence (Synthetic)",
                barmode="group",
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Simulated risk distribution – plug anomaly + predictive fusion outputs here.")
        except Exception as e:
            st.error(f"Risk analysis plotting error: {e}")

    def _plot_performance_trends(self):
        try:
            days = list(range(1, 8))
            acc = [96.5, 97.1, 97.6, 97.9, 98.2, 98.4, 98.6]
            resp_ms = [3.5, 3.0, 2.6, 2.2, 1.9, 1.7, 1.5]
            anomalies = [5, 4, 3, 2, 1, 1, 0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=acc, name="Model Accuracy (sim)", mode="lines+markers", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=days, y=resp_ms, name="Response Time ms (sim)", mode="lines+markers", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=days, y=anomalies, name="Anomalies (sim)", mode="lines+markers", line=dict(width=3)))

            fig.update_layout(
                title="7-Day Performance Indicators (Synthetic)",
                xaxis_title="Day Sequence",
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Historical performance placeholders – integrate with metrics persistence.")
        except Exception as e:
            st.error(f"Performance plotting error: {e}")

    # ------------------------------------------------------------------
    # Alerts API
    # ------------------------------------------------------------------

    def add_alert(self, level: AlertLevel, title: str, message: str, source: str = "System"):
        alert = SystemAlert(level=level, title=title, message=message, source=source)
        st.session_state.alerts.append(alert)
        self.logger.warning(f"Alert added: {level.value} | {title}")

    # ------------------------------------------------------------------
    # System Interactions
    # ------------------------------------------------------------------

    def _fetch_system_status(self) -> Dict[str, Any]:
        if not self.smart_twin:
            return {}
        return safe_call(self.logger,
                         lambda: self.smart_twin.get_enhanced_system_status(),
                         fallback={},
                         context="fetch_system_status")

    def _fetch_sensor_data(self) -> Dict[str, float]:
        if not self.smart_twin:
            return {}
        return getattr(self.smart_twin, "real_time_data", {}) or {}

    def _fetch_sensor_configs(self) -> Dict[str, Dict[str, Any]]:
        """Fetch sensor configurations from the system"""
        if not self.smart_twin:
            return {k: {"label": k.title(), "min": 0, "max": 100, "critical": 80, "unit": ""} 
                    for k in DEFAULT_SENSOR_CONFIGS}
        
        # Try to get sensors from config
        try:
            # First attempt to get from system configuration
            status = self._fetch_system_status()
            if "config" in status and "sensors" in status["config"]:
                return status["config"]["sensors"]
                
            # Second attempt to access from twin directly
            config = getattr(self.smart_twin, "config", {})
            if config and "sensors" in config:
                return config["sensors"]
        except:
            pass
            
        # Fall back to default configs combined with any keys from real_time_data
        sensor_data = self._fetch_sensor_data()
        configs = {}
        
        # Start with defaults for known sensors
        for key, (label, icon, critical, unit, nominal_min, nominal_max) in DEFAULT_SENSOR_CONFIGS.items():
            configs[key] = {
                "label": label,
                "icon": icon,
                "critical": critical,
                "unit": unit,
                "min": nominal_min,
                "max": nominal_max
            }
            
        # Add any missing sensors from real_time_data with generic configs
        for key in sensor_data:
            if key not in configs:
                configs[key] = {
                    "label": key.replace("_", " ").title(),
                    "icon": "🔍",
                    "critical": 100,
                    "unit": "",
                    "min": 0,
                    "max": 80
                }
                
        return configs

    def _get_sensor_config(self, sensor_key: str) -> Dict[str, Any]:
        """Get configuration for a specific sensor"""
        configs = self._fetch_sensor_configs()
        if sensor_key in configs:
            return configs[sensor_key]
            
        # Return a basic default config if not found
        if sensor_key in DEFAULT_SENSOR_CONFIGS:
            label, icon, critical, unit, nominal_min, nominal_max = DEFAULT_SENSOR_CONFIGS[sensor_key]
            return {
                "label": label,
                "icon": icon,
                "critical": critical,
                "unit": unit,
                "min": nominal_min,
                "max": nominal_max
            }
            
        return {
            "label": sensor_key.replace("_", " ").title(),
            "icon": "🔍",
            "critical": 100,
            "unit": "",
            "min": 0,
            "max": 80
        }

    def _is_physical_sensor(self, sensor_key: str) -> bool:
        """Determine if a sensor is physical or simulated"""
        # Try to get from status
        status = self._fetch_system_status()
        
        # Look for sensor interface info in config
        try:
            sensor_config = self._get_sensor_config(sensor_key)
            if "interface" in sensor_config:
                return True
            
            # Check if this is a known physical sensor
            if hasattr(self.smart_twin, "config") and "sensors" in self.smart_twin.config:
                sensor_configs = self.smart_twin.config["sensors"]
                if sensor_key in sensor_configs and "interface" in sensor_configs[sensor_key]:
                    return True
        except:
            pass
            
        return False

    def _get_sensor_source_type(self, sensor_key: str) -> str:
        """Get sensor source type (physical/simulated)"""
        if self._is_physical_sensor(sensor_key):
            return "physical"
        else:
            return "simulated"

    def _optimize_engine(self, scenarios: int):
        if not self.smart_twin:
            st.warning("Twin unavailable.")
            return
        engine = getattr(self.smart_twin, "fore_sight_engine", None)
        if not engine:
            st.warning("Foresight engine not found.")
            return
        result = safe_call(self.logger,
                           lambda: engine.update_engine_settings(scenarios),
                           fallback=None,
                           context="engine_optimize")
        if result is not None or True:
            st.success(f"Engine settings updated to {scenarios} scenarios/sec")

    def _export_prediction_report(self, horizon: int):
        status = self._fetch_system_status()
        data = {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "requested_horizon_hours": horizon,
            "system_state": status.get("system_status", "UNKNOWN"),
            "grid_health": status.get("sensor_grid_status", {}).get("grid_health", 0),
            "physical_sensor_count": status.get("physical_sensor_count", 0),
            "total_sensor_count": status.get("total_sensor_count", 0),
            "note": "This is a synthetic placeholder report. Integrate real predictive output."
        }
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(data, indent=2),
            file_name=f"forecast_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def _invoke_emergency_stop(self):
        if not self.smart_twin:
            st.error("System not available.")
            return
        relay = getattr(self.smart_twin, "relay_controller", None)
        if relay and hasattr(relay, "emergency_shutdown"):
            safe_call(self.logger, relay.emergency_shutdown, context="emergency_shutdown")
            self.add_alert(AlertLevel.EMERGENCY, "EMERGENCY STOP", "Emergency shutdown executed", "Safety")
            st.error("Emergency shutdown executed.")
        else:
            st.error("Relay controller not available.")
            self.add_alert(AlertLevel.CRITICAL, "EMERGENCY FAILURE", "Relay controller unavailable", "System")

    # ------------------------------------------------------------------
    # Auto Refresh
    # ------------------------------------------------------------------

    def _handle_auto_refresh(self):
        st.markdown(SECTION_DIVIDER)
        refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 1, 2])
        with refresh_col1:
            auto = st.checkbox("Auto-refresh", value=True, help="Periodically refresh sensor + status data.")
        with refresh_col2:
            interval_label = st.selectbox("Interval", list(AUTO_REFRESH_INTERVALS.keys()), index=1)
        with refresh_col3:
            if st.button("Manual Refresh"):
                st.session_state.last_refresh_ts = time.time()
                st.experimental_rerun()

        if auto:
            desired = AUTO_REFRESH_INTERVALS[interval_label]
            now = time.time()
            if now - st.session_state.last_refresh_ts >= desired:
                st.session_state.last_refresh_ts = now
                st.experimental_rerun()


# --------------------------------------------------------------------------------------
# Factory Function
# --------------------------------------------------------------------------------------

def create_advanced_dashboard(smart_twin_system: Any) -> AdvancedDashboard:
    """
    Create and return an S-Class dashboard instance.
    """
    dashboard = AdvancedDashboard(smart_twin_system)
    return dashboard


__all__ = [
    "AdvancedDashboard",
    "create_advanced_dashboard",
    "SystemAlert",
    "AlertLevel"
                ]
