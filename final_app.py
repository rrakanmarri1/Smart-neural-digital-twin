import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import platform
import time
import json
import uuid
import functools
from datetime import datetime, timedelta
import threading
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure page title, favicon, layout
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try OpenAI import, handle missing gracefully
try:
    import openai
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

# Try Twilio import
try:
    from twilio.rest import Client
    twilio_available = True
except ImportError:
    twilio_available = False

# Try additional packages
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

try:
    import graphviz
    graphviz_available = True
except ImportError:
    graphviz_available = False

# ----- LOGO SVG -----
logo_svg = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none">
  <circle cx="32" cy="32" r="32" fill="url(#grad1)"/>
  <defs>
    <linearGradient id="grad1" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
      <stop stop-color="#43cea2"/>
      <stop offset="1" stop-color="#185a9d"/>
    </linearGradient>
  </defs>
  <g>
    <ellipse cx="32" cy="32" rx="22" ry="10" fill="#fff" fill-opacity="0.18"/>
    <ellipse cx="32" cy="32" rx="12" ry="22" fill="#fff" fill-opacity="0.10"/>
    <path d="M20 32a12 12 0 1 0 24 0 12 12 0 1 0 -24 0" fill="#fff" fill-opacity="0.7"/>
    <path d="M32 16v32M16 32h32" stroke="#185a9d" stroke-width="2" stroke-linecap="round"/>
    <circle cx="32" cy="32" r="6" fill="#43cea2" stroke="#185a9d" stroke-width="2"/>
  </g>
</svg>
"""

# App version and last updated info
APP_VERSION = "2.5.1"
LAST_UPDATED = "2025-08-04 08:56:06"
DEVELOPER = "rrakanmarri1"

# ----- CONFIGURATION MANAGEMENT -----
def load_config() -> dict:
    """Load configuration from file with environment-specific overrides"""
    # Default configuration
    default_config = {
        "mqtt": {
            "broker": "test.mosquitto.org",
            "port": 1883,
            "topic": "digitaltwin/test/temperature",
            "fallback_enabled": True
        },
        "api": {
            "openai_model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.3
        },
        "ui": {
            "default_theme": "dark",
            "default_language": "en",
            "animation_enabled": True,
            "enable_voice_interface": False
        },
        "simulation": {
            "data_points": 96,
            "update_frequency": 5,
            "anomalies_enabled": True
        },
        "features": {
            "3d_visualization": True,
            "ai_chat": True,
            "heatmap": True,
            "dashboard": True,
            "enable_analytics": True,
            "error_tracking": True
        },
        "performance": {
            "cache_timeout": 300,  # seconds
            "lazy_loading": True,
            "prefetch_data": True
        }
    }
    
    # Try to load from config file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = default_config
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = json.load(f)
                
                # Merge configurations (nested update)
                def merge_dicts(d1, d2):
                    for k, v in d2.items():
                        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                            merge_dicts(d1[k], v)
                        else:
                            d1[k] = v
                
                merge_dicts(config, file_config)
    except Exception as e:
        st.warning(f"Failed to load config: {e}")
    
    # Environment variable overrides
    env_prefix = "DIGITAL_TWIN_"
    for key in os.environ:
        if key.startswith(env_prefix):
            parts = key[len(env_prefix):].lower().split('_')
            
            # Navigate to the correct nested dict
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value (convert types if needed)
            value = os.environ[key]
            try:
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                    value = float(value)
            except:
                pass
            
            current[parts[-1]] = value
    
    return config

# Load configuration
CONFIG = load_config()

# MQTT Config
MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_TOPIC = CONFIG["mqtt"]["topic"]

# Secure config via environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")
TWILIO_TO = os.environ.get("TWILIO_TO", "")

# ----- PERFORMANCE MONITORING & ERROR TRACKING -----
class PerformanceTracker:
    """Track performance metrics for the application"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def track_function(self, func):
        """Decorator to track function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            func_name = func.__name__
            if func_name not in self.metrics:
                self.metrics[func_name] = {
                    "calls": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0,
                    "last_call": datetime.now().isoformat()
                }
            
            metrics = self.metrics[func_name]
            metrics["calls"] += 1
            metrics["total_time"] += execution_time
            metrics["min_time"] = min(metrics["min_time"], execution_time)
            metrics["max_time"] = max(metrics["max_time"], execution_time)
            metrics["last_call"] = datetime.now().isoformat()
            
            return result
        
        return wrapper
    
    def get_app_metrics(self):
        """Get overall application metrics"""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "function_calls": sum(m["calls"] for m in self.metrics.values()),
            "slowest_function": max(self.metrics.items(), key=lambda x: x[1]["max_time"])[0] if self.metrics else None,
            "total_execution_time": sum(m["total_time"] for m in self.metrics.values())
        }
    
    def get_function_metrics(self):
        """Get metrics for all tracked functions"""
        result = {}
        for func_name, metrics in self.metrics.items():
            avg_time = metrics["total_time"] / metrics["calls"] if metrics["calls"] > 0 else 0
            result[func_name] = {
                "calls": metrics["calls"],
                "avg_time": avg_time,
                "min_time": metrics["min_time"] if metrics["min_time"] != float('inf') else 0,
                "max_time": metrics["max_time"],
                "last_call": metrics["last_call"]
            }
        return result

# Initialize performance tracker
perf_tracker = PerformanceTracker()

class ExceptionMonitor:
    """Monitor and track exceptions across the application"""
    
    def __init__(self):
        self.exceptions = []
        self.max_exceptions = 100  # Limit storage to prevent memory issues
    
    def capture(self, exception, context=None):
        """Capture an exception with optional context"""
        if len(self.exceptions) >= self.max_exceptions:
            self.exceptions.pop(0)  # Remove oldest exception
        
        self.exceptions.append({
            "exception": str(exception),
            "type": type(exception).__name__,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        })
    
    def get_summary(self):
        """Get a summary of recent exceptions"""
        if not self.exceptions:
            return "No exceptions captured"
        
        # Group exceptions by type
        grouped = {}
        for exc in self.exceptions:
            exc_type = exc["type"]
            if exc_type not in grouped:
                grouped[exc_type] = {"count": 0, "examples": []}
            
            grouped[exc_type]["count"] += 1
            if len(grouped[exc_type]["examples"]) < 3:  # Store up to 3 examples
                grouped[exc_type]["examples"].append(exc)
        
        # Format the summary
        summary = []
        for exc_type, data in grouped.items():
            summary.append(f"**{exc_type}**: {data['count']} occurrences")
            for example in data["examples"]:
                summary.append(f"  - {example['exception']} ({example['timestamp']})")
        
        return "\n".join(summary)
    
    def get_most_common(self, limit=5):
        """Get the most common exception types"""
        if not self.exceptions:
            return []
        
        # Count exceptions by type
        counts = {}
        for exc in self.exceptions:
            exc_type = exc["type"]
            counts[exc_type] = counts.get(exc_type, 0) + 1
        
        # Sort by count (descending)
        sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_types[:limit]
    
    def clear(self):
        """Clear all captured exceptions"""
        self.exceptions = []

# Initialize exception monitor
exception_monitor = ExceptionMonitor()

# ----- ANALYTICS TRACKING -----
class AnalyticsTracker:
    """Track user interactions and feature usage"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.interactions = []
        self.feature_usage = {}
        self.page_views = {}
        self.session_start = datetime.now()
    
    def track_interaction(self, event_type, details=None):
        """Track a user interaction"""
        if not CONFIG["features"]["enable_analytics"]:
            return
            
        self.interactions.append({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
    
    def track_feature_usage(self, feature_name):
        """Track usage of a specific feature"""
        if not CONFIG["features"]["enable_analytics"]:
            return
            
        if feature_name in self.feature_usage:
            self.feature_usage[feature_name] += 1
        else:
            self.feature_usage[feature_name] = 1
    
    def track_page_view(self, page_name):
        """Track a page view"""
        if not CONFIG["features"]["enable_analytics"]:
            return
            
        if page_name in self.page_views:
            self.page_views[page_name] += 1
        else:
            self.page_views[page_name] = 1
    
    def get_analytics_summary(self):
        """Get a summary of analytics data"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "total_interactions": len(self.interactions),
            "most_used_features": sorted(self.feature_usage.items(), key=lambda x: x[1], reverse=True),
            "page_views": self.page_views
        }

# Initialize analytics tracker
analytics = AnalyticsTracker()

# ----- CACHING MECHANISM -----
def timed_cache(timeout_seconds=300):
    """A decorator that caches the result of a function for a specified time period"""
    def decorator(func):
        # Store cache as a dictionary: {args: (timestamp, result)}
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if we have a cached result that's still valid
            if key in cache:
                timestamp, result = cache[key]
                if time.time() - timestamp <= timeout_seconds:
                    return result
            
            # No valid cached result, call the function
            result = func(*args, **kwargs)
            cache[key] = (time.time(), result)
            return result
        
        # Add a method to clear the cache
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    
    return decorator

# ----- UTILITY FUNCTIONS -----
def safe_execute(func, *args, context=None, **kwargs):
    """Execute a function safely, capturing any exceptions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exception_monitor.capture(e, context)
        raise  # Re-raise the exception after capturing

# Default demo data for API failures
MOCK_RESPONSES = {
    "en": {
        "temperature": "Based on the latest data, the current temperature is around 57.3°C. This is within normal operating parameters for this system. The trend shows a slight increase over the past hour but still within acceptable limits.",
        "why_high_temp": "The elevated temperature is likely due to three factors: 1) Increased production load over the past shift, 2) Seasonal ambient temperature rise, and 3) The compressor maintenance scheduled for last week was postponed. I recommend checking the cooling system filters and proceeding with the delayed maintenance.",
        "energy": "Energy analysis shows compressors are consuming 51% of total usage. You could reduce consumption by 12% by implementing a staggered startup sequence and running at 85% capacity during non-peak hours. I estimate annual savings of approximately $21,000.",
        "kpi": "Plant KPIs are generally positive. Efficiency is at 96% (target: 98%), water usage is optimized, but the incident rate is higher than target. The methane levels show concerning fluctuation in the past 48 hours - this requires investigation as it correlates with the temperature spikes.",
        "feedback": "Operator feedback indicates recurring concerns about compressor #2 vibration and intermittent sensor readings on the eastern pipeline segment. Multiple operators have reported notification delays in the alert system. These issues should be prioritized for maintenance."
    },
    "ar": {
        "temperature": "بناءً على أحدث البيانات، تبلغ درجة الحرارة الحالية حوالي 57.3 درجة مئوية. وهذا ضمن معايير التشغيل العادية لهذا النظام. يُظهر الاتجاه زيادة طفيفة على مدار الساعة الماضية ولكنه لا يزال ضمن الحدود المقبولة.",
        "why_high_temp": "يرجع ارتفاع درجة الحرارة على الأرجح إلى ثلاثة عوامل: 1) زيادة حمل الإنتاج خلال الوردية الماضية، 2) ارتفاع درجة الحرارة المحيطة الموسمية، 3) تأجيل صيانة الضاغط المقررة الأسبوع الماضي. أوصي بفحص مرشحات نظام التبريد والمضي قدماً في الصيانة المتأخرة.",
        "energy": "يظهر تحليل الطاقة أن الضواغط تستهلك 51٪ من إجمالي الاستخدام. يمكنك تقليل الاستهلاك بنسبة 12٪ من خلال تنفيذ تسلسل بدء تشغيل متدرج والتشغيل بسعة 85٪ خلال ساعات عدم الذروة. أقدر التوفير السنوي بحوالي 21,000 دولار.",
        "kpi": "مؤشرات أداء المصنع إيجابية بشكل عام. الكفاءة عند 96٪ (الهدف: 98٪)، استخدام المياه محسن، لكن معدل الحوادث أعلى من المستهدف. تظهر مستويات الميثان تقلبات مثيرة للقلق في الساعات الـ 48 الماضية - وهذا يتطلب التحقيق لأنه يرتبط بارتفاعات درجة الحرارة.",
        "feedback": "تشير ملاحظات المشغل إلى مخاوف متكررة بشأن اهتزاز الضاغط رقم 2 وقراءات المستشعر المتقطعة في قطاع خط الأنابيب الشرقي. أبلغ العديد من المشغلين عن تأخير الإخطارات في نظام التنبيه. يجب إعطاء الأولوية لهذه المشكلات للصيانة."
    }
}

# Generate consistent simulation data for demo
@perf_tracker.track_function
def generate_simulation_data():
    """Generate realistic plant simulation data"""
    np.random.seed(42)  # For reproducible results
    now = datetime.now()
    hours_24 = timedelta(hours=24)
    
    # Create time range
    times = pd.date_range(now - hours_24, now, periods=CONFIG["simulation"]["data_points"])
    
    # Base patterns with some randomness
    temp_base = 55 + 5 * np.sin(np.linspace(0, 4*np.pi, CONFIG["simulation"]["data_points"]))
    pressure_base = 7 + np.sin(np.linspace(0, 2*np.pi, CONFIG["simulation"]["data_points"])) * 0.8
    methane_base = 1.2 + 0.6 * np.sin(np.linspace(0, 3*np.pi, CONFIG["simulation"]["data_points"]))
    
    # Add realistic noise
    temp = temp_base + np.random.normal(0, 0.7, CONFIG["simulation"]["data_points"])
    pressure = pressure_base + np.random.normal(0, 0.2, CONFIG["simulation"]["data_points"])
    methane = np.clip(methane_base + np.random.normal(0, 0.15, CONFIG["simulation"]["data_points"]), 0, 5)
    
    if CONFIG["simulation"]["anomalies_enabled"]:
        # Add anomaly events
        # 1. Temperature spike
        temp[65:75] += np.linspace(0, 8, 10)  # Gradual rise
        temp[75:85] += np.linspace(8, 0, 10)  # Gradual fall
        
        # 2. Pressure drop
        pressure[40:45] -= 2
        
        # 3. Methane leak
        methane[70:85] += 1.5
    
    return pd.DataFrame({
        "time": times,
        "Temperature": temp,
        "Pressure": pressure,
        "Methane": methane
    })

# Function to initialize session state with sensible defaults
def initialize_app_state():
    defaults = {
        "lang": CONFIG["ui"]["default_language"],
        "scenario_step": 0,
        "solution_idx": 0,
        "theme": CONFIG["ui"]["default_theme"],
        "mqtt_temp": None,
        "mqtt_last": None,
        "mqtt_started": False,
        "sms_sent": False,
        "feedback_list": [],
        "chat_history": [],
        "simulation_data": generate_simulation_data(),
        "page_view_count": {},
        "performance_mode": "high",
        "disable_animations": False,
        "use_fallback_3d": False,
        "reduce_data_points": False,
        "onboarding_complete": False,
        "user_preferences": {},
        "ai_learning": {
            "user_queries": [],
            "common_topics": {},
            "feedback_ratings": {}
        }
    }
    
    # Only set if not already in session state
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Initialize app state
initialize_app_state()

# MQTT background thread
def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)
def on_message(client, userdata, msg):
    try:
        val = float(msg.payload.decode())
        st.session_state["mqtt_temp"] = val
        st.session_state["mqtt_last"] = datetime.now()
    except Exception:
        pass

@perf_tracker.track_function
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        if CONFIG["features"]["error_tracking"]:
            exception_monitor.capture(e, {"component": "mqtt_thread"})
        pass

if not st.session_state["mqtt_started"]:
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    st.session_state["mqtt_started"] = True

# Simulated MQTT temperature for demo
@perf_tracker.track_function
def simulate_mqtt_temp():
    if st.session_state["mqtt_temp"] is None and CONFIG["mqtt"]["fallback_enabled"]:
        st.session_state["mqtt_temp"] = np.random.normal(55, 5)
        st.session_state["mqtt_last"] = datetime.now()
        analytics.track_interaction("simulated_mqtt", {"value": st.session_state["mqtt_temp"]})

# OpenAI setup with fallback
class APIFactory:
    """Factory for creating API clients with proper error handling"""
    
    @staticmethod
    def create_openai_client(api_key=None):
        """Create an OpenAI client with fallback handling"""
        if not openai_available:
            return None
        
        try:
            api_key = api_key or OPENAI_API_KEY
            if not api_key:
                return None
            
            client = OpenAI(api_key=api_key)
            return client
        except Exception as e:
            if CONFIG["features"]["error_tracking"]:
                exception_monitor.capture(e, {"component": "openai_client_init"})
            return None
    
    @staticmethod
    def create_twilio_client(account_sid=None, auth_token=None):
        """Create a Twilio client with fallback handling"""
        if not twilio_available:
            return None
        
        try:
            account_sid = account_sid or TWILIO_SID
            auth_token = auth_token or TWILIO_AUTH
            
            if not (account_sid and auth_token):
                return None
            
            client = Client(account_sid, auth_token)
            return client
        except Exception as e:
            if CONFIG["features"]["error_tracking"]:
                exception_monitor.capture(e, {"component": "twilio_client_init"})
            return None

# Initialize API clients
openai_client = APIFactory.create_openai_client()
twilio_client = APIFactory.create_twilio_client()

@timed_cache(timeout_seconds=CONFIG["performance"]["cache_timeout"])
@perf_tracker.track_function
def ask_llm_advanced(prompt: str, lang: str, context: Optional[str] = None, root_cause: Optional[str] = None) -> str:
    """
    Enhanced AI Copilot with fallbacks and extensive error handling
    
    Args:
        prompt: The user's query
        lang: Language code (en or ar)
        context: Optional context to include
        root_cause: Optional root cause information
        
    Returns:
        AI response as string
    """
    analytics.track_feature_usage("ai_assistant")
    
    # Add user query to learning system
    if "ai_learning" in st.session_state:
        if "user_queries" not in st.session_state["ai_learning"]:
            st.session_state["ai_learning"]["user_queries"] = []
            
        st.session_state["ai_learning"]["user_queries"].append({
            "query": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep list manageable
        if len(st.session_state["ai_learning"]["user_queries"]) > 50:
            st.session_state["ai_learning"]["user_queries"] = st.session_state["ai_learning"]["user_queries"][-50:]
    
    # If OpenAI isn't available or fails, use mock responses
    if not openai_available or not OPENAI_API_KEY or openai_client is None:
        # Return mock responses based on keywords
        mock = MOCK_RESPONSES[lang]
        if "temperature" in prompt.lower():
            return mock["temperature"]
        elif any(w in prompt.lower() for w in ["why", "cause", "root", "سبب", "جذر", "لماذا"]):
            return mock["why_high_temp"]
        elif any(w in prompt.lower() for w in ["energy", "power", "طاقة"]):
            return mock["energy"]
        elif any(w in prompt.lower() for w in ["kpi", "performance", "أداء", "مؤشرات"]):
            return mock["kpi"]
        elif any(w in prompt.lower() for w in ["feedback", "comment", "ملاحظات"]):
            return mock["feedback"]
        else:
            return mock["temperature"]  # Default response

    # Add user preferences if they exist
    if "user_preferences" in st.session_state and st.session_state["user_preferences"]:
        if context:
            context = f"{context}\n\nUser preferences: {json.dumps(st.session_state['user_preferences'])}"
        else:
            context = f"User preferences: {json.dumps(st.session_state['user_preferences'])}"

    system_en = """You are an expert AI assistant for an industrial digital twin platform called 'Smart Neural Digital Twin'.
You have access to real-time plant data and advanced analytics. Your core capabilities include:
- Answering operational, troubleshooting, and data analysis questions.
- Performing root cause analysis: if a user asks "why" or "root cause", analyze past incidents and suggest a probable root.
- Giving actionable recommendations based on the plant's digital twin data, scenario playback, and forecast.
- Summarizing plant KPIs, risk factors, and energy optimization opportunities.
- If 'context' is provided, summarize and use it.
- If 'root_cause' is provided, use it to explain system failures or propose mitigations.

If asked about specific values, refer to the latest in-memory data if available. Reply in clear, concise, and helpful language.
Current Date and Time: 2025-08-04 08:56:06 UTC
"""
    system_ar = """أنت مساعد ذكاء صناعي خبير لمنصة التوأم الرقمي الصناعي المسماة 'التوأم الرقمي العصبي الذكي'.
لديك إمكانية الوصول إلى بيانات المصنع الحية والتحليلات المتقدمة. قدراتك الأساسية:
- الإجابة عن أسئلة التشغيل والتحليل وحل المشكلات.
- تحليل السبب الجذري: إذا سأل المستخدم "لماذا" أو "السبب الجذري"، قم بتحليل الحوادث واقترح السبب الممكن.
- إعطاء توصيات عملية بناءً على بيانات التوأم الرقمي وتشغيل السيناريو والتوقعات.
- تلخيص مؤشرات الأداء، عوامل المخاطر، وفرص تحسين الطاقة.
- إذا تم توفير 'context'، لخصه واستخدمه.
- إذا تم توفير 'root_cause'، فاشرح به أسباب الأعطال أو طرق المعالجة.

إذا سُئلت عن قيم معينة، استند للبيانات الأحدث المتوفرة بالذاكرة. أجب بوضوح واحترافية.
التاريخ والوقت الحالي: 2025-08-04 08:56:06 UTC
"""
    system = system_en if lang == "en" else system_ar

    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "system", "content": f"context: {context}"})
    if root_cause:
        messages.append({"role": "system", "content": f"root_cause: {root_cause}"})
    messages.append({"role": "user", "content": prompt})

    try:
        # Use the OpenAI client to create a chat completion
        resp = openai_client.chat.completions.create(
            model=CONFIG["api"]["openai_model"],
            messages=messages,
            temperature=CONFIG["api"]["temperature"],
            max_tokens=CONFIG["api"]["max_tokens"],
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Track the error
        if CONFIG["features"]["error_tracking"]:
            exception_monitor.capture(e, {"component": "openai_api", "prompt": prompt})
        
        # Fallback to mock responses on error
        st.warning(f"Using AI backup responses due to API limitations")
        
        # Use mock responses as fallback
        mock = MOCK_RESPONSES[lang]
        if "temperature" in prompt.lower():
            return mock["temperature"]
        elif any(w in prompt.lower() for w in ["why", "cause", "root", "سبب", "جذر", "لماذا"]):
            return mock["why_high_temp"]
        elif any(w in prompt.lower() for w in ["energy", "power", "طاقة"]):
            return mock["energy"]
        else:
            return mock["temperature"]  # Default response

# Twilio SMS
@perf_tracker.track_function
def send_sms(to: str, message: str) -> Tuple[bool, str]:
    """Send SMS using Twilio with error handling"""
    analytics.track_feature_usage("sms_alert")
    
    if not twilio_available:
        return False, "Twilio not installed."
    try:
        if not all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, to]):
            return False, "Twilio credentials or phone numbers not set."
        client = twilio_client or APIFactory.create_twilio_client()
        if not client:
            return False, "Failed to initialize Twilio client."
            
        message = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=to
        )
        return True, "Sent."
    except Exception as e:
        if CONFIG["features"]["error_tracking"]:
            exception_monitor.capture(e, {"component": "twilio_sms"})
        return False, str(e)

def to_arabic_numerals(num):
    """Convert Western numerals to Arabic numerals"""
    return str(num).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def highlight_metric(val, threshold, color="#fa709a"):
    """Create CSS style for highlighting metrics above threshold"""
    style = ""
    if val >= threshold:
        style = f"background:{color}22;border-radius:12px;padding:0.1em 0.4em;"
    return style

# Colorful palette for visualizations
colorful_palette = [
    "#43cea2", "#fa709a", "#ffb347", "#8fd3f4", "#185a9d",
    "#ffe259", "#ffa751", "#fdc830", "#eecda3", "#e0eafc", "#cfdef3", "#fe8c00", "#f83600"
]

# Translations (All sections, all labels, all solutions, all features, all about, complete)
texts = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "app_sub": "Intelligent Digital Plant Platform",
        "side_sections": [
            "Digital Twin", "Advanced Dashboard", "Predictive Analytics", "Scenario Playback",
            "Alerts & Fault Log", "Smart Solutions", "KPI Wall", "Plant Heatmap", "Root Cause Explorer", "AI Copilot Chat",
            "Live Plant 3D", "Incident Timeline", "Energy Optimization", "Future Insights", "Operator Feedback", "System Health", "About"
        ],
        "lang_en": "English",
        "lang_ar": "Arabic",
        "solution_btn": "Next Solution",
        "logo_alt": "Smart Neural Digital Twin Logo",
        "about_header": "Our Story",
        "about_story": "Our journey began with a simple question: <b>How can we detect gas leaks before they become disasters?</b> <span style=\"color:#fa709a;font-weight:bold\">We tried every solution, but realized we needed something smarter.</span> This digital twin platform was born from that need - combining AI, real-time sensing, and predictive analytics to create a nervous system for industrial plants. Now operators can see problems before they happen.",
        "about_colorful": [
            ("#43cea2", "AI at the Core"),
            ("#fa709a", "Real-time Sensing"),
            ("#ffb347", "Predictive Analytics"),
            ("#8fd3f4", "Instant Actions"),
            ("#185a9d", "Peace of Mind"),
            ("#ffe259", "Smart Monitoring"),
            ("#ffa751", "Safety First"),
        ],
        "features": [
            "Interactive plant schematic & overlays",
            "Advanced dashboards & KPIs",
            "AI-driven fault detection & smart solutions",
            "Root-cause explorer & scenario playback",
            "Live 3D plant visualization",
            "Bilingual support & vibrant design"
        ],
        "howto_extend": [
            "Connect to real plant historian data",
            "Add custom plant schematics & overlays",
            "Integrate with control and alerting systems",
            "Deploy on secure internal networks"
        ],
        "developers": [
            ("Rakan Almarri", "rakan.almarri.2@aramco.com", "0532559664"),
            ("Abdulrahman Alzahrani", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "Contact Info",
        "demo_note": "Demo use only: Not for live plant operation",
        "live3d_header": "Live Plant 3D",
        "live3d_intro": "Explore the interactive 3D model below. Use your mouse to zoom, rotate, and explore the plant!",
        "live3d_404": "The 3D model failed to load. View the static 3D plant image below.",
        "static_3d_caption": "Sample Plant 3D Visual",
        "ai_explain_btn": "Explain with AI",
        "ai_rootcause_btn": "Root Cause Analysis",
        "ai_whatif_btn": "What-if Scenario",
        "ai_kpi_btn": "Analyze KPIs",
        "ai_energy_btn": "Energy Optimization Advice",
        "ai_feedback_btn": "Summarize Feedback",
        "health_check": "System Health Check",
        "perf_check": "Performance Metrics",
        "app_version": "App Version",
        "last_updated": "Last Updated",
        "developer": "Lead Developer",
        "solutions": [
            {
                "title": "Automated Methane Leak Response",
                "desc": "Integrate advanced sensors with automated shutdown logic to instantly contain future methane leaks.",
                "steps": ["Deploy new IoT sensors", "Implement AI detection", "Link to emergency shutdown", "Train operators"],
                "priority": "High", "effectiveness": "High", "time": "3 days", "cost": "$4,000", "savings": "$25,000/year",
                "icon": "🛡️"
            },
            {
                "title": "Pump Predictive Maintenance",
                "desc": "Monitor vibration and temperature to predict pump failures before they occur.",
                "steps": ["Install vibration sensors", "Run ML models", "Alert on anomaly", "Schedule just-in-time maintenance"],
                "priority": "Medium", "effectiveness": "High", "time": "1 week", "cost": "$5,000", "savings": "$18,000/year",
                "icon": "🔧"
            },
            {
                "title": "Energy Use Optimization",
                "desc": "AI analyzes compressor schedule to cut energy waste by 11%.",
                "steps": ["Analyze compressor cycles", "Optimize schedule", "Implement load shifting", "Track savings"],
                "priority": "High", "effectiveness": "Medium", "time": "2 weeks", "cost": "$6,000", "savings": "$32,000/year",
                "icon": "⚡"
            },
        ]
    },
    "ar": {
        "app_title": "التوأم الرقمي العصبي الذكي",
        "app_sub": "منصة المصنع الذكي الرقمي",
        "side_sections": [
            "التوأم الرقمي", "لوحة القيادة المتقدمة", "التحليلات التنبؤية", "تشغيل السيناريو",
            "التنبيهات وسجل الأعطال", "الحلول الذكية", "جدار المؤشرات", "خريطة حرارة المصنع", "مستكشف السبب الجذري",
            "محادثة الذكاء الصناعي", "مصنع ثلاثي الأبعاد", "جدول الحوادث", "تحسين الطاقة", "رؤى مستقبلية", "ملاحظات المشغل", "صحة النظام", "حول"
        ],
        "lang_en": "الإنجليزية",
        "lang_ar": "العربية",
        "solution_btn": "الحل التالي",
        "logo_alt": "شعار التوأم الرقمي العصبي الذكي",
        "about_header": "قصتنا",
        "about_story": "بدأنا رحلتنا من سؤال بسيط: <b>كيف نكشف تسرب الغاز قبل أن يتحول إلى كارثة؟</b> <span style=\"color:#fa709a;font-weight:bold\">جربنا كل الحلول، لكن أدركنا أننا نحتاج شيئًا أكثر ذكاءً.</span> ولدت منصة التوأم الرقمي هذه من هذه الحاجة - تجمع بين الذكاء الاصطناعي والاستشعار في الوقت الحقيقي والتحليلات التنبؤية لإنشاء نظام عصبي للمصانع الصناعية. الآن يمكن للمشغلين رؤية المشكلات قبل حدوثها.",
        "about_colorful": [
            ("#43cea2", "الذكاء الاصطناعي في القلب"),
            ("#fa709a", "استشعار لحظي"),
            ("#ffb347", "تحليلات تنبؤية"),
            ("#8fd3f4", "إجراءات فورية"),
            ("#185a9d", "راحة البال"),
            ("#ffe259", "مراقبة ذكية"),
            ("#ffa751", "السلامة أولاً"),
        ],
        "features": [
            "مخطط مصنع تفاعلي وتراكب مباشر",
            "لوحات ومؤشرات متقدمة",
            "كشف أعطال ذكي وحلول فورية",
            "مستكشف السبب الجذري وتشغيل السيناريوهات",
            "رؤية ثلاثية الأبعاد للمصنع",
            "دعم لغتين وتصميم حيوي"
        ],
        "howto_extend": [
            "ربط مع بيانات المصنع الحقيقية",
            "إضافة مخططات وتراكب مخصص",
            "دمج مع أنظمة التحكم والتنبيه",
            "تشغيل داخلي آمن"
        ],
        "developers": [
            ("راكان المعاري", "rakan.almarri.2@aramco.com", "0532559664"),
            ("عبدالرحمن الزهراني", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
        "contact": "معلومات التواصل",
        "demo_note": "للعرض فقط: غير مخصص للتشغيل الفعلي",
        "live3d_header": "مصنع ثلاثي الأبعاد مباشر",
        "live3d_intro": "تفاعل مع النموذج الثلاثي الأبعاد أدناه. استخدم الماوس للتحريك والتكبير.",
        "live3d_404": "تعذر تحميل النموذج، شاهد صورة المصنع الثلاثي الأبعاد بالأسفل.",
        "static_3d_caption": "مشهد ثلاثي الأبعاد لمصنع صناعي",
        "ai_explain_btn": "شرح الذكاء الصناعي",
        "ai_rootcause_btn": "تحليل السبب الجذري",
        "ai_whatif_btn": "سيناريو افتراضي",
        "ai_kpi_btn": "تحليل مؤشرات الأداء",
        "ai_energy_btn": "توصيات توفير الطاقة",
        "ai_feedback_btn": "تلخيص الملاحظات",
        "health_check": "فحص صحة النظام",
        "perf_check": "مقاييس الأداء",
        "app_version": "إصدار التطبيق",
        "last_updated": "آخر تحديث",
        "developer": "المطور الرئيسي",
        "solutions": [
            {
                "title": "استجابة آلية لتسرب الميثان",
                "desc": "دمج حساسات متطورة مع منطق إيقاف تلقائي لاحتواء التسربات فوراً.",
                "steps": ["تركيب حساسات إنترنت الأشياء", "تفعيل كشف الذكاء الاصطناعي", "ربط بالإيقاف الطارئ", "تدريب المشغلين"],
                "priority": "عالية", "effectiveness": "عالية", "time": "٣ أيام", "cost": "$٤٬٠٠٠", "savings": "$٢٥٬٠٠٠/سنة",
                "icon": "🛡️"
            },
            {
                "title": "صيانة استباقية للمضخات",
                "desc": "مراقبة الاهتزازات والحرارة للتنبؤ بالأعطال قبل وقوعها.",
                "steps": ["تركيب حساسات الاهتزاز", "تشغيل نماذج التعلم الآلي", "تنبيه عند وجود شذوذ", "جدولة صيانة فورية"],
                "priority": "متوسطة", "effectiveness": "عالية", "time": "أسبوع", "cost": "$٥٬٠٠٠", "savings": "$١٨٬٠٠٠/سنة",
                "icon": "🔧"
            },
            {
                "title": "تحسين استهلاك الطاقة",
                "desc": "تحلل الذكاء الاصطناعي جدول الضواغط لخفض الهدر بنسبة ١١٪.",
                "steps": ["تحليل دورات الضواغط", "تحسين الجدولة", "تطبيق نقل الأحمال", "متابعة التوفير"],
                "priority": "عالية", "effectiveness": "متوسطة", "time": "أسبوعان", "cost": "$٦٬٠٠٠", "savings": "$٣٢٬٠٠٠/سنة",
                "icon": "⚡"
            },
        ]
    }
}

# ----- THEME & CSS -----
if st.sidebar.button("🌗 Theme", key="themebtn"):
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
    analytics.track_interaction("theme_toggle", {"new_theme": st.session_state["theme"]})

# Enhanced CSS with microinteractions and fixed rendering issues
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@700&family=Montserrat:wght@700&display=swap');
html, body, [class*="css"] {{
    background: {'#181a2a' if st.session_state["theme"] == "dark" else '#f6f6f7'} !important;
    color: {'#f9fcff' if st.session_state["theme"] == "dark" else '#232526'} !important;
    font-family: 'Montserrat', 'Cairo', sans-serif !important;
}}
.peak-card {{
    background: linear-gradient(135deg, #e0eafc 0%, #ffe259 100%);
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,.18);
    margin-bottom: 1.5em;
    padding: 1.5em 2em;
    animation: peakfade 0.8s;
    border-left: 8px solid #43cea2;
    transition: box-shadow 0.21s, transform 0.18s;
}}
.peak-card:hover {{
    box-shadow: 0 12px 38px 0 #fa709a55;
    transform: scale(1.018);
    border-left: 8px solid #fa709a;
}}
.kpi-card {{
    background: linear-gradient(135deg, #43cea2 0%, #fa709a 82%, #ffe259 100%);
    border-radius: 13px;
    color: #fff !important;
    font-size: 1.25em;
    font-weight: 700;
    box-shadow: 0 8px 24px 0 rgba(31,38,135,.10);
    padding: 1.3em 1.3em;
    text-align: center;
    margin-bottom: 1em;
    transition: box-shadow 0.18s, transform 0.16s;
    animation: peakfade 0.7s;
}}
.kpi-card:hover {{
    box-shadow: 0 8px 36px 0 #ffe25977;
    transform: scale(1.025);
}}
.sidebar-title {{
    font-size: 2em !important;
    font-weight: 900 !important;
    color: #43cea2 !important;
    letter-spacing: 0.5px;
    margin-bottom: 0.2em !important;
    text-shadow: 0 3px 10px #185a9d22;
}}
.sidebar-subtitle {{
    font-size: 1.15em !important;
    color: #fa709a !important;
    margin-bottom: 1em;
    margin-top: -.7em !important;
    text-shadow: 0 1px 6px #ffb34744;
}}
.gradient-header {{
    font-weight: 900;
    font-size: 2.1em;
    background: linear-gradient(90deg,#43cea2,#fa709a 60%,#ffe259 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3em;
    letter-spacing: .5px;
    text-shadow: 0 1px 6px #185a9d1c;
}}
.timeline-step {{
    border-left: 4px solid #43cea2;
    margin-left: 0.8em;
    padding-left: 1.2em;
    margin-bottom: 1em;
    position: relative;
    animation: peakfade 0.7s;
}}
.timeline-step:before {{
    content: '';
    position: absolute;
    left: -14px;
    top: 0.18em;
    width: 18px;
    height: 18px;
    background: #fa709a;
    border-radius: 100%;
    border: 2px solid #fff;
    box-shadow: 0 0 0 3px #ffe25933;
}}
.timeline-icon {{
    font-size: 1.5em;
    margin-right: 0.5em;
    vertical-align: middle;
}}
.about-bgcard {{
    background: linear-gradient(140deg,#43cea210,#fa709a10 60%,#ffe25910 100%);
    border-radius: 22px;
    padding: 2.2em 2.1em 1.8em 2.1em;
    margin-top: 1.6em;
    margin-bottom: 2.2em;
    box-shadow: 0 7px 32px 0 #43cea233;
    position: relative;
    animation: peakfade 0.9s;
}}
.about-story {{
    font-size: 1.18em;
    font-weight: 600;
    margin-bottom: 2em;
    color: {'#fff' if st.session_state["theme"] == "dark" else '#222'};
    line-height: 1.65em;
}}
.about-feature {{
    font-weight: 700;
    font-size: 1.16em;
    margin: .45em 0 .14em 0;
}}
.about-color {{
    font-weight: 900;
    font-size: 1.20em;
    margin-bottom: .45em;
    display: inline-block;
    padding: .18em .9em;
    border-radius: 12px;
    margin-right: .9em;
    margin-bottom: .5em;
    color: #232526;
    background: #fff;
    box-shadow: 0 2px 8px #185a9d22;
    border: 2px solid #43cea2;
}}
.about-color:nth-child(2) {{border-color: #fa709a;}}
.about-color:nth-child(3) {{border-color: #ffb347;}}
.about-color:nth-child(4) {{border-color: #8fd3f4;}}
.about-color:nth-child(5) {{border-color: #185a9d;}}
.about-color:nth-child(6) {{border-color: #ffe259;}}
.about-color:nth-child(7) {{border-color: #ffa751;}}
.feedback-bubble {{
    background: #43cea222;
    border-radius: 12px;
    padding: 0.8em 1.1em;
    margin-bottom: 0.7em;
    box-shadow: 0 2px 10px #43cea207;
}}
@keyframes peakfade {{
    0% {{ opacity: 0; transform: translateY(40px);}}
    100% {{ opacity: 1; transform: translateY(0);}}
}}
/* Microinteractions */
.hover-float:hover {{ transform: translateY(-3px); transition: transform 0.3s ease; }}
.click-pulse {{ animation: pulse 0.3s ease-in-out; }}

@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
    100% {{ transform: scale(1); }}
}}

/* Apply to elements */
.peak-card, .kpi-card {{ transition: transform 0.3s ease; }}
.stButton > button {{ transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; }}
.stButton > button:active {{ transform: scale(0.95) !important; }}

/* Mobile responsive adjustments */
@media (max-width: 768px) {{
    .kpi-card, .peak-card {{
        width: 100% !important;
        margin-right: 0 !important;
    }}
    
    .sidebar-title {{
        font-size: 1.5em !important;
    }}
    
    .gradient-header {{
        font-size: 1.8em !important;
    }}
}}

/* Proper error messages */
.error-container {{
    background: rgba(250,112,154,0.1);
    border-left: 4px solid #fa709a;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
    color: #fa709a;
}}

/* Enhance readability */
.stMarkdown, .stText {{
    font-family: 'Montserrat', 'Cairo', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}}

/* Better buttons */
.stButton > button {{
    background: linear-gradient(90deg,#43cea2,#185a9d) !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.5em 1em !important;
    border: none !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.15) !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
}}

/* Loader animations */
.stSpinner {{
    border: 4px solid #43cea2 !important;
    border-left-color: transparent !important;
    animation: spin 1s linear infinite !important;
}}

@keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}

/* Badge styling */
.badge {{
    display: inline-block;
    padding: 0.25em 0.6em;
    font-size: 0.85em;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 10rem;
    color: white;
    background-color: #43cea2;
    margin-right: 0.5em;
}}

.badge-primary {{
    background-color: #43cea2;
}}

.badge-warning {{
    background-color: #ffb347;
}}

.badge-danger {{
    background-color: #fa709a;
}}

.badge-info {{
    background-color: #8fd3f4;
}}

/* Enhanced tooltips */
.tooltip {{
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted #ccc;
    cursor: help;
}}

.tooltip .tooltip-text {{
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}}

.tooltip:hover .tooltip-text {{
    visibility: visible;
    opacity: 1;
}}

/* Status indicators */
.status-indicator {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}}

.status-online {{
    background-color: #43cea2;
}}

.status-warning {{
    background-color: #ffb347;
}}

.status-offline {{
    background-color: #fa709a;
}}

/* System stats cards */
.system-stat-card {{
    background: linear-gradient(135deg, rgba(24, 90, 157, 0.1), rgba(67, 206, 162, 0.1));
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    border-left: 4px solid #43cea2;
}}

/* Table enhancements */
.stDataFrame table {{
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}

.stDataFrame th {{
    background: linear-gradient(90deg, #43cea2, #185a9d) !important;
    color: white !important;
    font-weight: 600 !important;
}}

.stDataFrame tr:nth-child(even) {{
    background-color: rgba(67, 206, 162, 0.05) !important;
}}

.stDataFrame tr:hover {{
    background-color: rgba(67, 206, 162, 0.1) !important;
}}

/* Chat message styling */
.chat-message {{
    display: flex;
    margin-bottom: 10px;
}}

.chat-message.user {{
    justify-content: flex-end;
}}

.chat-message .message {{
    max-width: 80%;
    padding: 10px 15px;
    border-radius: 20px;
}}

.chat-message.user .message {{
    background-color: #43cea2;
    color: white;
    border-bottom-right-radius: 5px;
}}

.chat-message.ai .message {{
    background-color: #f0f0f0;
    color: #333;
    border-bottom-left-radius: 5px;
}}
</style>

<script>
// Add click animations to all buttons
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.add('click-pulse');
            setTimeout(() => this.classList.remove('click-pulse'), 300);
        });
    });

# Fix for the CSS JavaScript section - close the tags properly
st.markdown(f"""
<style>
...existing CSS...
</style>

<script>
// Add click animations to all buttons
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.add('click-pulse');
            setTimeout(() => this.classList.remove('click-pulse'), 300);
        });
    });
});
</script>
""", unsafe_allow_html=True)

# Add keyboard shortcuts
if CONFIG["ui"]["animation_enabled"]:
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Alt + D: Toggle dark mode
        if (e.altKey && e.key === 'd') {
            const themeBtn = document.querySelector('button[key="themebtn"]');
            if (themeBtn) themeBtn.click();
        }
        
        // Alt + number keys: Navigate sections
        if (e.altKey && !isNaN(parseInt(e.key)) && parseInt(e.key) > 0) {
            const sections = document.querySelectorAll('div[role="radio"]');
            const index = parseInt(e.key) - 1;
            if (index < sections.length) {
                sections[index].click();
            }
        }
        
        // Alt + S: Search
        if (e.altKey && e.key === 's') {
            const searchBox = document.querySelector('input[type="text"]');
            if (searchBox) {
                searchBox.focus();
                e.preventDefault();
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)
