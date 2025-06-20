import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium

# --- CSV Data Loader ---
@st.cache_data(show_spinner=False)
def load_sensor_csv():
    url = "https://raw.githubusercontent.com/rrakanmarri1/Smart-neural-digital-twin/master/sensor_data_simulated.csv"
    df = pd.read_csv(url)
    return df

# Load data into session state at startup
if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = load_sensor_csv()

from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
import time
from datetime import datetime, timedelta

# --- App Configuration ---
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Full Translations ---
translations = {
    "en": {
        "title": "Smart Neural Digital Twin",
        "nav_dashboard": "Dashboard", "nav_predictive": "Predictive Analysis", "nav_solutions": "Smart Solutions",
        "nav_locations": "Sensor Locations", "nav_alerts": "Real-time Alerts", "nav_sim_controls": "Simulation Controls",
        "nav_financial": "Financial Impact", "nav_anomaly": "AI Anomaly Log", "nav_3d": "3D Asset View",
        "nav_maintenance": "Maintenance Scheduler", "nav_energy": "Energy Dashboard", "nav_about": "About & Team", "nav_settings": "Settings",
        "language": "Language", "theme": "Theme", "auto_refresh": "Enable Auto-Refresh (15s)",
        "live_data_header": "Live Sensor Data", "temp": "Temperature", "pressure": "Pressure", "vibration": "Vibration", "methane": "Methane",
        "historical_data_header": "Historical Data (Last 7 Days)", "predictive_header": "72-Hour Predictive Forecast",
        "historical_vs_forecast": "Historical vs. Forecasted Data", "historical": "Historical", "forecasted": "Forecasted",
        "solutions_header": "Generated Smart Solution", "generate_solutions_button": "Generate New Recommendation",
        "task_title": "Task", "description": "Description", "priority": "Priority", "est_time": "Est. Time", "effectiveness": "Effectiveness",
        "high": "High", "medium": "Medium", "low": "Low", "schedule_maintenance": "Schedule Maintenance", "task_scheduled": "Task Scheduled!",
        "locations_header": "Sensor Locations Map", "alerts_header": "Real-time Alerts Dashboard", "alert_level": "Alert Level", "timestamp": "Timestamp",
        "about_header": "About the Project",
        "about_text": """
### 🚀 Revolutionizing Oilfield Operations!\n
Welcome to the command center for the next generation of energy production. The **Smart Neural Digital Twin** isn't just a tool; it's your new strategic partner. We turn complex data into decisive, money-saving action.\n
---\n
### ✨ **Key Features**\n
*   **🧠 Predictive AI:** We see the future! Our AI forecasts equipment health 72 hours in advance.\n
*   **💸 Financial Impact:** See your savings in real-time. We connect operational improvements directly to your bottom line.\n
*   **🔬 AI Anomaly Log:** Catches the sneakiest issues before they become catastrophic failures.\n
*   **🕹️ Simulation Controls:** Safely stress-test your assets under any condition you can imagine.\n
*   **⚡ Energy Dashboard:** Monitor and optimize energy consumption for greener, more efficient operations.\n
### ✅ **How It Helps You Win**\n
*   **📈 Maximize Uptime:** Proactive maintenance means less downtime and more production.\n
*   **🛡️ Enhance Safety:** Predict and prevent hazardous conditions before they happen.\n
*   **💰 Boost Profitability:** Lower maintenance costs and higher efficiency equals more profit.\n
*   **🎯 Data-Driven Decisions:** Stop guessing. Start making informed decisions backed by powerful AI insights.\n
""",
        "tech_stack": "Technology Stack", "team_header": "Main Developers", "contact": "Contact",
        "rakan_almarri": "Rakan Almarri", "abdulrahman_alzhrani": "Abdulrahman Alzhrani",
        "sim_controls_header": "Simulation Controls", "temp_baseline": "Temperature Baseline (°C)", "pressure_volatility": "Pressure Volatility (hPa)", "inject_anomaly": "Inject Temperature Anomaly",
        "financial_header": "Financial Impact Analysis", "cost_savings_desc": "Estimated cost savings based on proactively addressing generated solutions.",
        "prevented_failure_cost": "Prevented Failure Cost", "reduced_downtime_cost": "Reduced Downtime Cost", "total_savings": "Total Estimated Savings",
        "anomaly_log_header": "AI Anomaly Log", "anomaly_desc": "Subtle deviations detected by the AI that are being monitored.",
        "view_3d_header": "Interactive 3D Asset View", "maintenance_header": "Maintenance Scheduler", "pending_tasks": "Pending Tasks",
        "completed_tasks": "Completed Tasks", "mark_complete": "Mark as Complete", "energy_header": "Energy Consumption Dashboard",
        "live_consumption": "Live Energy Consumption (kWh)", "efficiency_rating": "Operational Efficiency"
    },
    "ar": {
        "title": "التوأم الرقمي العصبي الذكي",
        "nav_dashboard": "لوحة التحكم", "nav_predictive": "التحليل التنبؤي", "nav_solutions": "الحلول الذكية",
        "nav_locations": "مواقع الحساسات", "nav_alerts": "التنبيهات الفورية", "nav_sim_controls": "أدوات التحكم بالمحاكاة",
        "nav_financial": "الأثر المالي", "nav_anomaly": "سجل الشذوذ", "nav_3d": "عرض ثلاثي الأبعاد",
        "nav_maintenance": "جدولة الصيانة", "nav_energy": "لوحة الطاقة", "nav_about": "عن المشروع والفريق", "nav_settings": "الإعدادات",
        "language": "اللغة", "theme": "المظهر", "auto_refresh": "تفعيل التحديث التلقائي (15 ثانية)",
        "live_data_header": "بيانات الحساسات الحية", "temp": "درجة الحرارة", "pressure": "الضغط", "vibration": "الاهتزاز", "methane": "الميثان",
        "historical_data_header": "البيانات التاريخية (آخر 7 أيام)", "predictive_header": "توقعات تنبؤية لـ 72 ساعة",
        "historical_vs_forecast": "البيانات التاريخية مقابل المتوقعة", "historical": "تاريخي", "forecasted": "متوقع",
        "solutions_header": "الحل الذكي المقترح", "generate_solutions_button": "توليد توصية جديدة",
        "task_title": "المهمة", "description": "الوصف", "priority": "الأولوية", "est_time": "الوقت المقدر", "effectiveness": "الفعالية",
        "high": "عالية", "medium": "متوسطة", "low": "منخفضة", "schedule_maintenance": "جدولة الصيانة", "task_scheduled": "تمت جدولة المهمة!",
        "locations_header": "خريطة مواقع أجهزة الاستشعار", "alerts_header": "لوحة التنبيهات الفورية", "alert_level": "مستوى التنبيه", "timestamp": "الطابع الزمني",
        "about_header": "عن المشروع",
        "about_text": """
### 🚀 ثورة في عمليات حقول النفط!\n
أهلاً بك في مركز قيادة الجيل القادم من إنتاج الطاقة. **التوأم الرقمي العصبي الذكي** ليس مجرد أداة، بل هو شريكك الاستراتيجي الجديد. نحن نحوّل البيانات المعقدة إلى إجراءات حاسمة وموفرة للمال.\n
---\n
### ✨ **الميزات الرئيسية**\n
*   **🧠 ذكاء اصطناعي تنبؤي:** نحن نرى المستقبل! يتنبأ الذكاء الاصطناعي لدينا بصحة المعدات قبل 72 ساعة.\n
*   **💸 الأثر المالي:** شاهد مدخراتك في الوقت الفعلي. نربط التحسينات التشغيلية مباشرة بأرباحك.\n
*   **🔬 سجل شذوذ الذكاء الاصطناعي:** يكتشف المشاكل الخفية قبل أن تتحول إلى كوارث.\n
*   **🕹️ أدوات تحكم بالمحاكاة:** اختبر أصولك بأمان تحت أي ظرف يمكنك تخيله.\n
*   **⚡ لوحة استهلاك الطاقة:** راقب وحسّن استهلاك الطاقة لعمليات أكثر استدامة وكفاءة.\n
### ✅ **كيف يساعدك على التفوق**\n
*   **📈 زيادة وقت التشغيل:** صيانة استباقية تعني توقفًا أقل وإنتاجًا أكثر.\n
*   **🛡️ تعزيز السلامة:** تنبأ بالظروف الخطرة وامنعها قبل وقوعها.\n
*   **💰 زيادة الربحية:** تكاليف صيانة أقل وكفاءة أعلى تساوي أرباحًا أكثر.\n
*   **🎯 قرارات تعتمد على البيانات:** توقف عن التخمين. ابدأ في اتخاذ قرارات مستنيرة مدعومة برؤى الذكاء الاصطناعي القوية.\n
""",
        "tech_stack": "التقنيات المستخدمة", "team_header": "المطورون الرئيسيون", "contact": "للتواصل",
        "rakan_almarri": "راكان المري", "abdulrahman_alzhrani": "عبدالرحمن الزهراني",
        "sim_controls_header": "أدوات التحكم بالمحاكاة", "temp_baseline": "خط الأساس لدرجة الحرارة (مئوية)", "pressure_volatility": "تقلب الضغط (هكتوباسكال)", "inject_anomaly": "إدخال شذوذ في درجة الحرارة",
        "financial_header": "تحليل الأثر المالي", "cost_savings_desc": "تقدير وفورات التكاليف بناءً على المعالجة الاستباقية للحلول المقترحة.",
        "prevented_failure_cost": "تكلفة الفشل الذي تم منعه", "reduced_downtime_cost": "تكلفة تقليل وقت التوقف", "total_savings": "إجمالي الوفورات المقدرة",
        "anomaly_log_header": "سجل شذوذ الذكاء الاصطناعي", "anomaly_desc": "الانحرافات الطفيفة التي كشفها الذكاء الاصطناعي والتي يتم مراقبتها.",
        "view_3d_header": "عرض تفاعلي ثلاثي الأبعاد للأصل", "maintenance_header": "جدولة الصيانة", "pending_tasks": "المهام المعلقة",
        "completed_tasks": "المهام المكتملة", "mark_complete": "وضع علامة كمكتمل", "energy_header": "لوحة استهلاك الطاقة",
        "live_consumption": "استهلاك الطاقة المباشر (كيلوواط/ساعة)", "efficiency_rating": "كفاءة التشغيل"
    }
}

# --- Themes ---
themes = {
    'purple': {'primary': '#9B59B6', 'secondary': '#8E44AD', 'text': '#FFFFFF', 'bg': '#2C3E50'},
    'ocean': {'primary': '#3498DB', 'secondary': '#2980B9', 'text': '#FFFFFF', 'bg': '#1A5276'},
    'sunset': {'primary': '#E67E22', 'secondary': '#D35400', 'text': '#FFFFFF', 'bg': '#34495E'},
    'forest': {'primary': '#2ECC71', 'secondary': '#27AE60', 'text': '#FFFFFF', 'bg': '#1E8449'},
    'dark': {'primary': '#34495E', 'secondary': '#2C3E50', 'text': '#FFFFFF', 'bg': '#17202A'},
}

# --- Session State Initialization ---
def init_session_state():
    if 'lang' not in st.session_state: st.session_state.lang = 'en'
    if 'theme' not in st.session_state: st.session_state.theme = 'purple'
    if 'page' not in st.session_state: st.session_state.page = 'nav_dashboard'
    if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
    if 'solution' not in st.session_state: st.session_state.solution = None
    if 'maintenance_log' not in st.session_state: st.session_state.maintenance_log = []
    if 'anomaly_log' not in st.session_state: st.session_state.anomaly_log = pd.DataFrame(columns=["Timestamp", "Description"])
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = {'temp_baseline': 30.0, 'pressure_volatility': 10.0, 'anomaly_injected': False, 'last_anomaly_time': None}

init_session_state()

# --- Helper Functions ---
def get_text(key):
    return translations[st.session_state.lang].get(key, key)

def get_custom_css(theme_name):
    theme = themes[theme_name]
    return f"""<style>
    .stApp {{ background-color: {theme['bg']}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {theme['primary']}; }}
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3, .st-emotion-cache-1y4p8pa, .st-emotion-cache-1d3w5bk, .st-emotion-cache-10trblm, .st-emotion-cache-1r6slb0, .st-emotion-cache-1kyxreq, .st-emotion-cache-1xarl3l {{ color: {theme['text']}; }}
    .st-emotion-cache-6q9sum, .st-emotion-cache-13ln4jf, .stMetric {{ background-color: {theme['secondary']}; border-radius: 10px; padding: 15px; text-align: center; }}
    .stButton>button {{ background-color: {theme['primary']}; color: {theme['text']}; border: 2px solid {theme['secondary']}; border-radius: 8px; }}
    .stButton>button:hover {{ background-color: {theme['secondary']}; color: {theme['text']}; }}
    </style>"""
st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

# --- Data Simulation & Logic ---
def simulate_live_data():
    params = st.session_state.sim_params
    temp_anomaly = 0
    if params['anomaly_injected']:
        params['anomaly_injected'] = False
        params['last_anomaly_time'] = time.time()
        new_anomaly = pd.DataFrame([{'Timestamp': datetime.now(), 'Description': 'Manually Injected Temperature Anomaly'}])
        st.session_state.anomaly_log = pd.concat([new_anomaly, st.session_state.anomaly_log], ignore_index=True)
    if params['last_anomaly_time'] and (time.time() - params['last_anomaly_time'] < 300):
        temp_anomaly = 15 * ((300 - (time.time() - params['last_anomaly_time'])) / 300)
    temp = np.random.uniform(params['temp_baseline'] - 5, params['temp_baseline'] + 5) + np.sin(time.time() / 60) * 5 + temp_anomaly
    pressure = np.random.uniform(1000, 1020) + np.cos(time.time() / 60) * params['pressure_volatility']
    vibration = np.random.uniform(0.1, 0.5) + np.random.rand() * 0.1
    methane = np.random.uniform(1.8, 2.2) + np.random.rand() * 0.05
    energy = (temp/10) + (pressure/1000) + (vibration*5) + (methane*2) + np.random.uniform(-1,1)
    return temp, pressure, vibration, methane, energy

def get_historical_data(days=7):
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'))
    return pd.DataFrame({'Date': dates, 'Temperature': np.random.normal(30, 3, days), 'Pressure': np.random.normal(1010, 5, days)})

def get_future_forecast(hist_df, days=3):
    future_dates = pd.to_datetime(pd.date_range(start=hist_df['Date'].iloc[-1], periods=days+1, freq='D'))[1:]
    last_temp = hist_df['Temperature'].iloc[-1]
    last_pressure = hist_df['Pressure'].iloc[-1]
    return pd.DataFrame({'Date': future_dates, 'Temperature_Forecast': np.random.normal(last_temp, 2, days), 'Pressure_Forecast': np.random.normal(last_pressure, 4, days)})

def generate_smart_solution():
    solutions = [
        {"title": "Optimize Cooling System", "desc": "Adjust coolant flow rate based on predictive temperature analysis to prevent overheating.", "priority": "high", "time": "2 hours", "effectiveness": 95, "cost_saving": 5000},
        {"title": "Inspect Pipeline Section 3B", "desc": "Vibration levels are trending upwards. A physical inspection is recommended within 48 hours.", "priority": "medium", "time": "4 hours", "effectiveness": 80, "cost_saving": 2500},
        {"title": "Calibrate Methane Sensor #7", "desc": "Sensor readings show minor drift. Calibration will ensure data accuracy.", "priority": "low", "time": "1 hour", "effectiveness": 90, "cost_saving": 700},
        {"title": "Review Pressure Valve Protocols", "desc": "Forecasted pressure spikes require a review of automated pressure release valve settings.", "priority": "high", "time": "3 hours", "effectiveness": 92, "cost_saving": 6200}
    ]
    current_titles_in_log = [task['task'] for task in st.session_state.maintenance_log]
    available_solutions = [s for s in solutions if s['title'] not in current_titles_in_log]
    if not available_solutions: available_solutions = solutions # Fallback if all are scheduled
    st.session_state.solution = np.random.choice(available_solutions)

def get_real_time_alerts():
    return pd.DataFrame([
        {"level": "High", "desc": "Critical pressure warning in Sector A! Immediate action required.", "time": datetime.now() - timedelta(minutes=5)},
        {"level": 'Medium', "desc": "Unusual vibration pattern detected in Pump Station 2.", "time": datetime.now() - timedelta(minutes=25)},
        {"level": 'Low', "desc": "Temperature approaching upper threshold in cooling unit 4.", "time": datetime.now() - timedelta(hours=1)}
    ])

# --- Page Rendering Functions ---
def render_main_dashboard():
    if st.session_state.auto_refresh: st_autorefresh(interval=15 * 1000, key="data_refresh")
    st.header(get_text("live_data_header"))
    temp, pressure, vibration, methane, _ = simulate_live_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(get_text("temp"), f"{temp:.2f} °C", f"{np.random.uniform(-0.5, 0.5):.2f}")
    col2.metric(get_text("pressure"), f"{pressure:.2f} hPa", f"{np.random.uniform(-1, 1):.2f}")
    col3.metric(get_text("vibration"), f"{vibration:.3f} g", f"{np.random.uniform(-0.01, 0.01):.3f}")
    col4.metric(get_text("methane"), f"{methane:.2f} ppm", f"{np.random.uniform(-0.02, 0.02):.2f}")

def render_predictive_analysis():
    st.header(get_text("predictive_header"))
    hist_data = get_historical_data()
    forecast_data = get_future_forecast(hist_data)
    for metric, unit in [('Temperature', '°C'), ('Pressure', 'hPa')]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data[metric], name=get_text('historical'), mode='lines'))
        fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data[f"{metric}_Forecast"], name=get_text('forecasted'), mode='lines+markers', line=dict(dash='dash')))
        fig.update_layout(title=f"{get_text('historical_vs_forecast')}: {get_text(metric.lower())}", xaxis_title="Date", yaxis_title=f"{metric} ({unit})")
        st.plotly_chart(fig, use_container_width=True)

def render_smart_solutions():
    st.header(get_text("solutions_header"))
    if st.button(get_text("generate_solutions_button"), key="generate_solutions_button") or not st.session_state.solution:
        generate_smart_solution()
    sol = st.session_state.solution
    if sol:
        st.markdown(f"""<div style='border: 2px solid {themes[st.session_state.theme]['primary']}; border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
            <h3 style='color: {themes[st.session_state.theme]['primary']};'><b>{get_text('task_title')}: {sol['title']}</b></h3>
            <p><b>{get_text('description')}:</b> {sol['desc']}</p>
            <p><b>{get_text('priority')}:</b> {get_text(sol['priority'])} | <b>{get_text('est_time')}:</b> {sol['time']} | <b>{get_text('effectiveness')}:</b> {sol['effectiveness']}%</p>
        </div>""", unsafe_allow_html=True)
        if st.button(get_text("schedule_maintenance"), key=sol['title']):
            st.session_state.maintenance_log.insert(0, {"task": sol['title'], "status": "Pending", "scheduled_at": datetime.now(), "id": time.time()})
            st.session_state.solution = None # Clear after scheduling
            st.success(get_text("task_scheduled"))
            st.rerun()

def render_sensor_locations():
    st.header(get_text("locations_header"))
    df = st.session_state['sensor_data']
    # Simulate sensor IDs and assign random locations (for demo)
    np.random.seed(42)
    sensor_ids = [f"Sensor {i+1}" for i in range(6)]
    coords = {
        sid: [24 + np.random.uniform(-3, 3), 46 + np.random.uniform(-3, 3)]
        for sid in sensor_ids
    }
    # Assign each row a sensor id (round robin for demo)
    df['sensor_id'] = [sensor_ids[i % len(sensor_ids)] for i in range(len(df))]
    latest = df.groupby('sensor_id').last().reset_index()
    m = folium.Map(location=[24.7136, 46.6753], zoom_start=5)
    for _, row in latest.iterrows():
        sid = row['sensor_id']
        lat, lon = coords[sid]
        temp = row['temperature']
        popup = f"<b>{sid}</b><br>Temp: {temp:.1f}°C<br>Pressure: {row['pressure']:.1f} hPa<br>Vibration: {row['vibration']:.2f}<br>Methane: {row['methane']:.2f} ppm"
        color = 'red' if temp > 60 else 'orange' if temp > 45 else 'green'
        folium.Marker(
            [lat, lon],
            popup=popup,
            tooltip=f"{sid} (Click for details)",
            icon=folium.Icon(color=color)
        ).add_to(m)
    st_folium(m, width=725, height=500)
    st.caption("🗺️ Click markers for live sensor status. Locations are simulated for demo purposes.")


def render_real_time_alerts():
    st.header(get_text("alerts_header"))
    df = st.session_state['sensor_data']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    alerts = []
    # Define thresholds for alerting (customize as needed)
    for _, row in df.tail(100).iterrows():
        if row['temperature'] > 60:
            alerts.append({'level': 'High', 'desc': f"High temperature: {row['temperature']:.1f}°C", 'timestamp': row['timestamp']})
        elif row['pressure'] > 110:
            alerts.append({'level': 'High', 'desc': f"High pressure: {row['pressure']:.1f} hPa", 'timestamp': row['timestamp']})
        elif row['vibration'] > 0.7:
            alerts.append({'level': 'Medium', 'desc': f"Elevated vibration: {row['vibration']:.2f}", 'timestamp': row['timestamp']})
        elif row['methane'] > 2.5:
            alerts.append({'level': 'Medium', 'desc': f"Methane spike: {row['methane']:.2f} ppm", 'timestamp': row['timestamp']})
    if alerts:
        for alert in alerts[-10:][::-1]:
            color = st.error if alert['level'] == 'High' else st.warning if alert['level'] == 'Medium' else st.info
            color(f"**{alert['level']}**: {alert['desc']} ({alert['timestamp'].strftime('%Y-%m-%d %H:%M')})")
        st.dataframe(pd.DataFrame(alerts[-25:][::-1]), use_container_width=True)
    else:
        st.success("No critical alerts in the latest data! All systems normal.")
    st.info("⚡ Stay vigilant! Real-time alerts help prevent costly incidents.")


def render_simulation_controls():
    st.header(get_text("sim_controls_header"))
    st.session_state.sim_params['temp_baseline'] = st.slider(get_text("temp_baseline"), 10.0, 50.0, st.session_state.sim_params['temp_baseline'])
    st.session_state.sim_params['pressure_volatility'] = st.slider(get_text("pressure_volatility"), 5.0, 25.0, st.session_state.sim_params['pressure_volatility'])
    if st.button(get_text("inject_anomaly")):
        st.session_state.sim_params['anomaly_injected'] = True
        st.rerun()

def render_anomaly_log():
    st.header(get_text("anomaly_log_header"))
    st.info(get_text("anomaly_desc"))
    df = st.session_state['sensor_data']
    anomalies = []
    for _, row in df.iterrows():
        if row['temperature'] > 60:
            anomalies.append({"Sensor": row.get('sensor_id', 'N/A'), "Timestamp": row['timestamp'], "Description": f"🔥 High temperature: {row['temperature']:.1f}°C"})
        if row['pressure'] > 110:
            anomalies.append({"Sensor": row.get('sensor_id', 'N/A'), "Timestamp": row['timestamp'], "Description": f"⚠️ High pressure: {row['pressure']:.1f} hPa"})
        if row['vibration'] > 0.7:
            anomalies.append({"Sensor": row.get('sensor_id', 'N/A'), "Timestamp": row['timestamp'], "Description": f"🟠 Elevated vibration: {row['vibration']:.2f}"})
        if row['methane'] > 2.5:
            anomalies.append({"Sensor": row.get('sensor_id', 'N/A'), "Timestamp": row['timestamp'], "Description": f"💨 Methane spike: {row['methane']:.2f} ppm"})
    if anomalies:
        df_anom = pd.DataFrame(anomalies).sort_values("Timestamp", ascending=False).head(50)
        st.dataframe(df_anom, use_container_width=True)
    else:
        st.success("No anomalies detected in the current data!")


def render_3d_view():
    st.header(get_text("view_3d_header"))
    temp, _, _, _, _ = simulate_live_data()
    color = 'red' if temp > st.session_state.sim_params['temp_baseline'] + 10 else 'yellow' if temp > st.session_state.sim_params['temp_baseline'] + 5 else 'green'
    fig = go.Figure(data=[go.Scatter3d(x=[1], y=[1], z=[1], mode='markers', marker=dict(size=30, color=color, opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(xaxis_title='X-Axis', yaxis_title='Y-Axis', zaxis_title='Z-Axis'))
    st.plotly_chart(fig, use_container_width=True)

def render_maintenance_scheduler():
    st.header(get_text("maintenance_header"))
    st.subheader(get_text("pending_tasks"))
    for task in [t for t in st.session_state.maintenance_log if t['status'] == 'Pending']:
        with st.container():
            st.write(f"{task['task']} (Scheduled: {task['scheduled_at'].strftime('%Y-%m-%d %H:%M')})")
            if st.button(get_text("mark_complete"), key=f"complete_{task['id']}"):
                task['status'] = 'Completed'
                st.rerun()
    st.subheader(get_text("completed_tasks"))
    for task in [t for t in st.session_state.maintenance_log if t['status'] == 'Completed']:
        st.success(f"✓ {task['task']}")

def render_energy_dashboard():
    st.header(get_text("energy_header"))
    df = st.session_state['sensor_data']
    # Simulate energy as sum of all sensor readings (or use a real column if present)
    if 'energy' in df.columns:
        df['energy_kwh'] = df['energy']
    else:
        # Example: sum of temperature, pressure, vibration, methane for demo purposes
        df['energy_kwh'] = df[['temperature', 'pressure', 'vibration', 'methane']].sum(axis=1)

    # Show live gauge (latest value)
    latest_energy = df['energy_kwh'].iloc[-1]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_energy,
        title={'text': get_text("live_consumption")},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Show historical energy trend
    st.subheader("Energy Consumption (Last 24h)")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_24h = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(hours=24))]
        st.line_chart(df_24h.set_index('timestamp')['energy_kwh'])
    else:
        st.line_chart(df['energy_kwh'])

    # Show summary stats
    st.markdown(f"**Average (24h):** {df['energy_kwh'].tail(24).mean():.2f} kWh | **Max:** {df['energy_kwh'].max():.2f} kWh | **Min:** {df['energy_kwh'].min():.2f} kWh")
    st.info("💡 Tip: Monitor energy spikes to optimize efficiency and reduce costs!")


def render_about_page():
    st.header(get_text("about_header"))
    st.markdown(get_text("about_text"))
    st.subheader(get_text("tech_stack"))
    st.markdown("- Streamlit, Pandas, Plotly, Folium, Numpy")
    st.header(get_text("team_header"))
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**{get_text('rakan_almarri')}**\n* {get_text('contact')}: rakan.almarri.2@aramco.com | 0532559664")
    with col2:
        st.info(f"**{get_text('abdulrahman_alzhrani')}**\n* {get_text('contact')}: abdulrahman.alzhrani.1@aramco.com | 0549202574")

def render_settings():
    st.header(get_text("nav_settings"))
    lang_map = {"English": "en", "العربية": "ar"}
    selected_lang_name = st.selectbox(get_text("language"), options=list(lang_map.keys()), index=list(lang_map.values()).index(st.session_state.lang))
    if lang_map[selected_lang_name] != st.session_state.lang:
        st.session_state.lang = lang_map[selected_lang_name]
        st.rerun()
    theme_name = st.selectbox(get_text("theme"), options=list(themes.keys()), index=list(themes.keys()).index(st.session_state.theme))
    if theme_name != st.session_state.theme:
        st.session_state.theme = theme_name
        st.rerun()
    if st.toggle(get_text("auto_refresh"), value=st.session_state.auto_refresh):
        if not st.session_state.auto_refresh: 
            st.session_state.auto_refresh = True
            st.rerun()
    else:
        if st.session_state.auto_refresh: 
            st.session_state.auto_refresh = False
            st.rerun()

def render_financial_impact():
    st.header(get_text("financial_header"))
    st.info(get_text("cost_savings_desc"))
    # You can add charts, tables, or summary stats here as needed
    
# --- Main App Layout & Dispatcher ---
st.sidebar.title(get_text("title"))
page_options = {
    "nav_dashboard": ("🏠", get_text("nav_dashboard"), render_main_dashboard),
    "nav_predictive": ("📈", get_text("nav_predictive"), render_predictive_analysis),
    "nav_solutions": ("💡", get_text("nav_solutions"), render_smart_solutions),
    "nav_energy": ("⚡", get_text("nav_energy"), render_energy_dashboard),
    "nav_financial": ("💰", get_text("nav_financial"), render_financial_impact),
    "nav_maintenance": ("🛠️", get_text("nav_maintenance"), render_maintenance_scheduler),
    "nav_3d": ("🧊", get_text("nav_3d"), render_3d_view),
    "nav_locations": ("🗺️", get_text("nav_locations"), render_sensor_locations),
    "nav_alerts": ("🔔", get_text("nav_alerts"), render_real_time_alerts),
    "nav_anomaly": ("🔬", get_text("nav_anomaly"), render_anomaly_log),
    "nav_sim_controls": ("🎛️", get_text("nav_sim_controls"), render_simulation_controls),
    "nav_about": ("ℹ️", get_text("nav_about"), render_about_page),
    "nav_settings": ("⚙️", get_text("nav_settings"), render_settings),
}
solutions_data = [
        {"title": "Optimize Cooling System", "desc": "Adjust coolant flow rate based on predictive temperature analysis to prevent overheating.", "priority": "high", "time": "2 hours", "effectiveness": 95, "cost_saving": 5000},
        {"title": "Inspect Pipeline Section 3B", "desc": "Vibration levels are trending upwards. A physical inspection is recommended within 48 hours.", "priority": "medium", "time": "4 hours", "effectiveness": 80, "cost_saving": 2500},
        {"title": "Calibrate Methane Sensor #7", "desc": "Sensor readings show minor drift. Calibration will ensure data accuracy.", "priority": "low", "time": "1 hour", "effectiveness": 90, "cost_saving": 700},
        {"title": "Review Pressure Valve Protocols", "desc": "Forecasted pressure spikes require a review of automated pressure release valve settings.", "priority": "high", "time": "3 hours", "effectiveness": 92, "cost_saving": 6200}
    ]
for key, (icon, text, func) in page_options.items():
    if st.sidebar.button(f"{icon} {text}"):
        st.session_state.page = key
        st.rerun()

page_options[st.session_state.page][2]()

# --- Themes ---
themes = {
    'purple': {'primary': '#9B59B6', 'secondary': '#8E44AD', 'text': '#FFFFFF', 'bg': '#2C3E50'},
    'ocean': {'primary': '#3498DB', 'secondary': '#2980B9', 'text': '#FFFFFF', 'bg': '#1A5276'},
    'sunset': {'primary': '#E67E22', 'secondary': '#D35400', 'text': '#FFFFFF', 'bg': '#34495E'},
    'forest': {'primary': '#2ECC71', 'secondary': '#27AE60', 'text': '#FFFFFF', 'bg': '#1E8449'},
    'dark': {'primary': '#34495E', 'secondary': '#2C3E50', 'text': '#FFFFFF', 'bg': '#17202A'},
}

# --- Session State Initialization ---
def init_session_state():
    # Core state
    if 'lang' not in st.session_state: st.session_state.lang = 'en'
    if 'theme' not in st.session_state: st.session_state.theme = 'purple'
    if 'page' not in st.session_state: st.session_state.page = 'nav_dashboard'
    if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
    
    # Feature-specific state
    if 'solutions' not in st.session_state: st.session_state.solutions = []
    if 'maintenance_log' not in st.session_state: st.session_state.maintenance_log = []
    if 'anomaly_log' not in st.session_state: st.session_state.anomaly_log = []
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = {'temp_baseline': 30.0, 'pressure_volatility': 10.0, 'anomaly_injected': False}

init_session_state()

# --- Helper Functions ---
def get_text(key):
    return translations[st.session_state.lang].get(key, key)

def get_custom_css(theme_name):
    theme = themes[theme_name]
    # (CSS remains largely the same, can be collapsed for brevity)
    return f"""<style> ... </style>"""
st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

# --- Data Simulation & Logic ---
def simulate_live_data():
    params = st.session_state.sim_params
    base_temp = params['temp_baseline']
    temp_anomaly = 15 if params['anomaly_injected'] else 0
    temp = np.random.uniform(base_temp - 5, base_temp + 5) + np.sin(time.time() / 60) * 5 + temp_anomaly

    base_pressure_vol = params['pressure_volatility']
    pressure = np.random.uniform(1000, 1020) + np.cos(time.time() / 60) * base_pressure_vol
    
    vibration = np.random.uniform(0.1, 0.5) + np.random.rand() * 0.1
    methane = np.random.uniform(1.8, 2.2) + np.random.rand() * 0.05

    if params['anomaly_injected']:
        st.session_state.anomaly_log.insert(0, {'desc': 'Manually Injected Temperature Anomaly', 'time': datetime.now()})
        params['anomaly_injected'] = False # Reset after injection

    return temp, pressure, vibration, methane

def generate_smart_solutions():
    solutions = [
        {"title": "Optimize Cooling System", "desc": "Adjust coolant flow rate based on predictive temperature analysis to prevent overheating.", "priority": "high", "time": "2 hours", "effectiveness": 95, "cost_saving": 5000},
        {"title": "Inspect Pipeline Section 3B", "desc": "Vibration levels are trending upwards. A physical inspection is recommended within 48 hours.", "priority": "medium", "time": "4 hours", "effectiveness": 80, "cost_saving": 2500},
        {"title": "Calibrate Methane Sensor #7", "desc": "Sensor readings show minor drift. Calibration will ensure data accuracy.", "priority": "low", "time": "1 hour", "effectiveness": 90, "cost_saving": 700},
        {"title": "Review Pressure Valve Protocols", "desc": "Forecasted pressure spikes require a review of automated pressure release valve settings.", "priority": "high", "time": "3 hours", "effectiveness": 92, "cost_saving": 6200}
    ]
    st.session_state.solutions = [np.random.choice(solutions)] # Generate one solution

# (Other data functions like get_historical_data, get_future_forecast, get_real_time_alerts remain similar)

# --- Page Rendering Functions ---

def render_main_dashboard():
    # (Largely the same, but uses updated simulation)
    pass

def render_predictive_analysis():
    # (No major changes)
    pass

def render_smart_solutions():
    st.header(get_text("solutions_header"))
    if st.button(get_text("generate_solutions_button")) or not st.session_state.solutions:
        generate_smart_solutions()

    for sol in st.session_state.solutions:
        # (Display solution details)
        if st.button(get_text("schedule_maintenance"), key=sol['title']):
            st.session_state.maintenance_log.insert(0, {"task": sol['title'], "status": "Pending", "scheduled_at": datetime.now()})
            st.success(get_text("task_scheduled"))

def render_sensor_locations():
    # (No major changes)
    pass

def render_real_time_alerts():
    # (No major changes)
    pass

def render_simulation_controls():
    st.header(get_text("sim_controls_header"))
    st.session_state.sim_params['temp_baseline'] = st.slider(get_text("temp_baseline"), 10.0, 50.0, st.session_state.sim_params['temp_baseline'])
    st.session_state.sim_params['pressure_volatility'] = st.slider(get_text("pressure_volatility"), 5.0, 25.0, st.session_state.sim_params['pressure_volatility'])
    if st.button(get_text("inject_anomaly")):
        st.session_state.sim_params['anomaly_injected'] = True
        st.rerun()

def render_financial_impact():
    st.header(get_text("financial_header"))
    # (Display financial charts and metrics based on solutions)
    pass

def render_anomaly_log():
    st.header(get_text("anomaly_log_header"))
    st.write(get_text("anomaly_desc"))
    st.dataframe(pd.DataFrame(st.session_state.anomaly_log), use_container_width=True)

def render_3d_view():
    st.header(get_text("view_3d_header"))
    # (Use Plotly to render a 3D scatter plot with color-coded points)
    pass

def render_maintenance_scheduler():
    st.header(get_text("maintenance_header"))
    # (Display pending/completed tasks from st.session_state.maintenance_log)
    pass

def render_energy_dashboard():
    st.header(get_text("energy_header"))
    # (Display energy consumption charts)
    pass

def render_about_page():
    st.header(get_text("about_header"))
    st.markdown(get_text("about_text"))
    st.header(get_text("team_header"))
    # (Display developer contacts)

def render_settings():
    st.header(get_text("nav_settings"))
    # (Language, Theme, and Auto-Refresh controls moved here)
    pass

# --- Main App Layout ---
st.sidebar.title(get_text("title"))

page_options = {
    "nav_dashboard": ("🏠", get_text("nav_dashboard"), render_main_dashboard),
    "nav_predictive": ("📈", get_text("nav_predictive"), render_predictive_analysis),
    "nav_solutions": ("💡", get_text("nav_solutions"), render_smart_solutions),
    "nav_energy": ("⚡", get_text("nav_energy"), render_energy_dashboard),
    "nav_financial": ("💰", get_text("nav_financial"), render_financial_impact),
    "nav_maintenance": ("🛠️", get_text("nav_maintenance"), render_maintenance_scheduler),
    "nav_3d": ("🧊", get_text("nav_3d"), render_3d_view),
    "nav_locations": ("🗺️", get_text("nav_locations"), render_sensor_locations),
    "nav_alerts": ("🔔", get_text("nav_alerts"), render_real_time_alerts),
    "nav_anomaly": ("🔬", get_text("nav_anomaly"), render_anomaly_log),
    "nav_sim_controls": ("🎛️", get_text("nav_sim_controls"), render_simulation_controls),
    "nav_about": ("ℹ️", get_text("nav_about"), render_about_page),
    "nav_settings": ("⚙️", get_text("nav_settings"), render_settings),
}

# Create lists for the radio widget
page_keys = list(page_options.keys())
formatted_options = [f"{icon} {text}" for icon, text, func in page_options.values()]

# Find the index of the current page to set as default
try:
    current_index = page_keys.index(st.session_state.get('page', 'nav_dashboard'))
except ValueError:
    current_index = 0 # Default to dashboard if key not found

# Create the radio button for navigation
selected_option_formatted = st.sidebar.radio(
    "Navigation",
    formatted_options,
    index=current_index,
    label_visibility="collapsed" # Hides the "Navigation" label
)

# Get the key corresponding to the selected formatted option and update state
selected_index = formatted_options.index(selected_option_formatted)
st.session_state.page = page_keys[selected_index]

# --- Page Dispatcher ---
# The page has already been set in session_state by the radio button's interaction
page_options[st.session_state.page][2]()
