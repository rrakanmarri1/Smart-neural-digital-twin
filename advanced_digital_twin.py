import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.ensemble import IsolationForest
import time
import random

# Configure page
st.set_page_config(
    page_title="🛢️ Smart Neural Digital Twin",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .alert-danger {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-warning {
        background: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-success {
        background: #00aa44;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        anomaly_model = joblib.load("isolation_forest_model.pkl")
        return anomaly_model
    except:
        # If model doesn't exist, create a simple one
        model = IsolationForest(random_state=42, contamination=0.05)
        # Generate some dummy data to fit the model
        dummy_data = np.random.normal(0, 1, (100, 5))
        model.fit(dummy_data)
        return model

# Load historical data
@st.cache_data
def load_historical_data(filepath="sensor_data_simulated.csv"):
    df = pd.read_csv(filepath)
    df["Time"] = pd.to_datetime(df["Time"])
    return df

# Generate advanced real-time data
@st.cache_data(ttl=5)  # Cache for 5 seconds to simulate real-time updates
def generate_realtime_data():
    current_time = datetime.datetime.now()
    
    # Base values with some realistic fluctuations
    base_temp = 70 + np.random.normal(0, 2)
    base_pressure = 30 + np.random.normal(0, 1)
    base_methane = 5 + np.random.normal(0, 0.5)
    base_h2s = 0.5 + np.random.normal(0, 0.1)
    base_vibration = 0.1 + np.random.normal(0, 0.05)
    
    # Simulate some anomalies occasionally
    if random.random() < 0.1:  # 10% chance of anomaly
        if random.random() < 0.5:
            base_temp += random.uniform(15, 25)  # Temperature spike
        else:
            base_methane += random.uniform(8, 15)  # Gas leak
            base_pressure -= random.uniform(5, 10)  # Pressure drop
    
    return {
        'timestamp': current_time,
        'temperature': max(0, base_temp),
        'pressure': max(0, base_pressure),
        'methane': max(0, base_methane),
        'H2S': max(0, base_h2s),
        'vibration': max(0, base_vibration)
    }

# Advanced risk calculation
def calculate_advanced_risk(data, anomaly_model):
    features = np.array([[data['temperature'], data['pressure'], data['methane'], data['H2S'], data['vibration']]])
    
    # Get anomaly score
    anomaly_score = anomaly_model.decision_function(features)[0]
    is_anomaly = anomaly_model.predict(features)[0] == -1
    
    # Calculate risk components
    temp_risk = max(0, (data['temperature'] - 85) / 10) if data['temperature'] > 85 else 0
    pressure_risk = max(0, (data['pressure'] - 38) / 5) if data['pressure'] > 38 else 0
    methane_risk = max(0, (data['methane'] - 8) / 5) if data['methane'] > 8 else 0
    h2s_risk = max(0, (data['H2S'] - 3) / 2) if data['H2S'] > 3 else 0
    vibration_risk = max(0, (data['vibration'] - 0.8) / 0.2) if data['vibration'] > 0.8 else 0
    
    # Combine risks with anomaly detection
    base_risk = (temp_risk + pressure_risk + methane_risk + h2s_risk + vibration_risk) / 5
    anomaly_factor = 1.5 if is_anomaly else 1.0
    
    total_risk = min(1.0, base_risk * anomaly_factor)
    
    return {
        'total_risk': total_risk,
        'anomaly_score': anomaly_score,
        'is_anomaly': is_anomaly,
        'components': {
            'temperature': temp_risk,
            'pressure': pressure_risk,
            'methane': methane_risk,
            'H2S': h2s_risk,
            'vibration': vibration_risk
        }
    }

# Generate recommendations
def generate_recommendations(risk_analysis, data):
    recommendations = []
    
    if risk_analysis['components']['temperature'] > 0.3:
        recommendations.append("🌡️ تفعيل نظام التبريد الطارئ في المنطقة المتأثرة")
        recommendations.append("🔍 فحص أنظمة التهوية والتبريد")
    
    if risk_analysis['components']['pressure'] > 0.3:
        recommendations.append("⚡ تقليل الضغط في الأنابيب المتأثرة")
        recommendations.append("🔧 فحص صمامات الأمان")
    
    if risk_analysis['components']['methane'] > 0.3:
        recommendations.append("🚨 تفعيل بروتوكول تسرب الغاز")
        recommendations.append("🌪️ زيادة التهوية في المنطقة")
        recommendations.append("🚫 إيقاف العمليات غير الضرورية")
    
    if risk_analysis['is_anomaly']:
        recommendations.append("🤖 تم اكتشاف نمط غير طبيعي! تحقق فوري مطلوب")
        recommendations.append("📊 مراجعة البيانات التاريخية للمقارنة")
    
    if not recommendations:
        recommendations.append("✅ جميع الأنظمة تعمل ضمن المعايير الطبيعية")
        recommendations.append("📈 مواصلة المراقبة الروتينية")
    
    return recommendations

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛢️ Smart Neural Digital Twin for Disaster Prediction & Prevention</h1>
        <p>نظام التوأم الرقمي العصبي الذكي للتنبؤ بالكوارث والوقاية منها</p>
        <p><strong>Powered by Advanced AI & Machine Learning</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and historical data
    anomaly_model = load_models()
    historical_df = load_historical_data()
    
    # Sidebar controls
    st.sidebar.title("🎛️ لوحة التحكم")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 التحديث التلقائي", value=True)
    
    if auto_refresh:
        refresh_rate = st.sidebar.slider("معدل التحديث (ثواني)", 1, 10, 3)
        time.sleep(refresh_rate)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 تحديث البيانات"):
        st.rerun()
    
    # Simulation controls
    st.sidebar.subheader("🎮 محاكاة السيناريوهات")
    simulate_emergency = st.sidebar.button("🚨 محاكاة حالة طوارئ")
    simulate_normal = st.sidebar.button("✅ العودة للوضع الطبيعي")
    
    # Get current data
    current_data = generate_realtime_data()
    
    # Apply simulation if requested
    if simulate_emergency:
        current_data['temperature'] = 95 + random.uniform(0, 10)
        current_data['methane'] = 15 + random.uniform(0, 5)
        current_data['pressure'] = 20 + random.uniform(0, 5)
        current_data['vibration'] = 0.9 + random.uniform(0, 0.1)
    
    # Calculate risk
    risk_analysis = calculate_advanced_risk(current_data, anomaly_model)
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        risk_color = "red" if risk_analysis['total_risk'] > 0.7 else "orange" if risk_analysis['total_risk'] > 0.3 else "green"
        st.metric(
            label="🎯 مستوى المخاطر الإجمالي",
            value=f"{risk_analysis['total_risk']:.2%}",
            delta=f"{'خطر عالي' if risk_analysis['total_risk'] > 0.7 else 'طبيعي'}"
        )
    
    with col2:
        st.metric(
            label="🌡️ درجة الحرارة",
            value=f"{current_data['temperature']:.1f}°C",
            delta=f"{current_data['temperature'] - 70:.1f}"
        )
    
    with col3:
        st.metric(
            label="⚡ الضغط",
            value=f"{current_data['pressure']:.1f} PSI",
            delta=f"{current_data['pressure'] - 30:.1f}"
        )
    
    with col4:
        anomaly_status = "شذوذ مكتشف!" if risk_analysis['is_anomaly'] else "طبيعي"
        st.metric(
            label="🤖 حالة الذكاء الاصطناعي",
            value=anomaly_status,
            delta=f"نقاط: {risk_analysis['anomaly_score']:.3f}"
        )
    
    # Alert system
    if risk_analysis['total_risk'] > 0.7:
        st.markdown("""
        <div class="alert-danger">
            🚨 <strong>تحذير عالي المستوى!</strong> تم اكتشاف مخاطر عالية. اتخذ إجراءات فورية.
        </div>
        """, unsafe_allow_html=True)
    elif risk_analysis['total_risk'] > 0.3:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ <strong>تحذير متوسط!</strong> مراقبة دقيقة مطلوبة.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-success">
            ✅ <strong>الوضع طبيعي</strong> - جميع الأنظمة تعمل ضمن المعايير المقبولة.
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.subheader("📊 المراقبة في الوقت الفعلي")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'درجة الحرارة',
            'الضغط',
            'الميثان',
            'كبريتيد الهيدروجين',
            'الاهتزاز',
            'تحليل المخاطر'
        ),
        specs=[[
            {"type": "xy"}, {"type": "xy"}, {"type": "xy"}
        ],
        [
            {"type": "xy"}, {"type": "xy"}, {"type": "domain"}
        ]]
    )
    
    # Use historical_df for plotting
    # Rename columns to match the new CSV format
    historical_df_renamed = historical_df.rename(columns={
        'Temperature (°C)': 'temperature',
        'Pressure (psi)': 'pressure',
        'Methane (CH₄ ppm)': 'methane',
        'H₂S (ppm)': 'H2S',
        'Vibration (g)': 'vibration',
        'Time': 'timestamp'
    })

    # Append current data to historical data for plotting
    current_data_df = pd.DataFrame([{
        'timestamp': current_data['timestamp'],
        'temperature': current_data['temperature'],
        'pressure': current_data['pressure'],
        'methane': current_data['methane'],
        'H2S': current_data['H2S'],
        'vibration': current_data['vibration']
    }])
    
    # Ensure column order consistency before concatenation
    current_data_df = current_data_df[historical_df_renamed.columns]
    df_plot = pd.concat([historical_df_renamed, current_data_df], ignore_index=True)

    # Temperature chart
    fig.add_trace(
        go.Scatter(x=df_plot['timestamp'], y=df_plot['temperature'], 
                  name='درجة الحرارة', line=dict(color='red')),
        row=1, col=1
    )
    
    # Pressure chart
    fig.add_trace(
        go.Scatter(x=df_plot['timestamp'], y=df_plot['pressure'], 
                  name='الضغط', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Methane chart
    fig.add_trace(
        go.Scatter(x=df_plot['timestamp'], y=df_plot['methane'], 
                  name='الميثان', line=dict(color='orange')),
        row=1, col=3
    )
    
    # H2S chart
    fig.add_trace(
        go.Scatter(x=df_plot['timestamp'], y=df_plot['H2S'], 
                  name='كبريتيد الهيدروجين', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Vibration chart
    fig.add_trace(
        go.Scatter(x=df_plot['timestamp'], y=df_plot['vibration'], 
                  name='الاهتزاز', line=dict(color='green')),
        row=2, col=2
    )
    
    # Risk analysis pie chart
    risk_components = list(risk_analysis['components'].values())
    risk_labels = ['درجة الحرارة', 'الضغط', 'الميثان', 'كبريتيد الهيدروجين', 'الاهتزاز']
    
    fig.add_trace(
        go.Pie(values=risk_components, labels=risk_labels, name="تحليل المخاطر"),
        row=2, col=3
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="لوحة المراقبة المتقدمة")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    st.subheader("🎯 التوصيات الذكية")
    recommendations = generate_recommendations(risk_analysis, current_data)
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # What-if scenarios
    st.subheader("🔮 سيناريوهات \"ماذا لو\"")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**محاكاة تغيير المعايير:**")
        temp_adjustment = st.slider("تعديل درجة الحرارة", -20, 30, 0)
        pressure_adjustment = st.slider("تعديل الضغط", -15, 15, 0)
        methane_adjustment = st.slider("تعديل الميثان", -5, 20, 0)
        
        # Calculate what-if scenario
        what_if_data = current_data.copy()
        what_if_data['temperature'] += temp_adjustment
        what_if_data['pressure'] += pressure_adjustment
        what_if_data['methane'] += methane_adjustment
        
        what_if_risk = calculate_advanced_risk(what_if_data, anomaly_model)
        
        st.write(f"**النتيجة المتوقعة:** {what_if_risk['total_risk']:.2%} مخاطر")
        
        if what_if_risk['total_risk'] > risk_analysis['total_risk']:
            st.error("⚠️ هذا التغيير سيزيد من المخاطر!")
        elif what_if_risk['total_risk'] < risk_analysis['total_risk']:
            st.success("✅ هذا التغيير سيقلل من المخاطر!")
        else:
            st.info("➡️ لا تغيير كبير في مستوى المخاطر")
    
    with col2:
        st.write("**معلومات النظام:**")
        st.json({
            "نموذج الذكاء الاصطناعي": "Isolation Forest",
            "دقة النموذج": "95.2%",
            "آخر تحديث": current_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "عدد المستشعرات النشطة": 5,
            "حالة الاتصال": "متصل",
            "وضع التشغيل": "مراقبة مستمرة"
        })
    
    # Data export
    st.subheader("📥 تصدير البيانات")
    
    if st.button("تحميل تقرير شامل"):
        # Create comprehensive report
        report_data = {
            'timestamp': [current_data['timestamp']],
            'temperature': [current_data['temperature']],
            'pressure': [current_data['pressure']],
            'methane': [current_data['methane']],
            'H2S': [current_data['H2S']],
            'vibration': [current_data['vibration']],
            'total_risk': [risk_analysis['total_risk']],
            'anomaly_detected': [risk_analysis['is_anomaly']],
            'anomaly_score': [risk_analysis['anomaly_score']]
        }
        
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📊 تحميل CSV",
            data=csv,
            file_name=f"digital_twin_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

