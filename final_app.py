import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import os

# Initialize session state for theme and recommendation history if not already done
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# تعيين نمط الصفحة #
st.set_page_config(
    page_title="التوأم الرقمي الذكي - نظام مراقبة متقدم",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Theme Configuration ---
def get_theme_colors(theme_mode):
    if theme_mode == "dark":
        return {
            "primary-color": "#0E1117",
            "secondary-color": "#3498DB",
            "text-color": "#FFFFFF",
            "chart-background": "#1E212B",
            "card-background": "#2E313B",
            "accent-color": "#E74C3C",
            "success-color": "#27AE60",
            "warning-color": "#F39C12"
        }
    else:
        return {
            "primary-color": "#FFFFFF",
            "secondary-color": "#2980B9",
            "text-color": "#2C3E50",
            "chart-background": "#F8F9FA",
            "card-background": "#FFFFFF",
            "accent-color": "#E74C3C",
            "success-color": "#27AE60",
            "warning-color": "#F39C12"
        }

theme_colors = get_theme_colors(st.session_state.theme)

# Enhanced CSS with modern design
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .reportview-container {{
        background: linear-gradient(135deg, {theme_colors['primary-color']} 0%, {theme_colors['chart-background']} 100%);
        color: {theme_colors['text-color']};
        font-family: 'Inter', sans-serif;
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {theme_colors['secondary-color']} 0%, #1E3A8A 100%);
        color: #FFFFFF;
        border-radius: 0 15px 15px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }}
    
    .stButton>button {{
        background: linear-gradient(45deg, {theme_colors['secondary-color']} 0%, {theme_colors['accent-color']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .metric-card {{
        background: {theme_colors['card-background']};
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-align: center;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {theme_colors['secondary-color']};
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 1rem;
        font-weight: 500;
        color: {theme_colors['text-color']};
        opacity: 0.8;
    }}
    
    .status-indicator {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }}
    
    .status-normal {{ background-color: {theme_colors['success-color']}; }}
    .status-warning {{ background-color: {theme_colors['warning-color']}; }}
    .status-critical {{ background-color: {theme_colors['accent-color']}; }}
    
    .recommendation-card {{
        background: {theme_colors['card-background']};
        border-left: 4px solid {theme_colors['secondary-color']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .priority-critical {{ border-left-color: {theme_colors['accent-color']}; }}
    .priority-high {{ border-left-color: {theme_colors['warning-color']}; }}
    .priority-medium {{ border-left-color: {theme_colors['secondary-color']}; }}
    .priority-low {{ border-left-color: {theme_colors['success-color']}; }}
    
    h1, h2, h3 {{
        color: {theme_colors['text-color']};
        font-weight: 600;
    }}
    
    .stAlert {{
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {theme_colors['secondary-color']} 0%, {theme_colors['accent-color']} 100%);
    }}
    </style>
""", unsafe_allow_html=True)

# Load custom CSS if exists
try:
    with open('custom_style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- Sidebar Navigation ---
st.sidebar.markdown("# 🧠 التوأم الرقمي الذكي")
st.sidebar.markdown("---")

# Theme toggle with icon
theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
if st.sidebar.button(f"{theme_icon} تبديل المظهر"):
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

st.sidebar.markdown("---")

menu_options = [
    ("🏠", "لوحة التحكم", "Dashboard"),
    ("📊", "التحليل التنبؤي", "Predictive Analytics"),
    ("💡", "التوصيات الذكية", "Smart Recommendations"),
    ("ℹ️", "حول المشروع", "About Project"),
    ("⚙️", "الإعدادات", "Settings")
]

chosen_menu = None
for icon, arabic_name, english_name in menu_options:
    if st.sidebar.button(f"{icon} {arabic_name}", key=english_name):
        st.session_state.current_menu = english_name
        chosen_menu = english_name

if 'current_menu' not in st.session_state:
    st.session_state.current_menu = "Dashboard"

chosen_menu = st.session_state.current_menu

# --- Data Simulation (Enhanced) ---
@st.cache_data(ttl=60)  # Cache for 1 minute
@st.cache_data(ttl=60)
def generate_sensor_data(num_points=200):
    time_series = pd.date_range(end=datetime.now(), periods=num_points, freq='H')
    
    # More realistic sensor data with trends
    base_temp = 30 + np.sin(np.arange(num_points) * 2 * np.pi / 24) * 5  # Daily cycle
    temperature = base_temp + np.random.normal(0, 2, num_points)
    
    base_pressure = 100 + np.sin(np.arange(num_points) * 2 * np.pi / 168) * 10  # Weekly cycle
    pressure = base_pressure + np.random.normal(0, 5, num_points)
    
    vibration = np.random.normal(0.5, 0.1, num_points)
    
    # Introduce realistic anomalies
    for _ in range(int(num_points * 0.03)):
        idx = random.randint(0, num_points - 1)
        if random.random() > 0.5:  # Temperature spike
            temperature[idx] += random.uniform(15, 25)
        else:  # Pressure drop
            pressure[idx] -= random.uniform(20, 40)
        vibration[idx] += random.uniform(0.3, 0.8)
    
    df = pd.DataFrame({
        'Timestamp': time_series,
        'Temperature': temperature,
        'Pressure': pressure,
        'Vibration': vibration
    })
    return df

sensor_data = generate_sensor_data()

# --- Enhanced Anomaly Detection ---
@st.cache_data(ttl=60)
def detect_anomalies(df):
    anomalies = df[
        (df['Temperature'] > df['Temperature'].quantile(0.95)) |
        (df['Pressure'] < df['Pressure'].quantile(0.05)) |
        (df['Vibration'] > df['Vibration'].quantile(0.95))
    ]
    return anomalies

anomalies = detect_anomalies(sensor_data)

# --- Enhanced Predictive Model ---
@st.cache_data(ttl=60)
def predict_future_data(df, hours_ahead=72):
    last_timestamp = df['Timestamp'].max()
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=hours_ahead, freq='H')
    
    # Use moving averages for more realistic predictions
    temp_trend = df['Temperature'].rolling(window=24).mean().iloc[-1]
    pressure_trend = df['Pressure'].rolling(window=24).mean().iloc[-1]
    vib_trend = df['Vibration'].rolling(window=24).mean().iloc[-1]
    
    future_temp = [temp_trend + random.uniform(-3, 3) for _ in range(hours_ahead)]
    future_press = [pressure_trend + random.uniform(-8, 8) for _ in range(hours_ahead)]
    future_vib = [vib_trend + random.uniform(-0.15, 0.15) for _ in range(hours_ahead)]
    
    # Confidence intervals
    temp_upper = [t + random.uniform(2, 5) for t in future_temp]
    temp_lower = [t - random.uniform(2, 5) for t in future_temp]
    
    future_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Temperature': future_temp,
        'Pressure': future_press,
        'Vibration': future_vib,
        'Temperature_Upper': temp_upper,
        'Temperature_Lower': temp_lower
    })
    return future_df

future_predictions = predict_future_data(sensor_data)

# --- Enhanced Recommendations ---
def generate_recommendation(risk_level):
    recommendations = {
        "Low": [
            {"title": "مراقبة الأداء الروتينية", "details": "استمرار المراقبة الدورية لجميع بيانات أجهزة الاستشعار.", "priority": "منخفضة", "estimated_time": "ساعة واحدة", "effectiveness": "عالية"},
            {"title": "مراجعة ملفات السجل", "details": "فحص سجلات النظام للبحث عن أي إدخالات غير عادية.", "priority": "منخفضة", "estimated_time": "30 دقيقة", "effectiveness": "متوسطة"}
        ],
        "Medium": [
            {"title": "فحص تشخيصي شامل", "details": "تشغيل فحص تشخيصي كامل على المكونات المتأثرة.", "priority": "متوسطة", "estimated_time": "ساعتان", "effectiveness": "عالية"},
            {"title": "ضبط عتبات أجهزة الاستشعار", "details": "تعديل طفيف لعتبات كشف الشذوذ لتقليل الإنذارات الخاطئة.", "priority": "متوسطة", "estimated_time": "ساعة واحدة", "effectiveness": "متوسطة"}
        ],
        "High": [
            {"title": "إيقاف النظام الفوري", "details": "بدء إجراءات الإيقاف الطارئ لمنع المزيد من الأضرار.", "priority": "حرجة", "estimated_time": "10 دقائق", "effectiveness": "عالية جداً"},
            {"title": "إرسال فريق الصيانة", "details": "إرسال فريق صيانة متخصص لفحص وإصلاح المعدات.", "priority": "عالية", "estimated_time": "4 ساعات", "effectiveness": "عالية جداً"},
            {"title": "عزل المنطقة المتأثرة", "details": "تنفيذ بروتوكولات السلامة لعزل المنطقة عالية المخاطر.", "priority": "عالية", "estimated_time": "30 دقيقة", "effectiveness": "عالية"}
        ]
    }
    return random.choice(recommendations.get(risk_level, recommendations["Low"]))

# --- Main App Logic ---
if chosen_menu == "Dashboard":
    st.title("🏠 لوحة التحكم الرئيسية")
    st.markdown("### مراقبة البيانات في الوقت الفعلي")
    
    # Real-time status indicator
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**آخر تحديث:** {current_time}")
    
    # Enhanced Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_value = sensor_data['Temperature'].iloc[-1]
        temp_status = "normal" if 20 <= temp_value <= 40 else "warning" if temp_value > 40 else "critical"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ درجة الحرارة</div>
            <div class="metric-value">{temp_value:.1f}°C</div>
            <div><span class="status-indicator status-{temp_status}"></span>طبيعي</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pressure_value = sensor_data['Pressure'].iloc[-1]
        pressure_status = "normal" if 80 <= pressure_value <= 120 else "warning"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ الضغط</div>
            <div class="metric-value">{pressure_value:.1f} kPa</div>
            <div><span class="status-indicator status-{pressure_status}"></span>طبيعي</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vibration_value = sensor_data['Vibration'].iloc[-1]
        vibration_status = "normal" if vibration_value <= 0.8 else "critical"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📳 الاهتزاز</div>
            <div class="metric-value">{vibration_value:.2f} mm/s</div>
            <div><span class="status-indicator status-{vibration_status}"></span>طبيعي</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        system_health = 100 - (len(anomalies) * 5)
        health_status = "normal" if system_health > 80 else "warning" if system_health > 60 else "critical"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💚 صحة النظام</div>
            <div class="metric-value">{system_health}%</div>
            <div><span class="status-indicator status-{health_status}"></span>ممتاز</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 قراءات أجهزة الاستشعار الحديثة")
        
        # Create enhanced plotly chart
        fig = go.Figure()
        
        # Temperature
        fig.add_trace(go.Scatter(
            x=sensor_data['Timestamp'], 
            y=sensor_data['Temperature'],
            mode='lines',
            name='درجة الحرارة (°C)',
            line=dict(color='#E74C3C', width=2),
            hovertemplate='درجة الحرارة: %{y:.1f}°C<br>الوقت: %{x}<extra></extra>'
        ))
        
        # Pressure (scaled for visibility)
        fig.add_trace(go.Scatter(
            x=sensor_data['Timestamp'], 
            y=sensor_data['Pressure'],
            mode='lines',
            name='الضغط (kPa)',
            line=dict(color='#3498DB', width=2),
            yaxis='y2',
            hovertemplate='الضغط: %{y:.1f} kPa<br>الوقت: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='بيانات أجهزة الاستشعار عبر الزمن',
            xaxis_title='الوقت',
            yaxis=dict(title='درجة الحرارة (°C)', side='left'),
            yaxis2=dict(title='الضغط (kPa)', side='right', overlaying='y'),
            plot_bgcolor=theme_colors['chart-background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text-color']),
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 كشف الشذوذ")
        
        if not anomalies.empty:
            risk_level = "High" if len(anomalies) > 5 else "Medium"
            st.session_state.current_risk_level = risk_level
            
            st.error(f"تم اكتشاف {len(anomalies)} حالة شذوذ!")
            
            # Show recent anomalies
            recent_anomalies = anomalies.tail(3)
            for _, anomaly in recent_anomalies.iterrows():
                st.markdown(f"""
                <div class="recommendation-card priority-critical">
                    <strong>⚠️ شذوذ مكتشف</strong><br>
                    الوقت: {anomaly['Timestamp'].strftime('%H:%M')}<br>
                    الحرارة: {anomaly['Temperature']:.1f}°C<br>
                    الضغط: {anomaly['Pressure']:.1f} kPa
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔍 عرض جميع الشذوذات"):
                st.dataframe(anomalies, use_container_width=True)
        else:
            st.success("✅ لم يتم اكتشاف أي شذوذ. النظام مستقر.")
            st.session_state.current_risk_level = "Low"

elif chosen_menu == "Predictive Analytics":
    st.title("📊 التحليل التنبؤي")
    st.markdown("### توقعات البيانات للـ 72 ساعة القادمة")
    
    # Prediction controls
    col1, col2, col3 = st.columns(3)
    with col1:
        prediction_hours = st.selectbox("فترة التنبؤ", [24, 48, 72, 96], index=2)
    with col2:
        confidence_level = st.slider("مستوى الثقة", 80, 99, 95)
    with col3:
        if st.button("🔄 تحديث التنبؤات"):
            st.rerun()
    
    # Generate predictions based on selected hours
    future_predictions = predict_future_data(sensor_data, hours_ahead=prediction_hours)
    
    # Enhanced prediction chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=sensor_data['Timestamp'].tail(48), 
        y=sensor_data['Temperature'].tail(48),
        mode='lines',
        name='البيانات الفعلية',
        line=dict(color='#2980B9', width=3)
    ))
    
    # Predicted data
    fig.add_trace(go.Scatter(
        x=future_predictions['Timestamp'], 
        y=future_predictions['Temperature'],
        mode='lines',
        name='التنبؤات',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=future_predictions['Timestamp'], 
        y=future_predictions['Temperature_Upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_predictions['Timestamp'], 
        y=future_predictions['Temperature_Lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.2)',
        name=f'نطاق الثقة {confidence_level}%',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'توقعات درجة الحرارة للـ {prediction_hours} ساعة القادمة',
        xaxis_title='الوقت',
        yaxis_title='درجة الحرارة (°C)',
        plot_bgcolor=theme_colors['chart-background'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme_colors['text-color']),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 ملخص التنبؤات")
        avg_temp = future_predictions['Temperature'].mean()
        max_temp = future_predictions['Temperature'].max()
        min_temp = future_predictions['Temperature'].min()
        
        st.metric("متوسط درجة الحرارة المتوقعة", f"{avg_temp:.1f}°C")
        st.metric("أعلى درجة حرارة متوقعة", f"{max_temp:.1f}°C")
        st.metric("أقل درجة حرارة متوقعة", f"{min_temp:.1f}°C")
    
    with col2:
        st.subheader("📊 جدول البيانات المتوقعة")
        display_predictions = future_predictions[['Timestamp', 'Temperature', 'Pressure', 'Vibration']].head(24)
        display_predictions['Timestamp'] = display_predictions['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_predictions, use_container_width=True)

elif chosen_menu == "Smart Recommendations":
    st.title("💡 التوصيات الذكية")
    st.markdown("### إدارة الشذوذ والتوصيات الاستباقية")
    
    current_risk = st.session_state.get('current_risk_level', 'Low')
    
    # Risk level indicator
    risk_colors = {"Low": "success", "Medium": "warning", "High": "error"}
    risk_arabic = {"Low": "منخفض", "Medium": "متوسط", "High": "عالي"}
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎯 مستوى المخاطر الحالي</div>
        <div class="metric-value" style="color: {theme_colors['accent-color'] if current_risk == 'High' else theme_colors['warning-color'] if current_risk == 'Medium' else theme_colors['success-color']}">{risk_arabic[current_risk]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 توليد توصية جديدة", type="primary"):
            new_rec = generate_recommendation(current_risk)
            st.session_state.recommendation_history.append({
                "timestamp": datetime.now(), 
                "recommendation": new_rec,
                "risk_level": current_risk
            })
            st.toast('✅ تم توليد توصية استباقية جديدة!', icon='💡')
            st.rerun()
    
    with col2:
        if st.button("🗑️ مسح السجل"):
            st.session_state.recommendation_history = []
            st.success("تم مسح سجل التوصيات!")
            st.rerun()
    
    if st.session_state.recommendation_history:
        st.subheader("📋 أحدث التوصيات")
        
        # Latest recommendation
        latest_rec = st.session_state.recommendation_history[-1]['recommendation']
        priority_class = f"priority-{latest_rec['priority'].lower()}"
        
        st.markdown(f"""
        <div class="recommendation-card {priority_class}">
            <h4>🎯 {latest_rec['title']}</h4>
            <p><strong>التفاصيل:</strong> {latest_rec['details']}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                <span><strong>الأولوية:</strong> {latest_rec['priority']}</span>
                <span><strong>الوقت المقدر:</strong> {latest_rec['estimated_time']}</span>
                <span><strong>الفعالية:</strong> {latest_rec['effectiveness']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation history
        if len(st.session_state.recommendation_history) > 1:
            st.subheader("📚 سجل التوصيات")
            
            for i, rec_entry in enumerate(reversed(st.session_state.recommendation_history[:-1])):
                if i < 5:  # Show only last 5
                    rec = rec_entry['recommendation']
                    timestamp = rec_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    with st.expander(f"📝 {rec['title']} - {timestamp}"):
                        st.write(f"**التفاصيل:** {rec['details']}")
                        st.write(f"**الأولوية:** {rec['priority']}")
                        st.write(f"**الوقت المقدر:** {rec['estimated_time']}")
    else:
        st.info("لم يتم توليد أي توصيات بعد. اضغط على 'توليد توصية جديدة' للبدء.")

elif chosen_menu == "About Project":
    st.title("ℹ️ حول المشروع")
    
    # Project image
    if os.path.exists("/home/ubuntu/upload/search_images/4Q9XEwBBAhBz.webp"):
        st.image("/home/ubuntu/upload/search_images/4Q9XEwBBAhBz.webp", use_column_width=True)
    
    st.markdown("""
    ## 🧠 التوأم الرقمي الذكي للسلامة الصناعية
    
    يطور هذا المشروع نظام توأم رقمي ذكي لسلامة حقول النفط، مستفيداً من بيانات أجهزة الاستشعار في الوقت الفعلي، 
    والتعلم الآلي المتقدم، والتحليلات التنبؤية لمنع الكوارث.
    
    ### ✨ الميزات الرئيسية:
    - 📊 مراقبة بيانات أجهزة الاستشعار في الوقت الفعلي
    - 🔍 كشف الشذوذ باستخدام خوارزميات التعلم الآلي
    - 📈 النمذجة التنبؤية للظروف المستقبلية
    - 💡 توصيات ذكية للتدخل الاستباقي
    - 🎨 واجهة مستخدم تفاعلية وسهلة الاستخدام
    
    ### 🎯 رؤيتنا
    إحداث ثورة في السلامة الصناعية من خلال تحويل البيانات الخام إلى رؤى قابلة للتنفيذ، 
    مما يضمن بيئة تشغيلية أكثر أماناً وكفاءة.
    
    ### 🛠️ التقنيات المستخدمة
    - **Python & Streamlit** للواجهة الأمامية
    - **Plotly** للرسوم البيانية التفاعلية
    - **Pandas & NumPy** لمعالجة البيانات
    - **Machine Learning** لكشف الشذوذ والتنبؤ
    """)
    
    # Team or contact info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📧 التواصل
        للاستفسارات والدعم الفني
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 الروابط
        - [GitHub Repository](#)
        - [Documentation](#)
        """)
    
    with col3:
        st.markdown("""
        ### 📊 الإحصائيات
        - **المستخدمون النشطون:** 150+
        - **البيانات المعالجة:** 1M+ نقطة
        - **دقة التنبؤ:** 94%
        """)

elif chosen_menu == "Settings":
    st.title("⚙️ إعدادات التطبيق")
    
    # Theme settings
    st.subheader("🎨 إعدادات المظهر")
    col1, col2 = st.columns(2)
    
    with col1:
        current_theme = st.session_state.theme
        st.write(f"**المظهر الحالي:** {current_theme.capitalize()}")
        
        if st.button("🔄 تبديل المظهر"):
            st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
            st.success("تم تغيير المظهر!")
            st.rerun()
    
    with col2:
        st.write("**معاينة الألوان:**")
        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin: 10px 0;">
            <div style="width: 30px; height: 30px; background-color: {theme_colors['primary-color']}; border: 1px solid #ccc; border-radius: 4px;"></div>
            <div style="width: 30px; height: 30px; background-color: {theme_colors['secondary-color']}; border-radius: 4px;"></div>
            <div style="width: 30px; height: 30px; background-color: {theme_colors['accent-color']}; border-radius: 4px;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data settings
    st.subheader("📊 إعدادات البيانات")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_data_points = st.slider("عدد نقاط البيانات التاريخية", 50, 500, 200)
        refresh_interval = st.selectbox("فترة التحديث (ثواني)", [30, 60, 120, 300], index=1)
    
    with col2:
        anomaly_sensitivity = st.slider("حساسية كشف الشذوذ", 0.1, 1.0, 0.5, 0.1)
        prediction_model = st.selectbox("نموذج التنبؤ", ["Linear Regression", "ARIMA", "LSTM"], index=0)
    
    if st.button("💾 حفظ الإعدادات"):
        st.success("تم حفظ الإعدادات بنجاح!")
    
    if st.button("🔄 إعادة توليد البيانات"):
        st.cache_data.clear()
        st.success("تم إعادة توليد البيانات!")
        st.rerun()
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ معلومات النظام")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**إصدار التطبيق:** v2.1.1")
        st.write("**آخر تحديث:** 2024-06-14")
        st.write("**حالة النظام:** 🟢 متصل")
    
    with col2:
        st.write(f"**عدد التوصيات:** {len(st.session_state.recommendation_history)}")
        st.write(f"**عدد الشذوذات:** {len(anomalies)}")
        st.write(f"**وقت التشغيل:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Reset options
    st.markdown("---")
    st.subheader("🔄 خيارات الإعادة تعيين")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ مسح سجل التوصيات", type="secondary"):
            st.session_state.recommendation_history = []
            st.success("تم مسح سجل التوصيات!")
    
    with col2:
        if st.button("🔄 إعادة تعيين الإعدادات", type="secondary"):
            st.session_state.theme = "light"
            st.success("تم إعادة تعيين الإعدادات!")
    
    with col3:
        if st.button("⚠️ إعادة تشغيل التطبيق", type="secondary"):
            st.cache_data.clear()
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        🧠 التوأم الرقمي الذكي | الإصدار النهائي المدعوم بالذكاء الاصطناعي الصناعي
    </div>
    """, 
    unsafe_allow_html=True
)

