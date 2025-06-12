import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Initialize session state for theme and recommendation history if not already done
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# تعيين نمط الصفحة
st.set_page_config(
    page_title="Smart Neural Digital Twin - Enhanced",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Configuration ---
def get_theme_colors(theme_mode):
    if theme_mode == "dark":
        return {
            "primary-color": "#0E1117",  # Dark background
            "secondary-color": "#3498DB", # Blue accent
            "accent-color": "#F39C12",   # Orange accent
            "light-bg": "#161A25",    # Slightly lighter dark bg for cards
            "white": "#2E3440",       # Card background
            "text-color": "#ECEFF4",    # Light text
            "text-color-muted": "#D8DEE9",
            "grid-color": "rgba(200, 200, 200, 0.2)"
        }
    else: # Light theme (default)
        return {
            "primary-color": "#1A365D",
            "secondary-color": "#4A90E2",
            "accent-color": "#F5A623",
            "light-bg": "#F7F9FC",
            "white": "#FFFFFF",
            "text-color": "#333333",
            "text-color-muted": "#666666",
            "grid-color": "rgba(211, 211, 211, 0.5)"
        }

colors = get_theme_colors(st.session_state.theme)

# تخصيص CSS الديناميكي بناءً على الثيم
st.markdown(f"""
<style>
    /* الألوان الرئيسية */
    :root {{
        --primary-color: {colors["primary-color"]};
        --secondary-color: {colors["secondary-color"]};
        --accent-color: {colors["accent-color"]};
        --light-bg: {colors["light-bg"]};
        --white: {colors["white"]};
        --text-color: {colors["text-color"]};
        --text-color-muted: {colors["text-color-muted"]};
        --grid-color: {colors["grid-color"]};
    }}
    
    body {{
        background-color: var(--primary-color);
        color: var(--text-color);
    }}

    /* تنسيق العناوين */
    h1, h2, h3 {{
        font-family: 'Montserrat', sans-serif;
        color: var(--text-color); /* Adjusted for theme */
    }}
    
    h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    
    h2 {{
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
    }}
    
    h3 {{
        font-size: 1.4rem;
        font-weight: 600;
    }}
    
    /* تنسيق النصوص */
    p, li, div, span {{
        font-family: 'Roboto', sans-serif;
        color: var(--text-color); /* Adjusted for theme */
    }}
    p, li {{
        font-size: 1.1rem;
        line-height: 1.6;
    }}
    
    /* تنسيق البطاقات */
    .card {{
        background-color: var(--white);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--secondary-color);
    }}
    
    /* تنسيق الأزرار */
    .stButton > button {{
        background-color: var(--secondary-color);
        color: {colors["white"] if st.session_state.theme == 'light' else colors["text-color"]}; /* Text color on button */
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: var(--accent-color);
        transform: translateY(-2px);
    }}
    
    /* تنسيق الشريط الجانبي */
    .css-1d391kg {{
        background-color: var(--primary-color) !important; 
    }}
    
    .css-1d391kg .sidebar-content {{
        background-color: var(--primary-color) !important;
    }}
    .css-1d391kg .sidebar-content p, .css-1d391kg .sidebar-content div, .css-1d391kg .sidebar-content span, .css-1d391kg .sidebar-content label, .css-1d391kg .sidebar-content h3 {{
        color: {colors["text-color"]} !important; /* Ensure sidebar text is readable */
    }}
    
    /* تنسيق العنوان الرئيسي */
    .main-header {{
        background-color: {colors["primary-color"] if st.session_state.theme == 'light' else colors["light-bg"]};
        padding: 2rem;
        border-radius: 10px;
        color: {colors["text-color"]};
        text-align: center;
        margin-bottom: 2rem;
        background-image: linear-gradient(135deg, {colors["primary-color"]} 0%, {colors["secondary-color"]} 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}
    .main-header h1, .main-header p {{
        color: {colors["white"] if st.session_state.theme == 'light' else colors["text-color"]} !important;
    }}
    
    /* تنسيق البطاقات المميزة */
    .feature-card {{
        background-color: var(--white);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }}
    
    .feature-icon {{
        font-size: 2.5rem;
        color: var(--secondary-color);
        margin-bottom: 1rem;
    }}
    
    /* تنسيق المؤشرات */
    .metric-container {{
        background-color: var(--light-bg);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-bottom: 3px solid var(--secondary-color);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary-color); /* Highlight metric value */
    }}
    
    .metric-label {{
        font-size: 1rem;
        color: var(--text-color-muted);
    }}
    
    /* تنسيق التنبيهات */
    .alert {{
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: var(--text-color);
    }}
    
    .alert-warning {{
        background-color: rgba(245, 166, 35, 0.2);
        border-left: 4px solid var(--accent-color);
    }}
    
    .alert-danger {{
        background-color: rgba(255, 76, 76, 0.2);
        border-left: 4px solid #FF4C4C;
    }}
    .alert-success {{
        background-color: rgba(76, 175, 80, 0.2); /* Green for success */
        border-left: 4px solid #4CAF50;
    }}
    
    /* تنسيق الشريط العلوي */
    .stApp header {{
        background-color: var(--primary-color) !important;
    }}
    
    /* تنسيق الخريطة */
    .map-container {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* تنسيق الرسوم البيانية */
    .chart-container {{
        background-color: var(--white);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }}
    
    /* تنسيق الشعار */
    .logo-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }}
    
    .logo-text {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {colors["white"] if st.session_state.theme == 'light' else colors["text-color"]} !important;
        margin-left: 0.5rem;
    }}
    
    /* تنسيق القسم الرئيسي */
    .main-section {{
        padding: 2rem 0;
    }}
    
    /* تنسيق الفاصل */
    hr {{
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), {colors["grid-color"]}, rgba(0, 0, 0, 0));
    }}
    
    /* تنسيق الأقسام */
    .section-title {{
        position: relative;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    
    .section-title:after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 50px;
        height: 3px;
        background-color: var(--secondary-color);
    }}
    
    /* تنسيق الشريط التقدمي */
    .stProgress > div > div > div > div {{
        background-color: var(--secondary-color);
    }}

    /* Recommendation history styling */
    .recommendation-entry {{
        border: 1px solid var(--grid-color);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: var(--light-bg);
    }}
    .recommendation-entry p {{
        margin-bottom: 5px;
    }}

</style>
""", unsafe_allow_html=True)

# تعريف الدوال المساعدة
def generate_sensor_data(num_points=200): # Increased points for longer history
    """توليد بيانات المستشعر المحاكاة"""
    now = datetime.now()
    dates = [now - timedelta(hours=i) for i in range(num_points)] # Changed to hours
    dates.reverse()
    
    temperature = [random.uniform(70, 85) + random.uniform(-5, 5) * np.sin(i/24) for i in range(num_points)]
    pressure = [random.uniform(900, 1000) + random.uniform(-20, 20) * np.sin(i/36) for i in range(num_points)]
    vibration = [random.uniform(0.1, 0.3) + random.uniform(0, 0.5) * (1 if random.random() > 0.95 else 0) for i in range(num_points)]
    
    for i in range(num_points):
        if random.random() > 0.97:
            temperature[i] += random.uniform(10, 15)
            pressure[i] -= random.uniform(50, 100)
            vibration[i] += random.uniform(0.5, 1.0)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration
    })
    return df

def calculate_risk_score(temp, pressure, vibration):
    temp_score = max(0, min(100, (temp - 70) * 5))
    pressure_score = max(0, min(100, abs(pressure - 950) / 2))
    vibration_score = max(0, min(100, vibration * 200))
    return (temp_score * 0.3) + (pressure_score * 0.3) + (vibration_score * 0.4)

def get_risk_level(score):
    if score < 20: return "منخفض", "green"
    elif score < 40: return "متوسط-منخفض", "lightgreen"
    elif score < 60: return "متوسط", "yellow"
    elif score < 80: return "متوسط-مرتفع", "orange"
    else: return "مرتفع", "red"

def create_gauge_chart(value, title, theme_colors):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': theme_colors["text-color"]}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': theme_colors["text-color-muted"]},
            'bar': {'color': theme_colors["secondary-color"]},
            'bgcolor': theme_colors["white"],
            'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'green'},
                {'range': [20, 40], 'color': 'lightgreen'},
                {'range': [40, 60], 'color': 'yellow'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 100], 'color': 'red'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Arial"})
    return fig

def create_time_series(df, column, title, color_hex, theme_colors):
    fig = px.line(df, x='timestamp', y=column, title=title)
    fig.update_traces(line_color=color_hex, line_width=2)
    fig.update_layout(
        xaxis_title="الوقت", yaxis_title=title, hovermode="x unified",
        height=250, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': theme_colors["text-color"], 'family': "Arial"},
        xaxis=dict(gridcolor=theme_colors["grid-color"]),
        yaxis=dict(gridcolor=theme_colors["grid-color"])
    )
    return fig

def create_prediction_chart(df, theme_colors):
    last_temp = df['temperature'].iloc[-1]
    last_time = df['timestamp'].iloc[-1]
    
    # Predict for 72 hours (1 data point per hour)
    future_hours = 72
    future_times = [last_time + timedelta(hours=i) for i in range(1, future_hours + 1)]
    
    # Simulate prediction with confidence interval
    predicted_temps = []
    upper_bound = []
    lower_bound = []
    current_pred_temp = last_temp
    for i in range(future_hours):
        change = random.uniform(-0.5, 0.8) + (i * 0.02) # Slight upward trend over time
        current_pred_temp += change
        predicted_temps.append(current_pred_temp)
        confidence_margin = random.uniform(1, 3) + (i * 0.05) # Increasing uncertainty
        upper_bound.append(current_pred_temp + confidence_margin)
        lower_bound.append(current_pred_temp - confidence_margin)

    pred_df = pd.DataFrame({
        'timestamp': future_times,
        'predicted_temperature': predicted_temps,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines', name='البيانات الحقيقية', line=dict(color=theme_colors["secondary-color"], width=2)))
    fig.add_trace(go.Scatter(x=pred_df['timestamp'], y=pred_df['predicted_temperature'], mode='lines', name='التنبؤات', line=dict(color=theme_colors["accent-color"], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=pred_df['timestamp'], y=pred_df['upper_bound'], mode='lines', name='الحد الأعلى', line=dict(color=theme_colors["accent-color"], width=1, dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=pred_df['timestamp'], y=pred_df['lower_bound'], mode='lines', name='الحد الأدنى', line=dict(color=theme_colors["accent-color"], width=1, dash='dot'), opacity=0.5, fill='tonexty', fillcolor=f"rgba({int(theme_colors['accent-color'][1:3], 16)}, {int(theme_colors['accent-color'][3:5], 16)}, {int(theme_colors['accent-color'][5:7], 16)}, 0.2)"))

    fig.update_layout(
        title="التنبؤ بدرجة الحرارة للـ 72 ساعة القادمة",
        xaxis_title="الوقت", yaxis_title="درجة الحرارة", hovermode="x unified",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': theme_colors["text-color"], 'family': "Arial"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=theme_colors["grid-color"]),
        yaxis=dict(gridcolor=theme_colors["grid-color"])
    )
    fig.add_vrect(x0=last_time, x1=future_times[-1], fillcolor=f"rgba({int(theme_colors['secondary-color'][1:3], 16)}, {int(theme_colors['secondary-color'][3:5], 16)}, {int(theme_colors['secondary-color'][5:7], 16)}, 0.1)", layer="below", line_width=0)
    return fig

# --- Simulated Intervention Engine (Enhanced) ---
def suggest_enhanced_intervention(sensor_input, risk_score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recommendations = []
    base_intervention = ""
    priority = "عادي"
    estimated_time = "غير محدد"
    effectiveness = "غير محدد"

    if sensor_input['temperature'] > 85 or risk_score > 80:
        base_intervention = "خطر ارتفاع حرارة شديد! توصية بإيقاف النظام فوراً للفحص."
        recommendations.append("1. إيقاف تشغيل الوحدة المتأثرة فوراً.")
        recommendations.append("2. فحص نظام التبريد والمراوح.")
        recommendations.append("3. التحقق من عدم وجود عوائق لتدفق الهواء.")
        priority = "عاجل جداً"
        estimated_time = "فوري - ساعتان"
        effectiveness = "عالية جداً (لتجنب التلف)"
        alert_type = "danger"
    elif sensor_input['temperature'] > 80 or risk_score > 60:
        base_intervention = "ارتفاع حرارة ملحوظ. توصية بتفعيل التبريد الإضافي ومراقبة الوضع."
        recommendations.append("1. تفعيل وحدات التبريد الإضافية أو زيادة سرعة المراوح.")
        recommendations.append("2. مراقبة درجة الحرارة عن كثب خلال الـ 30 دقيقة القادمة.")
        recommendations.append("3. تجهيز خطة لتقليل الحمل إذا استمر الارتفاع.")
        priority = "مرتفع"
        estimated_time = "15-30 دقيقة"
        effectiveness = "عالية"
        alert_type = "warning"
    elif sensor_input['pressure'] < 920 or sensor_input['pressure'] > 980:
        base_intervention = "ضغط غير مستقر. توصية بفحص النظام بحثاً عن تسربات أو انسدادات."
        recommendations.append("1. فحص خطوط الأنابيب والصمامات بحثاً عن تسربات.")
        recommendations.append("2. التحقق من معايرة مستشعرات الضغط.")
        recommendations.append("3. مراجعة سجلات الصيانة الأخيرة المتعلقة بالضغط.")
        priority = "متوسط"
        estimated_time = "1-3 ساعات"
        effectiveness = "متوسطة إلى عالية"
        alert_type = "warning"
    else:
        base_intervention = "النظام يعمل بشكل طبيعي. لا حاجة لتدخل فوري."
        recommendations.append("1. الاستمرار في المراقبة الدورية.")
        recommendations.append("2. التأكد من إجراء الصيانة الوقائية المجدولة.")
        priority = "منخفض"
        alert_type = "success"

    # Add to history
    st.session_state.recommendation_history.insert(0, {
        "timestamp": timestamp,
        "base_intervention": base_intervention,
        "details": recommendations,
        "priority": priority,
        "estimated_time": estimated_time,
        "effectiveness": effectiveness,
        "alert_type": alert_type,
        "sensor_data": sensor_input.copy(),
        "risk_score": risk_score
    })
    # Keep history to a certain size, e.g., last 10 recommendations
    st.session_state.recommendation_history = st.session_state.recommendation_history[:10]
    
    return st.session_state.recommendation_history[0] # Return the latest one

# الشريط الجانبي
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <span style="font-size: 24px;">🧠</span>
        <div class="logo-text">Smart Neural DT</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("التنقل", ["الرئيسية", "لوحة التحكم", "التنبؤات", "التوصيات الذكية", "المعلومات", "الإعدادات"])
    st.markdown("---")
    st.subheader("إعدادات العرض")
    theme_options = {"فاتح": "light", "داكن": "dark"}
    selected_theme_label = st.selectbox("اختر الثيم", options=list(theme_options.keys()), index=0 if st.session_state.theme == "light" else 1)
    if theme_options[selected_theme_label] != st.session_state.theme:
        st.session_state.theme = theme_options[selected_theme_label]
        st.experimental_rerun() # Rerun to apply new theme

    st.markdown("---")
    st.subheader("إعدادات المحاكاة")
    update_interval = st.slider("فترة التحديث (ثواني)", min_value=1, max_value=10, value=3)
    risk_threshold = st.slider("عتبة المخاطر", min_value=0, max_value=100, value=60)
    st.markdown("---")
    st.subheader("معلومات النظام")
    st.markdown("**الإصدار:** 2.0.0 (محسّن)")
    st.markdown(f"**آخر تحديث:** {datetime.now().strftime('%d %B %Y')}")
    st.markdown("**الحالة:** نشط")

# توليد البيانات
sensor_data_df = generate_sensor_data()
latest_data_point = sensor_data_df.iloc[-1]
current_risk_score = calculate_risk_score(latest_data_point['temperature'], latest_data_point['pressure'], latest_data_point['vibration'])
current_risk_level, current_risk_color = get_risk_level(current_risk_score)

# --- Dynamic Notifications (Example) ---
if 'last_notified_risk' not in st.session_state:
    st.session_state.last_notified_risk = None

if current_risk_score > risk_threshold and current_risk_score != st.session_state.last_notified_risk:
    st.toast(f"⚠️ تنبيه خطر مرتفع! درجة المخاطرة: {current_risk_score:.1f}", icon="⚠️")
    st.session_state.last_notified_risk = current_risk_score
elif current_risk_score > risk_threshold * 0.7 and current_risk_score != st.session_state.last_notified_risk and current_risk_level != get_risk_level(st.session_state.last_notified_risk if st.session_state.last_notified_risk else 0)[0]:
    st.toast(f"🔔 تنبيه: درجة المخاطرة متوسطة الارتفاع: {current_risk_score:.1f}", icon="🔔")
    st.session_state.last_notified_risk = current_risk_score

# الصفحة الرئيسية
if page == "الرئيسية":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin-bottom: 0.5rem;">Smart Neural Digital Twin for Disaster Prediction & Prevention</h1>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">نظام التوأم الرقمي العصبي الذكي للتنبؤ بالكوارث والوقاية منها (نسخة محسنة)</p>
        <p style="font-size: 1rem;">Powered by Advanced AI & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("نظرة عامة على النظام")
    st.markdown("نظام **التوأم الرقمي العصبي الذكي** هو حل متقدم يجمع بين تقنيات الذكاء الاصطناعي والتعلم الآلي لإنشاء نسخة رقمية من الأنظمة الفيزيائية. يقوم النظام بمراقبة البيانات في الوقت الفعلي، وتحليلها، والتنبؤ بالمخاطر المحتملة قبل حدوثها، مما يتيح اتخاذ إجراءات وقائية مبكرة.")
    # ... (Rest of Home page content - kept similar for brevity, ensure text colors are correct for theme)
    st.markdown("## المميزات الرئيسية")
    col1, col2, col3 = st.columns(3)
    features = [
        {"icon": "🔍", "title": "مراقبة في الوقت الفعلي", "text": "مراقبة مستمرة للبيانات من مختلف المستشعرات والمصادر لضمان الكشف المبكر عن أي مؤشرات للمخاطر."},
        {"icon": "🧠", "title": "تحليل عصبي متقدم", "text": "استخدام خوارزميات الشبكات العصبية المتقدمة لتحليل الأنماط المعقدة وتحديد العلاقات غير الواضحة في البيانات."},
        {"icon": "⏱️", "title": "تنبؤ استباقي (72 ساعة)", "text": "التنبؤ بالمخاطر المحتملة لمدة تصل إلى 72 ساعة، مما يوفر وقتاً كافياً لاتخاذ الإجراءات الوقائية."},
        {"icon": "🔔", "title": "تنبيهات وتوصيات ذكية", "text": "نظام تنبيهات متكامل يرسل إشعارات فورية مع توصيات مفصلة للإجراءات عند اكتشاف مخاطر."},
        {"icon": "📊", "title": "تحليلات بصرية متقدمة", "text": "عرض البيانات والتحليلات بطريقة بصرية سهلة الفهم، مع فواصل ثقة للتنبؤات."},
        {"icon": "🔄", "title": "تعلم مستمر وتكيف", "text": "تحسين مستمر لدقة التنبؤات والتوصيات من خلال التعلم من البيانات التاريخية والتغذية الراجعة."}
    ]
    for i, feature in enumerate(features):
        with eval(f"col{i%3+1}"):
             st.markdown(f"""<div class="feature-card"><div class="feature-icon">{feature['icon']}</div><h3>{feature['title']}</h3><p>{feature['text']}</p></div>""", unsafe_allow_html=True)

# لوحة التحكم
elif page == "لوحة التحكم":
    st.markdown("<h1 class='section-title'>لوحة التحكم</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### درجة المخاطرة")
        st.plotly_chart(create_gauge_chart(current_risk_score, "درجة المخاطرة", colors), use_container_width=True)
        st.markdown(f"**مستوى المخاطرة:** <span style='color:{current_risk_color}; font-weight:bold;'>{current_risk_level}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("### درجة الحرارة")
        st.plotly_chart(create_gauge_chart(latest_data_point['temperature'], "درجة الحرارة (°F)", colors), use_container_width=True)
        st.markdown(f"**القراءة الحالية:** {latest_data_point['temperature']:.2f} °F")
    with col3:
        st.markdown("### الضغط")
        st.plotly_chart(create_gauge_chart((latest_data_point['pressure'] - 900) / 2, "الضغط (hPa)", colors), use_container_width=True)
        st.markdown(f"**القراءة الحالية:** {latest_data_point['pressure']:.2f} hPa")
    
    st.markdown("<h3 class='section-title'>التنبيهات النشطة</h3>", unsafe_allow_html=True)
    if current_risk_score > risk_threshold:
        st.markdown("<div class='alert alert-danger'><strong>تنبيه عالي المخاطر!</strong> تم اكتشاف مؤشرات لمخاطر محتملة. يرجى مراجعة صفحة التوصيات الذكية فوراً.</div>", unsafe_allow_html=True)
    elif current_risk_score > risk_threshold * 0.7:
        st.markdown("<div class='alert alert-warning'><strong>تنبيه!</strong> هناك زيادة في مؤشرات المخاطر. يرجى مراقبة الوضع عن كثب ومراجعة التوصيات.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert alert-success'>لا توجد تنبيهات نشطة حالياً. النظام يعمل بشكل طبيعي.</div>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-title'>بيانات المستشعرات</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_time_series(sensor_data_df, 'temperature', "درجة الحرارة (°F)", colors["secondary-color"], colors), use_container_width=True)
    with col2:
        st.plotly_chart(create_time_series(sensor_data_df, 'pressure', "الضغط (hPa)", colors["accent-color"], colors), use_container_width=True)
    # ... (rest of dashboard similar)

# التنبؤات
elif page == "التنبؤات":
    st.markdown("<h1 class='section-title'>التنبؤات والتحليلات المستقبلية (72 ساعة)</h1>", unsafe_allow_html=True)
    st.markdown("### التنبؤ بدرجة الحرارة مع فواصل الثقة")
    st.plotly_chart(create_prediction_chart(sensor_data_df, colors), use_container_width=True)
    # ... (rest of predictions page similar)

# --- Enhanced Smart Recommendations Page ---
elif page == "التوصيات الذكية":
    st.markdown("<h1 class='section-title'>التوصيات الذكية وسجل التدخلات</h1>", unsafe_allow_html=True)
    
    sensor_input_for_recommendation = {
        'pressure': latest_data_point['pressure'],
        'temperature': latest_data_point['temperature'],
        'flow_rate': random.uniform(25,35) # Simulated flow rate
    }
    latest_recommendation = suggest_enhanced_intervention(sensor_input_for_recommendation, current_risk_score)

    st.markdown("### أحدث توصية تدخل")
    alert_class = f"alert alert-{latest_recommendation['alert_type']}"
    st.markdown(f"<div class='{alert_class}'>", unsafe_allow_html=True)
    st.markdown(f"**{latest_recommendation['base_intervention']}**")
    st.markdown(f"**الأولوية:** {latest_recommendation['priority']}")
    st.markdown(f"**الوقت المقدر للتنفيذ:** {latest_recommendation['estimated_time']}")
    st.markdown(f"**الفعالية المتوقعة:** {latest_recommendation['effectiveness']}")
    st.markdown("**التفاصيل:**")
    for detail in latest_recommendation['details']:
        st.markdown(f"- {detail}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### سجل التوصيات الأخيرة")
    if not st.session_state.recommendation_history:
        st.info("لا توجد توصيات سابقة.")
    else:
        for rec in st.session_state.recommendation_history[1:]: # Skip the latest one already shown
            with st.expander(f"توصية بتاريخ: {rec['timestamp']} - الأولوية: {rec['priority']}"):
                st.markdown(f"<div class='recommendation-entry'>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>الوقت:</strong> {rec['timestamp']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>التوصية الأساسية:</strong> {rec['base_intervention']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>الأولوية:</strong> {rec['priority']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>الوقت المقدر:</strong> {rec['estimated_time']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>الفعالية:</strong> {rec['effectiveness']}</p>", unsafe_allow_html=True)
                st.markdown("<strong>التفاصيل:</strong>", unsafe_allow_html=True)
                for detail in rec['details']:
                    st.markdown(f"- {detail}")
                st.markdown("<strong>بيانات المستشعر وقتها:</strong>", unsafe_allow_html=True)
                st.json(rec['sensor_data'])
                st.markdown(f"<p><strong>درجة المخاطرة وقتها:</strong> {rec['risk_score']:.2f}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# المعلومات
elif page == "المعلومات":
    st.markdown("<h1 class='section-title'>معلومات عن النظام</h1>", unsafe_allow_html=True)
    # ... (Information page content - ensure text colors are correct for theme)

# الإعدادات
elif page == "الإعدادات":
    st.markdown("<h1 class='section-title'>الإعدادات</h1>", unsafe_allow_html=True)
    # ... (Settings page content - ensure text colors are correct for theme)

# Auto-update simulation (can be triggered by a button or timer in a real app)
if 'last_run_time' not in st.session_state:
    st.session_state.last_run_time = time.time()

if time.time() - st.session_state.last_run_time > update_interval:
    st.session_state.last_run_time = time.time()
    st.experimental_rerun()
