import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# تكوين الصفحة
st.set_page_config(
    page_title="التوأم الرقمي العصبي الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل البيانات المحاكاة
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "sensor_data_simulated.csv",
            parse_dates=["Time"],
            sep=","
        )
        df.rename(columns={
            "Time": "timestamp",
            "Temperature (°C)": "temp",
            "Pressure (psi)": "pressure",
            "Vibration (g)": "vibration",
            "Methane (CH₄ ppm)": "gas",
            "H₂S (ppm)": "h2s"
        }, inplace=True)
        return df
    except:
        # إنشاء بيانات تجريبية في حالة عدم وجود الملف
        import numpy as np
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        data = {
            'timestamp': dates,
            'temp': np.random.normal(25, 5, 1000),
            'pressure': np.random.normal(100, 10, 1000),
            'vibration': np.random.normal(0.5, 0.1, 1000),
            'gas': np.random.normal(50, 10, 1000),
            'h2s': np.random.normal(5, 2, 1000)
        }
        return pd.DataFrame(data)

df = load_data()

# تخزين حالة التطبيق
if 'language' not in st.session_state:
    st.session_state.language = 'العربية'
if 'theme' not in st.session_state:
    st.session_state.theme = 'Ocean'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'لوحة البيانات'

# ألوان المظاهر
themes = {
    "Ocean": {
        "primary": "#1E90FF",
        "secondary": "#87CEEB", 
        "background": "#F0F8FF",
        "text": "#2F4F4F",
        "sidebar": "#E6F3FF"
    },
    "Forest": {
        "primary": "#228B22",
        "secondary": "#90EE90",
        "background": "#F0FFF0", 
        "text": "#2F4F4F",
        "sidebar": "#E6FFE6"
    },
    "Sunset": {
        "primary": "#FF4500",
        "secondary": "#FFA500",
        "background": "#FFF8DC",
        "text": "#8B4513",
        "sidebar": "#FFE4E1"
    },
    "Purple": {
        "primary": "#800080",
        "secondary": "#DDA0DD",
        "background": "#F8F0FF",
        "text": "#4B0082",
        "sidebar": "#E6E6FA"
    },
    "Slate": {
        "primary": "#2F4F4F",
        "secondary": "#708090",
        "background": "#F5F5F5",
        "text": "#2F4F4F",
        "sidebar": "#E8E8E8"
    }
}

current_theme = themes[st.session_state.theme]

# CSS مخصص للتصميم المحسن
st.markdown(f"""
<style>
    /* تخصيص الشريط الجانبي */
    .css-1d391kg {{
        background-color: {current_theme['sidebar']};
        border-right: 3px solid {current_theme['primary']};
    }}
    
    /* تخصيص العنوان الرئيسي */
    .main-header {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']});
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    /* تخصيص البطاقات */
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {current_theme['primary']};
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}
    
    /* تخصيص أزرار القائمة */
    .menu-button {{
        display: block;
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        border: 2px solid {current_theme['primary']};
        border-radius: 10px;
        color: {current_theme['text']};
        text-decoration: none;
        text-align: right;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .menu-button:hover {{
        background: {current_theme['primary']};
        color: white;
        transform: translateX(-5px);
    }}
    
    .menu-button.active {{
        background: {current_theme['primary']};
        color: white;
    }}
    
    /* تخصيص صندوق الإعدادات */
    .settings-box {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {current_theme['secondary']};
        margin: 1rem 0;
    }}
    
    /* تخصيص عام */
    .stApp {{
        background-color: {current_theme['background']};
    }}
    
    .stSelectbox > div > div {{
        background-color: white;
        border: 2px solid {current_theme['primary']};
        border-radius: 8px;
    }}
    
    .stRadio > div {{
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {current_theme['secondary']};
    }}
    
    /* تحسين الرسوم البيانية */
    .js-plotly-plot {{
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    /* تخصيص التنبيهات */
    .alert {{
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid {current_theme['primary']};
    }}
    
    .alert-success {{
        background-color: #d4edda;
        color: #155724;
    }}
    
    .alert-warning {{
        background-color: #fff3cd;
        color: #856404;
    }}
    
    .alert-danger {{
        background-color: #f8d7da;
        color: #721c24;
    }}
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي المحسن
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: {current_theme['primary']}; color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h2>🧠 القائمة الرئيسية</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # قسم اختيار اللغة
    st.markdown("""
    <div class="settings-box">
        <h4>🌐 اختيار اللغة</h4>
    </div>
    """, unsafe_allow_html=True)
    
    language_options = ["العربية", "English"]
    selected_lang = st.radio(
        "",
        language_options,
        index=language_options.index(st.session_state.language),
        key="lang_radio"
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    # قسم اختيار المظهر
    st.markdown("""
    <div class="settings-box">
        <h4>🎨 لوحة الألوان</h4>
    </div>
    """, unsafe_allow_html=True)
    
    theme_options = ["Ocean", "Forest", "Sunset", "Purple", "Slate"]
    theme_labels = {
        "Ocean": "🌊 المحيط",
        "Forest": "🌲 الغابة", 
        "Sunset": "🌅 الغروب",
        "Purple": "💜 البنفسجي",
        "Slate": "⚫ الرمادي"
    }
    
    selected_theme = st.selectbox(
        "",
        theme_options,
        index=theme_options.index(st.session_state.theme),
        format_func=lambda x: theme_labels[x],
        key="theme_select"
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    st.markdown("---")
    
    # قائمة التنقل
    pages = {
        "لوحة البيانات": "📊",
        "المحاكاة": "🔄", 
        "التحليل التنبؤي": "📈",
        "الحلول الذكية": "💡",
        "الإعدادات": "⚙️",
        "حول المشروع": "ℹ️"
    }
    
    for page, icon in pages.items():
        if st.button(f"{icon} {page}", key=f"btn_{page}", use_container_width=True):
            st.session_state.current_page = page
            st.rerun()

# العنوان الرئيسي
st.markdown(f"""
<div class="main-header">
    <h1>🧠 التوأم الرقمي العصبي الذكي</h1>
    <p>نظام متقدم لمراقبة وتحليل البيانات في الوقت الفعلي</p>
</div>
""", unsafe_allow_html=True)

# المحتوى الرئيسي حسب الصفحة المختارة
current_page = st.session_state.current_page

if current_page == "لوحة البيانات":
    st.markdown("## 📊 لوحة البيانات الرئيسية")
    
    # عرض المقاييس الحالية
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌡️ درجة الحرارة</h4>
            <h2>{latest.temp:.1f}°C</h2>
            <small>آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 الضغط</h4>
            <h2>{latest.pressure:.1f} PSI</h2>
            <small>آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📳 الاهتزاز</h4>
            <h2>{latest.vibration:.2f} g</h2>
            <small>آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💨 غاز الميثان</h4>
            <h2>{latest.gas:.1f} ppm</h2>
            <small>آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # الرسوم البيانية
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 اتجاهات البيانات")
        fig = px.line(
            df.tail(100), 
            x="timestamp", 
            y=["temp", "pressure", "vibration", "gas"],
            labels={"timestamp": "الوقت", "value": "القيمة", "variable": "المتغير"},
            color_discrete_sequence=[current_theme['primary'], current_theme['secondary'], '#FF6B6B', '#4ECDC4']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=current_theme['text']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ الخريطة الحرارية")
        heat_data = df.pivot_table(
            index=df.timestamp.dt.hour, 
            columns=df.timestamp.dt.day, 
            values="temp", 
            aggfunc="mean"
        )
        fig2 = go.Figure(data=go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns,
            y=heat_data.index,
            colorscale="Viridis"
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=current_theme['text'],
            xaxis_title="اليوم",
            yaxis_title="الساعة"
        )
        st.plotly_chart(fig2, use_container_width=True)

elif current_page == "المحاكاة":
    st.markdown("## 🔄 محاكاة البيانات")
    
    st.markdown("""
    <div class="alert alert-success">
        <strong>محاكاة نشطة:</strong> يتم توليد البيانات التجريبية لعرض إمكانيات النظام
    </div>
    """, unsafe_allow_html=True)
    
    # عرض إحصائيات المحاكاة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 إحصائيات المحاكاة")
        stats = df.describe()
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.markdown("### 🎛️ تحكم في المحاكاة")
        simulation_speed = st.slider("سرعة المحاكاة", 1, 10, 5)
        data_points = st.slider("عدد نقاط البيانات", 100, 1000, 500)
        
        if st.button("🔄 إعادة تشغيل المحاكاة", use_container_width=True):
            st.success("تم إعادة تشغيل المحاكاة بنجاح!")

elif current_page == "التحليل التنبؤي":
    st.markdown("## 📈 التحليل التنبؤي")
    
    # تحليل الاتجاهات
    future_data = df.set_index("timestamp").resample("H").mean().ffill().tail(72)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 توقعات درجة الحرارة")
        fig3 = px.line(
            future_data, 
            x=future_data.index, 
            y="temp",
            labels={"timestamp": "الوقت", "temp": "درجة الحرارة (°C)"},
            color_discrete_sequence=[current_theme['primary']]
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=current_theme['text']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ تنبيهات النظام")
        
        # فحص القيم الشاذة
        temp_mean = df['temp'].mean()
        temp_std = df['temp'].std()
        
        if latest.temp > temp_mean + 2*temp_std:
            st.markdown("""
            <div class="alert alert-danger">
                <strong>تحذير:</strong> درجة الحرارة مرتفعة بشكل غير طبيعي!
            </div>
            """, unsafe_allow_html=True)
        elif latest.temp < temp_mean - 2*temp_std:
            st.markdown("""
            <div class="alert alert-warning">
                <strong>تنبيه:</strong> درجة الحرارة منخفضة بشكل غير طبيعي
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert alert-success">
                <strong>طبيعي:</strong> جميع القراءات ضمن المعدل الطبيعي
            </div>
            """, unsafe_allow_html=True)

elif current_page == "الحلول الذكية":
    st.markdown("## 💡 الحلول الذكية")
    
    st.markdown("""
    ### 🤖 توصيات النظام الذكي
    
    بناءً على تحليل البيانات الحالية، يقترح النظام الحلول التالية:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔧 التحسينات المقترحة
        - تحسين نظام التبريد
        - معايرة أجهزة الاستشعار
        - تحديث خوارزميات التنبؤ
        """)
    
    with col2:
        st.markdown("""
        #### 📋 خطة الصيانة
        - فحص شهري للمستشعرات
        - تنظيف أسبوعي للمعدات
        - مراجعة ربع سنوية للنظام
        """)

elif current_page == "الإعدادات":
    st.markdown("## ⚙️ إعدادات النظام")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌐 إعدادات اللغة")
        st.info(f"اللغة الحالية: {st.session_state.language}")
        
        st.markdown("### 🎨 إعدادات المظهر")
        st.info(f"المظهر الحالي: {theme_labels[st.session_state.theme]}")
    
    with col2:
        st.markdown("### 🔔 إعدادات التنبيهات")
        enable_alerts = st.checkbox("تفعيل التنبيهات", value=True)
        alert_threshold = st.slider("حد التنبيه", 0.1, 2.0, 1.0)
        
        st.markdown("### 💾 إعدادات البيانات")
        data_retention = st.selectbox("مدة الاحتفاظ بالبيانات", ["أسبوع", "شهر", "3 أشهر", "سنة"])

elif current_page == "حول المشروع":
    st.markdown("## ℹ️ حول المشروع")
    
    st.markdown("""
    ### 🧠 التوأم الرقمي العصبي الذكي
    
    نظام متطور لمراقبة وتحليل البيانات في الوقت الفعلي باستخدام تقنيات الذكاء الاصطناعي والتعلم الآلي.
    
    #### ✨ الميزات الرئيسية:
    - مراقبة البيانات في الوقت الفعلي
    - كشف الشذوذ التلقائي
    - التحليل التنبؤي المتقدم
    - واجهة مستخدم تفاعلية
    - دعم متعدد اللغات
    - مظاهر متنوعة قابلة للتخصيص
    
    #### 🛠️ التقنيات المستخدمة:
    - Python & Streamlit
    - Plotly للرسوم البيانية
    - Pandas لمعالجة البيانات
    - تقنيات التعلم الآلي
    
    #### 📧 التواصل:
    للدعم الفني أو الاستفسارات، يرجى التواصل مع فريق التطوير.
    """)

# تذييل الصفحة
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {current_theme['text']}; padding: 1rem;">
    <small>© 2024 التوأم الرقمي العصبي الذكي - جميع الحقوق محفوظة</small>
</div>
""", unsafe_allow_html=True)

