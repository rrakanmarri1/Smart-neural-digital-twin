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

# ألوان المظاهر - محدثة للمظهر الداكن
themes = {
    "Ocean": {
        "primary": "#1E90FF",
        "secondary": "#4169E1", 
        "background": "#1a1a1a",
        "text": "#ffffff",
        "sidebar": "#2d2d2d",
        "card": "#333333",
        "border": "#1E90FF"
    },
    "Forest": {
        "primary": "#32CD32",
        "secondary": "#228B22",
        "background": "#1a1a1a", 
        "text": "#ffffff",
        "sidebar": "#2d2d2d",
        "card": "#333333",
        "border": "#32CD32"
    },
    "Sunset": {
        "primary": "#FF6347",
        "secondary": "#FF4500",
        "background": "#1a1a1a",
        "text": "#ffffff",
        "sidebar": "#2d2d2d",
        "card": "#333333",
        "border": "#FF6347"
    },
    "Purple": {
        "primary": "#9370DB",
        "secondary": "#8A2BE2",
        "background": "#1a1a1a",
        "text": "#ffffff",
        "sidebar": "#2d2d2d",
        "card": "#333333",
        "border": "#9370DB"
    },
    "Slate": {
        "primary": "#708090",
        "secondary": "#2F4F4F",
        "background": "#1a1a1a",
        "text": "#ffffff",
        "sidebar": "#2d2d2d",
        "card": "#333333",
        "border": "#708090"
    }
}

current_theme = themes[st.session_state.theme]

# CSS مخصص للتصميم الداكن المحسن
st.markdown(f"""
<style>
    /* تخصيص الخلفية الرئيسية */
    .stApp {{
        background-color: {current_theme['background']};
        color: {current_theme['text']};
    }}
    
    /* تخصيص الشريط الجانبي */
    .css-1d391kg {{
        background-color: {current_theme['sidebar']};
        border-right: 3px solid {current_theme['primary']};
    }}
    
    /* تخصيص النصوص في الشريط الجانبي */
    .css-1d391kg .stMarkdown, .css-1d391kg .stSelectbox label, .css-1d391kg .stRadio label {{
        color: {current_theme['text']} !important;
    }}
    
    /* تخصيص العنوان الرئيسي */
    .main-header {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']});
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    
    /* تخصيص البطاقات */
    .metric-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {current_theme['primary']};
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}
    
    /* تخصيص أزرار القائمة */
    .stButton > button {{
        display: block;
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        background: {current_theme['card']};
        border: 2px solid {current_theme['primary']};
        border-radius: 10px;
        color: {current_theme['text']};
        text-decoration: none;
        text-align: right;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        background: {current_theme['primary']};
        color: white;
        transform: translateX(-5px);
    }}
    
    /* تخصيص صندوق الإعدادات */
    .settings-box {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {current_theme['border']};
        margin: 1rem 0;
    }}
    
    /* تخصيص عناصر الإدخال */
    .stSelectbox > div > div {{
        background-color: {current_theme['card']};
        color: {current_theme['text']};
        border: 2px solid {current_theme['primary']};
        border-radius: 8px;
    }}
    
    .stRadio > div {{
        background-color: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {current_theme['border']};
    }}
    
    /* تخصيص النصوص */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div {{
        color: {current_theme['text']} !important;
    }}
    
    /* تخصيص الرسوم البيانية */
    .js-plotly-plot {{
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        background-color: {current_theme['card']};
    }}
    
    /* تخصيص التنبيهات */
    .alert {{
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid {current_theme['primary']};
        color: {current_theme['text']};
    }}
    
    .alert-success {{
        background-color: rgba(40, 167, 69, 0.2);
        border-left-color: #28a745;
    }}
    
    .alert-warning {{
        background-color: rgba(255, 193, 7, 0.2);
        border-left-color: #ffc107;
    }}
    
    .alert-danger {{
        background-color: rgba(220, 53, 69, 0.2);
        border-left-color: #dc3545;
    }}
    
    /* تخصيص العناوين */
    .section-title {{
        color: {current_theme['primary']};
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        border-bottom: 2px solid {current_theme['primary']};
        padding-bottom: 0.5rem;
    }}
    
    /* تخصيص المحتوى الرئيسي */
    .main .block-container {{
        background-color: {current_theme['background']};
        color: {current_theme['text']};
    }}
    
    /* تخصيص شريط التمرير */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {current_theme['sidebar']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {current_theme['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {current_theme['secondary']};
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
    st.markdown(f"""
    <div class="settings-box">
        <h4 style="color: {current_theme['primary']};">🌐 اختيار اللغة</h4>
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
    st.markdown(f"""
    <div class="settings-box">
        <h4 style="color: {current_theme['primary']};">🎨 لوحة الألوان</h4>
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
    st.markdown(f'<div class="section-title">📊 لوحة البيانات الرئيسية</div>', unsafe_allow_html=True)
    
    # عرض المقاييس الحالية
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">🌡️ درجة الحرارة</h4>
            <h2 style="color: {current_theme['text']};">{latest.temp:.1f}°C</h2>
            <small style="color: {current_theme['text']};">آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📏 الضغط</h4>
            <h2 style="color: {current_theme['text']};">{latest.pressure:.1f} PSI</h2>
            <small style="color: {current_theme['text']};">آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📳 الاهتزاز</h4>
            <h2 style="color: {current_theme['text']};">{latest.vibration:.2f} g</h2>
            <small style="color: {current_theme['text']};">آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">💨 غاز الميثان</h4>
            <h2 style="color: {current_theme['text']};">{latest.gas:.1f} ppm</h2>
            <small style="color: {current_theme['text']};">آخر قراءة</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # الرسوم البيانية
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">📈 اتجاهات البيانات</div>', unsafe_allow_html=True)
        fig = px.line(
            df.tail(100), 
            x="timestamp", 
            y=["temp", "pressure", "vibration", "gas"],
            labels={"timestamp": "الوقت", "value": "القيمة", "variable": "المتغير"},
            color_discrete_sequence=[current_theme['primary'], current_theme['secondary'], '#FF6B6B', '#4ECDC4']
        )
        fig.update_layout(
            paper_bgcolor=current_theme['card'],
            plot_bgcolor=current_theme['card'],
            font_color=current_theme['text']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-title">🗺️ الخريطة الحرارية</div>', unsafe_allow_html=True)
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
            paper_bgcolor=current_theme['card'],
            plot_bgcolor=current_theme['card'],
            font_color=current_theme['text'],
            xaxis_title="اليوم",
            yaxis_title="الساعة"
        )
        st.plotly_chart(fig2, use_container_width=True)

elif current_page == "المحاكاة":
    st.markdown(f'<div class="section-title">🔄 محاكاة البيانات</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="alert alert-success">
        <strong>محاكاة نشطة:</strong> يتم توليد البيانات التجريبية لعرض إمكانيات النظام
    </div>
    """, unsafe_allow_html=True)
    
    # عرض إحصائيات المحاكاة
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">📊 إحصائيات المحاكاة</div>', unsafe_allow_html=True)
        stats = df.describe()
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-title">🎛️ تحكم في المحاكاة</div>', unsafe_allow_html=True)
        simulation_speed = st.slider("سرعة المحاكاة", 1, 10, 5)
        data_points = st.slider("عدد نقاط البيانات", 100, 1000, 500)
        
        if st.button("🔄 إعادة تشغيل المحاكاة", use_container_width=True):
            st.success("تم إعادة تشغيل المحاكاة بنجاح!")

elif current_page == "التحليل التنبؤي":
    st.markdown(f'<div class="section-title">📈 التحليل التنبؤي</div>', unsafe_allow_html=True)
    
    # تحليل الاتجاهات
    future_data = df.set_index("timestamp").resample("H").mean().ffill().tail(72)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">🔮 توقعات درجة الحرارة</div>', unsafe_allow_html=True)
        fig3 = px.line(
            future_data, 
            x=future_data.index, 
            y="temp",
            labels={"timestamp": "الوقت", "temp": "درجة الحرارة (°C)"},
            color_discrete_sequence=[current_theme['primary']]
        )
        fig3.update_layout(
            paper_bgcolor=current_theme['card'],
            plot_bgcolor=current_theme['card'],
            font_color=current_theme['text']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-title">⚠️ تنبيهات النظام</div>', unsafe_allow_html=True)
        
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
    st.markdown(f'<div class="section-title">💡 الحلول الذكية</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="section-title">🤖 توصيات النظام الذكي</div>
    
    <p style="color: {current_theme['text']};">بناءً على تحليل البيانات الحالية، يقترح النظام الحلول التالية:</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">🔧 التحسينات المقترحة</h4>
            <ul style="color: {current_theme['text']};">
                <li>تحسين نظام التبريد</li>
                <li>معايرة أجهزة الاستشعار</li>
                <li>تحديث خوارزميات التنبؤ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📋 خطة الصيانة</h4>
            <ul style="color: {current_theme['text']};">
                <li>فحص شهري للمستشعرات</li>
                <li>تنظيف أسبوعي للمعدات</li>
                <li>مراجعة ربع سنوية للنظام</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "الإعدادات":
    st.markdown(f'<div class="section-title">⚙️ إعدادات النظام</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">🌐 إعدادات اللغة</div>', unsafe_allow_html=True)
        st.info(f"اللغة الحالية: {st.session_state.language}")
        
        st.markdown(f'<div class="section-title">🎨 إعدادات المظهر</div>', unsafe_allow_html=True)
        st.info(f"المظهر الحالي: {theme_labels[st.session_state.theme]}")
    
    with col2:
        st.markdown(f'<div class="section-title">🔔 إعدادات التنبيهات</div>', unsafe_allow_html=True)
        enable_alerts = st.checkbox("تفعيل التنبيهات", value=True)
        alert_threshold = st.slider("حد التنبيه", 0.1, 2.0, 1.0)
        
        st.markdown(f'<div class="section-title">💾 إعدادات البيانات</div>', unsafe_allow_html=True)
        data_retention = st.selectbox("مدة الاحتفاظ بالبيانات", ["أسبوع", "شهر", "3 أشهر", "سنة"])

elif current_page == "حول المشروع":
    st.markdown(f'<div class="section-title">ℹ️ حول المشروع</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {current_theme['primary']};">🧠 التوأم الرقمي العصبي الذكي</h3>
        
        <p style="color: {current_theme['text']};">نظام متطور لمراقبة وتحليل البيانات في الوقت الفعلي باستخدام تقنيات الذكاء الاصطناعي والتعلم الآلي.</p>
        
        <h4 style="color: {current_theme['primary']};">✨ الميزات الرئيسية:</h4>
        <ul style="color: {current_theme['text']};">
            <li>مراقبة البيانات في الوقت الفعلي</li>
            <li>كشف الشذوذ التلقائي</li>
            <li>التحليل التنبؤي المتقدم</li>
            <li>واجهة مستخدم تفاعلية</li>
            <li>دعم متعدد اللغات</li>
            <li>مظاهر متنوعة قابلة للتخصيص</li>
        </ul>
        
        <h4 style="color: {current_theme['primary']};">🛠️ التقنيات المستخدمة:</h4>
        <ul style="color: {current_theme['text']};">
            <li>Python & Streamlit</li>
            <li>Plotly للرسوم البيانية</li>
            <li>Pandas لمعالجة البيانات</li>
            <li>تقنيات التعلم الآلي</li>
        </ul>
        
        <h4 style="color: {current_theme['primary']};">📧 التواصل:</h4>
        <p style="color: {current_theme['text']};">للدعم الفني أو الاستفسارات، يرجى التواصل مع فريق التطوير.</p>
    </div>
    """, unsafe_allow_html=True)

# تذييل الصفحة
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {current_theme['text']}; padding: 1rem;">
    <small>© 2024 التوأم الرقمي العصبي الذكي - جميع الحقوق محفوظة</small>
</div>
""", unsafe_allow_html=True)

