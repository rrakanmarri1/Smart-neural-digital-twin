import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import random
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import time
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# تكوين الصفحة
st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل البيانات المحاكاة
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["Time"], sep=",")
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
        dates = pd.date_range("2024-01-01", periods=1000, freq="h")
        data = {
            "timestamp": dates,
            "temp": np.random.normal(25, 5, 1000),
            "pressure": np.random.normal(100, 10, 1000),
            "vibration": np.random.normal(0.5, 0.1, 1000),
            "gas": np.random.normal(50, 10, 1000),
            "h2s": np.random.normal(5, 2, 1000)
        }
        return pd.DataFrame(data)

df = load_data()

# تخزين حالة التطبيق
if 'language' not in st.session_state:
    st.session_state.language = 'ar' # تم التعديل ليتوافق مع مفتاح قاموس الترجمة
if 'theme' not in st.session_state:
    st.session_state.theme = 'purple'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main_dashboard'
if 'generated_solutions' not in st.session_state:
    st.session_state.generated_solutions = []

# تعيين حالة الدخول مباشرة
st.session_state.logged_in = True
st.session_state.username = 'guest'
st.session_state.user_role = 'system_admin' # تعيين دور افتراضي
st.session_state.user_name = 'زائر' # تعيين اسم افتراضي

# قاموس الترجمة
translations = {
    'ar': {
        'title': 'التوأم الرقمي العصبي الذكي',
        'subtitle': 'نظام متقدم لمراقبة وتحليل البيانات في الوقت الفعلي',
        'login': 'تسجيل الدخول',
        'username': 'اسم المستخدم',
        'password': 'كلمة المرور',
        'login_button': 'تسجيل الدخول',
        'logout': 'تسجيل الخروج',
        'main_dashboard': 'لوحة البيانات الرئيسية',
        'predictive_analysis': 'التحليل التنبؤي المتقدم',
        'smart_solutions': 'الحلول الذكية والتوصيات',
        'sensor_locations': 'مواقع المستشعرات',
        'alerts': 'التنبيهات الفورية',
        'reports': 'التقارير والإحصائيات',
        'settings': 'الإعدادات والتخصيص',
        'about': 'حول المشروع والفريق',
        'temperature': 'درجة الحرارة',
        'pressure': 'الضغط',
        'vibration': 'الاهتزاز',
        'methane': 'غاز الميثان',
        'last_reading': 'آخر قراءة',
        'export_csv': 'تصدير CSV',
        'export_pdf': 'تصدير تقرير PDF',
        'generate_solutions': 'توليد حلول ذكية جديدة',
        'language_selection': 'اختيار اللغة',
        'theme_selection': 'اختيار المظهر',
        'welcome': 'مرحباً',
        'role': 'الدور',
        'system_admin': 'مدير النظام',
        'supervisor': 'مشرف',
        'operator': 'مشغل',
        'viewer': 'مراقب',
        'live_data': 'البيانات الحية',
        'auto_refresh': 'تحديث تلقائي',
        'last_updated': 'آخر تحديث'
    },
    'en': {
        'title': 'Smart Neural Digital Twin',
        'subtitle': 'Advanced Real-time Data Monitoring and Analysis System',
        'login': 'Login',
        'username': 'Username',
        'password': 'Password',
        'login_button': 'Login',
        'logout': 'Logout',
        'main_dashboard': 'Main Dashboard',
        'predictive_analysis': 'Advanced Predictive Analysis',
        'smart_solutions': 'Smart Solutions & Recommendations',
        'sensor_locations': 'Sensor Locations',
        'alerts': 'Real-time Alerts',
        'reports': 'Reports & Statistics',
        'settings': 'Settings & Customization',
        'about': 'About Project & Team',
        'temperature': 'Temperature',
        'pressure': 'Pressure',
        'vibration': 'Vibration',
        'methane': 'Methane Gas',
        'last_reading': 'Last Reading',
        'export_csv': 'Export CSV',
        'export_pdf': 'Export PDF Report',
        'generate_solutions': 'Generate New Smart Solutions',
        'language_selection': 'Language Selection',
        'theme_selection': 'Theme Selection',
        'welcome': 'Welcome',
        'role': 'Role',
        'system_admin': 'System Administrator',
        'supervisor': 'Supervisor',
        'operator': 'Operator',
        'viewer': 'Viewer',
        'live_data': 'Live Data',
        'auto_refresh': 'Auto Refresh',
        'last_updated': 'Last Updated'
    }
}

def get_text(key):
    return translations[st.session_state.language][key]

# دالة توليد البيانات الحية
@st.cache_data(ttl=5)  # تحديث كل 5 ثواني
def generate_live_sensor_data():
    current_time = datetime.now()
    
    # محاكاة تغيرات طبيعية في البيانات
    base_temp = 25 + 10 * np.sin(current_time.hour * 2 * np.pi / 24)
    base_pressure = 100 + 20 * np.sin(current_time.minute * 2 * np.pi / 60)
    base_vibration = 0.3 + 0.2 * np.sin(current_time.second * 2 * np.pi / 60)
    base_methane = 30 + 15 * np.sin(current_time.hour * 2 * np.pi / 12)
    
    # إضافة تشويش عشوائي
    data = {
        'temperature': base_temp + random.gauss(0, 2),
        'pressure': base_pressure + random.gauss(0, 5),
        'vibration': base_vibration + random.gauss(0, 0.1),
        'methane': base_methane + random.gauss(0, 3),
        'timestamp': current_time
    }
    
    return data

# دالة توليد البيانات التاريخية
@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', end='2024-12-31', freq='h')
    data = {
        'timestamp': dates,
        'temperature': 25 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7)) + np.random.normal(0, 2, len(dates)),
        'pressure': 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*30)) + np.random.normal(0, 5, len(dates)),
        'vibration': 0.3 + 0.2 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*3)) + np.random.normal(0, 0.1, len(dates)),
        'methane': 30 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*14)) + np.random.normal(0, 3, len(dates))
    }
    return pd.DataFrame(data)

# دالة إنشاء تقرير PDF
def create_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # عنوان التقرير
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue
    )
    story.append(Paragraph("تقرير التوأم الرقمي العصبي الذكي", title_style))
    story.append(Spacer(1, 12))
    
    # معلومات التقرير
    info_style = styles['Normal']
    story.append(Paragraph(f"تاريخ التقرير: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Paragraph(f"منشئ التقرير: {st.session_state.user_name} ({get_text(st.session_state.user_role)})", info_style))
    story.append(Spacer(1, 20))
    
    # البيانات الحية
    live_data = generate_live_sensor_data()
    story.append(Paragraph("البيانات الحية الحالية", styles['Heading2']))
    data_table = [
        ['المستشعر', 'القيمة الحالية', 'الوحدة'],
        ['درجة الحرارة', f"{live_data['temperature']:.1f}", '°C'],
        ['الضغط', f"{live_data['pressure']:.1f}", 'psi'],
        ['الاهتزاز', f"{live_data['vibration']:.2f}", 'g'],
        ['غاز الميثان', f"{live_data['methane']:.1f}", 'ppm']
    ]
    
    table = Table(data_table)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# دالة الحصول على ألوان المظهر
def get_theme_colors(theme):
    themes = {
        'purple': {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'accent': '#f093fb',
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        },
        'ocean': {
            'primary': '#2196F3',
            'secondary': '#21CBF3',
            'accent': '#03DAC6',
            'background': 'linear-gradient(135deg, #2196F3 0%, #21CBF3 100%)'
        },
        'sunset': {
            'primary': '#FF6B6B',
            'secondary': '#FF8E53',
            'accent': '#FF6B9D',
            'background': 'linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%)'
        },
        'forest': {
            'primary': '#4CAF50',
            'secondary': '#8BC34A',
            'accent': '#CDDC39',
            'background': 'linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)'
        },
        'dark': {
            'primary': '#424242',
            'secondary': '#616161',
            'accent': '#9E9E9E',
            'background': 'linear-gradient(135deg, #424242 0%, #616161 100%)'
        }
    }
    return themes.get(theme, themes['purple'])

# CSS مخصص للتصميم المتجاوب
def get_custom_css():
    theme_colors = get_theme_colors(st.session_state.theme)
    
    return f"""
<style>
    /* تصميم متجاوب للجوال */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: 2rem !important;
        }}
        .metric-card {{
            margin: 0.5rem 0 !important;
            padding: 1rem !important;
        }}
        .sidebar-button {{
            padding: 0.5rem !important;
            font-size: 0.9rem !important;
        }}
    }}
    
    .main-header {{
        background: {theme_colors['background']};
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .main-title {{
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
    }}
    
    .login-container {{
        background: {theme_colors['background']};
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 2rem auto;
    }}
    
    .sidebar-info {{
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }}
    
    .sidebar-button {{
        background: {theme_colors['background']};
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.25rem 0;
        width: 100%;
        text-align: right;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 1rem;
    }}
    
    .sidebar-button:hover {{
        transform: translateX(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .sidebar-button.active {{
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #333;
        font-weight: bold;
    }}
    
    .language-button {{
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .language-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .language-button.active {{
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #333;
    }}
    
    .theme-button {{
        background: {theme_colors['background']};
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }}
    
    .theme-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .theme-button.active {{
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #333;
    }}
    
    .export-button {{
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 0.25rem;
        transition: all 0.3s ease;
    }}
    
    .export-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .solution-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }}
    
    .success-message {{
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }}
    
    .live-indicator {{
        background: linear-gradient(45deg, #ff4444, #ff6666);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
        100% {{ opacity: 1; }}
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .about-section {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }}
    
    .aramco-logo {{
        background: linear-gradient(45deg, #0066cc, #004499);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }}
    
    .team-member {{
        background: {theme_colors['background']};
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .tech-spec {{
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }}
    
    .feature-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }}
    
    .project-stats-card {{
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .vision-section {{
        background: linear-gradient(45deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }}
    
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

# واجهة المستخدم
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Main application logic, assuming logged in
if st.session_state.logged_in:
    with st.sidebar:
        st.markdown(f"""<div class="sidebar-info">
            <h3>{get_text('welcome')}, {st.session_state.user_name}</h3>
            <p>{get_text('role')}: {get_text(st.session_state.user_role)}</p>
        </div>""", unsafe_allow_html=True)

        # أزرار التنقل الفوري
        if st.button(get_text('main_dashboard'), key='main_dashboard_btn', help=get_text('main_dashboard')):
            st.session_state.current_page = 'main_dashboard'
        if st.button(get_text('predictive_analysis'), key='predictive_analysis_btn', help=get_text('predictive_analysis')):
            st.session_state.current_page = 'predictive_analysis'
        if st.button(get_text('smart_solutions'), key='smart_solutions_btn', help=get_text('smart_solutions')):
            st.session_state.current_page = 'smart_solutions'
        if st.button(get_text('sensor_locations'), key='sensor_locations_btn', help=get_text('sensor_locations')):
            st.session_state.current_page = 'sensor_locations'
        if st.button(get_text('alerts'), key='alerts_btn', help=get_text('alerts')):
            st.session_state.current_page = 'alerts'
        if st.button(get_text('reports'), key='reports_btn', help=get_text('reports')):
            st.session_state.current_page = 'reports'
        if st.button(get_text('settings'), key='settings_btn', help=get_text('settings')):
            st.session_state.current_page = 'settings'
        if st.button(get_text('about'), key='about_btn', help=get_text('about')):
            st.session_state.current_page = 'about'

        st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)

        # أزرار اختيار اللغة
        st.markdown(f"<h4 style='color:white;'>{get_text('language_selection')}</h4>", unsafe_allow_html=True)
        col_lang1, col_lang2 = st.columns(2)
        with col_lang1:
            if st.button('العربية', key='lang_ar_btn', help='تغيير اللغة إلى العربية'):
                st.session_state.language = 'ar' # تم التعديل
                st.rerun()
        with col_lang2:
            if st.button('English', key='lang_en_btn', help='Change language to English'):
                st.session_state.language = 'en'
                st.rerun()

        # أزرار اختيار المظهر
        st.markdown(f"<h4 style='color:white;'>{get_text('theme_selection')}</h4>", unsafe_allow_html=True)
        col_theme1, col_theme2, col_theme3 = st.columns(3)
        with col_theme1:
            if st.button('Purple', key='theme_purple_btn', help='تغيير المظهر إلى البنفسجي'):
                st.session_state.theme = 'purple'
                st.rerun()
            if st.button('Ocean', key='theme_ocean_btn', help='تغيير المظهر إلى المحيط'):
                st.session_state.theme = 'ocean'
                st.rerun()
        with col_theme2:
            if st.button('Sunset', key='theme_sunset_btn', help='تغيير المظهر إلى الغروب'):
                st.session_state.theme = 'sunset'
                st.rerun()
        with col_theme3:
            if st.button('Forest', key='theme_forest_btn', help='تغيير المظهر إلى الغابة'):
                st.session_state.theme = 'forest'
                st.rerun()
            if st.button('Dark', key='theme_dark_btn', help='تغيير المظهر إلى الداكن'):
                st.session_state.theme = 'dark'
                st.rerun()

        st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)

    # محتوى الصفحة الرئيسية
    def main_dashboard():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('title')}</h1>
            <p style='color:white; font-size:1.2rem;'>{get_text('subtitle')}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='live-indicator'>{get_text('live_data')} - {get_text('auto_refresh')}</div>", unsafe_allow_html=True)
        st.write(f"<p style='text-align: right; font-size: 0.9rem; color: #aaa;'>{get_text('last_updated')}: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

        live_data = generate_live_sensor_data()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h3>{get_text('temperature')}</h3>
                <p style='font-size: 2rem; font-weight: bold;'>{live_data['temperature']:.1f} °C</p>
                <p>{get_text('last_reading')}</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h3>{get_text('pressure')}</h3>
                <p style='font-size: 2rem; font-weight: bold;'>{live_data['pressure']:.1f} psi</p>
                <p>{get_text('last_reading')}</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <h3>{get_text('vibration')}</h3>
                <p style='font-size: 2rem; font-weight: bold;'>{live_data['vibration']:.2f} g</p>
                <p>{get_text('last_reading')}</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <h3>{get_text('methane')}</h3>
                <p style='font-size: 2rem; font-weight: bold;'>{live_data['methane']:.1f} ppm</p>
                <p>{get_text('last_reading')}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # رسوم بيانية للبيانات التاريخية
        st.subheader(get_text('predictive_analysis'))
        historical_df = generate_historical_data()

        # درجة الحرارة
        fig_temp = px.line(historical_df, x='timestamp', y='temperature', title=get_text('temperature'))
        st.plotly_chart(fig_temp, use_container_width=True)

        # الضغط
        fig_pressure = px.line(historical_df, x='timestamp', y='pressure', title=get_text('pressure'))
        st.plotly_chart(fig_pressure, use_container_width=True)

        # الاهتزاز
        fig_vibration = px.line(historical_df, x='timestamp', y='vibration', title=get_text('vibration'))
        st.plotly_chart(fig_vibration, use_container_width=True)

        # غاز الميثان
        fig_methane = px.line(historical_df, x='timestamp', y='methane', title=get_text('methane'))
        st.plotly_chart(fig_methane, use_container_width=True)

        # أزرار التصدير
        st.markdown("<br>", unsafe_allow_html=True)
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv_data = historical_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=get_text('export_csv'),
                data=csv_data,
                file_name='sensor_data.csv',
                mime='text/csv',
                help='تصدير البيانات التاريخية إلى ملف CSV'
            )
        with col_export2:
            pdf_report = create_pdf_report()
            st.download_button(
                label=get_text('export_pdf'),
                data=pdf_report,
                file_name='sensor_report.pdf',
                mime='application/pdf',
                help='تصدير تقرير PDF للبيانات الحالية'
            )

    # دالة التحليل التنبؤي المتقدم (Placeholder)
    def predictive_analysis():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('predictive_analysis')}</h1>
            <p style='color:white; font-size:1.2rem;'>هذه الصفحة مخصصة لعرض التحليلات التنبؤية المتقدمة.</p>
        </div>""", unsafe_allow_html=True)
        st.write("سيتم إضافة محتوى التحليل التنبؤي هنا لاحقاً.")

    # دالة الحلول الذكية والتوصيات
    def smart_solutions():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('smart_solutions')}</h1>
            <p style='color:white; font-size:1.2rem;'>هذه الصفحة تعرض الحلول الذكية والتوصيات بناءً على بيانات المستشعرات.</p>
        </div>""", unsafe_allow_html=True)

        st.button(get_text('generate_solutions'), on_click=generate_smart_solution)

        if st.session_state.generated_solutions:
            st.subheader("الحلول المقترحة:")
            for i, solution in enumerate(st.session_state.generated_solutions):
                st.markdown(f"""<div class="solution-card">
                    <h4>حل مقترح #{i+1}</h4>
                    <p>{solution}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("لا توجد حلول مقترحة حتى الآن. اضغط على 'توليد حلول ذكية جديدة' لتوليد بعض الحلول.")

    def generate_smart_solution():
        # محاكاة توليد حلول ذكية
        solutions = [
            "تحسين معايرة مستشعر درجة الحرارة لزيادة الدقة بنسبة 5%.",
            "فحص نظام التهوية لتقليل مستويات غاز الميثان بنسبة 10%.",
            "جدولة صيانة وقائية للمعدات التي تظهر اهتزازات عالية.",
            "تحليل بيانات الضغط لتحديد الأنماط التي قد تشير إلى تسربات محتملة.",
            "تطبيق خوارزميات تعلم الآلة للتنبؤ بالأعطال قبل حدوثها."
        ]
        st.session_state.generated_solutions.append(random.choice(solutions))
        st.success("تم توليد حل ذكي جديد!")

    # دالة مواقع المستشعرات
    def sensor_locations():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('sensor_locations')}</h1>
            <p style='color:white; font-size:1.2rem;'>عرض مواقع المستشعرات على الخريطة.</p>
        </div>""", unsafe_allow_html=True)

        # بيانات مواقع المستشعرات (مثال)
        sensor_data = {
            'SensorID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'Latitude': [24.7136, 24.7500, 24.6800, 24.7200, 24.7000],
            'Longitude': [46.6753, 46.7000, 46.6500, 46.6800, 46.6600],
            'Location': ['Riyadh City Center', 'Industrial Area 1', 'Residential Zone A', 'Commercial Hub', 'Data Center 1'],
            'Status': ['Operational', 'Operational', 'Warning', 'Operational', 'Offline']
        }
        sensors_df = pd.DataFrame(sensor_data)

        # إنشاء خريطة Folium
        m = folium.Map(location=[sensors_df['Latitude'].mean(), sensors_df['Longitude'].mean()], zoom_start=12)

        # إضافة مجموعات علامات
        marker_cluster = MarkerCluster().add_to(m)

        # إضافة علامات للمستشعرات
        for idx, row in sensors_df.iterrows():
            color = 'green' if row['Status'] == 'Operational' else 'orange' if row['Status'] == 'Warning' else 'red'
            icon = folium.Icon(color=color, icon='info-sign')
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"<b>Sensor ID:</b> {row['SensorID']}<br><b>Location:</b> {row['Location']}<br><b>Status:</b> {row['Status']}",
                icon=icon
            ).add_to(marker_cluster)
        
        # عرض الخريطة في Streamlit
        st_folium(m, width=1000, height=600)

    # دالة التنبيهات الفورية (Placeholder)
    def alerts():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('alerts')}</h1>
            <p style='color:white; font-size:1.2rem;'>هذه الصفحة مخصصة لعرض التنبيهات الفورية.</p>
        </div>""", unsafe_allow_html=True)
        st.write("سيتم عرض التنبيهات هنا لاحقاً.")

    # دالة التقارير والإحصائيات (Placeholder)
    def reports():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('reports')}</h1>
            <p style='color:white; font-size:1.2rem;'>هذه الصفحة مخصصة لعرض التقارير والإحصائيات.</p>
        </div>""", unsafe_allow_html=True)
        st.write("سيتم توليد التقارير والإحصائيات هنا لاحقاً.")

    # دالة الإعدادات والتخصيص (Placeholder)
    def settings():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('settings')}</h1>
            <p style='color:white; font-size:1.2rem;'>هذه الصفحة مخصصة لإعدادات التطبيق وتخصيصه.</p>
        </div>""", unsafe_allow_html=True)
        st.write("سيتم توفير خيارات الإعدادات هنا لاحقاً.")

    # دالة حول المشروع والفريق (Placeholder)
    def about():
        st.markdown(f"""<div class="main-header">
            <h1 class="main-title">{get_text('about')}</h1>
            <p style='color:white; font-size:1.2rem;'>معلومات حول المشروع والفريق.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="about-section">
            <h2>حول التوأم الرقمي العصبي الذكي</h2>
            <p>يهدف هذا المشروع إلى تطوير نظام متكامل لمراقبة وتحليل البيانات في الوقت الفعلي باستخدام تقنيات التوأم الرقمي والذكاء الاصطناعي. يوفر النظام رؤى قيمة حول أداء الأنظمة والمعدات، ويساعد في اتخاذ قرارات مستنيرة لتحسين الكفاءة وتقليل المخاطر.</p>
            
            <h3>الرؤية</h3>
            <div class="vision-section">
                <p>أن نكون الرواد في توفير حلول التوأم الرقمي المبتكرة التي تمكن الصناعات من تحقيق أقصى قدر من الكفاءة التشغيلية والاستدامة.</p>
            </div>

            <h3>الميزات الرئيسية</h3>
            <div class="feature-card">
                <h4>مراقبة البيانات في الوقت الفعلي</h4>
                <p>عرض فوري لبيانات المستشعرات الحية مع تحديثات مستمرة.</p>
            </div>
            <div class="feature-card">
                <h4>تحليل تنبؤي متقدم</h4>
                <p>استخدام نماذج الذكاء الاصطناعي للتنبؤ بالأعطال والمشكلات المحتملة.</p>
            </div>
            <div class="feature-card">
                <h4>حلول ذكية وتوصيات</h4>
                <p>تقديم توصيات قابلة للتنفيذ لتحسين الأداء والصيانة.</p>
            </div>
            <div class="feature-card">
                <h4>تقارير شاملة ولوحات تحكم تفاعلية</h4>
                <p>توليد تقارير مفصلة ولوحات تحكم سهلة الاستخدام لاتخاذ القرارات.</p>
            </div>

            <h3>التقنيات المستخدمة</h3>
            <div class="tech-spec">
                <p><strong>الواجهة الأمامية:</strong> Streamlit, HTML, CSS</p>
            </div>
            <div class="tech-spec">
                <p><strong>معالجة البيانات:</strong> Pandas, NumPy</p>
            </div>
            <div class="tech-spec">
                <p><strong>الرسوم البيانية:</strong> Plotly</p>
            </div>
            <div class="tech-spec">
                <p><strong>توليد التقارير:</strong> ReportLab</p>
            </div>
            <div class="tech-spec">
                <p><strong>الخرائط:</strong> Folium, Streamlit-Folium</p>
            </div>

            <h3>إحصائيات المشروع</h3>
            <div class="project-stats-card">
                <p>عدد أسطر الكود: <b>أكثر من 1000</b></p>
                <p>عدد الميزات: <b>7+</b></p>
                <p>وقت التطوير: <b>3 أسابيع</b></p>
            </div>

            <h3>الشكر والتقدير</h3>
            <div class="aramco-logo">
                <p>هذا المشروع تم تطويره بدعم من <b>أرامكو السعودية</b></p>
            </div>

            <h2>فريق العمل</h2>
            <div class="team-member">
                <h3>[اسم العضو 1]</h3>
                <p>الدور: [دور العضو 1]</p>
            </div>
            <div class="team-member">
                <h3>[اسم العضو 2]</h3>
                <p>الدور: [دور العضو 2]</p>
            </div>
            <div class="team-member">
                <h3>[اسم العضو 3]</h3>
                <p>الدور: [دور العضو 3]</p>
            </div>
        </div>""", unsafe_allow_html=True)

    # توجيه الصفحات
    if st.session_state.current_page == 'main_dashboard':
        main_dashboard()
    elif st.session_state.current_page == 'predictive_analysis':
        predictive_analysis()
    elif st.session_state.current_page == 'smart_solutions':
        smart_solutions()
    elif st.session_state.current_page == 'sensor_locations':
        sensor_locations()
    elif st.session_state.current_page == 'alerts':
        alerts()
    elif st.session_state.current_page == 'reports':
        reports()
    elif st.session_state.current_page == 'settings':
        settings()
    elif st.session_state.current_page == 'about':
        about()

else:
    # صفحة تسجيل الدخول (إذا لم يكن المستخدم مسجلاً)
    st.markdown(f"""<div class="login-container">
        <h2 style='text-align: center; color: white;'>{get_text('login')}</h2>
        <form>
            <label style='color:white;'>{get_text('username')}</label>
            <input type='text' placeholder='{get_text('username')}' style='width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ddd;'>
            <label style='color:white;'>{get_text('password')}</label>
            <input type='password' placeholder='{get_text('password')}' style='width: 100%; padding: 10px; margin-bottom: 20px; border-radius: 5px; border: 1px solid #ddd;'>
            <button type='submit' style='width: 100%; padding: 10px; background: {get_theme_colors(st.session_state.theme)['primary']}; color: white; border: none; border-radius: 5px; cursor: pointer;'>{get_text('login_button')}</button>
        </form>
    </div>""", unsafe_allow_html=True)


