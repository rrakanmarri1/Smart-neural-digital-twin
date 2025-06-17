import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

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
        dates = pd.date_range('2024-01-01', periods=1000, freq='h')
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
    st.session_state.current_page = 'Dashboard'

# النصوص متعددة اللغات
texts = {
    'العربية': {
        'title': 'التوأم الرقمي العصبي الذكي',
        'subtitle': 'نظام متقدم لمراقبة وتحليل البيانات في الوقت الفعلي',
        'main_menu': '🧠 القائمة الرئيسية',
        'dashboard': '📊 لوحة البيانات',
        'simulation': '🔄 المحاكاة',
        'analytics': '📈 التحليل التنبؤي',
        'solutions': '💡 الحلول الذكية',
        'settings': '⚙️ الإعدادات',
        'about': 'ℹ️ حول المشروع',
        'reports': '📋 التقارير',
        'alerts': '🚨 التنبيهات',
        'maintenance': '🔧 الصيانة',
        'users': '👥 المستخدمين',
        'temperature': '🌡️ درجة الحرارة',
        'pressure': '📏 الضغط',
        'vibration': '📳 الاهتزاز',
        'gas': '💨 غاز الميثان',
        'last_reading': 'آخر قراءة',
        'language_settings': '🌐 إعدادات اللغة',
        'theme_settings': '🎨 إعدادات المظهر',
        'select_language': 'اختر اللغة:',
        'select_theme': 'اختر المظهر:',
        'ocean': '🌊 المحيط',
        'forest': '🌲 الغابة',
        'sunset': '🌅 الغروب',
        'purple': '💜 البنفسجي',
        'slate': '⚫ الرمادي',
        'contact_info': '📞 معلومات التواصل',
        'developer1': '👨‍💻 راكان المري',
        'developer2': '👨‍💻 عبدالرحمن الزهراني',
        'main_developer': 'مطور رئيسي',
        'co_developer': 'مطور مشارك',
        'email': '📧 البريد الإلكتروني:',
        'phone': '📱 الهاتف:',
        'company_info': '🏢 معلومات الشركة',
        'company_desc': 'تم تطوير هذا النظام كجزء من مشاريع الابتكار في أرامكو السعودية لتطوير حلول ذكية لمراقبة وتحليل البيانات الصناعية.',
        'tech_support': '📧 للدعم الفني:',
        'support_desc': 'للاستفسارات والدعم الفني، يرجى التواصل مع فريق التطوير عبر البريد الإلكتروني أو الهاتف المذكور أعلاه.',
        'features': '✨ الميزات الرئيسية:',
        'technologies': '🛠️ التقنيات المستخدمة:',
        'copyright': '© 2024 التوأم الرقمي العصبي الذكي - أرامكو السعودية - جميع الحقوق محفوظة',
        'project_description': 'نظام متطور لمراقبة وتحليل البيانات في الوقت الفعلي باستخدام تقنيات الذكاء الاصطناعي والتعلم الآلي.',
        'feature_list': [
            'مراقبة البيانات في الوقت الفعلي',
            'كشف الشذوذ التلقائي',
            'التحليل التنبؤي المتقدم',
            'واجهة مستخدم تفاعلية',
            'دعم متعدد اللغات',
            'مظاهر متنوعة قابلة للتخصيص'
        ],
        'tech_list': [
            'Python & Streamlit',
            'Plotly للرسوم البيانية',
            'Pandas لمعالجة البيانات',
            'تقنيات التعلم الآلي'
        ]
    },
    'English': {
        'title': 'Smart Neural Digital Twin',
        'subtitle': 'Advanced system for real-time data monitoring and analysis',
        'main_menu': '🧠 Main Menu',
        'dashboard': '📊 Dashboard',
        'simulation': '🔄 Simulation',
        'analytics': '📈 Predictive Analytics',
        'solutions': '💡 Smart Solutions',
        'settings': '⚙️ Settings',
        'about': 'ℹ️ About Project',
        'reports': '📋 Reports',
        'alerts': '🚨 Alerts',
        'maintenance': '🔧 Maintenance',
        'users': '👥 Users',
        'temperature': '🌡️ Temperature',
        'pressure': '📏 Pressure',
        'vibration': '📳 Vibration',
        'gas': '💨 Methane Gas',
        'last_reading': 'Last Reading',
        'language_settings': '🌐 Language Settings',
        'theme_settings': '🎨 Theme Settings',
        'select_language': 'Select Language:',
        'select_theme': 'Select Theme:',
        'ocean': '🌊 Ocean',
        'forest': '🌲 Forest',
        'sunset': '🌅 Sunset',
        'purple': '💜 Purple',
        'slate': '⚫ Slate',
        'contact_info': '📞 Contact Information',
        'developer1': '👨‍💻 Rakan Al-Marri',
        'developer2': '👨‍💻 Abdulrahman Al-Zahrani',
        'main_developer': 'Lead Developer',
        'co_developer': 'Co-Developer',
        'email': '📧 Email:',
        'phone': '📱 Phone:',
        'company_info': '🏢 Company Information',
        'company_desc': 'This system was developed as part of innovation projects at Saudi Aramco to develop smart solutions for monitoring and analyzing industrial data.',
        'tech_support': '📧 Technical Support:',
        'support_desc': 'For inquiries and technical support, please contact the development team via email or phone mentioned above.',
        'features': '✨ Key Features:',
        'technologies': '🛠️ Technologies Used:',
        'copyright': '© 2024 Smart Neural Digital Twin - Saudi Aramco - All Rights Reserved',
        'project_description': 'An advanced system for real-time data monitoring and analysis using artificial intelligence and machine learning technologies.',
        'feature_list': [
            'Real-time data monitoring',
            'Automatic anomaly detection',
            'Advanced predictive analytics',
            'Interactive user interface',
            'Multi-language support',
            'Customizable themes'
        ],
        'tech_list': [
            'Python & Streamlit',
            'Plotly for charts',
            'Pandas for data processing',
            'Machine learning techniques'
        ]
    }
}

# الحصول على النصوص الحالية
current_texts = texts[st.session_state.language]

# ألوان المظاهر
themes = {
    "Ocean": {"primary": "#1E90FF", "secondary": "#4169E1", "background": "#1a1a1a", "text": "#ffffff", "sidebar": "#2d2d2d", "card": "#333333", "border": "#1E90FF"},
    "Forest": {"primary": "#32CD32", "secondary": "#228B22", "background": "#1a1a1a", "text": "#ffffff", "sidebar": "#2d2d2d", "card": "#333333", "border": "#32CD32"},
    "Sunset": {"primary": "#FF6347", "secondary": "#FF4500", "background": "#1a1a1a", "text": "#ffffff", "sidebar": "#2d2d2d", "card": "#333333", "border": "#FF6347"},
    "Purple": {"primary": "#9370DB", "secondary": "#8A2BE2", "background": "#1a1a1a", "text": "#ffffff", "sidebar": "#2d2d2d", "card": "#333333", "border": "#9370DB"},
    "Slate": {"primary": "#708090", "secondary": "#2F4F4F", "background": "#1a1a1a", "text": "#ffffff", "sidebar": "#2d2d2d", "card": "#333333", "border": "#708090"}
}

current_theme = themes[st.session_state.theme]

# CSS مخصص
st.markdown(f"""
<style>
    .stApp {{
        background-color: {current_theme['background']};
        color: {current_theme['text']};
    }}
    
    .css-1d391kg {{
        background-color: {current_theme['sidebar']};
        border-right: 3px solid {current_theme['primary']};
    }}
    
    .css-1d391kg .stMarkdown, .css-1d391kg .stSelectbox label, .css-1d391kg .stRadio label {{
        color: {current_theme['text']} !important;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']});
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    
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
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        background: {current_theme['primary']};
        color: white;
        transform: translateX(-5px);
    }}
    
    .settings-box {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {current_theme['border']};
        margin: 1rem 0;
    }}
    
    .stSelectbox > div > div {{
        background-color: {current_theme['card']};
        color: {current_theme['text']};
        border: 2px solid {current_theme['primary']};
        border-radius: 8px;
    }}
    
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div {{
        color: {current_theme['text']} !important;
    }}
    
    .js-plotly-plot {{
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        background-color: {current_theme['card']};
    }}
    
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
    
    .section-title {{
        color: {current_theme['primary']};
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        border-bottom: 2px solid {current_theme['primary']};
        padding-bottom: 0.5rem;
    }}
    
    .main .block-container {{
        background-color: {current_theme['background']};
        color: {current_theme['text']};
    }}
    
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
    
    .contact-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid {current_theme['primary']};
        margin: 1rem 0;
        text-align: center;
    }}
    
    .info-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {current_theme['primary']};
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }}
    
    .feature-item {{
        color: {current_theme['text']};
        margin: 0.5rem 0;
        padding: 0.3rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي المحسن
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: {current_theme['primary']}; color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h2>{current_texts['main_menu']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # قائمة التنقل المحسنة
    pages = {
        "Dashboard": current_texts['dashboard'],
        "Simulation": current_texts['simulation'],
        "Analytics": current_texts['analytics'],
        "Solutions": current_texts['solutions'],
        "Reports": current_texts['reports'],
        "Alerts": current_texts['alerts'],
        "Maintenance": current_texts['maintenance'],
        "Users": current_texts['users'],
        "Settings": current_texts['settings'],
        "About": current_texts['about']
    }
    
    for page_key, page_name in pages.items():
        if st.button(page_name, key=f"btn_{page_key}", use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()

# العنوان الرئيسي
st.markdown(f"""
<div class="main-header">
    <h1>{current_texts['title']}</h1>
    <p>{current_texts['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# المحتوى الرئيسي حسب الصفحة المختارة
current_page = st.session_state.current_page

if current_page == "Dashboard":
    st.markdown(f'<div class="section-title">{current_texts["dashboard"]}</div>', unsafe_allow_html=True)
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">{current_texts['temperature']}</h4>
            <h2 style="color: {current_theme['text']};">{latest.temp:.1f}°C</h2>
            <small style="color: {current_theme['text']};">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">{current_texts['pressure']}</h4>
            <h2 style="color: {current_theme['text']};">{latest.pressure:.1f} PSI</h2>
            <small style="color: {current_theme['text']};">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">{current_texts['vibration']}</h4>
            <h2 style="color: {current_theme['text']};">{latest.vibration:.2f} g</h2>
            <small style="color: {current_theme['text']};">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">{current_texts['gas']}</h4>
            <h2 style="color: {current_theme['text']};">{latest.gas:.1f} ppm</h2>
            <small style="color: {current_theme['text']};">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">📈 Data Trends</div>', unsafe_allow_html=True)
        fig = px.line(df.tail(100), x="timestamp", y=["temp", "pressure", "vibration", "gas"],
                     labels={"timestamp": "Time", "value": "Value", "variable": "Variable"},
                     color_discrete_sequence=[current_theme['primary'], current_theme['secondary'], '#FF6B6B', '#4ECDC4'])
        fig.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], font_color=current_theme['text'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-title">🗺️ Heat Map</div>', unsafe_allow_html=True)
        heat_data = df.pivot_table(index=df.timestamp.dt.hour, columns=df.timestamp.dt.day, values="temp", aggfunc="mean")
        fig2 = go.Figure(data=go.Heatmap(z=heat_data.values, x=heat_data.columns, y=heat_data.index, colorscale="Viridis"))
        fig2.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], font_color=current_theme['text'],
                          xaxis_title="Day", yaxis_title="Hour")
        st.plotly_chart(fig2, use_container_width=True)

elif current_page == "Settings":
    st.markdown(f'<div class="section-title">{current_texts["settings"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">{current_texts["language_settings"]}</div>', unsafe_allow_html=True)
        
        language_options = ["العربية", "English"]
        selected_lang = st.selectbox(
            current_texts["select_language"],
            language_options,
            index=language_options.index(st.session_state.language),
            key="lang_settings"
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.success(f"Language changed to: {selected_lang}")
            st.rerun()
        
        st.markdown(f'<div class="section-title">{current_texts["theme_settings"]}</div>', unsafe_allow_html=True)
        
        theme_options = ["Ocean", "Forest", "Sunset", "Purple", "Slate"]
        theme_labels = {
            "Ocean": current_texts["ocean"],
            "Forest": current_texts["forest"],
            "Sunset": current_texts["sunset"],
            "Purple": current_texts["purple"],
            "Slate": current_texts["slate"]
        }
        
        selected_theme = st.selectbox(
            current_texts["select_theme"],
            theme_options,
            index=theme_options.index(st.session_state.theme),
            format_func=lambda x: theme_labels[x],
            key="theme_settings"
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.success(f"Theme changed to: {theme_labels[selected_theme]}")
            st.rerun()
    
    with col2:
        st.markdown(f'<div class="section-title">🔔 Alert Settings</div>', unsafe_allow_html=True)
        enable_alerts = st.checkbox("Enable Alerts", value=True)
        alert_threshold = st.slider("Alert Threshold", 0.1, 2.0, 1.0)
        
        st.markdown(f'<div class="section-title">💾 Data Settings</div>', unsafe_allow_html=True)
        data_retention = st.selectbox("Data Retention Period", ["1 Week", "1 Month", "3 Months", "1 Year"])

elif current_page == "About":
    st.markdown(f'<div class="section-title">{current_texts["about"]}</div>', unsafe_allow_html=True)
    
    # معلومات المشروع الأساسية
    st.markdown(f"""
    <div class="info-card">
        <h3 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['title']}</h3>
        <p style="color: {current_theme['text']}; font-size: 1.1rem; line-height: 1.6;">{current_texts['project_description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # الميزات الرئيسية
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['features']}</h4>
    """, unsafe_allow_html=True)
    
    for feature in current_texts['feature_list']:
        st.markdown(f'<div class="feature-item">• {feature}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # التقنيات المستخدمة
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['technologies']}</h4>
    """, unsafe_allow_html=True)
    
    for tech in current_texts['tech_list']:
        st.markdown(f'<div class="feature-item">• {tech}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # معلومات التواصل
    st.markdown(f'<div class="section-title">{current_texts["contact_info"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="contact-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['developer1']}</h4>
            <p style="color: {current_theme['text']}; margin-bottom: 0.5rem;">{current_texts['main_developer']}</p>
            <div style="text-align: left; margin-top: 1rem;">
                <p style="color: {current_theme['text']}; margin: 0.3rem 0;"><strong>{current_texts['email']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.3rem 0;">rakan.almarri.2@aramco.com</p>
                <p style="color: {current_theme['text']}; margin: 0.3rem 0;"><strong>{current_texts['phone']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.3rem 0;">0532559664</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="contact-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['developer2']}</h4>
            <p style="color: {current_theme['text']}; margin-bottom: 0.5rem;">{current_texts['co_developer']}</p>
            <div style="text-align: left; margin-top: 1rem;">
                <p style="color: {current_theme['text']}; margin: 0.3rem 0;"><strong>{current_texts['email']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.3rem 0;">abdulrahman.alzhrani.1@aramco.com</p>
                <p style="color: {current_theme['text']}; margin: 0.3rem 0;"><strong>{current_texts['phone']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.3rem 0;">0549202674</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # معلومات الشركة
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['company_info']}</h4>
        <p style="color: {current_theme['text']}; line-height: 1.6; margin-bottom: 1rem;">{current_texts['company_desc']}</p>
        
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['tech_support']}</h4>
        <p style="color: {current_theme['text']}; line-height: 1.6;">{current_texts['support_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

# إضافة صفحات جديدة
elif current_page == "Reports":
    st.markdown(f'<div class="section-title">{current_texts["reports"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📊 Daily Report</h4>
            <p style="color: {current_theme['text']};">Generate comprehensive daily performance reports</p>
            <button style="background: {current_theme['primary']}; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-top: 10px;">Generate Report</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📈 Weekly Analysis</h4>
            <p style="color: {current_theme['text']};">Detailed weekly trend analysis and insights</p>
            <button style="background: {current_theme['primary']}; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-top: 10px;">View Analysis</button>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Alerts":
    st.markdown(f'<div class="section-title">{current_texts["alerts"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="alert alert-danger">
        <strong>🚨 Critical Alert:</strong> Temperature exceeded threshold at 15:30
    </div>
    <div class="alert alert-warning">
        <strong>⚠️ Warning:</strong> Pressure fluctuation detected
    </div>
    <div class="alert alert-success">
        <strong>✅ Normal:</strong> All systems operating within normal parameters
    </div>
    """, unsafe_allow_html=True)

elif current_page == "Maintenance":
    st.markdown(f'<div class="section-title">{current_texts["maintenance"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">🔧 Scheduled Maintenance</h4>
            <ul style="color: {current_theme['text']};">
                <li>Sensor calibration - Due: Tomorrow</li>
                <li>System backup - Due: Next week</li>
                <li>Hardware inspection - Due: Next month</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']};">📋 Maintenance Log</h4>
            <ul style="color: {current_theme['text']};">
                <li>2024-06-15: Sensor cleaning completed</li>
                <li>2024-06-10: Software update installed</li>
                <li>2024-06-05: Routine inspection passed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Users":
    st.markdown(f'<div class="section-title">{current_texts["users"]}</div>', unsafe_allow_html=True)
    
    users_data = {
        'Name': ['Rakan Al-Marri', 'Abdulrahman Al-Zahrani', 'Admin User', 'Guest User'],
        'Role': ['Lead Developer', 'Co-Developer', 'Administrator', 'Viewer'],
        'Last Login': ['2024-06-16 08:00', '2024-06-16 07:30', '2024-06-15 18:00', '2024-06-14 12:00'],
        'Status': ['Active', 'Active', 'Active', 'Inactive']
    }
    
    users_df = pd.DataFrame(users_data)
    st.dataframe(users_df, use_container_width=True)

# إضافة المحتوى للصفحات الأخرى
elif current_page in ["Simulation", "Analytics", "Solutions"]:
    st.markdown(f'<div class="section-title">{pages[current_page]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: {current_theme['primary']};">Coming Soon</h4>
        <p style="color: {current_theme['text']};">This feature is under development and will be available soon.</p>
    </div>
    """, unsafe_allow_html=True)

# تذييل الصفحة
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {current_theme['text']}; padding: 1rem;">
    <small>{current_texts['copyright']}</small>
</div>
""", unsafe_allow_html=True)

