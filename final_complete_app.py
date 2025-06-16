import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import random

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
if 'generated_solutions' not in st.session_state:
    st.session_state.generated_solutions = []

# قاعدة بيانات الحلول الذكية الشاملة
smart_solutions_db = {
    'العربية': [
        {
            'name': 'نظام التبريد التكيفي الذكي',
            'description': 'تحسين نظام التبريد باستخدام خوارزميات التعلم الآلي المتقدمة لتقليل استهلاك الطاقة بنسبة تصل إلى 30% مع الحفاظ على الأداء الأمثل',
            'effectiveness': 92,
            'duration': '2-3 أسابيع',
            'priority': 'عالية جداً',
            'category': 'كفاءة الطاقة'
        },
        {
            'name': 'مراقبة الاهتزاز المتقدمة بالذكاء الاصطناعي',
            'description': 'تطبيق تقنيات الذكاء الاصطناعي والتعلم العميق لكشف الأعطال المبكرة في المعدات قبل حدوثها بـ 72 ساعة',
            'effectiveness': 96,
            'duration': '1-2 أسبوع',
            'priority': 'عالية جداً',
            'category': 'الصيانة التنبؤية'
        },
        {
            'name': 'تحسين ضغط النظام الديناميكي',
            'description': 'تطوير نظام تحكم ذكي متكيف لتحسين ضغط العمليات في الوقت الفعلي وتقليل الهدر بنسبة 25%',
            'effectiveness': 88,
            'duration': '3-4 أسابيع',
            'priority': 'عالية',
            'category': 'تحسين العمليات'
        },
        {
            'name': 'كشف تسرب الغازات الفوري',
            'description': 'نظام إنذار مبكر متطور لكشف تسرب الغازات الخطيرة باستخدام أجهزة استشعار ذكية مع دقة 99.8%',
            'effectiveness': 98,
            'duration': '1 أسبوع',
            'priority': 'عالية جداً',
            'category': 'السلامة والأمان'
        },
        {
            'name': 'تحليل البيانات التاريخية المتقدم',
            'description': 'استخدام البيانات التاريخية وخوارزميات التعلم الآلي لتحسين دقة التنبؤات المستقبلية إلى 94%',
            'effectiveness': 85,
            'duration': '2 أسبوع',
            'priority': 'متوسطة',
            'category': 'التحليل التنبؤي'
        },
        {
            'name': 'نظام الإنذار الذكي متعدد المستويات',
            'description': 'تطوير نظام إنذار متطور مع تصنيف أولويات التنبيهات وإشعارات فورية عبر قنوات متعددة',
            'effectiveness': 91,
            'duration': '1-2 أسبوع',
            'priority': 'عالية',
            'category': 'التنبيهات والإنذار'
        },
        {
            'name': 'تحسين جودة البيانات التلقائي',
            'description': 'تطبيق خوارزميات متقدمة لتنظيف البيانات وإزالة القيم الشاذة تلقائياً مع ضمان دقة 97%',
            'effectiveness': 82,
            'duration': '1 أسبوع',
            'priority': 'متوسطة',
            'category': 'جودة البيانات'
        },
        {
            'name': 'التحكم التلقائي الذكي في درجة الحرارة',
            'description': 'نظام تحكم تلقائي متطور لضبط درجة الحرارة بناءً على الظروف البيئية والتنبؤات الجوية',
            'effectiveness': 93,
            'duration': '2-3 أسابيع',
            'priority': 'عالية',
            'category': 'التحكم الآلي'
        },
        {
            'name': 'تحليل الأداء في الوقت الفعلي المتقدم',
            'description': 'لوحة تحكم متطورة لمراقبة الأداء وتحليل الاتجاهات في الوقت الفعلي مع تحديثات كل ثانية',
            'effectiveness': 89,
            'duration': '2 أسبوع',
            'priority': 'عالية',
            'category': 'المراقبة والتحليل'
        },
        {
            'name': 'نظام النسخ الاحتياطي الذكي المشفر',
            'description': 'نظام نسخ احتياطي تلقائي متطور مع ضغط البيانات وتشفيرها بمعايير عسكرية',
            'effectiveness': 95,
            'duration': '1 أسبوع',
            'priority': 'عالية',
            'category': 'أمان البيانات'
        },
        {
            'name': 'تحسين استهلاك الطاقة الذكي',
            'description': 'خوارزميات ذكية متطورة لتحسين استهلاك الطاقة وتقليل التكاليف التشغيلية بنسبة 35%',
            'effectiveness': 87,
            'duration': '3-4 أسابيع',
            'priority': 'عالية',
            'category': 'كفاءة الطاقة'
        },
        {
            'name': 'نظام التشخيص الذاتي المتقدم',
            'description': 'تطوير نظام تشخيص ذاتي للمعدات مع إصلاح تلقائي للمشاكل البسيطة وتقارير تفصيلية',
            'effectiveness': 90,
            'duration': '4-5 أسابيع',
            'priority': 'عالية',
            'category': 'الصيانة الذكية'
        },
        {
            'name': 'تحليل الاتجاهات بالتعلم العميق',
            'description': 'استخدام تقنيات التعلم العميق المتطورة لتحليل الاتجاهات طويلة المدى والتنبؤ بالمستقبل',
            'effectiveness': 94,
            'duration': '3 أسابيع',
            'priority': 'عالية جداً',
            'category': 'التحليل المتقدم'
        },
        {
            'name': 'نظام إدارة المخاطر الشامل',
            'description': 'تطوير نظام شامل ومتطور لتقييم وإدارة المخاطر التشغيلية مع نماذج تنبؤية',
            'effectiveness': 86,
            'duration': '2-3 أسابيع',
            'priority': 'عالية',
            'category': 'إدارة المخاطر'
        },
        {
            'name': 'تحسين واجهة المستخدم التفاعلية',
            'description': 'تطوير واجهة مستخدم أكثر تفاعلية وسهولة في الاستخدام مع تقنيات الواقع المعزز',
            'effectiveness': 79,
            'duration': '2 أسبوع',
            'priority': 'متوسطة',
            'category': 'تجربة المستخدم'
        }
    ],
    'English': [
        {
            'name': 'Smart Adaptive Cooling System',
            'description': 'Optimize cooling system using advanced machine learning algorithms to reduce energy consumption by up to 30% while maintaining optimal performance',
            'effectiveness': 92,
            'duration': '2-3 weeks',
            'priority': 'Very High',
            'category': 'Energy Efficiency'
        },
        {
            'name': 'AI-Powered Advanced Vibration Monitoring',
            'description': 'Apply AI and deep learning techniques for early fault detection in equipment 72 hours before failure occurs',
            'effectiveness': 96,
            'duration': '1-2 weeks',
            'priority': 'Very High',
            'category': 'Predictive Maintenance'
        },
        {
            'name': 'Dynamic System Pressure Optimization',
            'description': 'Develop smart adaptive control system to optimize process pressure in real-time and reduce waste by 25%',
            'effectiveness': 88,
            'duration': '3-4 weeks',
            'priority': 'High',
            'category': 'Process Optimization'
        },
        {
            'name': 'Instant Gas Leak Detection',
            'description': 'Advanced early warning system for detecting hazardous gas leaks using smart sensors with 99.8% accuracy',
            'effectiveness': 98,
            'duration': '1 week',
            'priority': 'Very High',
            'category': 'Safety & Security'
        },
        {
            'name': 'Advanced Historical Data Analysis',
            'description': 'Use historical data and machine learning algorithms to improve future prediction accuracy to 94%',
            'effectiveness': 85,
            'duration': '2 weeks',
            'priority': 'Medium',
            'category': 'Predictive Analytics'
        },
        {
            'name': 'Smart Multi-Level Alert System',
            'description': 'Develop advanced alert system with priority classification and instant notifications across multiple channels',
            'effectiveness': 91,
            'duration': '1-2 weeks',
            'priority': 'High',
            'category': 'Alerts & Notifications'
        },
        {
            'name': 'Automatic Data Quality Enhancement',
            'description': 'Apply advanced algorithms for automatic data cleaning and outlier removal with 97% accuracy guarantee',
            'effectiveness': 82,
            'duration': '1 week',
            'priority': 'Medium',
            'category': 'Data Quality'
        },
        {
            'name': 'Smart Automatic Temperature Control',
            'description': 'Advanced automatic control system to adjust temperature based on environmental conditions and weather forecasts',
            'effectiveness': 93,
            'duration': '2-3 weeks',
            'priority': 'High',
            'category': 'Automatic Control'
        },
        {
            'name': 'Advanced Real-time Performance Analysis',
            'description': 'Advanced dashboard for performance monitoring and trend analysis in real-time with updates every second',
            'effectiveness': 89,
            'duration': '2 weeks',
            'priority': 'High',
            'category': 'Monitoring & Analysis'
        },
        {
            'name': 'Smart Encrypted Backup System',
            'description': 'Advanced automatic backup system with data compression and military-grade encryption',
            'effectiveness': 95,
            'duration': '1 week',
            'priority': 'High',
            'category': 'Data Security'
        },
        {
            'name': 'Smart Energy Consumption Optimization',
            'description': 'Advanced smart algorithms to optimize energy consumption and reduce operational costs by 35%',
            'effectiveness': 87,
            'duration': '3-4 weeks',
            'priority': 'High',
            'category': 'Energy Efficiency'
        },
        {
            'name': 'Advanced Self-Diagnostic System',
            'description': 'Develop advanced self-diagnostic system for equipment with automatic repair for simple issues and detailed reports',
            'effectiveness': 90,
            'duration': '4-5 weeks',
            'priority': 'High',
            'category': 'Smart Maintenance'
        },
        {
            'name': 'Deep Learning Trend Analysis',
            'description': 'Use advanced deep learning techniques for long-term trend analysis and future prediction',
            'effectiveness': 94,
            'duration': '3 weeks',
            'priority': 'Very High',
            'category': 'Advanced Analytics'
        },
        {
            'name': 'Comprehensive Risk Management System',
            'description': 'Develop comprehensive and advanced system for operational risk assessment and management with predictive models',
            'effectiveness': 86,
            'duration': '2-3 weeks',
            'priority': 'High',
            'category': 'Risk Management'
        },
        {
            'name': 'Interactive User Interface Enhancement',
            'description': 'Develop more interactive and user-friendly interface with augmented reality technologies',
            'effectiveness': 79,
            'duration': '2 weeks',
            'priority': 'Medium',
            'category': 'User Experience'
        }
    ]
}

# النصوص متعددة اللغات
texts = {
    'العربية': {
        'title': 'التوأم الرقمي العصبي الذكي',
        'subtitle': 'نظام متقدم لمراقبة وتحليل البيانات في الوقت الفعلي',
        'main_menu': '🧠 القائمة الرئيسية',
        'dashboard': '📊 لوحة البيانات الرئيسية',
        'simulation': '🔄 المحاكاة والنمذجة',
        'analytics': '📈 التحليل التنبؤي المتقدم',
        'solutions': '💡 الحلول الذكية والتوصيات',
        'settings': '⚙️ الإعدادات والتخصيص',
        'about': 'ℹ️ حول المشروع والفريق',
        'reports': '📋 التقارير والإحصائيات',
        'alerts': '🚨 التنبيهات والإنذارات',
        'maintenance': '🔧 الصيانة والخدمة',
        'users': '👥 إدارة المستخدمين',
        'assets': '🏭 إدارة الأصول والمعدات',
        'operations': '⚡ التحكم في العمليات',
        'security': '🔒 الأمان والحماية',
        'finance': '💰 التحليل المالي والتكاليف',
        'energy': '🔋 إدارة الطاقة والاستدامة',
        'advanced_reports': '📊 التقارير المتقدمة والتحليلات',
        'quality': '🎯 ضمان الجودة والمعايير',
        'compliance': '📜 الامتثال والمعايير التنظيمية',
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
        ],
        'generate_solutions': '🚀 توليد حلول ذكية جديدة',
        'solution_name': 'اسم الحل',
        'solution_description': 'الوصف التفصيلي',
        'effectiveness': 'نسبة الفعالية',
        'duration': 'مدة التنفيذ',
        'priority': 'مستوى الأهمية',
        'category': 'فئة الحل',
        'no_solutions': 'لم يتم توليد حلول بعد. اضغط على الزر أعلاه لتوليد حلول ذكية مخصصة.',
        'solutions_generated': 'تم توليد الحلول الذكية بنجاح! 🎉'
    },
    'English': {
        'title': 'Smart Neural Digital Twin',
        'subtitle': 'Advanced system for real-time data monitoring and analysis',
        'main_menu': '🧠 Main Menu',
        'dashboard': '📊 Main Dashboard',
        'simulation': '🔄 Simulation & Modeling',
        'analytics': '📈 Advanced Predictive Analytics',
        'solutions': '💡 Smart Solutions & Recommendations',
        'settings': '⚙️ Settings & Customization',
        'about': 'ℹ️ About Project & Team',
        'reports': '📋 Reports & Statistics',
        'alerts': '🚨 Alerts & Notifications',
        'maintenance': '🔧 Maintenance & Service',
        'users': '👥 User Management',
        'assets': '🏭 Asset & Equipment Management',
        'operations': '⚡ Operations Control',
        'security': '🔒 Security & Protection',
        'finance': '💰 Financial Analysis & Costs',
        'energy': '🔋 Energy Management & Sustainability',
        'advanced_reports': '📊 Advanced Reports & Analytics',
        'quality': '🎯 Quality Assurance & Standards',
        'compliance': '📜 Compliance & Regulatory Standards',
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
        'tech_support': '📧 For Technical Support:',
        'support_desc': 'For inquiries and technical support, please contact the development team via email or phone mentioned above.',
        'features': '✨ Key Features:',
        'technologies': '🛠️ Technologies Used:',
        'copyright': '© 2024 Smart Neural Digital Twin - Saudi Aramco - All Rights Reserved',
        'project_description': 'Advanced system for real-time data monitoring and analysis using artificial intelligence and machine learning technologies.',
        'feature_list': [
            'Real-time data monitoring',
            'Automatic anomaly detection',
            'Advanced predictive analysis',
            'Interactive user interface',
            'Multi-language support',
            'Customizable themes'
        ],
        'tech_list': [
            'Python & Streamlit',
            'Plotly for charts',
            'Pandas for data processing',
            'Machine learning techniques'
        ],
        'generate_solutions': '🚀 Generate New Smart Solutions',
        'solution_name': 'Solution Name',
        'solution_description': 'Detailed Description',
        'effectiveness': 'Effectiveness Rate',
        'duration': 'Implementation Duration',
        'priority': 'Priority Level',
        'category': 'Solution Category',
        'no_solutions': 'No solutions generated yet. Click the button above to generate customized smart solutions.',
        'solutions_generated': 'Smart solutions generated successfully! 🎉'
    }
}

# ألوان المظاهر
themes = {
    'Ocean': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'background': '#0e1117',
        'surface': '#262730'
    },
    'Forest': {
        'primary': '#2ca02c',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'background': '#0e1117',
        'surface': '#262730'
    },
    'Sunset': {
        'primary': '#ff7f0e',
        'secondary': '#d62728',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'background': '#0e1117',
        'surface': '#262730'
    },
    'Purple': {
        'primary': '#9467bd',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'background': '#0e1117',
        'surface': '#262730'
    },
    'Slate': {
        'primary': '#7f7f7f',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'background': '#0e1117',
        'surface': '#262730'
    }
}

# تطبيق المظهر المخصص
def apply_custom_css():
    theme = themes[st.session_state.theme]
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {theme['background']};
        color: white;
    }}
    
    .main-header {{
        background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }}
    
    .metric-card {{
        background-color: {theme['surface']};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {theme['primary']};
        margin: 0.5rem 0;
        color: white;
    }}
    
    .sidebar .sidebar-content {{
        background-color: {theme['surface']};
        color: white;
    }}
    
    .stSelectbox > div > div {{
        background-color: {theme['surface']};
        color: white;
    }}
    
    .solution-card {{
        background-color: {theme['surface']};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {theme['primary']};
        margin: 1rem 0;
        color: white;
    }}
    
    .priority-high {{
        border-left: 4px solid {theme['danger']};
    }}
    
    .priority-medium {{
        border-left: 4px solid {theme['warning']};
    }}
    
    .priority-low {{
        border-left: 4px solid {theme['success']};
    }}
    
    .stButton > button {{
        background-color: {theme['primary']};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }}
    
    .stButton > button:hover {{
        background-color: {theme['secondary']};
    }}
    
    .sidebar-button {{
        width: 100%;
        text-align: left;
        background-color: {theme['primary']};
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        cursor: pointer;
    }}
    
    .sidebar-button:hover {{
        background-color: {theme['secondary']};
    }}
    
    .sidebar-button.active {{
        background-color: {theme['secondary']};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
    }}
    
    .stMarkdown {{
        color: white;
    }}
    
    .stDataFrame {{
        background-color: {theme['surface']};
    }}
    
    .stPlotlyChart {{
        background-color: {theme['surface']};
    }}
    </style>
    """, unsafe_allow_html=True)

# تطبيق CSS المخصص
apply_custom_css()

# الشريط الجانبي
with st.sidebar:
    st.markdown(f"<h2 style='color: white; text-align: center;'>{texts[st.session_state.language]['main_menu']}</h2>", unsafe_allow_html=True)
    
    # أزرار التنقل
    pages = [
        ('Dashboard', texts[st.session_state.language]['dashboard']),
        ('Assets', texts[st.session_state.language]['assets']),
        ('Operations', texts[st.session_state.language]['operations']),
        ('Security', texts[st.session_state.language]['security']),
        ('Finance', texts[st.session_state.language]['finance']),
        ('Energy', texts[st.session_state.language]['energy']),
        ('Quality', texts[st.session_state.language]['quality']),
        ('Compliance', texts[st.session_state.language]['compliance']),
        ('Simulation', texts[st.session_state.language]['simulation']),
        ('Analytics', texts[st.session_state.language]['analytics']),
        ('Solutions', texts[st.session_state.language]['solutions']),
        ('Reports', texts[st.session_state.language]['reports']),
        ('Alerts', texts[st.session_state.language]['alerts']),
        ('Maintenance', texts[st.session_state.language]['maintenance']),
        ('Settings', texts[st.session_state.language]['settings']),
        ('About', texts[st.session_state.language]['about'])
    ]
    
    for page_key, page_name in pages:
        if st.button(page_name, key=f"btn_{page_key}", use_container_width=True):
            st.session_state.current_page = page_key

# العنوان الرئيسي
st.markdown(f"""
<div class="main-header">
    <h1>{texts[st.session_state.language]['title']}</h1>
    <p>{texts[st.session_state.language]['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# المحتوى الرئيسي
if st.session_state.current_page == 'Dashboard':
    st.markdown(f"<h2 style='color: white;'>📊 {texts[st.session_state.language]['dashboard'].replace('📊 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الأداء الرئيسية
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_temp = df['temp'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['temperature']}</h4>
            <h2>{latest_temp:.1f} °C</h2>
            <p>{texts[st.session_state.language]['last_reading']}: {datetime.now().strftime('%H:%M %d-%m-%Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_pressure = df['pressure'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['pressure']}</h4>
            <h2>{latest_pressure:.1f} PSI</h2>
            <p>{texts[st.session_state.language]['last_reading']}: {datetime.now().strftime('%H:%M %d-%m-%Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        latest_vibration = df['vibration'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['vibration']}</h4>
            <h2>{latest_vibration:.1f} g</h2>
            <p>{texts[st.session_state.language]['last_reading']}: {datetime.now().strftime('%H:%M %d-%m-%Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        latest_gas = df['gas'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['gas']}</h4>
            <h2>{latest_gas:.1f} ppm</h2>
            <p>{texts[st.session_state.language]['last_reading']}: {datetime.now().strftime('%H:%M %d-%m-%Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # الرسوم البيانية
    st.markdown("<h3 style='color: white;'>اتجاهات البيانات (آخر 7 أيام)</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp = px.line(df.tail(168), x='timestamp', y='temp', 
                          title='درجة الحرارة', 
                          color_discrete_sequence=[themes[st.session_state.theme]['primary']])
        fig_temp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        fig_pressure = px.line(df.tail(168), x='timestamp', y='pressure', 
                              title='الضغط',
                              color_discrete_sequence=[themes[st.session_state.theme]['secondary']])
        fig_pressure.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_pressure, use_container_width=True)

elif st.session_state.current_page == 'Assets':
    st.markdown(f"<h2 style='color: white;'>🏭 {texts[st.session_state.language]['assets'].replace('🏭 ', '')}</h2>", unsafe_allow_html=True)
    
    # إحصائيات الأصول
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>إجمالي الأصول</h4>
            <h2>247</h2>
            <p>معدة نشطة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>معدل التشغيل</h4>
            <h2>94.2%</h2>
            <p>كفاءة عالية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>الصيانة المجدولة</h4>
            <h2>12</h2>
            <p>هذا الأسبوع</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>التنبيهات النشطة</h4>
            <h2>3</h2>
            <p>تحتاج متابعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني لحالة الأصول
    asset_status = ['تشغيل', 'صيانة', 'متوقف', 'في الانتظار']
    asset_counts = [234, 8, 3, 2]
    
    fig_assets = px.pie(values=asset_counts, names=asset_status, 
                       title='توزيع حالة الأصول',
                       color_discrete_sequence=px.colors.qualitative.Set3)
    fig_assets.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_assets, use_container_width=True)

elif st.session_state.current_page == 'Operations':
    st.markdown(f"<h2 style='color: white;'>⚡ {texts[st.session_state.language]['operations'].replace('⚡ ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الأداء التشغيلي
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>كفاءة العمليات</h4>
            <h2>96.8%</h2>
            <p>أداء ممتاز</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>الإنتاجية</h4>
            <h2>1,247</h2>
            <p>وحدة/ساعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>معدل الأخطاء</h4>
            <h2>0.12%</h2>
            <p>ضمن المعايير</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>وقت التوقف</h4>
            <h2>2.3</h2>
            <p>ساعات هذا الشهر</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني للإنتاجية
    hours = list(range(24))
    productivity = [np.random.normal(1200, 100) for _ in hours]
    
    fig_prod = px.bar(x=hours, y=productivity, 
                     title='الإنتاجية خلال 24 ساعة',
                     color_discrete_sequence=[themes[st.session_state.theme]['primary']])
    fig_prod.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_prod, use_container_width=True)

elif st.session_state.current_page == 'Security':
    st.markdown(f"<h2 style='color: white;'>🔒 {texts[st.session_state.language]['security'].replace('🔒 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الأمان
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>مستوى الأمان</h4>
            <h2>عالي</h2>
            <p>جميع الأنظمة آمنة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>محاولات الوصول</h4>
            <h2>1,247</h2>
            <p>اليوم</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>التهديدات المحجوبة</h4>
            <h2>23</h2>
            <p>آخر 24 ساعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>المستخدمون النشطون</h4>
            <h2>156</h2>
            <p>متصل حالياً</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني للتهديدات
    threat_types = ['فيروسات', 'هجمات شبكة', 'وصول غير مصرح', 'تسريب بيانات']
    threat_counts = [12, 8, 2, 1]
    
    fig_threats = px.bar(x=threat_types, y=threat_counts, 
                        title='أنواع التهديدات المحجوبة',
                        color_discrete_sequence=[themes[st.session_state.theme]['danger']])
    fig_threats.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_threats, use_container_width=True)

elif st.session_state.current_page == 'Finance':
    st.markdown(f"<h2 style='color: white;'>💰 {texts[st.session_state.language]['finance'].replace('💰 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات مالية
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>التكلفة التشغيلية</h4>
            <h2>$2.4M</h2>
            <p>هذا الشهر</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>العائد على الاستثمار</h4>
            <h2>18.5%</h2>
            <p>نمو إيجابي</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>توفير التكاليف</h4>
            <h2>$340K</h2>
            <p>مقارنة بالعام الماضي</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>كفاءة الطاقة</h4>
            <h2>92.1%</h2>
            <p>تحسن 5% عن الشهر الماضي</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني للتكاليف
    months = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو']
    costs = [2.1, 2.3, 2.2, 2.4, 2.3, 2.4]
    
    fig_costs = px.line(x=months, y=costs, 
                       title='التكاليف التشغيلية الشهرية (بالمليون دولار)',
                       color_discrete_sequence=[themes[st.session_state.theme]['success']])
    fig_costs.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_costs, use_container_width=True)

elif st.session_state.current_page == 'Energy':
    st.markdown(f"<h2 style='color: white;'>🔋 {texts[st.session_state.language]['energy'].replace('🔋 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الطاقة
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>استهلاك الطاقة</h4>
            <h2>847 kWh</h2>
            <p>آخر 24 ساعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>كفاءة الطاقة</h4>
            <h2>94.2%</h2>
            <p>أداء ممتاز</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>البصمة الكربونية</h4>
            <h2>1.2 طن</h2>
            <p>CO2 هذا الأسبوع</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>الطاقة المتجددة</h4>
            <h2>23%</h2>
            <p>من إجمالي الاستهلاك</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني لاستهلاك الطاقة
    hours = list(range(24))
    energy_consumption = [np.random.normal(35, 5) for _ in hours]
    
    fig_energy = px.area(x=hours, y=energy_consumption, 
                        title='استهلاك الطاقة خلال 24 ساعة (kWh)',
                        color_discrete_sequence=[themes[st.session_state.theme]['success']])
    fig_energy.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_energy, use_container_width=True)

elif st.session_state.current_page == 'Quality':
    st.markdown(f"<h2 style='color: white;'>🎯 {texts[st.session_state.language]['quality'].replace('🎯 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الجودة
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>معدل الجودة</h4>
            <h2>99.7%</h2>
            <p>ضمن المعايير العالمية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>العيوب المكتشفة</h4>
            <h2>3</h2>
            <p>آخر 1000 وحدة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>شهادات الجودة</h4>
            <h2>ISO 9001</h2>
            <p>معتمد ومحدث</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>رضا العملاء</h4>
            <h2>4.8/5</h2>
            <p>تقييم ممتاز</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني لمعدل الجودة
    days = ['الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
    quality_rates = [99.5, 99.8, 99.6, 99.7, 99.9, 99.4, 99.7]
    
    fig_quality = px.bar(x=days, y=quality_rates, 
                        title='معدل الجودة الأسبوعي (%)',
                        color_discrete_sequence=[themes[st.session_state.theme]['success']])
    fig_quality.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_quality, use_container_width=True)

elif st.session_state.current_page == 'Compliance':
    st.markdown(f"<h2 style='color: white;'>📜 {texts[st.session_state.language]['compliance'].replace('📜 ', '')}</h2>", unsafe_allow_html=True)
    
    # مؤشرات الامتثال
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>حالة الامتثال</h4>
            <h2>100%</h2>
            <p>جميع المعايير مستوفاة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>التراخيص النشطة</h4>
            <h2>47</h2>
            <p>جميعها سارية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>التدقيق الأخير</h4>
            <h2>15 مايو</h2>
            <p>نتائج ممتازة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>المخالفات</h4>
            <h2>0</h2>
            <p>آخر 12 شهر</p>
        </div>
        """, unsafe_allow_html=True)
    
    # رسم بياني لحالة التراخيص
    license_types = ['بيئية', 'سلامة', 'جودة', 'تشغيلية', 'أخرى']
    license_counts = [12, 15, 8, 10, 2]
    
    fig_licenses = px.pie(values=license_counts, names=license_types, 
                         title='توزيع التراخيص حسب النوع',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_licenses.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_licenses, use_container_width=True)

elif st.session_state.current_page == 'Simulation':
    st.markdown(f"<h2 style='color: white;'>🔄 {texts[st.session_state.language]['simulation'].replace('🔄 ', '')}</h2>", unsafe_allow_html=True)
    
    # محاكاة متقدمة للعمليات
    st.markdown("<h3 style='color: white;'>محاكاة العمليات الصناعية</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: white;'>معاملات المحاكاة</h4>", unsafe_allow_html=True)
        
        temp_range = st.slider("نطاق درجة الحرارة (°C)", 15, 40, (20, 30))
        pressure_range = st.slider("نطاق الضغط (PSI)", 80, 120, (90, 110))
        simulation_hours = st.selectbox("مدة المحاكاة", [1, 6, 12, 24, 48])
        
        if st.button("تشغيل المحاكاة", use_container_width=True):
            # توليد بيانات محاكاة
            sim_hours = list(range(simulation_hours))
            sim_temp = [np.random.uniform(temp_range[0], temp_range[1]) for _ in sim_hours]
            sim_pressure = [np.random.uniform(pressure_range[0], pressure_range[1]) for _ in sim_hours]
            
            # رسم نتائج المحاكاة
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=sim_hours, y=sim_temp, name='درجة الحرارة', line=dict(color=themes[st.session_state.theme]['primary'])))
            fig_sim.add_trace(go.Scatter(x=sim_hours, y=sim_pressure, name='الضغط', yaxis='y2', line=dict(color=themes[st.session_state.theme]['secondary'])))
            
            fig_sim.update_layout(
                title='نتائج المحاكاة',
                xaxis_title='الوقت (ساعات)',
                yaxis=dict(title='درجة الحرارة (°C)', side='left'),
                yaxis2=dict(title='الضغط (PSI)', side='right', overlaying='y'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: white;'>نماذج التنبؤ</h4>", unsafe_allow_html=True)
        
        # نموذج تنبؤي بسيط
        prediction_days = st.selectbox("فترة التنبؤ", [1, 3, 7, 14])
        
        if st.button("توليد تنبؤات", use_container_width=True):
            # توليد تنبؤات
            future_dates = pd.date_range(start=datetime.now(), periods=prediction_days*24, freq='h')
            predicted_temp = [25 + np.sin(i/12) * 3 + np.random.normal(0, 0.5) for i in range(len(future_dates))]
            
            fig_pred = px.line(x=future_dates, y=predicted_temp, 
                              title=f'تنبؤ درجة الحرارة لـ {prediction_days} أيام',
                              color_discrete_sequence=[themes[st.session_state.theme]['warning']])
            fig_pred.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pred, use_container_width=True)

elif st.session_state.current_page == 'Analytics':
    st.markdown(f"<h2 style='color: white;'>📈 {texts[st.session_state.language]['analytics'].replace('📈 ', '')}</h2>", unsafe_allow_html=True)
    
    # تحليلات متقدمة
    st.markdown("<h3 style='color: white;'>تحليلات الذكاء الاصطناعي</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: white;'>كشف الشذوذ</h4>", unsafe_allow_html=True)
        
        # محاكاة كشف الشذوذ
        anomaly_data = df.tail(100).copy()
        anomaly_data['anomaly'] = np.random.choice([0, 1], size=len(anomaly_data), p=[0.95, 0.05])
        
        fig_anomaly = px.scatter(anomaly_data, x='timestamp', y='temp', 
                                color='anomaly', 
                                title='كشف الشذوذ في درجة الحرارة',
                                color_discrete_map={0: themes[st.session_state.theme]['primary'], 1: themes[st.session_state.theme]['danger']})
        fig_anomaly.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: white;'>تحليل الاتجاهات</h4>", unsafe_allow_html=True)
        
        # تحليل الاتجاهات
        trend_data = df.tail(168).copy()  # آخر أسبوع
        trend_data['trend'] = trend_data['temp'].rolling(window=24).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_data['timestamp'], y=trend_data['temp'], 
                                      name='البيانات الفعلية', opacity=0.6,
                                      line=dict(color=themes[st.session_state.theme]['primary'])))
        fig_trend.add_trace(go.Scatter(x=trend_data['timestamp'], y=trend_data['trend'], 
                                      name='الاتجاه العام',
                                      line=dict(color=themes[st.session_state.theme]['danger'], width=3)))
        
        fig_trend.update_layout(
            title='تحليل اتجاه درجة الحرارة',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # مؤشرات التحليل المتقدم
    st.markdown("<h4 style='color: white;'>مؤشرات الأداء التنبؤي</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>دقة التنبؤ</h4>
            <h2>94.2%</h2>
            <p>نموذج محسن</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>الشذوذ المكتشف</h4>
            <h2>7</h2>
            <p>آخر 24 ساعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>التنبؤات النشطة</h4>
            <h2>156</h2>
            <p>نموذج يعمل</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>وقت المعالجة</h4>
            <h2>0.3s</h2>
            <p>استجابة سريعة</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == 'Solutions':
    st.markdown(f"<h2 style='color: white;'>💡 {texts[st.session_state.language]['solutions'].replace('💡 ', '')}</h2>", unsafe_allow_html=True)
    
    # زر توليد الحلول
    if st.button(texts[st.session_state.language]['generate_solutions'], use_container_width=True):
        # توليد حلول عشوائية من قاعدة البيانات
        available_solutions = smart_solutions_db[st.session_state.language]
        selected_solutions = random.sample(available_solutions, min(12, len(available_solutions)))
        st.session_state.generated_solutions = selected_solutions
        st.success(texts[st.session_state.language]['solutions_generated'])
    
    # عرض الحلول المولدة
    if st.session_state.generated_solutions:
        st.markdown("<h3 style='color: white;'>الحلول الذكية المولدة</h3>", unsafe_allow_html=True)
        
        for i, solution in enumerate(st.session_state.generated_solutions):
            priority_class = ""
            if solution['priority'] in ['عالية جداً', 'Very High']:
                priority_class = "priority-high"
            elif solution['priority'] in ['عالية', 'High']:
                priority_class = "priority-medium"
            else:
                priority_class = "priority-low"
            
            st.markdown(f"""
            <div class="solution-card {priority_class}">
                <h4>{solution['name']}</h4>
                <p>{solution['description']}</p>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <span><strong>{texts[st.session_state.language]['effectiveness']}:</strong> {solution['effectiveness']}%</span>
                    <span><strong>{texts[st.session_state.language]['duration']}:</strong> {solution['duration']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span><strong>{texts[st.session_state.language]['priority']}:</strong> {solution['priority']}</span>
                    <span><strong>{texts[st.session_state.language]['category']}:</strong> {solution['category']}</span>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="background-color: #333; border-radius: 10px; overflow: hidden;">
                        <div style="background-color: {themes[st.session_state.theme]['primary']}; height: 10px; width: {solution['effectiveness']}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(texts[st.session_state.language]['no_solutions'])

elif st.session_state.current_page == 'Reports':
    st.markdown(f"<h2 style='color: white;'>📋 {texts[st.session_state.language]['reports'].replace('📋 ', '')}</h2>", unsafe_allow_html=True)
    
    # تقارير شاملة
    st.markdown("<h3 style='color: white;'>التقارير التنفيذية</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: white;'>التقرير اليومي</h4>", unsafe_allow_html=True)
        
        daily_metrics = {
            'المؤشر': ['الإنتاجية', 'الجودة', 'الكفاءة', 'السلامة', 'التكلفة'],
            'القيمة الحالية': ['1,247 وحدة', '99.7%', '96.8%', '100%', '$2.4M'],
            'الهدف': ['1,200 وحدة', '99.5%', '95%', '100%', '$2.5M'],
            'الحالة': ['✅ متجاوز', '✅ متجاوز', '✅ متجاوز', '✅ مستوفي', '✅ أقل من المستهدف']
        }
        
        daily_df = pd.DataFrame(daily_metrics)
        st.dataframe(daily_df, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: white;'>التقرير الأسبوعي</h4>", unsafe_allow_html=True)
        
        weekly_metrics = {
            'اليوم': ['الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد'],
            'الإنتاجية': [1200, 1250, 1180, 1300, 1247, 1150, 1220],
            'الجودة (%)': [99.5, 99.8, 99.6, 99.7, 99.9, 99.4, 99.7],
            'التوقفات (دقيقة)': [15, 8, 22, 5, 12, 18, 10]
        }
        
        weekly_df = pd.DataFrame(weekly_metrics)
        st.dataframe(weekly_df, use_container_width=True)
    
    # رسم بياني للأداء الأسبوعي
    fig_weekly = px.line(weekly_df, x='اليوم', y='الإنتاجية', 
                        title='الأداء الأسبوعي للإنتاجية',
                        color_discrete_sequence=[themes[st.session_state.theme]['primary']])
    fig_weekly.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

elif st.session_state.current_page == 'Alerts':
    st.markdown(f"<h2 style='color: white;'>🚨 {texts[st.session_state.language]['alerts'].replace('🚨 ', '')}</h2>", unsafe_allow_html=True)
    
    # نظام التنبيهات المتقدم
    st.markdown("<h3 style='color: white;'>التنبيهات النشطة</h3>", unsafe_allow_html=True)
    
    # تنبيهات حسب الأولوية
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card priority-high">
            <h4>🔴 تنبيهات عالية</h4>
            <h2>2</h2>
            <p>تحتاج تدخل فوري</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card priority-medium">
            <h4>🟡 تنبيهات متوسطة</h4>
            <h2>5</h2>
            <p>تحتاج متابعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card priority-low">
            <h4>🟢 تنبيهات منخفضة</h4>
            <h2>12</h2>
            <p>للمراجعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    # قائمة التنبيهات
    alerts_data = {
        'الوقت': ['15:30', '14:45', '13:20', '12:10', '11:55'],
        'النوع': ['درجة حرارة عالية', 'ضغط منخفض', 'اهتزاز غير طبيعي', 'تسرب غاز', 'صيانة مجدولة'],
        'الأولوية': ['عالية', 'متوسطة', 'عالية', 'عالية جداً', 'منخفضة'],
        'الحالة': ['جديد', 'قيد المعالجة', 'جديد', 'تم الحل', 'مجدول'],
        'المعدة': ['مضخة-01', 'خزان-03', 'محرك-05', 'خط-أنابيب-02', 'ضاغط-04']
    }
    
    alerts_df = pd.DataFrame(alerts_data)
    st.dataframe(alerts_df, use_container_width=True)
    
    # رسم بياني لتوزيع التنبيهات
    alert_types = ['درجة حرارة', 'ضغط', 'اهتزاز', 'تسرب غاز', 'صيانة']
    alert_counts = [8, 5, 3, 2, 1]
    
    fig_alerts = px.bar(x=alert_types, y=alert_counts, 
                       title='توزيع التنبيهات حسب النوع',
                       color_discrete_sequence=[themes[st.session_state.theme]['warning']])
    fig_alerts.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_alerts, use_container_width=True)

elif st.session_state.current_page == 'Maintenance':
    st.markdown(f"<h2 style='color: white;'>🔧 {texts[st.session_state.language]['maintenance'].replace('🔧 ', '')}</h2>", unsafe_allow_html=True)
    
    # إدارة الصيانة المتقدمة
    st.markdown("<h3 style='color: white;'>جدولة الصيانة الذكية</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>الصيانة المجدولة</h4>
            <h2>12</h2>
            <p>هذا الأسبوع</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>الصيانة الطارئة</h4>
            <h2>2</h2>
            <p>قيد التنفيذ</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>معدل الإنجاز</h4>
            <h2>94%</h2>
            <p>في الوقت المحدد</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>التكلفة الشهرية</h4>
            <h2>$45K</h2>
            <p>ضمن الميزانية</p>
        </div>
        """, unsafe_allow_html=True)
    
    # جدول أعمال الصيانة
    maintenance_data = {
        'التاريخ': ['2024-06-17', '2024-06-18', '2024-06-19', '2024-06-20', '2024-06-21'],
        'المعدة': ['مضخة-01', 'ضاغط-02', 'محرك-03', 'خزان-04', 'مولد-05'],
        'نوع الصيانة': ['دورية', 'إصلاح', 'دورية', 'تنبؤية', 'دورية'],
        'المدة المتوقعة': ['2 ساعة', '4 ساعات', '1 ساعة', '3 ساعات', '2 ساعة'],
        'الحالة': ['مجدول', 'قيد التنفيذ', 'مجدول', 'مجدول', 'مجدول']
    }
    
    maintenance_df = pd.DataFrame(maintenance_data)
    st.dataframe(maintenance_df, use_container_width=True)
    
    # رسم بياني لتوزيع أنواع الصيانة
    maintenance_types = ['دورية', 'تنبؤية', 'إصلاح', 'طارئة']
    maintenance_counts = [15, 8, 5, 2]
    
    fig_maintenance = px.pie(values=maintenance_counts, names=maintenance_types, 
                            title='توزيع أنواع الصيانة',
                            color_discrete_sequence=px.colors.qualitative.Set2)
    fig_maintenance.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_maintenance, use_container_width=True)

elif st.session_state.current_page == 'Settings':
    st.markdown(f"<h2 style='color: white;'>⚙️ {texts[st.session_state.language]['settings'].replace('⚙️ ', '')}</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='color: white;'>{texts[st.session_state.language]['language_settings']}</h3>", unsafe_allow_html=True)
        
        new_language = st.selectbox(
            texts[st.session_state.language]['select_language'],
            ['العربية', 'English'],
            index=0 if st.session_state.language == 'العربية' else 1
        )
        
        if new_language != st.session_state.language:
            st.session_state.language = new_language
            st.rerun()
    
    with col2:
        st.markdown(f"<h3 style='color: white;'>{texts[st.session_state.language]['theme_settings']}</h3>", unsafe_allow_html=True)
        
        theme_options = {
            texts[st.session_state.language]['ocean']: 'Ocean',
            texts[st.session_state.language]['forest']: 'Forest',
            texts[st.session_state.language]['sunset']: 'Sunset',
            texts[st.session_state.language]['purple']: 'Purple',
            texts[st.session_state.language]['slate']: 'Slate'
        }
        
        selected_theme_display = st.selectbox(
            texts[st.session_state.language]['select_theme'],
            list(theme_options.keys()),
            index=list(theme_options.values()).index(st.session_state.theme)
        )
        
        new_theme = theme_options[selected_theme_display]
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

elif st.session_state.current_page == 'About':
    st.markdown(f"<h2 style='color: white;'>ℹ️ {texts[st.session_state.language]['about'].replace('ℹ️ ', '')}</h2>", unsafe_allow_html=True)
    
    # معلومات المشروع
    st.markdown(f"<h3 style='color: white;'>{texts[st.session_state.language]['project_description']}</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h4 style='color: white;'>{texts[st.session_state.language]['features']}</h4>", unsafe_allow_html=True)
        for feature in texts[st.session_state.language]['feature_list']:
            st.markdown(f"• {feature}")
        
        st.markdown(f"<h4 style='color: white;'>{texts[st.session_state.language]['technologies']}</h4>", unsafe_allow_html=True)
        for tech in texts[st.session_state.language]['tech_list']:
            st.markdown(f"• {tech}")
    
    with col2:
        st.markdown(f"<h4 style='color: white;'>{texts[st.session_state.language]['contact_info']}</h4>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['developer1']}</h4>
            <p>{texts[st.session_state.language]['main_developer']}</p>
            <p>{texts[st.session_state.language]['email']} rakan.almarri.2@aramco.com</p>
            <p>{texts[st.session_state.language]['phone']} 0532559664</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>{texts[st.session_state.language]['developer2']}</h4>
            <p>{texts[st.session_state.language]['co_developer']}</p>
            <p>{texts[st.session_state.language]['email']} abdulrahman.alzhrani.1@aramco.com</p>
            <p>{texts[st.session_state.language]['phone']} 0549202674</p>
        </div>
        """, unsafe_allow_html=True)
    
    # معلومات الشركة
    st.markdown(f"<h4 style='color: white;'>{texts[st.session_state.language]['company_info']}</h4>", unsafe_allow_html=True)
    st.markdown(texts[st.session_state.language]['company_desc'])
    
    st.markdown(f"<h4 style='color: white;'>{texts[st.session_state.language]['tech_support']}</h4>", unsafe_allow_html=True)
    st.markdown(texts[st.session_state.language]['support_desc'])

# تذييل الصفحة
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #888;'>{texts[st.session_state.language]['copyright']}</p>", unsafe_allow_html=True)

