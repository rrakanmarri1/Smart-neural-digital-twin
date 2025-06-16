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

# CSS مخصص محسن
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
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
        50% {{ box-shadow: 0 8px 25px rgba(30,144,255,0.4); }}
        100% {{ box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
    }}
    
    .metric-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {current_theme['primary']};
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left-color: {current_theme['secondary']};
    }}
    
    .solution-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid {current_theme['primary']};
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .solution-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {current_theme['primary']}, {current_theme['secondary']});
    }}
    
    .solution-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        border-color: {current_theme['secondary']};
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
        font-size: 1rem;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']});
        color: white;
        transform: translateX(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    
    .generate-btn {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']}) !important;
        color: white !important;
        border: none !important;
        padding: 1.5rem 3rem !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    
    .generate-btn:hover {{
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
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
    
    .section-title {{
        color: {current_theme['primary']};
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid {current_theme['primary']};
        padding-bottom: 0.5rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    
    .priority-very-high {{
        color: #ff3838;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 56, 56, 0.5);
        animation: glow 2s infinite;
    }}
    
    .priority-high {{
        color: #ff4757;
        font-weight: bold;
    }}
    
    .priority-medium {{
        color: #ffa502;
        font-weight: bold;
    }}
    
    @keyframes glow {{
        0%, 100% {{ text-shadow: 0 0 10px rgba(255, 56, 56, 0.5); }}
        50% {{ text-shadow: 0 0 20px rgba(255, 56, 56, 0.8); }}
    }}
    
    .effectiveness-container {{
        margin: 1rem 0;
    }}
    
    .effectiveness-bar {{
        background: rgba(255,255,255,0.1);
        height: 12px;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    
    .effectiveness-fill {{
        height: 100%;
        background: linear-gradient(90deg, {current_theme['primary']}, {current_theme['secondary']});
        border-radius: 6px;
        transition: width 1s ease;
        position: relative;
    }}
    
    .effectiveness-fill::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    .contact-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid {current_theme['primary']};
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .contact-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }}
    
    .info-card {{
        background: {current_theme['card']};
        color: {current_theme['text']};
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid {current_theme['primary']};
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }}
    
    .feature-item {{
        color: {current_theme['text']};
        margin: 0.8rem 0;
        padding: 0.5rem 0;
        font-size: 1.1rem;
    }}
    
    .category-badge {{
        background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']});
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }}
    
    .solution-stats {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }}
    
    .stat-item {{
        text-align: center;
    }}
    
    .stat-label {{
        color: {current_theme['primary']};
        font-weight: bold;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stat-value {{
        color: {current_theme['text']};
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }}
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي المحسن مع القوائم الجديدة
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, {current_theme['primary']}, {current_theme['secondary']}); color: white; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h2 style="margin: 0; font-size: 1.3rem;">{current_texts['main_menu']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # قائمة التنقل المحسنة مع القوائم الجديدة المرئية
    pages = {
        "Dashboard": current_texts['dashboard'],
        "Assets": current_texts['assets'],
        "Operations": current_texts['operations'],
        "Security": current_texts['security'],
        "Finance": current_texts['finance'],
        "Energy": current_texts['energy'],
        "Quality": current_texts['quality'],
        "Compliance": current_texts['compliance'],
        "Simulation": current_texts['simulation'],
        "Analytics": current_texts['analytics'],
        "Solutions": current_texts['solutions'],
        "Advanced_Reports": current_texts['advanced_reports'],
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
    <h1 style="margin: 0; font-size: 2.5rem;">{current_texts['title']}</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">{current_texts['subtitle']}</p>
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
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['temperature']}</h4>
            <h2 style="color: {current_theme['text']}; margin: 0; font-size: 2.5rem;">{latest.temp:.1f}°C</h2>
            <small style="color: {current_theme['text']}; opacity: 0.8;">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['pressure']}</h4>
            <h2 style="color: {current_theme['text']}; margin: 0; font-size: 2.5rem;">{latest.pressure:.1f} PSI</h2>
            <small style="color: {current_theme['text']}; opacity: 0.8;">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['vibration']}</h4>
            <h2 style="color: {current_theme['text']}; margin: 0; font-size: 2.5rem;">{latest.vibration:.2f} g</h2>
            <small style="color: {current_theme['text']}; opacity: 0.8;">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1rem;">{current_texts['gas']}</h4>
            <h2 style="color: {current_theme['text']}; margin: 0; font-size: 2.5rem;">{latest.gas:.1f} ppm</h2>
            <small style="color: {current_theme['text']}; opacity: 0.8;">{current_texts['last_reading']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title">📈 اتجاهات البيانات</div>', unsafe_allow_html=True)
        fig = px.line(df.tail(100), x="timestamp", y=["temp", "pressure", "vibration", "gas"],
                     labels={"timestamp": "الوقت", "value": "القيمة", "variable": "المتغير"},
                     color_discrete_sequence=[current_theme['primary'], current_theme['secondary'], '#FF6B6B', '#4ECDC4'])
        fig.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], 
                         font_color=current_theme['text'], title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-title">🗺️ الخريطة الحرارية</div>', unsafe_allow_html=True)
        heat_data = df.pivot_table(index=df.timestamp.dt.hour, columns=df.timestamp.dt.day, values="temp", aggfunc="mean")
        fig2 = go.Figure(data=go.Heatmap(z=heat_data.values, x=heat_data.columns, y=heat_data.index, 
                                        colorscale="Viridis", showscale=True))
        fig2.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], 
                          font_color=current_theme['text'], xaxis_title="اليوم", yaxis_title="الساعة")
        st.plotly_chart(fig2, use_container_width=True)

elif current_page == "Solutions":
    st.markdown(f'<div class="section-title">{current_texts["solutions"]}</div>', unsafe_allow_html=True)
    
    # زر توليد الحلول المحسن
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(current_texts['generate_solutions'], key="generate_solutions_btn", use_container_width=True):
            # توليد حلول عشوائية من قاعدة البيانات
            solutions_list = smart_solutions_db[st.session_state.language]
            selected_solutions = random.sample(solutions_list, min(12, len(solutions_list)))
            st.session_state.generated_solutions = selected_solutions
            st.success(current_texts['solutions_generated'])
            st.rerun()
    
    st.markdown("---")
    
    # عرض الحلول المولدة بتصميم محسن
    if st.session_state.generated_solutions:
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: {current_theme['primary']}; font-size: 1.5rem;">
                تم توليد {len(st.session_state.generated_solutions)} حل ذكي مخصص
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, solution in enumerate(st.session_state.generated_solutions):
            # تحديد لون الأولوية
            priority_class = ""
            if solution['priority'] in ['عالية جداً', 'Very High']:
                priority_class = "priority-very-high"
            elif solution['priority'] in ['عالية', 'High']:
                priority_class = "priority-high"
            else:
                priority_class = "priority-medium"
            
            # حساب عرض شريط الفعالية
            effectiveness_width = solution['effectiveness']
            
            st.markdown(f"""
            <div class="solution-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                    <h3 style="color: {current_theme['primary']}; margin: 0; font-size: 1.4rem; font-weight: bold;">
                        {solution['name']}
                    </h3>
                    <span class="{priority_class}" style="font-size: 1.1rem;">
                        {solution['priority']}
                    </span>
                </div>
                
                <p style="color: {current_theme['text']}; margin-bottom: 1.5rem; line-height: 1.7; font-size: 1.1rem;">
                    {solution['description']}
                </p>
                
                <div class="effectiveness-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: {current_theme['text']}; font-weight: bold;">{current_texts['effectiveness']}</span>
                        <span style="color: {current_theme['primary']}; font-weight: bold; font-size: 1.2rem;">{effectiveness_width}%</span>
                    </div>
                    <div class="effectiveness-bar">
                        <div class="effectiveness-fill" style="width: {effectiveness_width}%;"></div>
                    </div>
                </div>
                
                <div class="solution-stats">
                    <div class="stat-item">
                        <div class="stat-label">{current_texts['duration']}</div>
                        <div class="stat-value">{solution['duration']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">{current_texts['category']}</div>
                        <div class="stat-value">{solution['category']}</div>
                    </div>
                </div>
                
                <div class="category-badge">
                    {solution['category']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: {current_theme['primary']}; margin-bottom: 1rem;">
                🚀 مولد الحلول الذكية
            </h3>
            <p style="color: {current_theme['text']}; font-size: 1.2rem; line-height: 1.6;">
                {current_texts['no_solutions']}
            </p>
        </div>
        """, unsafe_allow_html=True)

# القوائم الجديدة المرئية
elif current_page == "Assets":
    st.markdown(f'<div class="section-title">{current_texts["assets"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🏭 نظرة عامة على الأصول</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">إجمالي الأصول: <strong style="color: {current_theme['primary']};">247</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الأصول النشطة: <strong style="color: #2ed573;">234</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تحت الصيانة: <strong style="color: #ffa502;">8</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">مشاكل حرجة: <strong style="color: #ff4757;">5</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📊 أداء الأصول</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الكفاءة الإجمالية: <strong style="color: {current_theme['primary']};">92.5%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">وقت التشغيل: <strong style="color: #2ed573;">98.2%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">معدل الاستخدام: <strong style="color: {current_theme['primary']};">87.3%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # رسم بياني لحالة الأصول
        asset_status = ['نشط', 'صيانة', 'حرج']
        asset_counts = [234, 8, 5]
        colors = ['#2ed573', '#ffa502', '#ff4757']
        
        fig = px.pie(values=asset_counts, names=asset_status, color_discrete_sequence=colors,
                    title="توزيع حالة الأصول")
        fig.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], 
                         font_color=current_theme['text'], title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)

elif current_page == "Operations":
    st.markdown(f'<div class="section-title">{current_texts["operations"]}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">⚡ حالة النظام</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: #2ed573; font-size: 1.1rem;">✅ الأنظمة الأساسية: متصلة</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ الأنظمة الاحتياطية: جاهزة</p>
                <p style="color: #ffa502; font-size: 1.1rem;">⚠️ نظام التبريد: تحذير</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ أنظمة السلامة: نشطة</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🎛️ لوحة التحكم</h4>
            <div style="margin: 1.5rem 0; text-align: center;">
                <button style="background: #2ed573; color: white; border: none; padding: 12px 24px; border-radius: 8px; margin: 5px; font-size: 1rem; cursor: pointer;">بدء العملية</button><br>
                <button style="background: #ff4757; color: white; border: none; padding: 12px 24px; border-radius: 8px; margin: 5px; font-size: 1rem; cursor: pointer;">إيقاف طارئ</button><br>
                <button style="background: {current_theme['primary']}; color: white; border: none; padding: 12px 24px; border-radius: 8px; margin: 5px; font-size: 1rem; cursor: pointer;">إعادة تشغيل</button><br>
                <button style="background: #ffa502; color: white; border: none; padding: 12px 24px; border-radius: 8px; margin: 5px; font-size: 1rem; cursor: pointer;">وضع الصيانة</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📈 مقاييس الأداء</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الإنتاجية: <strong style="color: {current_theme['primary']};">1,247 وحدة/ساعة</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الكفاءة: <strong style="color: #2ed573;">94.2%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">معدل الجودة: <strong style="color: #2ed573;">99.1%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">استهلاك الطاقة: <strong style="color: {current_theme['primary']};">2.3 ميجاوات</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Security":
    st.markdown(f'<div class="section-title">{current_texts["security"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🔒 حالة الأمان</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: #2ed573; font-size: 1.1rem;">✅ جدار الحماية: نشط</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ كشف التسلل: متصل</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ تشفير البيانات: مفعل</p>
                <p style="color: #ffa502; font-size: 1.1rem;">⚠️ آخر فحص أمني: منذ يومين</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🚨 أحداث الأمان الأخيرة</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تم حظر محاولة دخول فاشلة</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تم اكتشاف نشاط مشبوه في الشبكة</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تم تطبيق تحديث أمني بنجاح</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تم تحديث التحكم في الوصول</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">👥 التحكم في الوصول</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الجلسات النشطة: <strong style="color: {current_theme['primary']};">12</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">المستخدمون الإداريون: <strong style="color: {current_theme['primary']};">3</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">المستخدمون العاديون: <strong style="color: {current_theme['primary']};">9</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">المحاولات الفاشلة: <strong style="color: #ff4757;">2</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🛡️ توصيات الأمان</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تحديث سياسات الأمان</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تشغيل فحص الثغرات</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• مراجعة صلاحيات المستخدمين</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تفعيل المصادقة الثنائية</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Finance":
    st.markdown(f'<div class="section-title">{current_texts["finance"]}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">💰 تحليل التكاليف</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التكلفة التشغيلية الشهرية: <strong style="color: {current_theme['primary']};">$125,000</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تكاليف الطاقة: <strong style="color: #ffa502;">$45,000</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تكاليف الصيانة: <strong style="color: #ff4757;">$28,000</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تكاليف العمالة: <strong style="color: {current_theme['primary']};">$52,000</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📊 تحليل العائد على الاستثمار</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">العائد الحالي: <strong style="color: #2ed573;">18.5%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">العائد المتوقع: <strong style="color: {current_theme['primary']};">22.3%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">توفير التكاليف: <strong style="color: #2ed573;">$15,000/شهر</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">فترة الاسترداد: <strong style="color: {current_theme['primary']};">14 شهر</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📈 تتبع الميزانية</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">استخدام الميزانية: <strong style="color: {current_theme['primary']};">78%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الميزانية المتبقية: <strong style="color: #2ed573;">$275,000</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">دقة التوقعات: <strong style="color: {current_theme['primary']};">94%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التباين: <strong style="color: #ffa502;">-2.1%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Energy":
    st.markdown(f'<div class="section-title">{current_texts["energy"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🔋 استهلاك الطاقة</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الاستخدام الحالي: <strong style="color: {current_theme['primary']};">2.3 ميجاوات</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الذروة: <strong style="color: #ff4757;">3.1 ميجاوات</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">المتوسط: <strong style="color: {current_theme['primary']};">2.1 ميجاوات</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تقييم الكفاءة: <strong style="color: #2ed573;">87%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">💡 تحسين الطاقة</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التوفير المحتمل: <strong style="color: #2ed573;">15%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تكامل الشبكة الذكية: <strong style="color: {current_theme['primary']};">نشط</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الطاقة المتجددة: <strong style="color: #2ed573;">25%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">البصمة الكربونية: <strong style="color: {current_theme['primary']};">-12% مقارنة بالعام الماضي</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # رسم بياني لاستهلاك الطاقة
        hours = list(range(24))
        energy_consumption = [1.8 + 0.5 * np.sin(h * np.pi / 12) + np.random.normal(0, 0.1) for h in hours]
        
        fig = px.line(x=hours, y=energy_consumption, title="استهلاك الطاقة على مدار 24 ساعة",
                     labels={"x": "الساعة", "y": "الطاقة (ميجاوات)"})
        fig.update_traces(line_color=current_theme['primary'], line_width=3)
        fig.update_layout(paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], 
                         font_color=current_theme['text'], title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)

elif current_page == "Quality":
    st.markdown(f'<div class="section-title">{current_texts["quality"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🎯 مؤشرات الجودة</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">معدل الجودة الإجمالي: <strong style="color: #2ed573;">99.2%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">المنتجات المعيبة: <strong style="color: #ff4757;">0.8%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">رضا العملاء: <strong style="color: #2ed573;">96.5%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">الامتثال للمعايير: <strong style="color: #2ed573;">100%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📋 عمليات التفتيش</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تفتيش يومي: مكتمل ✅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تفتيش أسبوعي: مجدول لغداً 📅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تدقيق شهري: مكتمل ✅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• مراجعة سنوية: في التقدم 🔄</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📊 تحليل الاتجاهات</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تحسن الجودة: <strong style="color: #2ed573;">+2.3% هذا الشهر</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">تقليل العيوب: <strong style="color: #2ed573;">-15% مقارنة بالعام الماضي</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">كفاءة العمليات: <strong style="color: {current_theme['primary']};">94.8%</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">وقت الاستجابة: <strong style="color: #2ed573;">تحسن بنسبة 18%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">🏆 الشهادات والمعايير</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• ISO 9001:2015 ✅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• ISO 14001:2015 ✅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• OHSAS 18001 ✅</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• Six Sigma Green Belt 🎯</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Compliance":
    st.markdown(f'<div class="section-title">{current_texts["compliance"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📜 حالة الامتثال</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: #2ed573; font-size: 1.1rem;">✅ اللوائح البيئية: متوافق</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ معايير السلامة: متوافق</p>
                <p style="color: #ffa502; font-size: 1.1rem;">⚠️ تقرير سنوي: مطلوب التحديث</p>
                <p style="color: #2ed573; font-size: 1.1rem;">✅ تراخيص التشغيل: سارية</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📋 التدقيق والمراجعة</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• آخر تدقيق داخلي: منذ 3 أشهر</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• آخر تدقيق خارجي: منذ 6 أشهر</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• التدقيق القادم: خلال شهرين</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• نتائج التدقيق: ممتازة</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">⚖️ المتطلبات التنظيمية</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التراخيص النشطة: <strong style="color: #2ed573;">12</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التراخيص المنتهية: <strong style="color: #ff4757;">0</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">التجديدات المطلوبة: <strong style="color: #ffa502;">2</strong></p>
                <p style="color: {current_theme['text']}; font-size: 1.1rem;">معدل الامتثال: <strong style="color: #2ed573;">98.5%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {current_theme['primary']}; font-size: 1.3rem;">📊 تقارير الامتثال</h4>
            <div style="margin: 1.5rem 0;">
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تقرير شهري: مكتمل</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تقرير ربع سنوي: قيد الإعداد</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تقرير سنوي: مجدول</p>
                <p style="color: {current_theme['text']}; font-size: 1rem;">• تقارير خاصة: حسب الحاجة</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == "Settings":
    st.markdown(f'<div class="section-title">{current_texts["settings"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-title" style="font-size: 1.3rem;">{current_texts["language_settings"]}</div>', unsafe_allow_html=True)
        
        language_options = ["العربية", "English"]
        selected_lang = st.selectbox(
            current_texts["select_language"],
            language_options,
            index=language_options.index(st.session_state.language),
            key="lang_settings"
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.success(f"تم تغيير اللغة إلى: {selected_lang}")
            st.rerun()
        
        st.markdown(f'<div class="section-title" style="font-size: 1.3rem;">{current_texts["theme_settings"]}</div>', unsafe_allow_html=True)
        
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
            st.success(f"تم تغيير المظهر إلى: {theme_labels[selected_theme]}")
            st.rerun()
    
    with col2:
        st.markdown(f'<div class="section-title" style="font-size: 1.3rem;">🔔 إعدادات التنبيهات</div>', unsafe_allow_html=True)
        enable_alerts = st.checkbox("تفعيل التنبيهات", value=True)
        alert_threshold = st.slider("عتبة التنبيه", 0.1, 2.0, 1.0)
        
        st.markdown(f'<div class="section-title" style="font-size: 1.3rem;">💾 إعدادات البيانات</div>', unsafe_allow_html=True)
        data_retention = st.selectbox("فترة الاحتفاظ بالبيانات", ["أسبوع واحد", "شهر واحد", "3 أشهر", "سنة واحدة"])

elif current_page == "About":
    st.markdown(f'<div class="section-title">{current_texts["about"]}</div>', unsafe_allow_html=True)
    
    # معلومات المشروع الأساسية
    st.markdown(f"""
    <div class="info-card">
        <h3 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.5rem;">{current_texts['title']}</h3>
        <p style="color: {current_theme['text']}; font-size: 1.2rem; line-height: 1.8;">{current_texts['project_description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # الميزات الرئيسية
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['features']}</h4>
    """, unsafe_allow_html=True)
    
    for feature in current_texts['feature_list']:
        st.markdown(f'<div class="feature-item">• {feature}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # التقنيات المستخدمة
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['technologies']}</h4>
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
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['developer1']}</h4>
            <p style="color: {current_theme['text']}; margin-bottom: 1rem; font-size: 1.1rem;">{current_texts['main_developer']}</p>
            <div style="text-align: left; margin-top: 1.5rem;">
                <p style="color: {current_theme['text']}; margin: 0.5rem 0; font-size: 1rem;"><strong>{current_texts['email']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.5rem 0; font-size: 1rem;">rakan.almarri.2@aramco.com</p>
                <p style="color: {current_theme['text']}; margin: 0.5rem 0; font-size: 1rem;"><strong>{current_texts['phone']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.5rem 0; font-size: 1rem;">0532559664</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="contact-card">
            <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['developer2']}</h4>
            <p style="color: {current_theme['text']}; margin-bottom: 1rem; font-size: 1.1rem;">{current_texts['co_developer']}</p>
            <div style="text-align: left; margin-top: 1.5rem;">
                <p style="color: {current_theme['text']}; margin: 0.5rem 0; font-size: 1rem;"><strong>{current_texts['email']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.5rem 0; font-size: 1rem;">abdulrahman.alzhrani.1@aramco.com</p>
                <p style="color: {current_theme['text']}; margin: 0.5rem 0; font-size: 1rem;"><strong>{current_texts['phone']}</strong></p>
                <p style="color: {current_theme['primary']}; margin: 0.5rem 0; font-size: 1rem;">0549202674</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # معلومات الشركة
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['company_info']}</h4>
        <p style="color: {current_theme['text']}; line-height: 1.8; margin-bottom: 1.5rem; font-size: 1.1rem;">{current_texts['company_desc']}</p>
        
        <h4 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.3rem;">{current_texts['tech_support']}</h4>
        <p style="color: {current_theme['text']}; line-height: 1.8; font-size: 1.1rem;">{current_texts['support_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

# إضافة صفحات أخرى مع محتوى مطور
elif current_page in ["Reports", "Alerts", "Maintenance", "Users", "Simulation", "Analytics", "Advanced_Reports"]:
    st.markdown(f'<div class="section-title">{pages[current_page]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card" style="text-align: center; padding: 3rem;">
        <h3 style="color: {current_theme['primary']}; margin-bottom: 1.5rem; font-size: 1.5rem;">
            🚧 قيد التطوير
        </h3>
        <p style="color: {current_theme['text']}; font-size: 1.2rem; line-height: 1.6;">
            هذه الميزة قيد التطوير وستكون متاحة قريباً مع المزيد من الوظائف المتقدمة.
        </p>
    </div>
    """, unsafe_allow_html=True)

# تذييل الصفحة
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {current_theme['text']}; padding: 2rem; background: {current_theme['card']}; border-radius: 10px; margin-top: 2rem;">
    <small style="font-size: 1rem;">{current_texts['copyright']}</small>
</div>
""", unsafe_allow_html=True)

