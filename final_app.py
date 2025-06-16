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
        'solutions_generated': 'تم توليد الحلول الذكية بنجاح! 🎉',
        'under_development_title': '🚧 قيد التطوير',
        'under_development_message': 'هذه الميزة قيد التطوير وستكون متاحة قريباً مع المزيد من الوظائف المتقدمة.'
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
        'developer1': '👨‍💻 Rakan Almarri',
        'developer2': '👨‍💻 Abdulrahman Alzhrani',
        'main_developer': 'Main Developer',
        'co_developer': 'Co-Developer',
        'email': '📧 Email:',
        'phone': '📱 Phone:',
        'company_info': '🏢 Company Information',
        'company_desc': 'This system was developed as part of Saudi Aramco\'s innovation projects to develop smart solutions for industrial data monitoring and analysis.',
        'tech_support': '📧 For Technical Support:',
        'support_desc': 'For inquiries and technical support, please contact the development team via the email or phone mentioned above.',
        'features': '✨ Main Features:',
        'technologies': '🛠️ Technologies Used:',
        'copyright': '© 2024 Smart Neural Digital Twin - Saudi Aramco - All Rights Reserved',
        'project_description': 'An advanced system for real-time data monitoring and analysis using artificial intelligence and machine learning techniques.',
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
            'Machine Learning techniques'
        ],
        'generate_solutions': '🚀 Generate New Smart Solutions',
        'solution_name': 'Solution Name',
        'solution_description': 'Detailed Description',
        'effectiveness': 'Effectiveness Rate',
        'duration': 'Implementation Time',
        'priority': 'Priority Level',
        'category': 'Solution Category',
        'no_solutions': 'No solutions generated yet. Click the button above to generate custom smart solutions.',
        'solutions_generated': 'Smart solutions generated successfully! 🎉',
        'under_development_title': '🚧 Under Development',
        'under_development_message': 'This feature is under development and will be available soon with more advanced functionalities.'
    }
}

# الألوان للمظاهر المختلفة
theme_colors = {
    'Ocean': {'primary': '#1E90FF', 'secondary': '#ADD8E6', 'background': '#F0F8FF', 'text': '#000000', 'card_bg': '#FFFFFF', 'sidebar_bg': '#E0FFFF'},
    'Forest': {'primary': '#228B22', 'secondary': '#90EE90', 'background': '#F0FFF0', 'text': '#000000', 'card_bg': '#FFFFFF', 'sidebar_bg': '#E6FFE6'},
    'Sunset': {'primary': '#FF8C00', 'secondary': '#FFA07A', 'background': '#FFF5EE', 'text': '#000000', 'card_bg': '#FFFFFF', 'sidebar_bg': '#FFEFE6'},
    'Purple': {'primary': '#8A2BE2', 'secondary': '#D8BFD8', 'background': '#F8F0FF', 'text': '#000000', 'card_bg': '#FFFFFF', 'sidebar_bg': '#EFEOFF'},
    'Slate': {'primary': '#708090', 'secondary': '#B0C4DE', 'background': '#F5F5F5', 'text': '#000000', 'card_bg': '#FFFFFF', 'sidebar_bg': '#E8E8E8'},
    'Dark Ocean': {'primary': '#1E90FF', 'secondary': '#003366', 'background': '#001f3f', 'text': '#FFFFFF', 'card_bg': '#002b4f', 'sidebar_bg': '#001529'},
    'Dark Forest': {'primary': '#228B22', 'secondary': '#004d00', 'background': '#002b00', 'text': '#FFFFFF', 'card_bg': '#003d00', 'sidebar_bg': '#001a00'},
    'Dark Sunset': {'primary': '#FF8C00', 'secondary': '#8B4513', 'background': '#2b1a00', 'text': '#FFFFFF', 'card_bg': '#3d2500', 'sidebar_bg': '#1a0e00'},
    'Dark Purple': {'primary': '#8A2BE2', 'secondary': '#4B0082', 'background': '#1a0033', 'text': '#FFFFFF', 'card_bg': '#2c004f', 'sidebar_bg': '#100020'},
    'Dark Slate': {'primary': '#708090', 'secondary': '#2F4F4F', 'background': '#1c1c1c', 'text': '#FFFFFF', 'card_bg': '#2e2e2e', 'sidebar_bg': '#121212'}
}

current_theme_colors = theme_colors[st.session_state.theme]

# تطبيق الأنماط العامة
st.markdown(f"""
<style>
    body {{
        color: {current_theme_colors['text']};
        background-color: {current_theme_colors['background']};
    }}
    .stApp {{
        background-color: {current_theme_colors['background']};
    }}
    .stSidebar {{
        background-color: {current_theme_colors['sidebar_bg']} !important;
    }}
    .stButton>button {{
        background-color: {current_theme_colors['primary']};
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {current_theme_colors['secondary']};
    }}
    .stMetric {{
        background-color: {current_theme_colors['card_bg']};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: {current_theme_colors['text']};
    }}
    .stMetric label {{
        color: {current_theme_colors['text']} !important;
    }}
    .stMetric .st-emotion-cache-1g6go51 e1i5pmfg0 {{
         color: {current_theme_colors['text']} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {current_theme_colors['primary']};
    }}
    .card {{
        background-color: {current_theme_colors['card_bg']};
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }}
    .main-title {{
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        color: {current_theme_colors['primary']};
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    .subtitle {{
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 30px;
        color: {current_theme_colors['secondary'] if 'secondary' in current_theme_colors else current_theme_colors['text']};
    }}
    .section-header {{
        font-size: 2rem;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid {current_theme_colors['primary']};
        display: flex;
        align-items: center;
    }}
    .section-header i {{
        margin-right: 15px;
        font-size: 2.2rem;
    }}
    .sidebar-title {{
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 20px;
        color: {current_theme_colors['primary']};
        text-align: center;
    }}
    .footer {{
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        font-size: 0.9rem;
        color: {current_theme_colors['secondary'] if 'secondary' in current_theme_colors else current_theme_colors['text']};
        border-top: 1px solid {current_theme_colors['primary']};
    }}
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي
with st.sidebar:
    st.markdown(f'<p class="sidebar-title">{texts[st.session_state.language]["main_menu"]}</p>', unsafe_allow_html=True)
    
    menu_items = [
        ("dashboard", texts[st.session_state.language]["dashboard"]),
        ("assets", texts[st.session_state.language]["assets"]),
        ("operations", texts[st.session_state.language]["operations"]),
        ("security", texts[st.session_state.language]["security"]),
        ("finance", texts[st.session_state.language]["finance"]),
        ("energy", texts[st.session_state.language]["energy"]),
        ("quality", texts[st.session_state.language]["quality"]),
        ("compliance", texts[st.session_state.language]["compliance"]),
        ("simulation", texts[st.session_state.language]["simulation"]),
        ("analytics", texts[st.session_state.language]["analytics"]),
        ("solutions", texts[st.session_state.language]["solutions"]),
        ("reports", texts[st.session_state.language]["reports"]),
        ("alerts", texts[st.session_state.language]["alerts"]),
        ("maintenance", texts[st.session_state.language]["maintenance"]),
        ("users", texts[st.session_state.language]["users"]),
        ("settings", texts[st.session_state.language]["settings"]),
        ("about", texts[st.session_state.language]["about"])
    ]

    for item_key, item_name in menu_items:
        if st.button(item_name, key=f"sidebar_button_{item_key}", use_container_width=True):
            st.session_state.current_page = item_key

# دالة لعرض المحتوى تحت التطوير
def display_under_development():
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-tools"></i> {texts[lang]["under_development_title"]}</h2>', unsafe_allow_html=True)
    st.warning(texts[lang]["under_development_message"])
    st.image("https://via.placeholder.com/800x400.png?text=Under+Development", use_column_width=True)

# المحتوى الرئيسي
st.markdown(f'<h1 class="main-title">{texts[st.session_state.language]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{texts[st.session_state.language]["subtitle"]}</p>', unsafe_allow_html=True)

if st.session_state.current_page == "Dashboard":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-tachometer-alt"></i> {texts[lang]["dashboard"]}</h2>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    metrics = [("temperature", "temp"), ("pressure", "pressure"), ("vibration", "vibration"), ("gas", "gas")]
    units = ["°C", "PSI", "g", "ppm"]
    icons = ["🌡️", "📏", "📳", "💨"]

    for i, (metric_name, col_name) in enumerate(metrics):
        with cols[i]:
            latest_value = df[col_name].iloc[-1]
            st.metric(
                label=f"{icons[i]} {texts[lang][metric_name]}", 
                value=f"{latest_value:.1f} {units[i]}", 
                delta=f"{texts[lang]['last_reading']}: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
            )

    st.markdown(f'<h3 style="margin-top: 30px; color: {current_theme_colors["primary"]};">اتجاهات البيانات (آخر 7 أيام)</h3>', unsafe_allow_html=True)
    
    fig_temp = px.line(df.tail(24*7), x='timestamp', y='temp', title=texts[lang]["temperature"], labels={'timestamp': 'الوقت', 'temp': texts[lang]["temperature"]})
    fig_pressure = px.line(df.tail(24*7), x='timestamp', y='pressure', title=texts[lang]["pressure"], labels={'timestamp': 'الوقت', 'pressure': texts[lang]["pressure"]})
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_temp, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pressure, use_container_width=True)

elif st.session_state.current_page == "assets":
    display_under_development()
elif st.session_state.current_page == "operations":
    display_under_development()
elif st.session_state.current_page == "security":
    display_under_development()
elif st.session_state.current_page == "finance":
    display_under_development()
elif st.session_state.current_page == "energy":
    display_under_development()
elif st.session_state.current_page == "quality":
    display_under_development()
elif st.session_state.current_page == "compliance":
    display_under_development()

elif st.session_state.current_page == "simulation":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-cogs"></i> {texts[lang]["simulation"]}</h2>', unsafe_allow_html=True)
    st.write("هنا يمكنك إجراء محاكاة لسيناريوهات مختلفة وتأثيرها على النظام.")
    # محتوى المحاكاة هنا
    display_under_development() # مؤقتًا

elif st.session_state.current_page == "analytics":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-chart-line"></i> {texts[lang]["analytics"]}</h2>', unsafe_allow_html=True)
    st.write("هنا يتم عرض التحليلات التنبؤية المتقدمة بناءً على البيانات التاريخية.")
    # محتوى التحليل التنبؤي هنا
    display_under_development() # مؤقتًا

elif st.session_state.current_page == "solutions":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-lightbulb"></i> {texts[lang]["solutions"]}</h2>', unsafe_allow_html=True)
    
    if st.button(texts[lang]["generate_solutions"], key="generate_solutions_button"):
        num_solutions_to_generate = random.randint(10, 15) # توليد عدد عشوائي من الحلول بين 10 و 15
        st.session_state.generated_solutions = random.sample(smart_solutions_db[lang], k=min(num_solutions_to_generate, len(smart_solutions_db[lang])))
        st.success(texts[lang]["solutions_generated"])

    if not st.session_state.generated_solutions:
        st.info(texts[lang]["no_solutions"])
    else:
        for sol in st.session_state.generated_solutions:
            effectiveness_color = "#4CAF50" if sol['effectiveness'] >= 90 else ("#FFC107" if sol['effectiveness'] >= 80 else "#F44336")
            priority_color = {
                'عالية جداً': '#D32F2F', 'Very High': '#D32F2F',
                'عالية': '#FF7043', 'High': '#FF7043',
                'متوسطة': '#FFEE58', 'Medium': '#FFEE58',
                'منخفضة': '#66BB6A', 'Low': '#66BB6A'
            }.get(sol['priority'], '#757575')

            st.markdown(f"""
            <div class="card">
                <h3 style="color: {current_theme_colors['primary']};">{sol['name']}</h3>
                <p><strong>{texts[lang]['solution_description']}:</strong> {sol['description']}</p>
                <p><strong>{texts[lang]['category']}:</strong> <span style="background-color: {current_theme_colors['secondary']}; color: {current_theme_colors['primary']}; padding: 3px 8px; border-radius: 5px;">{sol['category']}</span></p>
                <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                    <p><strong>{texts[lang]['effectiveness']}:</strong> <span style="color: {effectiveness_color}; font-weight: bold;">{sol['effectiveness']}%</span></p>
                    <p><strong>{texts[lang]['duration']}:</strong> {sol['duration']}</p>
                    <p><strong>{texts[lang]['priority']}:</strong> <span style="color: {priority_color}; font-weight: bold;">{sol['priority']}</span></p>
                </div>
                <div style="background-color: #e0e0e0; border-radius: 5px; margin-top: 10px; height: 10px;">
                    <div style="background-color: {effectiveness_color}; width: {sol['effectiveness']}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.current_page == "reports":
    display_under_development()
elif st.session_state.current_page == "alerts":
    display_under_development()
elif st.session_state.current_page == "maintenance":
    display_under_development()
elif st.session_state.current_page == "users":
    display_under_development()

elif st.session_state.current_page == "settings":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-cog"></i> {texts[lang]["settings"]}</h2>', unsafe_allow_html=True)
    
    st.markdown(f'<h4>{texts[lang]["language_settings"]}</h4>', unsafe_allow_html=True)
    new_language = st.radio(
        texts[lang]["select_language"],
        ('العربية', 'English'), 
        index=0 if st.session_state.language == 'العربية' else 1,
        key="language_radio"
    )
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.rerun()

    st.markdown(f'<h4 style="margin-top: 30px;">{texts[lang]["theme_settings"]}</h4>', unsafe_allow_html=True)
    theme_options_display = {
        'Ocean': texts[lang]['ocean'], 'Forest': texts[lang]['forest'], 
        'Sunset': texts[lang]['sunset'], 'Purple': texts[lang]['purple'], 
        'Slate': texts[lang]['slate'],
        'Dark Ocean': f"{texts[lang]['ocean']} (Dark)", 'Dark Forest': f"{texts[lang]['forest']} (Dark)",
        'Dark Sunset': f"{texts[lang]['sunset']} (Dark)", 'Dark Purple': f"{texts[lang]['purple']} (Dark)",
        'Dark Slate': f"{texts[lang]['slate']} (Dark)"
    }
    theme_keys = list(theme_colors.keys())
    current_theme_index = theme_keys.index(st.session_state.theme)
    
    new_theme_display = st.selectbox(
        texts[lang]["select_theme"], 
        options=[theme_options_display[key] for key in theme_keys],
        index=current_theme_index,
        key="theme_selectbox"
    )
    # الحصول على مفتاح الثيم من الاسم المعروض
    new_theme_key = [key for key, display_name in theme_options_display.items() if display_name == new_theme_display][0]

    if new_theme_key != st.session_state.theme:
        st.session_state.theme = new_theme_key
        st.rerun()

elif st.session_state.current_page == "about":
    lang = st.session_state.language
    st.markdown(f'<h2 class="section-header"><i class="fas fa-info-circle"></i> {texts[lang]["about"]}</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <h3 style="color: {current_theme_colors['primary']};">{texts[lang]['project_description']}</h3>
        <p>{texts[lang]['company_desc']}</p>
        
        <h4 style="color: {current_theme_colors['primary']}; margin-top: 20px;">{texts[lang]['features']}</h4>
        <ul>
            {''.join([f'<li>{item}</li>' for item in texts[lang]['feature_list']])}
        </ul>
        
        <h4 style="color: {current_theme_colors['primary']}; margin-top: 20px;">{texts[lang]['technologies']}</h4>
        <ul>
            {''.join([f'<li>{item}</li>' for item in texts[lang]['tech_list']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<h3 style="color: {current_theme_colors["primary"]}; margin-top: 30px;">{texts[lang]["contact_info"]}</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>{texts[lang]['developer1']}</h4>
            <p><strong>{texts[lang]['main_developer']}</strong></p>
            <p>{texts[lang]['email']} rakan.almarri.2@aramco.com</p>
            <p>{texts[lang]['phone']} 0532559664</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>{texts[lang]['developer2']}</h4>
            <p><strong>{texts[lang]['co_developer']}</strong></p>
            <p>{texts[lang]['email']} abdulrahman.alzhrani.1@aramco.com</p>
            <p>{texts[lang]['phone']} 0549202674</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-top: 20px;">
        <h4 style="color: {current_theme_colors['primary']};">{texts[lang]['tech_support']}</h4>
        <p>{texts[lang]['support_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

# التذييل
st.markdown(f'<div class="footer">{texts[st.session_state.language]["copyright"]}</div>', unsafe_allow_html=True)

# إضافة أيقونات Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)


