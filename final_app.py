import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# ----- Custom CSS for Peak Visuals and RTL/LTR -----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@700&family=Montserrat:wght@700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Montserrat', 'Cairo', 'Arial', sans-serif;
        background: linear-gradient(120deg, #232526 0%, #414345 100%) !important;
    }
    .peak-card {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,.15);
        margin-bottom: 1.5em;
        padding: 1.5em 2em;
        transition: box-shadow 0.2s;
    }
    .peak-card:hover {
        box-shadow: 0 12px 38px 0 rgba(31,38,135,.28);
    }
    .kpi-card {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        border-radius: 13px;
        color: #fff !important;
        font-size: 1.25em;
        font-weight: 700;
        box-shadow: 0 8px 24px 0 rgba(31,38,135,.10);
        padding: 1.3em 1.3em;
        text-align: center;
        margin-bottom: 1em;
    }
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif !important;
    }
    .ltr {
        direction: ltr;
        text-align: left;
        font-family: 'Montserrat', sans-serif !important;
    }
    .color-accent {
        color: #43cea2 !important;
    }
    .color-accent2 {
        color: #ffb347 !important;
    }
    .color-title {
        color: #185a9d;
        font-weight:900;
    }
    .sidebar-title {
        font-size: 2em !important;
        font-weight: 900 !important;
        color: #43cea2 !important;
        letter-spacing: 0.5px;
        margin-bottom: 0.2em !important;
    }
    .sidebar-subtitle {
        font-size: 1.15em !important;
        color: #cfdef3 !important;
        margin-bottom: 1em;
        margin-top: -.7em !important;
    }
    .gradient-header {
        font-weight: 900;
        font-size: 2.1em;
        background: linear-gradient(90deg,#43cea2,#185a9d 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3em;
    }
    .gradient-ar {
        font-weight: 900;
        font-size: 2.1em;
        background: linear-gradient(90deg,#43cea2,#185a9d 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3em;
        font-family: 'Cairo', sans-serif !important;
    }
    .chat-bubble-user {
        background: #43cea2;
        color: #fff;
        border-radius: 17px 17px 2px 17px;
        padding: 1em 1.3em;
        margin: 1em 0;
        align-self: flex-end;
        font-size:1.1em;
        max-width: 70%;
        box-shadow: 0 1px 7px #185a9d33;
    }
    .chat-bubble-ai {
        background: #fff;
        color: #185a9d;
        border-radius: 17px 17px 17px 2px;
        padding: 1em 1.3em;
        margin: 1em 0;
        align-self: flex-start;
        font-size:1.1em;
        max-width: 70%;
        box-shadow: 0 1px 7px #185a9d22;
    }
    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 16px;
        margin-right: 8px;
        vertical-align: middle;
        background: #43cea2;
        display: inline-block;
        text-align: center;
        color: #fff;
        font-size: 1.5em;
        line-height: 1.6em;
    }
    .timeline-step {
        border-left: 4px solid #43cea2;
        margin-left: 0.8em;
        padding-left: 1.2em;
        margin-bottom: 1em;
        position: relative;
    }
    .timeline-step:before {
        content: '';
        position: absolute;
        left: -14px;
        top: 0.18em;
        width: 18px;
        height: 18px;
        background: #43cea2;
        border-radius: 100%;
        border: 2px solid #fff;
    }
    .timeline-icon {
        font-size: 1.5em;
        margin-right: 0.5em;
        vertical-align: middle;
    }
    .solution-btn {
        background: linear-gradient(90deg,#43cea2,#185a9d 80%);
        color: #fff;
        font-size:1.13em;
        border-radius: 9px;
        border: none;
        padding: 0.6em 1.6em;
        margin: 1em 0 1.6em 0;
        font-weight: 900;
        cursor: pointer;
        transition: background 0.17s;
    }
    .solution-btn:hover {
        background: linear-gradient(90deg,#185a9d,#43cea2 80%);
        color:#fff;
    }
    .alert-log-table td, .alert-log-table th {
        padding: 0.6em 1.2em;
        border-bottom: 1px solid #eee;
        font-size: 1.1em;
    }
    .alert-log-table th {
        background: #43cea233;
        color: #185a9d;
        font-weight: 700;
    }
    .alert-log-table tr:last-child td {
        border-bottom: none;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Translations -----
texts = {
    'en': {
        "app_title": "Smart Neural Digital Twin",
        "app_sub": "Intelligent Digital Plant Platform",
        "side_sections": [
            "Digital Twin", "Advanced Dashboard", "Predictive Analytics", "Scenario Playback",
            "Alerts & Fault Log", "Smart Solutions", "KPI Wall", "Plant Heatmap", "Root Cause Explorer",
            "AI Copilot Chat", "Live Plant 3D", "Incident Timeline", "Energy Optimization", "Future Insights", "About"
        ],
        "contact": "Contact Info",
        "lang_en": "English",
        "lang_ar": "Arabic",
        "restart": "Restart",
        "next": "Next",
        "solution_btn": "Next Solution",
        "see_details": "See Details",
        "acknowledge": "Acknowledge",
        "mission": "Mission Statement",
        "features": "Features",
        "howto": "How to extend",
        "developer": "Developer",
        "name": "Name",
        "demo_note": "Demo use only: Not for live plant operation",
        "kpi": ["Avg Temp", "Avg Pressure", "Avg Methane"],
        "kpi_wall": [
            ("Overall Efficiency", "✅", "#8fd3f4"),
            ("Energy Used (MWh)", "⚡", "#fe8c00"),
            ("Water Saved (m³)", "💧", "#43cea2"),
            ("Incidents This Year", "🛑", "#fa709a")
        ],
        "digital_twin_header": "Digital Plant Schematic",
        "digital_twin_intro": "Explore the interactive schematic below. Click any unit for live status and info.",
        "plant_units": ["Pump", "Compressor", "Reactor", "Separator", "Boiler"],
        "plant_status": [
            ("Healthy", "#43cea2"), ("Warning", "#ffb347"), ("Critical", "#fa709a")
        ],
        "plant_overlay": "Live Status: All systems nominal. Methane levels rising at Compressor 2.",
        "adv_dash_header": "Operations Dashboard",
        "adv_dash_intro": "Monitor plant KPIs, live sensor data, and distribution.",
        "pie_labels": ["Gas", "Oil", "Water"],
        "pie_title": "Product Distribution",
        "pred_header": "AI Forecast & Anomaly Detection",
        "pred_intro": "AI predicts temperature and methane spikes. Red markers show anomalies.",
        "scen_header": "Emergency Scenario Playback",
        "scen_steps": [
            {
                "title": "All Normal",
                "icon": "🟢",
                "desc": "Plant running steady, all KPIs within threshold.",
            },
            {
                "title": "Methane Leak Detected",
                "icon": "🟡",
                "desc": "AI detects methane spike at Compressor 2. Alert sent to operator.",
            },
            {
                "title": "Manual Response Delayed",
                "icon": "🔴",
                "desc": "Operator missed initial alert. Leak spreads, risk increases.",
            },
            {
                "title": "AI Emergency Shutdown",
                "icon": "🟢",
                "desc": "AI triggers rapid plant shutdown. Losses minimized.",
            },
            {
                "title": "Post-Incident Review",
                "icon": "🔵",
                "desc": "Root-cause analysis and smart solutions recommended.",
            },
        ],
        "alerts_header": "Recent Alerts & Faults",
        "alerts_table_head": ["Time", "Severity", "Unit", "Message", "Action"],
        "alerts_log": [
            ["2025-06-30 11:23", "Critical", "Compressor 2", "Methane leak detected", "See Details"],
            ["2025-06-30 11:20", "Warning", "Pump 1", "Flow rate below threshold", "See Details"],
            ["2025-06-30 10:50", "Info", "Reactor", "Temperature trend up", "See Details"]
        ],
        "alerts_colors": {"Critical": "#fa709a", "Warning": "#ffb347", "Info": "#43cea2"},
        "solutions_header": "AI Smart Solution",
        "solutions": [
            {
                "title": "Automated Methane Leak Response",
                "desc": "Integrate advanced sensors with automated shutdown logic to instantly contain future methane leaks.",
                "steps": ["Deploy new IoT sensors", "Implement AI detection", "Link to emergency shutdown", "Train operators"],
                "priority": "High", "effectiveness": "High", "time": "3 days", "cost": "$4,000", "savings": "$25,000/year",
                "icon": "🛡️"
            },
            {
                "title": "Pump Predictive Maintenance",
                "desc": "Monitor vibration and temperature to predict pump failures before they occur.",
                "steps": ["Install vibration sensors", "Run ML models", "Alert on anomaly", "Schedule just-in-time maintenance"],
                "priority": "Medium", "effectiveness": "High", "time": "1 week", "cost": "$5,000", "savings": "$18,000/year",
                "icon": "🔧"
            },
            {
                "title": "Energy Use Optimization",
                "desc": "AI analyzes compressor schedule to cut energy waste by 11%.",
                "steps": ["Analyze compressor cycles", "Optimize schedule", "Implement load shifting", "Track savings"],
                "priority": "High", "effectiveness": "Medium", "time": "2 weeks", "cost": "$6,000", "savings": "$32,000/year",
                "icon": "⚡"
            },
        ],
        "kpiwall_header": "Key Performance Indicators",
        "heatmap_header": "Plant Heatmap",
        "heatmap_intro": "High temperature and pressure zones are highlighted below.",
        "root_cause_header": "Root Cause Explorer",
        "root_cause_intro": "Trace issues to their origin. Sample propagation path shown below.",
        "ai_chat_header": "AI Copilot Chat",
        "ai_chat_intro": "Ask the AI about plant issues, troubleshooting, or improvements.",
        "chat_examples": [
            ("How do I fix a pump overheat?", "Check pump cooling, inspect for obstructions, and review maintenance logs."),
            ("Why is methane trending up?", "Possible leak detected. Check Compressor 2 sensors and run diagnostics."),
            ("How to reduce energy costs?", "Optimize compressor cycles and schedule maintenance during off-peak hours.")
        ],
        "live3d_header": "Live Plant 3D",
        "live3d_intro": "Explore the interactive 3D model below. Use your mouse to zoom, rotate, and explore the plant!",
        "timeline_header": "Incident Timeline",
        "timeline_steps": [
            ("2025-06-30 11:23", "🛑", "Methane leak detected at Compressor 2. Emergency shutdown triggered."),
            ("2025-06-30 10:58", "⚠️", "Flow rate anomaly at Pump 1. Operator notified."),
            ("2025-06-30 10:50", "ℹ️", "Temperature rising at Reactor. Trend within safe limits.")
        ],
        "energy_header": "Energy Optimization",
        "energy_intro": "Monitor and optimize plant energy use. AI recommendations below.",
        "energy_recos": [
            ("Reduce compressor load during peak hours", "⚡"),
            ("Schedule maintenance for low demand windows", "🛠️")
        ],
        "future_header": "Future Insights",
        "future_cards": [
            ("Predictive Risk Alert", "AI models forecast a risk spike for methane at Compressor 2 next week.", "🚨"),
            ("Efficiency Opportunity", "Upgrade control logic to boost plant efficiency by 3%.", "🌱")
        ],
        "about_header": "About the Platform",
        "mission_statement": "This digital twin combines AI, real-time data, and process expertise to deliver safer, more efficient operations.",
        "feature_list": [
            "Interactive digital twin schematic",
            "Advanced dashboards and KPIs",
            "AI-driven fault detection and smart solutions",
            "Root-cause explorer and scenario playback",
            "Live 3D plant visualization",
            "Bilingual support and peak design"
        ],
        "howto_extend": [
            "Connect to real plant historian data",
            "Add custom plant schematics and overlays",
            "Integrate with control and alerting systems",
            "Deploy on secure internal networks"
        ],
        "developers": [
            ("Rakan Almarri", "rakan.almarri.2@aramco.com", "0532559664"),
            ("Abdulrahman Alzahrani", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
    },
    'ar': {
        "app_title": "التوأم الرقمي العصبي الذكي",
        "app_sub": "منصة المصنع الذكي الرقمي",
        "side_sections": [
            "التوأم الرقمي", "لوحة القيادة المتقدمة", "التحليلات التنبؤية", "تشغيل السيناريو",
            "التنبيهات وسجل الأعطال", "الحلول الذكية", "جدار المؤشرات", "خريطة حرارة المصنع", "مستكشف السبب الجذري",
            "محادثة الذكاء الصناعي", "مصنع ثلاثي الأبعاد", "جدول الحوادث", "تحسين الطاقة", "رؤى مستقبلية", "حول النظام"
        ],
        "contact": "معلومات التواصل",
        "lang_en": "الإنجليزية",
        "lang_ar": "العربية",
        "restart": "إعادة تشغيل",
        "next": "التالي",
        "solution_btn": "الحل التالي",
        "see_details": "عرض التفاصيل",
        "acknowledge": "تأكيد",
        "mission": "بيان المهمة",
        "features": "الميزات",
        "howto": "كيفية التوسيع",
        "developer": "المطور",
        "name": "الاسم",
        "demo_note": "للعرض فقط: غير مخصص للتشغيل الفعلي",
        "kpi": ["متوسط الحرارة", "متوسط الضغط", "متوسط الميثان"],
        "kpi_wall": [
            ("الكفاءة العامة", "✅", "#8fd3f4"),
            ("الطاقة المستخدمة (ميغاواط)", "⚡", "#fe8c00"),
            ("الماء الموفر (م³)", "💧", "#43cea2"),
            ("الحوادث هذا العام", "🛑", "#fa709a")
        ],
        "digital_twin_header": "مخطط المصنع الرقمي",
        "digital_twin_intro": "تفاعل مع المخطط أدناه. اضغط على أي وحدة لرؤية حالتها ومعلوماتها المباشرة.",
        "plant_units": ["مضخة", "ضاغط", "مفاعل", "فاصل", "غلاية"],
        "plant_status": [
            ("سليم", "#43cea2"), ("تحذير", "#ffb347"), ("حرج", "#fa709a")
        ],
        "plant_overlay": "الحالة المباشرة: جميع الأنظمة مستقرة. ارتفاع مستويات الميثان عند الضاغط ٢.",
        "adv_dash_header": "لوحة العمليات",
        "adv_dash_intro": "راقب مؤشرات الأداء والبيانات الحية وتوزيع المنتجات.",
        "pie_labels": ["غاز", "نفط", "ماء"],
        "pie_title": "توزيع المنتجات",
        "pred_header": "توقعات وتحليل الذكاء الاصطناعي",
        "pred_intro": "يتوقع الذكاء الاصطناعي ارتفاع الحرارة والميثان. النقاط الحمراء تشير لشذوذ.",
        "scen_header": "تشغيل سيناريو الطوارئ",
        "scen_steps": [
            {
                "title": "الوضع طبيعي",
                "icon": "🟢",
                "desc": "تشغيل مستقر وجميع المؤشرات ضمن الحدود.",
            },
            {
                "title": "اكتشاف تسرب ميثان",
                "icon": "🟡",
                "desc": "كشف الذكاء الاصطناعي ارتفاع الميثان عند الضاغط ٢. تم إرسال تنبيه.",
            },
            {
                "title": "تأخر الاستجابة اليدوية",
                "icon": "🔴",
                "desc": "تجاهل المشغل التنبيه الأولي. انتشر التسرب وارتفع الخطر.",
            },
            {
                "title": "إيقاف طارئ آلي",
                "icon": "🟢",
                "desc": "قام الذكاء الاصطناعي بإيقاف المصنع بسرعة وتقليل الخسائر.",
            },
            {
                "title": "مراجعة ما بعد الحادث",
                "icon": "🔵",
                "desc": "تحليل السبب الجذري وتوصية حلول ذكية.",
            },
        ],
        "alerts_header": "التنبيهات والأعطال الأخيرة",
        "alerts_table_head": ["الوقت", "الخطورة", "الوحدة", "الرسالة", "الإجراء"],
        "alerts_log": [
            ["2025-06-30 11:23", "حرج", "ضاغط ٢", "تم اكتشاف تسرب ميثان", "عرض التفاصيل"],
            ["2025-06-30 11:20", "تحذير", "مضخة ١", "انخفاض معدل التدفق عن الحد", "عرض التفاصيل"],
            ["2025-06-30 10:50", "معلومات", "مفاعل", "اتجاه ارتفاع الحرارة", "عرض التفاصيل"]
        ],
        "alerts_colors": {"حرج": "#fa709a", "تحذير": "#ffb347", "معلومات": "#43cea2"},
        "solutions_header": "حل ذكي من الذكاء الاصطناعي",
        "solutions": [
            {
                "title": "استجابة آلية لتسرب الميثان",
                "desc": "دمج حساسات متطورة مع منطق إيقاف تلقائي لاحتواء التسربات فوراً.",
                "steps": ["تركيب حساسات إنترنت الأشياء", "تفعيل كشف الذكاء الاصطناعي", "ربط بالإيقاف الطارئ", "تدريب المشغلين"],
                "priority": "عالية", "effectiveness": "عالية", "time": "٣ أيام", "cost": "$٤٬٠٠٠", "savings": "$٢٥٬٠٠٠/سنة",
                "icon": "🛡️"
            },
            {
                "title": "صيانة استباقية للمضخات",
                "desc": "مراقبة الاهتزازات والحرارة للتنبؤ بالأعطال قبل وقوعها.",
                "steps": ["تركيب حساسات الاهتزاز", "تشغيل نماذج التعلم الآلي", "تنبيه عند وجود شذوذ", "جدولة صيانة فورية"],
                "priority": "متوسطة", "effectiveness": "عالية", "time": "أسبوع", "cost": "$٥٬٠٠٠", "savings": "$١٨٬٠٠٠/سنة",
                "icon": "🔧"
            },
            {
                "title": "تحسين استهلاك الطاقة",
                "desc": "تحلل الذكاء الاصطناعي جدول الضواغط لخفض الهدر بنسبة ١١٪.",
                "steps": ["تحليل دورات الضواغط", "تحسين الجدولة", "تطبيق نقل الأحمال", "متابعة التوفير"],
                "priority": "عالية", "effectiveness": "متوسطة", "time": "أسبوعان", "cost": "$٦٬٠٠٠", "savings": "$٣٢٬٠٠٠/سنة",
                "icon": "⚡"
            },
        ],
        "kpiwall_header": "مؤشرات الأداء الرئيسية",
        "heatmap_header": "خريطة حرارة المصنع",
        "heatmap_intro": "المناطق الحرجة للحرارة والضغط موضحة أدناه.",
        "root_cause_header": "مستكشف السبب الجذري",
        "root_cause_intro": "تتبع المشكلات إلى أصلها. سلسلة السبب والنتيجة أدناه.",
        "ai_chat_header": "محادثة الذكاء الصناعي",
        "ai_chat_intro": "اسأل الذكاء الصناعي عن الأعطال أو التحسينات أو الشرح.",
        "chat_examples": [
            ("كيف أعالج ارتفاع حرارة المضخة؟", "افحص التبريد، وتأكد من عدم وجود انسدادات وراجع سجل الصيانة."),
            ("لماذا يرتفع الميثان؟", "احتمال وجود تسرب. تحقق من حساسات الضاغط ٢ وشغل التشخيص."),
            ("كيف أخفض التكاليف؟", "حسن دورات الضواغط وجدول الصيانة خارج أوقات الذروة.")
        ],
        "live3d_header": "مصنع ثلاثي الأبعاد مباشر",
        "live3d_intro": "تفاعل مع النموذج ثلاثي الأبعاد أدناه. استخدم الماوس لتحريك وتكبير المصنع!",
        "timeline_header": "جدول الحوادث",
        "timeline_steps": [
            ("2025-06-30 11:23", "🛑", "اكتشاف تسرب ميثان عند الضاغط ٢. إيقاف طارئ تلقائي."),
            ("2025-06-30 10:58", "⚠️", "شذوذ تدفق عند مضخة ١. تم إخطار المشغل."),
            ("2025-06-30 10:50", "ℹ️", "ارتفاع الحرارة بالمفاعل. الاتجاه ضمن الحدود الآمنة.")
        ],
        "energy_header": "تحسين الطاقة",
        "energy_intro": "راقب وحسن استهلاك الطاقة. توصيات الذكاء الاصطناعي أدناه.",
        "energy_recos": [
            ("تخفيض تشغيل الضواغط أوقات الذروة", "⚡"),
            ("جدولة الصيانة أوقات الطلب المنخفض", "🛠️")
        ],
        "future_header": "رؤى مستقبلية",
        "future_cards": [
            ("تنبيه مخاطر تنبؤي", "يتوقع الذكاء الاصطناعي ارتفاع الميثان عند الضاغط ٢ الأسبوع القادم.", "🚨"),
            ("فرصة كفاءة", "تحديث منطق التحكم لرفع الكفاءة ٣٪.", "🌱")
        ],
        "about_header": "حول المنصة",
        "mission_statement": "يجمع هذا التوأم الرقمي بين الذكاء الاصطناعي والبيانات الحية وخبرة العمليات لتحقيق تشغيل أكثر أماناً وكفاءة.",
        "feature_list": [
            "مخطط توأم رقمي تفاعلي",
            "لوحات ومؤشرات متقدمة",
            "كشف أعطال ذكي وحلول فورية",
            "مستكشف السبب الجذري وتشغيل السيناريوهات",
            "رؤية ثلاثية الأبعاد للمصنع",
            "دعم لغتين وتصميم عصري"
        ],
        "howto_extend": [
            "ربط مع بيانات المصنع الحقيقية",
            "إضافة مخططات وتراكب مخصص",
            "دمج مع أنظمة التحكم والتنبيه",
            "تشغيل داخلي آمن"
        ],
        "developers": [
            ("راكان المعاري", "rakan.almarri.2@aramco.com", "0532559664"),
            ("عبدالرحمن الزهراني", "abdulrahman.alzhrani.2@aramco.com", "0549202574")
        ],
    }
}

# Language state
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "scenario_step" not in st.session_state:
    st.session_state["scenario_step"] = 0
if "solution_idx" not in st.session_state:
    st.session_state["solution_idx"] = 0

# Language selection
with st.sidebar:
    st.markdown(
        f"""<div class="sidebar-title">{texts[st.session_state["lang"]]["app_title"]}</div>
            <div class="sidebar-subtitle">{texts[st.session_state["lang"]]["app_sub"]}</div>""", unsafe_allow_html=True)
    lang_sel = st.radio(
        "", (texts["en"]["lang_en"], texts["en"]["lang_ar"]) if st.session_state["lang"] == "en" else (texts["ar"]["lang_en"], texts["ar"]["lang_ar"]),
        horizontal=True, index=0 if st.session_state["lang"] == "en" else 1
    )
    st.session_state["lang"] = "en" if lang_sel == texts["en"]["lang_en"] else "ar"
    # Section nav
    section_list = texts[st.session_state["lang"]]["side_sections"]
    section = st.radio(" ", section_list, index=0)

    st.markdown("---")
    st.markdown(f"<b>{texts[st.session_state['lang']]['contact']}</b>", unsafe_allow_html=True)
    for name, mail, phone in texts[st.session_state["lang"]]["developers"]:
        st.write(f"- {name}: {mail} ({phone})")

lang = st.session_state["lang"]
T = texts[lang]
rtl = True if lang == "ar" else False

def rtl_wrap(txt):
    return f'<div class="rtl">{txt}</div>' if rtl else f'<div class="ltr">{txt}</div>'

def kpi_number(val, unit="", trend=0):
    trend_icon = "▲" if trend > 0 else "▼" if trend < 0 else "●"
    trend_color = "#43cea2" if trend >= 0 else "#fa709a"
    return f"""<span style="font-size:1.6em;font-weight:900;">{val}{unit}</span><br>
            <span style="color:{trend_color};font-size:1.1em;">{trend_icon} {abs(trend):.2f}</span>"""

# ---------- DEMO DATA ----------
np.random.seed(0)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})
pie_data = [56, 31, 13]  # Gas, Oil, Water

# ---------- MAIN PAGE LOGIC -----------

# Section logic
if section == T["side_sections"][0]:  # Digital Twin
    st.markdown(
        f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['digital_twin_header']}</div>""",
        unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["digital_twin_intro"]), unsafe_allow_html=True)

    # Plant schematic
    fig = go.Figure()
    # Plant coordinates for units
    coords = [(2, 3), (4.5, 6), (7, 4), (5.5, 2), (3.2, 6.2)]
    colors = [T["plant_status"][0][1], T["plant_status"][1][1], T["plant_status"][2][1], T["plant_status"][0][1], T["plant_status"][0][1]]
    for i, (x, y) in enumerate(coords):
        fig.add_shape(type="circle", x0=x-0.7, y0=y-0.7, x1=x+0.7, y1=y+0.7,
                      fillcolor=colors[i], line=dict(color="#185a9d", width=3))
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="text",
            text=[f"<b style='font-size:1.3em'>{T['plant_units'][i]}</b>"],
            textposition="top center", textfont=dict(size=18, family="Cairo" if rtl else "Montserrat")
        ))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=350, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#e0eafc")
    st.plotly_chart(fig, use_container_width=True)
    # Status overlay cards
    st.markdown(f"""<div class="peak-card">{rtl_wrap(T['plant_overlay'])}</div>""", unsafe_allow_html=True)
    # Plant unit status cards
    st.markdown("<div style='display:flex;gap:1.5em;flex-wrap:wrap;'>", unsafe_allow_html=True)
    for i, (stat, clr) in enumerate(T["plant_status"]):
        st.markdown(f"""<div class="kpi-card" style="background:{clr}a8;min-width:160px;">
            {T['plant_units'][i]}<br>{stat}</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][1]:  # Advanced Dashboard
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['adv_dash_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["adv_dash_intro"]), unsafe_allow_html=True)
    # KPIs
    cols = st.columns(3)
    for i, c in enumerate(cols):
        c.markdown(f"""<div class="kpi-card" style='background:#43cea2c0'>{T['kpi'][i]}<br>
            {kpi_number(demo_df.iloc[-1, i+1], "°C" if i==0 else " bar" if i==1 else " ppm", demo_df.iloc[-1, i+1]-demo_df.iloc[-2, i+1])}</div>""", unsafe_allow_html=True)
    # Trend chart
    fig = go.Figure()
    for col in ["Temperature", "Pressure", "Methane"]:
        fig.add_trace(go.Scatter(x=demo_df["time"], y=demo_df[col], mode="lines", name=col if lang=="en" else {"Temperature":"الحرارة","Pressure":"الضغط","Methane":"الميثان"}[col]))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=270, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    # Pie chart
    fig2 = go.Figure(data=[go.Pie(labels=T["pie_labels"], values=pie_data, hole=.35)])
    fig2.update_traces(textinfo='label+percent', marker=dict(line=dict(color='#fff', width=2)))
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=210, title={"text":T["pie_title"], "y":0.93, "x":0.5, "xanchor":"center", "yanchor":"top"})
    st.plotly_chart(fig2, use_container_width=True)

elif section == T["side_sections"][2]:  # Predictive Analytics
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['pred_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["pred_intro"]), unsafe_allow_html=True)
    # AI forecast chart
    history = demo_df["Temperature"].values
    future = [history[-1] + 0.4*i + np.random.normal(0,0.4) for i in range(1, 13)]
    times = [demo_df["time"].iloc[-1] + timedelta(hours=i) for i in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=demo_df["time"], y=history, mode="lines+markers", name=T["kpi"][0]))
    fig.add_trace(go.Scatter(x=times, y=future, mode="lines+markers", name=("Forecast" if lang=="en" else "توقع"), line=dict(dash="dash", color="#fa709a")))
    # Mark 2 anomalies
    fig.add_trace(go.Scatter(x=[times[6], times[10]], y=[future[6], future[10]], mode="markers", marker=dict(color="#fa709a", size=18, symbol="star")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    # Info
    st.markdown(f"""<div class="peak-card">{rtl_wrap(("AI flagged 2 extreme values in the forecast. Review recommended!" if lang=="en" else "حدد الذكاء الاصطناعي نقطتين شاذتين في التوقع. يوصى بالمراجعة!"))}</div>""", unsafe_allow_html=True)

elif section == T["side_sections"][3]:  # Scenario Playback
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['scen_header']}</div>""", unsafe_allow_html=True)
    step = st.session_state["scenario_step"]
    scen = T["scen_steps"][step]
    # Progress bar
    st.progress((step+1)/len(T["scen_steps"]))
    st.markdown(f"""<div class="peak-card" style="background:#e0eafc;">
        <span class="timeline-icon">{scen['icon']}</span>
        <b>{scen['title']}</b><br>
        <span style='font-size:1.09em'>{scen['desc']}</span>
    </div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button(T["restart"]):
        st.session_state["scenario_step"] = 0
    if col2.button(T["next"]):
        st.session_state["scenario_step"] = min(len(T["scen_steps"])-1, step+1)
    # Timeline preview
    st.markdown(f"""<div style='margin-top:1.8em;'>
        <b>{T['timeline_header']}</b>
        <div style='margin-top:0.7em;'>
        {"".join([f"<div class='timeline-step'><span class='timeline-icon'>{s['icon']}</span>{s['title']}<br><span style='font-size:0.96em'>{s['desc']}</span></div>" for s in T["scen_steps"][:step+1]])}
        </div></div>""", unsafe_allow_html=True)

elif section == T["side_sections"][4]:  # Alerts & Fault Log
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['alerts_header']}</div>""", unsafe_allow_html=True)
    st.markdown("<div class='peak-card' style='padding:0;overflow-x:auto;'>", unsafe_allow_html=True)
    # Alerts table
    table = f"""<table class="alert-log-table"><tr>{''.join([f'<th>{h}</th>' for h in T['alerts_table_head']])}</tr>"""
    for row in T["alerts_log"]:
        color = T["alerts_colors"][row[1]]
        table += f"""<tr>
            <td>{row[0]}</td>
            <td style='color:{color};font-weight:900;'>{row[1]}</td>
            <td>{row[2]}</td>
            <td>{row[3]}</td>
            <td><button class="solution-btn" style="padding:.3em 1em;font-size:1em;">{row[4]}</button></td>
        </tr>"""
    table += "</table>"
    st.markdown(table, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][5]:  # Smart Solutions
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['solutions_header']}</div>""", unsafe_allow_html=True)
    idx = st.session_state["solution_idx"]
    sol = T["solutions"][idx]
    # Solution card
    steps_html = "".join([f"<li>{s}</li>" for s in sol["steps"]])
    st.markdown(f"""
    <div class="peak-card">
        <div style="font-size:2em;">{sol["icon"]}</div>
        <b style="font-size:1.3em">{sol["title"]}</b>
        <div style="margin:0.8em 0 0.5em 0;">{sol["desc"]}</div>
        <ul style="margin-bottom:0.7em;">{steps_html}</ul>
        <div style="display:flex;gap:0.9em;flex-wrap:wrap;">
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{('Priority' if lang=='en' else 'الأولوية')}: {sol['priority']}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{('Effectiveness' if lang=='en' else 'الفعالية')}: {sol['effectiveness']}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{('Time' if lang=='en' else 'المدة')}: {sol['time']}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{('Cost' if lang=='en' else 'التكلفة')}: {sol['cost']}</span>
            <span style="background:#185a9d12;padding:0.3em 1em;border-radius:6px;">{('Savings' if lang=='en' else 'التوفير')}: {sol['savings']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Next solution button
    if st.button(T["solution_btn"]):
        st.session_state["solution_idx"] = (idx + 1) % len(T["solutions"])

elif section == T["side_sections"][6]:  # KPI Wall
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['kpiwall_header']}</div>""", unsafe_allow_html=True)
    kpis = T["kpi_wall"]
    vals = [96, 272, 62, 1] if lang == "en" else [٩٦, ٢٧٢, ٦٢, ١]
    goals = [98, 250, 70, 0]
    st.markdown("<div style='display:flex;gap:1.3em;flex-wrap:wrap;'>", unsafe_allow_html=True)
    for i, (name, icon, color) in enumerate(kpis):
        st.markdown(f"""<div class="kpi-card" style="background:{color}c0;">
            <span style="font-size:2.1em;">{icon}</span><br>
            <b>{name}</b><br>
            <span style="font-size:2.3em;font-weight:900">{vals[i]}</span>
            <div style="font-size:.95em;color:#222;">{('Goal' if lang=='en' else 'الهدف')}: {goals[i]}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][7]:  # Plant Heatmap
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['heatmap_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["heatmap_intro"]), unsafe_allow_html=True)
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 8, 8)
    z = np.random.uniform(25, 70, (8, 10))
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='YlOrRd', colorbar=dict(title=('Temp °C' if lang=='en' else 'حرارة'))))
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

elif section == T["side_sections"][8]:  # Root Cause Explorer
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['root_cause_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["root_cause_intro"]), unsafe_allow_html=True)
    # Demo cause-effect chart (static)
    st.markdown("""
    <div style="margin-top:1em;display:flex;justify-content:center;">
    <svg width="340" height="180" viewBox="0 0 340 180">
      <rect x="20" y="70" width="80" height="38" rx="12" fill="#43cea2" opacity="0.89"/>
      <rect x="140" y="30" width="90" height="38" rx="12" fill="#ffb347" opacity="0.91"/>
      <rect x="140" y="110" width="90" height="38" rx="12" fill="#fa709a" opacity="0.91"/>
      <rect x="260" y="70" width="60" height="38" rx="12" fill="#8fd3f4" opacity="0.91"/>
      <text x="60" y="93" font-size="1.2em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="185" y="53" font-size="1.1em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="185" y="133" font-size="1.1em" fill="#fff" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <text x="290" y="93" font-size="1.1em" fill="#185a9d" font-family="Cairo,Montserrat" text-anchor="middle">{}</text>
      <line x1="100" y1="89" x2="140" y2="49" stroke="#43cea2" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="100" y1="89" x2="140" y2="129" stroke="#43cea2" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="230" y1="49" x2="260" y2="89" stroke="#ffb347" stroke-width="3" marker-end="url(#arrow)"/>
      <line x1="230" y1="129" x2="260" y2="89" stroke="#fa709a" stroke-width="3" marker-end="url(#arrow)"/>
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#185a9d" />
        </marker>
      </defs>
    </svg>
    </div>
    """.format(
        "Root: Methane leak" if lang=="en" else "الجذر: تسرب ميثان",
        "Compressor 2 Fault" if lang=="en" else "عطل الضاغط ٢",
        "Pump Overload" if lang=="en" else "تحميل زائد على المضخة",
        "Incident: Shutdown" if lang=="en" else "حادث: إيقاف"
    ), unsafe_allow_html=True)

elif section == T["side_sections"][9]:  # AI Copilot Chat
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['ai_chat_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["ai_chat_intro"]), unsafe_allow_html=True)
    # Example chat Q&A
    for user, ai in T["chat_examples"]:
        st.markdown(f"""<div style="display:flex;flex-direction:{'row-reverse' if rtl else 'row'};align-items:flex-start;">
            <div class="chat-avatar">{'👤' if not rtl else '👩‍🔧'}</div>
            <div class="chat-bubble-user">{user}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="display:flex;flex-direction:{'row-reverse' if rtl else 'row'};align-items:flex-start;">
            <div class="chat-avatar">{'🤖'}</div>
            <div class="chat-bubble-ai">{ai}</div>
        </div>""", unsafe_allow_html=True)
    st.text_input(("Ask AI a question..." if lang=="en" else "اكتب سؤالاً للذكاء الصناعي..."), key="ai_input")

elif section == T["side_sections"][10]:  # Live Plant 3D
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['live3d_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["live3d_intro"]), unsafe_allow_html=True)
    st.components.v1.iframe(
        "https://sketchfab.com/models/17e44c5d2828496bb7b132f6e1f13c3e/embed",
        height=480, scrolling=True
    )
    st.markdown(
        rtl_wrap(
            '<sup>3D model courtesy of <a href="https://sketchfab.com" target="_blank">Sketchfab</a></sup>' if lang=="en"
            else '<sup>النموذج ثلاثي الأبعاد مقدم من <a href="https://sketchfab.com" target="_blank">Sketchfab</a></sup>'
        ),
        unsafe_allow_html=True
    )
    st.image(
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80",
        caption=rtl_wrap("Sample Plant 3D Visual" if lang=="en" else "مشهد ثلاثي الأبعاد لمصنع صناعي")
    )

elif section == T["side_sections"][11]:  # Incident Timeline
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['timeline_header']}</div>""", unsafe_allow_html=True)
    for t, icon, desc in T["timeline_steps"]:
        st.markdown(
            f"""<div class="timeline-step"><span class="timeline-icon">{icon}</span>
            <b>{t}</b><br>
            <span style="font-size:1.07em">{desc}</span></div>""",
            unsafe_allow_html=True
        )

elif section == T["side_sections"][12]:  # Energy Optimization
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['energy_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["energy_intro"]), unsafe_allow_html=True)
    # Area energy chart
    energy_sect = ["Compressor", "Pump", "Lighting", "Other"] if lang=="en" else ["ضاغط", "مضخة", "إضاءة", "أخرى"]
    vals = [51, 28, 9, 12]
    fig = px.bar(x=energy_sect, y=vals, color=energy_sect, color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    # Recommendations
    for txt, icon in T["energy_recos"]:
        st.markdown(f"""<div class="peak-card" style="background:#e0eafc;margin-bottom:0.6em;">
            <span class="timeline-icon">{icon}</span> {txt}
        </div>""", unsafe_allow_html=True)

elif section == T["side_sections"][13]:  # Future Insights
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['future_header']}</div>""", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;gap:1.3em;flex-wrap:wrap;'>", unsafe_allow_html=True)
    for title, desc, icon in T["future_cards"]:
        st.markdown(f"""<div class="peak-card" style="min-width:220px;max-width:330px;">
            <span style="font-size:2.1em;">{icon}</span><br>
            <b>{title}</b><br>
            <span style="font-size:1.09em">{desc}</span>
        </div>""", unsafe_allow_html=True)
    # Future trend chart
    x = [datetime.now() + timedelta(days=i) for i in range(7)]
    y = [1.2,1.5,2.0,2.8,3.6,3.9,4.8]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#fa709a", width=3), name="Methane Risk"))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), title=("Methane Risk Forecast" if lang=="en" else "توقع مخاطر الميثان"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif section == T["side_sections"][14]:  # About
    st.markdown(f"""<div class="{ 'gradient-ar' if rtl else 'gradient-header' }">{T['about_header']}</div>""", unsafe_allow_html=True)
    st.markdown(rtl_wrap(T["mission_statement"]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<b>{T['features']}</b>"), unsafe_allow_html=True)
    st.markdown("<ul>"+"".join([f"<li>{f}</li>" for f in T["feature_list"]])+"</ul>", unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<b>{T['howto']}</b>"), unsafe_allow_html=True)
    st.markdown("<ul>"+"".join([f"<li>{f}</li>" for f in T["howto_extend"]])+"</ul>", unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<b>{T['developer']}</b>"), unsafe_allow_html=True)
    for name, mail, phone in T["developers"]:
        st.markdown(f"{T['name']}: {name}<br>Email: {mail}<br>Phone: {phone}<br>", unsafe_allow_html=True)
    st.markdown(rtl_wrap(f"<i>{T['demo_note']}</i>"), unsafe_allow_html=True)
