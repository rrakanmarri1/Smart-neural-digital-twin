
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# إعداد الصفحة
st.set_page_config(
    page_title="🧠 Smart Neural Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# اختيار اللغة
language = st.sidebar.selectbox("🌐 اختر اللغة | Choose Language", ["العربية", "English"])

# بيانات المحاكاة
data = {
    "temp": round(np.random.normal(36, 2), 2),
    "pressure": round(np.random.normal(95, 5), 2),
    "vibration": round(np.random.normal(0.5, 0.1), 2)
}

# ستايل
st.markdown("""
    <style>
    body {
        background-color: #f7fafd;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1565c0;
        color: white;
    }
    .main-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #0d47a1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-box {
        background-color: white;
        border-left: 6px solid #1976d2;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- اللغة العربية -------------------
if language == "العربية":
    st.markdown("<div class='main-title'>🧠 التوأم الرقمي الذكي - Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    st.markdown("### 🌍 "الكوارث لا تنتظر... ونحن أيضًا لا ننتظر"
#### 🔮 توقّع • 🛡️ وقاية • 🧯 حماية")
    st.markdown("---")

    menu = st.sidebar.radio("🚀 انتقل إلى:", ["الرئيسية", "المستشعرات", "التوصيات", "حول المشروع"])

    if menu == "الرئيسية":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📊 بيانات المستشعرات (محاكاة)")
        df = pd.DataFrame({
            "المستشعر": ["درجة الحرارة (°C)", "الضغط (kPa)", "الاهتزاز (mm/s)"],
            "القيمة": [data["temp"], data["pressure"], data["vibration"]]
        })
        st.table(df.set_index("المستشعر"))
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "التوصيات":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("💡 التوصية الذكية")
        if data["temp"] > 38:
            st.error("🔥 درجة حرارة عالية! فعّل نظام التبريد فورًا.")
        elif data["pressure"] < 85:
            st.warning("⚠️ الضغط منخفض. افحص الصمامات وخط الأنابيب.")
        else:
            st.success("✅ الوضع مستقر وآمن حاليًا.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "حول المشروع":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 🎯 رؤيتنا")
        st.markdown("🔬 **تحويل البيانات الصناعية إلى رؤى قابلة للتنفيذ لضمان بيئة تشغيل أكثر أمانًا وكفاءة.**")
        st.markdown("💡 هذا المشروع يستخدم الذكاء الاصطناعي لرصد وتحليل الحالات الخطرة واتخاذ قرارات ذكية استباقية.")
        st.markdown("📆 تاريخ آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- English Version -------------------
else:
    st.markdown("<div class='main-title'>🧠 Smart Neural Digital Twin</div>", unsafe_allow_html=True)
    st.markdown("### 🌍 "Disasters don't wait... and neither do we."
#### 🔮 Predict • 🛡️ Prevent • 🧯 Protect")
    st.markdown("---")

    menu = st.sidebar.radio("🚀 Navigate to:", ["Home", "Sensors", "Recommendations", "About"])

    if menu == "Home":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📊 Simulated Sensor Data")
        df = pd.DataFrame({
            "Sensor": ["Temperature (°C)", "Pressure (kPa)", "Vibration (mm/s)"],
            "Value": [data["temp"], data["pressure"], data["vibration"]]
        })
        st.table(df.set_index("Sensor"))
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "Recommendations":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("💡 Smart Recommendation")
        if data["temp"] > 38:
            st.error("🔥 High temperature detected! Activate the cooling system.")
        elif data["pressure"] < 85:
            st.warning("⚠️ Low pressure. Check the valves and pipeline.")
        else:
            st.success("✅ All systems are stable and safe.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "About":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Our Vision")
        st.markdown("🔬 **To revolutionize industrial safety by transforming raw data into actionable insights.**")
        st.markdown("💡 This project uses AI to detect and respond to anomalies in real-time.")
        st.markdown("📆 Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.markdown("</div>", unsafe_allow_html=True)
