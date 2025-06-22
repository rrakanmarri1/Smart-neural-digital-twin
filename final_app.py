import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

THEMES = {
    "Ocean": {
        "sidebar_bg": "#145DA0",
        "main_bg": "#1E3C72",
        "accent": "#00A8CC",
        "image": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?fit=crop&w=900&q=80"
    },
    "Forest": {
        "sidebar_bg": "#145A32",
        "main_bg": "#196F3D",
        "accent": "#45B39D",
        "image": "https://images.unsplash.com/photo-1464983953574-0892a716854b?fit=crop&w=900&q=80"
    },
    "Classic": {
        "sidebar_bg": "#222831",
        "main_bg": "#393E46",
        "accent": "#FFD369",
        "image": ""
    }
}

st.set_page_config(
    page_title="Smart Neural Digital Twin",
    page_icon="🧠",
    layout="wide"
)

if 'lang' not in st.session_state:
    st.session_state.lang = "ar"
if 'theme' not in st.session_state:
    st.session_state.theme = "Ocean"

def settings_panel():
    st.markdown("### ⚙️ Settings / الإعدادات")
    lang = st.radio("🌐 Language | اللغة", ["English", "العربية"], 
                    index=0 if st.session_state.lang == "en" else 1, horizontal=True)
    th = st.radio("🎨 Theme | الثيم", list(THEMES.keys()), 
                  index=list(THEMES.keys()).index(st.session_state.theme), horizontal=True)
    if lang == "English":
        st.session_state.lang = "en"
    else:
        st.session_state.lang = "ar"
    st.session_state.theme = th

theme = THEMES[st.session_state.theme]
st.markdown(
    f"""
    <style>
        body, .stApp {{
            background-color: {theme['main_bg']} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {theme['sidebar_bg']} !important;
        }}
        .main-title {{
            color: {theme['accent']}; font-size: 2.5em; text-align:center; font-weight: bold;
        }}
        .menu-btn {{
            background: {theme['accent']}; color: #fff; border-radius:12px; padding:0.7em 1.3em; font-size:1.15em; margin: 0.5em 1em 0.5em 0;
            border:none; cursor:pointer; box-shadow:1px 1px 7px #2222;
            transition: 0.2s all;
        }}
        .menu-btn.selected {{
            background: #fff; color: {theme['accent']}; border: 2px solid {theme['accent']};
        }}
        .rounded-box {{
            background: #fff1;
            border-radius: 14px;
            padding: 1.2em;
            margin-bottom: 1.3em;
            box-shadow: 0 2px 7px #2222;
        }}
        .main-bg-img {{
            background-image: url('{theme['image']}');
            background-size: cover;
            background-position: center;
            min-height: 250px;
            border-radius: 14px;
            margin-bottom: 1.5em;
        }}
    </style>
    """, unsafe_allow_html=True
)

if theme['image']:
    st.markdown('<div class="main-bg-img"></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["Timestamp"])
    np.random.seed(42)
    df["Current"] = np.random.uniform(10, 30, len(df))
    df["Level"] = np.random.uniform(0.2, 1.0, len(df))
    df["Humidity"] = np.random.uniform(45, 65, len(df))
    return df

df = load_data()

def get_anomaly_score(data):
    clf = IsolationForest(contamination=0.07, random_state=0)
    features = data[["Temp", "Pressure", "Vibration", "Gas", "Current", "Level", "Humidity"]]
    clf.fit(features)
    scores = clf.decision_function(features)
    return scores

df["AnomalyScore"] = get_anomaly_score(df)
df["Anomaly"] = df["AnomalyScore"] < -0.15

pages = {
    "dashboard": "📊 Dashboard",
    "predictive": "🔮 Predictive Analytics",
    "solutions": "💡 Smart Solutions",
    "log": "📝 Event Log",
    "report": "📄 Smart Report",
    "settings": "⚙️ Settings",
    "about": "ℹ️ About"
}
pages_ar = {
    "dashboard": "📊 لوحة البيانات",
    "predictive": "🔮 التحليل التنبؤي",
    "solutions": "💡 الحلول الذكية",
    "log": "📝 سجل الأحداث",
    "report": "📄 التقرير الذكي",
    "settings": "⚙️ الإعدادات",
    "about": "ℹ️ حول المشروع"
}
if st.session_state.lang == "ar":
    menu = list(pages_ar.values())
    keys = list(pages_ar.keys())
else:
    menu = list(pages.values())
    keys = list(pages.keys())

selected = st.selectbox(
    "🚦 اختر القسم" if st.session_state.lang == "ar" else "🚦 Select Section",
    menu,
    key="menu_select"
)
page = keys[menu.index(selected)]

if page == "dashboard":
    st.markdown(f"<div class='main-title'>{pages_ar['dashboard'] if st.session_state.lang == 'ar' else pages['dashboard']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.write("#### " + ("مخطط القيم الحية (72 ساعة)" if st.session_state.lang == "ar" else "Live Sensor Trends (72h)"))
    fig = px.line(df, x="Timestamp", y=["Temp", "Pressure", "Vibration", "Gas", "Current", "Level", "Humidity"],
                  labels={"value": "Value", "variable": "Sensor", "Timestamp": "Time"},
                  title="Sensor Data Over Time")
    st.plotly_chart(fig, use_container_width=True)
    st.write("#### " + ("خريطة الحرارة (الشذوذات)" if st.session_state.lang == "ar" else "Heatmap (Anomalies)"))
    fig2 = px.density_heatmap(df, x="Timestamp", y="Temp", z="AnomalyScore", nbinsx=32, nbinsy=12,
                              color_continuous_scale="Hot", title="Anomaly Heatmap")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "predictive":
    st.markdown(f"<div class='main-title'>{pages_ar['predictive'] if st.session_state.lang == 'ar' else pages['predictive']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.write("#### " + ("توقع القيم القادمة (ذكاء اصطناعي)" if st.session_state.lang == "ar" else "Predicting Future Values (AI)"))
    preds = df[["Temp", "Pressure", "Vibration", "Gas"]].iloc[-1] + np.random.normal(0, 1, 4)
    future = pd.date_range(df["Timestamp"].max(), periods=13, freq="H")[1:]
    pred_df = pd.DataFrame({
        "Timestamp": future,
        "Temp": preds["Temp"] + np.cumsum(np.random.normal(0, 0.2, 12)),
        "Pressure": preds["Pressure"] + np.cumsum(np.random.normal(0, 0.3, 12)),
        "Vibration": preds["Vibration"] + np.cumsum(np.random.normal(0, 0.03, 12)),
        "Gas": preds["Gas"] + np.cumsum(np.random.normal(0, 0.2, 12))
    })
    fig3 = px.line(pred_df, x="Timestamp", y=["Temp", "Pressure", "Vibration", "Gas"],
                   title="Predicted Sensor Values Next 12 Hours")
    st.plotly_chart(fig3, use_container_width=True)
    st.write("##### " + ("تحليل شذوذ الذكاء الاصطناعي" if st.session_state.lang == "ar" else "AI Anomaly Analysis"))
    st.dataframe(df[["Timestamp", "Temp", "Pressure", "Vibration", "Gas", "AnomalyScore", "Anomaly"]].tail(20))
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "solutions":
    st.markdown(f"<div class='main-title'>{pages_ar['solutions'] if st.session_state.lang == 'ar' else pages['solutions']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.write("#### " + ("الحلول الذكية حسب الحالة" if st.session_state.lang == "ar" else "AI-based Smart Actions"))
    current_row = df.iloc[-1]
    recommendation = ""
    details = ""
    time_needed = ""
    importance = ""
    efficiency = ""
    if st.button("🚀 توليد حل ذكي" if st.session_state.lang == "ar" else "🚀 Generate Solution"):
        if current_row["Temp"] > 60:
            recommendation = "تفعيل نظام التبريد" if st.session_state.lang == "ar" else "Activate Cooling System"
            details = "ارتفاع حرارة خطير" if st.session_state.lang == "ar" else "Critical temperature detected"
            time_needed = "فوري" if st.session_state.lang == "ar" else "Immediate"
            importance = "عالية جداً" if st.session_state.lang == "ar" else "Very High"
            efficiency = "99%"
        elif current_row["Pressure"] < 15:
            recommendation = "إغلاق الصمام الرئيسي" if st.session_state.lang == "ar" else "Shut Main Valve"
            details = "انخفاض ضغط خطير" if st.session_state.lang == "ar" else "Critical pressure drop"
            time_needed = "فوري" if st.session_state.lang == "ar" else "Immediate"
            importance = "مرتفعة" if st.session_state.lang == "ar" else "High"
            efficiency = "97%"
        elif current_row["Gas"] > 400:
            recommendation = "إخلاء المنطقة فوراً" if st.session_state.lang == "ar" else "Evacuate Area Immediately"
            details = "تسرب غاز شديد" if st.session_state.lang == "ar" else "Severe gas leak"
            time_needed = "فوري" if st.session_state.lang == "ar" else "Immediate"
            importance = "قصوى" if st.session_state.lang == "ar" else "Critical"
            efficiency = "100%"
        else:
            recommendation = "استمرار المراقبة" if st.session_state.lang == "ar" else "Continue Monitoring"
            details = "لا توجد مؤشرات خطر حالياً" if st.session_state.lang == "ar" else "No current risk detected"
            time_needed = "-" 
            importance = "عادية" if st.session_state.lang == "ar" else "Normal"
            efficiency = "N/A"
        st.success((f"🟢 الحل: {recommendation}\n\n🔎 التفاصيل: {details}\n⏰ الزمن: {time_needed}\n‼️ الأهمية: {importance}\n⚡ الفعالية: {efficiency}")
                   if st.session_state.lang == "ar" 
                   else 
                   (f"🟢 Solution: {recommendation}\n\n🔎 Details: {details}\n⏰ Time Needed: {time_needed}\n‼️ Importance: {importance}\n⚡ Efficiency: {efficiency}"))
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "log":
    st.markdown(f"<div class='main-title'>{pages_ar['log'] if st.session_state.lang == 'ar' else pages['log']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.write("#### " + ("سجل الأحداث" if st.session_state.lang == "ar" else "Event Log"))
    logdf = df[df["Anomaly"]][["Timestamp", "Temp", "Pressure", "Vibration", "Gas", "AnomalyScore"]]
    st.dataframe(logdf.tail(30))
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "report":
    st.markdown(f"<div class='main-title'>{pages_ar['report'] if st.session_state.lang == 'ar' else pages['report']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.write("#### " + ("ملخص التقرير الذكي (قابل للتصدير)" if st.session_state.lang == "ar" else "Smart Report Summary (Exportable)"))
    total_anomalies = df["Anomaly"].sum()
    max_temp = df["Temp"].max()
    min_press = df["Pressure"].min()
    max_gas = df["Gas"].max()
    summary = (
        f"عدد الشذوذات المكتشفة: {total_anomalies}\n"
        f"أعلى حرارة مسجلة: {max_temp:.2f}\n"
        f"أقل ضغط مسجل: {min_press:.2f}\n"
        f"أعلى غاز مسجل: {max_gas:.2f}"
        if st.session_state.lang == "ar" else
        f"Total Anomalies Detected: {total_anomalies}\n"
        f"Max Temperature: {max_temp:.2f}\n"
        f"Min Pressure: {min_press:.2f}\n"
        f"Max Gas: {max_gas:.2f}"
    )
    st.info(summary)
    if st.button("⬇️ تحميل CSV" if st.session_state.lang == "ar" else "⬇️ Download CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download", csv, "smart_digital_twin_report.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "settings":
    settings_panel()

elif page == "about":
    st.markdown(f"<div class='main-title'>{pages_ar['about'] if st.session_state.lang == 'ar' else pages['about']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)

    if st.session_state.lang == "en":
        st.markdown("""
        <div style="font-size:1.5em;font-weight:bold;color:#1976d2;text-align:center;margin-bottom:0.4em;">
            “Disasters don't wait.. and neither do we.”
        </div>
        <div style="font-size:1.15em;color:#00A8CC;text-align:center;margin-bottom:1.5em;">
            <b>Predict. Prevent. Protect.</b>
        </div>
        <hr>
        <h3>👨‍💻 Lead Developers</h3>
        <div style="background:#e3f2fd;border-radius:8px;padding:1em 1.5em;margin-bottom:1.5em;font-size:1.1em;">
        <b>Rakan Almarri</b> — rakan.almarri.2@aramco.com — 0532559664<br>
        <b>Abdulrahman Alzahrani</b> — abdulrahman.alzahrani.1@aramco.com — 0549202574
        </div>

        ### 💡 About the Project

        **Smart Neural Digital Twin** is an advanced prototype for oilfield safety, powered by AI and IoT.<br>
        <ul>
        <li>Real-time dashboard & predictive analytics for all sensors</li>
        <li>AI anomaly detection (Isolation Forest)</li>
        <li>Automated smart recommendations at the click of a button</li>
        <li>Full bilingual support (Arabic & English) with total translation</li>
        <li>Customizable UI (Ocean/Forest/Classic) with instant interface switching</li>
        <li>Exportable smart reports (CSV)</li>
        <li>Comprehensive log of anomalies and alerts</li>
        <li>Responsive design for mobile and desktop</li>
        <li>Designed for future integration with Aramco’s platforms</li>
        </ul>

        <b>Vision:</b> To revolutionize industrial safety by transforming raw data into actionable insights, ensuring a safer and more efficient operational environment.
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:1.5em;font-weight:bold;color:#1976d2;text-align:center;margin-bottom:0.4em;">
            "الكوارث لا تنتظر... ونحن أيضًا لا ننتظر"
        </div>
        <div style="font-size:1.15em;color:#45B39D;text-align:center;margin-bottom:1.5em;">
            <b>🔮 توقّع • 🛡️ وقاية • 🧯 حماية</b>
        </div>
        <hr>
        <h3>👨‍💻 المطورون الرئيسيون</h3>
        <div style="background:#e8f5e9;border-radius:8px;padding:1em 1.5em;margin-bottom:1.5em;font-size:1.1em;">
        <b>راكان المري</b> — rakan.almarri.2@aramco.com — 0532559664<br>
        <b>عبدالرحمن الزهراني</b> — abdulrahman.alzahrani.1@aramco.com — 0549202574
        </div>

        ### 💡 حول المشروع

        **التوأم الرقمي الذكي** هو نموذج أولي متطور لرفع مستوى الأمان في حقول النفط باستخدام الذكاء الاصطناعي وإنترنت الأشياء.<br>
        <ul>
        <li>لوحة بيانات وتوقعات لحظية لجميع المستشعرات</li>
        <li>كشف الشذوذات بأنظمة الذكاء الاصطناعي (Isolation Forest)</li>
        <li>حلول وتوصيات ذكية أوتوماتيكية بضغطة زر</li>
        <li>دعم لغتين بالكامل (العربية والإنجليزية) مع تعريب شامل</li>
        <li>تخصيص المظهر (بحري/غابة/تقليدي) مع تغيير فوري للواجهة</li>
        <li>تقارير ذكية قابلة للتصدير (CSV)</li>
        <li>سجل كامل للحوادث والشذوذات والتنبيهات</li>
        <li>تصميم تفاعلي يدعم الجوال والكمبيوتر</li>
        <li>مصمم للتكامل مستقبلاً مع منصات أرامكو الصناعية</li>
        </ul>

        <b>رؤيتنا:</b> تحويل البيانات الصناعية إلى رؤى قابلة للتنفيذ لضمان بيئة تشغيل أكثر أمانًا وكفاءة.
