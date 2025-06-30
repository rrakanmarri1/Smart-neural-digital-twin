import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# --- TRANSLATIONS ---
translations = {
    "en": {
        "Digital Twin": "Digital Twin",
        "Advanced Dashboard": "Advanced Dashboard",
        "Predictive Analytics": "Predictive Analytics",
        "Scenario Playback": "Scenario Playback",
        "Alerts & Fault Log": "Alerts & Fault Log",
        "Smart Solutions": "Smart Solutions",
        "KPI Wall": "KPI Wall",
        "Plant Heatmap": "Plant Heatmap",
        "Root Cause Explorer": "Root Cause Explorer",
        "AI Copilot Chat": "AI Copilot Chat",
        "Live Plant 3D": "Live Plant 3D",
        "Incident Timeline": "Incident Timeline",
        "Energy Optimization": "Energy Optimization",
        "Future Insights": "Future Insights",
        "About": "About",
        "Contact Info": "Contact Info",
        "English": "English",
        "Arabic": "Arabic",
        "Restart": "Restart",
        "Next": "Next",
        "Generate Solution": "Generate Solution",
        "Mission Statement": "Mission Statement",
        "Features": "Features",
        "How to extend": "How to extend",
        "Developer": "Developer",
        "Name": "Name",
        "Emails": "Emails",
        "Phones": "Phones",
        "Demo use only: Not for live plant operation": "Demo use only: Not for live plant operation",
    },
    "ar": {
        "Digital Twin": "التوأم الرقمي",
        "Advanced Dashboard": "لوحة القيادة المتقدمة",
        "Predictive Analytics": "التحليلات التنبؤية",
        "Scenario Playback": "تشغيل السيناريو",
        "Alerts & Fault Log": "التنبيهات وسجل الأعطال",
        "Smart Solutions": "الحلول الذكية",
        "KPI Wall": "جدار المؤشرات",
        "Plant Heatmap": "خريطة حرارة المصنع",
        "Root Cause Explorer": "مستكشف السبب الجذري",
        "AI Copilot Chat": "محادثة الذكاء الصناعي",
        "Live Plant 3D": "مصنع ثلاثي الأبعاد",
        "Incident Timeline": "جدول الحوادث",
        "Energy Optimization": "تحسين الطاقة",
        "Future Insights": "رؤى مستقبلية",
        "About": "حول النظام",
        "Contact Info": "معلومات التواصل",
        "English": "الإنجليزية",
        "Arabic": "العربية",
        "Restart": "إعادة تشغيل",
        "Next": "التالي",
        "Generate Solution": "توليد الحل",
        "Mission Statement": "بيان المهمة",
        "Features": "الميزات",
        "How to extend": "كيفية التوسيع",
        "Developer": "المطور",
        "Name": "الاسم",
        "Emails": "البريد الإلكتروني",
        "Phones": "أرقام الهواتف",
        "Demo use only: Not for live plant operation": "للعرض فقط: غير مخصص للتشغيل الفعلي",
    }
}
def get_label(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang].get(key, key)
def rtl(text):
    if st.session_state.get("lang", "en") == "ar":
        return f"<div dir='rtl' style='text-align:right'>{text}</div>"
    return text

# --- SESSION STATE INIT ---
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "scenario_step" not in st.session_state:
    st.session_state["scenario_step"] = 0
if "show_solutions" not in st.session_state:
    st.session_state["show_solutions"] = False

# --- SIDEBAR: Logo, Nav, Language ---
def render_logo():
    st.markdown("""
<div style="display:flex; align-items:center; gap:1em; margin-bottom:1em;">
<svg width="70" height="70" viewBox="0 0 80 80" fill="none">
    <circle cx="40" cy="40" r="34" fill="#E9F5FF" stroke="#0080FF" stroke-width="3"/>
    <path d="M40 12 C54 24, 66 38, 40 68" stroke="#0077B6" stroke-width="3" fill="none"/>
    <path d="M40 12 C26 24, 14 38, 40 68" stroke="#00B686" stroke-width="3" fill="none"/>
    <circle cx="40" cy="12" r="4" fill="#00B686" stroke="#0077B6" stroke-width="2"/>
    <circle cx="66" cy="38" r="3" fill="#0080FF"/>
    <circle cx="14" cy="38" r="3" fill="#00B686"/>
    <circle cx="40" cy="68" r="5" fill="#0077B6" stroke="#0080FF" stroke-width="2"/>
    <circle cx="54" cy="24" r="2" fill="#0080FF"/>
    <circle cx="26" cy="24" r="2" fill="#00B686"/>
    <circle cx="40" cy="40" r="3" fill="white" stroke="#0077B6" stroke-width="2"/>
</svg>
<div style="display:flex; flex-direction:column;">
  <span style="font-size:2.1em; font-weight:800; color:#0077B6; letter-spacing:1px;">Smart Neural Digital Twin</span>
  <span style="font-size:1.2em; font-weight:500; color:#00B686; letter-spacing:.5px;">التوأم الرقمي العصبي الذكي</span>
</div>
</div>
    """, unsafe_allow_html=True)

with st.sidebar:
    render_logo()
    SECTIONS = [
        "Digital Twin",
        "Advanced Dashboard",
        "Predictive Analytics",
        "Scenario Playback",
        "Alerts & Fault Log",
        "Smart Solutions",
        "KPI Wall",
        "Plant Heatmap",
        "Root Cause Explorer",
        "AI Copilot Chat",
        "Live Plant 3D",
        "Incident Timeline",
        "Energy Optimization",
        "Future Insights",
        "About"
    ]
    SECTIONS_AR = [
        "التوأم الرقمي",
        "لوحة القيادة المتقدمة",
        "التحليلات التنبؤية",
        "تشغيل السيناريو",
        "التنبيهات وسجل الأعطال",
        "الحلول الذكية",
        "جدار المؤشرات",
        "خريطة حرارة المصنع",
        "مستكشف السبب الجذري",
        "محادثة الذكاء الصناعي",
        "مصنع ثلاثي الأبعاد",
        "جدول الحوادث",
        "تحسين الطاقة",
        "رؤى مستقبلية",
        "حول النظام"
    ]
    sec = st.radio(
        " ",
        SECTIONS if st.session_state["lang"] == "en" else SECTIONS_AR,
        index=0
    )
    st.markdown("---")
    st.write(get_label("Contact Info"))
    st.write(
        "- Rakan Almarri: rakan.almarri.2@aramco.com (0532559664)\n"
        "- Abdulrahman Alzahrani: abdulrahman.alzhrani.2@aramco.com (0549202574)")
    lang_choice = st.radio(
        "",
        (get_label("English"), get_label("Arabic")),
        horizontal=True,
        index=0 if st.session_state["lang"] == "en" else 1
    )
    st.session_state["lang"] = "en" if lang_choice == translations["en"]["English"] else "ar"

page = SECTIONS[SECTIONS_AR.index(sec)] if st.session_state["lang"] == "ar" else sec

# --- DEMO DATA ---
np.random.seed(0)
demo_df = pd.DataFrame({
    "time": pd.date_range(datetime.now() - timedelta(hours=24), periods=48, freq="30min"),
    "Temperature": np.random.normal(55, 6, 48),
    "Pressure": np.random.normal(7, 1.2, 48),
    "Methane": np.clip(np.random.normal(1.4, 0.7, 48), 0, 6)
})

# --- PAGE LOGIC WITH VISUALS ---
if page == "Digital Twin":
    st.title(get_label("Digital Twin"))
    st.markdown(rtl("## Interactive Plant Schematic"))
    fig = go.Figure()
    area_x = [2, 5, 8]
    area_y = [4, 7, 4]
    titles = ["Pump", "Compressor", "Reactor"]
    icons = ["🛢️", "🌀", "⚗️"]
    for x, y, t, ic in zip(area_x, area_y, titles, icons):
        fig.add_shape(type="circle", x0=x-0.7, y0=y-0.7, x1=x+0.7, y1=y+0.7,
                      fillcolor="#e9f5ff", line=dict(color="#0077B6", width=3))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="text", text=[f"<span style='font-size:2em'>{ic}</span>"], textfont=dict(size=25)))
        fig.add_trace(go.Scatter(x=[x], y=[y+0.85], mode="text", text=[t], textfont=dict(size=15)))
    fig.add_trace(go.Scatter(x=area_x, y=area_y, mode="markers", marker=dict(size=36, color="#43cea2")))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=390, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.info(rtl("Click on a sensor in the real app to drill down. Visual overlays and propagation logic supported."))

elif page == "Advanced Dashboard":
    st.title(get_label("Advanced Dashboard"))
    st.markdown(rtl("## KPIs and Sensor Trends"))
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Avg Temp", f"{demo_df['Temperature'].mean():.1f}°C", f"{(demo_df['Temperature'].mean()-55):+.1f}")
    kpi_cols[1].metric("Avg Pressure", f"{demo_df['Pressure'].mean():.2f} bar", f"{(demo_df['Pressure'].mean()-7):+.2f}")
    kpi_cols[2].metric("Avg Methane", f"{demo_df['Methane'].mean():.2f} ppm", f"{(demo_df['Methane'].mean()-1.4):+.2f}")
    st.line_chart(demo_df.set_index("time")[["Temperature", "Pressure", "Methane"]], use_container_width=True)
    st.success(rtl("Health widgets, live graphs, and custom indicators all possible."))

elif page == "Predictive Analytics":
    st.title(get_label("Predictive Analytics"))
    st.markdown(rtl("## AI Forecast – Temperature"))
    last_temp = demo_df["Temperature"].iloc[-1]
    forecast = [last_temp + 0.3*i + np.random.normal(0,0.2) for i in range(10)]
    fut_idx = [demo_df["time"].iloc[-1] + timedelta(hours=i+1) for i in range(10)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=demo_df["time"], y=demo_df["Temperature"], mode="lines+markers", name="History"))
    fig.add_trace(go.Scatter(x=fut_idx, y=forecast, mode="lines+markers", name="Forecast", line=dict(dash="dash")))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.info(rtl("AI-driven anomaly prediction, what-if simulation and more."))

elif page == "Scenario Playback":
    st.title(get_label("Scenario Playback"))
    steps = [
        ("All sensors OK. Monitoring...", "#43cea2"),
        ("Methane rising! AI detects risk.", "#fee140"),
        ("Manual detection late! Leak escalates.", "#fa709a"),
        ("AI triggers rapid auto-shutdown. Loss minimized.", "#43cea2"),
    ]
    st.progress((st.session_state["scenario_step"]+1)/len(steps))
    step_msg, color = steps[st.session_state["scenario_step"]]
    st.markdown(f"<div style='background:{color}22;padding:1.2em;margin:.7em 0 .7em 0;border-radius:12px;'>{rtl(step_msg)}</div>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if st.button(get_label("Restart")):
            st.session_state["scenario_step"] = 0
    with colB:
        if st.button(get_label("Next")):
            st.session_state["scenario_step"] = min(st.session_state["scenario_step"]+1, len(steps)-1)
    st.info(rtl("Full scenario engine: add detection, escalation, and response logic here."))

elif page == "Alerts & Fault Log":
    st.title(get_label("Alerts & Fault Log"))
    st.markdown(rtl("## Live Alerts"))
    st.error(rtl("No faults detected in the last 24h."))
    st.warning(rtl("Warning: Methane trending up at Reactor."))
    st.info(rtl("Historical log, drilldown, and alert breakdowns supported."))

elif page == "Smart Solutions":
    st.title(get_label("Smart Solutions"))
    st.markdown(rtl("AI-generated actionable solutions for detected problems, optimization, and innovation."))
    if not st.session_state["show_solutions"]:
        if st.button(get_label("Generate Solution")):
            st.session_state["show_solutions"] = True
        st.info(rtl("Press 'Generate Solution' to view Smart Solutions."))
    else:
        # 8 demo solution cards
        solutions = [
            {
                "title_en": "Automated Leak Detection & Shutdown",
                "title_ar": "كشف التسرب الآلي والإيقاف",
                "details_en": "Integrate advanced methane sensors with auto-shutdown logic to instantly contain leaks.",
                "details_ar": "ربط حساسات الميثان مع منطق إيقاف فوري لاحتواء التسرب بشكل آلي.",
                "priority": "High",
                "effectiveness": "High",
                "time": "2 days",
                "cost": "$3,000",
                "savings": "$20,000/year"
            },
            {
                "title_en": "Energy Usage Optimization",
                "title_ar": "تحسين استهلاك الطاقة",
                "details_en": "Deploy AI-driven analytics to reduce compressor overuse and cut energy bills.",
                "details_ar": "استخدم التحليلات الذكية لتقليل تشغيل الضواغط وخفض استهلاك الطاقة.",
                "priority": "Medium",
                "effectiveness": "High",
                "time": "1 week",
                "cost": "$7,000",
                "savings": "$32,000/year"
            },
            {
                "title_en": "Predictive Maintenance for Pumps",
                "title_ar": "صيانة استباقية للمضخات",
                "details_en": "Monitor vibration and temperature for early pump failure prediction.",
                "details_ar": "راقب الاهتزاز والحرارة للتنبؤ بأعطال المضخات مبكراً.",
                "priority": "High",
                "effectiveness": "Medium",
                "time": "4 days",
                "cost": "$6,000",
                "savings": "$16,000/year"
            },
            {
                "title_en": "Digital Twin Operator Training",
                "title_ar": "تدريب المشغلين باستخدام التوأم الرقمي",
                "details_en": "Use the twin for scenario-based operator drills to improve safety culture.",
                "details_ar": "استخدم التوأم الرقمي لسيناريوهات تدريبية لتعزيز السلامة.",
                "priority": "Medium",
                "effectiveness": "Medium",
                "time": "2 weeks",
                "cost": "$12,000",
                "savings": "$8,000/year"
            },
            {
                "title_en": "H₂S Real-time Monitoring",
                "title_ar": "مراقبة H₂S اللحظية",
                "details_en": "Deploy real-time H₂S alarms in high-risk areas.",
                "details_ar": "ركب حساسات إنذار كبريتيد الهيدروجين في المناطق الحرجة.",
                "priority": "High",
                "effectiveness": "High",
                "time": "3 days",
                "cost": "$2,500",
                "savings": "Lives, incident cost"
            },
            {
                "title_en": "Water Consumption Analytics",
                "title_ar": "تحليل استهلاك المياه",
                "details_en": "AI identifies abnormal water usage and suggests process adjustments.",
                "details_ar": "الذكاء الاصطناعي يحدد استهلاك المياه غير الطبيعي ويقترح الحلول.",
                "priority": "Low",
                "effectiveness": "Medium",
                "time": "1 week",
                "cost": "$1,500",
                "savings": "$5,000/year"
            },
            {
                "title_en": "Asset Tracking Dashboard",
                "title_ar": "لوحة تتبع الأصول",
                "details_en": "Track all assets and maintenance state in one live dashboard.",
                "details_ar": "تتبع حالة الأصول والصيانة في لوحة واحدة حية.",
                "priority": "Medium",
                "effectiveness": "High",
                "time": "5 days",
                "cost": "$4,000",
                "savings": "$10,000/year"
            },
            {
                "title_en": "Remote Expert Collaboration",
                "title_ar": "التعاون مع الخبراء عن بعد",
                "details_en": "Enable video/AR remote support for field incidents.",
                "details_ar": "دعم عن بعد بالفيديو/الواقع المعزز للحوادث الميدانية.",
                "priority": "High",
                "effectiveness": "High",
                "time": "3 days",
                "cost": "$3,500",
                "savings": "$12,000/year"
            }
        ]
        lang = st.session_state["lang"]
        for i, sol in enumerate(solutions):
            c = ["#e3f6fc", "#e2ffe3", "#fffadd", "#ffe3e3"][i%4]
            st.markdown(
                f"""
<div style="background:{c};border-radius:16px;padding:1.6em 1.5em;box-shadow:0 4px 18px #ccc2; margin-bottom:1.2em;">
  <div style="font-size:1.7em;font-weight:900;color:#0077B6;margin-bottom:0.2em;">{sol['title_en'] if lang=="en" else sol['title_ar']}</div>
  <div style="font-size:1.13em;margin-bottom:1em;">{sol['details_en'] if lang=="en" else sol['details_ar']}</div>
  <div style="display:flex;gap:0.9em;flex-wrap:wrap;margin-bottom:0.7em;">
    <span style="background:#fff;padding:0.35em 0.9em;border-radius:7px;font-weight:700;">Priority: {sol['priority']}</span>
    <span style="background:#fff;padding:0.35em 0.9em;border-radius:7px;font-weight:700;">Effectiveness: {sol['effectiveness']}</span>
    <span style="background:#fff;padding:0.35em 0.9em;border-radius:7px;font-weight:700;">Time: {sol['time']}</span>
    <span style="background:#fff;padding:0.35em 0.9em;border-radius:7px;font-weight:700;">Cost: {sol['cost']}</span>
    <span style="background:#fff;padding:0.35em 0.9em;border-radius:7px;font-weight:700;">Savings: {sol['savings']}</span>
  </div>
</div>
""", unsafe_allow_html=True)
    st.info(rtl("More: Add live AI, solution feedback, export, and integration with plant data."))

elif page == "KPI Wall":
    st.title("📈 "+get_label("KPI Wall"))
    st.markdown(rtl("Live KPIs, health indexes, and dynamic plant performance."))
    cols = st.columns(4)
    kpis = [
        ("Overall Efficiency", 96, 98, "#43cea2"),
        ("Energy Used (kWh)", 272, 250, "#fee140"),
        ("Water Saved (m³)", 62, 70, "#43cea2"),
        ("Incidents This Year", 1, 0, "#fa709a"),
    ]
    for i, (name, val, goal, color) in enumerate(kpis):
        delta = val-goal if i!=3 else goal-val
        icon = "✅" if (i==0 and val >= goal) or (i==2 and val >= goal) else "⚠️" if i!=3 else "🛑"
        st.markdown(
            f"""<div style="background:{color}22;border-radius:12px;padding:1.2em;text-align:center;font-size:1.3em;margin-bottom:.5em;">
            <b>{get_label(name) if name in translations['en'] else name}</b><br>
            <span style="font-size:2.3em;font-weight:900">{icon} {val}</span>
            <div style="color:gray;font-size:.9em;">Goal: {goal}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown(rtl("Track all major KPIs at a glance. Customize this wall for your operation."))

elif page == "Plant Heatmap":
    st.title("🌡️ " + get_label("Plant Heatmap"))
    st.markdown(rtl("Visualize real-time temperature/pressure distribution across the plant."))
    x = np.linspace(0, 10, 12)
    y = np.linspace(0, 8, 12)
    z = np.random.uniform(28, 55, (12, 12))
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='YlGnBu'))
    fig.update_layout(height=420, margin=dict(l=12, r=12, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(rtl("High temperature/pressure zones are highlighted for quick action."))

elif page == "Root Cause Explorer":
    st.title("🔎 " + get_label("Root Cause Explorer"))
    st.markdown(rtl("Trace faults to their origin with interactive cause-effect mapping."))
    st.info(rtl("Click nodes to expand and understand the propagation of issues (demo coming soon)."))

elif page == "AI Copilot Chat":
    st.title("🤝 " + get_label("AI Copilot Chat"))
    st.markdown(rtl("Chat with your AI assistant for instant plant troubleshooting and learning."))
    st.info(rtl("Type a question about your plant, process, or code. (Full chatbot integration can be added.)"))

elif page == "Live Plant 3D":
    st.title("🪄 " + get_label("Live Plant 3D"))
    st.markdown(rtl("A 3D schematic of your plant. (For demo, view the enhanced 2D schematic in Digital Twin.)"))
    st.info(rtl("Upgrade to true 3D when deploying with specialized visualization tools."))

elif page == "Incident Timeline":
    st.title("🕒 " + get_label("Incident Timeline"))
    st.markdown(rtl("Review all major incidents and actions chronologically."))
    st.markdown(rtl("No major incidents in the last 30 days. (Simulate more in Scenario Playback.)"))

elif page == "Energy Optimization":
    st.title("⚡ " + get_label("Energy Optimization"))
    st.markdown(rtl("Monitor and optimize plant energy usage for sustainability and cost savings."))
    st.markdown(rtl("AI recommends reducing compressor runtime during off-peak hours for immediate savings."))

elif page == "Future Insights":
    st.title("🔮 " + get_label("Future Insights"))
    st.markdown(rtl("Predict emerging risks, improvement opportunities, and innovation pathways."))
    st.markdown(rtl("Upgrade to AI-powered forecasting for proactive plant management."))

elif page == "About":
    st.title("ℹ️ " + get_label("About"))
    st.markdown(rtl(f"""
### {get_label("Mission Statement")}
This digital twin demonstrates how advanced monitoring, predictive analytics, and real-time process intelligence can improve safety, efficiency, and sustainability in industrial operations.

---
### {get_label("Features")}
- 🌐 Digital Twin: Interactive plant schematic, overlays, and live simulation.
- 📊 Advanced Dashboard: KPIs, health widgets, and live sensor trends.
- 🤖 Smart Solutions: AI-generated solutions and code for plant challenges.
- 🧠 AI Copilot: Chat, guidance, and knowledge for your team.
- 🔥 Plant Heatmap: Visualize hotspots and pressure zones.
- 🔎 Root Cause Explorer: Drilldown issue tracing.
- 🏭 Live Plant 3D: Future-ready visual immersion.
- ⚡ Energy Optimization: Save cost and emissions.
- 🕒 Incident Timeline: All incidents at a glance.
- 🔮 Future Insights: Stay ahead with prediction.

---
### {get_label("How to extend")}
- Plug in real plant schematic and sensor data sources.
- Integrate with historian/data lake for live operation.
- Expand scenario logic, add dashboards, connect to control logic.
- Contact the developer for enhancements.

---
### {get_label("Developer")}
- **{get_label("Name")}:** Rakan Almarri (0532559664)  
  **Email:** rakan.almarri.2@aramco.com
- **{get_label("Name")}:** Abdulrahman Alzahrani (0549202574)  
  **Email:** abdulrahman.alzhrani.2@aramco.com
---
*{get_label('Demo use only: Not for live plant operation')}*
    """))
