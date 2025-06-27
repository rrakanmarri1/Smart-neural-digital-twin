import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain

# -------------------- TRANSLATIONS --------------------
translations = {
    "en": {
        "select_lang": "Select Language",
        "dashboard": "Dashboard",
        "predictive": "Predictive Analytics",
        "solutions": "Smart Solutions",
        "alerts": "Alerts",
        "cost": "Cost & Savings",
        "comparison": "AI vs. Manual",
        "milestones": "Roadmap & Changelog",
        "about": "About",
        "team": "Main Developers",
        "contact_us": "Contact Us",
        "ai_effectiveness": "AI Effectiveness",
        "smart_solutions": "Smart Solutions",
        "generate_solutions": "Generate Solutions",
        "solution_title": "Title",
        "solution_desc": "Description",
        "solution_eff": "Effectiveness",
        "solution_priority": "Priority",
        "solution_time": "Time Estimate",
        "solution_card": "Solution",
        "priority_high": "High",
        "priority_med": "Medium",
        "priority_low": "Low",
        "dashboard_title": "Smart Neural Digital Twin",
        "metrics_uptime": "Uptime (hrs)",
        "metrics_incidents": "Incidents Prevented",
        "metrics_savings": "Total Savings",
        "pred_analytics": "Predictive Analytics",
        "alert_title": "Active Alerts",
        "cost_title": "Cost & Savings",
        "savings_month": "Savings This Month",
        "interventions": "AI Interventions",
        "comparison_title": "AI vs. Manual Operations",
        "comparison_speed": "Detection Speed",
        "comparison_downtime": "Downtime Prevented",
        "comparison_savings": "Cost Savings",
        "comparison_ai": "AI System",
        "comparison_manual": "Manual",
        "simulation_title": "What-If Impact Simulator",
        "simulation_desc": "Adjust the AI sensitivity to see potential impact.",
        "simulation_sensitivity": "AI Sensitivity",
        "simulation_savings": "Projected Savings",
        "milestones_title": "Roadmap & Changelog",
        "milestones_roadmap": "Upcoming Features",
        "milestones_changelog": "Recent Updates",
        "about_title": "About the Project",
        "about_story": "Our Story",
        "about_story_desc": "We are two engineers passionate about leveraging AI to solve real-world plant challenges. This digital twin was built to maximize uptime, safety, and efficiency for critical operations.",
        "about_vision": "Vision",
        "about_vision_desc": "To set a new benchmark for industrial AI—real-time, actionable, and trusted by operators.",
        "about_team": "Main Developers",
        "about_contact": "Contact Us",
        "about_contact_desc": "For demo, partnership, or questions, reach out anytime.",
        "about_rakan": "Rakan Almarri",
        "about_abdulrahman": "Abdulrahman Alzahrani",
        "role_lead": "AI Lead",
        "role_engineer": "Engineer",
        "email": "Email",
        "phone": "Phone",
    },
    "ar": {
        "select_lang": "اختر اللغة",
        "dashboard": "لوحة التحكم",
        "predictive": "التحليلات التنبؤية",
        "solutions": "الحلول الذكية",
        "alerts": "التنبيهات",
        "cost": "التكلفة والمدخرات",
        "comparison": "مقارنة الذكاء الاصطناعي مع اليدوي",
        "milestones": "خارطة الطريق والتحديثات",
        "about": "حول",
        "team": "المطورون الرئيسيون",
        "contact_us": "تواصل معنا",
        "ai_effectiveness": "فعالية الذكاء الاصطناعي",
        "smart_solutions": "الحلول الذكية",
        "generate_solutions": "إنشاء حلول",
        "solution_title": "العنوان",
        "solution_desc": "الوصف",
        "solution_eff": "الفعالية",
        "solution_priority": "الأولوية",
        "solution_time": "الوقت المقدر",
        "solution_card": "الحل",
        "priority_high": "عالية",
        "priority_med": "متوسطة",
        "priority_low": "منخفضة",
        "dashboard_title": "التوأم الرقمي الذكي العصبي",
        "metrics_uptime": "مدة التشغيل (ساعات)",
        "metrics_incidents": "الحوادث الممنوعة",
        "metrics_savings": "إجمالي المدخرات",
        "pred_analytics": "التحليلات التنبؤية",
        "alert_title": "التنبيهات النشطة",
        "cost_title": "التكلفة والمدخرات",
        "savings_month": "المدخرات هذا الشهر",
        "interventions": "تدخلات الذكاء الاصطناعي",
        "comparison_title": "مقارنة الذكاء الاصطناعي مع العمليات اليدوية",
        "comparison_speed": "سرعة الاكتشاف",
        "comparison_downtime": "وقت التوقف الممنوع",
        "comparison_savings": "المدخرات",
        "comparison_ai": "النظام الذكي",
        "comparison_manual": "يدوي",
        "simulation_title": "محاكاة الأثر التقديري",
        "simulation_desc": "قم بتعديل حساسية الذكاء الاصطناعي لرؤية الأثر المحتمل.",
        "simulation_sensitivity": "حساسية الذكاء الاصطناعي",
        "simulation_savings": "المدخرات التقديرية",
        "milestones_title": "خارطة الطريق والتحديثات",
        "milestones_roadmap": "الميزات القادمة",
        "milestones_changelog": "آخر التحديثات",
        "about_title": "حول المشروع",
        "about_story": "قصتنا",
        "about_story_desc": "نحن مهندسان شغوفان بتسخير الذكاء الاصطناعي لحل تحديات المصانع الحقيقية. تم بناء هذا التوأم الرقمي لتعظيم وقت التشغيل والسلامة والكفاءة للعمليات الحرجة.",
        "about_vision": "الرؤية",
        "about_vision_desc": "أن نكون المعيار الجديد للذكاء الاصطناعي الصناعي — في الوقت الفعلي وقابل للتنفيذ وموثوق من قبل المشغلين.",
        "about_team": "المطورون الرئيسيون",
        "about_contact": "تواصل معنا",
        "about_contact_desc": "للحصول على عرض توضيحي أو شراكة أو استفسارات، تواصل معنا في أي وقت.",
        "about_rakan": "راكان المري",
        "about_abdulrahman": "عبدالرحمن الزهراني",
        "role_lead": "قائد الذكاء الاصطناعي",
        "role_engineer": "مهندس",
        "email": "البريد الإلكتروني",
        "phone": "رقم الجوال",
    }
}

def _(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang][key] if key in translations[lang] else key

# -------------------- SIDEBAR --------------------
def side_nav():
    nav_options = [
        ("dashboard", _("dashboard")),
        ("predictive", _("predictive")),
        ("solutions", _("solutions")),
        ("alerts", _("alerts")),
        ("cost", _("cost")),
        ("comparison", _("comparison")),
        ("milestones", _("milestones")),
        ("about", _("about")),
    ]
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3772/3772533.png", width=64)
    st.sidebar.title("Smart Neural Digital Twin")
    nav = st.sidebar.radio(
        "Navigation",
        options=[x[0] for x in nav_options],
        format_func=lambda k: dict(nav_options)[k],
        key="nav_radio"
    )
    st.sidebar.markdown("---")
    st.sidebar.selectbox(
        "🌍 "+_("select_lang"),
        options=[("en", "English"), ("ar", "العربية")],
        format_func=lambda x: x[1],
        key="lang_selectbox"
    )
    st.session_state["lang"] = st.session_state.get("lang_selectbox", ("en", "English"))[0]
    return nav

# -------------------- HEADER --------------------
def header():
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:10px;'>
            <img src="https://cdn-icons-png.flaticon.com/512/3772/3772533.png" width="40"/>
            <span style='font-size:2.2em;font-weight:bold;'>{_('dashboard_title')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- DASHBOARD --------------------
def dashboard():
    header()
    st.markdown(f"## {_('dashboard_title')}")
    st.markdown("##### AI-powered digital twin for real-time monitoring and optimization.")
    col1, col2, col3 = st.columns(3)
    col1.metric(_("metrics_uptime"), "984", "▲ 12")
    col2.metric(_("metrics_incidents"), "27", "▲ 6")
    col3.metric(_("metrics_savings"), "SAR 153,000", "▲ SAR 12,000")
    style_metric_cards()
    st.markdown("---")
    st.markdown(f"### {_('ai_effectiveness')}")
    st.progress(0.93, text="93%")
    rain(
        emoji="💡",
        font_size=32,
        falling_speed=5,
        animation_length="infinite"
    )

# -------------------- PREDICTIVE ANALYTICS --------------------
def predictive():
    header()
    st.markdown(f"## {_('pred_analytics')}")
    st.markdown("##### AI prediction of faults, failures, and maintenance needs.")
    # Fake time series
    df = pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
        "Methane": np.random.normal(150, 12, 30),
        "Pressure": np.random.normal(80, 5, 30),
        "Temp": np.random.normal(68, 2, 30)
    })
    feat = st.selectbox(_("solution_title"), ["Methane", "Pressure", "Temp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[feat], mode="lines+markers", name=feat))
    fig.update_layout(height=350, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["date", feat]].rename(columns={feat: _(feat.lower()) if feat.lower() in translations[st.session_state["lang"]] else feat}), hide_index=True)

# -------------------- SMART SOLUTIONS --------------------
def smart_solutions():
    header()
    st.markdown(f"## {_('smart_solutions')}")
    if st.button(_("generate_solutions")) or "solutions" not in st.session_state:
        # Demo 7 unique solutions
        st.session_state["solutions"] = [
            {
                "title": _("solution_card") + " 1: Methane spike",
                "desc": _("solution_desc") + ": Vent Tank 3 to stabilize pressure.",
                "eff": "96%",
                "priority": _("priority_high"),
                "time": "2h"
            },
            {
                "title": _("solution_card") + " 2: Pressure anomaly",
                "desc": _("solution_desc") + ": Inspect and recalibrate Valve 2.",
                "eff": "89%",
                "priority": _("priority_high"),
                "time": "4h"
            },
            {
                "title": _("solution_card") + " 3: Temperature drift",
                "desc": _("solution_desc") + ": Check Sensor 5 for calibration.",
                "eff": "83%",
                "priority": _("priority_med"),
                "time": "3h"
            },
            {
                "title": _("solution_card") + " 4: Vibration detected",
                "desc": _("solution_desc") + ": Tighten mounts on Compressor 1.",
                "eff": "80%",
                "priority": _("priority_med"),
                "time": "5h"
            },
            {
                "title": _("solution_card") + " 5: Flow irregularity",
                "desc": _("solution_desc") + ": Inspect Pipeline Section B.",
                "eff": "76%",
                "priority": _("priority_low"),
                "time": "8h"
            },
            {
                "title": _("solution_card") + " 6: Power fluctuation",
                "desc": _("solution_desc") + ": Replace Backup Battery A.",
                "eff": "90%",
                "priority": _("priority_high"),
                "time": "1h"
            },
            {
                "title": _("solution_card") + " 7: Moisture ingress",
                "desc": _("solution_desc") + ": Dry and reseal Control Panel 2.",
                "eff": "85%",
                "priority": _("priority_med"),
                "time": "6h"
            }
        ]
    # Show solution cards
    for s in st.session_state["solutions"]:
        with st.container():
            st.markdown(
                f"""
                <div style='background:linear-gradient(90deg,#182848,#566472);border-radius:12px;padding:16px 18px;margin-bottom:13px;box-shadow:0 2px 8px #0002'>
                    <div style='font-size:1.25em;font-weight:bold;margin-bottom:2px'>{s['title']}</div>
                    <div style='margin-bottom:6px'>{s['desc']}</div>
                    <span style='font-size:.98em'><b>{_('solution_eff')}:</b> {s['eff']}</span> |
                    <span style='font-size:.98em'><b>{_('solution_priority')}:</b> {s['priority']}</span> |
                    <span style='font-size:.98em'><b>{_('solution_time')}:</b> {s['time']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown(f"### {_('ai_effectiveness')}")
    st.progress(0.92, text="92%")

# -------------------- ALERTS --------------------
def alerts():
    header()
    st.markdown(f"## {_('alert_title')}")
    # Demo active alerts
    alerts_data = [
        {"type": "Methane", "level": "High", "msg": "Possible leak in Tank 3"},
        {"type": "Pressure", "level": "Medium", "msg": "Pressure anomaly in Section A"},
        {"type": "Temp", "level": "Low", "msg": "Temperature drift detected"}
    ]
    for a in alerts_data:
        st.warning(f"**{a['type']}** ({a['level']}) — {a['msg']}")

# -------------------- COST & SAVINGS --------------------
def cost():
    header()
    st.markdown(f"## {_('cost_title')}")
    # No more TypeError: show metrics safely
    savings = 22000
    interventions = 8
    col1, col2 = st.columns(2)
    col1.metric(_("savings_month"), f"SAR {savings:,}", "+SAR 4,200")
    col2.metric(_("interventions"), f"{interventions}", "+3")
    style_metric_cards()
    st.markdown("---")
    st.markdown("#### Last 6 Months")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    vals = np.random.randint(15000, 30000, 6)
    fig = go.Figure(data=[
        go.Bar(x=months, y=vals, marker_color="#44d"),
    ])
    fig.update_layout(height=320, template="plotly_dark", yaxis_title="SAR")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- AI vs. MANUAL (COMPARISON) --------------------
def comparison():
    header()
    st.markdown(f"## {_('comparison_title')}")
    st.markdown("#### " + _("comparison_title"))
    # Impressive: AI vs Manual metrics
    st.markdown("##### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(_("comparison_speed"), "3 sec", "AI")
    col2.metric(_("comparison_downtime"), "98%", "Prevented")
    col3.metric(_("comparison_savings"), "SAR 153,000", "Year")
    # Visual comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(name=_("comparison_ai"), x=["Detection Speed", "Downtime Prevented", "Cost Savings"], y=[3, 98, 153000], marker_color="#4da"))
    fig.add_trace(go.Bar(name=_("comparison_manual"), x=["Detection Speed", "Downtime Prevented", "Cost Savings"], y=[1800, 65, 23000], marker_color="#aaa"))
    fig.update_layout(barmode='group', height=340, template="plotly_dark", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    # What-If Simulator
    st.markdown(f"### {_('simulation_title')}")
    st.write(_( "simulation_desc" ))
    sensitivity = st.slider(_( "simulation_sensitivity" ), 50, 100, 92)
    projected_savings = int(23000 + (sensitivity-50)*2500)
    st.success(f"{_('simulation_savings')}: SAR {projected_savings:,}")

# -------------------- MILESTONES/ROADMAP & CHANGELOG --------------------
def milestones():
    header()
    st.markdown(f"## {_('milestones_title')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {_('milestones_roadmap')}")
        st.markdown("""
- **Finalize Hackathon Demo:** 2025-07-01
- **Pilot Deployment at Plant XYZ:** 2025-08-01
- **User Feedback Integration:** 2025-08-15
- **Mobile Monitoring App:** 2025-09-01
- **Energy Optimization Module:** 2025-10-01
- **Custom Alerts:** 2025-10-15
- **OPC-UA/SCADA Integration:** 2025-11-01
        """)
    with col2:
        st.markdown(f"#### {_('milestones_changelog')}")
        st.markdown("""
- **2025-06-27:** Added Arabic language support and fixed content bugs.
- **2025-06-26:** Improved Smart Solution generator.
- **2025-06-25:** Fixed cost & savings calculations.
- **2025-06-20:** Added AI vs. Manual Comparison and Impact Simulator.
- **2025-06-18:** Enhanced About page and team info.
        """)

# -------------------- ABOUT --------------------
def about():
    header()
    st.markdown(f"## {_('about_title')}")
    st.markdown(f"### {_('about_story')}")
    st.write(_( "about_story_desc" ))
    st.markdown(f"### {_('about_vision')}")
    st.write(_( "about_vision_desc" ))
    st.markdown(f"### {_('about_team')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#182848,#566472);border-radius:12px;padding:18px 18px 8px;margin-bottom:13px;box-shadow:0 2px 8px #0002'>
            <div style='font-size:1.2em;font-weight:bold'>{_('about_rakan')}</div>
            <div>{_('role_lead')}</div>
            <div><b>{_('email')}:</b> rakan.almarri.2@aramco.com</div>
            <div><b>{_('phone')}:</b> 0532559664</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#182848,#566472);border-radius:12px;padding:18px 18px 8px;margin-bottom:13px;box-shadow:0 2px 8px #0002'>
            <div style='font-size:1.2em;font-weight:bold'>{_('about_abdulrahman')}</div>
            <div>{_('role_lead')}</div>
            <div><b>{_('email')}:</b> abdulrahman.alzahrani.1@aramco.com</div>
            <div><b>{_('phone')}:</b> 0549202574</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"### {_('about_contact')}")
    st.info(_( "about_contact_desc" ))

# -------------------- PAGE ROUTER --------------------
def main():
    st.set_page_config(page_title="Smart Neural Digital Twin", layout="wide", page_icon="💡")
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    nav = side_nav()
    if nav == "dashboard":
        dashboard()
    elif nav == "predictive":
        predictive()
    elif nav == "solutions":
        smart_solutions()
    elif nav == "alerts":
        alerts()
    elif nav == "cost":
        cost()
    elif nav == "comparison":
        comparison()
    elif nav == "milestones":
        milestones()
    elif nav == "about":
        about()

if __name__ == "__main__":
    main()
