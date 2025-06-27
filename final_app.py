import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# ------------- IMAGE & ICONS -------------
AI_ICON_URL = "https://img.icons8.com/color/96/artificial-intelligence.png"
RA_AVATAR = "https://avatars.githubusercontent.com/u/22827311?v=4"
AA_AVATAR = "https://avatars.githubusercontent.com/u/125095317?v=4"

# ------------- TRANSLATIONS -------------
translations = {
    "en": {
        "app_title": "Smart Neural Digital Twin",
        "dashboard": "Dashboard",
        "predictive": "Predictive Analysis",
        "solutions": "Smart Solutions",
        "alerts": "Smart Alerts",
        "cost": "Cost & Savings",
        "achievements": "Achievements",
        "performance": "Performance",
        "comparison": "Comparison",
        "explorer": "Data Explorer",
        "about": "About",
        "select_lang": "Select Language",
        "kpi_temp": "Temperature",
        "kpi_pressure": "Pressure",
        "kpi_methane": "Methane",
        "kpi_vibration": "Vibration",
        "kpi_h2s": "H2S",
        "risk_assess": "Risk Assessment",
        "no_risk": "No risks detected.",
        "methane_risk": "⚠️ Methane spike detected!",
        "weekly_summary": "This Week's Summary",
        "future_forecast": "Future Forecast",
        "select_metric": "Select metric to forecast",
        "predicted_events": "Predicted Risk Events",
        "model_info": "About the Model",
        "arima_desc": "Forecasts use ARIMA models to predict future values from plant data.",
        "solution_reco": "Recommended Solution",
        "solution_impact": "Estimated Impact",
        "solution_history": "History of Similar Solutions",
        "effectiveness": "AI Effectiveness",
        "live_alerts": "Live Alerts",
        "alert_level": "Severity",
        "alert_time": "Time",
        "alert_location": "Location",
        "alert_type": "Type",
        "alert_status": "Status",
        "alert_summary": "Alert Summary",
        "resolved_open": "Resolved vs. Open Alerts",
        "filter_by": "Filter By",
        "savings": "Savings",
        "savings_month": "Monthly Savings",
        "savings_year": "Yearly Savings",
        "savings_counter": "You've saved",
        "savings_breakdown": "Savings Breakdown",
        "interventions": "Interventions",
        "economic_impact": "Economic Impact",
        "milestones": "Milestones",
        "progress": "Progress",
        "achieve_congrats": "Congratulations!",
        "current_streak": "Current Safe Streak",
        "longest_streak": "Longest Streak",
        "records": "Records",
        "performance_compare": "Performance Comparison",
        "delta_table": "Change Table",
        "improvement": "Improvement",
        "best_metric": "Best Metric",
        "needs_attention": "Needs Attention",
        "compare_by": "Compare By",
        "by_metric": "By Metric",
        "by_period": "By Period",
        "by_plant": "By Plant/Unit",
        "top_improver": "Top Improver",
        "biggest_opportunity": "Biggest Opportunity",
        "explore_data": "Explore Data",
        "select_var": "Select Variable",
        "download": "Download CSV",
        "about_story": """Our journey began with a simple question: How can we detect gas leaks before disaster strikes? We tried everything, even innovated with drones—and it worked. But we asked ourselves: Why wait for the problem at all?

Our dream was a smart digital twin that predicts danger before it happens—not impossible, but difficult. We made the difficult easy, connecting AI with plant data in a single platform that monitors, learns, and prevents disasters before they start.

Today, our platform is the first line of defense, changing the rules of industrial safety. This is the future.""",
        "about_vision": "Disasters don't wait… and neither do we. Our vision is a safer, smarter industrial world.",
        "about_features": [
            {"icon": "🤖", "title": "AI-powered Analytics", "desc": "Deep learning for anomaly detection & forecasting."},
            {"icon": "💡", "title": "Smart Solutions", "desc": "Actionable, AI-driven recommendations."},
            {"icon": "📈", "title": "Live Monitoring", "desc": "Real-time dashboards and alerts."},
            {"icon": "🌐", "title": "Multi-language", "desc": "Arabic & English support out of the box."},
            {"icon": "🎨", "title": "Beautiful UI", "desc": "Modern, intuitive interface."},
        ],
        "about_milestones": [
            {"icon": "🚀", "title": "MVP Launched", "date": "2024-06-01"},
            {"icon": "🏆", "title": "First Award Won", "date": "2024-11-15"},
            {"icon": "🛡️", "title": "100 Days Incident-Free", "date": "2025-03-10"},
            {"icon": "🤝", "title": "First Partnership", "date": "2025-05-01"},
        ],
        "about_team": [
            {"avatar": RA_AVATAR, "name": "Rakan Almarri", "role": "AI Lead", "email": "rrakanmarri1@gmail.com", "color": "#21e6c1"},
            {"avatar": AA_AVATAR, "name": "Ahmed Alotaibi", "role": "Data Engineer", "email": "Ahmadalotaibi2526@gmail.com", "color": "#278ea5"},
        ],
        "about_contact": "Want a demo or partnership? Reach out:",
        "about_contact_btn": "Contact Us",
    },
    "ar": {
        "app_title": "التوأم الرقمي الذكي العصبي",
        "dashboard": "لوحة القيادة",
        "predictive": "تحليل تنبؤي",
        "solutions": "الحلول الذكية",
        "alerts": "التنبيهات الذكية",
        "cost": "التكلفة والمدخرات",
        "achievements": "الإنجازات",
        "performance": "الأداء",
        "comparison": "المقارنة",
        "explorer": "مستكشف البيانات",
        "about": "عن المنصة",
        "select_lang": "اختر اللغة",
        "kpi_temp": "درجة الحرارة",
        "kpi_pressure": "الضغط",
        "kpi_methane": "الميثان",
        "kpi_vibration": "الاهتزاز",
        "kpi_h2s": "غاز H2S",
        "risk_assess": "تقييم المخاطر",
        "no_risk": "لا توجد مخاطر حالياً.",
        "methane_risk": "⚠️ تم رصد ارتفاع في الميثان!",
        "weekly_summary": "ملخص هذا الأسبوع",
        "future_forecast": "توقعات مستقبلية",
        "select_metric": "اختر المتغير للتنبؤ",
        "predicted_events": "الأحداث المتوقعة الخطرة",
        "model_info": "عن النموذج",
        "arima_desc": "يتم استخدام نماذج ARIMA للتنبؤ بقيم المستقبل بناءً على بيانات المصنع.",
        "solution_reco": "الحل المقترح",
        "solution_impact": "الأثر المتوقع",
        "solution_history": "سجل الحلول المشابهة",
        "effectiveness": "فعالية الذكاء الاصطناعي",
        "live_alerts": "التنبيهات الحية",
        "alert_level": "شدة",
        "alert_time": "الوقت",
        "alert_location": "الموقع",
        "alert_type": "النوع",
        "alert_status": "الحالة",
        "alert_summary": "ملخص التنبيهات",
        "resolved_open": "المنتهية مقابل المفتوحة",
        "filter_by": "تصفية حسب",
        "savings": "المدخرات",
        "savings_month": "المدخرات الشهرية",
        "savings_year": "المدخرات السنوية",
        "savings_counter": "لقد وفرت",
        "savings_breakdown": "تفصيل المدخرات",
        "interventions": "التدخلات",
        "economic_impact": "الأثر الاقتصادي",
        "milestones": "الإنجازات",
        "progress": "التقدم",
        "achieve_congrats": "تهانينا!",
        "current_streak": "سلسلة الأمان الحالية",
        "longest_streak": "أطول سلسلة",
        "records": "الأرقام القياسية",
        "performance_compare": "مقارنة الأداء",
        "delta_table": "جدول التغير",
        "improvement": "التحسن",
        "best_metric": "أفضل مؤشر",
        "needs_attention": "يحتاج متابعة",
        "compare_by": "المقارنة حسب",
        "by_metric": "حسب المؤشر",
        "by_period": "حسب الفترة",
        "by_plant": "حسب الوحدة",
        "top_improver": "الأكثر تحسنًا",
        "biggest_opportunity": "أكبر فرصة",
        "explore_data": "استكشاف البيانات",
        "select_var": "اختر المتغير",
        "download": "تحميل CSV",
        "about_story": """بدأت رحلتنا من سؤال بسيط كيف نكشف تسرب الغاز قبل أن يتحول إلى كارثة ؟ جربنا كل الحلول، وابتكرنا حتى استخدمنا الدرون بنجاح. لكن وقفنا وسألنا ليه ننتظر أصلاً؟ حلمنا كان بناء توأم رقمي ذكي يتوقع الخطر قبل حدوثه - مو مستحيل، لكن كان صعب إحنا أخذنا الصعب وخليناه سهل، وربطنا الذكاء الاصطناعي مع بيانات المصنع في منصة واحدة، تراقب وتتعلم وتمنع الكوارث قبل أن تبدأ. اليوم، منصتنا هي خط الدفاع الأول، تغير قواعد الأمان الصناعي من أساسها. هذا هو المستقبل.""",
        "about_vision": "الكوارث لا تنتظر... ونحن أيضاً لا ننتظر. رؤيتنا: عالم صناعي أكثر أمانًا وذكاء.",
        "about_features": [
            {"icon": "🤖", "title": "تحليلات مدعومة بالذكاء الاصطناعي", "desc": "تعلم عميق لاكتشاف الشذوذ والتنبؤ."},
            {"icon": "💡", "title": "حلول ذكية", "desc": "توصيات قابلة للتنفيذ مدفوعة بالذكاء الاصطناعي."},
            {"icon": "📈", "title": "مراقبة حية", "desc": "لوحات بيانات وتنبيهات في الوقت الحقيقي."},
            {"icon": "🌐", "title": "دعم متعدد اللغات", "desc": "العربية والإنجليزية افتراضيًا."},
            {"icon": "🎨", "title": "واجهة جميلة", "desc": "تصميم عصري وسهل الاستخدام."},
        ],
        "about_milestones": [
            {"icon": "🚀", "title": "إطلاق MVP", "date": "2024-06-01"},
            {"icon": "🏆", "title": "أول جائزة", "date": "2024-11-15"},
            {"icon": "🛡️", "title": "100 يوم بلا حوادث", "date": "2025-03-10"},
            {"icon": "🤝", "title": "أول شراكة", "date": "2025-05-01"},
        ],
        "about_team": [
            {"avatar": RA_AVATAR, "name": "راكان المرّي", "role": "قائد الذكاء الاصطناعي", "email": "rrakanmarri1@gmail.com", "color": "#21e6c1"},
            {"avatar": AA_AVATAR, "name": "أحمد العتيبي", "role": "مهندس بيانات", "email": "Ahmadalotaibi2526@gmail.com", "color": "#278ea5"},
        ],
        "about_contact": "هل ترغب بعرض توضيحي أو شراكة؟ تواصل معنا:",
        "about_contact_btn": "تواصل معنا",
    }
}

# ------------- LANGUAGE SUPPORT -------------
def _(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang][key] if key in translations[lang] else key

def _list(key):
    lang = st.session_state.get("lang", "en")
    return translations[lang][key]

# ------------- PAGE SETUP & NAVIGATION -------------
st.set_page_config(page_title="Smart Neural Digital Twin", layout="wide", page_icon=AI_ICON_URL)

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# Language selector
col_lang, col_spacer = st.columns([1, 8])
with col_lang:
    st.selectbox(
        label="🌏 "+_("select_lang"),
        options=[("English", "en"), ("العربية", "ar")],
        index=0 if st.session_state["lang"]=="en" else 1,
        key="lang",
        format_func=lambda x: x[0],
        label_visibility="collapsed"
    )

# ------------- HEADER: ICON + TITLE -------------
def header():
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:10px;">
            <img src="{AI_ICON_URL}" width="52" style="border-radius:22px; border:3px solid #21e6c1; background:#fff;"/>
            <span style="font-size:2.2em; font-weight:900; letter-spacing:1.5px; color:#153243;">
                {_("app_title")}
            </span>
        </div>
        """, unsafe_allow_html=True
    )

# ------------- SIDEBAR NAVIGATION -------------
PAGES = [
    ("dashboard", "📊"),
    ("predictive", "🔮"),
    ("solutions", "💡"),
    ("alerts", "🚨"),
    ("cost", "💸"),
    ("achievements", "🏆"),
    ("performance", "📈"),
    ("comparison", "⚖️"),
    ("explorer", "🗂️"),
    ("about", "ℹ️"),
]
with st.sidebar:
    st.image(AI_ICON_URL, width=68)
    st.markdown(f"<h2 style='margin-bottom:0'>{_('app_title')}</h2>", unsafe_allow_html=True)
    page = st.radio(
        label="Navigation",
        options=[p[0] for p in PAGES],
        format_func=lambda x: f"{dict(PAGES)[x]} {_(x)}"
    )

# ----------------------------------------------
# ------------- PAGE LOGIC ---------------------
# ----------------------------------------------

# ---------- 1. DASHBOARD ----------
if page == "dashboard":
    header()
    st.subheader(f"📊 {_(page)}")
    # Fake KPI data
    kpi_vals = {
        "kpi_temp": np.random.normal(80, 2),
        "kpi_pressure": np.random.normal(7.5, 0.3),
        "kpi_methane": np.random.normal(0.7, 0.05),
        "kpi_vibration": np.random.normal(2, 0.5),
        "kpi_h2s": np.random.normal(0.05, 0.01),
    }
    cols = st.columns(5)
    for i, k in enumerate(["kpi_temp", "kpi_pressure", "kpi_methane", "kpi_vibration", "kpi_h2s"]):
        delta = np.random.uniform(-2, 2)
        cols[i].metric(_(k), f"{kpi_vals[k]:.2f}", f"{delta:+.2f}")
    style_metric_cards()
    # Status
    if kpi_vals["kpi_methane"] > 0.8:
        st.warning(_( "methane_risk" ))
    else:
        st.success(_( "no_risk" ))
    # KPI trends
    st.markdown(f"### {_( 'weekly_summary' )}")
    kpi_chart_data = {
        k: np.cumsum(np.random.normal(0, 0.3, 30)) + v
        for k, v in kpi_vals.items()
    }
    chart_df = pd.DataFrame(kpi_chart_data)
    chart_df.index = [ (datetime.now() - timedelta(days=(29-i))).strftime("%b %d") for i in range(30) ]
    fig = go.Figure()
    for k, col in zip(chart_df.columns, ["#278ea5", "#21e6c1", "#fbb13c", "#a3cef1", "#e84545"]):
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[k], mode="lines+markers", name=_(k), line=dict(color=col)))
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=25,b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    # Risk assessment
    st.markdown(f"### {_( 'risk_assess' )}")
    st.info(_( "no_risk" ) if kpi_vals["kpi_methane"] < 0.8 else _( "methane_risk" ))

# ---------- 2. PREDICTIVE ANALYSIS ----------
elif page == "predictive":
    header()
    st.subheader(f"🔮 {_(page)}")
    metric = st.selectbox(_( "select_metric" ), [
        _( "kpi_temp" ), _( "kpi_pressure" ), _( "kpi_methane" ), _( "kpi_vibration" ), _( "kpi_h2s" )
    ])
    # Fake forecast
    days = np.arange(0, 14)
    base = 80 if "حرارة" in metric or "Temp" in metric else 0.7
    trend = np.sin(days / 2) + np.random.normal(0, 0.2, len(days))
    forecast = base + trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[(datetime.now() + timedelta(days=int(d))).strftime("%b %d") for d in days],
        y=forecast,
        mode="lines+markers", name=metric,
        line=dict(color="#278ea5", width=3)
    ))
    fig.add_vrect(x0=10, x1=13, fillcolor="#e84545", opacity=0.2, line_width=0)
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=25,b=0))
    st.markdown(f"#### {_( 'future_forecast' )} ({metric})")
    st.plotly_chart(fig, use_container_width=True)
    # Upcoming predicted events
    st.markdown(f"#### {_( 'predicted_events' )}")
    event_rows = []
    for i in [10, 12]:
        event_rows.append({
            "Time": (datetime.now() + timedelta(days=i)).strftime("%b %d"),
            "Metric": metric,
            "Severity": "High" if i==10 else "Moderate",
            "Recommendation": _( "solution_reco" ) + f": {_( 'solution_impact' )}: {np.random.randint(1,5)}h saved"
        })
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True)
    # Model info
    with st.expander(_( "model_info" )):
        st.info(_( "arima_desc" ))

# ---------- 3. SMART SOLUTIONS ----------
elif page == "solutions":
    header()
    st.subheader(f"💡 {_(page)}")
    # Example solution cards
    for i, (risk, act, impact, hist) in enumerate([
        ("Methane spike", "Vent Tank 3", "+12h uptime", "2 similar incidents prevented"),
        ("Pressure anomaly", "Check Valve 2", "+8h uptime", "No downtime in 6 months"),
        ("Temp drift", "Inspect Sensor 5", "+5% stability", "Reduced false alerts"),
    ]):
        cols = st.columns([1,4])
        with cols[0]: st.markdown(f"<div style='font-size:2.5em'>{'⚠️' if i==0 else '🔧'}</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                f"""
                <div style="background:linear-gradient(90deg,#21e6c1 20%,#a3cef1 100%);padding:18px 22px;border-radius:18px;box-shadow:0 3px 18px #278ea533;">
                    <b>{_( 'solution_reco' )}: {risk}</b><br>
                    <span style="color:#278ea5;font-weight:500;">→ {act}</span><br>
                    <span style="font-size:1.05em;">{_( 'solution_impact' )}: <b>{impact}</b></span><br>
                    <span style="font-size:0.95em;color:#153243;">{_( 'solution_history' )}: {hist}</span>
                </div>
                """, unsafe_allow_html=True
            )
    # AI effectiveness
    st.markdown(f"#### {_( 'effectiveness' )}")
    st.progress(0.92, text="92%")

# ---------- 4. SMART ALERTS ----------
elif page == "alerts":
    header()
    st.subheader(f"🚨 {_(page)}")
    # Fake alerts table
    alert_df = pd.DataFrame([
        {"Time": (datetime.now() - timedelta(hours=i*2)).strftime("%b %d %H:%M"), "Severity": np.random.choice(["High","Medium","Low"], p=[0.4,0.4,0.2]), "Location": f"Tank {np.random.randint(1,5)}", "Type": np.random.choice(["Methane","Pressure","Temperature"]), "Status": np.random.choice(["Open","Resolved"], p=[0.6,0.4])}
        for i in range(12)
    ])
    st.markdown(f"#### {_( 'live_alerts' )}")
    st.dataframe(alert_df, use_container_width=True)
    # Alert summary
    with st.expander(_( "alert_summary" )):
        st.metric(_( "alert_level" ), alert_df["Severity"].value_counts().idxmax())
        st.metric(_( "alert_status" ), alert_df["Status"].value_counts().idxmax())
        # Resolved vs open
        open_count = (alert_df["Status"]=="Open").sum()
        resolved_count = (alert_df["Status"]=="Resolved").sum()
        st.markdown(f"{_( 'resolved_open' )}: <b style='color:#21e6c1'>{resolved_count}</b> / <b style='color:#e84545'>{open_count}</b>", unsafe_allow_html=True)
        # Pie
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=alert_df["Severity"], values=alert_df["Severity"].value_counts(), hole=0.45))
        fig.update_layout(margin=dict(l=30,r=30,t=30,b=30), height=220)
        st.plotly_chart(fig, use_container_width=True)

# ---------- 5. COST & SAVINGS ----------
elif page == "cost":
    header()
    st.subheader(f"💸 {_(page)}")
    # Savings cards
    col1, col2, col3, _ = st.columns([2,2,2,2])
    total_saved = np.random.randint(70000, 150000)
    col1.metric(_( "savings_month" ), f"SAR {total_saved // 12:,}")
    col2.metric(_( "savings_year" ), f"SAR {total_saved:,}")
    col3.metric(_( "savings" ), f"SAR {total_saved // 3:,}")
    style_metric_cards()
    # Counter
    st.markdown(
        f"<div style='font-size:1.5em;font-weight:700;color:#21e6c1;background:rgba(33,230,193,0.12);padding:12px 24px;border-radius:18px;display:inline-block;'>{_( 'savings_counter' )} SAR {total_saved:,}!</div>",
        unsafe_allow_html=True
    )
    # Savings breakdown chart
    breakdown = pd.Series({
        _( "interventions" ): total_saved * 0.45,
        _( "economic_impact" ): total_saved * 0.35,
        _( "savings" ): total_saved * 0.20,
    })
    fig = go.Figure(go.Pie(labels=breakdown.index, values=breakdown.values, hole=0.55))
    fig.update_layout(margin=dict(l=30,r=30,t=30,b=30), height=230)
    st.markdown(f"#### {_( 'savings_breakdown' )}")
    st.plotly_chart(fig, use_container_width=True)
    # Table
    st.markdown(f"#### {_( 'interventions' )}")
    st.table(pd.DataFrame({
        _( "interventions" ): ["Methane fix", "Valve repair", "Sensor upgrade"],
        _( "economic_impact" ): [f"SAR {int(total_saved*0.15):,}", f"SAR {int(total_saved*0.12):,}", f"SAR {int(total_saved*0.18):,}"]
    }))

# ---------- 6. ACHIEVEMENTS ----------
elif page == "achievements":
    header()
    st.subheader(f"🏆 {_(page)}")
    # Milestones timeline
    ms = _list( "about_milestones" )
    timeline = ""
    for m in ms:
        timeline += f"<div style='margin-bottom:12px'><span style='font-size:2em'>{m['icon']}</span> <b>{m['title']}</b> <span style='color:#aaa;font-size:0.95em'>({m['date']})</span></div>"
    st.markdown(f"""
    <div style="background:linear-gradient(100deg,#21e6c1 25%,#278ea5 90%);padding:22px 30px;border-radius:22px;box-shadow:0 5px 32px #21e6c144;">
        <b style="font-size:1.35em">{_( 'milestones' )}</b>
        <div style="margin-top:18px">{timeline}</div>
    </div>
    """, unsafe_allow_html=True)
    # Progress bars/records
    st.markdown(f"#### {_( 'progress' )}")
    st.markdown(f"<b>{_( 'current_streak' )}:</b> 109 days", unsafe_allow_html=True)
    st.progress(109/180, text="60% to next milestone (6 months)")
    st.markdown(f"<b>{_( 'longest_streak' )}:</b> 132 days", unsafe_allow_html=True)
    # Achievements cards
    col1, col2, col3 = st.columns(3)
    col1.success(f"🥇 {_( 'achieve_congrats' )}!\n\n{_( 'records' )}: Zero incidents in 100+ days.")
    col2.info("📉 Biggest cost reduction: SAR 23,000 in one month!")
    col3.warning("🔥 5 high-severity events prevented this year.")

# ---------- 7. PERFORMANCE ----------
elif page == "performance":
    header()
    st.subheader(f"📈 {_(page)}")
    # Compare current vs previous KPIs
    kpis = ["kpi_temp", "kpi_pressure", "kpi_methane", "kpi_vibration", "kpi_h2s"]
    now_vals = np.random.normal([80,7.5,0.7,2,0.05], [2,0.3,0.05,0.5,0.01])
    prev_vals = now_vals + np.random.uniform(-2,2,len(now_vals))
    cols = st.columns(5)
    for i, k in enumerate(kpis):
        delta = now_vals[i] - prev_vals[i]
        cols[i].metric(_(k), f"{now_vals[i]:.2f}", f"{delta:+.2f}")
    style_metric_cards()
    # Trend chart
    df = pd.DataFrame(
        np.c_[now_vals, prev_vals],
        columns=["Current", "Previous"],
        index=[_(k) for k in kpis]
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Current"], name="Current", marker_color="#21e6c1"))
    fig.add_trace(go.Bar(x=df.index, y=df["Previous"], name="Previous", marker_color="#278ea5"))
    fig.update_layout(barmode="group", height=300)
    st.plotly_chart(fig, use_container_width=True)
    # Change table
    delta_df = pd.DataFrame({
        "Metric": [_(k) for k in kpis],
        _( "delta_table" ): [f"{now_vals[i]-prev_vals[i]:+.2f}" for i in range(len(kpis))],
        _( "improvement" ): ["✅" if now_vals[i] > prev_vals[i] else "⚠️" for i in range(len(kpis))]
    })
    st.dataframe(delta_df, use_container_width=True, hide_index=True)
    # Best/needs attention
    best = np.argmax(now_vals - prev_vals)
    worst = np.argmin(now_vals - prev_vals)
    st.success(f"{_( 'best_metric' )}: {_(kpis[best])}")
    st.error(f"{_( 'needs_attention' )}: {_(kpis[worst])}")

# ---------- 8. COMPARISON ----------
elif page == "comparison":
    header()
    st.subheader(f"⚖️ {_(page)}")
    # Compare by
    compare_type = st.radio(_( "compare_by" ), [_( "by_metric" ), _( "by_period" ), _( "by_plant" )], horizontal=True)
    # Fake comparison data
    items = ["Metric A", "Metric B", "Metric C"] if compare_type==_( "by_metric" ) else ["Jan", "Feb", "Mar"] if compare_type==_( "by_period" ) else ["Plant 1", "Plant 2", "Plant 3"]
    vals1 = np.random.normal(100, 10, 3)
    vals2 = vals1 + np.random.normal(-12, 12, 3)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=items, y=vals1, name="This Period", marker_color="#21e6c1"))
    fig.add_trace(go.Bar(x=items, y=vals2, name="Previous", marker_color="#278ea5"))
    fig.update_layout(barmode="group", height=320)
    st.plotly_chart(fig, use_container_width=True)
    # Table
    comp_df = pd.DataFrame({
        "Item": items,
        "This Period": vals1,
        "Previous": vals2,
        "Δ": vals1 - vals2
    })
    st.dataframe(comp_df, use_container_width=True)
    # Top/bottom
    st.success(f"{_( 'top_improver' )}: {items[np.argmax(vals1-vals2)]}")
    st.warning(f"{_( 'biggest_opportunity' )}: {items[np.argmin(vals1-vals2)]}")

# ---------- 9. DATA EXPLORER ----------
elif page == "explorer":
    header()
    st.subheader(f"🗂️ {_(page)}")
    # Fake plant data
    variables = ["Temperature","Pressure","Methane","Vibration","H2S"]
    df = pd.DataFrame({
        "Date": pd.date_range(datetime.now()-timedelta(days=59), periods=60),
        **{v: np.cumsum(np.random.normal(0, 0.5, 60)) + 80 + i*0.5 for i,v in enumerate(variables)}
    })
    var = st.selectbox(_( "select_var" ), variables)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[var], mode="lines+markers", name=var, line=dict(color="#278ea5", width=3)))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["Date", var]], use_container_width=True)
    st.download_button(_( "download" ), df.to_csv(index=False), file_name="data.csv")

# ---------- 10. ABOUT (WOW SECTION) ----------
elif page == "about":
    header()
    # Hero card/story
    st.markdown(
        f'''
        <div style="background: linear-gradient(120deg, #21e6c1 30%, #278ea5 100%); border-radius:24px; padding:28px 36px; margin-bottom:24px; box-shadow:0 6px 32px rgba(0,0,0,0.10); color:#153243; font-size:1.23em;">
            <img src="{AI_ICON_URL}" width="56" style="vertical-align:middle; margin-right:18px;"/>
            <span style="font-weight:bold; font-size:1.45em;">{_( 'app_title' )}</span>
            <hr style="border:1px solid #21e6c1;">
            <div style="margin-top:12px; line-height:1.9;">
                {_( 'about_story' )}
            </div>
        </div>
        ''', unsafe_allow_html=True
    )
    # Vision
    st.markdown(
        f"""
        <div style="background:linear-gradient(100deg,#fff3e0 10%,#21e6c1 90%); border-radius:18px; padding:18px 28px; margin-bottom:18px; box-shadow:0 4px 14px #278ea522;">
            <span style="font-size:1.3em; font-style:italic;">“{_( 'about_vision' )}”</span>
        </div>
        """, unsafe_allow_html=True
    )
    # Features
    st.markdown(f"<b style='font-size:1.17em'>{'🌟 '+_('about_features')[0]['title'].split()[0]} {_( 'about' )}:</b>", unsafe_allow_html=True)
    feat_cols = st.columns(len(_list( "about_features" )))
    for col, f in zip(feat_cols, _list( "about_features" )):
        with col:
            st.markdown(
                f"""
                <div style='background:linear-gradient(105deg,#21e6c1 55%,#fff 100%);border-radius:14px;padding:16px 7px;margin-bottom:7px;box-shadow:0 2px 10px #278ea522; text-align:center;'>
                    <span style='font-size:2em'>{f['icon']}</span><br>
                    <b>{f['title']}</b><br>
                    <span style='font-size:0.96em;color:#278ea5'>{f['desc']}</span>
                </div>
                """, unsafe_allow_html=True
            )
    # Milestones
    ms = _list( "about_milestones" )
    st.markdown(f"<b style='font-size:1.13em'>{_('milestones')}</b>", unsafe_allow_html=True)
    st.markdown(
        "<ul style='list-style:none;padding-left:0;'>"
        + "".join([f"<li style='margin-bottom:8px'><span style='font-size:1.3em'>{m['icon']}</span> <b>{m['title']}</b> <span style='color:#aaa;font-size:0.97em'>({m['date']})</span></li>" for m in ms])
        + "</ul>",
        unsafe_allow_html=True
    )
    # Team
    st.markdown(f"<b style='font-size:1.13em'>👥 {_( 'about_team' )[0]['role' if st.session_state['lang']=='en' else 'role']}</b>", unsafe_allow_html=True)
    team_cols = st.columns(len(_list( "about_team" )))
    for col, t in zip(team_cols, _list( "about_team" )):
        with col:
            st.markdown(
                f"""
                <div style='background:linear-gradient(115deg,{t['color']} 60%,#fff 100%);border-radius:16px;padding:10px 0 14px 0;margin-bottom:7px;box-shadow:0 2px 10px #278ea522; text-align:center;'>
                    <img src="{t['avatar']}" width="54" style="border-radius:50%;border:3px solid #fff;margin-bottom:8px;"/><br>
                    <b>{t['name']}</b><br>
                    <span style='font-size:0.96em;color:#153243'>{t['role']}</span><br>
                    <a href="mailto:{t['email']}" style="font-size:0.93em;color:#278ea5;text-decoration:none;">{t['email']}</a>
                </div>
                """, unsafe_allow_html=True
            )
    # Contact
    st.markdown(
        f"""
        <div style="background:linear-gradient(100deg,#278ea5 10%,#fff3e0 90%); border-radius:18px; padding:18px 28px; box-shadow:0 4px 16px #21e6c133; text-align:center;">
            <span style="font-size:1.18em;">{_( 'about_contact' )}</span><br>
            <a href="mailto:rrakanmarri1@gmail.com,Ahmadalotaibi2526@gmail.com">
                <button style="background:#21e6c1;color:#153243;font-weight:bold;font-size:1.08em;border-radius:9px; border:none; padding:11px 36px; margin-top:10px; box-shadow:0 1px 8px #278ea522; cursor:pointer;">
                    {_( 'about_contact_btn' )}
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True
    )
    rain(
        emoji="💡",
        font_size=28,
        falling_speed=4,
        animation_length="infinite"
)
