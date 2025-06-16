import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from fpdf import FPDF
import streamlit_authenticator as stauth
from datetime import timedelta, datetime
import json

# 1. Page config
st.set_page_config(page_title="Smart Neural Digital Twin", page_icon="🧠", layout="wide")

# 2. Authentication
users = {"rakan.almarri.2@aramco.com": {"name": "Rakan Almarri", "password": "password1"},
         "abdulrahman.alzhrani.1@aramco.com": {"name": "Abdulrahman Alzhrani", "password": "password2"}}
credentials = {
    "usernames": {u: {"name": users[u]["name"], "password": stauth.Hasher([users[u]["password"]]).generate()[0]} 
                  for u in users}
}
auth = stauth.Authenticate(credentials, "smart_digital_twin", "abcdef", cookie_expiry_days=1)
name, authentication_status, username = auth.login("Login", "main")
if not authentication_status:
    st.stop()

# 3. Session defaults
if "lang" not in st.session_state: st.session_state.lang = "en"
if "palette" not in st.session_state: st.session_state.palette = "Ocean"
if "last_toast" not in st.session_state: st.session_state.last_toast = None

# 4. Immediate language/theme switch
def rerun(): st.experimental_rerun()

# 5. Palettes & backgrounds
PALETTES = {
    "Ocean": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Forest": "https://images.unsplash.com/photo-1501785888041-af3ef285b470",
    "Sunset": "https://images.unsplash.com/photo-1518837695005-2083093ee35b",
    "Purple": "https://images.unsplash.com/photo-1522661067900-22e2879de181",
    "Slate": "https://images.unsplash.com/photo-1531959871807-30c29ab64749"
}
bg = PALETTES[st.session_state.palette]
st.markdown(f"""
    <style>
    .stApp {{ background: url('{bg}?auto=compress&cs=tinysrgb&w=1260') no-repeat center fixed; background-size: cover; }}
    .btn-gen > button {{ background-color: #FF5722 !important; color: white !important; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# 6. Sidebar controls
lang = st.sidebar.radio("Language", ["English", "العربية"], index=0 if st.session_state.lang=="en" else 1, on_change=rerun, key="lang")
st.session_state.lang = lang
palette = st.sidebar.radio("Palette" if lang=="en" else "لوحة الألوان", list(PALETTES.keys()),
                           index=list(PALETTES.keys()).index(st.session_state.palette), on_change=rerun, key="palette")
st.session_state.palette = palette

# 7. Translations
T = {
    "en": {
        "Dashboard":"Dashboard", "Simulation":"Simulation","Predictive Analysis":"Predictive","Smart Solutions":"Solutions",
        "Settings":"Settings","About":"About","Generate Solution":"Generate Solution","No data":"No data available",
        "Export CSV":"Export CSV","Export PDF":"Export PDF","Sensor Map":"Sensor Locations","Anomaly Sensitivity":"Anomaly Sensitivity"
    },
    "العربية": {
        "Dashboard":"لوحة البيانات","Simulation":"المحاكاة","Predictive Analysis":"التحليل التنبؤي",
        "Smart Solutions":"الحلول الذكية","Settings":"الإعدادات","About":"حول","Generate Solution":"توليد الحل",
        "No data":"لا توجد بيانات","Export CSV":"تصدير CSV","Export PDF":"تصدير PDF","Sensor Map":"مواقع المستشعرات","Anomaly Sensitivity":"حساسية الشذوذ"
    }
}[lang]

# PDF helper
def create_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Recent Sensor Data", ln=1)
    for _, row in data.tail(10).iterrows():
        pdf.cell(0, 10,
                 f"{row['timestamp']} T:{row['temp']:.1f} P:{row['pressure']:.1f} "
                 f"V:{row['vibration']:.1f} G:{row['gas']:.1f}", ln=1)
    return pdf.output(dest="S").encode('latin1')

# 8. Load data
df = pd.read_csv("sensor_data_simulated.csv", parse_dates=["Time"])
df.rename(columns={"Time":"timestamp", "Temperature (°C)":"temp","Pressure (psi)":"pressure",
                   "Vibration (g)":"vibration","Methane (CH₄ ppm)":"gas"}, inplace=True)

# 9. Real-time streaming (auto-refresh)
st_autorefresh = st.experimental_memo.clear  # handled by cache
st_autorefresh()  # clear on each run

# 10. Page navigation
pages = [T["Dashboard"], T["Simulation"], T["Predictive Analysis"], T["Smart Solutions"], T["Settings"], T["About"]]
page = st.radio("", pages, horizontal=True)

# 11. Dashboard
if page==T["Dashboard"]:
    st.header(T["Dashboard"])
    if df.empty:
        st.info(T["No data"])
    else:
        latest = df.iloc[-1]
        cols=st.columns(4)
        cols[0].metric("🌡️ Temp", f"{latest.temp:.2f}°C")
        cols[1].metric("⚡ Pressure", f"{latest.pressure:.2f} psi")
        cols[2].metric("📳 Vibration", f"{latest.vibration:.2f} g")
        cols[3].metric("🛢️ Methane", f"{latest.gas:.2f} ppm")
        # Sensor Map
        st.subheader(T["Sensor Map"])
        locs = pd.DataFrame({
            "lat":[27.0,26.8,27.1],"lon":[49.6,49.7,49.65],
            "sensor":["A","B","C"]
        })
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=locs,
            get_position="[lon, lat]",
            get_color=[0, 0, 255, 160],
            get_radius=5000,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=locs.lat.mean(), longitude=locs.lon.mean(), zoom=6)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                tooltip={"text": "{sensor}"}))

        # Export
        csv = df.to_csv(index=False)
        st.download_button(T["Export CSV"], data=csv, file_name="data.csv", mime="text/csv")
        pdf = create_pdf_report(df)
        st.download_button(T["Export PDF"], data=pdf, file_name="report.pdf", mime="application/pdf")

# 12. Simulation
elif page==T["Simulation"]:
    st.header(T["Simulation"])
    if df.empty:
        st.info(T["No data"])
    else:
        sens = st.slider(T["Anomaly Sensitivity"], 0.01,1.0,0.5)
        sd = {
            "temp": st.slider("Temp",20.0,60.0,float(df.temp.iloc[-1])),
            "pressure": st.slider("Pressure",80.0,150.0,float(df.pressure.iloc[-1])),
            "vibration": st.slider("Vib",0.0,2.0,float(df.vibration.iloc[-1])),
            "gas": st.slider("Gas",0.0,20.0,float(df.gas.iloc[-1]))
        }
        fig=px.imshow(pd.DataFrame([sd]).T, labels={"x":"Sensor","value":"Value"})
        st.plotly_chart(fig,use_container_width=True)

# 13. Predictive Analysis
elif page==T["Predictive Analysis"]:
    st.header(T["Predictive Analysis"])
    if df.empty:
        st.info(T["No data"])
    else:
        last=df.timestamp.max()
        fut=pd.DataFrame({
            "timestamp":[last+timedelta(hours=i) for i in range(1,73)],
            "temp":np.linspace(df.temp.iloc[-1],df.temp.iloc[-1]+2,72)
        })
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.timestamp.tail(24),y=df.temp.tail(24),name="Actual"))
        fig.add_trace(go.Scatter(x=fut.timestamp,y=fut.temp,name="Predicted",line=dict(dash="dash")))
        st.plotly_chart(fig,use_container_width=True)

# 14. Smart Solutions
elif page==T["Smart Solutions"]:
    st.header(T["Smart Solutions"])
    if df.empty:
        st.info(T["No data"])
    else:
        if st.button(T["Generate Solution"], key="gen", help="Click to get a solution"):
            sol = {"Solution":["Activate cooling"],"Duration":["15m"],"Priority":["High"],"Effectiveness":["95%"]}
            sol_df = pd.DataFrame(sol)
            fig=px.pie(sol_df, names="Solution", values="Effectiveness", title="Effectiveness")
            st.plotly_chart(fig,use_container_width=True)
            st.table(sol_df)

# 15. Settings
elif page==T["Settings"]:
    st.header(T["Settings"])
    st.write(f"{T['select_lang']} : {lang}")
    st.write(f"{T['select_pal']} : {palette}")

# 16. About
else:
    st.header(T["About"])
    # Aramco logo + links
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/00/Aramco_logo.svg", width=150)
    if lang=="en":
        st.write("A Smart Neural Digital Twin for industrial safety, real-time monitoring, anomaly detection, predictive analytics, and smart recommendations.")
        st.write("[GitHub](https://github.com/rrakanmarri1/Smart-neural-digital-twin) • [Confluence](#)")
    else:
        st.write("نظام توأم رقمي عصبي ذكي للسلامة الصناعية، مراقبة لحظية، كشف شذوذ، تحليل تنبؤي، وتوصيات ذكية.")
        st.write("[GitHub](https://github.com/rrakanmarri1/Smart-neural-digital-twin) • [Confluence](#)")

# 17. Toast notifications
if not df.empty and df.temp.iloc[-1] > 80:
    if st.session_state.last_toast != "high_temp":
        st.toast("⚠️ High temperature!", icon="⚠️")
        st.session_state.last_toast = "high_temp"
