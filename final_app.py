import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from core_systems import (
    logger, theme_manager, translator, logo_svg, 
    show_logo, show_system_status_banner, show_notification_history,
    mqtt_client, real_pi_controller, demo_df
)
from ai_systems import (
    lifelong_memory, ai_analyzer, sndt_chat, 
    generate_ai_response, generate_fallback_response
)
from advanced_systems import (
    digital_twin_optimizer, predictive_maintenance, 
    emergency_response, self_healing, sustainability_monitor
)

# -------------------- أقسام التطبيق --------------------
def dashboard_section():
    """لوحة التحكم الرئيسية"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[0]}</div>', unsafe_allow_html=True)
    
    show_system_status_banner()
    
    # عرض المقاييس الرئيسية
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('temperature')}</h3>
            <h2>{st.session_state.get('mqtt_temp', 55):.1f}°م</h2>
            <p>{'▲ عالية' if st.session_state.get('mqtt_temp', 55) > 58 else '▼ طبيعية'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('pressure')}</h3>
            <h2>{st.session_state.get('pressure', 7.2):.1f} بار</h2>
            <p>{'▲ مرتفع' if st.session_state.get('pressure', 7.2) > 7.5 else '▼ طبيعي'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{translator.get_text('methane')}</h3>
            <h2>{st.session_state.get('methane', 1.4):.1f} ppm</h2>
            <p>{'▲ مرتفع' if st.session_state.get('methane', 1.4) > 2.0 else '▼ منخفض'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        efficiency = sustainability_monitor.calculate_energy_efficiency()
        st.markdown(f"""
        <div class="metric-card">
            <h3>كفاءة الطاقة</h3>
            <h2>{efficiency:.1f}%</h2>
            <p>{'▲ جيدة' if efficiency > 80 else '▼ تحتاج تحسين'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # مخططات البيانات
    st.markdown(f'<div class="section-header">البيانات المباشرة</div>', unsafe_allow_html=True)
    
    live_data = pd.DataFrame({
        "time": [datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)],
        "Temperature": np.random.normal(st.session_state.get("mqtt_temp", 55), 1.5, 30),
        "Pressure": np.random.normal(st.session_state.get("pressure", 7.2), 0.2, 30),
        "Methane": np.random.normal(st.session_state.get("methane", 1.4), 0.1, 30)
    })
    
    fig_temp = px.line(live_data, x="time", y="Temperature", title="درجة الحرارة خلال last 30 minutes")
    fig_temp.update_layout(height=300, xaxis_title="الوقت", yaxis_title="درجة الحرارة (°م)")
    st.plotly_chart(fig_temp, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pressure = px.line(live_data, x="time", y="Pressure", title="الضغط خلال last 30 minutes")
        fig_pressure.update_layout(height=300, xaxis_title="الوقت", yaxis_title="الضغط (بار)")
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col2:
        fig_methane = px.line(live_data, x="time", y="Methane", title="الميثان خلال last 30 minutes")
        fig_methane.update_layout(height=300, xaxis_title="الوقت", yaxis_title="الميثان (ppm)")
        st.plotly_chart(fig_methane, use_container_width=True)
    
    # التنبيهات والتوصيات
    st.markdown(f'<div class="section-header">التنبيهات والتوصيات</div>', unsafe_allow_html=True)
    
    # التحقق من طوارئ
    current_sensor_data = {
        "mqtt_temp": st.session_state.get("mqtt_temp", 55),
        "pressure": st.session_state.get("pressure", 7.2),
        "methane": st.session_state.get("methane", 1.4),
        "vibration": st.session_state.get("vibration", 4.5),
        "flow_rate": st.session_state.get("flow_rate", 110)
    }
    
    emergencies = emergency_response.check_emergency_conditions(current_sensor_data)
    
    if emergencies:
        for emergency in emergencies:
            st.error(f"**تنبيه طوارئ**: {emergency['message']} (المستوى: {emergency['level']})")
            
            with st.expander("إجراءات الطوارئ المطلوبة"):
                for action in emergency['actions']:
                    st.write(f"• {action}")
    else:
        st.success("لا توجد تنبيهات طوارئ حالية")
    
    # توصيات التحسين
    optimizations = digital_twin_optimizer.analyze_current_state()
    
    if optimizations:
        st.warning("**توصيات التحسين**:")
        for opt in optimizations:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {opt['action']} (الأهمية: {opt['impact']})")
            with col2:
                if st.button("تطبيق", key=f"apply_opt_{opt['rule']}"):
                    success, message = digital_twin_optimizer.apply_optimization(opt)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
    
    # الإصلاح الذاتي
    healing_actions = self_healing.monitor_and_heal(current_sensor_data)
    if healing_actions:
        st.info("**تم تنفيذ الإصلاح الذاتي**:")
        for action in healing_actions:
            st.write(f"• {action['name']}: {action['result']}")

def analytics_ai_section():
    """التحليلات والذكاء الاصطناعي"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[1]}</div>', unsafe_allow_html=True)
    
    # تحليل البيانات
    st.markdown(f'<div class="section-header">تحليل البيانات المتقدم</div>', unsafe_allow_html=True)
    
    if "analytics_df" not in st.session_state:
        st.session_state["analytics_df"] = demo_df.copy()
    
    analysis_type = st.selectbox("نوع التحليل", [
        "كشف الشذوذ", 
        "التجميع", 
        "التنبؤ بالاتجاه",
        "الرؤى الذكية"
    ])
    
    if st.button("تشغيل التحليل"):
        with st.spinner("جاري تحليل البيانات..."):
            if analysis_type == "كشف الشذوذ":
                anomalies, analyzed_df = ai_analyzer.detect_anomalies(st.session_state["analytics_df"])
                st.session_state["analytics_df"] = analyzed_df
                
                if not anomalies.empty:
                    st.warning(f"تم كشف {len(anomalies)} نقطة شاذة في البيانات")
                    
                    fig = px.scatter(analyzed_df, x="time", y="Temperature", 
                                    color="anomaly", title="كشف الشذوذ في درجة الحرارة")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("عرض البيانات الشاذة"):
                        st.dataframe(anomalies)
                else:
                    st.success("لا توجد شذوذ كبير في البيانات")
            
            elif analysis_type == "التجميع":
                clustered_df = ai_analyzer.cluster_data(st.session_state["analytics_df"])
                st.session_state["analytics_df"] = clustered_df
                
                st.success("تم تجميع البيانات بنجاح")
                
                fig = px.scatter(clustered_df, x="time", y="Temperature", 
                                color="cluster", title="تجميع البيانات")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "التنبؤ بالاتجاه":
                target = st.selectbox("اختر المتغير للتنبؤ", ["Temperature", "Pressure", "Methane"])
                hours = st.slider("عدد الساعات للتنبؤ", 1, 24, 6)
                
                analyzed_df, predictions = ai_analyzer.predict_trend(
                    st.session_state["analytics_df"], target, hours
                )
                
                if predictions is not None:
                    st.session_state["analytics_df"] = analyzed_df
                    
                    combined_df = pd.concat([
                        analyzed_df.assign(is_prediction=False),
                        predictions.assign(is_prediction=True)
                    ])
                    
                    fig = px.line(combined_df, x="time", y=target, 
                                 color="is_prediction", title=f"التنبؤ بـ {target}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("عرض بيانات التنبؤ"):
                        st.dataframe(predictions)
                else:
                    st.error("فشل في إنشاء التنبؤات. تأكد من وجود بيانات كافية.")
            
            elif analysis_type == "الرؤى الذكية":
                insights = ai_analyzer.generate_insights(st.session_state["analytics_df"])
                
                if insights:
                    st.success(f"تم توليد {len(insights)} رؤى من البيانات")
                    
                    for insight in insights:
                        if insight.get('type') == 'correlation_analysis':
                            st.markdown("**تحليل العلاقات القوية:**")
                            for corr in insight['correlations']:
                                st.write(f"- {corr['variables']}: علاقة {corr['type']} ({corr['correlation']:.2f})")
                        elif insight.get('type') == 'hourly_patterns':
                            st.markdown("**أنماط الأداء حسب الوقت:**")
                            for metric, pattern in insight['patterns'].items():
                                st.write(f"- {metric}: ذروة عند الساعة {pattern['peak_hour']}, أدنى عند الساعة {pattern['low_hour']}")
                        else:
                            st.write(f"""
                            **{insight['metric']}**:
                            - المتوسط: {insight['mean']:.2f}
                            - الاستقرار: {insight['stability']:.2f}
                            - المدى: {insight['range']}
                            - الاتجاه: {insight['trend']}
                            - القيمة الحالية: {insight.get('current_value', 'N/A'):.2f}
                            """)
                else:
                    st.error("فشل في توليد الرؤى. تأكد من وجود بيانات كافية.")
    
    # الصيانة التنبؤية
    st.markdown(f'<div class="section-header">الصيانة التنبؤية</div>', unsafe_allow_html=True)
    
    if st.button("تحليل صحة المكونات"):
        with st.spinner("جاري تحليل صحة المكونات..."):
            current_sensor_data = {
                "mqtt_temp": st.session_state.get("mqtt_temp", 55),
                "pressure": st.session_state.get("pressure", 7.2),
                "methane": st.session_state.get("methane", 1.4),
                "vibration": st.session_state.get("vibration", 4.5),
                "flow_rate": st.session_state.get("flow_rate", 110)
            }
            
            predictive_maintenance.update_component_health(current_sensor_data)
            predictions = predictive_maintenance.predict_failures()
            
            if predictions:
                st.warning("**تنبؤات الأعطال المحتملة:**")
                
                for pred in predictions:
                    progress_value = pred["failure_probability"] / 100
                    color = "red" if pred["urgency"] == "high" else "orange" if pred["urgency"] == "medium" else "blue"
                    
                    st.write(f"**{pred['component']}**")
                    st.write(f"صحة المكون: {pred['health']:.1f}%")
                    st.write(f"احتمالية العطل: {pred['failure_probability']:.1f}%")
                    st.progress(progress_value, text=f"احتمالية العطل: {pred['failure_probability']:.1f}%")
                    
                    if st.button(f"جدولة صيانة {pred['component']}", key=f"maint_{pred['component']}"):
                        schedule_date = predictive_maintenance.schedule_maintenance(
                            pred["component"], pred["recommended_action"]
                        )
                        st.success(f"تم جدولة الصيانة للتاريخ: {schedule_date.strftime('%Y-%m-%d')}")
                    
                    st.divider()
            else:
                st.success("لا توجد تنبؤات بأعطال محتملة في الوقت الحالي")
    
    # الذاكرة الدائمة والتعلم
    st.markdown(f'<div class="section-header">الذاكرة الدائمة والتعلم</div>', unsafe_allow_html=True)
    
    if st.button("عرض التوصيات بناءً على الخبرة"):
        recommendations = lifelong_memory.get_recommendations("optimization", "الحالة الحالية")
        
        if recommendations:
            st.info("**توصيات مستندة على الخبرة السابقة:**")
            for rec in recommendations:
                st.write(f"- {rec['recommendation']}")
                st.write(f"  الثقة: {rec['confidence']:.0%}")
                st.write(f"  مستند على: {rec['based_on']}")
                st.divider()
        else:
            st.info("لا توجد توصيات مستندة على الخبرة yet. سيتم توليدها مع مرور الوقت.")
    
    # تحليل الاتجاهات
    if st.button("تحليل الاتجاهات من التجارب السابقة"):
        trends = lifelong_memory.analyze_trends("optimization")
        
        st.metric("معدل النجاح", f"{trends['success_rate']:.0%}")
        
        if trends["common_issues"]:
            st.write("**المشاكل الشائعة:**")
            for issue, count in trends["common_issues"]:
                st.write(f"- {issue} (حدث {count} مرات)")

def operations_control_section():
    """العمليات والتحكم"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[2]}</div>', unsafe_allow_html=True)
    
    # إحصائيات العمليات
    st.markdown(f'<div class="section-header">إحصائيات العمليات</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("الإنتاج اليومي", "2,450 طن", "+3.2%")
    
    with col2:
        st.metric("كفاءة الطاقة", "87.5%", "+1.8%")
    
    with col3:
        st.metric("الجودة", "98.2%", "-0.4%")
    
    # مخطط أداء العمليات
    operation_data = pd.DataFrame({
        "الفترة": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو"],
        "الإنتاج": [2200, 2350, 2400, 2300, 2450, 2500],
        "الكفاءة": [82, 85, 84, 87, 86, 88],
        "الجودة": [97, 98, 97.5, 98.2, 97.8, 98.5]
    })
    
    fig = px.line(operation_data, x="الفترة", y=["الإنتاج", "الكفاءة", "الجودة"],
                 title="أداء العمليات خلال الأشهر الستة الماضية",
                 labels={"value": "القيمة", "variable": "المؤشر"})
    st.plotly_chart(fig, use_container_width=True)
    
    # إدارة الجودة
    st.markdown(f'<div class="section-header">إدارة الجودة</div>', unsafe_allow_html=True)
    
    quality_data = pd.DataFrame({
        "البند": ["النقاوة", "اللزوجة", "الكثافة", "اللون", "التركيب"],
        "القيمة": [98.5, 96.8, 99.2, 97.5, 98.8],
        "المعيار": [95, 95, 98, 96, 97]
    })
    
    fig = px.bar(quality_data, x="البند", y=["القيمة", "المعيار"],
                title="مقارنة جودة المنتج بالمعايير",
                barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # التحكم بالأجهزة
    st.markdown(f'<div class="section-header">التحكم بالأجهزة</div>', unsafe_allow_html=True)
    
    # حالة Raspberry Pi
    pi_status = st.session_state.get("pi_status", "disconnected")
    status_color = "#2ecc71" if pi_status == "connected" else "#e74c3c"
    
    st.markdown(f"""
    <div style="padding:1rem; background:#f8f9fa; border-radius:0.5rem; margin-bottom:1rem;">
        <strong>الحالة:</strong> <span style="color:{status_color}">{pi_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # الاتصال بـ Raspberry Pi
    if pi_status != "connected":
        with st.form("connect_pi_form"):
            st.write("إعدادات الاتصال بـ Raspberry Pi")
            ip_address = st.text_input("IP Address", "192.168.1.100")
            username = st.text_input("Username", "pi")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("الاتصال"):
                with st.spinner("جاري الاتصال..."):
                    success, message = real_pi_controller.connect_to_pi(ip_address, username, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
    else:
        if st.button("قطع الاتصال"):
            real_pi_controller.disconnect()
            st.success("تم قطع الاتصال بـ Raspberry Pi")
            st.rerun()
    
    # التحكم في المنافذ
    if pi_status == "connected":
        st.markdown(f'<div class="section-header">التحكم في المنافذ</div>', unsafe_allow_html=True)
        
        # تهيئة GPIO
        if not real_pi_controller.gpio_initialized:
            if st.button("تهيئة منافذ GPIO"):
                success, message = real_pi_controller.initialize_gpio()
                if success:
                    st.success(message)
                else:
                    st.error(message)
                st.rerun()
        else:
            st.success("تم تهيئة منافذ GPIO")
            
            # التحكم في منافذ الإخراج
            st.subheader("منافذ الإخراج")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("تشغيل المنفذ 1"):
                    success, message = real_pi_controller.control_output(1, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 1"):
                    success, message = real_pi_controller.control_output(1, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col2:
                if st.button("تشغيل المنفذ 2"):
                    success, message = real_pi_controller.control_output(2, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 2"):
                    success, message = real_pi_controller.control_output(2, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col3:
                if st.button("تشغيل المنفذ 3"):
                    success, message = real_pi_controller.control_output(3, True)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                if st.button("إيقاف المنفذ 3"):
                    success, message = real_pi_controller.control_output(3, False)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            # قراءة منافذ الإدخال
            st.subheader("منافذ الإدخال")
            if st.button("قراءة المنفذ 4"):
                success, message, value = real_pi_controller.read_input(4)
                if success:
                    st.info(message)
                else:
                    st.error(message)
            
            if st.button("قراءة المنفذ 5"):
                success, message, value = real_pi_controller.read_input(5)
                if success:
                    st.info(message)
                else:
                    st.error(message)
    
    # إعدادات MQTT
    st.markdown(f'<div class="section-header">إعدادات اتصال MQTT</div>', unsafe_allow_html=True)
    
    mqtt_connected = st.session_state.get("mqtt_connected", False)
    st.write(f"الحالة: {'متصل' if mqtt_connected else 'غير متصل'}")
    
    if mqtt_connected:
        if st.button("قطع الاتصال MQTT"):
            mqtt_client.disconnect()
            st.success("تم قطع الاتصال MQTT")
            st.rerun()
    else:
        if st.button("إعادة الاتصال MQTT"):
            success = mqtt_client.connect_with_retry()
            if success:
                st.success("تم الاتصال MQTT بنجاح")
            else:
                st.error("فشل الاتصال MQTT")
            st.rerun()
    
    # إرسال رسالة MQTT
    st.subheader("إرسال رسالة MQTT")
    topic = st.selectbox("الموضوع", ["sndt/temperature", "sndt/pressure", "sndt/methane", "sndt/control"])
    message = st.text_input("الرسالة", "25.5")
    
    if st.button("إرسال الرسالة"):
        if mqtt_client.publish(topic, message):
            st.success("تم إرسال الرسالة بنجاح")
        else:
            st.error("فشل إرسال الرسالة")

def safety_emergency_section():
    """السلامة والطوارئ"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[3]}</div>', unsafe_allow_html=True)
    
    # نظرة عامة على السلامة
    st.markdown(f'<div class="section-header">نظرة عامة على السلامة</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_since = (datetime.now() - datetime(2023, 1, 1)).days
        st.metric("الأيام بدون حوادث", f"{days_since} يوم")
    
    with col2:
        st.metric("التنبيهات النشطة", "3", "-1 من الأسبوع الماضي")
    
    with col3:
        st.metric("مستوى المخاطر", "منخفض", "2%")
    
    # خريطة الحرارة للمخاطر
    risk_data = pd.DataFrame({
        "المنطقة": ["التفاعل", "التخزين", "المناولة", "التحكم", "الخدمات"],
        "مستوى المخاطرة": [8, 6, 7, 3, 4]
    })
    
    fig = px.bar(risk_data, x="المنطقة", y="مستوى المخاطرة", 
                title="مستويات المخاطرة حسب المنطقة",
                color="مستوى المخاطرة", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)
    
    # التنبيهات الحالية
    st.markdown(f'<div class="section-header">التنبيهات الحالية</div>', unsafe_allow_html=True)
    show_notification_history()
    
    # إعدادات التنبيهات
    st.markdown(f'<div class="section-header">إعدادات التنبيهات</div>', unsafe_allow_html=True)
    
    twilio_enabled = st.toggle("تفعيل تنبيهات SMS", value=st.session_state.get("twilio_enabled", True))
    st.session_state["twilio_enabled"] = twilio_enabled
    
    if twilio_enabled:
        phone_number = st.text_input("رقم الهاتف للتنبيهات", value=st.session_state.get("alert_phone_number", ""))
        st.session_state["alert_phone_number"] = phone_number
        
        # اختبار إرسال رسالة
        if st.button("اختبار إرسال رسالة"):
            if phone_number:
                from core_systems import send_twilio_alert
                success, message = send_twilio_alert("هذه رسالة اختبار من نظام SNDT", phone_number)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("يرجى إدخال رقم الهاتف أولاً")
    
    # عتبات التنبيه
    st.subheader("عتبات التنبيه")
    
    temp_threshold = st.slider("عتبة درجة الحرارة (°م)", 50, 80, 65)
    pressure_threshold = st.slider("عتبة الضغط (بار)", 5, 12, 9)
    methane_threshold = st.slider("عتبة الميثان (ppm)", 1, 5, 3)
    
    if st.button("حفظ العتبات"):
        st.session_state["alert_thresholds"] = {
            "temperature": temp_threshold,
            "pressure": pressure_threshold,
            "methane": methane_threshold
        }
        st.success("تم حفظ عتبات التنبيه")
    
    # بروتوكولات الطوارئ
    st.markdown(f'<div class="section-header">بروتوكولات الطوارئ</div>', unsafe_allow_html=True)
    
    emergency_level = st.selectbox("مستوى الطوارئ", ["منخفض", "متوسط", "مرتفع", "حرج"])
    
    procedures = emergency_response.get_emergency_procedures(emergency_level.lower())
    
    st.write(f"**إجراءات الطوارئ لمستوى {emergency_level}:**")
    for procedure in procedures:
        st.write(f"• {procedure}")
    
    # محاكاة طوارئ
    if st.button("محاكاة حالة طوارئ", type="secondary"):
        st.session_state["disaster_simulated"] = True
        
        # محاكاة ارتفاع مفاجئ في درجة الحرارة
        st.session_state["mqtt_temp"] = 78.5
        st.session_state["pressure"] = 9.8
        st.session_state["methane"] = 3.7
        
        st.error("تم تفعيل محاكاة الطوارئ! تم رفع قيم الاستشعار إلى مستويات خطيرة.")
        st.rerun()
    
    if st.session_state.get("disaster_simulated", False):
        if st.button("إنهاء محاكاة الطوارئ"):
            st.session_state["disaster_simulated"] = False
            
            # إعادة القيم إلى وضعها الطبيعي
            st.session_state["mqtt_temp"] = 55.0
            st.session_state["pressure"] = 7.2
            st.session_state["methane"] = 1.4
            
            st.success("تم إنهاء محاكاة الطوارئ وأعيدت القيم إلى وضعها الطبيعي.")
            st.rerun()

def sustainability_energy_section():
    """الاستدامة والطاقة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[4]}</div>', unsafe_allow_html=True)
    
    # البصمة الكربونية
    st.markdown(f'<div class="section-header">البصمة الكربونية</div>', unsafe_allow_html=True)
    
    # حساب البصمة الكربونية الحالية
    sensor_data = {
        "mqtt_temp": st.session_state.get("mqtt_temp", 55),
        "pressure": st.session_state.get("pressure", 7.2),
        "flow_rate": st.session_state.get("flow_rate", 110)
    }
    
    footprint = sustainability_monitor.calculate_carbon_footprint(sensor_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("انبعاثات CO₂", f"{footprint.get('co2_emissions', 0):.1f} kg", "-2.3%")
    
    with col2:
        st.metric("استهلاك الطاقة", f"{footprint.get('energy_consumption', 0):.1f} kWh", "-1.8%")
    
    with col3:
        st.metric("استهلاك المياه", f"{footprint.get('water_usage', 0):.1f} m³", "-3.1%")
    
    # مخطط البصمة الكربونية
    footprint_data = pd.DataFrame({
        "الشهر": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو"],
        "انبعاثات CO₂": [1450, 1380, 1320, 1280, 1250, 1220],
        "الهدف": [1300, 1250, 1200, 1150, 1100, 1050]
    })
    
    fig = px.line(footprint_data, x="الشهر", y=["انبعاثات CO₂", "الهدف"],
                 title="اتجاه البصمة الكربونية خلال الأشهر الستة الماضية",
                 labels={"value": "انبعاثات CO₂ (kg)", "variable": "المتغير"})
    st.plotly_chart(fig, use_container_width=True)
    
    # كفاءة الطاقة
    st.markdown(f'<div class="section-header">كفاءة الطاقة</div>', unsafe_allow_html=True)
    
    efficiency = sustainability_monitor.calculate_energy_efficiency()
    st.metric("كفاءة الطاقة الإجمالية", f"{efficiency:.1f}%")
    
    efficiency_data = pd.DataFrame({
        "المعدة": ["المفاعل", "المضخات", "التبريد", "التحكم", "الإضاءة"],
        "الكفاءة": [85, 78, 92, 88, 95],
        "استهلاك الطاقة": [45, 25, 15, 10, 5]  # نسب مئوية
    })
    
    fig = px.bar(efficiency_data, x="المعدة", y="الكفاءة",
                title="كفاءة الطاقة حسب المعدة",
                color="الكفاءة", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)
    
    # تقرير الاستدامة
    st.markdown(f'<div class="section-header">تقرير الاستدامة</div>', unsafe_allow_html=True)
    
    if st.button("إنشاء تقرير الاستدامة"):
        with st.spinner("جاري إنشاء تقرير الاستدامة..."):
            report = sustainability_monitor.generate_sustainability_report()
            
            st.success("تم إنشاء تقرير الاستدامة بنجاح!")
            
            st.write(f"**تاريخ التقرير:** {datetime.fromisoformat(report['report_date']).strftime('%Y-%m-%d %H:%M')}")
            st.metric("كفاءة الطاقة", f"{report['energy_efficiency']:.1f}%")
            st.metric("البصمة الكربونية", f"{report['carbon_footprint']:.1f} kg CO₂")
            st.metric("استهلاك المياه", f"{report['water_usage']:.1f} m³")
            
            if report['recommendations']:
                st.write("**توصيات تحسين الاستدامة:**")
                for recommendation in report['recommendations']:
                    st.write(f"• {recommendation}")
    
    # أهداف الاستدامة
    st.markdown(f'<div class="section-header">أهداف الاستدامة</div>', unsafe_allow_html=True)
    
    goals_data = pd.DataFrame({
        "الهدف": [
            "خفض انبعاثات CO₂ بنسبة 20%",
            "تقليل استهلاك الطاقة بنسبة 15%",
            "خفض استهلاك المياه بنسبة 25%",
            "زيادة إعادة التدوير إلى 75%",
            "تحقيق الصفر من النفايات الخطرة"
        ],
        "التقدم": [65, 80, 45, 70, 90],
        "الموعد النهائي": ["2023-12-31", "2023-10-31", "2024-03-31", "2023-09-30", "2024-06-30"]
    })
    
    for _, goal in goals_data.iterrows():
        st.write(f"**{goal['الهدف']}**")
        st.progress(goal['التقدم'] / 100, text=f"{goal['التقدم']}% مكتمل")
        st.write(f"الموعد النهائي: {goal['الموعد النهائي']}")
        st.divider()

def smart_assistant_section():
    """المساعد الذكي"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[5]}</div>', unsafe_allow_html=True)
    
    # واجهة الدردشة
    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4 style="margin:0;">المساعد الذكي SNDT</h4>
        <p style="margin:0.5rem 0 0 0; color: #666;">اسألني عن حالة النظام، التحليلات، التنبؤات، أو أي استفسار آخر</p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض سجل المحادثة
    chat_history = st.session_state.get("chat_history", [])
    
    for message in chat_history[-10:]:
        if message["user"]:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: left;">
                <strong>You:</strong> {message["user"]}
            </div>
            """, unsafe_allow_html=True)
        
        if message["assistant"]:
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: right;">
                <strong>المساعد:</strong> {message["assistant"]}
            </div>
            """, unsafe_allow_html=True)
    
    # إدخال الرسالة
    user_input = st.chat_input("اكتب رسالتك هنا...")
    
    if user_input:
        # عرض رسالة المستخدم فوراً
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: left;">
            <strong>You:</strong> {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # توليد الرد
        with st.spinner("جاري التفكير..."):
            response = generate_ai_response(user_input)
            
            # عرض رد المساعد
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: right;">
                <strong>المساعد:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
    
    # إمكانيات المساعد
    st.markdown(f'<div class="section-header">إمكانيات المساعد</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**يمكنني المساعدة في:**")
        st.write("• مراقبة حالة النظام")
        st.write("• تحليل البيانات والاتجاهات")
        st.write("• التنبؤ بالمشاكل المحتملة")
        st.write("• تقديم التوصيات الذكية")
        st.write("• الإجابة على الأسئلة العامة")
    
    with col2:
        st.write("**اسألني أمثلة:**")
        st.write("• ما هي حالة النظام الحالية؟")
        st.write("• كيف تبدو درجة الحرارة الآن؟")
        st.write("• هل هناك أي مشاكل متوقعة؟")
        st.write("• ما هي توصياتك لتحسين الأداء؟")
        st.write("• ما هو الطقس اليوم؟")
    
    # إعدادات الذكاء الاصطناعي
    st.markdown(f'<div class="section-header">إعدادات الذكاء الاصطناعي</div>', unsafe_allow_html=True)
    
    openai_enabled = st.toggle("تفعيل OpenAI (GPT)", value=st.session_state.get("openai_enabled", False))
    st.session_state["openai_enabled"] = openai_enabled
    
    if openai_enabled:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
        st.session_state["openai_api_key"] = api_key
        
        if api_key:
            st.success("تم تفعيل الذكاء الاصطناعي المتقدم")
        else:
            st.warning("يرجى إدخال مفتاح API لتفعيل الذكاء الاصطناعي المتقدم")
    else:
        st.info("يستخدم النظام الذكاء الاصطناعي المدمج (بدون OpenAI)")

def settings_help_section():
    """الإعدادات والمساعدة"""
    st.markdown(f'<div class="main-header">{translator.get_text("side_sections")[6]}</div>', unsafe_allow_html=True)
    
    # الإعدادات
    st.markdown(f'<div class="section-header">الإعدادات</div>', unsafe_allow_html=True)
    
    # إعدادات اللغة
    lang = st.radio("اللغة", ["العربية", "English"], horizontal=True, index=0 if st.session_state.get("lang", "ar") == "ar" else 1)
    st.session_state["lang"] = "ar" if lang == "العربية" else "en"
    
    # إعدادات الثيم
    theme = st.radio("المظهر", ["فاتح", "داكن"], horizontal=True, index=0 if st.session_state.get("theme", "light") == "light" else 1)
    st.session_state["theme"] = "light" if theme == "فاتح" else "dark"
    theme_manager.apply_theme_styles()
    
    # إعدادات النظام
    st.subheader("إعدادات النظام")
    
    simulation_active = st.toggle("وضع المحاكاة النشط", value=st.session_state.get("simulation_active", True))
    st.session_state["simulation_active"] = simulation_active
    
    data_refresh = st.slider("معدل تحديث البيانات (ثواني)", 1, 60, 5)
    st.session_state["data_refresh_rate"] = data_refresh
    
    if st.button("حفظ الإعدادات"):
        st.success("تم حفظ الإعدادات بنجاح")
    
    # المعلومات والمساعدة
    st.markdown(f'<div class="section-header">المعلومات والمساعدة</div>', unsafe_allow_html=True)
    
    with st.expander("عن التطبيق"):
        st.write("""
        ### منصة SNDT - التوأم الرقمي الذكي
        
        **الإصدار:** 1.0.0
        **تاريخ البناء:** 2025-07-01
        
        منصة SNDT هي نظام متكامل لإدارة المصانع والعمليات الصناعية باستخدام تقنيات التوأم الرقمي والذكاء الاصطناعي.
        
        **المميزات الرئيسية:**
        - مراقبة البيانات في الوقت الحقيقي
        - التحليلات التنبؤية والذكاء الاصطناعي
        - الصيانة التنبؤية
        - إدارة السلامة والطوارئ
        - تحليل الاستدامة والكفاءة
        - المساعد الذكي المتقدم
        
        **التقنيات المستخدمة:**
        - Python, Streamlit
        - MQTT للاتصال بأجهزة IoT
        - Redis للتخزين المؤقت
        - TensorFlow/PyTorch للذكاء الاصطناعي
        - Plotly للتصورات
        """)
    
    with st.expander("دليل المستخدم"):
        st.write("""
        ### دليل استخدام منصة SNDT
        
        **لوحة التحكم الرئيسية:**
        - عرض البيانات المباشرة من أجهزة الاستشعار
        - متابعة المقاييس الرئيسية لأداء النظام
        - الاطلاع على التنبيهات والتوصيات
        
        **التحليلات والذكاء الاصطناعي:**
        - تحليل البيانات التاريخية
        - كشف الشذوذ والأنماط
        - التنبؤ بالاتجاهات المستقبلية
        - الصيانة التنبؤية
        
        **العمليات والتحكم:**
        - متابعة إحصائيات الإنتاج
        - إدارة الجودة
        - التحكم بالأجهزة والأنظمة
        - إعدادات الاتصالات
        
        **السلامة والطوارئ:**
        - متابعة تنبيهات السلامة
        - إعدادات التنبيهات
        - بروتوكولات الطوارئ
        
        **الاستدامة والطاقة:**
        - متابعة البصمة الكربونية
        - تحليل كفاءة الطاقة
        - أهداف الاستدامة
        
        **المساعد الذكي:**
        - التفاعل مع النظام باستخدام الذكاء الاصطناعي
        - الحصول على إجابات لاستفساراتك
        - تلقي التوصيات الذكية
        """)
    
    with st.expander("استكشاف الأخطاء وإصلاحها"):
        st.write("""
        ### استكشاف الأخطاء وإصلاحها
        
        **لا يتم تحميل البيانات:**
        1. تحقق من اتصال الإنترنت
        2. تأكد من أن خادم MQTT يعمل
        3. تحقق من إعدادات الاتصال
        
        **المساعد الذكي لا يستجيب:**
        1. تحقق من اتصال OpenAI API (إذا كان مفعلاً)
        2. تأكد من صحة مفتاح API
        
        **لا يمكن الاتصال بـ Raspberry Pi:**
        1. تحقق من عنوان IP وبيانات الاعتماد
        2. تأكد من أن خدمة SSH مفعلة على Raspberry Pi
        3. تحقق من إعدادات الشبكة
        
        **البيانات لا تتحدث:**
        1. تحقق من أن وضع المحاكاة مفعل
        2. تأكد من اتصال MQTT
        3. تحقق من إعدادات تحديث البيانات
        
        **للحصول على مساعدة إضافية:**
        - راجع documentation المرفق
        - اتصل بفريق الدعم الفني
        - تحقق من forums المجتمع
        """)
    
    # معلومات الاتصال
    st.markdown(f'<div class="section-header">الدعم والاتصال</div>', unsafe_allow_html=True)
    
    st.write("**ساعات الدعم:** الأحد - الخميس، 8 ص - 5 م")
    st.write("**هاتف الدعم:** +966 12 345 6789")
    st.write("**البريد الإلكتروني:** support@sndt.com")
    st.write("**الموقع الإلكتروني:** https://sndt.com")
    
    if st.button("طلب دعم فني"):
        st.info("تم إرسال طلب الدعم الفني. سيتصل بك فريق الدعم خلال 24 ساعة.")

# -------------------- التطبيق الرئيسي --------------------
def main():
    # تطبيق أنماط الثيم
    theme_manager.apply_theme_styles()
    
    # الشريط الجانبي
    with st.sidebar:
        show_logo()
        st.markdown(f'<div style="text-align:center; font-size:1.5rem; font-weight:bold; margin-bottom:1.5rem;">SNDT Platform</div>', unsafe_allow_html=True)
        
        # اختيار القسم
        sections = translator.get_text("side_sections")
        selected_section = st.radio("اختر القسم", sections, index=0)
        
        st.divider()
        
        # معلومات النظام
        st.write("**معلومات النظام:**")
        st.write(f"الحالة: {'متصل' if st.session_state.get('mqtt_connected', False) else 'غير متصل'}")
        st.write(f"آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}")
        
        if st.session_state.get("pi_connected", False):
            st.success("✓ Raspberry Pi متصل")
        else:
            st.error("✗ Raspberry Pi غير متصل")
        
        st.divider()
        
        # الإصدار
        st.write("الإصدار: 1.0.0")
    
    # عرض القسم المحدد
    section_index = sections.index(selected_section)
    
    if section_index == 0:
        dashboard_section()
    elif section_index == 1:
        analytics_ai_section()
    elif section_index == 2:
        operations_control_section()
    elif section_index == 3:
        safety_emergency_section()
    elif section_index == 4:
        sustainability_energy_section()
    elif section_index == 5:
        smart_assistant_section()
    elif section_index == 6:
        settings_help_section()

if __name__ == "__main__":
    main()
