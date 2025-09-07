import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import streamlit as st
from core_systems import logger, event_system
from ai_systems import lifelong_memory, ai_analyzer

# -------------------- نظام التحسين الذاتي --------------------
class DigitalTwinOptimizer:
    """نظام تحسين ذاتي للتوأم الرقمي باستخدام التعلم المعزز"""
    def __init__(self):
        self.optimization_history = []
        self.optimization_rules = self.load_optimization_rules()
    
    def load_optimization_rules(self):
        return {
            "temperature_high": {
                "condition": lambda data: data.get("mqtt_temp", 0) > 60,
                "action": "خفض درجة حرارة التشغيل بنسبة 5%",
                "impact": "high",
                "effect": lambda: setattr(st.session_state, "mqtt_temp", st.session_state.get("mqtt_temp", 55) * 0.95)
            },
            "temperature_low": {
                "condition": lambda data: data.get("mqtt_temp", 0) < 50,
                "action": "زيادة درجة حرارة التشغيل بنسبة 5%",
                "impact": "medium",
                "effect": lambda: setattr(st.session_state, "mqtt_temp", st.session_state.get("mqtt_temp", 55) * 1.05)
            },
            "pressure_high": {
                "condition": lambda data: data.get("pressure", 0) > 8.5,
                "action": "تقليل ضغط التشغيل إلى 7.0 بار",
                "impact": "high",
                "effect": lambda: setattr(st.session_state, "pressure", 7.0)
            },
            "pressure_low": {
                "condition": lambda data: data.get("pressure", 0) < 6.5,
                "action": "زيادة ضغط التشغيل إلى 7.2 بار",
                "impact": "medium",
                "effect": lambda: setattr(st.session_state, "pressure", 7.2)
            },
            "methane_high": {
                "condition": lambda data: data.get("methane", 0) > 2.5,
                "action": "تفعيل نظام التهوية الإضافي",
                "impact": "critical",
                "effect": lambda: setattr(st.session_state, "methane", st.session_state.get("methane", 1.4) * 0.7)
            },
            "energy_optimization": {
                "condition": lambda data: data.get("flow_rate", 0) < 100 and data.get("mqtt_temp", 0) < 50,
                "action": "ضبط معدل التدفق لتحسين كفاءة الطاقة",
                "impact": "medium",
                "effect": lambda: setattr(st.session_state, "flow_rate", 110)
            }
        }
    
    def analyze_current_state(self):
        current_data = {
            "mqtt_temp": st.session_state.get("mqtt_temp", 55),
            "pressure": st.session_state.get("pressure", 7.2),
            "methane": st.session_state.get("methane", 1.4),
            "vibration": st.session_state.get("vibration", 4.5),
            "flow_rate": st.session_state.get("flow_rate", 110.0)
        }
        
        recommendations = []
        
        for rule_name, rule in self.optimization_rules.items():
            if rule["condition"](current_data):
                recommendations.append({
                    "rule": rule_name,
                    "action": rule["action"],
                    "impact": rule["impact"],
                    "timestamp": datetime.now().isoformat()
                })
        
        if recommendations:
            st.session_state["recommendations"] = recommendations
            for rec in recommendations:
                self.optimization_history.append(rec)
                lifelong_memory.add_experience(
                    "optimization", 
                    f"حالة النظام: {current_data}", 
                    f"تم اقتراح: {rec['action']} بسبب {rec['rule']}"
                )
        
        logger.info(f"تم توليد {len(recommendations)} توصية تحسين")
        return recommendations
    
    def apply_optimization(self, optimization):
        try:
            action = optimization["action"]
            
            # تطبيق التأثير الفعلي للتحسين
            for rule_name, rule in self.optimization_rules.items():
                if rule["action"] == action:
                    rule["effect"]()
                    break
            
            optimization["applied_at"] = datetime.now().isoformat()
            optimization["status"] = "applied"
            
            st.session_state["optimization_history"].append(optimization)
            lifelong_memory.add_experience(
                "optimization_applied", 
                f"تم تطبيق: {optimization['action']}", 
                "تم تطبيق التحسين بنجاح"
            )
            
            logger.info(f"تم تطبيق التحسين: {optimization['action']}")
            return True, f"تم تطبيق التحسين: {optimization['action']}"
            
        except Exception as e:
            logger.error(f"خطأ في تطبيق التحسين: {e}")
            return False, f"خطأ في تطبيق التحسين: {str(e)}"
    
    def get_optimization_history(self, limit=10):
        return st.session_state.get("optimization_history", [])[-limit:]

digital_twin_optimizer = DigitalTwinOptimizer()

# -------------------- نظام الصيانة التنبؤية --------------------
class PredictiveMaintenance:
    """نظام الصيانة التنبؤية المدعوم بالذكاء الاصطناعي"""
    def __init__(self):
        self.maintenance_schedule = {}
    
    def update_component_health(self, sensor_data):
        temp = sensor_data.get("mqtt_temp", 55)
        pressure = sensor_data.get("pressure", 7.2)
        vibration = sensor_data.get("vibration", 4.5)
        
        # تحديث صحة المكونات بناءً على بيانات الاستشعار
        component_health = st.session_state.get("component_health", {})
        
        if temp > 65:
            component_health["heat_exchanger"]["health"] = max(0, component_health["heat_exchanger"]["health"] - 0.5)
            component_health["valves"]["health"] = max(0, component_health["valves"]["health"] - 0.3)
        
        if pressure > 8.0:
            component_health["compressor"]["health"] = max(0, component_health["compressor"]["health"] - 0.7)
            component_health["pumps"]["health"] = max(0, component_health["pumps"]["health"] - 0.4)
        
        if vibration > 6.0:
            component_health["compressor"]["health"] = max(0, component_health["compressor"]["health"] - 0.6)
            component_health["pumps"]["health"] = max(0, component_health["pumps"]["health"] - 0.5)
        
        # استعادة تدريجية للصحة مع مرور الوقت (إذا كانت الظروف طبيعية)
        if temp < 60 and pressure < 8.0 and vibration < 5.0:
            for component in component_health:
                component_health[component]["health"] = min(100, component_health[component]["health"] + 0.1)
        
        st.session_state["component_health"] = component_health
        logger.info("تم تحديث صحة المكونات بناءً على بيانات الاستشعار")
    
    def predict_failures(self):
        predictions = []
        component_health = st.session_state.get("component_health", {})
        
        for component, data in component_health.items():
            health = data["health"]
            last_maintenance = datetime.fromisoformat(data["last_maintenance"])
            days_since_maintenance = (datetime.now() - last_maintenance).days
            
            if health < 70:
                failure_prob = min(95, 100 - health + days_since_maintenance / 2)
                urgency = "high"
            elif health < 85:
                failure_prob = min(70, 100 - health + days_since_maintenance / 4)
                urgency = "medium"
            else:
                failure_prob = max(5, (100 - health) / 2)
                urgency = "low"
            
            if failure_prob > 30:
                predictions.append({
                    "component": component,
                    "health": health,
                    "failure_probability": failure_prob,
                    "urgency": urgency,
                    "recommended_action": f"صيانة {component}",
                    "days_since_maintenance": days_since_maintenance
                })
        
        st.session_state["maintenance_predictions"] = predictions
        logger.info(f"تم إنشاء {len(predictions)} تنبؤات صيانة")
        return predictions
    
    def schedule_maintenance(self, component, action):
        schedule_date = datetime.now() + timedelta(days=7)
        
        self.maintenance_schedule[component] = {
            "scheduled_date": schedule_date.isoformat(),
            "action": action,
            "scheduled_at": datetime.now().isoformat(),
            "status": "scheduled"
        }
        
        # تحديث تاريخ آخر صيانة للمكون
        component_health = st.session_state.get("component_health", {})
        if component in component_health:
            component_health[component]["last_maintenance"] = datetime.now().isoformat()
            st.session_state["component_health"] = component_health
        
        lifelong_memory.add_experience(
            "maintenance", 
            f"جدولة صيانة {component}", 
            f"تم جدولة {action} للتاريخ {schedule_date.strftime('%Y-%m-%d')}"
        )
        
        logger.info(f"تم جدولة صيانة {component} للتاريخ {schedule_date}")
        return schedule_date

predictive_maintenance = PredictiveMaintenance()

# -------------------- نظام الاستجابة للطوارئ --------------------
class EmergencyResponseSystem:
    """نظام متقدم للاستجابة للطوارئ والكوارث"""
    def __init__(self):
        self.emergency_protocols = self.load_emergency_protocols()
        self.emergency_levels = {
            "low": {"color": "#f39c12", "action": "مراقبة عن كثب"},
            "medium": {"color": "#e67e22", "action": "اتخاذ إجراء وقائي"},
            "high": {"color": "#e74c3c", "action": "إجراء عاجل مطلوب"},
            "critical": {"color": "#c0392b", "action": "إخلاء وإغلاق الطوارئ"}
        }
    
    def load_emergency_protocols(self):
        return {
            "temperature_extreme": {
                "condition": lambda data: data.get("mqtt_temp", 0) > 75 or data.get("mqtt_temp", 0) < 40,
                "message": "خطر: درجة حرارة غير طبيعية",
                "level_func": lambda data: "critical" if data.get("mqtt_temp", 0) > 75 else "high",
                "actions": [
                    "إيقاف النظام فوراً",
                    "تفعيل نظام التبريد/التسخين الاحتياطي",
                    "إخطار فريق الصيانة العاجل"
                ]
            },
            "pressure_extreme": {
                "condition": lambda data: data.get("pressure", 0) > 10 or data.get("pressure", 0) < 5,
                "message": "خطر: ضغط غير طبيعي",
                "level_func": lambda data: "critical",
                "actions": [
                    "تفعيل صمامات الأمان تلقائياً",
                    "تقليل ضغط التشغيل فوراً",
                    "إخلاء المنطقة إذا لزم الأمر"
                ]
            },
            "methane_leak": {
                "condition": lambda data: data.get("methane", 0) > 4.0,
                "message": "تحذير: تسرب غاز محتمل",
                "level_func": lambda data: "critical",
                "actions": [
                    "تفعيل نظام التهوية القصوى",
                    "إيقاف مصادر الاشتعال",
                    "إخلاء المنطقة فوراً",
                    "إخطار فريق الطوارئ"
                ]
            },
            "vibration_extreme": {
                "condition": lambda data: data.get("vibration", 0) > 8.0,
                "message": "خطر: اهتزازات غير طبيعية",
                "level_func": lambda data: "high",
                "actions": [
                    "إيقاف المعدات المتحركة",
                    "تفعيل نظام التثبيت الطارئ",
                    "إخلاء المنطقة المجاورة"
                ]
            }
        }
    
    def check_emergency_conditions(self, sensor_data):
        emergencies = []
        
        for protocol_name, protocol in self.emergency_protocols.items():
            if protocol["condition"](sensor_data):
                emergency = {
                    "protocol": protocol_name,
                    "message": protocol["message"],
                    "level": protocol["level_func"](sensor_data),
                    "actions": protocol["actions"],
                    "timestamp": datetime.now().isoformat(),
                    "sensor_data": sensor_data
                }
                
                emergencies.append(emergency)
                
                st.session_state["incident_timeline"].append(emergency)
                lifelong_memory.add_experience(
                    "emergency", 
                    f"تفعيل بروتوكول الطوارئ: {protocol_name}", 
                    f"{protocol['message']} - المستوى: {protocol['level']}"
                )
                
                if protocol["level"] in ["high", "critical"]:
                    self.trigger_emergency_alert(emergency)
        
        return emergencies
    
    def trigger_emergency_alert(self, emergency):
        alert_message = f"تنبيه طوارئ: {emergency['message']} | المستوى: {emergency['level']}"
        
        st.session_state["notification_history"].append({
            "type": "emergency_alert",
            "message": alert_message,
            "level": emergency["level"],
            "timestamp": datetime.now().isoformat()
        })
        
        if st.session_state.get("twilio_enabled", False):
            phone_number = st.session_state.get("alert_phone_number", "")
            if phone_number:
                from core_systems import send_twilio_alert
                send_twilio_alert(alert_message, phone_number)
        
        st.session_state["last_emergency_alert"] = emergency
        st.session_state["emergency_protocols_activated"] = True
        logger.warning(f"تم تفعيل تنبيه الطوارئ: {alert_message}")
    
    def get_emergency_procedures(self, level):
        procedures = {
            "low": [
                "مراقبة المؤشرات عن كثب",
                "إعداد تقرير للمشرف",
                "الاستعداد لإجراءات وقائية"
            ],
            "medium": [
                "تنبيه فريق الصيانة",
                "بدء التسجيل التفصيلي للبيانات",
                "تحضير المعدات للتدخل"
            ],
            "high": [
                "إخطار مدير المصنع",
                "بدء الإجراءات التصحيحية",
                "تحضير خطة إخلاء احتياطية"
            ],
            "critical": [
                "إخلاء المنطقة فوراً",
                "إيقاف النظام بالكامل",
                "إخطار الدفاع المدني والطوارئ"
            ]
        }
        
        return procedures.get(level, [])

emergency_response = EmergencyResponseSystem()

# -------------------- نظام الإصلاح الذاتي --------------------
class SelfHealingSystem:
    """نظام الإصلاح الذاتي التلقائي"""
    def __init__(self):
        self.healing_actions = []
        self.load_healing_protocols()
    
    def load_healing_protocols(self):
        self.healing_actions = [
            {
                "name": "ضبط درجة الحرارة التلقائي",
                "condition": lambda data: data.get("mqtt_temp", 0) > 62,
                "action": lambda: self.adjust_temperature(),
                "priority": "high"
            },
            {
                "name": "موازنة الضغط التلقائية",
                "condition": lambda data: data.get("pressure", 0) > 8.2,
                "action": lambda: self.balance_pressure(),
                "priority": "high"
            },
            {
                "name": "تحسين كفاءة الطاقة",
                "condition": lambda data: data.get("flow_rate", 0) < 100 and data.get("mqtt_temp", 0) < 50,
                "action": lambda: self.optimize_energy(),
                "priority": "medium"
            },
            {
                "name": "تقليل انبعاثات الميثان",
                "condition": lambda data: data.get("methane", 0) > 2.5,
                "action": lambda: self.reduce_methane(),
                "priority": "critical"
            }
        ]
    
    def monitor_and_heal(self, sensor_data):
        applied_actions = []
        
        for action in self.healing_actions:
            if action["condition"](sensor_data):
                try:
                    result = action["action"]()
                    applied_actions.append({
                        "name": action["name"],
                        "priority": action["priority"],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"تم تطبيق إصلاح ذاتي: {action['name']}")
                    
                except Exception as e:
                    logger.error(f"فشل الإصلاح الذاتي {action['name']}: {e}")
        
        if applied_actions:
            for action in applied_actions:
                st.session_state["self_healing_actions"] = st.session_state.get("self_healing_actions", []) + [action]
            
            lifelong_memory.add_experience(
                "self_healing", 
                f"تم تطبيق {len(applied_actions)} إجراء إصلاح ذاتي", 
                f"الإجراءات: {[a['name'] for a in applied_actions]}"
            )
        
        return applied_actions
    
    def adjust_temperature(self):
        current_temp = st.session_state.get("mqtt_temp", 55)
        new_temp = current_temp * 0.93
        st.session_state["mqtt_temp"] = new_temp
        
        return f"تم خفض درجة الحرارة من {current_temp:.1f} إلى {new_temp:.1f}°م"
    
    def balance_pressure(self):
        current_pressure = st.session_state.get("pressure", 7.2)
        new_pressure = 7.0
        st.session_state["pressure"] = new_pressure
        
        return f"تم ضبط الضغط من {current_pressure:.1f} إلى {new_pressure:.1f} بار"
    
    def optimize_energy(self):
        current_flow = st.session_state.get("flow_rate", 110)
        new_flow = 110
        st.session_state["flow_rate"] = new_flow
        
        return f"تم تحسين معدل التدفق من {current_flow} إلى {new_flow} لتحسين كفاءة الطاقة"
    
    def reduce_methane(self):
        current_methane = st.session_state.get("methane", 1.4)
        new_methane = current_methane * 0.7
        st.session_state["methane"] = new_methane
        
        return f"تم تقليل انبعاثات الميثان من {current_methane:.1f} إلى {new_methane:.1f} ppm"

self_healing = SelfHealingSystem()

# -------------------- نظام الاستدامة --------------------
class SustainabilityMonitor:
    """مراقبة وتحليل استدامة العمليات"""
    def __init__(self):
        self.carbon_footprint = {
            "energy_consumption": 0,
            "co2_emissions": 0,
            "water_usage": 0,
            "waste_production": 0
        }
        self.initialize_sustainability_metrics()
    
    def initialize_sustainability_metrics(self):
        st.session_state["carbon_footprint"] = {
            "total_emissions": 1250,
            "energy_consumption": 3500,
            "water_usage": 120,
            "recycling_rate": 65,
            "last_calculated": datetime.now().isoformat()
        }
    
    def calculate_carbon_footprint(self, sensor_data):
        temp = sensor_data.get("mqtt_temp", 55)
        pressure = sensor_data.get("pressure", 7.2)
        flow_rate = sensor_data.get("flow_rate", 110)
        
        # حساب أكثر دقة للبصمة الكربونية بناءً على البيانات الحالية
        energy_consumption = (temp * 0.8) + (pressure * 12) + (flow_rate * 0.15)
        co2_emissions = energy_consumption * 0.85
        water_usage = flow_rate * 0.05
        
        sustainability_data = {
            "energy_consumption": energy_consumption,
            "co2_emissions": co2_emissions,
            "water_usage": water_usage,
            "waste_production": energy_consumption * 0.02,
            "last_updated": datetime.now().isoformat()
        }
        
        st.session_state["carbon_footprint"] = sustainability_data
        logger.info("تم تحديث بيانات الاستدامة بناءً على قراءات الاستشعار")
        
        return sustainability_data
    
    def calculate_energy_efficiency(self):
        sensor_data = {
            "mqtt_temp": st.session_state.get("mqtt_temp", 55),
            "pressure": st.session_state.get("pressure", 7.2),
            "flow_rate": st.session_state.get("flow_rate", 110)
        }
        
        # قيم مثالية للنظام
        ideal_temp, ideal_pressure, ideal_flow = 55, 7.2, 110
        
        # حساب الكفاءة بناءً على الانحراف عن القيم المثالية
        temp_efficiency = max(0, 100 - abs(sensor_data["mqtt_temp"] - ideal_temp) * 2)
        pressure_efficiency = max(0, 100 - abs(sensor_data["pressure"] - ideal_pressure) * 10)
        flow_efficiency = max(0, 100 - abs(sensor_data["flow_rate"] - ideal_flow) * 0.5)
        
        # حساب الكفاءة العامة مع أوزان مختلفة
        overall_efficiency = (temp_efficiency * 0.4 + pressure_efficiency * 0.3 + flow_efficiency * 0.3)
        
        return overall_efficiency
    
    def generate_sustainability_report(self):
        footprint = st.session_state.get("carbon_footprint", {})
        efficiency = self.calculate_energy_efficiency()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "energy_efficiency": efficiency,
            "carbon_footprint": footprint.get("co2_emissions", 0),
            "water_usage": footprint.get("water_usage", 0),
            "waste_management": footprint.get("waste_production", 0) * 0.3,
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self):
        recommendations = []
        current_temp = st.session_state.get("mqtt_temp", 55)
        current_pressure = st.session_state.get("pressure", 7.2)
        current_flow = st.session_state.get("flow_rate", 110)
        
        if current_temp > 58:
            recommendations.append("خفض درجة حرارة التشغيل بنسبة 5% لتحسين كفاءة الطاقة")
        elif current_temp < 52:
            recommendations.append("زيادة درجة حرارة التشغيل بنسبة 3% لتحسين الكفاءة")
        
        if current_pressure > 7.5:
            recommendations.append("ضبط ضغط التشغيل إلى 7.0 بار لتقليل الاستهلاك")
        elif current_pressure < 6.8:
            recommendations.append("زيادة ضغط التشغيل إلى 7.2 بار لتحسين الأداء")
        
        if current_flow < 105:
            recommendations.append("تحسين معدل التدفق إلى 110 لتعزيز الكفاءة")
        
        if not recommendations:
            recommendations.append("النظام يعمل بكفاءة جيدة، الاستمرار في المراقبة الروتينية")
        
        return recommendations

sustainability_monitor = SustainabilityMonitor()
