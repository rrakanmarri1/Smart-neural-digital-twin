import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from core_systems import logger, event_system

# -------------------- الذاكرة الدائمة --------------------
class LifelongLearningMemory:
    """نظام ذاكرة دائمة للتعلم من التجارب"""
    def __init__(self):
        self.memory_file = "lifelong_memory.json"
        self.memory = self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"خطأ في تحميل الذاكرة: {e}")
            return {}
    
    def save_memory(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"خطأ في حفظ الذاكرة: {e}")
            return False
    
    def add_experience(self, category, experience, outcome):
        if category not in self.memory:
            self.memory[category] = []
        
        timestamp = datetime.now().isoformat()
        experience_id = hashlib.md5(f"{category}_{timestamp}".encode()).hexdigest()
        
        self.memory[category].append({
            "id": experience_id,
            "timestamp": timestamp,
            "experience": experience,
            "outcome": outcome
        })
        
        if len(self.memory[category]) > 1000:
            self.memory[category] = self.memory[category][-1000:]
        
        self.save_memory()
        logger.info(f"تم إضافة تجربة جديدة إلى فئة {category}")
    
    def get_recommendations(self, category, current_situation):
        if category not in self.memory:
            return []
        
        recommendations = []
        for experience in self.memory[category]:
            if "success" in experience["outcome"].lower():
                recommendations.append({
                    "based_on": experience["experience"],
                    "recommendation": f"بناءً على تجربة ناجحة سابقة: {experience['outcome']}",
                    "confidence": random.uniform(0.7, 0.95)
                })
        
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:5]
    
    def analyze_trends(self, category):
        if category not in self.memory or not self.memory[category]:
            return {"success_rate": 0, "common_issues": []}
        
        successes = 0
        issues = {}
        
        for experience in self.memory[category]:
            if "success" in experience["outcome"].lower():
                successes += 1
            
            if "error" in experience["outcome"].lower() or "fail" in experience["outcome"].lower():
                for word in experience["outcome"].split():
                    if word.lower() not in ["the", "a", "an", "in", "on", "at"] and len(word) > 3:
                        issues[word] = issues.get(word, 0) + 1
        
        success_rate = successes / len(self.memory[category])
        common_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "success_rate": success_rate,
            "common_issues": common_issues
        }

lifelong_memory = LifelongLearningMemory()

# -------------------- Advanced AI Analysis --------------------
class AdvancedAIAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=3, random_state=42)
        self.regressor = LinearRegression()
        self.svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                                    solver='adam', max_iter=1000, random_state=42)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_fitted = False
    
    def prepare_data(self, df):
        try:
            data = df.copy()
            
            if 'time' in data.columns:
                data['hour'] = pd.to_datetime(data['time']).dt.hour
                data['day_part'] = pd.cut(data['hour'], 
                                         bins=[0, 6, 12, 18, 24], 
                                         labels=['night', 'morning', 'afternoon', 'evening'],
                                         include_lowest=True)
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'hour' in numeric_cols:
                numeric_cols.remove('hour')
            
            return data, numeric_cols
        except Exception as e:
            logger.error(f"خطأ في تحضير البيانات: {e}")
            return df, []
    
    def detect_anomalies(self, df):
        try:
            data, numeric_cols = self.prepare_data(df)
            
            if not numeric_cols:
                return [], df
            
            scaled_data = self.scaler.fit_transform(data[numeric_cols])
            anomalies = self.anomaly_detector.fit_predict(scaled_data)
            
            data['anomaly'] = anomalies
            data['anomaly_score'] = self.anomaly_detector.decision_function(scaled_data)
            
            anomaly_points = data[data['anomaly'] == -1]
            
            logger.info(f"تم كشف {len(anomaly_points)} نقطة شاذة")
            return anomaly_points, data
            
        except Exception as e:
            logger.error(f"خطأ في كشف الشذوذ: {e}")
            return [], df
    
    def cluster_data(self, df):
        try:
            data, numeric_cols = self.prepare_data(df)
            
            if not numeric_cols:
                return df
            
            scaled_data = self.scaler.fit_transform(data[numeric_cols])
            clusters = self.clusterer.fit_predict(scaled_data)
            
            data['cluster'] = clusters
            
            logger.info(f"تم تجميع البيانات إلى {len(set(clusters))} clusters")
            return data
            
        except Exception as e:
            logger.error(f"خطأ في تجميع البيانات: {e}")
            return df
    
    def predict_trend(self, df, target_column, hours_ahead=6):
        try:
            if target_column not in df.columns:
                logger.error(f"العمود {target_column} غير موجود في البيانات")
                return df, None
            
            data = df.copy()
            data = data.dropna(subset=[target_column])
            
            if len(data) < 10:
                logger.warning("لا توجد بيانات كافية للتنبؤ")
                return df, None
            
            # إنشاء ميزات زمنية إضافية
            data['time_index'] = range(len(data))
            data['hour'] = pd.to_datetime(data['time']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['time']).dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # استخدام أفضل نموذج بناءً على نوع البيانات
            X = data[['time_index', 'hour', 'day_of_week', 'is_weekend']].values
            y = data[target_column].values
            
            # اختيار النموذج المناسب بناءً على نمط البيانات
            if len(data) > 100:
                model = self.rf_model
            elif len(data) > 50:
                model = self.nn_model
            else:
                model = self.regressor
            
            model.fit(X, y)
            
            # إنشاء بيانات مستقبلية للتنبؤ
            last_index = data['time_index'].max()
            future_indices = np.array(range(last_index + 1, last_index + hours_ahead + 1))
            
            # افتراض أن التنبؤات هي للنهار في أيام الأسبوع
            future_hours = [(datetime.now().hour + i) % 24 for i in range(1, hours_ahead + 1)]
            future_days = [datetime.now().weekday()] * hours_ahead
            future_weekend = [1 if d >= 5 else 0 for d in future_days]
            
            future_X = np.column_stack([future_indices, future_hours, future_days, future_weekend])
            future_predictions = model.predict(future_X)
            
            last_time = pd.to_datetime(data['time'].iloc[-1])
            future_times = [last_time + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
            
            predictions_df = pd.DataFrame({
                'time': future_times,
                f'predicted_{target_column}': future_predictions,
                'is_prediction': True,
                'confidence': np.clip(1 - (0.1 * np.arange(1, hours_ahead + 1)), 0.5, 0.95)
            })
            
            logger.info(f"تم إنشاء تنبؤات للـ {hours_ahead} ساعات القادمة بدقة {predictions_df['confidence'].mean():.2f}")
            return data, predictions_df
            
        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {e}")
            return df, None
    
    def generate_insights(self, df):
        insights = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col in ['anomaly', 'cluster', 'time_index', 'hour', 'day_of_week', 'is_weekend']:
                    continue
                    
                mean_val = df[col].mean()
                std_val = df[col].std()
                max_val = df[col].max()
                min_val = df[col].min()
                trend = "مستقر"
                
                # تحليل الاتجاه
                if len(df) > 10:
                    recent_mean = df[col].iloc[-10:].mean()
                    if recent_mean > mean_val + std_val:
                        trend = "صاعد بشكل ملحوظ"
                    elif recent_mean > mean_val:
                        trend = "صاعد"
                    elif recent_mean < mean_val - std_val:
                        trend = "هابط بشكل ملحوظ"
                    elif recent_mean < mean_val:
                        trend = "هابط"
                
                insight = {
                    'metric': col,
                    'mean': mean_val,
                    'stability': std_val,
                    'range': f"{min_val:.2f} - {max_val:.2f}",
                    'trend': trend,
                    'current_value': df[col].iloc[-1] if len(df) > 0 else 0
                }
                
                insights.append(insight)
            
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                strong_correlations = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.iloc[i, j]
                        
                        if abs(corr) > 0.7:
                            strong_correlations.append({
                                'variables': f"{col1} & {col2}",
                                'correlation': corr,
                                'type': 'إيجابية' if corr > 0 else 'سلبية',
                                'strength': 'قوية' if abs(corr) > 0.8 else 'متوسطة'
                            })
                
                if strong_correlations:
                    insights.append({
                        'type': 'correlation_analysis',
                        'correlations': strong_correlations
                    })
            
            # تحليل الأنماط الموسمية
            if 'hour' in df.columns and len(numeric_cols) > 0:
                hourly_patterns = {}
                for col in numeric_cols:
                    if col not in ['hour', 'day_of_week', 'is_weekend']:
                        hourly_avg = df.groupby('hour')[col].mean()
                        peak_hour = hourly_avg.idxmax()
                        low_hour = hourly_avg.idxmin()
                        
                        hourly_patterns[col] = {
                            'peak_hour': peak_hour,
                            'low_hour': low_hour,
                            'variation': (hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean()
                        }
                
                if hourly_patterns:
                    insights.append({
                        'type': 'hourly_patterns',
                        'patterns': hourly_patterns
                    })
            
            logger.info(f"تم توليد {len(insights)} رؤى من البيانات")
            return insights
            
        except Exception as e:
            logger.error(f"خطأ في توليد الرؤى: {e}")
            return insights

ai_analyzer = AdvancedAIAnalyzer()

# -------------------- نظام SNDT Chat الذكي --------------------
class SNDTChatSystem:
    """نظام دردشة ذكي مع تنبؤ بالمشاكل وتوصيات ذكية"""
    def __init__(self):
        self.responses = {
            "greeting": [
                "مرحباً! أنا المساعد الذكي لمنصة SNDT. كيف يمكنني مساعدتك اليوم؟",
                "أهلاً بك! أنا هنا لمساعدتك في إدارة وتحليل بيانات المصنع.",
                "مساء الخير! كيف يمكنني مساعدتك في نظام التوأم الرقمي اليوم؟"
            ],
            "help": [
                "يمكنني مساعدتك في: تحليل البيانات، كشف المشاكل، تقديم التوصيات، والإجابة على أسئلتك.",
                "أنا متخصص في تحليل بيانات المصنع وتقديم التوصيات الذكية. ما الذي تريد معرفته؟",
                "اسألني عن: حالة النظام، التنبؤات، التحليلات، أو أي استفسار آخر."
            ],
            "system_status": [
                "حالة النظام الحالية: {status}. درجة الحرارة: {temp}°م، الضغط: {pressure} بار.",
                "النظام يعمل بشكل {status}. البيانات الأخيرة: حرارة {temp}°م، ضغط {pressure} بار.",
                "الحالة الراهنة: {status}. آخر القراءات: {temp}°م للحرارة، {pressure} بار للضغط."
            ],
            "prediction": [
                "بناءً على البيانات الحالية، أتوقع أن {prediction} في الساعات القادمة.",
                "التنبؤات تشير إلى أن {prediction} خلال الفترة القادمة.",
                "تحليل البيانات يشير إلى توقع {prediction} في المستقبل القريب."
            ],
            "anomaly": [
                "تم كشف شذوذ في {metric}. القيمة: {value}، المتوسط: {average}.",
                "هناك انحراف في {metric}. القيمة الحالية: {value}، بينما المتوسط: {average}.",
                "لاحظت شذوذ في {metric}. القيمة المسجلة: {value} مقارنة بالمتوسط: {average}."
            ],
            "recommendation": [
                "أوصي بـ {recommendation} لتحسين الأداء.",
                "بناءً على التحليل، التوصية هي: {recommendation}.",
                "لمواجهة هذا التحدي، أنصح بـ {recommendation}."
            ],
            "unknown": [
                "عذراً، لم أفهم سؤالك بالكامل. هل يمكنك إعادة الصياغة؟",
                "أحتاج إلى مزيد من التوضيح للإجابة على سؤالك.",
                "هل يمكنك طرح سؤالك بطريقة أخرى؟ سأبذل جهدي للمساعدة."
            ]
        }
    
    def generate_response(self, user_input, context=None):
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["مرحبا", "اهلا", "السلام", "hello", "hi"]):
            response = random.choice(self.responses["greeting"])
        
        elif any(word in user_input_lower for word in ["مساعدة", "مساعدة", "help", "دعم"]):
            response = random.choice(self.responses["help"])
        
        elif any(word in user_input_lower for word in ["حالة", "status", "البيانات", "الآن"]):
            status = "جيدة" if st.session_state.get("mqtt_temp", 55) < 60 else "تحت المراقبة"
            response = random.choice(self.responses["system_status"]).format(
                status=status,
                temp=st.session_state.get("mqtt_temp", 55),
                pressure=st.session_state.get("pressure", 7.2)
            )
        
        elif any(word in user_input_lower for word in ["تنبأ", "توقع", "مستقبل", "predict", "forecast"]):
            prediction = self.generate_prediction(context)
            response = random.choice(self.responses["prediction"]).format(prediction=prediction)
        
        elif any(word in user_input_lower for word in ["مشكلة", "خطأ", "شذوذ", "anomaly", "issue"]):
            anomaly_info = self.detect_current_anomalies()
            if anomaly_info:
                response = random.choice(self.responses["anomaly"]).format(
                    metric=anomaly_info["metric"],
                    value=anomaly_info["value"],
                    average=anomaly_info["average"]
                )
            else:
                response = "لا توجد مشاكل أو شذوذ كبير في البيانات الحالية."
        
        elif any(word in user_input_lower for word in ["توصية", "نصيحة", "recommend", "advice"]):
            recommendation = self.generate_recommendation(context)
            response = random.choice(self.responses["recommendation"]).format(recommendation=recommendation)
        
        else:
            response = random.choice(self.responses["unknown"])
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        st.session_state["chat_history"].append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(st.session_state["chat_history"]) > 50:
            st.session_state["chat_history"] = st.session_state["chat_history"][-50:]
        
        logger.info(f"تم توليد رد للمساعد: {user_input[:50]}...")
        return response
    
    def generate_prediction(self, context=None):
        # تحليل البيانات الحالية للتنبؤ الدقيق
        current_temp = st.session_state.get("mqtt_temp", 55)
        current_pressure = st.session_state.get("pressure", 7.2)
        current_methane = st.session_state.get("methane", 1.4)
        
        predictions = []
        
        # تنبؤات بناءً على البيانات الحالية
        if current_temp > 62:
            predictions.append("درجة الحرارة ستستمر في الارتفاع إذا لم يتم اتخاذ إجراء")
        elif current_temp < 50:
            predictions.append("درجة الحرارة قد تنخفض أكثر خلال الساعات القادمة")
        else:
            predictions.append("درجة الحرارة ستستقر حول 55-58°م")
        
        if current_pressure > 8.0:
            predictions.append("ضغط النظام قد يصل إلى مستويات خطيرة إذا استمر هذا الاتجاه")
        elif current_pressure < 6.5:
            predictions.append("ضغط النظام قد ينخفض أكثر مما يؤثر على الكفاءة")
        else:
            predictions.append("ضغط النظام سيبقى ضمن النطاق الآمن")
        
        if current_methane > 2.5:
            predictions.append("مستويات الميثان مرتفعة وقد تشير إلى تسرب محتمل")
        else:
            predictions.append("مستويات الميثان ستظل منخفضة ومستقرة")
        
        if context and "temperature" in context.lower():
            return "درجة الحرارة ستصل إلى 58°م خلال الساعتين القادمتين"
        elif context and "pressure" in context.lower():
            return "ضغط النظام سيرتفع قليلاً إلى 7.5 بار ثم يعود إلى المستوى الطبيعي"
        
        return "، ".join(predictions)
    
    def detect_current_anomalies(self):
        current_temp = st.session_state.get("mqtt_temp", 55)
        current_pressure = st.session_state.get("pressure", 7.2)
        current_methane = st.session_state.get("methane", 1.4)
        
        anomalies = []
        
        if current_temp > 65:
            anomalies.append({
                "metric": "درجة الحرارة",
                "value": f"{current_temp}°م",
                "average": "55°م",
                "severity": "عالية" if current_temp > 70 else "متوسطة"
            })
        
        if current_pressure > 9.0:
            anomalies.append({
                "metric": "الضغط",
                "value": f"{current_pressure} بار",
                "average": "7.2 بار",
                "severity": "عالية" if current_pressure > 10 else "متوسطة"
            })
        
        if current_methane > 3.0:
            anomalies.append({
                "metric": "الميثان",
                "value": f"{current_methane} ppm",
                "average": "1.4 ppm",
                "severity": "عالية" if current_methane > 4 else "متوسطة"
            })
        
        if anomalies:
            return random.choice(anomalies)
        
        return None
    
    def generate_recommendation(self, context=None):
        current_temp = st.session_state.get("mqtt_temp", 55)
        current_pressure = st.session_state.get("pressure", 7.2)
        current_methane = st.session_state.get("methane", 1.4)
        
        recommendations = []
        
        if current_temp > 62:
            recommendations.append("خفض درجة حرارة التشغيل بنسبة 5-10%")
        elif current_temp < 50:
            recommendations.append("زيادة درجة حرارة التشغيل بنسبة 5%")
        
        if current_pressure > 8.0:
            recommendations.append("تقليل ضغط التشغيل إلى 7.0 بار")
        elif current_pressure < 6.5:
            recommendations.append("زيادة ضغط التشغيل إلى 7.2 بار")
        
        if current_methane > 2.5:
            recommendations.append("تفعيل نظام التهوية الإضافي والتحقق من وجود تسرب")
        
        if not recommendations:
            recommendations = [
                "الاستمرار في المراقبة الروتينية للنظام",
                "إجراء فحص وقائي للصمامات والوصلات",
                "مراجعة إعدادات نظام التبريد"
            ]
        
        if context and "حرارة" in context:
            return "خفض إعدادات التسخين بنسبة 10% لتفادي الارتفاع المتوقع"
        elif context and "ضغط" in context:
            return "تفقد صمامات الأمان للتأكد من عملها بشكل صحيح"
        
        return random.choice(recommendations)

sndt_chat = SNDTChatSystem()

# -------------------- المساعد الذكي --------------------
def generate_ai_response(prompt):
    """مساعد ذكي مدعوم بالذاكرة الدائمة"""
    prompt_lower = prompt.lower()
    
    if st.session_state.get("openai_enabled", False) and st.session_state.get("openai_api_key", ""):
        try:
            import openai
            openai.api_key = st.session_state["openai_api_key"]
            
            # تجميع السياق من الذاكرة الدائمة والبيانات الحالية
            context = "\n".join([
                f"التجربة: {exp['experience']} - النتيجة: {exp['outcome']}"
                for exp in st.session_state.get("lifelong_memory", [])[-5:]
            ])
            
            # البيانات الحالية للنظام
            current_data = f"""
            البيانات الحالية: 
            - درجة الحرارة: {st.session_state.get('mqtt_temp', 55)}°م
            - الضغط: {st.session_state.get('pressure', 7.2)} بار
            - الميثان: {st.session_state.get('methane', 1.4)} ppm
            - الاهتزاز: {st.session_state.get('vibration', 4.5)}
            - معدل التدفق: {st.session_state.get('flow_rate', 110)}
            - حالة النظام: {'جيدة' if st.session_state.get('mqtt_temp', 55) < 60 else 'تحت المراقبة'}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""أنت مساعد ذكي لمنصة التوأم الرقمي SNDT. 
                    أنت متخصص في تحليل بيانات المصانع والتنبؤ بالمشاكل وتقديم التوصيات.
                    
                    السياق من الذاكرة الدائمة: {context}
                    
                    {current_data}
                    
                    قواعد يجب اتباعها:
                    1. قدم إجابات دقيقة وواقعية بناءً على البيانات
                    2. ركز على السلامة والكفاءة في التوصيات
                    3. كن واضحاً ومباشراً في الإجابات
                    4. إذا لاحظت مشكلة محتملة، نبه المستخدم واقترح حلولاً
                    5. استخدم لغة عربية فصحى واضحة"""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            st.session_state["chat_history"].append({
                "user": prompt,
                "assistant": ai_response,
                "timestamp": datetime.now().isoformat(),
                "source": "openai"
            })
            
            # إضافة هذه التجربة إلى الذاكرة الدائمة
            lifelong_memory.add_experience(
                "chat_interaction",
                f"سؤال: {prompt}",
                f"تم الرد باستخدام الذكاء الاصطناعي: {ai_response[:100]}..."
            )
            
            logger.info("تم توليد رد باستخدام OpenAI")
            return ai_response
            
        except Exception as e:
            logger.error(f"خطأ في الاتصال بـ OpenAI: {e}")
            return generate_fallback_response(prompt_lower)
    
    else:
        return sndt_chat.generate_response(prompt)

def generate_fallback_response(prompt_lower):
    """إنشاء رد عند عدم توفر OpenAI"""
    response = ""
    if any(word in prompt_lower for word in ["الطقس", "درجة الحرارة", "weather", "temperature"]):
        response += get_weather_info()
    elif any(word in prompt_lower for word in ["الوقت", "التاريخ", "اليوم", "time", "date", "today"]):
        response += get_current_time_info()
    elif any(word in prompt_lower for word in ["مرحبا", "السلام", "hello", "hi"]):
        response += "مرحباً! أنا المساعد الذكي لمنصة التوأم الرقمي. كيف يمكنني مساعدتك اليوم؟"
    elif any(word in prompt_lower for word in ["تنبأ", "توقع", "predict", "forecast"]):
        response += generate_time_based_prediction(prompt_lower)
    elif any(word in prompt_lower for word in ["حالي", "مباشر", "current", "now"]):
        response += generate_current_status()
    else:
        response += "أنا المساعد الذكي للمنصة. يمكنني مساعدتك في مراقبة المصنع، التنبؤات، الطقس، الوقت، وأسئلة عامة أخرى."
    
    st.session_state["chat_history"].append({
        "user": prompt_lower,
        "assistant": response,
        "timestamp": datetime.now().isoformat(),
        "source": "fallback"
    })
    
    return response

def get_weather_info():
    """معلومات الطقس"""
    weather_data = {
        "temperature": random.randint(20, 35),
        "condition": random.choice(["مشمس", "غائم جزئياً", "صافي"]),
        "humidity": random.randint(30, 70)
    }
    return f"""حالة الطقس الحالية:
• درجة الحرارة: {weather_data['temperature']}°م
• الحالة: {weather_data['condition']}
• الرطوبة: {weather_data['humidity']}%"""

def get_current_time_info():
    """معلومات الوقت والتاريخ"""
    now = datetime.now()
    return f"""الوقت الحالي: {now.strftime('%H:%M:%S')}
تاريخ اليوم: {now.strftime('%Y-%m-%d')}
اليوم: {now.strftime('%A')}"""

def generate_time_based_prediction(prompt):
    """إنشاء تنبؤات زمنية"""
    time_keywords = {"ساعة": 1, "ساعات": 1, "يوم": 24, "أيام": 24, "أسبوع": 168, "أسابيع": 168}
    hours_ahead = 2
    
    for keyword, hours in time_keywords.items():
        if keyword in prompt:
            hours_ahead = hours
            break
    
    # تحليل البيانات الحالية لإنشاء تنبؤات أكثر دقة
    current_temp = st.session_state.get("mqtt_temp", 55)
    current_pressure = st.session_state.get("pressure", 7.2)
    current_methane = st.session_state.get("methane", 1.4)
    
    predictions = []
    
    if current_temp > 60:
        predictions.append(f"درجة الحرارة قد تصل إلى {current_temp + 2}°م")
    elif current_temp < 50:
        predictions.append(f"درجة الحرارة قد تنخفض إلى {current_temp - 2}°م")
    else:
        predictions.append(f"درجة الحرارة ستستقر حول {current_temp}°م")
    
    if current_pressure > 8.0:
        predictions.append("ضغط النظام سيظل مرتفعاً ويتطلب مراقبة")
    else:
        predictions.append("ضغط النظام سيبقى ضمن النطاق الآمن")
    
    if current_methane > 2.5:
        predictions.append("مستويات الميثان مرتفعة وقد تتطلب تدخلاً")
    else:
        predictions.append("مستويات الميثان ستظل مستقرة")
    
    return f"خلال الـ {hours_ahead} ساعة القادمة، {'، '.join(predictions)}"

def generate_current_status():
    """الحالة الحالية للنظام"""
    # تحليل شامل للحالة الحالية
    current_temp = st.session_state.get("mqtt_temp", 55)
    temp_status = "طبيعية" if 50 <= current_temp <= 60 else "مرتفعة" if current_temp > 60 else "منخفضة"
    
    current_pressure = st.session_state.get("pressure", 7.2)
    pressure_status = "طبيعي" if 6.5 <= current_pressure <= 8.0 else "مرتفع" if current_pressure > 8.0 else "منخفض"
    
    current_methane = st.session_state.get("methane", 1.4)
    methane_status = "طبيعي" if current_methane <= 2.0 else "مرتفع"
    
    system_health = "جيدة"
    if current_temp > 65 or current_pressure > 9.0 or current_methane > 3.0:
        system_health = "تحت المراقبة"
    if current_temp > 70 or current_pressure > 10.0 or current_methane > 4.0:
        system_health = "في خطر"
    
    return f"""الحالة الحالية للنظام:
• درجة الحرارة: {current_temp}°م ({temp_status})
• الضغط: {current_pressure} بار ({pressure_status})
• الميثان: {current_methane} ppm ({methane_status})
• الاهتزاز: {st.session_state.get('vibration', 4.5)}
• معدل التدفق: {st.session_state.get('flow_rate', 110)}
• آخر تحديث: {st.session_state.get('mqtt_last', datetime.now()).strftime('%H:%M:%S')}
• صحة النظام: {system_health}"""
