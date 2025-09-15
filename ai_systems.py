import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from enum import Enum

class ModelType(Enum):
    """أنواع النماذج المتاحة للاختيار الديناميكي"""
    LSTM = "lstm"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    MONTE_CARLO = "monte_carlo"
    GRADIENT_BOOSTING = "gradient_boosting"
    LIGHT_GBM = "light_gbm"

class ModelPerformance:
    """تتبع أداء النماذج للاختيار الذكي"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.accuracy_history: List[float] = []
        self.latency_history: List[float] = []
        self.memory_usage_history: List[float] = []
        self.last_used: Optional[datetime] = None
        self.success_count = 0
        self.failure_count = 0
    
    def update_performance(self, accuracy: float, latency: float, 
                         memory_usage: float, success: bool):
        """تحديث أداء النموذج"""
        self.accuracy_history.append(accuracy)
        self.latency_history.append(latency)
        self.memory_usage_history.append(memory_usage)
        self.last_used = datetime.now()
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # الاحتفاظ فقط بـ 100 قياس حديثة
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
            self.latency_history = self.latency_history[-100:]
            self.memory_usage_history = self.memory_usage_history[-100:]
    
    def get_performance_score(self) -> float:
        """حساب درجة أداء النموذج"""
        if not self.accuracy_history:
            return 0.0
        
        # حساب المتوسط المرجح
        accuracy = np.mean(self.accuracy_history[-10:]) if self.accuracy_history else 0.0
        latency = np.mean(self.latency_history[-10:]) if self.latency_history else 1.0
        memory = np.mean(self.memory_usage_history[-10:]) if self.memory_usage_history else 1.0
        
        # تحويل الكمون والذاكرة إلى عوامل تصحيح (كلما قلوا كلما كان أفضل)
        latency_factor = 1.0 / max(0.1, latency)
        memory_factor = 1.0 / max(0.1, memory)
        
        # حساب الدرجة النهائية
        score = accuracy * 0.6 + latency_factor * 0.2 + memory_factor * 0.2
        return max(0.0, min(1.0, score))

class DynamicModelSelector:
    """
    نظام الاختيار الديناميكي للنماذج - ميزة براءة الاختراع
    
    يختار تلقائياً أفضل نموذج بناءً على:
    - أداء النموذج التاريخي
    - قيود موارد النظام الحالية
    - خصائص البيانات المدخلة
    - متطلبات المهمة
    
    يعمل بدون أي تدخل بشري ويحسن نفسه باستمرار
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_models: Dict[ModelType, Any] = {}
        self.model_performance: Dict[ModelType, ModelPerformance] = {}
        self.current_model: Optional[ModelType] = None
        self.logger = logging.getLogger(__name__)
        
        self.initialize_models()
    
    def initialize_models(self):
        """تهيئة جميع النماذج المتاحة"""
        for model_type in ModelType:
            self.model_performance[model_type] = ModelPerformance(model_type)
        
        self.logger.info("✅ Dynamic Model Selector initialized")
    
    def register_model(self, model_type: ModelType, model_instance: Any):
        """تسجيل نموذج جديد في النظام"""
        self.available_models[model_type] = model_instance
        self.logger.info(f"✅ Registered model: {model_type.value}")
    
    def select_best_model(self, data_characteristics: Dict[str, Any], 
                        resource_constraints: Dict[str, float]) -> ModelType:
        """
        اختيار أفضل نموذج بناءً على البيانات والموارد
        
        Args:
            data_characteristics: خصائص البيانات المدخلة
            resource_constraints: قيود الموارد الحالية (CPU, RAM, etc.)
            
        Returns:
            ModelType: أفضل نموذج للاستخدام
        """
        # تحليل خصائص البيانات
        data_complexity = self._analyze_data_complexity(data_characteristics)
        data_size = data_characteristics.get('size', 0)
        
        # تحليل قيود الموارد
        cpu_available = resource_constraints.get('cpu_available', 1.0)
        memory_available = resource_constraints.get('memory_available', 1.0)
        
        # حساب درجات الملائمة لكل نموذج
        model_scores = {}
        
        for model_type, performance in self.model_performance.items():
            if model_type not in self.available_models:
                continue
            
            # درجة الأداء التاريخي
            performance_score = performance.get_performance_score()
            
            # درجة الملائمة للبيانات
            suitability_score = self._calculate_model_suitability(
                model_type, data_complexity, data_size
            )
            
            # درجة كفاءة الموارد
            efficiency_score = self._calculate_resource_efficiency(
                model_type, cpu_available, memory_available
            )
            
            # الدرجة النهائية (مرجحة)
            final_score = (
                performance_score * 0.4 +
                suitability_score * 0.3 +
                efficiency_score * 0.3
            )
            
            model_scores[model_type] = final_score
        
        if not model_scores:
            self.logger.warning("No models available, using default")
            return ModelType.LSTM
        
        # اختيار النموذج بأعلى درجة
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        self.current_model = best_model
        
        self.logger.info(f"🎯 Selected model: {best_model.value} (score: {model_scores[best_model]:.3f})")
        return best_model
    
    def _analyze_data_complexity(self, data_characteristics: Dict[str, Any]) -> float:
        """تحليل تعقيد البيانات"""
        size = data_characteristics.get('size', 0)
        dimensions = data_characteristics.get('dimensions', 0)
        variability = data_characteristics.get('variability', 0)
        
        # حساب درجة التعقيد (0-1)
        complexity = min(1.0, (size / 10000 + dimensions / 10 + variability) / 3)
        return complexity
    
    def _calculate_model_suitability(self, model_type: ModelType, 
                                   data_complexity: float, data_size: int) -> float:
        """حساب ملائمة النموذج للبيانات"""
        suitability_scores = {
            ModelType.LSTM: min(1.0, 0.8 + data_complexity * 0.2),  # جيد للبيانات المعقدة
            ModelType.ISOLATION_FOREST: 0.9 if data_size > 1000 else 0.6,
            ModelType.ONE_CLASS_SVM: 0.7 + data_complexity * 0.3,
            ModelType.MONTE_CARLO: 0.8 if data_complexity < 0.5 else 0.6,
            ModelType.GRADIENT_BOOSTING: 0.9 if data_size > 500 else 0.7,
            ModelType.LIGHT_GBM: 0.85 if data_size > 1000 else 0.65
        }
        
        return suitability_scores.get(model_type, 0.5)
    
    def _calculate_resource_efficiency(self, model_type: ModelType,
                                    cpu_available: float, memory_available: float) -> float:
        """حساب كفاءة النموذج في استخدام الموارد"""
        # احتياجات الموارد التقريبية لكل نموذج
        resource_needs = {
            ModelType.LSTM: {'cpu': 0.8, 'memory': 0.7},
            ModelType.ISOLATION_FOREST: {'cpu': 0.5, 'memory': 0.4},
            ModelType.ONE_CLASS_SVM: {'cpu': 0.6, 'memory': 0.5},
            ModelType.MONTE_CARLO: {'cpu': 0.9, 'memory': 0.6},
            ModelType.GRADIENT_BOOSTING: {'cpu': 0.7, 'memory': 0.5},
            ModelType.LIGHT_GBM: {'cpu': 0.6, 'memory': 0.4}
        }
        
        needs = resource_needs.get(model_type, {'cpu': 0.7, 'memory': 0.5})
        
        # حساب الكفاءة (كلما قل الاستخدام كلما كانت الكفاءة أعلى)
        cpu_efficiency = 1.0 - max(0, needs['cpu'] - cpu_available)
        memory_efficiency = 1.0 - max(0, needs['memory'] - memory_available)
        
        return (cpu_efficiency + memory_efficiency) / 2
    
    def update_model_performance(self, model_type: ModelType, accuracy: float,
                               latency: float, memory_usage: float, success: bool):
        """تحديث أداء النموذج بعد الاستخدام"""
        if model_type in self.model_performance:
            self.model_performance[model_type].update_performance(
                accuracy, latency, memory_usage, success
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """الحصول على تقرير أداء النماذج"""
        report = {}
        for model_type, performance in self.model_performance.items():
            report[model_type.value] = {
                'performance_score': performance.get_performance_score(),
                'success_rate': performance.success_count / max(1, performance.success_count + performance.failure_count),
                'last_used': performance.last_used.isoformat() if performance.last_used else None,
                'usage_count': performance.success_count + performance.failure_count
            }
        return report
    
        def auto_optimize(self):
        """التحسين التلقائي للنماذج بناءً على الأداء"""
        try:
            # إعادة تدريب النماذج ذات الأداء المنخفض
            for model_type, performance in self.model_performance.items():
                if performance.get_performance_score() < 0.6:
                    self._retrain_model(model_type)
            
            # تحميل نماذج جديدة إذا كانت الموارد تسمح
            if self._check_resource_availability():
                self._load_additional_models()
                
            self.logger.info("✅ Model auto-optimization completed")
            
        except Exception as e:
            self.logger.error(f"❌ Auto-optimization failed: {e}")
    
    def _retrain_model(self, model_type: ModelType):
        """إعادة تدريب النموذج"""
        if model_type in self.available_models:
            try:
                model = self.available_models[model_type]
                if hasattr(model, 'retrain'):
                    model.retrain()
                    self.logger.info(f"✅ Retrained model: {model_type.value}")
            except Exception as e:
                self.logger.error(f"❌ Failed to retrain {model_type.value}: {e}")
    
    def _check_resource_availability(self) -> bool:
        """التحقق من توفر الموارد لتحميل نماذج إضافية"""
        # محاكاة التحقق من الموارد
        return True  # في الواقع سيتم التحقق من CPU و RAM
    
    def _load_additional_models(self):
        """تحميل نماذج إضافية إذا كانت الموارد تسمح"""
        # يمكن إضافة نماذج جديدة هنا
        pass

# دالة مساعدة
def create_model_selector(config: Dict[str, Any]) -> DynamicModelSelector:
    return DynamicModelSelector(config)
