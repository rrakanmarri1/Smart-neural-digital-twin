import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# استيراد مكتبات preprocessing المتقدمة
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
    from sklearn.impute import KNNImputer
    from sklearn.feature_selection import mutual_info_regression
    from scipy import stats
    HAS_ADVANCED_PREPROCESSING = True
except ImportError:
    HAS_ADVANCED_PREPROCESSING = False

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_data(data)
    
    def preprocess_realtime_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_realtime_data(data)
        
class DataQualityLevel(Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    ENGINEERED = "engineered"
    NORMALIZED = "normalized"
    READY_FOR_TRAINING = "ready"

@dataclass
class DataQualityReport:
    timestamp: datetime
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    missing_values: int
    outliers_removed: int
    quality_level: DataQualityLevel
    processing_time_seconds: float
    features_added: List[str]
    features_removed: List[str]

class AdvancedDataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        self.quality_reports: List[DataQualityReport] = []
        self.setup_logging()
        
    def setup_logging(self):
        """تهيئة نظام التسجيل"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess(self, filepath: str, 
                          target_columns: List[str] = None) -> pd.DataFrame:
        """تحميل ومعالجة البيانات بشكل كامل"""
        start_time = datetime.now()
        
        # 1. تحميل البيانات
        self.logger.info(f"📂 Loading data from {filepath}")
        raw_df = self._load_data(filepath)
        
        # 2. التحقق من الجودة الأولية
        initial_report = self._generate_quality_report(raw_df, DataQualityLevel.RAW)
        
        # 3. التنظيف الأساسي
        cleaned_df = self._clean_data(raw_df)
        
        # 4. معالجة القيم المفقودة
        imputed_df = self._handle_missing_values(cleaned_df)
        
        # 5. إضافة Features هندسية
        engineered_df = self._feature_engineering(imputed_df, target_columns)
        
        # 6. معالجة القيم المتطرفة
        processed_df = self._handle_outliers(engineered_df)
        
        # 7. التطبيع
        normalized_df = self._normalize_data(processed_df)
        
        # 8. تقرير الجودة النهائي
        processing_time = (datetime.now() - start_time).total_seconds()
        final_report = DataQualityReport(
            timestamp=datetime.now(),
            original_shape=raw_df.shape,
            final_shape=normalized_df.shape,
            missing_values=raw_df.isnull().sum().sum(),
            outliers_removed=len(raw_df) - len(processed_df),
            quality_level=DataQualityLevel.READY_FOR_TRAINING,
            processing_time_seconds=processing_time,
            features_added=list(set(normalized_df.columns) - set(raw_df.columns)),
            features_removed=list(set(raw_df.columns) - set(normalized_df.columns))
        )
        
        self.quality_reports.append(final_report)
        self.logger.info(f"✅ Data preprocessing completed in {processing_time:.2f}s")
        
        return normalized_df
    
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """تحميل البيانات مع معالجة الأخطاء"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif filepath.endswith('.feather'):
                df = pd.read_feather(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # التحقق من وجود بيانات
            if df.empty:
                raise ValueError("Loaded dataframe is empty")
                
            self.logger.info(f"📊 Loaded data shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """تنظيف البيانات الأساسي"""
        cleaned_df = df.copy()
        
        # 1. إصلاح أسماء الأعمدة
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
        
        # 2. تحويل التواريخ
        date_columns = [col for col in cleaned_df.columns if 'date' in col or 'time' in col]
        for col in date_columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
            except:
                self.logger.warning(f"Could not parse date column: {col}")
        
        # 3. إزالة الأعمدة غير المرغوبة
        columns_to_drop = self.config.get('columns_to_drop', [])
        cleaned_df = cleaned_df.drop(columns=columns_to_drop, errors='ignore')
        
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المفقودة بذكاء"""
        imputed_df = df.copy()
        missing_percentage = imputed_df.isnull().mean()
        
        for column in imputed_df.columns:
            if imputed_df[column].isnull().any():
                missing_pct = missing_percentage[column]
                
                if missing_pct > 0.3:  # إذا كانت النسبة عالية
                    # حذف العمود إذا كانت القيم المفقودة أكثر من 30%
                    if missing_pct > 0.3:
                        imputed_df = imputed_df.drop(columns=[column])
                        self.logger.info(f"🗑️ Dropped column {column} ({missing_pct:.1%} missing)")
                
                elif missing_pct > 0.05 and HAS_ADVANCED_PREPROCESSING:
                    # استخدام KNN Imputer للقيم المفقودة
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        imputed_df[column] = imputer.fit_transform(imputed_df[[column]]).flatten()
                        self.imputers[column] = imputer
                    except Exception as e:
                        self.logger.warning(f"KNN imputation failed for {column}: {e}")
                        imputed_df[column] = imputed_df[column].fillna(imputed_df[column].median())
                
                else:
                    # تعويض بسيط
                    if imputed_df[column].dtype in ['float64', 'int64']:
                        imputed_df[column] = imputed_df[column].fillna(imputed_df[column].median())
                    else:
                        imputed_df[column] = imputed_df[column].fillna(imputed_df[column].mode()[0])
        
        return imputed_df
    
    def _feature_engineering(self, df: pd.DataFrame, target_columns: List[str] = None) -> pd.DataFrame:
        """هندسة الميزات المتقدمة"""
        engineered_df = df.copy()
        
        # 1. إضافة ميزات زمنية
        datetime_columns = [col for col in engineered_df.columns if engineered_df[col].dtype == 'datetime64[ns]']
        for col in datetime_columns:
            engineered_df[f'{col}_hour'] = engineered_df[col].dt.hour
            engineered_df[f'{col}_dayofweek'] = engineered_df[col].dt.dayofweek
            engineered_df[f'{col}_month'] = engineered_df[col].dt.month
            engineered_df[f'{col}_quarter'] = engineered_df[col].dt.quarter
        
        # 2. إضافة ميزات إحصائية
        numeric_columns = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            # Rolling statistics
            engineered_df[f'{col}_rolling_mean_6'] = engineered_df[col].rolling(6).mean()
            engineered_df[f'{col}_rolling_std_6'] = engineered_df[col].rolling(6).std()
            
            # Difference features
            engineered_df[f'{col}_diff_1'] = engineered_df[col].diff()
            engineered_df[f'{col}_pct_change'] = engineered_df[col].pct_change()
        
        # 3. إضافة ميزات التفاعل
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    engineered_df[f'{col1}_{col2}_ratio'] = engineered_df[col1] / (engineered_df[col2] + 1e-8)
                    engineered_df[f'{col1}_{col2}_product'] = engineered_df[col1] * engineered_df[col2]
        
        # 4. حساب أهمية الميزات إذا كان هناك target
        if target_columns and HAS_ADVANCED_PREPROCESSING:
            self._calculate_feature_importance(engineered_df, target_columns)
        
        # تنظيف القيم اللانهائية
        engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
        engineered_df = engineered_df.fillna(0)
        
        return engineered_df
    
    def _calculate_feature_importance(self, df: pd.DataFrame, target_columns: List[str]):
        """حساب أهمية الميزات"""
        try:
            X = df.drop(columns=target_columns).select_dtypes(include=[np.number])
            y = df[target_columns[0]]  # أول target فقط
            
            # حساب Mutual Information
            mi_scores = mutual_info_regression(X, y)
            self.feature_importance = dict(zip(X.columns, mi_scores))
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المتطرفة بذكاء"""
        processed_df = df.copy()
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # استخدام IQR مع حدود قابلة للتعديل
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 2.5 * IQR  # ⬆️ زيادة التسامح قليلاً
            upper_bound = Q3 + 2.5 * IQR
            
            # استبدال القيم المتطرفة بحدود IQR
            processed_df[col] = np.where(processed_df[col] < lower_bound, lower_bound, processed_df[col])
            processed_df[col] = np.where(processed_df[col] > upper_bound, upper_bound, processed_df[col])
        
        return processed_df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """تطبيع البيانات بشكل انتقائي"""
        normalized_df = df.copy()
        numeric_columns = normalized_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # اختيار طريقة التطبيع المناسبة
            if self._is_normal_distribution(normalized_df[col]):
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()  # أفضل للبيانات غير الطبيعية
            
            normalized_df[col] = scaler.fit_transform(normalized_df[[col]].values.reshape(-1, 1))
            self.scalers[col] = scaler
        
        return normalized_df
    
    def _is_normal_distribution(self, series: pd.Series) -> bool:
        """التحقق إذا كانت البيانات تتبع توزيعاً طبيعياً"""
        try:
            stat, p_value = stats.normaltest(series.dropna())
            return p_value > 0.05  # إذا كانت p-value > 0.05 فالبيانات طبيعية
        except:
            return False
    
    def _generate_quality_report(self, df: pd.DataFrame, quality_level: DataQualityLevel) -> DataQualityReport:
        """توليد تقرير جودة البيانات"""
        return DataQualityReport(
            timestamp=datetime.now(),
            original_shape=df.shape,
            final_shape=df.shape,
            missing_values=df.isnull().sum().sum(),
            outliers_removed=0,
            quality_level=quality_level,
            processing_time_seconds=0,
            features_added=[],
            features_removed=[]
        )
    
    def get_quality_report(self) -> Optional[DataQualityReport]:
        """الحصول على آخر تقرير جودة"""
        if self.quality_reports:
            return self.quality_reports[-1]
        return None
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """حفظ البيانات المعالجة"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix == '.parquet':
                df.to_parquet(output_path, index=False)
            elif output_path.suffix == '.feather':
                df.to_feather(output_path)
            
            self.logger.info(f"💾 Saved processed data to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving processed data: {e}")
            return False

# دالة مساعدة
def create_data_preprocessor(config: Dict[str, Any]) -> AdvancedDataPreprocessor:
    return AdvancedDataPreprocessor(config)
    
