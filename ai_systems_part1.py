from __future__ import annotations

import json
import logging
import os
import pickle
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------------------
# Logging Setup (module-level logger)
# -------------------------------------------------------------------------------------
logger = logging.getLogger("SmartNeural.AI.AnomalyCore")
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------------------
# Enumerations & Data Models
# -------------------------------------------------------------------------------------

class DetectorType(Enum):
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOF = "local_outlier_factor"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble_fusion"


@dataclass
class DataQualityIssue:
    feature: str
    issue_type: str
    description: str
    severity: str = "INFO"


@dataclass
class DataQualityReport:
    issues: List[DataQualityIssue]
    total_features: int
    outlier_features: int
    invalid_range_features: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issues": [asdict(i) for i in self.issues],
            "total_features": self.total_features,
            "outlier_features": self.outlier_features,
            "invalid_range_features": self.invalid_range_features
        }


@dataclass
class AnomalyModelMetadata:
    detector: DetectorType
    trained_at: datetime
    version: str
    params: Dict[str, Any]
    calibration_reference: Dict[str, Any]


@dataclass
class AnomalyDetectionResult:
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    adaptive_threshold: float
    fused_details: Dict[str, Any]
    data_quality: Dict[str, Any]
    critical_anomalies: List[Dict[str, Any]]
    temporal_context: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "risk_level": self.risk_level,
            "adaptive_threshold": self.adaptive_threshold,
            "fused_details": self.fused_details,
            "data_quality": self.data_quality,
            "critical_anomalies": self.critical_anomalies,
            "temporal_context": self.temporal_context,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


# -------------------------------------------------------------------------------------
# Sensor Feature Mapping
# -------------------------------------------------------------------------------------

class SensorFeatureMapper:
    """
    Maintains canonical ordering & importance weighting for sensor features.
    Supports safe evolution (adding new sensors won't reorder existing).
    Enhanced to support dynamic feature sets with backward compatibility.
    """

    def __init__(self, ordered_features: List[str]):
        if len(ordered_features) != len(set(ordered_features)):
            raise ValueError("Duplicate feature names detected in mapper initialization.")
        self.feature_names: List[str] = list(ordered_features)
        self.feature_index: Dict[str, int] = {f: i for i, f in enumerate(self.feature_names)}
        self.importances: Dict[str, float] = {f: 1.0 for f in self.feature_names}
        self._original_feature_set: Set[str] = set(ordered_features)
        self._features_modified = False

    def ordered_vector(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert sensor data dictionary to ordered feature vector
        Works with any sensor data, handling missing or extra features safely
        """
        return np.array([float(sensor_data.get(f, 0.0)) for f in self.feature_names], dtype=float)

    def update_features(self, new_features: List[str]) -> bool:
        """
        Update feature list while preserving original ordering where possible
        Returns True if features were modified
        """
        if not new_features:
            return False
            
        # If identical, no change needed
        if set(new_features) == set(self.feature_names):
            return False
            
        # Check for new features
        current_set = set(self.feature_names)
        new_set = set(new_features)
        
        # Find truly new features (weren't in the original set)
        added_features = new_set - current_set
        
        if not added_features:
            return False
            
        # Preserve original ordering and append new features
        updated_features = list(self.feature_names)
        for feature in new_features:
            if feature not in current_set:
                updated_features.append(feature)
                
        # Update our internal state
        self.feature_names = updated_features
        self.feature_index = {f: i for i, f in enumerate(self.feature_names)}
        
        # Add default importance for new features
        for f in added_features:
            self.importances[f] = 1.0
            
        self._features_modified = True
        logger.info(f"Feature set updated, added {len(added_features)} new features: {', '.join(added_features)}")
        return True

    def check_compatibility(self, feature_set: Set[str]) -> Tuple[bool, Set[str], Set[str]]:
        """
        Check compatibility between current feature set and provided feature set
        Returns (is_compatible, missing_features, new_features)
        """
        current_set = set(self.feature_names)
        missing_features = current_set - feature_set
        new_features = feature_set - current_set
        
        # Simple heuristic: if missing > 25% of current, considered incompatible
        is_compatible = len(missing_features) <= len(current_set) * 0.25
        
        return is_compatible, missing_features, new_features

    def align_data_for_model(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw sensor data to align with the feature set expected by the model
        Handles missing features by using 0.0 as default
        """
        # Only process if we need to - if the mapper has been modified
        if not self._features_modified:
            return sensor_data
            
        aligned_data = {}
        
        # Copy all features that the model expects
        for feature in self.feature_names:
            aligned_data[feature] = sensor_data.get(feature, 0.0)
            
        return aligned_data

    def apply_weighting(self, vector: np.ndarray) -> np.ndarray:
        weights = np.array([self.importances.get(f, 1.0) for f in self.feature_names], dtype=float)
        return vector * weights

    def update_importances(self, new_importances: Dict[str, float]):
        for k, v in new_importances.items():
            if k in self.importances and v > 0:
                self.importances[k] = float(v)

    def summary(self) -> Dict[str, float]:
        return dict(self.importances)
        
    def features_modified(self) -> bool:
        """Check if features have been modified since initialization"""
        return self._features_modified


# -------------------------------------------------------------------------------------
# Preprocessor
# -------------------------------------------------------------------------------------

class AdvancedPreprocessor:
    """
    Responsible for:
        * Feature ordering
        * Multi-scaler fitting (standard, minmax, robust) for different downstream consumers
        * Statistical profiling for anomaly calibration & data quality
        * Handling dynamic feature sets and sensor configurations
    """

    def __init__(self, config: Dict[str, Any], feature_order: Optional[List[str]] = None):
        self.config = config
        feature_list = feature_order or list(config.get("sensors", {}).keys())
        self.mapper = SensorFeatureMapper(feature_list)
        self.scalers: Dict[str, Any] = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        self.fitted: bool = False
        self.stats: Dict[str, np.ndarray] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.random_seed: Optional[int] = None
        self.needs_retraining: bool = False
        self.last_feature_update: Optional[datetime] = None

    def set_seed(self, seed: int):
        self.random_seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def update_feature_set(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Check if sensor data contains new features and update mapper if needed
        Returns True if features were updated
        """
        current_features = set(self.mapper.feature_names)
        new_features = set(sensor_data.keys())
        
        # Check for truly new features
        added_features = new_features - current_features
        
        if not added_features:
            return False
            
        # Update the feature mapper with all available features
        updated = self.mapper.update_features(list(new_features))
        
        if updated:
            self.needs_retraining = True
            self.last_feature_update = datetime.utcnow()
            logger.info(f"Feature set updated with {len(added_features)} new sensors, retraining recommended")
            
        return updated

    def fit(self, data_records: List[Dict[str, Any]]):
        if not data_records:
            raise ValueError("No data provided for preprocessor fitting.")
            
        # First check if we need to update our feature set based on the data
        all_features = set()
        for record in data_records:
            all_features.update(record.keys())
            
        self.mapper.update_features(list(all_features))
        
        # Now process the data with our updated feature set
        matrix = np.vstack([self.mapper.ordered_vector(r) for r in data_records])
        
        # Fit scalers
        for s in self.scalers.values():
            s.fit(matrix)
            
        # Compute statistics
        self.stats = {
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0) + 1e-8,  # avoid division by zero
            "median": np.median(matrix, axis=0),
            "q1": np.percentile(matrix, 25, axis=0),
            "q3": np.percentile(matrix, 75, axis=0),
            "min": matrix.min(axis=0),
            "max": matrix.max(axis=0),
            "skew": stats.skew(matrix, axis=0),
            "kurtosis": stats.kurtosis(matrix, axis=0)
        }
        
        # Correlations
        df = pd.DataFrame(matrix, columns=self.mapper.feature_names)
        self.correlation_matrix = df.corr()
        self.fitted = True
        self.needs_retraining = False
        logger.info(f"Preprocessor fitted successfully on {matrix.shape[0]} records with {matrix.shape[1]} features.")

    def transform(self, sensor_data: Dict[str, Any], scaler: str = "minmax") -> np.ndarray:
        """
        Transform sensor data using specified scaler
        Handles potential feature mismatches with dynamic sensor sets
        """
        if scaler not in self.scalers:
            raise ValueError(f"Scaler '{scaler}' not available.")
            
        # Check for feature updates
        self.update_feature_set(sensor_data)
        
        # Align data with the feature set expected by the mapper
        aligned_data = self.mapper.align_data_for_model(sensor_data)
        
        # Get the feature vector
        vec = self.mapper.ordered_vector(aligned_data).reshape(1, -1)
        
        if not self.fitted:
            # Light fallback normalization (bounded scaling)
            max_ranges = np.array([
                self.config["sensors"].get(f, {}).get("max", 1.0) or 1.0
                for f in self.mapper.feature_names
            ], dtype=float)
            return np.clip(vec / max_ranges, 0, 1)
        
        # Handle feature dimension mismatch with fitted scaler
        expected_features = self.scalers[scaler].n_features_in_
        if vec.shape[1] != expected_features:
            logger.warning(f"Feature count mismatch: model expects {expected_features}, got {vec.shape[1]}")
            if vec.shape[1] > expected_features:
                # Truncate extra features
                vec = vec[:, :expected_features]
            else:
                # Pad with zeros (this shouldn't happen with our feature alignment)
                padding = np.zeros((1, expected_features - vec.shape[1]))
                vec = np.hstack([vec, padding])
        
        # Apply the transformation
        return self.scalers[scaler].transform(vec)

    def detect_quality(self, sensor_data: Dict[str, Any]) -> DataQualityReport:
        """
        Detect quality issues in sensor data
        Enhanced to handle dynamic feature sets
        """
        # Update feature set if needed
        self.update_feature_set(sensor_data)
        
        # Align data with expected features
        aligned_data = self.mapper.align_data_for_model(sensor_data)
        
        issues: List[DataQualityIssue] = []
        vector = self.mapper.ordered_vector(aligned_data)
        outlier_count = 0
        invalid_range_count = 0

        for idx, feature in enumerate(self.mapper.feature_names):
            # Only check features actually present in the data
            if feature not in sensor_data:
                continue
                
            cfg = self.config["sensors"].get(feature, {})
            v = vector[idx]
            min_v = cfg.get("min", -np.inf)
            max_v = cfg.get("max", np.inf)

            # Range check
            if not (min_v <= v <= max_v):
                invalid_range_count += 1
                issues.append(
                    DataQualityIssue(
                        feature=feature,
                        issue_type="RANGE_VIOLATION",
                        description=f"value {v} outside [{min_v}, {max_v}]",
                        severity="WARNING"
                    )
                )

            # Statistical (only if fitted)
            if self.fitted and idx < len(self.stats["mean"]):
                mu = self.stats["mean"][idx]
                sigma = self.stats["std"][idx]
                z = abs(v - mu) / sigma
                if z > 3.5:  # fairly strict
                    outlier_count += 1
                    issues.append(
                        DataQualityIssue(
                            feature=feature,
                            issue_type="OUTLIER",
                            description=f"z-score={z:.2f} (>3.5)",
                            severity="INFO"
                        )
                    )

        return DataQualityReport(
            issues=issues,
            total_features=len([f for f in self.mapper.feature_names if f in sensor_data]),
            outlier_features=outlier_count,
            invalid_range_features=invalid_range_count
        )


# -------------------------------------------------------------------------------------
# Autoencoder (Tabular Reconstruction)
# -------------------------------------------------------------------------------------

class TabularAutoencoder(nn.Module):
    """
    Symmetric feedforward autoencoder for compact latent reconstruction anomaly scoring.
    - Non-linear latent representation (Tanh bottleneck)
    - Optional dropout & batch normalization
    - Final linear output to allow unconstrained reconstruction (L2 loss)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.2
    ):
        super().__init__()
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        enc_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# -------------------------------------------------------------------------------------
# Ensemble Anomaly Detector
# -------------------------------------------------------------------------------------

class EnsembleAnomalyDetector:
    """
    Wraps classical detectors:
      * Isolation Forest
      * One-Class SVM
      * Local Outlier Factor (optional novelty scoring)
    Provides unified scoring interface. Each model returns binary anomaly decision.
    Enhanced with graceful handling of feature dimension changes.
    """

    def __init__(self, random_state: int = 42, enable_lof: bool = True):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.enable_lof = enable_lof
        self.feature_dimensions = 0
        self._init_models()

    def _init_models(self):
        self.models["isolation_forest"] = IsolationForest(
            n_estimators=256,
            contamination=0.08,  # tune defensively
            max_features=1.0,
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models["one_class_svm"] = OneClassSVM(
            nu=0.05, kernel="rbf", gamma="scale"
        )
        if self.enable_lof:
            # LOF: novelty=True allows calling .predict on unseen data
            self.models["lof"] = LocalOutlierFactor(
                n_neighbors=25,
                contamination=0.1,
                novelty=True
            )

    def fit(self, X: np.ndarray):
        """
        Fit all models in the ensemble to the data
        Store feature dimensions for future compatibility checks
        """
        self.feature_dimensions = X.shape[1]
        for name, model in self.models.items():
            model.fit(X)

    def predict(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Returns per-model dict:
          { model_name: { 'decision': 0/1, 'score': float } }
        decision: 1 = anomaly, 0 = normal
        
        Enhanced to handle feature dimension mismatches safely
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        # Handle feature dimension mismatch
        if self.feature_dimensions > 0 and X.shape[1] != self.feature_dimensions:
            if X.shape[1] > self.feature_dimensions:
                # Truncate extra features
                X = X[:, :self.feature_dimensions]
                logger.warning(f"Input features truncated from {X.shape[1]} to {self.feature_dimensions}")
            else:
                # Pad with zeros
                padding = np.zeros((X.shape[0], self.feature_dimensions - X.shape[1]))
                X = np.hstack([X, padding])
                logger.warning(f"Input features padded from {X.shape[1] - padding.shape[1]} to {self.feature_dimensions}")
        
        for name, model in self.models.items():
            try:
                prediction = model.predict(X)  # -1 for anomaly in sklearn conventions
                is_anom = (prediction == -1).astype(int)
                # Decision function or negative_outlier_factor
                if hasattr(model, "decision_function"):
                    raw = model.decision_function(X)
                    # Lower scores often correspond to anomaly; invert for intuitive scaling
                    scaled = (-raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                elif hasattr(model, "score_samples"):
                    raw = model.score_samples(X)
                    scaled = (-raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                else:
                    scaled = is_anom.astype(float)
                results[name] = {
                    "decision": int(is_anom[0]),
                    "score": float(scaled[0])
                }
            except Exception as e:
                logger.error(f"Detector {name} failed during predict: {e}")
        return results


# -------------------------------------------------------------------------------------
# Advanced Anomaly System
# -------------------------------------------------------------------------------------

class AdvancedAnomalySystem:
    """
    Coordinates:
        - Preprocessing (fitted once)
        - Ensemble detectors
        - Autoencoder reconstruction scoring
        - Confidence fusion + adaptive thresholding
        - Temporal pattern smoothing
        - Dynamic sensor support & feature adaptation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_dir: str = "models/anomaly",
        latent_dim: int = 32,
        autoencoder_epochs: int = 80,
        autoencoder_batch_size: int = 32,
        learning_rate: float = 1e-3,
        fuse_weights: Dict[str, float] = None,
        seed: int = 42,
        enable_lof: bool = True
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Core components
        feature_order = list(config.get("sensors", {}).keys())
        self.preprocessor = AdvancedPreprocessor(config, feature_order=feature_order)
        self.detectors = EnsembleAnomalyDetector(random_state=seed, enable_lof=enable_lof)
        
        # Create initial autoencoder (will be updated if feature set changes)
        input_dim = len(feature_order)
        self.autoencoder = TabularAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim
        )

        self.ae_epochs = autoencoder_epochs
        self.ae_batch_size = autoencoder_batch_size
        self.ae_lr = learning_rate

        # Fusion weights (classical ensemble vs AE)
        self.fuse_weights = fuse_weights or {
            "ensemble": 0.55,
            "autoencoder": 0.45
        }

        # Runtime state
        self.threshold: float = 0.5
        self.is_trained: bool = False
        self.history: List[AnomalyDetectionResult] = []
        self.max_history: int = 1200
        
        # Dynamic sensor adaptation
        self.needs_retraining: bool = False
        self.last_feature_update: Optional[datetime] = None
        self.original_feature_count: int = len(feature_order)

        # Calibration caches
        self._reconstruction_distribution: List[float] = []
        self._version = "1.1.0"  # Updated for dynamic sensor support

        logger.info("AdvancedAnomalySystem initialized (foundation layer with dynamic sensor support).")

    # ------------------------------------------------------------------
    # Training & Calibration
    # ------------------------------------------------------------------
    
    def check_retraining_needed(self, sensor_data: Dict[str, Any]) -> bool:
        """Check if model retraining is needed due to feature changes"""
        # First, update feature set in preprocessor
        updated = self.preprocessor.update_feature_set(sensor_data)
        
        # If features were updated, or we have a pending retraining flag
        if updated or self.preprocessor.needs_retraining:
            self.needs_retraining = True
            return True
            
        # Check if our autoencoder's input dimension matches current feature set
        if hasattr(self.autoencoder, "input_dim") and len(self.preprocessor.mapper.feature_names) != self.autoencoder.input_dim:
            self.needs_retraining = True
            return True
            
        return False

    def train(self, training_records: List[Dict[str, Any]]):
        if not training_records:
            raise ValueError("Training data empty.")

        logger.info("Starting anomaly system training on %d records.", len(training_records))
        # Preprocess
        self.preprocessor.fit(training_records)
        
        # Update autoencoder if feature dimensions changed
        feature_count = len(self.preprocessor.mapper.feature_names)
        if not hasattr(self.autoencoder, "input_dim") or self.autoencoder.input_dim != feature_count:
            logger.info(f"Recreating autoencoder for {feature_count} features (was: {getattr(self.autoencoder, 'input_dim', 0)})")
            self.autoencoder = TabularAutoencoder(
                input_dim=feature_count,
                latent_dim=min(32, max(8, feature_count // 2))  # Adaptive latent dimension
            )
            
        X = np.vstack([self.preprocessor.mapper.ordered_vector(r) for r in training_records])
        X_scaled = self.preprocessor.scalers["minmax"].transform(X)

        # Train ensemble detectors
        self.detectors.fit(X)

        # Train autoencoder
        self._train_autoencoder(X_scaled)

        # Calibrate threshold using reconstruction errors
        rec_errors = self._compute_reconstruction_errors(X_scaled)
        self._reconstruction_distribution = rec_errors

        # Use robust percentile + MAD-based backup
        perc95 = np.percentile(rec_errors, 95)
        mad = stats.median_abs_deviation(rec_errors)
        fallback = np.median(rec_errors) + 3 * (mad + 1e-8)
        self.threshold = max(perc95, fallback * 0.9)
        self.is_trained = True
        self.needs_retraining = False
        self.original_feature_count = feature_count

        logger.info("Anomaly system trained. Calibration threshold=%.5f (p95=%.5f, fallback=%.5f)",
                    self.threshold, perc95, fallback)

        # Persist
        self._persist_all()

    def _train_autoencoder(self, X_scaled: np.ndarray):
        dataset = TensorDataset(torch.from_numpy(X_scaled).float())
        loader = DataLoader(dataset, batch_size=self.ae_batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr, weight_decay=1e-5)
        self.autoencoder.train()
        for epoch in range(self.ae_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, self.ae_epochs // 4) == 0 or epoch == self.ae_epochs - 1:
                logger.info("AE Training epoch %d/%d - avg_loss=%.6f",
                            epoch + 1, self.ae_epochs, epoch_loss / len(loader))

    def _compute_reconstruction_errors(self, X_scaled: np.ndarray) -> List[float]:
        self.autoencoder.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled).float()
            recon = self.autoencoder(tensor)
            errors = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy().tolist()
        return errors

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, sensor_sample: Dict[str, Any]) -> AnomalyDetectionResult:
        if not self.is_trained:
            return self._untrained_response()
            
        # Check if we need retraining due to feature changes
        if self.check_retraining_needed(sensor_sample):
            # We still proceed, but adjust confidence and add recommendation
            retraining_needed = True
            confidence_penalty = 0.2
            retrain_rec = "Retrain anomaly detection model to incorporate new sensors."
        else:
            retraining_needed = False
            confidence_penalty = 0.0
            retrain_rec = None

        try:
            # Data quality - works with updated feature set
            quality_report = self.preprocessor.detect_quality(sensor_sample)
            quality_dict = quality_report.to_dict()

            # Get feature vector from sensor data
            aligned_data = self.preprocessor.mapper.align_data_for_model(sensor_sample)
            raw_vec = self.preprocessor.mapper.ordered_vector(aligned_data).reshape(1, -1)
            scaled_vec = self.preprocessor.transform(sensor_sample, scaler="minmax")

            # Ensemble predictions
            model_preds = self.detectors.predict(raw_vec)

            # Autoencoder reconstruction
            ae_score, ae_conf = self._score_autoencoder(scaled_vec)
            
            # Apply confidence penalty if retraining needed
            if retraining_needed:
                ae_conf = max(0.0, ae_conf - confidence_penalty)

            fusion = self._fuse_scores(model_preds, ae_score, ae_conf)

            criticals = self._evaluate_critical_conditions(sensor_sample)
            temporal = self._temporal_context(fusion["fused_score"])
            risk_level = self._map_risk(fusion["fused_score"], criticals)

            recs = self._generate_recommendations(fusion["fused_score"], criticals)
            
            # Add retraining recommendation if needed
            if retraining_needed and retrain_rec:
                recs.insert(0, retrain_rec)
            
            result = AnomalyDetectionResult(
                is_anomaly=fusion["fused_score"] > self.threshold,
                anomaly_score=float(fusion["fused_score"]),
                risk_level=risk_level,
                adaptive_threshold=float(self.threshold),
                fused_details={
                    "per_model": model_preds,
                    "autoencoder_score": ae_score,
                    "autoencoder_confidence": ae_conf,
                    "weights": self.fuse_weights,
                    "retraining_needed": retraining_needed
                },
                data_quality=quality_dict,
                critical_anomalies=criticals,
                temporal_context=temporal,
                recommendations=recs,
                confidence=float(max(0.1, fusion["confidence"] - (confidence_penalty if retraining_needed else 0.0)))
            )
            self._update_history(result)
            self._adaptive_threshold_refine()
            return result
        except Exception as e:
            logger.error(f"Detection failure: {e}\n{traceback.format_exc()}")
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                risk_level="UNKNOWN",
                adaptive_threshold=self.threshold,
                fused_details={"error": str(e)},
                data_quality={},
                critical_anomalies=[],
                temporal_context={},
                recommendations=["Check logs for anomaly pipeline error."],
                confidence=0.0
            )

    def _score_autoencoder(self, scaled_vec: np.ndarray) -> Tuple[float, float]:
        """
        Score using autoencoder with safety checks for dimension mismatch
        """
        self.autoencoder.eval()
        with torch.no_grad():
            # Check dimensions
            input_dim = getattr(self.autoencoder, "input_dim", 0)
            if input_dim > 0 and scaled_vec.shape[1] != input_dim:
                if scaled_vec.shape[1] > input_dim:
                    # Truncate extra dimensions
                    scaled_vec = scaled_vec[:, :input_dim]
                else:
                    # Pad with zeros
                    padding = np.zeros((scaled_vec.shape[0], input_dim - scaled_vec.shape[1]))
                    scaled_vec = np.hstack([scaled_vec, padding])
            
            tens = torch.from_numpy(scaled_vec).float()
            recon = self.autoencoder(tens)
            mse = torch.mean((tens - recon) ** 2).item()
            
        # Confidence heuristic: ratio relative to threshold
        conf = 1.0 - min(1.0, mse / (self.threshold + 1e-8))
        return mse, max(0.0, conf)

    def _fuse_scores(self, model_preds: Dict[str, Dict[str, Any]], ae_score: float, ae_conf: float) -> Dict[str, float]:
        """
        Convert model binary decisions & their normalized scores into a blended anomaly score.
        1) Ensemble: average of model 'score' weighted by their binary decision emphasis
        2) Combine with AE using configured fuse_weights
        """
        # Ensemble aggregated
        if model_preds:
            ensemble_raw_scores = []
            decisions = []
            for d in model_preds.values():
                ensemble_raw_scores.append(d["score"])
                decisions.append(d["decision"])
            # Decision emphasis: weighted boost if multiple detectors agree
            decision_agreement = sum(decisions) / max(1, len(decisions))
            ensemble_score = float(np.mean(ensemble_raw_scores) * (0.5 + 0.5 * decision_agreement))
        else:
            ensemble_score = 0.0

        # Normalize autoencoder error relative to threshold
        ae_norm = min(1.0, ae_score / (self.threshold + 1e-8))

        fused = (
            self.fuse_weights["ensemble"] * ensemble_score +
            self.fuse_weights["autoencoder"] * ae_norm
        )
        fused_conf = (0.5 + 0.5 * ae_conf) * (0.5 + 0.5 * (1 - abs(0.5 - ensemble_score)))

        return {
            "fused_score": float(fused),
            "confidence": float(max(0.0, min(1.0, fused_conf)))
        }

    # ------------------------------------------------------------------
    # Supporting Logic
    # ------------------------------------------------------------------

    def _evaluate_critical_conditions(self, sensor_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        criticals: List[Dict[str, Any]] = []
        for sensor, cfg in self.config.get("sensors", {}).items():
            if sensor not in sensor_sample:
                continue
            val = sensor_sample[sensor]
            critical = cfg.get("critical")
            warn = critical * 0.8 if critical is not None else None
            if critical is not None and val >= critical:
                criticals.append({
                    "sensor": sensor,
                    "value": val,
                    "threshold": critical,
                    "severity": "CRITICAL"
                })
            elif warn is not None and val >= warn:
                criticals.append({
                    "sensor": sensor,
                    "value": val,
                    "threshold": warn,
                    "severity": "WARNING"
                })
        return criticals

    def _temporal_context(self, current_score: float) -> Dict[str, Any]:
        # Use last N scores to derive trend, volatility
        window = 25
        scores = [h.anomaly_score for h in self.history[-window:]]
        if len(scores) < 5:
            return {"trend": "insufficient_history", "volatility": 0.0, "recent_mean": current_score}
        arr = np.array(scores)
        # Linear trend
        x = np.arange(len(arr))
        slope, _, _, _, _ = stats.linregress(x, arr)
        trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        return {
            "trend": trend,
            "slope": float(slope),
            "volatility": float(np.std(arr)),
            "recent_mean": float(arr.mean())
        }

    def _map_risk(self, fused_score: float, criticals: List[Dict[str, Any]]) -> str:
        crit_count = sum(1 for c in criticals if c["severity"] == "CRITICAL")
        warn_count = sum(1 for c in criticals if c["severity"] == "WARNING")
        # Composite
        composite = fused_score + crit_count * 0.25 + warn_count * 0.1
        if composite >= 0.85 or crit_count >= 2:
            return "CRITICAL"
        if composite >= 0.65 or crit_count >= 1:
            return "HIGH"
        if composite >= 0.45 or warn_count >= 2:
            return "MEDIUM"
        return "LOW"

    def _generate_recommendations(self, fused_score: float, criticals: List[Dict[str, Any]]) -> List[str]:
        recs: List[str] = []
        if fused_score >= 0.85:
            recs.extend([
                "Initiate immediate safety protocol review.",
                "Escalate to operations supervisor.",
                "Increase sensor polling frequency (fast mode)."
            ])
        elif fused_score >= 0.65:
            recs.extend([
                "Schedule near-term inspection of affected subsystems.",
                "Enable heightened monitoring interval (short-term)."
            ])
        if not recs:
            recs.append("Continue routine monitoring.")
            
        # Sensor-specific references
        for c in criticals:
            if c["severity"] == "CRITICAL":
                recs.append(f"Immediate technical check: sensor '{c['sensor']}' exceeded critical threshold.")
            elif c["severity"] == "WARNING":
                recs.append(f"Monitor sensor '{c['sensor']}' – nearing critical threshold.")
                
        # Add retraining recommendation if feature set changed
        if self.needs_retraining:
            recs.append("Retrain anomaly detection model to incorporate sensor changes.")
            
        # Deduplicate preserving order
        seen = set()
        unique = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:10]

    def _update_history(self, result: AnomalyDetectionResult):
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _adaptive_threshold_refine(self):
        # Optional smoothing using recent accepted non-anomalous scores
        recent = [r.anomaly_score for r in self.history[-100:] if not r.is_anomaly]
        if len(recent) >= 25:
            p95 = np.percentile(recent, 95)
            self.threshold = 0.9 * self.threshold + 0.1 * p95

    def _untrained_response(self) -> AnomalyDetectionResult:
        return AnomalyDetectionResult(
            is_anomaly=False,
            anomaly_score=0.0,
            risk_level="LOW",
            adaptive_threshold=self.threshold,
            fused_details={"warning": "models_not_trained"},
            data_quality={"note": "preprocessor not fitted"},
            critical_anomalies=[],
            temporal_context={},
            recommendations=["Train anomaly system before production usage."],
            confidence=0.1
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_all(self):
        try:
            meta = {
                "version": self._version,
                "trained_at": datetime.utcnow().isoformat(),
                "threshold": self.threshold,
                "feature_order": self.preprocessor.mapper.feature_names
            }
            # Classical models
            for name, mdl in self.detectors.models.items():
                path = self.model_dir / f"{name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(mdl, f)
            # Scalers / stats
            with open(self.model_dir / "preprocessor.pkl", "wb") as f:
                pickle.dump({
                    "scalers": self.preprocessor.scalers,
                    "stats": self.preprocessor.stats,
                    "feature_names": self.preprocessor.mapper.feature_names
                }, f)
            # Autoencoder weights
            torch.save(self.autoencoder.state_dict(), self.model_dir / "autoencoder.pt")
            # Save additional metadata about feature dimensions
            meta["input_dimensions"] = getattr(self.autoencoder, "input_dim", len(self.preprocessor.mapper.feature_names))
            # Metadata
            with open(self.model_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info("Anomaly models persisted successfully.")
        except Exception as e:
            logger.error(f"Model persistence failed: {e}")

    def load(self):
        try:
            # Preprocessor
            with open(self.model_dir / "preprocessor.pkl", "rb") as f:
                obj = pickle.load(f)
            self.preprocessor.mapper = SensorFeatureMapper(obj["feature_names"])
            self.preprocessor.scalers = obj["scalers"]
            self.preprocessor.stats = obj["stats"]
            self.preprocessor.fitted = True
            
            # Store original feature count
            self.original_feature_count = len(obj["feature_names"])

            # Load metadata
            meta_path = self.model_dir / "metadata.json"
            input_dim = self.original_feature_count
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.threshold = float(meta.get("threshold", self.threshold))
                input_dim = meta.get("input_dimensions", self.original_feature_count)
                logger.info("Loaded anomaly system metadata: version=%s threshold=%.5f",
                            meta.get("version"), self.threshold)

            # Classical detectors
            for name in list(self.detectors.models.keys()):
                path = self.model_dir / f"{name}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        self.detectors.models[name] = pickle.load(f)
                        
            # Store feature dimensions in detector for compatibility check
            self.detectors.feature_dimensions = input_dim

            # Recreate autoencoder with correct input dimension
            self.autoencoder = TabularAutoencoder(
                input_dim=input_dim,
                latent_dim=min(32, max(8, input_dim // 2))
            )
            
            # Load autoencoder weights
            ae_path = self.model_dir / "autoencoder.pt"
            if ae_path.exists():
                self.autoencoder.load_state_dict(torch.load(ae_path, map_location="cpu"))
                self.autoencoder.eval()

            self.is_trained = True
            logger.info(f"Anomaly system loaded from disk with {input_dim} feature dimensions.")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.is_trained = False

    # ------------------------------------------------------------------
    # Public Introspection
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        current_features = len(self.preprocessor.mapper.feature_names)
        return {
            "trained": self.is_trained,
            "threshold": self.threshold,
            "history_size": len(self.history),
            "feature_importances": self.preprocessor.mapper.summary(),
            "original_feature_count": self.original_feature_count,
            "current_feature_count": current_features,
            "features_changed": current_features != self.original_feature_count,
            "needs_retraining": self.needs_retraining,
            "last_feature_update": self.last_feature_update.isoformat() if self.last_feature_update else None,
            "last_result": self.history[-1].to_dict() if self.history else None
        }


# -------------------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------------------

def create_anomaly_system(config: Dict[str, Any], **kwargs) -> AdvancedAnomalySystem:
    """
    Factory with safe defaults. Pass additional kwargs to override training parameters.
    """
    return AdvancedAnomalySystem(config=config, **kwargs)


# -------------------------------------------------------------------------------------
# __all__
# -------------------------------------------------------------------------------------

__all__ = [
    "DetectorType",
    "DataQualityIssue",
    "DataQualityReport",
    "AnomalyModelMetadata",
    "AnomalyDetectionResult",
    "SensorFeatureMapper",
    "AdvancedPreprocessor",
    "TabularAutoencoder",
    "EnsembleAnomalyDetector",
    "AdvancedAnomalySystem",
    "create_anomaly_system"
            ]
