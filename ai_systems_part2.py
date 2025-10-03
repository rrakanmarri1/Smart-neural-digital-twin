from __future__ import annotations

import json
import logging
import math
import os
import pickle
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset, random_split

# Import Part 1 components
try:
    from ai_systems_part1 import (
        AdvancedAnomalySystem,
        AnomalyDetectionResult,
        create_anomaly_system
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Part 1 (ai_systems_part1.py) must be available before using Part 2. "
        f"Import failed: {e}"
    )

logger = logging.getLogger("SmartNeural.AI.Part2")
logger.setLevel(logging.INFO)

# =====================================================================================
# Domain Data Models
# =====================================================================================

@dataclass
class ForecastResult:
    horizons: Dict[str, List[float]]
    per_feature_confidence: Dict[str, float]
    aggregate_confidence: float
    risk_level: str
    risk_factors: List[str]
    trends: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_used: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizons": self.horizons,
            "per_feature_confidence": self.per_feature_confidence,
            "aggregate_confidence": self.aggregate_confidence,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "trends": self.trends,
            "timestamp": self.timestamp.isoformat(),
            "model_used": self.model_used,
            "recommendations": self.recommendations
        }


@dataclass
class DriftReport:
    model_drift_detected: bool
    feature_drift_detected: bool
    model_drift_p_value: float
    feature_drift_summary: Dict[str, Any]
    recent_error_mean: float
    baseline_error_mean: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveLearningSummary:
    sampled_for_retrain: bool
    feedback_buffer_size: int
    drift_report: DriftReport
    next_retrain_due: Optional[datetime]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["next_retrain_due"] = self.next_retrain_due.isoformat() if self.next_retrain_due else None
        return d


@dataclass
class UnifiedAIStepResult:
    anomaly: AnomalyDetectionResult
    forecast: ForecastResult
    adaptive: AdaptiveLearningSummary
    overall_risk: Dict[str, Any]
    recommendations: List[str]
    processing_time_s: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly": self.anomaly.to_dict(),
            "forecast": self.forecast.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "overall_risk": self.overall_risk,
            "recommendations": self.recommendations,
            "processing_time_s": self.processing_time_s,
            "timestamp": self.timestamp.isoformat()
        }


# =====================================================================================
# Utility
# =====================================================================================

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_config_get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# =====================================================================================
# Positional Encoding (Transformer)
# =====================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        return x + self.pe[:, : x.size(1)]


# =====================================================================================
# Sequence Models
# =====================================================================================

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, output_size: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        last = attn_out[:, -1, :]
        return self.head(last)


class TransformerForecast(nn.Module):
    def __init__(self, input_size: int, model_dim: int = 128, n_heads: int = 4, num_layers: int = 3, ff_dim: int = 256, dropout: float = 0.1, output_size: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(input_size, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim, output_size)
        )
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        enc = self.encoder(h)
        last = enc[:, -1, :]
        return self.head(last)


class HybridFusionModel(nn.Module):
    """
    Parallel LSTM (bidirectional) + shallow Transformer branch.
    Fuses final representations and predicts next-step vector.
    """
    def __init__(self, input_size: int, hidden_lstm: int = 96, model_dim: int = 96, output_size: int = 6):
        super().__init__()
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size,
            hidden_lstm,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        # Transformer branch
        enc_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pos_enc = PositionalEncoding(input_size)

        fusion_in = hidden_lstm * 2 + input_size
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, output_size)
        )
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.kaiming_uniform_(p, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]
        # Transformer branch
        tx_in = self.pos_enc(x)
        tx_out = self.transformer(tx_in)
        tx_feat = tx_out[:, -1, :]
        fused = torch.cat([lstm_feat, tx_feat], dim=1)
        return self.head(fused)


# =====================================================================================
# Prediction Engine
# =====================================================================================

class PredictionEngine:
    """
    Handles:
        - Sequence preparation
        - Model training (multiple architectures)
        - Early stopping (validation-based)
        - Multi-horizon forecasting via iterative refeeding
        - Confidence estimation based on residual scale & horizon distance
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_order: List[str],
        model_dir: str = "models/prediction",
        seed: int = 42,
        device: Optional[str] = None
    ):
        self.config = config
        self.feature_order = feature_order
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        set_all_seeds(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = int(safe_config_get(config, "prediction.sequence_length", 50))
        horizons_cfg = safe_config_get(config, "prediction.horizons", {"short": 6, "medium": 24, "long": 72})
        self.horizons = {k: int(v) for k, v in horizons_cfg.items()}

        self.scaler_mean: np.ndarray = None
        self.scaler_std: np.ndarray = None

        input_dim = len(feature_order)
        self.models: Dict[str, nn.Module] = {
            "lstm_attn": LSTMWithAttention(input_dim=input_dim, output_size=input_dim),
            "transformer": TransformerForecast(input_size=input_dim, output_size=input_dim),
            "hybrid": HybridFusionModel(input_size=input_dim, output_size=input_dim)
        }
        self.active_model_name: Optional[str] = None
        self.active_model: Optional[nn.Module] = None
        self.training_metrics: Dict[str, Any] = {}
        self.residual_history: List[float] = []
        self.max_residuals = 1000
        self.is_trained = False

    # ------------------------------------------------------------------
    # Data Utilities
    # ------------------------------------------------------------------

    def _prepare_array(self, records: List[Dict[str, Any]]) -> np.ndarray:
        mat = []
        for r in records:
            mat.append([float(r.get(f, 0.0)) for f in self.feature_order])
        return np.asarray(mat, dtype=float)

    def _build_sequences(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for i in range(len(matrix) - self.seq_len):
            seqs.append(matrix[i:i + self.seq_len])
            targets.append(matrix[i + self.seq_len])
        if not seqs:
            return np.empty((0, self.seq_len, matrix.shape[1])), np.empty((0, matrix.shape[1]))
        return np.array(seqs), np.array(targets)

    def _fit_standard_scaler(self, array: np.ndarray):
        self.scaler_mean = array.mean(axis=0)
        self.scaler_std = array.std(axis=0) + 1e-8

    def _scale(self, array: np.ndarray) -> np.ndarray:
        return (array - self.scaler_mean) / self.scaler_std

    def _inverse_scale(self, array: np.ndarray) -> np.ndarray:
        return array * self.scaler_std + self.scaler_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, records: List[Dict[str, Any]], val_split: float = 0.15, max_epochs: int = 120, patience: int = 12, lr: float = 1e-3):
        if len(records) < self.seq_len + 5:
            raise ValueError("Insufficient records for training.")

        matrix = self._prepare_array(records)
        self._fit_standard_scaler(matrix)
        seqs, targets = self._build_sequences(matrix)
        seqs_scaled = self._scale(seqs)
        targets_scaled = self._scale(targets)

        dataset = TensorDataset(
            torch.from_numpy(seqs_scaled).float(),
            torch.from_numpy(targets_scaled).float()
        )

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        if val_size < 5:  # ensure minimum validation batch
            val_size = min(5, len(dataset) // 5)
            train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        results = {}
        criterion = nn.MSELoss()

        for name, model in self.models.items():
            model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            best_val = float("inf")
            best_state = None
            epochs_no_improve = 0
            losses = {"train": [], "val": []}

            for epoch in range(max_epochs):
                model.train()
                epoch_loss = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                train_avg = epoch_loss / max(1, len(train_loader))

                # Validation
                model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        preds = model(xb)
                        v_loss = criterion(preds, yb)
                        val_loss_accum += v_loss.item()
                val_avg = val_loss_accum / max(1, len(val_loader))

                losses["train"].append(train_avg)
                losses["val"].append(val_avg)

                improved = val_avg < best_val - 1e-5
                if improved:
                    best_val = val_avg
                    best_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if (epoch + 1) % 20 == 0 or improved or epoch == max_epochs - 1:
                    logger.info(
                        f"[Prediction] {name} epoch {epoch+1}/{max_epochs} "
                        f"train={train_avg:.5f} val={val_avg:.5f} best={best_val:.5f}"
                    )

                if epochs_no_improve >= patience:
                    logger.info(f"[Prediction] Early stopping {name} at epoch {epoch+1}")
                    break

            # Restore best weights
            if best_state:
                model.load_state_dict(best_state)

            results[name] = {
                "best_val_loss": best_val,
                "train_curve": losses["train"],
                "val_curve": losses["val"]
            }

        # Select best model
        best_name = min(results.keys(), key=lambda n: results[n]["best_val_loss"])
        self.active_model_name = best_name
        self.active_model = self.models[best_name]
        self.training_metrics = results
        self._persist_prediction_state()
        self.is_trained = True
        logger.info(f"Active forecast model selected: {best_name}")

    # ------------------------------------------------------------------
    # Multi-Horizon Forecasting
    # ------------------------------------------------------------------

    def forecast(self, recent_records: List[Dict[str, Any]]) -> ForecastResult:
        if not self.is_trained or self.active_model is None:
            return self._untrained_forecast_response()

        if len(recent_records) < self.seq_len:
            return self._insufficient_data_response()

        matrix = self._prepare_array(recent_records[-self.seq_len:])
        scaled_seq = self._scale(matrix)
        seq_tensor = torch.from_numpy(scaled_seq).float().unsqueeze(0).to(self.device)

        self.active_model.eval()
        with torch.no_grad():
            horizon_outputs: Dict[str, List[float]] = {}
            per_feature_conf: Dict[str, float] = {}
            all_conf_factors = []

            for horizon_name, steps in self.horizons.items():
                working_seq = seq_tensor.clone()
                preds_scaled_accum = []
                for step in range(steps):
                    pred = self.active_model(working_seq)
                    preds_scaled_accum.append(pred.cpu().numpy())
                    # iterative append
                    appended = torch.cat([working_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
                    working_seq = appended

                preds_scaled_array = np.array(preds_scaled_accum)  # shape (steps, features)
                preds_array = self._inverse_scale(preds_scaled_array)
                horizon_outputs[horizon_name] = preds_array.tolist()

                # Basic confidence: lower variance across iterative predictions => higher confidence
                per_feature_var = preds_scaled_array.var(axis=0)
                conf = 1.0 / (1.0 + per_feature_var)  # shrink
                for i, f in enumerate(self.feature_order):
                    if f not in per_feature_conf:
                        per_feature_conf[f] = 0.0
                    per_feature_conf[f] += conf[i] / len(self.horizons)
                all_conf_factors.extend(conf.tolist())

            agg_conf = float(np.mean(all_conf_factors)) if all_conf_factors else 0.0

            # Risk assessment (basic – threshold against critical percentages)
            risk_level, risk_factors = self._forecast_risk_analysis(horizon_outputs)

            trends = self._extract_trends(horizon_outputs)

            recommendations = self._forecast_recommendations(risk_level, risk_factors, trends)

            return ForecastResult(
                horizons=horizon_outputs,
                per_feature_confidence={k: float(min(1.0, max(0.0, v))) for k, v in per_feature_conf.items()},
                aggregate_confidence=float(min(1.0, max(0.0, agg_conf))),
                risk_level=risk_level,
                risk_factors=risk_factors,
                trends=trends,
                model_used=self.active_model_name or "unknown",
                recommendations=recommendations
            )

    def _forecast_risk_analysis(self, horizon_outputs: Dict[str, List[List[float]]]) -> Tuple[str, List[str]]:
        factors: List[str] = []
        # Use the furthest horizon as worst-case
        if not horizon_outputs:
            return "LOW", factors
        # Choose the longest horizon key
        longest_key = max(horizon_outputs.keys(), key=lambda k: len(horizon_outputs[k]))
        final_step = horizon_outputs[longest_key][-1]
        # Map features
        final_map = {f: final_step[i] for i, f in enumerate(self.feature_order)}
        for f, val in final_map.items():
            cfg = self.config["sensors"].get(f, {})
            critical = cfg.get("critical")
            max_v = cfg.get("max")
            if critical and val >= 0.9 * critical:
                factors.append(f"{f} near critical ({val:.2f} >= 0.9*{critical})")
            elif max_v and val >= 0.95 * max_v:
                factors.append(f"{f} approaching max capacity")
        if len(factors) >= 3:
            return "HIGH", factors
        if len(factors) >= 1:
            return "MEDIUM", factors
        return "LOW", factors

    def _extract_trends(self, horizon_outputs: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        if not horizon_outputs:
            return {}
        # Use short horizon
        short_key = min(horizon_outputs.keys(), key=lambda k: len(horizon_outputs[k]))
        arr = np.array(horizon_outputs[short_key])  # shape (steps, features)
        slopes = {}
        for i, f in enumerate(self.feature_order):
            series = arr[:, i]
            if len(series) < 2:
                slopes[f] = 0.0
            else:
                x = np.arange(len(series))
                slope, _, _, _, _ = stats.linregress(x, series)
                slopes[f] = float(slope)
        dominant = sorted(slopes.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return {"feature_slopes": slopes, "dominant": dominant}

    def _forecast_recommendations(self, risk_level: str, factors: List[str], trends: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        if risk_level == "HIGH":
            recs.extend([
                "Escalate to control room supervisor.",
                "Enable high-frequency telemetry capture.",
                "Prepare mitigation protocol checklist."
            ])
        elif risk_level == "MEDIUM":
            recs.append("Schedule targeted inspection window.")
        if not recs:
            recs.append("Maintain standard monitoring procedures.")
        # Factor references
        for f in factors:
            if "critical" in f:
                recs.append("Pre-emptively validate safety interlocks for critical parameter.")
        return list(dict.fromkeys(recs))[:10]

    # ------------------------------------------------------------------
    # Forecast Response Helpers
    # ------------------------------------------------------------------

    def _untrained_forecast_response(self) -> ForecastResult:
        return ForecastResult(
            horizons={},
            per_feature_confidence={},
            aggregate_confidence=0.0,
            risk_level="UNKNOWN",
            risk_factors=["Model not trained"],
            trends={},
            model_used="None",
            recommendations=["Train prediction engine."]
        )

    def _insufficient_data_response(self) -> ForecastResult:
        return ForecastResult(
            horizons={},
            per_feature_confidence={},
            aggregate_confidence=0.0,
            risk_level="LOW",
            risk_factors=["Insufficient sequence length"],
            trends={},
            model_used="None",
            recommendations=["Accumulate more data for valid forecasting."]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_prediction_state(self):
        try:
            meta = {
                "active_model": self.active_model_name,
                "trained_at": datetime.utcnow().isoformat(),
                "seq_len": self.seq_len,
                "feature_order": self.feature_order
            }
            # Save scaler
            with open(self.model_dir / "scaler.pkl", "wb") as f:
                pickle.dump({"mean": self.scaler_mean, "std": self.scaler_std}, f)
            # Save each model state
            for name, model in self.models.items():
                torch.save(model.state_dict(), self.model_dir / f"{name}.pt")
            # Save training metrics
            with open(self.model_dir / "training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.training_metrics, f, indent=2)
            # Save meta
            with open(self.model_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info("Prediction engine persisted.")
        except Exception as e:
            logger.error(f"Prediction engine persistence failed: {e}")

    def load(self):
        try:
            with open(self.model_dir / "scaler.pkl", "rb") as f:
                sc = pickle.load(f)
            self.scaler_mean = sc["mean"]
            self.scaler_std = sc["std"]

            with open(self.model_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.seq_len = meta.get("seq_len", self.seq_len)
            self.active_model_name = meta.get("active_model")
            # Load training metrics if exist
            tm_file = self.model_dir / "training_metrics.json"
            if tm_file.exists():
                with open(tm_file, "r", encoding="utf-8") as f:
                    self.training_metrics = json.load(f)

            # Load models
            for name, model in self.models.items():
                path = self.model_dir / f"{name}.pt"
                if path.exists():
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    model.to(self.device)
            self.active_model = self.models.get(self.active_model_name, None)
            if self.active_model:
                self.is_trained = True
                logger.info("Prediction engine loaded. Active model=%s", self.active_model_name)
            else:
                logger.warning("Active model '%s' not found – forecasting disabled.", self.active_model_name)
        except Exception as e:
            logger.error(f"Prediction engine load failed: {e}")
            self.is_trained = False


# =====================================================================================
# Drift & Adaptive Learning
# =====================================================================================

class DriftMonitor:
    """
    Tracks residual distributions (post-inference) and feature distributions
    to detect drift via:
        - Welch t-test for residual mean shifts
        - KS-test for feature distribution changes
    """

    def __init__(self, residual_window: int = 200, feature_window: int = 400):
        self.residual_window = residual_window
        self.feature_window = feature_window
        self.residuals: List[float] = []
        self.feature_history: List[np.ndarray] = []

    def update(self, residual_vector: np.ndarray, latest_features: np.ndarray):
        # residual_vector shape: (features,)
        # Use mean absolute residual as scalar
        mean_res = float(np.mean(np.abs(residual_vector)))
        self.residuals.append(mean_res)
        if len(self.residuals) > self.residual_window * 2:
            self.residuals = self.residuals[-self.residual_window * 2:]

        self.feature_history.append(latest_features.copy())
        if len(self.feature_history) > self.feature_window * 2:
            self.feature_history = self.feature_history[-self.feature_window * 2:]

    def report(self) -> DriftReport:
        # Model drift
        if len(self.residuals) >= self.residual_window * 2:
            recent = self.residuals[-self.residual_window:]
            baseline = self.residuals[-self.residual_window * 2: -self.residual_window]
            # Welch's t-test (unequal variances)
            t_stat, p_val = stats.ttest_ind(recent, baseline, equal_var=False)
            model_drift = (p_val < 0.05) and (np.mean(recent) > np.mean(baseline) * 1.05)
            recent_mean = float(np.mean(recent))
            baseline_mean = float(np.mean(baseline))
        else:
            model_drift = False
            p_val = 1.0
            recent_mean = baseline_mean = 0.0

        # Feature drift
        feature_drift_detected = False
        feature_summary: Dict[str, Any] = {}
        if len(self.feature_history) >= self.feature_window * 2:
            hist_arr = np.array(self.feature_history)
            recent_f = hist_arr[-self.feature_window:]
            baseline_f = hist_arr[-self.feature_window * 2: -self.feature_window]
            drift_count = 0
            for i in range(hist_arr.shape[1]):
                ks_stat, ks_p = stats.ks_2samp(baseline_f[:, i], recent_f[:, i])
                if ks_p < 0.01:
                    drift_count += 1
            feature_drift_detected = drift_count >= max(1, hist_arr.shape[1] // 3)
            feature_summary = {
                "drifted_dimensions": drift_count,
                "total_dimensions": hist_arr.shape[1]
            }

        return DriftReport(
            model_drift_detected=model_drift,
            feature_drift_detected=feature_drift_detected,
            model_drift_p_value=float(p_val),
            feature_drift_summary=feature_summary,
            recent_error_mean=recent_mean,
            baseline_error_mean=baseline_mean
        )


class FeedbackBuffer:
    """
    Stores user/automatic feedback entries with capped retention.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add(self, prediction: ForecastResult, actual_sample: Dict[str, Any], meta: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction.to_dict(),
            "actual": actual_sample,
            "meta": meta
        }
        self.buffer.append(entry)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def size(self) -> int:
        return len(self.buffer)


class ActiveLearningController:
    """
    Decides whether to sample a prediction result for future retraining
    based on low confidence or high forecast risk.
    """

    def __init__(self, confidence_threshold: float = 0.55, high_risk_levels: Tuple[str, ...] = ("HIGH", "CRITICAL")):
        self.confidence_threshold = confidence_threshold
        self.high_risk_levels = high_risk_levels

    def should_sample(self, forecast: ForecastResult) -> bool:
        if forecast.aggregate_confidence < self.confidence_threshold:
            return True
        if forecast.risk_level in self.high_risk_levels:
            return True
        return False


# =====================================================================================
# AI System Manager (Aggregation)
# =====================================================================================

class AISystemManager:
    """
    Orchestrates:
        - Anomaly Detection (from Part 1)
        - Prediction Engine
        - Drift Monitoring & Adaptive triggers

    Usage:
        manager = AISystemManager(config)
        manager.train_all(training_records)
        result = manager.process(single_sensor_sample)
    """

    def __init__(self, config: Dict[str, Any], model_root: str = "models", seed: int = 42):
        self.config = config
        self.seed = seed
        set_all_seeds(seed)
        sensors = list(config.get("sensors", {}).keys())
        if not sensors:
            raise ValueError("Config sensors section is empty or missing.")

        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

        # Subsystems
        self.anomaly_system = create_anomaly_system(config)
        self.prediction_engine = PredictionEngine(
            config,
            feature_order=sensors,
            model_dir=str(self.model_root / "prediction"),
            seed=seed
        )
        self.drift_monitor = DriftMonitor()
        self.feedback_buffer = FeedbackBuffer()
        self.active_learning = ActiveLearningController()

        # Scheduling
        self.last_retrain: Optional[datetime] = None
        self.retrain_interval = timedelta(hours=float(safe_config_get(config, "prediction.train.retrain_hours", 12)))

        # State
        self.system_status = "initializing"
        self.history: List[UnifiedAIStepResult] = []
        self.max_history = 500

        logger.info("AISystemManager initialized.")

    # ------------------------------------------------------------------
    # Training Orchestration
    # ------------------------------------------------------------------

    def train_all(self, training_records: List[Dict[str, Any]]):
        if len(training_records) < 100:
            raise ValueError("Provide at least 100 records for initial training to ensure representativeness.")

        logger.info("Starting unified AI training pipeline (records=%d)", len(training_records))
        # Anomaly
        self.anomaly_system.train(training_records)
        # Forecast
        pred_train_cfg = safe_config_get(self.config, "prediction.train", {})
        epochs = int(pred_train_cfg.get("epochs", 100))
        patience = int(pred_train_cfg.get("patience", 12))
        lr = float(pred_train_cfg.get("lr", 1e-3))
        self.prediction_engine.train(
            training_records,
            val_split=pred_train_cfg.get("val_split", 0.15),
            max_epochs=epochs,
            patience=patience,
            lr=lr
        )
        self.system_status = "trained"
        self.last_retrain = datetime.utcnow()
        logger.info("Unified AI training complete.")

    # ------------------------------------------------------------------
    # Processing Pipeline
    # ------------------------------------------------------------------

    def process(self, sensor_sample: Dict[str, Any], recent_history: Optional[List[Dict[str, Any]]] = None) -> UnifiedAIStepResult:
        """
        sensor_sample: latest observation (dict of sensor -> value)
        recent_history: required for forecasting; if not supplied, forecast will fallback gracefully
        """
        start = datetime.utcnow()

        anomaly_result = self.anomaly_system.detect(sensor_sample)

        if recent_history is None:
            forecast_result = self.prediction_engine._insufficient_data_response()
        else:
            forecast_result = self.prediction_engine.forecast(recent_history)

        # Adaptive learning / drift
        drift_report = self._update_drift(forecast_result, sensor_sample, recent_history)
        sampled = False
        if self.active_learning.should_sample(forecast_result):
            self.feedback_buffer.add(forecast_result, sensor_sample, {"reason": "active_learning_trigger"})
            sampled = True

        adaptive_summary = AdaptiveLearningSummary(
            sampled_for_retrain=sampled,
            feedback_buffer_size=self.feedback_buffer.size(),
            drift_report=drift_report,
            next_retrain_due=self._next_retrain_time()
        )

        overall_risk = self._overall_risk_fusion(anomaly_result, forecast_result)
        combined_recs = self._combine_recommendations(anomaly_result, forecast_result, adaptive_summary)

        elapsed = (datetime.utcnow() - start).total_seconds()

        step = UnifiedAIStepResult(
            anomaly=anomaly_result,
            forecast=forecast_result,
            adaptive=adaptive_summary,
            overall_risk=overall_risk,
            recommendations=combined_recs,
            processing_time_s=elapsed
        )
        self._append_history(step)
        self._conditional_retrain()
        return step

    # ------------------------------------------------------------------
    # Drift & Adaptive Logic
    # ------------------------------------------------------------------

    def _update_drift(self, forecast: ForecastResult, sensor_sample: Dict[str, Any], recent_history: Optional[List[Dict[str, Any]]]) -> DriftReport:
        try:
            if not forecast.horizons or not recent_history:
                return DriftReport(
                    model_drift_detected=False,
                    feature_drift_detected=False,
                    model_drift_p_value=1.0,
                    feature_drift_summary={},
                    recent_error_mean=0.0,
                    baseline_error_mean=0.0
                )
            # Use the shortest horizon predicted first-step vs actual as residual vector
            shortest_key = min(forecast.horizons.keys(), key=lambda k: len(forecast.horizons[k]))
            first_step_pred = np.array(forecast.horizons[shortest_key][0])  # shape (features,)
            actual_vec = np.array([sensor_sample.get(f, 0.0) for f in self.prediction_engine.feature_order])
            residual_vec = actual_vec - first_step_pred
            # Use last observation features for distribution drift
            self.drift_monitor.update(residual_vec, actual_vec)
            return self.drift_monitor.report()
        except Exception as e:
            logger.error(f"Drift update failed: {e}")
            return DriftReport(
                model_drift_detected=False,
                feature_drift_detected=False,
                model_drift_p_value=1.0,
                feature_drift_summary={"error": str(e)},
                recent_error_mean=0.0,
                baseline_error_mean=0.0
            )

    def _next_retrain_time(self) -> Optional[datetime]:
        if not self.last_retrain:
            return None
        return self.last_retrain + self.retrain_interval

    def _conditional_retrain(self):
        """
        Lightweight placeholder:
        In a production deployment this would:
            - Evaluate drift significance
            - Check retrain window alignment (e.g., low-load windows)
            - Trigger asynchronous job
        """
        if not self.last_retrain:
            return
        if datetime.utcnow() >= self.last_retrain + self.retrain_interval:
            logger.info("Scheduled retraining window reached (placeholder – manual invocation required).")
            self.last_retrain = datetime.utcnow()

    # ------------------------------------------------------------------
    # Risk Fusion
    # ------------------------------------------------------------------

    def _overall_risk_fusion(self, anomaly: AnomalyDetectionResult, forecast: ForecastResult) -> Dict[str, Any]:
        priority_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 0}
        anomaly_rank = priority_order.get(anomaly.risk_level, 0)
        forecast_rank = priority_order.get(forecast.risk_level, 0)
        composite_rank = max(anomaly_rank, forecast_rank)
        inverse_map = {v: k for k, v in priority_order.items()}
        overall_level = inverse_map.get(composite_rank, "LOW")
        return {
            "overall_level": overall_level,
            "anomaly_level": anomaly.risk_level,
            "forecast_level": forecast.risk_level,
            "anomaly_score": anomaly.anomaly_score,
            "forecast_confidence": forecast.aggregate_confidence
        }

    def _combine_recommendations(
        self,
        anomaly: AnomalyDetectionResult,
        forecast: ForecastResult,
        adaptive: AdaptiveLearningSummary
    ) -> List[str]:
        recs = []
        recs.extend(anomaly.recommendations)
        recs.extend(forecast.recommendations)
        if adaptive.drift_report.model_drift_detected:
            recs.append("Investigate model performance degradation (model drift detected).")
        if adaptive.drift_report.feature_drift_detected:
            recs.append("Review sensor calibration / process changes (feature distribution drift).")
        if adaptive.sampled_for_retrain:
            recs.append("Sample flagged for potential retraining dataset enrichment.")
        # Deduplicate
        seen = set()
        uniq = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq[:15]

    # ------------------------------------------------------------------
    # History & Introspection
    # ------------------------------------------------------------------

    def _append_history(self, step: UnifiedAIStepResult):
        self.history.append(step)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def status(self) -> Dict[str, Any]:
        return {
            "system_status": self.system_status,
            "trained": self.system_status == "trained",
            "anomaly_status": self.anomaly_system.status(),
            "prediction_trained": self.prediction_engine.is_trained,
            "active_prediction_model": self.prediction_engine.active_model_name,
            "history_count": len(self.history),
            "last_step": self.history[-1].to_dict() if self.history else None,
            "next_retrain_due": self._next_retrain_time().isoformat() if self._next_retrain_time() else None
        }


# =====================================================================================
# Factory
# =====================================================================================

def create_ai_system_manager(config: Dict[str, Any], **kwargs) -> AISystemManager:
    return AISystemManager(config, **kwargs)


# =====================================================================================
# __all__
# =====================================================================================

__all__ = [
    "ForecastResult",
    "DriftReport",
    "AdaptiveLearningSummary",
    "UnifiedAIStepResult",
    "PredictionEngine",
    "DriftMonitor",
    "FeedbackBuffer",
    "ActiveLearningController",
    "AISystemManager",
    "create_ai_system_manager"
      ]
