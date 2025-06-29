import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[AdvancedPredictionEngine] %(levelname)s - %(message)s'
)

SENSOR_COLUMNS = [
    'Temperature (°C)', 'Pressure (psi)', 'Vibration (g)', 'Methane (CH₄ ppm)', 'H₂S (ppm)'
]

MODEL_PATH = "models/advanced_prediction_models_v2.pkl"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all time/cyclical/lag features for time series prediction."""
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["hour"] = df["Time"].dt.hour
    df["day"] = df["Time"].dt.day
    df["day_of_week"] = df["Time"].dt.dayofweek
    df["time_numeric"] = (df["Time"] - df["Time"].min()).dt.total_seconds() / 3600
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    # Lag/rolling
    for sensor in SENSOR_COLUMNS:
        df[f'{sensor}_lag1'] = df[sensor].shift(1)
        df[f'{sensor}_lag2'] = df[sensor].shift(2)
        df[f'{sensor}_rolling_mean'] = df[sensor].rolling(window=3).mean()
    return df.dropna()


def get_feature_columns() -> List[str]:
    """Return all feature columns used for prediction."""
    base = ['time_numeric', 'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    lagged = []
    for sensor in SENSOR_COLUMNS:
        lagged += [f'{sensor}_lag1', f'{sensor}_lag2', f'{sensor}_rolling_mean']
    return base + lagged


def train_models(
    data: pd.DataFrame,
    save_path: str = MODEL_PATH
) -> Dict[str, Dict[str, Any]]:
    """Train RandomForest models for each sensor and save to disk."""
    logging.info("Generating features for training...")
    df = create_features(data)
    feature_columns = get_feature_columns()
    models = {}
    for sensor in SENSOR_COLUMNS:
        logging.info(f"Training model for {sensor}...")
        X, y = df[feature_columns], df[sensor]
        model = RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        models[sensor] = {
            'model': model,
            'feature_columns': feature_columns,
            'mean': y.mean(),
            'std': y.std(),
            'min': y.min(),
            'max': y.max()
        }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(models, save_path)
    logging.info(f"Saved trained models to {save_path}")
    return models


def load_models(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """Load trained models from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")
    return joblib.load(model_path)


def predict_future(
    models: Dict[str, Any],
    recent_data: pd.DataFrame,
    hours_ahead: int = 72
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Predict sensor values for the next `hours_ahead` hours, using recent_data for lag features.
    """
    predictions = {sensor: [] for sensor in SENSOR_COLUMNS}
    df_hist = create_features(recent_data)
    feature_columns = get_feature_columns()

    for h in range(1, hours_ahead + 1):
        pred_time = recent_data["Time"].max() + pd.Timedelta(hours=h)
        new_row = {
            "Time": pred_time,
            "hour": pred_time.hour,
            "day": pred_time.day,
            "day_of_week": pred_time.dayofweek,
            "time_numeric": (pred_time - recent_data["Time"].min()).total_seconds() / 3600,
            "hour_sin": np.sin(2 * np.pi * pred_time.hour / 24),
            "hour_cos": np.cos(2 * np.pi * pred_time.hour / 24),
            "day_sin": np.sin(2 * np.pi * pred_time.dayof_week / 7),
            "day_cos": np.cos(2 * np.pi * pred_time.day_of_week / 7),
        }
        # Prepare lag/rolling features per sensor:
        for sensor in SENSOR_COLUMNS:
            lag1 = df_hist[sensor].iloc[-1] if not df_hist.empty else 0
            lag2 = df_hist[sensor].iloc[-2] if len(df_hist) > 1 else lag1
            rolling = df_hist[sensor].iloc[-3:].mean() if len(df_hist) >= 3 else lag1
            new_row[f"{sensor}_lag1"] = lag1
            new_row[f"{sensor}_lag2"] = lag2
            new_row[f"{sensor}_rolling_mean"] = rolling

        # Predict for each sensor
        for sensor in SENSOR_COLUMNS:
            model_data = models[sensor]
            X_pred = pd.DataFrame([new_row])[model_data["feature_columns"]]
            value = model_data["model"].predict(X_pred)[0]
            # Clamp value to min/max
            value = float(np.clip(value, model_data["min"], model_data["max"]))
            predictions[sensor].append({
                "time": pred_time,
                "value": value,
                "hours_ahead": h
            })
            # Update historical DataFrame for next iteration (simulate rolling window)
            df_hist = pd.concat([
                df_hist,
                pd.DataFrame({sensor: [value]}, index=[df_hist.index[-1] + 1 if not df_hist.empty else 0])
            ], axis=0, ignore_index=True)
    return predictions


def get_prediction_summary(predictions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Return summary stats (min/max/mean) for each sensor's predicted series."""
    summary = {}
    for sensor, preds in predictions.items():
        values = [p["value"] for p in preds]
        summary[sensor] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values))
        }
    return summary


# Example usage (can be removed in prod):
if __name__ == "__main__":
    df = pd.read_csv("sensor_data_simulated.csv")
    models = train_models(df)
    preds = predict_future(models, df.tail(100), hours_ahead=72)
    print(get_prediction_summary(preds))
