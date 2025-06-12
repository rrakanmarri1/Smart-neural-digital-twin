import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_advanced_prediction_model():
    """Create an advanced time series prediction model for 72-hour forecasting"""
    # Load historical data
    df = pd.read_csv("sensor_data_simulated.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    
    # Create comprehensive time-based features
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['time_numeric'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 3600  # hours since start
    
    # Create cyclical features for better time series modeling
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Create lag features for better temporal modeling
    sensor_columns = ['Temperature (°C)', 'Pressure (psi)', 'Vibration (g)', 'Methane (CH₄ ppm)', 'H₂S (ppm)']
    
    for sensor in sensor_columns:
        df[f'{sensor}_lag1'] = df[sensor].shift(1)
        df[f'{sensor}_lag2'] = df[sensor].shift(2)
        df[f'{sensor}_rolling_mean'] = df[sensor].rolling(window=3).mean()
    
    # Drop rows with NaN values created by lag features
    df = df.dropna()
    
    # Prepare features for prediction
    feature_columns = ['time_numeric', 'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    # Add lag features
    for sensor in sensor_columns:
        feature_columns.extend([f'{sensor}_lag1', f'{sensor}_lag2', f'{sensor}_rolling_mean'])
    
    models = {}
    
    # Train advanced models for each sensor type
    for sensor in sensor_columns:
        print(f"Training model for {sensor}...")
        
        X = df[feature_columns]
        y = df[sensor]
        
        # Use Random Forest for better long-term predictions
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Store model and statistics
        models[sensor] = {
            'model': model,
            'feature_columns': feature_columns,
            'mean': df[sensor].mean(),
            'std': df[sensor].std(),
            'min': df[sensor].min(),
            'max': df[sensor].max()
        }
    
    # Save models
    joblib.dump(models, "advanced_prediction_models.pkl")
    print("Advanced prediction models trained and saved!")
    
    return models

def predict_future_values_72h(models, hours_ahead=72):
    """Predict sensor values for the next 72 hours"""
    current_time = pd.Timestamp.now()
    predictions = {}
    
    # Load recent data for lag features
    df = pd.read_csv("sensor_data_simulated.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    recent_data = df.tail(5)  # Get last 5 rows for lag features
    
    sensor_columns = ['Temperature (°C)', 'Pressure (psi)', 'Vibration (g)', 'Methane (CH₄ ppm)', 'H₂S (ppm)']
    
    for sensor, model_data in models.items():
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        future_predictions = []
        
        # Initialize with recent values for lag features
        last_values = {
            f'{s}_lag1': recent_data[s].iloc[-1] if s in recent_data.columns else model_data['mean']
            for s in sensor_columns
        }
        last_values.update({
            f'{s}_lag2': recent_data[s].iloc[-2] if len(recent_data) > 1 and s in recent_data.columns else model_data['mean']
            for s in sensor_columns
        })
        last_values.update({
            f'{s}_rolling_mean': recent_data[s].tail(3).mean() if s in recent_data.columns else model_data['mean']
            for s in sensor_columns
        })
        
        for h in range(1, hours_ahead + 1):
            future_time = current_time + pd.Timedelta(hours=h)
            
            # Create features for future time
            features = {
                'time_numeric': h,  # hours from now
                'hour': future_time.hour,
                'day_of_week': future_time.dayofweek,
                'hour_sin': np.sin(2 * np.pi * future_time.hour / 24),
                'hour_cos': np.cos(2 * np.pi * future_time.hour / 24),
                'day_sin': np.sin(2 * np.pi * future_time.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * future_time.dayofweek / 7)
            }
            
            # Add lag features
            features.update(last_values)
            
            # Create feature vector
            X_future = np.array([[features[col] for col in feature_columns]])
            
            # Predict
            pred = model.predict(X_future)[0]
            
            # Add realistic noise and constraints
            noise_factor = min(0.1, h / 72 * 0.2)  # Increase uncertainty over time
            noise = np.random.normal(0, model_data['std'] * noise_factor)
            pred += noise
            
            # Apply constraints
            pred = max(model_data['min'] * 0.8, min(model_data['max'] * 1.2, pred))
            
            # Update lag features for next prediction
            if h > 1:
                last_values[f'{sensor}_lag2'] = last_values[f'{sensor}_lag1']
            last_values[f'{sensor}_lag1'] = pred
            
            # Update rolling mean
            if h >= 3:
                recent_preds = [future_predictions[i]['value'] for i in range(max(0, len(future_predictions)-2), len(future_predictions))]
                recent_preds.append(pred)
                last_values[f'{sensor}_rolling_mean'] = np.mean(recent_preds)
            
            future_predictions.append({
                'time': future_time,
                'value': pred,
                'hours_ahead': h,
                'confidence': max(0.5, 1 - (h / 72) * 0.4)  # Decreasing confidence over time
            })
        
        predictions[sensor] = future_predictions
    
    return predictions

def get_prediction_summary(predictions, time_windows=[6, 24, 48, 72]):
    """Get summary statistics for different time windows"""
    summary = {}
    
    for sensor, pred_list in predictions.items():
        summary[sensor] = {}
        
        for window in time_windows:
            window_preds = [p for p in pred_list if p['hours_ahead'] <= window]
            if window_preds:
                values = [p['value'] for p in window_preds]
                summary[sensor][f'{window}h'] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                    'volatility': np.std(values)
                }
    
    return summary

if __name__ == "__main__":
    print("Creating advanced 72-hour prediction models...")
    models = create_advanced_prediction_model()
    
    print("Testing 72-hour predictions...")
    predictions = predict_future_values_72h(models, 72)
    
    print("Sample predictions for first sensor:")
    first_sensor = list(predictions.keys())[0]
    for i in [0, 11, 23, 47, 71]:  # Show predictions at 1h, 12h, 24h, 48h, 72h
        if i < len(predictions[first_sensor]):
            pred = predictions[first_sensor][i]
            print(f"  {pred['hours_ahead']}h: {pred['value']:.2f} (confidence: {pred['confidence']:.2f})")
    
    summary = get_prediction_summary(predictions)
    print("Prediction summary created successfully!")

