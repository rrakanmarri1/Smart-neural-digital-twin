import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib

def create_prediction_model():
    """Create a simple time series prediction model"""
    # Load historical data
    df = pd.read_csv("sensor_data_simulated.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    
    # Create time-based features
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    df['time_numeric'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 3600  # hours since start
    
    # Prepare features for prediction
    features = ['time_numeric', 'hour']
    
    models = {}
    
    # Train models for each sensor type
    sensor_columns = ['Temperature (°C)', 'Pressure (psi)', 'Vibration (g)', 'Methane (CH₄ ppm)', 'H₂S (ppm)']
    
    for sensor in sensor_columns:
        # Create polynomial features for better fitting
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(df[features])
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, df[sensor])
        
        # Store model and polynomial transformer
        models[sensor] = {
            'model': model,
            'poly_features': poly_features,
            'mean': df[sensor].mean(),
            'std': df[sensor].std()
        }
    
    # Save models
    joblib.dump(models, "prediction_models.pkl")
    print("Prediction models trained and saved!")
    
    return models

def predict_future_values(models, hours_ahead=6):
    """Predict sensor values for the next few hours"""
    current_time = pd.Timestamp.now()
    predictions = {}
    
    for sensor, model_data in models.items():
        model = model_data['model']
        poly_features = model_data['poly_features']
        
        future_predictions = []
        
        for h in range(1, hours_ahead + 1):
            future_time = current_time + pd.Timedelta(hours=h)
            
            # Create features for future time
            time_numeric = h  # hours from now
            hour = future_time.hour
            
            # Transform features
            X_future = poly_features.transform([[time_numeric, hour]])
            
            # Predict
            pred = model.predict(X_future)[0]
            
            # Add some realistic noise
            noise = np.random.normal(0, model_data['std'] * 0.1)
            pred += noise
            
            future_predictions.append({
                'time': future_time,
                'value': max(0, pred),  # Ensure non-negative values
                'hours_ahead': h
            })
        
        predictions[sensor] = future_predictions
    
    return predictions

if __name__ == "__main__":
    models = create_prediction_model()
    predictions = predict_future_values(models)
    print("Sample predictions:", predictions)

