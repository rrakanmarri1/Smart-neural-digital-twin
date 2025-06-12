import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import datetime

# Function to train and save prediction models for each sensor
def train_and_save_prediction_models(df, features, target_columns, model_path="advanced_prediction_models.pkl"):
    models = {}
    scalers = {}
    
    for col in target_columns:
        # Prepare data
        X = df[features].values
        y = df[col].values
        
        # Scale features
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Train a simple RandomForestRegressor for demonstration
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        models[col] = model
        scalers[col] = scaler_X
        
    joblib.dump({"models": models, "scalers": scalers}, model_path)
    print(f"Prediction models saved to {model_path}")
    return models, scalers

# Function to predict future values for 72 hours
def predict_future_values_72h(historical_df, trained_models_data, hours=72):
    models = trained_models_data["models"]
    scalers = trained_models_data["scalers"]
    
    last_data_point = historical_df.iloc[-1]
    future_timestamps = [last_data_point["Time"] + datetime.timedelta(hours=i) for i in range(1, hours + 1)]
    
    future_data = []
    for i in range(hours):
        # For simplicity, we use the last known values as features for prediction
        # In a more complex model, you might use predicted values as features for subsequent predictions
        features_for_prediction = np.array([[last_data_point["temperature"], last_data_point["pressure"], last_data_point["methane"], last_data_point["H2S"], last_data_point["vibration"]]])
        
        predicted_values = {}
        for col, model in models.items():
            scaler_X = scalers[col]
            scaled_features = scaler_X.transform(features_for_prediction)
            predicted_values[col] = model.predict(scaled_features)[0]
            
        # Update last_data_point for the next prediction step (simple autoregressive)
        last_data_point = {**last_data_point, **predicted_values}
        last_data_point["Time"] = future_timestamps[i]
        future_data.append(last_data_point)
        
    future_df = pd.DataFrame(future_data)
    future_df = future_df[["Time"] + list(models.keys())] # Ensure order
    return future_df

# Function to get a summary of future predictions and potential risks
def get_prediction_summary(future_predictions_df):
    summary = ""
    
    # Overall trends
    summary += "**ملخص التنبؤات لسلوك المستشعرات على مدى الـ 72 ساعة القادمة:**\n"
    for col in future_predictions_df.columns[1:-1]: # Exclude Time and total_risk
        max_val = future_predictions_df[col].max()
        min_val = future_predictions_df[col].min()
        avg_val = future_predictions_df[col].mean()
        summary += f"- **{col.capitalize()}**: تتراوح بين {min_val:.2f} و {max_val:.2f} بمتوسط {avg_val:.2f}.\n"
        
    # Risk summary
    high_risk_events = future_predictions_df[future_predictions_df["total_risk"] >= 0.7]
    if not high_risk_events.empty:
        summary += f"\n- **تحذيرات حرجة:** تم توقع {len(high_risk_events)} حدثًا حرجًا. أول حدث متوقع عند {high_risk_events["Time"].iloc[0].strftime("%Y-%m-%d %H:%M")}.\n"
    else:
        summary += "\n- **لا توجد تحذيرات حرجة** متوقعة خلال الـ 72 ساعة القادمة.\n"
        
    return summary

# Example of how to train models (this part would typically run once)
if __name__ == "__main__":
    # Load your simulated historical data
    try:
        df = pd.read_csv("sensor_data_simulated.csv")
        df["Time"] = pd.to_datetime(df["Time"])
        
        features = [col for col in df.columns if col != "Time"]
        target_columns = features # Predicting all sensor values
        
        train_and_save_prediction_models(df, features, target_columns)
        
        # Test prediction
        trained_data = joblib.load("advanced_prediction_models.pkl")
        future_df = predict_future_values_72h(df, trained_data)
        print("\nFuture Predictions (first 5 rows):\n", future_df.head())
        print("\nFuture Predictions (last 5 rows):\n", future_df.tail())
        
    except FileNotFoundError:
        print("Error: sensor_data_simulated.csv not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
