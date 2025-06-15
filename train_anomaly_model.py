
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

def train_anomaly_detection_model(filepath="advanced_sensor_data.csv"):
    df = pd.read_csv(filepath)
    features = ["temperature", "pressure", "methane", "H2S", "vibration"]
    data = df[features]

    # Train Isolation Forest model
    model = IsolationForest(random_state=42, contamination=0.05) # contamination is the proportion of outliers in the data set
    model.fit(data)

    # Save the model
    joblib.dump(model, "isolation_forest_model.pkl")
    print("Isolation Forest model trained and saved as isolation_forest_model.pkl")

    # Predict anomalies on the training data to get anomaly scores
    df["anomaly_score"] = model.decision_function(data)
    df["anomaly"] = model.predict(data) # -1 for outliers, 1 for inliers

    # Determine a threshold for anomaly detection (e.g., based on the 5th percentile of scores)
    # A lower score indicates a higher likelihood of being an anomaly
    anomaly_threshold = df["anomaly_score"] .quantile(0.05)
    print(f"Anomaly score threshold: {anomaly_threshold}")
    
    return anomaly_threshold

if __name__ == "__main__":
    train_anomaly_detection_model()


