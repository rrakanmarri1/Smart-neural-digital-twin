
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath=\'advanced_sensor_data.csv\', sequence_length=60):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")

    # Define features (sensor readings)
    features = [\'temperature\', \'pressure\', \'methane\', \'H2S\', \'vibration\']
    data = df[features].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        # For simplicity, let's try to predict the next 'vibration' as an anomaly indicator
        # In a real scenario, 'y' would be a more complex target (e.g., risk level, anomaly flag)
        y.append(scaled_data[i+sequence_length, features.index(\'vibration\')]) 

    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, features

if __name__ == \'__main__\':
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Data preprocessing complete.")


