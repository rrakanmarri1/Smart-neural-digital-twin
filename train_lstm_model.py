
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# Function to preprocess data (copied from preprocess_data.py for self-containment)
def preprocess_data(filepath="advanced_sensor_data.csv", sequence_length=60):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")

    features = ["temperature", "pressure", "methane", "H2S", "vibration"]
    data = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, features.index("vibration")]) # Predicting vibration

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, features

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output a single value (predicted vibration)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data()

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Callbacks for training
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("best_lstm_model.keras", save_best_only=True, monitor="val_loss", mode="min")

    print("Training LSTM model...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping, model_checkpoint], verbose=1)

    # Save the scaler for later use in prediction
    joblib.dump(scaler, "scaler.pkl")
    print("LSTM model trained and saved as best_lstm_model.keras")
    print("Scaler saved as scaler.pkl")


