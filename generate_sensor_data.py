
import pandas as pd
import numpy as np
import datetime

def generate_advanced_sensor_data(num_samples=1000, start_date=None):
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)

    timestamps = [start_date + datetime.timedelta(minutes=i) for i in range(num_samples)]

    # Simulate normal operating conditions with some fluctuations
    temperature = np.random.normal(loc=70, scale=2, size=num_samples)
    pressure = np.random.normal(loc=30, scale=1, size=num_samples)
    methane = np.random.normal(loc=5, scale=0.5, size=num_samples)
    h2s = np.random.normal(loc=0.5, scale=0.1, size=num_samples)
    vibration = np.random.normal(loc=0.1, scale=0.05, size=num_samples)

    # Introduce some anomalies/spikes to simulate potential disaster indicators
    # Anomaly 1: Temperature spike
    temp_spike_start = np.random.randint(100, num_samples - 200)
    temperature[temp_spike_start:temp_spike_start+50] = np.random.normal(loc=90, scale=5, size=50)

    # Anomaly 2: Pressure drop and methane increase (simulating a leak)
    leak_start = np.random.randint(temp_spike_start + 100, num_samples - 100)
    pressure[leak_start:leak_start+30] = np.random.normal(loc=20, scale=2, size=30)
    methane[leak_start:leak_start+30] = np.random.normal(loc=15, scale=3, size=30)

    # Anomaly 3: Vibration spike (simulating equipment malfunction)
    vibration_spike_start = np.random.randint(leak_start + 50, num_samples - 50)
    vibration[vibration_spike_start:vibration_spike_start+20] = np.random.normal(loc=0.8, scale=0.2, size=20)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'pressure': pressure,
        'methane': methane,
        'H2S': h2s,
        'vibration': vibration
    })
    return df

if __name__ == '__main__':
    # Generate a larger dataset for training advanced AI models
    sensor_data = generate_advanced_sensor_data(num_samples=5000)
    sensor_data.to_csv('advanced_sensor_data.csv', index=False)
    print("Generated advanced_sensor_data.csv with 5000 samples.")

    # Generate a smaller dataset for real-time simulation
    realtime_data = generate_advanced_sensor_data(num_samples=500, start_date=datetime.datetime.now() - datetime.timedelta(hours=5))
    realtime_data.to_csv('realtime_sensor_data.csv', index=False)
    print("Generated realtime_sensor_data.csv with 500 samples.")


