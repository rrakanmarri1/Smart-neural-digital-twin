import pandas as pd
from generate_sensor_data import generate_advanced_sensor_data

def test_generate_sensor_data():
    df = generate_advanced_sensor_data(num_samples=5)
    assert len(df) == 5
    expected_cols = {"timestamp", "temperature", "pressure", "methane", "H2S", "vibration"}
    assert expected_cols.issubset(df.columns)
