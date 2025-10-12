# Smart Neural Digital Twin

[![Python](https://img.shields.io/badge/Python-100%25-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/rrakanmarri1/Smart-neural-digital-twin)](LICENSE)
[![Issues](https://img.shields.io/github/issues/rrakanmarri1/Smart-neural-digital-twin)](https://github.com/rrakanmarri1/Smart-neural-digital-twin/issues)

## Overview

**Smart Neural Digital Twin** is an advanced Python-based platform for real-time monitoring, predictive analytics, and anomaly detection in industrial environments—especially tailored for oil field disaster prevention. Leveraging AI, dynamic sensor grids (including Raspberry Pi and I2C sensors), and a modern Streamlit dashboard, this system provides operational insight, early warning, and robust disaster mitigation capabilities.

---

## Features

- **Dynamic Sensor Detection**: Automatically detects, manages, and integrates new physical or simulated sensors at runtime.
- **Real-Time Anomaly Detection**: Uses ensemble ML models and autoencoders for fast, adaptive anomaly identification.
- **Predictive Analytics**: Multi-horizon forecasting powered by LSTM, Transformer, and Hybrid AI models.
- **Smart Dashboard**: Interactive Streamlit dashboard with real-time visualization, sensor management, and system introspection.
- **Raspberry Pi Integration**: Supports GPIO and I2C sensors for seamless physical deployment.
- **Adaptive Learning**: Monitors drift, triggers retraining, and adapts to changing sensor configurations.
- **Comprehensive Logging and Configuration**: Centralized config, structured logging, and easy-to-use UI controls.

---

## Architecture

- **core_systems.py**: Initializes the digital twin, handles sensor detection, and manages the main loop.
- **advanced_systems.py**: Implements the Streamlit dashboard and advanced UI functions.
- **ai_systems_part1.py**: Handles anomaly detection (ensemble, autoencoder, data quality).
- **ai_systems_part2.py**: Implements forecasting, drift monitoring, and adaptive learning.
- **config_and_logging.py**: Central configuration and advanced logging, with dynamic sensor registry.
- **final_app.py**: Entry point for the Streamlit app, ties all components together.

---

## Getting Started

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip (latest version)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rrakanmarri1/Smart-neural-digital-twin.git
   cd Smart-neural-digital-twin
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > On Raspberry Pi, use `sudo` if necessary and ensure I2C and GPIO are enabled.

3. **(Optional) Configure environment variables:**
   - Copy `.env.example` to `.env` and set your environment variables as needed.

4. **(Optional) Hardware Setup:**
   - Connect your physical sensors per the `sensor_pins` and `i2c_scan_addresses` in `config_and_logging.py`.

---

## Usage

### Run the Digital Twin Dashboard

```bash
streamlit run final_app.py
```

- Access the dashboard at `http://localhost:8501` in your browser.

### Key UI Features

- **Sensor Management**: Add, remove, and monitor sensors in real time.
- **System Status & Logs**: View performance metrics, logs, and configuration in the sidebar.
- **AI Insights**: Explore anomaly scores, predicted trends, and make data-driven decisions.

---

## Directory Structure

```
Smart-neural-digital-twin/
├── core_systems.py
├── advanced_systems.py
├── ai_systems_part1.py
├── ai_systems_part2.py
├── config_and_logging.py
├── final_app.py
├── requirements.txt
├── README.md
└── ...
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by modern digital twin and AI-based safety systems.
- Uses open-source libraries: Streamlit, PyTorch, scikit-learn, Adafruit CircuitPython, and more.

---
