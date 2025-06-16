# Smart Neural Digital Twin - Industrial Monitoring System

## Overview
An advanced neural digital twin solution for industrial equipment monitoring, featuring real-time analytics, AI-powered anomaly detection, and predictive maintenance capabilities.

## 🌟 Key Features
- **Real-time Monitoring**: Live sensor data visualization with interactive charts
- **AI Anomaly Detection**: Automatic detection of abnormal equipment behavior
- **Predictive Maintenance**: Advanced analytics for equipment health assessment
- **3D Equipment Visualization**: Interactive 3D model of industrial equipment
- **Bilingual Support**: English/Arabic interface
- **Dark Mode**: Optimized for extended monitoring sessions
- **Interactive Maps**: View sensor locations on an interactive Mapbox map
- **PDF/CSV Export**: Download recent readings as CSV or PDF reports
- **User Authentication**: Streamlit based login with role support
- **Toast Alerts**: Instant notifications for critical values
- **Docker & CI**: Dockerfile and GitHub Actions for automated testing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
### Installation
```bash
# Clone the repository
git clone [https://github.com/rrakanmarri1/Smart-neural-digital-twin.git](https://github.com/rrakanmarri1/Smart-neural-digital-twin.git)
cd Smart-neural-digital-twin

# Install dependencies
pip install -r requirements.txt

### Running
```bash
# Launch the application
streamlit run final_app.py

# Run tests
pytest

# Build container
docker build -t digital-twin .
```
