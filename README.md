# Crop Disease Prediction using Machine Learning

## 🌟 Description
This project is an intelligent crop monitoring system that combines deep learning, machine learning, and IoT sensors for early disease detection and environmental risk prediction. It helps farmers monitor crop health, detect diseases, and take preventive actions.

---

## 🛠️ Technologies & Tools Used
* **Programming Language**: Python
* **Web Framework**: Flask
* **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras, XGBoost, Joblib, OpenCV (`opencv-python`)
* **Deep Learning & Machine Learning Models**: 
  * CNN with VGG16 (Image-based disease detection)
  * LSTM (Time-series environmental forecasting)
  * XGBoost (Disease risk classification)
  * Autoencoder (Anomaly detection for abnormal environmental conditions)
* **Database**: SQLite

---

## 🔌 Hardware Used
* **Microcontroller**: ESP32
* **Sensors & Modules**:
  * DHT11 Sensor (Temperature and humidity)
  * Soil Moisture Sensor (Soil water content)
  * LDR Sensor (Light intensity)
  * NPK Sensor (Soil nutrients: Nitrogen, Phosphorus, Potassium)
  * GPS Module (Location tracking)

---

## 📦 Required Model Files
Ensure the following files are present in your project directory:
* `hybrid_cnn_vgg16_final.h5`
* `lstm_sensor_forecast.h5`
* `sensor_autoencoder.h5`
* `sensor_scaler.pkl`
* `xgb_risk_model.json`

---

## 🔄 System Architecture & Workflow
* **IoT Telemetry & Data Collection**: The ESP32 hardware connects to physical sensors, continuously collecting microclimatic and soil parameters and transmitting live data to the backend at regular intervals.
* **Image Diagnostics**: Users upload a crop leaf image through the Flask web interface, which is preprocessed and analyzed by the CNN-VGG16 model to detect specific plant diseases.
* **Multimodal Risk Assessment**: The system simultaneously processes real-time environmental data using the LSTM model for time-series forecasting, the XGBoost model to predict risk levels (Low, Medium, High), and the Autoencoder to spot environmental anomalies.
* **Interactive Dashboard**: The frontend displays comprehensive output including disease name, prediction confidence, risk level, forecast results, graphical visualizations, multilingual text output, and voice alerts. All data is recorded in the database for ongoing monitoring.

---

## 🚀 Execution Steps

1. **Install Dependencies**: Open your terminal and install all required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras xgboost flask joblib opencv-python

**Run the Application:** Open a terminal in the project directory and start the Flask backend server:python app.py
