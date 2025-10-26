# 🌤️ Beirut Weather Forecast – Deep Learning Model

This project predicts Beirut's weather conditions for the next **three days** using a **TensorFlow Keras model** trained on historical meteorological data.  
It fetches recent hourly weather data from the **Open-Meteo API**, processes it into model-ready features, scales them, and then predicts temperature, pressure, humidity, cloud cover, and wind characteristics.

---

## 🚀 Features

- Fetches the last 48 hours of weather data for Beirut from [Open-Meteo Archive API](https://open-meteo.com/).
- Preprocesses data into a numerical feature array with time-based, lag, and rolling features.
- Uses trained scalers (`scaler2_X.pkl`, `scaler2_y.pkl`) for input/output normalization.
- Loads a pre-trained deep learning model (`weather_forecast_model_3targets_v3.keras`) for multi-day forecasting.
- Prints predictions for the next **1, 2, and 3 days** in a readable format.

---

## 🧠 Model Overview

- **Model Type:** Deep Neural Network (TensorFlow / Keras)
- **Input:** 29 engineered features (temperature, pressure, humidity, wind data, rolling averages, time-based features, etc.)
- **Output:** 18 predicted values (6 weather attributes × 3 forecast days)
- **Scalers:** StandardScaler from scikit-learn (`joblib` format)

---

## 📂 Project Structure
```bash
├── beirut_weather_dataset_6years.csv
├── beirut_weather_dataset.csv
├── model_training.py
├── notebook_preprocessing.ipynb
├── notebook_preprocessing2.ipynb
├── preprocessing.py
├── preprocessing2.py
├── processed_weather_data_6years_3targets.csv
├── processed_weather_data_6years.csv
├── processed_weather_data.csv
├── scaler_X.pkl
├── scaler_y.pkl
├── scaler2_X.pkl
├── scaler2_y.pkl
├── server.py
├── test_backend.py
├── train_model.ipynb
├── train_model2.ipynb
├── weather_forecast_model_3targets_v2.keras
├── weather_forecast_model_3targets_v3.keras
├── weather_forecast_model_v1.keras
├── weather_forecast_model_v2.keras
└── weather_forecast_model_v3.keras
```

## ⚙️ How It Works

1. **Fetch latest data**
   ```python
   response = requests.get(API_URL)
   data = response.json()
   ```

2. **Feature generation**
    - Calculates lag features (previous hour)
    - Computes 3-hour and 24-hour rolling averages
    - Adds time-based and temperature summary features

3. **Scaling**
    ```python
    scaler_X = joblib.load('scaler2_X.pkl')
    X_scaled = scaler_X.transform(features)
    ```

4. **Prediction**
    ```python
    model = load_model('weather_forecast_model_3targets_v3.keras')
    y_pred = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred)
    ```

## 🧪 Running Locally

### Prerequisites

Install dependencies:
```bash
pip install flask tensorflow scikit-learn pandas numpy requests joblib
```

### Run the server
```bash
python server.py
```

## Example Output
The date right now is : 2025-03-14 03:00:00

After one day, the weather will be:
Temp: 21.5
Pressure: 1013.2
Humidity: 62.4
Clouds: 47.0
Wind Speed: 4.1
Wind Degree: 210.0

After two days, the weather will be:
...

After three days, the weather will be:
...