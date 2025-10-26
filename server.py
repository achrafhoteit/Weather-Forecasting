import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize Flask app
app = Flask(__name__)

LAT, LON = 33.83, 35.83  # Coordinates for Beirut
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Get the current date and time
current_time = datetime.now()

# Calculate yesterday's date by subtracting 1 day from the current time
two_days_back = current_time - timedelta(days=2) # to make sure we get a full data (over the whole day)

# Format both start and end dates (only the date part)
end_date = two_days_back.strftime("%Y-%m-%d")  # Get yesterday's date in YYYY-MM-DD format

# Calculate the start date by subtracting 48 (2 days before) hours from the current time
start_time = two_days_back - timedelta(hours=48)

# Format both start and end dates (only the date part)
# start_date = start_time.strftime("%Y-%m-%d")  # Only year-month-day (no time)
start_date = start_time.strftime("%Y-%m-%d")  # Use the current date as the end date

# API call to get hourly weather data for the last 24 hours
url = (
    f"{API_URL}?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,cloud_cover,wind_speed_10m,wind_direction_10m"
    f"&timezone=Asia/Beirut"
)




# Function to get the weather data from Open-Meteo API for the current day
def get_feature_array_for_current_weather():

    response = requests.get(url)
    data = response.json()

    # Convert the data into a DataFrame
    hourly_data = data['hourly']
    df = pd.DataFrame({
        'timestamp': hourly_data['time'],
        'temp': hourly_data['temperature_2m'],
        'humidity': hourly_data['relative_humidity_2m'],
        'pressure': hourly_data['pressure_msl'],
        'clouds': hourly_data['cloud_cover'],
        'wind_speed': hourly_data['wind_speed_10m'],
        'wind_deg': hourly_data['wind_direction_10m']
    })

    # Convert the timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Expanding minimum temperature
    expanding_min_temp = df.groupby(df['timestamp'].dt.date)['temp'].apply(lambda x: x.expanding(1).min())

    # Expanding maximum temperature
    expanding_max_temp = df.groupby(df['timestamp'].dt.date)['temp'].apply(lambda x: x.expanding(1).max())

    # Expanding average temperature
    expanding_avg_temp = df.groupby(df['timestamp'].dt.date)['temp'].apply(lambda x: x.expanding(1).mean())

    expanding_min_temp = expanding_min_temp.reset_index(drop=True)
    expanding_max_temp = expanding_max_temp.reset_index(drop=True)
    expanding_avg_temp = expanding_avg_temp.reset_index(drop=True)

    # print(expanding_min_temp)
    # print(expanding_max_temp)
    # print(expanding_avg_temp)

    # Assign the result back to the original dataframe
    df['hourly_temp_min'] = expanding_min_temp
    df['hourly_temp_max'] = expanding_max_temp
    df['hourly_avg_temp'] = expanding_avg_temp


    # Format the datetime object to match the timestamp format in your DataFrame
    formatted_date = two_days_back.strftime('%Y-%m-%d %H:00:00')  # Example: '2025-03-14 03:00:00'

    # Now you can use this formatted date to fetch the correct row
    current_data = df[df['timestamp'] == formatted_date]
    # Extract the timestamp from current_data (it's a pandas Series)
    current_timestamp = current_data['timestamp'].values[0]
    # print(type(current_timestamp))

    # Convert numpy.datetime64 to Python datetime object if it's a numpy.datetime64
    if isinstance(current_timestamp, np.datetime64):
        current_timestamp = current_timestamp.astype('datetime64[ns]').tolist()

    # Ensure it's a datetime object (if it's an integer, handle that)
    if isinstance(current_timestamp, int):
        current_timestamp = pd.to_datetime(current_timestamp)

    # Calculate the previous hour's timestamp (lag 1)
    previous_timestamp = current_timestamp - timedelta(hours=1)

    # print(previous_timestamp)

    # Fetch the row corresponding to the previous hour
    lag_data = df[df['timestamp'] == previous_timestamp]

    # print("lag data", lag_data)

    # Extract the relevant lag values for the previous hour
    # time_lag_1 = lag_data['timestamp'].values[0]
    # temp_lag_1 = lag_data['temp'].values[0]
    # humidity_lag_1 = lag_data['humidity'].values[0]
    # pressure_lag_1 = lag_data['pressure'].values[0]
    # clouds_lag_1 = lag_data['clouds'].values[0]
    # wind_speed_lag_1 = lag_data['wind_speed'].values[0]
    # wind_deg_lag_1 = lag_data['wind_deg'].values[0]


    # Filter the data to include only rows where the timestamp is <= current_timestamp
    filtered_df = df[df['timestamp'] <= current_timestamp]

    rolling_window = 24

    # Get the last 24 rows (representing the last 24 hours before the current time)
    df_24h = filtered_df.tail(rolling_window)  # Last 24 hours

    # Calculate the rolling averages for temperature, humidity, and wind speed
    temp_rolling_avg = df_24h['temp'].mean()
    humidity_rolling_avg = df_24h['humidity'].mean()
    wind_speed_rolling_avg = df_24h['wind_speed'].mean()

    # print(filtered_df)

    # Print the rolling averages
    # print("\n24-Hour Rolling Averages:")
    # print(f"Temperature (rolling average): {temp_rolling_avg}")
    # print(f"Humidity (rolling average): {humidity_rolling_avg}")
    # print(f"Wind Speed (rolling average): {wind_speed_rolling_avg}")


    # Calculate the 3-hour rolling average for temperature
    df_3h = filtered_df.tail(3)
    rolling_avg_3hr = df_3h['temp'].mean()

    #print("rolling_avg_3hr", rolling_avg_3hr)

    # print("current data", current_data)

    # Time-based features
    # Get the hour of the day
    hour_of_day = current_data['timestamp'].dt.hour

    # Get the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = current_data['timestamp'].dt.dayofweek

    is_weekend = ((day_of_week >= 5)).values[0]
    is_morning = ((hour_of_day >= 6) & (hour_of_day < 12)).values[0]
    is_afternoon = ((hour_of_day >= 12) & (hour_of_day < 18)).values[0]
    is_evening = ((hour_of_day >= 18) & (hour_of_day < 22)).values[0]
    is_night = ((hour_of_day >= 22) | (hour_of_day < 6)).values[0]

    elapsed_time_of_day = (hour_of_day + (current_data['timestamp'].dt.minute / 60)).values[0]
    month_of_year = (current_data['timestamp'].dt.month).values[0]
    day_of_year = (current_data['timestamp'].dt.dayofyear).values[0]


    # Create an array to hold the feature values (one value for each attribute)
    feature_array = [
        current_data['temp'].values[0],               # temp
        current_data['pressure'].values[0],           # pressure
        current_data['humidity'].values[0],           # humidity
        current_data['clouds'].values[0],             # clouds
        current_data['wind_speed'].values[0],         # wind_speed
        current_data['wind_deg'].values[0],           # wind_deg

        lag_data['temp'].values[0],         # temp_lag_1
        lag_data['pressure'].values[0],     # pressure_lag_1
        lag_data['humidity'].values[0],     # humidity_lag_1
        lag_data['clouds'].values[0],       # clouds_lag_1
        lag_data['wind_speed'].values[0],   # wind_speed_lag_1
        lag_data['wind_deg'].values[0],     # wind_deg_lag_1

        rolling_avg_3hr,    # rolling_avg_3hr
        temp_rolling_avg,   # temp_rolling_avg
        humidity_rolling_avg,# humidity_rolling_avg
        wind_speed_rolling_avg, # wind_speed_rolling_avg

        (hour_of_day).values[0],
        (day_of_week).values[0],

        is_weekend,
        is_morning,
        is_afternoon,
        is_evening,
        is_night,

        elapsed_time_of_day,
        month_of_year,      # month_of_year
        day_of_year,       # day_of_year

        current_data['hourly_avg_temp'].values[0],    # hourly_avg_temp
        current_data['hourly_temp_max'].values[0],    # hourly_temp_max
        current_data['hourly_temp_min'].values[0]     # hourly_temp_min
    ]

    # print("before", feature_array)

    # Convert the list to a numpy array if needed
    feature_array = np.array(feature_array)

    return feature_array




# Example prediction function (just a mock-up for now)
def predict_weather(np_feature_array):

    # Load the pre-trained model
    model = load_model('weather_forecast_model_3targets_v3.keras')  # Provide the correct path to your saved model

    # Load the scalers used during training
    scaler_X = joblib.load('scaler2_X.pkl')  # Load the feature scaler
    scaler_y = joblib.load('scaler2_y.pkl')  # Load the target scaler

    # Reshape the feature array to match the input shape required by the model
    feature_array = np_feature_array.reshape(1, 1, -1)  # Reshaping to (1, 1, 29)

    feature_array_scaled = scaler_X.transform(feature_array.reshape(1, -1))

    # Make the prediction using the model
    prediction = model.predict(feature_array_scaled.reshape(1, 1, -1))

    predicted_values = scaler_y.inverse_transform(prediction)

    # Extracting predicted values (assuming the model outputs 18 values: temp, pressure, humidity, etc.)
    # predicted_temp = predicted_values[0][0]
    # predicted_pressure = predicted_values[0][1]
    # predicted_humidity = predicted_values[0][2]
    # predicted_clouds = predicted_values[0][3]
    # predicted_wind_speed = predicted_values[0][4]
    # predicted_wind_deg = predicted_values[0][5]

    return predicted_values


# print(two_days_back.date())
# print(two_days_back.time())
formatted_date = two_days_back.strftime('%Y-%m-%d %H:00:00')

# print(formatted_date)
array = get_feature_array_for_current_weather()

prediction = predict_weather(array)

# print(prediction)
print(f"The date right now is : {formatted_date}")

print(f"After one day, the weather will be:")
print(f"Temp: {prediction[0][0]}")
print(f"Pressure: {prediction[0][1]}")
print(f"Humidity: {prediction[0][2]}")
print(f"Clouds: {prediction[0][3]}")
print(f"Wind Speed: {prediction[0][4]}")
print(f"Wind Degree: {prediction[0][5]}")

print("\n\n")

print(f"After two days, the weather will be:")
print(f"Temp: {prediction[0][6]}")
print(f"Pressure: {prediction[0][7]}")
print(f"Humidity: {prediction[0][8]}")
print(f"Clouds: {prediction[0][9]}")
print(f"Wind Speed: {prediction[0][10]}")
print(f"Wind Degree: {prediction[0][11]}")

print("\n\n")

print(f"After three days, the weather will be:")
print(f"Temp: {prediction[0][12]}")
print(f"Pressure: {prediction[0][13]}")
print(f"Humidity: {prediction[0][14]}")
print(f"Clouds: {prediction[0][15]}")
print(f"Wind Speed: {prediction[0][16]}")
print(f"Wind Degree: {prediction[0][17]}")


