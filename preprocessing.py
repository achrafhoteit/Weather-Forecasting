import pandas as pd
import numpy as np
from datetime import datetime

# Display all columns and rows (optional, you can limit this if needed)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows (be careful if the dataset is large)

# Load the weather data CSV
df = pd.read_csv("beirut_weather_dataset.csv")

# Convert time to datetime object for easier handling
df['dt'] = pd.to_datetime(df['dt'])

# Set the index to the datetime column
df.set_index('dt', inplace=True) # setting the index column as date for easier queries

# Check the first few rows of the data to ensure it's correct
print(df.head())

# Extract additional time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear

# Add lag features (previous hour's temperature, humidity, wind speed, etc.)
df['temp_lag_1'] = df['temp'].shift(1)  # Previous hour's temperature
df['pressure_lag_1'] = df['pressure'].shift(1)  # Previous hour's pressure
df['humidity_lag_1'] = df['humidity'].shift(1)  # Previous hour's humidity
df['clouds_lag_1'] = df['clouds'].shift(1)  # Previous hour's cloud cover
df['wind_speed_lag_1'] = df['wind_speed'].shift(1)  # Previous hour's wind speed
df['wind_deg_lag_1'] = df['wind_deg'].shift(1)  # Previous hour's wind direction


# Rolling statistics (e.g., 24-hour rolling mean for temperature)
df['temp_rolling_avg'] = df['temp'].rolling(window=24).mean()  # 24-hour rolling mean for temperature
df['humidity_rolling_avg'] = df['humidity'].rolling(window=24).mean()  # 24-hour rolling mean for humidity
df['wind_speed_rolling_avg'] = df['wind_speed'].rolling(window=24).mean()  # 24-hour rolling mean for wind speed

# Drop rows with NaN values caused by lag/rolling features
df.dropna(inplace=True)

# Create target variables for the next day (shift by 24 hours)
df['temp_next_day'] = df['temp'].shift(-24)  # Shift by 24 hours to predict next day's temperature
df['pressure_next_day'] = df['pressure'].shift(-24)  # Same for pressure
df['humidity_next_day'] = df['humidity'].shift(-24)  # Same for humidity
df['clouds_next_day'] = df['clouds'].shift(-24)  # Same for clouds
df['wind_speed_next_day'] = df['wind_speed'].shift(-24)  # Same for wind speed
df['wind_deg_next_day'] = df['wind_deg'].shift(-24)  # Same for wind direction

# Drop rows with NaN values caused by the shift (the last 24 rows will be NaN)
df.dropna(subset=['temp_next_day', 'pressure_next_day', 'humidity_next_day', 
                  'clouds_next_day', 'wind_speed_next_day', 'wind_deg_next_day'], inplace=True)

# Check the new features
print(df.head())

# Save the DataFrame with all features to a CSV file
df.to_csv("processed_weather_data.csv", index=False)

print("Data saved to 'processed_weather_data.csv'")

