import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed weather data from the CSV file
df = pd.read_csv("processed_weather_data.csv")

# Features (X) and multiple target variables (y)
X = df[['temp_lag_1', 'humidity_lag_1', 'wind_speed_lag_1', 'temp_rolling_avg', 'hour', 
        'day_of_week', 'month', 'day_of_year']]
y = df[['temp_next_day', 'pressure_next_day', 'humidity_next_day', 'clouds_next_day', 
        'wind_speed_next_day', 'wind_deg_next_day']]

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)  # 80% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check the shape of the training and testing sets
print(X_train.shape, X_test.shape)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model for each target variable
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")