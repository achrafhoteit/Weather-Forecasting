import requests

# Define the data for today's weather
data = {
    'temp': 25,  # Example temperature
    'humidity': 60,  # Example humidity
    'windSpeed': 5,  # Example wind speed
}

# Send POST request to the Flask backend
response = requests.post('http://localhost:5000/predict', json=data)

# Check the response
if response.status_code == 200:
    predictions = response.json()
    print("Predicted weather for the next 3 days:")
    for i, prediction in enumerate(predictions):
        print(f"Day {i + 1}: Temp: {prediction['temp']}°C, Humidity: {prediction['humidity']}%, Wind Speed: {prediction['wind_speed']} m/s")
else:
    print(f"Error: {response.status_code}")
