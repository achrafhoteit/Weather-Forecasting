import pandas as pd
import numpy as np
from datetime import datetime

# Display all columns and rows (optional, you can limit this if needed)
pd.set_option('display.max_columns', None)  # Show all columns

# Load the weather data CSV
df = pd.read_csv("beirut_weather_dataset_6years.csv")

# Convert time to datetime object for easier handling
df['dt'] = pd.to_datetime(df['dt'])

# Set the index to the datetime column
df.set_index('dt', inplace=True) # setting the index column as date for easier queries

# Sort the data by the datetime index (in case it's not already sorted)
df.sort_index(inplace=True)

# Check the first few rows of the data to ensure it's correct
print(df.head())

# Check the first and last date in the dataset
print("First date:", df.index.min())
print("Last date:", df.index.max())
