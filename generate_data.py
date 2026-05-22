import pandas as pd
import numpy as np

# Generate 3 years of daily dates
date_range = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")

# Simulate temperature with a seasonal sinusoidal wave + some noise
np.random.seed(42)
base_temp = 15 + 10 * np.sin(2 * np.pi * date_range.dayofyear / 365)
temp_noise = np.random.normal(0, 3, size=len(date_range))
temperature = base_temp + temp_noise

# Simulate humidity and rainfall
humidity = np.clip(np.random.normal(65, 15, size=len(date_range)), 30, 100)
rainfall = np.random.exponential(scale=2, size=len(date_range))
rainfall = np.where(rainfall < 1.5, 0, rainfall)  # Make some days completely dry

# Build the DataFrame
df_mock = pd.DataFrame({
    'Date': date_range,
    'Temperature_C': temperature,
    'Humidity_Pct': humidity,
    'Rainfall_mm': rainfall
})

# Inject a few missing values on purpose for data-cleaning practice
df_mock.loc[df_mock.sample(frac=0.01).index, 'Temperature_C'] = np.nan

# Save to CSV
df_mock.to_csv('weather_data.csv', index=False)
print("Created 'weather_data.csv' successfully!")