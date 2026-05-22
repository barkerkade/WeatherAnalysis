import pandas as pd

def load_and_clean_data(filepath):
    print("--- Loading and Cleaning Data ---")
    # Load data and ensure 'Date' is parsed as actual datetime objects
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Check for missing values
    print("Missing values before cleaning:\n", df.isnull().sum())
    
    # Fill missing temperature values using forward-fill (takes the previous day's temp)
    df['Temperature_C'] = df['Temperature_C'].ffill()
    
    # Set Date as the index — crucial for time-series operations
    df.set_index('Date', inplace=True)
    return df

def extract_time_features(df):
    # Extract structural components out of the index for grouping
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    return df

def perform_statistical_analysis(df):
    print("\n--- Basic Summary Statistics ---")
    print(df[['Temperature_C', 'Humidity_Pct', 'Rainfall_mm']].describe())
    
    print("\n--- Monthly Averages (Climate Trends) ---")
    # Group by month to see seasonal shifts across all years
    monthly_trends = df.groupby('Month')[['Temperature_C', 'Rainfall_mm']].mean()
    print(monthly_trends.round(2))
    
    print("\n--- Annual Rainfall Totals ---")
    annual_rain = df.groupby('Year')['Rainfall_mm'].sum()
    print(annual_rain)
    
    return monthly_trends

def calculate_moving_averages(df):
    print("\n--- Calculating Rolling Indicators ---")
    # A 7-day moving average smoothes out short-term spikes to show the true trend
    df['Temp_7Day_MA'] = df['Temperature_C'].rolling(window=7, min_periods=1).mean()
    
    # Identify anomalies: Days where temperature was 3 standard deviations away from monthly mean
    df['Month_Mean'] = df.groupby('Month')['Temperature_C'].transform('mean')
    df['Month_Std'] = df.groupby('Month')['Temperature_C'].transform('std')
    df['Temp_ZScore'] = (df['Temperature_C'] - df['Month_Mean']) / df['Month_Std']
    
    anomalies = df[df['Temp_ZScore'].abs() > 3]
    print(f"Detected {len(anomalies)} extreme temperature anomalies.")
    return df, anomalies

if __name__ == "__main__":
    # Execute the full pipeline
    weather_df = load_and_clean_data('weather_data.csv')
    weather_df = extract_time_features(weather_df)
    monthly_summary = perform_statistical_analysis(weather_df)
    weather_df, extreme_days = calculate_moving_averages(weather_df)