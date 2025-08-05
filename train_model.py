import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Đọc dữ liệu
energy_url = 'https://raw.githubusercontent.com/phantuan1311/DA_with_Python/refs/heads/main/energy_dataset.csv'
energy_data = pd.read_csv(energy_url)
energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)

# Load filtered_data đã xử lý trước
filtered_data = pd.read_csv("filtered_data.csv")
filtered_data['time'] = pd.to_datetime(filtered_data['time'], utc=True)

# Merge, tạo feature thời gian
merged = pd.merge(filtered_data, energy_data[['time', 'price actual', 'generation solar',
                                               'generation wind onshore', 'generation fossil hard coal',
                                               'generation hydro pumped storage consumption']],
                  on='time', how='left')
merged['hour'] = merged['time'].dt.hour
merged['day_of_week'] = merged['time'].dt.dayofweek
merged['month'] = merged['time'].dt.month
if 'temp_c' not in merged.columns:
    merged['temp_c'] = merged['temp'] - 273.15
merged.fillna(merged.mean(numeric_only=True), inplace=True)

# Feature và target
X = merged[['temp_c', 'humidity', 'pressure', 'wind_speed', 'rain_1h', 'snow_3h', 'clouds_all',
            'hour', 'day_of_week', 'month', 'price actual',
            'generation solar', 'generation wind onshore',
            'generation fossil hard coal', 'generation hydro pumped storage consumption']]
y = merged['total load actual']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train with lightweight config
rf = RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.4f}")

# Save scaler + model
joblib.dump(scaler, "scaler.pkl", compress=3)
joblib.dump(rf, "random_forest.pkl", compress=3)
print("Đã lưu scaler.pkl và random_forest.pkl")
