import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Load the dataset
file_path = 'C:\\Users\\johan\\TDT4259_Data_Science\\consumption_temp.csv'  # Replace with your file path
df = pd.read_csv(file_path)


# Filter data for Oslo
df_oslo = df[df['location'] == 'oslo'].copy()

# Convert 'time' to datetime and sort
df_oslo['time'] = pd.to_datetime(df_oslo['time'])
df_oslo.sort_values('time', inplace=True)

# Generate time-based features
df_oslo['hour'] = df_oslo['time'].dt.hour
df_oslo['day_of_week'] = df_oslo['time'].dt.dayofweek
df_oslo['month'] = df_oslo['time'].dt.month

# Prepare features and target
X = df_oslo[['temperature', 'hour', 'day_of_week', 'month']]
y = df_oslo['consumption']

# Initialize some variables for the rolling forecast
window_size = 24  # Define the size of the test window (e.g., 30 days)
step_size = 24  # Define the step size for rolling (e.g., 7 days)
n_splits = int((len(X) - window_size) / step_size)

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Store RMSE for each split
rmse_list = []
mae_list = []

# Rolling Forecast Origin loop
for i in range(n_splits):
    train_end = i * step_size
    test_start = train_end + 1
    test_end = test_start + window_size
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_list.append(mae)
    print(f"Split {i+1}, MAE: {mae}")
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_list.append(rmse)
    print(f"Split {i+1}, RMSE: {rmse}")

# Final model training on the entire dataset
xgb_model.fit(X, y)

# Average RMSE
print("Average RMSE across all splits:", np.mean(rmse_list))
print("Average MAE across all splits:", np.mean(mae_list))
