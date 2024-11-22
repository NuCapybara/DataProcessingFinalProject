import pandas as pd
import matplotlib.pyplot as plt

# Define the file path to the combined data
file_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_scale/h0/H_r2deg22h0_segmented/RL_imu_combined_formatted.csv"

# Load the data
df = pd.read_csv(file_path)

# Remove leading dots from column names
df.columns = df.columns.str.lstrip('.')

# Convert 'time' column to datetime for proper plotting
df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')

# Check for NaN values and ensure data alignment
columns_to_check = [
    'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
    'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
    'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z'
]
df = df.dropna(subset=['time'] + columns_to_check)

# Debugging information
print(df[['time', 'orientation.x']].head())
print(df[['time', 'orientation.x']].isnull().sum())

# Plot Orientation
plt.figure(figsize=(12, 6))
plt.plot(df['time'].values, df['orientation.x'].values, label='Orientation X')
plt.plot(df['time'].values, df['orientation.y'].values, label='Orientation Y')
plt.plot(df['time'].values, df['orientation.z'].values, label='Orientation Z')
plt.plot(df['time'].values, df['orientation.w'].values, label='Orientation W')
plt.xlabel('Time')
plt.ylabel('Orientation')
plt.title('Orientation Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Angular Velocity
plt.figure(figsize=(12, 6))
plt.plot(df['time'].values, df['angular_velocity.x'].values, label='Angular Velocity X')
plt.plot(df['time'].values, df['angular_velocity.y'].values, label='Angular Velocity Y')
plt.plot(df['time'].values, df['angular_velocity.z'].values, label='Angular Velocity Z')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.title('Angular Velocity Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Linear Acceleration
plt.figure(figsize=(12, 6))
plt.plot(df['time'].values, df['linear_acceleration.x'].values, label='Linear Acceleration X')
plt.plot(df['time'].values, df['linear_acceleration.y'].values, label='Linear Acceleration Y')
plt.plot(df['time'].values, df['linear_acceleration.z'].values, label='Linear Acceleration Z')
plt.xlabel('Time')
plt.ylabel('Linear Acceleration')
plt.title('Linear Acceleration Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
