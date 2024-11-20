import pandas as pd
import matplotlib.pyplot as plt

# Define the file path to the combined data (update the path as necessary)
file_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_combined/H_r2deg157h0_segmented/RL_imu_combined.csv"  # Replace with RU if needed

# Load the data
df = pd.read_csv(file_path)

# Convert 'time' column to datetime for proper plotting
df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')

# Plot Orientation
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['.orientation.x'], label='Orientation X')
plt.plot(df['time'], df['.orientation.y'], label='Orientation Y')
plt.plot(df['time'], df['.orientation.z'], label='Orientation Z')
plt.plot(df['time'], df['.orientation.w'], label='Orientation W')
plt.xlabel('Time')
plt.ylabel('Orientation')
plt.title('Orientation Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Angular Velocity
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['.angular_velocity.x'], label='Angular Velocity X')
plt.plot(df['time'], df['.angular_velocity.y'], label='Angular Velocity Y')
plt.plot(df['time'], df['.angular_velocity.z'], label='Angular Velocity Z')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.title('Angular Velocity Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Linear Acceleration
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['.linear_acceleration.x'], label='Linear Acceleration X')
plt.plot(df['time'], df['.linear_acceleration.y'], label='Linear Acceleration Y')
plt.plot(df['time'], df['.linear_acceleration.z'], label='Linear Acceleration Z')
plt.xlabel('Time')
plt.ylabel('Linear Acceleration')
plt.title('Linear Acceleration Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
