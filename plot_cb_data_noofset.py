import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the function to load and plot data from a given file
def load_and_plot_data(file_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Update the key based on the structure of your .mat file
    data_key = 'data'  # Replace with the correct key if it's different
    if data_key not in mat_data:
        print(f"Key '{data_key}' not found in {file_path}")
        return
    
    data = mat_data[data_key]

    # Convert to DataFrame and set column names
    df = pd.DataFrame(data, columns=[
        'Ax', 'Ay', 'Az',  # Accelerometer data
        'Gx', 'Gy', 'Gz',  # Gyroscope data
        'Mx', 'My', 'Mz',  # Magnetometer data
        'S1', 'S2', 'S3', 'S4', 'S5', 'S6',  # Sensor/Force data
        'time'  # Timestamp column
    ])

    # Convert 'time' column to seconds if necessary
    df['time'] = (df['time'] - df['time'].min()) / 1000  # Adjust if time is in milliseconds

    # Ensure each column is in 1D array format for plotting
    time_data = df['time'].values
    # ax_data = df['Ax'].values
    # ay_data = df['Ay'].values
    # az_data = df['Az'].values

    # # Plot Accelerometer data (Ax, Ay, Az)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_data, ax_data, label='Ax')
    # plt.plot(time_data, ay_data, label='Ay')
    # plt.plot(time_data, az_data, label='Az')
    # plt.title(f"Accelerometer Data over Time - {os.path.basename(file_path)}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (m/s²)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot Gyroscope data (Gx, Gy, Gz)
    # gx_data = df['Gx'].values
    # gy_data = df['Gy'].values
    # gz_data = df['Gz'].values
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_data, gx_data, label='Gx')
    # plt.plot(time_data, gy_data, label='Gy')
    # plt.plot(time_data, gz_data, label='Gz')
    # plt.title(f"Gyroscope Data over Time - {os.path.basename(file_path)}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angular Velocity (rad/s)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot Magnetometer data (Mx, My, Mz)
    # mx_data = df['Mx'].values
    # my_data = df['My'].values
    # mz_data = df['Mz'].values
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_data, mx_data, label='Mx')
    # plt.plot(time_data, my_data, label='My')
    # plt.plot(time_data, mz_data, label='Mz')
    # plt.title(f"Magnetometer Data over Time - {os.path.basename(file_path)}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Magnetic Field (µT)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Plot Sensor/Force Data (S1 to S6)
    plt.figure(figsize=(12, 6))
    for i in range(1, 7):
        sensor_data = df[f'S{i}'].values
        plt.plot(time_data, sensor_data, label=f'S{i}')
    plt.title(f"Sensor/Force Data over Time - {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force or Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Directory containing the .mat files
directory_path = 'ArmCubeDatah1'

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {file_path}")
        load_and_plot_data(file_path)
    else:
        continue 