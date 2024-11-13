import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the function to load, modify, and plot data from a given file
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

    # Balance each S1 to S6 to a base of 0 by removing the average offset after time > 40 seconds
    for i in range(1, 7):
        offset = df.loc[df['time'] > 60, f'S{i}'].mean()  # Use the average value after time > 40 seconds as the offset
        df[f'S{i}'] = df[f'S{i}'] - offset
        print(f"Offset for S{i}: {offset}")


    # Ensure each column is in 1D array format for plotting
    time_data = df['time'].values
    # Plot Sensor/Force Data (S1 to S6)
    plt.figure(figsize=(12, 6))
    plotList = [4, 1, 3]
    for i in plotList:
        sensor_data = df[f'S{i}'].values
        plt.plot(time_data, sensor_data, label=f'S{i}')


    plt.title(f"Sensor/Force Data over Time - {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force or Sensor Value (Balanced)")
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
