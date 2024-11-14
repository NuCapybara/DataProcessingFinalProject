import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

# Base directory containing the EMG and IMU CSV files in subfolders
base_directory = 'emg_csv_data/h0/'

# Function to parse the _data column and extract array values
def parse_emg_array(data_str):
    try:
        array_str = data_str.split("array('h', ")[1].rstrip(")")
        return np.array(ast.literal_eval(array_str))
    except (ValueError, IndexError) as e:
        print(f"Error parsing data: {e}")
        return np.zeros(8)

# Traverse all subdirectories to find EMG and IMU CSV files
for root, dirs, files in os.walk(base_directory):
    emg_file = None
    imu_file = None

    # Identify EMG and IMU files in the current subfolder
    for file_name in files:
        if file_name.endswith('_emg.csv'):
            emg_file = os.path.join(root, file_name)
        elif file_name.endswith('_imu.csv'):
            imu_file = os.path.join(root, file_name)

    # Process only if both EMG and IMU files are present
    if emg_file and imu_file:
        print(f"Processing EMG file: {emg_file}")
        print(f"Processing IMU file: {imu_file}")

        # Load EMG data
        emg_data_raw = pd.read_csv(emg_file)
        if '_data' not in emg_data_raw.columns:
            print(f"Skipping {emg_file} as it does not contain '_data' column.")
            continue

        emg_data_raw['_data'] = emg_data_raw['_data'].apply(parse_emg_array)
        emg_data = pd.DataFrame(emg_data_raw['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
        emg_data['Time'] = (emg_data_raw['timestamp'] - emg_data_raw['timestamp'].iloc[0]) / 1e9  # Convert timestamp to seconds
        emg_data['Amplitude'] = emg_data[[f'channel_{i}' for i in range(8)]].abs().mean(axis=1)

        # Identify 4 prominent clusters in EMG data
        total_time = emg_data['Time'].max() - emg_data['Time'].min()
        segment_duration = total_time / 4
        clusters = []
        right_expansion = 0.4  # Extend the right boundary by 0.4 seconds

        for i in range(4):
            start_time = emg_data['Time'].min() + i * segment_duration
            end_time = start_time + segment_duration
            segment_data = emg_data[(emg_data['Time'] >= start_time) & (emg_data['Time'] < end_time)]

            max_amp_index = segment_data['Amplitude'].idxmax()
            peak_time = segment_data.loc[max_amp_index, 'Time']
            window_size = segment_duration * 0.3
            peak_start_time = max(start_time, peak_time - window_size / 2)
            peak_end_time = min(end_time, peak_time + window_size / 2 + right_expansion)
            clusters.append((peak_start_time, peak_end_time))

        # Load IMU data
        imu_data = pd.read_csv(imu_file)
        imu_data['Time'] = (imu_data['timestamp'] - imu_data['timestamp'].iloc[0]) / 1e9  # Convert timestamp to seconds

        # Plot EMG and IMU data for each cluster window
        for idx, (start_time, end_time) in enumerate(clusters):
            emg_window = emg_data[(emg_data['Time'] >= start_time) & (emg_data['Time'] <= end_time)]
            imu_window = imu_data[(imu_data['Time'] >= start_time) & (imu_data['Time'] <= end_time)]

            # Plot EMG data
            plt.figure(figsize=(14, 6))
            for channel in range(8):
                channel_name = f'channel_{channel}'
                plt.plot(emg_window['Time'], emg_window[channel_name], label=f'EMG {channel_name}')
            plt.xlabel('Time (s)')
            plt.ylabel('EMG Signal')
            plt.title(f'EMG Data - Cluster {idx + 1} (Time: {start_time:.2f}s to {end_time:.2f}s)')
            plt.legend()
            plt.show()

            # Plot IMU data (assuming IMU has columns named 'accel_x', 'accel_y', 'accel_z' for accelerometer data)
            plt.figure(figsize=(14, 6))
            if 'accel_x' in imu_window.columns and 'accel_y' in imu_window.columns and 'accel_z' in imu_window.columns:
                plt.plot(imu_window['Time'], imu_window['accel_x'], label='IMU Accel X')
                plt.plot(imu_window['Time'], imu_window['accel_y'], label='IMU Accel Y')
                plt.plot(imu_window['Time'], imu_window['accel_z'], label='IMU Accel Z')
            else:
                print("IMU data does not contain expected accelerometer columns.")
            plt.xlabel('Time (s)')
            plt.ylabel('IMU Signal')
            plt.title(f'IMU Data - Cluster {idx + 1} (Time: {start_time:.2f}s to {end_time:.2f}s)')
            plt.legend()
            plt.show()
