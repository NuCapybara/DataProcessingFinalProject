import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import re

# Base directory containing the EMG and IMU CSV files in subfolders
base_directory = 'emg_csv_data/h0/'

# Dictionary to store time windows for each RL_emg file
windows_dict = {}

# Function to parse the _data column and extract array values
def parse_emg_array(data_str):
    try:
        array_str = data_str.split("array('h', ")[1].rstrip(")")
        return np.array(ast.literal_eval(array_str))
    except (ValueError, IndexError) as e:
        print(f"Error parsing data: {e}")
        return np.zeros(8)

# Functions to parse IMU data
def parse_quaternion(quaternion_str):
    match = re.findall(r"-?\d+\.\d+", quaternion_str)
    return list(map(float, match))

def parse_vector3(vector3_str):
    match = re.findall(r"-?\d+\.\d+", vector3_str)
    return list(map(float, match))

# Parameters for trimming and clustering
amplitude_threshold_factor = 0.2  # Factor of max amplitude to define stable regions
right_expansion = 0.4  # Additional expansion for the right boundary

# Traverse all subdirectories to find EMG and IMU CSV files
for root, dirs, files in os.walk(base_directory):
    rl_emg_file = None
    ru_emg_file = None
    rl_imu_file = None
    ru_imu_file = None
    print("lalala")
    # Identify RL_emg, RU_emg, RL_imu, and RU_imu files in the current subfolder
    for file_name in files:
        if file_name.endswith('RL_emg.csv'):
            rl_emg_file = os.path.join(root, file_name)
        elif file_name.endswith('RU_emg.csv'):
            ru_emg_file = os.path.join(root, file_name)
        elif file_name.endswith('RL_imu.csv'):
            rl_imu_file = os.path.join(root, file_name)
        elif file_name.endswith('RU_imu.csv'):
            ru_imu_file = os.path.join(root, file_name)

    # Process only if all RL_emg, RU_emg, RL_imu, and RU_imu files are present
    if rl_emg_file and ru_emg_file and rl_imu_file and ru_imu_file:
        print(f"Processing RL EMG file: {rl_emg_file}")
        print(f"Processing RU EMG file: {ru_emg_file}")
        print(f"Processing RL IMU file: {rl_imu_file}")
        print(f"Processing RU IMU file: {ru_imu_file}")

        # Load RL EMG data and identify clusters
        rl_emg_data_raw = pd.read_csv(rl_emg_file)
        if '_data' not in rl_emg_data_raw.columns:
            print(f"Skipping {rl_emg_file} as it does not contain '_data' column.")
            continue

        rl_emg_data_raw['_data'] = rl_emg_data_raw['_data'].apply(parse_emg_array)
        rl_emg_data = pd.DataFrame(rl_emg_data_raw['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
        rl_emg_data['Time'] = (rl_emg_data_raw['timestamp'] - rl_emg_data_raw['timestamp'].iloc[0]) / 1e9  # Convert timestamp to seconds
        rl_emg_data['Amplitude'] = rl_emg_data[[f'channel_{i}' for i in range(8)]].abs().mean(axis=1)

        # Set amplitude threshold dynamically based on data
        amplitude_threshold = amplitude_threshold_factor * rl_emg_data['Amplitude'].max()

        # Trim the stable regions at the beginning and end
        active_data = rl_emg_data[rl_emg_data['Amplitude'] > amplitude_threshold]
        trimmed_start_time = active_data['Time'].iloc[0]
        trimmed_end_time = active_data['Time'].iloc[-1]
        active_data = rl_emg_data[(rl_emg_data['Time'] >= trimmed_start_time) & (rl_emg_data['Time'] <= trimmed_end_time)]

        # Divide the active data into 4 equal time segments and identify clusters
        total_active_time = active_data['Time'].max() - active_data['Time'].min()
        segment_duration = total_active_time / 4
        clusters = []

        for i in range(4):
            start_time = active_data['Time'].min() + i * segment_duration
            end_time = start_time + segment_duration
            segment_data = active_data[(active_data['Time'] >= start_time) & (active_data['Time'] < end_time)]
            max_amp_index = segment_data['Amplitude'].idxmax()
            peak_time = segment_data.loc[max_amp_index, 'Time']
            window_size = segment_duration * 0.3
            peak_start_time = max(start_time, peak_time - window_size / 2)
            peak_end_time = min(end_time, peak_time + window_size / 2 + right_expansion)
            clusters.append((peak_start_time, peak_end_time))

        # Save windows to dictionary
        rl_filename = os.path.basename(rl_emg_file)
        windows_dict[rl_filename] = clusters

        # Load RU EMG data
        ru_emg_data_raw = pd.read_csv(ru_emg_file)
        ru_emg_data_raw['_data'] = ru_emg_data_raw['_data'].apply(parse_emg_array)
        ru_emg_data = pd.DataFrame(ru_emg_data_raw['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
        ru_emg_data['Time'] = (ru_emg_data_raw['timestamp'] - ru_emg_data_raw['timestamp'].iloc[0]) / 1e9

        # Load and process RL IMU data
        rl_imu_data_raw = pd.read_csv(rl_imu_file)
        rl_imu_data_raw['timestamp'] = (rl_imu_data_raw['timestamp'] - rl_imu_data_raw['timestamp'].min()) / 1e9
        rl_orientation = rl_imu_data_raw['_orientation'].apply(parse_quaternion)
        rl_angular_velocity = rl_imu_data_raw['_angular_velocity'].apply(parse_vector3)
        rl_linear_acceleration = rl_imu_data_raw['_linear_acceleration'].apply(parse_vector3)
        rl_orientation_df = pd.DataFrame(rl_orientation.tolist(), columns=['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
        rl_angular_velocity_df = pd.DataFrame(rl_angular_velocity.tolist(), columns=['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
        rl_linear_acceleration_df = pd.DataFrame(rl_linear_acceleration.tolist(), columns=['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
        rl_imu_data = pd.concat([rl_imu_data_raw['timestamp'], rl_orientation_df, rl_angular_velocity_df, rl_linear_acceleration_df], axis=1)

        # Load and process RU IMU data
        ru_imu_data_raw = pd.read_csv(ru_imu_file)
        ru_imu_data_raw['timestamp'] = (ru_imu_data_raw['timestamp'] - ru_imu_data_raw['timestamp'].min()) / 1e9
        ru_orientation = ru_imu_data_raw['_orientation'].apply(parse_quaternion)
        ru_angular_velocity = ru_imu_data_raw['_angular_velocity'].apply(parse_vector3)
        ru_linear_acceleration = ru_imu_data_raw['_linear_acceleration'].apply(parse_vector3)
        ru_orientation_df = pd.DataFrame(ru_orientation.tolist(), columns=['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
        ru_angular_velocity_df = pd.DataFrame(ru_angular_velocity.tolist(), columns=['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
        ru_linear_acceleration_df = pd.DataFrame(ru_linear_acceleration.tolist(), columns=['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
        ru_imu_data = pd.concat([ru_imu_data_raw['timestamp'], ru_orientation_df, ru_angular_velocity_df, ru_linear_acceleration_df], axis=1)


        # Add these print statements for debugging
        print("RL EMG Data:", rl_emg_data.head())         # Check if data is loaded and parsed
        print("RU EMG Data:", ru_emg_data.head())
        print("RL IMU Data:", rl_imu_data.head())
        print("RU IMU Data:", ru_imu_data.head())
        print("Clusters (start and end times):", clusters)  # Check if clusters are calculated correctly

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

        # Plot RL EMG data
        for channel in range(8):
            axs[0].plot(rl_emg_data['Time'], rl_emg_data[f'channel_{channel}'], label=f'RL_channel_{channel}')
        axs[0].set_ylabel('RL EMG Signal')
        axs[0].legend()

        # Plot RU EMG data
        for channel in range(8):
            axs[1].plot(ru_emg_data['Time'], ru_emg_data[f'channel_{channel}'], label=f'RU_channel_{channel}')
        axs[1].set_ylabel('RU EMG Signal')
        axs[1].legend()

        # Plot RL IMU Linear Acceleration

        for col in rl_linear_acceleration_df.columns:
            axs[2].plot(rl_imu_data['timestamp'], rl_imu_data[col], label=f'RL_{col}')
        axs[2].set_ylabel('RL IMU Linear Acceleration (m/s^2)')
        axs[2].legend()

        # Plot RU IMU Linear Acceleration
        for col in ru_linear_acceleration_df.columns:
            axs[3].plot(ru_imu_data['timestamp'], ru_imu_data[col], label=f'RU_{col}')
        axs[3].set_ylabel('RU IMU Linear Acceleration (m/s^2)')
        axs[3].set_xlabel('Time (s)')
        axs[3].legend()

        # Apply the same cluster windows from RL to all subplots
        for start_time, end_time in clusters:
            axs[0].axvspan(start_time, end_time, color='blue', alpha=0.1)
            axs[1].axvspan(start_time, end_time, color='blue', alpha=0.1)
            axs[2].axvspan(start_time, end_time, color='blue', alpha=0.1)
            axs[3].axvspan(start_time, end_time, color='blue', alpha=0.1)

        # Set titles for clarity
        axs[0].set_title(f'RL EMG Data with Clusters Highlighted (File: {os.path.basename(rl_emg_file)})')
        axs[1].set_title(f'RU EMG Data with Clusters Highlighted (File: {os.path.basename(ru_emg_file)})')
        axs[2].set_title(f'RL IMU Linear Acceleration with Clusters Highlighted (File: {os.path.basename(rl_imu_file)})')
        axs[3].set_title(f'RU IMU Linear Acceleration with Clusters Highlighted (File: {os.path.basename(ru_imu_file)})')

        plt.tight_layout()
        plt.show()

