import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import re

# Base directory containing the segmented EMG and IMU CSV files
base_directory = 'emg_csv_data/h1_segmented/'
output_directory = 'emg_csv_data/h1_seg_plot'
os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

# Parsing Functions
def parse_emg_array(data_str):
    try:
        if pd.isna(data_str):
            return np.zeros(8)  # Default value for missing data
        array_str = data_str.split("array('h', ")[1].rstrip(")")
        return np.array(ast.literal_eval(array_str))
    except (ValueError, IndexError) as e:
        print(f"Error parsing EMG data: {e}")
        return np.zeros(8)

def parse_quaternion(quaternion_str):
    try:
        if pd.isna(quaternion_str):
            return [0, 0, 0, 1]  # Default quaternion
        match = re.findall(r"-?\d+\.\d+", quaternion_str)
        return list(map(float, match))
    except Exception as e:
        print(f"Error parsing quaternion: {e}")
        return [0, 0, 0, 1]

def parse_vector3(vector3_str):
    try:
        if pd.isna(vector3_str):
            return [0, 0, 0]  # Default vector
        match = re.findall(r"-?\d+\.\d+", vector3_str)
        return list(map(float, match))
    except Exception as e:
        print(f"Error parsing Vector3: {e}")
        return [0, 0, 0]

# Process Each Subfolder
for root, dirs, files in os.walk(base_directory):
    # Group files by trial and type
    trial_files = {}
    for file_name in files:
        if file_name.endswith('.csv'):
            # Extract trial and type information
            match = re.match(r"(H_.*?_seg\d+)\.csv", file_name)
            if match:
                trial_name = match.group(1).rsplit("_", 1)[0]  # Extract base trial name
                trial_files.setdefault(trial_name, []).append(file_name)

    # Process Each Trial
    for trial_name, segmented_files in trial_files.items():
        print(f"Processing trial: {trial_name}")

        # Create output folder for this trial
        trial_output_directory = os.path.join(output_directory, trial_name)
        os.makedirs(trial_output_directory, exist_ok=True)

        # Separate EMG and IMU Files
        emg_files = [f for f in segmented_files if '_emg_' in f]
        imu_files = [f for f in segmented_files if '_imu_' in f]

        # Plot EMG Data
        for emg_file in emg_files:
            emg_file_path = os.path.join(root, emg_file)
            emg_data_raw = pd.read_csv(emg_file_path)
            emg_data_raw['_data'] = emg_data_raw['_data'].apply(parse_emg_array)
            emg_data = pd.DataFrame(emg_data_raw['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
            emg_data['Time'] = (emg_data_raw['timestamp'] - emg_data_raw['timestamp'].iloc[0]) / 1e9  # Convert to seconds

            # Plot EMG Channels
            plt.figure(figsize=(14, 8))
            for channel in range(8):
                plt.plot(emg_data['Time'], emg_data[f'channel_{channel}'], label=f'Channel {channel}')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal Amplitude')
            plt.title(f"EMG Data: {emg_file}")
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_path = os.path.join(trial_output_directory, f"{os.path.splitext(emg_file)[0]}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved EMG plot for {emg_file} to {plot_path}")

        # Plot IMU Data
        for imu_file in imu_files:
            imu_file_path = os.path.join(root, imu_file)
            imu_data_raw = pd.read_csv(imu_file_path)
            imu_data_raw['timestamp'] = (imu_data_raw['timestamp'] - imu_data_raw['timestamp'].min()) / 1e9  # Convert to seconds
            linear_acceleration = imu_data_raw['_linear_acceleration'].apply(parse_vector3)
            linear_acceleration_df = pd.DataFrame(linear_acceleration.tolist(), columns=['x', 'y', 'z'])
            imu_data = pd.concat([imu_data_raw['timestamp'], linear_acceleration_df], axis=1)

            # Plot IMU Linear Acceleration
            plt.figure(figsize=(14, 8))
            plt.plot(imu_data['timestamp'], imu_data['x'], label='x')
            plt.plot(imu_data['timestamp'], imu_data['y'], label='y')
            plt.plot(imu_data['timestamp'], imu_data['z'], label='z')
            plt.xlabel('Time (s)')
            plt.ylabel('Linear Acceleration (m/s^2)')
            plt.title(f"IMU Data: {imu_file}")
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_path = os.path.join(trial_output_directory, f"{os.path.splitext(imu_file)[0]}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved IMU plot for {imu_file} to {plot_path}")
