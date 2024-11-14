import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

# Base directory containing the EMG CSV files in subfolders
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

# Parameters for trimming and clustering
amplitude_threshold_factor = 0.2  # Factor of max amplitude to define stable regions
right_expansion = 0.4  # Additional expansion for the right boundary

# Traverse all subdirectories to find EMG CSV files
for root, dirs, files in os.walk(base_directory):
    rl_emg_file = None
    ru_emg_file = None

    # Identify RL_emg and RU_emg files in the current subfolder
    for file_name in files:
        if file_name.endswith('RL_emg.csv'):
            rl_emg_file = os.path.join(root, file_name)
        elif file_name.endswith('RU_emg.csv'):
            ru_emg_file = os.path.join(root, file_name)

    # Process only if both RL_emg and RU_emg files are present
    if rl_emg_file and ru_emg_file:
        print(f"Processing RL EMG file: {rl_emg_file}")
        print(f"Processing RU EMG file: {ru_emg_file}")

        # Load RL EMG data
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

        # Divide the active data into 4 equal time segments
        total_active_time = active_data['Time'].max() - active_data['Time'].min()
        segment_duration = total_active_time / 4
        clusters = []

        for i in range(4):
            # Define the start and end time for each segment
            start_time = active_data['Time'].min() + i * segment_duration
            end_time = start_time + segment_duration
            
            # Extract data within this time segment
            segment_data = active_data[(active_data['Time'] >= start_time) & (active_data['Time'] < end_time)]
            
            # Find the time range within the segment that has the highest amplitude
            max_amp_index = segment_data['Amplitude'].idxmax()
            peak_time = segment_data.loc[max_amp_index, 'Time']
            
            # Define a narrow window around the peak within the segment
            window_size = segment_duration * 0.3  # 30% of the segment duration
            peak_start_time = max(start_time, peak_time - window_size / 2)
            peak_end_time = min(end_time, peak_time + window_size / 2 + right_expansion)  # Expand right boundary

            clusters.append((peak_start_time, peak_end_time))

        # Save windows to dictionary using the RL_emg filename as the key
        rl_filename = os.path.basename(rl_emg_file)
        windows_dict[rl_filename] = clusters

        # Now apply the same time frames to RU EMG data
        ru_emg_data_raw = pd.read_csv(ru_emg_file)
        ru_emg_data_raw['_data'] = ru_emg_data_raw['_data'].apply(parse_emg_array)
        ru_emg_data = pd.DataFrame(ru_emg_data_raw['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
        ru_emg_data['Time'] = (ru_emg_data_raw['timestamp'] - ru_emg_data_raw['timestamp'].iloc[0]) / 1e9  # Convert timestamp to seconds

        # Plot RL and RU EMG data in separate subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot RL EMG data
        for channel in range(8):
            channel_name = f'RL_channel_{channel}'
            axs[0].plot(rl_emg_data['Time'], rl_emg_data[f'channel_{channel}'], label=channel_name)
        axs[0].set_ylabel('RL EMG Signal')
        axs[0].set_title(f'RL EMG Clusters Highlighted (File: {os.path.basename(rl_emg_file)})')
        axs[0].legend()

        # Plot RU EMG data
        for channel in range(8):
            channel_name = f'RU_channel_{channel}'
            axs[1].plot(ru_emg_data['Time'], ru_emg_data[f'channel_{channel}'], label=channel_name)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('RU EMG Signal')
        axs[1].set_title(f'RU EMG (File: {os.path.basename(ru_emg_file)})')
        axs[1].legend()

        # Highlight each cluster with shaded rectangles in both subplots
        for start_time, end_time in clusters:
            axs[0].axvspan(start_time, end_time, color='blue', alpha=0.1)
            axs[1].axvspan(start_time, end_time, color='blue', alpha=0.1)

        plt.show()

