import pandas as pd
import numpy as np
import os
import ast

# Base directory containing the EMG CSV files in subfolders
emg_base_directory = 'emg_csv_data/h0/'
output_directory = 'output_emg_windows/'  # Directory to save window times

# Create output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Function to parse the _data column and extract array values
def parse_emg_array(data_str):
    try:
        array_str = data_str.split("array('h', ")[1].rstrip(")")
        return np.array(ast.literal_eval(array_str))
    except (ValueError, IndexError) as e:
        print(f"Error parsing data: {e}")
        return np.zeros(8)

# Traverse all subdirectories to find EMG CSV files
for root, dirs, files in os.walk(emg_base_directory):
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            print(f"Processing EMG file: {file_path}")

            # Load the EMG data
            data = pd.read_csv(file_path)

            # Check if '_data' column exists
            if '_data' not in data.columns:
                print(f"Skipping file {file_name} as it does not contain '_data' column.")
                continue

            # Apply parsing function to create EMG data for each channel
            data['_data'] = data['_data'].apply(parse_emg_array)
            emg_data = pd.DataFrame(data['_data'].tolist(), columns=[f'channel_{i}' for i in range(8)])
            emg_data['Time'] = (data['timestamp'] - data['timestamp'].iloc[0]) / 1e9  # Convert timestamp to seconds

            # Calculate the overall amplitude (magnitude) across all channels
            emg_data['Amplitude'] = emg_data[[f'channel_{i}' for i in range(8)]].abs().mean(axis=1)

            # Divide the data into 4 equal time segments
            total_time = emg_data['Time'].max() - emg_data['Time'].min()
            segment_duration = total_time / 4

            clusters = []
            right_expansion = 0.4  # 0.4 seconds extension on the right side

            # Record start and end times for each window
            for i in range(4):
                # Define the start and end time for each segment
                start_time = emg_data['Time'].min() + i * segment_duration
                end_time = start_time + segment_duration

                # Extract data within this time segment
                segment_data = emg_data[(emg_data['Time'] >= start_time) & (emg_data['Time'] < end_time)]

                # Find the time range within the segment that has the highest amplitude sum
                max_amp_index = segment_data['Amplitude'].idxmax()
                peak_time = segment_data.loc[max_amp_index, 'Time']

                # Define a narrow window around the peak within the segment
                window_size = segment_duration * 0.3  # 30% of the segment duration
                peak_start_time = max(start_time, peak_time - window_size / 2)
                peak_end_time = min(end_time, peak_time + window_size / 2 + right_expansion)  # Expand right boundary

                clusters.append((peak_start_time, peak_end_time))

            # Save the cluster windows to a CSV file
            output_file = os.path.join(output_directory, f"{file_name}_windows.csv")
            window_df = pd.DataFrame(clusters, columns=['start_time', 'end_time'])
            window_df.to_csv(output_file, index=False)
            print(f"Saved EMG window times to {output_file}")
