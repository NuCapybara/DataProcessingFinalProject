import os
import pandas as pd
from scipy.signal import resample
import numpy as np

def parse_emg_data(data_str):
    """
    Parse EMG _data column (e.g., "array('h', [...])").
    """
    return list(eval(data_str.replace("array('h',", "").replace(")", "")))

def downsample_to_shortest(trail_folder):
    """
    Adjust all segment pairs in a trail to match the shortest segment's length.
    """
    emg_files = []
    imu_files = []

    # Collect EMG and IMU file paths
    for file_name in os.listdir(trail_folder):
        if "_emg_seg" in file_name:
            emg_files.append(os.path.join(trail_folder, file_name))
        elif "_imu_seg" in file_name:
            imu_files.append(os.path.join(trail_folder, file_name))

    # Sort to align corresponding EMG and IMU files
    emg_files.sort()
    imu_files.sort()

    # Verify matching segment pairs
    assert len(emg_files) == len(imu_files), f"Mismatched EMG and IMU segments in {trail_folder}"

    # Determine the shortest segment length
    segment_lengths = []
    for emg_file, imu_file in zip(emg_files, imu_files):
        emg_data = pd.read_csv(emg_file)
        imu_data = pd.read_csv(imu_file)
        segment_lengths.append(len(emg_data))  # Assuming EMG and IMU are synchronized
    shortest_length = min(segment_lengths)

    print(f"Shortest segment length in {trail_folder}: {shortest_length}")

    # Downsample all segments to the shortest length
    for emg_file, imu_file in zip(emg_files, imu_files):
        emg_data = pd.read_csv(emg_file)
        imu_data = pd.read_csv(imu_file)

        # Downsample EMG data
        emg_data['_data'] = emg_data['_data'].apply(parse_emg_data)  # Parse the EMG _data column
        emg_data_resampled = resample(
            np.array(emg_data['_data'].tolist()), shortest_length, axis=0
        )
        emg_data_resampled_df = emg_data.iloc[:shortest_length].copy()  # Slice DataFrame to shortest length
        emg_data_resampled_df['_data'] = [f"array('h', {list(map(int, row))})" for row in emg_data_resampled]

        # Downsample IMU data
        imu_data_resampled_df = imu_data.iloc[:shortest_length].copy()

        # Save the downsampled data
        emg_data_resampled_df.to_csv(emg_file, index=False)
        imu_data_resampled_df.to_csv(imu_file, index=False)

        print(f"Downsampled and saved: {emg_file}, {imu_file}")

# Process all trails
base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Sync_Data_EMGIMU/h1_segmented"
for trail_folder_name in os.listdir(base_dir):
    trail_folder_path = os.path.join(base_dir, trail_folder_name)
    if os.path.isdir(trail_folder_path):
        downsample_to_shortest(trail_folder_path)
