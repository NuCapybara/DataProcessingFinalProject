import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np

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

    # Calculate sampling rate
    time_intervals = np.diff(df['time'].values)
    avg_time_interval = np.mean(time_intervals)
    sampling_rate = 1 / avg_time_interval  # in Hz
    print(f"Estimated Sampling Rate: {sampling_rate:.2f} Hz")
    
    # Define the minimum distance between peaks in terms of samples
    min_peak_distance = int(4 * sampling_rate)  # Minimum 4 seconds apart

    # Balance each S1 to S6 to a base of 0 by removing the average offset after time > 40 seconds
    for i in range(1, 7):
        offset = df.loc[df['time'] > 60, f'S{i}'].mean()  # Use the average value after time > 40 seconds as the offset
        df[f'S{i}'] = df[f'S{i}'] - offset
        print(f"Offset for S{i}: {offset}")

    # Specify sensors of interest
    sensors_of_interest = ['S1', 'S3']

    # Threshold for near-zero filtering
    near_zero_threshold = 0.1  # Adjust as necessary based on your data scale

    # Analyze each sensor
    for sensor in sensors_of_interest:
        sensor_data = df[sensor].values

        # Filter out near-zero values
        mask = np.abs(sensor_data) > near_zero_threshold
        filtered_time_data = df['time'].values[mask]
        filtered_sensor_data = sensor_data[mask]

        # Smooth the data using a Gaussian filter
        smoothed_data = gaussian_filter1d(filtered_sensor_data, sigma=2)

        # Define a minimum peak height based on the observed smallest true peak
        min_peak_height = 50000  # Adjust this threshold based on your data's scale

        # Find peaks with the specified minimum distance, prominence, and height
        peaks, _ = find_peaks(
            smoothed_data, 
            distance=min_peak_distance, 
            prominence=0.5, 
            height=min_peak_height  # Only peaks with values above this height will be detected
        )

        # Ensure we have at least 4 peaks; otherwise, skip
        if len(peaks) < 4:
            print(f"Insufficient peaks found in {sensor} for segmentation.")
            continue
        
        # Keep only the first four peaks for segmentation
        peaks = peaks[:4]

        # Extend the segmentation range to capture full peaks
        segments = []
        for i, peak in enumerate(peaks):
            # Define start and end boundaries for each segment
            if i == 0:
                # For the first peak, start from the beginning of the dataset
                start = 0
            else:
                # Start at the midpoint between the current peak and the previous peak
                start = (peaks[i - 1] + peak) // 2

            if i == len(peaks) - 1:
                # For the last peak, go until the end of the dataset
                end = len(filtered_sensor_data)
            else:
                # End at the midpoint between the current peak and the next peak
                end = (peak + peaks[i + 1]) // 2

            # Extract segment within the start and end boundaries
            segment_time = filtered_time_data[start:end]
            segment_data = filtered_sensor_data[start:end]
            
            segments.append((segment_time, segment_data))

        # For the fourth segment, truncate data after stabilization
        if len(segments) >= 4:
            fourth_segment_time, fourth_segment_data = segments[3]

            # Define stabilization criteria: continuous values within [-500, 500]
            stable_lower = -300
            stable_upper = 300
            stabilization_window = int(20 * sampling_rate)  # e.g., 20 seconds of stable data

            # Find stabilization point only after the fourth peak is fully included
            cutoff_index = len(fourth_segment_data)  # Default to full length if no stabilization point found
            for idx in range(len(fourth_segment_data) - stabilization_window):
                # Define a window from the current point to stabilization_window length
                window = fourth_segment_data[idx:idx + stabilization_window]
                if np.all((window >= stable_lower) & (window <= stable_upper)):
                    cutoff_index = idx  # Mark where stabilization begins
                    break

            # Truncate the fourth segment after the stabilization point
            fourth_segment_time = fourth_segment_time[:cutoff_index]
            fourth_segment_data = fourth_segment_data[:cutoff_index]

            # Update the fourth segment in the list
            segments[3] = (fourth_segment_time, fourth_segment_data)

        # Plot each segment separately
        plt.figure(figsize=(10, 6))
        for idx, (time_seg, data_seg) in enumerate(segments):
            plt.plot(time_seg, data_seg, label=f'Segment {idx + 1}')

        plt.title(f"{sensor} - Segmented by Peaks - {os.path.basename(file_path)}")
        plt.xlabel("Time (s)")
        plt.ylabel(f"{sensor} Value (Balanced)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Directory containing the .mat files
directory_path = 'ArmCubeDatah0'

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {file_path}")
        load_and_plot_data(file_path)
    else:
        continue
