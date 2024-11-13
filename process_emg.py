import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import ruptures as rpt
import re

# Load the CSV file
df = pd.read_csv('emg_csv_data/h0/H_r1deg0h0/H_r1deg0h0_RL_emg.csv')

# Convert timestamp to seconds for easier interpretation (optional)
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

# Define a function to parse the _data field
def parse_emg_data(data_str):
    # Use regex to find all integers in the string
    return list(map(int, re.findall(r"-?\d+", data_str)))

# Apply parsing function to _data column
emg_data = df['_data'].apply(parse_emg_data)

# Expand parsed EMG data into separate columns
emg_df = pd.DataFrame(emg_data.tolist(), columns=[f'channel_{i}' for i in range(len(emg_data.iloc[0]))])

# Concatenate timestamp and expanded EMG data
df = pd.concat([df['timestamp'], emg_df], axis=1)

# Define a function to apply a bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Filter parameters
lowcut = 20.0    # Lower cutoff frequency in Hz
highcut = 450.0  # Upper cutoff frequency in Hz
fs = 1000.0      # Sampling frequency in Hz (adjust based on your data)

# Select a specific channel for analysis, e.g., channel_1
channel_list = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
for channel in channel_list:
    signal = emg_df[channel].values

    # Apply bandpass filter to the selected channel
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs)

    # Apply change point detection using the ruptures library
    # Create a model instance for change point detection with a different model and lower penalty
    model = "l2"  # Using "l2" model instead of "rbf" for detecting mean changes
    algo = rpt.Pelt(model=model).fit(filtered_signal)

    # Try with a lower penalty value
    penalty = 5  # Lower penalty for more sensitivity
    change_points = algo.predict(pen=penalty)

    # Print change points to see if any were found
    print(f"Detected change points: {change_points}")

    # Filter out change points that exceed the signal length
    valid_change_points = [cp for cp in change_points if cp < len(filtered_signal)]

    # Plot the original signal with detected change points
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], filtered_signal, label='Filtered Signal', color='b')

    # Plot detected change points if any were found
    if len(valid_change_points) > 0:  # Make sure there are detected points within a valid range
        for cp in valid_change_points:
            plt.axvline(x=df['timestamp'].iloc[cp], color='r', linestyle='--', label='Change Point' if cp == valid_change_points[0] else "")

    plt.title("Change Point Detection in EMG Data (Filtered) - {channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("EMG Signal (Filtered)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
