import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Load the data
file_path = 'emg_csv_data/h0/H_r1deg45h0/H_r1deg45h0_RU_emg.csv'
data = pd.read_csv(file_path)

# Function to parse the _data column and extract array values
def parse_emg_array(data_str):
    try:
        array_str = data_str.split("array('h', ")[1].rstrip(")")
        return np.array(ast.literal_eval(array_str))
    except (ValueError, IndexError) as e:
        print(f"Error parsing data: {e}")
        return np.zeros(8)

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

    # Get data within this peak window
    cluster_data = emg_data[(emg_data['Time'] >= peak_start_time) & (emg_data['Time'] <= peak_end_time)]
    clusters.append((peak_start_time, peak_end_time, cluster_data))

# Plot the data with exactly 4 clusters highlighted
plt.figure(figsize=(12, 6))
for channel in range(8):
    channel_name = f'channel_{channel}'
    plt.plot(emg_data['Time'], emg_data[channel_name], label=channel_name)

# Highlight each cluster with rectangles
for start_time, end_time, _ in clusters:
    plt.axvspan(start_time, end_time, color='blue', alpha=0.1)

plt.xlabel('Time (s)')
plt.ylabel('EMG Signal')
plt.title('EMG Clusters Highlighted (4 Prominent Clusters, Right-Expanded Boundaries)')
plt.legend()
plt.show()
