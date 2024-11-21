import pandas as pd
import matplotlib.pyplot as plt

# Path to the RU_emg_combined.csv file
file_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined/h0/H_r2deg22h0_segmented/transformed/RU_emg_combined.csv"  # Replace with the actual file path

# Load the data
df = pd.read_csv(file_path)

# Convert 'time' to datetime format
df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')

# Split '.data' into separate columns for each channel
df[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']] = (
    df['.data']
    .str.extract(r'\(([-\d\s]+)\)')[0]  # Extract the data inside parentheses
    .str.split(' ', expand=True)  # Split by spaces into individual channels
    .apply(pd.to_numeric)  # Convert to numeric
)

# Ensure 'time' and channel columns are 1D arrays for plotting
time_values = df['time'].values  # Convert to numpy array
channel_data = [df[f'ch{i}'].values for i in range(1, 9)]  # List of 1D arrays for each channel

# Plot each channel over time
plt.figure(figsize=(12, 6))
for i, channel in enumerate(channel_data, start=1):  # Loop through channels ch1 to ch8
    plt.plot(time_values, channel, label=f'Channel {i}')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.title('EMG Signal Over Time (RU_emg_combined.csv)')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
