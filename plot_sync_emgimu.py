import pandas as pd
import matplotlib.pyplot as plt
import os


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_imu_field(field_str):
    """
    Parse IMU fields (e.g., geometry_msgs.msg.Vector3(x=..., y=..., z=...)).
    """
    start_idx = field_str.find("(") + 1
    end_idx = field_str.find(")")
    values = field_str[start_idx:end_idx].split(",")
    return [float(val.split("=")[1]) for val in values]

def plot_emg_imu_detailed(emg_file, imu_file):
    # Load synchronized EMG and IMU data
    emg_data = pd.read_csv(emg_file)
    imu_data = pd.read_csv(imu_file)
    
    # Extract timestamps (convert nanoseconds to seconds for better readability)
    emg_time = emg_data['timestamp'].values / 1e9
    imu_time = imu_data['timestamp'].values / 1e9

    # Extract EMG values (convert _data column to a usable format)
    emg_values = emg_data['_data'].apply(lambda x: eval(x.replace("array('h',", "").replace(")", ""))).tolist()

    # Extract IMU data (full details for linear acceleration)
    imu_linear_acceleration = imu_data['_linear_acceleration'].apply(parse_imu_field).tolist()
    imu_linear_acceleration = np.array(imu_linear_acceleration)  # Convert to NumPy array for easier handling

    # Plot all EMG channels
    plt.figure(figsize=(14, 8))
    for i in range(len(emg_values[0])):  # Number of EMG channels
        emg_channel = [val[i] for val in emg_values]
        plt.plot(emg_time, emg_channel, label=f'EMG Channel {i + 1}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EMG Signal Value')
    plt.title('Detailed EMG Signal Over Time')
    plt.legend()
    plt.grid()
    # plt.savefig(os.path.join(output_plot_path, "emg_detailed_plot.png"))
    # print(f"EMG plot saved to: {os.path.join(output_plot_path, 'emg_detailed_plot.png')}")
    plt.show()

    # Plot IMU linear acceleration (x, y, z)
    plt.figure(figsize=(14, 8))
    plt.plot(imu_time, imu_linear_acceleration[:, 0], label='IMU Linear Acceleration X', linestyle='-')
    plt.plot(imu_time, imu_linear_acceleration[:, 1], label='IMU Linear Acceleration Y', linestyle='-')
    plt.plot(imu_time, imu_linear_acceleration[:, 2], label='IMU Linear Acceleration Z', linestyle='-.')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Linear Acceleration (m/s^2)')
    plt.title('IMU Linear Acceleration Over Time')
    plt.legend()
    plt.grid()
    # plt.savefig(os.path.join(output_plot_path, "imu_detailed_plot.png"))
    # print(f"IMU plot saved to: {os.path.join(output_plot_path, 'imu_detailed_plot.png')}")
    plt.show()

# Example usage
emg_file_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_smooth_data/h0_segmented/H_r1deg0h0_segmented/RL_emg_combined.csv"  # Replace with actual synchronized EMG file path
imu_file_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_smooth_data/h0_segmented/H_r1deg0h0_segmented/RL_imu_combined.csv"  # Replace with actual synchronized IMU file path
# output_plot_path = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Sync_Data_EMGIMU/h0_segmented/H_r1deg0h0_segmented"  # Replace with desired output plot path

plot_emg_imu_detailed(emg_file_path, imu_file_path)

