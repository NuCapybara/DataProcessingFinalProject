import os
import pandas as pd
import numpy as np
from scipy.signal import decimate

def process_robot_velocity(robot_file, emg_length):
    # Step 1: Load robot data
    df_robot = pd.read_csv(robot_file)
    samples_per_repeat = emg_length // 4  # Calculate target length for one repetition

    # Step 2: Extract joint velocities and timestamps
    velocity_cols = [col for col in df_robot.columns if 'velocity' in col]  # Joint velocity columns
    timestamps = df_robot['Timestamp'].values

    # Step 3: Low-pass filter and downsample for velocities
    filtered_velocity_data = {}
    for col in velocity_cols:
        # Filter and downsample velocity data
        filtered_velocity_data[col] = decimate(df_robot[col].values, max(1, len(df_robot) // samples_per_repeat), ftype='iir')
        filtered_velocity_data[col] = filtered_velocity_data[col][:samples_per_repeat]  # Truncate to ensure exact length

    # Step 4: Build downsampled DataFrame for velocities
    downsampled_velocities = pd.DataFrame(filtered_velocity_data)
    downsampled_velocities['Timestamp'] = np.linspace(
        timestamps[0],
        timestamps[-1] / 4,  # Divide total time by 4
        len(downsampled_velocities)
    )

    # Step 5: Repeat the downsampled velocity data 4 times
    repeated_velocities = pd.concat([downsampled_velocities] * 4, ignore_index=True)

    # Step 6: Adjust timestamps for continuity
    repeated_velocities['Timestamp'] = np.linspace(
        timestamps[0],
        timestamps[-1],  # Extend to the original total duration
        len(repeated_velocities)
    )

    # Step 7: Add static points to match final length
    while len(repeated_velocities) < emg_length:
        repeated_velocities = pd.concat([repeated_velocities, repeated_velocities.iloc[-1:].copy()], ignore_index=True)

    return repeated_velocities[:emg_length]  # Ensure exact match to emg_length

def automate_velocity_processing(emg_dir, robot_dir, velocity_output_dir):
    for root, dirs, files in os.walk(emg_dir):
        for subdir in dirs:
            emg_folder = os.path.join(root, subdir)
            robot_folder_name = subdir.replace('H_', 'R_').replace('_segmented', '')  # Match robot folder name convention
            robot_folder = os.path.join(robot_dir, robot_folder_name)

            if not os.path.exists(robot_folder):
                print(f"Skipping: {robot_folder} does not exist")
                continue

            # Find EMG combined file
            emg_file = os.path.join(emg_folder, 'RL_emg_combined.csv')
            if not os.path.exists(emg_file):
                print(f"Skipping: {emg_file} does not exist")
                continue

            # Read EMG data to determine target length
            emg_length = sum(1 for _ in open(emg_file)) - 1  # Subtract header line

            # Process each robot file in the robot folder
            for robot_file in os.listdir(robot_folder):
                if robot_file.endswith('.csv'):
                    robot_file_path = os.path.join(robot_folder, robot_file)
                    processed_velocities = process_robot_velocity(robot_file_path, emg_length)

                    # Save the processed robot velocity data
                    output_velocity_file = os.path.join(velocity_output_dir, f"processed_velocities_{robot_file}")
                    os.makedirs(velocity_output_dir, exist_ok=True)
                    processed_velocities.to_csv(output_velocity_file, index=False)

                    print(f"Processed and saved: {output_velocity_file}")

# Define paths
emg_dir = '/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_smooth_data'
robot_dir = '/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/cropped_data'
velocity_output_dir = '/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/processed_robot_velocity_data'

# Run automation
automate_velocity_processing(emg_dir, robot_dir, velocity_output_dir)
