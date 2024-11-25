import os
import pandas as pd
import numpy as np

def process_all_robot_emg_files(robot_dir_base, emg_dir_base, output_dir_base):
    """
    Process all robot and EMG files across directories (h0_segmented and h1_segmented).
    """
    # Iterate over both h0 and h1 segmented directories
    for emg_segment in ["h0_segmented", "h1_segmented"]:
        emg_dir = os.path.join(emg_dir_base, emg_segment)
        
        for emg_folder in os.listdir(emg_dir):
            emg_folder_path = os.path.join(emg_dir, emg_folder)
            
            # Skip if not a directory
            if not os.path.isdir(emg_folder_path):
                continue

            # Generate the corresponding robot directory name
            robot_folder_name = emg_folder.replace("H_", "R_").replace("_segmented", "")
            robot_folder_path = os.path.join(robot_dir_base, robot_folder_name)

            # Check if robot folder exists
            if not os.path.exists(robot_folder_path) or not os.path.isdir(robot_folder_path):
                print(f"Robot folder not found for {robot_folder_name}")
                continue

            # Find all robot CSV files in the corresponding robot folder
            robot_files = [
                os.path.join(robot_folder_path, f)
                for f in os.listdir(robot_folder_path)
                if f.endswith(".csv")
            ]

            # EMG combined file path
            emg_file_path = os.path.join(emg_folder_path, "RL_emg_combined.csv")

            # Ensure the EMG file exists
            if not os.path.exists(emg_file_path):
                print(f"EMG file not found: {emg_file_path}")
                continue

            # Output directory for processed robot data
            output_subdir = os.path.join(output_dir_base, emg_segment, robot_folder_name)
            os.makedirs(output_subdir, exist_ok=True)

            # Process each robot file
            for robot_file in robot_files:
                # Construct output file path
                output_file_name = os.path.basename(robot_file).replace(".csv", "_processed.csv")
                output_file_path = os.path.join(output_subdir, output_file_name)

                # Process the robot file to match the length of the EMG data
                process_robot_data(robot_file, emg_file_path, output_file_path)


def process_robot_data(robot_csv_path, emg_csv_path, output_path):
    """
    Process robot data to match the length of EMG data by downsampling and repeating.
    """
    # Load robot and EMG data
    robot_data = pd.read_csv(robot_csv_path)
    emg_data = pd.read_csv(emg_csv_path)
    
    # Target length calculation
    target_length = len(emg_data) // 4  # Downsample to 1/4th of EMG data length

    # Downsample robot data
    robot_data_downsampled = downsample_data(robot_data, target_length)

    # Repeat robot data 4 times and adjust timestamps
    final_robot_data = repeat_and_adjust_timestamps(robot_data_downsampled, len(emg_data))

    # Save the final robot data to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_robot_data.to_csv(output_path, index=False)
    print(f"Processed robot data saved to: {output_path}")


def downsample_data(data, target_length):
    """
    Downsample data to the target length using linear interpolation.
    """
    # Generate indices for target length
    indices = np.linspace(0, len(data) - 1, target_length)
    downsampled_data = data.iloc[indices.astype(int)].reset_index(drop=True)
    return downsampled_data


def repeat_and_adjust_timestamps(data, final_length):
    """
    Repeat data 4 times and adjust timestamps for continuity.
    """
    repeated_data = pd.DataFrame()
    total_segments = 4
    segment_length = final_length // total_segments

    for i in range(total_segments):
        # Copy the data for this segment
        segment_data = data.copy()

        # Adjust timestamps for continuity
        if i > 0:
            time_offset = repeated_data['timestamp'].iloc[-1] - segment_data['timestamp'].iloc[0] + 1
            segment_data['timestamp'] += time_offset

        repeated_data = pd.concat([repeated_data, segment_data], ignore_index=True)

    # Ensure the final length matches exactly
    return repeated_data.iloc[:final_length]


# Directories
robot_dir_base = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/cropped_data"
emg_dir_base = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_data"
output_dir_base = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/processed_data"

# Run the automation
process_all_robot_emg_files(robot_dir_base, emg_dir_base, output_dir_base)
