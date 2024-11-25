import os
import pandas as pd

def combine_imu_segments(input_dir, output_dir):
    """
    Combine IMU segments into RL_imu_combined.csv and RU_imu_combined.csv for each trail folder.
    """
    for root, dirs, files in os.walk(input_dir):
        for trail_dir in dirs:
            trail_path = os.path.join(root, trail_dir)
            
            # Collect RL and RU segment files
            rl_files = sorted(
                [
                    os.path.join(trail_path, f)
                    for f in os.listdir(trail_path)
                    if "RL_imu_seg" in f and f.endswith(".csv")
                ]
            )
            ru_files = sorted(
                [
                    os.path.join(trail_path, f)
                    for f in os.listdir(trail_path)
                    if "RU_imu_seg" in f and f.endswith(".csv")
                ]
            )
            
            # Combine RL IMU files
            if rl_files:
                combine_and_save_segments(rl_files, trail_path, input_dir, output_dir, "RL_imu_combined.csv")
            else:
                print(f"No RL IMU segments found in {trail_path}")

            # Combine RU IMU files
            if ru_files:
                combine_and_save_segments(ru_files, trail_path, input_dir, output_dir, "RU_imu_combined.csv")
            else:
                print(f"No RU IMU segments found in {trail_path}")


def combine_and_save_segments(segment_files, trail_path, input_dir, output_dir, output_filename):
    """
    Combine segment files into a single file with continuous timestamps.
    """
    combined_imu = pd.DataFrame()
    previous_end_time = None

    for segment_file in segment_files:
        imu_data = pd.read_csv(segment_file)

        # Adjust timestamps for continuity
        if previous_end_time is not None:
            time_offset = previous_end_time - imu_data['timestamp'].iloc[0] + 1
            imu_data['timestamp'] += time_offset

        previous_end_time = imu_data['timestamp'].iloc[-1]
        combined_imu = pd.concat([combined_imu, imu_data], ignore_index=True)

    # Create the corresponding output subdirectory
    relative_path = os.path.relpath(trail_path, input_dir)
    output_trail_path = os.path.join(output_dir, relative_path)
    os.makedirs(output_trail_path, exist_ok=True)

    # Save the combined IMU data
    output_file = os.path.join(output_trail_path, output_filename)
    combined_imu.to_csv(output_file, index=False)
    print(f"Combined IMU data saved to: {output_file}")


# Input and output directories
input_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Sync_Data_EMGIMU"
output_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_data"

# Combine IMU segments
combine_imu_segments(input_base_dir, output_base_dir)
