import os
import pandas as pd


def combine_segments_with_continuous_timestamps(subdir, prefix, output_dir, output_name):
    """
    Combine EMG segment files with a specific prefix into one file, ensuring continuous timestamps,
    and save the combined file to the specified output directory.
    """
    # Collect all segment files containing the prefix
    segment_files = [
        os.path.join(subdir, file)
        for file in os.listdir(subdir)
        if prefix in file and file.endswith(".csv")
    ]

    # If no matching files are found, skip
    if not segment_files:
        print(f"No matching files with prefix '{prefix}' in {subdir}")
        return

    # Sort the segment files (assumes naming convention orders segments logically)
    segment_files.sort()

    combined_df = pd.DataFrame()  # Initialize an empty DataFrame
    last_timestamp = None  # To store the last timestamp of the previous segment

    for segment_file in segment_files:
        df = pd.read_csv(segment_file)

        # Adjust timestamps for continuity
        if last_timestamp is not None:
            # Calculate the offset to make timestamps continuous
            time_offset = last_timestamp - df['timestamp'].iloc[0] + 1
            df['timestamp'] += time_offset

        # Update last_timestamp to the last value in this segment
        last_timestamp = df['timestamp'].iloc[-1]

        # Append the current segment to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined file
    output_path = os.path.join(output_dir, output_name)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined file with continuous timestamps saved: {output_path}")


def process_trails(root_dir, output_base_dir):
    """
    Process each trail in the root directory to combine EMG segments
    and save the combined files to the corresponding subfolders in the output base directory.
    """
    for subdir, _, _ in os.walk(root_dir):
        # Determine the relative path for the subfolder
        rel_path = os.path.relpath(subdir, root_dir)
        output_dir = os.path.join(output_base_dir, rel_path)

        # Combine RL_emg segments
        combine_segments_with_continuous_timestamps(subdir, "RL_emg_", output_dir, "RL_emg_combined.csv")

        # Combine RU_emg segments
        combine_segments_with_continuous_timestamps(subdir, "RU_emg_", output_dir, "RU_emg_combined.csv")


# Define root directory for processing
root_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Sync_Data_EMGIMU"
# Define base output directory
output_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined_sync_data"

# Process all trails
process_trails(root_dir, output_base_dir)
