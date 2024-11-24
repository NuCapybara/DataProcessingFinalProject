import os
import pandas as pd
import numpy as np
import ast


def parse_emg_data(data_str):
    return list(ast.literal_eval(data_str.replace("array('h',", "").replace(")", "")))


def parse_imu_field(field_str):
    start_idx = field_str.find("(") + 1
    end_idx = field_str.find(")")
    values = field_str[start_idx:end_idx].split(",")
    return [float(val.split("=")[1]) for val in values]


def approximate_match_emg_to_imu(emg_data, imu_data, tolerance=0.01):
    """
    Match each IMU timestamp to the nearest EMG timestamp within a given tolerance.
    """
    emg_times = emg_data['time'].values
    imu_times = imu_data['time'].values

    matched_emg_indices = []
    matched_imu_indices = []

    for imu_idx, imu_time in enumerate(imu_times):
        # Find the closest EMG timestamp within the tolerance
        closest_idx = np.abs(emg_times - imu_time).argmin()
        time_diff = abs(emg_times[closest_idx] - imu_time)

        if time_diff <= tolerance:
            matched_imu_indices.append(imu_idx)
            matched_emg_indices.append(closest_idx)

    # Create matched datasets
    matched_emg = emg_data.iloc[matched_emg_indices].reset_index(drop=True)
    matched_imu = imu_data.iloc[matched_imu_indices].reset_index(drop=True)

    return matched_emg, matched_imu


def align_and_resample(emg_file, imu_file, output_dir, tolerance=0.01):
    try:
        # Load EMG and IMU data
        emg_data = pd.read_csv(emg_file)
        imu_data = pd.read_csv(imu_file)

        # Convert timestamps to seconds
        emg_data['time'] = emg_data['timestamp'] / 1e9
        imu_data['time'] = imu_data['timestamp'] / 1e9

        # Parse EMG and IMU data
        emg_data['_data_parsed'] = emg_data['_data'].apply(parse_emg_data)
        imu_data['orientation'] = imu_data['_orientation'].apply(parse_imu_field)
        imu_data['angular_velocity'] = imu_data['_angular_velocity'].apply(parse_imu_field)
        imu_data['linear_acceleration'] = imu_data['_linear_acceleration'].apply(parse_imu_field)

        # Approximate time matching
        matched_emg, matched_imu = approximate_match_emg_to_imu(emg_data, imu_data, tolerance)

        if matched_emg.empty or matched_imu.empty:
            print(f"Skipped (No matched data within tolerance): EMG: {emg_file}, IMU: {imu_file}")
            return False

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the matched data
        emg_output_file = os.path.join(output_dir, os.path.basename(emg_file))
        imu_output_file = os.path.join(output_dir, os.path.basename(imu_file))

        matched_emg.to_csv(emg_output_file, index=False)
        matched_imu.to_csv(imu_output_file, index=False)

        # Validate saved files
        if not os.path.exists(emg_output_file) or not os.path.exists(imu_output_file):
            print(f"Error saving files: EMG ({emg_output_file}), IMU ({imu_output_file})")
        else:
            print(f"Saved EMG: {emg_output_file}")
            print(f"Saved IMU: {imu_output_file}")

        return True

    except Exception as e:
        print(f"Error processing EMG: {emg_file}, IMU: {imu_file} -> {e}")
        return False


def process_folder(input_base_dir, output_base_dir, tolerance=0.01):
    """
    Recursively process all subfolders under the input directory, ignoring the `transformed` folder.
    """
    unmatched_emg_files = []
    unmatched_imu_files = []
    processed_pairs = set()

    for root, dirs, files in os.walk(input_base_dir):
        # Skip 'transformed' folder
        dirs[:] = [d for d in dirs if d != "transformed"]

        emg_files = [os.path.join(root, f) for f in files if "_emg_seg" in f]
        imu_files = [os.path.join(root, f) for f in files if "_imu_seg" in f]

        for emg_file in emg_files:
            # Extract segment ID and prefix (e.g., RU, RL) from EMG file name
            segment_id = emg_file.split("_emg_seg")[1].split(".csv")[0]
            prefix = emg_file.split("/")[-1].split("_")[2]

            # Find corresponding IMU file with the same segment ID and prefix
            imu_file = next(
                (
                    imu
                    for imu in imu_files
                    if f"_{prefix}_imu_seg{segment_id}" in imu
                ),
                None,
            )

            if imu_file:
                # Generate output directory
                relative_path = os.path.relpath(root, input_base_dir)
                output_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Process and save data
                success = align_and_resample(emg_file, imu_file, output_dir, tolerance)
                if success:
                    processed_pairs.add((emg_file, imu_file))
                else:
                    unmatched_emg_files.append(emg_file)
                    unmatched_imu_files.append(imu_file)
            else:
                unmatched_emg_files.append(emg_file)

    # Log processing summary
    print("\nProcessing Summary:")
    print(f"Processed Unique Pairs: {len(processed_pairs)}")
    print(f"Unmatched EMG files: {len(unmatched_emg_files)}")
    for emg_file in unmatched_emg_files:
        print(f"  - {emg_file}")
    print(f"Unmatched IMU files: {len(unmatched_imu_files)}")
    for imu_file in unmatched_imu_files:
        print(f"  - {imu_file}")


# Base directories
input_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Raw_Data_EMGIMU"
output_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Segmented_Sync_Data_EMGIMU"

# Process folders
process_folder(input_base_dir, output_base_dir)
