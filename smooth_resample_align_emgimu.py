import os
import pandas as pd
import numpy as np
import ast
from scipy.signal import detrend
from scipy.signal import resample

def smooth_and_rectify(emg_signal, window_size=50):
    """
    Smooth and rectify an EMG signal.

    Args:
        emg_signal (pd.Series or np.ndarray): Raw EMG signal.
        window_size (int): Size of the rolling window for smoothing.

    Returns:
        pd.Series: Processed EMG signal.
    """
    # Remove any constant offset
    emg_signal = detrend(emg_signal, type='constant')
    # Full-wave rectification
    emg_signal = np.abs(emg_signal)
    # Smooth using a moving average
    emg_signal = pd.Series(emg_signal).rolling(window=window_size, min_periods=1).mean()
    return emg_signal


def downsample_emg(emg_data, target_length):
    """
    Downsample the EMG data to match the target length.

    Args:
        emg_data (pd.DataFrame): DataFrame containing the EMG data to downsample.
        target_length (int): The target number of samples after downsampling.

    Returns:
        pd.DataFrame: Downsampled EMG data.
    """
    # Use scipy.signal.resample for accurate downsampling
    return pd.DataFrame(resample(emg_data, target_length, axis=0))


def parse_emg_data(data_str):
    return list(ast.literal_eval(data_str.replace("array('h',", "").replace(")", "")))


def parse_imu_field(field_str):
    start_idx = field_str.find("(") + 1
    end_idx = field_str.find(")")
    values = field_str[start_idx:end_idx].split(",")
    return [float(val.split("=")[1]) for val in values]

def align_and_resample(emg_file, imu_file, output_dir, tolerance=0.01):
    try:
        # Load EMG and IMU data
        emg_data = pd.read_csv(emg_file)
        imu_data = pd.read_csv(imu_file)

        # Parse EMG data
        emg_data['_data_parsed'] = emg_data['_data'].apply(parse_emg_data)

        # Smooth and rectify EMG data
        emg_channels = pd.DataFrame(emg_data['_data_parsed'].tolist())
        smoothed_channels = emg_channels.apply(lambda x: smooth_and_rectify(x, window_size=50), axis=0)

        # Downsample EMG data to match IMU segment length
        imu_segment_length = len(imu_data)
        downsampled_emg = downsample_emg(smoothed_channels, imu_segment_length)

        # Replace the `_data` column with the downsampled data
        emg_data = emg_data.iloc[:imu_segment_length]  # Keep only rows matching IMU length
        emg_data['_data'] = downsampled_emg.apply(lambda row: f"array('h', {row.tolist()})", axis=1)

        # Drop unnecessary columns
        emg_data = emg_data[['timestamp', '_data', '_check_fields']]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the processed EMG data
        emg_output_file = os.path.join(output_dir, os.path.basename(emg_file))
        emg_data.to_csv(emg_output_file, index=False)

        # Save IMU data without changes
        imu_output_file = os.path.join(output_dir, os.path.basename(imu_file))
        imu_data.to_csv(imu_output_file, index=False)

        print(f"Saved EMG: {emg_output_file}")
        print(f"Saved IMU: {imu_output_file}")

        return True

    except Exception as e:
        print(f"Error processing EMG: {emg_file}, IMU: {imu_file} -> {e}")
        return False


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
output_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/Sedmengted_sync_smooth_Data_IMUEMG"

# Process folders
process_folder(input_base_dir, output_base_dir)
