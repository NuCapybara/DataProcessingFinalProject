import os
import pandas as pd
import shutil

# Base directories
BASE_EMG_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined"
TARGET_EMG_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_scale"

# Desired time span (in seconds)
DESIRED_TIME_SPAN = 14

# Subdirectories for h0 and h1
SUB_DIRS = ["h0", "h1"]

def scale_time_span(file_path, output_path, desired_time_span):
    """
    Scales the time span of the data in a CSV file to the desired time span.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the scaled output CSV file.
        desired_time_span (float): Desired time span in seconds.

    Returns:
        None
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')

        # Calculate the current time span
        start_time = df['time'].iloc[0]
        end_time = df['time'].iloc[-1]
        current_time_span = (end_time - start_time).total_seconds()

        # Calculate the scaling factor
        scaling_factor = desired_time_span / current_time_span

        # Scale the time column
        original_time_seconds = (df['time'] - start_time).dt.total_seconds()
        scaled_time_seconds = original_time_seconds * scaling_factor
        df['time'] = pd.to_timedelta(scaled_time_seconds, unit='s') + start_time

        # Save the scaled data
        df.to_csv(output_path, index=False)
        print(f"Scaled and saved: {output_path}")
    except Exception as e:
        print(f"Error scaling file {file_path}: {e}")

def format_time_column(input_file, output_file):
    """
    Reformats the 'time' column in a CSV file to the desired format.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the reformatted CSV file.

    Returns:
        None
    """
    try:
        # Load the data
        df = pd.read_csv(input_file)

        # Convert 'time' column to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Format the 'time' column back to the desired format
        df['time'] = df['time'].dt.strftime('%Y/%m/%d/%H:%M:%S.%f').str[:-3]  # Keep milliseconds

        # Save the reformatted data
        df.to_csv(output_file, index=False)
        print(f"Formatted and saved to {output_file}")
    except Exception as e:
        print(f"Error formatting file {input_file}: {e}")

def process_emg_files(base_dir, target_dir, sub_dirs, desired_time_span):
    """
    Processes all EMG files in the specified base directory and subdirectories.
    Scales the time span of each file, reformats the 'time' column, and saves
    the formatted files to the target directory.

    Args:
        base_dir (str): Base directory containing EMG data.
        target_dir (str): Directory to save formatted files.
        sub_dirs (list): List of subdirectories (e.g., ["h0", "h1"]).
        desired_time_span (float): Desired time span in seconds.

    Returns:
        None
    """
    for sub_dir in sub_dirs:
        main_dir = os.path.join(base_dir, sub_dir)

        if not os.path.exists(main_dir):
            print(f"Directory not found: {main_dir}")
            continue

        for subfolder in os.listdir(main_dir):
            transformed_dir = os.path.join(main_dir, subfolder, "transformed")
            target_subfolder_dir = os.path.join(target_dir, subfolder)

            # Ensure the target subfolder exists
            os.makedirs(target_subfolder_dir, exist_ok=True)

            if not os.path.exists(transformed_dir):
                print(f"No 'transformed' folder in {os.path.join(main_dir, subfolder)}")
                continue

            # Process all CSV files in the transformed directory
            for file in os.listdir(transformed_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(transformed_dir, file)
                    scaled_output_path = os.path.join(
                        transformed_dir, f"{os.path.splitext(file)[0]}_scaled.csv"
                    )
                    formatted_output_path = os.path.join(
                        target_subfolder_dir, f"{os.path.splitext(file)[0]}_formatted.csv"
                    )

                    # Scale the time span
                    scale_time_span(file_path, scaled_output_path, desired_time_span)

                    # Format the time column and save it to the target directory
                    format_time_column(scaled_output_path, formatted_output_path)

if __name__ == "__main__":
    print("Processing EMG files...")
    process_emg_files(BASE_EMG_DIR, TARGET_EMG_DIR, SUB_DIRS, DESIRED_TIME_SPAN)
    print("EMG processing complete.")
