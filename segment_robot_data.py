import os
import pandas as pd

def crop_robot_data_recursive(trail_csv, data_root, output_directory):
    """
    Recursively crop robot data based on timestamps from the provided CSV file.

    :param trail_csv: Path to the CSV file with trail names and timestamps.
    :param data_root: Root directory containing the robot data CSV files.
    :param output_directory: Directory to save the cropped files.
    """
    # Load the timestamps from the CSV file
    try:
        timestamps_df = pd.read_csv(trail_csv)
    except Exception as e:
        print(f"Error reading the trail CSV file: {e}")
        return
    
    if "Trail Name" not in timestamps_df.columns or "Timestamp" not in timestamps_df.columns:
        print("Error: The CSV must contain 'Trail Name' and 'Timestamp' columns.")
        return

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create a dictionary for quick lookup of timestamps
    timestamps_dict = dict(zip(timestamps_df["Trail Name"], timestamps_df["Timestamp"]))

    # Recursively walk through the data_root directory
    for subdir, _, files in os.walk(data_root):
        trail_name = os.path.basename(subdir)  # Subfolder name

        # Check if this trail name exists in the timestamps CSV
        if trail_name not in timestamps_dict:
            continue

        start_timestamp = timestamps_dict[trail_name]

        # Process all CSV files in this subdirectory
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(file_path)

                    if "Timestamp" not in df.columns:
                        print(f"Skipping {file_path}: 'Timestamp' column not found.")
                        continue
                    
                    # Convert Timestamp to numeric
                    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                    df = df.dropna(subset=["Timestamp"])

                    # Crop data
                    cropped_df = df[df["Timestamp"] >= start_timestamp]

                    if cropped_df.empty:
                        print(f"No data after {start_timestamp} in {file_path}. Skipping.")
                        continue

                    # Save the cropped data
                    relative_subdir = os.path.relpath(subdir, data_root)
                    output_subdir = os.path.join(output_directory, relative_subdir)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file_path = os.path.join(output_subdir, file)
                    cropped_df.to_csv(output_file_path, index=False)
                    print(f"Cropped data saved to {output_file_path}.")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# Example usage
h0_csv = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/single_velocity_timestamps_h0.csv"
h1_csv = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/single_velocity_timestamps_h1.csv"
data_directory = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data"
output_directory = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/cropped_data"

# Process h0 and h1 subfolders separately
crop_robot_data_recursive(h0_csv, os.path.join(data_directory, "h0"), output_directory)
crop_robot_data_recursive(h1_csv, os.path.join(data_directory, "h1"), output_directory)
