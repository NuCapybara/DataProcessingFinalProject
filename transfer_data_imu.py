import os
import pandas as pd
import datetime

# Function to convert nanoseconds to James's timestamp format
def convert_nanoseconds_to_james_format(nanoseconds):
    dt = datetime.datetime.fromtimestamp(nanoseconds / 1e9)  # Convert nanoseconds to seconds
    return dt.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]

# Function to transform IMU data from your format to James's format
def transform_imu_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Dynamically detect the timestamp column
    timestamp_column = None
    for col in df.columns:
        if "timestamp" in col.lower():
            timestamp_column = col
            break

    if not timestamp_column:
        raise KeyError(f"Timestamp column not found in file: {input_file}")

    # Convert the detected timestamp column to James's format
    df["time"] = df[timestamp_column].apply(convert_nanoseconds_to_james_format)

    # Extract header details
    df[".header.seq"] = df.index + 1  # Sequence numbers
    df[".header.stamp.secs"] = df["_header"].str.extract(r"sec=(\d+)").astype(int)
    df[".header.stamp.nsecs"] = df["_header"].str.extract(r"nanosec=(\d+)").astype(int)
    df[".header.frame_id"] = df["_header"].str.extract(r"frame_id='([^']+)'")

    # Extract orientation
    df[".orientation.x"] = df["_orientation"].str.extract(r"x=([-\d.]+)").astype(float)
    df[".orientation.y"] = df["_orientation"].str.extract(r"y=([-\d.]+)").astype(float)
    df[".orientation.z"] = df["_orientation"].str.extract(r"z=([-\d.]+)").astype(float)
    df[".orientation.w"] = df["_orientation"].str.extract(r"w=([-\d.]+)").astype(float)

    # Extract angular velocity
    df[".angular_velocity.x"] = df["_angular_velocity"].str.extract(r"x=([-\d.]+)").astype(float)
    df[".angular_velocity.y"] = df["_angular_velocity"].str.extract(r"y=([-\d.]+)").astype(float)
    df[".angular_velocity.z"] = df["_angular_velocity"].str.extract(r"z=([-\d.]+)").astype(float)

    # Extract linear acceleration
    df[".linear_acceleration.x"] = df["_linear_acceleration"].str.extract(r"x=([-\d.]+)").astype(float)
    df[".linear_acceleration.y"] = df["_linear_acceleration"].str.extract(r"y=([-\d.]+)").astype(float)
    df[".linear_acceleration.z"] = df["_linear_acceleration"].str.extract(r"z=([-\d.]+)").astype(float)

    # Drop unnecessary columns and reorder
    james_columns = [
        "time", ".header.seq", ".header.stamp.secs", ".header.stamp.nsecs", ".header.frame_id",
        ".orientation.x", ".orientation.y", ".orientation.z", ".orientation.w",
        ".angular_velocity.x", ".angular_velocity.y", ".angular_velocity.z",
        ".linear_acceleration.x", ".linear_acceleration.y", ".linear_acceleration.z"
    ]
    transformed_df = df[james_columns]

    # Save the transformed file
    transformed_df.to_csv(output_file, index=False)
    print(f"Transformed and saved: {output_file}")

# Function to combine segments while maintaining time continuity
def combine_segments(transformed_dir, combined_dir, subfolder_name, prefix, output_name):
    # Ensure subfolder in combined_dir exists
    combined_subfolder = os.path.join(combined_dir, subfolder_name)
    os.makedirs(combined_subfolder, exist_ok=True)

    # Collect all segment files containing the prefix
    segment_files = [
        os.path.join(transformed_dir, file)
        for file in os.listdir(transformed_dir)
        if prefix in file and file.endswith(".csv")
    ]
    
    # If no matching files are found, skip
    if not segment_files:
        print(f"No matching files with prefix '{prefix}' in {transformed_dir}")
        return
    
    # Sort the segment files
    segment_files.sort()
    
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame
    prev_end_time = None  # To store the last time of the previous segment

    for segment_file in segment_files:
        df = pd.read_csv(segment_file)

        # Convert 'time' column to datetime for adjustment
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')
        
        if prev_end_time is not None:
            # Calculate time offset
            time_offset = prev_end_time - df['time'].iloc[0]
            # Adjust times in the current segment
            df['time'] += time_offset
        
        # Update `prev_end_time` to the last time in this segment
        prev_end_time = df['time'].iloc[-1]
        
        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined file
    output_path = os.path.join(combined_subfolder, output_name)
    combined_df.to_csv(output_path, index=False, date_format='%Y/%m/%d/%H:%M:%S.%f')
    print(f"Combined file saved: {output_path}")

# Main processing loop
root_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/h0_segmented"
combined_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_combined"

for subdir, dirs, files in os.walk(root_dir):
    if "transformed" in dirs:  # Skip the 'transformed' subfolder
        dirs.remove("transformed")

    if not any("imu" in file for file in files):  # Skip folders without IMU files
        continue

    subfolder_name = os.path.basename(subdir)  # Get the subfolder name
    transformed_dir = os.path.join(combined_dir, subfolder_name)  # Store transformed files in imu_combined/<subfolder>
    os.makedirs(transformed_dir, exist_ok=True)

    for file in files:
        if "imu" in file and file.endswith(".csv"):  # Check for IMU files
            file_path = os.path.join(subdir, file)
            transformed_file_path = os.path.join(transformed_dir, f"{os.path.splitext(file)[0]}_transformed.csv")
            transform_imu_data(file_path, transformed_file_path)
    
    # Combine RL_imu_transformed and RU_imu_transformed
    combine_segments(transformed_dir, combined_dir, subfolder_name, "RL_imu_", "RL_imu_combined.csv")
    combine_segments(transformed_dir, combined_dir, subfolder_name, "RU_imu_", "RU_imu_combined.csv")
