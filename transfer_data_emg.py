import os
import pandas as pd
import datetime
import shutil

# Function to delete the 'transformed' folder if it exists
def delete_transformed_folder(subdir):
    transformed_dir = os.path.join(subdir, "transformed")
    if os.path.exists(transformed_dir):
        shutil.rmtree(transformed_dir)  # Delete the folder and its contents
        print(f"Deleted transformed folder: {transformed_dir}")
    return transformed_dir

# Define the function to convert timestamp
def convert_timestamp_from_nanoseconds(nanoseconds):
    # Convert nanoseconds to seconds and create datetime
    dt = datetime.datetime.fromtimestamp(nanoseconds / 1e9)
    return dt.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]  # Format to match the target

# Define the function to convert data format
def convert_data(data):
    # Extract numbers from the array representation
    numbers = data.split('[')[1].split(']')[0]
    return f"({numbers.replace(',', '')})"

# Directory containing the data
root_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/h1_segmented"

# Process all subfolders
for subdir, _, files in os.walk(root_dir):
    delete_transformed_folder(subdir)
    for file in files:
        if "emg" in file and file.endswith(".csv"):  # Check for "emg" in the file name
            file_path = os.path.join(subdir, file)
            
            # Create a 'transformed' subfolder in the current folder
            transformed_dir = os.path.join(subdir, "transformed")
            os.makedirs(transformed_dir, exist_ok=True)
            
            # Define the transformed file path
            transformed_file_path = os.path.join(
                transformed_dir, f"{os.path.splitext(file)[0]}_transformed.csv"
            )
            
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                
                # Check the structure of the file and perform transformation
                if "_data" in df.columns and "timestamp" in df.columns:
                    # Convert timestamp and data
                    df["time"] = df["timestamp"].apply(convert_timestamp_from_nanoseconds)
                    df[".data"] = df["_data"].apply(convert_data)
                    
                    # Keep only the required columns in the desired order
                    transformed_df = df[["time", ".data"]]
                    
                    # Save the transformed file
                    transformed_df.to_csv(transformed_file_path, index=False)
                    print(f"Transformed and saved: {transformed_file_path}")
                else:
                    print(f"File structure not supported: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Function to combine files with a specific prefix, ensuring time continuity
def combine_segments(subdir, prefix, output_name):
    transformed_dir = os.path.join(subdir, "transformed")
    
    # Check if the transformed folder exists
    if not os.path.exists(transformed_dir):
        print(f"No transformed folder in {subdir}")
        return
    
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
    
    # Sort the segment files (assumes naming convention orders segments logically)
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
    output_path = os.path.join(transformed_dir, output_name)
    combined_df.to_csv(output_path, index=False, date_format='%Y/%m/%d/%H:%M:%S.%f')
    print(f"Combined file saved: {output_path}")

# Process all subfolders
for subdir, _, _ in os.walk(root_dir):
    
    # Combine RL_emg_transformed segments
    combine_segments(subdir, "RL_emg_", "RL_emg_combined.csv")
    
    # Combine RU_emg_transformed segments
    combine_segments(subdir, "RU_emg_", "RU_emg_combined.csv")
