import os
import pandas as pd
import numpy as np

def find_first_velocity_pattern(df, tolerance=1e-3):
    """
    Identify the first timestamp where velocity transitions 0 -> some value -> 0.
    """
    for col in [col for col in df.columns if "velocity" in col]:
        velocity = df[col].values
        timestamp = df["Timestamp"].values
        for i in range(1, len(velocity) - 1):
            if (
                abs(velocity[i - 1]) <= tolerance
                and abs(velocity[i]) > tolerance
                and abs(velocity[i + 1]) <= tolerance
            ):
                return timestamp[i + 1]  # Return the first match
    return None  # No match found

def process_and_export_single_timestamp(directory, output_csv, tolerance=1e-3):
    results = []
    for subdir, _, files in os.walk(directory):
        subfolder_name = os.path.basename(subdir)
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(file_path)
                    if "Timestamp" not in df.columns:
                        continue
                    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                    df = df.dropna(subset=["Timestamp"])
                    for col in df.columns:
                        if col != "Timestamp":
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    
                    timestamp = find_first_velocity_pattern(df, tolerance)
                    if timestamp is not None:
                        results.append({"Trail Name": subfolder_name, "Timestamp": timestamp})
                        break  # Only keep the first timestamp per trail
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Export to CSV
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Timestamps successfully exported to {output_csv}")
    else:
        print("No matching patterns found.")

# Example usage
directories_to_process = [
    "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/h1"
    # "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/h0"
]
output_csv = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/robot_csv_data/single_velocity_timestamps_h1.csv"

for directory in directories_to_process:
    process_and_export_single_timestamp(directory, output_csv)
