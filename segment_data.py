import os
import pandas as pd

# Define base paths
input_base = "emg_csv_data"
output_windows_base = "output_emg_windows"
output_segmented_base = "emg_csv_data"

# Define subfolders for different conditions
subfolders = ["h0", "h1"]

# Function to segment data and save each segment separately
def segment_data(raw_file, window_file, output_folder, file_prefix):
    # Read the raw data and time window
    raw_data = pd.read_csv(raw_file)
    windows = pd.read_csv(window_file)

    # Normalize raw data timestamps to start from 0
    raw_data["timestamp"] -= raw_data["timestamp"].iloc[0]

    # Convert start and end times to nanoseconds (if necessary)
    windows["Start_Time"] *= 1e9
    windows["End_Time"] *= 1e9

    # Process each window and save as a separate file
    for i, row in windows.iterrows():
        start, end = row["Start_Time"], row["End_Time"]
        segment = raw_data[(raw_data["timestamp"] >= start) & (raw_data["timestamp"] <= end)]

        if segment.empty:
            print(f"No data found in {raw_file} for window {i + 1}. Skipping...")
            continue

        # Save the segmented data
        segment_file = os.path.join(output_folder, f"{file_prefix}_seg{i + 1}.csv")
        segment.to_csv(segment_file, index=False)
        print(f"Saved segment {i + 1} to {segment_file}")

# Process each subfolder
for subfolder in subfolders:
    input_path = os.path.join(input_base, subfolder)
    window_path = os.path.join(output_windows_base, subfolder)
    output_path = os.path.join(output_segmented_base, f"{subfolder}_segmented")

    # Iterate over window files
    for window_file in os.listdir(window_path):
        if not window_file.endswith("_emg_windows.csv"):
            continue
        
        # Determine corresponding trial folder
        trial_name = window_file.replace("_RL_emg_windows.csv", "")
        trial_folder = os.path.join(input_path, trial_name)

        if not os.path.isdir(trial_folder):
            print(f"Missing trial folder: {trial_folder}. Skipping...")
            continue

        # Create output folder for the trial
        trial_output_folder = os.path.join(output_path, f"{trial_name}_segmented")
        os.makedirs(trial_output_folder, exist_ok=True)

        # Construct raw file paths
        raw_files = [
            os.path.join(trial_folder, f"{trial_name}_RL_emg.csv"),
            os.path.join(trial_folder, f"{trial_name}_RU_emg.csv"),
            os.path.join(trial_folder, f"{trial_name}_RL_imu.csv"),
            os.path.join(trial_folder, f"{trial_name}_RU_imu.csv"),
        ]

        # Process each raw file
        for raw_file in raw_files:
            if os.path.exists(raw_file):
                print(f"Processing {raw_file} with {window_file}...")
                file_prefix = os.path.splitext(os.path.basename(raw_file))[0]
                segment_data(raw_file, os.path.join(window_path, window_file), trial_output_folder, file_prefix)
            else:
                print(f"Missing raw file: {raw_file}. Skipping...")

print("Segmentation complete!")
