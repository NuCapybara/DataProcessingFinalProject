import os
import shutil

# Root directories for h0_segmented and h1_segmented
root_dirs = [
    "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/h0_segmented",
    "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/h1_segmented"
]

# Target directory for combined EMG files
combined_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined"

# Ensure the combined directory exists
os.makedirs(combined_dir, exist_ok=True)

# Function to copy combined EMG files and preserve uniqueness
def copy_combined_emg_files(root_dirs, combined_dir):
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                # Look for files ending with "_emg_combined.csv"
                if file.endswith("_emg_combined.csv"):
                    # Get the relative path to preserve subfolder hierarchy
                    relative_path = os.path.relpath(subdir, root_dir)
                    target_subfolder = os.path.join(combined_dir, relative_path)
                    os.makedirs(target_subfolder, exist_ok=True)

                    # Source and destination file paths
                    source_path = os.path.join(subdir, file)
                    destination_path = os.path.join(target_subfolder, file)

                    # Copy the file
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")

# Execute the copy
copy_combined_emg_files(root_dirs, combined_dir)
