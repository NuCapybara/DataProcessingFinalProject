import os
import shutil

# Base directories
BASE_EMG_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined"
BASE_IMU_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_combined"
TARGET_EMG_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_scale"
TARGET_IMU_DIR = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_scale"

# Subdirectories for h0 and h1
SUB_DIRS = ["h0", "h1"]

def copy_formatted_files(source_base_dir, target_base_dir):
    """
    Copies all _formatted.csv files from the source directory to the target directory
    while preserving subfolder names.

    Args:
        source_base_dir (str): Base directory where the formatted files are located.
        target_base_dir (str): Base directory where the formatted files should be copied.

    Returns:
        None
    """
    for sub_dir in SUB_DIRS:
        main_dir = os.path.join(source_base_dir, sub_dir)

        if not os.path.exists(main_dir):
            print(f"Directory not found: {main_dir}")
            continue

        for subfolder in os.listdir(main_dir):
            source_subfolder_dir = os.path.join(main_dir, subfolder)
            target_subfolder_dir = os.path.join(target_base_dir, subfolder)

            # Ensure the target subfolder exists
            os.makedirs(target_subfolder_dir, exist_ok=True)

            for file in os.listdir(source_subfolder_dir):
                if file.endswith("combined_formatted.csv"):
                    source_file_path = os.path.join(source_subfolder_dir, file)
                    target_file_path = os.path.join(target_subfolder_dir, file)

                    # Copy the file
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"Copied: {source_file_path} -> {target_file_path}")

if __name__ == "__main__":
    print("Organizing EMG files...")
    copy_formatted_files(BASE_EMG_DIR, TARGET_EMG_DIR)

    print("Organizing IMU files...")
    copy_formatted_files(BASE_IMU_DIR, TARGET_IMU_DIR)

    print("File organization complete.")
