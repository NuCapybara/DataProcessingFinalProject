import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_robot_data(file_path, subfolder_name):
    # Load the CSV data into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return
    
    # Ensure "Timestamp" is present and numeric
    if "Timestamp" not in df.columns:
        print(f"Error: 'Timestamp' column not found in the data for file {file_path}")
        return
    
    # Convert "Timestamp" to numeric
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    
    # Drop rows with NaN in Timestamp
    df = df.dropna(subset=["Timestamp"])
    
    # Convert all other columns to numeric
    for col in df.columns:
        if col != "Timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Plot positions
    plt.figure(figsize=(10, 6))
    for col in [col for col in df.columns if "position" in col]:
        plt.plot(df["Timestamp"].values, df[col].values, label=col)
    plt.title(f"{subfolder_name} - Joint Positions Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot velocities
    plt.figure(figsize=(10, 6))
    for col in [col for col in df.columns if "velocity" in col]:
        plt.plot(df["Timestamp"].values, df[col].values, label=col)
    plt.title(f"{subfolder_name} - Joint Velocities Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot efforts
    plt.figure(figsize=(10, 6))
    for col in [col for col in df.columns if "effort" in col]:
        plt.plot(df["Timestamp"].values, df[col].values, label=col)
    plt.title(f"{subfolder_name} - Joint Efforts Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Effort")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot specific finger joint positions
    plt.figure(figsize=(10, 6))
    if "panda_finger_joint1_position" in df.columns:
        plt.plot(df["Timestamp"].values, df["panda_finger_joint1_position"].values, label="panda_finger_joint1_position")
    if "panda_finger_joint2_position" in df.columns:
        plt.plot(df["Timestamp"].values, df["panda_finger_joint2_position"].values, label="panda_finger_joint2_position")
    plt.title(f"{subfolder_name} - Panda Finger Joint Positions Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_robot_data(directory):
    for subdir, _, files in os.walk(directory):
        subfolder_name = os.path.basename(subdir)
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                plot_robot_data(file_path, subfolder_name)

# Example usage
robot_data_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/processed_robot_velocity_data"
plot_all_robot_data(robot_data_dir)
