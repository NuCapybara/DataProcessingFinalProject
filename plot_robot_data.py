import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_robot_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Load the CSV data into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return
    
    # Ensure "Timestamp" is present and numeric
    if "Timestamp" not in df.columns:
        print("Error: 'Timestamp' column not found in the data.")
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
    plt.title("Joint Positions Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot velocities
    plt.figure(figsize=(10, 6))
    for col in [col for col in df.columns if "velocity" in col]:
        plt.plot(df["Timestamp"].values, df[col].values, label=col)
    plt.title("Joint Velocities Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot efforts
    plt.figure(figsize=(10, 6))
    for col in [col for col in df.columns if "effort" in col]:
        plt.plot(df["Timestamp"].values, df[col].values, label=col)
    plt.title("Joint Efforts Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Effort")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot specific finger joint positions
    plt.figure(figsize=(10, 6))
    # plt.plot(df["Timestamp"].values, df["panda_finger_joint1_position"].values, label="panda_finger_joint1_position")
    plt.plot(df["Timestamp"].values, df["panda_finger_joint2_position"].values, label="panda_finger_joint2_position")
    plt.title("Panda Finger Joint Positions Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
robot_data_file = "robot_csv_data/R_r1deg0h0/R_r1deg0h0_0.csv"
plot_robot_data(robot_data_file)
