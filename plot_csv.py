import os
import pandas as pd
import matplotlib.pyplot as plt
import re

### This file plots the emg & imu data from the csv files we have currently in emg_csv_data folder

# Ensure the output directory for all graphs exists
output_graph_dir = "emg_csv_data/h1_graph"
os.makedirs(output_graph_dir, exist_ok=True)

# Function to parse EMG data
def parse_emg_data(data_str):
    # Use regex to find all integers in the string
    return list(map(int, re.findall(r"-?\d+", str(data_str))))

# Function to parse Quaternion data from a string
def parse_quaternion(quaternion_str):
    """Extract x, y, z, w values from Quaternion string."""
    match = re.findall(r"-?\d+\.\d+", str(quaternion_str))
    return list(map(float, match))

# Function to parse Vector3 data from a string
def parse_vector3(vector3_str):
    """Extract x, y, z values from Vector3 string."""
    match = re.findall(r"-?\d+\.\d+", str(vector3_str))
    return list(map(float, match))

def plot_csv_files(data_base_dir):
    # Iterate over each subfolder inside the given data directory
    for subfolder in os.listdir(data_base_dir):
        subfolder_path = os.path.join(data_base_dir, subfolder)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Find all CSV files in the current subfolder
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]

        # Plot each CSV file
        for csv_file in csv_files:
            csv_path = os.path.join(subfolder_path, csv_file)

            # Load the CSV file
            df = pd.read_csv(csv_path)

            # Handle '_data' column parsing if it is an EMG data file
            if '_emg' in csv_file and '_data' in df.columns:
                # Convert timestamp to seconds for easier interpretation
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

                # Apply parsing function to _data column
                emg_data = df['_data'].apply(parse_emg_data)

                # Expand parsed EMG data into separate columns
                emg_df = pd.DataFrame(emg_data.tolist(), columns=[f'channel_{i}' for i in range(len(emg_data.iloc[0]))])

                # Concatenate timestamp and expanded EMG data
                df = pd.concat([df['timestamp'], emg_df], axis=1)

                # Plot EMG Data
                plt.figure(figsize=(12, 6))
                for col in emg_df.columns:
                    plt.plot(df['timestamp'], df[col], label=col)
                plt.title(f"EMG Data over Time - {csv_file}")
                plt.xlabel("Time (s)")
                plt.ylabel("EMG Signal")
                plt.legend(title="Channels", loc="upper right")
                plt.grid(True)

                # Save the plot as a PNG file in the central output directory
                plot_filename = f"{subfolder}_{csv_file.replace('.csv', '_plot.png')}"
                plot_path = os.path.join(output_graph_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")

                # Clear the figure to avoid overlap when plotting the next file
                plt.close()

            # Handle IMU data parsing and plotting
            elif '_imu' in csv_file and '_orientation' in df.columns and '_angular_velocity' in df.columns and '_linear_acceleration' in df.columns:
                # Convert timestamp to seconds for easier interpretation
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

                # Parse orientation, angular velocity, and linear acceleration data
                orientation_data = df['_orientation'].apply(parse_quaternion)
                angular_velocity_data = df['_angular_velocity'].apply(parse_vector3)
                linear_acceleration_data = df['_linear_acceleration'].apply(parse_vector3)

                # Convert parsed data into separate DataFrames with named columns
                orientation_df = pd.DataFrame(orientation_data.tolist(), columns=['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
                angular_velocity_df = pd.DataFrame(angular_velocity_data.tolist(), columns=['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
                linear_acceleration_df = pd.DataFrame(linear_acceleration_data.tolist(), columns=['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])

                # Concatenate timestamp and all IMU data
                df = pd.concat([df['timestamp'], orientation_df, angular_velocity_df, linear_acceleration_df], axis=1)

                # Plot Orientation Data (Quaternion)
                plt.figure(figsize=(12, 6))
                for col in orientation_df.columns:
                    plt.plot(df['timestamp'], df[col], label=col)
                plt.title(f"Orientation (Quaternion) over Time - {csv_file}")
                plt.xlabel("Time (s)")
                plt.ylabel("Orientation")
                plt.legend(loc="upper right")
                plt.grid(True)
                plot_filename = f"{subfolder}_{csv_file.replace('.csv', '_orientation_plot.png')}"
                plot_path = os.path.join(output_graph_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plt.close()

                # Plot Angular Velocity Data
                plt.figure(figsize=(12, 6))
                for col in angular_velocity_df.columns:
                    plt.plot(df['timestamp'], df[col], label=col)
                plt.title(f"Angular Velocity over Time - {csv_file}")
                plt.xlabel("Time (s)")
                plt.ylabel("Angular Velocity (rad/s)")
                plt.legend(loc="upper right")
                plt.grid(True)
                plot_filename = f"{subfolder}_{csv_file.replace('.csv', '_angular_velocity_plot.png')}"
                plot_path = os.path.join(output_graph_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plt.close()

                # Plot Linear Acceleration Data
                plt.figure(figsize=(12, 6))
                for col in linear_acceleration_df.columns:
                    plt.plot(df['timestamp'], df[col], label=col)
                plt.title(f"Linear Acceleration over Time - {csv_file}")
                plt.xlabel("Time (s)")
                plt.ylabel("Linear Acceleration (m/s^2)")
                plt.legend(loc="upper right")
                plt.grid(True)
                plot_filename = f"{subfolder}_{csv_file.replace('.csv', '_linear_acceleration_plot.png')}"
                plot_path = os.path.join(output_graph_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plt.close()

            else:
                # Skip files that do not match expected EMG or IMU
                print(f"Skipping file {csv_file} as it doesn't match EMG or IMU data patterns.")

if __name__ == "__main__":
    # Directory containing the processed CSV data
    data_base_dir = "emg_csv_data/h1"  # Adjust this path accordingly

    # Run the plotting function
    plot_csv_files(data_base_dir)
