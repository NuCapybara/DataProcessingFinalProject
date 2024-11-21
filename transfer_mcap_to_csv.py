import os
import pandas as pd
import re
from mcap.reader import make_reader
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def rosbag2_to_csv(bag_path, output_base_dir):
    # Iterate over all subfolders inside the given bag_path directory
    for subfolder in os.listdir(bag_path):
        subfolder_path = os.path.join(bag_path, subfolder)
        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # The expected .mcap file name inside each subfolder
        mcap_file_name = f"{subfolder}_0.mcap"
        mcap_file_path = os.path.join(subfolder_path, mcap_file_name)

        # Skip if the expected .mcap file does not exist
        if not os.path.exists(mcap_file_path):
            print(f"Skipping {subfolder_path}, {mcap_file_name} not found.")
            continue

        # Prepare the output directory for the CSV files
        output_dir = os.path.join(output_base_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        with open(mcap_file_path, "rb") as f:
            reader = make_reader(f)

            # Dictionary to store dataframes for each topic
            dataframes = {}

            for schema, channel, message in reader.iter_messages():
                topic_name = channel.topic
                msg_type = schema.name
                timestamp = message.log_time

                # Dynamically get message type and deserialize
                try:
                    msg_class = get_message(msg_type)
                    msg = deserialize_message(message.data, msg_class)
                    msg_dict = {"timestamp": timestamp}

                    # Convert message fields to dictionary
                    msg_dict.update(
                        {field: getattr(msg, field) for field in msg.__slots__}
                    )
                except Exception as e:
                    print(f"Failed to deserialize message on topic {topic_name}: {e}")
                    continue

                # Accumulate data for each topic in a dataframe
                if topic_name not in dataframes:
                    dataframes[topic_name] = []
                dataframes[topic_name].append(msg_dict)

            # Write each topic's data to a CSV file with the required renaming pattern
            for topic_name, records in dataframes.items():
                df = pd.DataFrame(records)

                # Define suffix based on topic name for CSV file naming
                if topic_name == "/RL_myo/emg":
                    suffix = "_RL_emg"
                elif topic_name == "/RL_myo/imu":
                    suffix = "_RL_imu"
                elif topic_name == "/RU_myo/emg":
                    suffix = "_RU_emg"
                elif topic_name == "/RU_myo/imu":
                    suffix = "_RU_imu"
                else:
                    print(f"Skipping unknown topic {topic_name} in {subfolder}")
                    continue

                # Generate CSV file path with the appropriate naming
                csv_filename = f"{subfolder}{suffix}.csv"
                csv_file_path = os.path.join(output_dir, csv_filename)

                # Save the dataframe as a CSV file
                df.to_csv(csv_file_path, index=False)
                print(f"Saved {topic_name} to {csv_file_path}")


if __name__ == "__main__":
    # Adjust these paths accordingly
    bag_path = "HumanElbowData/h1"  # Base path containing subfolders with .mcap files
    output_base_dir = "emg_csv_data/h1"  # Output directory for CSV files

    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Run the conversion process
    rosbag2_to_csv(bag_path, output_base_dir)
