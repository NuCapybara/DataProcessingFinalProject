import os
import csv
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from mcap.reader import make_reader

def read_mcap_with_rclpy(file_path):
    rclpy.init()
    timestamps = []
    joint_names = []
    positions = []
    velocities = []
    efforts = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages(topics="/joint_states"):
            joint_state_msg = deserialize_message(message.data, JointState)

            # Extract data from the deserialized message
            timestamps.append(joint_state_msg.header.stamp.sec + joint_state_msg.header.stamp.nanosec / 1e9)
            joint_names = joint_state_msg.name  # Assuming joint names remain constant
            positions.append(joint_state_msg.position)
            velocities.append(joint_state_msg.velocity)
            efforts.append(joint_state_msg.effort)

    rclpy.shutdown()
    return timestamps, joint_names, positions, velocities, efforts

def save_to_csv(timestamps, joint_names, positions, velocities, efforts, output_csv):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Write data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["Timestamp"] + [f"{name}_position" for name in joint_names] + \
                 [f"{name}_velocity" for name in joint_names] + \
                 [f"{name}_effort" for name in joint_names]
        writer.writerow(header)

        # Write rows
        for i in range(len(timestamps)):
            # Convert array.array to list for compatibility
            row = [timestamps[i]] + list(positions[i]) + list(velocities[i]) + list(efforts[i])
            writer.writerow(row)


def process_mcap_to_csv(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mcap"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_csv = os.path.join(output_folder, relative_path, file.replace(".mcap", ".csv"))
                
                print(f"Processing {input_path} -> {output_csv}")
                
                timestamps, joint_names, positions, velocities, efforts = read_mcap_with_rclpy(input_path)
                save_to_csv(timestamps, joint_names, positions, velocities, efforts, output_csv)
                print(f"Saved CSV: {output_csv}")

if __name__ == "__main__":
    input_folder = "RobotJointData/h1"  # Replace with your MCAP folder path
    output_folder = "robot_csv_data"    # Output folder for CSV files
    
    process_mcap_to_csv(input_folder, output_folder)
