import os
import pandas as pd

# Base directory for combined IMU data
imu_base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/imu_combined"

# Subdirectories for h0 and h1
sub_dirs = ["h0", "h1"]

# Function to parse time column dynamically
def parse_time_column(time_series):
    try:
        # Attempt auto-parsing
        parsed_time = pd.to_datetime(time_series, errors='coerce')
        if parsed_time.isnull().any():
            print("Warning: Some time values could not be parsed. Check input format.")
        return parsed_time
    except Exception as e:
        print(f"Time parsing failed: {e}")
        return pd.NaT

# Function to calculate time spans
def calculate_time_spans(base_dir, file_suffix):
    time_spans = {}
    total_time_span = 0
    file_count = 0

    for sub_dir in sub_dirs:
        main_dir = os.path.join(base_dir, sub_dir)
        print(f"Processing main directory: {main_dir}")

        if not os.path.exists(main_dir):
            print(f"Directory not found: {main_dir}")
            continue

        for subfolder in os.listdir(main_dir):
            subfolder_dir = os.path.join(main_dir, subfolder)

            if not os.path.isdir(subfolder_dir):
                continue

            for file in os.listdir(subfolder_dir):
                if file.endswith(file_suffix):  # Process only scaled files
                    file_path = os.path.join(subfolder_dir, file)
                    print(f"Processing file: {file_path}")

                    try:
                        # Read the file
                        df = pd.read_csv(file_path)

                        # Dynamically parse the time column
                        df['time'] = parse_time_column(df['time'])

                        if df['time'].isnull().any():
                            print(f"Skipping file due to unparseable time values: {file_path}")
                            continue

                        # Calculate time span
                        start_time = df['time'].iloc[0]
                        end_time = df['time'].iloc[-1]
                        time_span = (end_time - start_time).total_seconds()

                        # Store time span
                        time_spans[file_path] = time_span
                        total_time_span += time_span
                        file_count += 1
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    if time_spans:
        shortest_file = min(time_spans, key=time_spans.get)
        shortest_time_span = time_spans[shortest_file]
        average_time_span = total_time_span / file_count if file_count > 0 else 0

        # Print results
        print("\nTime Spans for Each File:")
        for file, span in time_spans.items():
            print(f"{file}: {span:.2f} seconds")

        print(f"\nShortest Time Span: {shortest_time_span:.2f} seconds (File: {shortest_file})")
        print(f"Average Time Span: {average_time_span:.2f} seconds")
    else:
        print("No valid CSV files found.")

# Process scaled IMU files
calculate_time_spans(imu_base_dir, "_scaled.csv")
