import os
import pandas as pd

# Base directory for combined EMG data
base_dir = "/home/jialuyu/Data_Final_Project/DataProcessingFinalProject/emg_csv_data/emg_combined"

# Subdirectories for h0 and h1
sub_dirs = ["h0", "h1"]

# Initialize variables
time_spans = {}  # Dictionary to store file name and its time span
total_time_span = 0
file_count = 0

# Loop through h0 and h1 directories
for sub_dir in sub_dirs:
    main_dir = os.path.join(base_dir, sub_dir)
    print(f"Processing main directory: {main_dir}")  # Debugging info

    if not os.path.exists(main_dir):
        print(f"Directory not found: {main_dir}")
        continue

    # Loop through subfolders in h0/h1
    for subfolder in os.listdir(main_dir):
        transformed_dir = os.path.join(main_dir, subfolder, "transformed")
        print(f"Checking 'transformed' folder: {transformed_dir}")  # Debugging info

        if not os.path.exists(transformed_dir):
            print(f"No 'transformed' folder in {os.path.join(main_dir, subfolder)}")
            continue

        # Process all CSV files in the transformed directory
        for file in os.listdir(transformed_dir):
            if file.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(transformed_dir, file)
                print(f"Processing file: {file_path}")  # Debugging info

                try:
                    # Read the file
                    df = pd.read_csv(file_path)
                    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d/%H:%M:%S.%f')

                    # Calculate the time span
                    start_time = df['time'].iloc[0]
                    end_time = df['time'].iloc[-1]
                    time_span = (end_time - start_time).total_seconds()

                    # Store the time span
                    time_spans[file_path] = time_span
                    total_time_span += time_span
                    file_count += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# Calculate the shortest and average time spans
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
    print("No CSV files found.")
