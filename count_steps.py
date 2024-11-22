import os
import csv

def count_rows_in_csv(folder_path):
    """
    Loop through CSV files in a folder and count the number of rows in each file.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                try:
                    # Count rows in the CSV file
                    with open(file_path, mode='r') as csv_file:
                        reader = csv.reader(csv_file)
                        row_count = sum(1 for row in reader) - 1  # Subtract 1 for the header row

                    print(f"{file}: {row_count} rows")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    folder_path = "emg_csv_data/emg_scale/h0"  # Replace with your CSV folder path
    count_rows_in_csv(folder_path)
