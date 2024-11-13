import os

def delete_plot_files(data_base_dir):
    # Iterate over each subfolder inside the given data directory
    for subfolder in os.listdir(data_base_dir):
        subfolder_path = os.path.join(data_base_dir, subfolder)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Find all PNG files (plots) in the current subfolder
        plot_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]

        # Delete each plot file found
        for plot_file in plot_files:
            plot_file_path = os.path.join(subfolder_path, plot_file)
            try:
                os.remove(plot_file_path)
                print(f"Deleted plot file: {plot_file_path}")
            except OSError as e:
                print(f"Error: Could not delete file {plot_file_path}: {e}")

if __name__ == "__main__":
    # Directory containing the processed CSV data and plots
    data_base_dir = "emg_csv_data"  # Adjust this path accordingly

    # Run the deletion function
    delete_plot_files(data_base_dir)
