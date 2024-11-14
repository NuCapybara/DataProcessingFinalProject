import pandas as pd
import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("emg_csv_data/h0/H_r1deg0h0/H_r1deg0h0_RL_emg.csv")
# Parse the _data column to extract arrays and create a column for channel_1
parsed_channel_1 = []

for row in data['_data']:
    # Convert the string representation of the array to a list of integers
    array = ast.literal_eval(row.split("'h', ")[1].strip(")"))
    # Extract only channel_1 (assuming it’s the second element in the array)
    parsed_channel_1.append(array[1])

# Add parsed channel_1 data to the DataFrame
data['channel_1'] = parsed_channel_1

# Prepare data for clustering: cluster based on sample index only
# Add a sample index column
data['sample_index'] = data.index

# Apply K-Means clustering on sample_index to divide horizontally
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
data['Cluster'] = kmeans.fit_predict(data[['sample_index']])

# Plot the channel_1 data, color-coded by cluster
plt.figure(figsize=(12, 6))
plt.scatter(data['sample_index'], data['channel_1'], c=data['Cluster'], cmap="viridis", alpha=0.6)
plt.colorbar(label="Cluster")
plt.xlabel("Sample Index")
plt.ylabel("Channel 1 EMG Signal")
plt.title("Channel 1 EMG Data Clustering into 8 Clusters (Horizontal Clustering)")
plt.show()
