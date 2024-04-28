import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import joblib

# Load the dataset
data = pd.read_csv("network_dataset.csv")

# Visualize Length distribution using a bar chart
plt.figure(figsize=(10, 6))
plt.hist(data["Length"], bins=30, color='skyblue', edgecolor='black')
plt.title('Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Load the dataset
data = pd.read_csv("network_dataset.csv")

# Calculate Protocol distribution
protocol_counts = data["Protocol"].value_counts()

# Plot Protocol distribution using a pie chart
plt.figure(figsize=(8, 8))
plt.pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Protocol Distribution')
plt.axis('equal')
plt.show()

# Count occurrences of unique source-destination pairs
traffic_flow = data.groupby(['Source', 'Destination']).size().unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(traffic_flow, cmap="Blues", annot=True, fmt="d")
plt.title("Traffic Flow (Source vs Destination)")
plt.xlabel("Destination IP Address")
plt.ylabel("Source IP Address")
plt.show()

# Convert the 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'], unit='s')

# Group by 'Time' and 'Protocol' and count the packets
protocol_counts = data.groupby(['Time', 'Protocol']).size().unstack(fill_value=0)

# Plot the packet protocol over time
plt.figure(figsize=(12, 8))

# Loop through each protocol and plot its count over time
for protocol in protocol_counts.columns:
    plt.plot(protocol_counts.index, protocol_counts[protocol], label=protocol)

plt.xlabel('Time')
plt.ylabel('Packet Count')
plt.title('Packet Protocol Over Time')
plt.legend()
plt.show()

# Create a directed graph
G = nx.from_pandas_edgelist(data, source='Source', target='Destination', create_using=nx.DiGraph())

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray', arrowsize=20)
plt.title("Network Graph of Traffic Flow")
plt.show()

# Calculate total traffic volume (e.g., count of packets) per time unit (e.g., per second)
traffic_volume = data.groupby('Time').size()

# Set threshold for anomaly detection (e.g., based on mean and standard deviation)
threshold = traffic_volume.mean() + 2 * traffic_volume.std()

# Detect anomalies
anomalies = traffic_volume[traffic_volume > threshold]

# Plot traffic volume over time
plt.figure(figsize=(12, 6))
plt.plot(traffic_volume.index, traffic_volume.values, color='blue', label='Traffic Volume')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume Over Time')
plt.legend()
plt.show()

print("Detected Anomalies:")
print(anomalies)


import joblib

# Assuming 'clf' is your trained DecisionTreeClassifier model

# Save the model to a file
joblib.dump(None, 'decision_tree_model.pkl')