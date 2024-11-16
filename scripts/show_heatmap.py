import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure a CSV file is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <csv_file>")
    sys.exit(1)

# Read the CSV file from the command-line argument
csv_file = sys.argv[1]

# Load the data from the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file)

# Convert DataFrame to a NumPy array (if necessary)
data_matrix = data.values

# Create a heatmap with grayscale (black to white), with color normalization
plt.figure(figsize=(8, 6))
sns.heatmap(data_matrix, cbar=True)

# Set labels and title (optional)
plt.title('Heatmap from CSV')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the heatmap
plt.show()

