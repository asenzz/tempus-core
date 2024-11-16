import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Load data from CSV file
data = np.loadtxt('/tmp/tune_params.csv', delimiter=' ', skiprows=0)

# Extract columns for X, Y, Z
x = data[:, 1]  # First column
y = data[:, 2]  # Second column
z = data[:, 0]  # Third column

# Create a grid for X and Y values
X_unique = np.unique(x)
Y_unique = np.unique(y)
X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)

# Interpolate Z values on the grid
Z_grid = griddata((x, y), z, (X_grid, Y_grid), method='linear')

# Create a figure for the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Surface Plot from CSV Data')

# Add a color bar to indicate values
fig.colorbar(surf)

# Show the plot
plt.show()

