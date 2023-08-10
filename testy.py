import numpy as np

center_coordinates = [(0, 0), (0, 0.05), (-0.043, 0.025), (0.043, 0.025), (-0.043, -0.025), (0.043, -0.025), (0, -0.05)]
gaussRes = 512

x2 = np.linspace(-0.2, 0.2, gaussRes)
y2 = np.linspace(-0.2, 0.2, gaussRes)

# Function to map coordinates to indices
def map_coordinates_to_indices(coordinates, x_array, y_array):
    x_indices = np.argmin(np.abs(x_array[:, np.newaxis] - np.array([x[0] for x in coordinates])), axis=0)
    y_indices = np.argmin(np.abs(y_array[:, np.newaxis] - np.array([x[1] for x in coordinates])), axis=0)
    return x_indices, y_indices

# Get the mapped indices for center_coordinates
x_indices, y_indices = map_coordinates_to_indices(center_coordinates, x2, y2)

# Create a new array with the same structure as center_coordinates
new_coordinates = [(x2[x_indices[i]], y2[y_indices[i]]) for i in range(len(center_coordinates))]

print(new_coordinates)
