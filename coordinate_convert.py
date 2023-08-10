import numpy as np


def coordinate_shift(center_coordinates, gaussRes, x2, y2):
    def map_coordinates_to_indices(coordinates, x_array, y_array):
        x_indices = np.round((np.array([x[0] for x in coordinates]) + 0.2) * (gaussRes - 1) / 0.4).astype(int)
        y_indices = np.round((np.array([x[1] for x in coordinates]) + 0.2) * (gaussRes - 1) / 0.4).astype(int)
        return x_indices, y_indices

    # Get the mapped indices for center_coordinates
    x_indices, y_indices = map_coordinates_to_indices(center_coordinates, x2, y2)

    # Convert indices to 0 to gaussRes - 1 range
    x_indices = np.clip(x_indices, 0, gaussRes - 1)
    y_indices = np.clip(y_indices, 0, gaussRes - 1)

    # Create a new array with the same structure as center_coordinates
    new_coordinates = [(x_indices[i], y_indices[i]) for i in range(len(center_coordinates))]
    return new_coordinates
    