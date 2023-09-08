import numpy as np
import matplotlib.pyplot as plt

def generate_filled_toroid(R, r, num_points):
    toroid_points = []

    for theta in np.linspace(0, 2*np.pi, num_points):
        for phi in np.linspace(0, 2*np.pi, num_points):
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = (R + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            toroid_points.append((x, y, z))

    return toroid_points

def voxelization(toroid_points, grid_size):
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    grid_min = -grid_size // 2
    grid_max = grid_size // 2

    for point in toroid_points:
        x, y, z = point
        if grid_min <= x < grid_max and grid_min <= y < grid_max and grid_min <= z < grid_max:
            x_idx = int(x + grid_size // 2)
            y_idx = int(y + grid_size // 2)
            z_idx = int(z + grid_size // 2)
            voxel_grid[x_idx, y_idx, z_idx] = 1

    return voxel_grid

# Toroid parameters
R = 5
r = 2
num_points = 1000
grid_size = 30  # Adjust grid size as needed

# Generate filled toroid points
toroid_points = generate_filled_toroid(R, r, num_points)

# Voxelization
voxel_grid = voxelization(toroid_points, grid_size)

# Visualization (example slice along the x-axis)
plt.imshow(voxel_grid[grid_size // 3, :, :], cmap='binary')
plt.title("Voxelized Toroid Slice")
plt.show()