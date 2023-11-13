import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate points on a torus
def generate_torus(R, r, num_points):
    phi = np.linspace(0, 2*np.pi, num_points)
    theta = np.linspace(0, 2*np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

# Number of points on the torus
num_points = 100

# Radii of the torus
R = 25  # Major radius (increase for bigger torus)
r = 22   # Minor radius (decrease for smaller torus)

# Grid parameters
grid_size = 200  # Increase for a larger grid
voxel_resolution = 1.0

# Create a 3D plot
fig = plt.figure()

# Plot the original toroids
ax_original = fig.add_subplot(121, projection='3d')
ax_original.set_title('Original Toroids')

# List to store toroid positions
toroid_positions = []

# Generate and plot 10 toroids with spacing and collision detection
for _ in range(100):
    toroid_collides = True

    while toroid_collides:
        # Generate torus points
        x, y, z = generate_torus(R, r, num_points)

        # Apply random rotation
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, 2*np.pi)

        x_rotated = x * np.cos(theta) * np.cos(phi) - y * np.sin(phi) * np.cos(theta) + z * np.sin(theta)
        y_rotated = x * np.sin(phi) + y * np.cos(phi)
        z_rotated = -x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(theta)

        # Scale the rotated torus down
        x_scaled = x_rotated / 5  # Adjust the scaling factor as needed
        y_scaled = y_rotated / 5
        z_scaled = z_rotated / 5

        # Set the toroid position with spacing
        x_offset = np.random.uniform(-100, 100)  # Adjust the range for spacing
        y_offset = np.random.uniform(-100, 100)
        z_offset = np.random.uniform(-100, 100)

        x_final = x_scaled + x_offset
        y_final = y_scaled + y_offset
        z_final = z_scaled + z_offset

        # Check for collisions with existing toroids
        toroid_collides = any(
            np.linalg.norm(np.array([x_final, y_final, z_final]) - pos) < 2 * r
            for pos in toroid_positions
        )

    # Store the toroid position
    toroid_positions.append(np.array([x_final, y_final, z_final]))

    # Plot the scaled and positioned torus
    ax_original.plot_surface(x_final, y_final, z_final, cmap='viridis', alpha=0.5)

# Set labels
ax_original.set_xlabel('X')
ax_original.set_ylabel('Y')
ax_original.set_zlabel('Z')

# Voxelization and plotting of the voxelized grid
voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
for pos in toroid_positions:
    x_indices = np.clip((pos[0] + grid_size / 2).astype(int), 0, grid_size - 1)
    y_indices = np.clip((pos[1] + grid_size / 2).astype(int), 0, grid_size - 1)
    z_indices = np.clip((pos[2] + grid_size / 2).astype(int), 0, grid_size - 1)
    voxel_grid[x_indices, y_indices, z_indices] = 1

# Plot the voxelized version
ax_voxel = fig.add_subplot(122, projection='3d')
ax_voxel.set_title('Voxelized Toroids')
ax_voxel.voxels(voxel_grid, facecolors='red',edgecolor='r')

plt.show()