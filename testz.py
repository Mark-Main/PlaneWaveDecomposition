import numpy as np
import matplotlib.pyplot as plt

def generate_torus(R, r, num_points):
    phi = np.linspace(0, 2*np.pi, num_points)
    theta = np.linspace(0, 2*np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

def generate_voxelized_toroids(num_toroids, R, r, num_points, grid_size, voxel_resolution):
    toroid_positions = []
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    for _ in range(num_toroids):
        toroid_collides = True

        while toroid_collides:
            x, y, z = generate_torus(R, r, num_points)

            phi = np.random.uniform(0, 2*np.pi)
            theta = np.random.uniform(0, 2*np.pi)

            x_rotated = x * np.cos(theta) * np.cos(phi) - y * np.sin(phi) * np.cos(theta) + z * np.sin(theta)
            y_rotated = x * np.sin(phi) + y * np.cos(phi)
            z_rotated = -x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(theta)

            x_scaled = x_rotated / 5
            y_scaled = y_rotated / 5
            z_scaled = z_rotated / 5

            x_offset = np.random.uniform(-100, 100)
            y_offset = np.random.uniform(-100, 100)
            z_offset = np.random.uniform(-100, 100)

            x_final = x_scaled + x_offset
            y_final = y_scaled + y_offset
            z_final = z_scaled + z_offset

            toroid_collides = any(
                np.linalg.norm(np.array([x_final, y_final, z_final]) - pos) < 2 * r
                for pos in toroid_positions
            )

        toroid_positions.append(np.array([x_final, y_final, z_final]))

        x_indices = np.clip((x_final + grid_size / 2).astype(int), 0, grid_size - 1)
        y_indices = np.clip((y_final + grid_size / 2).astype(int), 0, grid_size - 1)
        z_indices = np.clip((z_final + grid_size / 2).astype(int), 0, grid_size - 1)
        voxel_grid[x_indices, y_indices, z_indices] = 1

    return voxel_grid

# Parameters
num_toroids = 100
R = 25
r = 22
num_points = 100
grid_size = 200
voxel_resolution = 1.0

# Generate voxelized grid with toroids
voxel_grid = generate_voxelized_toroids(num_toroids, R, r, num_points, grid_size, voxel_resolution)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D voxel grid
ax.voxels(voxel_grid, facecolors='red', edgecolor='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
