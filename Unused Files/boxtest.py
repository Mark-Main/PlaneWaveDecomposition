import numpy as np
import random
import matplotlib.pyplot as plt

# Define parameters
grid_size = 100
num_spheres = 50  # Adjust the number of spheres
min_radius = 1
max_radius = 10
sphere_radii = np.random.uniform(min_radius, max_radius, num_spheres)
voxel_resolution = 0.1  # Increase the resolution for each sphere

# Initialize the 3D grid
space = np.zeros((grid_size, grid_size, grid_size), dtype=int)

# Place spheres randomly within the grid
for radius in sphere_radii:
    is_overlap = True
    while is_overlap:
        sphere_center = (
            random.randint(0, grid_size - max_radius),
            random.randint(0, grid_size - max_radius),
            random.randint(0, grid_size - max_radius)
        )
        
        is_overlap = False
        int_radius = int(radius)
        for x in range(max(0, sphere_center[0] - int_radius), min(grid_size, sphere_center[0] + int_radius + 1)):
            for y in range(max(0, sphere_center[1] - int_radius), min(grid_size, sphere_center[1] + int_radius + 1)):
                for z in range(max(0, sphere_center[2] - int_radius), min(grid_size, sphere_center[2] + int_radius + 1)):
                    distance = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
                    if distance <= radius:
                        if space[x, y, z] == 1:
                            is_overlap = True
                            break
                if is_overlap:
                    break
            if is_overlap:
                break
        
    # Update the space with the new sphere
    int_voxel_resolution = int(voxel_resolution * radius)
    for x in range(max(0, sphere_center[0] - int_radius), min(grid_size, sphere_center[0] + int_radius + 1)):
        for y in range(max(0, sphere_center[1] - int_radius), min(grid_size, sphere_center[1] + int_radius + 1)):
            for z in range(max(0, sphere_center[2] - int_radius), min(grid_size, sphere_center[2] + int_radius + 1)):
                distance = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
                if distance <= radius:
                    space[x, y, z] = 1

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D voxel grid
ax.voxels(space, facecolors='red', edgecolor='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
