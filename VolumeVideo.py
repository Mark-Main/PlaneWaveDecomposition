import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bloodVolumeCreator

# Example usage
grid_size = 100
num_toroids = 10  # Changed from num_spheres for toroids
major_radius_range = (5, 7)  # Adjust the range as needed
minor_radius_range = (1, 2)    # Adjust the range as needed
voxel_resolution = 0.01

resulting_space, x_slices = bloodVolumeCreator.generate_toroids(grid_size, num_toroids, major_radius_range, minor_radius_range, voxel_resolution)
print("Hello")

# Create an animation to go through slices like frames in a video
""" fig, ax = plt.subplots()
slice_start = 0
slice_end = grid_size - 1
slice_index = slice_start

def update(frame):
    global slice_index
    ax.clear()
    ax.imshow(x_slices[slice_index], cmap='gray')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title(f'Slice at X = {slice_index}')
    slice_index = (slice_index + 1) % (slice_end + 1)

ani = FuncAnimation(fig, update, interval=200)  # Interval in milliseconds
plt.show() """


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D voxel grid
ax.voxels(resulting_space, facecolors='red', edgecolor='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()