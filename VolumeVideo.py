import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bloodVolumeCreator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Example usage
grid_size = 100
num_toroids = 60  # Changed from num_spheres for toroids
r=20
R=25  # Adjust the range as needed
voxel_resolution = 1

resulting_space, x_slices = bloodVolumeCreator.generate_voxelized_toroids(grid_size, num_toroids, R,r)
print("Hello")

# Create an animation to go through slices like frames in a video
fig, ax = plt.subplots()
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
plt.show() 

'''
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D voxel grid
ax.voxels(resulting_space, facecolors='red', edgecolor='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
'''