import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bloodVolumeCreator

# Python3 implementation of the floodFill algorithm
def isValid(screen, m, n, x, y, prevC, newC):
    if x < 0 or x >= m or y < 0 or y >= n or screen[x][y] != prevC or screen[x][y] == newC:
        return False
    return True

def floodFill(screen, m, n, x, y, prevC, newC):
    queue = [(x, y)]
    screen[x][y] = newC
        
    while queue:
        posX, posY = queue.pop(0)
        
        if posX + 1 < m and screen[posX + 1][posY] == prevC:
            queue.append((posX + 1, posY))
            screen[posX + 1][posY] = newC
        
        if posX - 1 >= 0 and screen[posX - 1][posY] == prevC:
            queue.append((posX - 1, posY))
            screen[posX - 1][posY] = newC
        
        if posY + 1 < n and screen[posX][posY + 1] == prevC:
            queue.append((posX, posY + 1))
            screen[posX][posY + 1] = newC
        
        if posY - 1 >= 0 and screen[posX][posY - 1] == prevC:
            queue.append((posX, posY - 1))
            screen[posX][posY - 1] = newC

# Example usage
grid_size = 100
num_toroids = 3
r = 40
R = 60
voxel_resolution = 1

resulting_space, x_slices = bloodVolumeCreator.generate_voxelized_toroids(grid_size, num_toroids, R, r)

# Apply flood fill to each slice in x_slices
filled_slices = []
x = 0  # Co-ordinate provided by the user
y = 0
prevC = 0  # Current color at that co-ordinate
newC = 255  # New color that has to be filled

for slice in x_slices:
    filled_slice = np.copy(slice)
    floodFill(filled_slice, len(slice), len(slice[0]), x, y, prevC, newC)
    filled_slices.append(filled_slice)

# Create an animation to cycle through slices
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

slice_start = 0
slice_end = grid_size - 1
slice_index = slice_start

def update(frame):
    global slice_index
    axs[0].clear()
    axs[0].imshow(x_slices[slice_index], cmap='gray')
    axs[0].set_title(f'Original Slice at X = {slice_index}')
    axs[1].clear()
    axs[1].imshow(filled_slices[slice_index], cmap='gray')
    axs[1].set_title(f'Filled Slice at X = {slice_index}')
    slice_index = (slice_index + 1) % (slice_end + 1)

ani = FuncAnimation(fig, update, interval=100)  # Interval in milliseconds
plt.show()