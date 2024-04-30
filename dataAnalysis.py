import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
from scipy import stats
import random
import pandas as pd

base_directory = r'/Volumes/DeBroglie/SVDSphereModal5_98Sphere'
output_directory = r'/Volumes/DeBroglie/DataPlots'


#Analysis of full run 
# Analysis and plotting of the results
num_files = 100
file_pattern = 'Overlap_Values_Run_{}.npy'  # File naming pattern

# Initialize a list to hold the data
data = []

# Load the data from each file
for i in range(1, num_files + 1):
    file_path = os.path.join(base_directory, file_pattern.format(i))
    array = np.load(file_path)
    data.append(array)

# Convert the list of arrays into a 3D numpy array
data_stack = np.stack(data, axis=2)

# Calculate average, median,  mode and sd
average = np.mean(data_stack, axis=2)
median = np.median(data_stack, axis=2)
mode = stats.mode(data_stack, axis=2).mode.squeeze()
std_dev = np.std(data_stack, axis=2)

# Save the average (mean) to a CSV file
mean_csv_file_path = os.path.join(output_directory, 'SVDSphere98_Modal5.csv')
pd.DataFrame(mode).to_csv(mean_csv_file_path, index=False, header=False)


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 22))  # Slightly increase figure height to give more room at the top

cmap = 'viridis'  # Color map

# Add an overall title with more space at the top
fig.suptitle('SVD Sphere 99.9%', fontsize=20, y=0.95)  # Adjust y to lower the title position

# Plot Average
im0 = axes[0, 0].imshow(average, cmap=cmap)
axes[0, 0].set_title('Average')
fig.colorbar(im0, ax=axes[0, 0], label='Value')

# Plot Median
im1 = axes[0, 1].imshow(median, cmap=cmap)
axes[0, 1].set_title('Median')
fig.colorbar(im1, ax=axes[0, 1], label='Value')

# Plot Mode
im2 = axes[1, 0].imshow(mode, cmap=cmap)
axes[1, 0].set_title('Mode')
fig.colorbar(im2, ax=axes[1, 0], label='Value')

# Plot Standard Deviation
im3 = axes[1, 1].imshow(std_dev, cmap=cmap)
axes[1, 1].set_title('Standard Deviation')
fig.colorbar(im3, ax=axes[1, 1], label='Value')

# Adjust layout to ensure everything fits
fig.subplots_adjust(top=0.9)  # Lower the top adjustment to provide more space for the title

plt.show()