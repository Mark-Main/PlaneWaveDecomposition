import os
import matplotlib.pyplot as plt
import numpy as np

# Define the base directory
base_directory = r'/Users/Mark/Documents/Test'
intensity_data = [[[] for _ in range(3)] for _ in range(3)]

for p in range(3):
    for l in range(3):
        output_directory = os.path.join(base_directory, f'P={p}, L={l}')
        os.makedirs(output_directory, exist_ok=True)  # Create folder if it doesn't exist
        for i in range(800):
            # Load the file
            loaded_data = np.load(f'{output_directory}//BinaryDataSlice{i}_p={p}_l={l}.npy')
            intensity_data[p][l].append(np.abs(loaded_data) ** 2 / np.max(np.abs(loaded_data) ** 2))
        print ("p = ",p, "l= ",l)

# Create a 6x6 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Create a new folder to save all the plots
output_folder = os.path.join(base_directory, 'AllPlots')
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

for i in range(800) :  # Assuming you have 5 slices
    # Create a 6x6 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for p in range(3):
        for l in range(3):

            axs[p, l].imshow(intensity_data[p][l][i], cmap='inferno')
            axs[p, l].axis('off')

            if p == 0:
                axs[p, l].text(0.5, 1.15, f'L={l}', va='center', ha='center', fontsize=16, transform=axs[p, l].transAxes)
            if l == 0:
                axs[p, l].text(-0.15, 0.5, f'P={p}', va='center', ha='right', rotation='vertical', fontsize=16, transform=axs[p, l].transAxes)

    # Add a label indicating the current slice
    fig.text(0.95, 0.95, f'Slice = {i}', fontsize=16, ha='right')

    # Save the plot to the specified folder
    output_filename = os.path.join(output_folder, f'plot_{i}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # Save the plot
    plt.close()

