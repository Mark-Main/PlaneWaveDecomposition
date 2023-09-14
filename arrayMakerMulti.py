import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

# Define the base directory
base_directory = r'D:\MarkSimulations\SphericalTrialLP6'
intensity_data = [[[] for _ in range(6)] for _ in range(6)]

def process_data(p, l):
    output_directory = os.path.join(base_directory, f'P={p}, L={l}')
    os.makedirs(output_directory, exist_ok=True)  # Create folder if it doesn't exist
    data = []
    for i in range(600):
        loaded_data = np.load(f'{output_directory}\\BinaryDataSlice{i}_p={p}_l={l}.npy')
        data.append(np.abs(loaded_data) ** 2 / np.max(np.abs(loaded_data) ** 2))
    return p, l, data

def generate_plot(args):
    p, l, data = args
    fig, axs = plt.subplots(6, 6, figsize=(15, 15))
    for i in range(600):  # Assuming you have 5 slices
        for p_idx in range(6):
            for l_idx in range(6):
                axs[p_idx, l_idx].imshow(data[p_idx][l_idx][i], cmap='inferno')
                axs[p_idx, l_idx].axis('off')
                if p_idx == 0:
                    axs[p_idx, l_idx].text(0.5, 1.15, f'L={l_idx}', va='center', ha='center', fontsize=16,
                                           transform=axs[p_idx, l_idx].transAxes)
                if l_idx == 0:
                    axs[p_idx, l_idx].text(-0.15, 0.5, f'P={p_idx}', va='center', ha='right', rotation='vertical',
                                           fontsize=16, transform=axs[p_idx, l_idx].transAxes)
        fig.text(0.95, 0.95, f'Slice = {i}', fontsize=16, ha='right')
        output_folder = os.path.join(base_directory, 'AllPlots')
        os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
        output_filename = os.path.join(output_folder, f'plot_{i}_p={p}_l={l}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # Save the plot

if __name__ == '__main__':
    # Use multiprocessing to load data in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_data, [(p, l) for p in range(6) for l in range(6)])
    
    for result in results:
        p, l, data = result
        intensity_data[p][l] = data
    
    # Generate plots in parallel
    with Pool(cpu_count()) as pool:
        pool.map(generate_plot, [(p, l, intensity_data[p][l]) for p in range(6) for l in range(6)])
