import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
from scipy import stats
import random
import pandas as pd

# Import your custom modules
import bloodGenerator
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import computeWave
import propogator
import itertools

# Global parameters
base_directory = r'/Volumes/DeBroglie/SVDCheck'
mask_directory = r'/Volumes/DeBroglie/DiscScreens'
waveRes = 500
sliceThickness = 40
discocyte_radius = 16
num_normal_discocytes = 0
num_bumpy_discocytes = 120
num_crescents = 0
bump_radius = 10
number_of_bumps = 15
cellSize = 8e-6
w0 = 20e-6
x = np.linspace(-250e-6, 250e-6, waveRes)
y = np.linspace(-250e-6, 250e-6, waveRes)
X, Y = np.meshgrid(x, y)
grid_size = waveRes
screen_dimensions = (waveRes, waveRes, sliceThickness)
num_processes = min(9, multiprocessing.cpu_count())

run_count = 1

# Function to generate and save phase screen
def generate_and_save_phase_screen(slice_number):
    simulated_volume, _ = bloodGenerator.simulate_discocytes(
        screen_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes, num_crescents, bump_radius, number_of_bumps)

    # Constants
    n_white = 1.5  # Refractive index for parts represented by 1 in simulated_volume
    n_non_white = 1.3  # Refractive index for parts represented by 0 in simulated_volume
    wavelength = 850e-9  # Wavelength in meters
    voxel_size = cellSize/(discocyte_radius*2) # Size of one voxel in meters

    # Assign refractive indices
    refractive_indices = np.where(simulated_volume == 1, n_white, n_non_white)

    # Calculate Optical Path Length (OPL) and phase shift
    OPL = refractive_indices * voxel_size
    phase_shift = (2 * np.pi / wavelength) * OPL

    # Project 3D phase data onto 2D plane
    phase_screen = np.sum(phase_shift, axis=2)

    # Save the phase screen
    output_directory = os.path.join(base_directory, f'PhaseScreens')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'phase_screen_{slice_number}.npy'), phase_screen)

    return f'Phase screen {slice_number} saved'

# Function to process l
def process_l(l):
    phaseshifttip_init = 0
    phaseshifttilt_init = 0
    s = 0.2  # Aperture size
    λ = 850e-9  # Wavelength of light
    computeStep = 1  # Computation step distance
    finalDistance = 10  # Total propagation distance
    scatter_event_count = 10

    oldwave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, l, w0)
    tiptilt = tilt_func.tilttip(waveRes, phaseshifttip_init, phaseshifttilt_init)
    oldwave = computeWave.makeWave(oldwave, tiptilt)
    oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, waveRes, s, λ))

    overlap_importWave = np.trapz(np.conj(oldwave) * oldwave, dx=x[1] - x[0], axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)
    oldwave = oldwave / np.sqrt(overlapy_importWave)

    output_directory = os.path.join(base_directory, f'L={l}')

    for i in tqdm(range(finalDistance // computeStep), desc=f"L={l}", ascii=False, ncols=75):
        if i < scatter_event_count:
            random_index = random.randint(0, 4999)  # Use a random integer from 0 to 4999
            csv_file_path = os.path.join(mask_directory, f'phase_screen_{random_index}.csv')
            if os.path.exists(csv_file_path):
                # Load the CSV file, convert it to a NumPy array, and then save it as .npy for future use
                scattermask = pd.read_csv(csv_file_path, header=None).to_numpy()
                
                npy_file_path = csv_file_path.replace('.csv', '.npy')
                np.save(npy_file_path, scattermask)
            else:
                # Load the NPY file if the CSV conversion has already been done
                npy_file_path = os.path.join(mask_directory, f'phase_screen_{random_index}.npy')
                scattermask = np.load(npy_file_path)

            wave = propogator.propogateScatterMask(oldwave, computeStep * 1e-5, waveRes, s, λ, scattermask)
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep, waveRes, s, λ)
            oldwave = wave

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_l={l}.npy'), oldwave)


# Function to perform decomposition
def decompose(inputL_outputL):
    inputL, outputL = inputL_outputL

    refWave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, inputL, w0)
    output_directory = os.path.join(base_directory, f'L={outputL}')
    importWave = np.load(os.path.join(output_directory, f'BinaryDataSlice_l={outputL}.npy'))

    overlap_importWave = np.trapz(np.conj(importWave)*importWave, dx=x[1] - x[0], axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)

    overlap_refWave = np.trapz(np.conj(refWave)*refWave, dx=x[1] - x[0])
    overlapy_refWave = np.trapz(overlap_refWave)

    normalized_importWave = importWave / np.sqrt(overlapy_importWave)
    normalized_refWave = refWave / np.sqrt(overlapy_refWave)

    overlap = np.trapz(np.conj(normalized_refWave)*normalized_importWave, dx=x[1] - x[0], axis=0)
    overlap = np.trapz(overlap)

    calculated_value = np.abs(overlap)

    return inputL, outputL, calculated_value



if __name__ == '__main__':
    for run_number in range(run_count):
        print(f"Run {run_number + 1}/{run_count}")

        # Task 2: Perform Propagation
        pool = multiprocessing.Pool(processes=num_processes)
        l_values = list(range(50))  # Adjust as needed
        for _ in tqdm(pool.imap_unordered(process_l, l_values), total=len(l_values), desc="Processing l values"):
            pass
        pool.close()
        pool.join()

        # Task 3: Decompose with multiprocessing
        pool = multiprocessing.Pool(processes=num_processes)
        all_combinations = list(itertools.product(range(50), repeat=2))  # Create all (inputL, outputL) combinations

        # Initialize overlap_values array
        overlap_values = np.zeros((50, 50))

        # Start multiprocessing pool for decomposition
        results = pool.imap_unordered(decompose, all_combinations)

        for inputL, outputL, calculated_value in tqdm(results, total=len(all_combinations), desc="Decomposing", ascii=False, ncols=75):
            overlap_values[inputL][outputL] = calculated_value

        pool.close()
        pool.join()

        # Save overlap values data
        np.save(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.npy'), overlap_values)
        np.savetxt(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.csv'), overlap_values, delimiter=',')

        # Plotting and Saving the Matrix
        plt.figure(figsize=(10, 8))  # Set the figure size (width, height)
        plt.imshow(overlap_values[::-1], origin='lower', extent=(0, 49, 0, 49))
        plt.colorbar(label='Overlap Value')
        plt.xlabel('Input L', fontsize=18)
        plt.ylabel('Output L', fontsize=18)
        plt.xticks(np.arange(0, 50, step=1), fontsize=12)
        plt.yticks(np.arange(0, 50, step=1), fontsize=12)
        plt.title(f'Overlap Values for Discs - Run {run_number + 1}', fontsize=16)
        plt.savefig(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.png'), dpi=600)
        plt.close()

    # Generate reference LG modes
# Assuming generateLaguerre2DnoZ.laguerre_gaussian, X, Y, and w0 are already defined
        
    data = []
    file_pattern = 'Overlap_Values_Run_{}.npy'
    # Load the data from each file
    for i in range(1, run_count + 1):
        file_path = os.path.join(base_directory, file_pattern.format(i))
        array = np.load(file_path)
        data.append(array)

    # Convert the list of arrays into a 3D numpy array
    data_stack = np.stack(data, axis=2)

    # Calculate average, median,  mode and sd
    average = np.mean(data_stack, axis=2)
    refLGstorage = [None] * 50  # Pre-allocate storage for 50 modes

    for l in range(50):
        # Generate each Laguerre-Gaussian mode
        mode = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, l, w0)
        
        # Compute the power of the mode
        power = np.abs(mode)**2
        
        # Normalize the power to range from 0 to 1
        normalized_power = (power - np.min(power)) / (np.max(power) - np.min(power))
        
        # Store the normalized power (or mode, depending on your requirement) back in refLGstorage
        refLGstorage[l] = normalized_power 

    def multiply_lg_waves_by_svd(lg_waves, svd_vector):
        result = np.zeros_like(lg_waves[0])
        for i in range(len(lg_waves)):
            result += lg_waves[i] * svd_vector[i]
        return result
    

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # Create a grid of 3 rows and 5 columns for the subplots
    U, S, VT = np.linalg.svd(average)
    for mode in range(15):  # Iterate through the first 15 modes
        row = mode // 5  # Determine the row of the current mode
        col = mode % 5   # Determine the column of the current mode
        
        svd_mode_intensity = np.abs(multiply_lg_waves_by_svd(refLGstorage, VT[mode]))  # Calculate mode intensity
        
        ax = axes[row, col]  # Select the appropriate subplot
        cax = ax.imshow(svd_mode_intensity, cmap='viridis')  # Display the mode intensity as an image
        ax.set_title(f'SVD Mode {mode + 1}')  # Set the title for the subplot
        
        fig.colorbar(cax, ax=ax)  # Add a colorbar for each subplot to indicate intensity scale

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.show()  # Display the figure

    num_modes = 15  # Number of modes to compare
    dot_product_matrix = np.zeros((num_modes, num_modes))
    
    for i in range(num_modes):
        for j in range(num_modes):
            # Calculate the dot product between the decomposed vectors for SVD modes i and j
            dot_product_matrix[i, j] = np.dot(VT[i], np.conj(VT[j]))

    # Plot the dot product matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(dot_product_matrix, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title('Dot Product Matrix of SVD Modes')
    ax.set_xlabel('SVD Mode Index')
    ax.set_ylabel('SVD Mode Index')
    ax.set_xticks(np.arange(num_modes))
    ax.set_yticks(np.arange(num_modes))
    ax.set_xticklabels(np.arange(1, num_modes + 1))
    ax.set_yticklabels(np.arange(1, num_modes + 1))
    plt.show()
   