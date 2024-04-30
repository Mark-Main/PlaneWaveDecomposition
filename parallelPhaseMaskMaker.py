import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
import bloodGenerator

base_directory = r'/Volumes/DeBroglie/PhaseScreens/'
# Parameters
# Laguerre-Gaussian parameters

waveRes = 500
# Set simulation parameters
sliceThickness = 40
volume_dimensions = (waveRes, waveRes,sliceThickness)
discocyte_radius = 16  # Radius for discocytes
num_normal_discocytes1 = 140 # Number of normal discocytes
num_bumpy_discocytes1 =0  # Number of bumpy discocytes
num_normal_discocytes2 = 0 # Number of normal discocytes
num_bumpy_discocytes2 =140  # Number of bumpy discocytes
num_crescents1 =000
bump_radius = 8 # Radius of the bumps
number_of_bumps = 15  # Number of bumps
cellSize = 8e-6


def generate_and_save_phase_screen1(slice_number):
    simulated_volume, _ = bloodGenerator.simulate_discocytes(
        volume_dimensions, discocyte_radius, num_normal_discocytes1, num_bumpy_discocytes1, num_crescents1, bump_radius, number_of_bumps)

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
    output_directory = os.path.join(base_directory, f'PhaseScreensDisc')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    #np.save(os.path.join(output_directory, f'phase_screen_{slice_number}.csv'), phase_screen)
    np.savetxt(os.path.join(base_directory, f'phase_screen_{slice_number}.csv'), phase_screen, delimiter=',')
    # plt.imshow(phase_screen, cmap='gray')  # 'gray' colormap is typical for single-channel images
    # plt.colorbar()  # Optional, to display a color bar representing the data scale
    # plt.title('Visual Representation of the .npy File')
    # plt.show()
    return f'Phase screen {slice_number} saved'

def generate_and_save_phase_screen2(slice_number):
    simulated_volume, _ = bloodGenerator.simulate_discocytes(
        volume_dimensions, discocyte_radius, num_normal_discocytes2, num_bumpy_discocytes2, num_crescents1, bump_radius, number_of_bumps)

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
    output_directory = os.path.join(base_directory, f'PhaseScreensBump')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    #np.save(os.path.join(output_directory, f'phase_screen_{slice_number}.csv'), phase_screen)
    np.savetxt(os.path.join(base_directory, f'phase_screen_{slice_number}.csv'), phase_screen, delimiter=',')
    # plt.imshow(phase_screen, cmap='gray')  # 'gray' colormap is typical for single-channel images
    # plt.colorbar()  # Optional, to display a color bar representing the data scale
    # plt.title('Visual Representation of the .npy File')
    # plt.show()
    return f'Phase screen {slice_number} saved'


if __name__ == '__main__':
    num_processes = min(10, 10)  # Use up to 25 processes, or the number of CPUs, whichever is smaller
    pool = multiprocessing.Pool(processes=num_processes)

    run_count = 10000  # Set the number of phase screens you want to generate

    # Use tqdm for progress bar
    for _ in tqdm(pool.imap_unordered(generate_and_save_phase_screen1, range(run_count)), total=run_count, desc="Generating phase screens", ascii=False, ncols=75):
        pass

    pool.close()
    pool.join()

    pool = multiprocessing.Pool(processes=num_processes)

    run_count = 10000  # Set the number of phase screens you want to generate

    # Use tqdm for progress bar
    for _ in tqdm(pool.imap_unordered(generate_and_save_phase_screen2, range(run_count)), total=run_count, desc="Generating phase screens", ascii=False, ncols=75):
        pass

    pool.close()
    pool.join()
