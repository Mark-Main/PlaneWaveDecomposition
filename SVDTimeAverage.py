import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
from scipy import stats
import random
import pandas as pd
from scipy import interpolate, signal
from scipy.interpolate import interp2d
from matplotlib.animation import FuncAnimation

# Import your custom modules
import bloodGenerator
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import computeWave
import propogator
import itertools



# Global parameters
base_directory = r'/Volumes/DeBroglie/TimeAverage99'
mask_directory = r'/Volumes/DeBroglie/Sphere99/PhaseScreensSphere'

if not os.path.exists(base_directory):
    os.makedirs(base_directory)


sliceThickness = 40
discocyte_radius = 16
num_normal_discocytes = 0
num_bumpy_discocytes = 120
num_crescents = 0
bump_radius = 10
number_of_bumps = 15
cellSize = 8e-6
waveRes = 500
w0 = 20e-6
x = np.linspace(-250e-6, 250e-6, waveRes)
y = np.linspace(-250e-6, 250e-6, waveRes)
X, Y = np.meshgrid(x, y)
grid_size = waveRes
screen_dimensions = (waveRes, waveRes,sliceThickness)
num_processes = min(9, 9)

scatter_event_count = 20
run_count = 1000




def process_l(args):
    l, scatter_masks_index = args
    phaseshifttip_init = 0
    phaseshifttilt_init = 0
    s = 700e-6  # Aperture size
    λ = 850e-9  # Wavelength of light
    computeStep = 1  # Computation step distance
    finalDistance = 21  # Total propagation distance


    oldwave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, l, w0)
    tiptilt = tilt_func.tilttip(waveRes, phaseshifttip_init, phaseshifttilt_init)
    oldwave = computeWave.makeWave(oldwave, tiptilt)
    oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, waveRes, s, λ))

    overlap_importWave = np.trapz(np.conj(oldwave) * oldwave, dx=x[1] - x[0], axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)
    oldwave = oldwave / np.sqrt(overlapy_importWave)

    output_directory = os.path.join(base_directory, f'L={l}')

    for i in tqdm(range(finalDistance // computeStep), desc=f"L={l}", ascii=False, ncols=75):
        if i < 20:
            
            mask_number = scatter_masks_index[i]
            
            csv_file_path = os.path.join(mask_directory, f'phase_screen_{mask_number}.csv')
            
            if os.path.exists(csv_file_path):
                # Load the CSV file, convert it to a NumPy array, and then save it as .npy for future use
                scattermask = pd.read_csv(csv_file_path, header=None).to_numpy()
                
                npy_file_path = csv_file_path.replace('.csv', '.npy')
                np.save(npy_file_path, scattermask)
            else:
                # Load the NPY file if the CSV conversion has already been done
                npy_file_path = os.path.join(mask_directory, f'phase_screen_{mask_number}.npy')
                scattermask = np.load(npy_file_path)

            wave = propogator.propogateScatterMask(oldwave, computeStep * 1e-5, waveRes, s, λ, scattermask)
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep*1e-4, waveRes, s, λ)
            oldwave = wave

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_lg={l}.npy'), oldwave)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 8))  # Adjusted to create a 2x2 grid

    #    # Intensity of the reference wavefront
    # axes[0].imshow(np.abs(oldwave)**2, cmap='viridis')
    # axes[0].set_title('Reference Wavefront Intensity')
    # axes[1].imshow(np.angle(oldwave), cmap='viridis')
    # axes[0].set_title('Reference Wavefront phase')
    # plt.tight_layout()
    # plt.show()

def process_svd(args):
    mode, scatter_masks_index, svd_mode_wave = args
    s = 700e-6  # Aperture size
    λ = 850e-9  # Wavelength of light
    computeStep = 1  # Computation step distance
    finalDistance = 21

    oldwave = svd_mode_wave


    output_directory = os.path.join(base_directory, f'SVD={mode}')

    for i in tqdm(range(finalDistance // computeStep), desc=f"SVD={mode}", ascii=False, ncols=75):
        if i < 20:
            mask_number = scatter_masks_index[i]
            if mode == 0:
                print(mask_number)
            csv_file_path = os.path.join(mask_directory, f'phase_screen_{mask_number}.csv')
            if os.path.exists(csv_file_path):
                # Load the CSV file, convert it to a NumPy array, and then save it as .npy for future use
                scattermask = pd.read_csv(csv_file_path, header=None).to_numpy()
                
                npy_file_path = csv_file_path.replace('.csv', '.npy')
                np.save(npy_file_path, scattermask)
            else:
                # Load the NPY file if the CSV conversion has already been done
                npy_file_path = os.path.join(mask_directory, f'phase_screen_{mask_number}.npy')
                scattermask = np.load(npy_file_path)

            wave = propogator.propogateScatterMask(oldwave, computeStep * 1e-5, waveRes, s, λ, scattermask)
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep*1e-5, waveRes, s, λ) 
            oldwave = wave
            
        
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_svd={mode}.npy'), oldwave)

# Function to perform decomposition
def decompose(inputL_outputL_modeType):
    inputL, outputL, modeType = inputL_outputL_modeType
    if modeType == 'LG':
        output_directory = os.path.join(base_directory, f'L={outputL}')
    elif modeType == 'SVD':
        output_directory = os.path.join(base_directory, f'SVD={outputL}')
    elif modeType == 'SVDstore':
        output_directory = os.path.join(base_directory, f'SVDstore={outputL}')    

    importWave_path = os.path.join(output_directory, f'BinaryDataSlice_{modeType.lower()}={outputL}.npy')
    importWave = np.load(importWave_path)

    refWave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, inputL, w0)

    overlap_importWave = np.trapz(np.conj(importWave)*importWave, dx=x[1] - x[0], axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)

    overlap_refWave = np.trapz(np.conj(refWave)*refWave, dx=x[1] - x[0])
    overlapy_refWave = np.trapz(overlap_refWave)

    normalized_importWave = importWave / np.sqrt(overlapy_importWave)
    normalized_refWave = refWave / np.sqrt(overlapy_refWave)

    overlap = np.trapz(np.conj(normalized_refWave)*normalized_importWave, dx=x[1] - x[0], axis=0)
    overlap = np.trapz(overlap)


    return inputL, outputL, overlap


def apply_svd_mode_to_wavefront(VT_element, reference_wavefronts):
    # Initialize a new wavefront as zeros with the same shape as one of the reference wavefronts
    new_wavefront = np.zeros_like(reference_wavefronts[0])
    # Apply each element of the VT_element to its corresponding reference wavefront
    for i in range(len(VT_element)):
        #normalise reference waveform
        
        overlapx_reference_wavefronts = np.trapz(np.conj(reference_wavefronts[i])*reference_wavefronts[i], dx=x[1] - x[0], axis=0)
        overlapy_reference_wavefronts = np.trapz(overlapx_reference_wavefronts)
        normalised_reference_wavefronts = reference_wavefronts[i] / np.sqrt(overlapy_reference_wavefronts)


        new_wavefront += np.conj(VT_element[i]) * normalised_reference_wavefronts

        #normalise reference waveform
        overlapx_new_wavefront = np.trapz(np.conj(new_wavefront)*new_wavefront, dx=x[1] - x[0], axis=0)
        overlapy_new_wavefront = np.trapz(overlapx_new_wavefront)
        normalised_new_wavefront = new_wavefront / np.sqrt(overlapy_new_wavefront)


    return normalised_new_wavefront

def process_mode(mode, num_processes=4):
    pool = multiprocessing.Pool(processes=num_processes)
    all_inputLs = [(inputL, mode) for inputL in range(50)]  # Create all (inputL, mode) tuples for the current mode

    # Initialize results array for the current mode
    results_vector = np.zeros(50)

    # Start multiprocessing pool for decomposition
    results = pool.imap_unordered(decompose, all_inputLs)

    for inputL, calculated_value in tqdm(results, total=len(all_inputLs), desc=f"Decomposing mode {mode}", ascii=False, ncols=75):
        results_vector[inputL] = calculated_value

    # Save the results_vector for the current mode
    np.save(os.path.join(base_directory, f'results_vector_mode_{mode}.npy'), results_vector)

    pool.close()
    pool.join()


if __name__ == '__main__':

    for run_number in range(run_count):

        # Generate a list of unique random values within the range 0 to 4999
        scatter_masks_index = random.sample(range(0, 499), scatter_event_count)
        

        print(f"Run {run_number + 1}/{run_count}")

        #Task 1: Perform Propagation on LG basis set
        pool = multiprocessing.Pool(processes=num_processes)
        l_values = list(range(50))  # Adjust as needed
        # Adjusted call to include X, Y, and waveRes in args_for_pool
        args_for_pool = [(l, scatter_masks_index) for l in l_values]
        for _ in tqdm(pool.imap_unordered(process_l, args_for_pool), total=len(l_values), desc="Processing l values"):
            pass

        pool.close()
        pool.join()


        # Task 2: Decompose LG with multiprocessing
        pool = multiprocessing.Pool(processes=num_processes)
        all_combinations = [(inputL, outputL, 'LG') for inputL in range(50) for outputL in range(50)]  # Specify 'LG' for LG mode decomposition

        # Initialize overlap_values array
        overlap_values = np.zeros((50, 50),dtype=complex)

        # Start multiprocessing pool for decomposition
        results = pool.imap_unordered(decompose, all_combinations)

        for inputL, outputL, calculated_value in tqdm(results, total=len(all_combinations), desc="Decomposing LG", ascii=False, ncols=75):
            overlap_values[inputL][outputL] = calculated_value

        pool.close()
        pool.join()

        # Task 4: Process first 15 SVD modes
        reference_wavefronts = [generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, l, w0) for l in range(50)]

        # New list to hold the conjugated wavefronts
        conjugated_wavefronts = [np.conj(wavefront) for wavefront in reference_wavefronts]

        # If you need to overwrite the original list name with the new list
        reference_wavefronts = conjugated_wavefronts
        
        U, S, VT = np.linalg.svd(overlap_values)
     
        # VT = np.transpose(VT)

        for i in range(15):  # For each of the 15 modes
            mode = i
            svd_mode = VT[i]  # Select the i-th row of VT for the i-th mode
            svd_mode_wave = np.conj(apply_svd_mode_to_wavefront(svd_mode, reference_wavefronts))# Generate the new wavefront
            svd_store_directory = os.path.join(base_directory, f'SVDstore={mode}')
            if not os.path.exists(svd_store_directory):
                os.makedirs(svd_store_directory)
            np.save(os.path.join(svd_store_directory, f'BinaryDataSlice_runnumber={run_number}.npy'), svd_mode_wave)
        
        fig, axs = plt.subplots(5, 3, figsize=(15, 10))

        for i in range(15):  # For each of the 15 modes
            mode = i
            svd_mode = VT[i]  # Select the i-th row of VT for the i-th mode
            # Generate the new wavefront
            svd_mode_wave = np.conj(apply_svd_mode_to_wavefront(svd_mode, reference_wavefronts))
            # Take the square of the absolute values
            squared_absolute_values = np.abs(svd_mode_wave) ** 2
            
            # Plotting
            ax = axs[mode // 3, mode % 3]  # Determine subplot position
            ax.plot(squared_absolute_values)
            ax.set_title(f'Mode {mode}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Squared Magnitude')
            
            # Directory for storing the results
            svd_store_directory = os.path.join(base_directory, f'SVDstore={mode}')
            if not os.path.exists(svd_store_directory):
                os.makedirs(svd_store_directory)
            # Save the squared absolute values as a binary file
            np.save(os.path.join(svd_store_directory, f'BinaryDataSlice_runnumber={run_number}.npy'), squared_absolute_values)

        plt.tight_layout()
        plt.show()

        
  
        




