import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
import pandas as pd
import random
import itertools

# Import your custom modules
import bloodGenerator
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import computeWave
import propogator

# Global parameters
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
run_count = 300

def process_l(l, base_directory, mask_directory):
    phaseshifttip_init = 0
    phaseshifttilt_init = 0
    s = 700e-6  # Aperture size
    λ = 850e-9  # Wavelength of light
    computeStep = 1  # Computation step distance
    finalDistance = 21  # Total propagation distance
    scatter_event_count = 20

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
            random_index = random.randint(0, 499)  # Use a random integer
            csv_file_path = os.path.join(mask_directory, f'phase_screen_{random_index}.csv')
            if os.path.exists(csv_file_path):
                scattermask = pd.read_csv(csv_file_path, header=None).to_numpy()
                npy_file_path = csv_file_path.replace('.csv', '.npy')
                np.save(npy_file_path, scattermask)
            else:
                npy_file_path = os.path.join(mask_directory, f'phase_screen_{random_index}.npy')
                scattermask = np.load(npy_file_path)

            wave = propogator.propogateScatterMask(oldwave, computeStep * 1e-5, waveRes, s, λ, scattermask)
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep*1e-5, waveRes, s, λ)
            oldwave = wave
            

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_l={l}.npy'), oldwave)

def decompose(inputL_outputL_base_mask):
    inputL, outputL, base_directory, mask_directory = inputL_outputL_base_mask

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

def process_l_worker(args):
    # Unpack arguments
    l, base_directory, mask_directory = args
    # Call the original function with the unpacked arguments
    return process_l(l, base_directory, mask_directory)

def decompose_worker(args):
    # Unpack arguments, including base_directory and mask_directory for decompose function
    inputL, outputL, base_directory, mask_directory = args
    # Call the original decompose function with the unpacked arguments
    return decompose((inputL, outputL, base_directory, mask_directory))




if __name__ == '__main__':
    configurations = [
        {
            'base_directory': r'/Volumes/DeBroglie/Sphere100OutputNew',
            'mask_directory': r'/Volumes/DeBroglie/Sphere100/PhaseScreensSphere',
        },
        {
            'base_directory': r'/Volumes/DeBroglie/Sphere99_9OutputNew',
            'mask_directory': r'/Volumes/DeBroglie/Sphere99_9/PhaseScreensSphere',
        },
        {
            'base_directory': r'/Volumes/DeBroglie/Sphere99OutputNew',
            'mask_directory': r'/Volumes/DeBroglie/Sphere99/PhaseScreensSphere',
        },
        {
            'base_directory': r'/Volumes/DeBroglie/Sphere98OutputNew',
            'mask_directory': r'/Volumes/DeBroglie/Sphere98/PhaseScreensSphere',
        },
        {
            'base_directory': r'/Volumes/DeBroglie/Sphere90OutputNew',
            'mask_directory': r'/Volumes/DeBroglie/Sphere90/PhaseScreensSphere',
        }
    ]

    for config in configurations:
        base_directory = config['base_directory']
        mask_directory = config['mask_directory']

        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        for run_number in range(run_count):
            print(f"Run {run_number + 1}/{run_count} for {base_directory}")

             # Main processing logic for each configuration
            pool = multiprocessing.Pool(processes=num_processes)
            
            # Prepare arguments for process_l_worker
            l_values = [(l, base_directory, mask_directory) for l in range(50)]
            for _ in tqdm(pool.imap_unordered(process_l_worker, l_values), total=len(l_values), desc=f"Processing l values for {base_directory}"):
                pass
            pool.close()
            pool.join()

            # Prepare arguments for decompose_worker
            pool = multiprocessing.Pool(processes=num_processes)
            all_combinations = [(inputL, outputL, base_directory, mask_directory) for inputL, outputL in itertools.product(range(50), repeat=2)]
            results = pool.imap_unordered(decompose_worker, all_combinations)
            
            overlap_values = np.zeros((50, 50))
            for inputL, outputL, calculated_value in tqdm(results, total=len(all_combinations), desc=f"Decomposing for {base_directory}"):
                overlap_values[inputL][outputL] = calculated_value
            
            pool.close()
            pool.join()

            np.save(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.npy'), overlap_values)
            np.savetxt(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.csv'), overlap_values, delimiter=',')

            plt.figure(figsize=(10, 8))
            plt.imshow(overlap_values[::-1], origin='lower', extent=(0, 49, 0, 49))
            plt.colorbar(label='Overlap Value')
            plt.xlabel('Input L')
            plt.ylabel('Output L')
            plt.xticks(np.arange(0, 50, step=1))
            plt.yticks(np.arange(0, 50, step=1))
            plt.title(f'Overlap Values - Run {run_number + 1}')
            plt.savefig(os.path.join(base_directory, f'Overlap_Values_Run_{run_number + 1}.png'), dpi=600)
            plt.close()
