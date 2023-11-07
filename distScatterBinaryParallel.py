import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import computeWave 
import propogator
import bloodVolumeCreator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import os
from tqdm import tqdm
import multiprocessing

base_directory = r'/Users/Mark/Documents/DataForSVD14'
# Parameters
# Laguerre-Gaussian parameters

waveRes = 512
x = np.linspace(-200e-6, 200e-6, waveRes)
y = np.linspace(-200e-6, 200e-6, waveRes)
X, Y = np.meshgrid(x, y)

grid_size = waveRes
num_spheres = 25500
min_radius = 3
max_radius = 3
voxel_resolution = waveRes/512
bloodVol, bloodSlices = bloodVolumeCreator.generate_spheres(grid_size, num_spheres, min_radius, max_radius, voxel_resolution)
if not os.path.exists(base_directory):
    os.makedirs(base_directory)
np.save(os.path.join(base_directory, f'bloodslices.npy'), bloodSlices)

def process_l(l):
    w0 = 20e-6  # Waist parameter
    phaseshifttip_init = 0
    phaseshifttilt_init = 0
    s = 0.2 # Aperature size
    λ = 600e-9 # Wavelength of light
    computeStep = 1 # How far does the wave propagate at each computation step
    finalDistance = 800 # How far does the wave propagate in total
    intensity_data = []
    phase_data = []

    oldwave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, l, w0)
    tiptilt = tilt_func.tilttip(waveRes, phaseshifttip_init, phaseshifttilt_init)
    oldwave = computeWave.makeWave(oldwave, tiptilt)
    oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, waveRes, s, λ))

    overlap_importWave = np.trapz(np.conj(oldwave)*oldwave, dx=x[1] - x[0], axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)
    oldwave = oldwave / np.sqrt(overlapy_importWave)

    output_directory = os.path.join(base_directory, f'L={l}')

    for i in tqdm(range(0, finalDistance, computeStep), desc=f"L={l}", ascii=False, ncols=75):
        if i < grid_size:
            wave = propogator.propogateScatterMask(oldwave, computeStep*1e-6, waveRes, s, λ, bloodSlices[i])
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep*1e-1, waveRes, s, λ)
            oldwave = wave  

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_l={l}.npy'), wave)

if __name__ == '__main__':
    num_processes = 8  # Adjust this based on your system's capabilities
    pool = multiprocessing.Pool(processes=num_processes)

    l_values = list(range(50))  # List of l values

    results = list(tqdm(pool.imap(process_l, l_values), total=len(l_values), desc="Processing l values", ascii=False, ncols=75))

    pool.close()
    pool.join()
