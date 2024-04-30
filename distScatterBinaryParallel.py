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
import bloodGenerator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import os
from tqdm import tqdm
import multiprocessing

base_directory = r'/Users/Mark/Desktop/NewSimData/PhaseMaskTest2'
mask_directory = r'/Users/Mark/Desktop/NewSimData/PhaseScreenTrialMix/PhaseScreens'
# Parameters
# Laguerre-Gaussian parameters

waveRes = 500
x = np.linspace(-200e-6, 200e-6, waveRes)
y = np.linspace(-200e-6, 200e-6, waveRes)
X, Y = np.meshgrid(x, y)

grid_size = waveRes

'''num_spheres = 25500
min_radius = 3
max_radius = 3
voxel_resolution = waveRes/512
bloodVol, bloodSlices = bloodVolumeCreator.generate_spheres(grid_size, num_spheres, min_radius, max_radius, voxel_resolution)

# Set simulation parameters
volume_dimensions = (waveRes, waveRes, waveRes)
discocyte_radius = 20  # Radius for discocytes
num_normal_discocytes = 0 # Number of normal discocytes
num_bumpy_discocytes =0  # Number of bumpy discocytes
num_crescents =100
bump_radius = 2  # Radius of the bumps
number_of_bumps = 50  # Number of bumps

# Run the simulation
simulated_volume, bloodSlices = bloodGenerator.simulate_discocytes(
    volume_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes,num_crescents, bump_radius, number_of_bumps)



if not os.path.exists(base_directory):
    os.makedirs(base_directory)
np.save(os.path.join(base_directory, f'bloodslices.npy'), bloodSlices)'''

def process_l(l):
    w0 = 20e-6  # Waist parameter
    phaseshifttip_init = 0
    phaseshifttilt_init = 0
    s = 0.2 # Aperature size
    λ = 850e-9 # Wavelength of light
    computeStep = 1 # How far does the wave propagate at each computation step
    finalDistance = 11 # How far does the wave propagate in total
    scatter_event_count = 10
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
        if i <= scatter_event_count:
            scattermask = np.load(os.path.join(mask_directory, f'phase_screen_{i}.npy'))
            wave = propogator.propogateScatterMask(oldwave, computeStep*1e-5, waveRes, s, λ,  scattermask)
            oldwave = wave
        else:
            wave = propogator.propogate(oldwave, computeStep, waveRes, s, λ)
            oldwave = wave  

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, f'BinaryDataSlice_l={l}.npy'), wave)

if __name__ == '__main__':
    num_processes = 8  # Adjust this based on your system's capabilities
    pool = multiprocessing.Pool(processes=num_processes)

    l_values = list(range(30))  # List of l values

    results = list(tqdm(pool.imap(process_l, l_values), total=len(l_values), desc="Processing l values", ascii=False, ncols=75))

    pool.close()
    pool.join()

