import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
import bloodGenerator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import os
from tqdm import tqdm
import multiprocessing

base_directory = r'/Users/Mark/Desktop/NewSimData/Disc_bigfile_1024'
# Parameters
# Laguerre-Gaussian parameters

waveRes = 512
x = np.linspace(-100e-6, 100e-6, waveRes)
y = np.linspace(-100e-6, 100e-6, waveRes)
X, Y = np.meshgrid(x, y)
grid_size = waveRes
# Set simulation parameters
volume_dimensions = (waveRes, waveRes,25)
discocyte_radius = 20  # Radius for discocytes
num_normal_discocytes = 120 # Number of normal discocytes
num_bumpy_discocytes =0  # Number of bumpy discocytes
num_crescents =000
bump_radius = 10  # Radius of the bumps
number_of_bumps = 25  # Number of bumps

# Run the simulation
simulated_volume, bloodSlices = bloodGenerator.simulate_discocytes(
    volume_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes,num_crescents, bump_radius, number_of_bumps)

'''
ani = FuncAnimation(plt.gcf(), update, frames=range(25), repeat=True, interval=200)
base_directory = r'/Users/Mark/Documents/PlaneWaveDecomposition'
# Save the animation as a GIF with improved quality
output_file = os.path.join(base_directory, 'animation11.gif')
ani.save(output_file, writer='pillow', fps=20, dpi=300)'''

# Constants
n_white = 1.5  # Refractive index for parts represented by 1 in simulated_volume
n_non_white = 1.3  # Refractive index for parts represented by 0 in simulated_volume
wavelength = 850e-9  # Wavelength in meters

# Assuming each dimension of the volume is given in microns, convert to meters
# Assuming cubic voxels, we use one dimension to calculate the voxel size
voxel_size = 25e-6 / waveRes  # Size of one voxel in meters (along one dimension)

# Assign refractive indices based on the values in simulated_volume
refractive_indices = np.where(simulated_volume == 1, n_white, n_non_white)

# Calculate Optical Path Length (OPL) for each voxel
# Multiply the refractive index by the voxel size for each point in the volume
OPL = refractive_indices * voxel_size

# Calculate the phase shift
phase_shift = (2 * np.pi / wavelength) * OPL

# Sum along one axis to project 3D phase data onto a 2D plane
phase_screen = np.sum(phase_shift, axis=2)

# Plot the phase screen
phase_image = plt.imshow(phase_screen, cmap='gray')

# Determine the maximum phase shift in multiples of pi
max_phase = np.max(phase_screen)
num_pi_intervals = np.ceil(max_phase / np.pi)

# Create ticks for the color bar at multiples of pi within the range of phase shifts
pi_ticks = np.arange(0, num_pi_intervals + 1) * np.pi
pi_labels = [f'{n:.0f}π' for n in np.arange(0, num_pi_intervals + 1)]

# Create a color bar with labels in multiples of pi
colorbar = plt.colorbar(phase_image, label='Phase Shift (in multiples of π)')
colorbar.set_ticks(pi_ticks)
colorbar.set_ticklabels(pi_labels)

plt.title('2D Phase Screen')
plt.show()