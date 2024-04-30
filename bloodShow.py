import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import bloodGenerator
import os
from matplotlib.animation import FuncAnimation

waveRes = 500
# Set simulation parameters
sliceThickness = 40
volume_dimensions = (waveRes, waveRes,sliceThickness)
discocyte_radius = 16  # Radius for discocytes
num_normal_discocytes = 120 # Number of normal discocytes
num_bumpy_discocytes =0  # Number of bumpy discocytes
num_crescents =000
bump_radius = 10 # Radius of the bumps
number_of_bumps = 15  # Number of bumps

# Run the simulation
simulated_volume, bloodSlices = bloodGenerator.simulate_discocytes(
    volume_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes, num_crescents, bump_radius, number_of_bumps)
# Assuming waveRes is defined somewhere in your code
 # Example value, replace with the actual value used in your simulation

# Function to plot 3D volume
def plot_3d_volume(volume):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Preparing data for 3D plot
    x, y, z = np.indices(volume.shape)
    voxels = volume > 0  # Assuming the volume contains binary data

    # Plotting the 3D volume
    ax.voxels(x, y, z, voxels, edgecolor='k')
    plt.show()

def update(frame):
    plt.clf()  # Clear the figure
    plt.imshow(bloodSlices[frame], cmap='gray')
    plt.axis('off')
    plt.title(f'Slice {frame}')


ani = FuncAnimation(plt.gcf(), update, frames=range(sliceThickness), repeat=True, interval=200)
base_directory = r'/Users/Mark/Documents/PlaneWaveDecomposition'
# Save the animation as a GIF with improved quality
output_file = os.path.join(base_directory, 'animation7.gif')
ani.save(output_file, writer='pillow', fps=20, dpi=300)