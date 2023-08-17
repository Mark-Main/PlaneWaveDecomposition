import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import randomScatter
import supergaussian
import computeWave 
import propogator
import bloodVolumeCreator

# Parameters
# Laguerre-Gaussian parameters

waveRes = 512
x = np.linspace(-0.05, 0.05, waveRes)
y = np.linspace(-0.05, 0.05, waveRes)
X, Y = np.meshgrid(x, y)
p_init = 2 # Initial radial mode
l_init = 3  # Initial azimuthal mode
w0 = 0.01  # Waist parameter

# Tip Tilt parameters

phaseshifttip_init = 0
phaseshifttilt_init = 0

#------------------------------------------------------------

# Distance parameters for computation 

s = 0.1 # Aperature size
λ = 600e-9 # Wavelength of light

#------------------------------------------------------------

# Defining distance steps with or without scattering 

computeStep = 1 # How far does the wave propogate at each computation step
finalDistance = 100 # How far does the wave propogate in total
displayPoints = [0,20,40,60,80,95] # Points at which the wave will be displayed
# Creating plot arrays 

intensity_data = []
phase_data = []


#------------------------------------------------------------


# Create the initial wavefront 

oldwave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p_init, l_init, w0)
tiptilt = tilt_func.tilttip(waveRes, phaseshifttip_init, phaseshifttilt_init)

oldwave = computeWave.makeWave(oldwave, tiptilt)
oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, waveRes, s, λ))
plt.imshow(np.abs(oldwave) ** 2, cmap='inferno')
plt.imshow(np.angle(oldwave), cmap='inferno')

#------------------------------------------------------------

# Generate Blood volume and slices

grid_size = waveRes
num_spheres =50000
min_radius = 3
max_radius = 3
voxel_resolution = waveRes/100

bloodVol, bloodSlices = bloodVolumeCreator.generate_spheres(grid_size, num_spheres, min_radius, max_radius, voxel_resolution)

#------------------------------------------------------------

# Propogate the wavefront

for i in range(0, finalDistance, computeStep):
    wave = propogator.propogateScatterMask(oldwave, computeStep*1e-6, waveRes, s, λ, bloodSlices[i])
    intensity_data.append(np.abs(wave) ** 2 / np.max(np.abs(wave) ** 2))
    phase_data.append(np.angle(wave))
    oldwave = wave
    print(i)

#------------------------------------------------------------

# Create the figure and axes for the 3D plots

fig = plt.figure(figsize=(20, 25))
ax = fig.add_subplot(121, projection='3d')  # Left subplot
ax2 = fig.add_subplot(122, projection='3d')  # Right subplot


for i in range(len(displayPoints)):
    cmap = plt.cm.inferno
    ax.plot_surface(X, Y, displayPoints[i] * np.ones_like(X), facecolors=cmap(intensity_data[i]),
                    rstride=1, cstride=1, shade=False)

    ax2.plot_surface(X, Y, displayPoints[i] * np.ones_like(X), facecolors=cmap(phase_data[i]),
                    rstride=1, cstride=1, shade=False)


ax.set_zlim(0, max(displayPoints))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Distance')
ax.set_title('2D Intensity Plots in 3D')

ax2.set_zlim(0, max(displayPoints))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Distance')
ax2.set_title('2D Phase Plots in 3D')

# Set the rotation angles
ax.view_init(elev=-50, azim=0, roll=270)
ax2.view_init(elev=-50, azim=0, roll=270)

# Set the aspect ratio to make it a cuboid
ax.set_box_aspect([15, 15, 50])
ax2.set_box_aspect([15, 15, 50])

# Show the plots
fig.canvas.draw_idle()

plt.show()



