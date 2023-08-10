import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
import generateWave
import distanceTerm
import tilt_func
import randomScatter
import supergaussian
import computeWave 
import propogator
import computeFSO
import coordinate_convert as con

# Parameters
# Laguerre-Gaussian parameters

waveRes = 16
x = np.linspace(-0.0125, 0.0125, waveRes)
y = np.linspace(-0.0125, 0.0125, waveRes)
X, Y = np.meshgrid(x, y)
p_init = 0 # Initial radial mode
l_init = 0  # Initial azimuthal mode
w0 = 0.1  # Waist parameter

# Tip Tilt parameters

phaseshifttip_init = 0
phaseshifttilt_init = 0

# Super Gaussian parameters (probably don't need to change these)

gaussRes = 1024
x2 = np.linspace(-0.4, 0.4, gaussRes)
y2 = np.linspace(-0.4, 0.4, gaussRes)
X2, Y2 = np.meshgrid(x2, y2)
sigma = 1.0  # Standard deviation of the super Gaussian
exponent = 4  # Exponent controlling the shape of the super Gaussian

# Tip Tilt parameters

phaseshifttip_init = 0
phaseshifttilt_init = 0

#------------------------------------------------------------

# Distance parameters for computation 

fullRes = 1024
s = 0.2 # Aperature size
λ = 1550e-9 # Wavelength of light

#------------------------------------------------------------

# Defining distance steps with or without scattering 

computeStep = 50 # How far does the wave propogate at each computation step
finalDistance = 1000 # How far does the wave propogate in total
scatterPoints = []# Points at which scattering will be carried out
displayPoints = [0,200,400,600,800,1000] # Points at which the wave will be displayed

center_coordinates = [(0,0), (0,0.05), (-0.043, 0.025), (0.043, 0.025), (-0.043, -0.025), (0.043, -0.025), (0, -0.05)]
# Function to map coordinates to indices
new_coordinates = con.coordinate_shift(center_coordinates, gaussRes, x2, y2)
# Creating plot arrays  

intensity_data = []
phase_data = []


#------------------------------------------------------------


# Create the initial wavefront 

oldwave = generateWave.laguerre_gaussian(X, Y, p_init, l_init, w0)
#oldwave = generateWave.gaussian(X, Y, 0, 0, 0.5, 0.5, 0.6)
tiptilt = tilt_func.tilttip(gaussRes, phaseshifttip_init, phaseshifttilt_init)

oldwave = computeFSO.makeWave(oldwave, tiptilt, X2, Y2, sigma, exponent, new_coordinates)
oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, gaussRes, s, λ))
plt.imshow(np.abs(oldwave) ** 2, cmap='inferno')
#plt.imshow(np.angle(oldwave), cmap='inferno')
intensity_data.append(np.abs(oldwave) ** 2 / np.max(np.abs(oldwave) ** 2))
phase_data.append(np.angle(oldwave))

#------------------------------------------------------------

# Propogate the wavefront

for i in range(0, finalDistance, computeStep):
    if i in scatterPoints:
        wave = propogator.propogateScatter(oldwave, computeStep, fullRes, s, λ, 0.3)
    else:
        wave = propogator.propogate(oldwave, computeStep, fullRes, s, λ)
    if i in displayPoints:
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
    cmap = plt.cm.viridis
    ax.plot_surface(X2, Y2, displayPoints[i] * np.ones_like(X2), facecolors=cmap(intensity_data[i]),
                    rstride=1, cstride=1, shade=False)

    ax2.plot_surface(X2, Y2, displayPoints[i] * np.ones_like(X2), facecolors=cmap(phase_data[i]),
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



