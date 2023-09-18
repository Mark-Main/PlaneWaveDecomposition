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


# Parameters
# Laguerre-Gaussian parameters

waveRes = 500
x = np.linspace(-100e-6, 100e-6, waveRes)
y = np.linspace(-100e-6, 100e-6, waveRes)
X, Y = np.meshgrid(x, y)


grid_size = waveRes
num_spheres =25500
min_radius = 3
max_radius = 3
voxel_resolution = waveRes/500
bloodVol, bloodSlices = bloodVolumeCreator.generate_spheres(grid_size, num_spheres, min_radius, max_radius, voxel_resolution)

for p in range(10):
    for l in range(10):
        w0 = 20e-6  # Waist parameter

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
        finalDistance = 800 # How far does the wave propogate in total
        displayPoints = [10,20,30,40,50,60,70,80,90,100] # Points at which the wave will be displayed
        # Creating plot arrays 

        intensity_data = []
        phase_data = []


        #------------------------------------------------------------


        # Create the initial wavefront 

        oldwave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p, l, w0)
        tiptilt = tilt_func.tilttip(waveRes, phaseshifttip_init, phaseshifttilt_init)

        oldwave = computeWave.makeWave(oldwave, tiptilt)
        oldwave = np.fft.ifft2(oldwave * distanceTerm.disStep(0, waveRes, s, λ))
        plt.imshow(np.abs(oldwave) ** 2, cmap='inferno')
        plt.imshow(np.angle(oldwave), cmap='inferno')

        #------------------------------------------------------------

        base_directory = r'D:\MarkSimulations\SphericalTrialLP6_2'


        output_directory = os.path.join(base_directory, f'P={p}, L={l}')

        #------------------------------------------------------------

        # Propogate the wavefront

        for i in range(0, finalDistance, computeStep):
            if i < grid_size:
                wave = propogator.propogateScatterMask(oldwave, computeStep*1e-6, waveRes, s, λ, bloodSlices[i])
               # intensity_data.append(np.abs(wave) ** 2 / np.max(np.abs(wave) ** 2))
                #phase_data.append(np.angle(wave))
                oldwave = wave
            else:
                wave = propogator.propogate(oldwave, computeStep*1e-1, waveRes, s, λ)
                #intensity_data.append(np.abs(wave) ** 2 / np.max(np.abs(wave) ** 2))
                #phase_data.append(np.angle(wave))
                oldwave = wave  
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            np.save(os.path.join(output_directory, f'BinaryDataSlice{i}_p={p}_l={l}.npy'), wave)      
            print(i)

        #------------------------------------------------------------

        # Show the plots
        '''
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots

        slice_start = 0
        slice_end = finalDistance - 1
        slice_index = slice_start

        def update(frame):
            global slice_index
            axs[0].clear()
            axs[1].clear()

            axs[0].imshow(intensity_data[slice_index], cmap='inferno')
            axs[0].set_title(f'Intensity Slice at X = {slice_index}')

            axs[1].imshow(phase_data[slice_index], cmap='inferno')  # Assuming 'viridis' colormap for phase
            axs[1].set_title(f'Phase Slice at X = {slice_index}')

            slice_index = (slice_index + 1) % (slice_end + 1)

        ani = FuncAnimation(fig, update, interval=300)  # Interval in milliseconds
        plt.show()'''



        def save_frames(output_directory):
            os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

            for i, intensity_frame in enumerate(intensity_data):
                plt.imshow(intensity_frame, cmap='inferno')
                plt.title(f'Intensity Slice at Z = {i}')
                plt.savefig(os.path.join(output_directory, f'intensity_frame_{i}.png'))
                plt.close()

            for i, phase_frame in enumerate(phase_data):
                plt.imshow(phase_frame, cmap='inferno')
                plt.title(f'Phase Slice at Z = {i}')
                plt.savefig(os.path.join(output_directory, f'phase_frame_{i}.png'))
                plt.close()

        

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Uncomment the line below to save frames
        save_frames(output_directory)

        plt.show()








# Create the figure and axes for the 3D plots

'''fig = plt.figure(figsize=(20, 25))
ax = fig.add_subplot(131, projection='3d')  # Left subplot
ax2 = fig.add_subplot(132, projection='3d')  # Right subplot
ax3 = fig.add_subplot(133, projection='3d')  # Right subplot



for display_point in displayPoints:
    print(display_point)
    cmap = plt.cm.inferno
    cmap2 = plt.cm.binary
    index = displayPoints.index(display_point)
    ax.plot_surface(X, Y, display_point * np.ones_like(X), facecolors=cmap(intensity_data[index]),
                    rstride=1, cstride=1, shade=False)

    ax2.plot_surface(X, Y, display_point * np.ones_like(X), facecolors=cmap(phase_data[index]),
                    rstride=1, cstride=1, shade=False)
    ax3.plot_surface(X, Y, display_point * np.ones_like(X), facecolors=cmap2(bloodSlices[index]),
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
ax3.view_init(elev=-50, azim=0, roll=270)


# Set the aspect ratio to make it a cuboid
ax.set_box_aspect([15, 15, 50])
ax2.set_box_aspect([15, 15, 50])
ax3.set_box_aspect([15, 15, 50])


# Show the plots
fig.canvas.draw_idle()

plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 5))

slice_start = 0
slice_end = grid_size - 1
slice_index = slice_start

def update(frame):
    global slice_index
    axs.clear()
    axs.imshow(bloodSlices[slice_index], cmap='gray')
    axs.set_title(f'Original Slice at X = {slice_index}')
    slice_index = (slice_index + 1) % (slice_end + 1)

ani = FuncAnimation(fig, update, interval=100)  # Interval in milliseconds
plt.show()'''



