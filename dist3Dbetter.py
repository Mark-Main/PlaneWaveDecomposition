import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import genlaguerre
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func
import csv

# Parameters
res = 8
res2 = 64
x = np.linspace(-0.25, 0.25, res)
y = np.linspace(-0.25, 0.25, res)
X, Y = np.meshgrid(x, y)

x2 = np.linspace(-4, 4, res2)
y2 = np.linspace(-4, 4, res2)
X2, Y2 = np.meshgrid(x2, y2)

p_init = 1  # Initial radial mode
l_init = 2  # Initial azimuthal mode
w0 = 0.1  # Waist parameter
z_init = [0, 3, 10, 20]  # Propagation distances for slices

s = 0.025
λ = 600 / 1000000000

sigma = 1.0  # Standard deviation of the super Gaussian
exponent = 4  # Exponent controlling the shape of the super Gaussian

phaseshifttip_init = 0
phaseshifttilt_init = 0

# Create the figure and axes for the 3D plots
fig = plt.figure(figsize=(20, 25))
ax = fig.add_subplot(121, projection='3d')  # Left subplot
ax2 = fig.add_subplot(122, projection='3d')  # Right subplot

# Create text boxes for phase shift
tip_text_box = TextBox(plt.axes([0.85, 0.9, 0.1, 0.05]), 'Phase Shift Tip', initial=str(phaseshifttip_init))
tilt_text_box = TextBox(plt.axes([0.85, 0.85, 0.1, 0.05]), 'Phase Shift Tilt', initial=str(phaseshifttilt_init))


# Update function for text boxes
def update(val):
    global phaseshifttip_init, phaseshifttilt_init

    try:
        phaseshifttip = float(tip_text_box.text)
        phaseshifttilt = float(tilt_text_box.text)
    except ValueError:
        tip_text_box.set_val(str(phaseshifttip_init))
        tilt_text_box.set_val(str(phaseshifttilt_init))
        return

    phaseshifttip_init = phaseshifttip
    phaseshifttilt_init = phaseshifttilt

    # Clear the current plot
    ax.cla()
    ax2.cla()

    # Iterate over the slices
    for i in range(len(z_init)):
        # Generate the initial Laguerre-Gaussian beam
        wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p_init, l_init, w0)
        L, M, N = tilt_func.tilttip(res2, phaseshifttip, phaseshifttilt)  # Modify size of N

        # Generate the super Gaussian function
        R = np.sqrt((X2) ** 2 + (Y2) ** 2)
        super_gaussian = np.exp(-(R / sigma) ** exponent)

        # Normalize the super Gaussian
        super_gaussian /= np.max(super_gaussian)

        # Calculate the dimensions of the wave array
        wave_size = wave.shape[0]
        super_gaussian_size = super_gaussian.shape[0]

        # Calculate the indices for adding the wave to the center of the super Gaussian
        x_start = (super_gaussian_size - wave_size) // 2
        x_end = x_start + wave_size
        y_start = (super_gaussian_size - wave_size) // 2
        y_end = y_start + wave_size

        # Convert the super Gaussian array to complex
        super_gaussian = super_gaussian.astype(complex)

        # Create an array of the same size as the super Gaussian to add the wave
        wave_padded = np.ones_like(super_gaussian)
        wave_padded[x_start:x_end, y_start:y_end] = wave

        # Add the wave to the super Gaussian
        super_gaussian *= wave_padded

        # Create distance array matching the size of the super Gaussian
        distance = distanceTerm.disStep(z_init[i], res2, s, λ)  # Modify size of distance

        # Multiply wave and distance arrays
        product = np.multiply(super_gaussian, np.exp(1j * N))

        # Take Fourier transform
        fft_result = np.fft.fft2(product)

        # Perform inverse Fourier transform
        ifft_result = np.fft.ifft2(fft_result * distance)

        # Normalize the intensity
        intensity = np.abs(ifft_result) ** 2
        intensity /= np.max(intensity)

        filename = 'array_data.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(intensity)

        print(f"Array data saved as {filename}")

        phase = np.angle(ifft_result)

        # Create custom colormap with alpha channel for transparency
        cmap = plt.cm.inferno

        # Plot the 2D slice in the 3D plot with transparent areas
        ax.plot_surface(X2, Y2, z_init[i] * np.ones_like(X2), facecolors=cmap(intensity),
                        rstride=1, cstride=1, shade=False)

        ax2.plot_surface(X2, Y2, z_init[i] * np.ones_like(X2), facecolors=cmap(phase),
                         rstride=1, cstride=1, shade=False)

    # Set limits and labels for the 3D plots

    ax.set_zlim(0, max(z_init))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance')
    ax.set_title('2D Intensity Plots in 3D')

    ax2.set_zlim(0, max(z_init))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Distance')
    ax2.set_title('2D Phase Plots in 3D')

    # Set the rotation angles
    ax.view_init(elev=-20, azim=0, roll=270)
    ax2.view_init(elev=-20, azim=0, roll=270)

    # Set the aspect ratio to make it a cuboid
    ax.set_box_aspect([15, 15, 50])
    ax2.set_box_aspect([15, 15, 50])

    # Show the plots
    fig.canvas.draw_idle()


# Connect the update function to the text box events
tip_text_box.on_submit(update)
tilt_text_box.on_submit(update)

# Call the update function to plot with initial tip and tilt values
update(None)

# Show the plots
plt.show()
