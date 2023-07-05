import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import genlaguerre
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func

# Parameters
res = 64
x = np.linspace(-0.25, 0.25, res)
y = np.linspace(-0.25, 0.25, res)
X, Y = np.meshgrid(x, y)
p_init = 2  # Initial radial mode
l_init = 3  # Initial azimuthal mode
w0 = 0.002  # Waist parameter
z_init = [ 20]  # Propagation distances for slices


s = 0.25
λ = 600 / 1000000000

phaseshifttip_init = 0
phaseshifttilt_init = 0

# Create the figure and axis for the 3D plot
fig = plt.figure(figsize=(20, 25))
ax = fig.add_subplot(111, projection='3d')

# Create text boxes for phase shift values
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

    # Iterate over the slices
    for i in range(len(z_init)):
        # Generate the initial Laguerre-Gaussian beam
        wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p_init, l_init, w0)
        distance = distanceTerm.disStep(z_init[i], res, s, λ)
        L, M, N = tilt_func.tilttip(res, phaseshifttip, phaseshifttilt)

        # Multiply wave and distance arrays
        product = np.multiply(wave, np.exp(1j * N))

        # Take Fourier transform
        fft_result = np.fft.fft2(product)

        # Perform inverse Fourier transform
        ifft_result = np.fft.ifft2(fft_result * distance)

        # Normalize the intensity
        intensity = np.abs(ifft_result) ** 2
        intensity /= np.max(intensity)

        # Plot the 2D slice in the 3D plot
        ax.plot_surface(X, Y, z_init[i] * np.ones_like(X), facecolors=plt.cm.viridis(intensity),
                        rstride=1, cstride=1, shade=False)

    # Set limits and labels for the 3D plot
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(0, max(z_init))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance')
    ax.set_title('2D Intensity Plots in 3D')



    # Set the rotation angles
    ax.view_init(elev=-20, azim=0, roll=270)

    # Set the aspect ratio to make it a cuboid
    ax.set_box_aspect([15, 15, 50])

    # Show the plot
    plt.draw()

# Connect the update function to the text box events
tip_text_box.on_submit(update)
tilt_text_box.on_submit(update)

# Call the update function to plot with initial tip and tilt values
update(None)

# Show the plot
plt.show()
