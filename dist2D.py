import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import genlaguerre
import generateLaguerre2DnoZ
import distanceTerm

# Parameters
x = np.linspace(-10, 10, 512)
y = np.linspace(-10, 10, 512)
X, Y = np.meshgrid(x, y)
p_init = 0  # Initial radial mode
l_init = 0  # Initial azimuthal mode
w0 = 2  # Waist parameter
z_init = 1  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

res = 512
s = 1
λ = 400 / 1000000000

plt.close('all')

# Create the figure and axis for intensity
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
plt.subplots_adjust(left=0.25, bottom=0.3)

# Create and customize the first subplot using ax[0]
ax[0].set_xlabel('X1')
ax[0].set_ylabel('Y1')
ax[0].set_title('Intensity')

# Create and customize the second subplot using ax[1]
ax[1].set_xlabel('X2')
ax[1].set_ylabel('Y2')
ax[1].set_title('Phase')

# Generate the initial Laguerre-Gaussian beam
wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p_init, l_init, w0, k)
distance = distanceTerm.disStep(z_init, res, s, λ)

# Multiply wave and distance arrays
product = np.multiply(wave, distance)

# Take Fourier transform
fft_result = np.fft.fft2(product)

# Perform inverse Fourier transform
ifft_result = np.fft.ifft2(fft_result)

# Normalize the intensity
intensity = np.abs(ifft_result) ** 2

# Identify the phase
phase = np.angle(ifft_result)

# Plot the initial Laguerre-Gaussian beam
cax = ax[0].imshow(intensity, extent=[x[0], x[-1], y[0], y[-1]], cmap='inferno', origin='lower')
cax2 = ax[1].imshow(phase, extent=[x[0], x[-1], y[0], y[-1]], cmap='inferno', origin='lower')

# Create the colorbars
colorbar1 = plt.colorbar(cax, ax=ax[0])
colorbar2 = plt.colorbar(cax2, ax=ax[1])

# Set the minimum value to 0 for both colorbars
colorbar1.mappable.set_clim(vmin=0)
colorbar2.mappable.set_clim(vmin=0)


# Create the slider axes
ax_l = plt.axes([0.1, 0.3, 0.05, 0.6])
ax_p = plt.axes([0.2, 0.3, 0.05, 0.6])
ax_z = plt.axes([0, 0.3, 0.05, 0.6])

# Create the sliders
slider_l = Slider(ax=ax_l, label='l', valmin=0, valmax=10, valstep=1, valinit=l_init, orientation='vertical')
slider_p = Slider(ax=ax_p, label='p', valmin=0, valmax=10, valstep=1, valinit=p_init, orientation='vertical')
slider_z = Slider(ax=ax_z, label='z', valmin=1, valmax=100, valstep=1, valinit=z_init, orientation='vertical')


# Update function for the sliders
def update(val):
    # Get the updated values of l and p
    l = int(slider_l.val)
    p = int(slider_p.val)
    z = int(slider_z.val)

    # Generate the Laguerre-Gaussian beam
    wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p, l, w0, k)
    distance = distanceTerm.disStep(z, res, s, λ)

    # Multiply wave and distance arrays
    product = np.multiply(wave, distance)

    # Take Fourier transform
    fft_result = np.fft.fft2(product)

    # Perform inverse Fourier transform
    ifft_result = np.fft.ifft2(fft_result)

    # Normalize the intensity
    intensity = np.abs(ifft_result) ** 2

    # Identify the phase
    phase = np.angle(ifft_result)

    # Update the plot
    cax.set_data(intensity)
    cax2.set_data(phase)
    fig.canvas.draw_idle()


# Register the update function with the sliders
slider_l.on_changed(update)
slider_p.on_changed(update)
slider_z.on_changed(update)

# Show the plot
plt.show()