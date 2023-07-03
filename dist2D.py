import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import genlaguerre
import generateLaguerre2DnoZ
import distanceTerm
import tilt_func

# Parameters
x = np.linspace(-10, 10, 512)
y = np.linspace(-10, 10, 512)
X, Y = np.meshgrid(x, y)
p_init = 2  # Initial radial mode
l_init = 3  # Initial azimuthal mode
w0 = 2  # Waist parameter
z_init = 1  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

res = 512
s = 10
λ = 600 / 1000000000

phaseshifttip_init = 3.0
phaseshifttilt_init = 1.0



# Create the figure and axis for intensity
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))

plt.subplots_adjust(left=0.25, bottom=0.3)

# Create and customize the first subplot using ax[0, 0]
ax[0, 0].set_xlabel('X1')
ax[0, 0].set_ylabel('Y1')
ax[0, 0].set_title('Intensity')

# Create and customize the second subplot using ax[0, 1]
ax[0, 1].set_xlabel('X2')
ax[0, 1].set_ylabel('Y2')
ax[0, 1].set_title('Phase')

# Create and customize the third subplot using ax[1, 0]
ax[1, 0].set_xlabel('X3')
ax[1, 0].set_ylabel('Y3')
ax[1, 0].set_title('Tiptilt')


# Generate the initial Laguerre-Gaussian beam
wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p_init, l_init, w0, k)
distance = distanceTerm.disStep(z_init, res, s, λ)
L, M, N = tilt_func.tilttip(res, phaseshifttip_init, phaseshifttilt_init)

# Multiply wave and distance arrays
product = np.multiply(wave, np.exp(1j * N))


# Take Fourier transform
fft_result = np.fft.fft2(product)

# Perform inverse Fourier transform
ifft_result = np.fft.ifft2(fft_result)

# Normalize the intensity
intensity = np.abs(ifft_result * distance) ** 2

# Identify the phase
phase = np.angle(ifft_result * distance)

# Plot the initial Laguerre-Gaussian beam
cax = ax[0,0].imshow(intensity, extent=[x[10], x[-10], y[1], y[-1]], cmap='inferno', origin='lower')
cax2 = ax[0,1].imshow(phase, extent=[x[0], x[-1], y[0], y[-1]], cmap='inferno', origin='lower')
cax3 = ax[1,0].imshow(N, cmap='inferno', origin='lower')

# Create the colorbars
colorbar1 = plt.colorbar(cax, ax=ax[0,0])
colorbar2 = plt.colorbar(cax2, ax=ax[0,1])

# Set the minimum value to 0 for both colorbars
colorbar1.mappable.set_clim(vmin=0, vmax=1)
colorbar2.mappable.set_clim(vmin=-3)


# Create the slider axes
ax_l = plt.axes([0.2, 0.3, 0.05, 0.6])
ax_p = plt.axes([0.15, 0.3, 0.05, 0.6])
ax_z = plt.axes([0.1, 0.3, 0.05, 0.6])
ax_tip = plt.axes([0.05, 0.3, 0.05, 0.6])
ax_tilt = plt.axes([0.0, 0.3, 0.05, 0.6])

# Create the sliders
slider_l = Slider(ax=ax_l, label='l', valmin=0, valmax=10, valstep=1, valinit=l_init, orientation='vertical')
slider_p = Slider(ax=ax_p, label='p', valmin=0, valmax=10, valstep=1, valinit=p_init, orientation='vertical')
slider_z = Slider(ax=ax_z, label='z', valmin=1, valmax=100, valstep=1, valinit=z_init, orientation='vertical')
phaseshifttip_slider = Slider(ax_tip, 'Phase Shift Tip', 0.1, 10.0, valinit=phaseshifttip_init, valstep=0.1, orientation='vertical')
phaseshifttilt_slider = Slider(ax_tilt, 'Phase Shift Tilt', 0.1, 10.0, valinit=phaseshifttilt_init, valstep=0.1, orientation='vertical')


# Update function for the sliders
def update(val):
    # Get the updated values of l and p
    l = int(slider_l.val)
    p = int(slider_p.val)
    z = slider_z.val
    phaseshifttip = phaseshifttip_slider.val
    phaseshifttilt = phaseshifttilt_slider.val

    # Generate the Laguerre-Gaussian beam
    L, M, N = tilt_func.tilttip(res, phaseshifttip, phaseshifttilt)
    wave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p, l, w0, k)
    distance = distanceTerm.disStep(z, res, s, λ)


    # Multiply wave and distance arrays
    product = np.multiply(wave, np.exp(1j * N))

    # Take Fourier transform
    fft_result = np.fft.fft2(product)

    # Perform inverse Fourier transform
    ifft_result = np.fft.ifft2(fft_result)

    # Normalize the intensity
    intensity = np.abs(ifft_result * distance) ** 2

    # Identify the phase
    phase = np.angle(ifft_result * distance)

    # Update the plot
    cax.set_data(intensity)
    cax2.set_data(phase)
    cax3.set_array(N)
    fig.canvas.draw_idle()



# Register the update function with the sliders
slider_l.on_changed(update)
slider_p.on_changed(update)
slider_z.on_changed(update)
phaseshifttip_slider.on_changed(update)
phaseshifttilt_slider.on_changed(update)

# Show the plot
plt.show()
