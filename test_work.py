import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def laguerre_gaussian(x, y, p, l, w0, z, k):
    """
    Generate a Laguerre-Gaussian (LG) beam.

    Parameters:
        x (ndarray): X-axis coordinates.
        y (ndarray): Y-axis coordinates.
        p (int): Radial mode.
        l (int): Azimuthal mode.
        w0 (float): Waist parameter.
        z (float): Propagation distance.
        k (float): Wave number.

    Returns:
        ndarray: Complex field amplitude.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Calculate the radial and azimuthal parts
    radial_part = np.sqrt(2 / (np.pi * w0**2)) * (np.sqrt(2) * r / w0)**l * np.exp(-r**2 / w0**2)
    azimuthal_part = np.exp(1j * l * theta)

    # Calculate the Gaussian factor
    gaussian_factor = np.exp(-1j * k * r**2 / (2 * (z + 1j * k * w0**2)))

    # Calculate the Laguerre polynomial
    laguerre_poly = np.polyval(np.poly1d([np.math.factorial(p + l) / (np.math.factorial(p) * np.math.factorial(l)), -1])**p, (2 * r**2) / w0**2)

    # Calculate the Laguerre-Gaussian beam
    laguerre_gaussian_beam = radial_part * azimuthal_part * gaussian_factor * laguerre_poly

    return laguerre_gaussian_beam

# Parameters
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
p_init = 2  # Initial radial mode
l_init = 3  # Initial azimuthal mode
w0 = 2  # Waist parameter
z = 1  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.3)

# Generate the initial Laguerre-Gaussian beam
Z = laguerre_gaussian(X, Y, p_init, l_init, w0, z, k)

# Normalize the intensity
intensity = np.abs(Z)**2
intensity /= np.max(intensity)

# Plot the initial Laguerre-Gaussian beam
cax = ax.imshow(intensity, extent=[x[0], x[-1], y[0], y[-1]], cmap='hot', origin='lower')
plt.colorbar(cax)

# Create the slider axes
ax_l = plt.axes([0.1, 0.3, 0.05, 0.6])
ax_p = plt.axes([0.2, 0.3, 0.05, 0.6])

# Create the sliders
slider_l = Slider(ax=ax_l, label='l', valmin=0, valmax=10, valstep=1, valinit=l_init, orientation='vertical')
slider_p = Slider(ax=ax_p, label='p', valmin=0, valmax=10, valstep=1, valinit=p_init, orientation='vertical')

# Update function for the sliders
def update(val):
    # Get the updated values of l and p
    l = int(slider_l.val)
    p = int(slider_p.val)

    # Generate the Laguerre-Gaussian beam
    Z = laguerre_gaussian(X, Y, p, l, w0, z, k)

    # Normalize the intensity
    intensity = np.abs(Z)**2
    intensity /= np.max(intensity)

    # Update the plot
    cax.set_data(intensity)
    fig.canvas.draw_idle()

# Register the update function with the sliders
slider_l.on_changed(update)
slider_p.on_changed(update)

# Show the plot
plt.show()
