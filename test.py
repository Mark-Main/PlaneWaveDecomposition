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
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    radial = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + np.abs(l))))
    laguerre = np.polynomial.Legendre.basis(l)(2 * rho**2 / w0**2)
    gauss = np.exp(-rho**2 / w0**2)

    phase = np.exp(1j * (l * phi + k * rho**2 / (2 * z)))

    lg_beam = np.sqrt(2 / np.pi) * radial * laguerre * gauss * phase

    return lg_beam

# Parameters
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
p = 1  # Radial mode
l = 2  # Azimuthal mode
w0 = 2  # Waist parameter
z = 100  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

# Create subplot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# Generate initial LG wave
lg_wave = laguerre_gaussian(X, Y, p, l, w0, z, k)

# Plot the intensity profile in 3D
intensity = np.abs(lg_wave)**2
surf = ax.plot_surface(X, Y, intensity, cmap='hot')

# Create axes for p and l sliders
ax_p = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_l = plt.axes([0.25, 0.1, 0.65, 0.03])

# Create p and l sliders with initial values
slider_p = Slider(ax_p, 'p', 0, 5, valinit=p, valstep=1)
slider_l = Slider(ax_l, 'l', -5, 5, valinit=l, valstep=1)

# Function to update the plot when the sliders' values change
def update_plot(val):
    p_val = int(slider_p.val)
    l_val = int(slider_l.val)

    # Generate LG wave with the updated p and l values
    lg_wave = laguerre_gaussian(X, Y, p_val, l_val, w0, z, k)

    # Update the intensity profile
    intensity = np.abs(lg_wave)**2
    surf.set_array(intensity.ravel())
    fig.canvas.draw_idle()

# Link the sliders to the update_plot function
slider_p.on_changed(update_plot)
slider_l.on_changed(update_plot)

# Display the sliders and initial plot
plt.show()

