import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

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

    radial = np.sqrt(2 * sp.factorial(p) / (np.pi * sp.factorial(p + np.abs(l))))
    laguerre = sp.assoc_laguerre(np.abs(l), p, 2 * rho**2 / w0**2)
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
z = 10  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

# Function to update the plot when the slider value changes
def update_plot(l):
    # Generate LG wave with the updated l value
    lg_wave = laguerre_gaussian(X, Y, p, l, w0, z, k)

    # Clear previous plot
    ax.clear()

    # Plot the intensity profile in 3D
    intensity = np.abs(lg_wave)**2
    ax.plot_surface(X, Y, intensity, cmap='hot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title(f'Laguerre-Gaussian (p={p}, l={l}) at z={z}')

    plt.show()

# Create the figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate initial LG wave
lg_wave = laguerre_gaussian(X, Y, p, l, w0, z, k)

# Plot the intensity profile in 3D
intensity = np.abs(lg_wave)**2
ax.plot_surface(X, Y, intensity, cmap='hot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
ax.set_title(f'Laguerre-Gaussian (p={p}, l={l}) at z={z}')

# Create the slider
l_slider = widgets.IntSlider(value=l, min=-5, max=5, description='l')

# Link the slider to the update_plot function
l_slider.observe(lambda change: update_plot(change['new']), names='value')

# Display the slider and initial plot
display(l_slider)
plt.show()
