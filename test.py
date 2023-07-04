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
p = 2  # Radial mode
l = 3  # Azimuthal mode
w0 = 2  # Waist parameter
z = 1  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

# Create subplot for the 3D plot
fig = plt.figure()
ax_3d = fig.add_subplot(121, projection='3d')

# Generate initial LG wave
lg_wave = laguerre_gaussian(X, Y, p, l, w0, z, k)

# Plot the intensity profile in 3D
intensity = np.abs(lg_wave)**2

surf = ax_3d.plot_surface(X, Y, intensity, cmap='jet')

# Create axes for p and l sliders
ax_p = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_l = plt.axes([0.25, 0.01, 0.65, 0.03])

# Create p and l sliders with initial values
slider_p = Slider(ax_p, 'p', 0, 5, valinit=p, valstep=1)
slider_l = Slider(ax_l, 'l', 0, 5, valinit=l, valstep=1)

# Function to update the plots when the sliders' values change
def update_plot(val):
    global surf  # Declare surf as global
    p_val = int(slider_p.val)
    l_val = int(slider_l.val)

    # Generate LG wave with the updated p and l values
    lg_wave = laguerre_gaussian(X, Y, p_val, l_val, w0, z, k)

    # Update the intensity profile
    intensity = np.abs(lg_wave)**2

    # Remove the previous plot in the 3D subplot
    ax_3d.collections.remove(surf)

    # Plot the new intensity profile in the 3D subplot
    surf = ax_3d.plot_surface(X, Y, intensity, cmap='jet')

    # Adjust the plot labels in the 3D subplot
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Intensity')
    ax_3d.set_title(f'Laguerre-Gaussian (p={p_val}, l={l_val}) at z={z}')

    # Update the 2D subplot
    ax_2d.imshow(intensity.reshape(X.shape), cmap='jet', origin='lower')
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Y')
    ax_2d.set_title(f'Laguerre-Gaussian (p={p_val}, l={l_val}) at z={z}')

    # Update the plots
    fig.canvas.draw_idle()


# Link the sliders to the update_plot function
slider_p.on_changed(update_plot)
slider_l.on_changed(update_plot)

# Create subplot for the 2D plot
ax_2d = fig.add_subplot(122)

# Generate initial LG wave for the 2D plot
lg_wave_2d = laguerre_gaussian(X, Y, p, l, w0, z, k)

# Plot the intensity profile in 2D
intensity_2d = np.abs(lg_wave_2d)**2
ax_2d.imshow(intensity_2d.reshape(X.shape), cmap='jet', origin='lower')
ax_2d.set_xlabel('X')
ax_2d.set_ylabel('Y')
ax_2d.set_title(f'Laguerre-Gaussian (p={p}, l={l}) at z={z}')

# Display the sliders and initial plots
plt.show()
