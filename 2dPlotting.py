import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import genlaguerre
import generateLaguerre2D


# Parameters
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
p_init = 2  # Initial radial mode
l_init = 3  # Initial azimuthal mode
w0 = 2  # Waist parameter
z_init = 0  # Propagation distance
k = 2 * np.pi / 0.5  # Wave number

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
Z = generateLaguerre2D.laguerre_gaussian(X, Y, p_init, l_init, w0, z_init, k)

# Normalize the intensity
intensity = np.abs(Z)**2
intensity /= np.max(intensity)

#Identify the phase

phase = np.angle(Z)

# Plot the initial Laguerre-Gaussian beam
cax = ax[0].imshow(intensity, extent=[x[0], x[-1], y[0], y[-1]], cmap='inferno', origin='lower')
cax2 = ax[1].imshow(phase, extent=[x[0], x[-1], y[0], y[-1]], cmap='inferno', origin='lower')
plt.colorbar(cax)
plt.colorbar(cax2)

# Create the slider axes
ax_l = plt.axes([0.1, 0.3, 0.05, 0.6])
ax_p = plt.axes([0.2, 0.3, 0.05, 0.6])
ax_z = plt.axes([0, 0.3, 0.05, 0.6])

# Create the sliders
slider_l = Slider(ax=ax_l, label='l', valmin=0, valmax=10, valstep=1, valinit=l_init, orientation='vertical')
slider_p = Slider(ax=ax_p, label='p', valmin=0, valmax=10, valstep=1, valinit=p_init, orientation='vertical')
slider_z = Slider(ax=ax_z, label='z', valmin=0, valmax=100, valstep=5, valinit=z_init, orientation='vertical')

# Update function for the sliders
def update(val):
    # Get the updated values of l and p
    l = int(slider_l.val)
    p = int(slider_p.val)
    z = int(slider_z.val)

    # Generate the Laguerre-Gaussian beam
    Z = generateLaguerre2D.laguerre_gaussian(X, Y, p, l, w0, z, k)

    # Normalize the intensity
    intensity = np.abs(Z)**2
    intensity /= np.max(intensity)

    #Identify the phase

    phase = np.angle(Z)

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
