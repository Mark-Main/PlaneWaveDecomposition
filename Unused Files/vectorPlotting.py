import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import generateLaguerre3D

# Define the parameters
p = 1     # Radial mode
l = 2     # Azimuthal mode
w0 = 1.0  # Waist parameter
z0 = 10.0 # Rayleigh range
k = 2 * np.pi / 0.5  # Wave number

# Generate the grid of coordinates
x = np.linspace(-5, 5, 103)  # Adjusted dimension
y = np.linspace(-5, 5, 103)  # Adjusted dimension
z = np.linspace(-10, 10, 103)  # Adjusted dimension
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Compute the Laguerre-Gaussian beam
beam = generateLaguerre3D.laguerre_gaussian(X, Y, Z, p, l, w0, z0, k)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Compute the magnitude of the beam
magnitude = np.abs(beam)

# Normalize the beam magnitudes for vector length
normalized_magnitude = magnitude / np.max(magnitude)

# Define the vector components
u = normalized_magnitude * np.cos(np.angle(beam))
v = normalized_magnitude * np.sin(np.angle(beam))
w = np.zeros_like(u)  # Set the z-component to zero, as we're plotting in 3D

# Plot the vectors
ax.quiver(X.flatten(), Y.flatten(), Z.flatten(), u.flatten(), v.flatten(), w.flatten(), length=0.2, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Laguerre-Gaussian Beam (Vectors)')
plt.show()
