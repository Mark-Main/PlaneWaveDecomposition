import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define lens shape (parabolic for simplicity)
def lens_shape(x):
    return 0.02 * x**2

# Define a ray
def ray(x, y, z):
    return (x, y, z)

# Generate lens shape
x = np.linspace(-10, 10, 100)
y = lens_shape(x)

# Generate z coordinates (height of the lens)
z = np.linspace(0, 0, len(x))

# Generate ray coordinates
x_ray = np.linspace(-10, 10, 100)
y_ray = np.linspace(0, lens_shape(0), 100)
z_ray = np.linspace(0, 0, 100)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the lens
ax.plot(x, y, z)

# Plot the ray
ax.plot(x_ray, y_ray, z_ray, 'r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
