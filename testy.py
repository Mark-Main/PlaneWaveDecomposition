import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate points on a torus
def generate_torus(R, r, num_points):
    phi = np.linspace(0, 2*np.pi, num_points)
    theta = np.linspace(0, 2*np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

# Number of points on the torus
num_points = 100

# Radii of the torus
R = 10  # Major radius (increase for bigger torus)
r = 8   # Minor radius (decrease for smaller torus)

# Grid parameters
grid_size = 200  # Increase for a larger grid
voxel_resolution = 1.0

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# List to store toroid positions
toroid_positions = []

# Generate and plot 10 toroids with spacing and collision detection
for _ in range(200):
    toroid_collides = True

    while toroid_collides:
        # Generate torus points
        x, y, z = generate_torus(R, r, num_points)

        # Apply random rotation
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, 2*np.pi)

        x_rotated = x * np.cos(theta) * np.cos(phi) - y * np.sin(phi) * np.cos(theta) + z * np.sin(theta)
        y_rotated = x * np.sin(phi) + y * np.cos(phi)
        z_rotated = -x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(theta)

        # Scale the rotated torus down
        x_scaled = x_rotated / 5  # Adjust the scaling factor as needed
        y_scaled = y_rotated / 5
        z_scaled = z_rotated / 5

        # Set the toroid position with spacing
        x_offset = np.random.uniform(-50, 50)  # Adjust the range for spacing
        y_offset = np.random.uniform(-50, 50)
        z_offset = np.random.uniform(-50, 50)

        x_final = x_scaled + x_offset
        y_final = y_scaled + y_offset
        z_final = z_scaled + z_offset

        # Check for collisions with existing toroids
        toroid_collides = any(
            np.linalg.norm(np.array([x_final, y_final, z_final]) - pos) < 2 * r
            for pos in toroid_positions
        )

    # Store the toroid position
    toroid_positions.append(np.array([x_final, y_final, z_final]))

    # Plot the scaled and positioned torus
    ax.plot_surface(x_final, y_final, z_final, cmap='viridis', alpha=0.5)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Grid with Small Toroids and Spacing')

plt.show()