import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters
D = 1.0  # Diameter of the disc

# Coefficient values for demonstration (you can adjust these)
a0 = 1.0
a1 = 0.2
a2 = 0.1

# Create a grid of r values
r = np.linspace(0, D / 2, 100)

# Create a grid of theta values (angle)
theta = np.linspace(0, 2 * np.pi, 100)

# Create a grid of (r, theta) values
R, Theta = np.meshgrid(r, theta)

# Calculate corresponding X, Y, and Z values using the original equation
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = D * np.sqrt(1 - (4 * R**2 / D**2)) * (a0 + a1 * (R**2 / (D/2)**2) + a2 * (R**4 / (D/2)**4))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Biconcave Disc Surface')

# Show the plot
plt.show()