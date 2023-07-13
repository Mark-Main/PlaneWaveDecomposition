import numpy as np
import matplotlib.pyplot as plt
res =128
x = np.linspace(-1,1, res)
y = np.linspace(-1, 1, res)
X, Y = np.meshgrid(x, y)


sigma = 0.5  # Standard deviation of the super Gaussian
exponent = 2  # Exponent controlling the shape of the super Gaussian


R = np.sqrt(X**2 + Y**2)
super_gaussian = np.exp(-(R/sigma)**exponent)

#bigplot = np.zeros((res*10, res*10), dtype=complex)
#bigplot[:super_gaussian.shape[0], :super_gaussian.shape[1]] = super_gaussian

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 30))

plt.subplots_adjust(left=0.25, bottom=0.3)

cax = ax.imshow(super_gaussian, cmap='inferno', origin='lower')
colorbar1 = plt.colorbar(cax, ax=ax)

# Show the plot
plt.show()