import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

def tilttip(res, phaseshifttip, phaseshifttilt):
    return [[x * phaseshifttip / res + y * phaseshifttilt / res for y in range(1, res + 1)] for x in range(1, res + 1)]

def plot_tilttip(res, x, y):
    data = tilttip(res, x, y)
    plt.imshow(np.mod(data, 2 * np.pi), cmap='hsv')
    plt.colorbar()
    plt.show()

res = 512
x_range = np.arange(-30 * np.pi, 30 * np.pi, 0.1 * np.pi)
y_range = np.arange(-30 * np.pi, 30 * np.pi, 0.1 * np.pi)

interact(plot_tilttip, res=(1, 1000), x=x_range, y=y_range)
