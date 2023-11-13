import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import genlaguerre

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
    laguerre_poly = genlaguerre(p, l)(2 * r**2 / w0**2)

    # Calculate the Laguerre-Gaussian beam
    laguerre_gaussian_beam = radial_part * azimuthal_part * gaussian_factor * laguerre_poly

    return laguerre_gaussian_beam