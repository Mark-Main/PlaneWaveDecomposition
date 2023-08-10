import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import genlaguerre

def laguerre_gaussian(x, y, p, l, w0):
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
   # gaussian_factor = np.exp(-1j * k * r**2 / (2 * (z + 1j * k * w0**2)))

    # Calculate the Laguerre polynomial
    laguerre_poly = genlaguerre(p, l)(2 * r**2 / w0**2)

    # Calculate the Laguerre-Gaussian beam
    laguerre_gaussian_beam = radial_part * azimuthal_part * laguerre_poly

    return laguerre_gaussian_beam

def gaussian(x, y, x0, y0, sigma_x, sigma_y, max_intensity):
    """
    Simulate a Gaussian laser beam intensity profile.

    Parameters:
        x (np.ndarray): X-coordinates of the grid.
        y (np.ndarray): Y-coordinates of the grid.
        x0 (float): X-coordinate of the beam center.
        y0 (float): Y-coordinate of the beam center.
        sigma_x (float): Standard deviation of the Gaussian in the x-direction.
        sigma_y (float): Standard deviation of the Gaussian in the y-direction.
        max_intensity (float, optional): Maximum intensity of the Gaussian beam. Default is 1.0.

    Returns:
        np.ndarray: 2D array representing the Gaussian beam intensity profile.
    """
    intensity = max_intensity * np.exp(-((x - x0) ** 2) / (2 * sigma_x ** 2) - ((y - y0) ** 2) / (2 * sigma_y ** 2))
    return intensity