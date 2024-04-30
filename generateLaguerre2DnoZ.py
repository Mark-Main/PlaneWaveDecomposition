import numpy as np
from scipy.special import genlaguerre

def laguerre_gaussian(x, y, p, l, w0, z=0, k=1):
    """
    Generate a Laguerre-Gaussian (LG) beam, allowing for negative l values.

    Parameters:
        x (ndarray): X-axis coordinates.
        y (ndarray): Y-axis coordinates.
        p (int): Radial mode.
        l (int): Azimuthal mode.
        w0 (float): Waist parameter.
        z (float): Propagation distance (included but not used in this calculation).
        k (float): Wave number (included but not used in this calculation).

    Returns:
        ndarray: Complex field amplitude.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Use absolute value of l for radial part and Laguerre polynomial to handle negative l values
    radial_part = np.sqrt(2 / (np.pi * w0**2)) * (np.sqrt(2) * r / w0)**abs(l) * np.exp(-r**2 / w0**2)
    azimuthal_part = np.exp(1j * l * theta)  # This naturally handles negative l values

    # Calculate the Laguerre polynomial for the absolute value of l
    laguerre_poly = genlaguerre(p, abs(l))(2 * r**2 / w0**2)

    # Calculate the Laguerre-Gaussian beam
    laguerre_gaussian_beam = radial_part * azimuthal_part * laguerre_poly

    return laguerre_gaussian_beam
