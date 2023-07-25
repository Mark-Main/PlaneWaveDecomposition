import numpy as np

def super_gaussian(X2, Y2, sigma, exponent):
    """
    Generate the super Gaussian function.

    Parameters:
        X2 (float or numpy array): X-coordinate squared.
        Y2 (float or numpy array): Y-coordinate squared.
        sigma (float): Width parameter controlling the spread of the function.
        exponent (float): Exponent parameter determining the shape of the function.

    Returns:
        numpy array: The result of the super Gaussian function evaluation.
    """
    R = np.sqrt(X2 ** 2 + Y2 ** 2)
    super_gaussian = np.exp(-(R / sigma) ** exponent)
    return super_gaussian
