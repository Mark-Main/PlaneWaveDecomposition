import numpy as np


def add_random_scatterer(array, amplitude):
    shape = array.shape
    random_phase = amplitude * np.random.random(shape)
    return array * np.exp(1j * random_phase)