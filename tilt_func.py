import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def tilttip(res, phaseshifttip, phaseshifttilt):
    x = np.arange(1, res + 1, 1)
    y = np.arange(1, res + 1, 1)
    X, Y = np.meshgrid(x, y)
    Z = (X * phaseshifttip / res) + (Y * phaseshifttilt / res)
    return  Z