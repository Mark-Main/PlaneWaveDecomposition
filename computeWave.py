# Generate the initial Laguerre-Gaussian beam
import numpy as np
import supergaussian   
import distanceTerm   

def makeWave(wave, N):

    # Multiply wave and tiptilt arrays
    product = np.multiply(wave, np.exp(1j * N))

    # Take Fourier transform
    fft_result = np.fft.fft2(product)

    return fft_result

