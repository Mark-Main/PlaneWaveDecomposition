# Generate the initial Laguerre-Gaussian beam
import numpy as np
import supergaussian   
import distanceTerm   

def makeWave(wave, N, X2, Y2, sigma, exponent):

    # Generate the super Gaussian function
    super_gaussian = supergaussian.super_gaussian(X2, Y2, sigma, exponent)

    # Normalize the super Gaussian
    super_gaussian /= np.max(super_gaussian)

    # Calculate the dimensions of the wave array
    wave_size = wave.shape[0]
    super_gaussian_size = super_gaussian.shape[0]
    

    # Calculate the indices for adding the wave to the center of the super Gaussian
    x_start = (super_gaussian_size - wave_size) // 2
    print(x_start)
    x_end = x_start + wave_size
    y_start = (super_gaussian_size - wave_size) // 2
    y_end = y_start + wave_size

    # Convert the super Gaussian array to complex
    super_gaussian = super_gaussian.astype(complex)

    # Create an array of the same size as the super Gaussian to add the wave
    wave_padded = np.ones_like(super_gaussian)    
    wave_padded[x_start:x_end, y_start:y_end] = wave

    # Add the wave to the super Gaussian
    super_gaussian *= wave_padded

    # Multiply wave and distance arrays
    product = np.multiply(super_gaussian, np.exp(1j * N))

    # Take Fourier transform
    fft_result = np.fft.fft2(product)

    return fft_result

