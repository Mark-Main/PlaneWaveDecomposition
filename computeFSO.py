# Generate the initial Laguerre-Gaussian beam
import numpy as np
import supergaussian   
import distanceTerm   

def makeWave(wave, N, X2, Y2, sigma, exponent, center_coordinates):

     # Generate the super Gaussian function
    super_gaussian = supergaussian.super_gaussian(X2, Y2, sigma, exponent)

    # Normalize the super Gaussian
    super_gaussian /= np.max(super_gaussian)

    # Calculate the dimensions of the wave array
    wave_size = wave.shape[0]
    super_gaussian_size = super_gaussian.shape[0]
    

    # Convert the super Gaussian array to complex
    super_gaussian = super_gaussian.astype(complex)

    # Create an array of the same size as the super Gaussian to add the waves
    wave_padded = np.ones_like(super_gaussian)

    # Loop over the center_coordinates and place waves at those positions
    for coord in center_coordinates:

        x_center, y_center = coord
        
        x_start = x_center - wave_size // 2
        
        x_end = x_start + wave_size
        
        y_start = y_center - wave_size // 2
        y_end = y_start + wave_size


        # Make sure the wave fits within the super_gaussian array
        # Note: You may want to consider a padding strategy if the wave doesn't fit.
        
        
        wave_padded[x_start:x_end, y_start:y_end] = wave

    # Add the waves to the super Gaussian
    #super_gaussian *= wave_padded

    # Multiply wave and distance arrays
    product = np.multiply(wave_padded, np.exp(1j * N))

    # Take Fourier transform
    fft_result = np.fft.fft2(product)

    return fft_result