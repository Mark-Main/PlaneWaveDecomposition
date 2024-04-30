import numpy as np
import distanceTerm
import randomScatter
import supergaussian


def propogateScatterMask(wave, computationStep, res2, s , λ, scattermask):

        # Create distance array matching the size of the super Gaussian
        distance = distanceTerm.disStep(computationStep, res2, s, λ)  # Modify size of distance

        # Multiply wave and distance arrays
        # wave = np.multiply(wave, np.exp(1j * N))

        #Apply phase mask


        modified_wave = np.multiply(wave, np.exp(1j * scattermask))

        
        # Take Fourier transform
        fft_result = np.fft.fft2(modified_wave)

        # Perform inverse Fourier transform
        ifft_result = np.fft.ifft2(fft_result * distance)
       
        

        return ifft_result

def propogateScatter(wave, computationStep, res2, s , λ, scattersize):

        # Create distance array matching the size of the super Gaussian
        distance = distanceTerm.disStep(computationStep, res2, s, λ)  # Modify size of distance

        # Multiply wave and distance arrays
        # wave = np.multiply(wave, np.exp(1j * N))

        # Add random scatterer
        product = randomScatter.add_random_scatterer(wave, scattersize)
        
        # Take Fourier transform
        fft_result = np.fft.fft2(product)

        # Perform inverse Fourier transform
        ifft_result = np.fft.ifft2(fft_result * distance)
        print(computationStep)

        return ifft_result


def propogate(wave, computationStep, res2, s , λ):

        # Create distance array matching the size of the super Gaussian
        distance = distanceTerm.disStep(computationStep, res2, s, λ)  # Modify size of distance
        
        # Take Fourier transform
        fft_result = np.fft.fft2(wave)

        # Perform inverse Fourier transform
        ifft_result = np.fft.ifft2(fft_result * distance)

        return ifft_result