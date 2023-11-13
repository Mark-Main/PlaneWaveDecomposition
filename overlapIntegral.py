import numpy as np
import generateLaguerre2DnoZ
import matplotlib.pyplot as plt
import os

# taken from disScatter. Always use the same params
waveRes = 500
x = np.linspace(-100e-6, 100e-6, waveRes)
y = np.linspace(-100e-6, 100e-6, waveRes)
X, Y = np.meshgrid(x, y)
w0 = 20e-6

base_directory = r'D:\MarkSimulations\SphericalTrialLP6_2'
output_directory = os.path.join(base_directory, f'P={1}, L={3}')
importWave = np.load(f'{output_directory}//BinaryDataSlice{799}_p={1}_l={3}.npy')

overlap_values = np.zeros((10, 10))

for p in range (10):
    for l in range (10):

        refWave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, p, l, w0)


        overlap_importWave = np.trapz(np.conj(importWave)*importWave, dx=x[1] - x[0], axis=0)
        overlapy_importWave = np.trapz(overlap_importWave)

        overlap_refWave = np.trapz(np.conj(refWave)*refWave, dx=x[1] - x[0])
        overlapy_refWave = np.trapz(overlap_refWave)

        normalized_importWave = importWave / np.sqrt(overlapy_importWave)
        normalized_refWave = refWave / np.sqrt(overlapy_refWave)

        overlap = np.trapz(np.conj(normalized_refWave)*normalized_importWave, dx=x[1] - x[0], axis=0)
        overlap = np.trapz(overlap)

        overlap_values[p][l] = overlap
        print("p = ",p,"l= ",l)
        print (overlap_values[p][l])




# Create a 2D plot of overlap values against l and p
plt.imshow(overlap_values[::-1], origin='lower', extent=(0, 9, 0, 9), cmap='viridis')  # Invert the p axis
plt.colorbar(label='Overlap Value')
plt.xlabel('L', fontsize=18)
plt.ylabel('P', fontsize=18)

# Set the x and y ticks with labels
plt.xticks(np.arange(0, 10, step=1), [f'l={i}' for i in range(10)], fontsize=12)
plt.yticks(np.arange(0, 10, step=1), [f'p={i}' for i in range(9, -1, -1)], fontsize=12)

plt.title('Overlap Values', fontsize=16)
plt.show()