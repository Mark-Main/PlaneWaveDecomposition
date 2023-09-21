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

base_directory = r'/Users/Mark/Documents/Test'


overlap_values = np.zeros((10, 10))

for inputL in range (10):
    for outputL in range (10):

        refWave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, inputL, w0)
        output_directory = os.path.join(base_directory, f'P={0}, L={outputL}')
        importWave = np.load(f'{output_directory}//BinaryDataSlice{420}_p={0}_l={outputL}.npy')

        overlap_importWave = np.trapz(np.conj(importWave)*importWave, dx=x[1] - x[0], axis=0)
        overlapy_importWave = np.trapz(overlap_importWave)

        overlap_refWave = np.trapz(np.conj(refWave)*refWave, dx=x[1] - x[0])
        overlapy_refWave = np.trapz(overlap_refWave)

        normalized_importWave = importWave / np.sqrt(overlapy_importWave)
        normalized_refWave = refWave / np.sqrt(overlapy_refWave)

        overlap = np.trapz(np.conj(normalized_refWave)*normalized_importWave, dx=x[1] - x[0], axis=0)
        overlap = np.trapz(overlap)

        overlap_values[inputL][outputL] = np.abs(overlap)
        print("Input L = ",inputL,"OutputL= ",outputL)
        print (overlap_values[inputL][outputL])




# Create a 2D plot of overlap values against l and p
plt.imshow(overlap_values[::-1], origin='lower', extent=(0, 9, 0, 9), cmap='viridis')  # Invert the p axis
plt.colorbar(label='Overlap Value')
plt.xlabel('Input L', fontsize=18)
plt.ylabel('Output L', fontsize=18)

# Set the x and y ticks with labels
plt.xticks(np.arange(0, 10, step=1), [f'l={i}' for i in range(10)], fontsize=12)
plt.yticks(np.arange(0, 10, step=1), [f'l={i}' for i in range(9, -1, -1)], fontsize=12)

plt.title('Overlap Values', fontsize=16)
plt.show()