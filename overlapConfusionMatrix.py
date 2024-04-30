import numpy as np
import generateLaguerre2DnoZ
import matplotlib.pyplot as plt
import os

# taken from disScatter. Always use the same params
waveRes = 500
x = np.linspace(-200e-6, 200e-6, waveRes)
y = np.linspace(-200e-6, 200e-6, waveRes)
X, Y = np.meshgrid(x, y)
w0 = 20e-6

base_directory = r'/Users/Mark/Desktop/NewSimData/PhasemaskTest2'


overlap_values = np.zeros((30, 30))

for inputL in range (30):
    for outputL in range (30):

        refWave = generateLaguerre2DnoZ.laguerre_gaussian(X, Y, 0, inputL, w0)
        output_directory = os.path.join(base_directory, f'L={outputL}')
        importWave = np.load(f'{output_directory}//BinaryDataSlice_l={outputL}.npy')

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
plt.imshow(overlap_values[::-1], origin='lower', extent=(0, 29, 0, 29))  # Invert the p axis
plt.colorbar(label='Overlap Value')
plt.xlabel('Input L', fontsize=18)
plt.ylabel('Output L', fontsize=18)

# Set the x and y ticks with labels
plt.xticks(np.arange(0, 30, step=1), [f'{i}' for i in range(30)], fontsize=12)
plt.yticks(np.arange(0, 30, step=1), [f'{i}' for i in range(29, -1, -1)], fontsize=12)

plt.title('Overlap Values for Discocyte with 10% Bumps', fontsize=16)
plt.show()