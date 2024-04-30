import numpy as np
import os
import generateLaguerre2DnoZ
import matplotlib.pyplot as plt
import propogator
from tqdm import tqdm

waveRes = 500
x = np.linspace(-200e-6, 200e-6, waveRes)
y = np.linspace(-200e-6, 200e-6, waveRes)
X, Y = np.meshgrid(x, y)
w0 = 20e-6

lgCount = 30

base_directory = r'/Users/Mark/Desktop/NewSimData/PhasemaskTest2'


# Generate reference LG modes
refLGstorage = [[] for _ in range(lgCount)]

for l in range(lgCount):
    refLGstorage[l] = generateLaguerre2DnoZ.laguerre_gaussian(X,Y,0,l,w0)



# Take the overlap integreal of each output with all input LGs
# Gives a values


overlap_values = np.zeros((lgCount, lgCount))

for inputL in range (lgCount):
    for outputL in range (lgCount):

        refWave = refLGstorage[inputL]
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




#Perform SVd on the resulting matrix

U, S, VT = np.linalg.svd(overlap_values)


#Multiply each pixel of the top row by the same i LG
#Sum those values

def multiply_lg_waves_by_svd(lg_waves, svd):
    

    result = np.zeros_like(lg_waves[0])

    for i in range(len(lg_waves)):
        result += lg_waves[i] * svd[0][i]

    return result

svd_mode = multiply_lg_waves_by_svd(refLGstorage,VT)
svd_mode_intensity = np.abs(svd_mode)
plt.imshow(svd_mode_intensity)
plt.show()



#-------------------------
bloodSlices = np.load(f'{base_directory}//bloodSlices.npy')

grid_size,finalDistance, computeStep,s,λ = waveRes, 11,1,0.1,850e-9
oldwave = svd_mode

overlap_importWave = np.trapz(np.conj(oldwave)*oldwave, dx=x[1] - x[0], axis=0)
overlapy_importWave = np.trapz(overlap_importWave)
oldwave = oldwave / np.sqrt(overlapy_importWave)

for i in tqdm (range (0, finalDistance, computeStep), desc="Loading…", ascii=False, ncols=75):
    if i <= grid_size:
        wave = propogator.propogateScatterMask(oldwave, computeStep*1e-6, waveRes, s, λ, bloodSlices[i])
        # intensity_data.append(np.abs(wave) ** 2 / np.max(np.abs(wave) ** 2))
        #phase_data.append(np.angle(wave))
        oldwave = wave
    else:
        wave = propogator.propogate(oldwave, computeStep*1e-2, waveRes, s, λ)
        #intensity_data.append(np.abs(wave) ** 2 / np.max(np.abs(wave) ** 2))
        #phase_data.append(np.angle(wave))
        oldwave = wave



print("SVD sum ",np.sum(np.abs(oldwave) ** 2) )
for mode in range(30):
    output_directory = os.path.join(base_directory, f'L={mode}')
    input = np.load(f'{output_directory}//BinaryDataSlice_l={mode}.npy')
    print("LG sum ",mode, np.sum(np.abs(input) ** 2))

plt.imshow(np.abs(svd_mode)**2)
plt.show()
plt.imshow(np.abs(oldwave)**2)
plt.show()





