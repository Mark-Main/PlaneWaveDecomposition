import numpy as np
import generateLaguerre2DnoZ
import propogator
import bloodVolumeCreator
import matplotlib.pyplot as plt

for p in range (10):
    for l in range (10):


    overlap_importWave = np.trapz(np.conj(importWave)*importWave, dx=1, axis=0)
    overlapy_importWave = np.trapz(overlap_importWave)

    overlap_refWave = np.trapz(np.conj(refWave)*refWave, dx=1)
    overlapy_refWave = np.trapz(overlap_refWave)

    normalized_importWave = importWave / np.sqrt(overlapy_importWave)
    normalized_refWave = refWave / np.sqrt(overlapy_refWave)

    overlap = np.trapz(np.conj(normalized_refWave)*normalized_importWave, dx=1, axis=0)
    overlap = np.trapz(overlap)

    print(overlap, np.abs(overlap**2))