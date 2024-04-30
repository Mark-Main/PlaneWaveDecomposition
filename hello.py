import numpy as np
import matplotlib.pyplot as plt

# Replace this with your .npy file path
file_path = '/Users/Mark/Desktop/NewSimData/PhaseScreenTrialMix/PhaseScreens/phase_screen_2.npy'

# Load the array
data = np.load(file_path)

# Plot the array
plt.imshow(data, cmap='gray')  # 'gray' colormap is typical for single-channel images
plt.colorbar()  # Optional, to display a color bar representing the data scale
plt.title('Visual Representation of the .npy File')
plt.show()
