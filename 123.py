import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Base path to the directories
base_path = Path("/Volumes/DeBroglie/TimeAverage")

# Initialize the figure with subplots
fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # Adjust the grid size as needed
axes = axes.flatten()

# Store the initial data and images for each subplot
average_datas = []
ims = []

for i, ax in enumerate(axes):
    directory_path = base_path / f"SVDstore={i}"

    file_paths = sorted(directory_path.glob('BinaryDataSlice_runnumber=*.npy'), key=lambda x: int(x.stem.split('=')[-1]))
    data = np.abs(np.load(file_paths[0]))**2
    average_data = np.copy(data)
    average_datas.append(average_data)
    
    im = ax.imshow(average_data, animated=True, cmap='inferno')
    ims.append(im)

# Position the frame number text at the top center of the figure
# Adjust y=0.95 to move it lower or higher as needed
frame_text = fig.text(0.5, 0.97, '', ha='center', va='center', fontsize='large')

# Function to update each figure and the frame number
def updatefig(i):
    global frame_text
    for store_index, (im, ax) in enumerate(zip(ims, axes)):
        directory_path = base_path / f"SVDstore={store_index}"
        file_paths = sorted(directory_path.glob('BinaryDataSlice_runnumber=*.npy'), key=lambda x: int(x.stem.split('=')[-1]))
        
        if i < min(100, len(file_paths)):
            new_data = np.abs(np.load(file_paths[i]))**2
            average_datas[store_index] = ((average_datas[store_index] * i) + new_data) / (i + 1)
            im.set_array(average_datas[store_index])
        
    # Update the frame number, adjust formatting as needed
    frame_text.set_text(f'Frame {i+1}')

    return ims + [frame_text]

# Create an animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(1, 1000), interval=1, blit=False)

plt.tight_layout(pad=3.0)
plt.show()
