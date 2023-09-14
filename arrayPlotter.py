import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the directory containing the plots
base_directory = r'D:\MarkSimulations\SphericalTrialLP6\AllPlots2'

# Get a list of all the plot files
plot_files = [os.path.join(base_directory, f'plot_{i}.png') for i in range(500,600)]

def update(frame):
    plt.clf()  # Clear the figure
    img = plt.imread(plot_files[frame])
    plt.imshow(img)
    plt.axis('off')

# Create the animation
ani = FuncAnimation(plt.gcf(), update, frames=len(plot_files), repeat=True, interval=200)

# Save the animation as a GIF with improved quality
output_file = os.path.join(base_directory, 'animation5.gif')
ani.save(output_file, writer='pillow', fps=20, dpi=300)

