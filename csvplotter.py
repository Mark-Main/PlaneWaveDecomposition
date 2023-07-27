import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = np.loadtxt('array_data2.csv', delimiter=',')

# Create a meshgrid for plotting
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# Plot the data with a colorbar
plt.pcolormesh(X, Y, data, cmap='jet')
plt.colorbar()

# Set axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSV Data with Colorbar')

# Display the plot
plt.show()
