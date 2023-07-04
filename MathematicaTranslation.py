import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def disStep(d, res, s, λ):
    x, y = np.meshgrid(np.arange(res), np.arange(res))
    phase = np.sqrt((6.28318 / λ) ** 2 - ((2 * np.pi * ((np.mod(res / 2 + x, res) - res / 2) / s)) ** 2 +
                  (2 * np.pi * ((np.mod(res / 2 + y, res) - res / 2) / s)) ** 2))
    return np.exp(d * phase * (0 + 1j))

def complexForm(r, theta):
    return r * np.cos(theta) + r * np.sin(theta) * 1j

def lens(res, width, f, lam):
    x, y = np.meshgrid(np.linspace(-width/2, width/2, res), np.linspace(-width/2, width/2, res))
    return np.exp(-1j * (2 * np.pi / lam) * (x**2 + y**2) / (2 * f))

def lgNorm(res, width, l, p, w0):
    x, y = np.meshgrid(np.linspace(-width/2, width/2, res), np.linspace(-width/2, width/2, res))
    norm = np.where((x == 0) & (y == 0), 0, (2 * (x**2 + y**2) / w0**2) ** abs(l) * np.exp(- (x**2 + y**2) / w0**2))
    return norm * np.exp(1j * l * np.arctan2(x, y))

def tilttip(res, phaseshifttip, phaseshifttilt):
    x, y = np.meshgrid(np.linspace(-30 * np.pi, 30 * np.pi, res), np.linspace(-30 * np.pi, 30 * np.pi, res))
    return x * phaseshifttip / res + y * phaseshifttilt / res

# Parameters
λ = 600 * 10**(-9)
res = 512

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Initialize plots
x_val = 0
y_val = 0

phasemask = np.mod(tilttip(res, x_val, y_val), 2 * np.pi)
binarygrating = np.where(phasemask > np.pi, 1, 0)

lg_norm = lgNorm(res, 0.025, -2, 2, 0.002)
tilttip_phase = tilttip(res, x_val, y_val)
beam = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(lg_norm) * disStep(2, res, 0.025, λ) * np.exp(1j * tilttip_phase)))

img1 = axs[0].imshow(phasemask, cmap='gray')
img2 = axs[1].imshow(binarygrating, cmap='gray')
img3 = axs[2].imshow(np.abs(beam), cmap='gray')

# Update plots
def update_plots(x_val, y_val):
    phasemask = np.mod(tilttip(res, x_val, y_val), 2 * np.pi)
    binarygrating = np.where(phasemask > np.pi, 1, 0)
    tilttip_phase = tilttip(res, x_val, y_val)
    beam = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(lg_norm) * disStep(2, res, 0.025, λ) * np.exp(1j * tilttip_phase)))

    img1.set_array(phasemask)
    img2.set_array(binarygrating)
    img3.set_array(np.abs(beam))

    fig.canvas.draw()

# Slider widgets
x_slider = widgets.FloatSlider(min=-30*np.pi, max=30*np.pi, step=0.1*np.pi, value=0, description='X')
y_slider = widgets.FloatSlider(min=-30*np.pi, max=30*np.pi, step=0.1*np.pi, value=0, description='Y')

# Update plots when sliders change
widgets.interact(update_plots, x_val=x_slider, y_val=y_slider)

# Display sliders and plots
display(widgets.HBox([x_slider, y_slider]))
plt.show()
