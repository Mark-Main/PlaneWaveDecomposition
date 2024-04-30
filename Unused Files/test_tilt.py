import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def tilttip(res, phaseshifttip, phaseshifttilt):
    x = np.arange(1, res + 1, 1)
    y = np.arange(1, res + 1, 1)
    X, Y = np.meshgrid(x, y)
    Z = X * phaseshifttip / res + Y * phaseshifttilt / res
    return X, Y, Z

# Initial parameter values
res = 512
phaseshifttip_init = 1.0
phaseshifttilt_init = 1.0

# Create initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')

X, Y, Z = tilttip(res, phaseshifttip_init, phaseshifttilt_init)
cax = ax.imshow(Z, cmap='inferno', origin='lower')

# Create sliders
phaseshifttip_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
phaseshifttilt_slider_ax = plt.axes([0.25, 0.10, 0.65, 0.03])

phaseshifttip_slider = Slider(phaseshifttip_slider_ax, 'Phase Shift Tip', 0.1, 10.0, valinit=phaseshifttip_init, valstep=0.1)
phaseshifttilt_slider = Slider(phaseshifttilt_slider_ax, 'Phase Shift Tilt', 0.1, 10.0, valinit=phaseshifttilt_init, valstep=0.1)

# Update plot when slider values change
def update(val):
    phaseshifttip = phaseshifttip_slider.val
    phaseshifttilt = phaseshifttilt_slider.val
    X, Y, Z = tilttip(res, phaseshifttip, phaseshifttilt)
    cax.set_array(Z)
    fig.canvas.draw_idle()

phaseshifttip_slider.on_changed(update)
phaseshifttilt_slider.on_changed(update)

plt.show()
