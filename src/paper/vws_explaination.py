import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk
import matplotlib.patches as patches

img = np.ones((500, 500))

not_ws_signals_locations = [[100, 100], [150, 250], [220, 380], [350, 150]]
ws_signals_locations = [[100, 100], [100, 250], [220, 380], [350, 150]]
signal = np.ones(shape=(100, 100))

rr, cc = disk((50, 50), 50)
signal[rr, cc] = 0

fig, ax = plt.subplots(1, 2)

img = np.ones((500, 500))
for loc in ws_signals_locations:

    img[loc[0]: loc[0] + 100, loc[1]:loc[1] + 100] = signal
    rect = patches.Rectangle((loc[1], loc[0]), 100, 100, linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
    ax[0].add_patch(rect)
    ax[0].scatter(loc[1], loc[0], color='red')
    ax[0].plot([0, 500], [loc[0], loc[0]], color='gold', linestyle='--')

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('This example is well separated.')

img = np.ones((500, 500))
for loc in not_ws_signals_locations:

    img[loc[0]: loc[0] + 100, loc[1]:loc[1] + 100] = signal
    rect = patches.Rectangle((loc[1], loc[0]), 100, 100, linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
    ax[1].add_patch(rect)
    ax[1].scatter(loc[1], loc[0], color='red')
    ax[1].plot([0, 500], [loc[0], loc[0]], color='gold', linestyle='--')

    ax[1].imshow(img, cmap='gray')
    ax[1].set_title('This example is not well separated.')

plt.show()
