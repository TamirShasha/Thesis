import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk

img = np.zeros((100, 100))

disk_size = 10
centers = [(20, 20)]

for center in centers:
    rr, cc = disk(center, disk_size // 2)
    img[rr, cc] = 0

fig, ax = plt.subplots()
# ax.imshow(img, cmap='gray')
# ax.imshow(img)
ax.set_ylim(0, 100)
ax.set_xlim(0, 100)
for center in centers:
    ax.add_patch(plt.Circle(center, 5))
ax.set_aspect("equal", adjustable="datalim")
# ax.autoscale()

plt.show()

# fig6, ax = plt.subplots()
#
# ax.add_patch(plt.Circle((5, 3), 1))
# ax.set_aspect("equal", adjustable="datalim")
# ax.autoscale()
#
# plt.show()
