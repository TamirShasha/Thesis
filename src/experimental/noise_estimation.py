import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D

simulator = DataSimulator2D(rows=1000,
                            columns=1000,
                            signal_length=80,
                            signal_power=1,
                            signal_fraction=1 / 6,
                            signal_gen=Shapes2D.disk,
                            noise_std=10,
                            noise_mean=0,
                            apply_ctf=False)

data = simulator.simulate()

# plt.imshow(data, cmap='gray')
# plt.show()

patch_size = 10
rows, cols = data.shape

win_mean = np.array(ndimage.uniform_filter(data, (patch_size, patch_size)))
win_sqr_mean = np.array(ndimage.uniform_filter(np.square(data), (patch_size, patch_size)))
win_std = np.sqrt(win_sqr_mean - np.square(win_mean))

# plt.imshow(win_mean)
# plt.colorbar()
# plt.show()
print(np.nanmean(data))
plt.hist(win_mean.reshape(-1), bins=50)
plt.show()


print(np.nanstd(data))
# plt.imshow(win_std)
# plt.colorbar()
# plt.show()

