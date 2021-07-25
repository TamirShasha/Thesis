from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import bezier_curve, disk

sim_data = DataSimulator2D(rows=4000,
                           columns=4000,
                           # signal_length=PARTICLE_200.particle_length,
                           signal_length=200,
                           signal_power=1,
                           signal_fraction=1 / 6,
                           # signal_gen=PARTICLE_200.get_signal_gen(),
                           signal_gen=Shapes2D.sphere,
                           contamination=False,
                           # signal_gen=sig_gen,
                           noise_std=10,
                           noise_mean=0,
                           apply_ctf=False)

img = sim_data.simulate()

# np.random.seed(503)
width = 31
rows, columns = img.shape
width_buffer = (width - 1) // 2

n = img.shape[0]
i = 0

curves = []
while i < 10:

    curve = np.zeros(n)

    r0, c0 = np.random.randint(width_buffer, n - width_buffer, size=2)
    r1, c1 = np.random.randint(width_buffer, n - width_buffer, size=2)
    r2, c2 = np.random.randint(width_buffer, n - width_buffer, size=2)
    weight = 2
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, weight, img.shape)

    if len(rr) < n:
        continue

    rr, cc = rr[len(rr) - n:], cc[len(cc) - n:]

    if np.random.binomial(1, p=0.5):
        cc = n - cc

    for j, (row, column) in enumerate(zip(rr, cc)):
        rr_d, cc_d = disk((row, column), width_buffer)
        curve[j] = np.nanmean(img[rr_d, cc_d])
        # img[rr_d, cc_d] = 10

    curves.append(curve)
    i += 1

plt.imshow(img, cmap='gray')
plt.show()
plt.plot(curves[0])
plt.show()
