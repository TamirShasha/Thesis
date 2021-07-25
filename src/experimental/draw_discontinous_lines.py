from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
import matplotlib.pyplot as plt
import numpy as np
# from skimage.draw import line
import cv2

sim_data = DataSimulator2D(rows=1000,
                           columns=1000,
                           # signal_length=PARTICLE_200.particle_length,
                           signal_length=10,
                           signal_power=1,
                           signal_fraction=1 / 6,
                           # signal_gen=PARTICLE_200.get_signal_gen(),
                           signal_gen=Shapes2D.sphere,
                           contamination=False,
                           # signal_gen=sig_gen,
                           noise_std=3,
                           noise_mean=0,
                           apply_ctf=False)

img = sim_data.simulate()

np.random.seed(503)
width = 31
rows, columns = img.shape
width_buffer = (width - 1) // 2
n = img.shape[0]

# i = 0
# curves = []
# while i < 1:
#     curve = []
#
#     r0, c0 = np.random.randint(width_buffer, n - width_buffer, size=2)
#     while len(curve) < n:
#         r1, c1 = np.random.randint(width_buffer, n - width_buffer, size=2)
#
#         width_curve = []
#         for j in range(width):
#             _r0, _c0, _r1, _c1 = np.array([r0, c0, r1, c1]) - width_buffer + j
#             rr, cc = line(_r0, _c0, _r1, _c1)
#             img[rr, cc] = 10
#             width_curve.append(img[rr, cc])
#
#         width_curve = np.nanmean(width_curve, axis=0)
#         if len(curve) + len(width_curve) > n:
#             width_curve = width_curve[:n - len(curve)]
#
#         curve = np.concatenate([curve, width_curve])
#         r0, c0 = r1, c1
#
#     curves.append(curve)
#     i += 1

# x = [0, 0]
# y = [500, 500]
# x_coor = np.arange(1, 500)
# y_coor = np.interp(x_coor, x, y)
# coordinates = np.column_stack((x_coor, y_coor))
# print(coordinates)

bw = np.zeros((n, n), np.uint8)
cv2.line(bw, (50, 50), (500, 500), 255, 50, lineType=cv2.LINE_AA)
plt.imshow(bw, cmap='gray')
plt.show()
