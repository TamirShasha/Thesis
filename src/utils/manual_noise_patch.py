import numpy as np
from src.constants import ROOT_DIR
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

# Open the image form working directory

data_dir = os.path.join(ROOT_DIR, os.pardir, 'datasets_locs')

# FoilHole_17935664_Data_17936289_17936290_20170418_1651-12280-frames.mrc
# w, h = 520, 569
# true_w, true_h = 3837, 3709
#
# xs = np.array([[1, 160],
#                [1, 330],
#                [180, 507],
#                [333, 462],
#                [475, 288],
#                [58, 229]], dtype=int)

'''
manual_noise_positions(:, 1) = [1042, 7];
manual_noise_positions(:, 2) = [2151, 7];
manual_noise_positions(:, 3) = [3304, 1328];
manual_noise_positions(:, 4) = [3011, 2457];
manual_noise_positions(:, 5) = [1877, 3304];
manual_noise_positions(:, 6) = [1492, 427];
'''

# /data/yoelsh/datasets/10081/micrographs/HCN1apo_0015_2xaligned.mrc
# img_name = 'HCN1apo_0015_2xaligned.png'
# true_w, true_h = 3710, 3838

'''
manual_noise_positions(:, 1) = [41, 62];
manual_noise_positions(:, 2) = [809, 399];
manual_noise_positions(:, 3) = [2861, 1075];
manual_noise_positions(:, 4) = [1374, 1717];
manual_noise_positions(:, 5) = [139, 2261];
manual_noise_positions(:, 6) = [907, 2799];
'''

# img_name = 'FoilHole_17935665_Data_17936289_17936290_20170418_1654-12282-frames.png'
# true_w, true_h = 3710, 3838

img_name = 'FoilHole_17935666_Data_17936289_17936290_20170418_1656-12284-frames.png'
true_w, true_h = 3710, 3838

image = np.array(Image.open(os.path.join(data_dir, img_name)))
x, y = np.where(np.all(image == [237, 28, 36], axis=2))
xs = np.array(list(zip(x, y)))
w, h, c = image.shape
mrc_xs = np.array(xs / [w, h] * [true_w, true_h], dtype=int)
for i, x in enumerate(mrc_xs):
    print(f'manual_noise_positions(:, {i + 1}) = [{x[0]}, {x[1]}];')
