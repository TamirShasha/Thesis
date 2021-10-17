import mrcfile
import os
import numpy as np
from scipy.io import loadmat, savemat


def mat_to_npy(file_name):
    if '.mat' not in file_name:
        file_name += '.mat'
    full_mat = loadmat(file_name)
    key = None
    for k in full_mat:
        if '__' not in k:
            key = k
    return full_mat[key]


def read_file(file_path):
    file_name, file_extention = os.path.splitext(file_path)
    return


def write_mrc(file_path, x):
    # For now it is transposed, when moving to C aligned this should be removed
    with mrcfile.new(file_path, overwrite=True) as mrc_fh:
        mrc_fh.set_data(x.astype('float32').T)
    return


def read_mrc(file_path):
    mrc = np.ascontiguousarray(mrcfile.open(file_path).data.T)
    # mrc /= np.max(mrc)
    return mrc


# import matplotlib.pyplot as plt
# from src.algorithms.utils import cryo_downsample

# file_path = '../../../data/001.mrc'
# x = read_mrc(file_path)
# x = mat_to_npy(file_path)
# x = cryo_downsample(x, (500, 500))
# write_mrc('../../../data/EMD-2984_0010.mrc', x)
# x -= np.min(x)
# x /= np.max(x)
# plt.imshow(x, cmap='gray', vmin=0, vmax=0.1)
# plt.show()
