import pathlib
from enum import Enum
import numpy as np
import mrcfile
import os

from PIL import Image
from skimage.filters import gaussian
from numpy.core.multiarray import ndarray
from scipy.io import loadmat
from scipy.stats import iqr
import matplotlib.pyplot as plt
from skimage.io import imread
import scipy.fftpack as fp
from scipy.ndimage import fourier_gaussian

from src.utils.logger import logger
from src.constants import ROOT_DIR
from src.algorithms.utils import cryo_downsample


def mat_to_npy(file_name):
    if '.mat' not in file_name:
        file_name += '.mat'
    full_mat = loadmat(file_name)
    key = None
    for k in full_mat:
        if '__' not in k:
            key = k
    return full_mat[key]


def write_mrc(file_path, x):
    # For now it is transposed, when moving to C aligned this should be removed
    with mrcfile.new(file_path, overwrite=True) as mrc_fh:
        mrc_fh.set_data(x.astype('float32').T)
    return


def read_mrc(file_path):
    mrc = np.ascontiguousarray(mrcfile.open(file_path).data.T)
    # mrc /= np.max(mrc)
    return mrc


class NoiseNormalizationMethod(Enum):
    NoNormalization = -1
    Simple = 0
    Whitening = 1


class Micrograph:
    def __init__(self,
                 file_path,
                 downsample=-1,
                 noise_std=None,
                 noise_mean=None,
                 clip_outliers=False,
                 load_micrograph=False,
                 low_pass_filter=0,
                 plot=False):
        self.file_path = file_path
        self.name = os.path.basename(self.file_path)
        self.downsample = downsample
        self.noise_std_param = noise_std
        self.noise_mean_param = noise_mean
        self.clip_outliers = clip_outliers
        self.low_pass_filter = low_pass_filter
        self.plot = plot

        self.noise_mean = None
        self.noise_std = None
        self.img = None

        if load_micrograph:
            self.img = self.load_micrograph()

    def get_micrograph(self):
        if self.img is None:
            self.img = self.load_micrograph()

        return self.img

    def load_micrograph(self):

        logger.info(f'Loading micrograph {self.name} ...')

        file_extension = pathlib.Path(self.file_path).suffix
        if file_extension == '.npy':
            mrc = np.load(self.file_path)
        elif file_extension == '.mat':
            mrc = mat_to_npy(self.file_path)
        elif file_extension == '.mrc':
            mrc = read_mrc(self.file_path)
        else:
            raise Exception('Unsupported File Extension!')

        logger.info(f'Loaded micrograph size is {mrc.shape}')

        # Flipping for positive signal power, cutting for square dimension
        # plt.imshow(mrc, cmap='gray')
        # plt.show()
        mrc = mrc[:min(mrc.shape), :min(mrc.shape)]
        mrc = -mrc

        if self.clip_outliers:
            mrc = Micrograph.clip_outliers(mrc)

        sigma = 100
        mrc_blurred = gaussian(mrc, sigma=sigma, mode='nearest', truncate=2.0)

        mrc = mrc - mrc_blurred
        mrc = mrc[2 * sigma: -2 * sigma, 2 * sigma: -2 * sigma]

        mrc = self.normalize_noise(mrc)

        sigma = mrc.shape[0] // 300
        logger.info(f'Applying gaussian averaging with sigma = {sigma}')
        mrc = gaussian(mrc, sigma=sigma, mode='nearest', truncate=2.0)

        if 0 < self.downsample < mrc.shape[0]:
            logger.info(f'Downsample to size ({self.downsample}, {self.downsample}) from {mrc.shape}')
            original_size = mrc.shape[0]
            mrc = cryo_downsample(mrc, (self.downsample, self.downsample))
            downsample_factor = (original_size / self.downsample)
            self.noise_std = self.noise_std / downsample_factor

        logger.info(f'Loaded Micrograph of size {mrc.shape}')

        return mrc

    def normalize_noise(self, mrc):

        # sigma = 300
        # blurred = np.fft.ifft2(fourier_gaussian(np.fft.fft2(mrc), sigma=sigma)).real
        # mrc -= blurred
        # mrc = mrc[sigma:-sigma, sigma:-sigma]
        #
        # blurred_2 = np.fft.ifft2(fourier_gaussian(np.fft.fft2(mrc), sigma=sigma)).real
        #
        # fig, axs = plt.subplots(1, 3, figsize=(16, 8))
        # axs[0].imshow(mrc, cmap='gray')
        # axs[1].imshow(blurred, cmap='gray')
        # axs[2].imshow(blurred_2, cmap='gray')
        # plt.show()

        logger.info('Normalizing noise ..')
        old_mean, old_std = np.nanmean(mrc), np.nanstd(mrc)

        noise_mean = self.noise_mean_param if self.noise_mean_param is not None else np.nanmean(mrc)
        noise_std = self.noise_std_param if self.noise_std_param is not None else np.nanstd(mrc)

        mrc -= noise_mean
        mrc /= noise_std

        self.noise_mean = 0
        self.noise_std = 1

        logger.info(f'Normalized MRC mean/std to (0,1) from ({format(old_mean, ".3f")}/{format(old_std, ".3f")})')
        return mrc

    @staticmethod
    def clip_outliers(mrc: np.ndarray):
        low, high, c = 25, 75, 2
        mrc_iqr = iqr(mrc, rng=(low, high))
        low_ths = np.percentile(mrc, low) - c * mrc_iqr
        high_ths = np.percentile(mrc, high) + c * mrc_iqr

        # import matplotlib.pyplot as plt
        # plt.title('Histogram of gray values in HCN1apo_0002_2xaligned.mrc')
        # plt.hist(mrc.flatten(), bins=100)
        # plt.axvline(x=low_ths, color='r', linestyle='-')
        # plt.axvline(x=high_ths, color='r', linestyle='-')
        # plt.show()

        total_clipped_low_perc = np.round((np.nansum(mrc < low_ths) / np.nanprod(mrc.shape)) * 100, 3)
        total_clipped_high_perc = np.round((np.nansum(mrc > high_ths) / np.nanprod(mrc.shape)) * 100, 3)
        clipped_mrc = mrc.clip(low_ths, high_ths)

        logger.info(f'Clipping outliers by IQR method, using ranges ({low}, {high}), '
                    f'total of ({total_clipped_low_perc}%, {total_clipped_high_perc}%) pixels where clipped')
        return clipped_mrc


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)


def apply_low_pass_filter(img, n_pass=5, plot=False):
    F1 = fp.fft2((img.astype(float)))
    F2 = fp.fftshift(F1)

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow((20 * np.log10(0.1 + F2)).astype(int), cmap='gray')
        plt.show()

    (w, h) = img.shape
    half_w, half_h = int(w / 2), int(h / 2)

    # high pass filter
    F2[half_w - n_pass:half_w + n_pass + 1,
    half_h - n_pass:half_h + n_pass + 1] = 0  # select all but the first 50x50 (low) frequencies
    im1 = fp.ifft2(fp.ifftshift(F2)).real

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow((20 * np.log10(0.1 + F2)).astype(int))
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.imshow(im1, cmap='gray')
        plt.axis('off')
        plt.show()

    return im1


if __name__ == '__main__':
    mrc = Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\10028\005.mrc',
                     clip_outliers=True, plot=True)
    # mrc = Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0155_frames_sum.mrc',
    #                  clip_outliers=True, plot=True)
    plt.imshow(mrc.get_micrograph())
    plt.show()

    # Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0155_frames_sum.mrc',
    #            clip_outliers=True)
    #
    # # mrc = Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\HCN1apo_0016_2xaligned.mrc',
    # mrc = Micrograph(
    #     # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10049\stack_0250_2x_SumCorr - Copy.mrc',
    #     file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10061\EMD-2984_0775.mrc',
    #     # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10061\EMD-2984_1249.mrc',
    #     # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0155_frames_sum.mrc',
    #     # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10081\HCN1apo_0035_2xaligned.mrc',
    #     downsample=-1,
    #     clip_outliers=True,
    #     load_micrograph=True)
    #
    # plt.imshow(mrc.img, cmap='gray')
    # plt.show()
    #
    # img_gaus_10 = gaussian(mrc.img, sigma=10, mode='nearest', truncate=2.0)
    # plt.imshow(img_gaus_10, cmap='gray')
    # plt.show()
    # img_gaus_5 = gaussian(mrc.img, sigma=5, mode='nearest', truncate=2.0)
    # plt.imshow(img_gaus_5, cmap='gray')
    # plt.show()
    # img_gaus_3 = gaussian(mrc.img, sigma=3, mode='nearest', truncate=2.0)
    # plt.imshow(img_gaus_3, cmap='gray')
    # plt.show()
    # img__ = gaussian(mrc.img - img_, sigma=100, mode='nearest', truncate=2.0)
    #
    # fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    # axs[0, 0].imshow(mrc.img, cmap='gray')
    # axs[0, 1].imshow(img_, cmap='gray')
    # axs[1, 0].imshow(mrc.img - img_, cmap='gray')
    # axs[1, 1].imshow(img__, cmap='gray')
    # plt.show()

    # img = apply_low_pass_filter(mrc.img, n_pass=10, plot=True)

    # sigma = 300
    # blurred = np.fft.ifft2(fourier_gaussian(np.fft.fft2(img), sigma=sigma)).real
    #
    # plt.imshow(img, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(blurred, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # img = mrc.img
    #
    # sigma = 300
    # blurred = np.fft.ifft2(fourier_gaussian(np.fft.fft2(img), sigma=sigma)).real
    #
    # plt.imshow(img, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(blurred, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(np.fft.ifft2(fourier_gaussian(np.fft.fft2(img - blurred), sigma=sigma)).real, cmap='gray')
    # plt.colorbar()
    # plt.show()

    print()
