import pathlib
from enum import Enum
import numpy as np
import mrcfile
import os
from scipy.io import loadmat
from scipy.stats import iqr

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
                 downsample=1000,
                 noise_normalization_method=NoiseNormalizationMethod.Simple,
                 noise_std=1,
                 noise_mean=0,
                 clip_outliers=True,
                 load_micrograph=False):
        self.file_path = file_path
        self.name = os.path.basename(self.file_path)
        self.downsample = downsample
        self.noise_normalization_method = noise_normalization_method
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.clip_outliers = clip_outliers
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

        mrc = mrc[:min(mrc.shape), :min(mrc.shape)]

        if self.clip_outliers:
            mrc = Micrograph.clip_outliers(mrc)

        mrc = self.normalize_noise(mrc)

        if self.downsample > 0:
            logger.info(f'Downsample to size ({self.downsample, self.downsample})')
            original_size = mrc.shape[0]
            mrc = cryo_downsample(mrc, (self.downsample, self.downsample))
            downsample_factor = (original_size / self.downsample)
            self.noise_std = self.noise_std / downsample_factor

        return mrc

    def normalize_noise(self, mrc):
        logger.info('Normalizing noise ..')
        if self.noise_normalization_method == NoiseNormalizationMethod.NoNormalization:
            mrc -= self.noise_mean
            mrc /= self.noise_std
        elif self.noise_normalization_method == NoiseNormalizationMethod.Simple:
            mrc -= np.nanmean(mrc)
            mrc /= np.nanstd(mrc)
        else:
            raise Exception('Whitening is unsupported at the moment')

        logger.info(f'MRC mean/std is {np.nanmean(mrc)}/{np.nanstd(mrc)}')
        return mrc

    @staticmethod
    def clip_outliers(mrc: np.ndarray):
        low, high = 25, 75
        mrc_iqr = iqr(mrc, rng=(low, high))
        low_ths = np.percentile(mrc, low) - mrc_iqr
        high_ths = np.percentile(mrc, high) + mrc_iqr
        total_clipped_low_perc = np.round((np.nansum(mrc < low_ths) / np.prod(mrc.shape)) * 100, 3)
        total_clipped_high_perc = np.round((np.nansum(mrc > high_ths) / np.prod(mrc.shape)) * 100, 3)
        clipped_mrc = mrc.clip(low_ths, high_ths)

        logger.info(f'Clipping outliers by IQR method, using ranges ({low}, {high}), '
                    f'total of ({total_clipped_low_perc}%, {total_clipped_high_perc}%) pixels where clipped')
        return clipped_mrc


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)
