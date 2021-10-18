import os
import pathlib
from enum import Enum
import numpy as np

from src.constants import ROOT_DIR
from src.utils.mrc import read_mrc, mat_to_npy
from src.algorithms.utils import cryo_downsample


class NoiseNormalizationMethod(Enum):
    NoNormalization = -1
    Simple = 0
    Whitening = 1


class Micrograph:
    def __init__(self,
                 file_path,
                 downsample=None,
                 noise_normalization_method=NoiseNormalizationMethod.Simple,
                 noise_std=1,
                 noise_mean=0,
                 load_micrograph=False):
        self.file_path = file_path
        self.name = os.path.basename(self.file_path)
        self.downsample = downsample
        self.noise_normalization_method = noise_normalization_method
        self.noise_std = noise_std
        self.noise_mean = noise_mean

        if load_micrograph:
            self.img = self.load_micrograph()

    def get_micrograph(self):
        if self.img is None:
            self.img = self.load_micrograph()

        return self.img

    def load_micrograph(self):

        file_extension = pathlib.Path(self.file_path).suffix
        if file_extension == '.mat':
            mrc = mat_to_npy(self.file_path)
        elif file_extension == '.mrc':
            mrc = read_mrc(self.file_path)
        else:
            raise Exception('Unsupported File Extension!')

        mrc = mrc[:min(mrc.shape), :min(mrc.shape)]

        mrc = self.normalize_noise(mrc)

        if self.downsample:
            mrc = cryo_downsample(mrc, (self.downsample, self.downsample))
            downsample_factor = (mrc.shape[0] / self.downsample)
            self.noise_std = self.noise_std / downsample_factor

        return mrc

    def normalize_noise(self, mrc):
        if self.noise_normalization_method == NoiseNormalizationMethod.NoNormalization:
            mrc -= self.noise_mean
            mrc /= self.noise_std
        if self.noise_normalization_method == NoiseNormalizationMethod.Simple:
            mrc -= np.nanmean(mrc)
            mrc /= np.nanstd(mrc)
        else:
            raise Exception('Whitening is unsupported at the moment')
        return mrc


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)

# MICROGRAPHS = {
#     'simple_0': Micrograph(name='clean_one.mrc', size=301, occurrences=80, noise_std=0, noise_mean=0),
#     'simple_3': Micrograph(name='clean_one_std_3.mrc', size=301, occurrences=80, noise_std=3, noise_mean=0),
#     'simple_5': Micrograph(name='clean_one_std_5.mrc', size=301, occurrences=80, noise_std=5, noise_mean=0),
#     '002': Micrograph(name='002.mrc', size=350, occurrences=80, noise_std=1, noise_mean=0),
#     'whitened002': Micrograph(name='002white.mrc', size=350, occurrences=80, noise_std=1, noise_mean=0),
#     'whitened002_x10': Micrograph(name='002white_x10.mrc', size=350, occurrences=80, noise_std=10, noise_mean=0),
#     '002_whitened': Micrograph(name='002_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
#                                noise_mean=0),
#     '001_whitened_normalized': Micrograph(name='001_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
#                                           noise_mean=0),
#     '006_whitened_normalized': Micrograph(name='006_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
#                                           noise_mean=0),
#     '001_normalized': Micrograph(name='001_normalized.mrc', size=350, occurrences=80),
#     '002_normalized': Micrograph(name='002_normalized.mrc', size=350, occurrences=80),
#     '006_normalized': Micrograph(name='006_normalized.mrc', size=350, occurrences=80),
#     '001_raw': Micrograph(name='001_raw.mrc', size=350),
#     '002_raw': Micrograph(name='002_raw.mrc', size=350),
#     '006_raw': Micrograph(name='006_raw.mrc', size=350),
#     '001_automatic_normalized': Micrograph(name='001_automatic_normalized.mrc', size=350),
#     '002_automatic_normalized': Micrograph(name='002_automatic_normalized.mrc', size=350),
#     '006_automatic_normalized': Micrograph(name='006_automatic_normalized.mrc', size=350),
#     '_002': Micrograph(name='002.mrc', size=350),
#     'EMD-2984_0010': Micrograph(name='EMD-2984_0010.mrc', size=350),
#     'EMD-2984_0010_500': Micrograph(name='EMD-2984_0010_500.mrc', size=350),
#     'EMD-2984_0010_1000': Micrograph(name='EMD-2984_0010_1000.mrc', size=350),
#
# }
