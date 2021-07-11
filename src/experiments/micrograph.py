import os

from src.constants import ROOT_DIR
from src.utils.mrc import read_mrc


class Micrograph:
    def __init__(self, name, size, occurrences=0, noise_std=1, noise_mean=0):
        self.name = name
        self.signal_length = size
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.occurrences = occurrences

    def load_mrc(self):
        return read_mrc(_get_path(self.name))


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)


MICROGRAPHS = {
    'simple_0': Micrograph(name='clean_one.mrc', size=301, occurrences=80, noise_std=0, noise_mean=0),
    'simple_3': Micrograph(name='clean_one_std_3.mrc', size=301, occurrences=80, noise_std=3, noise_mean=0),
    'simple_5': Micrograph(name='clean_one_std_5.mrc', size=301, occurrences=80, noise_std=5, noise_mean=0),
    '002': Micrograph(name='002.mrc', size=350, occurrences=80, noise_std=1, noise_mean=0),
    'whitened002': Micrograph(name='002white.mrc', size=350, occurrences=80, noise_std=1, noise_mean=0),
    'whitened002_x10': Micrograph(name='002white_x10.mrc', size=350, occurrences=80, noise_std=10, noise_mean=0),
    '002_whitened': Micrograph(name='002_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
                               noise_mean=0),
    '001_whitened_normalized': Micrograph(name='001_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
                                          noise_mean=0),
    '006_whitened_normalized': Micrograph(name='006_whitened_normalized.mrc', size=350, occurrences=80, noise_std=1,
                                          noise_mean=0),
    '001_normalized': Micrograph(name='001_normalized.mrc', size=350, occurrences=80),
    '002_normalized': Micrograph(name='002_normalized.mrc', size=350, occurrences=80),
    '006_normalized': Micrograph(name='006_normalized.mrc', size=350, occurrences=80),
    '001_raw': Micrograph(name='001_raw.mrc', size=350),
    '002_raw': Micrograph(name='002_raw.mrc', size=350),
    '006_raw': Micrograph(name='006_raw.mrc', size=350),
    '001_automatic_normalized': Micrograph(name='001_automatic_normalized.mrc', size=350),
    '002_automatic_normalized': Micrograph(name='002_automatic_normalized.mrc', size=350),
    '006_automatic_normalized': Micrograph(name='006_automatic_normalized.mrc', size=350),
    '_002': Micrograph(name='002.mrc', size=350),
    'EMD-2984_0010': Micrograph(name='EMD-2984_0010.mrc', size=350),
    'EMD-2984_0010_500': Micrograph(name='EMD-2984_0010_500.mrc', size=350),
    'EMD-2984_0010_1000': Micrograph(name='EMD-2984_0010_1000.mrc', size=350),

}
