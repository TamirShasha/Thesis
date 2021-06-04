import os

from src.constants import ROOT_DIR
from src.utils.mrc import read_mrc


class Micrograph:
    def __init__(self, name, length, occurrences, noise_std, noise_mean):
        self.name = name
        self.signal_length = length
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.occurrences = occurrences

    def load_mrc(self):
        return read_mrc(_get_path(self.name))


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)


MICROGRAPHS = {
    'simple_0': Micrograph(name='clean_one.mrc', length=301, occurrences=80, noise_std=0, noise_mean=0),
    'simple_3': Micrograph(name='clean_one_std_3.mrc', length=301, occurrences=80, noise_std=3, noise_mean=0),
    'simple_5': Micrograph(name='clean_one_std_5.mrc', length=301, occurrences=80, noise_std=5, noise_mean=0),
    'whitened002': Micrograph(name='002white.mrc', length=350, occurrences=80, noise_std=1, noise_mean=0)
}
