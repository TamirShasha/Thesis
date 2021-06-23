import os
import numpy as np

from src.constants import ROOT_DIR
from src.utils.mrc import mat_to_npy
from src.algorithms.utils import crop


def _get_path(mrc_name):
    return os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)


class _ParticleProjectionsDatasets:

    def __init__(self, file_name: str, particle_length: int,
                 preprocess=('average', 'positive', 'normalize', 'crop')):
        self._file_name = file_name
        self._pre_process = preprocess
        self.particle_length = particle_length

    @staticmethod
    def average(signal):
        avg_signal = np.copy(signal)
        temp = signal
        for i in range(3):
            temp = np.rot90(temp)
            avg_signal += temp
        return avg_signal / 4

    @staticmethod
    def positive(signal):
        return signal * (signal > 0)

    @staticmethod
    def normalize(signal, power):
        return power * signal / np.nansum(signal) * np.nansum(signal > 0)

    def crop(self, signal):
        return crop(signal, (self.particle_length, self.particle_length))

    def get_signal_gen(self):
        file_path = _get_path(self._file_name)
        signals = np.load(file_path)
        h, w, num_of_projections = signals.shape

        def signal_gen(l, p):
            idx = np.random.randint(num_of_projections)
            signal = signals[:, :, idx]

            if 'crop' in self._pre_process:
                signal = crop(signal, (self.particle_length, self.particle_length))
            if 'average' in self._pre_process:
                signal = _ParticleProjectionsDatasets.average(signal)
            if 'positive' in self._pre_process:
                signal = _ParticleProjectionsDatasets.positive(signal)
            if 'normalize' in self._pre_process:
                signal = _ParticleProjectionsDatasets.normalize(signal, p)

            return signal

        return signal_gen


PARTICLE_250 = _ParticleProjectionsDatasets('projections.npy', 200, preprocess=('crop', 'normalize'))
