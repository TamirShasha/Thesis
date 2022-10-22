import numpy as np

from src.algorithms.utils import random_1d_ws_positions
from src.utils.logger import logger


def add_pulses(y, signal, positions):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    d = signal.shape[0]
    pivot = 0
    for pos in positions[:-1]:
        new_y[pos + pivot:pos + pivot + d] = signal
        pivot += pos + d
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape)
    return y + noise


def simulate_data(n, d, p, k, noise_std, signal=None):
    if signal is None:
        signal = np.full(d, p)
    y = np.zeros(n)
    positions = np.array(random_1d_ws_positions(n, k, d), dtype=int)
    pulses = add_pulses(y, signal, positions)
    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses


class DataSimulator1D:
    def __init__(self,
                 size=2000,
                 signal_fraction=1 / 6,
                 signal_size=10,
                 signal_margin=0.02,
                 signal_gen=lambda d: np.ones(d),
                 noise_std=3,
                 noise_mean=0,
                 num_of_instances=None,
                 method='BF'):
        self.size = size
        self.signal_fraction = signal_fraction
        self.signal_margin = signal_margin
        self.signal_gen = signal_gen
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.num_of_instances = num_of_instances
        self.method = method
        self.signal_size = signal_size

        self.signal_padded_size = self.signal_size
        self.clean_data = np.zeros(self.size)

        if self.num_of_instances is None:
            self.num_of_instances = int(self.signal_fraction * self.size / self.signal_size)

        self.unmarginized_signal_gen = self.signal_gen
        if self.signal_margin:
            margin = int(self.signal_margin * self.size)
            self.signal_gen = lambda d: np.pad(signal_gen(d), (margin, margin), 'constant', constant_values=(0, 0))
            self.signal_padded_size = self.signal_size + 2 * margin

    @staticmethod
    def _paste_signals(data, signal, positions):
        """
        Add to y the signal at positions signal_masl == 1
        """
        new_y = np.copy(data)
        d = signal.shape[0]
        for pos in positions:
            new_y[pos:pos + d] = signal
        return new_y

    def simulate(self):
        positions = np.array(random_1d_ws_positions(self.size, self.num_of_instances, self.signal_padded_size),
                             dtype=int)

        signal_instance = self.signal_gen(self.signal_size)
        self.clean_data = self._paste_signals(self.clean_data, signal_instance, positions)
        noise = np.random.normal(self.noise_mean, self.noise_std, self.size)
        noisy_data = self.clean_data + noise

        return noisy_data
