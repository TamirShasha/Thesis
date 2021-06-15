import numpy as np
from skimage.draw import ellipse, disk
import matplotlib.pyplot as plt

from src.utils.logger import logger
from src.algorithms.utils import dynamic_programming_2d, log_size_S_1d, random_1d_ws_positions
from src.utils.CTF_Relion import apply_CTF, cryo_CTF_Relion


class Shapes2D:

    @staticmethod
    def ellipse(major_diameter, minor_diameter, fill_value, rotation='random'):
        if rotation == 'random':
            rotation = -np.pi + 2 * np.pi * np.random.rand()

        signal = np.zeros(shape=(major_diameter, major_diameter))

        r, c = major_diameter // 2, major_diameter // 2
        r_radius, c_radius = major_diameter // 2, minor_diameter // 2
        rr, cc = ellipse(r, c, r_radius, c_radius, rotation=rotation)
        signal[rr, cc] = fill_value

        return signal

    @staticmethod
    def disk(diameter, fill_value):
        signal = np.zeros(shape=(diameter, diameter))

        r, c = diameter // 2, diameter // 2
        rr, cc = disk((r, c), diameter // 2)
        signal[rr, cc] = fill_value

        return signal

    @staticmethod
    def square(length, fill_value):
        return np.full((length, length), fill_value)

    @staticmethod
    def gaussian(length, sigma):
        x = np.linspace(-1, 1, length)
        y = np.linspace(-1, 1, length)
        X, Y = np.meshgrid(x, y)
        term1 = 1 / (2 * np.pi * np.square(sigma))
        term2 = np.square(X / sigma) + np.square(Y / sigma)
        signal = term1 * np.exp(-term2 / 2)
        # signal2 = np.square(X / sigma) + np.square(Y / sigma)
        # signal2 = np.sqrt(np.max(signal2) - signal2)
        return signal

    @staticmethod
    def double_disk(large_diam, small_diam, fill_value_large, fill_value_small):
        signal = np.zeros(shape=(large_diam, large_diam))

        r, c = large_diam // 2, large_diam // 2
        rr, cc = disk((r, c), large_diam // 2)
        signal[rr, cc] = fill_value_large

        rr, cc = disk((r, c), small_diam // 2)
        signal[rr, cc] = fill_value_small

        return signal

    @staticmethod
    def sphere(length, power):
        signal = np.zeros(shape=(length, length))
        radius = length // 2
        rr, cc = disk((radius, radius), radius)
        signal[rr, cc] = power * np.sqrt(1 - np.square((rr - radius) / radius) - np.square((cc - radius) / radius))

        return signal


class DataSimulator2D:
    def __init__(self, rows=2000, columns=2000, signal_length=100, signal_power=1, signal_fraction=1 / 6,
                 signal_gen=lambda d, p: Shapes2D.disk(d, p), noise_std=3, noise_mean=0, method='BF',
                 collision_threshold=100, apply_ctf=False):
        self.rows = rows
        self.columns = columns
        self.signal_fraction = signal_fraction
        self.signal_gen = signal_gen
        self.signal_length = signal_length
        self.signal_power = signal_power
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.method = method
        self.collision_threshold = collision_threshold
        self.apply_ctf = apply_ctf

        self.signal_area = np.count_nonzero(self.signal_gen(signal_length, signal_power))
        self.signal_shape = (signal_length, signal_length)
        self.mrc_area = self.rows * self.columns
        self.occurrences = int(self.signal_fraction * self.mrc_area / self.signal_area)
        self.snr = self._calc_snr()

        logger.info(f'SNR is {self.snr}db')

    def simulate(self):

        if self.method == 'BF':
            data = self._simulate_signal_bf()
        elif self.method == 'VWS':
            data = self._simulate_signal_vws()
        else:
            raise ValueError('method = {} is not supported, try bf or vws'.format(self.method))
        logger.info(f'Total signal area fraction is {np.count_nonzero(data) / np.prod(data.shape)}\n')

        # add noise
        noise = self._random_gaussian_noise()

        simulated_data = data + noise

        return simulated_data

    def create_signal_instance(self):
        signal = self.signal_gen(self.signal_length, self.signal_power)
        if self.apply_ctf:
            signal = self._apply_ctf_on_signal(signal)
        return signal

    def _simulate_signal_bf(self):
        data = np.zeros((self.rows, self.columns))

        # add signals
        for o in range(self.occurrences):
            signal = self.create_signal_instance()

            for t in range(self.collision_threshold):  # trying to find clean location for new signal, max of threshold
                row = np.random.randint(self.rows - self.signal_shape[0])
                column = np.random.randint(self.columns - self.signal_shape[1])
                if np.all(data[row:row + self.signal_shape[0], column:column + self.signal_shape[1]] == 0):
                    data[row:row + self.signal_shape[0], column:column + self.signal_shape[1]] += signal
                    break

            if t == self.collision_threshold - 1:
                logger.warning(f'Failed to simulate dataset with {self.occurrences} occurrences. '
                               f'Reduced to {o + 1}')
                self.occurrences = o + 1
                break

        return data

    def _simulate_signal_vws(self):
        data = np.zeros((self.rows, self.columns))
        n = self.rows
        d = self.signal_shape[0]
        k = self.occurrences

        # Get number of signals per row
        max_k_in_row = n // d
        log_size_S_per_k = np.zeros((max_k_in_row))
        for k_in_row in range(1, max_k_in_row + 1):
            log_size_S_per_k[k_in_row - 1] = log_size_S_1d(n, k_in_row, d)
        log_size_S_per_k = log_size_S_per_k[::-1].copy()
        constants = np.zeros((n - d + 1, n - d + 1))
        q = dynamic_programming_2d(n, k, d, constants)

        occurrences_left = k
        curr_row = 0
        positions = []
        while occurrences_left > 0:
            curr_options = [q[curr_row + 1, occurrences_left]]
            curr_max_k = min(occurrences_left, max_k_in_row)
            for i in range(1, curr_max_k + 1):
                curr_options.append(q[curr_row + d, occurrences_left - i] + log_size_S_per_k[-i])

            curr_options = np.array(curr_options)
            prob_per_option = np.exp(curr_options - q[curr_row, occurrences_left])
            num_occurences_picked = np.random.choice(np.arange(len(prob_per_option)), p=prob_per_option)
            if num_occurences_picked > 0:
                positions_1d = random_1d_ws_positions(n, num_occurences_picked, d)
                for col in positions_1d:
                    positions.append([curr_row, col])
                occurrences_left -= num_occurences_picked
                curr_row += d
            else:
                curr_row += 1

        positions = np.array(positions, dtype='int')
        switch_rows_and_cols = np.random.choice([0, 1])
        if switch_rows_and_cols:
            positions = positions[:, 1::-1]

        # add signals
        for row, col in positions:
            signal = self.signal_gen(self.signal_length, self.signal_power)
            data[row:row + self.signal_shape[0], col:col + self.signal_shape[1]] += signal

        return data

    def _random_gaussian_noise(self):
        return np.random.normal(self.noise_mean, self.noise_std, (self.rows, self.columns))

    def _apply_ctf_on_signal(self, data):
        pixel_size = 1.3399950228756292
        defocus_u = 2334.4699219
        defocus_v = 2344.5949219
        defocus_angle = 0.6405358529352114
        spherical_aberration = 2.0
        amplitude_contrast = 0.1

        CTF = cryo_CTF_Relion(data.shape[0], pixel_size, defocus_u, defocus_v, defocus_angle, spherical_aberration,
                              amplitude_contrast)

        return -np.real(apply_CTF(data, CTF))

    def _calc_snr(self):
        signal = self.signal_gen(self.signal_length, self.signal_power)
        signal_support = signal != 0
        if self.apply_ctf:
            signal = self._apply_ctf_on_signal(signal)

        avg_signal_power = np.nansum(np.square(signal * signal_support)) / np.nansum(signal_support)
        fraction = self.occurrences * self.signal_area / self.mrc_area
        snr = (avg_signal_power / np.square(self.noise_std)) * fraction

        db = int(10 * np.log10(snr))
        return db
