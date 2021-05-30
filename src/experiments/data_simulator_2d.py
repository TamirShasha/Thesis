import numpy as np
from skimage.draw import ellipse, disk
from src.algorithms.utils import dynamic_programming_2d, log_size_S_1d, random_1d_ws_positions
import logging


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


class DataSimulator2D:
    def __init__(self, rows=2000, columns=2000, signal_length=100, signal_power=1, signal_fraction=1 / 6,
                 signal_gen=lambda d, p: Shapes2D.disk(d, p), noise_std=3, noise_mean=0, method='BF', collision_threshold=100):
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

        self.signal_area = np.count_nonzero(self.signal_gen(signal_length, signal_power))
        self.signal_shape = (signal_length, signal_length)
        self.occurrences = int(self.signal_fraction * self.rows * self.columns / self.signal_area)

    def simulate(self):

        if self.method == 'BF':
            data = self._simulate_signal_bf()
        elif self.method == 'VWS':
            data = self._simulate_signal_vws()
        else:
            raise ValueError('method = {} is not supported, try bf or vws'.format(self.method))
        logging.info(f'Total signal area fraction is {np.count_nonzero(data) / np.prod(data.shape)}\n')

        # add noise
        noise = self._random_gaussian_noise()

        return data + noise

    def _simulate_signal_bf(self):
        data = np.zeros((self.rows, self.columns))

        # add signals
        for o in range(self.occurrences):
            signal = self.signal_gen(self.signal_length, self.signal_power)

            for t in range(self.collision_threshold):  # trying to find clean location for new signal, max of threshold
                row = np.random.randint(self.rows - self.signal_shape[0])
                column = np.random.randint(self.columns - self.signal_shape[1])
                # first insert the signal
                data[row:row + self.signal_shape[0], column:column + self.signal_shape[1]] += signal
                # second, check if collision has occurred, if so, remove the signal, if not, continue to next
                if np.any(data == 2 * self.signal_power):
                    data[row:row + self.signal_shape[0], column:column + self.signal_shape[1]] -= signal
                else:
                    break

            if t == self.collision_threshold - 1:
                logging.warning(f'Failed to simulate dataset with {self.occurrences} occurrences. '
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
