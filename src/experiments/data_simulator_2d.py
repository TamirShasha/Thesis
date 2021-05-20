import numpy as np
from skimage.draw import ellipse, disk


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


# def simulate_data(shape, signal_generator, signal_shape, occurrences, noise_std, noise_mean, collision_threshold=100):
#     data = np.zeros(shape)
#
#     # add signals
#     for _ in range(occurrences):
#         signal = signal_generator()
#
#         for t in range(collision_threshold):  # trying to find clean location for new signal, max of threshold times
#             row = np.random.randint(shape[0] - signal_shape[0])
#             column = np.random.randint(shape[1] - signal_shape[1])
#             if np.all(data[row:row + signal_shape[0], column:column + signal_shape[1]] == 0):
#                 data[row:row + signal_shape[0], column:column + signal_shape[1]] += signal
#                 break
#
#     # add noise
#     noise = np.random.normal(noise_mean, noise_std, data.shape)
#
#     return data + noise


class DataSimulator2D:
    def __init__(self, rows=2000, columns=2000,
                 signal_length=100, signal_power=1, signal_fraction=1 / 6, signal_gen=lambda d, p: Shapes2D.disk(d, p),
                 noise_std=3, noise_mean=0,
                 collision_threshold=100):
        self.rows = rows
        self.columns = columns
        self.signal_fraction = signal_fraction
        self.signal_gen = signal_gen
        self.signal_length = signal_length
        self.signal_power = signal_power
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.collision_threshold = collision_threshold

        self.signal_area = np.count_nonzero(self.signal_gen(signal_length, signal_power))
        self.signal_shape = (signal_length, signal_length)
        self.occurrences = int(self.signal_fraction * self.rows * self.columns / self.signal_area)

    def simulate(self):
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
                print(f'Failed to simulate dataset with {self.occurrences} instances. '
                      f'Reduced to {o + 1}')
                self.occurrences = o + 1
                break

        # add noise
        noise = np.random.normal(self.noise_mean, self.noise_std, data.shape)

        return data + noise
