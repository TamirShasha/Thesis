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
        rr, cc = ellipse(r, c, r_radius, c_radius, rotation)
        signal[rr, cc] = fill_value

        return signal

    @staticmethod
    def disk(diameter, fill_value):
        signal = np.zeros(shape=(diameter, diameter))

        r, c = diameter // 2, diameter // 2
        rr, cc = disk((r, c), diameter)
        signal[rr, cc] = fill_value

        return signal

    @staticmethod
    def square(length, fill_value):
        return np.full((length, length), fill_value)


def simulate_data(shape, signal_generator, signal_shape, occurrences, noise_std, noise_mean, collision_threshold=100):
    data = np.zeros(shape)

    # add signals
    for _ in range(occurrences):
        signal = signal_generator()

        for t in range(collision_threshold):  # trying to find clean location for new signal, max of threshold times
            row = np.random.randint(shape[0] - signal_shape[0])
            column = np.random.randint(shape[1] - signal_shape[1])
            if np.all(data[row:row + signal_shape[0], column:column + signal_shape[1]] == 0):
                data[row:row + signal_shape[0], column:column + signal_shape[1]] += signal
                break

    # add noise
    noise = np.random.normal(noise_mean, noise_std, data.shape)

    return data + noise
