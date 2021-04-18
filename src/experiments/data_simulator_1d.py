import numpy as np

from src.algorithms.utils import create_random_k_tuple_sum_to_n


def add_pulses(y, signal_mask, signal):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    x_len = signal.shape[0]
    s_cum = np.cumsum(signal_mask)
    for i in np.arange(s_cum.shape[0] - 1):
        start = s_cum[i] + x_len * i
        new_y[start:start + x_len] = signal
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape[0])
    return y + noise


def simulate_data(n, d, p, k, noise_std):
    signal = np.full(d, p)
    y = np.zeros(n)
    signal_mask = create_random_k_tuple_sum_to_n(n - d * k, k + 1)
    pulses = add_pulses(y, signal_mask, signal)
    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses