import numpy as np

from src.algorithms.utils import random_1d_ws_positions


def add_pulses(y, signal, positions):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    d = signal.shape[0]
    for pos in positions:
        new_y[pos:pos + d] = signal
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
