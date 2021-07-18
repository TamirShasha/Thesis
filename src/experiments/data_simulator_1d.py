import numpy as np

from src.algorithms.utils import random_1d_ws_positions


def add_pulses(y, signal, positions):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    d = signal.shape[0]
    start_idx, end_idx = 0, 0
    for pos in positions[:-1]:
        start_idx += pos
        end_idx = start_idx + d
        new_y[start_idx:end_idx] = signal
        start_idx = end_idx
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape)
    return y + noise


def simulate_data(n, d, p, k, noise_std):
    signal = np.full(d, p)
    y = np.zeros(n)
    positions = random_1d_ws_positions(n, k, d)
    pulses = add_pulses(y, signal, positions)
    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses
