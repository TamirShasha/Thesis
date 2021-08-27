import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.filter_estimator_1d import FilterEstimator1D, _create_chebyshev_basis


def shift_signal(signal, shift=None, dist=None):
    """
    Circular Shifts given signal upon given dist
    """
    n = len(signal)
    if shift is None:
        if dist is None:
            shift = np.random.randint(n)
        else:
            shift = np.random.choice(np.arange(n), p=dist)
    shifted_signal = np.roll(signal, shift)
    return shifted_signal


def relative_error(estimated_signal, true_signal):
    """
    Calculate the relative error between estimated signal and true signal up to circular shift
    :return: relative error
    """
    n = len(true_signal)
    corr = [np.linalg.norm(true_signal - np.roll(estimated_signal, i)) for i in range(n)]
    shift = np.argmin(corr)
    error = np.min(corr) / np.linalg.norm(true_signal)
    return error, shift


def generate_shift_dist(s, L):
    """
    Generates experiment distribution of length L
    :param s: regulation param
    :param L: length
    """
    shift_dist = np.array([np.exp(-np.square(t / s)) for t in np.arange(1, L + 1)])
    shift_dist /= np.sum(shift_dist)
    return shift_dist


def _add_gaussian_noise(signal, sigma):
    """
    :param signal: Clean signal
    :param sigma: Noise STD
    :return: Noisy signal
    """
    noise = np.random.normal(0, sigma, len(signal))
    return signal + noise


def create_mra_data_circularly(signal, window_size, num_of_samples, sigma, shift_dist=None):
    full_signal = np.zeros(window_size)
    full_signal[:signal.shape[0]] = signal
    signals = []
    for i in range(num_of_samples):
        shifted_signal = shift_signal(full_signal, dist=shift_dist)
        noisy_signal = _add_gaussian_noise(shifted_signal, sigma)
        signals.append(noisy_signal)

    return np.array(signals)


def create_mra_data(signal, window_size, num_of_samples, sigma):
    signals = []
    signal_length = signal.shape[0]
    for i in range(num_of_samples):
        start_point = np.random.randint(0, window_size - signal_length)
        shifted_signal = np.zeros(window_size)
        shifted_signal[start_point:start_point + signal_length] = signal
        noisy_signal = _add_gaussian_noise(shifted_signal, sigma)
        signals.append(noisy_signal)

    return np.array(signals)


def classic_signal():
    signal = np.zeros(100)
    signal[:50] = 0.35
    signal[50:] = -0.35
    return signal


def circle_signal():
    L = 100
    xs = np.linspace(-1, 1, L)
    signal = 1 - np.square(xs)
    return signal


def pulse_signal():
    L = 100
    signal = np.ones(L)
    return signal


def experiment():
    signal = circle_signal()
    N = 100
    L = 200
    noise_std = 1
    mra_data = create_mra_data(signal, L, N, noise_std)
    # mra_data = create_mra_data_circularly(signal, L, N, noise_std)

    augmented_data = np.concatenate([mra_data, mra_data], axis=1)

    filter_basis = _create_chebyshev_basis(signal.shape[0], 30)
    filter_estimator = FilterEstimator1D(augmented_data, filter_basis, 2, noise_std)
    l, ps = filter_estimator.estimate()
    signal_est = filter_basis.T.dot(ps)

    plt.title(f'error:{np.linalg.norm(signal - signal_est)}')
    plt.plot(signal)
    plt.plot(signal_est)
    plt.show()


if __name__ == '__main__':
    experiment()
