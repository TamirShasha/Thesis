import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import create_random_k_tuple_sum_to_n, add_pulses, add_gaus_noise
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator

np.random.seed(500)


def arange_data(n, d, p, k, std):
    signal = np.full(d, p)
    y = np.zeros(n)
    signal_mask = create_random_k_tuple_sum_to_n(n - d * k, k + 1)
    y = add_pulses(y, signal_mask, signal)
    y = add_gaus_noise(y, 0, std)
    return y


def experiment():
    d, p, noise_std = 10, 1, 2.5
    fraction = 1 / 4
    l = 5000
    N = 1000
    k = int(fraction * l / d)
    y = arange_data(l, d, p, k, 0)

    true_signal_power = k * p * d
    n_options = np.arange(10, N + 1, 10)
    expectation_errs = np.zeros(len(n_options))
    power_errs = np.zeros(len(n_options))
    m = 10
    for j in range(m):
        print(f'j={j}')
        for i, n in enumerate(n_options):

            total_expectation = 0
            total_power_squared = 0
            for _ in np.arange(n):
                noisy_y = add_gaus_noise(y, 0, noise_std)
                total_expectation += estimate_signal_power(noisy_y, noise_std, 0, SignalPowerEstimator.FirstMoment)
                total_power_squared += estimate_signal_power(noisy_y, noise_std, 0, SignalPowerEstimator.SecondMoment)

            expectation_estimation = total_expectation / n
            expectation_errs[i] += (np.abs(true_signal_power - expectation_estimation))

            power_estimation = total_power_squared / n
            power_errs[i] += (np.abs(true_signal_power - power_estimation))

    expectation_errs /= m
    power_errs /= m

    plt.figure()
    plt.plot(np.log(n_options), np.log(expectation_errs))
    plt.plot(np.log(n_options), np.log(power_errs))
    plt.show()


def __main__():
    experiment()


if __name__ == '__main__':
    __main__()
