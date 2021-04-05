import numpy as np
import matplotlib.pyplot as plt

from src.utils import create_random_k_tuple_sum_to_n, add_pulses, add_gaus_noise

np.random.seed(500)


# y^2 = (x+e)^2 = x^2 + 2xe + e^2
# x^2 = y^2 - 2xe - e^2
# x^2 = E(y^2) - E(2xe) - E(e^2) = E(y^2) - E(e^2) = E(y^2) - std^2
# z = x^2 = E(y^2) - std^2
# E(y^2) ~ (1/n)(y1^2+ .. + yn^2) = mu
# z = mu - std^2
# x = sqrt(z)


def calc_expected_signal_power_using_moment(y, noise_std):
    y_power = np.sum(np.power(y, 2))
    noise_power = (noise_std ** 2) * y.shape[0]
    all_signal_power_squared = y_power - noise_power
    return all_signal_power_squared


def calc_expected_signal_power_using_expectation(y):
    return np.sum(y)


def calc_expected_k_using_power(y, d, p, noise_std):
    all_signal_power = calc_expected_signal_power_using_moment(y, noise_std)
    single_signal_power = d * p
    k = int(np.round(all_signal_power / single_signal_power))
    return k


def calc_expected_k_using_expectation(y, d, p):
    all_signal_power = calc_expected_signal_power_using_expectation(y)
    single_signal_power = d * p
    k = int(np.round(all_signal_power / single_signal_power))
    return k


def arange_data(n, d, p, k, std):
    signal = np.full(d, p)
    y = np.zeros(n)
    signal_mask = create_random_k_tuple_sum_to_n(n - d * k, k + 1)
    y = add_pulses(y, signal_mask, signal)
    y = add_gaus_noise(y, 0, std)
    return y


def experiment2():
    d, p, noise_std = 10, 1, 2.5
    fraction = 1 / 4
    l = 5000
    N = 1000
    k = int(fraction * l / d)
    y = arange_data(l, d, p, k, 0)

    true_signal_power = k * p * d
    n_options = np.arange(10, N+1, 10)
    expectation_errs = np.zeros(len(n_options))
    power_errs = np.zeros(len(n_options))
    m = 10
    for j in range(m):
        print(f'j={j}')
        for i, n in enumerate(n_options):

            total_expectation = 0
            total_power_squared = 0
            for t in np.arange(n):
                noisy_y = add_gaus_noise(y, 0, noise_std)
                total_expectation += calc_expected_signal_power_using_expectation(noisy_y)
                total_power_squared += calc_expected_signal_power_using_moment(noisy_y, noise_std)

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


def experiment1():
    d = 10
    p = 10
    k = 30
    std = 2.5

    true_signal_power = k * p * d

    power_errs = []
    expectation_errs = []
    n_options = np.arange(1e3, 5e4, 5000, dtype=int)
    for n in n_options:
        y = arange_data(n, d, p, k, std)

        expected_power = calc_expected_signal_power_using_expectation(y)
        expectation_errs.append(np.abs(true_signal_power - expected_power))

        expected_power = calc_expected_signal_power_using_moment(y, std)
        power_errs.append(np.abs(true_signal_power - expected_power))

        # expected_k = calc_expected_k_using_expectation(y, d, p)
        # err = np.abs(expected_k - k)
        # expectation_errs.append(err)

        # expected_k = calc_expected_k_using_power(y, d, p, std)
        # err = np.abs(expected_k - k)
        # power_errs.append(err)

        print(f'n={n}')

    plt.figure()
    plt.plot(power_errs)
    plt.plot(expectation_errs)
    plt.show()


def run_experiment():
    n = 1000
    d = 10
    p = 2
    k = 30
    std = 2.5

    n_errs = []
    n_options = np.arange(1e3, 2e4, 1000, dtype=int)
    for n in n_options:
        k = int(0.03 * n)
        y = arange_data(n, d, p, k, std)
        expected_k = calc_expected_k_using_expectation(y, d, p)
        err = np.abs(expected_k - k)
        n_errs.append(err)
        print(f'For n={n}, err={err}')

    # std_errs = []
    # std_options = np.arange(1, 5, 0.5)
    # for std in std_options:
    #     y = arange_data(n, d, p, k, std)
    #     expected_k = calc_expected_k(y, d, p, std)
    #     std_errs.append(np.abs(expected_k - k))

    plt.plot(n_errs)
    plt.show()


def __main__():
    experiment2()


if __name__ == '__main__':
    __main__()
