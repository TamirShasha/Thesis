import matplotlib.pyplot as plt
import numpy as np
import time

from src.algorithms.filter_estimator_1d import FilterEstimator1D, create_filter_basis
from src.experiments.data_simulator_1d import simulate_data
from src.algorithms.utils import relative_error


def exp():
    n, d, p, k, sig = 4000, 300, 1, 3, 20
    M = 300

    # signal = np.array([1 - np.square(i) for i in np.linspace(-1, 1, d)])
    signal = np.ones(d)

    # sigmas = [0.1, 1, 5, 10, 15, 20]
    sigmas = [5]

    T = 1
    # Ms = [1, 5, 10, 50, 100, 300]
    Ms = [20]
    lns = np.arange(600, 601, 100)
    powers = np.zeros(shape=(len(Ms), len(sigmas), len(lns), T))
    likelihoods = np.zeros(shape=(len(Ms), len(sigmas), len(lns), T))
    likelihoods_opt = np.zeros(shape=(len(sigmas), len(lns), T))

    from src.algorithms.utils import calc_most_likelihood_and_optimized_power_1d
    for mi, m in enumerate(Ms):
        for l, sigma in enumerate(sigmas):
            print(f'At sigma={sigma}')
            for j, _d in enumerate(lns):
                for t in range(T):
                    data = []
                    for i in range(m):
                        _data, pulses = simulate_data(n, d, p, k, sigma, signal)
                        # plt.plot(_data)
                        # plt.show()
                        data.append(_data)
                    data = np.array(data)
                    # filter_basis = create_symmetric_basis(_d, 7)
                    filter_basis = create_filter_basis(_d, 7, 'classic')
                    # plt.plot(filter_basis)
                    # plt.show()
                    # filter_basis = create_span_basis(_d, 10)

                    fil_est = FilterEstimator1D(data, filter_basis, k, sigma)

                    # f, p = fil_est.estimate_max()
                    f, p = fil_est.estimate()
                    # print(p)

                    # f, p = calc_most_likelihood_and_optimized_power_1d(data,
                    #                                                    filter_basis[0],
                    #                                                    k,
                    #                                                    sigma)
                    powers[mi, l, j, t] = p
                    likelihoods[mi, l, j, t] = f
                    # likelihoods_opt[l, j, t] = f2

                    # f2, p2 = calc_most_likelihood_and_optimized_power_1d(data,
                    #                                                      filter_basis[0],
                    #                                                      k // 2,
                    #                                                      sig)
                    # plt.plot(filter_basis[0] * p2)
                    # plt.plot(filter_basis.T.dot(p))
                    # plt.plot(signal)
                    # plt.show()

    # for i in range(len(sigmas)):
    #     plt.plot(lns, powers[i], label=f'{sigmas[i]}')
    # plt.legend()
    # plt.show()

    errs = np.abs(powers - 1).mean(axis=3)[:, 0, 0]
    plt.plot(Ms, errs)
    plt.show()

    # mean_powers = powers.mean(axis=2)
    # var_powers = powers.var(axis=2)
    #
    # for l in range(len(lns)):
    #     for s in range(len(sigmas)):
    #         print(
    #             f'For sigma {sigmas[s]}, length {lns[l]}, mean is {mean_powers[s, l]}, variance is {var_powers[s, l]}')
    #     plt.plot(sigmas, mean_powers[:, l])
    #
    # plt.xlabel('STD')
    # plt.ylabel('power')
    # plt.show()

    # mean_likelihoods = likelihoods.mean(axis=2)
    # mean_likelihoods_opt = likelihoods_opt.mean(axis=2)

    # for l in range(len(lns)):
    #     plt.plot(sigmas, mean_likelihoods[:, l], label='opt filter')
    #     plt.plot(sigmas, mean_likelihoods_opt[:, l], label='1 filter')
    # plt.legend()
    # plt.xlabel('Likelihoods')
    # plt.ylabel('power')
    # plt.show()


def exp2():
    n, d, p, k, sig = 8000, 300, 1, 3, 20

    # signal = np.array([1 - np.square(i) for i in np.linspace(-1, 1, d)])
    signal = np.concatenate([np.full(d // 8, 1 / 3),
                             np.full(d // 8, -2 / 3),
                             np.full(d // 8, -1),
                             np.full(d // 8, 2 / 3),
                             np.full(d // 8, 2 / 3),
                             np.full(d // 8, -1),
                             np.full(d // 8, -2 / 3),
                             np.full(d // 8 + d % 8, 1 / 3)])

    signal2 = np.concatenate([np.full(d // 8, 1 / 3),
                              np.full(d // 8, 2 / 3),
                              np.full(d // 8, 1),
                              np.full(d // 8, 2 / 3),
                              np.full(d // 8, 2 / 3),
                              np.full(d // 8, 1),
                              np.full(d // 8, 2 / 3),
                              np.full(d // 8 + d % 8, 1 / 3)])
    #
    # signal = np.concatenate([np.full(d // 4, 1),
    #                          np.full(d // 4, -1 / 2),
    #                          np.full(d // 4, 1 / 4),
    #                          np.full(d // 4 + d % 4, 1 / 8)])
    signal = np.ones(d)
    # signal = np.concatenate([
    #     np.full(75, -0.5),
    #     np.array([1 - 1.5 * np.square(i) for i in np.linspace(-1, 1, d - 150)]),
    #     np.full(75, -0.5)])

    # sigmas = [0.1, 1, 5, 10, 15, 20]
    sigmas = [1]

    T = 1
    # Ms = [1, 5, 10, 50, 100, 300]
    Ms = [1]
    lns = np.arange(300, 301, 100)
    powers = np.zeros(shape=(len(Ms), len(sigmas), len(lns), T))
    likelihoods = np.zeros(shape=(len(Ms), len(sigmas), len(lns), T))

    for mi, m in enumerate(Ms):
        for l, sigma in enumerate(sigmas):
            print(f'At sigma={sigma}')
            for j, _d in enumerate(lns):
                for t in range(T):
                    data = []
                    for i in range(m):
                        _k = np.random.choice(np.arange(1, 6))
                        _data, pulses = simulate_data(n, d, p, k, sigma, signal)
                        # plt.plot(_data)
                        # plt.show()
                        data.append(_data)
                    data = np.array(data)
                    filter_basis = create_filter_basis(_d, 7, 'classic_symmetric')
                    # filter_basis = create_span_basis(_d, 10)
                    # plt.plot(filter_basis)
                    # plt.show()
                    # filter_basis = create_span_basis(_d, 10)

                    __k = np.random.choice(np.arange(1, 6))
                    fil_est = FilterEstimator1D(data, filter_basis, k, sigma)

                    # f, p = fil_est.estimate_max()
                    t0 = time()
                    f, p = fil_est.estimate()
                    t1 = time()
                    print(f'took {t1 - t0} seconds')
                    # print(p)

                    # f, p = calc_most_likelihood_and_optimized_power_1d(data,
                    #                                                    filter_basis[0],
                    #                                                    k,
                    #                                                    sigma)
                    # powers[mi, l, j, t] = p
                    # likelihoods[mi, l, j, t] = f

                    est_signal = filter_basis.T.dot(p)
                    if len(signal) <= len(est_signal):
                        padded_signal = np.concatenate([signal, np.zeros(len(est_signal) - len(signal))])
                        err, shift = relative_error(est_signal, padded_signal)

                        plt.title(f'Error is {err}')
                        plt.plot(est_signal, label='filter')
                        plt.plot(np.roll(padded_signal, -shift), label='signal')
                        plt.legend()
                        plt.show()


if __name__ == '__main__':
    exp2()
