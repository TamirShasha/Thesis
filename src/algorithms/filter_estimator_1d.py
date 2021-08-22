import numpy as np
from scipy.signal import convolve
from time import time

from src.algorithms.utils import log_size_S_1d, calc_mapping_1d, _calc_term_two_derivative_1d, log_prob_all_is_noise, \
    _gradient_descent, calc_most_likelihood_and_optimized_power_1d, relative_error, _calc_mapping_1d_many
from src.experimental.dev3 import heuristic_dp

np.random.seed(500)


class FilterEstimator1D:

    def __init__(self,
                 unnormalized_data: np.ndarray,
                 unnormalized_filter_basis: np.ndarray,
                 num_of_instances: int,
                 noise_std=1.,
                 noise_mean=0.):
        """
        initialize the filter estimator
        :param unnormalized_data: 1d data
        :param unnormalized_filter_basis: list of orthonormal basis
        """

        if len(unnormalized_filter_basis) == 0:
            raise Exception('Must provide basis')

        if len(unnormalized_data.shape) == 1:
            unnormalized_data = np.array([unnormalized_data])

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances = num_of_instances
        self.noise_std = noise_std

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.num_of_data_samples = unnormalized_data.shape[0]
        self.sample_length = unnormalized_data.shape[1]
        self.filter_basis_size = len(unnormalized_filter_basis)
        self.filter_length = len(unnormalized_filter_basis[0])
        self.filter_basis, self.basis_norms = self.normalize_basis()

        # self.convolved_basis, self.convolved_filter = self.calc_constants()
        self.convolved_basis = self.convolve_basis()
        self.term_one = -self.calc_log_size_s()
        self.term_two = log_prob_all_is_noise(self.data, 1)
        self.term_three_const = self.calc_term_three_const()

    def normalize_basis(self):
        """
        :return: normalized basis and respected norms
        """
        normalized_basis = np.zeros_like(self.unnormalized_filter_basis)
        basis_norms = np.zeros(self.filter_basis_size)
        for i, fil in enumerate(self.unnormalized_filter_basis):
            norm = np.linalg.norm(fil)
            basis_norms[i] = norm
            normalized_basis[i] = fil / norm

        return normalized_basis, basis_norms

    def calc_log_size_s(self):
        return log_size_S_1d(self.sample_length, self.num_of_instances, self.filter_length)

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        term = -self.num_of_instances / 2
        return term

    def convolve_basis_element(self, filter_element):
        """
        convolve one basis element with each of the data samples
        :param filter_element: one element from filter basis
        :return: (T, n-d) while T is basis size, n is data length and d is filter length
        """
        flipped_signal_filter = np.flip(filter_element)  # Flipping to cross-correlate
        conv = np.array([convolve(self.data[i], flipped_signal_filter, mode='valid')
                         for i in range(self.num_of_data_samples)])
        return conv

    def convolve_basis(self):
        """
        convolve each normalized basis element with data
        :return: (m, T, n-d) array where T is basis size, n is data length, d is filter length and m is num of samples
        """
        constants = np.zeros(shape=(self.num_of_data_samples,
                                    self.filter_basis_size,
                                    self.sample_length - self.filter_length + 1))
        for i, fil in enumerate(self.filter_basis):
            constants[:, i, :] = self.convolve_basis_element(fil)
        return constants

    def calc_convolved_filter(self, sample_index, filter_coeffs):
        """
        :param sample_index: index of the sample among m data samples
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        return np.inner(self.convolved_basis[sample_index].T, filter_coeffs)

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_1d(self.sample_length,
                               self.num_of_instances,
                               self.filter_length,
                               convolved_filter)

    def calc_mappings(self, convolved_filters):
        """
        :param convolved_filters: array of dot product between data sample and a filter
        :return: likelihood mapping
        """
        return _calc_mapping_1d_many(self.sample_length,
                                     self.num_of_instances,
                                     self.filter_length,
                                     convolved_filters)

    def calc_gradient(self, sample_index, filter_coeffs, convolved_filter, mapping=None):
        """
        calculate sample gradient with respect to each filter coefficient
        :param sample_index: index of the data sample
        :param filter_coeffs: filter coefficients
        :param convolved_filter: convolution of filter with data sample
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """
        if mapping is None:
            mapping = self.calc_mapping(convolved_filter)

        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            gradient[i] = _calc_term_two_derivative_1d(self.sample_length,
                                                       self.num_of_instances,
                                                       self.filter_length,
                                                       self.convolved_basis[sample_index][i],
                                                       convolved_filter,
                                                       mapping)
        return gradient + 2 * self.term_three_const * filter_coeffs

    def calc_gradient_discrete(self, sample_index, filter_coeffs, mapping=None):
        """
        calculate sample gradient with respect to each filter coefficient
        :param sample_index: index of the data sample
        :param filter_coeffs: filter coefficients
        :param convolved_filter: convolution of filter with data sample
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """
        likelihood = self.term_one + \
                     self.term_two + \
                     self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                     mapping[0, -1]

        eps = 1e-4
        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            filter_coeffs_perturbation = filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
            convolved_filter = self.calc_convolved_filter(sample_index, filter_coeffs_perturbation)
            likelihood_perturbation = self.calc_mapping(convolved_filter)[0, -1] + \
                                      self.term_one + \
                                      self.term_two + \
                                      self.term_three_const * np.inner(filter_coeffs_perturbation,
                                                                       filter_coeffs_perturbation)
            gradient[i] = (likelihood_perturbation - likelihood) / eps

        return gradient

    def calc_likelihood_and_gradient(self, filter_coeffs):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """
        likelihoods = np.zeros(self.num_of_data_samples)
        gradients = np.zeros(shape=(self.num_of_data_samples, self.filter_basis_size))

        for i, sample in enumerate(self.data):
            convolved_filter = self.calc_convolved_filter(i, filter_coeffs)
            mapping = self.calc_mapping(convolved_filter)
            # gradients[i] = self.calc_gradient(i, filter_coeffs, convolved_filter, mapping)
            gradients[i] = self.calc_gradient_discrete(i, filter_coeffs, mapping)
            likelihoods[i] = self.term_one + \
                             self.term_two + \
                             self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                             mapping[0, self.num_of_instances]

        # return likelihoods.mean(axis=0), gradients.mean(axis=0)
        return likelihoods.sum(axis=0), gradients.sum(axis=0)

    def calc_likelihood_and_gradient2(self, filter_coeffs, eps=1e-4):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """
        t_start = time()
        likelihoods = np.zeros(self.num_of_data_samples)
        gradients = np.zeros(shape=(self.num_of_data_samples, self.filter_basis_size))

        convolved_filters = np.zeros(shape=(self.num_of_data_samples,
                                            self.filter_basis_size + 1,
                                            self.sample_length - self.filter_length + 1))
        # TODO: do all with 1 command
        filter_coeffs_with_perturbations = np.array([filter_coeffs] +
                                                    [filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
                                                     for i in range(self.filter_basis_size)])
        for i, sample in enumerate(self.data):
            for t in range(self.filter_basis_size + 1):
                convolved_filters[i, t] = self.calc_convolved_filter(i, filter_coeffs_with_perturbations[t])

        t0 = time()
        mappings = self.calc_mappings(convolved_filters.reshape(-1, self.sample_length - self.filter_length + 1))
        print(f'took {time() - t0} seconds')
        mappings = mappings[:, 0, self.num_of_instances].reshape(self.num_of_data_samples, self.filter_basis_size + 1)

        for i in range(self.num_of_data_samples):
            term_three = self.term_three_const * np.array([np.inner(fc, fc) for fc in filter_coeffs_with_perturbations])
            sample_likelihoods = self.term_one + \
                                 self.term_two + \
                                 term_three + \
                                 mappings[i, :]
            gradients[i] = (sample_likelihoods[1:] - sample_likelihoods[0]) / eps
            likelihoods[i] = sample_likelihoods[0]

        # return likelihoods.mean(axis=0), gradients.mean(axis=0)
        print(f'all took {time() - t_start} seconds')
        return likelihoods.sum(axis=0), gradients.sum(axis=0)

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        def _calc_likelihood_and_gradient(filter_coeffs):
            return self.calc_likelihood_and_gradient(filter_coeffs)

        initial_coeffs, t, epsilon, max_iter = np.zeros(self.filter_basis_size), 0.1, 1e-4, 100
        # l, initial_coeffs = self.estimate_max()
        # initial_coeffs = np.array(np.arange(self.filter_basis_size) / sum(np.arange(self.filter_basis_size)))
        # plt.plot(self.filter_basis.T.dot(initial_coeffs))
        # plt.show()
        likelihood, normalized_optimal_coeffs = _gradient_descent(_calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        # likelihood2, p2 = _calc_likelihood_and_gradient(self.basis_norms / self.noise_std)
        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms

        # a = self.term_three_const
        # b1 = self.calc_mapping(self.convolved_basis[0, 0, :])[0, -1]
        # b2 = self.calc_mapping(-self.convolved_basis[0, 0, :])[0, -1]
        # p1 = -b1 / (2 * a)
        # p2 = b2 / (2 * a)
        #
        # print(p1 * self.noise_std / self.basis_norms)
        # print(p2 * self.noise_std / self.basis_norms)
        #
        # likelihoods = []
        # gradients = []
        # xs = np.linspace(0.8, 1.2, 1000)
        # for p in xs:
        #     l, g = self.calc_likelihood_and_gradient(p * self.basis_norms / self.noise_std)
        #     likelihoods.append(l)
        #     gradients.append(g[0])
        #     # if g[0] >= 0:
        #     #     plt.arrow(p, l, 0, 5, width=0.03)
        #     # else:
        #     #     plt.arrow(p, l, 0, -5, width=0.03)
        # plt.plot(xs, likelihoods)
        # x_opt = xs[np.argmax(likelihoods)]
        # plt.title(f'optimum at x={x_opt}')
        # plt.axvline(x=1, color='red', linestyle='--')
        # # plt.scatter(optimal_coeffs, [likelihood])
        # plt.show()
        #
        # likelihoods_gradients = np.gradient(likelihoods, xs[1] - xs[0])
        # fac = 1
        # plt.plot(xs, gradients, label='analytic')
        # plt.plot(xs, likelihoods_gradients * fac, label=f'discrete')
        # plt.title(f'Gradients')
        # plt.axvline(x=1, color='red', linestyle='--')
        # plt.axhline(y=0, color='green', linestyle='--')
        # # plt.scatter(optimal_coeffs, 0)
        # plt.legend()
        # plt.show()
        #
        # exit()

        return likelihood, optimal_coeffs

    def estimate_max(self):

        _p = np.zeros(self.filter_basis_size)
        likelihoods_pos = []
        likelihoods_neg = []
        p_pos = []
        p_neg = []
        for i in range(self.num_of_data_samples):
            print(i)
            c = self.convolved_basis[i].T
            likelihood_pos, locations_pos = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
                                                         c)
            likelihood_neg, locations_neg = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
                                                         -c)

            # if likelihood_pos > likelihood_neg:
            #     likelihood = likelihood_pos
            #     locations = locations_pos
            #
            # else:
            #     c = -c
            #     likelihood = likelihood_neg
            #     locations = locations_neg

            p_pos.append(np.sum(c[locations_pos], 0) / (2 * self.term_three_const))
            p_neg.append(np.sum(-c[locations_neg], 0) / (2 * self.term_three_const))
            likelihoods_pos.append(likelihood_pos)
            likelihoods_neg.append(likelihood_neg)

            # p = np.sum(c[locations], 0) / (2 * self.term_three_const)
            # _p += p

        likelihood_pos = sum(likelihoods_pos)
        likelihood_neg = sum(likelihoods_neg)

        if likelihood_pos > likelihood_neg:
            p = np.nanmean(p_pos, 0)
        else:
            p = np.nanmean(p_neg, 0)

        # _p /= self.num_of_data_samples
        optimal_coeffs = p * self.noise_std / self.basis_norms
        return 0, p


def create_symmetric_basis(length, size):
    """
    creates basis for symmetric signals of given size
    the higher the size, the finer the basis
    :param length: each element length
    :param size: num of elements in returned basis
    """
    step_width = length // (size * 2)
    basis = np.zeros(shape=(size, length))

    for i in range(size):
        pos = i * step_width
        basis[i, pos: pos + step_width] = 1

    basis += np.flip(basis, axis=1)
    basis[size - 1, size * step_width:size * step_width + length % (size * 2)] = 1

    return basis


def create_span_basis(length, size):
    """
    creates basis for discrete signals of given size
    the higher the size, the finer the basis
    :param length: each element length
    :param size: num of elements in returned basis
    """
    step_width = length // size
    basis = np.zeros(shape=(size, length))

    end = 0
    for i in range(size):
        start = end
        end = start + step_width
        if i < length % size:
            end += 1
        basis[i, start: end] = 1

    return basis


from src.experiments.data_simulator_1d import simulate_data
import matplotlib.pyplot as plt


# b = create_span_basis(19, 10)
# print(b)
# for a in b:
#     plt.plot(a)
# plt.show()


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
                    filter_basis = create_span_basis(_d, 7)
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
                    filter_basis = create_symmetric_basis(_d, 1)
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
